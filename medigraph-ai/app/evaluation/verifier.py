"""
Verifies a single claim against the retrieved evidence set.

Two backends:
  1. LLMJudgeVerifier   — asks Groq/Llama to classify SUPPORTED / UNSUPPORTED /
     CONTRADICTED with a one-line rationale. This is the accurate path and the
     default in production.
  2. LexicalOverlapVerifier — a zero-dependency, zero-network fallback used
     when no API key is configured (e.g. CI, offline unit tests) so the
     hallucination-metrics *pipeline* itself always has full test coverage,
     independent of whether a live LLM is reachable.

Both implement the same `VerifierProtocol`, so `hallucination_metrics.py`
never needs to know which one it's using.
"""
from __future__ import annotations

import json
import re
from typing import Protocol

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import get_settings
from app.models.schemas import ClaimVerdict, ClaimVerification
from app.utils.logger import logger

_JUDGE_SYSTEM_PROMPT = """You are a strict fact-checking judge for a medical AI system. \
Given a CLAIM and a list of EVIDENCE lines, decide whether the claim is:
- "supported": the evidence directly backs this claim
- "contradicted": the evidence directly conflicts with this claim
- "unsupported": the evidence neither confirms nor conflicts with it (including no relevant evidence)

Respond with ONLY a JSON object, no other text:
{"verdict": "supported|contradicted|unsupported", "matched_evidence": "<the evidence line, or null>", "rationale": "<one short sentence>"}"""


class VerifierProtocol(Protocol):
    def verify(self, claim: str, evidence: list[str]) -> ClaimVerification: ...


class LLMJudgeVerifier:
    def __init__(self, model: str | None = None):
        settings = get_settings()
        self._settings = settings
        self._model_name = model or settings.hallucination_judge_model
        self._llm = None

    def _get_llm(self) -> ChatGroq:
        if self._llm is None:
            if not self._settings.groq_api_key:
                raise RuntimeError("GROQ_API_KEY is not set; cannot run LLM-judge verification.")
            self._llm = ChatGroq(model=self._model_name, temperature=0.0, api_key=self._settings.groq_api_key, timeout=20)
        return self._llm

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=4), reraise=False)
    def _call_judge(self, claim: str, evidence: list[str]) -> str:
        llm = self._get_llm()
        evidence_block = "\n".join(f"- {e}" for e in evidence) if evidence else "(no evidence retrieved)"
        content = f"CLAIM: {claim}\n\nEVIDENCE:\n{evidence_block}"
        response = llm.invoke([SystemMessage(content=_JUDGE_SYSTEM_PROMPT), HumanMessage(content=content)])
        return response.content

    def verify(self, claim: str, evidence: list[str]) -> ClaimVerification:
        try:
            raw = self._call_judge(claim, evidence)
            parsed = _parse_judge_json(raw)
            return ClaimVerification(
                claim=claim,
                verdict=ClaimVerdict(parsed["verdict"]),
                matched_evidence=parsed.get("matched_evidence"),
                judge_rationale=parsed.get("rationale"),
            )
        except Exception as exc:
            logger.warning("LLM-judge verification failed for claim {!r}: {}. Falling back to lexical check.", claim, exc)
            return LexicalOverlapVerifier().verify(claim, evidence)


class LexicalOverlapVerifier:
    """
    Deterministic fallback: flags a claim as SUPPORTED only if it shares a
    strong keyword overlap with at least one evidence line. Deliberately
    conservative (biased toward UNSUPPORTED) since a fallback verifier's job
    is to never silently mask a real hallucination.
    """

    _STOPWORDS = {
        "the", "a", "an", "is", "are", "was", "were", "of", "to", "in", "on", "for",
        "and", "or", "with", "as", "by", "this", "that", "it", "be", "may", "can",
    }

    def _tokens(self, text: str) -> set[str]:
        words = re.findall(r"[a-zA-Z][a-zA-Z\-]+", text.lower())
        return {self._stem(w) for w in words if w not in self._STOPWORDS and len(w) > 2}

    @staticmethod
    def _stem(word: str) -> str:
        # Deliberately crude suffix-stripping (not a real stemmer) so that
        # "treat"/"treated"/"treatment" and "indicate"/"indicates" overlap —
        # good enough for a keyword-overlap fallback, not for anything precise.
        for suffix in ("ments", "ment", "ing", "ated", "ates", "ate", "ed", "es", "s"):
            if word.endswith(suffix) and len(word) - len(suffix) >= 3:
                return word[: -len(suffix)]
        return word

    def verify(self, claim: str, evidence: list[str]) -> ClaimVerification:
        claim_tokens = self._tokens(claim)
        if not claim_tokens or not evidence:
            return ClaimVerification(claim=claim, verdict=ClaimVerdict.UNSUPPORTED, judge_rationale="No evidence available.")

        best_overlap, best_line = 0.0, None
        for line in evidence:
            line_tokens = self._tokens(line)
            if not line_tokens:
                continue
            overlap = len(claim_tokens & line_tokens) / len(claim_tokens)
            if overlap > best_overlap:
                best_overlap, best_line = overlap, line

        if best_overlap >= 0.5:
            return ClaimVerification(
                claim=claim, verdict=ClaimVerdict.SUPPORTED, matched_evidence=best_line,
                judge_rationale=f"Lexical overlap {best_overlap:.0%} (fallback verifier).",
            )
        return ClaimVerification(
            claim=claim, verdict=ClaimVerdict.UNSUPPORTED, matched_evidence=best_line,
            judge_rationale=f"Best lexical overlap only {best_overlap:.0%} (fallback verifier).",
        )


def _parse_judge_json(raw: str) -> dict:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in judge response: {raw!r}")
    data = json.loads(match.group(0))
    if data.get("verdict") not in {"supported", "unsupported", "contradicted"}:
        raise ValueError(f"Invalid verdict in judge response: {data!r}")
    return data


def get_default_verifier() -> VerifierProtocol:
    """Picks the LLM judge when an API key is configured, else the offline fallback."""
    settings = get_settings()
    if settings.groq_api_key:
        return LLMJudgeVerifier()
    logger.warning("No GROQ_API_KEY configured — using LexicalOverlapVerifier for hallucination scoring.")
    return LexicalOverlapVerifier()
