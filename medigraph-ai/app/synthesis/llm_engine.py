"""
Synthesis engine: turns a RetrievalContext into a grounded, cited answer.

Reliability choices:
- If the firewall already rejected the context, we never call the LLM at all —
  abstention is free and instant, and it's the #1 hallucination-reduction lever.
- The prompt requires every clinical claim to cite the evidence line it came
  from (e.g. "[E3]"); `_extract_citations` parses those back out for the
  hallucination-metrics stage to check.
- All Groq calls are wrapped in retry + a timeout-safe fallback: on repeated
  failure we return a clearly-labeled degraded answer instead of raising a
  500 to the caller.
"""
from __future__ import annotations

import re
import time

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import get_settings
from app.models.schemas import RetrievalContext, SynthesizedAnswer
from app.utils.logger import logger

_SYSTEM_PROMPT = """You are a clinical decision-support assistant. You must answer ONLY using \
the numbered evidence lines provided. Rules:
1. Every clinical claim you make must end with a citation tag like [E1] or [E2] referencing \
the evidence line number it comes from.
2. If the evidence does not support a claim, do NOT make that claim.
3. If the evidence is insufficient to answer the question at all, say so plainly instead of guessing.
4. Be concise and structured. This is decision support, not a diagnosis — always recommend \
clinician confirmation for any treatment decision."""

_ABSTAIN_ANSWER = (
    "I don't have enough verified evidence in the knowledge graph or reference corpus to answer "
    "this safely. Rather than guess, I'm declining to answer — please consult a clinician or "
    "expand the knowledge base with relevant literature."
)

_CITATION_PATTERN = re.compile(r"\[E(\d+)\]")


class LLMSynthesisEngine:
    def __init__(self, model: str | None = None, temperature: float = 0.0):
        settings = get_settings()
        self._settings = settings
        self._model_name = model or settings.groq_model
        self._temperature = temperature
        self._llm = None  # lazy: don't require a valid API key just to import this module

    def _get_llm(self) -> ChatGroq:
        if self._llm is None:
            if not self._settings.groq_api_key:
                raise RuntimeError(
                    "GROQ_API_KEY is not set. Add it to your .env file before requesting synthesis."
                )
            self._llm = ChatGroq(
                model=self._model_name,
                temperature=self._temperature,
                api_key=self._settings.groq_api_key,
                timeout=30,
            )
        return self._llm

    @staticmethod
    def _build_evidence_block(context: RetrievalContext) -> tuple[str, list[str]]:
        evidence_lines = context.evidence_texts()
        numbered = [f"[E{i+1}] {line}" for i, line in enumerate(evidence_lines)]
        return "\n".join(numbered), evidence_lines

    @staticmethod
    def _extract_citations(answer_text: str) -> list[str]:
        return sorted(set(_CITATION_PATTERN.findall(answer_text)), key=int)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8), reraise=False)
    def _call_llm(self, question: str, evidence_block: str, patient_context_text: str) -> str:
        llm = self._get_llm()
        user_content = (
            f"Patient context:\n{patient_context_text or 'None provided.'}\n\n"
            f"Evidence:\n{evidence_block}\n\n"
            f"Question: {question}"
        )
        response = llm.invoke([SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=user_content)])
        return response.content

    def synthesize(self, context: RetrievalContext, patient_context_text: str = "") -> SynthesizedAnswer:
        start = time.perf_counter()

        if not context.firewall_passed or context.is_empty():
            logger.info("Firewall blocked synthesis for query={!r}: {}", context.query, context.firewall_reason)
            return SynthesizedAnswer(
                query=context.query,
                answer=_ABSTAIN_ANSWER,
                citations=[],
                abstained=True,
                latency_seconds=time.perf_counter() - start,
                retrieval_context=context,
            )

        evidence_block, _ = self._build_evidence_block(context)
        try:
            answer_text = self._call_llm(context.query, evidence_block, patient_context_text)
        except Exception as exc:
            logger.error("LLM synthesis failed after retries for query={!r}: {}", context.query, exc)
            return SynthesizedAnswer(
                query=context.query,
                answer=(
                    "The reasoning engine is temporarily unavailable, so I'm not able to generate a "
                    "verified answer right now rather than risk an unverified one. Please retry shortly."
                ),
                citations=[],
                abstained=True,
                latency_seconds=time.perf_counter() - start,
                retrieval_context=context,
            )

        citations = self._extract_citations(answer_text)
        return SynthesizedAnswer(
            query=context.query,
            answer=answer_text,
            citations=citations,
            abstained=False,
            latency_seconds=time.perf_counter() - start,
            retrieval_context=context,
        )
