"""
Extracts (subject, predicate, object) clinical triplets from a paper abstract
using the same Groq/Llama model as synthesis, constrained to a fixed
predicate vocabulary so the resulting graph stays queryable and consistent
with `data/sample_clinical_triplets.json`.

A malformed or unparseable LLM response drops that one abstract's triplets
and logs a warning — it never aborts a whole ingestion batch.
"""
from __future__ import annotations

import json
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import get_settings
from app.ingestion.openalex_client import OpenAlexWork
from app.models.schemas import ClinicalTriplet
from app.utils.logger import logger

# Keep this vocabulary in sync with the relationship types used in
# data/sample_clinical_triplets.json and scripts/seed_neo4j.py.
_ALLOWED_PREDICATES = (
    "INDICATES", "TREATED_BY", "REQUIRES", "CONTRAINDICATED_WITH", "CAUSES", "PREVENTS",
)

_EXTRACTION_SYSTEM_PROMPT = f"""You are a clinical relation-extraction system. Given a paper abstract, \
extract clinical relationship triplets using ONLY these predicates: {", ".join(_ALLOWED_PREDICATES)}.

Rules:
- Only extract relationships the abstract explicitly states or strongly implies — do not infer beyond the text.
- subject and object must be concise clinical entity names (a symptom, disease, or drug), not full sentences.
- If the abstract contains no extractable clinical relationship, return an empty list.
- Extract at most 5 triplets per abstract.

Respond with ONLY a JSON array, no other text:
[{{"subject": "...", "predicate": "INDICATES", "object": "..."}}, ...]"""


class TripletExtractor:
    def __init__(self, model: str | None = None):
        settings = get_settings()
        self._settings = settings
        self._model_name = model or settings.groq_model
        self._llm = None

    def _get_llm(self) -> ChatGroq:
        if self._llm is None:
            if not self._settings.groq_api_key:
                raise RuntimeError("GROQ_API_KEY is not set; cannot run triplet extraction.")
            self._llm = ChatGroq(model=self._model_name, temperature=0.0, api_key=self._settings.groq_api_key, timeout=30)
        return self._llm

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=6), reraise=False)
    def _call_llm(self, abstract: str) -> str:
        llm = self._get_llm()
        response = llm.invoke(
            [SystemMessage(content=_EXTRACTION_SYSTEM_PROMPT), HumanMessage(content=f"Abstract:\n{abstract}")]
        )
        return response.content

    def extract(self, work: OpenAlexWork, authority_score: float) -> list[ClinicalTriplet]:
        if not work.abstract or len(work.abstract.split()) < 20:
            logger.debug("Skipping triplet extraction for {!r}: abstract too short/missing.", work.title)
            return []

        try:
            raw = self._call_llm(work.abstract)
        except Exception as exc:
            logger.warning("Triplet extraction LLM call failed for work {!r}: {}", work.id, exc)
            return []

        try:
            candidates = _parse_triplet_json(raw)
        except Exception as exc:
            logger.warning("Could not parse triplet extraction response for work {!r}: {}. Raw: {!r}", work.id, exc, raw[:200])
            return []

        triplets: list[ClinicalTriplet] = []
        for c in candidates:
            try:
                predicate = c["predicate"].strip().upper()
                if predicate not in _ALLOWED_PREDICATES:
                    logger.debug("Dropping triplet with disallowed predicate {!r} from work {!r}", predicate, work.id)
                    continue
                triplets.append(
                    ClinicalTriplet(
                        subject=c["subject"].strip(),
                        predicate=predicate,
                        object=c["object"].strip(),
                        source=work.id,
                        # Triplet confidence is capped by the source's authority — a paper's
                        # own certainty language can't outweigh how authoritative the source is.
                        confidence=round(min(0.95, authority_score), 4),
                    )
                )
            except (KeyError, AttributeError) as exc:
                logger.warning("Skipping malformed triplet candidate {} from work {!r}: {}", c, work.id, exc)
        return triplets


def _parse_triplet_json(raw: str) -> list[dict]:
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        raise ValueError("No JSON array found in extraction response")
    data = json.loads(match.group(0))
    if not isinstance(data, list):
        raise ValueError("Extraction response JSON is not a list")
    return data
