"""
Authority scoring: combines citation count, recency, open-access availability,
and abstract completeness into a single 0-1 score used to (a) rank which
works get ingested into the KG/vector store, and (b) weight the `confidence`
field on any triplet extracted from that work.

This is the piece the semantic firewall ultimately depends on being honest:
a triplet sourced from a low-authority work should carry a lower confidence
score, not the same 1.0 as a well-cited systematic review.

Weights are intentionally simple and named constants (not learned) so the
scoring is auditable — a reviewer can see exactly why a work scored 0.62.
"""
from __future__ import annotations

import math
from datetime import date

from app.ingestion.openalex_client import OpenAlexWork
from app.utils.logger import logger

# Weights sum to 1.0 — adjust here, not inline, if the scoring needs retuning.
_WEIGHT_CITATIONS = 0.45
_WEIGHT_RECENCY = 0.20
_WEIGHT_OPEN_ACCESS = 0.10
_WEIGHT_ABSTRACT_QUALITY = 0.25

_CITATION_SATURATION = 500  # citation count at which the citation sub-score saturates near 1.0
_RECENCY_HALF_LIFE_YEARS = 10  # a paper this many years old scores ~0.5 on recency


def _citation_subscore(cited_by_count: int) -> float:
    if cited_by_count <= 0:
        return 0.0
    # Log-scaled so a jump from 5->50 citations matters more than 500->550.
    return min(1.0, math.log1p(cited_by_count) / math.log1p(_CITATION_SATURATION))


def _recency_subscore(publication_year: int | None) -> float:
    if not publication_year:
        return 0.3  # unknown year — neither reward nor heavily penalize
    age_years = max(0, date.today().year - publication_year)
    return 0.5 ** (age_years / _RECENCY_HALF_LIFE_YEARS)


def _abstract_quality_subscore(abstract: str) -> float:
    if not abstract:
        return 0.0
    word_count = len(abstract.split())
    if word_count < 30:
        return 0.2  # present but too thin to extract reliable clinical relationships from
    return min(1.0, word_count / 200)


def score_work(work: OpenAlexWork) -> float:
    """Returns an authority score in [0, 1]. Never raises — malformed input scores 0.0."""
    try:
        citation_score = _citation_subscore(work.cited_by_count)
        recency_score = _recency_subscore(work.publication_year)
        oa_score = 1.0 if work.is_open_access else 0.0
        abstract_score = _abstract_quality_subscore(work.abstract)

        total = (
            _WEIGHT_CITATIONS * citation_score
            + _WEIGHT_RECENCY * recency_score
            + _WEIGHT_OPEN_ACCESS * oa_score
            + _WEIGHT_ABSTRACT_QUALITY * abstract_score
        )
        return round(min(1.0, max(0.0, total)), 4)
    except Exception as exc:
        logger.warning("Authority scoring failed for work id={!r}: {}", getattr(work, "id", "?"), exc)
        return 0.0


def rank_works(works: list[OpenAlexWork], min_score: float = 0.0) -> list[tuple[OpenAlexWork, float]]:
    """Returns (work, score) pairs sorted by score descending, filtered by min_score."""
    scored = [(w, score_work(w)) for w in works]
    scored = [pair for pair in scored if pair[1] >= min_score]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored
