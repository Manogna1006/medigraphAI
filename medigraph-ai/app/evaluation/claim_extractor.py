"""
Splits a generated answer into atomic, checkable claims.

Kept deliberately simple and dependency-free (regex sentence splitting +
citation-tag stripping) rather than another LLM call, so that computing
hallucination metrics never itself becomes a source of nondeterminism or a
second point of failure. This is a reasonable v1; swap in a proper clinical
sentence segmenter (e.g. scispaCy) if claim boundaries need to be more precise.
"""
from __future__ import annotations

import re

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_CITATION_TAG = re.compile(r"\[E\d+\]")
_BULLET_PREFIX = re.compile(r"^[\s\-\*\u2022\d\.\)]+")


def extract_claims(answer_text: str) -> list[str]:
    """
    Returns a list of atomic claim strings, citation tags stripped, with
    boilerplate/non-claim lines (headers, disclaimers, empty lines) filtered out.
    """
    if not answer_text or not answer_text.strip():
        return []

    # Split on both newlines (bullet points) and sentence boundaries.
    raw_lines: list[str] = []
    for line in answer_text.split("\n"):
        line = line.strip()
        if not line:
            continue
        raw_lines.extend(_SENTENCE_SPLIT.split(line))

    claims: list[str] = []
    for line in raw_lines:
        cleaned = _CITATION_TAG.sub("", line).strip()
        cleaned = _BULLET_PREFIX.sub("", cleaned).strip()
        if _is_checkable_claim(cleaned):
            claims.append(cleaned)
    return claims


def _is_checkable_claim(text: str) -> bool:
    if len(text) < 8:
        return False
    lowered = text.lower()
    # Filter boilerplate disclaimers / meta-sentences that aren't factual claims.
    non_claim_markers = (
        "i don't have enough",
        "please consult",
        "always recommend",
        "consult a clinician",
        "this is decision support",
        "not a diagnosis",
    )
    if any(marker in lowered for marker in non_claim_markers):
        return False
    return True
