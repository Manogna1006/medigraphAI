"""
Shared fixtures. Every fixture here is a mock/stub — the test suite runs
fully offline with no Neo4j instance and no Groq API key required, so it can
run in CI on every commit.
"""
from __future__ import annotations

import pytest

from app.models.schemas import ClinicalTriplet, RetrievalContext, SynthesizedAnswer, VectorChunk


@pytest.fixture
def sample_triplets() -> list[ClinicalTriplet]:
    return [
        ClinicalTriplet(subject="Fever", predicate="INDICATES", object="Influenza", source="PMID:1", confidence=0.8),
        ClinicalTriplet(subject="Influenza", predicate="TREATED_BY", object="Oseltamivir", source="PMID:1", confidence=0.75),
    ]


@pytest.fixture
def sample_vector_chunks() -> list[VectorChunk]:
    return [
        VectorChunk(text="Oseltamivir reduces influenza symptom duration when given within 48 hours.", source="doc-1", distance=0.12),
        VectorChunk(text="Unrelated distant chunk about dermatology.", source="doc-2", distance=0.91),
    ]


@pytest.fixture
def grounded_context(sample_triplets, sample_vector_chunks) -> RetrievalContext:
    return RetrievalContext(
        query="What treats influenza?",
        triplets=sample_triplets,
        vector_chunks=[sample_vector_chunks[0]],  # only the close one, as the firewall would keep
        firewall_passed=True,
    )


@pytest.fixture
def empty_context() -> RetrievalContext:
    return RetrievalContext(query="What treats a made-up disease?", triplets=[], vector_chunks=[], firewall_passed=False,
                             firewall_reason="Insufficient grounded evidence.")


@pytest.fixture
def grounded_answer(grounded_context) -> SynthesizedAnswer:
    return SynthesizedAnswer(
        query=grounded_context.query,
        answer=(
            "Influenza is indicated by fever [E1]. "
            "Oseltamivir is a recommended treatment for influenza [E2]. "
            "It works best when given within 48 hours of symptom onset [E3]."
        ),
        citations=["1", "2", "3"],
        abstained=False,
        latency_seconds=1.2,
        retrieval_context=grounded_context,
    )


@pytest.fixture
def hallucinated_answer(grounded_context) -> SynthesizedAnswer:
    return SynthesizedAnswer(
        query=grounded_context.query,
        answer=(
            "Patients with influenza should undergo emergency cardiac catheterization immediately. "
            "A full-body MRI is required before any treatment can begin."
        ),
        citations=[],
        abstained=False,
        latency_seconds=1.0,
        retrieval_context=grounded_context,
    )


@pytest.fixture
def abstained_answer(empty_context) -> SynthesizedAnswer:
    return SynthesizedAnswer(
        query=empty_context.query,
        answer="I don't have enough verified evidence to answer this safely.",
        citations=[],
        abstained=True,
        latency_seconds=0.01,
        retrieval_context=empty_context,
    )
