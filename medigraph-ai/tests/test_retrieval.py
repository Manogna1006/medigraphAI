from __future__ import annotations

from unittest.mock import MagicMock

from app.models.schemas import ClinicalTriplet, VectorChunk
from app.retrieval.hybrid_retriever import HybridRetriever


def _make_retriever(triplets=None, vector_chunks=None):
    graph = MagicMock()
    graph.get_clinical_triplets.return_value = triplets or []
    vectors = MagicMock()
    vectors.query.return_value = vector_chunks or []
    return HybridRetriever(graph_client=graph, vector_store=vectors)


def test_firewall_passes_with_kg_evidence_only():
    triplets = [ClinicalTriplet(subject="Fever", predicate="INDICATES", object="Flu")]
    retriever = _make_retriever(triplets=triplets, vector_chunks=[])
    ctx = retriever.retrieve("what causes fever")
    assert ctx.firewall_passed is True
    assert len(ctx.triplets) == 1


def test_firewall_passes_with_close_vector_chunk_only():
    chunks = [VectorChunk(text="relevant chunk", distance=0.1)]
    retriever = _make_retriever(triplets=[], vector_chunks=chunks)
    ctx = retriever.retrieve("some question")
    assert ctx.firewall_passed is True
    assert len(ctx.vector_chunks) == 1


def test_firewall_blocks_when_no_evidence():
    retriever = _make_retriever(triplets=[], vector_chunks=[])
    ctx = retriever.retrieve("nonexistent disease question")
    assert ctx.firewall_passed is False
    assert ctx.firewall_reason is not None


def test_firewall_drops_far_vector_chunks():
    far_chunk = VectorChunk(text="irrelevant", distance=0.99)
    retriever = _make_retriever(triplets=[], vector_chunks=[far_chunk])
    ctx = retriever.retrieve("some question")
    assert ctx.firewall_passed is False
    assert len(ctx.vector_chunks) == 0


def test_empty_query_is_rejected():
    retriever = _make_retriever()
    ctx = retriever.retrieve("   ")
    assert ctx.firewall_passed is False


def test_extra_kg_terms_pull_in_additional_triplets():
    graph = MagicMock()
    # Return different triplets depending on the term queried.
    def fake_get(term):
        if term == "chest pain":
            return [ClinicalTriplet(subject="Chest Pain", predicate="INDICATES", object="MI")]
        if term == "shortness of breath":
            return [ClinicalTriplet(subject="Shortness of Breath", predicate="INDICATES", object="CHF")]
        return []

    graph.get_clinical_triplets.side_effect = fake_get
    vectors = MagicMock()
    vectors.query.return_value = []
    retriever = HybridRetriever(graph_client=graph, vector_store=vectors)

    ctx = retriever.retrieve("chest pain", extra_kg_terms=["shortness of breath"])
    subjects = {t.subject for t in ctx.triplets}
    assert subjects == {"Chest Pain", "Shortness of Breath"}


def test_extra_kg_terms_deduplicates_repeated_triplets():
    graph = MagicMock()
    triplet = ClinicalTriplet(subject="Fever", predicate="INDICATES", object="Flu")
    graph.get_clinical_triplets.return_value = [triplet]
    vectors = MagicMock()
    vectors.query.return_value = []
    retriever = HybridRetriever(graph_client=graph, vector_store=vectors)

    ctx = retriever.retrieve("fever", extra_kg_terms=["fever", "fever"])
    assert len(ctx.triplets) == 1
