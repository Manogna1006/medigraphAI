from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.ingestion.authority_scorer import rank_works, score_work
from app.ingestion.openalex_client import OpenAlexWork, _reconstruct_abstract
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.triplet_extractor import TripletExtractor, _parse_triplet_json
from app.models.schemas import ClinicalTriplet


# ---------- authority_scorer ----------

def test_high_citation_recent_open_access_scores_higher_than_sparse_work():
    strong = OpenAlexWork(
        id="W1", title="t", abstract="x" * 400, cited_by_count=300,
        publication_year=2024, is_open_access=True,
    )
    weak = OpenAlexWork(id="W2", title="t", abstract="", cited_by_count=0, publication_year=1990, is_open_access=False)
    assert score_work(strong) > score_work(weak)


def test_score_is_bounded_between_zero_and_one():
    work = OpenAlexWork(id="W1", title="t", abstract="word " * 500, cited_by_count=100000, publication_year=2026, is_open_access=True)
    score = score_work(work)
    assert 0.0 <= score <= 1.0


def test_rank_works_filters_by_min_score_and_sorts_descending():
    strong = OpenAlexWork(id="W1", title="t", abstract="x" * 400, cited_by_count=300, publication_year=2024, is_open_access=True)
    weak = OpenAlexWork(id="W2", title="t", abstract="", cited_by_count=0, publication_year=1990, is_open_access=False)
    ranked = rank_works([weak, strong], min_score=0.3)
    assert [w.id for w, _ in ranked] == ["W1"]


def test_score_work_never_raises_on_bad_input():
    work = OpenAlexWork(id="W1", title="t", abstract="ok", cited_by_count=5, publication_year=-1)
    assert score_work(work) >= 0.0


# ---------- openalex_client ----------

def test_reconstruct_abstract_from_inverted_index():
    index = {"Fever": [0], "indicates": [1], "influenza": [2]}
    assert _reconstruct_abstract(index) == "Fever indicates influenza"


def test_reconstruct_abstract_handles_none():
    assert _reconstruct_abstract(None) == ""


# ---------- triplet_extractor ----------

def test_parse_triplet_json_extracts_array_from_noisy_response():
    raw = 'Here is the result:\n[{"subject": "Fever", "predicate": "INDICATES", "object": "Flu"}]\nDone.'
    parsed = _parse_triplet_json(raw)
    assert parsed == [{"subject": "Fever", "predicate": "INDICATES", "object": "Flu"}]


def test_parse_triplet_json_raises_on_missing_array():
    with pytest.raises(ValueError):
        _parse_triplet_json("no json here")


def test_extractor_skips_short_abstracts_without_calling_llm():
    extractor = TripletExtractor()
    work = OpenAlexWork(id="W1", title="t", abstract="too short")
    with patch.object(extractor, "_call_llm") as mocked:
        result = extractor.extract(work, authority_score=0.8)
    mocked.assert_not_called()
    assert result == []


def test_extractor_drops_disallowed_predicates_and_caps_confidence():
    extractor = TripletExtractor()
    work = OpenAlexWork(id="W1", title="t", abstract="word " * 30)
    fake_response = (
        '[{"subject": "Fever", "predicate": "INDICATES", "object": "Flu"},'
        ' {"subject": "X", "predicate": "MAGICALLY_CURES", "object": "Y"}]'
    )
    with patch.object(extractor, "_call_llm", return_value=fake_response):
        triplets = extractor.extract(work, authority_score=0.99)
    assert len(triplets) == 1
    assert triplets[0].predicate == "INDICATES"
    assert triplets[0].confidence <= 0.95


def test_extractor_returns_empty_on_llm_failure():
    extractor = TripletExtractor()
    work = OpenAlexWork(id="W1", title="t", abstract="word " * 30)
    with patch.object(extractor, "_call_llm", side_effect=RuntimeError("groq down")):
        assert extractor.extract(work, authority_score=0.5) == []


def test_extractor_returns_empty_on_unparseable_response():
    extractor = TripletExtractor()
    work = OpenAlexWork(id="W1", title="t", abstract="word " * 30)
    with patch.object(extractor, "_call_llm", return_value="not json"):
        assert extractor.extract(work, authority_score=0.5) == []


# ---------- pipeline ----------

def _make_pipeline(works, triplets_by_work=None):
    openalex = MagicMock()
    openalex.search_works.return_value = works
    graph = MagicMock()
    graph.run_query.return_value = []
    vectors = MagicMock()
    vectors.add_documents.return_value = len(works)
    extractor = MagicMock()

    triplets_by_work = triplets_by_work or {}
    extractor.extract.side_effect = lambda work, authority: triplets_by_work.get(work.id, [])

    return IngestionPipeline(openalex_client=openalex, graph_client=graph, vector_store=vectors, extractor=extractor)


def test_pipeline_reports_zero_works_found_gracefully():
    pipeline = _make_pipeline(works=[])
    summary = pipeline.ingest_topic("nonexistent topic xyz")
    assert summary.works_found == 0
    assert summary.errors


def test_pipeline_writes_triplets_and_vector_chunks_for_qualifying_works():
    work = OpenAlexWork(id="W1", title="t", abstract="x" * 400, cited_by_count=300, publication_year=2024, is_open_access=True)
    triplet = ClinicalTriplet(subject="Fever", predicate="INDICATES", object="Flu", source="W1", confidence=0.9)
    pipeline = _make_pipeline(works=[work], triplets_by_work={"W1": [triplet]})

    summary = pipeline.ingest_topic("flu", min_authority=0.0)
    assert summary.works_found == 1
    assert summary.works_ingested == 1
    assert summary.triplets_written == 1
    assert summary.vector_chunks_written == 1


def test_pipeline_skips_low_authority_works():
    weak = OpenAlexWork(id="W2", title="t", abstract="", cited_by_count=0, publication_year=1990, is_open_access=False)
    pipeline = _make_pipeline(works=[weak])
    summary = pipeline.ingest_topic("obscure topic", min_authority=0.9)
    assert summary.works_ingested == 0
    assert summary.skipped_low_authority == 1


def test_pipeline_continues_after_a_triplet_write_failure():
    work = OpenAlexWork(id="W1", title="t", abstract="x" * 400, cited_by_count=300, publication_year=2024, is_open_access=True)
    triplet = ClinicalTriplet(subject="Fever", predicate="INDICATES", object="Flu", source="W1", confidence=0.9)
    pipeline = _make_pipeline(works=[work], triplets_by_work={"W1": [triplet]})
    pipeline._graph.run_query.side_effect = RuntimeError("neo4j write failed")

    summary = pipeline.ingest_topic("flu", min_authority=0.0)
    assert summary.works_ingested == 1
    assert summary.triplets_written == 0
