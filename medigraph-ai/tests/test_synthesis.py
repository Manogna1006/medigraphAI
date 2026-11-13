from __future__ import annotations

from unittest.mock import patch

from app.synthesis.llm_engine import LLMSynthesisEngine


def test_synthesis_abstains_immediately_when_firewall_blocked(empty_context):
    engine = LLMSynthesisEngine()
    answer = engine.synthesize(empty_context)
    assert answer.abstained is True
    assert "consult a clinician" in answer.answer.lower() or "declining" in answer.answer.lower()
    assert answer.citations == []


def test_synthesis_calls_llm_when_firewall_passes(grounded_context):
    engine = LLMSynthesisEngine()
    with patch.object(engine, "_call_llm", return_value="Oseltamivir treats influenza [E2].") as mocked:
        answer = engine.synthesize(grounded_context)
    mocked.assert_called_once()
    assert answer.abstained is False
    assert answer.citations == ["2"]


def test_synthesis_degrades_gracefully_on_repeated_llm_failure(grounded_context):
    engine = LLMSynthesisEngine()
    with patch.object(engine, "_call_llm", side_effect=RuntimeError("groq is down")):
        answer = engine.synthesize(grounded_context)
    assert answer.abstained is True
    assert "temporarily unavailable" in answer.answer.lower()


def test_citation_extraction_handles_multiple_and_duplicate_tags():
    text = "Claim one [E1]. Claim two [E2]. Repeated evidence [E1] again."
    citations = LLMSynthesisEngine._extract_citations(text)
    assert citations == ["1", "2"]
