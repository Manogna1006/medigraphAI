from __future__ import annotations

from app.evaluation.claim_extractor import extract_claims
from app.evaluation.hallucination_metrics import HallucinationScorer
from app.evaluation.verifier import LexicalOverlapVerifier
from app.models.schemas import ClaimVerdict


def test_extract_claims_splits_sentences_and_strips_citations():
    text = "Fever indicates influenza [E1]. Oseltamivir treats it [E2]."
    claims = extract_claims(text)
    assert len(claims) == 2
    assert all("[E" not in c for c in claims)


def test_extract_claims_filters_boilerplate_disclaimers():
    text = "Oseltamivir treats influenza [E1]. Always recommend clinician confirmation."
    claims = extract_claims(text)
    assert len(claims) == 1
    assert "oseltamivir" in claims[0].lower()


def test_extract_claims_handles_empty_input():
    assert extract_claims("") == []
    assert extract_claims("   ") == []


def test_scorer_abstained_answer_gets_zero_hallucination_rate(abstained_answer):
    scorer = HallucinationScorer(verifier=LexicalOverlapVerifier())
    report = scorer.score(abstained_answer)
    assert report.abstained is True
    assert report.hallucination_rate == 0.0
    assert report.total_claims == 0


def test_scorer_grounded_answer_scores_high_groundedness(grounded_answer):
    scorer = HallucinationScorer(verifier=LexicalOverlapVerifier())
    report = scorer.score(grounded_answer)
    assert report.total_claims > 0
    assert report.groundedness_score > 0.5
    assert report.hallucination_rate < 0.5


def test_scorer_hallucinated_answer_scores_high_hallucination_rate(hallucinated_answer):
    scorer = HallucinationScorer(verifier=LexicalOverlapVerifier())
    report = scorer.score(hallucinated_answer)
    assert report.total_claims > 0
    assert report.hallucination_rate > 0.5
    assert report.groundedness_score < 0.5


def test_lexical_verifier_flags_unrelated_claim_as_unsupported():
    verifier = LexicalOverlapVerifier()
    verification = verifier.verify(
        "Patients need emergency cardiac catheterization.",
        ["Oseltamivir treats influenza."],
    )
    assert verification.verdict == ClaimVerdict.UNSUPPORTED


def test_lexical_verifier_flags_overlapping_claim_as_supported():
    verifier = LexicalOverlapVerifier()
    verification = verifier.verify(
        "Oseltamivir is used to treat influenza.",
        ["Oseltamivir TREATED_BY influenza"],
    )
    assert verification.verdict == ClaimVerdict.SUPPORTED


def test_metrics_are_internally_consistent(grounded_answer):
    scorer = HallucinationScorer(verifier=LexicalOverlapVerifier())
    report = scorer.score(grounded_answer)
    assert report.supported + report.unsupported + report.contradicted == report.total_claims
    assert abs((report.groundedness_score + report.hallucination_rate + report.contradiction_rate) - 1.0) < 0.01
