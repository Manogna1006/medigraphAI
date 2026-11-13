"""
Computes the hallucination-rate metrics for a single synthesized answer.

Definitions (these are the numbers a reviewer/professor will ask you to justify):

    hallucination_rate   = unsupported_claims / total_claims
                            (claims made with no backing evidence at all —
                             the core "did the model make things up" signal)

    contradiction_rate   = contradicted_claims / total_claims
                            (claims that actively conflict with retrieved evidence —
                             arguably worse than "unsupported" and reported separately)

    groundedness_score   = supported_claims / total_claims
                            (1 - hallucination_rate - contradiction_rate; the
                             complement, reported because it reads better on a dashboard)

An abstained answer (the semantic firewall refused to answer) is reported with
hallucination_rate = 0.0 and abstained=True — refusing to guess is the
*correct* behavior, not a hallucination, and should never be penalized as one.
"""
from __future__ import annotations

from app.evaluation.claim_extractor import extract_claims
from app.evaluation.verifier import VerifierProtocol, get_default_verifier
from app.models.schemas import ClaimVerdict, HallucinationReport, SynthesizedAnswer
from app.utils.logger import logger


class HallucinationScorer:
    def __init__(self, verifier: VerifierProtocol | None = None):
        self._verifier = verifier or get_default_verifier()

    def score(self, answer: SynthesizedAnswer) -> HallucinationReport:
        if answer.abstained:
            return HallucinationReport(
                query=answer.query,
                total_claims=0,
                supported=0,
                unsupported=0,
                contradicted=0,
                hallucination_rate=0.0,
                groundedness_score=1.0,
                contradiction_rate=0.0,
                claim_verifications=[],
                abstained=True,
            )

        evidence = answer.retrieval_context.evidence_texts()
        claims = extract_claims(answer.answer)

        if not claims:
            logger.warning("No checkable claims extracted from answer for query={!r}", answer.query)
            return HallucinationReport(
                query=answer.query, total_claims=0, supported=0, unsupported=0, contradicted=0,
                hallucination_rate=0.0, groundedness_score=1.0, contradiction_rate=0.0,
                claim_verifications=[], abstained=False,
            )

        verifications = []
        for claim in claims:
            try:
                verifications.append(self._verifier.verify(claim, evidence))
            except Exception as exc:
                logger.error("Verifier crashed on claim {!r}: {}. Counting as unsupported.", claim, exc)
                verifications.append(
                    ClaimVerdict.UNSUPPORTED  # placeholder to keep counts aligned below
                )

        supported = sum(1 for v in verifications if getattr(v, "verdict", v) == ClaimVerdict.SUPPORTED)
        contradicted = sum(1 for v in verifications if getattr(v, "verdict", v) == ClaimVerdict.CONTRADICTED)
        total = len(verifications)
        unsupported = total - supported - contradicted

        # Normalize any raw ClaimVerdict placeholders (from a verifier crash) into real objects.
        from app.models.schemas import ClaimVerification
        clean_verifications = [
            v if isinstance(v, ClaimVerification) else ClaimVerification(claim="(unavailable)", verdict=v)
            for v in verifications
        ]

        return HallucinationReport(
            query=answer.query,
            total_claims=total,
            supported=supported,
            unsupported=unsupported,
            contradicted=contradicted,
            hallucination_rate=round(unsupported / total, 4) if total else 0.0,
            groundedness_score=round(supported / total, 4) if total else 1.0,
            contradiction_rate=round(contradicted / total, 4) if total else 0.0,
            claim_verifications=clean_verifications,
            abstained=False,
        )
