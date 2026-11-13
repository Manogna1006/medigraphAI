"""
Pydantic schemas shared across the retrieval, synthesis, and evaluation layers.

Keeping these in one place means every module validates against the same
contract, which is most of what "no errors, no faults" buys you in practice:
a bad shape fails at the boundary, not three function calls later.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class ClinicalTriplet(BaseModel):
    """One edge pulled from the Neo4j knowledge graph, e.g. (Symptom)-[INDICATES]->(Disease)."""

    subject: str
    predicate: str
    object: str
    source: Optional[str] = Field(default=None, description="Citation / provenance, e.g. PMID or OpenAlex work ID")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    def as_text(self) -> str:
        return f"{self.subject} {self.predicate} {self.object}".strip()


class VectorChunk(BaseModel):
    """One retrieved chunk from ChromaDB."""

    text: str
    source: Optional[str] = None
    distance: float = Field(..., description="Lower = more similar. Used by the semantic firewall.")


class RetrievalContext(BaseModel):
    """Everything retrieved for a single query, before synthesis."""

    query: str
    triplets: list[ClinicalTriplet] = Field(default_factory=list)
    vector_chunks: list[VectorChunk] = Field(default_factory=list)
    firewall_passed: bool = Field(
        default=True,
        description="False if evidence was too sparse/distant and synthesis should abstain.",
    )
    firewall_reason: Optional[str] = None

    def evidence_texts(self) -> list[str]:
        return [t.as_text() for t in self.triplets] + [c.text for c in self.vector_chunks]

    def is_empty(self) -> bool:
        return not self.triplets and not self.vector_chunks


class PatientContext(BaseModel):
    """Optional structured patient data (vitals) that can accompany a query."""

    age: Optional[int] = Field(default=None, ge=0, le=130)
    symptoms: list[str] = Field(default_factory=list)
    vitals: dict[str, float] = Field(default_factory=dict)
    notes: Optional[str] = None


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    patient_context: Optional[PatientContext] = None

    @field_validator("question")
    @classmethod
    def strip_question(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("question must not be empty")
        return v


class ClaimVerdict(str, Enum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    CONTRADICTED = "contradicted"


class ClaimVerification(BaseModel):
    claim: str
    verdict: ClaimVerdict
    matched_evidence: Optional[str] = None
    judge_rationale: Optional[str] = None


class HallucinationReport(BaseModel):
    """Result of scoring one generated answer against its retrieval context."""

    query: str
    total_claims: int
    supported: int
    unsupported: int
    contradicted: int
    hallucination_rate: float = Field(..., ge=0.0, le=1.0)
    groundedness_score: float = Field(..., ge=0.0, le=1.0)
    contradiction_rate: float = Field(..., ge=0.0, le=1.0)
    claim_verifications: list[ClaimVerification] = Field(default_factory=list)
    abstained: bool = Field(default=False, description="True if the system refused to answer due to the firewall")


class SynthesizedAnswer(BaseModel):
    query: str
    answer: str
    citations: list[str] = Field(default_factory=list)
    abstained: bool = False
    latency_seconds: float = 0.0
    retrieval_context: RetrievalContext


class QueryResponse(BaseModel):
    answer: SynthesizedAnswer
    hallucination_report: Optional[HallucinationReport] = None
