"""
Hybrid retriever: combines Neo4j knowledge-graph traversal with ChromaDB
semantic search, and applies the "semantic firewall" — the single gate that
decides whether there is *enough grounded evidence* to let the LLM answer at
all. This is the primary lever for reducing hallucination, so its logic is
kept explicit and unit-tested rather than buried in a prompt.

Firewall rule (tunable via .env):
    - Reject (abstain) if there are zero KG triplets AND zero vector chunks
      under the distance threshold.
    - Otherwise pass, but only the vector chunks under threshold are kept —
      distant/irrelevant chunks are dropped before they ever reach the LLM.
"""
from __future__ import annotations

from app.config import get_settings
from app.graph.neo4j_client import Neo4jClient
from app.models.schemas import RetrievalContext
from app.utils.logger import logger
from app.vector.chroma_store import ChromaStore


class HybridRetriever:
    def __init__(self, graph_client: Neo4jClient | None = None, vector_store: ChromaStore | None = None):
        self._settings = get_settings()
        self._graph = graph_client or Neo4jClient()
        self._vectors = vector_store or ChromaStore()

    def retrieve(self, query: str, top_k_vectors: int = 5, extra_kg_terms: list[str] | None = None) -> RetrievalContext:
        """
        `extra_kg_terms` lets callers (e.g. the API layer, seeding patient
        symptoms) pull in additional KG triplets beyond what the raw question
        text alone would traverse — this is the multi-source part: a
        question plus a patient's reported symptoms should surface more
        relevant evidence than the question in isolation.
        """
        query = (query or "").strip()
        if not query:
            return RetrievalContext(
                query=query, firewall_passed=False, firewall_reason="Empty query."
            )

        triplets = self._graph.get_clinical_triplets(query)
        for term in extra_kg_terms or []:
            if not term or not term.strip():
                continue
            for t in self._graph.get_clinical_triplets(term.strip()):
                if t not in triplets:
                    triplets.append(t)

        raw_chunks = self._vectors.query(query, top_k=top_k_vectors)

        threshold = self._settings.vector_distance_threshold
        passing_chunks = [c for c in raw_chunks if c.distance <= threshold]
        dropped = len(raw_chunks) - len(passing_chunks)
        if dropped:
            logger.debug("Semantic firewall dropped {} vector chunks over distance threshold {}", dropped, threshold)

        min_triplets = self._settings.min_kg_triplets_required
        has_enough_evidence = len(triplets) >= min_triplets or len(passing_chunks) > 0

        if not has_enough_evidence:
            return RetrievalContext(
                query=query,
                triplets=triplets,
                vector_chunks=[],
                firewall_passed=False,
                firewall_reason=(
                    f"Insufficient grounded evidence: found {len(triplets)} KG triplets "
                    f"(need >= {min_triplets}) and 0 vector chunks under distance {threshold}."
                ),
            )

        return RetrievalContext(
            query=query,
            triplets=triplets,
            vector_chunks=passing_chunks,
            firewall_passed=True,
        )

    def close(self) -> None:
        self._graph.close()
