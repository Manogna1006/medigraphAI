"""
Thin, defensive wrapper around the Neo4j driver.

Design choices that matter for reliability:
- A connection failure at startup is caught and logged, not raised past the
  caller uninvited — callers explicitly call `.verify_connectivity()` if they
  need to fail fast (e.g. at app startup).
- Every query goes through `run_query`, which retries transient errors
  (connection resets, leader-switch during Aura maintenance, etc.) with
  exponential backoff, and always closes its session.
- Query results are mapped into `ClinicalTriplet` so nothing downstream ever
  touches a raw Neo4j Record.
"""
from __future__ import annotations

from typing import Any

from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError, ServiceUnavailable
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.config import get_settings
from app.models.schemas import ClinicalTriplet
from app.utils.logger import logger

# Cypher is parameterized everywhere below — never string-format user input into a query.
_TRIPLET_QUERY = """
MATCH (s:Entity)-[r]->(o:Entity)
WHERE toLower(s.name) CONTAINS toLower($term) OR toLower(o.name) CONTAINS toLower($term)
RETURN s.name AS subject, type(r) AS predicate, o.name AS object,
       r.source AS source, coalesce(r.confidence, 1.0) AS confidence
LIMIT $limit
"""


class Neo4jClient:
    def __init__(self, uri: str | None = None, username: str | None = None, password: str | None = None):
        settings = get_settings()
        self._uri = uri or settings.neo4j_uri
        self._username = username or settings.neo4j_username
        self._password = password or settings.neo4j_password
        self._driver = None

    def _get_driver(self):
        if self._driver is None:
            self._driver = GraphDatabase.driver(self._uri, auth=(self._username, self._password))
        return self._driver

    def verify_connectivity(self) -> bool:
        """Explicit health check. Call this at app startup, not implicitly on every query."""
        try:
            self._get_driver().verify_connectivity()
            logger.info("Neo4j connectivity verified at {}", self._uri)
            return True
        except ServiceUnavailable as exc:
            logger.error("Neo4j is unreachable at {}: {}", self._uri, exc)
            return False

    @retry(
        retry=retry_if_exception_type((ServiceUnavailable,)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
        reraise=True,
    )
    def run_query(self, query: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        driver = self._get_driver()
        with driver.session() as session:
            try:
                result = session.run(query, parameters or {})
                return [record.data() for record in result]
            except Neo4jError as exc:
                logger.error("Neo4j query failed: {} | query={}", exc, query)
                raise

    def get_clinical_triplets(self, term: str, limit: int = 30) -> list[ClinicalTriplet]:
        """
        Traverse the graph for triplets relevant to `term` (a symptom, disease, or drug name).
        Returns [] on any failure rather than raising — callers (the retriever) treat an
        empty KG result as a signal for the semantic firewall, not a crash.
        """
        if not term or not term.strip():
            return []
        try:
            rows = self.run_query(_TRIPLET_QUERY, {"term": term.strip(), "limit": limit})
        except (ServiceUnavailable, Neo4jError) as exc:
            logger.warning("KG traversal failed for term={!r}: {}. Returning empty result.", term, exc)
            return []

        triplets: list[ClinicalTriplet] = []
        for row in rows:
            try:
                triplets.append(
                    ClinicalTriplet(
                        subject=row["subject"],
                        predicate=row["predicate"],
                        object=row["object"],
                        source=row.get("source"),
                        confidence=row.get("confidence", 1.0),
                    )
                )
            except Exception as exc:  # malformed row shouldn't crash the whole retrieval
                logger.warning("Skipping malformed KG row {}: {}", row, exc)
        return triplets

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()
            self._driver = None
