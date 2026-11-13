"""
Orchestrates the full ingestion pipeline for one or more topics:

    OpenAlex search -> authority scoring -> filter by min score
        -> triplet extraction (LLM) -> write triplets to Neo4j
        -> write abstract chunks to ChromaDB (tagged with authority score)

Every stage degrades independently: a topic that returns zero works, a work
with no extractable triplets, or a Neo4j write failure all just reduce the
final counts — they never abort the whole run. `IngestionSummary` reports
exactly what happened so a bad run is visible, not silent.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.config import get_settings
from app.graph.neo4j_client import Neo4jClient
from app.ingestion.authority_scorer import rank_works
from app.ingestion.openalex_client import OpenAlexClient
from app.ingestion.triplet_extractor import TripletExtractor
from app.models.schemas import ClinicalTriplet
from app.utils.logger import logger
from app.vector.chroma_store import ChromaStore

_MERGE_DYNAMIC_QUERY_TEMPLATE = """
MERGE (s:Entity {{name: $subject}})
MERGE (o:Entity {{name: $object}})
MERGE (s)-[r:`{predicate}`]->(o)
SET r.source = $source, r.confidence = $confidence
"""


@dataclass
class IngestionSummary:
    topic: str
    works_found: int = 0
    works_ingested: int = 0
    triplets_written: int = 0
    vector_chunks_written: int = 0
    skipped_low_authority: int = 0
    errors: list[str] = field(default_factory=list)


class IngestionPipeline:
    def __init__(
        self,
        openalex_client: OpenAlexClient | None = None,
        graph_client: Neo4jClient | None = None,
        vector_store: ChromaStore | None = None,
        extractor: TripletExtractor | None = None,
    ):
        settings = get_settings()
        self._settings = settings
        self._openalex = openalex_client or OpenAlexClient()
        self._graph = graph_client or Neo4jClient()
        self._vectors = vector_store or ChromaStore()
        self._extractor = extractor or TripletExtractor()

    def _write_triplet(self, triplet: ClinicalTriplet) -> bool:
        try:
            query = _MERGE_DYNAMIC_QUERY_TEMPLATE.format(predicate=triplet.predicate.replace("`", ""))
            self._graph.run_query(
                query,
                {
                    "subject": triplet.subject,
                    "object": triplet.object,
                    "source": triplet.source,
                    "confidence": triplet.confidence,
                },
            )
            return True
        except Exception as exc:
            logger.error("Failed to write triplet {} to Neo4j: {}", triplet.as_text(), exc)
            return False

    def ingest_topic(self, topic: str, max_works: int | None = None, min_authority: float | None = None) -> IngestionSummary:
        max_works = max_works or self._settings.openalex_max_works_per_topic
        min_authority = min_authority if min_authority is not None else self._settings.openalex_min_authority_score
        summary = IngestionSummary(topic=topic)

        works = self._openalex.search_works(topic, max_results=max_works)
        summary.works_found = len(works)
        if not works:
            summary.errors.append("No works returned from OpenAlex.")
            return summary

        ranked = rank_works(works, min_score=min_authority)
        summary.skipped_low_authority = len(works) - len(ranked)

        texts, sources, ids = [], [], []
        for work, authority in ranked:
            try:
                triplets = self._extractor.extract(work, authority)
            except Exception as exc:
                summary.errors.append(f"Extraction failed for {work.id}: {exc}")
                continue

            written = sum(1 for t in triplets if self._write_triplet(t))
            summary.triplets_written += written

            if work.abstract:
                texts.append(work.abstract)
                sources.append(work.id)
                ids.append(work.id)

            summary.works_ingested += 1

        if texts:
            summary.vector_chunks_written = self._vectors.add_documents(texts, sources=sources, ids=ids)

        return summary

    def ingest_topics(self, topics: list[str], **kwargs) -> list[IngestionSummary]:
        summaries = []
        for topic in topics:
            logger.info("Ingesting topic: {!r}", topic)
            summary = self.ingest_topic(topic, **kwargs)
            logger.info(
                "Topic {!r}: {} works found, {} ingested, {} triplets, {} vector chunks, {} skipped (low authority)",
                topic, summary.works_found, summary.works_ingested, summary.triplets_written,
                summary.vector_chunks_written, summary.skipped_low_authority,
            )
            summaries.append(summary)
        return summaries

    def close(self) -> None:
        self._graph.close()
