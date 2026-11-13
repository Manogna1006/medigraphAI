"""
Usage:
    python -m scripts.ingest_openalex --topics "bacterial pneumonia" "type 2 diabetes"
    python -m scripts.ingest_openalex --topics-file data/topics.json --max-works 30
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.ingestion.pipeline import IngestionPipeline
from app.utils.logger import logger


def load_topics_file(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not all(isinstance(t, str) for t in data):
        raise ValueError(f"{path} must be a JSON array of topic strings")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest OpenAlex literature into the MediGraph AI knowledge graph")
    parser.add_argument("--topics", nargs="+", help="One or more topic strings, e.g. 'bacterial pneumonia'")
    parser.add_argument("--topics-file", type=Path, help="Path to a JSON array of topic strings")
    parser.add_argument("--max-works", type=int, default=None, help="Max works to fetch per topic (default from .env)")
    parser.add_argument("--min-authority", type=float, default=None, help="Min authority score 0-1 (default from .env)")
    args = parser.parse_args()

    if not args.topics and not args.topics_file:
        parser.error("Provide --topics or --topics-file")

    topics = list(args.topics or [])
    if args.topics_file:
        topics.extend(load_topics_file(args.topics_file))
    if not topics:
        parser.error("No topics to ingest.")

    pipeline = IngestionPipeline()
    try:
        summaries = pipeline.ingest_topics(topics, max_works=args.max_works, min_authority=args.min_authority)
    finally:
        pipeline.close()

    total_triplets = sum(s.triplets_written for s in summaries)
    total_chunks = sum(s.vector_chunks_written for s in summaries)
    logger.info(
        "Ingestion complete: {} triplets, {} vector chunks written across {} topics",
        total_triplets, total_chunks, len(topics),
    )

    any_errors = any(s.errors for s in summaries)
    if any_errors:
        logger.warning("Some topics had errors — see log above for details.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
