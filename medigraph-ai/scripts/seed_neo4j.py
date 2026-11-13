"""
Seed Neo4j with the sample clinical knowledge graph.

Usage:
    python -m scripts.seed_neo4j
    python -m scripts.seed_neo4j --wipe   # drop existing Entity nodes first
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.graph.neo4j_client import Neo4jClient
from app.utils.logger import logger

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "sample_clinical_triplets.json"

_MERGE_QUERY = """
MERGE (s:Entity {name: $subject})
MERGE (o:Entity {name: $object})
MERGE (s)-[r:REL {type: $predicate}]->(o)
SET r.source = $source, r.confidence = $confidence
"""
# Note: relationship *type* in Neo4j must be static, so we also set a generic
# REL type carrying `predicate` as a property, AND create a dynamically-typed
# relationship via APOC-free raw Cypher below for readability in Neo4j Browser.
_MERGE_DYNAMIC_QUERY_TEMPLATE = """
MERGE (s:Entity {{name: $subject}})
MERGE (o:Entity {{name: $object}})
MERGE (s)-[r:`{predicate}`]->(o)
SET r.source = $source, r.confidence = $confidence
"""


def load_triplets(path: Path = DATA_PATH) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Seed data not found at {path}")
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError("Seed data must be a non-empty JSON array")
    return data


def wipe_graph(client: Neo4jClient) -> None:
    logger.warning("Wiping all :Entity nodes and relationships")
    client.run_query("MATCH (n:Entity) DETACH DELETE n")


def seed(client: Neo4jClient, triplets: list[dict]) -> int:
    inserted = 0
    for t in triplets:
        try:
            query = _MERGE_DYNAMIC_QUERY_TEMPLATE.format(predicate=t["predicate"].replace("`", ""))
            client.run_query(
                query,
                {
                    "subject": t["subject"],
                    "object": t["object"],
                    "source": t.get("source"),
                    "confidence": t.get("confidence", 1.0),
                },
            )
            inserted += 1
        except Exception as exc:
            logger.error("Failed to insert triplet {}: {}", t, exc)
    return inserted


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed Neo4j with sample clinical triplets")
    parser.add_argument("--wipe", action="store_true", help="Delete existing Entity nodes first")
    parser.add_argument("--data", type=Path, default=DATA_PATH, help="Path to a triplets JSON file")
    args = parser.parse_args()

    client = Neo4jClient()
    if not client.verify_connectivity():
        logger.error("Cannot reach Neo4j. Check NEO4J_URI / credentials in .env")
        return 1

    if args.wipe:
        wipe_graph(client)

    triplets = load_triplets(args.data)
    inserted = seed(client, triplets)
    logger.info("Seeded {} / {} triplets", inserted, len(triplets))
    client.close()
    return 0 if inserted == len(triplets) else 1


if __name__ == "__main__":
    sys.exit(main())
