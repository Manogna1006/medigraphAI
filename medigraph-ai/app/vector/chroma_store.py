"""
ChromaDB-backed vector store.

The "semantic firewall" concept from the reference project lives partly here:
`query` returns raw (chunk, distance) pairs and never decides on its own
whether they're "good enough" — that threshold decision belongs to the
retriever (app/retrieval/hybrid_retriever.py), which is the one place that
combines KG + vector evidence and decides whether to abstain. Keeping the
threshold check out of this class keeps it a dumb, testable data layer.
"""
from __future__ import annotations

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import get_settings
from app.models.schemas import VectorChunk
from app.utils.logger import logger


class ChromaStore:
    def __init__(self, persist_dir: str | None = None, collection_name: str | None = None):
        settings = get_settings()
        self._persist_dir = persist_dir or settings.chroma_persist_dir
        self._collection_name = collection_name or settings.chroma_collection_name
        self._client = None
        self._collection = None

    def _get_collection(self):
        if self._collection is None:
            self._client = chromadb.PersistentClient(
                path=self._persist_dir,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def add_documents(self, texts: list[str], sources: list[str] | None = None, ids: list[str] | None = None) -> int:
        if not texts:
            return 0
        sources = sources or [None] * len(texts)
        if len(sources) != len(texts):
            raise ValueError("sources must be the same length as texts")
        ids = ids or [f"doc-{i}-{abs(hash(t)) % 10**8}" for i, t in enumerate(texts)]
        try:
            collection = self._get_collection()
            collection.add(
                documents=texts,
                metadatas=[{"source": s or "unknown"} for s in sources],
                ids=ids,
            )
            return len(texts)
        except Exception as exc:
            logger.error("Failed to add documents to Chroma: {}", exc)
            return 0

    def query(self, query_text: str, top_k: int = 5) -> list[VectorChunk]:
        """
        Returns up to `top_k` chunks sorted by relevance. Returns [] (never raises)
        on any backend failure or empty collection — the retriever treats that the
        same as "no evidence found," which is the safe direction to fail in.
        """
        if not query_text or not query_text.strip():
            return []
        try:
            collection = self._get_collection()
            if collection.count() == 0:
                logger.warning("Chroma collection '{}' is empty; nothing to retrieve.", self._collection_name)
                return []
            result = collection.query(query_texts=[query_text], n_results=min(top_k, collection.count()))
        except Exception as exc:
            logger.error("Chroma query failed: {}", exc)
            return []

        chunks: list[VectorChunk] = []
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        dists = (result.get("distances") or [[]])[0]
        for doc, meta, dist in zip(docs, metas, dists):
            chunks.append(VectorChunk(text=doc, source=(meta or {}).get("source"), distance=float(dist)))
        return chunks
