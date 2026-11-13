"""
Extracts text from a clinical PDF record (discharge summary, lab report,
referral letter, etc.), chunks it, and adds it to ChromaDB tagged with a
patient/record identifier so it can be retrieved alongside the general
literature corpus.

This is deliberately a *separate* ingestion path from OpenAlex literature —
a patient's own PDF record is evidence about that patient, not general
medical knowledge, so `source` is tagged distinctly (`patient_record:<id>`)
and callers should scope retrieval per-patient in a multi-tenant deployment
rather than mixing records across patients in one shared collection.
"""
from __future__ import annotations

import re
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.utils.logger import logger
from app.vector.chroma_store import ChromaStore

_WHITESPACE_RUN = re.compile(r"\s+")


def extract_text_from_pdf(path: str | Path) -> str:
    """Returns extracted text, or '' if the PDF can't be read (encrypted,
    corrupted, scanned-image-only with no text layer, etc.) — logged, not raised,
    since one bad record shouldn't halt an ingestion batch."""
    path = Path(path)
    if not path.exists():
        logger.error("PDF not found: {}", path)
        return ""
    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted:
            logger.warning("PDF {} is encrypted and cannot be read without a password.", path)
            return ""
        pages_text = [page.extract_text() or "" for page in reader.pages]
        text = "\n".join(pages_text).strip()
        if not text:
            logger.warning(
                "No extractable text in {} — likely a scanned image PDF with no text layer (OCR not implemented).",
                path,
            )
        return text
    except (PdfReadError, Exception) as exc:  # pypdf raises assorted exceptions on malformed files
        logger.error("Failed to read PDF {}: {}", path, exc)
        return ""


def chunk_text(text: str, max_words: int = 200, overlap_words: int = 30) -> list[str]:
    """Splits text into overlapping word-count chunks. Overlap preserves context
    across a chunk boundary so a fact split across two chunks isn't lost entirely."""
    text = _WHITESPACE_RUN.sub(" ", text).strip()
    if not text:
        return []
    words = text.split(" ")
    if len(words) <= max_words:
        return [text]

    chunks = []
    step = max(1, max_words - overlap_words)
    for start in range(0, len(words), step):
        chunk_words = words[start : start + max_words]
        if chunk_words:
            chunks.append(" ".join(chunk_words))
        if start + max_words >= len(words):
            break
    return chunks


def ingest_pdf_record(
    path: str | Path,
    patient_id: str,
    vector_store: ChromaStore | None = None,
    max_words_per_chunk: int = 200,
) -> int:
    """
    Extracts, chunks, and indexes one clinical PDF. Returns the number of
    chunks successfully written (0 on any failure, never raises).
    """
    vector_store = vector_store or ChromaStore()
    text = extract_text_from_pdf(path)
    if not text:
        return 0

    chunks = chunk_text(text, max_words=max_words_per_chunk)
    if not chunks:
        return 0

    source_tag = f"patient_record:{patient_id}"
    ids = [f"{source_tag}:{Path(path).stem}:{i}" for i in range(len(chunks))]
    written = vector_store.add_documents(chunks, sources=[source_tag] * len(chunks), ids=ids)
    logger.info("Ingested {} chunk(s) from {} for patient_id={}", written, path, patient_id)
    return written
