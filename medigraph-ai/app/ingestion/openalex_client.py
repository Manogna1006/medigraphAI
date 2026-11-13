"""
Thin client for the OpenAlex Works API (https://docs.openalex.org).

Design choices:
- Uses OpenAlex's "polite pool" (a `mailto` param) when configured, which
  gets much higher and more stable rate limits than anonymous requests.
- Every network call is retried on transient errors (timeouts, 429, 5xx)
  with exponential backoff, and ultimately fails to an empty list rather
  than raising — a single bad topic shouldn't kill a multi-topic ingestion run.
- Returns plain `OpenAlexWork` Pydantic objects, never raw API JSON, so
  nothing downstream needs to know OpenAlex's field-naming quirks
  (e.g. `cited_by_count`, inverted abstract index).
"""
from __future__ import annotations

import httpx
from pydantic import BaseModel, Field
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.config import get_settings
from app.utils.logger import logger


class OpenAlexWork(BaseModel):
    id: str
    title: str
    abstract: str = ""
    cited_by_count: int = 0
    publication_year: int | None = None
    is_open_access: bool = False
    venue: str | None = None
    concepts: list[str] = Field(default_factory=list)


def _reconstruct_abstract(inverted_index: dict[str, list[int]] | None) -> str:
    """OpenAlex stores abstracts as {word: [positions]} to respect publisher copyright.
    Reconstructing plain text from it is standard practice and documented by OpenAlex itself."""
    if not inverted_index:
        return ""
    position_map: dict[int, str] = {}
    for word, positions in inverted_index.items():
        for pos in positions:
            position_map[pos] = word
    if not position_map:
        return ""
    return " ".join(position_map[i] for i in sorted(position_map))


class OpenAlexClient:
    def __init__(self, base_url: str | None = None, mailto: str | None = None, timeout: float = 15.0):
        settings = get_settings()
        self._base_url = (base_url or settings.openalex_base_url).rstrip("/")
        self._mailto = mailto if mailto is not None else settings.openalex_mailto
        self._timeout = timeout

    @retry(
        retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=15),
        reraise=True,
    )
    def _get(self, params: dict) -> dict:
        if self._mailto:
            params = {**params, "mailto": self._mailto}
        with httpx.Client(timeout=self._timeout) as client:
            response = client.get(f"{self._base_url}/works", params=params)
            if response.status_code == 429 or response.status_code >= 500:
                # Let tenacity retry these — they're typically transient.
                response.raise_for_status()
            response.raise_for_status()
            return response.json()

    def search_works(self, topic: str, max_results: int = 50, min_year: int | None = None) -> list[OpenAlexWork]:
        """
        Search OpenAlex for works matching `topic`, sorted by citation count
        (a simple relevance proxy — the authority scorer refines this further).
        Returns [] on any failure after retries, logged but never raised, so a
        single failed topic doesn't abort a batch ingestion job.
        """
        topic = (topic or "").strip()
        if not topic:
            return []

        filters = [f"title_and_abstract.search:{topic}"]
        if min_year:
            filters.append(f"publication_year:>{min_year - 1}")

        params = {
            "filter": ",".join(filters),
            "sort": "cited_by_count:desc",
            "per_page": min(max_results, 200),
        }

        try:
            payload = self._get(params)
        except (httpx.TransportError, httpx.HTTPStatusError) as exc:
            logger.error("OpenAlex search failed for topic={!r}: {}", topic, exc)
            return []

        works: list[OpenAlexWork] = []
        for result in payload.get("results", [])[:max_results]:
            try:
                works.append(
                    OpenAlexWork(
                        id=result.get("id", ""),
                        title=result.get("title") or "",
                        abstract=_reconstruct_abstract(result.get("abstract_inverted_index")),
                        cited_by_count=result.get("cited_by_count", 0),
                        publication_year=result.get("publication_year"),
                        is_open_access=(result.get("open_access") or {}).get("is_oa", False),
                        venue=((result.get("primary_location") or {}).get("source") or {}).get("display_name"),
                        concepts=[c.get("display_name", "") for c in (result.get("concepts") or [])[:5]],
                    )
                )
            except Exception as exc:
                logger.warning("Skipping malformed OpenAlex result: {}", exc)
        return works
