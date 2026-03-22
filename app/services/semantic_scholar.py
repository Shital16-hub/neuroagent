"""
Semantic Scholar API client — async, with optional API key and exponential backoff.

Searches https://api.semanticscholar.org/graph/v1/paper/search
No API key required for basic use. Providing a key raises rate limits significantly.

Fields requested: title, abstract, authors, year, citationCount, openAccessPdf, externalIds
Results returned as Paper Pydantic models.
"""

import asyncio
from typing import Any, Optional

import httpx
from loguru import logger
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)
import logging

from app.config import get_settings
from app.models.paper import Paper
from app.utils.text_utils import clean_text, extract_year_from_date

_S2_BASE_URL = "https://api.semanticscholar.org/graph/v1"
_S2_FIELDS = "title,abstract,authors,year,citationCount,openAccessPdf,externalIds"

# Without key: 100 req/5min → sleep 3s between requests to stay safe
# With key: 1 req/sec
_RATE_LIMIT_DELAY_NO_KEY = 3.0
_RATE_LIMIT_DELAY_WITH_KEY = 1.0

_tenacity_logger = logging.getLogger(__name__)


def _parse_paper(data: dict[str, Any]) -> Optional[Paper]:
    """
    Convert a Semantic Scholar paper dict into a Paper model.
    Returns None if required fields are missing or invalid.
    """
    try:
        title = clean_text((data.get("title") or "").replace("\n", " "))
        abstract = clean_text((data.get("abstract") or "").replace("\n", " "))

        if not title or not abstract:
            logger.debug(
                "Skipping S2 paper — missing title or abstract | s2id={}",
                data.get("paperId", "?"),
            )
            return None

        s2_id: str = data.get("paperId", "")
        external_ids: dict[str, str] = {
            k.lower(): v
            for k, v in (data.get("externalIds") or {}).items()
            if v
        }

        # Prefer DOI as canonical identifier; fall back to arXiv ID, then S2 ID
        doi: Optional[str] = external_ids.get("doi")
        arxiv_id: Optional[str] = external_ids.get("arxiv")

        if arxiv_id:
            paper_id = arxiv_id
        elif doi:
            paper_id = doi
        else:
            paper_id = f"s2:{s2_id}"

        authors = [
            a.get("name", "").strip()
            for a in (data.get("authors") or [])
            if a.get("name")
        ]

        year_raw = data.get("year")
        year: Optional[int] = int(year_raw) if year_raw else None

        pdf_url: Optional[str] = None
        open_access = data.get("openAccessPdf")
        if open_access and open_access.get("url"):
            pdf_url = open_access["url"]
        elif arxiv_id:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"

        if s2_id:
            external_ids["s2"] = s2_id

        return Paper(
            paper_id=paper_id,
            title=title,
            abstract=abstract,
            authors=authors,
            year=year,
            pdf_url=pdf_url,
            doi=doi,
            source="semantic_scholar",
            citation_count=data.get("citationCount"),
            external_ids=external_ids,
        )
    except Exception as exc:
        logger.warning(
            "Failed to parse S2 paper | s2id={} error={}",
            data.get("paperId", "?"),
            exc,
        )
        return None


class SemanticScholarClient:
    """
    Async client for the Semantic Scholar Graph API.

    Automatically uses SEMANTIC_SCHOLAR_API_KEY from settings if set,
    which raises rate limits from 100 req/5min to 1 req/sec.

    Usage:
        async with SemanticScholarClient() as client:
            papers = await client.search("retrieval augmented generation", max_results=10)
    """

    def __init__(self, timeout: float = 30.0) -> None:
        settings = get_settings()
        self._api_key: Optional[str] = settings.semantic_scholar_api_key
        self._timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
        self._rate_delay = (
            _RATE_LIMIT_DELAY_WITH_KEY if self._api_key else _RATE_LIMIT_DELAY_NO_KEY
        )

    async def __aenter__(self) -> "SemanticScholarClient":
        headers: dict[str, str] = {}
        if self._api_key:
            headers["x-api-key"] = self._api_key
            logger.debug("SemanticScholar: using authenticated client")
        else:
            logger.debug("SemanticScholar: using unauthenticated client (lower rate limits)")

        self._client = httpx.AsyncClient(
            headers=headers,
            timeout=self._timeout,
        )
        return self

    async def __aexit__(self, *_) -> None:
        if self._client:
            await self._client.aclose()

    async def search(
        self,
        query: str,
        max_results: int = 10,
        exclude_doi_set: Optional[set[str]] = None,
    ) -> list[Paper]:
        """
        Search Semantic Scholar for papers matching `query`.

        Automatically retries on 429 (rate limit) with exponential backoff.

        Args:
            query: Free-text search query.
            max_results: Maximum number of results to return.
            exclude_doi_set: Set of DOIs already fetched (from arXiv) to skip.

        Returns:
            List of Paper models (deduped against exclude_doi_set). Empty on failure.
        """
        if self._client is None:
            raise RuntimeError(
                "SemanticScholarClient must be used as an async context manager"
            )

        exclude = exclude_doi_set or set()

        logger.info(
            "Semantic Scholar search | query='{}' max_results={}", query, max_results
        )

        params = {
            "query": query,
            "limit": min(max_results, 100),
            "fields": _S2_FIELDS,
        }

        raw_data: list[dict[str, Any]] = []

        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(5),
                wait=wait_exponential(multiplier=2, min=2, max=60),
                retry=retry_if_exception_type(httpx.HTTPStatusError),
                before_sleep=before_sleep_log(_tenacity_logger, logging.WARNING),
                reraise=True,
            ):
                with attempt:
                    response = await self._client.get(
                        f"{_S2_BASE_URL}/paper/search", params=params
                    )
                    if response.status_code == 429:
                        logger.warning(
                            "Semantic Scholar rate limited — will retry with backoff"
                        )
                        response.raise_for_status()  # triggers tenacity retry
                    response.raise_for_status()
                    raw_data = response.json().get("data", [])

        except httpx.HTTPStatusError as exc:
            logger.error(
                "Semantic Scholar HTTP error | status={}", exc.response.status_code
            )
            return []
        except httpx.RequestError as exc:
            logger.error("Semantic Scholar request failed | error={}", exc)
            return []

        papers: list[Paper] = []
        for item in raw_data:
            paper = _parse_paper(item)
            if paper is None:
                continue
            # Deduplicate against arXiv results by DOI
            if paper.doi and paper.doi in exclude:
                logger.debug(
                    "Skipping S2 paper — DOI already in arXiv results | doi={}", paper.doi
                )
                continue
            papers.append(paper)

        logger.info(
            "Semantic Scholar search complete | query='{}' fetched={} parsed={} deduped={}",
            query,
            len(raw_data),
            len(papers),
            len(raw_data) - len(papers),
        )

        await asyncio.sleep(self._rate_delay)
        return papers


# ── Standalone Test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import asyncio

    async def _test() -> None:
        async with SemanticScholarClient() as client:
            papers = await client.search("retrieval augmented generation", max_results=5)
            print(f"\nFetched {len(papers)} papers from Semantic Scholar:\n")
            for p in papers:
                print(f"  [{p.year}] {p.title[:80]}")
                print(f"         Citations: {p.citation_count} | ID: {p.paper_id}")
                print()

    asyncio.run(_test())
