"""
arXiv API client — async, with rate-limiting and retry.

Searches http://export.arxiv.org/api/query (Atom XML feed).
No API key required. arXiv asks for max 1 request per 3 seconds.

Parsed with feedparser; results returned as Paper Pydantic models.
"""

import asyncio
import re
from typing import Optional

import feedparser
import httpx
from loguru import logger

from app.models.paper import Paper
from app.utils.retry import async_retry_network
from app.utils.text_utils import clean_text, extract_year_from_date

# arXiv politely requests no more than 1 req / 3 seconds
_ARXIV_BASE_URL = "http://export.arxiv.org/api/query"
_RATE_LIMIT_DELAY = 3.0  # seconds between requests

# Words to strip from natural-language queries before passing to arXiv.
# These are filler/question words that produce irrelevant matches.
_FILLER_WORDS = {
    "what", "how", "why", "when", "where", "which", "who", "whose",
    "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "can", "could", "would", "should", "will",
    "a", "an", "the", "and", "or", "of", "in", "on", "at", "to",
    "for", "with", "about", "that", "this", "these", "those",
    "by", "from", "as", "its", "it", "i", "me", "my",
    "please", "tell", "explain", "describe", "list", "give", "show",
    "find", "get", "use", "using", "used", "uses", "review",
    "research", "study", "paper", "papers", "between", "across",
    "some", "any", "all", "most", "more", "very", "just", "also",
}


def _extract_search_terms(query: str, max_terms: int = 8) -> str:
    """
    Extract key technical terms from a natural-language research question.

    Strips question words, articles, and filler words so that arXiv receives
    a focused keyword query rather than a full English sentence.

    Examples:
        "What are the limitations of RAG systems in production?"
            → "limitations RAG systems production"
        "transformer attention mechanism efficiency"
            → "transformer attention mechanism efficiency"  (unchanged)
    """
    cleaned = re.sub(r"[^\w\s-]", " ", query.lower())
    words = cleaned.split()
    terms = [w.strip("-") for w in words if len(w) > 1 and w not in _FILLER_WORDS]
    return " ".join(terms[:max_terms]) if terms else query


def _extract_arxiv_id(entry_id: str) -> str:
    """
    Extract the canonical arXiv ID from a full entry URL.

    Examples:
        "http://arxiv.org/abs/2307.09288v2"  → "2307.09288"
        "http://arxiv.org/abs/cs.LG/0601001" → "cs.LG/0601001"
    """
    # Strip version suffix (v1, v2, …)
    raw = entry_id.split("/abs/")[-1]
    return re.sub(r"v\d+$", "", raw)


def _extract_pdf_url(entry_id: str, links: list) -> Optional[str]:
    """
    Build the PDF URL from feedparser links or fall back to the standard pattern.
    """
    for link in links:
        href = link.get("href", "")
        title = link.get("title", "")
        mime = link.get("type", "")
        if mime == "application/pdf" or title.lower() == "pdf":
            return href
    # Standard arXiv PDF URL pattern
    arxiv_id = _extract_arxiv_id(entry_id)
    return f"https://arxiv.org/pdf/{arxiv_id}"


def _parse_entry(entry) -> Optional[Paper]:
    """
    Convert a single feedparser entry into a Paper model.
    Returns None if the entry is missing required fields.
    """
    try:
        paper_id = _extract_arxiv_id(entry.get("id", ""))
        if not paper_id:
            return None

        title = clean_text(entry.get("title", "").replace("\n", " "))
        abstract = clean_text(entry.get("summary", "").replace("\n", " "))

        if not title or not abstract:
            logger.debug("Skipping arXiv entry — missing title or abstract | id=%s", paper_id)
            return None

        authors = [
            a.get("name", "").strip()
            for a in entry.get("authors", [])
            if a.get("name")
        ]

        year = extract_year_from_date(entry.get("published", ""))

        pdf_url = _extract_pdf_url(entry.get("id", ""), entry.get("links", []))

        # DOI is exposed as entry.arxiv_doi in feedparser (may not exist)
        doi: Optional[str] = getattr(entry, "arxiv_doi", None) or None

        return Paper(
            paper_id=paper_id,
            title=title,
            abstract=abstract,
            authors=authors,
            year=year,
            pdf_url=pdf_url,
            doi=doi,
            source="arxiv",
            external_ids={"arxiv": paper_id},
        )
    except Exception as exc:
        logger.warning("Failed to parse arXiv entry | error=%s", exc)
        return None


class ArxivClient:
    """
    Async client for the arXiv public API.

    Usage:
        async with ArxivClient() as client:
            papers = await client.search("retrieval augmented generation", max_results=10)
    """

    def __init__(self, timeout: float = 30.0) -> None:
        self._timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self) -> "ArxivClient":
        self._client = httpx.AsyncClient(
            timeout=self._timeout,
            follow_redirects=True,  # arXiv API issues 301 redirects
        )
        return self

    async def __aexit__(self, *_) -> None:
        if self._client:
            await self._client.aclose()

    @async_retry_network
    async def search(
        self,
        query: str,
        max_results: int = 10,
        sort_by: str = "relevance",
    ) -> list[Paper]:
        """
        Search arXiv for papers matching `query`.

        Args:
            query: Free-text search query (e.g. "retrieval augmented generation").
            max_results: Maximum number of results to return.
            sort_by: "relevance" | "lastUpdatedDate" | "submittedDate"

        Returns:
            List of Paper models. Empty list on API error or zero results.
        """
        if self._client is None:
            raise RuntimeError("ArxivClient must be used as an async context manager")

        # Strip question/filler words so arXiv gets technical keywords, not full sentences.
        search_terms = _extract_search_terms(query)
        params = {
            "search_query": f"all:{search_terms}",
            "max_results": max_results,
            "sortBy": sort_by,
            "sortOrder": "descending",
        }

        logger.info(
            "arXiv search | query='{}' search_terms='{}' max_results={}",
            query,
            search_terms,
            max_results,
        )

        try:
            response = await self._client.get(_ARXIV_BASE_URL, params=params)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.error("arXiv HTTP error | status={} url={}", exc.response.status_code, exc.request.url)
            return []
        except httpx.RequestError as exc:
            logger.error("arXiv request failed | error={}", exc)
            raise  # Let tenacity retry on network errors

        feed = feedparser.parse(response.text)

        if feed.bozo and feed.bozo_exception:
            logger.warning("arXiv feed parse warning | error={}", feed.bozo_exception)

        entries = feed.get("entries", [])
        papers: list[Paper] = []
        for entry in entries:
            paper = _parse_entry(entry)
            if paper:
                papers.append(paper)

        logger.info(
            "arXiv search complete | query='{}' fetched={} parsed={}",
            query,
            len(entries),
            len(papers),
        )

        # Respect arXiv's rate limit — wait before any subsequent call
        await asyncio.sleep(_RATE_LIMIT_DELAY)

        return papers


# ── Standalone Test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import asyncio

    async def _test() -> None:
        async with ArxivClient() as client:
            papers = await client.search("retrieval augmented generation", max_results=5)
            print(f"\nFetched {len(papers)} papers from arXiv:\n")
            for p in papers:
                print(f"  [{p.year}] {p.title[:80]}")
                print(f"         Authors: {', '.join(p.authors[:2])}")
                print(f"         ID: {p.paper_id} | PDF: {p.pdf_url}")
                print()

    asyncio.run(_test())
