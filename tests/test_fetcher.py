"""
Tests for the Fetcher Agent and API clients.

These tests use mocking to avoid hitting live APIs.
Integration tests (marked with @pytest.mark.integration) require
Qdrant running locally and network access to arXiv/S2.
"""

import pytest
import httpx
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

from app.models.paper import Paper
from app.models.state import initial_state
from app.agents.fetcher import FetcherAgent, _deduplicate
from app.services.arxiv_client import _extract_search_terms


def _make_paper(
    paper_id: str = "test-001",
    title: str = "Test Paper",
    doi: Optional[str] = None,
    arxiv_id: Optional[str] = None,
    source: str = "arxiv",
) -> Paper:
    external_ids = {}
    if arxiv_id:
        external_ids["arxiv"] = arxiv_id
    return Paper(
        paper_id=paper_id,
        title=title,
        abstract="This is a test abstract for the paper.",
        authors=["Author A", "Author B"],
        year=2024,
        source=source,
        doi=doi,
        external_ids=external_ids,
    )


class TestDeduplication:
    def test_no_duplicates_unchanged(self):
        papers = [
            _make_paper("p1", doi="10.1/a"),
            _make_paper("p2", doi="10.1/b"),
            _make_paper("p3", doi="10.1/c"),
        ]
        result = _deduplicate(papers)
        assert len(result) == 3

    def test_dedup_by_doi(self):
        papers = [
            _make_paper("p1", doi="10.1/same", source="arxiv"),
            _make_paper("p2", doi="10.1/same", source="semantic_scholar"),
        ]
        result = _deduplicate(papers)
        assert len(result) == 1
        # First one wins
        assert result[0].paper_id == "p1"

    def test_dedup_by_arxiv_id(self):
        papers = [
            _make_paper("2307.09288", arxiv_id="2307.09288", source="arxiv"),
            _make_paper("s2:abc123", arxiv_id="2307.09288", source="semantic_scholar"),
        ]
        result = _deduplicate(papers)
        assert len(result) == 1

    def test_dedup_by_paper_id(self):
        papers = [
            _make_paper("same-id"),
            _make_paper("same-id"),
        ]
        result = _deduplicate(papers)
        assert len(result) == 1

    def test_empty_list(self):
        assert _deduplicate([]) == []

    def test_single_paper_unchanged(self):
        papers = [_make_paper("solo")]
        result = _deduplicate(papers)
        assert len(result) == 1


class TestFetcherAgent:
    def _make_agent(self):
        qdrant = AsyncMock()
        qdrant.upsert_papers = AsyncMock()
        mongodb = AsyncMock()
        mongodb.save_papers = AsyncMock()
        return FetcherAgent(qdrant=qdrant, mongodb=mongodb), qdrant, mongodb

    @pytest.mark.asyncio
    async def test_run_returns_papers_on_success(self):
        agent, qdrant, mongodb = self._make_agent()

        mock_papers = [
            _make_paper("arxiv-001", doi="10.1/a"),
            _make_paper("arxiv-002", doi="10.1/b"),
        ]

        with patch.object(agent, "_fetch_parallel", new=AsyncMock(return_value=(mock_papers, []))):
            state = initial_state(query="RAG", session_id="sess-test")
            result = await agent.run(state)

        assert len(result["papers"]) == 2
        assert result["errors"] == []
        qdrant.upsert_papers.assert_awaited_once()
        mongodb.save_papers.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_run_graceful_on_both_api_failure(self):
        agent, qdrant, mongodb = self._make_agent()

        with patch.object(
            agent, "_fetch_parallel", new=AsyncMock(side_effect=RuntimeError("Both APIs failed"))
        ):
            state = initial_state(query="RAG", session_id="sess-test")
            result = await agent.run(state)

        assert result["papers"] == []
        assert len(result["errors"]) == 1
        assert "Both APIs failed" in result["errors"][0]
        qdrant.upsert_papers.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_run_continues_if_qdrant_fails(self):
        agent, qdrant, mongodb = self._make_agent()

        mock_papers = [_make_paper("p1", doi="10.1/a")]
        qdrant.upsert_papers = AsyncMock(side_effect=Exception("Qdrant down"))

        with patch.object(agent, "_fetch_parallel", new=AsyncMock(return_value=(mock_papers, []))):
            state = initial_state(query="RAG", session_id="sess-test")
            result = await agent.run(state)

        # Papers should still be in state despite Qdrant failure
        assert len(result["papers"]) == 1
        assert any("Qdrant" in e for e in result["errors"])

    @pytest.mark.asyncio
    async def test_run_continues_if_mongodb_fails(self):
        agent, qdrant, mongodb = self._make_agent()

        mock_papers = [_make_paper("p1", doi="10.1/a")]
        mongodb.save_papers = AsyncMock(side_effect=Exception("MongoDB down"))

        with patch.object(agent, "_fetch_parallel", new=AsyncMock(return_value=(mock_papers, []))):
            state = initial_state(query="RAG", session_id="sess-test")
            result = await agent.run(state)

        assert len(result["papers"]) == 1
        assert any("MongoDB" in e for e in result["errors"])

    @pytest.mark.asyncio
    async def test_run_adds_warning_if_no_papers_found(self):
        agent, _, _ = self._make_agent()

        with patch.object(agent, "_fetch_parallel", new=AsyncMock(return_value=([], []))):
            state = initial_state(query="very obscure query xyz123", session_id="sess-test")
            result = await agent.run(state)

        assert result["papers"] == []
        assert len(result["errors"]) == 1
        assert "No papers found" in result["errors"][0]

    @pytest.mark.asyncio
    async def test_s2_429_handled_gracefully(self):
        """
        When Semantic Scholar returns 429 (rate limit), the fetcher must still
        return the arXiv papers and not raise — S2 failure is non-fatal.
        """
        agent, qdrant, mongodb = self._make_agent()
        arxiv_papers = [_make_paper("arxiv-001", doi="10.1/a")]

        mock_request = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 429

        # Patch the S2 client inside _fetch_parallel to raise 429
        mock_arxiv_instance = AsyncMock()
        mock_arxiv_instance.search = AsyncMock(return_value=arxiv_papers)
        mock_s2_instance = AsyncMock()
        mock_s2_instance.search = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "429 Too Many Requests",
                request=mock_request,
                response=mock_response,
            )
        )

        with (
            patch("app.agents.fetcher.ArxivClient") as MockArxiv,
            patch("app.agents.fetcher.SemanticScholarClient") as MockS2,
        ):
            MockArxiv.return_value.__aenter__ = AsyncMock(return_value=mock_arxiv_instance)
            MockArxiv.return_value.__aexit__ = AsyncMock(return_value=None)
            MockS2.return_value.__aenter__ = AsyncMock(return_value=mock_s2_instance)
            MockS2.return_value.__aexit__ = AsyncMock(return_value=None)

            state = initial_state(query="RAG limitations", session_id="sess-429-test")
            result = await agent.run(state)

        # arXiv papers are returned despite S2 failure
        assert len(result["papers"]) == 1
        assert result["papers"][0].paper_id == "arxiv-001"
        # S2 429 is a warning, not a fatal error — no pipeline errors added
        assert result["errors"] == []


# ── Query extraction ───────────────────────────────────────────────────────────

class TestQueryExtraction:
    """Tests for the search term extractor that cleans natural-language queries."""

    def test_rag_acronym_expanded(self):
        terms = _extract_search_terms("What are the limitations of RAG systems?")
        assert "retrieval" in terms.lower()
        assert "RAG" not in terms  # expanded, not kept as acronym

    def test_llm_acronym_expanded(self):
        terms = _extract_search_terms("How do LLMs perform on reasoning tasks?")
        assert "language" in terms.lower()

    def test_filler_words_removed(self):
        terms = _extract_search_terms("What are the limitations of fine-tuning?")
        for filler in ("what", "are", "the", "of"):
            assert filler not in terms.split()

    def test_technical_terms_preserved(self):
        terms = _extract_search_terms("transformer attention mechanism efficiency")
        assert "transformer" in terms
        assert "attention" in terms
        assert "mechanism" in terms

    def test_acronyms_are_fully_expanded(self):
        """RLHF and LLM should both be expanded to full phrases."""
        terms = _extract_search_terms("RLHF and LLM research")
        assert "reinforcement" in terms.lower()
        assert "language" in terms.lower()

    def test_empty_after_stripping_falls_back_to_original(self):
        """If all words are filler words, fall back to the original query."""
        result = _extract_search_terms("what is the")
        assert len(result) > 0
