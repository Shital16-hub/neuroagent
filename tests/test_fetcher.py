"""
Tests for the Fetcher Agent and API clients.

These tests use mocking to avoid hitting live APIs.
Integration tests (marked with @pytest.mark.integration) require
Qdrant running locally and network access to arXiv/S2.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from app.models.paper import Paper
from app.models.state import initial_state
from app.agents.fetcher import FetcherAgent, _deduplicate


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


from typing import Optional


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
