"""
FastAPI endpoint tests with all services mocked.

Tests verify contract correctness — structure, status codes, validation —
without hitting any live external service (Qdrant, MongoDB, Groq, Neo4j).

Approach:
  - Patch app.main.lifespan with a no-op that injects mock services.
  - Use FastAPI's TestClient (sync) which triggers the lifespan correctly.
  - Each test class corresponds to one route group.
"""

import pytest
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient


# ── Shared mock factory ────────────────────────────────────────────────────────

def _make_mock_orchestrator() -> AsyncMock:
    """Orchestrator that returns a minimal but valid research result."""
    orch = AsyncMock()
    orch.run_research = AsyncMock(return_value={
        "papers": [
            MagicMock(paper_id="2307.09288"),
            MagicMock(paper_id="2310.01714"),
        ],
        "summaries": [MagicMock(), MagicMock()],
        "concepts": ["retrieval", "augmented", "generation", "fine-tuning"],
        "contradiction_report": None,
        "final_synthesis": (
            "RAG systems face several production challenges including "
            "latency, retrieval quality, and hallucination [2307.09288]."
        ),
        "evaluation": None,
        "errors": [],
        "qdrant_context": [],
    })
    return orch


def _make_mock_mongodb() -> AsyncMock:
    db = AsyncMock()
    db.get_evaluations = AsyncMock(return_value=[])
    db.get_session = AsyncMock(return_value=None)
    return db


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_orchestrator():
    return _make_mock_orchestrator()


@pytest.fixture
def mock_mongodb():
    return _make_mock_mongodb()


@pytest.fixture
def client(mock_orchestrator, mock_mongodb):
    """
    TestClient with the FastAPI app and all services replaced by mocks.

    Patches the lifespan context manager so no real service connections
    are attempted during test startup.
    """
    from app.main import create_app
    from app import dependencies

    mock_qdrant = MagicMock()

    @asynccontextmanager
    async def mock_lifespan(app):
        dependencies.set_services(
            qdrant=mock_qdrant,
            mongodb=mock_mongodb,
            neo4j=None,
            mem0=None,
            orchestrator=mock_orchestrator,
        )
        yield

    with patch("app.main.lifespan", mock_lifespan):
        app = create_app()
        with TestClient(app, raise_server_exceptions=True) as test_client:
            yield test_client


# ── /health ────────────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_returns_status_and_services_keys(self, client):
        data = client.get("/health").json()
        assert "status" in data
        assert "services" in data

    def test_services_block_contains_all_components(self, client):
        services = client.get("/health").json()["services"]
        for key in ("qdrant", "mongodb", "neo4j", "mem0", "orchestrator"):
            assert key in services

    def test_status_ok_when_core_services_up(self, client):
        """qdrant and orchestrator are mocked (non-None) → overall status is 'ok'."""
        data = client.get("/health").json()
        assert data["status"] == "ok"

    def test_root_endpoint_returns_service_info(self, client):
        data = client.get("/").json()
        assert data["service"] == "NeuroAgent"
        assert "docs" in data
        assert "health" in data


# ── POST /api/research ─────────────────────────────────────────────────────────

class TestResearchEndpoint:
    def test_returns_200_on_valid_query(self, client):
        resp = client.post("/api/research", json={"query": "What are the limitations of RAG?"})
        assert resp.status_code == 200

    def test_returns_session_id(self, client):
        data = client.post("/api/research", json={"query": "RAG limitations"}).json()
        assert "session_id" in data
        assert len(data["session_id"]) == 36  # UUID format

    def test_returns_full_response_structure(self, client):
        data = client.post("/api/research", json={"query": "RAG limitations"}).json()
        for field in ("session_id", "query", "papers_fetched", "summaries_generated",
                      "concepts", "contradictions", "final_synthesis", "errors"):
            assert field in data, f"Missing field: {field}"

    def test_papers_fetched_count_matches_pipeline_output(self, client, mock_orchestrator):
        data = client.post("/api/research", json={"query": "RAG limitations"}).json()
        # Mock orchestrator returned 2 papers
        assert data["papers_fetched"] == 2

    def test_calls_orchestrator_run_research(self, client, mock_orchestrator):
        client.post("/api/research", json={"query": "fine-tuning vs prompting"})
        mock_orchestrator.run_research.assert_awaited_once()

    def test_passes_query_to_orchestrator(self, client, mock_orchestrator):
        client.post("/api/research", json={"query": "chain of thought prompting"})
        call_kwargs = mock_orchestrator.run_research.call_args.kwargs
        assert call_kwargs["query"] == "chain of thought prompting"

    def test_passes_user_id_when_provided(self, client, mock_orchestrator):
        client.post("/api/research", json={"query": "RAG systems", "user_id": "user-123"})
        call_kwargs = mock_orchestrator.run_research.call_args.kwargs
        assert call_kwargs["user_id"] == "user-123"

    def test_rejects_query_shorter_than_3_chars(self, client):
        resp = client.post("/api/research", json={"query": "ab"})
        assert resp.status_code == 422

    def test_rejects_max_papers_over_30(self, client):
        resp = client.post("/api/research", json={"query": "RAG systems", "max_papers": 31})
        assert resp.status_code == 422

    def test_rejects_max_papers_below_1(self, client):
        resp = client.post("/api/research", json={"query": "RAG systems", "max_papers": 0})
        assert resp.status_code == 422

    def test_missing_query_field_returns_422(self, client):
        resp = client.post("/api/research", json={"max_papers": 10})
        assert resp.status_code == 422


# ── GET /api/evaluations/stats ─────────────────────────────────────────────────

class TestEvaluationsEndpoint:
    def test_stats_returns_200(self, client):
        assert client.get("/api/evaluations/stats").status_code == 200

    def test_stats_returns_correct_keys(self, client):
        data = client.get("/api/evaluations/stats").json()
        for key in ("total_evaluations", "mean_faithfulness",
                    "mean_answer_relevancy", "mean_context_precision", "pass_rate"):
            assert key in data, f"Missing key: {key}"

    def test_stats_zero_totals_when_no_evaluations(self, client):
        data = client.get("/api/evaluations/stats").json()
        assert data["total_evaluations"] == 0
        assert data["mean_faithfulness"] is None

    def test_stats_with_stored_evaluations(self, client, mock_mongodb):
        """When evaluations exist, stats should compute correctly."""
        mock_mongodb.get_evaluations = AsyncMock(return_value=[
            {"faithfulness": 0.8, "answer_relevancy": 0.9, "context_precision": 0.7,
             "session_id": "sess-1", "query": "test"},
            {"faithfulness": 0.6, "answer_relevancy": 0.7, "context_precision": 0.5,
             "session_id": "sess-2", "query": "test2"},
        ])
        data = client.get("/api/evaluations/stats").json()
        assert data["total_evaluations"] == 2
        assert data["mean_faithfulness"] == pytest.approx(0.7, abs=0.01)

    def test_list_evaluations_returns_200(self, client):
        assert client.get("/api/evaluations").status_code == 200

    def test_list_evaluations_returns_pagination_fields(self, client):
        data = client.get("/api/evaluations").json()
        assert "evaluations" in data
        assert "count" in data
        assert "limit" in data
        assert "skip" in data


# ── GET /api/graph/concepts ────────────────────────────────────────────────────

class TestGraphEndpoint:
    def test_graph_concepts_503_when_neo4j_unavailable(self, client):
        """Neo4j is None in our mock setup → should return 503."""
        resp = client.get("/api/graph/concepts")
        assert resp.status_code == 503

    def test_graph_concepts_error_detail_is_informative(self, client):
        data = client.get("/api/graph/concepts").json()
        assert "Neo4j" in data["detail"] or "not available" in data["detail"]
