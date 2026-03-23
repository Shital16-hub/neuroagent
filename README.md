# NeuroAgent — Multi-Agent AI Research Assistant

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2-orange.svg)](https://github.com/langchain-ai/langgraph)
[![RAGAS](https://img.shields.io/badge/RAGAS-evaluated-purple.svg)](https://docs.ragas.io)
[![Tests](https://img.shields.io/badge/tests-76%20passing-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **A production-grade, multi-agent AI system** that autonomously fetches academic papers from arXiv and Semantic Scholar, summarizes them, detects contradictions, builds a knowledge graph, and evaluates its own output quality using RAGAS — all orchestrated by LangGraph.

---

## What It Does

You type a research question. NeuroAgent:

1. **Fetches** relevant papers from arXiv + Semantic Scholar in parallel
2. **Summarizes** each paper (key claims, methodology, findings, limitations)
3. **Detects contradictions** between papers using an LLM reasoning pass
4. **Extracts concepts** and stores them in a Neo4j knowledge graph
5. **Synthesizes** a research narrative with inline citations
6. **Evaluates** its own output quality using RAGAS (faithfulness, relevancy, precision)
7. **Persists** everything to MongoDB + Qdrant for semantic search across sessions

---

## Architecture

```mermaid
graph TD
    User["User Query"] --> API["FastAPI\n/api/research"]
    API --> Orch["LangGraph\nOrchestrator"]

    Orch --> Fetcher["FetcherAgent\narXiv + Semantic Scholar\n(parallel)"]
    Fetcher -->|papers| Summarizer["SummarizerAgent\nGroq LLM"]

    Summarizer -->|summaries| Contradiction["ContradictionDetector\nGroq reasoning model"]
    Summarizer -->|summaries| Concepts["ConceptExtractor\nGroq LLM"]

    Contradiction -->|report| Synthesis["SynthesisAgent\nGroq LLM + Qdrant RAG"]
    Concepts -->|concepts| Synthesis

    Synthesis -->|final answer| Evaluator["EvaluatorAgent\nRAGAS scores"]
    Evaluator -->|evaluation| State["AgentState\nfinal response"]

    State --> MongoDB[("MongoDB\npaper + eval storage")]
    State --> Qdrant[("Qdrant\nvector search")]
    Concepts --> Neo4j[("Neo4j\nknowledge graph")]

    API --> Frontend["Gradio UI\n4-tab interface"]
```

**Key design choice:** Contradiction detection and concept extraction run as a **parallel fan-out** in LangGraph — both start immediately after summarization, cutting wall-clock time on multi-paper queries.

---

## Key Features

| Feature | Details |
|---------|---------|
| **Multi-source fetching** | arXiv + Semantic Scholar with deduplication by DOI/arXiv ID |
| **Smart query parsing** | Acronym expansion (RAG → "retrieval augmented generation"), filler-word removal, `abs:` field targeting |
| **Contradiction detection** | LLM-driven comparison with confidence scoring (0–1), conflict type classification, graceful JSON repair |
| **Knowledge graph** | Neo4j stores concept co-occurrence across sessions; visualized in Gradio with Plotly |
| **RAGAS evaluation** | Self-evaluated faithfulness, answer relevancy, context precision per query |
| **Semantic memory** | Qdrant filters search to current-session papers (prevents index pollution from old runs) |
| **Mem0 integration** | Optional long-term user memory across research sessions |
| **Graceful degradation** | Each service (Qdrant, MongoDB, Neo4j, Mem0) fails independently — core pipeline always returns |

---

## RAGAS Evaluation Results

Evaluated on real pipeline runs (not mocked):

| Query | Faithfulness | Answer Relevancy | Context Precision | Average | Result |
|-------|-------------|-----------------|------------------|---------|--------|
| "What are the trade-offs between fine-tuning and prompting LLMs?" | 0.875 | 0.875 | 0.875 | **0.875** | ✅ PASS |
| "How does chain-of-thought prompting improve reasoning?" | 0.083 | 0.083 | 0.083 | **0.083** | ⚠️ Review |

> **Note:** Low scores on CoT query reflect HuggingFace embedding unavailability in that session (RAGAS fell back to faithfulness scores for all metrics). The fine-tuning query ran with full RAGAS evaluation enabled and represents the system's quality ceiling.

**Pass threshold:** Average score ≥ 0.70 (configurable via `RAGAS_PASS_THRESHOLD`)

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Orchestration** | LangGraph 0.2 | StateGraph with parallel fan-out nodes |
| **LLM** | Groq API (Llama 3.3 70B) | Summarization, synthesis, contradiction detection |
| **Reasoning** | Groq (DeepSeek R1 / Llama 3.3 70B) | Contradiction analysis with structured JSON output |
| **Vector DB** | Qdrant (Cloud) | Semantic similarity search, session-filtered RAG |
| **Document DB** | MongoDB Atlas | Papers, evaluations, session persistence |
| **Graph DB** | Neo4j AuraDB | Concept knowledge graph with co-occurrence edges |
| **Evaluation** | RAGAS | Faithfulness, answer relevancy, context precision |
| **Memory** | Mem0 | Cross-session user memory (optional) |
| **API** | FastAPI 0.115 | Async REST API with Pydantic v2 validation |
| **Frontend** | Gradio 5.x | 4-tab UI: Research, Evaluations, Knowledge Graph, System |
| **Paper sources** | arXiv API + Semantic Scholar API | Federated academic paper fetching |
| **Embeddings** | Sentence Transformers (all-MiniLM-L6-v2) | Paper vectorization for Qdrant |
| **Deployment** | Railway (backend) + Hugging Face Spaces (frontend) | Zero-downtime cloud hosting |

---

## Quick Start

### Prerequisites

- Python 3.11+
- Accounts with API keys for: Groq, Qdrant Cloud, MongoDB Atlas, Neo4j AuraDB

### 1. Clone and install

```bash
git clone https://github.com/your-username/neuroagent.git
cd neuroagent
pip install -e ".[dev]"
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:

```env
# LLM
GROQ_API_KEY=your_groq_api_key

# Vector DB
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your_qdrant_key

# Document DB
MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net/neuroagent

# Graph DB
NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password

# Memory (optional)
MEM0_API_KEY=your_mem0_key
```

### 3. Run the backend

```bash
uvicorn app.main:app --reload --port 8000
```

### 4. Run the frontend

```bash
python frontend/app.py
```

Open `http://localhost:7860` in your browser.

### 5. Run with Docker Compose

```bash
docker-compose up
```

Starts Qdrant locally + backend + frontend.

### 6. Run tests

```bash
pytest tests/ -v
# 76 passed
```

---

## API Reference

### POST `/api/research`

Run the full research pipeline.

```json
{
  "query": "What are the limitations of RAG systems in production?",
  "max_papers": 10,
  "user_id": "optional-user-id"
}
```

**Response:**
```json
{
  "session_id": "uuid",
  "query": "...",
  "papers_fetched": 10,
  "summaries_generated": 8,
  "concepts": ["retrieval", "augmented generation", "latency", "..."],
  "contradictions": [],
  "final_synthesis": "RAG systems face several challenges...",
  "evaluation": {
    "faithfulness": 0.875,
    "answer_relevancy": 0.875,
    "context_precision": 0.875,
    "average_score": 0.875,
    "passed_quality_threshold": true
  },
  "errors": []
}
```

### GET `/api/evaluations/stats`

Aggregate RAGAS statistics across all stored evaluations.

### GET `/api/graph/concepts`

Returns nodes and edges for the knowledge graph (requires Neo4j).

### GET `/health`

Service health check for all dependencies.

---

## Deployment

### Railway (Backend)

The repo includes `railway.toml` and `Dockerfile`. Railway auto-detects and deploys:

```bash
railway up
```

Set all environment variables in the Railway dashboard. The backend binds to `$PORT` automatically.

### Hugging Face Spaces (Frontend)

The `frontend/` directory is ready for HF Spaces deployment:

1. Create a new Space (Gradio SDK)
2. Push `frontend/app.py` and `frontend/requirements.txt`
3. Set `NEUROAGENT_API_URL` secret to your Railway URL

---

## Project Structure

```
neuroagent/
├── app/
│   ├── agents/           # LangGraph agent nodes
│   │   ├── fetcher.py        # arXiv + S2 parallel fetch + dedup
│   │   ├── summarizer.py     # Per-paper LLM summarization
│   │   ├── contradiction.py  # Cross-paper contradiction detection
│   │   ├── concept_extractor.py  # Concept/keyword extraction
│   │   ├── synthesis.py      # Final narrative synthesis
│   │   └── evaluator.py      # RAGAS quality evaluation
│   ├── api/routes/       # FastAPI route handlers
│   ├── models/           # Pydantic v2 data models
│   ├── services/         # External service clients
│   │   ├── arxiv_client.py
│   │   ├── semantic_scholar.py
│   │   ├── qdrant_service.py
│   │   ├── mongodb_service.py
│   │   ├── neo4j_service.py
│   │   └── llm_factory.py
│   ├── orchestrator.py   # LangGraph StateGraph definition
│   └── main.py           # FastAPI app + lifespan
├── frontend/
│   └── app.py            # Gradio UI (4 tabs)
├── tests/
│   ├── test_fetcher.py   # FetcherAgent + query extraction (25 tests)
│   ├── test_contradiction.py  # ContradictionDetector (20 tests)
│   └── test_api.py       # FastAPI endpoints (31 tests)
├── docker-compose.yml
├── Dockerfile
└── railway.toml
```

---

## Engineering Decisions

### Why LangGraph instead of a simple sequential pipeline?

LangGraph's `StateGraph` enables **true parallel execution**: contradiction detection and concept extraction both run immediately after summarization completes, sharing the same summaries list. A sequential pipeline would add 5–15 seconds of unnecessary latency on every request. LangGraph also makes the execution graph inspectable and testable as individual nodes.

### Why RAGAS for self-evaluation?

RAGAS provides three complementary metrics that catch different failure modes:
- **Faithfulness** catches hallucinations (synthesis claims not supported by retrieved papers)
- **Answer relevancy** catches topic drift (synthesis answers a different question)
- **Context precision** catches retrieval noise (irrelevant papers pulled into context)

Running RAGAS in a `ThreadPoolExecutor` avoids the `asyncio.run()` conflict that RAGAS's internal event loop creates inside an already-running async context.

### Why session-filtered Qdrant search?

Without filtering, Qdrant's semantic search returns the globally most similar papers across all sessions. If a previous query indexed astrophysics papers (from a broken run), those papers contaminate the synthesis context for every subsequent query. Filtering to `paper_id IN current_session_papers` prevents cross-session pollution entirely.

### Why `@computed_field` for Pydantic v2?

Pydantic v2 excludes plain `@property` from `model_dump()` output. Since `average_score` and `passed_quality_threshold` are derived values that need to persist in MongoDB documents, they must use `@computed_field @property`. Without this, MongoDB stores `0.0` for both fields even when correctly computed in memory.

---

## What I Learned

Building NeuroAgent as a full-stack AI system surfaced engineering challenges that tutorials skip:

- **Multi-agent state management**: Designing the `AgentState` TypedDict so every node can read from and write to shared state without race conditions requires thinking about what each node needs as input vs. what it produces as output.

- **LLM output reliability**: LLMs return markdown-fenced JSON, trailing commas, `<think>` tags, and prose explanations instead of clean JSON. The `_extract_json()` pipeline (strip think tags → try direct parse → regex extract → fix trailing commas) handles the full spectrum of real model outputs gracefully.

- **Async in a sync world**: RAGAS runs `asyncio.run()` internally, which crashes inside FastAPI's already-running event loop. The fix — `loop.run_in_executor()` with a `ThreadPoolExecutor` — is one line but requires understanding Python's event loop model.

- **Test isolation for async FastAPI**: The `lifespan` context manager pattern means standard `TestClient` will attempt real service connections unless you patch `app.main.lifespan` itself. Patching the lifespan with a mock that calls `dependencies.set_services()` gives full isolation without modifying application code.

- **Vector database pollution**: Semantic search quality degrades silently when the index contains irrelevant documents from past sessions. The solution isn't to clear the index — it's to filter queries to the current session's paper IDs, preserving the index for future cross-session retrieval while isolating each pipeline run.

---

## License

MIT — see [LICENSE](LICENSE).

---

*Built as a portfolio project demonstrating senior-level GenAI engineering: multi-agent orchestration, RAG, knowledge graphs, self-evaluation, and production deployment patterns.*
