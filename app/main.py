"""
NeuroAgent FastAPI application entry point.

Starts the async application, configures middleware, registers routers,
and manages lifespan (startup/shutdown) for all service connections.

Services initialized at startup (all non-fatal except Qdrant):
  - Qdrant (required — vector search)
  - MongoDB (optional — session + eval persistence)
  - Neo4j (optional — knowledge graph)
  - Mem0 (optional — user memory)
  - Orchestrator (always — wires all agents)
"""

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.dependencies import set_services, get_service_status
from app.services.qdrant_service import QdrantService
from app.services.mongodb_service import MongoDBService
from app.services.neo4j_service import Neo4jService
from app.services.mem0_service import Mem0Service
from app.agents.orchestrator import Orchestrator
from app.utils.logger import setup_logging

logger = logging.getLogger(__name__)

# Service instances (referenced in shutdown)
_qdrant: Optional[QdrantService] = None
_mongodb: Optional[MongoDBService] = None
_neo4j: Optional[Neo4jService] = None
_mem0: Optional[Mem0Service] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage startup and shutdown of all service connections.
    FastAPI calls this once at startup and once at shutdown.
    """
    global _qdrant, _mongodb, _neo4j, _mem0

    # ── Startup ────────────────────────────────────────────────────────────────
    setup_logging()
    settings = get_settings()
    logger.info(
        "NeuroAgent starting up | host=%s port=%d", settings.app_host, settings.app_port
    )

    # Qdrant (required — fail fast if unavailable)
    _qdrant = QdrantService()
    await _qdrant.connect()
    logger.info("Qdrant connected")

    # MongoDB (optional — log warning and continue if unavailable)
    _mongodb = None
    try:
        _mongodb = MongoDBService()
        await _mongodb.connect()
        logger.info("MongoDB connected")
    except Exception as exc:
        logger.warning("MongoDB unavailable — sessions will not be persisted | error=%s", exc)

    # Neo4j (optional)
    _neo4j = None
    try:
        _neo4j = Neo4jService()
        await _neo4j.connect()
        logger.info("Neo4j connected")
    except Exception as exc:
        logger.warning("Neo4j unavailable — graph writes disabled | error=%s", exc)

    # Mem0 (optional)
    _mem0 = Mem0Service()
    await _mem0.connect()
    if _mem0.is_available:
        logger.info("Mem0 initialized")
    else:
        logger.warning("Mem0 unavailable — user memory disabled")

    # Orchestrator (wires all agents with connected services)
    orchestrator = Orchestrator(
        qdrant=_qdrant,
        mongodb=_mongodb,
        neo4j=_neo4j,
        mem0=_mem0,
    )

    # Register all services with the dependency injection module
    set_services(
        qdrant=_qdrant,
        mongodb=_mongodb,
        neo4j=_neo4j,
        mem0=_mem0,
        orchestrator=orchestrator,
    )

    logger.info("NeuroAgent startup complete | all services initialized")
    yield

    # ── Shutdown ───────────────────────────────────────────────────────────────
    logger.info("NeuroAgent shutting down")
    if _qdrant:
        await _qdrant.close()
    if _mongodb:
        await _mongodb.close()
    if _neo4j:
        await _neo4j.close()
    if _mem0:
        await _mem0.close()
    logger.info("All services closed")


def create_app() -> FastAPI:
    """Application factory — creates and configures the FastAPI instance."""
    settings = get_settings()

    app = FastAPI(
        title="NeuroAgent",
        description=(
            "Multi-agent AI research assistant. "
            "Fetches academic papers, summarizes them, detects contradictions, "
            "extracts concepts, synthesizes answers, and evaluates quality with RAGAS."
        ),
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── CORS ───────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers ────────────────────────────────────────────────────────────────
    from app.api.routes.research import router as research_router
    from app.api.routes.evaluations import router as evaluations_router
    from app.api.routes.graph import router as graph_router

    app.include_router(research_router, prefix="/api")
    app.include_router(evaluations_router, prefix="/api")
    app.include_router(graph_router, prefix="/api")

    # ── System endpoints ───────────────────────────────────────────────────────

    @app.get("/health", tags=["System"], summary="Health check")
    async def health_check():
        """
        Health check endpoint.
        Returns status of all service connections.
        """
        status_map = get_service_status()
        all_core_ok = status_map["qdrant"] and status_map["orchestrator"]
        return {
            "status": "ok" if all_core_ok else "degraded",
            "services": status_map,
        }

    @app.get("/", tags=["System"], include_in_schema=False)
    async def root():
        return {
            "service": "NeuroAgent",
            "version": "0.1.0",
            "docs": "/docs",
            "health": "/health",
        }

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=True,
        log_level=settings.log_level.lower(),
    )
