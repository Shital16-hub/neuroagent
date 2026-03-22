"""
NeuroAgent FastAPI application entry point.

Starts the async application, configures middleware, registers routers,
and manages lifespan (startup/shutdown) for all service connections.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.utils.logger import setup_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage startup and shutdown of all service connections.
    FastAPI calls this once at startup and once at shutdown.
    """
    # ── Startup ────────────────────────────────────────────────────────────────
    setup_logging()
    settings = get_settings()
    logger.info("NeuroAgent starting up | host=%s port=%d", settings.app_host, settings.app_port)

    # Services will be initialized here in subsequent sessions:
    # - Qdrant collection setup
    # - MongoDB index setup
    # - Neo4j schema / constraints setup

    yield

    # ── Shutdown ───────────────────────────────────────────────────────────────
    logger.info("NeuroAgent shutting down")


def create_app() -> FastAPI:
    """Application factory — creates and configures the FastAPI instance."""
    settings = get_settings()

    app = FastAPI(
        title="NeuroAgent",
        description="Multi-agent AI research assistant with RAGAS evaluation",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routers will be registered here in subsequent sessions
    # from app.api.routes import research, sessions, evaluations, graph
    # app.include_router(research.router, prefix="/api")
    # ...

    @app.get("/health", tags=["System"])
    async def health_check():
        """Health check endpoint. Returns status of all service connections."""
        return {"status": "ok", "service": "neuroagent"}

    @app.get("/metrics", tags=["System"])
    async def metrics():
        """System metrics endpoint."""
        return {"status": "metrics endpoint — to be implemented"}

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
