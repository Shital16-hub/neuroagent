"""
FastAPI dependency injection — global service singletons.

All services are initialized once in the FastAPI lifespan (app/main.py)
and stored here as module-level singletons. Route handlers access them
via FastAPI's Depends() injection.

Pattern:
    # In a route:
    from fastapi import Depends
    from app.dependencies import get_qdrant
    async def my_route(qdrant: QdrantService = Depends(get_qdrant)):
        ...

Services default to None and are set by the lifespan context manager.
Dependency functions raise 503 if the service failed to initialize.
"""

from typing import Optional

from fastapi import HTTPException, status

from app.config import Settings, get_settings
from app.services.qdrant_service import QdrantService
from app.services.mongodb_service import MongoDBService
from app.services.neo4j_service import Neo4jService
from app.services.mem0_service import Mem0Service
from app.agents.orchestrator import Orchestrator


# ── Service singletons (set by lifespan) ──────────────────────────────────────

_qdrant: Optional[QdrantService] = None
_mongodb: Optional[MongoDBService] = None
_neo4j: Optional[Neo4jService] = None
_mem0: Optional[Mem0Service] = None
_orchestrator: Optional[Orchestrator] = None


def set_services(
    qdrant: QdrantService,
    mongodb: Optional[MongoDBService],
    neo4j: Optional[Neo4jService],
    mem0: Optional[Mem0Service],
    orchestrator: Orchestrator,
) -> None:
    """Called once by the FastAPI lifespan to register initialized services."""
    global _qdrant, _mongodb, _neo4j, _mem0, _orchestrator
    _qdrant = qdrant
    _mongodb = mongodb
    _neo4j = neo4j
    _mem0 = mem0
    _orchestrator = orchestrator


# ── FastAPI dependency functions ──────────────────────────────────────────────

def get_config() -> Settings:
    """Inject application settings."""
    return get_settings()


def get_qdrant() -> QdrantService:
    if _qdrant is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Qdrant service is not initialized",
        )
    return _qdrant


def get_mongodb() -> MongoDBService:
    if _mongodb is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="MongoDB service is not initialized",
        )
    return _mongodb


def get_neo4j() -> Optional[Neo4jService]:
    """Returns Neo4j service or None if unavailable (non-fatal)."""
    return _neo4j


def get_mem0() -> Optional[Mem0Service]:
    """Returns Mem0 service or None if unavailable (non-fatal)."""
    return _mem0


def get_orchestrator() -> Orchestrator:
    if _orchestrator is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Orchestrator is not initialized",
        )
    return _orchestrator


def get_service_status() -> dict:
    """Return health status of all services (for /health endpoint)."""
    return {
        "qdrant": _qdrant is not None,
        "mongodb": _mongodb is not None,
        "neo4j": _neo4j is not None,
        "mem0": _mem0 is not None and _mem0.is_available,
        "orchestrator": _orchestrator is not None,
    }
