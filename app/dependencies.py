"""
FastAPI dependency injection.

All service instances are created once and shared via FastAPI's dependency
injection system. This prevents multiple connections to the same database
and makes testing easy (swap dependencies in tests).
"""

from functools import lru_cache
from app.config import Settings, get_settings


def get_config() -> Settings:
    """Inject the application settings."""
    return get_settings()


# Database and service dependencies will be added here in subsequent sessions:
# async def get_qdrant() -> QdrantService: ...
# async def get_mongodb() -> MongoDBService: ...
# async def get_neo4j() -> Neo4jService: ...
# async def get_mem0() -> Mem0Service: ...
