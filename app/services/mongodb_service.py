"""
MongoDB service — async client using motor.

Handles all application data: paper metadata, research sessions,
RAGAS evaluation scores, and query history.

Supports both local (mongodb://localhost:27017) and Atlas cloud
(mongodb+srv://...) connections. Atlas uses TLS by default via the
+srv URI scheme — no ssl flags needed or wanted.

Usage:
    from app.services.mongodb_service import MongoDBService

    service = MongoDBService()
    await service.connect()
    await service.save_paper(paper)
    await service.close()
"""

import logging
from datetime import datetime
from typing import Any, Optional

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo import ASCENDING, DESCENDING, IndexModel

from app.config import get_settings
from app.models.evaluation import EvaluationResult
from app.models.paper import Paper

logger = logging.getLogger(__name__)


class MongoDBService:
    """
    Async MongoDB client wrapping motor.

    Connection string comes exclusively from settings.mongodb_url —
    never hardcoded. The motor client handles both local and Atlas URIs
    transparently, including TLS for mongodb+srv:// URIs.
    """

    # Collection names
    PAPERS_COLLECTION = "papers"
    SESSIONS_COLLECTION = "sessions"
    EVALUATIONS_COLLECTION = "evaluations"
    QUERIES_COLLECTION = "queries"

    def __init__(self) -> None:
        self._client: Optional[AsyncIOMotorClient] = None
        self._db: Optional[AsyncIOMotorDatabase] = None

    async def connect(self) -> None:
        """
        Open the motor connection and ensure indexes exist.

        Called once at application startup via FastAPI lifespan.
        Uses settings.mongodb_url — works for both local and Atlas URIs.
        No ssl=False or trust flags — Atlas TLS is handled by the URI.
        """
        settings = get_settings()
        logger.info("Connecting to MongoDB | url_prefix=%s", settings.mongodb_url[:30])

        self._client = AsyncIOMotorClient(
            settings.mongodb_url,
            serverSelectionTimeoutMS=10_000,
        )
        self._db = self._client[settings.mongodb_db_name]

        # Verify the connection is live
        await self._client.admin.command("ping")
        logger.info("MongoDB connected | db=%s", settings.mongodb_db_name)

        await self._ensure_indexes()

    async def close(self) -> None:
        """Close the motor connection. Called at application shutdown."""
        if self._client:
            self._client.close()
            logger.info("MongoDB connection closed")

    @property
    def db(self) -> AsyncIOMotorDatabase:
        """Return the database handle, raising if not connected."""
        if self._db is None:
            raise RuntimeError("MongoDBService.connect() has not been called")
        return self._db

    # ── Papers ─────────────────────────────────────────────────────────────────

    async def save_paper(self, paper: Paper) -> None:
        """
        Upsert a paper document by paper_id.
        Safe to call multiple times — won't create duplicates.
        """
        doc = paper.model_dump(mode="json")
        await self.db[self.PAPERS_COLLECTION].update_one(
            {"paper_id": paper.paper_id},
            {"$set": doc},
            upsert=True,
        )
        logger.debug("Paper saved | paper_id=%s", paper.paper_id)

    async def save_papers(self, papers: list[Paper]) -> None:
        """Bulk upsert a list of papers."""
        for paper in papers:
            await self.save_paper(paper)
        logger.info("Saved %d papers to MongoDB", len(papers))

    async def get_paper(self, paper_id: str) -> Optional[dict[str, Any]]:
        """Retrieve a single paper document by paper_id."""
        return await self.db[self.PAPERS_COLLECTION].find_one(
            {"paper_id": paper_id}, {"_id": 0}
        )

    # ── Sessions ───────────────────────────────────────────────────────────────

    async def save_session(self, session_id: str, data: dict[str, Any]) -> None:
        """Create or update a research session document."""
        await self.db[self.SESSIONS_COLLECTION].update_one(
            {"session_id": session_id},
            {"$set": {**data, "updated_at": datetime.utcnow()}},
            upsert=True,
        )
        logger.debug("Session saved | session_id=%s", session_id)

    async def get_session(self, session_id: str) -> Optional[dict[str, Any]]:
        """Retrieve a session document by session_id."""
        return await self.db[self.SESSIONS_COLLECTION].find_one(
            {"session_id": session_id}, {"_id": 0}
        )

    # ── RAGAS Evaluations ──────────────────────────────────────────────────────

    async def save_evaluation(self, result: EvaluationResult) -> None:
        """Persist a RAGAS evaluation result."""
        doc = result.model_dump(mode="json")
        await self.db[self.EVALUATIONS_COLLECTION].insert_one(doc)
        logger.info(
            "Evaluation saved | session_id=%s avg_score=%.3f",
            result.session_id,
            result.average_score,
        )

    async def get_evaluations(
        self,
        limit: int = 50,
        skip: int = 0,
    ) -> list[dict[str, Any]]:
        """
        Retrieve recent RAGAS evaluation results, newest first.

        Args:
            limit: Maximum number of results to return.
            skip: Pagination offset.
        """
        cursor = (
            self.db[self.EVALUATIONS_COLLECTION]
            .find({}, {"_id": 0})
            .sort("evaluated_at", DESCENDING)
            .skip(skip)
            .limit(limit)
        )
        return await cursor.to_list(length=limit)

    # ── Indexes ────────────────────────────────────────────────────────────────

    async def _ensure_indexes(self) -> None:
        """Create indexes if they don't already exist. Idempotent."""
        await self.db[self.PAPERS_COLLECTION].create_indexes([
            IndexModel([("paper_id", ASCENDING)], unique=True),
            IndexModel([("source", ASCENDING)]),
            IndexModel([("year", DESCENDING)]),
        ])
        await self.db[self.SESSIONS_COLLECTION].create_indexes([
            IndexModel([("session_id", ASCENDING)], unique=True),
            IndexModel([("updated_at", DESCENDING)]),
        ])
        await self.db[self.EVALUATIONS_COLLECTION].create_indexes([
            IndexModel([("session_id", ASCENDING)]),
            IndexModel([("evaluated_at", DESCENDING)]),
        ])
        logger.debug("MongoDB indexes verified")
