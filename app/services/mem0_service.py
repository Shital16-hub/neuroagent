"""
Mem0 service — persistent user research memory (self-hosted).

Stores and retrieves per-user research context across sessions using
Mem0's open-source Memory class (no API key required).

Configured with:
  - LLM:          Groq (llama-3.1-8b-instant) — for memory extraction
  - Embedder:     HuggingFace (BAAI/bge-small-en-v1.5) — local, no cost
  - Vector store: Qdrant — same instance used by the paper store,
                  but in a separate 'mem0_memories' collection

All mem0 operations are synchronous — they run in a thread pool executor
to avoid blocking the async event loop.

If Groq is not configured or initialization fails, all methods silently
no-op and return empty strings. The pipeline never crashes due to missing
memory context.
"""

import asyncio
from typing import Optional

from loguru import logger

from app.config import get_settings


_MEM0_COLLECTION = "mem0_memories"
_EXPECTED_DIM = 384  # bge-small-en-v1.5 / all-MiniLM-L6-v2


def _reset_mem0_collection_if_needed(qdrant_url: str, qdrant_api_key) -> None:
    """
    Delete the mem0_memories collection if it has the wrong vector dimension.

    Mem0 creates the collection on first use. If the environment previously
    used a different embedding model (e.g., OpenAI's 1536-dim ada-002), the
    stale collection must be deleted so Mem0 recreates it with 384 dims.
    Safe to call on every startup — no-op if the collection has the right dim
    or doesn't exist.
    """
    try:
        from qdrant_client import QdrantClient

        kwargs = {"url": qdrant_url}
        if qdrant_api_key:
            kwargs["api_key"] = qdrant_api_key

        client = QdrantClient(**kwargs)
        try:
            info = client.get_collection(_MEM0_COLLECTION)
            existing_dim = info.config.params.vectors.size
            if existing_dim != _EXPECTED_DIM:
                logger.warning(
                    "Mem0: mem0_memories has dim={} (expected {}), deleting stale collection",
                    existing_dim,
                    _EXPECTED_DIM,
                )
                client.delete_collection(_MEM0_COLLECTION)
        except Exception:
            # Collection doesn't exist or get_collection failed — both fine
            pass
    except Exception as exc:
        logger.warning("Mem0: collection reset check failed (non-fatal) | error={}", exc)


class Mem0Service:
    """
    Async wrapper around the synchronous mem0 Memory client.

    Usage (application startup):
        service = Mem0Service()
        await service.connect()
        await service.add_memory(user_id="user123", content="User is studying RAG systems")
        context = await service.get_memory(user_id="user123", query="RAG limitations")
        await service.close()
    """

    def __init__(self) -> None:
        self._memory = None
        self._available = False

    async def connect(self) -> None:
        """
        Initialize Mem0 with Groq LLM + Qdrant vector store.

        Non-fatal — logs a warning if setup fails (e.g., Groq key missing,
        Qdrant unavailable) and leaves service in unavailable state.
        """
        settings = get_settings()

        if not settings.has_groq:
            logger.warning("Mem0Service: Groq API key not set — memory disabled")
            return

        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._init_memory, settings)
            self._available = True
            logger.info("Mem0Service initialized | vector_store=qdrant collection=mem0_memories")
        except Exception as exc:
            logger.warning("Mem0Service init failed — memory disabled | error={}", exc)
            self._available = False

    def _init_memory(self, settings) -> None:
        """Initialize Memory synchronously (runs in thread pool)."""
        from mem0 import Memory

        # HuggingFace bge-small-en-v1.5 produces 384-dim vectors.
        # If the mem0_memories collection was previously created with a different
        # dimension (e.g., 1536 from OpenAI), delete it so Mem0 can recreate it.
        _reset_mem0_collection_if_needed(settings.qdrant_url, settings.qdrant_api_key)

        vector_store_config: dict = {
            "collection_name": "mem0_memories",
            "url": settings.qdrant_url,
        }
        if settings.qdrant_api_key:
            vector_store_config["api_key"] = settings.qdrant_api_key

        config = {
            "llm": {
                "provider": "groq",
                "config": {
                    "model": settings.default_llm_model,
                    "api_key": settings.groq_api_key,
                    "temperature": 0.1,
                    "max_tokens": 1024,
                },
            },
            "embedder": {
                "provider": "huggingface",
                "config": {
                    "model": settings.embedding_model,
                },
            },
            "vector_store": {
                "provider": "qdrant",
                "config": vector_store_config,
            },
        }
        self._memory = Memory.from_config(config)

    async def add_memory(self, user_id: str, content: str) -> None:
        """
        Add a memory entry for the user (non-blocking).

        Mem0 internally extracts key facts from `content` using the LLM,
        then stores them as searchable embeddings in Qdrant.

        Args:
            user_id: Unique user identifier.
            content: Text content to store as memory (e.g., research queries,
                     findings, or preferences).
        """
        if not self._available or not self._memory:
            return

        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self._memory.add(content, user_id=user_id),
            )
            logger.debug("Mem0: memory added | user_id={}", user_id)
        except Exception as exc:
            logger.warning("Mem0: add_memory failed | user_id={} error={}", user_id, exc)

    async def get_memory(self, user_id: str, query: str, limit: int = 5) -> str:
        """
        Retrieve relevant memories for a user given a query.

        Args:
            user_id: Unique user identifier.
            query: Search query to find relevant memories (e.g., the user's
                   current research question).
            limit: Maximum number of memory entries to return.

        Returns:
            Formatted string of relevant past research context (bullet list),
            or empty string if no memories exist or memory is unavailable.
        """
        if not self._available or not self._memory:
            return ""

        try:
            loop = asyncio.get_event_loop()
            raw = await loop.run_in_executor(
                None,
                lambda: self._memory.search(query, user_id=user_id, limit=limit),
            )

            if not raw:
                return ""

            # Handle both list[dict] (older API) and SearchResults object (newer API)
            items = raw
            if hasattr(raw, "results"):
                items = raw.results  # mem0 >= 0.1.0 wraps results

            memories: list[str] = []
            for item in items:
                if isinstance(item, dict):
                    text = item.get("memory") or item.get("text") or str(item)
                else:
                    text = getattr(item, "memory", None) or str(item)
                if text:
                    memories.append(text)

            if not memories:
                return ""

            context = "\n".join(f"- {m}" for m in memories)
            logger.debug(
                "Mem0: retrieved {} memories | user_id={}", len(memories), user_id
            )
            return context

        except Exception as exc:
            logger.warning(
                "Mem0: get_memory failed | user_id={} error={}", user_id, exc
            )
            return ""

    async def close(self) -> None:
        """No cleanup needed for the in-process Mem0 client."""
        pass

    @property
    def is_available(self) -> bool:
        """True if Mem0 was successfully initialized and is ready to use."""
        return self._available
