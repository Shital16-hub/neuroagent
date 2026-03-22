"""
Qdrant vector store service.

Manages two collections:
  - 'papers'   — title + abstract embeddings for semantic paper search
  - 'concepts' — concept name embeddings (written by ConceptExtractorAgent)

Embeddings are generated locally using sentence-transformers (no API cost).
Model: sentence-transformers/all-MiniLM-L6-v2 — 384 dimensions, ~80MB on disk.

The model is loaded once at connect() and reused for all encode() calls.
Encoding runs in a thread pool to avoid blocking the async event loop.
"""

import asyncio
import hashlib
import uuid
from typing import Any, Optional

from loguru import logger
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
)
from sentence_transformers import SentenceTransformer

from app.config import get_settings
from app.models.paper import Paper

# Vector dimension for all-MiniLM-L6-v2
_VECTOR_DIM = 384
_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _paper_id_to_uuid(paper_id: str) -> str:
    """
    Convert an arbitrary paper_id string into a deterministic UUID.
    Qdrant point IDs must be UUIDs or unsigned integers.
    Uses uuid5 (SHA-1 hash) so the same paper_id always yields the same UUID.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, paper_id))


def _paper_to_point(paper: Paper, vector: list[float]) -> PointStruct:
    """
    Build a Qdrant PointStruct from a Paper model and its embedding vector.

    Metadata stored in payload (all searchable / filterable):
      paper_id, title, authors, year, source, doi, pdf_url, citation_count
    """
    return PointStruct(
        id=_paper_id_to_uuid(paper.paper_id),
        vector=vector,
        payload={
            "paper_id": paper.paper_id,
            "title": paper.title,
            "abstract": paper.abstract[:1000],  # cap payload size
            "authors": paper.authors[:10],
            "year": paper.year,
            "source": paper.source,
            "doi": paper.doi,
            "pdf_url": paper.pdf_url,
            "citation_count": paper.citation_count,
        },
    )


def _payload_to_paper(payload: dict[str, Any]) -> Optional[Paper]:
    """Reconstruct a Paper model from a Qdrant point payload."""
    try:
        return Paper(
            paper_id=payload["paper_id"],
            title=payload["title"],
            abstract=payload.get("abstract", "No abstract available."),
            authors=payload.get("authors", []),
            year=payload.get("year"),
            source=payload["source"],
            doi=payload.get("doi"),
            pdf_url=payload.get("pdf_url"),
            citation_count=payload.get("citation_count"),
        )
    except Exception as exc:
        logger.warning("Failed to reconstruct Paper from Qdrant payload | error={}", exc)
        return None


class QdrantService:
    """
    Async Qdrant client wrapper.

    Handles collection creation, paper upsert, and semantic search.
    The SentenceTransformer model is loaded once during connect() and
    all encode() calls run in a thread pool executor to stay async-safe.

    Usage (application startup):
        service = QdrantService()
        await service.connect()
        # ... use service ...
        await service.close()
    """

    def __init__(self) -> None:
        self._client: Optional[AsyncQdrantClient] = None
        self._model: Optional[SentenceTransformer] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    async def connect(self) -> None:
        """
        Initialize Qdrant client and load the embedding model.

        Called once at application startup. Works for both local Docker
        (http://localhost:6333) and Qdrant Cloud (https://...) based on
        QDRANT_URL and QDRANT_API_KEY in settings.
        """
        settings = get_settings()

        logger.info("Connecting to Qdrant | url={}", settings.qdrant_url)
        self._client = AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
        )

        # Load embedding model in thread pool (CPU-bound, blocks event loop)
        logger.info("Loading embedding model | model={}", _EMBEDDING_MODEL)
        self._loop = asyncio.get_event_loop()
        self._model = await self._loop.run_in_executor(
            None,
            lambda: SentenceTransformer(_EMBEDDING_MODEL),
        )
        logger.info("Embedding model loaded | dim={}", _VECTOR_DIM)

        await self._ensure_collections(settings)

    async def close(self) -> None:
        """Close the Qdrant client. Called at application shutdown."""
        if self._client:
            await self._client.close()
            logger.info("Qdrant connection closed")

    # ── Embedding ──────────────────────────────────────────────────────────────

    async def _encode(self, texts: list[str]) -> list[list[float]]:
        """
        Encode a list of texts into embedding vectors.
        Runs in a thread pool to avoid blocking the async event loop.
        """
        if self._model is None:
            raise RuntimeError("QdrantService.connect() has not been called")

        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False),
        )
        return [e.tolist() for e in embeddings]

    async def _encode_single(self, text: str) -> list[float]:
        vectors = await self._encode([text])
        return vectors[0]

    # ── Papers Collection ──────────────────────────────────────────────────────

    async def upsert_papers(self, papers: list[Paper]) -> None:
        """
        Embed and upsert a list of papers into the 'papers' collection.

        Each paper is embedded as: "{title}. {abstract}" (combined for richer semantics).
        Duplicate paper_ids are safely overwritten (upsert semantics).

        Args:
            papers: List of Paper models to index.
        """
        if not papers:
            return

        settings = get_settings()

        # Build embedding texts: title + abstract concatenated
        texts = [f"{p.title}. {p.abstract}" for p in papers]

        logger.info("Embedding {} papers for Qdrant...", len(papers))
        vectors = await self._encode(texts)

        points = [_paper_to_point(paper, vec) for paper, vec in zip(papers, vectors)]

        await self._client.upsert(
            collection_name=settings.qdrant_papers_collection,
            points=points,
            wait=True,
        )
        logger.info(
            "Upserted {} papers into Qdrant collection='{}'",
            len(points),
            settings.qdrant_papers_collection,
        )

    async def search_similar(
        self,
        query: str,
        limit: int = 5,
        score_threshold: float = 0.3,
    ) -> list[Paper]:
        """
        Semantic search over the 'papers' collection.

        Args:
            query: Natural language query string.
            limit: Maximum number of results.
            score_threshold: Minimum cosine similarity score (0.0–1.0).

        Returns:
            List of Paper models ordered by similarity (most similar first).
        """
        settings = get_settings()

        query_vector = await self._encode_single(query)

        results = await self._client.search(
            collection_name=settings.qdrant_papers_collection,
            query_vector=query_vector,
            limit=limit,
            score_threshold=score_threshold,
            with_payload=True,
        )

        papers: list[Paper] = []
        for hit in results:
            paper = _payload_to_paper(hit.payload)
            if paper:
                papers.append(paper)

        logger.info(
            "Qdrant semantic search | query='{}' results={} threshold={}",
            query[:60],
            len(papers),
            score_threshold,
        )
        return papers

    async def get_paper(self, paper_id: str) -> Optional[Paper]:
        """
        Retrieve a single paper from Qdrant by its paper_id (payload field).

        Args:
            paper_id: The paper's canonical ID (arXiv ID, DOI, or s2:xxx).

        Returns:
            Paper model if found, None otherwise.
        """
        settings = get_settings()

        results, _ = await self._client.scroll(
            collection_name=settings.qdrant_papers_collection,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="paper_id",
                        match=MatchValue(value=paper_id),
                    )
                ]
            ),
            limit=1,
            with_payload=True,
        )

        if not results:
            return None

        return _payload_to_paper(results[0].payload)

    async def collection_count(self) -> int:
        """Return the number of vectors in the papers collection."""
        settings = get_settings()
        info = await self._client.get_collection(settings.qdrant_papers_collection)
        return info.points_count

    # ── Collection Setup ───────────────────────────────────────────────────────

    async def _ensure_collections(self, settings) -> None:
        """
        Create Qdrant collections if they don't already exist.
        Safe to call on every startup (checks before creating).
        """
        for collection_name in [
            settings.qdrant_papers_collection,
            settings.qdrant_concepts_collection,
        ]:
            existing = await self._client.collection_exists(collection_name)
            if not existing:
                await self._client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=_VECTOR_DIM,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info("Created Qdrant collection '{}'", collection_name)
            else:
                logger.debug("Qdrant collection '{}' already exists", collection_name)


# ── Standalone Test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import asyncio
    from app.models.paper import Paper
    from datetime import datetime

    async def _test() -> None:
        service = QdrantService()
        await service.connect()

        # Insert a dummy paper
        dummy = Paper(
            paper_id="test-rag-001",
            title="Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
            abstract=(
                "We explore a general-purpose fine-tuning recipe for retrieval-augmented "
                "generation (RAG), where seq2seq models are combined with dense vector "
                "retrieval for open-domain QA and other knowledge-intensive NLP tasks."
            ),
            authors=["Patrick Lewis", "Ethan Perez"],
            year=2020,
            source="arxiv",
        )
        await service.upsert_papers([dummy])

        count = await service.collection_count()
        print(f"\nQdrant papers collection count: {count}")

        results = await service.search_similar("retrieval augmented generation", limit=3)
        print(f"\nSearch results ({len(results)} found):")
        for p in results:
            print(f"  {p.title[:70]}")

        await service.close()

    asyncio.run(_test())
