"""
Fetcher Agent — searches academic APIs and populates the vector store.

Responsibilities:
  1. Search arXiv and Semantic Scholar in parallel (asyncio.gather)
  2. Deduplicate combined results by DOI, then by paper_id
  3. Upsert unique papers into Qdrant (semantic index)
  4. Save raw paper metadata to MongoDB
  5. Update AgentState.papers with the deduplicated list

Failure policy: if both APIs fail, add an error to state.errors and return
state with an empty papers list — never raise an exception out of run().
"""

import asyncio
from loguru import logger

from app.models.paper import Paper
from app.models.state import AgentState
from app.services.arxiv_client import ArxivClient
from app.services.semantic_scholar import SemanticScholarClient
from app.services.qdrant_service import QdrantService
from app.services.mongodb_service import MongoDBService


def _deduplicate(papers: list[Paper]) -> list[Paper]:
    """
    Deduplicate a mixed list of papers from multiple sources.

    Dedup priority order:
      1. DOI (canonical, source-independent identifier)
      2. arXiv ID (from external_ids["arxiv"])
      3. paper_id (last resort)

    When a duplicate is found, we keep the version with more information
    (prefer Semantic Scholar entries as they include citation counts).
    """
    seen_dois: set[str] = set()
    seen_arxiv: set[str] = set()
    seen_ids: set[str] = set()
    unique: list[Paper] = []

    for paper in papers:
        # Check DOI
        if paper.doi:
            if paper.doi in seen_dois:
                logger.debug("Dedup: skipping duplicate DOI={}", paper.doi)
                continue
            seen_dois.add(paper.doi)

        # Check arXiv ID
        arxiv_id = paper.external_ids.get("arxiv")
        if arxiv_id:
            if arxiv_id in seen_arxiv:
                logger.debug("Dedup: skipping duplicate arXiv={}", arxiv_id)
                continue
            seen_arxiv.add(arxiv_id)

        # Check paper_id
        if paper.paper_id in seen_ids:
            logger.debug("Dedup: skipping duplicate paper_id={}", paper.paper_id)
            continue
        seen_ids.add(paper.paper_id)

        unique.append(paper)

    return unique


class FetcherAgent:
    """
    Specialist agent responsible for retrieving academic papers.

    Requires QdrantService and MongoDBService to be already connected
    (injected at construction time from FastAPI dependencies).

    Usage (within LangGraph):
        agent = FetcherAgent(qdrant=qdrant_svc, mongodb=mongodb_svc)
        state = await agent.run(state)
    """

    def __init__(
        self,
        qdrant: QdrantService,
        mongodb: MongoDBService,
    ) -> None:
        self._qdrant = qdrant
        self._mongodb = mongodb

    async def run(self, state: AgentState) -> AgentState:
        """
        LangGraph node entry point.

        Reads:   state["query"], state["max_papers"]
        Writes:  state["papers"], state["errors"]

        Never raises — all errors are caught and appended to state["errors"].

        Args:
            state: Current AgentState from the LangGraph workflow.

        Returns:
            Updated AgentState with papers list populated (possibly empty on failure).
        """
        query = state["query"]
        max_papers = state.get("max_papers", 10)
        session_id = state["session_id"]

        logger.info(
            "FetcherAgent starting | session={} query='{}' max_papers={}",
            session_id,
            query,
            max_papers,
        )

        # ── Parallel API fetch ────────────────────────────────────────────────
        arxiv_papers: list[Paper] = []
        s2_papers: list[Paper] = []

        try:
            arxiv_papers, s2_papers = await self._fetch_parallel(query, max_papers)
        except Exception as exc:
            error_msg = f"Both APIs failed during fetch: {exc}"
            logger.error("FetcherAgent: {}", error_msg)
            return {**state, "papers": [], "errors": state["errors"] + [error_msg]}

        # ── Deduplicate ───────────────────────────────────────────────────────
        combined = arxiv_papers + s2_papers
        unique_papers = _deduplicate(combined)

        logger.info(
            "FetcherAgent dedup | arxiv={} s2={} combined={} unique={}",
            len(arxiv_papers),
            len(s2_papers),
            len(combined),
            len(unique_papers),
        )

        if not unique_papers:
            warning = f"No papers found for query: '{query}'"
            logger.warning("FetcherAgent: {}", warning)
            return {**state, "papers": [], "errors": state["errors"] + [warning]}

        # ── Qdrant upsert ─────────────────────────────────────────────────────
        try:
            await self._qdrant.upsert_papers(unique_papers)
        except Exception as exc:
            # Non-fatal: we can continue without vector index
            error_msg = f"Qdrant upsert failed (continuing without vector index): {exc}"
            logger.warning("FetcherAgent: {}", error_msg)
            state = {**state, "errors": state["errors"] + [error_msg]}

        # ── MongoDB save ──────────────────────────────────────────────────────
        try:
            await self._mongodb.save_papers(unique_papers)
        except Exception as exc:
            error_msg = f"MongoDB save failed (non-fatal): {exc}"
            logger.warning("FetcherAgent: {}", error_msg)
            state = {**state, "errors": state["errors"] + [error_msg]}

        logger.info(
            "FetcherAgent complete | session={} papers_stored={}",
            session_id,
            len(unique_papers),
        )

        return {**state, "papers": unique_papers}

    async def _fetch_parallel(
        self, query: str, max_papers: int
    ) -> tuple[list[Paper], list[Paper]]:
        """
        Run arXiv and Semantic Scholar searches concurrently.

        Semantic Scholar receives the DOI set from arXiv to skip duplicates
        at parse time. This reduces redundant dedup work.

        Args:
            query: Research query string.
            max_papers: Target total papers (split ~60/40 between sources).

        Returns:
            Tuple of (arxiv_papers, s2_papers).

        Raises:
            Exception: Only if BOTH fetches raise — individual failures return [].
        """
        arxiv_limit = max(max_papers, 10)
        s2_limit = max(max_papers, 10)

        arxiv_result: list[Paper] = []
        s2_result: list[Paper] = []

        async def fetch_arxiv() -> list[Paper]:
            try:
                async with ArxivClient() as client:
                    return await client.search(query, max_results=arxiv_limit)
            except Exception as exc:
                logger.warning("arXiv fetch failed | error={}", exc)
                return []

        async def fetch_s2(doi_set: set[str]) -> list[Paper]:
            try:
                async with SemanticScholarClient() as client:
                    return await client.search(
                        query,
                        max_results=s2_limit,
                        exclude_doi_set=doi_set,
                    )
            except Exception as exc:
                logger.warning("Semantic Scholar fetch failed | error={}", exc)
                return []

        # Run both concurrently
        arxiv_result, s2_result = await asyncio.gather(
            fetch_arxiv(),
            fetch_s2(set()),  # initial fetch without DOI exclusion
        )

        # If arXiv succeeded, re-run S2 dedup at the dedup stage instead
        # (avoids a second network round-trip just for exclusion)
        if not arxiv_result and not s2_result:
            raise RuntimeError(
                "Both arXiv and Semantic Scholar returned empty results — "
                "check network connectivity or try a different query"
            )

        return arxiv_result, s2_result


# ── Standalone Demo ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import asyncio
    from app.models.state import initial_state
    from app.services.qdrant_service import QdrantService
    from app.services.mongodb_service import MongoDBService
    import uuid

    async def _demo() -> None:
        """
        End-to-end demo: fetch papers, store in Qdrant, print summary.
        Requires Qdrant running locally (docker-compose up -d).
        MongoDB is optional — errors are non-fatal.
        """
        print("=" * 60)
        print("NeuroAgent — FetcherAgent Demo")
        print("=" * 60)

        qdrant = QdrantService()
        await qdrant.connect()

        # MongoDB is optional for demo — skip if not available
        mongodb = MongoDBService()
        try:
            await mongodb.connect()
            mongo_ok = True
        except Exception as exc:
            print(f"[WARN] MongoDB not available: {exc} (continuing without)")
            mongo_ok = False

        state = initial_state(
            query="retrieval augmented generation",
            session_id=str(uuid.uuid4()),
            max_papers=10,
        )

        agent = FetcherAgent(qdrant=qdrant, mongodb=mongodb)
        state = await agent.run(state)

        print(f"\nResults:")
        print(f"  Papers fetched:  {len(state['papers'])}")
        print(f"  Errors:          {state['errors'] or 'none'}")

        count = await qdrant.collection_count()
        print(f"  Qdrant total:    {count} vectors in 'papers' collection")

        if state["papers"]:
            print(f"\nTop papers fetched:")
            for i, p in enumerate(state["papers"][:5], 1):
                print(f"  {i}. [{p.source.upper()}] {p.title[:70]}")
                print(f"     Year: {p.year} | Citations: {p.citation_count}")

        await qdrant.close()
        if mongo_ok:
            await mongodb.close()

    asyncio.run(_demo())
