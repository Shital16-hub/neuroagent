"""
End-to-end pipeline test: Fetch → Summarize → Detect Contradictions → Extract Concepts

Run with:
    python -m tests.test_pipeline

Requires:
    - GROQ_API_KEY in .env (for summarization + contradiction detection)
    - QDRANT_URL + QDRANT_API_KEY in .env (for vector store)
    - NEO4J_URI (optional — graph write skipped if unavailable)
    - MongoDB (optional — non-fatal if unavailable)
"""

import asyncio
import uuid

from loguru import logger

from app.models.state import initial_state
from app.services.qdrant_service import QdrantService
from app.services.mongodb_service import MongoDBService
from app.services.neo4j_service import Neo4jService
from app.agents.fetcher import FetcherAgent
from app.agents.summarizer import SummarizerAgent
from app.agents.contradiction import ContradictionDetectorAgent
from app.agents.concept_extractor import ConceptExtractorAgent
from app.config import get_settings


async def run_pipeline(query: str, max_papers: int = 8) -> None:
    settings = get_settings()

    print("=" * 65)
    print(f"NeuroAgent — Full Pipeline Test")
    print(f"Query: '{query}'")
    print("=" * 65)

    # ── Service setup ─────────────────────────────────────────────────────────
    qdrant = QdrantService()
    await qdrant.connect()

    mongodb = MongoDBService()
    try:
        await mongodb.connect()
        mongo_ok = True
    except Exception as exc:
        print(f"[WARN] MongoDB unavailable: {exc}")
        mongo_ok = False

    neo4j: Neo4jService | None = None
    try:
        neo4j = Neo4jService()
        await neo4j.connect()
        neo4j_ok = True
        print("[OK] Neo4j connected")
    except Exception as exc:
        print(f"[WARN] Neo4j unavailable: {exc}")
        neo4j_ok = False
        neo4j = None

    # ── Initial state ─────────────────────────────────────────────────────────
    state = initial_state(
        query=query,
        session_id=str(uuid.uuid4()),
        max_papers=max_papers,
    )

    # ── Step 1: Fetch ─────────────────────────────────────────────────────────
    print(f"\n[1/4] Fetching papers...")
    fetcher = FetcherAgent(qdrant=qdrant, mongodb=mongodb)
    state = await fetcher.run(state)
    print(f"      Papers fetched: {len(state['papers'])}")
    if state["errors"]:
        print(f"      Errors: {state['errors']}")

    if not state["papers"]:
        print("\n[ABORT] No papers fetched — cannot continue pipeline.")
        return

    # ── Step 2: Summarize ─────────────────────────────────────────────────────
    print(f"\n[2/4] Summarizing {len(state['papers'])} papers (parallel, concurrency={settings.summarizer_concurrency})...")
    summarizer = SummarizerAgent()
    state = await summarizer.run(state)
    print(f"      Summaries generated: {len(state['summaries'])}/{len(state['papers'])}")

    if state["summaries"]:
        print(f"\n      Sample summary — '{state['summaries'][0].paper_id}':")
        s = state["summaries"][0]
        for i, claim in enumerate(s.key_claims, 1):
            print(f"        Claim {i}: {claim}")
        print(f"        Method: {s.methodology}")
        print(f"        Findings: {s.findings}")

    # ── Step 3: Detect Contradictions ─────────────────────────────────────────
    print(f"\n[3/4] Detecting contradictions across {len(state['summaries'])} summaries...")
    contradiction_agent = ContradictionDetectorAgent()
    state = await contradiction_agent.run(state)

    report = state["contradiction_report"]
    if report:
        print(f"      Total conflicts found: {len(report.conflicts)}")
        print(f"      High-confidence (>=0.7): {len(report.high_confidence_conflicts)}")
        if report.conflicts:
            print(f"\n      Conflicts detected:")
            for c in report.conflicts:
                print(f"        [{c.conflict_type.upper()} | conf={c.confidence:.2f}]")
                print(f"          {c.description}")
                if c.claim_a:
                    print(f"          A: {c.claim_a[:100]}")
                if c.claim_b:
                    print(f"          B: {c.claim_b[:100]}")
        else:
            print("      No conflicts detected (papers may be complementary).")
    else:
        print("      Contradiction report: None")

    # ── Step 4: Extract Concepts ──────────────────────────────────────────────
    print(f"\n[4/4] Extracting concepts...")
    concept_agent = ConceptExtractorAgent(neo4j=neo4j)
    state = await concept_agent.run(state)

    print(f"      Concepts extracted: {len(state['concepts'])}")
    if state["concepts"]:
        top5 = state["concepts"][:5]
        print(f"      Top 5 concepts: {', '.join(top5)}")
        if len(state["concepts"]) > 5:
            remaining = state["concepts"][5:]
            print(f"      Others: {', '.join(remaining)}")

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("Pipeline Summary")
    print("=" * 65)
    print(f"  Papers fetched:      {len(state['papers'])}")
    print(f"  Papers summarized:   {len(state['summaries'])}")
    print(f"  Contradictions:      {len(report.conflicts) if report else 0}")
    print(f"  Concepts extracted:  {len(state['concepts'])}")
    print(f"  Neo4j graph:         {'written' if neo4j_ok and neo4j else 'skipped (unavailable)'}")
    print(f"  Errors:              {len(state['errors'])}")
    if state["errors"]:
        for err in state["errors"]:
            print(f"    - {err}")

    # ── Teardown ──────────────────────────────────────────────────────────────
    await qdrant.close()
    if mongo_ok:
        await mongodb.close()
    if neo4j_ok and neo4j:
        await neo4j.close()


if __name__ == "__main__":
    asyncio.run(run_pipeline("transformer attention mechanism efficiency"))
