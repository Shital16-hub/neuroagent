"""
End-to-end orchestrator test: full LangGraph pipeline via Orchestrator class.

Run with:
    python -m tests.test_orchestrator

Requires:
    - GROQ_API_KEY in .env
    - QDRANT_URL + QDRANT_API_KEY in .env
    - NEO4J_URI in .env (optional — skipped if unavailable)
    - MONGODB_URL in .env (optional — skipped if unavailable)
"""

import asyncio
import uuid

from loguru import logger

from app.agents.orchestrator import Orchestrator
from app.services.qdrant_service import QdrantService
from app.services.mongodb_service import MongoDBService
from app.services.neo4j_service import Neo4jService
from app.services.mem0_service import Mem0Service


async def run_orchestrator_test(query: str) -> None:
    print("=" * 65)
    print("NeuroAgent — Full Orchestrator Test (LangGraph)")
    print(f"Query: '{query}'")
    print("=" * 65)

    # ── Service setup ──────────────────────────────────────────────────────────
    qdrant = QdrantService()
    await qdrant.connect()
    print("[OK] Qdrant connected")

    mongodb: MongoDBService | None = None
    try:
        mongodb = MongoDBService()
        await mongodb.connect()
        print("[OK] MongoDB connected")
    except Exception as exc:
        print(f"[WARN] MongoDB unavailable: {exc}")
        mongodb = None

    neo4j: Neo4jService | None = None
    try:
        neo4j = Neo4jService()
        await neo4j.connect()
        print("[OK] Neo4j connected")
    except Exception as exc:
        print(f"[WARN] Neo4j unavailable: {exc}")
        neo4j = None

    mem0: Mem0Service | None = None
    try:
        mem0 = Mem0Service()
        await mem0.connect()
        if mem0.is_available:
            print("[OK] Mem0 initialized")
        else:
            print("[WARN] Mem0 disabled (Groq key not set or init failed)")
    except Exception as exc:
        print(f"[WARN] Mem0 unavailable: {exc}")
        mem0 = None

    # ── Run pipeline ───────────────────────────────────────────────────────────
    print("\nRunning LangGraph pipeline...")
    orchestrator = Orchestrator(
        qdrant=qdrant,
        mongodb=mongodb,
        neo4j=neo4j,
        mem0=mem0,
    )

    session_id = str(uuid.uuid4())
    result = await orchestrator.run_research(
        query=query,
        user_id="test_user_001",
        session_id=session_id,
    )

    # ── Print results ──────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("Pipeline Results")
    print("=" * 65)
    print(f"  Session ID:          {session_id}")
    print(f"  Papers fetched:      {len(result.get('papers', []))}")
    print(f"  Papers summarized:   {len(result.get('summaries', []))}")

    report = result.get("contradiction_report")
    print(f"  Contradictions:      {len(report.conflicts) if report else 0}")
    print(f"  Concepts:            {len(result.get('concepts', []))}")
    print(f"  Synthesis length:    {len(result.get('final_synthesis', ''))} chars")
    print(f"  Qdrant ctx chunks:   {len(result.get('qdrant_context', []))}")

    evaluation = result.get("evaluation")
    if evaluation:
        print(f"\n  RAGAS Scores:")
        if evaluation.evaluation_error:
            print(f"    Error: {evaluation.evaluation_error[:100]}")
            print(f"    Scores: 0.0 / 0.0 / 0.0 (defaulted due to error)")
        else:
            print(f"    Faithfulness:     {evaluation.faithfulness:.3f}")
            print(f"    Answer Relevancy: {evaluation.answer_relevancy:.3f}")
            print(f"    Context Precision:{evaluation.context_precision:.3f}")
            print(f"    Average:          {evaluation.average_score:.3f}")
            print(f"    Quality passed:   {evaluation.passed_quality_threshold}")
    else:
        print("  RAGAS Evaluation:    None")

    print(f"\n  Errors: {len(result.get('errors', []))}")
    for err in result.get("errors", []):
        print(f"    - {err}")

    # ── Print synthesis excerpt ────────────────────────────────────────────────
    synthesis = result.get("final_synthesis", "")
    if synthesis:
        print(f"\n{'=' * 65}")
        print("Synthesis Preview (first 600 chars)")
        print("=" * 65)
        print(synthesis[:600])
        if len(synthesis) > 600:
            print(f"\n... [{len(synthesis) - 600} more chars]")

    # ── Concepts ───────────────────────────────────────────────────────────────
    concepts = result.get("concepts", [])
    if concepts:
        print(f"\nConcepts: {', '.join(concepts[:10])}")
        if len(concepts) > 10:
            print(f"... and {len(concepts) - 10} more")

    # ── Teardown ───────────────────────────────────────────────────────────────
    await qdrant.close()
    if mongodb:
        await mongodb.close()
    if neo4j:
        await neo4j.close()
    if mem0:
        await mem0.close()

    print(f"\n[DONE] Pipeline completed")


if __name__ == "__main__":
    asyncio.run(
        run_orchestrator_test("what are the limitations of RAG systems in production?")
    )
