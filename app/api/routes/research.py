"""
Research API routes.

POST /api/research        — Run the full NeuroAgent pipeline for a query.
                           Returns complete results synchronously.
GET  /api/research/{id}  — Retrieve a previously-saved research session.
"""

import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.agents.orchestrator import Orchestrator
from app.dependencies import get_orchestrator, get_mongodb
from app.services.mongodb_service import MongoDBService

router = APIRouter(prefix="/research", tags=["Research"])


# ── Request / Response schemas ─────────────────────────────────────────────────

class ResearchRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="The research question to investigate",
        examples=["What are the limitations of RAG systems in production?"],
    )
    user_id: Optional[str] = Field(
        default=None,
        description="Optional user ID for Mem0 personalization",
    )
    max_papers: int = Field(
        default=10,
        ge=1,
        le=30,
        description="Maximum number of papers to fetch",
    )


class ConflictResponse(BaseModel):
    paper_a_id: str
    paper_b_id: str
    description: str
    confidence: float
    conflict_type: str


class EvaluationResponse(BaseModel):
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    average_score: float
    passed_quality_threshold: bool
    evaluation_error: Optional[str] = None


class ResearchResponse(BaseModel):
    session_id: str
    query: str
    papers_fetched: int
    summaries_generated: int
    concepts: list[str]
    contradictions: list[ConflictResponse]
    final_synthesis: str
    evaluation: Optional[EvaluationResponse]
    errors: list[str]


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post(
    "",
    response_model=ResearchResponse,
    summary="Run full research pipeline",
    description=(
        "Fetches papers from arXiv and Semantic Scholar, summarizes them, "
        "detects contradictions, extracts concepts, generates a synthesis, "
        "and evaluates quality using RAGAS. All steps run in a LangGraph pipeline."
    ),
)
async def run_research(
    request: ResearchRequest,
    orchestrator: Orchestrator = Depends(get_orchestrator),
) -> ResearchResponse:
    """Run the full NeuroAgent pipeline and return structured results."""
    session_id = str(uuid.uuid4())

    result = await orchestrator.run_research(
        query=request.query,
        user_id=request.user_id,
        session_id=session_id,
    )

    # Build contradiction list
    contradictions: list[ConflictResponse] = []
    report = result.get("contradiction_report")
    if report and report.conflicts:
        for c in report.conflicts:
            contradictions.append(ConflictResponse(
                paper_a_id=c.paper_a_id,
                paper_b_id=c.paper_b_id,
                description=c.description,
                confidence=c.confidence,
                conflict_type=c.conflict_type,
            ))

    # Build evaluation response
    evaluation: Optional[EvaluationResponse] = None
    if result.get("evaluation"):
        ev = result["evaluation"]
        evaluation = EvaluationResponse(
            faithfulness=ev.faithfulness,
            answer_relevancy=ev.answer_relevancy,
            context_precision=ev.context_precision,
            average_score=ev.average_score,
            passed_quality_threshold=ev.passed_quality_threshold,
            evaluation_error=ev.evaluation_error,
        )

    return ResearchResponse(
        session_id=session_id,
        query=request.query,
        papers_fetched=len(result.get("papers", [])),
        summaries_generated=len(result.get("summaries", [])),
        concepts=result.get("concepts", []),
        contradictions=contradictions,
        final_synthesis=result.get("final_synthesis", ""),
        evaluation=evaluation,
        errors=result.get("errors", []),
    )


@router.get(
    "/{session_id}",
    summary="Get research session",
    description="Retrieve a previously-saved research session from MongoDB.",
)
async def get_session(
    session_id: str,
    mongodb: MongoDBService = Depends(get_mongodb),
):
    """Retrieve a research session by ID."""
    doc = await mongodb.get_session(session_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found",
        )
    return doc
