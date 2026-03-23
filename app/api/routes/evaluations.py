"""
Evaluations API routes.

GET /api/evaluations        — List recent RAGAS evaluation results (paginated).
GET /api/evaluations/stats  — Aggregate statistics across all evaluations.
"""

from typing import Any

from fastapi import APIRouter, Depends, Query

from app.dependencies import get_mongodb
from app.services.mongodb_service import MongoDBService

router = APIRouter(prefix="/evaluations", tags=["Evaluations"])


@router.get(
    "",
    summary="List RAGAS evaluations",
    description="Returns recent RAGAS evaluation results, newest first. Supports pagination.",
)
async def list_evaluations(
    limit: int = Query(default=20, ge=1, le=100, description="Max results to return"),
    skip: int = Query(default=0, ge=0, description="Pagination offset"),
    mongodb: MongoDBService = Depends(get_mongodb),
) -> dict[str, Any]:
    """Retrieve paginated RAGAS evaluation results."""
    results = await mongodb.get_evaluations(limit=limit, skip=skip)
    return {
        "evaluations": results,
        "count": len(results),
        "skip": skip,
        "limit": limit,
    }


@router.get(
    "/stats",
    summary="Evaluation statistics",
    description=(
        "Aggregate statistics across all stored RAGAS evaluations: "
        "mean scores, pass rate, and total count."
    ),
)
async def evaluation_stats(
    mongodb: MongoDBService = Depends(get_mongodb),
) -> dict[str, Any]:
    """Compute aggregate statistics over all RAGAS evaluations."""
    results = await mongodb.get_evaluations(limit=500, skip=0)

    if not results:
        return {
            "total_evaluations": 0,
            "mean_faithfulness": None,
            "mean_answer_relevancy": None,
            "mean_context_precision": None,
            "mean_average_score": None,
            "pass_rate": None,
        }

    # Filter out failed evaluations (where evaluation_error is set)
    valid = [r for r in results if not r.get("evaluation_error")]

    def _mean(key: str) -> float:
        vals = [r[key] for r in valid if key in r]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    mean_faith = _mean("faithfulness")
    mean_relevancy = _mean("answer_relevancy")
    mean_precision = _mean("context_precision")
    mean_avg = round((mean_faith + mean_relevancy + mean_precision) / 3, 4)

    pass_rate = round(
        sum(1 for r in valid if (r.get("faithfulness", 0) + r.get("answer_relevancy", 0) + r.get("context_precision", 0)) / 3 >= 0.7)
        / len(valid)
        if valid else 0.0,
        4,
    )

    return {
        "total_evaluations": len(results),
        "valid_evaluations": len(valid),
        "mean_faithfulness": mean_faith,
        "mean_answer_relevancy": mean_relevancy,
        "mean_context_precision": mean_precision,
        "mean_average_score": mean_avg,
        "pass_rate": pass_rate,
    }
