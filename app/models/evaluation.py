"""
Evaluation data models.

EvaluationResult — RAGAS scores for a single research session's synthesis.
Scores are persisted to MongoDB and retrievable via the /api/evaluations endpoint.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator


class EvaluationResult(BaseModel):
    """
    RAGAS evaluation scores for one research synthesis.

    Metrics:
    - faithfulness: Is the answer grounded in the retrieved context? (hallucination check)
    - answer_relevancy: Is the answer relevant to the query?
    - context_precision: Are the retrieved chunks actually useful?
    - context_recall: Did we retrieve all the relevant chunks? (requires ground truth)
    """

    session_id: str = Field(description="Links to the research session this eval belongs to")
    query: str = Field(description="The original user research query")
    faithfulness: float = Field(
        ge=0.0,
        le=1.0,
        description="RAGAS faithfulness score (0-1). Measures hallucination resistance.",
    )
    answer_relevancy: float = Field(
        ge=0.0,
        le=1.0,
        description="RAGAS answer relevancy score (0-1). Measures answer-to-query alignment.",
    )
    context_precision: float = Field(
        ge=0.0,
        le=1.0,
        description="RAGAS context precision score (0-1). Measures retrieved chunk quality.",
    )
    context_recall: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="RAGAS context recall score (0-1). Only available if ground truth is provided.",
    )
    model_used: str = Field(description="LLM model ID used for synthesis")
    num_papers_used: int = Field(
        ge=0,
        description="Number of papers that contributed to the synthesis context",
    )
    evaluated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="UTC timestamp when evaluation was run",
    )
    evaluation_error: Optional[str] = Field(
        default=None,
        description="If RAGAS evaluation failed, the error message is stored here",
    )

    @field_validator("faithfulness", "answer_relevancy", "context_precision")
    @classmethod
    def score_in_range(cls, v: float) -> float:
        return round(v, 4)

    @computed_field
    @property
    def average_score(self) -> float:
        """Mean of the three core RAGAS metrics (excludes context_recall).
        Decorated with @computed_field so Pydantic v2 includes it in model_dump()."""
        return round(
            (self.faithfulness + self.answer_relevancy + self.context_precision) / 3,
            4,
        )

    @computed_field
    @property
    def passed_quality_threshold(self) -> bool:
        """Returns True if average score >= 0.7.
        Decorated with @computed_field so Pydantic v2 includes it in model_dump()."""
        return self.average_score >= 0.7

    model_config = ConfigDict(
        protected_namespaces=(),  # allow fields prefixed with 'model_'
        json_schema_extra={"example": {
            "session_id": "sess_abc123",
            "query": "What are the limitations of RAG systems?",
            "faithfulness": 0.87,
            "answer_relevancy": 0.91,
            "context_precision": 0.83,
            "context_recall": None,
            "model_used": "llama-3.1-8b-instant",
            "num_papers_used": 8,
            "evaluated_at": "2025-01-15T10:30:00Z",
        }},
    )
