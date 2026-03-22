"""
Summary and contradiction data models.

PaperSummary — structured output from the Summarizer Agent.
Conflict / ContradictionReport — output from the Contradiction Detector Agent.
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator


class PaperSummary(BaseModel):
    """
    Structured LLM-generated summary of a single academic paper.
    Produced by the Summarizer Agent from a paper's title + abstract.
    """

    paper_id: str = Field(description="References Paper.paper_id")
    key_claims: list[str] = Field(
        description="Main claims or contributions of the paper (3-5 bullet points)"
    )
    methodology: str = Field(
        description="Brief description of the research methodology or approach"
    )
    findings: str = Field(
        description="Key results and empirical findings"
    )
    limitations: str = Field(
        description="Stated or inferred limitations of the work"
    )
    summary_model: str = Field(
        description="LLM model ID used to generate this summary"
    )

    @field_validator("key_claims")
    @classmethod
    def at_least_one_claim(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("key_claims must contain at least one claim")
        return [claim.strip() for claim in v if claim.strip()]

    model_config = {"json_schema_extra": {"example": {
        "paper_id": "2307.09288",
        "key_claims": [
            "Llama 2 outperforms open-source models on most benchmarks",
            "RLHF fine-tuning improves safety without sacrificing helpfulness",
        ],
        "methodology": "Supervised fine-tuning followed by RLHF with human preference data",
        "findings": "Llama 2-Chat achieves comparable performance to GPT-3.5 on many tasks",
        "limitations": "Still underperforms GPT-4; safety evaluations limited to English",
        "summary_model": "llama-3.1-8b-instant",
    }}}


class Conflict(BaseModel):
    """
    A single detected conflict between two papers.
    Produced by the Contradiction Detector Agent.
    """

    paper_a_id: str = Field(description="ID of the first paper in the conflict")
    paper_b_id: str = Field(description="ID of the second paper in the conflict")
    description: str = Field(
        description="Human-readable explanation of the conflict"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score for this conflict (0.0 = uncertain, 1.0 = definitive)",
    )
    conflict_type: Literal["direct", "methodological", "scope"] = Field(
        description=(
            "Type of conflict: "
            "'direct' = contradictory factual claims, "
            "'methodological' = different methods yield incompatible conclusions, "
            "'scope' = different conditions / datasets make claims incomparable"
        )
    )
    claim_a: Optional[str] = Field(
        default=None, description="The specific claim from paper A involved in this conflict"
    )
    claim_b: Optional[str] = Field(
        default=None, description="The specific claim from paper B involved in this conflict"
    )

    model_config = {"json_schema_extra": {"example": {
        "paper_a_id": "2307.09288",
        "paper_b_id": "2303.08774",
        "description": "Paper A claims RLHF improves safety; Paper B finds RLHF reduces factual accuracy",
        "confidence": 0.75,
        "conflict_type": "direct",
        "claim_a": "RLHF fine-tuning significantly improves model safety",
        "claim_b": "RLHF training reduces factual grounding in model responses",
    }}}


class ContradictionReport(BaseModel):
    """
    Full contradiction analysis across a set of papers.
    Produced by the Contradiction Detector Agent.
    """

    conflicts: list[Conflict] = Field(
        default_factory=list,
        description="All detected conflicts between paper pairs",
    )
    total_papers_compared: int = Field(
        ge=0,
        description="Number of papers analyzed to produce this report",
    )
    has_conflicts: bool = Field(
        default=False,
        description="Convenience flag — True if at least one conflict was detected",
    )

    def model_post_init(self, __context: object) -> None:
        """Derive has_conflicts from the conflicts list after initialization."""
        self.has_conflicts = len(self.conflicts) > 0

    @property
    def high_confidence_conflicts(self) -> list[Conflict]:
        """Returns only conflicts with confidence >= 0.7."""
        return [c for c in self.conflicts if c.confidence >= 0.7]
