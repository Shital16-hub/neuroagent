"""
Paper data models.

Represents a single academic paper fetched from arXiv, Semantic Scholar, or PubMed.
All fields that may be missing from a given source are Optional.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class Paper(BaseModel):
    """A single academic paper from any data source."""

    paper_id: str = Field(
        description="Unique identifier — arXiv ID, DOI, or Semantic Scholar ID"
    )
    title: str = Field(description="Full paper title")
    abstract: str = Field(description="Paper abstract text")
    authors: list[str] = Field(default_factory=list, description="List of author names")
    year: Optional[int] = Field(default=None, description="Publication year")
    pdf_url: Optional[str] = Field(default=None, description="Direct URL to PDF")
    doi: Optional[str] = Field(default=None, description="Digital Object Identifier")
    source: str = Field(
        description="Data source: 'arxiv' | 'semantic_scholar' | 'pubmed'"
    )
    fetched_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="UTC timestamp when this paper was fetched",
    )
    citation_count: Optional[int] = Field(
        default=None, description="Number of citations (if available from source)"
    )
    external_ids: dict[str, str] = Field(
        default_factory=dict,
        description="Additional IDs: {'arxiv': '...', 'doi': '...', 's2': '...'}",
    )

    @field_validator("source")
    @classmethod
    def validate_source(cls, v: str) -> str:
        allowed = {"arxiv", "semantic_scholar", "pubmed"}
        if v not in allowed:
            raise ValueError(f"source must be one of {allowed}, got '{v}'")
        return v

    @field_validator("abstract")
    @classmethod
    def abstract_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("abstract cannot be empty")
        return v.strip()

    @field_validator("title")
    @classmethod
    def title_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("title cannot be empty")
        return v.strip()

    model_config = {"json_schema_extra": {"example": {
        "paper_id": "2307.09288",
        "title": "Llama 2: Open Foundation and Fine-Tuned Chat Models",
        "abstract": "In this work, we develop and release Llama 2...",
        "authors": ["Hugo Touvron", "Louis Martin"],
        "year": 2023,
        "pdf_url": "https://arxiv.org/pdf/2307.09288",
        "doi": None,
        "source": "arxiv",
        "citation_count": 4200,
    }}}
