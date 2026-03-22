"""
Application configuration.

All settings are read from environment variables (via .env file).
Pydantic Settings handles type coercion and validation automatically.
Never import this module's Settings directly — use the `get_settings()` dependency.
"""

from functools import lru_cache
from typing import Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Typed configuration for NeuroAgent.
    Values are read from environment variables or a .env file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM Providers ──────────────────────────────────────────────────────────
    groq_api_key: Optional[str] = Field(default=None, description="Groq API key (primary LLM)")
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key (fallback)")
    ollama_base_url: Optional[str] = Field(default=None, description="Ollama base URL (local)")

    default_llm_model: str = Field(
        default="llama-3.1-8b-instant",
        description="Default Groq model for fast tasks",
    )
    reasoning_llm_model: str = Field(
        default="llama-3.3-70b-versatile",
        description="Groq model for reasoning-heavy tasks",
    )
    embedding_model: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="HuggingFace embedding model (runs locally)",
    )

    # ── Vector Database ────────────────────────────────────────────────────────
    qdrant_url: str = Field(
        default="http://localhost:6333",
        description="Qdrant server URL",
    )
    qdrant_api_key: Optional[str] = Field(
        default=None,
        description="Qdrant Cloud API key (leave blank for local)",
    )
    qdrant_papers_collection: str = Field(
        default="papers",
        description="Qdrant collection name for paper vectors",
    )
    qdrant_concepts_collection: str = Field(
        default="concepts",
        description="Qdrant collection name for concept vectors",
    )

    # ── Graph Database ─────────────────────────────────────────────────────────
    neo4j_uri: str = Field(
        default="bolt://localhost:7687",
        description="Neo4j Bolt URI",
    )
    neo4j_username: str = Field(default="neo4j", description="Neo4j username")
    neo4j_password: str = Field(
        default="neuroagent_dev",
        description="Neo4j password",
    )

    # ── Application Database ───────────────────────────────────────────────────
    mongodb_url: str = Field(
        default="mongodb://localhost:27017",
        description="MongoDB connection string",
    )
    mongodb_db_name: str = Field(
        default="neuroagent",
        description="MongoDB database name",
    )

    # ── External APIs ──────────────────────────────────────────────────────────
    semantic_scholar_api_key: Optional[str] = Field(
        default=None,
        description="Semantic Scholar API key (optional, increases rate limits)",
    )
    ncbi_api_key: Optional[str] = Field(
        default=None,
        description="NCBI/PubMed API key (optional, increases rate limits)",
    )

    # ── Observability ──────────────────────────────────────────────────────────
    langchain_tracing_v2: bool = Field(
        default=False,
        description="Enable LangSmith tracing",
    )
    langchain_api_key: Optional[str] = Field(
        default=None,
        description="LangSmith API key",
    )
    langchain_project: str = Field(
        default="neuroagent",
        description="LangSmith project name",
    )
    langchain_endpoint: str = Field(
        default="https://api.smith.langchain.com",
        description="LangSmith API endpoint",
    )

    # ── Application ────────────────────────────────────────────────────────────
    app_host: str = Field(default="0.0.0.0", description="FastAPI host")
    app_port: int = Field(default=8000, description="FastAPI port")
    log_level: str = Field(default="INFO", description="Logging level")
    max_papers_per_query: int = Field(
        default=15,
        description="Maximum papers to fetch per query",
    )
    summarizer_concurrency: int = Field(
        default=5,
        description="Number of parallel LLM calls in Summarizer Agent",
    )
    ragas_background_eval: bool = Field(
        default=True,
        description="Run RAGAS evaluation as a background task (non-blocking)",
    )

    # ── Security ───────────────────────────────────────────────────────────────
    secret_key: str = Field(
        default="change_this_in_production",
        description="Secret key for session signing",
    )
    allowed_origins: str = Field(
        default="http://localhost:3000,http://localhost:7860",
        description="Comma-separated CORS allowed origins",
    )

    # ── Derived Properties ─────────────────────────────────────────────────────
    @property
    def cors_origins(self) -> list[str]:
        """Parse allowed_origins string into a list."""
        return [origin.strip() for origin in self.allowed_origins.split(",")]

    @property
    def has_groq(self) -> bool:
        return bool(self.groq_api_key)

    @property
    def has_openai(self) -> bool:
        return bool(self.openai_api_key)

    @property
    def has_ollama(self) -> bool:
        return bool(self.ollama_base_url)

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in allowed:
            raise ValueError(f"log_level must be one of {allowed}")
        return v_upper


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return a cached Settings instance.
    Using lru_cache ensures we only read .env once per process lifetime.
    In tests, call get_settings.cache_clear() to reset.
    """
    return Settings()
