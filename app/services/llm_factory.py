"""
LLM Factory — abstract LLM provider with automatic fallback.

Priority order:
  1. Groq (primary — free tier, fast)
  2. OpenAI (fallback — if OPENAI_API_KEY is set)
  3. Ollama (local fallback — if OLLAMA_BASE_URL is set)

Usage:
    from app.services.llm_factory import LLMFactory

    llm = LLMFactory.get_llm()                          # default model
    reasoning_llm = LLMFactory.get_reasoning_llm()     # heavier reasoning model
    llm = LLMFactory.get_llm(model="llama-3.1-70b-versatile")  # explicit model
"""

import logging
from enum import Enum
from typing import Optional

from langchain_core.language_models import BaseChatModel

from app.config import get_settings

logger = logging.getLogger(__name__)


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    GROQ = "groq"
    OPENAI = "openai"
    OLLAMA = "ollama"


class LLMFactory:
    """
    Factory for creating LangChain-compatible LLM instances.

    All returned LLMs implement the BaseChatModel interface, so they are
    interchangeable throughout the codebase regardless of provider.
    """

    # Groq model identifiers (free tier)
    GROQ_FAST_MODEL = "llama-3.1-8b-instant"
    GROQ_REASONING_MODEL = "deepseek-r1-distill-llama-70b"
    GROQ_BALANCED_MODEL = "llama-3.1-70b-versatile"

    # OpenAI model identifiers
    OPENAI_DEFAULT_MODEL = "gpt-4o-mini"
    OPENAI_REASONING_MODEL = "gpt-4o"

    @classmethod
    def get_llm(
        cls,
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        provider: Optional[LLMProvider] = None,
    ) -> BaseChatModel:
        """
        Return the best available LLM for standard tasks.

        Args:
            model: Explicit model name. If None, uses config default.
            temperature: Sampling temperature (lower = more deterministic).
            max_tokens: Maximum tokens in the response.
            provider: Force a specific provider. If None, uses priority order.

        Returns:
            A LangChain BaseChatModel instance.

        Raises:
            RuntimeError: If no LLM provider is configured.
        """
        settings = get_settings()
        target_model = model or settings.default_llm_model

        if provider:
            return cls._build_llm(provider, target_model, temperature, max_tokens)

        # Auto-select provider by priority
        if settings.has_groq:
            logger.debug("LLMFactory: using Groq provider, model=%s", target_model)
            return cls._build_groq(target_model, temperature, max_tokens)

        if settings.has_openai:
            logger.warning(
                "LLMFactory: Groq not configured, falling back to OpenAI"
            )
            fallback_model = cls.OPENAI_DEFAULT_MODEL
            return cls._build_openai(fallback_model, temperature, max_tokens)

        if settings.has_ollama:
            logger.warning(
                "LLMFactory: Groq/OpenAI not configured, falling back to Ollama"
            )
            return cls._build_ollama(target_model, temperature, max_tokens)

        raise RuntimeError(
            "No LLM provider configured. "
            "Set at least one of: GROQ_API_KEY, OPENAI_API_KEY, or OLLAMA_BASE_URL "
            "in your .env file."
        )

    @classmethod
    def get_reasoning_llm(
        cls,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> BaseChatModel:
        """
        Return an LLM optimized for multi-step reasoning tasks.

        Used by: Contradiction Detector Agent, Synthesis Agent.
        On Groq free tier, this uses deepseek-r1-distill-llama-70b.

        Args:
            temperature: Lower temperature for more deterministic reasoning.
            max_tokens: Higher token limit for longer reasoning chains.

        Returns:
            A LangChain BaseChatModel instance.
        """
        settings = get_settings()

        if settings.has_groq:
            model = settings.reasoning_llm_model
            logger.debug("LLMFactory: reasoning LLM — Groq/%s", model)
            return cls._build_groq(model, temperature, max_tokens)

        if settings.has_openai:
            logger.warning("LLMFactory: reasoning LLM — falling back to OpenAI")
            return cls._build_openai(cls.OPENAI_REASONING_MODEL, temperature, max_tokens)

        # Fall back to default if no specialized reasoning model available
        logger.warning(
            "LLMFactory: no reasoning-capable provider configured, using default LLM"
        )
        return cls.get_llm(temperature=temperature, max_tokens=max_tokens)

    @classmethod
    def get_provider(cls) -> LLMProvider:
        """Return the currently active provider based on configuration."""
        settings = get_settings()
        if settings.has_groq:
            return LLMProvider.GROQ
        if settings.has_openai:
            return LLMProvider.OPENAI
        if settings.has_ollama:
            return LLMProvider.OLLAMA
        raise RuntimeError("No LLM provider is configured.")

    # ── Private Builders ───────────────────────────────────────────────────────

    @classmethod
    def _build_llm(
        cls,
        provider: LLMProvider,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> BaseChatModel:
        """Dispatch to the correct provider builder."""
        if provider == LLMProvider.GROQ:
            return cls._build_groq(model, temperature, max_tokens)
        if provider == LLMProvider.OPENAI:
            return cls._build_openai(model, temperature, max_tokens)
        if provider == LLMProvider.OLLAMA:
            return cls._build_ollama(model, temperature, max_tokens)
        raise ValueError(f"Unknown provider: {provider}")

    @classmethod
    def _build_groq(
        cls, model: str, temperature: float, max_tokens: int
    ) -> BaseChatModel:
        """Build a Groq LLM instance."""
        from langchain_groq import ChatGroq

        settings = get_settings()
        return ChatGroq(
            model=model,
            api_key=settings.groq_api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    @classmethod
    def _build_openai(
        cls, model: str, temperature: float, max_tokens: int
    ) -> BaseChatModel:
        """Build an OpenAI LLM instance."""
        from langchain_openai import ChatOpenAI

        settings = get_settings()
        return ChatOpenAI(
            model=model,
            api_key=settings.openai_api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    @classmethod
    def _build_ollama(
        cls, model: str, temperature: float, max_tokens: int
    ) -> BaseChatModel:
        """Build an Ollama LLM instance (local)."""
        from langchain_community.chat_models import ChatOllama

        settings = get_settings()
        return ChatOllama(
            model=model,
            base_url=settings.ollama_base_url,
            temperature=temperature,
            num_predict=max_tokens,
        )
