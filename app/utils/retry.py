"""
Retry decorators with exponential backoff using tenacity.

Usage:
    from app.utils.retry import async_retry, async_retry_on_rate_limit

    @async_retry_on_rate_limit
    async def call_groq_api(...):
        ...
"""

import logging
from functools import wraps
from typing import Callable, Type

from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

logger = logging.getLogger(__name__)


def async_retry(
    max_attempts: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 60.0,
    exception_types: tuple[Type[Exception], ...] = (Exception,),
) -> Callable:
    """
    Generic async retry decorator with exponential backoff.

    Args:
        max_attempts: Total number of attempts before giving up.
        min_wait: Minimum wait between retries (seconds).
        max_wait: Maximum wait between retries (seconds).
        exception_types: Only retry on these exception types.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(max_attempts),
                wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
                retry=retry_if_exception_type(exception_types),
                before_sleep=before_sleep_log(logger, logging.WARNING),
                reraise=True,
            ):
                with attempt:
                    return await func(*args, **kwargs)
        return wrapper
    return decorator


# Pre-configured decorators for common use cases

async_retry_on_rate_limit = async_retry(
    max_attempts=5,
    min_wait=2.0,
    max_wait=30.0,
    exception_types=(Exception,),
)
"""Use this on Groq/OpenAI calls that may hit 429 rate limits."""

async_retry_network = async_retry(
    max_attempts=3,
    min_wait=1.0,
    max_wait=15.0,
    exception_types=(Exception,),
)
"""Use this on HTTP calls to external APIs (arXiv, Semantic Scholar)."""
