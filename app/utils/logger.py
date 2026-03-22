"""
Structured logging setup using loguru.

All modules should import `logger` from here, not from loguru directly.
This ensures log format, level, and sinks are configured in one place.

Usage:
    from app.utils.logger import logger

    logger.info("Fetcher started", query=query, max_papers=max_papers)
    logger.error("Neo4j write failed", error=str(e), paper_id=paper_id)
"""

import sys
from app.config import get_settings
from loguru import logger as _loguru_logger


def setup_logging() -> None:
    """
    Configure loguru with structured JSON-compatible output.
    Call once at application startup (in main.py lifespan).
    """
    settings = get_settings()

    # Remove default sink
    _loguru_logger.remove()

    # Add stderr sink with structured format
    _loguru_logger.add(
        sys.stderr,
        level=settings.log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "{message}"
        ),
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    # Add file sink for persistent logs (rotation: 10 MB, retention: 7 days)
    _loguru_logger.add(
        "logs/neuroagent_{time:YYYY-MM-DD}.log",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {name}:{function}:{line} | {message}",
        rotation="10 MB",
        retention="7 days",
        compression="gz",
        enqueue=True,  # thread-safe async logging
    )

    _loguru_logger.info("Logging configured at level={}", settings.log_level)


# Module-level export — import this everywhere
logger = _loguru_logger
