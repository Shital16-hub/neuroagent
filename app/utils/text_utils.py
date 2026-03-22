"""
Text processing utilities.

Used across multiple agents for chunking, cleaning, and formatting.
"""

import re
from typing import Optional


def clean_text(text: str) -> str:
    """
    Remove excessive whitespace and normalize Unicode from text.

    Args:
        text: Raw text (e.g., from arXiv abstract).

    Returns:
        Cleaned, normalized text.
    """
    # Normalize unicode whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove common LaTeX artifacts
    text = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', text)
    text = re.sub(r'\\[a-zA-Z]+', '', text)
    return text.strip()


def chunk_text(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[str]:
    """
    Split text into overlapping chunks for vector indexing.

    Args:
        text: Input text to chunk.
        chunk_size: Target characters per chunk.
        chunk_overlap: Characters to overlap between chunks.

    Returns:
        List of text chunks.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        # Try to break on a sentence boundary
        if end < len(text):
            # Look for '. ' near the end of the chunk
            boundary = text.rfind('. ', start, end)
            if boundary > start + chunk_size // 2:
                end = boundary + 1
        chunks.append(text[start:end].strip())
        start = end - chunk_overlap

    return [c for c in chunks if c]


def truncate_text(text: str, max_chars: int = 3000, suffix: str = "...") -> str:
    """
    Truncate text to max_chars, ending at a word boundary.

    Used to prevent exceeding LLM token limits.
    """
    if len(text) <= max_chars:
        return text
    truncated = text[: max_chars - len(suffix)]
    # Break at last space
    last_space = truncated.rfind(' ')
    if last_space > max_chars // 2:
        truncated = truncated[:last_space]
    return truncated + suffix


def extract_year_from_date(date_str: Optional[str]) -> Optional[int]:
    """
    Extract a 4-digit year from a date string.

    Handles formats: '2023-01-15', '2023', 'January 2023', etc.

    Args:
        date_str: Any date string.

    Returns:
        Year as int, or None if extraction fails.
    """
    if not date_str:
        return None
    match = re.search(r'\b(19|20)\d{2}\b', date_str)
    if match:
        return int(match.group())
    return None


def format_authors(authors: list[str], max_authors: int = 3) -> str:
    """
    Format a list of author names for display.

    Args:
        authors: List of author name strings.
        max_authors: Show this many authors before appending 'et al.'

    Returns:
        Formatted string like "Smith J, Jones A, Lee B et al."
    """
    if not authors:
        return "Unknown Authors"
    if len(authors) <= max_authors:
        return ", ".join(authors)
    shown = ", ".join(authors[:max_authors])
    return f"{shown} et al."
