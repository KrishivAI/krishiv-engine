"""Error text for failed queries: the exception class and message, truncated, as recorded in result JSON."""

from __future__ import annotations

MAX_ERROR_CHARS = 500


def describe_error(exc: BaseException) -> str:
    """``"Class: message"`` with the message truncated to ``MAX_ERROR_CHARS``."""
    return f"{type(exc).__name__}: {str(exc)[:MAX_ERROR_CHARS]}"


def error_fields(exc: BaseException) -> dict[str, str]:
    """The result-JSON keys for a failed query: ``error`` (class and message) plus both parts on their own."""
    return {
        "error": describe_error(exc),
        "error_class": type(exc).__name__,
        "error_message": str(exc)[:MAX_ERROR_CHARS],
    }
