"""Module-level holder for the active RunContext.

The agent tool functions are passed by reference to the agent framework, so they
need a stable place to read/write run state. This module provides that singleton
without exposing a global at import time.
"""

from __future__ import annotations

from .models import RunContext

_active_context: RunContext | None = None


def set_active_context(context: RunContext) -> None:
    """Install the run context that tool functions will operate on."""
    global _active_context
    _active_context = context


def get_active_context() -> RunContext:
    """Return the active run context, raising if no run has been started."""
    if _active_context is None:
        raise RuntimeError("No active run context. Call set_active_context() first.")
    return _active_context


def build_redacted_text_view() -> str:
    """Reconstruct the visible document text after applying current redactions.

    Redacted spans are replaced with the literal token ``[REDACTED]`` so the
    reviewer agent can scan the textual view without OCR.
    """
    context = get_active_context()
    parts: list[str] = []
    current_page = -1
    for word in context.words:
        if word.page != current_page:
            if current_page != -1:
                parts.append("\n")
            parts.append(f"\n--- Page {word.page + 1} ---\n")
            current_page = word.page
        is_redacted = (word.page, word.index) in context.redacted_word_keys
        token = "[REDACTED]" if is_redacted else word.text
        parts.append(token)
    return " ".join(parts)
