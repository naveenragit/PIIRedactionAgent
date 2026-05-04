"""Plain data models used across the redaction pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PageWord:
    """A single word extracted from a PDF page, with its bounding box in PDF points."""

    page: int
    index: int
    text: str
    x0: float
    top: float
    x1: float
    bottom: float


@dataclass
class RunContext:
    """Mutable state for one end-to-end redaction run."""

    input_pdf: Path
    iteration: int = 0
    words: list[PageWord] = field(default_factory=list)
    page_sizes: list[tuple[float, float]] = field(default_factory=list)
    redacted_word_keys: set[tuple[int, int]] = field(default_factory=set)
    current_pdf: Path | None = None
    history: list[dict] = field(default_factory=list)
