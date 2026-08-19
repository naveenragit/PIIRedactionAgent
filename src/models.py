"""Plain data models used across the redaction pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .document_context import DocumentContext


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
class PageRegion:
    """A non-text visual region (logo / image / figure) on a PDF page.

    Coordinates are in PDF points. ``strategy`` is resolved at extraction time
    and tells the renderer how to handle a redaction:

    - ``"inline"``      Paint a black rectangle over the region in place.
    - ``"page_split"``  The region is a large background/watermark; redacting
                        it would visually destroy overlaid foreground content,
                        so the renderer emits the fully-blacked page followed
                        by a clean reflow page containing only the non-redacted
                        foreground text.
    """

    page: int
    index: int
    kind: str  # "logo" | "image" | "figure"
    x0: float
    top: float
    x1: float
    bottom: float
    confidence: float = 0.0
    label: str | None = None
    strategy: str = "inline"


@dataclass
class RunContext:
    """Mutable state for one end-to-end redaction run."""

    input_pdf: Path
    iteration: int = 0
    words: list[PageWord] = field(default_factory=list)
    page_sizes: list[tuple[float, float]] = field(default_factory=list)
    redacted_word_keys: set[tuple[int, int]] = field(default_factory=set)
    regions: list[PageRegion] = field(default_factory=list)
    redacted_region_keys: set[tuple[int, int]] = field(default_factory=set)
    page_split_pages: set[int] = field(default_factory=set)
    # Pages that emitted a tagged supplemental page in the last render.
    supplemental_pages: set[int] = field(default_factory=set)
    current_pdf: Path | None = None
    history: list[dict] = field(default_factory=list)
    # Auto-detected client / preparer identity. Populated by the
    # orchestrator before agent turns run.
    doc_context: "DocumentContext | None" = None
    # Per-tool attribution counters used by the metrics module.
    # Shape: {"tool_name": {"calls": int, "words_added": int, "regions_added": int}}.
    tool_counters: dict[str, dict[str, int]] = field(default_factory=dict)
