"""Centralized configuration: paths, runtime limits, and environment loading."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Project layout
PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parent
SAMPLES_DIR = PROJECT_DIR / "samples"
OUTPUT_DIR = PROJECT_DIR / "output"

# Load environment variables. Prefers a project-local `.env` (next to main.py)
# so the deliverable is self-contained, but falls back to the first `.env`
# found while walking parent directories. This lets developers test against a
# repo-root `.env` without duplicating secrets into this folder.
_local_env = PROJECT_DIR / ".env"
if _local_env.exists():
    load_dotenv(_local_env)
else:
    for _candidate_dir in PROJECT_DIR.parents:
        _candidate_env = _candidate_dir / ".env"
        if _candidate_env.exists():
            load_dotenv(_candidate_env)
            break

# Orchestration limits. ``REDACTION_MAX_ITERATIONS`` env var overrides the
# default — useful when running against unusually large or noisy decks.
MIN_ITERATIONS = 2
try:
    MAX_ITERATIONS = max(MIN_ITERATIONS, int(os.getenv("REDACTION_MAX_ITERATIONS", "8")))
except ValueError:
    MAX_ITERATIONS = 8

# When the redactor uses `redact_bbox` to act on reviewer-supplied logo
# coordinates, inflate the box by this many PDF points on each side so we
# also cover halos / glow / anti-aliased edges around the visual mark.
REDACT_BBOX_INFLATE_POINTS = 5.0

# Logo-finding dedupe threshold (IoU). When a reviewer-side detection
# overlaps an already-redacted region by at least this fraction it is
# suppressed before being reported as a Logo miss — prevents the loop
# from re-flagging the same blacked-out region every iteration.
LOGO_DEDUPE_IOU_THRESHOLD = 0.4

# Visual rendering DPI for the redacted PDF (200 is a good readability/size tradeoff).
RENDER_DPI = 200

# Maximum characters of redacted text passed to the reviewer in one call.
REVIEWER_TEXT_CHAR_LIMIT = 60_000

# Default name for the input sample.
DEFAULT_INPUT_FILENAME = "sample_input.pdf"

# Some PDFs (typically print-to-PDF or scanned filings) carry a hidden text
# layer whose glyph boxes are near-zero-sized and positioned nowhere near the
# visible text. Redacting against such a layer paints slivers in the wrong
# place, so the extractor falls back to OCR when the median word height on
# most pages drops below this many points.
MIN_PLAUSIBLE_WORD_HEIGHT_POINTS = 3.0

# ---------------------------------------------------------------------------
# Logo / visual-region handling
# ---------------------------------------------------------------------------
# A visual region is classified as a background / watermark logo (and therefore
# rendered via the "page_split" strategy) when either of the following holds:
#   - The region area / page area ratio exceeds BG_AREA_RATIO, OR
#   - The region bounding box encloses at least BG_WORD_OVERLAP_THRESHOLD
#     foreground words that are not themselves part of the region.
BG_AREA_RATIO = 0.35
BG_WORD_OVERLAP_THRESHOLD = 8

# Minimum confidence to accept a logo detection from Document Intelligence
# layout figures during the reviewer pass.
LOGO_DETECTION_MIN_CONFIDENCE = 0.0

# ---------------------------------------------------------------------------
# Template logo propagation
# ---------------------------------------------------------------------------
# Detection services report a mark only on the pages where they happen to find
# it. A brand mark stamped at the same coordinates across many pages is a
# template element, so once a detection cluster repeats on this many pages it
# is replicated onto the pages where detection missed it.
TEMPLATE_MIN_PAGES = 3

# Two boxes on different pages count as the same mark at or above this IoU.
TEMPLATE_IOU_THRESHOLD = 0.6

# Only marks smaller than this share of the page are propagated. Guards against
# replicating page-body rasters or full-slide figures across the document.
TEMPLATE_MAX_AREA_RATIO = 0.25

# The preparer's own brand mark is expected to remain in the deliverable, so
# it is excluded from logo redaction. Flip to True to redact it as well.
REDACT_PREPARER_LOGO = False

# ---------------------------------------------------------------------------
# Supplemental pages
# ---------------------------------------------------------------------------
# When a redacted visual region (typically a background logo or watermark)
# sits behind foreground text that is NOT itself redacted, blacking out the
# region would also hide that legitimate text. In that case the renderer
# appends a tagged supplemental page immediately after the redacted page,
# carrying the page's non-redacted text on a clean canvas.
#
# A page qualifies once at least this many non-redacted words are overlapped
# by a redacted region.
SUPPLEMENTAL_MIN_OCCLUDED_WORDS = 1

# Banner drawn across the top of every supplemental page. ``{page}`` is the
# 1-based number of the redacted page this supplement belongs to.
SUPPLEMENTAL_PAGE_TAG = "SUPPLEMENTAL - non-redacted text from page {page}"

# Height of the reserved banner strip, in PDF points. Page content is
# compressed vertically to fit beneath it so no text is lost.
SUPPLEMENTAL_BANNER_HEIGHT_POINTS = 22.0

# Azure AI Vision Image Analysis — optional second source of logo detections.
# If set, the visual extractor calls Image Analysis on each rasterized page and
# fuses its Object + DenseCaption detections with Document Intelligence figures.
# NOTE: Caption / DenseCaptions are only available in a subset of Azure regions
# (East US, West US, France Central, North Europe, West Europe, Southeast Asia,
# East Asia, Korea Central). Outside those regions the extractor automatically
# falls back to the features the resource does support.
VISION_IOU_MERGE_THRESHOLD = 0.5
VISION_LOGO_KEYWORDS = ("logo", "brand", "emblem", "trademark", "insignia", "watermark")
VISION_ANALYSIS_DPI = 150  # rasterization DPI for the Vision API call

# Font candidates (in priority order) used by the page-split "reflow" page to
# re-render non-redacted text on a clean canvas. The first existing path wins;
# if none are present we fall back to PIL's built-in bitmap font.
REFLOW_FONT_CANDIDATES = (
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/Library/Fonts/Arial.ttf",
)

# Ensure output directory exists at import time so logging/artifacts can be written.
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
