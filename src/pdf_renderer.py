"""Render a redacted PDF as a flattened raster image PDF with black-box overlays.

The output is a flattened raster PDF, so the original text stream is gone —
redactions cannot be reversed via copy/paste or text extraction, and the black
rectangles are visually unambiguous.

When a redacted visual region (a logo or watermark) sits behind foreground text
that is *not* itself redacted, blacking out the region would also hide that
legitimate text. For those pages the renderer emits two pages in sequence:

1. The original page with all redactions applied (the region is blacked out,
   which also covers any foreground content sitting on top of it).
2. A tagged "supplemental" page carrying only the **non-redacted** words of
   that page, re-rendered at their original positions on a clean white canvas
   beneath a banner identifying which page they came from.
"""

from __future__ import annotations

from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image, ImageDraw, ImageFont

from .config import (
    REFLOW_FONT_CANDIDATES,
    RENDER_DPI,
    SUPPLEMENTAL_BANNER_HEIGHT_POINTS,
    SUPPLEMENTAL_MIN_OCCLUDED_WORDS,
    SUPPLEMENTAL_PAGE_TAG,
)
from .logger import log
from .models import PageRegion, PageWord


# Pad applied around each word bounding box (in PDF points) before scaling
# to pixels. Keeps descenders covered and avoids hairline gaps.
_BBOX_PAD_POINTS = 1.0

# Cached font lookup so we don't re-probe the filesystem per page.
_FONT_CACHE: dict[int, ImageFont.ImageFont] = {}
_FONT_PATH: str | None = None
_FONT_PATH_RESOLVED = False


def _resolve_font_path() -> str | None:
    """Return the first font path from REFLOW_FONT_CANDIDATES that exists."""
    global _FONT_PATH, _FONT_PATH_RESOLVED
    if _FONT_PATH_RESOLVED:
        return _FONT_PATH
    for candidate in REFLOW_FONT_CANDIDATES:
        if Path(candidate).exists():
            _FONT_PATH = candidate
            break
    _FONT_PATH_RESOLVED = True
    if _FONT_PATH is None:
        log.warning(
            "No TTF font found in REFLOW_FONT_CANDIDATES; "
            "page-split reflow will use PIL's bitmap font."
        )
    return _FONT_PATH


def _get_font(pixel_size: int) -> ImageFont.ImageFont:
    """Return a cached PIL font sized to ``pixel_size`` pixels (best-effort)."""
    pixel_size = max(6, int(pixel_size))
    if pixel_size in _FONT_CACHE:
        return _FONT_CACHE[pixel_size]
    font_path = _resolve_font_path()
    if font_path:
        try:
            font = ImageFont.truetype(font_path, pixel_size)
        except OSError:
            font = ImageFont.load_default()
    else:
        font = ImageFont.load_default()
    _FONT_CACHE[pixel_size] = font
    return font


def _draw_redacted_overlay(
    image: Image.Image,
    word_redactions: list[PageWord],
    region_redactions: list[PageRegion],
    points_to_pixels: float,
) -> None:
    """Paint black rectangles for the given word and region redactions."""
    draw = ImageDraw.Draw(image)
    for word in word_redactions:
        # pdfplumber bbox is top-down in points; PIL is top-down in pixels —
        # so just scale.
        x0 = (word.x0 - _BBOX_PAD_POINTS) * points_to_pixels
        y0 = (word.top - _BBOX_PAD_POINTS) * points_to_pixels
        x1 = (word.x1 + _BBOX_PAD_POINTS) * points_to_pixels
        y1 = (word.bottom + _BBOX_PAD_POINTS) * points_to_pixels
        draw.rectangle([x0, y0, x1, y1], fill="black")
    for region in region_redactions:
        x0 = (region.x0 - _BBOX_PAD_POINTS) * points_to_pixels
        y0 = (region.top - _BBOX_PAD_POINTS) * points_to_pixels
        x1 = (region.x1 + _BBOX_PAD_POINTS) * points_to_pixels
        y1 = (region.bottom + _BBOX_PAD_POINTS) * points_to_pixels
        draw.rectangle([x0, y0, x1, y1], fill="black")


# Above this many words per distinct text row, a page's coordinates are
# treated as unusable for positional replay (see _layout_is_degenerate).
_DEGENERATE_WORDS_PER_ROW = 25


def _layout_is_degenerate(words: list[PageWord]) -> bool:
    """True when word y-positions collapse onto very few rows.

    Some print-to-PDF documents carry correct text in an unusable layout
    (hundreds of words sharing one baseline). Replaying those words at their
    original coordinates yields an illegible smear, so the supplemental page
    reflows them into wrapped lines instead.
    """
    if not words:
        return False
    rows = {round(word.top) for word in words}
    return len(words) / max(1, len(rows)) > _DEGENERATE_WORDS_PER_ROW


def _draw_wrapped_words(
    draw: ImageDraw.ImageDraw,
    words: list[PageWord],
    width_px: int,
    height_px: int,
    start_y_px: int,
    points_to_pixels: float,
) -> None:
    """Draw ``words`` in reading order as wrapped lines of body text."""
    font_px = max(9, int(9 * points_to_pixels))
    font = _get_font(font_px)
    margin = int(14 * points_to_pixels)
    line_height = int(font_px * 1.4)
    max_width = width_px - (2 * margin)
    bottom_limit = height_px - margin - line_height

    y = start_y_px + margin
    line: list[str] = []
    for word in sorted(words, key=lambda w: w.index):
        candidate = line + [word.text]
        if line and draw.textlength(" ".join(candidate), font=font) > max_width:
            draw.text((margin, y), " ".join(line), fill="black", font=font)
            y += line_height
            if y > bottom_limit:
                return
            line = [word.text]
        else:
            line = candidate
    if line and y <= bottom_limit:
        draw.text((margin, y), " ".join(line), fill="black", font=font)


def _occluded_visible_words(
    page_words: list[PageWord],
    redacted_word_keys: set[tuple[int, int]],
    redacted_regions: list[PageRegion],
) -> list[PageWord]:
    """Return non-redacted words whose bbox intersects a redacted region.

    These are words that would become unreadable once the region is blacked
    out, so they are the reason a supplemental page is needed.
    """
    if not redacted_regions:
        return []
    occluded: list[PageWord] = []
    for word in page_words:
        if (word.page, word.index) in redacted_word_keys:
            continue
        for region in redacted_regions:
            if (
                word.x0 < region.x1
                and word.x1 > region.x0
                and word.top < region.bottom
                and word.bottom > region.top
            ):
                occluded.append(word)
                break
    return occluded


def _render_supplemental_page(
    page_size_points: tuple[float, float],
    visible_words: list[PageWord],
    points_to_pixels: float,
    banner_text: str,
) -> Image.Image:
    """Build a tagged white page carrying ``visible_words`` at their original bboxes.

    Content is compressed vertically to sit beneath the banner strip so that no
    word is pushed off the bottom of the page.
    """
    width_px = max(1, int(page_size_points[0] * points_to_pixels))
    height_px = max(1, int(page_size_points[1] * points_to_pixels))
    canvas = Image.new("RGB", (width_px, height_px), "white")
    draw = ImageDraw.Draw(canvas)

    banner_px = max(1, int(SUPPLEMENTAL_BANNER_HEIGHT_POINTS * points_to_pixels))
    draw.rectangle([0, 0, width_px, banner_px], fill=(0, 0, 0))
    banner_font = _get_font(int(banner_px * 0.55))
    try:
        draw.text(
            (int(6 * points_to_pixels), int(banner_px * 0.22)),
            banner_text,
            fill="white",
            font=banner_font,
        )
    except Exception:  # pragma: no cover - defensive
        pass

    page_height_px = max(1.0, page_size_points[1] * points_to_pixels)
    squeeze = (page_height_px - banner_px) / page_height_px

    if _layout_is_degenerate(visible_words):
        _draw_wrapped_words(
            draw, visible_words, width_px, height_px, banner_px, points_to_pixels
        )
        return canvas

    for word in visible_words:
        box_height_points = max(1.0, word.bottom - word.top)
        # Use ~95% of the bbox height as the font size — visually approximates
        # the original glyphs without overflowing the box.
        font_size_pixels = int(box_height_points * points_to_pixels * 0.95 * squeeze)
        font = _get_font(font_size_pixels)
        x = word.x0 * points_to_pixels
        y = banner_px + (word.top * points_to_pixels * squeeze)
        try:
            draw.text((x, y), word.text, fill="black", font=font)
        except Exception:  # pragma: no cover - defensive
            continue

    return canvas


def render_redacted_pdf(
    source_pdf: Path,
    words: list[PageWord],
    redacted_word_keys: set[tuple[int, int]],
    output_path: Path,
    regions: list[PageRegion] | None = None,
    redacted_region_keys: set[tuple[int, int]] | None = None,
    page_split_pages: set[int] | None = None,
) -> set[int]:
    """Rasterize ``source_pdf`` and overlay black rectangles over redactions.

    Args:
        source_pdf: Original PDF to render from.
        words: All extracted words (with bounding boxes in PDF points).
        redacted_word_keys: Set of ``(page_index, word_index)`` pairs to mask.
        output_path: Destination PDF path.
        regions: Visual regions (logos / images / figures) discovered in the PDF.
        redacted_region_keys: Set of ``(page_index, region_index)`` pairs marked
            for redaction.
        page_split_pages: Pages already known to carry a background watermark;
            these always produce a supplemental page.

    Returns:
        The set of 0-based source page indices that emitted a supplemental page.
    """
    regions = regions or []
    redacted_region_keys = redacted_region_keys or set()
    page_split_pages = page_split_pages or set()

    word_lookup = {(w.page, w.index): w for w in words}
    redactions_by_page: dict[int, list[PageWord]] = {}
    for key in redacted_word_keys:
        word = word_lookup.get(key)
        if word is not None:
            redactions_by_page.setdefault(word.page, []).append(word)

    region_lookup = {(r.page, r.index): r for r in regions}
    redacted_regions_by_page: dict[int, list[PageRegion]] = {}
    for key in redacted_region_keys:
        region = region_lookup.get(key)
        if region is not None:
            redacted_regions_by_page.setdefault(region.page, []).append(region)

    # Words grouped by page — used to build the reflow page.
    words_by_page: dict[int, list[PageWord]] = {}
    for w in words:
        words_by_page.setdefault(w.page, []).append(w)

    points_to_pixels = RENDER_DPI / 72.0
    pdf = pdfium.PdfDocument(str(source_pdf))
    page_images: list[Image.Image] = []
    supplemental_pages: set[int] = set()
    try:
        for page_index in range(len(pdf)):
            page = pdf[page_index]
            pil_image = page.render(scale=points_to_pixels).to_pil().convert("RGB")

            page_word_redactions = redactions_by_page.get(page_index, [])
            page_region_redactions = redacted_regions_by_page.get(page_index, [])

            _draw_redacted_overlay(
                pil_image,
                page_word_redactions,
                page_region_redactions,
                points_to_pixels,
            )
            page_images.append(pil_image)

            # A supplemental page is needed when a redacted region hides text
            # that is not itself redacted.
            all_page_words = words_by_page.get(page_index, [])
            occluded = _occluded_visible_words(
                all_page_words, redacted_word_keys, page_region_redactions
            )
            needs_supplement = bool(page_region_redactions) and (
                page_index in page_split_pages
                or len(occluded) >= SUPPLEMENTAL_MIN_OCCLUDED_WORDS
            )
            if not needs_supplement:
                continue

            visible_words = [
                w for w in all_page_words if (w.page, w.index) not in redacted_word_keys
            ]
            if not visible_words:
                continue

            page_size = page.get_size()  # (width_points, height_points)
            page_size_points = (float(page_size[0]), float(page_size[1]))
            page_images.append(
                _render_supplemental_page(
                    page_size_points=page_size_points,
                    visible_words=visible_words,
                    points_to_pixels=points_to_pixels,
                    banner_text=SUPPLEMENTAL_PAGE_TAG.format(page=page_index + 1),
                )
            )
            supplemental_pages.add(page_index)
    finally:
        pdf.close()

    if not page_images:
        raise RuntimeError("No pages rendered from input PDF.")

    page_images[0].save(
        output_path,
        save_all=True,
        append_images=page_images[1:],
        format="PDF",
        resolution=float(RENDER_DPI),
    )

    if supplemental_pages:
        log.info(
            "Supplemental pages appended after source pages: %s",
            sorted(p + 1 for p in supplemental_pages),
        )
    return supplemental_pages
