"""Render a redacted PDF as a flattened raster image PDF with black-box overlays.

The output is a flattened raster PDF, so the original text stream is gone —
redactions cannot be reversed via copy/paste or text extraction, and the black
rectangles are visually unambiguous.

For pages that contain a "background" logo or watermark (a large region whose
redaction would also obscure overlaid foreground content), the renderer emits
two pages in sequence:

1. The original page with all redactions applied (the background region is
   blacked out, which also covers the foreground content sitting on top).
2. A freshly synthesized "reflow" page containing only the **non-redacted**
   foreground words, re-rendered at their original positions on a clean white
   canvas.
"""

from __future__ import annotations

from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image, ImageDraw, ImageFont

from .config import REFLOW_FONT_CANDIDATES, RENDER_DPI
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


def _page_has_page_number(
    page_index: int,
    page_size_points: tuple[float, float],
    page_words: list[PageWord],
) -> bool:
    """Heuristic: does this page show a printed page number?

    Looks for a numeric token in the bottom 15% of the page whose value matches
    the 1-based page index. Tolerates simple decorations like "5", "- 5 -",
    "Page 5", "5/12".
    """
    _, page_height = page_size_points
    if page_height <= 0:
        return False
    footer_threshold = page_height * 0.85
    expected = str(page_index + 1)
    for word in page_words:
        if word.top < footer_threshold:
            continue
        token = word.text.strip().strip(".,;:()[]{}\"'`-/")
        if token == expected:
            return True
        # "5/12" style
        head = token.split("/", 1)[0]
        if head == expected:
            return True
    return False


def _render_reflow_page(
    page_size_points: tuple[float, float],
    visible_words: list[PageWord],
    points_to_pixels: float,
    footer_text: str,
) -> Image.Image:
    """Build a clean white page with ``visible_words`` placed at their original bboxes."""
    width_px = max(1, int(page_size_points[0] * points_to_pixels))
    height_px = max(1, int(page_size_points[1] * points_to_pixels))
    canvas = Image.new("RGB", (width_px, height_px), "white")
    draw = ImageDraw.Draw(canvas)

    for word in visible_words:
        box_height_points = max(1.0, word.bottom - word.top)
        # Use ~95% of the bbox height as the font size — visually approximates
        # the original glyphs without overflowing the box.
        font_size_pixels = int(box_height_points * points_to_pixels * 0.95)
        font = _get_font(font_size_pixels)
        x = word.x0 * points_to_pixels
        y = word.top * points_to_pixels
        try:
            draw.text((x, y), word.text, fill="black", font=font)
        except Exception:  # pragma: no cover - defensive
            continue

    if footer_text:
        footer_font = _get_font(max(10, int(8 * points_to_pixels)))
        try:
            draw.text(
                (10, height_px - int(14 * points_to_pixels)),
                footer_text,
                fill=(120, 120, 120),
                font=footer_font,
            )
        except Exception:  # pragma: no cover
            pass

    return canvas


def render_redacted_pdf(
    source_pdf: Path,
    words: list[PageWord],
    redacted_word_keys: set[tuple[int, int]],
    output_path: Path,
    regions: list[PageRegion] | None = None,
    redacted_region_keys: set[tuple[int, int]] | None = None,
    page_split_pages: set[int] | None = None,
) -> None:
    """Rasterize ``source_pdf`` and overlay black rectangles over redactions.

    Args:
        source_pdf: Original PDF to render from.
        words: All extracted words (with bounding boxes in PDF points).
        redacted_word_keys: Set of ``(page_index, word_index)`` pairs to mask.
        output_path: Destination PDF path.
        regions: Visual regions (logos / images / figures) discovered in the PDF.
        redacted_region_keys: Set of ``(page_index, region_index)`` pairs marked
            for redaction.
        page_split_pages: Pages whose redacted regions are background watermarks;
            each such page produces two output pages (blacked original + reflow).
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

            # If this page has a background logo redaction, add the reflow page.
            if page_index in page_split_pages and page_region_redactions:
                redacted_keys_on_page = {
                    (page_index, w.index) for w in page_word_redactions
                }
                visible_words = [
                    w
                    for w in words_by_page.get(page_index, [])
                    if (w.page, w.index) not in redacted_keys_on_page
                ]
                page_size = page.get_size()  # (width_points, height_points)
                page_size_points = (float(page_size[0]), float(page_size[1]))
                if _page_has_page_number(
                    page_index, page_size_points, words_by_page.get(page_index, [])
                ):
                    footer_text = f"Page {page_index + 1} (continued)"
                else:
                    footer_text = ""
                reflow_image = _render_reflow_page(
                    page_size_points=page_size_points,
                    visible_words=visible_words,
                    points_to_pixels=points_to_pixels,
                    footer_text=footer_text,
                )
                page_images.append(reflow_image)
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
