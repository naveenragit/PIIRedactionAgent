"""Render a redacted PDF as a flattened raster image PDF with black-box overlays.

The output is a flattened raster PDF, so the original text stream is gone —
redactions cannot be reversed via copy/paste or text extraction, and the black
rectangles are visually unambiguous.
"""

from __future__ import annotations

from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image, ImageDraw

from .config import RENDER_DPI
from .models import PageWord


# Pad applied around each word bounding box (in PDF points) before scaling
# to pixels. Keeps descenders covered and avoids hairline gaps.
_BBOX_PAD_POINTS = 1.0


def render_redacted_pdf(
    source_pdf: Path,
    words: list[PageWord],
    redacted_word_keys: set[tuple[int, int]],
    output_path: Path,
) -> None:
    """Rasterize ``source_pdf`` and overlay black rectangles over redacted words.

    Args:
        source_pdf: Original PDF to render from.
        words: All extracted words (with bounding boxes in PDF points).
        redacted_word_keys: Set of ``(page_index, word_index)`` pairs to mask.
        output_path: Destination PDF path.
    """
    word_lookup = {(w.page, w.index): w for w in words}
    redactions_by_page: dict[int, list[PageWord]] = {}
    for key in redacted_word_keys:
        word = word_lookup.get(key)
        if word is not None:
            redactions_by_page.setdefault(word.page, []).append(word)

    points_to_pixels = RENDER_DPI / 72.0
    pdf = pdfium.PdfDocument(str(source_pdf))
    page_images: list[Image.Image] = []
    try:
        for page_index in range(len(pdf)):
            page = pdf[page_index]
            pil_image = page.render(scale=points_to_pixels).to_pil().convert("RGB")
            draw = ImageDraw.Draw(pil_image)
            for word in redactions_by_page.get(page_index, []):
                # pdfplumber bbox is top-down in points; PIL is top-down in
                # pixels — so just scale.
                x0 = (word.x0 - _BBOX_PAD_POINTS) * points_to_pixels
                y0 = (word.top - _BBOX_PAD_POINTS) * points_to_pixels
                x1 = (word.x1 + _BBOX_PAD_POINTS) * points_to_pixels
                y1 = (word.bottom + _BBOX_PAD_POINTS) * points_to_pixels
                draw.rectangle([x0, y0, x1, y1], fill="black")
            page_images.append(pil_image)
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
