"""Word-level text extraction from PDFs.

Uses ``pdfplumber`` for native (text-based) PDFs and falls back to Azure AI
Document Intelligence (``prebuilt-read``) for scanned/image-only PDFs.
"""

from __future__ import annotations

import statistics
from pathlib import Path

import pdfplumber
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeResult

from .azure_clients import get_azure_credential, resolve_document_intelligence_endpoint
from .config import MIN_PLAUSIBLE_WORD_HEIGHT_POINTS
from .logger import log
from .models import PageWord


def _extract_words_via_ocr(pdf_path: Path) -> tuple[list[PageWord], list[tuple[float, float]]]:
    """OCR-based extraction via Azure AI Document Intelligence ``prebuilt-read``.

    Returns words with bounding boxes in PDF points so the rest of the pipeline
    is unit-consistent with the pdfplumber path.
    """
    endpoint = resolve_document_intelligence_endpoint()
    if not endpoint:
        raise RuntimeError(
            "Scanned/image-only PDF detected and no Document Intelligence "
            "endpoint configured. Set AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT or "
            "AI_FOUNDRY_PROJECT_ENDPOINT in .env."
        )

    log.info("OCR fallback: calling Document Intelligence (prebuilt-read) at %s", endpoint)
    client = DocumentIntelligenceClient(endpoint=endpoint, credential=get_azure_credential())

    with open(pdf_path, "rb") as fp:
        pdf_bytes = fp.read()

    poller = client.begin_analyze_document(
        model_id="prebuilt-read",
        body=pdf_bytes,
        content_type="application/pdf",
    )
    result: AnalyzeResult = poller.result()

    words: list[PageWord] = []
    page_sizes: list[tuple[float, float]] = []
    for page_index, page in enumerate(result.pages or []):
        # DI returns units in inches/pixels/points. Convert to PDF points.
        unit = (page.unit or "inch").lower()
        if unit == "inch":
            scale = 72.0
        elif unit == "pixel":
            # 96 DPI is DI's default for image inputs.
            scale = 72.0 / 96.0
        else:  # "point"
            scale = 1.0

        page_width = float(page.width or 0) * scale
        page_height = float(page.height or 0) * scale
        page_sizes.append((page_width, page_height))

        for word_index, word in enumerate(page.words or []):
            polygon = word.polygon  # [x1,y1,x2,y2,x3,y3,x4,y4]
            xs = [polygon[k] * scale for k in (0, 2, 4, 6)]
            ys = [polygon[k] * scale for k in (1, 3, 5, 7)]
            words.append(
                PageWord(
                    page=page_index,
                    index=word_index,
                    text=word.content,
                    x0=min(xs),
                    top=min(ys),
                    x1=max(xs),
                    bottom=max(ys),
                )
            )

    log.info("OCR extracted %d words across %d pages", len(words), len(page_sizes))
    return words, page_sizes


def _text_layer_geometry_is_usable(words: list[PageWord]) -> bool:
    """True when word boxes are large enough to redact against.

    A hidden search layer reports glyph boxes around a point tall, sitting at
    coordinates unrelated to the rendered text. Painting redactions over those
    boxes produces invisible slivers in the wrong place, so such a layer must
    not be trusted.
    """
    heights = [w.bottom - w.top for w in words]
    if not heights:
        return False
    return statistics.median(heights) >= MIN_PLAUSIBLE_WORD_HEIGHT_POINTS


def extract_words_with_bboxes(
    pdf_path: Path,
) -> tuple[list[PageWord], list[tuple[float, float]]]:
    """Extract word-level text + bounding boxes from a PDF.

    Falls back to OCR via Azure AI Document Intelligence when pdfplumber finds
    zero words (scanned/image-only PDFs) or when the embedded text layer's
    geometry is too degenerate to redact against.
    """
    words: list[PageWord] = []
    page_sizes: list[tuple[float, float]] = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_index, page in enumerate(pdf.pages):
            page_sizes.append((page.width, page.height))
            page_words = page.extract_words(use_text_flow=True)
            for word_index, word in enumerate(page_words):
                words.append(
                    PageWord(
                        page=page_index,
                        index=word_index,
                        text=word["text"],
                        x0=float(word["x0"]),
                        top=float(word["top"]),
                        x1=float(word["x1"]),
                        bottom=float(word["bottom"]),
                    )
                )

    if not words:
        log.warning(
            "pdfplumber found 0 words in %s — falling back to OCR via Azure AI Document Intelligence.",
            pdf_path.name,
        )
        return _extract_words_via_ocr(pdf_path)

    if not _text_layer_geometry_is_usable(words):
        median_height = statistics.median([w.bottom - w.top for w in words])
        log.warning(
            "%s has a degenerate text layer (median word height %.2fpt < %.1fpt); "
            "redactions drawn against it would be misplaced. Falling back to OCR.",
            pdf_path.name,
            median_height,
            MIN_PLAUSIBLE_WORD_HEIGHT_POINTS,
        )
        return _extract_words_via_ocr(pdf_path)

    return words, page_sizes
