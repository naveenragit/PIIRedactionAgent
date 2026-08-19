"""Detect non-text visual regions (logos / images / figures) in a PDF.

Uses two complementary sources fused by IOU:

- **Azure AI Document Intelligence** ``prebuilt-layout`` — reports a ``figure``
  for each non-text region (charts, embedded images, logo blocks).
- **Azure AI Vision Image Analysis** — runs on each rasterized page and
  contributes Object + DenseCaption detections whose tags / captions match a
  logo-related keyword (``logo``, ``brand``, ``watermark``, ...).

Each region is then classified as either an inline graphic (small, doesn't
overlap much foreground text) or a background watermark (large, covers many
foreground words), which determines how the renderer should handle its
redaction.
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import pypdfium2 as pdfium
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeResult

from .azure_clients import get_azure_credential, resolve_document_intelligence_endpoint
from .config import (
    BG_AREA_RATIO,
    BG_WORD_OVERLAP_THRESHOLD,
    LOGO_DETECTION_MIN_CONFIDENCE,
    TEMPLATE_IOU_THRESHOLD,
    TEMPLATE_MAX_AREA_RATIO,
    TEMPLATE_MIN_PAGES,
    VISION_ANALYSIS_DPI,
    VISION_IOU_MERGE_THRESHOLD,
    VISION_LOGO_KEYWORDS,
)
from .logger import log
from .models import PageRegion, PageWord


# Feature set negotiated once per process. Caption / DenseCaptions are only
# available in some regions, so the richest supported set is discovered on the
# first analyzed page and reused thereafter.
_VISION_FEATURES: list | None = None


def _resolve_vision_endpoint() -> str | None:
    endpoint = os.getenv("AZURE_VISION_ENDPOINT", "").strip()
    if endpoint and not endpoint.startswith("<"):
        return endpoint.rstrip("/") + "/"
    log.warning(
        "AZURE_VISION_ENDPOINT is not set — Azure AI Vision logo detection is "
        "disabled; falling back to Document Intelligence figures only."
    )
    return None


def _unit_scale_to_points(unit: str | None) -> float:
    """Convert a Document Intelligence page unit to PDF points."""
    u = (unit or "inch").lower()
    if u == "inch":
        return 72.0
    if u == "pixel":
        return 72.0 / 96.0
    return 1.0  # "point"


def _bbox_from_polygon(polygon: list[float], scale: float) -> tuple[float, float, float, float]:
    """Return (x0, top, x1, bottom) in points from a DI polygon."""
    xs = [polygon[k] * scale for k in range(0, len(polygon), 2)]
    ys = [polygon[k] * scale for k in range(1, len(polygon), 2)]
    return min(xs), min(ys), max(xs), max(ys)


def _bbox_iou(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    """Intersection-over-union of two (x0, top, x1, bottom) boxes."""
    ix0 = max(a[0], b[0])
    iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2])
    iy1 = min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _word_in_region(word: PageWord, region: tuple[float, float, float, float]) -> bool:
    """True if the word's center lies inside the region bbox."""
    cx = (word.x0 + word.x1) / 2.0
    cy = (word.top + word.bottom) / 2.0
    x0, top, x1, bottom = region
    return x0 <= cx <= x1 and top <= cy <= bottom


def _classify_strategy(
    bbox: tuple[float, float, float, float],
    page_size: tuple[float, float],
    page_words: list[PageWord],
) -> str:
    """Decide whether a region should be redacted inline or via page-split."""
    page_w, page_h = page_size
    if page_w <= 0 or page_h <= 0:
        return "inline"
    x0, top, x1, bottom = bbox
    region_area = max(0.0, x1 - x0) * max(0.0, bottom - top)
    page_area = page_w * page_h
    area_ratio = region_area / page_area if page_area else 0.0

    enclosed_words = sum(1 for w in page_words if _word_in_region(w, bbox))

    if area_ratio >= BG_AREA_RATIO:
        return "page_split"
    if enclosed_words >= BG_WORD_OVERLAP_THRESHOLD:
        return "page_split"
    return "inline"


def _negotiate_vision_features(client, probe_image: bytes) -> list:
    """Return the richest Image Analysis feature set this resource supports.

    ``DenseCaptions`` drives the keyword-based logo matching but is only
    offered in a subset of Azure regions; when it is unavailable the call is
    downgraded to ``Objects`` alone rather than failing on every page.
    """
    global _VISION_FEATURES
    if _VISION_FEATURES is not None:
        return _VISION_FEATURES

    from azure.ai.vision.imageanalysis.models import VisualFeatures

    candidates = [
        [VisualFeatures.OBJECTS, VisualFeatures.DENSE_CAPTIONS],
        [VisualFeatures.OBJECTS],
    ]
    for features in candidates:
        try:
            client.analyze(image_data=probe_image, visual_features=features)
        except Exception as exc:  # pragma: no cover - service-side
            log.warning(
                "Vision feature set %s unavailable: %s",
                [str(f) for f in features],
                str(exc).splitlines()[0][:120],
            )
            continue
        if len(features) == 1:
            log.warning(
                "Vision running WITHOUT DenseCaptions (region limitation); "
                "logo recall from this source will be limited."
            )
        _VISION_FEATURES = features
        return features

    _VISION_FEATURES = []
    return []


# ---------------------------------------------------------------------------
# Azure AI Vision Image Analysis
# ---------------------------------------------------------------------------
def _vision_logo_detections(
    pdf_path: Path,
    page_sizes: list[tuple[float, float]],
) -> dict[int, list[tuple[tuple[float, float, float, float], float, str | None]]]:
    """Per page, return logo-like detections from Image Analysis.

    Output mapping: ``page_index -> list of (bbox_in_points, confidence, label)``.
    Returns an empty mapping if no Vision endpoint is configured or the call
    fails — callers must tolerate that.
    """
    endpoint = _resolve_vision_endpoint()
    if not endpoint:
        return {}

    # Lazy import so the SDK is optional.
    try:
        from azure.ai.vision.imageanalysis import ImageAnalysisClient
    except ImportError:  # pragma: no cover
        log.warning("azure-ai-vision-imageanalysis not installed; skipping Vision logo detection.")
        return {}

    client = ImageAnalysisClient(endpoint=endpoint, credential=get_azure_credential())
    keywords = tuple(k.lower() for k in VISION_LOGO_KEYWORDS)
    points_per_pixel = 72.0 / VISION_ANALYSIS_DPI

    detections: dict[int, list[tuple[tuple[float, float, float, float], float, str | None]]] = {}

    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        for page_index in range(len(pdf)):
            try:
                page = pdf[page_index]
                pil_image = (
                    page.render(scale=VISION_ANALYSIS_DPI / 72.0).to_pil().convert("RGB")
                )
                buf = io.BytesIO()
                pil_image.save(buf, format="JPEG", quality=85)
                image_bytes = buf.getvalue()

                features = _negotiate_vision_features(client, image_bytes)
                if not features:
                    log.warning(
                        "Vision Image Analysis unusable for this resource; "
                        "skipping Vision logo detection entirely."
                    )
                    break

                result = client.analyze(
                    image_data=image_bytes,
                    visual_features=features,
                )
            except Exception as exc:  # pragma: no cover - service-side
                log.warning("Vision analyze failed on page %d: %s", page_index, exc)
                continue

            page_hits: list[tuple[tuple[float, float, float, float], float, str | None]] = []

            objects = getattr(result, "objects", None)
            if objects and getattr(objects, "list", None):
                for obj in objects.list:
                    tags = getattr(obj, "tags", None) or []
                    matched_label = None
                    matched_conf = 0.0
                    for tag in tags:
                        name = (getattr(tag, "name", "") or "").lower()
                        if any(k in name for k in keywords):
                            matched_label = name
                            matched_conf = max(matched_conf, float(getattr(tag, "confidence", 0.0)))
                    if matched_label is None:
                        continue
                    if matched_conf < LOGO_DETECTION_MIN_CONFIDENCE:
                        continue
                    box = getattr(obj, "bounding_box", None)
                    if box is None:
                        continue
                    x0_pt = float(box.x) * points_per_pixel
                    y0_pt = float(box.y) * points_per_pixel
                    x1_pt = (float(box.x) + float(box.width)) * points_per_pixel
                    y1_pt = (float(box.y) + float(box.height)) * points_per_pixel
                    page_hits.append(((x0_pt, y0_pt, x1_pt, y1_pt), matched_conf, matched_label))

            captions = getattr(result, "dense_captions", None)
            if captions and getattr(captions, "list", None):
                for cap in captions.list:
                    text = (getattr(cap, "text", "") or "").lower()
                    if not any(k in text for k in keywords):
                        continue
                    conf = float(getattr(cap, "confidence", 0.0))
                    if conf < LOGO_DETECTION_MIN_CONFIDENCE:
                        continue
                    box = getattr(cap, "bounding_box", None)
                    if box is None:
                        continue
                    x0_pt = float(box.x) * points_per_pixel
                    y0_pt = float(box.y) * points_per_pixel
                    x1_pt = (float(box.x) + float(box.width)) * points_per_pixel
                    y1_pt = (float(box.y) + float(box.height)) * points_per_pixel
                    page_hits.append(((x0_pt, y0_pt, x1_pt, y1_pt), conf, text))

            if page_hits:
                detections[page_index] = page_hits
    finally:
        pdf.close()

    if detections:
        total = sum(len(v) for v in detections.values())
        log.info("Vision logo detections: %d hits across %d pages", total, len(detections))
    return detections


def _propagate_template_regions(
    regions: list[PageRegion],
    page_sizes: list[tuple[float, float]],
    words_by_page: dict[int, list[PageWord]],
) -> list[PageRegion]:
    """Replicate repeated small marks onto pages where detection missed them.

    Detection reports a logo only on the pages where it happens to be found,
    which leaves gaps on visually identical pages. Small boxes that recur at
    the same coordinates on at least ``TEMPLATE_MIN_PAGES`` pages are treated
    as template chrome and stamped onto every remaining page.
    """
    if not regions or not page_sizes:
        return []

    # Only small marks are eligible; large figures must never be replicated.
    candidates: list[PageRegion] = []
    for region in regions:
        page_w, page_h = page_sizes[region.page]
        page_area = page_w * page_h
        if page_area <= 0:
            continue
        area = max(0.0, region.x1 - region.x0) * max(0.0, region.bottom - region.top)
        if area / page_area <= TEMPLATE_MAX_AREA_RATIO:
            candidates.append(region)
    if not candidates:
        return []

    # Greedy clustering of boxes that land at the same spot on different pages.
    clusters: list[list[PageRegion]] = []
    for region in candidates:
        box = (region.x0, region.top, region.x1, region.bottom)
        for cluster in clusters:
            head = cluster[0]
            if _bbox_iou(box, (head.x0, head.top, head.x1, head.bottom)) >= TEMPLATE_IOU_THRESHOLD:
                cluster.append(region)
                break
        else:
            clusters.append([region])

    next_index_by_page: dict[int, int] = {}
    for region in regions:
        next_index_by_page[region.page] = max(
            next_index_by_page.get(region.page, 0), region.index + 1
        )

    added: list[PageRegion] = []
    for cluster in clusters:
        covered_pages = {r.page for r in cluster}
        if len(covered_pages) < TEMPLATE_MIN_PAGES:
            continue

        # Average the cluster to a stable representative box.
        count = len(cluster)
        rep = (
            sum(r.x0 for r in cluster) / count,
            sum(r.top for r in cluster) / count,
            sum(r.x1 for r in cluster) / count,
            sum(r.bottom for r in cluster) / count,
        )
        label = next((r.label for r in cluster if r.label), None)

        for page_index in range(len(page_sizes)):
            if page_index in covered_pages:
                continue
            already = any(
                _bbox_iou(rep, (r.x0, r.top, r.x1, r.bottom)) >= TEMPLATE_IOU_THRESHOLD
                for r in regions
                if r.page == page_index
            )
            if already:
                continue
            new_index = next_index_by_page.get(page_index, 0)
            next_index_by_page[page_index] = new_index + 1
            added.append(
                PageRegion(
                    page=page_index,
                    index=new_index,
                    kind="logo",
                    x0=rep[0],
                    top=rep[1],
                    x1=rep[2],
                    bottom=rep[3],
                    confidence=min(r.confidence for r in cluster),
                    label=label or "template-logo",
                    strategy=_classify_strategy(
                        rep, page_sizes[page_index], words_by_page.get(page_index, [])
                    ),
                )
            )

    if added:
        log.info(
            "Template propagation: +%d regions across %d pages (from %d repeated marks)",
            len(added),
            len({r.page for r in added}),
            sum(1 for c in clusters if len({r.page for r in c}) >= TEMPLATE_MIN_PAGES),
        )
    return added


def extract_visual_regions(
    pdf_path: Path,
    words: list[PageWord],
    page_sizes: list[tuple[float, float]],
) -> list[PageRegion]:
    """Return logo / image regions for ``pdf_path``.

    Fuses Document Intelligence ``prebuilt-layout`` figures with Azure AI Vision
    Image Analysis detections. Quietly returns an empty list if neither source
    is configured.
    """
    di_endpoint = resolve_document_intelligence_endpoint()
    di_result: AnalyzeResult | None = None
    if di_endpoint:
        log.info("Visual region extraction: DI prebuilt-layout at %s", di_endpoint)
        try:
            client = DocumentIntelligenceClient(
                endpoint=di_endpoint, credential=get_azure_credential()
            )
            with open(pdf_path, "rb") as fp:
                pdf_bytes = fp.read()
            poller = client.begin_analyze_document(
                model_id="prebuilt-layout",
                body=pdf_bytes,
                content_type="application/pdf",
            )
            di_result = poller.result()
        except Exception as exc:  # pragma: no cover
            log.warning("prebuilt-layout call failed (%s); continuing without DI figures.", exc)

    # Group words by page once for the classifier.
    words_by_page: dict[int, list[PageWord]] = {}
    for word in words:
        words_by_page.setdefault(word.page, []).append(word)

    # Collect DI bboxes per page.
    di_by_page: dict[int, list[tuple[tuple[float, float, float, float], float, str | None]]] = {}
    if di_result is not None:
        page_scale: dict[int, float] = {}
        for di_page in di_result.pages or []:
            page_index = int(di_page.page_number) - 1
            page_scale[page_index] = _unit_scale_to_points(di_page.unit)
        for figure in getattr(di_result, "figures", None) or []:
            for region in getattr(figure, "bounding_regions", None) or []:
                page_index = int(region.page_number) - 1
                if page_index < 0 or page_index >= len(page_sizes):
                    continue
                polygon = list(region.polygon or [])
                if len(polygon) < 8:
                    continue
                bbox = _bbox_from_polygon(polygon, page_scale.get(page_index, 72.0))
                di_by_page.setdefault(page_index, []).append((bbox, 1.0, None))

    # Vision detections, fused by IOU with the DI list.
    vision_by_page = _vision_logo_detections(pdf_path, page_sizes)

    regions: list[PageRegion] = []
    for page_index in range(len(page_sizes)):
        combined: list[tuple[tuple[float, float, float, float], float, str | None]] = list(
            di_by_page.get(page_index, [])
        )
        for v_bbox, v_conf, v_label in vision_by_page.get(page_index, []):
            merged = False
            for i, (c_bbox, c_conf, c_label) in enumerate(combined):
                if _bbox_iou(v_bbox, c_bbox) >= VISION_IOU_MERGE_THRESHOLD:
                    # Union of bboxes, prefer Vision's label.
                    merged_bbox = (
                        min(v_bbox[0], c_bbox[0]),
                        min(v_bbox[1], c_bbox[1]),
                        max(v_bbox[2], c_bbox[2]),
                        max(v_bbox[3], c_bbox[3]),
                    )
                    combined[i] = (
                        merged_bbox,
                        max(v_conf, c_conf),
                        v_label or c_label,
                    )
                    merged = True
                    break
            if not merged:
                combined.append((v_bbox, v_conf, v_label))

        for idx, (bbox, conf, label) in enumerate(combined):
            strategy = _classify_strategy(
                bbox,
                page_sizes[page_index],
                words_by_page.get(page_index, []),
            )
            regions.append(
                PageRegion(
                    page=page_index,
                    index=idx,
                    kind="logo" if (label or strategy == "inline") else "figure",
                    x0=bbox[0],
                    top=bbox[1],
                    x1=bbox[2],
                    bottom=bbox[3],
                    confidence=conf,
                    label=label,
                    strategy=strategy,
                )
            )

    regions.extend(_propagate_template_regions(regions, page_sizes, words_by_page))

    log.info(
        "Visual region extraction: %d regions across %d pages (page_split=%d)",
        len(regions),
        len({r.page for r in regions}),
        sum(1 for r in regions if r.strategy == "page_split"),
    )
    return regions


def _di_figures_by_page(
    pdf_path: Path,
) -> dict[int, list[tuple[tuple[float, float, float, float], float, str | None]]]:
    """Return DI ``prebuilt-layout`` figures grouped by 0-based page index."""
    out: dict[int, list[tuple[tuple[float, float, float, float], float, str | None]]] = {}
    di_endpoint = resolve_document_intelligence_endpoint()
    if not di_endpoint:
        return out
    try:
        client = DocumentIntelligenceClient(endpoint=di_endpoint, credential=get_azure_credential())
        with open(pdf_path, "rb") as fp:
            pdf_bytes = fp.read()
        poller = client.begin_analyze_document(
            model_id="prebuilt-layout",
            body=pdf_bytes,
            content_type="application/pdf",
        )
        result: AnalyzeResult = poller.result()
    except Exception as exc:  # pragma: no cover
        log.warning("Reviewer logo detection (DI) failed: %s", exc)
        return out

    page_scale: dict[int, float] = {}
    for di_page in result.pages or []:
        page_index = int(di_page.page_number) - 1
        page_scale[page_index] = _unit_scale_to_points(di_page.unit)

    for figure in getattr(result, "figures", None) or []:
        for region in getattr(figure, "bounding_regions", None) or []:
            page_index = int(region.page_number) - 1
            polygon = list(region.polygon or [])
            if len(polygon) < 8:
                continue
            scale = page_scale.get(page_index, 72.0)
            bbox = _bbox_from_polygon(polygon, scale)
            out.setdefault(page_index, []).append((bbox, 1.0, None))
    return out


def detect_remaining_logos(pdf_path: Path) -> list[dict]:
    """Run logo detection on the current redacted PDF and return findings.

    Runs **both** Azure AI Vision Image Analysis on each rasterized page
    **and** Document Intelligence ``prebuilt-layout`` figures, then unions
    the two sets — merging boxes whose IoU exceeds
    ``VISION_IOU_MERGE_THRESHOLD`` (taking the union bbox, max confidence,
    and preferring Vision's label). Vision catches stylized raster logos
    that DI misses; DI catches structural figure blocks that Vision scores
    below threshold. The union maximizes recall.

    Returns ``[{"page", "bbox": [x0, top, x1, bottom], "confidence",
    "label"}]`` with coordinates in PDF points.
    """
    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        page_sizes = [(float(p.get_size()[0]), float(p.get_size()[1])) for p in pdf]
    finally:
        pdf.close()

    vision_by_page = _vision_logo_detections(pdf_path, page_sizes)
    di_by_page = _di_figures_by_page(pdf_path)

    findings: list[dict] = []
    pages = set(vision_by_page) | set(di_by_page)
    for page_index in sorted(pages):
        combined: list[tuple[tuple[float, float, float, float], float, str | None]] = list(
            di_by_page.get(page_index, [])
        )
        for v_bbox, v_conf, v_label in vision_by_page.get(page_index, []):
            merged = False
            for i, (c_bbox, c_conf, c_label) in enumerate(combined):
                if _bbox_iou(v_bbox, c_bbox) >= VISION_IOU_MERGE_THRESHOLD:
                    combined[i] = (
                        (
                            min(v_bbox[0], c_bbox[0]),
                            min(v_bbox[1], c_bbox[1]),
                            max(v_bbox[2], c_bbox[2]),
                            max(v_bbox[3], c_bbox[3]),
                        ),
                        max(v_conf, c_conf),
                        v_label or c_label,
                    )
                    merged = True
                    break
            if not merged:
                combined.append((v_bbox, v_conf, v_label))

        for bbox, conf, label in combined:
            findings.append(
                {
                    "page": page_index,
                    "bbox": [bbox[0], bbox[1], bbox[2], bbox[3]],
                    "confidence": conf,
                    "label": label,
                }
            )

    log.info(
        "detect_remaining_logos: vision=%d, di=%d, fused=%d",
        sum(len(v) for v in vision_by_page.values()),
        sum(len(v) for v in di_by_page.values()),
        len(findings),
    )
    return findings

