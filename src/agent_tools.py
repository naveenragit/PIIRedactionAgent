"""Tool functions exposed to the redactor and reviewer agents.

Each function reads/writes the active :class:`RunContext` via
``redaction_state``. Tools are kept module-level (not bound methods) because the
agent framework introspects their signatures and docstrings to build tool
schemas.
"""

from __future__ import annotations

import json
import os
from typing import Annotated

from azure.ai.textanalytics import TextAnalyticsClient
from pydantic import Field

from .azure_clients import get_azure_credential
from .config import (
    LOGO_DEDUPE_IOU_THRESHOLD,
    OUTPUT_DIR,
    REDACT_BBOX_INFLATE_POINTS,
    REVIEWER_TEXT_CHAR_LIMIT,
)
from .logger import log
from .models import PageRegion
from .pdf_renderer import render_redacted_pdf
from .pdf_visual_extractor import detect_remaining_logos
from .redaction_state import build_redacted_text_view, get_active_context, record_tool_use


def _loads_spans_tolerant(raw: str, tool_name: str) -> tuple[object | None, int]:
    """Parse a span payload, salvaging model output that was cut off mid-array.

    Large tool-call arguments get truncated by the model's output limit, which
    would otherwise discard an entire batch of valid redactions. Returns the
    parsed value plus the number of characters dropped during salvage.
    """
    try:
        return json.loads(raw), 0
    except json.JSONDecodeError:
        pass

    # Walk the payload tracking brace depth (ignoring braces inside strings) to
    # find the end of the last complete top-level object.
    depth = 0
    in_string = False
    escaped = False
    last_complete = -1
    for position, char in enumerate(raw):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                last_complete = position

    if last_complete < 0:
        return None, 0

    head = raw.lstrip()
    repaired = raw[: last_complete + 1]
    if head.startswith("["):
        repaired += "]"

    try:
        salvaged = json.loads(repaired)
    except json.JSONDecodeError:
        return None, 0

    dropped = max(0, len(raw) - (last_complete + 1))
    log.warning(
        "%s: payload was truncated; salvaged %d complete entr%s and dropped %d trailing chars.",
        tool_name,
        len(salvaged) if isinstance(salvaged, list) else 1,
        "y" if isinstance(salvaged, list) and len(salvaged) == 1 else "ies",
        dropped,
    )
    return salvaged, dropped


# ---------------------------------------------------------------------------
# Redactor tools
# ---------------------------------------------------------------------------
def extract_pdf_words(
    only_unredacted: Annotated[
        bool,
        Field(
            description=(
                "If true, omit already-redacted words. Use this on iterations "
                "after the first."
            )
        ),
    ] = False,
) -> str:
    """Return word-level text + indices for the input PDF.

    Output JSON shape::

      [{"page": 0, "words": [{"i": 0, "text": "...", "redacted": false}, ...]}, ...]

    Use the ``(page, i)`` pair when calling :func:`apply_redactions`.
    """
    context = get_active_context()
    pages_out: dict[int, list[dict]] = {}
    for word in context.words:
        is_redacted = (word.page, word.index) in context.redacted_word_keys
        if only_unredacted and is_redacted:
            continue
        pages_out.setdefault(word.page, []).append(
            {"i": word.index, "text": word.text, "redacted": is_redacted}
        )

    result = [{"page": page, "words": words} for page, words in sorted(pages_out.items())]
    payload = json.dumps(result)
    total_words = sum(len(p["words"]) for p in result)
    log.info(
        "extract_pdf_words(only_unredacted=%s) -> %d pages, %d words",
        only_unredacted,
        len(result),
        total_words,
    )
    return payload


def apply_redactions(
    spans_json: Annotated[
        str,
        Field(
            description=(
                'JSON list like [{"page":0,"word_indices":[3,4,5,12]}, ...]. '
                "Indices come from extract_pdf_words. Keep each call to at "
                "most ~10 pages and split larger sets across several calls — "
                "long payloads get truncated in transit and lose redactions."
            )
        ),
    ],
) -> str:
    """Record redactions for the listed words and rebuild the visual PDF.

    Returns the path to the freshly saved redacted PDF for this iteration.
    """
    context = get_active_context()
    log.debug("apply_redactions called with %d bytes", len(spans_json))

    spans, _ = _loads_spans_tolerant(spans_json, "apply_redactions")
    if spans is None:
        log.error("apply_redactions: unparseable JSON | head=%r", spans_json[:200])
        return json.dumps(
            {
                "error": "invalid JSON",
                "hint": (
                    "The payload could not be parsed, most likely because it was "
                    "too long and got cut off. Resend in smaller batches of at "
                    "most 10 pages per call."
                ),
            }
        )

    # Be tolerant of agents that wrap the list in a dict like {"spans":[...]}.
    if isinstance(spans, dict):
        for key in ("spans", "redactions", "items", "data"):
            value = spans.get(key)
            if isinstance(value, list):
                spans = value
                break
        else:
            list_values = [v for v in spans.values() if isinstance(v, list)]
            spans = list_values[0] if list_values else []

    if not isinstance(spans, list):
        log.error("apply_redactions: expected list, got %s", type(spans).__name__)
        return json.dumps({"error": "expected JSON list of {page, word_indices} entries"})

    word_lookup = {(w.page, w.index): w for w in context.words}
    added = 0
    invalid = 0

    for entry in spans:
        if not isinstance(entry, dict) or "page" not in entry:
            invalid += 1
            continue
        try:
            page = int(entry["page"])
        except (TypeError, ValueError):
            invalid += 1
            continue
        for raw_index in entry.get("word_indices", []):
            try:
                word_index = int(raw_index)
            except (TypeError, ValueError):
                invalid += 1
                continue
            if (page, word_index) not in word_lookup:
                invalid += 1
                continue
            key = (page, word_index)
            if key not in context.redacted_word_keys:
                context.redacted_word_keys.add(key)
                added += 1

    output_path = OUTPUT_DIR / f"iteration_{context.iteration}_redacted.pdf"
    context.supplemental_pages = render_redacted_pdf(
        source_pdf=context.input_pdf,
        words=context.words,
        redacted_word_keys=context.redacted_word_keys,
        output_path=output_path,
        regions=context.regions,
        redacted_region_keys=context.redacted_region_keys,
        page_split_pages=context.page_split_pages,
    )
    context.current_pdf = output_path

    record_tool_use("apply_redactions", words_added=added)
    log.info(
        "apply_redactions: added=%d total=%d invalid=%d -> %s",
        added,
        len(context.redacted_word_keys),
        invalid,
        output_path.name,
    )

    return json.dumps(
        {
            "output_path": str(output_path),
            "words_added_this_call": added,
            "total_redacted": len(context.redacted_word_keys),
            "invalid_entries": invalid,
        }
    )


def redact_all_matching_terms(
    terms_json: Annotated[
        str,
        Field(
            description=(
                'JSON list of terms to redact across the entire document, e.g. '
                '["Tribune", "TRB", "Sam Zell"]. Matching is case-insensitive '
                "and matches whole words. Use this for client identity terms "
                "(name, ticker, abbreviations, executive names) so EVERY "
                "occurrence on EVERY page is redacted deterministically."
            )
        ),
    ],
) -> str:
    """Sweep all pages and redact every word matching any of the given terms.

    Multi-word terms (e.g. ``"Sam Zell"``) are matched as a contiguous word
    sequence on the same page. Returns counts and the path to the regenerated
    redacted PDF.
    """
    context = get_active_context()

    try:
        terms = json.loads(terms_json)
    except json.JSONDecodeError as exc:
        return json.dumps({"error": f"invalid JSON: {exc}"})

    if not isinstance(terms, list) or not all(isinstance(term, str) for term in terms):
        return json.dumps({"error": "expected JSON list of strings"})

    def _tokenize(text: str) -> list[str]:
        """Lower-case and strip surrounding punctuation from each token."""
        return [
            tok.strip(".,;:()[]{}\"'`").lower()
            for tok in text.split()
            if tok.strip()
        ]

    term_token_lists: list[list[str]] = [t for t in (_tokenize(s) for s in terms) if t]
    if not term_token_lists:
        return json.dumps({"error": "no usable terms after normalization"})

    # Group words by page in original order.
    words_by_page: dict[int, list] = {}
    for word in context.words:
        words_by_page.setdefault(word.page, []).append(word)

    added = 0
    match_counts_by_term: dict[str, int] = {}
    for page_words in words_by_page.values():
        page_words_sorted = sorted(page_words, key=lambda w: w.index)
        normalized = [
            w.text.strip(".,;:()[]{}\"'`$").lower() for w in page_words_sorted
        ]
        page_word_count = len(normalized)
        for token_list in term_token_lists:
            term_length = len(token_list)
            if term_length == 0 or term_length > page_word_count:
                continue
            for start in range(0, page_word_count - term_length + 1):
                if normalized[start : start + term_length] == token_list:
                    for offset in range(term_length):
                        word = page_words_sorted[start + offset]
                        key = (word.page, word.index)
                        if key not in context.redacted_word_keys:
                            context.redacted_word_keys.add(key)
                            added += 1
                    label = " ".join(token_list)
                    match_counts_by_term[label] = match_counts_by_term.get(label, 0) + 1

    output_path = OUTPUT_DIR / f"iteration_{context.iteration}_redacted.pdf"
    context.supplemental_pages = render_redacted_pdf(
        source_pdf=context.input_pdf,
        words=context.words,
        redacted_word_keys=context.redacted_word_keys,
        output_path=output_path,
        regions=context.regions,
        redacted_region_keys=context.redacted_region_keys,
        page_split_pages=context.page_split_pages,
    )
    context.current_pdf = output_path

    record_tool_use("redact_all_matching_terms", words_added=added)
    log.info(
        "redact_all_matching_terms: terms=%d added=%d total=%d -> %s",
        len(term_token_lists),
        added,
        len(context.redacted_word_keys),
        output_path.name,
    )
    return json.dumps(
        {
            "output_path": str(output_path),
            "terms_evaluated": len(term_token_lists),
            "match_counts_by_term": match_counts_by_term,
            "words_added_this_call": added,
            "total_redacted": len(context.redacted_word_keys),
        }
    )


def list_visual_regions(
    only_unredacted: Annotated[
        bool,
        Field(
            description=(
                "If true, omit regions already marked for redaction. Use this "
                "on iterations after the first."
            )
        ),
    ] = False,
) -> str:
    """List logo / image / figure regions discovered in the input PDF.

    Each region has a ``strategy`` field:

    - ``"inline"``       — small graphic. Redacting it just paints a black
                           rectangle over it; surrounding content is unaffected.
    - ``"page_split"``   — large background watermark. Redacting it would also
                           cover overlaid foreground text, so the renderer will
                           emit the blacked page followed by a clean reflow
                           page containing only the non-redacted text.

    Output JSON shape::

      [{"page": 0,
        "regions": [{"i": 0, "kind": "logo", "label": "Tribune",
                     "strategy": "inline", "confidence": 0.92,
                     "redacted": false}, ...]}, ...]

    Use the ``(page, i)`` pair when calling :func:`redact_visual_regions`.
    """
    context = get_active_context()
    pages_out: dict[int, list[dict]] = {}
    for region in context.regions:
        is_redacted = (region.page, region.index) in context.redacted_region_keys
        if only_unredacted and is_redacted:
            continue
        pages_out.setdefault(region.page, []).append(
            {
                "i": region.index,
                "kind": region.kind,
                "label": region.label,
                "strategy": region.strategy,
                "confidence": round(region.confidence, 3),
                "bbox": [region.x0, region.top, region.x1, region.bottom],
                "redacted": is_redacted,
            }
        )

    result = [{"page": page, "regions": regions} for page, regions in sorted(pages_out.items())]
    total = sum(len(p["regions"]) for p in result)
    log.info(
        "list_visual_regions(only_unredacted=%s) -> %d pages, %d regions",
        only_unredacted,
        len(result),
        total,
    )
    return json.dumps(result)


def redact_visual_regions(
    spans_json: Annotated[
        str,
        Field(
            description=(
                'JSON list like [{"page":0,"region_indices":[0,1]}, ...]. '
                "Indices come from list_visual_regions. Each region's "
                "strategy is honored automatically — page_split regions add "
                "the page to the renderer's split list."
            )
        ),
    ],
) -> str:
    """Mark the listed visual regions for redaction and rebuild the visual PDF."""
    context = get_active_context()

    spans, _ = _loads_spans_tolerant(spans_json, "redact_visual_regions")
    if spans is None:
        log.error("redact_visual_regions: unparseable JSON | head=%r", spans_json[:200])
        return json.dumps(
            {
                "error": "invalid JSON",
                "hint": "Payload may have been cut off. Resend in smaller batches.",
            }
        )

    if isinstance(spans, dict):
        for key in ("spans", "regions", "items", "data"):
            value = spans.get(key)
            if isinstance(value, list):
                spans = value
                break
        else:
            list_values = [v for v in spans.values() if isinstance(v, list)]
            spans = list_values[0] if list_values else []

    if not isinstance(spans, list):
        return json.dumps({"error": "expected JSON list of {page, region_indices} entries"})

    region_lookup = {(r.page, r.index): r for r in context.regions}
    added = 0
    invalid = 0
    page_split_added: set[int] = set()

    for entry in spans:
        if not isinstance(entry, dict) or "page" not in entry:
            invalid += 1
            continue
        try:
            page = int(entry["page"])
        except (TypeError, ValueError):
            invalid += 1
            continue
        for raw_index in entry.get("region_indices", []) or entry.get("indices", []) or []:
            try:
                region_index = int(raw_index)
            except (TypeError, ValueError):
                invalid += 1
                continue
            region = region_lookup.get((page, region_index))
            if region is None:
                invalid += 1
                continue
            key = (page, region_index)
            if key not in context.redacted_region_keys:
                context.redacted_region_keys.add(key)
                added += 1
                if region.strategy == "page_split":
                    if page not in context.page_split_pages:
                        context.page_split_pages.add(page)
                        page_split_added.add(page)

    output_path = OUTPUT_DIR / f"iteration_{context.iteration}_redacted.pdf"
    context.supplemental_pages = render_redacted_pdf(
        source_pdf=context.input_pdf,
        words=context.words,
        redacted_word_keys=context.redacted_word_keys,
        output_path=output_path,
        regions=context.regions,
        redacted_region_keys=context.redacted_region_keys,
        page_split_pages=context.page_split_pages,
    )
    context.current_pdf = output_path

    record_tool_use("redact_visual_regions", regions_added=added)
    log.info(
        "redact_visual_regions: added=%d total=%d invalid=%d page_split_added=%s -> %s",
        added,
        len(context.redacted_region_keys),
        invalid,
        sorted(page_split_added),
        output_path.name,
    )
    return json.dumps(
        {
            "output_path": str(output_path),
            "regions_added_this_call": added,
            "total_regions_redacted": len(context.redacted_region_keys),
            "page_split_pages": sorted(context.page_split_pages),
            "invalid_entries": invalid,
        }
    )


def redact_bbox(
    spans_json: Annotated[
        str,
        Field(
            description=(
                'JSON list like [{"page": 0, "bbox": [x0, top, x1, bottom], '
                '"reason": "logo"}]. Coordinates are in PDF points. Use this '
                "to act directly on reviewer Logo misses that include a "
                "bbox — no need to re-query list_visual_regions. The bbox "
                "is inflated by a small margin to cover halos."
            )
        ),
    ],
) -> str:
    """Redact arbitrary page bounding boxes (typically reviewer-supplied logo misses).

    Synthesizes a new ``PageRegion`` entry on the active :class:`RunContext`
    (so the renderer treats it like any other visual region) and rebuilds
    the redacted PDF. Strategy is always ``inline``: a black rectangle is
    painted at the given coordinates.
    """
    context = get_active_context()

    spans, _ = _loads_spans_tolerant(spans_json, "redact_bbox")
    if spans is None:
        log.error("redact_bbox: unparseable JSON | head=%r", spans_json[:200])
        return json.dumps(
            {
                "error": "invalid JSON",
                "hint": "Payload may have been cut off. Resend in smaller batches.",
            }
        )

    if isinstance(spans, dict):
        for key in ("spans", "boxes", "items", "data", "missed"):
            value = spans.get(key)
            if isinstance(value, list):
                spans = value
                break
        else:
            list_values = [v for v in spans.values() if isinstance(v, list)]
            spans = list_values[0] if list_values else []

    if not isinstance(spans, list):
        return json.dumps({"error": "expected JSON list of {page, bbox} entries"})

    # Reserve indices on each page that don't collide with existing regions.
    next_index_by_page: dict[int, int] = {}
    for region in context.regions:
        next_index_by_page[region.page] = max(
            next_index_by_page.get(region.page, -1) + 1,
            region.index + 1,
        )

    added = 0
    invalid = 0
    page_count = len(context.page_sizes)
    inflate = float(REDACT_BBOX_INFLATE_POINTS)

    for entry in spans:
        if not isinstance(entry, dict):
            invalid += 1
            continue
        try:
            page = int(entry["page"])
            bbox = entry["bbox"]
        except (KeyError, TypeError, ValueError):
            invalid += 1
            continue
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            invalid += 1
            continue
        try:
            x0, top, x1, bottom = (float(v) for v in bbox)
        except (TypeError, ValueError):
            invalid += 1
            continue
        if not (0 <= page < page_count):
            invalid += 1
            continue
        if x1 <= x0 or bottom <= top:
            invalid += 1
            continue

        page_w, page_h = context.page_sizes[page]
        x0 = max(0.0, x0 - inflate)
        top = max(0.0, top - inflate)
        x1 = min(float(page_w), x1 + inflate)
        bottom = min(float(page_h), bottom + inflate)

        new_index = next_index_by_page.get(page, 0)
        next_index_by_page[page] = new_index + 1
        label = str(entry.get("reason") or entry.get("label") or "bbox")[:120]

        context.regions.append(
            PageRegion(
                page=page,
                index=new_index,
                kind="logo",
                x0=x0,
                top=top,
                x1=x1,
                bottom=bottom,
                confidence=float(entry.get("confidence", 1.0) or 1.0),
                label=label,
                strategy="inline",
            )
        )
        context.redacted_region_keys.add((page, new_index))
        added += 1

    if added:
        output_path = OUTPUT_DIR / f"iteration_{context.iteration}_redacted.pdf"
        context.supplemental_pages = render_redacted_pdf(
            source_pdf=context.input_pdf,
            words=context.words,
            redacted_word_keys=context.redacted_word_keys,
            output_path=output_path,
            regions=context.regions,
            redacted_region_keys=context.redacted_region_keys,
            page_split_pages=context.page_split_pages,
        )
        context.current_pdf = output_path
        output_str = str(output_path)
    else:
        output_str = str(context.current_pdf) if context.current_pdf else ""

    record_tool_use("redact_bbox", regions_added=added)
    log.info(
        "redact_bbox: added=%d total_regions=%d invalid=%d",
        added,
        len(context.redacted_region_keys),
        invalid,
    )
    return json.dumps(
        {
            "output_path": output_str,
            "boxes_added_this_call": added,
            "total_regions_redacted": len(context.redacted_region_keys),
            "invalid_entries": invalid,
        }
    )


# ---------------------------------------------------------------------------
# Reviewer tools
# ---------------------------------------------------------------------------
def read_redacted_text(
    _unused: Annotated[
        str,
        Field(description="Ignored. Kept for tool-call compatibility."),
    ] = "",
) -> str:
    """Return the current redacted text. Redacted spans are replaced with ``[REDACTED]``."""
    text = build_redacted_text_view()[:REVIEWER_TEXT_CHAR_LIMIT]
    context = get_active_context()
    log.info(
        "read_redacted_text -> %d chars (redactions=%d)",
        len(text),
        len(context.redacted_word_keys),
    )
    return text


def detect_pii_with_language_service(
    text: Annotated[
        str,
        Field(description="Text to scan for PII via Azure AI Language."),
    ],
) -> str:
    """Call Azure AI Language ``recognize_pii_entities`` and return entities as JSON."""
    endpoint = os.environ["AZURE_LANGUAGE_ENDPOINT"]
    client = TextAnalyticsClient(endpoint=endpoint, credential=get_azure_credential())

    chunks = [text[i : i + 5000] for i in range(0, len(text), 5000)] or [""]
    found: list[dict] = []
    for chunk in chunks:
        if not chunk.strip():
            continue
        result = client.recognize_pii_entities([chunk])
        for doc in result:
            if doc.is_error:
                log.warning("language service error: %s", doc.error)
                continue
            for entity in doc.entities:
                found.append(
                    {
                        "text": entity.text,
                        "category": entity.category,
                        "subcategory": entity.subcategory,
                        "confidence": entity.confidence_score,
                    }
                )

    log.info("detect_pii_with_language_service -> %d entities", len(found))
    return json.dumps({"entities": found, "count": len(found)})


def detect_logos_on_rendered_pdf(
    _unused: Annotated[
        str,
        Field(description="Ignored. Kept for tool-call compatibility."),
    ] = "",
) -> str:
    """Scan the **current redacted PDF** for remaining logos / watermarks.

    Uses Azure AI Vision Image Analysis on each rasterized page (with a
    Document Intelligence ``prebuilt-layout`` fallback) — so it sees the
    document exactly as a human reviewer would. Findings are then filtered:

    - Already-redacted regions are dropped (IoU vs.
      ``context.redacted_region_keys`` exceeds
      :data:`LOGO_DEDUPE_IOU_THRESHOLD`). Prevents re-flagging the same
      logo that was just blacked out.
    - Findings whose underlying page-words name the auto-detected PREPARER
      (e.g. "Goldman Sachs") are dropped. Catches the named-preparer-logo
      case even when the detector returned ``label=None``.
    - Findings whose ``label`` field substring-matches a preparer alias
      are dropped (preserved for completeness; Vision labels populate this
      path).

    Returns JSON ``{"findings": [{"page", "bbox", "confidence", "label"}],
    "count": int}``. An empty list means no remaining logos were detected.
    Treat every returned entry as a missed ``Logo`` item.
    """
    context = get_active_context()
    if context.current_pdf is None or not context.current_pdf.exists():
        log.warning("detect_logos_on_rendered_pdf: no current redacted PDF yet.")
        return json.dumps({"findings": [], "count": 0})

    findings = detect_remaining_logos(context.current_pdf)
    if not findings:
        log.info("detect_logos_on_rendered_pdf -> 0 findings")
        return json.dumps({"findings": [], "count": 0})

    # Findings are indexed against the rendered PDF, which interleaves
    # supplemental pages. Translate back to source page indices so the
    # redactor's (page, bbox) pairs address the right page.
    page_map = _rendered_to_source_pages(context)
    remapped: list[dict] = []
    dropped_synthetic = 0
    for finding in findings:
        try:
            rendered_page = int(finding.get("page"))
        except (TypeError, ValueError):
            remapped.append(finding)
            continue
        source_page = page_map.get(rendered_page, rendered_page)
        if source_page is None:
            dropped_synthetic += 1
            continue
        finding["page"] = source_page
        remapped.append(finding)
    if dropped_synthetic:
        log.info(
            "detect_logos_on_rendered_pdf: dropped %d finding(s) on supplemental pages",
            dropped_synthetic,
        )
    findings = remapped

    preparer_terms = _preparer_identity_terms(context)
    skipped_dedupe = 0
    skipped_label = 0
    skipped_text = 0
    kept: list[dict] = []

    # Pre-index existing redacted regions by page for O(N) IoU checks.
    redacted_boxes_by_page: dict[int, list[tuple[float, float, float, float]]] = {}
    for region in context.regions:
        if (region.page, region.index) in context.redacted_region_keys:
            redacted_boxes_by_page.setdefault(region.page, []).append(
                (region.x0, region.top, region.x1, region.bottom)
            )

    # Words by page for the OCR-style preparer text check. Page indices are
    # realigned to source space above, so this is always safe.
    page_indices_aligned = True
    words_by_page: dict[int, list] = {}
    if page_indices_aligned and preparer_terms:
        for w in context.words:
            words_by_page.setdefault(w.page, []).append(w)

    for finding in findings:
        page = finding.get("page")
        bbox = finding.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4 or page is None:
            kept.append(finding)
            continue
        try:
            page_idx = int(page)
            fbox = tuple(float(v) for v in bbox)
        except (TypeError, ValueError):
            kept.append(finding)
            continue

        # Fix A: drop if already covered by a redacted region.
        if any(
            _bbox_iou(fbox, rbox) >= LOGO_DEDUPE_IOU_THRESHOLD
            for rbox in redacted_boxes_by_page.get(page_idx, [])
        ):
            skipped_dedupe += 1
            continue

        # Existing label-based preparer filter.
        label = str(finding.get("label") or "").lower()
        if preparer_terms and label and any(term in label for term in preparer_terms):
            skipped_label += 1
            continue

        # Fix B: substring-match preparer aliases against the page words
        # that fall inside this finding's bbox.
        if preparer_terms and page_indices_aligned:
            region_text = " ".join(
                w.text for w in words_by_page.get(page_idx, [])
                if _word_center_in_box(w, fbox)
            ).lower()
            if region_text and any(term in region_text for term in preparer_terms):
                skipped_text += 1
                continue

        kept.append(finding)

    if skipped_dedupe or skipped_label or skipped_text:
        log.info(
            "detect_logos_on_rendered_pdf: filtered dedupe=%d label=%d text=%d",
            skipped_dedupe,
            skipped_label,
            skipped_text,
        )

    log.info("detect_logos_on_rendered_pdf -> %d findings", len(kept))
    return json.dumps({"findings": kept, "count": len(kept)})


def _rendered_to_source_pages(context) -> dict[int, int | None]:
    """Map rendered PDF page index to source page index.

    Supplemental pages are synthetic and map to ``None`` so findings landing on
    them can be discarded rather than misattributed to a real page.
    """
    mapping: dict[int, int | None] = {}
    rendered = 0
    for source_page in range(len(context.page_sizes)):
        mapping[rendered] = source_page
        rendered += 1
        if source_page in context.supplemental_pages:
            mapping[rendered] = None
            rendered += 1
    return mapping


def _preparer_identity_terms(context) -> list[str]:
    """Lowercased preparer aliases for filtering logo detections."""
    doc_ctx = getattr(context, "doc_context", None)
    preparer = getattr(doc_ctx, "preparer", None)
    if preparer is None:
        return []
    try:
        terms = preparer.all_identity_terms()
    except AttributeError:
        return []
    return [t.lower() for t in terms if t]


def _bbox_iou(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    """Intersection-over-union for two (x0, top, x1, bottom) boxes."""
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


def _word_center_in_box(word, bbox: tuple[float, float, float, float]) -> bool:
    """True if a PageWord's center lies inside the bbox."""
    cx = (word.x0 + word.x1) / 2.0
    cy = (word.top + word.bottom) / 2.0
    return bbox[0] <= cx <= bbox[2] and bbox[1] <= cy <= bbox[3]
