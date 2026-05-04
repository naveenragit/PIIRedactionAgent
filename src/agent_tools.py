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
from .config import OUTPUT_DIR, REVIEWER_TEXT_CHAR_LIMIT
from .logger import log
from .pdf_renderer import render_redacted_pdf
from .redaction_state import build_redacted_text_view, get_active_context


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
                "Indices come from extract_pdf_words."
            )
        ),
    ],
) -> str:
    """Record redactions for the listed words and rebuild the visual PDF.

    Returns the path to the freshly saved redacted PDF for this iteration.
    """
    context = get_active_context()
    log.debug("apply_redactions called with %d bytes", len(spans_json))

    try:
        spans = json.loads(spans_json)
    except json.JSONDecodeError as exc:
        log.error("apply_redactions: invalid JSON: %s | head=%r", exc, spans_json[:200])
        return json.dumps({"error": f"invalid JSON: {exc}"})

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
    render_redacted_pdf(
        source_pdf=context.input_pdf,
        words=context.words,
        redacted_word_keys=context.redacted_word_keys,
        output_path=output_path,
    )
    context.current_pdf = output_path

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
    render_redacted_pdf(
        source_pdf=context.input_pdf,
        words=context.words,
        redacted_word_keys=context.redacted_word_keys,
        output_path=output_path,
    )
    context.current_pdf = output_path

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
