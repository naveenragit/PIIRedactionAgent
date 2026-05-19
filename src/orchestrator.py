"""End-to-end orchestration: run the redactor/reviewer loop until APPROVED or max iterations."""

from __future__ import annotations

import json
import re
from pathlib import Path

from .agents import build_redactor_agent, build_reviewer_agent
from .azure_clients import build_chat_client
from .config import MAX_ITERATIONS, MIN_ITERATIONS, OUTPUT_DIR
from .document_context import DocumentContext, detect_document_context
from .logger import log
from .models import RunContext
from .pdf_text_extractor import extract_words_with_bboxes
from .pdf_visual_extractor import extract_visual_regions
from .redaction_policy import CUSTOM_REDACTION_RULES
from .redaction_state import set_active_context


# Matches a balanced JSON object — used to extract the reviewer's verdict block.
_JSON_OBJECT_PATTERN = re.compile(r"\{(?:[^{}]|(?:\{[^{}]*\}))*\}", re.DOTALL)


_REDACTOR_FIRST_TURN_TEMPLATE = """\
Active document: {input_pdf}

{document_context}

{policy}

Suggested workflow:
0. {seed_step}
1. Call `extract_pdf_words` and read the document to expand the CLIENT
   identity term list (legal name, brand name, common abbreviations,
   ticker, aliases, subsidiary or product names that uniquely identify the
   client, every CLIENT executive or contact full name, every
   precedent-deal target name and counterparty name visible). When in
   doubt about the CLIENT side, include the term. DO NOT include the
   PREPARER's firm name in this sweep. Call `redact_all_matching_terms`
   with that full list before any `apply_redactions` call so identity
   coverage is consistent across pages.
2. Call `extract_pdf_words(only_unredacted=true)` and locate residual items
   per the policy: currency plus numeric tokens that disclose valuation,
   revenue, EBITDA, or deal sizes (cover "$", the number, and the unit "B"
   or "M" together); specific transaction dates and identifying event
   descriptors; value labels in charts and tables; standard PII identifiers
   (SSN, EIN, phone, email, etc.) — including direct contact info of named
   preparer bankers. Call `apply_redactions` with a JSON spans list
   covering all of it.
3. Call `list_visual_regions` and redact EVERY CLIENT logo, brand mark, or
   watermark by calling `redact_visual_regions` with their `{{page, i}}`
   pairs. DO NOT redact the preparer's firm logo. Trust the `strategy`
   field returned by the tool: `inline` regions are simply blacked out;
   `page_split` regions are large background watermarks and the renderer
   will automatically produce a clean continuation page after the blacked
   original.
4. End with a short summary: client identity terms swept (with match
   counts), residual items covered, logo regions redacted (with strategy),
   and the final output path.
"""


_REDACTOR_FOLLOWUP_TEMPLATE = """\
{document_context}

Reviewer feedback from the previous round (please address every item):
{feedback}

{logo_bbox_block}\
Treat every missed CLIENT name, alias, or ticker as a new term and call
`redact_all_matching_terms` with the expanded list before any per-page
work. For missed numeric or event items, call `apply_redactions` with the
appropriate {{page, word_indices}} spans. For missed logos / watermarks,
prefer calling `redact_bbox` directly on the (page, bbox) coordinates the
reviewer provided — this works even when the region was not in the
original visual-region enumeration. Fall back to
`list_visual_regions(only_unredacted=true)` + `redact_visual_regions` only
if no bbox was provided.
"""


_REVIEWER_PROMPT_TEMPLATE = """\
Review the current sanitized document text and decide whether it is ready
to share.

{document_context}

{policy}

Steps:
1. Call `read_redacted_text` to get the current document text. Covered
   spans appear as the literal token [REDACTED].
2. Call `detect_pii_with_language_service` on that text to catch standard
   PII identifiers (names, contacts, government IDs, financials).
3. Call `detect_logos_on_rendered_pdf` to scan the rasterized output for
   any remaining company logos, brand marks, or watermarks (foreground OR
   background). Each finding must be reported as a `Logo` miss — but DO
   NOT flag the PREPARER's firm logo as a miss; that is expected to
   remain. When you report a `Logo` miss, always include the `bbox` field
   so the redactor can act on it directly.
4. Check structural integrity (page order preserved; section headers,
   titles, bullet structure, table layouts intact). Note: pages with a
   background watermark are intentionally split into a fully-blacked page
   followed by a `Page N (continued)` reflow page; this is expected and
   should NOT be flagged as a structure issue.
5. Confirm complete coverage of personal names, contacts, and identifiers;
   valuation, revenue, and deal-size figures; CLIENT identity terms,
   tickers, and brand names. Flag any leftover items, including initials,
   partial names, or numeric leakage through ranges or deltas. The
   PREPARER's firm name appearing in headers / footers / disclaimers is
   NOT a miss.
6. Run a consistency audit across the entire document and check that
   redactions preserve readability without distorting the banker's
   judgment.

Verdict mapping:
- "APPROVED"               => No issues found, safe for sharing.
- "APPROVED_WITH_ISSUES"   => Cosmetic or minor consistency issues only;
                              still safe for sharing. The orchestration
                              treats this as approved.
- "REVISE"                 => Sensitive content remains, client identity
                              is still inferable, or structural failure.

End your response with valid JSON only, no markdown fence, in this shape:
{{
  "verdict": "APPROVED" | "APPROVED_WITH_ISSUES" | "REVISE",
  "summary": ["3-5 bullet findings"],
  "missed": [
    {{
      "page": <int or null>,
      "type": "PII" | "MNPI" | "Logo" | "Consistency" | "Structure",
      "text": "exact offending text or description",
      "bbox": [x0, top, x1, bottom] | null,
      "reason": "why this is a violation"
    }}
  ],
  "required_actions": "Concrete instructions for the redactor if REVISE; empty string otherwise.",
  "feedback": "Concise actionable feedback for the redactor."
}}
"""


def _render_seed_step(doc_context: DocumentContext) -> str:
    """Build the iteration-1 seed step that pre-loads detected client terms."""
    terms = doc_context.client.all_identity_terms()
    if not terms:
        return (
            "No high-confidence client identity was auto-detected. Begin by "
            "reading the document to determine the client, then proceed to "
            "step 1."
        )
    quoted = ", ".join(json.dumps(t) for t in terms)
    people_clause = ""
    if doc_context.client.people:
        people_quoted = ", ".join(json.dumps(p) for p in doc_context.client.people)
        people_clause = f" Also seed the client-people list with: [{people_quoted}]."
    return (
        f"Immediately call `redact_all_matching_terms` with this "
        f"auto-detected CLIENT identity seed list before anything else: "
        f"[{quoted}].{people_clause} Then move to step 1 to expand on it."
    )


def _format_logo_bbox_block(verdict: dict) -> str:
    """Render reviewer-supplied Logo misses with bboxes for the redactor."""
    items: list[dict] = []
    for miss in verdict.get("missed", []) or []:
        if not isinstance(miss, dict):
            continue
        if str(miss.get("type", "")).lower() != "logo":
            continue
        bbox = miss.get("bbox")
        page = miss.get("page")
        if not isinstance(bbox, list) or len(bbox) != 4 or page is None:
            continue
        items.append(
            {
                "page": page,
                "bbox": bbox,
                "reason": str(miss.get("reason", "logo")),
            }
        )
    if not items:
        return ""
    return (
        "Logo misses with explicit bounding boxes — call `redact_bbox` with "
        "this JSON list FIRST, before any other tool, to cover them "
        "deterministically:\n"
        f"{json.dumps(items, indent=2)}\n\n"
    )



def parse_reviewer_verdict(text: str) -> dict:
    """Extract the last JSON object containing ``verdict`` from reviewer output."""
    matches = _JSON_OBJECT_PATTERN.findall(text)
    for candidate in reversed(matches):
        try:
            obj = json.loads(candidate)
            if "verdict" in obj:
                return obj
        except json.JSONDecodeError:
            continue
    return {
        "verdict": "REVISE",
        "missed": [],
        "feedback": "Could not parse reviewer JSON; please retry.",
    }


async def run_redaction_loop(input_pdf: Path) -> dict:
    """Run the redactor/reviewer loop end-to-end and return the audit trail."""
    words, page_sizes = extract_words_with_bboxes(input_pdf)
    regions = extract_visual_regions(input_pdf, words, page_sizes)
    context = RunContext(
        input_pdf=input_pdf,
        words=words,
        page_sizes=page_sizes,
        regions=regions,
    )
    set_active_context(context)
    log.info(
        "Loaded %d words across %d pages and %d visual regions from %s",
        len(words),
        len(page_sizes),
        len(regions),
        input_pdf.name,
    )

    chat_client = build_chat_client()

    # Detect client / preparer identity once, before any agent runs.
    doc_context = await detect_document_context(chat_client, words, len(page_sizes))
    context.doc_context = doc_context
    doc_context_block = doc_context.render_for_prompt()

    redactor = build_redactor_agent(chat_client)
    reviewer = build_reviewer_agent(chat_client)

    last_verdict: dict = {}
    feedback_for_redactor: str | None = None
    last_logo_bbox_block = ""

    # Convergence-stall tracking: bail out early if neither side is making
    # progress for two consecutive iterations.
    zero_delta_streak = 0
    prev_missed_count: int | None = None
    stall_streak = 0

    for iteration in range(1, MAX_ITERATIONS + 1):
        context.iteration = iteration
        log.info("--- Iteration %d/%d ---", iteration, MAX_ITERATIONS)

        # ---- Redactor turn ----
        if iteration == 1:
            redactor_prompt = _REDACTOR_FIRST_TURN_TEMPLATE.format(
                input_pdf=context.input_pdf,
                document_context=doc_context_block,
                policy=CUSTOM_REDACTION_RULES.strip(),
                seed_step=_render_seed_step(doc_context),
            )
        else:
            redactor_prompt = _REDACTOR_FOLLOWUP_TEMPLATE.format(
                document_context=doc_context_block,
                feedback=feedback_for_redactor,
                logo_bbox_block=last_logo_bbox_block,
            )

        redactions_before = len(context.redacted_word_keys)
        regions_before = len(context.redacted_region_keys)
        redactor_result = await redactor.run(redactor_prompt)
        redactor_text = str(redactor_result)
        word_delta = len(context.redacted_word_keys) - redactions_before
        region_delta = len(context.redacted_region_keys) - regions_before
        total_delta = word_delta + region_delta
        if total_delta == 0:
            zero_delta_streak += 1
            log.warning(
                "Redactor added 0 redactions this turn (streak=%d).",
                zero_delta_streak,
            )
            log.warning("Redactor response: %s", redactor_text[:600])
        else:
            zero_delta_streak = 0
            log.info(
                "Redactor done: +%d words, +%d regions (total words=%d, regions=%d)",
                word_delta,
                region_delta,
                len(context.redacted_word_keys),
                len(context.redacted_region_keys),
            )

        # ---- Reviewer turn ----
        reviewer_prompt = _REVIEWER_PROMPT_TEMPLATE.format(
            document_context=doc_context_block,
            policy=CUSTOM_REDACTION_RULES.strip(),
        )
        reviewer_result = await reviewer.run(reviewer_prompt)
        reviewer_text = str(reviewer_result)

        verdict = parse_reviewer_verdict(reviewer_text)
        last_verdict = verdict
        missed_count = len(verdict.get("missed", []))
        log.info(
            "Reviewer verdict: %s (missed=%d)",
            verdict.get("verdict"),
            missed_count,
        )

        # Stall detection: missed count stopped decreasing AND redactor added nothing.
        if prev_missed_count is not None and missed_count >= prev_missed_count and total_delta == 0:
            stall_streak += 1
        else:
            stall_streak = 0
        prev_missed_count = missed_count

        context.history.append(
            {
                "iteration": iteration,
                "redacted_pdf": str(context.current_pdf) if context.current_pdf else None,
                "total_redactions": len(context.redacted_word_keys),
                "total_regions_redacted": len(context.redacted_region_keys),
                "page_split_pages": sorted(context.page_split_pages),
                "verdict": verdict.get("verdict"),
                "missed_count": missed_count,
                "redactor_word_delta": word_delta,
                "redactor_region_delta": region_delta,
                "feedback": verdict.get("feedback", ""),
            }
        )

        approved = verdict.get("verdict") in ("APPROVED", "APPROVED_WITH_ISSUES")
        print(
            f"\n[Loop] verdict={verdict.get('verdict')} "
            f"missed={missed_count} "
            f"total_redactions={len(context.redacted_word_keys)}"
        )

        if approved and iteration >= MIN_ITERATIONS:
            print(f"[Loop] APPROVED at iteration {iteration} (>= min {MIN_ITERATIONS}). Exiting.")
            break
        if approved and iteration < MIN_ITERATIONS:
            print(
                f"[Loop] APPROVED but iteration {iteration} < min {MIN_ITERATIONS}. "
                "Forcing another pass."
            )
            feedback_for_redactor = json.dumps(
                {
                    "missed": [],
                    "feedback": (
                        "Reviewer approved, but the minimum-iterations policy requires another pass. "
                        "Re-scan strictly under the custom rules — especially borderline names, "
                        "addresses, and identifiers."
                    ),
                },
                indent=2,
            )
            last_logo_bbox_block = ""
            continue
        if stall_streak >= 2 and iteration >= MIN_ITERATIONS:
            print(
                f"[Loop] Convergence stalled for {stall_streak} iterations "
                f"(no new redactions, missed count not decreasing). Stopping."
            )
            break
        if iteration == MAX_ITERATIONS:
            print(f"[Loop] Reached max_iterations ({MAX_ITERATIONS}). Stopping with last result.")
            break

        feedback_for_redactor = json.dumps(
            {
                "missed": verdict.get("missed", []),
                "feedback": verdict.get("feedback", ""),
            },
            indent=2,
        )
        last_logo_bbox_block = _format_logo_bbox_block(verdict)

    # Copy the latest iteration to a stable filename.
    final_pdf_path = OUTPUT_DIR / "final_redacted.pdf"
    if context.current_pdf and context.current_pdf.exists():
        final_pdf_path.write_bytes(context.current_pdf.read_bytes())

    audit = {
        "input_pdf": str(context.input_pdf),
        "final_pdf": str(final_pdf_path) if final_pdf_path.exists() else None,
        "iterations_run": context.iteration,
        "min_iterations": MIN_ITERATIONS,
        "max_iterations": MAX_ITERATIONS,
        "total_redactions": len(context.redacted_word_keys),
        "total_regions_redacted": len(context.redacted_region_keys),
        "page_split_pages": sorted(context.page_split_pages),
        "document_context": doc_context.to_dict(),
        "final_verdict": last_verdict,
        "history": context.history,
    }
    (OUTPUT_DIR / "audit_trail.json").write_text(json.dumps(audit, indent=2))
    return audit
