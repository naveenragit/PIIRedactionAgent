"""End-to-end orchestration: run the redactor/reviewer loop until APPROVED or max iterations."""

from __future__ import annotations

import json
import re
from pathlib import Path

from .agents import build_redactor_agent, build_reviewer_agent
from .azure_clients import build_chat_client
from .config import MAX_ITERATIONS, MIN_ITERATIONS, OUTPUT_DIR
from .logger import log
from .models import RunContext
from .pdf_text_extractor import extract_words_with_bboxes
from .redaction_policy import CUSTOM_REDACTION_RULES
from .redaction_state import set_active_context


# Matches a balanced JSON object — used to extract the reviewer's verdict block.
_JSON_OBJECT_PATTERN = re.compile(r"\{(?:[^{}]|(?:\{[^{}]*\}))*\}", re.DOTALL)


_REDACTOR_FIRST_TURN_TEMPLATE = """\
Active document: {input_pdf}

{policy}

Suggested workflow:
1. Call `extract_pdf_words` and read the document to build the complete
   list of client identity terms (legal name, brand name, common
   abbreviations, ticker, aliases, subsidiary or product names that
   uniquely identify the client, every executive or contact full name,
   every precedent-deal target name and counterparty name visible). When in
   doubt, include the term. Then call `redact_all_matching_terms` with that
   full list before any `apply_redactions` call so identity coverage is
   consistent across pages.
2. Call `extract_pdf_words(only_unredacted=true)` and locate residual items
   per the policy: currency plus numeric tokens that disclose valuation,
   revenue, EBITDA, or deal sizes (cover "$", the number, and the unit "B"
   or "M" together); specific transaction dates and identifying event
   descriptors; value labels in charts and tables; standard PII identifiers
   (SSN, EIN, phone, email, etc.). Call `apply_redactions` with a JSON
   spans list covering all of it.
3. End with a short summary: client identity terms swept (with match
   counts), residual items covered, and the final output path.
"""


_REDACTOR_FOLLOWUP_TEMPLATE = """\
Reviewer feedback from the previous round (please address every item):
{feedback}

Treat every missed name, alias, or ticker as a new term and call
`redact_all_matching_terms` with the expanded list before any per-page
work. For missed numeric or event items, call `apply_redactions` with the
appropriate {{page, word_indices}} spans.
"""


_REVIEWER_PROMPT_TEMPLATE = """\
Review the current sanitized document text and decide whether it is ready
to share.

{policy}

Steps:
1. Call `read_redacted_text` to get the current document text. Covered
   spans appear as the literal token [REDACTED].
2. Call `detect_pii_with_language_service` on that text to catch standard
   PII identifiers (names, contacts, government IDs, financials).
3. Check structural integrity (page order preserved; section headers,
   titles, bullet structure, table layouts intact).
4. Confirm complete coverage of personal names, contacts, and identifiers;
   valuation, revenue, and deal-size figures; client identity terms,
   tickers, and brand names. Flag any leftover items, including initials,
   partial names, or numeric leakage through ranges or deltas.
5. Run a consistency audit across the entire document and check that
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
      "reason": "why this is a violation"
    }}
  ],
  "required_actions": "Concrete instructions for the redactor if REVISE; empty string otherwise.",
  "feedback": "Concise actionable feedback for the redactor."
}}
"""


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
    context = RunContext(input_pdf=input_pdf, words=words, page_sizes=page_sizes)
    set_active_context(context)
    log.info(
        "Loaded %d words across %d pages from %s",
        len(words),
        len(page_sizes),
        input_pdf.name,
    )

    chat_client = build_chat_client()
    redactor = build_redactor_agent(chat_client)
    reviewer = build_reviewer_agent(chat_client)

    last_verdict: dict = {}
    feedback_for_redactor: str | None = None

    for iteration in range(1, MAX_ITERATIONS + 1):
        context.iteration = iteration
        log.info("--- Iteration %d/%d ---", iteration, MAX_ITERATIONS)

        # ---- Redactor turn ----
        if iteration == 1:
            redactor_prompt = _REDACTOR_FIRST_TURN_TEMPLATE.format(
                input_pdf=context.input_pdf,
                policy=CUSTOM_REDACTION_RULES.strip(),
            )
        else:
            redactor_prompt = _REDACTOR_FOLLOWUP_TEMPLATE.format(
                feedback=feedback_for_redactor,
            )

        redactions_before = len(context.redacted_word_keys)
        redactor_result = await redactor.run(redactor_prompt)
        redactor_text = str(redactor_result)
        delta = len(context.redacted_word_keys) - redactions_before
        if delta == 0:
            log.warning("Redactor added 0 redactions this turn.")
            log.warning("Redactor response: %s", redactor_text[:600])
        else:
            log.info(
                "Redactor done: +%d redactions (total=%d)",
                delta,
                len(context.redacted_word_keys),
            )

        # ---- Reviewer turn ----
        reviewer_prompt = _REVIEWER_PROMPT_TEMPLATE.format(
            policy=CUSTOM_REDACTION_RULES.strip(),
        )
        reviewer_result = await reviewer.run(reviewer_prompt)
        reviewer_text = str(reviewer_result)

        verdict = parse_reviewer_verdict(reviewer_text)
        last_verdict = verdict
        log.info(
            "Reviewer verdict: %s (missed=%d)",
            verdict.get("verdict"),
            len(verdict.get("missed", [])),
        )

        context.history.append(
            {
                "iteration": iteration,
                "redacted_pdf": str(context.current_pdf) if context.current_pdf else None,
                "total_redactions": len(context.redacted_word_keys),
                "verdict": verdict.get("verdict"),
                "missed_count": len(verdict.get("missed", [])),
                "feedback": verdict.get("feedback", ""),
            }
        )

        approved = verdict.get("verdict") in ("APPROVED", "APPROVED_WITH_ISSUES")
        print(
            f"\n[Loop] verdict={verdict.get('verdict')} "
            f"missed={len(verdict.get('missed', []))} "
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
            continue
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
        "final_verdict": last_verdict,
        "history": context.history,
    }
    (OUTPUT_DIR / "audit_trail.json").write_text(json.dumps(audit, indent=2))
    return audit
