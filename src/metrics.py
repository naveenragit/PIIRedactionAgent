"""Per-run redaction metrics for comparing prompt versions across runs.

Each call to :func:`write_run_metrics` produces two artifacts under
``output/metrics/``:

- ``run_<timestamp>_<label>.json`` — full per-run breakdown, including
  per-iteration deltas, per-tool attribution, and reviewer misses by type.
- ``runs_summary.csv`` — one row per run, appended in chronological order.
  Open this in Excel to compare runs side-by-side.

The CSV is the primary tool for prompt A/B comparisons; the JSON file is
there when you need to drill into a specific run.
"""

from __future__ import annotations

import csv
import json
import re
from datetime import datetime
from pathlib import Path

from .config import OUTPUT_DIR
from .models import RunContext


METRICS_DIR = OUTPUT_DIR / "metrics"
RUNS_SUMMARY_CSV = METRICS_DIR / "runs_summary.csv"

# Canonical reviewer miss categories. Anything outside these buckets goes
# under "Other" so the CSV columns stay stable across runs.
MISS_TYPES = ("PII", "MNPI", "Logo", "Consistency", "Structure", "Other")

# Redactor tools we attribute words / regions to. Listed explicitly so the
# CSV has stable columns even when a tool wasn't called in a given run.
REDACTOR_WORD_TOOLS = ("redact_all_matching_terms", "apply_redactions")
REDACTOR_REGION_TOOLS = ("redact_visual_regions", "redact_bbox")

CSV_COLUMNS = [
    "timestamp",
    "run_label",
    "input_pdf",
    "iterations_run",
    "final_verdict",
    "duration_seconds",
    "total_words_redacted",
    "total_regions_redacted",
    "proactive_words",
    "proactive_regions",
    "reviewer_driven_words",
    "reviewer_driven_regions",
    "words_via_redact_all_matching_terms",
    "words_via_apply_redactions",
    "regions_via_redact_visual_regions",
    "regions_via_redact_bbox",
    "final_missed_count",
    "total_misses_flagged",
    "misses_pii",
    "misses_mnpi",
    "misses_logo",
    "misses_consistency",
    "misses_structure",
    "misses_other",
]


def _safe_label(label: str) -> str:
    """Make a label safe to embed in a filename."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", label.strip())
    return cleaned.strip("-") or "run"


def classify_miss_type(raw: object) -> str:
    """Map a reviewer ``missed[*].type`` value to a canonical bucket."""
    if not isinstance(raw, str):
        return "Other"
    key = raw.strip().lower()
    for canonical in MISS_TYPES:
        if key == canonical.lower():
            return canonical
    return "Other"


def count_missed_by_type(missed: list) -> dict[str, int]:
    """Bucket a reviewer verdict's ``missed`` list into canonical categories."""
    counts: dict[str, int] = {t: 0 for t in MISS_TYPES}
    if not isinstance(missed, list):
        return counts
    for entry in missed:
        if not isinstance(entry, dict):
            counts["Other"] += 1
            continue
        counts[classify_miss_type(entry.get("type"))] += 1
    return counts


def snapshot_tool_counters(context: RunContext) -> dict[str, dict[str, int]]:
    """Deep-copy the per-tool counter map so we can diff against it later."""
    return {
        tool: dict(counters) for tool, counters in context.tool_counters.items()
    }


def diff_tool_counters(
    before: dict[str, dict[str, int]],
    after: dict[str, dict[str, int]],
) -> dict[str, dict[str, int]]:
    """Return the per-tool deltas between two snapshots (after - before)."""
    diff: dict[str, dict[str, int]] = {}
    for tool, after_counters in after.items():
        before_counters = before.get(tool, {})
        tool_diff = {
            key: after_counters.get(key, 0) - before_counters.get(key, 0)
            for key in ("calls", "words_added", "regions_added")
        }
        if any(value != 0 for value in tool_diff.values()):
            diff[tool] = tool_diff
    return diff


def compute_run_metrics(
    context: RunContext,
    audit: dict,
    *,
    run_label: str,
    started_at: datetime,
    finished_at: datetime,
) -> dict:
    """Build the metrics dict for one full redaction run."""
    history = audit.get("history", []) or []

    # Redactor attribution: iteration 1 is "proactive"; subsequent
    # iterations only happen because the reviewer asked for more, so we
    # attribute them as "reviewer-driven".
    proactive_words = 0
    proactive_regions = 0
    reviewer_driven_words = 0
    reviewer_driven_regions = 0
    for entry in history:
        word_delta = int(entry.get("redactor_word_delta", 0) or 0)
        region_delta = int(entry.get("redactor_region_delta", 0) or 0)
        if entry.get("iteration") == 1:
            proactive_words += word_delta
            proactive_regions += region_delta
        else:
            reviewer_driven_words += word_delta
            reviewer_driven_regions += region_delta

    # Per-tool totals.
    words_by_tool = {
        tool: int(context.tool_counters.get(tool, {}).get("words_added", 0))
        for tool in REDACTOR_WORD_TOOLS
    }
    regions_by_tool = {
        tool: int(context.tool_counters.get(tool, {}).get("regions_added", 0))
        for tool in REDACTOR_REGION_TOOLS
    }

    # Reviewer miss aggregates across iterations.
    total_misses_flagged_by_type = {t: 0 for t in MISS_TYPES}
    for entry in history:
        per_iter = entry.get("missed_by_type") or {}
        for bucket in MISS_TYPES:
            total_misses_flagged_by_type[bucket] += int(per_iter.get(bucket, 0))

    final_verdict_obj = audit.get("final_verdict") or {}
    final_missed = final_verdict_obj.get("missed") or []
    final_missed_by_type = count_missed_by_type(final_missed)

    duration_seconds = round((finished_at - started_at).total_seconds(), 2)

    metrics = {
        "run_label": run_label,
        "timestamp": started_at.isoformat(timespec="seconds"),
        "finished_at": finished_at.isoformat(timespec="seconds"),
        "duration_seconds": duration_seconds,
        "input_pdf": str(context.input_pdf),
        "iterations_run": int(audit.get("iterations_run", context.iteration)),
        "final_verdict": final_verdict_obj.get("verdict"),
        "totals": {
            "words_redacted": int(audit.get("total_redactions", 0)),
            "regions_redacted": int(audit.get("total_regions_redacted", 0)),
            "page_split_pages": list(audit.get("page_split_pages", []) or []),
            "final_missed_count": len(final_missed),
        },
        "redactor_attribution": {
            "proactive": {
                "words": proactive_words,
                "regions": proactive_regions,
            },
            "reviewer_driven": {
                "words": reviewer_driven_words,
                "regions": reviewer_driven_regions,
            },
        },
        "redactor_by_tool": {
            "words_by_tool": words_by_tool,
            "regions_by_tool": regions_by_tool,
            "raw_counters": {
                tool: dict(counters)
                for tool, counters in context.tool_counters.items()
            },
        },
        "reviewer_misses": {
            "total_flagged_across_iterations": sum(
                total_misses_flagged_by_type.values()
            ),
            "by_type_across_iterations": total_misses_flagged_by_type,
            "final_iteration_count": len(final_missed),
            "final_iteration_by_type": final_missed_by_type,
        },
        "per_iteration": history,
    }
    return metrics


def write_run_metrics(metrics: dict) -> Path:
    """Persist the per-run JSON file and return its path."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    ts = metrics.get("timestamp", datetime.now().isoformat(timespec="seconds"))
    # Compact timestamp for filenames: 2026-06-09T14:23:45 -> 20260609-142345.
    file_ts = re.sub(r"[^0-9]", "", ts)[:14] or "00000000000000"
    file_ts = f"{file_ts[:8]}-{file_ts[8:14]}" if len(file_ts) >= 14 else file_ts
    label = _safe_label(str(metrics.get("run_label") or "run"))
    path = METRICS_DIR / f"run_{file_ts}_{label}.json"
    path.write_text(json.dumps(metrics, indent=2, default=str))
    return path


def _row_from_metrics(metrics: dict) -> dict[str, object]:
    """Flatten a metrics dict into the CSV row shape."""
    totals = metrics.get("totals", {}) or {}
    attribution = metrics.get("redactor_attribution", {}) or {}
    proactive = attribution.get("proactive", {}) or {}
    reviewer_driven = attribution.get("reviewer_driven", {}) or {}
    by_tool = metrics.get("redactor_by_tool", {}) or {}
    words_by_tool = by_tool.get("words_by_tool", {}) or {}
    regions_by_tool = by_tool.get("regions_by_tool", {}) or {}
    misses = metrics.get("reviewer_misses", {}) or {}
    misses_by_type = misses.get("by_type_across_iterations", {}) or {}

    input_pdf = metrics.get("input_pdf", "")
    input_basename = Path(input_pdf).name if input_pdf else ""

    return {
        "timestamp": metrics.get("timestamp", ""),
        "run_label": metrics.get("run_label", ""),
        "input_pdf": input_basename,
        "iterations_run": metrics.get("iterations_run", ""),
        "final_verdict": metrics.get("final_verdict", ""),
        "duration_seconds": metrics.get("duration_seconds", ""),
        "total_words_redacted": totals.get("words_redacted", 0),
        "total_regions_redacted": totals.get("regions_redacted", 0),
        "proactive_words": proactive.get("words", 0),
        "proactive_regions": proactive.get("regions", 0),
        "reviewer_driven_words": reviewer_driven.get("words", 0),
        "reviewer_driven_regions": reviewer_driven.get("regions", 0),
        "words_via_redact_all_matching_terms": words_by_tool.get(
            "redact_all_matching_terms", 0
        ),
        "words_via_apply_redactions": words_by_tool.get("apply_redactions", 0),
        "regions_via_redact_visual_regions": regions_by_tool.get(
            "redact_visual_regions", 0
        ),
        "regions_via_redact_bbox": regions_by_tool.get("redact_bbox", 0),
        "final_missed_count": totals.get("final_missed_count", 0),
        "total_misses_flagged": misses.get(
            "total_flagged_across_iterations", 0
        ),
        "misses_pii": misses_by_type.get("PII", 0),
        "misses_mnpi": misses_by_type.get("MNPI", 0),
        "misses_logo": misses_by_type.get("Logo", 0),
        "misses_consistency": misses_by_type.get("Consistency", 0),
        "misses_structure": misses_by_type.get("Structure", 0),
        "misses_other": misses_by_type.get("Other", 0),
    }


def append_runs_summary_csv(metrics: dict) -> Path:
    """Append a single-line summary row to ``runs_summary.csv``."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    row = _row_from_metrics(metrics)
    write_header = not RUNS_SUMMARY_CSV.exists()
    with RUNS_SUMMARY_CSV.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    return RUNS_SUMMARY_CSV


def format_metrics_table(metrics: dict) -> str:
    """Return a human-readable summary table for stdout."""
    totals = metrics.get("totals", {}) or {}
    attribution = metrics.get("redactor_attribution", {}) or {}
    proactive = attribution.get("proactive", {}) or {}
    reviewer_driven = attribution.get("reviewer_driven", {}) or {}
    by_tool = metrics.get("redactor_by_tool", {}) or {}
    misses = metrics.get("reviewer_misses", {}) or {}
    misses_by_type = misses.get("by_type_across_iterations", {}) or {}

    lines: list[str] = []
    lines.append("Run metrics")
    lines.append("-" * 70)
    lines.append(f"Run label:           {metrics.get('run_label', '')}")
    lines.append(f"Timestamp:           {metrics.get('timestamp', '')}")
    lines.append(f"Duration (s):        {metrics.get('duration_seconds', '')}")
    lines.append(f"Iterations:          {metrics.get('iterations_run', '')}")
    lines.append(f"Final verdict:       {metrics.get('final_verdict', '')}")
    lines.append("")
    lines.append("Redactions (totals)")
    lines.append(f"  Words:             {totals.get('words_redacted', 0)}")
    lines.append(f"  Regions:           {totals.get('regions_redacted', 0)}")
    lines.append("")
    lines.append("Redactor catches by source")
    lines.append(
        f"  Proactive  (iter 1): words={proactive.get('words', 0):>5}  "
        f"regions={proactive.get('regions', 0):>3}"
    )
    lines.append(
        f"  Reviewer-driven    : words={reviewer_driven.get('words', 0):>5}  "
        f"regions={reviewer_driven.get('regions', 0):>3}"
    )
    lines.append("")
    lines.append("Redactor catches by tool")
    for tool, count in (by_tool.get("words_by_tool") or {}).items():
        lines.append(f"  {tool:<30} words={count}")
    for tool, count in (by_tool.get("regions_by_tool") or {}).items():
        lines.append(f"  {tool:<30} regions={count}")
    lines.append("")
    lines.append("Reviewer misses flagged (across all iterations)")
    lines.append(
        f"  Total flagged:     {misses.get('total_flagged_across_iterations', 0)}"
    )
    for bucket in MISS_TYPES:
        lines.append(f"  {bucket:<18} {misses_by_type.get(bucket, 0)}")
    lines.append("")
    lines.append(
        f"Reviewer misses remaining at end:  {totals.get('final_missed_count', 0)}"
    )
    return "\n".join(lines)
