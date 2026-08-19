"""CLI entrypoint for the PDF redaction agent loop.

Usage::

    python main.py --input path/to/your.pdf

If ``--input`` is omitted, the script looks for ``samples/sample_input.pdf``.
Outputs (per-iteration PDFs, ``final_redacted.pdf``, ``audit_trail.json``)
land in the ``output/`` folder next to this script.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from src.config import DEFAULT_INPUT_FILENAME, OUTPUT_DIR, SAMPLES_DIR
from src.orchestrator import run_redaction_loop


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=str(SAMPLES_DIR / DEFAULT_INPUT_FILENAME),
        help="Path to input PDF.",
    )
    parser.add_argument(
        "--run-label",
        default=None,
        help=(
            "Short tag for this run (e.g. 'old-prompt', 'new-prompt-v2'). "
            "Used in the per-run metrics filename and the runs_summary.csv "
            "row so you can compare prompt revisions side-by-side."
        ),
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help=(
            "Cap the redactor/reviewer loop to this many iterations. Use 1 for a "
            "single-pass baseline run (the minimum is clamped to match). "
            "Defaults to the configured maximum."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_pdf = Path(args.input)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    if not input_pdf.exists():
        print(
            f"Input PDF not found: {input_pdf}\n"
            f"   Drop a PDF at {SAMPLES_DIR / DEFAULT_INPUT_FILENAME} or pass --input <path>.",
            file=sys.stderr,
        )
        sys.exit(2)

    audit = asyncio.run(
        run_redaction_loop(
            input_pdf,
            run_label=args.run_label,
            max_iterations=args.max_iterations,
        )
    )

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"Final PDF:   {audit['final_pdf']}")
    print(f"Iterations:  {audit['iterations_run']}")
    print(f"Verdict:     {audit['final_verdict'].get('verdict')}")
    print(f"Audit trail: {OUTPUT_DIR / 'audit_trail.json'}")
    if audit.get("metrics_path"):
        print(f"Metrics:     {audit['metrics_path']}")
    if audit.get("metrics_summary_csv"):
        print(f"Summary CSV: {audit['metrics_summary_csv']}")


if __name__ == "__main__":
    main()
