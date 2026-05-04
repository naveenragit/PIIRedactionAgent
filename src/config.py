"""Centralized configuration: paths, runtime limits, and environment loading."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

# Project layout
PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parent
SAMPLES_DIR = PROJECT_DIR / "samples"
OUTPUT_DIR = PROJECT_DIR / "output"

# Load environment variables. Prefers a project-local `.env` (next to main.py)
# so the deliverable is self-contained, but falls back to the first `.env`
# found while walking parent directories. This lets developers test against a
# repo-root `.env` without duplicating secrets into this folder.
_local_env = PROJECT_DIR / ".env"
if _local_env.exists():
    load_dotenv(_local_env)
else:
    for _candidate_dir in PROJECT_DIR.parents:
        _candidate_env = _candidate_dir / ".env"
        if _candidate_env.exists():
            load_dotenv(_candidate_env)
            break

# Orchestration limits
MIN_ITERATIONS = 2
MAX_ITERATIONS = 5

# Visual rendering DPI for the redacted PDF (200 is a good readability/size tradeoff).
RENDER_DPI = 200

# Maximum characters of redacted text passed to the reviewer in one call.
REVIEWER_TEXT_CHAR_LIMIT = 60_000

# Default name for the input sample.
DEFAULT_INPUT_FILENAME = "sample_input.pdf"

# Ensure output directory exists at import time so logging/artifacts can be written.
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
