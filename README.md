# PDF Redaction Agent

A two-agent pipeline that automatically redacts **PII** and **MNPI** (Material
Non-Public Information) from a banker's pitch deck PDF, then audits its own
work until a reviewer agent approves the result or a maximum number of
iterations is reached.

```
            ┌──────────────┐         ┌──────────────┐
  PDF ───▶  │   Redactor   │ ──────▶ │   Reviewer   │ ──┐
            │    Agent     │         │    Agent     │   │
            └──────────────┘         └──────────────┘   │
                  ▲                                     │
                  └───── REVISE feedback (≤ max iters) ─┘
```

- **Redactor** — extracts every word + bounding box from the input PDF, sweeps
  client-identity terms across all pages, and applies per-page redactions for
  numerics, dates, and other MNPI. After each tool call a flattened raster PDF
  is regenerated with black rectangles overlaid on the source pages.
- **Reviewer** — reads the post-redaction text view and runs Azure AI Language
  PII detection on it, then returns a strict JSON verdict
  (`APPROVED` / `APPROVED_WITH_ISSUES` / `REVISE`) with a list of missed items.

Output is a flattened raster PDF: the original text stream is gone, so
redactions cannot be reversed via copy/paste or text extraction.

---

## Project layout

```
document-intelligence/
├── README.md                  # this file
├── requirements.txt
├── .env.example
├── main.py                    # CLI entrypoint
├── samples/                   # drop your input PDFs here
├── output/                    # generated artifacts (created on first run)
└── src/
    ├── config.py              # paths, iteration limits, .env loading
    ├── logger.py              # shared logger
    ├── models.py              # PageWord, RunContext dataclasses
    ├── redaction_state.py     # active-run state holder + redacted text view
    ├── redaction_policy.py    # the redaction rules used by both agents
    ├── azure_clients.py       # AAD credential, chat client, DI endpoint
    ├── pdf_text_extractor.py  # pdfplumber + Document Intelligence OCR fallback
    ├── pdf_renderer.py        # rasterize + overlay black-box redactions
    ├── agent_tools.py         # tools exposed to the agents
    ├── agents.py              # redactor + reviewer agent factories
    └── orchestrator.py        # run_redaction_loop + verdict parsing
```

---

## Prerequisites

- **Python 3.11+** (tested on 3.13).
- **Azure CLI** (`az login` completed) — the pipeline authenticates with the
  signed-in identity via `AzureCliCredential` (with an interactive browser
  fallback). The identity needs the **Cognitive Services User** role (or
  higher) on each Azure resource referenced in `.env`.
- An **Azure OpenAI** or **Microsoft Foundry** chat model deployment.
- An **Azure AI Language** resource for PII detection.
- (Optional) An **Azure AI Document Intelligence** resource — only needed if
  you intend to redact scanned / image-only PDFs.

---

## Setup

```powershell
# 1. (Recommended) create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment variables
Copy-Item .env.example .env
# then edit .env with your endpoints and deployment names

# 4. Sign in to Azure (used by AzureCliCredential)
az login
```

Required environment variables (see [.env.example](.env.example)):

| Variable | Purpose |
| --- | --- |
| `AI_FOUNDRY_PROJECT_ENDPOINT` *(or `AZURE_OPENAI_ENDPOINT`)* | Chat model host |
| `AZURE_AI_MODEL_DEPLOYMENT_NAME` *(or `AZURE_OPENAI_CHAT_DEPLOYMENT_NAME`)* | Chat model deployment name |
| `AZURE_LANGUAGE_ENDPOINT` | Azure AI Language endpoint (PII detection) |
| `AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT` *(optional)* | Forced OCR endpoint for image PDFs |

---

## Running

Drop your input PDF at `samples/sample_input.pdf` (or pass an explicit path):

```powershell
python main.py --input "samples/your-pitch-deck.pdf"
```

To tag a run for prompt A/B comparison, pass `--run-label`:

```powershell
python main.py --input "samples/your-pitch-deck.pdf" --run-label "old-prompt"
# ...edit the redactor / reviewer prompt...
python main.py --input "samples/your-pitch-deck.pdf" --run-label "new-prompt-v2"
```

Artifacts written to `output/`:

| File | Description |
| --- | --- |
| `iteration_<N>_redacted.pdf` | Redacted PDF after each pass (1..N) |
| `final_redacted.pdf` | Copy of the last iteration's PDF |
| `audit_trail.json` | Full run log: per-iteration verdicts, missed items, totals |
| `metrics/run_<timestamp>_<label>.json` | Detailed per-run metrics (see below) |
| `metrics/runs_summary.csv` | One row per run, appended in chronological order |

The orchestration enforces **min 2 / max 5** iterations. Adjust in
[src/config.py](src/config.py) (`MIN_ITERATIONS`, `MAX_ITERATIONS`).

---

## Comparing prompt versions (metrics)

Every run writes:

- a detailed JSON file at
  `output/metrics/run_<timestamp>_<label>.json`, and
- a single appended row to `output/metrics/runs_summary.csv`.

Use `--run-label` to give each run a memorable tag (defaults to the run
timestamp). Open `runs_summary.csv` in Excel to compare runs side-by-side.

What gets counted:

- **Totals** — words and visual regions redacted across the whole run.
- **Redactor catches by source** — words/regions caught *proactively* on
  iteration 1 (no reviewer guidance yet) vs. *reviewer-driven* (added on
  iterations 2+ in response to reviewer feedback). A prompt revision that
  raises the proactive share and lowers the reviewer-driven share is doing
  more work up-front.
- **Redactor catches by tool** — words via `redact_all_matching_terms`
  (client-identity sweeps) vs. `apply_redactions` (per-page PII/MNPI
  spans); regions via `redact_visual_regions` (logos discovered up-front)
  vs. `redact_bbox` (logos the reviewer pointed at).
- **Reviewer misses flagged** — count of items the reviewer flagged as
  missed, bucketed by canonical type (`PII`, `MNPI`, `Logo`,
  `Consistency`, `Structure`, `Other`), both across all iterations and at
  the final iteration. Fewer total flags + fewer final flags = better
  prompt.

A short summary table is also printed to stdout at the end of every run.

---

## Customizing the policy

The redaction rules — what counts as PII / MNPI, what to leave intact, what to
sweep document-wide — live in
[src/redaction_policy.py](src/redaction_policy.py). Both the redactor and the
reviewer load the same policy text, so editing it once keeps the two agents in
sync.

---

## Troubleshooting

**`pdfplumber found 0 words … falling back to OCR`** — the input is a
scanned/image-only PDF. Set `AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT` (or rely on
`AI_FOUNDRY_PROJECT_ENDPOINT`).

**`Set AI_FOUNDRY_PROJECT_ENDPOINT or AZURE_OPENAI_ENDPOINT in .env`** — the
chat client could not resolve an endpoint; verify your `.env` and that
placeholder values starting with `<` have been replaced.

**Reviewer never reaches APPROVED** — this is iterative scrubbing of complex
documents; inspect `output/audit_trail.json` (`history[*].missed`) to see what
the reviewer is still flagging, then tighten the rules in
`redaction_policy.py` or extend the redactor prompt.

**Authentication errors against Language / Document Intelligence** — confirm
`az login` is current and that your identity has the **Cognitive Services
User** role on the resource.
