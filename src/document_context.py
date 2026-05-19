"""Auto-detect client vs. preparer identity from the input document.

Pitch decks have two distinct identities that the redactor and reviewer must
treat differently:

- **Client** — the company that is the subject of the deck. Aggressively
  redact its name, aliases, tickers, subsidiaries, executives, logos, and
  financials. This is *what the deck is about*.
- **Preparer** — the bank or advisory firm that authored the deck. Its
  firm-level name and branding are typically left intact (they're the
  presenter, not sensitive client data). Named bankers' direct contact
  info (phone, email) should still be covered as PII.

This module runs **one** LLM call at the start of a redaction run, before
either agent turns, and returns a :class:`DocumentContext` consumed by the
prompt templates.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field

from agent_framework import ChatAgent
from agent_framework.azure import AzureOpenAIChatClient

from .logger import log
from .models import PageWord


# Pages used as evidence for identity detection. Cover slide(s) usually
# declare the client + preparer; the closing slide(s) often carry banker
# contact info and preparer disclaimers.
_FRONT_PAGES = 2
_BACK_PAGES = 1
_MAX_EVIDENCE_CHARS = 8000


_JSON_OBJECT_PATTERN = re.compile(r"\{(?:[^{}]|(?:\{[^{}]*\}))*\}", re.DOTALL)


@dataclass
class PartyContext:
    """One side of the document — either the client or the preparer."""

    name: str = ""
    aliases: list[str] = field(default_factory=list)
    tickers: list[str] = field(default_factory=list)
    subsidiaries: list[str] = field(default_factory=list)
    people: list[str] = field(default_factory=list)
    industry: str = ""
    evidence: str = ""

    def all_identity_terms(self) -> list[str]:
        """Flat de-duplicated list of every identifier for this party."""
        seen: set[str] = set()
        out: list[str] = []
        for term in (
            [self.name, *self.aliases, *self.tickers, *self.subsidiaries]
        ):
            term = (term or "").strip()
            if term and term.lower() not in seen:
                seen.add(term.lower())
                out.append(term)
        return out


@dataclass
class DocumentContext:
    """Auto-detected client / preparer identity for a redaction run."""

    client: PartyContext = field(default_factory=PartyContext)
    preparer: PartyContext = field(default_factory=PartyContext)
    confidence: str = "low"  # "high" | "medium" | "low"
    raw_response: str = ""

    def to_dict(self) -> dict:
        return {
            "client": asdict(self.client),
            "preparer": asdict(self.preparer),
            "confidence": self.confidence,
        }

    def render_for_prompt(self) -> str:
        """Render as a short block suitable for injection into agent prompts."""
        if not self.client.name and not self.preparer.name:
            return (
                "Document context: auto-detection did not produce a confident "
                "result. Infer client / preparer identity yourself from the "
                "document content."
            )

        lines = [f"Document context (auto-detected, confidence={self.confidence}):"]
        if self.client.name:
            client_terms = ", ".join(self.client.all_identity_terms())
            lines.append(
                f"  CLIENT (subject of the deck — REDACT aggressively on every page): "
                f"{client_terms}"
            )
            if self.client.people:
                lines.append(
                    f"    Client people to redact: {', '.join(self.client.people)}"
                )
            if self.client.industry:
                lines.append(f"    Client industry hint: {self.client.industry}")
        if self.preparer.name:
            preparer_terms = ", ".join(self.preparer.all_identity_terms())
            lines.append(
                f"  PREPARER (authoring firm — DO NOT redact the firm name, logo, "
                f"or branding; this is the presenter, not the sensitive client): "
                f"{preparer_terms}"
            )
            if self.preparer.people:
                lines.append(
                    f"    Preparer bankers named in the deck (redact ONLY their "
                    f"direct contact info — phone/email/address — keep their "
                    f"names if they're attribution-only): "
                    f"{', '.join(self.preparer.people)}"
                )
        return "\n".join(lines)


_DETECTION_SYSTEM_INSTRUCTIONS = (
    "You analyze the first and last pages of an investment-banking pitch "
    "deck and identify two parties: the CLIENT (the company the deck is "
    "about / pitched to / analyzed) and the PREPARER (the bank or advisory "
    "firm that authored the deck). You return strict JSON only. Do not "
    "guess if evidence is weak; leave fields empty and set confidence "
    "accordingly."
)


_DETECTION_USER_TEMPLATE = """\
Identify the CLIENT and the PREPARER for this pitch deck. The client is the
company that is the subject of the deck. The preparer is the bank / advisory
firm that wrote and presents the deck (e.g. Goldman Sachs, Morgan Stanley,
JP Morgan, Lazard, Evercore, Centerview, Houlihan Lokey, etc.).

Heuristics:
- Cover slide typically reads like: "<Preparer> — Discussion materials
  prepared for <Client>", or "Project <codename> — Presentation to
  <Client>".
- Disclaimers, footers, and contact pages usually name the preparer.
- Tickers, executive bios, financial detail, and historical events are
  about the CLIENT.
- If the deck names a "Project <codename>", that is the client's
  transaction codename — include it as a client alias.

Document evidence (front + back pages):
---
{evidence}
---

Return ONLY this JSON object, no markdown fence, no commentary:
{{
  "client": {{
    "name": "primary legal or brand name, empty string if unknown",
    "aliases": ["other names, brand names, project codenames, abbreviations"],
    "tickers": ["stock tickers if any"],
    "subsidiaries": ["subsidiaries or product lines uniquely tied to the client"],
    "people": ["client executives or board members named (full names)"],
    "industry": "short industry descriptor or empty string",
    "evidence": "one short sentence quoting where this was inferred"
  }},
  "preparer": {{
    "name": "authoring firm name, empty string if unknown",
    "aliases": ["abbreviations or short forms (e.g. GS, JPM)"],
    "tickers": [],
    "subsidiaries": [],
    "people": ["named bankers / authors visible in the deck (full names)"],
    "industry": "",
    "evidence": "one short sentence quoting where this was inferred"
  }},
  "confidence": "high" | "medium" | "low"
}}
"""


def _collect_evidence_text(
    words: list[PageWord],
    total_pages: int,
) -> str:
    """Gather text from the first ``_FRONT_PAGES`` and last ``_BACK_PAGES`` pages."""
    if total_pages == 0:
        return ""
    front = set(range(min(_FRONT_PAGES, total_pages)))
    back = set(range(max(0, total_pages - _BACK_PAGES), total_pages))
    target_pages = sorted(front | back)

    by_page: dict[int, list[str]] = {p: [] for p in target_pages}
    for w in words:
        if w.page in by_page:
            by_page[w.page].append(w.text)

    chunks = []
    for p in target_pages:
        if by_page[p]:
            chunks.append(f"[Page {p + 1}]\n" + " ".join(by_page[p]))
    text = "\n\n".join(chunks)
    if len(text) > _MAX_EVIDENCE_CHARS:
        text = text[:_MAX_EVIDENCE_CHARS] + "\n... [truncated]"
    return text


def _parse_party(node: object) -> PartyContext:
    if not isinstance(node, dict):
        return PartyContext()

    def _str_list(key: str) -> list[str]:
        raw = node.get(key, [])
        if not isinstance(raw, list):
            return []
        return [str(x).strip() for x in raw if isinstance(x, (str, int, float)) and str(x).strip()]

    return PartyContext(
        name=str(node.get("name", "")).strip(),
        aliases=_str_list("aliases"),
        tickers=_str_list("tickers"),
        subsidiaries=_str_list("subsidiaries"),
        people=_str_list("people"),
        industry=str(node.get("industry", "")).strip(),
        evidence=str(node.get("evidence", "")).strip(),
    )


def _parse_response(text: str) -> DocumentContext:
    """Extract the JSON object and project it into a :class:`DocumentContext`."""
    for candidate in reversed(_JSON_OBJECT_PATTERN.findall(text)):
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        if "client" not in obj and "preparer" not in obj:
            continue
        confidence = str(obj.get("confidence", "low")).lower().strip()
        if confidence not in ("high", "medium", "low"):
            confidence = "low"
        return DocumentContext(
            client=_parse_party(obj.get("client")),
            preparer=_parse_party(obj.get("preparer")),
            confidence=confidence,
            raw_response=text,
        )
    return DocumentContext(raw_response=text)


async def detect_document_context(
    chat_client: AzureOpenAIChatClient,
    words: list[PageWord],
    total_pages: int,
) -> DocumentContext:
    """Run one LLM call to detect the client vs. preparer identities.

    Returns a populated :class:`DocumentContext` on success, or one with
    empty parties and ``confidence="low"`` if detection fails. Failures
    never raise — redaction proceeds with the agents inferring identity
    themselves.
    """
    evidence = _collect_evidence_text(words, total_pages)
    if not evidence.strip():
        log.warning("Document context: no evidence text on cover/back pages; skipping.")
        return DocumentContext()

    # A throwaway agent with no tools — we only want the model's text response.
    detector = ChatAgent(
        name="DocumentContextDetector",
        description="One-shot client/preparer identity detection.",
        instructions=_DETECTION_SYSTEM_INSTRUCTIONS,
        chat_client=chat_client,
    )
    prompt = _DETECTION_USER_TEMPLATE.format(evidence=evidence)

    try:
        result = await detector.run(prompt)
    except Exception as exc:  # pragma: no cover - network/credential failures
        log.warning("Document context detection failed: %s", exc)
        return DocumentContext()

    ctx = _parse_response(str(result))
    log.info(
        "Detected client=%r, preparer=%r, confidence=%s",
        ctx.client.name or "<unknown>",
        ctx.preparer.name or "<unknown>",
        ctx.confidence,
    )
    return ctx
