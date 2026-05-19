"""Agent factory functions for the redactor and reviewer.

The system instructions are kept intentionally short. Detailed policy lives
in :mod:`redaction_policy` and is injected via the user message at the start
of each iteration in :mod:`orchestrator`. Long, imperative system prompts
tend to trigger Azure OpenAI Prompt Shields jailbreak heuristics.
"""

from __future__ import annotations

from agent_framework import ChatAgent
from agent_framework.azure import AzureOpenAIChatClient

from .agent_tools import (
    apply_redactions,
    detect_logos_on_rendered_pdf,
    detect_pii_with_language_service,
    extract_pdf_words,
    list_visual_regions,
    read_redacted_text,
    redact_all_matching_terms,
    redact_bbox,
    redact_visual_regions,
)


_REDACTOR_INSTRUCTIONS = (
    "You help prepare business documents for external sharing by identifying "
    "text that should be covered before the document is shared. You do this "
    "by calling tools that mark word spans; a separate process draws the "
    "black rectangles. The user message will provide the policy rules and "
    "the suggested workflow. Follow them and use your tools to mark every "
    "span that the policy describes. Favor coverage and consistency, and "
    "only use word indices returned by `extract_pdf_words`."
)


_REVIEWER_INSTRUCTIONS = (
    "You audit a sanitized business document and decide whether it is ready "
    "to share. You do not modify files. Use your tools to read the current "
    "text and to scan it for sensitive entities, then return a JSON verdict "
    "in the format the user message asks for."
)


def build_redactor_agent(chat_client: AzureOpenAIChatClient) -> ChatAgent:
    """Build the redactor agent that marks sensitive spans for coverage."""
    return ChatAgent(
        name="RedactorAgent",
        description="Marks sensitive spans in a business document.",
        instructions=_REDACTOR_INSTRUCTIONS,
        chat_client=chat_client,
        tools=[
            extract_pdf_words,
            redact_all_matching_terms,
            apply_redactions,
            list_visual_regions,
            redact_visual_regions,
            redact_bbox,
        ],
    )


def build_reviewer_agent(chat_client: AzureOpenAIChatClient) -> ChatAgent:
    """Build the reviewer agent that audits the sanitized document."""
    return ChatAgent(
        name="ReviewerAgent",
        description="Audits a sanitized business document.",
        instructions=_REVIEWER_INSTRUCTIONS,
        chat_client=chat_client,
        tools=[
            read_redacted_text,
            detect_pii_with_language_service,
            detect_logos_on_rendered_pdf,
        ],
    )
