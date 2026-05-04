"""Azure client and credential factories.

Centralizes:
  - AAD credential chain (az CLI -> interactive browser fallback).
  - Azure OpenAI / Microsoft Foundry chat client construction.
  - Document Intelligence endpoint resolution for the OCR fallback.
"""

from __future__ import annotations

import os
import re

from agent_framework.azure import AzureOpenAIChatClient
from azure.identity import (
    AzureCliCredential,
    ChainedTokenCredential,
    InteractiveBrowserCredential,
)

from .logger import log


def get_azure_credential() -> ChainedTokenCredential:
    """Return an AAD credential chain (az CLI first, interactive browser fallback)."""
    return ChainedTokenCredential(
        AzureCliCredential(),
        InteractiveBrowserCredential(),
    )


def resolve_document_intelligence_endpoint() -> str | None:
    """Resolve a Document Intelligence endpoint for OCR fallback.

    Prefers the dedicated ``AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT`` and falls
    back to the Foundry resource hosting the project (``AI_FOUNDRY_PROJECT_ENDPOINT``).
    Returns ``None`` if neither is configured.
    """
    di_endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT", "").strip()
    if di_endpoint and not di_endpoint.startswith("<"):
        return di_endpoint.rstrip("/") + "/"

    foundry_project = os.getenv("AI_FOUNDRY_PROJECT_ENDPOINT", "").strip()
    if foundry_project and not foundry_project.startswith("<"):
        # Foundry/AI Services resource exposes Document Intelligence at the
        # same services.ai.azure.com host.
        return foundry_project.split("/api/projects/", 1)[0].rstrip("/") + "/"
    return None


def build_chat_client() -> AzureOpenAIChatClient:
    """Build the Azure OpenAI chat client used by both agents.

    Resolution order:
      1. ``AZURE_OPENAI_ENDPOINT`` + ``AZURE_OPENAI_CHAT_DEPLOYMENT_NAME``
         (a dedicated Azure OpenAI resource).
      2. ``AI_FOUNDRY_PROJECT_ENDPOINT`` + ``AZURE_AI_MODEL_DEPLOYMENT_NAME``
         (a Microsoft Foundry resource that hosts the model deployment).
    """
    foundry_project = os.getenv("AI_FOUNDRY_PROJECT_ENDPOINT", "").strip()
    aoai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()

    endpoint: str | None = None
    deployment: str | None = None

    if aoai_endpoint and not aoai_endpoint.startswith("<"):
        endpoint = aoai_endpoint
        deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
    elif foundry_project and not foundry_project.startswith("<"):
        # Foundry resources expose OpenAI inference at the `openai.azure.com`
        # subdomain (which accepts AAD tokens), not the `services.ai.azure.com`
        # agents endpoint.
        # e.g. https://<resource>.services.ai.azure.com/api/projects/<p>
        #   -> https://<resource>.openai.azure.com/
        host = foundry_project.split("/api/projects/", 1)[0]
        match = re.match(r"https?://([^.]+)\.services\.ai\.azure\.com", host)
        if match:
            endpoint = f"https://{match.group(1)}.openai.azure.com/"
        else:
            endpoint = host.rstrip("/") + "/"
        deployment = os.getenv("AZURE_AI_MODEL_DEPLOYMENT_NAME")

    if not endpoint:
        raise RuntimeError(
            "Set AI_FOUNDRY_PROJECT_ENDPOINT (preferred) or AZURE_OPENAI_ENDPOINT in .env"
        )
    if not deployment:
        raise RuntimeError(
            "Set AZURE_AI_MODEL_DEPLOYMENT_NAME (Foundry) or AZURE_OPENAI_CHAT_DEPLOYMENT_NAME in .env"
        )

    log.info("Chat client endpoint: %s (deployment=%s)", endpoint, deployment)

    # Drop any placeholder/legacy AZURE_OPENAI_API_KEY so the openai SDK doesn't
    # silently auto-pick it up and override the AAD bearer token.
    api_key = os.environ.get("AZURE_OPENAI_API_KEY", "")
    if not api_key or api_key.startswith("<"):
        os.environ.pop("AZURE_OPENAI_API_KEY", None)

    return AzureOpenAIChatClient(
        deployment_name=deployment,
        endpoint=endpoint,
        credential=get_azure_credential(),
    )
