"""The deployment facts the harness reads, and nothing else (every one is in ``.env.example``).

Everything that is a design decision — limits, timeouts, prompts, sample rates — is a named
constant next to the code that uses it. A setting here says *where* a service is, *whether* it
exists in this deployment, and the credentials it is reached with. Who the deployment is (its
tenant, whether it may write memory) is not configured: the memory service says so about
``TRELLIS_API_KEY`` (``GET /v1/keys/self``).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from urllib.parse import unquote

from pydantic import BaseModel, ConfigDict, Field


class Settings(BaseModel):
    """One deployment. ``Settings.from_env()`` is what ``Harness()`` uses."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Bifrost's OpenAI-compatible ``/v1`` base: MCP tools, ``ReAct`` models.
    bifrost_url: str | None = None
    #: The agent's Bifrost virtual key: which models and MCP tools it may use, its budget —
    #: and the key the memory service's own LLM work for this agent is billed to.
    bifrost_virtual_key: str | None = None
    #: The one Trellis key, for the memory service and agent-runs alike.
    api_key: str | None = None
    #: The memory service: memory is on exactly when this is set.
    memory_url: str | None = None
    #: agent-runs: durable runs, the worker queue, the inbox and schedules (else in process).
    runs_url: str | None = None
    #: OTLP/HTTP traces endpoint (Langfuse's, or a collector's).
    otlp_endpoint: str | None = None
    #: OTLP headers (``OTEL_EXPORTER_OTLP_HEADERS``, parsed).
    otlp_headers: dict[str, str] = Field(default_factory=dict)

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Settings:
        env = os.environ if environ is None else environ

        def get(name: str) -> str | None:
            value = env.get(name, "").strip()
            return value or None

        return cls(
            bifrost_url=get("BIFROST_URL"),
            bifrost_virtual_key=get("BIFROST_VIRTUAL_KEY"),
            api_key=get("TRELLIS_API_KEY"),
            memory_url=get("MEMORY_URL"),
            runs_url=get("RUNS_URL"),
            otlp_endpoint=get("OTEL_EXPORTER_OTLP_ENDPOINT"),
            otlp_headers=parse_headers(get("OTEL_EXPORTER_OTLP_HEADERS") or ""),
        )


def parse_headers(text: str) -> dict[str, str]:
    """``OTEL_EXPORTER_OTLP_HEADERS`` as the OTel spec writes it: ``k1=v1,k2=v2``, values
    URL-encoded. Keys are kept lower-case (headers are case-insensitive)."""
    headers: dict[str, str] = {}
    for pair in text.split(","):
        key, sep, value = pair.partition("=")
        if sep and key.strip():
            headers[key.strip().lower()] = unquote(value.strip())
    return headers
