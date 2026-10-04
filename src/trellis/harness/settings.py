"""The deployment facts the harness reads, and nothing else (every one is in ``.env.example``).

Everything that is a design decision — limits, timeouts, prompts — is a named constant next to
the code that uses it. A setting here says *where* a service is, *whether* it exists in this
deployment, the credentials it is reached with, the facts only the host knows (where
undelivered writes may be kept on disk, how many runs a worker process takes at once), and how
much of the traffic is checked, and by which model — costs the deployment chooses (the
grounding and judge samples, the judge's model and virtual key). Who the
deployment is (its tenant) is not configured: the memory service says so about
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
    #: A directory memory writes this process could not deliver are kept in, replayed at
    #: the next start (``writes.py``); unset: they are logged and counted, then lost.
    spool_dir: str | None = None
    #: Runs a worker executes at once (``h.worker``, ``python -m trellis.harness.worker``); unset:
    #: the machine's CPU count, between 1 and 8.
    worker_concurrency: int | None = Field(default=None, ge=1)
    #: The share of successful runs whose answer is checked against the context it was given
    #: (the memory service's ``/v1/verify``), 0 to 1; chosen by the run id, so a run is either
    #: always or never sampled.
    grounding_sample: float = Field(default=0.1, ge=0.0, le=1.0)
    #: The model ``llm_judge`` asks, a Bifrost model name (through ``BIFROST_URL``); unset: the
    #: judged agent's own model (a ``ReAct``'s), logged once — a different, stronger model
    #: than the agent's avoids a model grading itself.
    judge_model: str | None = None
    #: The virtual key the judge's calls go through (its own budget and limits); unset:
    #: ``BIFROST_VIRTUAL_KEY``.
    judge_virtual_key: str | None = None
    #: The share of successful runs the online judges (``Harness(judges=[...])``) score, 0 to
    #: 1, chosen by the run id; unset: 0.1 when there are judges.
    judge_sample: float | None = Field(default=None, ge=0.0, le=1.0)

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
            spool_dir=get("TRELLIS_SPOOL_DIR"),
            worker_concurrency=get("TRELLIS_WORKER_CONCURRENCY"),  # type: ignore[arg-type]
            judge_model=get("TRELLIS_JUDGE_MODEL"),
            judge_virtual_key=get("TRELLIS_JUDGE_VIRTUAL_KEY"),
            judge_sample=get("TRELLIS_JUDGE_SAMPLE"),  # type: ignore[arg-type]
            **_given(grounding_sample=get("TRELLIS_GROUNDING_SAMPLE")),  # type: ignore[arg-type]
        )


def _given(**values: str | None) -> dict[str, str]:
    """The variables that are set: an unset one keeps its field's default."""
    return {name: value for name, value in values.items() if value is not None}


def parse_headers(text: str) -> dict[str, str]:
    """``OTEL_EXPORTER_OTLP_HEADERS`` as the OTel spec writes it: ``k1=v1,k2=v2``, values
    URL-encoded. Keys are kept lower-case (headers are case-insensitive)."""
    headers: dict[str, str] = {}
    for pair in text.split(","):
        key, sep, value = pair.partition("=")
        if sep and key.strip():
            headers[key.strip().lower()] = unquote(value.strip())
    return headers
