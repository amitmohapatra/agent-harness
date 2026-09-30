"""The deployment facts the harness reads, and nothing else (every one is in ``.env.example``).

Everything that is a design decision — limits, timeouts, prompts, sample floors — is a named
constant next to the code that uses it. A setting here says *where* a service is and *whether*
it exists in this deployment.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, Field

#: The tenant a single-tenant deployment runs as when a call names none.
DEFAULT_TENANT = "default"


class Settings(BaseModel):
    """One deployment. ``Settings.from_env()`` is what ``Harness()`` uses."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant: str = DEFAULT_TENANT
    bifrost_url: str | None = None
    bifrost_virtual_key: str | None = None
    memory_url: str | None = None
    memory_api_key: str | None = None
    memory_model_key: str | None = None
    runs_url: str | None = None
    runs_api_key: str | None = None
    eval_sample: float = Field(default=0.1, ge=0.0, le=1.0)
    otlp_endpoint: str | None = None
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str | None = None

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Settings:
        env = os.environ if environ is None else environ

        def get(name: str) -> str | None:
            value = env.get(name, "").strip()
            return value or None

        sample = get("TRELLIS_EVAL_SAMPLE")
        return cls(
            tenant=get("TRELLIS_TENANT") or DEFAULT_TENANT,
            bifrost_url=get("BIFROST_URL"),
            bifrost_virtual_key=get("BIFROST_VIRTUAL_KEY"),
            memory_url=get("MEMORY_URL"),
            memory_api_key=get("MEMORY_API_KEY"),
            memory_model_key=get("TRELLIS_MEMORY_MODEL_KEY"),
            runs_url=get("RUNS_URL"),
            runs_api_key=get("RUNS_API_KEY"),
            eval_sample=float(sample) if sample is not None else 0.1,
            otlp_endpoint=get("OTEL_EXPORTER_OTLP_ENDPOINT"),
            langfuse_public_key=get("LANGFUSE_PUBLIC_KEY"),
            langfuse_secret_key=get("LANGFUSE_SECRET_KEY"),
            langfuse_host=get("LANGFUSE_HOST"),
        )
