"""The deployment facts the harness reads from the environment, and nothing else (every one
is in ``.env.example``): where the models are and how they are reached (``LLM_BASE_URL``,
``LLM_API_KEY``, ``LLM_GATEWAY``, ``LLM_AUTH``, ``TRELLIS_JUDGE_VIRTUAL_KEY``), where the
services are and the key they are reached with (``MEMORY_URL``, ``TRELLIS_API_KEY``,
``RUNS_URL``) and where undelivered writes are kept (``TRELLIS_SPOOL_DIR``). Each setting
lives in exactly one layer:

* **the environment** — what differs per deployment: the nine above. The tools' own variables
  are read by their own SDKs, or under their own names, never mirrored here: ``OTEL_*`` (the
  OpenTelemetry SDK; ``service.version`` in ``OTEL_RESOURCE_ATTRIBUTES`` is the agents'
  version), ``LANGFUSE_*`` (its scores, datasets and — with no OpenTelemetry exporter variable
  — its traces), the providers' keys (``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``...: their SDKs),
  ``DOCKER_HOST``;
* **the code** — the agent's behaviour: models, prompts, tools, budgets, the worker's
  concurrency, the sandbox image;
* **the owning services** — the model gateway (Bifrost: routing, fallbacks, retries, rate
  limits, budgets, caching, stored prompts, the judge's model alias), the memory catalog
  (approval rules, tool risk), agent-runs (schedules);
* **internal facts** — thresholds (constants: a fixed share of runs is grounded), the model's
  window (its metadata), the tenant (the memory service says who ``TRELLIS_API_KEY`` is).

Explicit code (``Harness(config=Settings(...))``) beats the environment, and the environment
beats what is detected (the spool's default folder). ``python -m trellis.harness.doctor``
prints the effective settings and where each came from.
"""

from __future__ import annotations

import base64
import functools
import logging
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final
from urllib.parse import unquote

from pydantic import BaseModel, ConfigDict

log = logging.getLogger("trellis.settings")

#: Langfuse's own defaults, as its SDK has them: its cloud, and where on a host it takes OTLP
#: (and so how its host is recognised in an endpoint).
LANGFUSE_CLOUD: Final = "https://cloud.langfuse.com"
LANGFUSE_OTLP_PATH: Final = "/api/public/otel"
#: The header Langfuse v4 (and its Cloud) processes the OTLP spans it is sent in real time with.
LANGFUSE_INGESTION_HEADER: Final = "x-langfuse-ingestion-version"
LANGFUSE_INGESTION_VERSION: Final = "4"
#: The OpenTelemetry SDK's own exporter variables: with any set, the SDK's exporter reads them
#: itself (the harness derives nothing from Langfuse's keys).
OTEL_EXPORTER_ENV: Final = (
    "OTEL_EXPORTER_OTLP_ENDPOINT",
    "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
)
#: The variables that mark a container or a serverless runtime, where a folder on local disk
#: does not outlive the process: the spool's default is not taken there.
EPHEMERAL_MARKERS: Final = ("K_SERVICE", "KUBERNETES_SERVICE_HOST", "AWS_LAMBDA_FUNCTION_NAME")
#: Each setting, by the variable it is read from.
VARIABLES: Final = {
    "llm_base_url": "LLM_BASE_URL",
    "llm_api_key": "LLM_API_KEY",
    "llm_gateway": "LLM_GATEWAY",
    "llm_auth": "LLM_AUTH",
    "api_key": "TRELLIS_API_KEY",
    "memory_url": "MEMORY_URL",
    "runs_url": "RUNS_URL",
    "judge_virtual_key": "TRELLIS_JUDGE_VIRTUAL_KEY",
    "spool_dir": "TRELLIS_SPOOL_DIR",
}


class Settings(BaseModel):
    """One deployment. ``Settings.from_env()`` is what ``Harness()`` uses."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The OpenAI-compatible ``/v1`` root every model name is sent to (``trellis.harness.
    #: gateways``); unset: each provider's own SDK.
    llm_base_url: str | None = None
    #: The endpoint's bearer key (Bifrost's: the agent's virtual key — which models and MCP
    #: tools it may use, its budget); the memory service's own model work for the agent is
    #: sent to the same endpoint with it.
    llm_api_key: str | None = None
    #: The route's name in the gateways' registry: ``openai-compatible`` (unset, with
    #: ``LLM_BASE_URL``), ``bifrost``, or one a package registers.
    llm_gateway: str | None = None
    #: How the endpoint's requests are authenticated: ``bearer`` (unset; ``LLM_API_KEY``),
    #: ``google`` (a refreshed Google token), ``aws`` (SigV4) — bifrost-sdk's strategies.
    llm_auth: str | None = None
    #: The one Trellis key, for the memory service and agent-runs alike.
    api_key: str | None = None
    #: The memory service: memory is on exactly when this is set.
    memory_url: str | None = None
    #: agent-runs: durable runs, the worker queue, the inbox and schedules (else in process).
    runs_url: str | None = None
    #: A directory memory writes this process could not deliver are kept in, replayed at
    #: the next start (``writes.py``): ``TRELLIS_SPOOL_DIR``, else the XDG state folder where
    #: it lasts (:func:`default_spool`); unset: they are logged and counted, then lost.
    spool_dir: str | None = None
    #: The virtual key the judge's calls go through — its own budget, rate limits and models
    #: in Bifrost, so evaluation spend never exhausts the agents'; unset: ``LLM_API_KEY``.
    judge_virtual_key: str | None = None

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Settings:
        given: dict[str, Any] = {name: value for name, (value, _) in sourced(environ).items()}
        return cls(**given)


def sourced(environ: Mapping[str, str] | None = None) -> dict[str, tuple[object, str]]:
    """Each setting the environment gives, with where it came from (the variable's name, or
    ``detected``): what :meth:`Settings.from_env` builds from and what ``doctor`` prints."""
    env = os.environ if environ is None else environ
    found: dict[str, tuple[object, str]] = {}
    for field, name in VARIABLES.items():
        value = env.get(name, "").strip()
        if value:
            found[field] = (value, name)
    if "spool_dir" not in found:
        spool = default_spool(env)
        if spool is not None:
            found["spool_dir"] = (spool, "detected (the XDG state folder)")
    return found


def service_version(environ: Mapping[str, str]) -> str | None:
    """``service.version`` in ``OTEL_RESOURCE_ATTRIBUTES``, read by the OpenTelemetry spec's
    format for it (``k=v,k2=v2``, values URL-encoded) — what ``Resource.create()`` reads."""
    attributes = parse_headers(environ.get("OTEL_RESOURCE_ATTRIBUTES", ""))
    return attributes.get("service.version") or None


def default_spool(environ: Mapping[str, str]) -> str | None:
    """The XDG state folder for undelivered writes (``$XDG_STATE_HOME/trellis``, else
    ``~/.local/state/trellis``) — only where a folder on this disk outlives the process: not in
    a container or a serverless runtime (:data:`EPHEMERAL_MARKERS`, ``/.dockerenv``), and only
    when it can be written. ``None`` otherwise, said once: an undelivered write is then lost
    at a restart."""
    if any(environ.get(m) for m in EPHEMERAL_MARKERS) or _container():
        _no_spool("a container or a serverless runtime")
        return None
    base = environ.get("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
    folder = Path(base) / "trellis"
    try:
        folder.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=folder):
            pass
    except OSError:
        _no_spool(f"{folder} cannot be written")
        return None
    return str(folder)


def _container() -> bool:
    """Whether this process runs in a Docker container (its ``/.dockerenv``)."""
    return Path("/.dockerenv").exists()


@functools.cache
def _no_spool(why: str) -> None:
    log.warning(
        "no spool for undelivered memory writes (%s): they will not survive a restart; set "
        "TRELLIS_SPOOL_DIR to a lasting folder",
        why,
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


def basic_auth(public_key: str, secret_key: str) -> str:
    """The ``Authorization`` value Langfuse takes its key pair as: ``Basic base64(pk:sk)``."""
    return "Basic " + base64.b64encode(f"{public_key}:{secret_key}".encode()).decode("ascii")


def langfuse_keys(environ: Mapping[str, str]) -> tuple[str, str, str] | None:
    """Langfuse's host and key pair, by the names its own SDK reads (``LANGFUSE_BASE_URL``,
    unset: its cloud; ``LANGFUSE_PUBLIC_KEY``, ``LANGFUSE_SECRET_KEY``); ``None`` without both
    keys."""
    public = environ.get("LANGFUSE_PUBLIC_KEY", "").strip()
    secret = environ.get("LANGFUSE_SECRET_KEY", "").strip()
    if not (public and secret):
        return None
    host = environ.get("LANGFUSE_BASE_URL", "").strip() or LANGFUSE_CLOUD
    return host.rstrip("/"), public, secret


def langfuse_otlp(host: str | None, public_key: str, secret_key: str) -> tuple[str, dict[str, str]]:
    """The OTLP endpoint and headers Langfuse's keys mean, as its SDK exports to: the host's
    ``/api/public/otel`` (``host`` unset: Langfuse Cloud) with Basic auth."""
    endpoint = (host or LANGFUSE_CLOUD).rstrip("/") + LANGFUSE_OTLP_PATH
    headers = {
        "authorization": basic_auth(public_key, secret_key),
        LANGFUSE_INGESTION_HEADER: LANGFUSE_INGESTION_VERSION,
    }
    return endpoint, headers
