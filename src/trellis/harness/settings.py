"""The deployment facts the harness reads from the environment, and nothing else (every one
is in ``.env.example``). Each setting lives in exactly one layer:

* **the environment** — what differs per deployment: where the services are and the keys they
  are reached with (``BIFROST_URL``, ``BIFROST_VIRTUAL_KEY``, ``TRELLIS_JUDGE_VIRTUAL_KEY``,
  ``MEMORY_URL``, ``TRELLIS_API_KEY``, ``RUNS_URL``), the share of runs evaluated
  (``TRELLIS_EVAL_SAMPLE``), where undelivered writes are kept (``TRELLIS_SPOOL_DIR``), and
  ``SANDBOX``. The tools' own standard variables (``OTEL_*``, ``LANGFUSE_*``, ``DOCKER_HOST``)
  are read by their SDKs or by the harness under their own names, never mirrored;
* **the code** — the agent's behaviour: models, prompts and skills (their sources given in
  code), tools, budgets, the worker's concurrency, the sandbox image (``SandboxSpec``);
* **the owning services** — Bifrost (routing, fallbacks, retries, rate limits, budgets,
  caching, the judge's model alias), the memory catalog (approval rules, tool risk),
  agent-runs (schedules);
* **internal facts** — thresholds (constants), the model's window (its metadata), the tenant
  (the memory service says who ``TRELLIS_API_KEY`` is).

Explicit code (``Harness(config=Settings(...))``) beats the environment, and the environment
beats what is detected (the spool's default folder). Each variable a release retired still
works for one release, with a warning naming what replaced it (:data:`DEPRECATED`).
``python -m trellis.harness.doctor`` prints the effective settings and where each came from.
"""

from __future__ import annotations

import base64
import functools
import logging
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final, Literal
from urllib.parse import unquote

from pydantic import BaseModel, ConfigDict, Field, model_validator

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
#: The share of successful runs evaluated (grounding and the online judges) when
#: ``TRELLIS_EVAL_SAMPLE`` is unset.
EVAL_SAMPLE: Final = 0.1
#: The variables that mark a container or a serverless runtime, where a folder on local disk
#: does not outlive the process: the spool's default is not taken there.
EPHEMERAL_MARKERS: Final = ("K_SERVICE", "KUBERNETES_SERVICE_HOST", "AWS_LAMBDA_FUNCTION_NAME")
#: Retired variables, each read for one more release with a warning naming its replacement.
DEPRECATED: Final[dict[str, str]] = {
    "PROMPTS_DIR": "Harness(prompts=[prompts_dir(...)]) in code",
    "SKILLS_DIR": "Harness(skills=[skills_dir(...)]) in code",
    "TRELLIS_AGENT_VERSION": "OTEL_RESOURCE_ATTRIBUTES=service.version=... (or h.wrap(version=))",
    "TRELLIS_WORKER_CONCURRENCY": "h.worker(concurrency=) or the worker's --concurrency",
    "SANDBOX_IMAGE": "SandboxSpec(image=) in code",
    "TRELLIS_JUDGE_MODEL": "llm_judge(model=) in code, or the gateway's judge alias",
    "TRELLIS_GROUNDING_SAMPLE": "TRELLIS_EVAL_SAMPLE",
    "TRELLIS_JUDGE_SAMPLE": "TRELLIS_EVAL_SAMPLE",
    "LANGFUSE_HOST": "LANGFUSE_BASE_URL",
}


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
    #: An OTLP/HTTP endpoint the harness exports its traces to itself: given in code, or
    #: Langfuse's, derived from its keys when no OpenTelemetry exporter variable is set
    #: (:data:`OTEL_EXPORTER_ENV`; with one set, the SDK's exporter reads its own variables).
    otlp_endpoint: str | None = None
    #: The headers sent with it (Langfuse's Basic auth, when derived).
    otlp_headers: dict[str, str] = Field(default_factory=dict)
    #: A directory memory writes this process could not deliver are kept in, replayed at
    #: the next start (``writes.py``): ``TRELLIS_SPOOL_DIR``, else the XDG state folder where
    #: it lasts (:func:`default_spool`); unset: they are logged and counted, then lost.
    spool_dir: str | None = None
    #: Deprecated (``h.worker(concurrency=)``): runs a worker executes at once; unset: the
    #: machine's CPU count, between 1 and 8.
    worker_concurrency: int | None = Field(default=None, ge=1)
    #: The version of the agents' code this deployment runs, recorded with every run they
    #: start: ``OTEL_RESOURCE_ATTRIBUTES``' ``service.version`` (bake it in at build time);
    #: ``h.wrap(..., version=)`` names one agent's own.
    agent_version: str | None = Field(default=None, max_length=128)
    #: The share of successful runs evaluated — checked against the context they were given
    #: (the memory service's ``/v1/verify``) and scored by the online judges — 0 to 1, chosen
    #: by the run id, so a run is either always or never sampled.
    eval_sample: float = Field(default=EVAL_SAMPLE, ge=0.0, le=1.0)
    #: Deprecated (``llm_judge(model=)``, or the gateway's judge alias): the judge's model.
    judge_model: str | None = None
    #: The virtual key the judge's calls go through — its own budget, rate limits and models
    #: in Bifrost, so evaluation spend never exhausts the agents'; unset: ``BIFROST_VIRTUAL_KEY``.
    judge_virtual_key: str | None = None
    #: Deprecated (``Harness(prompts=[prompts_dir(...)])``): a folder of prompts.
    prompts_dir: str | None = None
    #: Deprecated (``Harness(skills=[skills_dir(...)])``): a folder of Agent Skills.
    skills_dir: str | None = None
    #: Langfuse, by the names its own SDK reads (``LANGFUSE_BASE_URL``, unset: its cloud; the
    #: key pair): its prompts, its scores and datasets — and, with no OpenTelemetry exporter
    #: variable set, the traces too.
    langfuse_host: str | None = None
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    #: Where ``sandbox()`` makes the sandboxes its tools work in when it is given no provider:
    #: ``docker``, the Docker daemon ``DOCKER_HOST`` (else the docker context, else the local
    #: socket) names; unset: none — granting code execution is explicit.
    sandbox: Literal["docker"] | None = None
    #: Deprecated (``SandboxSpec(image=)``): the image sandboxes are made from.
    sandbox_image: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _renamed(cls, given: Any) -> Any:
        """The retired fields ``grounding_sample`` and ``judge_sample``, read as ``eval_sample``
        for one more release (said once)."""
        found = dict(given)  # its keywords: Settings(...) and from_env
        for old in ("grounding_sample", "judge_sample"):
            if old in found:
                _deprecated_field(old)
                value = found.pop(old)
                if value is not None:
                    found.setdefault("eval_sample", value)
        return found

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Settings:
        given: dict[str, Any] = {name: value for name, (value, _) in sourced(environ).items()}
        return cls(**given)


def sourced(environ: Mapping[str, str] | None = None) -> dict[str, tuple[object, str]]:
    """Each setting the environment gives, with where it came from (the variable's name,
    ``derived from ...``, ``detected``): what :meth:`Settings.from_env` builds from and what
    ``doctor`` prints. A retired variable still read is warned about once."""
    env = os.environ if environ is None else environ

    def get(name: str) -> str | None:
        value = env.get(name, "").strip()
        if value and name in DEPRECATED:
            _deprecated(name)
        return value or None

    found: dict[str, tuple[object, str]] = {}

    def put(field: str, value: object, source: str) -> None:
        if value is not None:
            found[field] = (value, source)

    for field, name in (
        ("bifrost_url", "BIFROST_URL"),
        ("bifrost_virtual_key", "BIFROST_VIRTUAL_KEY"),
        ("api_key", "TRELLIS_API_KEY"),
        ("memory_url", "MEMORY_URL"),
        ("runs_url", "RUNS_URL"),
        ("judge_virtual_key", "TRELLIS_JUDGE_VIRTUAL_KEY"),
        ("sandbox", "SANDBOX"),
        ("langfuse_public_key", "LANGFUSE_PUBLIC_KEY"),
        ("langfuse_secret_key", "LANGFUSE_SECRET_KEY"),
        ("prompts_dir", "PROMPTS_DIR"),
        ("skills_dir", "SKILLS_DIR"),
        ("worker_concurrency", "TRELLIS_WORKER_CONCURRENCY"),
        ("judge_model", "TRELLIS_JUDGE_MODEL"),
        ("sandbox_image", "SANDBOX_IMAGE"),
    ):
        put(field, get(name), name)
    for name in ("LANGFUSE_BASE_URL", "LANGFUSE_HOST"):
        if "langfuse_host" not in found:
            put("langfuse_host", get(name), name)
    for name in ("TRELLIS_EVAL_SAMPLE", "TRELLIS_GROUNDING_SAMPLE", "TRELLIS_JUDGE_SAMPLE"):
        if "eval_sample" not in found:
            put("eval_sample", get(name), name)
    version = service_version(env)
    if version is not None:
        put("agent_version", version, "OTEL_RESOURCE_ATTRIBUTES (service.version)")
    else:
        put("agent_version", get("TRELLIS_AGENT_VERSION"), "TRELLIS_AGENT_VERSION")
    spool = get("TRELLIS_SPOOL_DIR")
    if spool is not None:
        put("spool_dir", spool, "TRELLIS_SPOOL_DIR")
    else:
        put("spool_dir", default_spool(env), "detected (the XDG state folder)")
    public, secret = get("LANGFUSE_PUBLIC_KEY"), get("LANGFUSE_SECRET_KEY")
    if public and secret and not any(get(n) for n in OTEL_EXPORTER_ENV):
        host = found.get("langfuse_host", (None, ""))[0]
        endpoint, headers = langfuse_otlp(host if isinstance(host, str) else None, public, secret)
        put("otlp_endpoint", endpoint, "derived from LANGFUSE_PUBLIC_KEY/SECRET_KEY")
        put("otlp_headers", headers, "derived from LANGFUSE_PUBLIC_KEY/SECRET_KEY")
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


@functools.cache
def _deprecated_field(name: str) -> None:
    log.warning("Settings(%s=) is deprecated and goes in the next release: use eval_sample", name)


@functools.cache
def _deprecated(name: str) -> None:
    log.warning("%s is deprecated and goes in the next release: use %s", name, DEPRECATED[name])


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


def langfuse_otlp(host: str | None, public_key: str, secret_key: str) -> tuple[str, dict[str, str]]:
    """The OTLP endpoint and headers Langfuse's keys mean, as its SDK exports to: the host's
    ``/api/public/otel`` (``host`` unset: Langfuse Cloud) with Basic auth."""
    endpoint = (host or LANGFUSE_CLOUD).rstrip("/") + LANGFUSE_OTLP_PATH
    headers = {
        "authorization": basic_auth(public_key, secret_key),
        LANGFUSE_INGESTION_HEADER: LANGFUSE_INGESTION_VERSION,
    }
    return endpoint, headers
