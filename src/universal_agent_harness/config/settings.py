"""Harness configuration: a YAML file, environment variables, or plain Python.

Two rules keep this surface honest:

* **one source of truth per concern** — capture policy and sampling are defined once, in
  ``telemetry``, and every backend (including Langfuse) obeys them. A second, near-identical
  block per backend is how observability configuration rots;
* **no setting that does nothing** — a provider is enabled by *passing* it
  (``AgentHarness(policy=...)``), not by a flag that must agree with it.

Everything is optional and every default is safe: no memory writes without a memory client,
no payloads in telemetry, non-blocking observability, retries off unless asked for.
Validation happens at construction, so a misconfigured process fails at startup rather than
on the first agent execution.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

FailureMode = Literal["non_blocking", "fail_closed"]


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class MemoryConfig(_Section):
    """When and what the harness reads from and writes to the Memory Service."""

    enabled: bool = True
    #: Fetch a context bundle before the agent runs.
    retrieve_before: bool = True
    #: What is written back after the result is produced.
    observe_input: bool = True
    observe_output: bool = True
    observe_claims: bool = True
    record_outcome: bool = True
    #: Tool outputs frequently contain customer data, so they are not written by default.
    observe_tool_results: bool = False
    #: Mark everything this harness writes as visible only to the agent run.
    private_by_default: bool = False
    #: Also record the turn as chat messages (user/assistant), not only observations.
    record_messages: bool = False
    token_budget: int | None = None
    #: Writes happen after the result is returned, so the turn never waits for them.
    writeback: bool = True
    #: ``non_blocking``: a memory failure degrades the run (a warning is attached).
    #: ``fail_closed``: a memory failure fails the run.
    failure_mode: FailureMode = "non_blocking"


class CaptureConfig(_Section):
    """What telemetry may contain. Payload capture is opt-in, per tenant/environment."""

    #: Prompts, model inputs, tool arguments, agent inputs.
    inputs: bool = False
    #: Model completions, tool results, agent results.
    outputs: bool = False
    #: Retrieved memory text. Separate because it is the most sensitive of the three.
    memory_content: bool = False
    #: Identity attributes. ``user_id`` reaches a backend only when this is on.
    user_id: bool = False
    thread_id: bool = True


class SamplingConfig(_Section):
    """Head sampling. Errors and critical agents can always be kept."""

    sample_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    error_sample_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    critical_agent_sample_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    critical_agents: tuple[str, ...] = ()


class TelemetryConfig(_Section):
    """OpenTelemetry is the canonical contract; every backend rides on these spans."""

    enabled: bool = True
    service_name: str = "agent-harness"
    #: Configure an OTel SDK provider ourselves. Off by default: most applications already
    #: configure OpenTelemetry, and the harness must not fight them for the global provider.
    configure_sdk: bool = False
    exporter: Literal["none", "console", "otlp"] = "none"
    endpoint: str | None = None
    metrics_enabled: bool = True
    capture: CaptureConfig = CaptureConfig()
    sampling: SamplingConfig = SamplingConfig()


class LangfuseConfig(_Section):
    """Langfuse as an optional backend. It obeys ``telemetry.capture`` and
    ``telemetry.sampling`` — there is no second capture policy to keep in sync."""

    enabled: bool = False
    #: ``sdk`` uses the installed Langfuse SDK; ``otlp`` exports OTel spans carrying
    #: Langfuse attributes straight to its OTLP endpoint (no SDK needed); ``auto`` prefers
    #: the SDK and falls back to ``otlp``.
    mode: Literal["auto", "sdk", "otlp"] = "auto"
    base_url: str | None = None
    public_key: str | None = None
    secret_key: str | None = None
    environment: str | None = None
    release: str | None = None

    @property
    def configured(self) -> bool:
        return bool(self.public_key and self.secret_key)


class ObservabilityConfig(_Section):
    langfuse: LangfuseConfig = LangfuseConfig()
    #: JSON logs with trace/agent identifiers on every line.
    structured_logging: bool = True
    log_level: str = "INFO"
    #: Applies to every telemetry backend. ``fail_closed`` makes an observability outage a
    #: business outage — only for environments that genuinely require it.
    failure_mode: FailureMode = "non_blocking"


class RetryConfig(_Section):
    """Retries are explicit, category-gated and only for agents marked idempotent."""

    enabled: bool = False
    max_attempts: int = Field(default=2, ge=1, le=10)
    backoff_seconds: float = 0.1
    #: Only these error categories are ever retried.
    retry_categories: tuple[str, ...] = ("TIMEOUT", "RATE_LIMIT", "DEPENDENCY")


class TimeoutConfig(_Section):
    """Hierarchical deadlines; a child never outlives its parent."""

    default_seconds: float | None = 30.0
    memory_seconds: float | None = 10.0
    model_seconds: float | None = 60.0
    tool_seconds: float | None = 30.0


class ModelsConfig(_Section):
    default_model: str | None = None


class ToolsConfig(_Section):
    #: Record invocations in the Memory Service's tool memory.
    record_to_memory: bool = True
    #: Look up the Memory Service tool cache before executing a cacheable tool.
    use_memory_cache: bool = False


class ArtifactsConfig(_Section):
    enabled: bool = True
    #: Anything larger than this in a result's ``data`` is moved to the artifact store.
    inline_max_bytes: int = 64_000


class EvaluationConfig(_Section):
    enabled: bool = False
    #: Evaluation is asynchronous by default; synchronous scoring blocks the result.
    synchronous: bool = False
    sample_rate: float = Field(default=1.0, ge=0.0, le=1.0)


class HarnessConfig(BaseModel):
    """The whole configuration surface. Immutable once built."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    memory: MemoryConfig = MemoryConfig()
    telemetry: TelemetryConfig = TelemetryConfig()
    observability: ObservabilityConfig = ObservabilityConfig()
    models: ModelsConfig = ModelsConfig()
    tools: ToolsConfig = ToolsConfig()
    retries: RetryConfig = RetryConfig()
    timeouts: TimeoutConfig = TimeoutConfig()
    artifacts: ArtifactsConfig = ArtifactsConfig()
    evaluation_events: EvaluationConfig = EvaluationConfig()

    @model_validator(mode="after")
    def _validate(self) -> Self:
        lf = self.observability.langfuse
        if lf.enabled and not lf.configured:
            raise ValueError(
                "observability.langfuse: enabled without public_key/secret_key "
                "(set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY)"
            )
        if lf.enabled and not self.telemetry.enabled:
            raise ValueError("observability.langfuse requires telemetry.enabled")
        return self

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(
        cls,
        source: str | Path | dict[str, Any] | None = None,
        *,
        env: bool = True,
        overrides: dict[str, Any] | None = None,
    ) -> HarnessConfig:
        """Build a config from a YAML file/dict, then environment variables, then overrides.

        The YAML may be the whole document (with a ``harness:`` key, as in the
        documentation) or just the harness section.
        """
        data: dict[str, Any] = {}
        if isinstance(source, str | Path):
            data = _read_yaml(Path(source))
        elif isinstance(source, dict):
            data = dict(source)
        if "harness" in data:
            data = dict(data["harness"] or {})
        if env:
            data = _deep_merge(data, env_overrides(os.environ))
        if overrides:
            data = _deep_merge(data, overrides)
        return cls.model_validate(data)


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"harness config not found: {path}")
    loaded = yaml.safe_load(path.read_text()) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"harness config must be a mapping: {path}")
    return loaded


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _bool(raw: str) -> bool:
    return raw.strip().lower() in ("1", "true", "yes", "on")


#: Documented environment variables -> config path.
_ENV_MAP: dict[str, tuple[tuple[str, ...], Any]] = {
    "UAH_MEMORY_ENABLED": (("memory", "enabled"), _bool),
    "UAH_MEMORY_RETRIEVE_BEFORE": (("memory", "retrieve_before"), _bool),
    "UAH_MEMORY_FAILURE_MODE": (("memory", "failure_mode"), str),
    "UAH_OTEL_ENABLED": (("telemetry", "enabled"), _bool),
    "UAH_OTEL_EXPORTER": (("telemetry", "exporter"), str),
    "UAH_OTEL_ENDPOINT": (("telemetry", "endpoint"), str),
    "UAH_OTEL_CONFIGURE_SDK": (("telemetry", "configure_sdk"), _bool),
    "UAH_SERVICE_NAME": (("telemetry", "service_name"), str),
    "UAH_SAMPLE_RATE": (("telemetry", "sampling", "sample_rate"), float),
    "UAH_CAPTURE_INPUTS": (("telemetry", "capture", "inputs"), _bool),
    "UAH_CAPTURE_OUTPUTS": (("telemetry", "capture", "outputs"), _bool),
    "UAH_LANGFUSE_ENABLED": (("observability", "langfuse", "enabled"), _bool),
    "UAH_LANGFUSE_MODE": (("observability", "langfuse", "mode"), str),
    "UAH_OBSERVABILITY_FAILURE_MODE": (("observability", "failure_mode"), str),
    "LANGFUSE_HOST": (("observability", "langfuse", "base_url"), str),
    "LANGFUSE_BASE_URL": (("observability", "langfuse", "base_url"), str),
    "LANGFUSE_PUBLIC_KEY": (("observability", "langfuse", "public_key"), str),
    "LANGFUSE_SECRET_KEY": (("observability", "langfuse", "secret_key"), str),
    "LANGFUSE_TRACING_ENVIRONMENT": (("observability", "langfuse", "environment"), str),
    "UAH_DEFAULT_TIMEOUT": (("timeouts", "default_seconds"), float),
    "UAH_MODEL_TIMEOUT": (("timeouts", "model_seconds"), float),
    "UAH_TOOL_TIMEOUT": (("timeouts", "tool_seconds"), float),
    "UAH_RETRIES_ENABLED": (("retries", "enabled"), _bool),
    "UAH_RETRIES_MAX_ATTEMPTS": (("retries", "max_attempts"), int),
    "UAH_EVAL_EVENTS_ENABLED": (("evaluation_events", "enabled"), _bool),
    "UAH_LOG_LEVEL": (("observability", "log_level"), str),
}


def env_overrides(environ: Any) -> dict[str, Any]:
    """The subset of the configuration expressed by environment variables."""
    out: dict[str, Any] = {}
    for name, (path, cast) in _ENV_MAP.items():
        raw = environ.get(name)
        if raw is None or raw == "":
            continue
        try:
            value = cast(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid value for {name}: {raw!r}") from exc
        node = out
        for key in path[:-1]:
            node = node.setdefault(key, {})
        node[path[-1]] = value
    return out
