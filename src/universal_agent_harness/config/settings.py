"""Harness configuration (§66/§67): a YAML file, environment overrides, or plain Python.

Everything is optional and every default is safe: no memory writes without a memory client,
no raw prompts in telemetry, non-blocking observability, retries off unless asked for.
Validation happens at construction, so a misconfigured process fails at startup rather
than on the first agent execution.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

TelemetryProviderName = Literal["opentelemetry", "noop"]
FailureMode = Literal["non_blocking", "fail_closed"]


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class MemoryConfig(_Section):
    """When and what the harness reads from and writes to the Memory Service (§11)."""

    enabled: bool = True
    retrieve_before: bool = True
    observe_after: bool = True
    observe_input: bool = True
    observe_output: bool = True
    observe_tool_results: bool = False
    observe_claims: bool = True
    private_by_default: bool = False
    record_messages: bool = False
    token_budget: int | None = None
    require_evidence: bool = False
    #: ``non_blocking``: a memory failure degrades the run (a warning is attached).
    #: ``fail_closed``: a memory failure fails the run (§77).
    failure_mode: FailureMode = "non_blocking"
    #: Observations are written after the result is returned; this bounds that work.
    writeback: bool = True
    writeback_max_pending: int = 256
    retrieval_timeout_seconds: float | None = 10.0
    observation_timeout_seconds: float | None = 15.0


class CaptureConfig(_Section):
    """What telemetry may contain (§26). Raw content is opt-in, per tenant/environment."""

    agents: bool = True
    models: bool = True
    tools: bool = True
    memory_operations: bool = True
    retrieval_metadata: bool = True

    raw_prompts: bool = False
    raw_agent_inputs: bool = False
    raw_agent_outputs: bool = False
    raw_model_inputs: bool = False
    raw_model_outputs: bool = False
    raw_tool_inputs: bool = False
    raw_tool_outputs: bool = False
    raw_memory_content: bool = False
    #: Identity attributes. ``user_id`` reaches a backend only when this is on and policy allows.
    user_id: bool = False
    thread_id: bool = True


class SamplingConfig(_Section):
    """Head sampling (§28). Errors and critical agents can always be kept."""

    sample_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    error_sample_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    critical_agent_sample_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    critical_agents: tuple[str, ...] = ()


class LangfuseConfig(_Section):
    """Langfuse as an optional provider. Business logic never depends on it (§15/§32)."""

    enabled: bool = False
    #: ``sdk`` uses the installed Langfuse SDK; ``otlp`` exports OTel spans with Langfuse
    #: attributes straight to Langfuse's OTLP endpoint (no SDK needed); ``auto`` prefers the
    #: SDK and falls back to ``otlp`` when it is not installed.
    mode: Literal["auto", "sdk", "otlp"] = "auto"
    base_url: str | None = None
    public_key: str | None = None
    secret_key: str | None = None
    environment: str | None = None
    release: str | None = None
    #: A Langfuse outage must not fail business execution unless explicitly required (§32).
    failure_mode: FailureMode = "non_blocking"
    flush_on_exit: bool = True
    debug: bool = False
    capture: CaptureConfig = CaptureConfig()
    sampling: SamplingConfig = SamplingConfig()

    @property
    def configured(self) -> bool:
        return bool(self.public_key and self.secret_key)


class TelemetryConfig(_Section):
    """OpenTelemetry is the canonical contract; Langfuse rides on top of it (§21)."""

    enabled: bool = True
    provider: TelemetryProviderName = "opentelemetry"
    service_name: str = "agent-harness"
    service_version: str | None = None
    #: Configure an OTel SDK provider ourselves. Off by default: most applications already
    #: configure OpenTelemetry, and the harness must not fight them for the global provider.
    configure_sdk: bool = False
    exporter: Literal["none", "console", "otlp"] = "none"
    endpoint: str | None = None
    metrics_enabled: bool = True
    capture: CaptureConfig = CaptureConfig()
    sampling: SamplingConfig = SamplingConfig()
    failure_mode: FailureMode = "non_blocking"


class ObservabilityConfig(_Section):
    langfuse: LangfuseConfig = LangfuseConfig()
    #: JSON logs with trace/agent identifiers on every line (§36).
    structured_logging: bool = True
    log_level: str = "INFO"


class RetryConfig(_Section):
    """Retries are explicit and category-gated (§40)."""

    enabled: bool = False
    max_attempts: int = Field(default=2, ge=1, le=10)
    initial_backoff_seconds: float = 0.1
    max_backoff_seconds: float = 5.0
    backoff_multiplier: float = 2.0
    jitter: bool = True
    #: Only these error categories are ever retried, and only when the agent is idempotent.
    retry_categories: tuple[str, ...] = ("TIMEOUT", "RATE_LIMIT", "DEPENDENCY")


class TimeoutConfig(_Section):
    """Hierarchical deadlines; a child never outlives its parent (§38)."""

    default_seconds: float | None = 30.0
    memory_seconds: float | None = 10.0
    model_seconds: float | None = 60.0
    tool_seconds: float | None = 30.0


class ModelsConfig(_Section):
    provider: str = "direct"
    default_model: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)


class ToolsConfig(_Section):
    provider: str = "local"
    record_to_memory: bool = True
    #: Look up the Memory Service tool cache before executing a cacheable tool.
    use_memory_cache: bool = False
    options: dict[str, Any] = Field(default_factory=dict)


class ArtifactsConfig(_Section):
    enabled: bool = True
    provider: str = "memory"
    #: Anything larger than this in a result's ``data`` is a candidate for artifact offload.
    inline_max_bytes: int = 64_000
    options: dict[str, Any] = Field(default_factory=dict)


class EvaluationConfig(_Section):
    enabled: bool = False
    #: Evaluation is async by default; synchronous scoring blocks the result (§29).
    synchronous: bool = False
    sample_rate: float = Field(default=1.0, ge=0.0, le=1.0)


class RegistryConfig(_Section):
    enabled: bool = False
    provider: str = "noop"


class PolicyConfig(_Section):
    enabled: bool = False
    provider: str = "noop"
    #: With no policy provider configured, a ``fail_closed`` policy refuses to execute.
    failure_mode: FailureMode = "non_blocking"


class FrameworksConfig(_Section):
    langgraph: bool = True
    crewai: bool = False


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
    registry: RegistryConfig = RegistryConfig()
    policy: PolicyConfig = PolicyConfig()
    frameworks: FrameworksConfig = FrameworksConfig()

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

        The YAML may be the whole document (with ``harness:`` / ``frameworks:`` keys, as in
        the documentation) or just the harness section.
        """
        data: dict[str, Any] = {}
        if isinstance(source, str | Path):
            data = _read_yaml(Path(source))
        elif isinstance(source, dict):
            data = dict(source)
        data = _flatten_document(data)
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


def _flatten_document(data: dict[str, Any]) -> dict[str, Any]:
    """Accept both ``{harness: {...}, frameworks: {...}}`` and a bare harness mapping."""
    if "harness" not in data:
        return data
    out = dict(data["harness"] or {})
    if "frameworks" in data:
        out["frameworks"] = data["frameworks"]
    return out


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


#: Documented environment variables (§67) -> config path. Everything else stays file/code.
_ENV_MAP: dict[str, tuple[tuple[str, ...], Any]] = {
    "UAH_MEMORY_ENABLED": (("memory", "enabled"), _bool),
    "UAH_MEMORY_RETRIEVE_BEFORE": (("memory", "retrieve_before"), _bool),
    "UAH_MEMORY_OBSERVE_AFTER": (("memory", "observe_after"), _bool),
    "UAH_MEMORY_FAILURE_MODE": (("memory", "failure_mode"), str),
    "UAH_OTEL_ENABLED": (("telemetry", "enabled"), _bool),
    "UAH_OTEL_EXPORTER": (("telemetry", "exporter"), str),
    "UAH_OTEL_ENDPOINT": (("telemetry", "endpoint"), str),
    "UAH_OTEL_CONFIGURE_SDK": (("telemetry", "configure_sdk"), _bool),
    "UAH_SERVICE_NAME": (("telemetry", "service_name"), str),
    "UAH_LANGFUSE_ENABLED": (("observability", "langfuse", "enabled"), _bool),
    "UAH_LANGFUSE_MODE": (("observability", "langfuse", "mode"), str),
    "UAH_LANGFUSE_SAMPLE_RATE": (("observability", "langfuse", "sampling", "sample_rate"), float),
    "UAH_LANGFUSE_FAILURE_MODE": (("observability", "langfuse", "failure_mode"), str),
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
