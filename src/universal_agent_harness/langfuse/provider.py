"""Langfuse as a first-class *optional* telemetry backend (§22/§23).

Design in one line: **Langfuse rides on the harness's OpenTelemetry spans; it does not
replace them.** Langfuse's SDK attaches a span processor to the TracerProvider it is given
(the application's, when one is registered), so enabling Langfuse adds an exporter to the
existing span tree instead of building a second one. This provider therefore:

1. constructs the Langfuse client (the exporter side), telling it to export harness spans
   as well as its own via the SDK's public ``should_export_span`` hook;
2. enriches each harness span with Langfuse's documented OTel attributes so the span shows
   up as the right observation type — agent / generation / tool / retriever;
3. flushes on demand, and swallows its own failures in ``non_blocking`` mode (§32).

If the SDK is not installed, ``otlp`` mode does the same enrichment and relies on an OTLP
exporter aimed at Langfuse — no Langfuse code in the process at all.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

from universal_agent_harness.config.settings import LangfuseConfig
from universal_agent_harness.langfuse import attributes as LA
from universal_agent_harness.runtime.logging import get_logger
from universal_agent_harness.telemetry.otel import TRACER_NAME, OtelSpan
from universal_agent_harness.telemetry.span import NOOP_SPAN

log = get_logger("universal_agent_harness.langfuse")


class LangfuseSpanEnricher:
    """Wraps a harness span and adds Langfuse attributes to it (never a second span)."""

    __slots__ = ("_inner", "_observation_type")

    def __init__(self, inner: Any, observation_type: str) -> None:
        self._inner = inner
        self._observation_type = observation_type
        inner.set_attribute(LA.OBSERVATION_TYPE, observation_type)

    def set_attribute(self, key: str, value: Any) -> None:
        self._inner.set_attribute(key, value)
        mapped = _MAPPED.get(key)
        if mapped is not None:
            self._inner.set_attribute(mapped, _stringify(value))

    def set_attributes(self, attributes: Mapping[str, Any]) -> None:
        self._inner.set_attributes(attributes)
        extra = {
            _MAPPED[k]: _stringify(v)
            for k, v in attributes.items()
            if k in _MAPPED and v is not None
        }
        usage = _usage_details(attributes)
        if usage:
            extra[LA.OBSERVATION_USAGE_DETAILS] = json.dumps(usage)
        cost = attributes.get("gen_ai.usage.cost")
        if cost is not None:
            extra[LA.OBSERVATION_COST_DETAILS] = json.dumps({"total": cost})
        if extra:
            self._inner.set_attributes(extra)

    def add_event(self, name: str, attributes: Mapping[str, Any] | None = None) -> None:
        self._inner.add_event(name, attributes)

    def record_error(self, error: BaseException | str, **attributes: Any) -> None:
        self._inner.record_error(error, **attributes)
        message = (
            f"{type(error).__name__}: {error}" if isinstance(error, BaseException) else str(error)
        )
        self._inner.set_attributes(
            {LA.OBSERVATION_LEVEL: "ERROR", LA.OBSERVATION_STATUS_MESSAGE: message[:500]}
        )

    def set_status_ok(self) -> None:
        self._inner.set_status_ok()

    @property
    def inner(self) -> Any:
        return self._inner


#: Harness attribute -> Langfuse attribute. Payload attributes are already capture-gated by
#: the tracer, so anything arriving here is allowed to be exported.
_MAPPED: dict[str, str] = {
    "input.value": LA.OBSERVATION_INPUT,
    "output.value": LA.OBSERVATION_OUTPUT,
    "gen_ai.request.model": LA.OBSERVATION_MODEL,
    # The response model is the one actually used; set later, it overwrites the request's.
    "gen_ai.response.model": LA.OBSERVATION_MODEL,
    "gen_ai.prompt.id": LA.OBSERVATION_PROMPT_NAME,
    "gen_ai.prompt.version": LA.OBSERVATION_PROMPT_VERSION,
}


class LangfuseTelemetryProvider:
    """Adds Langfuse export and attribute mapping to the harness's OTel spans."""

    name = "langfuse"

    def __init__(self, config: LangfuseConfig, *, tracer_provider: Any = None) -> None:
        self.config = config
        self.strict = config.failure_mode == "fail_closed"
        self.client: Any = None
        self.mode = "disabled"
        if not config.enabled:
            return
        if config.mode in ("auto", "sdk"):
            self.client = _create_client(config, tracer_provider)
            if self.client is not None:
                self.mode = "sdk"
        if self.client is None:
            if config.mode == "sdk" and self.strict:
                raise RuntimeError("langfuse mode 'sdk' requested but the SDK is not installed")
            self.mode = "otlp"
            log.info("langfuse.mode", mode="otlp", reason="sdk unavailable or not requested")

    # -- telemetry provider port -------------------------------------------------
    @contextmanager
    def start_span(
        self, name: str, *, kind: str = "internal", attributes: Mapping[str, Any] | None = None
    ) -> Iterator[Any]:
        """Langfuse does not open a span of its own: it decorates the current OTel span.

        The composite provider calls every provider's ``start_span``; ours yields an
        enricher bound to the span OpenTelemetry has just made current, which keeps exactly
        one span per logical operation.
        """
        # The try/except deliberately covers only the *setup*: wrapping the ``yield`` would
        # swallow exceptions thrown into this context manager by the code it wraps.
        try:
            from opentelemetry import trace as otel_trace  # noqa: PLC0415

            span: Any = LangfuseSpanEnricher(
                OtelSpan(otel_trace.get_current_span()), LA.observation_type(name, kind)
            )
            if attributes:
                span.set_attributes(attributes)
        except Exception:
            if self.strict:
                raise
            log.debug("langfuse span enrichment failed")
            span = NOOP_SPAN
        yield span

    def record_event(self, name: str, attributes: Mapping[str, Any] | None = None) -> None:
        return None

    def record_metric(
        self,
        name: str,
        value: float,
        *,
        unit: str = "",
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        return None  # metrics go to OpenTelemetry; Langfuse consumes traces and scores

    def flush(self, timeout_seconds: float = 5.0) -> None:
        if self.client is None:
            return
        try:
            self.client.flush()
        except Exception:
            if self.strict:
                raise
            log.warning("langfuse flush failed")

    def shutdown(self) -> None:
        if self.client is not None:
            try:
                self.client.shutdown()
            except Exception:  # pragma: no cover
                log.debug("langfuse shutdown failed")

    # -- trace-level attributes (§24) ---------------------------------------------
    def decorate_trace(
        self,
        span: Any,
        *,
        trace_name: str,
        session_id: str | None = None,
        user_id: str | None = None,
        tags: list[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Set the Langfuse trace fields on the root agent span."""
        payload: dict[str, Any] = {LA.TRACE_NAME: trace_name}
        if session_id:
            payload[LA.TRACE_SESSION_ID] = session_id
        if user_id:
            payload[LA.TRACE_USER_ID] = user_id
        if tags:
            payload[LA.TRACE_TAGS] = list(tags)
        if metadata:
            payload[LA.TRACE_METADATA] = json.dumps(dict(metadata), default=str)[:8000]
        if self.config.environment:
            payload[LA.ENVIRONMENT] = self.config.environment
        if self.config.release:
            payload[LA.RELEASE] = self.config.release
        try:
            span.set_attributes(payload)
        except Exception:
            if self.strict:
                raise
            log.debug("langfuse trace decoration failed")


def _create_client(config: LangfuseConfig, tracer_provider: Any) -> Any:
    """Construct the Langfuse client, or return ``None`` when the SDK is unavailable."""
    try:
        from langfuse import Langfuse, is_default_export_span  # noqa: PLC0415
    except ImportError:
        return None
    kwargs: dict[str, Any] = {
        "public_key": config.public_key,
        "secret_key": config.secret_key,
        "debug": config.debug,
        "environment": config.environment,
        "release": config.release,
        "sample_rate": config.sampling.sample_rate,
        # Export the harness's own spans in addition to what Langfuse exports by default.
        "should_export_span": lambda span: (
            is_default_export_span(span)
            or (
                span.instrumentation_scope is not None
                and span.instrumentation_scope.name == TRACER_NAME
            )
        ),
    }
    if config.base_url:
        kwargs["host"] = config.base_url
    if tracer_provider is not None:
        kwargs["tracer_provider"] = tracer_provider
    try:
        return Langfuse(**{k: v for k, v in kwargs.items() if v is not None})
    except Exception as exc:  # pragma: no cover - misconfiguration at startup
        log.warning("langfuse client could not be created", error=str(exc))
        return None


def otlp_headers(config: LangfuseConfig) -> dict[str, str]:
    """Headers for exporting straight to Langfuse's OTLP endpoint (``otlp`` mode)."""
    token = base64.b64encode(f"{config.public_key}:{config.secret_key}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


def otlp_endpoint(config: LangfuseConfig) -> str:
    base = (config.base_url or "https://cloud.langfuse.com").rstrip("/")
    return f"{base}/api/public/otel/v1/traces"


def _usage_details(attributes: Mapping[str, Any]) -> dict[str, Any]:
    keys = {
        "gen_ai.usage.input_tokens": "input",
        "gen_ai.usage.output_tokens": "output",
        "gen_ai.usage.total_tokens": "total",
    }
    return {name: attributes[key] for key, name in keys.items() if attributes.get(key) is not None}


def _stringify(value: Any) -> Any:
    if isinstance(value, str | int | float | bool):
        return value
    try:
        return json.dumps(value, default=str, ensure_ascii=False)[:8000]
    except (TypeError, ValueError):  # pragma: no cover
        return str(value)[:8000]
