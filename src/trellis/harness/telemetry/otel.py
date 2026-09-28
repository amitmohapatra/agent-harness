"""The OpenTelemetry telemetry provider — the harness's canonical backend (§21).

Only the OTel **API** is required at runtime: if an application has not configured an SDK,
the API's no-op implementation is used and nothing breaks. ``configure_sdk`` exists for
applications that want the harness to set a provider up (console/OTLP), but it never
overrides a provider the application already registered.
"""

from __future__ import annotations

import logging
import re
import secrets
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any, Final

from opentelemetry import context as otel_context
from opentelemetry import metrics as otel_metrics
from opentelemetry import trace as otel_trace
from opentelemetry.trace import SpanKind, Status, StatusCode

from trellis.harness.config.settings import TelemetryConfig

log = logging.getLogger("trellis.harness.telemetry")

#: A W3C trace id: 32 lower-case hex characters.
W3C_TRACE_ID: Final = re.compile(r"[0-9a-f]{32}")

TRACER_NAME = "trellis.harness"

_SPAN_KINDS = {
    "agent": SpanKind.INTERNAL,
    "generation": SpanKind.CLIENT,
    "tool": SpanKind.CLIENT,
    "retriever": SpanKind.CLIENT,
    "internal": SpanKind.INTERNAL,
    "client": SpanKind.CLIENT,
    "server": SpanKind.SERVER,
}


class OtelSpan:
    """Adapts an OTel span to :class:`~trellis.harness.telemetry.span.HarnessSpan`."""

    __slots__ = ("_span",)

    def __init__(self, span: otel_trace.Span) -> None:
        self._span = span

    @property
    def otel_span(self) -> otel_trace.Span:
        return self._span

    def set_attribute(self, key: str, value: Any) -> None:
        if value is None:
            return
        self._span.set_attribute(key, _coerce(value))

    def set_attributes(self, attributes: Mapping[str, Any]) -> None:
        clean = {k: _coerce(v) for k, v in attributes.items() if v is not None}
        if clean:
            self._span.set_attributes(clean)

    def add_event(self, name: str, attributes: Mapping[str, Any] | None = None) -> None:
        self._span.add_event(name, {k: _coerce(v) for k, v in (attributes or {}).items()})

    def record_error(self, error: BaseException | str, **attributes: Any) -> None:
        if isinstance(error, BaseException):
            self._span.record_exception(error)
            message = f"{type(error).__name__}: {error}"
        else:
            message = str(error)
        self.set_attributes(attributes)
        self._span.set_status(Status(StatusCode.ERROR, message[:500]))

    def set_status_ok(self) -> None:
        self._span.set_status(Status(StatusCode.OK))


class OpenTelemetryTelemetryProvider:
    """Spans and metrics through the OTel API. Safe to construct without an SDK."""

    name = "opentelemetry"

    def __init__(self, config: TelemetryConfig | None = None) -> None:
        self.config = config or TelemetryConfig()
        if self.config.configure_sdk:
            configure_sdk(self.config)
        self._tracer = otel_trace.get_tracer(TRACER_NAME)
        self._meter = otel_metrics.get_meter(TRACER_NAME)
        self._histograms: dict[str, Any] = {}
        self._counters: dict[str, Any] = {}

    @contextmanager
    def start_span(
        self, name: str, *, kind: str = "internal", attributes: Mapping[str, Any] | None = None
    ) -> Iterator[OtelSpan]:
        span_kind = _SPAN_KINDS.get(kind, SpanKind.INTERNAL)
        clean = {k: _coerce(v) for k, v in (attributes or {}).items() if v is not None}
        with self._tracer.start_as_current_span(
            name, kind=span_kind, attributes=clean, record_exception=False
        ) as span:
            yield OtelSpan(span)

    def record_event(self, name: str, attributes: Mapping[str, Any] | None = None) -> None:
        span = otel_trace.get_current_span()
        if span is not None:
            span.add_event(name, {k: _coerce(v) for k, v in (attributes or {}).items()})

    def record_metric(
        self,
        name: str,
        value: float,
        *,
        unit: str = "",
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        """Counters end in ``.count``/``.total``; everything else is a histogram."""
        if not self.config.metrics_enabled:
            return
        labels = {k: _coerce(v) for k, v in (attributes or {}).items() if v is not None}
        if name.endswith((".count", ".total")):
            counter = self._counters.get(name)
            if counter is None:
                counter = self._counters[name] = self._meter.create_counter(name, unit=unit)
            counter.add(value, labels)
            return
        hist = self._histograms.get(name)
        if hist is None:
            hist = self._histograms[name] = self._meter.create_histogram(name, unit=unit)
        hist.record(value, labels)

    def flush(self, timeout_seconds: float = 5.0) -> None:
        provider = otel_trace.get_tracer_provider()
        force_flush = getattr(provider, "force_flush", None)
        if callable(force_flush):
            try:
                force_flush(int(timeout_seconds * 1000))
            except Exception:  # pragma: no cover - exporter-specific
                log.debug("otel force_flush failed", exc_info=True)

    def current_trace_id(self) -> str | None:
        ctx = otel_trace.get_current_span().get_span_context()
        if ctx and ctx.trace_id:
            return format(ctx.trace_id, "032x")
        return None

    @contextmanager
    def bind_trace(self, trace_id: str | None) -> Iterator[None]:
        """Start the spans opened inside in the trace ``trace_id`` names, when no span is
        active: the harness's spans, the SDKs' ``traceparent`` and the execution context
        then agree on one trace id (design §4), and every service a request touches is
        joined on it. Inside an active span nothing changes: the caller's trace wins."""
        current = otel_trace.get_current_span().get_span_context()
        if not trace_id or not W3C_TRACE_ID.fullmatch(trace_id) or current.is_valid:
            yield
            return
        parent = otel_trace.NonRecordingSpan(
            otel_trace.SpanContext(
                trace_id=int(trace_id, 16),
                span_id=secrets.randbits(64) or 1,
                is_remote=True,
                trace_flags=otel_trace.TraceFlags(otel_trace.TraceFlags.SAMPLED),
            )
        )
        token = otel_context.attach(otel_trace.set_span_in_context(parent))
        try:
            yield
        finally:
            otel_context.detach(token)
        return None


def configure_sdk(config: TelemetryConfig) -> bool:
    """Register an OTel SDK provider if — and only if — the application has not.

    Returns whether a provider was installed. Never replaces an existing provider: fighting
    the application for the global provider is how you lose an application's own traces.
    """
    current = otel_trace.get_tracer_provider()
    if not isinstance(current, otel_trace.ProxyTracerProvider):
        log.debug("OpenTelemetry provider already configured; harness will not replace it")
        return False
    try:
        # The SDK is an optional extra: the API alone is enough to run (§69).
        from opentelemetry.sdk.resources import Resource  # noqa: PLC0415
        from opentelemetry.sdk.trace import TracerProvider  # noqa: PLC0415
        from opentelemetry.sdk.trace.export import (  # noqa: PLC0415
            BatchSpanProcessor,
            ConsoleSpanExporter,
        )
    except ImportError:  # pragma: no cover - documented degradation (§77)
        log.warning("telemetry.configure_sdk requested but opentelemetry-sdk is not installed")
        return False

    from trellis.harness.harness import __version__  # noqa: PLC0415 - avoids a cycle

    provider = TracerProvider(
        resource=Resource.create(
            {"service.name": config.service_name, "service.version": __version__}
        )
    )
    if config.exporter == "console":
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    elif config.exporter == "otlp":
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (  # noqa: PLC0415
                OTLPSpanExporter,
            )
        except ImportError:  # pragma: no cover
            log.warning("otlp exporter requested but opentelemetry-exporter-otlp is missing")
        else:
            provider.add_span_processor(
                BatchSpanProcessor(OTLPSpanExporter(endpoint=config.endpoint))
                if config.endpoint
                else BatchSpanProcessor(OTLPSpanExporter())
            )
    otel_trace.set_tracer_provider(provider)
    return True


def _coerce(value: Any) -> Any:
    """OTel attributes are scalars or homogeneous sequences; anything else is stringified."""
    if isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, list | tuple):
        return [v if isinstance(v, bool | int | float | str) else str(v) for v in value]
    return str(value)
