"""OpenTelemetry: one span per run and per tool call, a handful of counters, and exporters.

The harness speaks the OTel *API* only. When ``OTEL_EXPORTER_OTLP_ENDPOINT`` is set (or a
Langfuse key pair is), :func:`configure` installs an SDK tracer provider with OTLP exporters —
Langfuse is reached through its OTLP endpoint, so there is no Langfuse SDK in the process. An
application that configured OTel itself keeps its own provider: ``configure`` never replaces
one that is already installed. Every attribute passes through the redactor first.
"""

from __future__ import annotations

import base64
import logging
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any, Final

from opentelemetry import metrics as otel_metrics
from opentelemetry import trace

from trellis.harness.redaction import redact_attributes
from trellis.harness.settings import Settings

log = logging.getLogger("trellis.telemetry")

TRACER_NAME: Final = "trellis"
#: Langfuse's OTLP traces endpoint, relative to its host.
LANGFUSE_OTLP_PATH: Final = "/api/public/otel/v1/traces"
LANGFUSE_DEFAULT_HOST: Final = "https://cloud.langfuse.com"

_tracer = trace.get_tracer(TRACER_NAME)
_meter = otel_metrics.get_meter(TRACER_NAME)
_runs = _meter.create_counter("trellis.runs", description="runs finished, by outcome")
_tools = _meter.create_counter("trellis.tool_calls", description="tool calls, by status")
_writes = _meter.create_counter(
    "trellis.writes.failed", description="background writes that failed"
)
_judged = _meter.create_histogram("trellis.judge.score", description="judge scores")


class _Metrics:
    """The counters, behind names that say what happened."""

    @staticmethod
    def run_finished(agent_id: str, outcome: str) -> None:
        _runs.add(1, {"agent": agent_id, "outcome": outcome})

    @staticmethod
    def tool_called(tool: str, status: str) -> None:
        _tools.add(1, {"tool": tool, "status": status})

    @staticmethod
    def write_failed(label: str) -> None:
        _writes.add(1, {"write": label})

    @staticmethod
    def judged(agent_id: str, score: float, method: str) -> None:
        _judged.record(score, {"agent": agent_id, "method": method})


metrics = _Metrics()


@contextmanager
def span(name: str, attributes: Mapping[str, Any]) -> Iterator[trace.Span]:
    """A span with redacted attributes; exceptions are recorded and re-raised."""
    with _tracer.start_as_current_span(name, attributes=redact_attributes(attributes)) as current:
        yield current


def configure(settings: Settings) -> bool:
    """Install OTLP exporters when the deployment asked for them. Returns whether it did."""
    exporters = _exporters(settings)
    if not exporters:
        return False
    if not isinstance(trace.get_tracer_provider(), trace.ProxyTracerProvider):
        log.info("an OpenTelemetry tracer provider is already installed; keeping it")
        return False
    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (  # noqa: PLC0415
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource  # noqa: PLC0415
        from opentelemetry.sdk.trace import TracerProvider  # noqa: PLC0415
        from opentelemetry.sdk.trace.export import BatchSpanProcessor  # noqa: PLC0415
    except ImportError:
        log.warning(
            "OTLP export is configured but not installed: pip install 'trellis-harness[otel]'"
        )
        return False
    provider = TracerProvider(resource=Resource.create({"service.name": "trellis-harness"}))
    for endpoint, headers in exporters:
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, headers=headers))
        )
    trace.set_tracer_provider(provider)
    return True


def _exporters(settings: Settings) -> list[tuple[str | None, dict[str, str]]]:
    """(endpoint, headers) per exporter. ``None`` lets the OTLP exporter read its own env."""
    found: list[tuple[str | None, dict[str, str]]] = []
    if settings.otlp_endpoint:
        found.append((None, {}))
    if settings.langfuse_public_key and settings.langfuse_secret_key:
        pair = f"{settings.langfuse_public_key}:{settings.langfuse_secret_key}".encode()
        host = (settings.langfuse_host or LANGFUSE_DEFAULT_HOST).rstrip("/")
        found.append(
            (
                host + LANGFUSE_OTLP_PATH,
                {"Authorization": "Basic " + base64.b64encode(pair).decode()},
            )
        )
    return found
