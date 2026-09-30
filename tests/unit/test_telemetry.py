from __future__ import annotations

import base64

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from trellis import Settings
from trellis.harness import telemetry
from trellis.harness.redaction import REDACTED


def test_langfuse_is_an_otlp_exporter_with_basic_auth() -> None:
    settings = Settings(
        langfuse_public_key="pk", langfuse_secret_key="sk", langfuse_host="http://lf/"
    )
    [(endpoint, headers)] = telemetry._exporters(settings)
    assert endpoint == "http://lf/api/public/otel/v1/traces"
    assert headers == {"Authorization": "Basic " + base64.b64encode(b"pk:sk").decode()}
    assert telemetry._exporters(Settings()) == []
    assert telemetry._exporters(Settings(otlp_endpoint="http://otel")) == [(None, {})]


def test_nothing_configured_installs_nothing() -> None:
    assert telemetry.configure(Settings()) is False


def test_span_attributes_are_redacted() -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("t")
    original = telemetry._tracer
    telemetry._tracer = tracer
    try:
        with telemetry.span("trellis.tool", {"tool.name": "x", "api_key": "secret"}):
            pass
    finally:
        telemetry._tracer = original
    [span] = exporter.get_finished_spans()
    assert span.attributes is not None and span.attributes["api_key"] == REDACTED
    assert isinstance(trace.get_tracer_provider(), trace.ProxyTracerProvider | TracerProvider)
