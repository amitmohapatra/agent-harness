"""Spans carry the GenAI semantic conventions and the trace attributes Langfuse maps; every
attempt of a run shares one trace; scores reach Langfuse with the OTLP credentials."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Iterator

import httpx
import pytest
import respx
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from trellis import Settings
from trellis.harness import telemetry
from trellis.harness.redaction import REDACTED
from trellis.harness.telemetry import RunTrace, Scores


@pytest.fixture
def spans(monkeypatch: pytest.MonkeyPatch) -> Iterator[InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(telemetry, "_tracer", provider.get_tracer("t"))
    yield exporter


RUN = RunTrace(
    run_id="run_1", agent_id="support", tenant="acme", user="u1", thread="th", framework="react"
)


def test_the_agent_span_is_an_invoke_agent_span_with_the_trace_attributes(spans) -> None:
    with telemetry.agent_span(RUN, "where is my order?") as span:
        telemetry.output(span, "on its way")
    [found] = spans.get_finished_spans()
    attributes = dict(found.attributes or {})
    assert found.name == "invoke_agent support"
    assert attributes["gen_ai.operation.name"] == "invoke_agent"
    assert attributes["gen_ai.agent.name"] == "support"
    assert attributes["gen_ai.conversation.id"] == "th"
    assert attributes["langfuse.trace.name"] == "support"
    assert attributes["user.id"] == "u1"
    assert attributes["session.id"] == "th"
    assert attributes["langfuse.trace.tags"] == ("support", "react")
    assert attributes["langfuse.trace.metadata.tenant"] == "acme"
    assert attributes["trellis.run_id"] == "run_1"
    assert attributes["langfuse.observation.type"] == "agent"
    assert attributes["langfuse.observation.input"] == "where is my order?"
    assert attributes["langfuse.observation.output"] == "on its way"
    assert found.context is not None
    assert found.context.trace_id == telemetry.trace_id_of("run_1")


def test_every_attempt_and_score_of_a_run_share_its_trace(spans) -> None:
    for attempt in (1, 2):
        with telemetry.agent_span(dataclasses.replace(RUN, attempt=attempt), "q"):
            pass
    telemetry.score_span("run_1", "grounding", 0.9, None)
    traces = {s.context.trace_id for s in spans.get_finished_spans() if s.context}
    assert traces == {telemetry.trace_id_of("run_1")}
    assert telemetry.trace_hex("run_1") == format(telemetry.trace_id_of("run_1"), "032x")
    score = spans.get_finished_spans()[-1]
    assert score.name == "score grounding"
    assert (score.attributes or {})["trellis.score.value"] == 0.9


def test_tool_and_model_spans_follow_the_genai_conventions(spans) -> None:
    with telemetry.tool_span(
        "erp-get_stock",
        "call_1",
        {"sku": "a", "api_key": "sk-abcdefghijklmnop"},
        source="mcp",
        tier="auto",
    ) as span:
        telemetry.output(span, {"units": 3}, key="gen_ai.tool.call.result")
    with telemetry.model_span("openai/gpt-4.1-nano", [{"role": "user", "content": "hi"}]) as span:
        telemetry.usage(
            span,
            {
                "model": "gpt-4.1-nano",
                "usage": {"prompt_tokens": 12, "completion_tokens": 3},
                "choices": [{"finish_reason": "stop"}],
            },
        )
    tool_span, model_span = spans.get_finished_spans()
    tool = dict(tool_span.attributes or {})
    assert tool_span.name == "execute_tool erp-get_stock"
    assert tool["gen_ai.operation.name"] == "execute_tool"
    assert tool["gen_ai.tool.name"] == "erp-get_stock"
    assert tool["gen_ai.tool.call.id"] == "call_1"
    assert tool["gen_ai.tool.type"] == "extension"
    assert "sk-abcdefghijklmnop" not in tool["gen_ai.tool.call.arguments"]
    assert json.loads(tool["gen_ai.tool.call.result"].replace("'", '"')) == {"units": 3}
    model = dict(model_span.attributes or {})
    assert model_span.name == "chat openai/gpt-4.1-nano"
    assert model["gen_ai.request.model"] == "openai/gpt-4.1-nano"
    assert model["gen_ai.provider.name"] == "openai"
    assert model["gen_ai.usage.input_tokens"] == 12
    assert model["gen_ai.usage.output_tokens"] == 3
    assert model["gen_ai.response.finish_reasons"] == ("stop",)
    assert model["langfuse.observation.type"] == "generation"


def test_sensitive_attributes_are_redacted(spans) -> None:
    with telemetry.retrieval_span("q") as span:
        span.set_attributes(telemetry.redact_attributes({"api_key": "secret"}))
    [found] = spans.get_finished_spans()
    assert (found.attributes or {})["api_key"] == REDACTED


def test_nothing_configured_installs_nothing() -> None:
    assert telemetry.configure(Settings()) is False
    assert telemetry.traces_url("http://c:4318") == "http://c:4318/v1/traces"
    assert telemetry.traces_url("http://lf/api/public/otel/v1/traces/") == (
        "http://lf/api/public/otel/v1/traces"
    )


@pytest.mark.parametrize(
    ("endpoint", "headers", "host"),
    [
        ("https://lf.example/api/public/otel", {"authorization": "Basic x"}, "https://lf.example"),
        (
            "http://collector:4318",
            {"authorization": "Basic x", "x-langfuse-host": "https://lf.example"},
            "https://lf.example",
        ),
        ("http://collector:4318", {"authorization": "Basic x"}, None),
        ("https://lf.example/api/public/otel", {}, None),
        ("https://lf.example/api/public/otel", {"authorization": "Bearer t"}, None),
    ],
    ids=["langfuse-direct", "collector-named-host", "collector-only", "no-auth", "not-basic"],
)
def test_the_scores_api_is_reached_only_with_langfuse_credentials(endpoint, headers, host) -> None:
    scores = Scores.of(Settings(otlp_endpoint=endpoint, otlp_headers=headers))
    assert (scores.host if scores is not None else None) == host


@respx.mock
async def test_a_score_is_posted_on_the_runs_trace_idempotently() -> None:
    route = respx.post("https://lf.example/api/public/scores").mock(
        return_value=httpx.Response(200, json={"id": "k"})
    )
    scores = Scores("https://lf.example", "Basic eHg6eXk=")
    await scores.post("run_1", "grounding", 0.75, data_type="NUMERIC", key="run_1:grounding")
    request = route.calls[0].request
    assert request.headers["authorization"] == "Basic eHg6eXk="
    assert json.loads(request.content) == {
        "id": "run_1:grounding",
        "traceId": telemetry.trace_hex("run_1"),
        "name": "grounding",
        "value": 0.75,
        "dataType": "NUMERIC",
    }
    await scores.aclose()


@pytest.fixture
def installed(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """The global tracer provider, as configure() sees and sets it (never really set: a
    process gets one, and an exporter must not try the network)."""
    from opentelemetry import trace

    providers: list[object] = []
    monkeypatch.setattr(trace, "get_tracer_provider", trace.ProxyTracerProvider)
    monkeypatch.setattr(trace, "set_tracer_provider", providers.append)
    return providers


def test_an_otlp_endpoint_installs_one_exporter_to_its_traces_url(installed: list[object]) -> None:
    settings = Settings(
        otlp_endpoint="http://collector:4318/", otlp_headers={"authorization": "Basic x"}
    )
    assert telemetry.configure(settings) is True
    [provider] = installed
    assert isinstance(provider, TracerProvider)
    [processor] = provider._active_span_processor._span_processors  # type: ignore[attr-defined]
    exporter = processor.span_exporter  # type: ignore[attr-defined]
    assert exporter._endpoint == "http://collector:4318/v1/traces"
    assert exporter._headers == {"authorization": "Basic x"}
    assert provider.resource.attributes["service.name"] == "trellis-harness"
    provider.shutdown()


def test_an_application_provider_is_kept(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from opentelemetry import trace

    own = TracerProvider()
    monkeypatch.setattr(trace, "get_tracer_provider", lambda: own)
    with caplog.at_level("INFO", logger="trellis.telemetry"):
        assert telemetry.configure(Settings(otlp_endpoint="http://c:4318")) is False
    assert "already installed; keeping it" in caplog.text


def test_otlp_without_the_extra_installed_says_how_to_get_it(
    installed: list[object], monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "opentelemetry.exporter.otlp.proto.http.trace_exporter", None)
    with caplog.at_level("WARNING", logger="trellis.telemetry"):
        assert telemetry.configure(Settings(otlp_endpoint="http://c:4318")) is False
    assert "pip install 'trellis-harness[otel]'" in caplog.text
    assert installed == []


def test_counters_count_what_happened(monkeypatch: pytest.MonkeyPatch) -> None:
    added: list[tuple[str, int, dict[str, str]]] = []

    class Counter:
        def __init__(self, name: str) -> None:
            self.name = name

        def add(self, amount: int, attributes: dict[str, str]) -> None:
            added.append((self.name, amount, attributes))

    for name in ("_runs", "_tools", "_writes"):
        monkeypatch.setattr(telemetry, name, Counter(name))
    telemetry.metrics.run_finished("support", "success")
    telemetry.metrics.tool_called("refund", "ok")
    telemetry.metrics.write_failed("memory.transcript")
    assert added == [
        ("_runs", 1, {"agent": "support", "outcome": "success"}),
        ("_tools", 1, {"tool": "refund", "status": "ok"}),
        ("_writes", 1, {"write": "memory.transcript"}),
    ]


@respx.mock
async def test_a_score_with_a_comment_carries_it() -> None:
    route = respx.post("https://lf.example/api/public/scores").mock(
        return_value=httpx.Response(200, json={"id": "k"})
    )
    scores = Scores("https://lf.example/", "Basic eHg6eXk=")
    await scores.post(
        "run_1", "feedback", 0.0, data_type="NUMERIC", key="run_1:feedback", comment="13"
    )
    assert json.loads(route.calls[0].request.content)["comment"] == "13"
    assert scores.host == "https://lf.example"
    await scores.aclose()
