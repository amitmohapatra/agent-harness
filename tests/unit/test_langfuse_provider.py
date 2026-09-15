"""Langfuse provider internals (§22, §23, §24, §32) — no network, no live project."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from universal_agent_harness.config.settings import LangfuseConfig
from universal_agent_harness.langfuse import attributes as LA
from universal_agent_harness.langfuse.provider import (
    LangfuseSpanEnricher,
    LangfuseTelemetryProvider,
    _create_client,
    otlp_endpoint,
    otlp_headers,
)


class RecordingSpan:
    def __init__(self) -> None:
        self.attributes: dict[str, object] = {}
        self.status: str | None = None

    def set_attribute(self, key, value):
        self.attributes[key] = value

    def set_attributes(self, attributes):
        self.attributes.update(attributes)

    def add_event(self, name, attributes=None):
        pass

    def record_error(self, error, **attributes):
        self.status = "error"

    def set_status_ok(self):
        self.status = "ok"


def test_observation_types_follow_the_documented_mapping():
    assert LA.observation_type("agent.run", "agent") == "agent"
    assert LA.observation_type("agent.model.invoke", "generation") == "generation"
    assert LA.observation_type("agent.tool.call", "tool") == "tool"
    assert LA.observation_type("agent.memory.retrieve", "retriever") == "retriever"
    assert LA.observation_type("something.else", "internal") == "span"


def test_enricher_maps_harness_attributes_to_langfuse_ones():
    span = RecordingSpan()
    enricher = LangfuseSpanEnricher(span, "generation")
    enricher.set_attributes(
        {
            "input.value": "prompt text",
            "output.value": "answer",
            "gen_ai.request.model": "gpt-x",
            "gen_ai.usage.input_tokens": 10,
            "gen_ai.usage.output_tokens": 4,
            "gen_ai.usage.total_tokens": 14,
            "gen_ai.usage.cost": 0.002,
        }
    )
    assert span.attributes[LA.OBSERVATION_TYPE] == "generation"
    assert span.attributes[LA.OBSERVATION_INPUT] == "prompt text"
    assert span.attributes[LA.OBSERVATION_OUTPUT] == "answer"
    assert span.attributes[LA.OBSERVATION_MODEL] == "gpt-x"
    assert json.loads(span.attributes[LA.OBSERVATION_USAGE_DETAILS]) == {
        "input": 10, "output": 4, "total": 14
    }
    assert json.loads(span.attributes[LA.OBSERVATION_COST_DETAILS]) == {"total": 0.002}


def test_enricher_marks_errors_at_langfuse_level():
    span = RecordingSpan()
    LangfuseSpanEnricher(span, "span").record_error(RuntimeError("bad"))
    assert span.attributes[LA.OBSERVATION_LEVEL] == "ERROR"
    assert "RuntimeError" in span.attributes[LA.OBSERVATION_STATUS_MESSAGE]


def test_disabled_provider_creates_no_client():
    provider = LangfuseTelemetryProvider(LangfuseConfig())
    assert provider.client is None and provider.mode == "disabled"


def test_otlp_mode_needs_no_sdk():
    provider = LangfuseTelemetryProvider(
        LangfuseConfig(enabled=True, mode="otlp", public_key="pk", secret_key="sk")
    )
    assert provider.mode == "otlp"
    assert provider.client is None


def test_otlp_endpoint_and_headers_follow_langfuse_s_otel_contract():
    config = LangfuseConfig(enabled=True, mode="otlp", public_key="pk", secret_key="sk",
                            base_url="https://lf.example.com/")
    assert otlp_endpoint(config) == "https://lf.example.com/api/public/otel/v1/traces"
    assert otlp_headers(config)["Authorization"].startswith("Basic ")


def test_sdk_client_is_constructed_with_documented_public_arguments(monkeypatch):
    """The harness must configure Langfuse through its public constructor only — including
    the hook that also exports the harness's own spans (§22)."""
    captured: dict[str, object] = {}

    class FakeLangfuse:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    import langfuse

    monkeypatch.setattr(langfuse, "Langfuse", FakeLangfuse)
    config = LangfuseConfig(
        enabled=True, mode="sdk", public_key="pk", secret_key="sk",
        base_url="https://lf.example.com", environment="staging", release="1.2.3",
    )
    client = _create_client(config, tracer_provider=None, sample_rate=0.5)

    assert isinstance(client, FakeLangfuse)
    assert captured["public_key"] == "pk"
    assert captured["host"] == "https://lf.example.com"
    assert captured["environment"] == "staging"
    assert captured["release"] == "1.2.3"
    assert captured["sample_rate"] == 0.5          # sampling comes from telemetry config
    assert callable(captured["should_export_span"])


def test_harness_spans_are_included_in_what_langfuse_exports(monkeypatch):
    captured: dict[str, object] = {}

    class FakeLangfuse:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    import langfuse

    monkeypatch.setattr(langfuse, "Langfuse", FakeLangfuse)
    _create_client(LangfuseConfig(enabled=True, mode="sdk", public_key="pk", secret_key="sk"), None)
    should_export = captured["should_export_span"]

    harness_span = SimpleNamespace(
        instrumentation_scope=SimpleNamespace(name="universal_agent_harness"), attributes={}
    )
    unrelated_span = SimpleNamespace(
        instrumentation_scope=SimpleNamespace(name="some.other.library"), attributes={}
    )
    assert should_export(harness_span) is True
    assert should_export(unrelated_span) is False


def test_sdk_creation_failure_degrades_instead_of_raising(monkeypatch):
    class Exploding:
        def __init__(self, **kwargs):
            raise RuntimeError("bad credentials")

    import langfuse

    monkeypatch.setattr(langfuse, "Langfuse", Exploding)
    provider = LangfuseTelemetryProvider(
        LangfuseConfig(enabled=True, mode="auto", public_key="pk", secret_key="sk")
    )
    assert provider.mode == "otlp"  # falls back rather than failing the process (§32)


def test_flush_failure_is_swallowed_in_non_blocking_mode():
    provider = LangfuseTelemetryProvider(LangfuseConfig())
    provider.client = SimpleNamespace(flush=lambda: (_ for _ in ()).throw(RuntimeError("down")))
    provider.flush()  # must not raise


def test_flush_failure_propagates_in_fail_closed_mode():
    provider = LangfuseTelemetryProvider(LangfuseConfig())
    provider.strict = True
    provider.client = SimpleNamespace(flush=lambda: (_ for _ in ()).throw(RuntimeError("down")))
    with pytest.raises(RuntimeError):
        provider.flush()


async def test_evaluation_provider_creates_scores_off_the_event_loop():
    from universal_agent_harness.langfuse.evaluation import LangfuseEvaluationProvider

    calls: list[dict] = []

    class FakeClient:
        def create_score(self, **kwargs):
            calls.append(kwargs)

    provider = LangfuseEvaluationProvider(FakeClient())
    await provider.score("groundedness", 0.93, trace_id="t1", comment="deepeval")
    assert calls[0]["name"] == "groundedness"
    assert calls[0]["value"] == 0.93
    assert calls[0]["trace_id"] == "t1"
    assert calls[0]["data_type"] == "NUMERIC"


async def test_evaluation_failures_never_reach_the_caller():
    from universal_agent_harness.langfuse.evaluation import LangfuseEvaluationProvider

    class FailingClient:
        def create_score(self, **kwargs):
            raise RuntimeError("langfuse down")

    await LangfuseEvaluationProvider(FailingClient()).score("x", 1.0)  # must not raise


async def test_prompt_provider_compiles_variables():
    from universal_agent_harness.langfuse.evaluation import LangfusePromptProvider

    class FakePrompt:
        def compile(self, **vars):
            return f"hello {vars['name']}"

    class FakeClient:
        def get_prompt(self, name, **kwargs):
            assert name == "greeting"
            return FakePrompt()

    assert await LangfusePromptProvider(FakeClient()).get_prompt("greeting", name="world") == "hello world"
