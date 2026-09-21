"""Context propagation (§33), composite telemetry (§32), model/policy providers."""

from __future__ import annotations

import asyncio

import pytest
from universal_agent_contracts.errors import ConfigurationError, ModelError

from universal_agent_harness import ModelRequest
from universal_agent_harness.config.settings import TelemetryConfig
from universal_agent_harness.models.providers import DirectModelClient, UnconfiguredModelClient
from universal_agent_harness.policy.providers import (
    AllowListPolicyProvider,
    CallablePolicyProvider,
)
from universal_agent_harness.runtime.propagation import (
    baggage_fields,
    bind,
    current_context,
    current_runtime,
    extract_trace_id,
    require_runtime,
    trace_headers,
)
from universal_agent_harness.telemetry.composite import CompositeTelemetryProvider
from universal_agent_harness.telemetry.noop import NoOpTelemetryProvider
from universal_agent_harness.telemetry.otel import OpenTelemetryTelemetryProvider, configure_sdk

# --------------------------------------------------------------------------- propagation


def test_bind_sets_and_restores(context):
    assert current_context() is None
    with bind(context):
        assert current_context() is context
    assert current_context() is None


def test_bind_is_reentrant(context):
    child = context.for_agent("child")
    with bind(context):
        with bind(child):
            assert current_context() is child
        assert current_context() is context


async def test_child_tasks_inherit_the_context_but_cannot_leak_back(context):
    """``create_task`` copies the current context (that is how a nested agent inherits
    lineage); what must never happen is a binding inside a task escaping to its parent."""
    child = context.for_agent("child")

    async def inside() -> tuple[object, object]:
        inherited = current_context()
        with bind(child):
            rebound = current_context()
        return inherited, rebound

    with bind(context):
        inherited, rebound = await asyncio.create_task(inside())
        assert inherited is context
        assert rebound is child
        assert current_context() is context  # the parent is untouched


def test_require_runtime_explains_itself():
    assert current_runtime() is None
    with pytest.raises(RuntimeError, match="no AgentRuntime is bound"):
        require_runtime()


def test_trace_headers_carry_correlation_ids(context):
    headers = trace_headers(context)
    assert headers["x-request-id"] == context.request_id
    assert headers["x-correlation-id"] == context.correlation_id


def test_baggage_carries_ids_only(context):
    fields = baggage_fields(context)
    assert set(fields) <= {"tenant_id", "agent_id", "agent_run_id", "correlation_id"}
    assert "user_id" not in fields  # no principal, no content, no secrets (§33)


def test_extract_trace_id_returns_none_without_a_span():
    assert extract_trace_id() in (None, "00000000000000000000000000000000")


# --------------------------------------------------------------------------- telemetry


def test_composite_isolates_a_failing_provider():
    class Broken:
        def start_span(self, name, *, kind="internal", attributes=None):
            raise RuntimeError("provider is down")

        def record_event(self, *a, **k):
            raise RuntimeError("down")

        def record_metric(self, *a, **k):
            raise RuntimeError("down")

        def flush(self, timeout_seconds=5.0):
            raise RuntimeError("down")

    composite = CompositeTelemetryProvider([Broken(), NoOpTelemetryProvider()])
    with composite.start_span("agent.run") as span:
        span.set_attribute("k", "v")  # the healthy provider still works
    composite.record_metric("m", 1.0)
    composite.flush()


def test_composite_strict_mode_propagates():
    class Broken:
        def start_span(self, name, *, kind="internal", attributes=None):
            raise RuntimeError("provider is down")

    composite = CompositeTelemetryProvider([Broken()], strict=True)
    with pytest.raises(RuntimeError), composite.start_span("agent.run"):
        pass


def test_empty_composite_yields_a_noop_span():
    with CompositeTelemetryProvider([]).start_span("agent.run") as span:
        span.set_attributes({"a": 1})


def test_configure_sdk_never_replaces_an_application_provider():
    # the test session already registered a provider
    assert configure_sdk(TelemetryConfig(configure_sdk=True, exporter="console")) is False


def test_metrics_can_be_switched_off():
    provider = OpenTelemetryTelemetryProvider(TelemetryConfig(metrics_enabled=False))
    provider.record_metric("agent.executions.count", 1)  # must be a no-op, not an error


def test_otel_provider_exposes_the_current_trace_id():
    provider = OpenTelemetryTelemetryProvider()
    with provider.start_span("probe"):
        assert provider.current_trace_id()


# --------------------------------------------------------------------------- model providers


async def test_direct_client_accepts_a_plain_async_callable():
    client = DirectModelClient(lambda prompt: f"echo {prompt}", model="m1")
    response = await client.invoke("hi")
    assert response.text == "echo hi"
    assert response.model == "m1"


async def test_direct_client_prefers_ainvoke_on_an_object():
    class Provider:
        async def ainvoke(self, prompt, **kwargs):
            return {"text": "from ainvoke"}

    assert (await DirectModelClient(Provider()).invoke("q")).text == "from ainvoke"


async def test_direct_client_passes_messages_when_present():
    seen = {}

    async def call(payload, **kwargs):
        seen["payload"] = payload
        return "ok"

    await DirectModelClient(call).invoke(ModelRequest(messages=[{"role": "user", "content": "hi"}]))
    assert seen["payload"] == [{"role": "user", "content": "hi"}]


async def test_direct_client_streams_from_a_sync_iterable():
    class Provider:
        async def ainvoke(self, prompt, **kwargs):
            return "x"

        def stream(self, prompt, **kwargs):
            return iter(["a", "b"])

    chunks = [c async for c in DirectModelClient(Provider()).stream("q")]
    assert chunks == ["a", "b"]


async def test_direct_client_without_streaming_says_so():
    with pytest.raises(ModelError, match="does not stream"):
        DirectModelClient(lambda p: "x").stream("q")


def test_unusable_model_target_is_rejected_at_construction():
    with pytest.raises(ConfigurationError, match="not usable as a model client"):
        DirectModelClient(object())


async def test_unconfigured_client_explains_the_fix():
    with pytest.raises(ConfigurationError, match="no model client is configured"):
        await UnconfiguredModelClient().invoke("q")


# --------------------------------------------------------------------------- policy


async def test_allow_list_checks_each_dimension(context):
    from universal_agent_harness import AgentRequest, ToolCall

    policy = AllowListPolicyProvider(
        agents={"inv"}, tools={"search"}, models={"m1"}, tenants={"acme"}
    )
    allowed = AgentRequest.create(context.with_fields(agent_id="inv"))
    assert await policy.authorize_execution(allowed) is True
    denied = AgentRequest.create(context.with_fields(agent_id="other"))
    assert isinstance(await policy.authorize_execution(denied), str)
    assert await policy.authorize_tool(context, ToolCall(tool="search")) is True
    assert isinstance(await policy.authorize_tool(context, ToolCall(tool="rm")), str)
    assert await policy.authorize_model(context, ModelRequest(model="m1")) is True
    assert isinstance(await policy.authorize_model(context, ModelRequest(model="m2")), str)


async def test_allow_list_without_restrictions_allows(context):
    from universal_agent_harness import AgentRequest, ToolCall

    policy = AllowListPolicyProvider()
    assert await policy.authorize_execution(AgentRequest.create(context)) is True
    assert await policy.authorize_tool(context, ToolCall(tool="anything")) is True


async def test_callable_policy_adapts_sync_and_async_functions(context):
    from universal_agent_harness import AgentRequest, ToolCall

    async def deny_tool(ctx, call):
        return f"{call.tool} is not allowed here"

    policy = CallablePolicyProvider(execution=lambda request: True, tool=deny_tool)
    assert await policy.authorize_execution(AgentRequest.create(context)) is True
    assert await policy.authorize_tool(context, ToolCall(tool="rm")) == "rm is not allowed here"
    assert await policy.authorize_model(context, ModelRequest()) is True  # unset -> allow
