"""Tool instrumentation (§12, §13, §14, §41)."""

from __future__ import annotations

import asyncio

import pytest
from tests.support import span_by_name, span_names
from universal_agent_contracts.errors import ToolNotFoundError

from universal_agent_harness import AgentHarness, ToolError, ToolSpec
from universal_agent_harness.tools.local import LocalToolClient


async def test_runtime_tool_client_is_instrumented(memory, context, spans):
    async def inventory_db(sku: str) -> dict:
        """Look up stock for a SKU."""
        return {"sku": sku, "on_hand": 3}

    harness = AgentHarness(memory=memory, tools=[inventory_db], defaults={"tenant_id": "acme"})

    async def agent(payload, runtime):
        outcome = await runtime.tools.call("inventory_db", sku="SKU-1")
        assert outcome.ok and outcome.output == {"sku": "SKU-1", "on_hand": 3}
        assert outcome.latency_ms is not None
        return outcome.output

    result = await harness.wrap(agent, agent_id="inv")(None, context=context)
    assert result.metrics["tool_calls"] == 1

    span = span_by_name(spans, "agent.tool.call")
    assert span.attributes["tool.name"] == "inventory_db"
    assert span.attributes["status"] == "ok"
    assert span.attributes["tool.idempotency_key"]
    assert span.attributes["tool.args.schema"] == ("sku",)


async def test_tool_arguments_are_not_captured_by_default(memory, context, spans):
    async def secret_tool(password: str) -> str:
        return "done"

    harness = AgentHarness(memory=memory, tools=[secret_tool], defaults={"tenant_id": "acme"})

    async def agent(payload, runtime):
        return (await runtime.tools.call("secret_tool", password="hunter2")).output

    await harness.wrap(agent, agent_id="inv")(None, context=context)
    span = span_by_name(spans, "agent.tool.call")
    assert "hunter2" not in str(dict(span.attributes))
    assert "input.value" not in span.attributes


async def test_tool_inputs_are_captured_only_when_configured(memory, context, spans):
    async def echo(text: str) -> str:
        return text

    harness = AgentHarness(
        memory=memory,
        tools=[echo],
        defaults={"tenant_id": "acme"},
        config={"telemetry": {"capture": {"inputs": True, "outputs": True}}},
    )

    async def agent(payload, runtime):
        return (await runtime.tools.call("echo", text="visible")).output

    await harness.wrap(agent, agent_id="inv")(None, context=context)
    span = span_by_name(spans, "agent.tool.call")
    assert "visible" in span.attributes["input.value"]
    assert "visible" in span.attributes["output.value"]


async def test_tool_idempotency_key_is_stable_across_replays(memory, context):
    keys: list[str] = []

    async def note(text: str) -> str:
        return text

    harness = AgentHarness(memory=memory, tools=[note], defaults={"tenant_id": "acme"})

    async def agent(payload, runtime):
        await runtime.tools.call("note", text="x")
        keys.append(runtime.tool_calls[-1]["idempotency_key"])
        return "ok"

    wrapped = harness.wrap(agent, agent_id="inv")
    await wrapped(None, context=context)
    await wrapped(None, context=context)
    assert keys[0] == keys[1]  # same logical step -> same key (§41/§42)


async def test_tool_errors_are_normalized_and_traced(memory, context, spans):
    async def flaky() -> None:
        raise RuntimeError("upstream exploded")

    harness = AgentHarness(memory=memory, tools=[flaky], defaults={"tenant_id": "acme"})

    async def agent(payload, runtime):
        with pytest.raises(ToolError) as exc:
            await runtime.tools.call("flaky")
        assert "upstream exploded" in str(exc.value)
        return "handled"

    await harness.wrap(agent, agent_id="inv")(None, context=context)
    span = span_by_name(spans, "agent.tool.call")
    assert span.attributes["status"] == "error"


async def test_unknown_tool_reports_what_is_registered(memory, context):
    async def known() -> None:
        return None

    harness = AgentHarness(memory=memory, tools=[known], defaults={"tenant_id": "acme"})

    async def agent(payload, runtime):
        with pytest.raises(ToolNotFoundError, match="known"):
            await runtime.tools.call("unknown")
        return "ok"

    await harness.wrap(agent, agent_id="inv")(None, context=context)


async def test_no_tool_runtime_gives_an_actionable_error(harness, context):
    async def agent(payload, runtime):
        with pytest.raises(ToolNotFoundError, match="no tool runtime"):
            await runtime.tools.call("anything")
        return "ok"

    await harness.wrap(agent, agent_id="inv")(None, context=context)


async def test_tool_invocations_are_recorded_in_tool_memory(memory, context):
    async def search(q: str) -> str:
        return "result"

    harness = AgentHarness(memory=memory, tools=[search], defaults={"tenant_id": "acme"})

    async def agent(payload, runtime):
        return (await runtime.tools.call("search", q="stock")).output

    await harness.wrap(agent, agent_id="inv")(None, context=context)
    records = memory.of("tools.record")
    assert records and records[0]["tool"] == "search"
    assert records[0]["output"] is None  # tool results are not stored by default (§11)


async def test_tool_outputs_reach_memory_only_when_the_policy_allows(memory, context):
    async def search(q: str) -> str:
        return "the answer"

    harness = AgentHarness(
        memory=memory,
        tools=[search],
        defaults={"tenant_id": "acme"},
        config={"memory": {"observe_tool_results": True, "writeback": False}},
    )

    async def agent(payload, runtime):
        return (await runtime.tools.call("search", q="stock")).output

    await harness.wrap(agent, agent_id="inv")(None, context=context)
    assert memory.of("tools.record")[0]["output"] == "the answer"


async def test_wrap_tool_instruments_direct_calls(harness, context, spans):
    @harness.wrap_tool
    async def pricing_api(sku: str) -> float:
        """Current price."""
        return 9.99

    async def agent(payload, runtime):
        return await pricing_api(sku="SKU-1")  # called directly, not through runtime.tools

    result = await harness.wrap(agent, agent_id="inv")(None, context=context)
    assert result.data == 9.99
    assert "agent.tool.call" in span_names(spans)
    assert span_by_name(spans, "agent.tool.call").attributes["tool.name"] == "pricing_api"


async def test_wrapped_tool_outside_an_execution_is_a_pass_through(harness, spans):
    @harness.wrap_tool
    async def standalone(x: int) -> int:
        return x + 1

    assert await standalone(1) == 2  # still works...
    assert "agent.tool.call" not in span_names(spans)  # ...just not instrumented (§13)


def test_wrapped_sync_tool_keeps_working_outside_an_execution(harness):
    @harness.wrap_tool
    def add(a: int, b: int) -> int:
        return a + b

    assert add(1, 2) == 3
    assert add.tool_spec.name == "add"


async def test_wrapped_sync_tool_is_instrumented_inside_an_execution(harness, context, spans):
    @harness.wrap_tool
    def add(a: int, b: int) -> int:
        return a + b

    async def agent(payload, runtime):
        return await asyncio.to_thread(add, 2, 3)

    assert (await harness.wrap(agent, agent_id="inv")(None, context=context)).data == 5
    assert "agent.tool.call" in span_names(spans)


async def test_positional_tool_arguments_are_recorded_by_name(harness, context, spans):
    @harness.wrap_tool
    async def lookup(sku: str, region: str = "eu") -> str:
        return sku

    async def agent(payload, runtime):
        return await lookup("SKU-9")

    await harness.wrap(agent, agent_id="inv")(None, context=context)
    assert set(span_by_name(spans, "agent.tool.call").attributes["tool.args.schema"]) == {
        "sku",
        "region",
    }


async def test_tool_specs_are_derived_from_the_signature():
    client = LocalToolClient()
    spec = client.register(lambda sku, limit=5: None, name="stock")
    assert isinstance(spec, ToolSpec)
    assert spec.input_schema["required"] == ["sku"]
    assert set(spec.input_schema["properties"]) == {"sku", "limit"}
