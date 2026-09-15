"""Plain Python is first class (§57, §81): every calling convention works unchanged."""

from __future__ import annotations

import asyncio

import pytest

from universal_agent_harness import (
    AgentExecutionContext,
    AgentHarness,
    AgentResult,
    AgentRuntime,
    current_context,
)


async def test_wrap_async_callable(harness, context):
    async def agent(payload):
        return {"seen": payload}

    wrapped = harness.wrap(agent, agent_id="echo")
    result = await wrapped("hello", context=context)
    assert isinstance(result, AgentResult)
    assert result.data == {"seen": "hello"}
    assert result.status == "SUCCESS"


def test_wrap_sync_callable_stays_sync(harness, context):
    def agent(payload):
        return payload.upper()

    wrapped = harness.wrap(agent, agent_id="upper")
    assert not asyncio.iscoroutinefunction(wrapped)
    assert wrapped("hi", context=context).data == "HI"


async def test_wrap_callable_object(harness, context):
    class Agent:
        def __init__(self) -> None:
            self.calls = 0

        async def __call__(self, payload):
            self.calls += 1
            return payload

    agent = Agent()
    wrapped = harness.wrap(agent, agent_id="obj")
    await wrapped("x", context=context)
    assert agent.calls == 1


async def test_zero_argument_agents_are_supported(harness, context):
    async def agent():
        return "no input needed"

    assert (await harness.wrap(agent, agent_id="noargs")(context=context)).data == "no input needed"


async def test_decorator_injects_the_runtime(harness, context):
    seen: dict[str, object] = {}

    @harness.agent(agent_id="inventory-agent", skills=["inventory.analysis"])
    async def inventory_agent(state, agent: AgentRuntime):
        seen["runtime"] = agent
        seen["agent_id"] = agent.agent_id
        return AgentResult.ok({"stock": state["sku"]})

    result = await inventory_agent({"sku": "SKU-1"}, context=context)
    assert result.data == {"stock": "SKU-1"}
    assert isinstance(seen["runtime"], AgentRuntime)
    assert seen["agent_id"] == "inventory-agent"
    assert inventory_agent.descriptor.skill_ids == ["inventory.analysis"]


async def test_state_mapper_owns_the_output_shape(harness, context):
    async def agent(payload):
        return {"total": 7}

    wrapped = harness.wrap(
        agent, agent_id="inv", state_mapper=lambda result: {"inventory_result": result.data}
    )
    assert await wrapped(None, context=context) == {"inventory_result": {"total": 7}}


async def test_execution_context_manager(harness, context):
    async with harness.execution(context, agent_id="block-agent", input="question?") as runtime:
        assert runtime.agent_id == "block-agent"
        assert current_context() is runtime.context
        runtime.state["result"] = AgentResult.ok("done")
    assert current_context() is None


async def test_context_is_bound_during_execution_and_cleared_after(harness, context):
    async def agent(_):
        ctx = current_context()
        assert ctx is not None and ctx.agent_id == "bound"
        return "ok"

    await harness.wrap(agent, agent_id="bound")(None, context=context)
    assert current_context() is None


async def test_nested_agents_inherit_identity_and_record_lineage(harness, context):
    inner_context: dict[str, AgentExecutionContext] = {}

    async def child(payload):
        inner_context["ctx"] = current_context()
        return "child done"

    wrapped_child = harness.wrap(child, agent_id="child-agent")

    async def parent(payload):
        await wrapped_child("x")  # no context passed: it is inherited
        return "parent done"

    await harness.wrap(parent, agent_id="parent-agent")(None, context=context)

    child_ctx = inner_context["ctx"]
    assert child_ctx.agent_id == "child-agent"
    assert child_ctx.tenant_id == context.tenant_id
    assert child_ctx.thread_id == context.thread_id
    assert child_ctx.trace_id == context.trace_id
    assert child_ctx.parent_agent_run_id is not None
    assert child_ctx.parent_agent_run_id != child_ctx.agent_run_id


async def test_harness_run_without_keeping_a_wrapper(harness, context):
    async def agent(payload):
        return payload * 2

    result = await harness.run(agent, 21, agent_id="doubler", context=context)
    assert result.data == 42


def test_run_sync_outside_a_loop(memory):
    harness = AgentHarness(memory=memory, defaults={"tenant_id": "acme"})

    async def agent(payload):
        await asyncio.sleep(0)
        return "done"

    assert harness.run_sync(agent, None, agent_id="sync").data == "done"


async def test_run_sync_inside_a_running_loop_does_not_nest_asyncio_run(harness, context):
    def sync_agent(payload):
        return "from sync"

    wrapped = harness.wrap(sync_agent, agent_id="sync-in-loop")
    # Calling the sync wrapper from inside a loop must not raise "asyncio.run() cannot be
    # called from a running event loop" (§72).
    result = await asyncio.to_thread(wrapped, None, context=context)
    assert result.data == "from sync"


async def test_defaults_supply_tenant_when_no_context_is_passed(harness):
    async def agent(_):
        return current_context().tenant_id

    assert (await harness.wrap(agent, agent_id="defaults")(None)).data == "acme"


async def test_missing_tenant_is_a_clear_error(memory):
    harness = AgentHarness(memory=memory)

    async def agent(_):
        return None

    with pytest.raises(ValueError, match="tenant_id"):
        await harness.wrap(agent, agent_id="no-tenant")(None)


async def test_callable_object_is_not_mistaken_for_a_runtime_aware_agent(harness, context):
    """``__call__``'s ``self`` must not be counted as a positional argument, or a plain
    callable object would be handed the runtime as its payload."""

    class Agent:
        async def __call__(self, payload):
            return {"got": payload}

    assert (await harness.wrap(Agent(), agent_id="obj")("x", context=context)).data == {"got": "x"}


async def test_callable_object_can_be_runtime_aware(harness, context):
    class Agent:
        async def __call__(self, payload, runtime):
            return runtime.agent_id

    assert (await harness.wrap(Agent(), agent_id="obj2")("x", context=context)).data == "obj2"


async def test_execution_block_emits_lifecycle_events_like_a_wrapped_agent(harness, context):
    seen: list[str] = []
    harness.on(lambda event, payload: seen.append(event))

    async with harness.execution(context, agent_id="block-agent", input="q") as runtime:
        runtime.state["result"] = AgentResult.ok("done")

    assert seen[0] == "on_agent_start"
    assert "on_agent_success" in seen
    assert seen[-1] == "on_agent_finish"


async def test_execution_block_reports_failures(harness, context):
    seen: list[str] = []
    harness.on(lambda event, payload: seen.append(event))

    with pytest.raises(RuntimeError, match="inside the block"):
        async with harness.execution(context, agent_id="block-agent"):
            raise RuntimeError("inside the block")

    assert "on_agent_error" in seen and seen[-1] == "on_agent_finish"


async def test_wrap_model_adapts_an_existing_client(harness, context):
    class Provider:
        async def ainvoke(self, prompt, **kwargs):
            return {"text": f"provider says {prompt}", "usage": {"input_tokens": 3}}

    harness.model_client = harness.wrap_model(Provider(), provider="acme-ai")
    harness.runtime_builder.model_client = harness.model_client

    async def agent(payload, runtime):
        response = await runtime.model.invoke("hello")
        assert response.usage.input_tokens == 3
        return response.text

    result = await harness.wrap(agent, agent_id="inv")(None, context=context)
    assert result.data == "provider says hello"


async def test_register_tool_after_construction(harness, context):
    async def late_tool(x: int) -> int:
        return x * 2

    spec = harness.register_tool(late_tool)
    assert spec.name == "late_tool"

    async def agent(payload, runtime):
        return (await runtime.tools.call("late_tool", x=21)).output

    assert (await harness.wrap(agent, agent_id="inv")(None, context=context)).data == 42
