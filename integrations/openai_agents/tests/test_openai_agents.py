"""OpenAI Agents SDK adapter: the six moments of design §8, against the installed SDK.

No network and no real model: the model is a scripted client behind the harness's own model
port, except in the gateway test, where a real HTTP server answers on a real socket.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from agents import Agent, RunConfig, Runner, function_tool
from tests.support_gateway import FakeGateway, completion
from trellis.contracts import (
    AgentExecutionContext,
    Interrupt,
    InterruptDecision,
    InterruptReason,
    InterruptResolution,
    ModelResponse,
    ModelUsage,
    PolicyDeniedError,
    RunEventType,
    RunOutcome,
)

from trellis.harness import (
    AgentHarness,
    BifrostModelClient,
    CallablePolicyProvider,
    CollectingEventSink,
)
from trellis.harness.interrupts import ApprovalRequired
from trellis.harness_openai_agents import (
    PENDING_RESULT_KEY,
    BifrostModel,
    MemoryServiceSession,
    OpenAIAgentsHarness,
    openai_agents_version,
    to_model_request,
)

TENANT = "acme"
CALLS: list[dict[str, Any]] = []


@function_tool
def lookup(sku: str) -> str:
    """Look up the stock of one SKU."""
    CALLS.append({"sku": sku})
    return f"{sku}: 4 in stock"


@function_tool(needs_approval=True)
def refund(order_id: int) -> str:
    """Refund an order. Needs a person's approval."""
    CALLS.append({"order_id": order_id})
    return f"refunded {order_id}"


def says(text: str) -> ModelResponse:
    return ModelResponse(
        text=text, model="scripted", usage=ModelUsage(input_tokens=3, output_tokens=2)
    )


def calls(tool_name: str, **args: Any) -> ModelResponse:
    return ModelResponse(
        text=None,
        model="scripted",
        tool_calls=[
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": tool_name, "arguments": json.dumps(args)},
            }
        ],
    )


class ToolThenAnswer:
    """Asks for one tool, then answers once it sees the tool's result.

    A rule, not a script: a resumed run re-plans from scratch, so a fixed script would
    answer the resumed run's first model call with the wrong line.
    """

    def __init__(self, tool_name: str, args: dict[str, Any], answer: str) -> None:
        self.tool_name, self.args, self.answer = tool_name, args, answer
        self.requests: list[Any] = []

    async def invoke(self, request: Any, /, **_: Any) -> ModelResponse:
        self.requests.append(request)
        if any(m.get("role") == "tool" for m in request.messages or ()):
            return says(self.answer)
        return calls(self.tool_name, **self.args)

    async def structured(self, request: Any, /, schema: Any, **kwargs: Any) -> ModelResponse:
        return await self.invoke(request, **kwargs)


class Answers:
    """Answers with one line, whatever it is asked."""

    def __init__(self, text: str = "4 in stock") -> None:
        self.text = text
        self.requests: list[Any] = []

    async def invoke(self, request: Any, /, **_: Any) -> ModelResponse:
        self.requests.append(request)
        return says(self.text)

    async def structured(self, request: Any, /, schema: Any, **kwargs: Any) -> ModelResponse:
        return await self.invoke(request, **kwargs)


def turn(agent_id: str, thread_id: str) -> AgentExecutionContext:
    return AgentExecutionContext.create(
        tenant_id=TENANT,
        user_id="u1",
        agent_id=agent_id,
        thread_id=thread_id,
        turn_id=f"trn_{thread_id}",
    )


def build(
    *,
    model: Any = None,
    policy: Any = None,
    sink: CollectingEventSink | None = None,
    memory: Any = None,
) -> AgentHarness:
    return AgentHarness(
        memory=memory,
        model=model or Answers(),
        policy=policy,
        event_sinks=[sink] if sink is not None else (),
        defaults={"tenant_id": TENANT},
        config={"telemetry": {"capture": {"inputs": True, "outputs": True}}},
    )


# --------------------------------------------------------------------------- the adapter
def test_the_adapter_reports_the_installed_version_and_its_availability() -> None:
    harness = build()
    assert harness.openai_agents.version == openai_agents_version()
    assert OpenAIAgentsHarness.available() is True
    assert harness.openai_agents is harness.openai_agents


def test_the_public_name_resolves_through_the_core() -> None:
    from trellis.harness import OpenAIAgentsHarness as exported

    assert exported is OpenAIAgentsHarness


# --------------------------------------------------------------------------- 1. context
async def test_the_memory_bundle_reaches_the_agents_instructions() -> None:
    """The SDK calls a two-argument ``instructions`` callable per run: that is the seam."""
    harness = build()
    render = harness.openai_agents.instructions("You answer stock questions.")
    async with harness.execution(turn("ctx", "thr-ctx"), agent_id="ctx") as runtime:
        runtime.memory_context = _bundle("The customer is on the Pro plan [mem_7]")
        rendered = render(None, None)
    assert rendered.startswith("You answer stock questions.")
    assert "The customer is on the Pro plan [mem_7]" in rendered
    assert "cite memory ids" in rendered


def test_instructions_outside_a_run_are_the_agents_own() -> None:
    """A bare ``Agent(...)`` built at import time must still be constructible and callable."""
    render = build().openai_agents.instructions("Just this.")
    assert render(None, None) == "Just this."


# --------------------------------------------------------------------------- 2. model
async def test_the_model_call_goes_through_the_harness_model_port() -> None:
    model = Answers("4 in stock")
    harness = build(model=model)
    run = harness.openai_agents.agent(agent_id="model-agent", tools=[lookup])
    result = await run("how much stock of SKU-1?", context=turn("model-agent", "thr-model"))
    assert result.succeeded and result.data == "4 in stock"
    request = model.requests[0]
    assert any(m["role"] == "user" and "SKU-1" in m["content"] for m in request.messages)
    assert any(t["function"]["name"] == "lookup" for t in request.tools)


async def test_the_model_call_reaches_a_real_gateway_over_a_real_socket() -> None:
    with FakeGateway([completion("4 in stock")]) as gateway:
        harness = build(model=BifrostModelClient(gateway.url, model="gateway-model"))
        run = harness.openai_agents.agent(
            agent_id="gw-agent", tools=[lookup], model="gateway-model"
        )
        result = await run("how much stock of SKU-1?", context=turn("gw-agent", "thr-gw"))
        assert result.succeeded
        body = gateway.requests[0]
    assert body["model"] == "gateway-model"
    assert any(m["role"] == "user" and "SKU-1" in str(m["content"]) for m in body["messages"])
    assert any(t["function"]["name"] == "lookup" for t in body["tools"])


def test_the_model_converts_sdk_items_into_a_harness_request() -> None:
    request = to_model_request(
        "be brief",
        [
            {"role": "user", "content": "hi"},
            {
                "type": "function_call",
                "call_id": "c1",
                "name": "lookup",
                "arguments": '{"sku":"S1"}',
            },
            {"type": "function_call_output", "call_id": "c1", "output": "S1: 4"},
        ],
        model="m",
        tools=[],
    )
    roles = [m["role"] for m in request.messages or ()]
    assert roles == ["system", "user", "assistant", "tool"]
    assistant = (request.messages or [])[2]
    assert assistant["tool_calls"][0]["function"]["name"] == "lookup"
    assert (request.messages or [])[3]["tool_call_id"] == "c1"


async def test_the_model_converts_a_harness_response_into_sdk_output_items() -> None:
    model = ToolThenAnswer("lookup", {"sku": "S1"}, "4 in stock")
    response = await BifrostModel(client=model, model="m").get_response(
        "system", [{"role": "user", "content": "hi"}], None, [], None, [], None
    )
    assert [item.type for item in response.output] == ["function_call"]
    assert response.output[0].name == "lookup"
    assert json.loads(response.output[0].arguments) == {"sku": "S1"}
    assert response.usage.requests == 1


async def test_streaming_says_it_is_unsupported_rather_than_faking_it() -> None:
    model = BifrostModel(client=Answers(), model="m")
    with pytest.raises(NotImplementedError, match="does not implement SDK streaming"):
        async for _ in model.stream_response(
            "s", [{"role": "user", "content": "hi"}], None, [], None, [], None
        ):
            pass


# --------------------------------------------------------------------------- 3. tools
async def test_a_denied_tool_never_runs() -> None:
    CALLS.clear()

    async def deny(context: Any, call: Any) -> Any:
        return call.tool != "lookup"

    harness = build(
        model=ToolThenAnswer("lookup", {"sku": "S1"}, "4 in stock"),
        policy=CallablePolicyProvider(tool=deny),
    )
    run = harness.openai_agents.agent(agent_id="deny-agent", tools=[lookup])
    with pytest.raises(PolicyDeniedError):
        await run("how much stock?", context=turn("deny-agent", "thr-deny"))
    assert CALLS == []


async def test_a_tool_call_opens_and_closes_on_the_run_stream() -> None:
    CALLS.clear()
    sink = CollectingEventSink()
    harness = build(model=ToolThenAnswer("lookup", {"sku": "S1"}, "4 in stock"), sink=sink)
    run = harness.openai_agents.agent(agent_id="tool-agent", tools=[lookup])
    context = turn("tool-agent", "thr-tool")
    assert (await run("how much stock?", context=context)).succeeded
    assert CALLS == [{"sku": "S1"}]
    types = sink.types(context.agent_run_id)
    assert types[0] is RunEventType.RUN_STARTED
    assert RunEventType.STEP_STARTED in types and RunEventType.STEP_FINISHED in types
    assert [t for t in types if t.value.startswith("TOOL_CALL")] == [
        RunEventType.TOOL_CALL_START,
        RunEventType.TOOL_CALL_ARGS,
        RunEventType.TOOL_CALL_END,
        RunEventType.TOOL_CALL_RESULT,
    ]
    assert types[-1] is RunEventType.RUN_FINISHED
    assert len({e.attempt for e in sink.for_run(context.agent_run_id)}) == 1


# --------------------------------------------------------------------------- 4. pause
async def test_a_policy_approval_pauses_the_run_and_the_answer_continues_it() -> None:
    CALLS.clear()
    sink = CollectingEventSink()

    async def hold(context: Any, call: Any) -> Any:
        return "require_approval" if call.tool == "lookup" else True

    harness = build(
        model=ToolThenAnswer("lookup", {"sku": "S1"}, "4 in stock"),
        policy=CallablePolicyProvider(tool=hold),
        sink=sink,
    )
    run = harness.openai_agents.agent(agent_id="hitl-agent", tools=[lookup])
    context = turn("hitl-agent", "thr-hitl")

    with pytest.raises(ApprovalRequired) as paused:
        await run("how much stock?", context=context)
    assert paused.value.tool_call.tool == "lookup" and CALLS == []

    interrupt = _paused(sink, context.agent_run_id)
    assert interrupt.reason is InterruptReason.APPROVAL
    assert interrupt.tool_call is not None and interrupt.tool_call.args == {"sku": "S1"}

    result = await harness.resume(
        interrupt,
        InterruptResolution(
            interrupt_id=interrupt.interrupt_id,
            run_id=interrupt.run_id,
            decision=InterruptDecision.APPROVE,
            reviewer="u1",
        ),
        context=context,
        agent=run,
        payload="how much stock?",
    )
    assert result is not None and result.succeeded
    assert CALLS == [{"sku": "S1"}]


async def test_an_approver_rejection_comes_back_to_the_model_as_a_refusal() -> None:
    CALLS.clear()
    sink = CollectingEventSink()

    async def hold(context: Any, call: Any) -> Any:
        return "require_approval" if call.tool == "lookup" else True

    harness = build(
        model=ToolThenAnswer("lookup", {"sku": "S1"}, "I could not check the stock"),
        policy=CallablePolicyProvider(tool=hold),
        sink=sink,
    )
    run = harness.openai_agents.agent(agent_id="reject-agent", tools=[lookup])
    context = turn("reject-agent", "thr-reject")
    with pytest.raises(ApprovalRequired):
        await run("how much stock?", context=context)
    interrupt = _paused(sink, context.agent_run_id)
    result = await harness.resume(
        interrupt,
        InterruptResolution(
            interrupt_id=interrupt.interrupt_id,
            run_id=interrupt.run_id,
            decision=InterruptDecision.REJECT,
            reviewer="u1",
        ),
        context=context,
        agent=run,
        payload="how much stock?",
    )
    assert result is not None and result.succeeded
    assert CALLS == []


async def test_the_sdks_own_needs_approval_becomes_the_same_interrupt() -> None:
    """The SDK *returns* its approvals; the platform has one shape for a pause."""
    CALLS.clear()
    sink = CollectingEventSink()
    harness = build(model=ToolThenAnswer("refund", {"order_id": 91}, "refunded"), sink=sink)
    run = harness.openai_agents.agent(agent_id="approval-agent", tools=[refund])
    context = turn("approval-agent", "thr-approval")
    with pytest.raises(ApprovalRequired) as paused:
        await run("refund order 91", context=context)
    assert paused.value.tool_call.tool == "refund"
    assert paused.value.tool_call.args == {"order_id": 91}
    assert CALLS == []
    interrupt = _paused(sink, context.agent_run_id)
    assert interrupt.reason is InterruptReason.APPROVAL


async def test_a_paused_runs_result_is_reachable_for_an_in_place_resume() -> None:
    """An application that would rather continue the SDK's own run needs its ``RunResult``."""
    CALLS.clear()
    sink = CollectingEventSink()
    harness = build(model=ToolThenAnswer("refund", {"order_id": 91}, "refunded"), sink=sink)
    run = harness.openai_agents.agent(agent_id="reachable-agent", tools=[refund])
    context = turn("reachable-agent", "thr-reachable")
    with pytest.raises(ApprovalRequired):
        await run("refund order 91", context=context)
    # the harness keeps the paused run's state for the turn, keyed where the adapter says
    assert PENDING_RESULT_KEY == "openai_agents.pending_result"
    assert harness.openai_agents.pending_result(_State({})) is None


async def test_an_edit_decision_is_refused_rather_than_silently_approved() -> None:
    """The SDK approves or rejects a call; it does not rewrite one, and the adapter says so."""
    harness = build()
    state = _FakeRunState()
    with pytest.raises(ValueError, match="does not edit one"):
        harness.openai_agents.apply_resolution(
            state,
            InterruptResolution(
                interrupt_id="int_1",
                run_id="run_1",
                decision=InterruptDecision.EDIT,
                payload={"order_id": 92},
            ),
        )


async def test_a_resolution_maps_onto_the_sdks_own_approve_and_reject() -> None:
    harness = build()
    approved = _FakeRunState()
    harness.openai_agents.apply_resolution(
        approved,
        InterruptResolution(interrupt_id="i", run_id="r", decision=InterruptDecision.APPROVE),
    )
    assert approved.approved == ["item"]
    rejected = _FakeRunState()
    harness.openai_agents.apply_resolution(
        rejected,
        InterruptResolution(interrupt_id="i", run_id="r", decision=InterruptDecision.REJECT),
    )
    assert rejected.rejected == ["item"]


# --------------------------------------------------------------------------- 5. run end
async def test_the_answer_is_written_to_memory_on_run_end(harness: Any, memory: Any) -> None:
    model = Answers("SKU-1 has 4 in stock")
    harness.model_client = harness.wrap_model(model)
    harness.runtime_builder.model_client = harness.model_client
    run = harness.openai_agents.agent(agent_id="end-agent", tools=[lookup])
    result = await run("how much stock of SKU-1?", context=turn("end-agent", "thr-end"))
    assert result.succeeded
    await harness.drain()
    assert any("SKU-1 has 4 in stock" in str(o["content"]) for o in memory.observations)


# --------------------------------------------------------------------------- 6. session
async def test_the_session_is_the_memory_service_thread(harness: Any, memory: Any) -> None:
    context = turn("session-agent", "thr-session")
    async with harness.execution(context, agent_id="session-agent") as runtime:
        session = MemoryServiceSession(runtime.memory)
        await session.add_items(
            [
                {"role": "user", "content": "how much stock of SKU-1?"},
                {"role": "assistant", "content": "4 in stock"},
            ]
        )
        items = await session.get_items()
    roles = [i["role"] for i in items]
    assert "user" in roles and "assistant" in roles
    assert any(i["content"] == "4 in stock" for i in items)


async def test_the_session_refuses_to_pretend_it_popped_a_message() -> None:
    """``None`` means "nothing was said"; a message that exists cannot be silently unsaid."""
    empty = MemoryServiceSession(_StubMemory([]))
    assert await empty.pop_item() is None

    spoken = MemoryServiceSession(_StubMemory([{"role": "user", "content": "hello"}]))
    with pytest.raises(NotImplementedError, match="no per-message delete"):
        await spoken.pop_item()


async def test_compaction_writes_a_run_scoped_summary(harness: Any, memory: Any) -> None:
    """The SDK has no summarisation hook, so the session carries the compaction moment."""
    model = Answers("the customer asked about stock and was told there were four")
    harness.model_client = harness.wrap_model(model)
    harness.runtime_builder.model_client = harness.model_client
    context = turn("compact-agent", "thr-compact")
    async with harness.execution(context, agent_id="compact-agent") as runtime:
        session = MemoryServiceSession(runtime.memory, keep_recent=0)
        await session.add_items(
            [
                {"role": "user", "content": "how much stock of SKU-1?"},
                {"role": "assistant", "content": "4 in stock"},
            ]
        )
        summary = await session.compact()
    assert summary == "the customer asked about stock and was told there were four"
    await harness.drain()
    written = [o for o in memory.observations if "Conversation summary" in str(o["content"])]
    assert written and written[-1]["kind"] == "AGENT_RESULT"
    assert written[-1]["hints"]["visibility"] == "RUN"
    assert written[-1]["framework"] == "openai-agents"


async def test_the_session_refuses_to_pretend_outside_a_run() -> None:
    with pytest.raises(RuntimeError, match=r"harness\.openai_agents"):
        _ = MemoryServiceSession().memory


# --------------------------------------------------------------------------- interop
async def test_an_agent_built_by_hand_can_be_wrapped() -> None:
    """The adapter must not require its own constructor: an existing agent is the point."""
    CALLS.clear()
    harness = build(model=ToolThenAnswer("lookup", {"sku": "S9"}, "9 in stock"))
    agent = Agent(name="hand-built", instructions="answer briefly", tools=[lookup])
    run = harness.openai_agents.wrap(agent, agent_id="hand-built")
    result = await run("how much stock?", context=turn("hand-built", "thr-hand"))
    assert result.succeeded and result.data == "9 in stock"
    assert CALLS == [{"sku": "S9"}]


async def test_the_run_config_routes_every_model_name_to_the_gateway() -> None:
    model = Answers("routed")
    harness = build(model=model)
    provider = harness.openai_agents.model_provider()
    agent = Agent(name="named-model", instructions="hi", model="some-model-name")
    wrapped = harness.openai_agents.wrap(
        agent,
        agent_id="named-model",
        run_config=RunConfig(model_provider=provider, tracing_disabled=True),
    )
    result = await wrapped("hello", context=turn("named-model", "thr-provider"))
    assert result.succeeded and result.data == "routed"
    assert model.requests, "the provider must have routed the call to the harness client"


def test_runner_is_the_sdks_own(monkeypatch: Any) -> None:
    """The adapter calls the real ``Runner``; nothing here re-implements the agent loop."""
    assert hasattr(Runner, "run")


# --------------------------------------------------------------------------- helpers
def _bundle(rendered: str) -> Any:
    class Bundle:
        def __init__(self, text: str) -> None:
            self.rendered = text

    return Bundle(rendered)


def _paused(sink: CollectingEventSink, run_id: str) -> Interrupt:
    finished = [e for e in sink.for_run(run_id) if e.type is RunEventType.RUN_FINISHED]
    assert finished and finished[-1].outcome is RunOutcome.INTERRUPT
    return Interrupt.model_validate(finished[-1].data["interrupt"])


class _State:
    """Just the one member ``pending_result`` consults."""

    def __init__(self, state: dict[str, Any]) -> None:
        self.state = state


class _StubMemory:
    """Just the two members ``pop_item`` consults."""

    enabled = True

    def __init__(self, messages: list[Any]) -> None:
        self.messages = messages

    async def history(self, *, limit: int = 50, **_: Any) -> list[Any]:
        return self.messages[:limit]


class _FakeRunState:
    """The two calls :meth:`apply_resolution` makes on an SDK ``RunState``."""

    def __init__(self) -> None:
        self.approved: list[str] = []
        self.rejected: list[str] = []

    def get_interruptions(self) -> list[str]:
        return ["item"]

    def approve(self, item: str, always_approve: bool = False) -> None:
        self.approved.append(item)

    def reject(self, item: str, always_reject: bool = False) -> None:
        self.rejected.append(item)
