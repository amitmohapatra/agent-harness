"""Deep Agents adapter: the six moments of design §8, against the installed framework.

No network and no real model: the model is a scripted client behind the harness's own model
port, except where a test is specifically about the gateway — there a real HTTP server
answers on a real socket (``tests.support_gateway.FakeGateway``), because "the request the
framework would send" is only believable if something received it.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
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
from trellis.harness_deepagents import (
    BifrostChatModel,
    DeepAgentsHarness,
    MemoryServiceBackend,
    TrellisMiddleware,
    deepagents_version,
    to_model_request,
)

TENANT = "acme"
CALLS: list[dict[str, Any]] = []


@tool
def lookup(sku: str) -> str:
    """Look up the stock of one SKU."""
    CALLS.append({"sku": sku})
    return f"{sku}: 4 in stock"


class ScriptedModel:
    """The harness's model port, answering from a script. Records every request."""

    def __init__(self, *responses: ModelResponse) -> None:
        self.script = list(responses)
        self.requests: list[Any] = []

    async def invoke(self, request: Any, /, **_: Any) -> ModelResponse:
        self.requests.append(request)
        if not self.script:
            return ModelResponse(text="done", model="scripted")
        return self.script.pop(0) if len(self.script) > 1 else self.script[0]

    async def structured(self, request: Any, /, schema: Any, **kwargs: Any) -> ModelResponse:
        return await self.invoke(request, **kwargs)


class ToolThenAnswer:
    """Asks for one tool, then answers once it sees a tool result.

    A rule rather than a fixed script, because a resumed run re-plans from scratch: a script
    would answer the resumed run's *first* model call with the second run's line and the
    tool would never be asked for again.
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


def turn(agent_id: str, thread_id: str) -> AgentExecutionContext:
    """A context naming its turn, so every run of it is the same run (what a resume needs)."""
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
        model=model or ScriptedModel(says("done")),
        policy=policy,
        event_sinks=[sink] if sink is not None else (),
        defaults={"tenant_id": TENANT},
        config={"telemetry": {"capture": {"inputs": True, "outputs": True}}},
    )


def ask(text: str = "how much stock of SKU-1?") -> dict[str, Any]:
    return {"messages": [{"role": "user", "content": text}]}


# --------------------------------------------------------------------------- the adapter
def test_the_adapter_reports_the_installed_version_and_its_availability() -> None:
    harness = build()
    assert harness.deepagents.version == deepagents_version()
    assert DeepAgentsHarness.available() is True
    assert harness.deepagents.supports(lookup) is True
    # the same object every time: the adapter is per harness, not per call
    assert harness.deepagents is harness.deepagents


def test_the_public_name_resolves_through_the_core() -> None:
    from trellis.harness import DeepAgentsHarness as exported

    assert exported is DeepAgentsHarness


# --------------------------------------------------------------------------- 1. context
async def test_the_memory_bundle_is_rendered_into_the_system_message() -> None:
    """The context moment: what the harness remembered reaches the framework's prompt."""
    harness = build()
    context = turn("ctx-agent", "thr-ctx")
    middleware = harness.deepagents.middleware()
    async with harness.execution(context, agent_id="ctx-agent") as runtime:
        runtime.memory_context = _bundle("The customer is on the Pro plan [mem_7]")
        seen: list[Any] = []

        async def handler(request: Any) -> Any:
            seen.append(request)
            return AIMessage(content="ok")

        await middleware.awrap_model_call(
            _model_request(system="You are an inventory agent."), handler
        )
    prompt = seen[0].system_message.text
    assert prompt.startswith("You are an inventory agent.")
    assert "The customer is on the Pro plan [mem_7]" in prompt
    assert "cite memory ids" in prompt


async def test_without_a_bundle_the_agents_own_prompt_is_untouched() -> None:
    harness = build()
    middleware = harness.deepagents.middleware()
    async with harness.execution(turn("ctx-agent", "thr-ctx2"), agent_id="ctx-agent"):
        seen: list[Any] = []

        async def handler(request: Any) -> Any:
            seen.append(request)
            return AIMessage(content="ok")

        original = _model_request(system="Just this.")
        await middleware.awrap_model_call(original, handler)
    assert seen[0].system_message.text == "Just this."


# --------------------------------------------------------------------------- 2. model
async def test_the_model_call_goes_through_the_harness_model_port() -> None:
    model = ScriptedModel(says("4 in stock"))
    harness = build(model=model)
    run = harness.deepagents.agent(agent_id="model-agent", tools=[lookup])
    result = await run(ask(), context=turn("model-agent", "thr-model"))
    assert result.succeeded
    # the framework's call arrived as a harness ModelRequest, not as a provider payload
    request = model.requests[0]
    assert next(m["role"] for m in request.messages) == "system"
    assert any(m["role"] == "user" for m in request.messages)
    assert any(t["function"]["name"] == "lookup" for t in request.tools)


async def test_the_model_call_reaches_a_real_gateway_over_a_real_socket() -> None:
    """Assert on the body the framework's model call actually put on the wire."""
    with FakeGateway([completion("4 in stock")]) as gateway:
        harness = build(model=BifrostModelClient(gateway.url, model="gateway-model"))
        run = harness.deepagents.agent(agent_id="gw-agent", tools=[lookup], model="gateway-model")
        result = await run(ask(), context=turn("gw-agent", "thr-gw"))
        assert result.succeeded
        body = gateway.requests[0]
    assert body["model"] == "gateway-model"
    assert any(m["role"] == "user" and "SKU-1" in str(m["content"]) for m in body["messages"])
    assert any(t["function"]["name"] == "lookup" for t in body["tools"])


def test_the_chat_model_converts_messages_and_tool_calls_both_ways() -> None:
    request = to_model_request(
        [
            SystemMessage(content="be brief"),
            HumanMessage(content="hi"),
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "lookup", "args": {"sku": "S1"}, "id": "c1", "type": "tool_call"}
                ],
            ),
            ToolMessage(content="S1: 4", tool_call_id="c1"),
        ],
        model="m",
    )
    roles = [m["role"] for m in request.messages or ()]
    assert roles == ["system", "user", "assistant", "tool"]
    assistant = (request.messages or [])[2]
    assert assistant["tool_calls"][0]["function"]["name"] == "lookup"
    assert json.loads(assistant["tool_calls"][0]["function"]["arguments"]) == {"sku": "S1"}
    assert (request.messages or [])[3]["tool_call_id"] == "c1"


async def test_the_chat_model_binds_tools_in_the_gateways_schema() -> None:
    model = ScriptedModel(calls("lookup", sku="S1"))
    bound = BifrostChatModel(client=model, model_name="m").bind_tools([lookup])
    message = await bound.ainvoke([HumanMessage(content="hi")])
    assert model.requests[0].tools[0]["function"]["name"] == "lookup"
    assert message.tool_calls[0]["name"] == "lookup"
    assert message.tool_calls[0]["args"] == {"sku": "S1"}


async def test_malformed_tool_arguments_become_an_invalid_tool_call_not_a_crash() -> None:
    broken = ModelResponse(
        text=None,
        model="scripted",
        tool_calls=[{"id": "c9", "function": {"name": "lookup", "arguments": "{not json"}}],
    )
    message = await BifrostChatModel(client=ScriptedModel(broken), model_name="m").ainvoke(
        [HumanMessage(content="hi")]
    )
    assert message.tool_calls == []
    assert message.invalid_tool_calls[0]["id"] == "c9"


def test_the_chat_model_never_carries_a_credential() -> None:
    model = BifrostChatModel(client=ScriptedModel(), model_name="gemini-3.8-flash")
    assert "gemini-3.8-flash" in repr(model._identifying_params)
    assert "api_key" not in repr(model) and "token" not in repr(model).lower()


# --------------------------------------------------------------------------- 3. tools
async def test_a_denied_tool_never_runs() -> None:
    CALLS.clear()

    async def deny(context: Any, call: Any) -> Any:
        return call.tool != "lookup"

    harness = build(
        model=ToolThenAnswer("lookup", {"sku": "S1"}, "4 in stock"),
        policy=CallablePolicyProvider(tool=deny),
    )
    run = harness.deepagents.agent(agent_id="deny-agent", tools=[lookup])
    with pytest.raises(PolicyDeniedError):
        await run(ask(), context=turn("deny-agent", "thr-deny"))
    assert CALLS == []


async def test_a_tool_call_opens_and_closes_on_the_run_stream() -> None:
    CALLS.clear()
    sink = CollectingEventSink()
    harness = build(model=ToolThenAnswer("lookup", {"sku": "S1"}, "4 in stock"), sink=sink)
    run = harness.deepagents.agent(agent_id="tool-agent", tools=[lookup])
    context = turn("tool-agent", "thr-tool")
    assert (await run(ask(), context=context)).succeeded
    assert CALLS == [{"sku": "S1"}]
    types = sink.types(context.agent_run_id)
    assert types[0] is RunEventType.RUN_STARTED
    assert RunEventType.STEP_STARTED in types and RunEventType.STEP_FINISHED in types
    quartet = [t for t in types if t.value.startswith("TOOL_CALL")]
    assert quartet == [
        RunEventType.TOOL_CALL_START,
        RunEventType.TOOL_CALL_ARGS,
        RunEventType.TOOL_CALL_END,
        RunEventType.TOOL_CALL_RESULT,
    ]
    assert types[-1] is RunEventType.RUN_FINISHED
    finished = [
        e for e in sink.for_run(context.agent_run_id) if e.type is RunEventType.RUN_FINISHED
    ]
    assert finished[-1].outcome is RunOutcome.SUCCESS
    assert len({e.attempt for e in sink.for_run(context.agent_run_id)}) == 1


# --------------------------------------------------------------------------- 4. pause
@pytest.mark.parametrize(
    ("decision", "expected_calls"),
    [
        (InterruptDecision.APPROVE, [{"sku": "S1"}]),
        (InterruptDecision.REJECT, []),
    ],
)
async def test_an_approval_pauses_the_run_and_the_answer_continues_it(
    decision: InterruptDecision, expected_calls: list[dict[str, Any]]
) -> None:
    CALLS.clear()
    sink = CollectingEventSink()

    async def hold(context: Any, call: Any) -> Any:
        return "require_approval" if call.tool == "lookup" else True

    harness = build(
        model=ToolThenAnswer("lookup", {"sku": "S1"}, "4 in stock"),
        policy=CallablePolicyProvider(tool=hold),
        sink=sink,
    )
    run = harness.deepagents.agent(agent_id="hitl-agent", tools=[lookup])
    context = turn("hitl-agent", f"thr-hitl-{decision.value.lower()}")

    with pytest.raises(ApprovalRequired) as paused:
        await run(ask(), context=context)
    assert paused.value.tool_call.tool == "lookup" and CALLS == []

    interrupt = _paused(sink, context.agent_run_id)
    assert interrupt.reason is InterruptReason.APPROVAL
    assert interrupt.tool_call is not None and interrupt.tool_call.args == {"sku": "S1"}

    resolution = InterruptResolution(
        interrupt_id=interrupt.interrupt_id,
        run_id=interrupt.run_id,
        decision=decision,
        reviewer="u1",
    )
    result = await harness.resume(interrupt, resolution, context=context, agent=run, payload=ask())
    assert result is not None and result.succeeded
    assert expected_calls == CALLS


# --------------------------------------------------------------------------- 5. run end
async def test_the_answer_is_written_to_memory_on_run_end(harness: Any, memory: Any) -> None:
    """The run-end moment, against the real Memory Service: the framework's last message is
    what the platform records as the answer."""
    model = ScriptedModel(says("SKU-1 has 4 in stock"))
    harness.model_client = harness.wrap_model(model)
    harness.runtime_builder.model_client = harness.model_client
    run = harness.deepagents.agent(agent_id="end-agent", tools=[lookup])
    context = turn("end-agent", "thr-end")
    result = await run(ask(), context=context)
    assert result.succeeded
    await harness.drain()
    observed = [o["content"] for o in memory.observations]
    assert any("SKU-1 has 4 in stock" in text for text in observed)


# --------------------------------------------------------------------------- 6. compaction
async def test_a_summarisation_summary_becomes_a_run_scoped_observation(
    harness: Any, memory: Any
) -> None:
    """Compaction: LangChain has no post-summary hook, so the adapter reads the summary out
    of the assembled model request and writes it exactly as ``ContextAssembler.compact`` does."""
    middleware = harness.deepagents.middleware()
    context = turn("compact-agent", "thr-compact")
    summary = HumanMessage(
        content="Here is a summary of the conversation to date:\n\nthe customer wants a refund",
        additional_kwargs={"lc_source": "summarization"},
    )
    async with harness.execution(context, agent_id="compact-agent"):

        async def handler(request: Any) -> Any:
            return AIMessage(content="ok")

        request = _model_request(system="s", messages=[summary])
        await middleware.awrap_model_call(request, handler)
        # a second model call must not write the same summary twice
        await middleware.awrap_model_call(request, handler)
    await harness.drain()
    written = [o for o in memory.observations if "Conversation summary" in str(o["content"])]
    assert len(written) == 1
    assert written[0]["kind"] == "AGENT_RESULT"
    assert written[0]["hints"]["visibility"] == "RUN"
    assert written[0]["framework"] == "deepagents"


# --------------------------------------------------------------------------- the backend
async def test_the_memory_backend_serves_memories_off_the_service(
    harness: Any, memory: Any
) -> None:
    """Deep Agents' ``/memories/*`` files, addressed through the Memory Service."""
    context = turn("fs-agent", "thr-fs")
    async with harness.execution(context, agent_id="fs-agent") as runtime:
        backend = MemoryServiceBackend(runtime.memory, visibility="USER")
        written = await backend.awrite("/memories/refund-policy.md", "Refunds within 30 days.")
        assert written.error is None and written.path.startswith("/memories/")
        # the service names the note, not the caller: the requested stem is not the path
        assert "refund-policy" not in written.path
        listed = await backend.als("/memories")
        assert listed.error is None and listed.entries is not None
        assert any(e["path"] == written.path for e in listed.entries)
        read = await backend.aread(written.path)
        assert read.error is None
        assert "Refunds within 30 days." in read.file_data["content"]
        found = await backend.agrep("30 days")
        assert found.error is None and found.matches
        globbed = await backend.aglob("*.md")
        assert globbed.error is None and globbed.matches
        edited = await backend.aedit(written.path, "30 days", "60 days")
        assert edited.error is None and edited.occurrences == 1
        removed = await backend.adelete(edited.path)
        assert removed.error is None


async def test_the_backend_refuses_paths_it_does_not_own(harness: Any) -> None:
    async with harness.execution(turn("fs-agent", "thr-fs2"), agent_id="fs-agent") as runtime:
        backend = MemoryServiceBackend(runtime.memory)
        assert "serves /memories/ only" in (await backend.als("/etc")).error
        assert "serves /memories/ only" in (await backend.aread("/etc/passwd")).error
        assert "serves /memories/ only" in (await backend.awrite("/tmp/x", "y")).error
        assert (await backend.agrep("[")).error.startswith("invalid pattern")


async def test_the_pieces_refuse_to_pretend_outside_a_run() -> None:
    """A graph run outside the harness must fail loudly, not quietly record nothing."""
    with pytest.raises(RuntimeError, match=r"harness\.deepagents\.agent"):
        _ = MemoryServiceBackend().memory
    with pytest.raises(RuntimeError, match=r"harness\.deepagents\.agent"):
        _ = TrellisMiddleware().runtime


# --------------------------------------------------------------------------- helpers
def _bundle(rendered: str) -> Any:
    class Bundle:
        def __init__(self, text: str) -> None:
            self.rendered = text

    return Bundle(rendered)


def _model_request(*, system: str, messages: list[Any] | None = None) -> Any:
    from langchain.agents.middleware.types import ModelRequest

    return ModelRequest(
        model=BifrostChatModel(client=ScriptedModel(), model_name="m"),
        messages=messages if messages is not None else [HumanMessage(content="hi")],
        system_message=SystemMessage(content=system),
    )


def _paused(sink: CollectingEventSink, run_id: str) -> Interrupt:
    finished = [e for e in sink.for_run(run_id) if e.type is RunEventType.RUN_FINISHED]
    assert finished and finished[-1].outcome is RunOutcome.INTERRUPT
    return Interrupt.model_validate(finished[-1].data["interrupt"])
