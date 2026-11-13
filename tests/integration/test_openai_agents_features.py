"""The OpenAI Agents SDK's own features, run through the harness with the real ``Runner`` and a
scripted ``Model``: each does what the SDK says it does, and the harness's part still holds for
it — the calls made inside it governed and recorded, the memory context pushed and the turn
written, a pause inside it resumed, a cancel honoured.

Function tools (their failure policy, their time limit, the run's context object), dynamic
instructions, agent and run lifecycle hooks, guardrails (input, output, tool), structured
output, ``tool_use_behavior``, model settings, handoffs (nested, described, filtered), agents
as tools (an approval of the SDK's own inside one), and the tools the SDK runs itself (a
shell). ``test_today_...`` records what the harness does today where that is not what the SDK
does without it."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from agents import (
    Agent,
    AgentHooks,
    GuardrailFunctionOutput,
    ModelSettings,
    RunConfig,
    RunContextWrapper,
    RunHooks,
    SessionSettings,
    ShellTool,
    StopAtTools,
    ToolGuardrailFunctionOutput,
    Usage,
    agent_tool_state,
    function_tool,
    handoff,
    input_guardrail,
    output_guardrail,
    tool_input_guardrail,
    tool_output_guardrail,
)
from agents.agent import ToolsToFinalOutputResult
from pydantic import BaseModel

from tests.support.arm import declaring
from tests.support.memory import FakeMemoryService
from tests.support.openai_model import Items, ScriptedModel
from trellis import Budget, Harness, Settings
from trellis.contracts import InterruptReason, RunEventType, RunOutcome, RunStatus
from trellis.harness import budget
from trellis.harness.adapters.openai_agents import _asked_about
from trellis.harness.runs import LocalRuns
from trellis.harness.tools import bridge

ran: list[str] = []


@pytest.fixture(autouse=True)
def _reset() -> None:
    ran.clear()


@function_tool
def look(order: str) -> str:
    """Look an order up."""
    ran.append(f"look {order}")
    return f"{order}: shipped"


@function_tool
def send(order: str) -> str:
    """Send an order."""
    ran.append(f"send {order}")
    return f"sent {order}"


@function_tool(needs_approval=True)
def gated(order: str) -> str:
    """A tool behind the SDK's own approval."""
    ran.append(f"gated {order}")
    return f"gated {order}"


def recorded(memory_service: FakeMemoryService) -> list[tuple[str, str]]:
    """The tool calls recorded in the memory service: each tool and how it ended."""
    return [(c.body["tool"], c.body["status"]) for c in memory_service.named("record_tool")]


def outputs(model: ScriptedModel) -> list[Any]:
    """The tool results the model's latest call was given, in order."""
    return [
        item["output"]
        for item in model.inputs[-1]
        if isinstance(item, dict) and item.get("type") == "function_call_output"
    ]


def written(memory_service: FakeMemoryService) -> list[tuple[str, str]]:
    """The chat turns written to the memory service's thread: who said what."""
    return [
        (m["role"], m["content"])
        for call in memory_service.named("messages")
        for m in call.body["messages"]
        if m["role"] in ("USER", "ASSISTANT") and m["content"]
    ]


async def children(h: Harness, parent_run_id: str) -> list[Any]:
    found = [s async for s in h.runs.iterate(parent_run_id=parent_run_id)]
    return [await h.runs.get(s.run_id) for s in found]


class Offered(ScriptedModel):
    """A scripted model that keeps what each call was given beside its input: its settings and
    its handoffs."""

    def __init__(self, turns: list[Any], **kwargs: Any) -> None:
        super().__init__(turns, **kwargs)
        self.settings: list[Any] = []
        self.handoffs: list[list[tuple[str, str]]] = []

    def _offered(
        self, tools: Any, _schema: Any = None, handoffs: Any = (), *_: Any, **k: Any
    ) -> None:
        super()._offered(tools, _schema, handoffs, *_, **k)
        given = k.get("handoffs", handoffs)
        self.handoffs.append([(h.tool_name, h.tool_description) for h in given])

    async def get_response(  # type: ignore[override]
        self, system_instructions: Any, input: Any, model_settings: Any = None, *a: Any, **k: Any
    ) -> Any:
        self.settings.append(model_settings)
        return await super().get_response(system_instructions, input, model_settings, *a, **k)


# --------------------------------------------------------------------------- function tools
async def test_a_failing_tool_tells_the_model_what_its_failure_function_says(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    """A tool that raises: the model reads the SDK's default failure text, or the tool's own
    ``failure_error_function``'s, and the run goes on — each call governed as the run's."""

    @function_tool
    def boom(order: str) -> str:
        raise ValueError("it broke")

    @function_tool(failure_error_function=lambda _context, error: f"custom: {error}")
    def crack(order: str) -> str:
        raise ValueError("it cracked")

    model = ScriptedModel([("boom", {"order": "o1"}), ("crack", {"order": "o1"}), "both failed"])
    h = declaring(memory_harness, boom="read", crack="irreversible")
    agent = h.wrap(Agent(name="desk", model=model, tools=[boom, crack]), id="failing")
    paused = await agent.run("try both", user="u")
    assert paused.interrupt is not None and paused.interrupt.tool_call is not None
    assert paused.interrupt.tool_call.tool == "crack"  # the catalog's rule asks first
    done = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="r")
    assert done.status is RunStatus.SUCCESS and done.answer == "both failed"
    assert outputs(model) == [
        "An error occurred while running the tool. Please try again. Error: it broke",
        "custom: it cracked",
    ]


async def test_today_a_failure_the_sdk_formats_is_recorded_as_an_ok_call(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    """Today: the SDK turns the tool's exception into text inside the tool's own invoke, which
    is all the harness sees of it — the call is recorded ``ok``, its output the failure text.
    (The SDK gives no public sign that the text is a failure.)"""

    @function_tool
    def boom(order: str) -> str:
        raise ValueError("it broke")

    model = ScriptedModel([("boom", {"order": "o1"}), "failed"])
    h = declaring(memory_harness, boom="read")
    events = [
        e
        async for e in h.wrap(Agent(name="desk", model=model, tools=[boom]), id="ok").stream(
            "try", user="u"
        )
    ]
    await h.writes.drain()
    assert recorded(memory_service) == [("boom", "ok")]
    [result] = [e.data for e in events if e.type is RunEventType.TOOL_CALL_RESULT]
    assert result["status"] == "ok" and "it broke" in result["output"]


async def test_today_a_tool_that_asks_to_raise_does_not_end_the_run(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    """Today: ``failure_error_function=None`` (the SDK: the exception ends the run) does not end
    a harness run — every failure of a governed call is an outcome its model reads
    (``tools.bridge.call``), recorded as an error."""

    @function_tool(failure_error_function=None)
    def boom(order: str) -> str:
        raise ValueError("it broke")

    model = ScriptedModel([("boom", {"order": "o1"}), "it failed"])
    h = declaring(memory_harness, boom="read")
    result = await h.wrap(Agent(name="desk", model=model, tools=[boom]), id="raises").run(
        "try", user="u"
    )
    await h.writes.drain()
    assert result.status is RunStatus.SUCCESS and result.answer == "it failed"
    assert outputs(model) == ["boom failed: it broke"]
    assert recorded(memory_service) == [("boom", "error")]


async def test_a_tools_own_time_limit_is_the_sdks(harness: Harness) -> None:
    """``function_tool(timeout=...)``: past it the model reads the SDK's timeout text and the
    run goes on; with ``timeout_behavior="raise_exception"`` the run fails with the SDK's
    ``ToolTimeoutError``."""

    @function_tool(timeout=0.05)
    async def slow(order: str) -> str:
        await asyncio.sleep(5)
        return "late"

    @function_tool(timeout=0.05, timeout_behavior="raise_exception")
    async def strict(order: str) -> str:
        await asyncio.sleep(5)
        return "late"

    h = declaring(harness, slow="read", strict="read")
    model = ScriptedModel([("slow", {"order": "o1"}), "it timed out"])
    events = [
        e
        async for e in h.wrap(Agent(name="desk", model=model, tools=[slow]), id="slow").stream(
            "try", user="u"
        )
    ]
    assert events[-1].outcome is RunOutcome.SUCCESS
    assert outputs(model) == ["Tool 'slow' timed out after 0.05 seconds."]
    # today: the SDK cancels the call it timed out, which the harness reports as a call cut
    # short by a cancelled run
    [cut] = [e.data for e in events if e.type is RunEventType.TOOL_CALL_RESULT]
    assert cut["status"] == bridge.CANCELLED
    model = ScriptedModel([("strict", {"order": "o1"}), "never"])
    failed = await h.wrap(Agent(name="desk", model=model, tools=[strict]), id="strict").run(
        "try", user="u"
    )
    assert failed.status is RunStatus.ERROR and failed.error is not None
    assert failed.error.code == "ToolTimeoutError" and len(model.inputs) == 1


async def test_the_runs_context_object_reaches_tools_instructions_and_agent_hooks(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    """``context=`` (a framework option) is the SDK's ``RunContextWrapper.context`` in a tool,
    in the agent's dynamic instructions and in its ``AgentHooks``; the instructions the
    function returns are what the model is given, the day after them; the tool's call is
    recorded."""
    seen: list[Any] = []

    @function_tool
    def who(context: RunContextWrapper[dict[str, str]], order: str) -> str:
        """Who asks about an order."""
        return f"{context.context['name']} asks about {order}"

    def instructions(context: RunContextWrapper[dict[str, str]], agent: Agent[Any]) -> str:
        return f"You serve {context.context['name']}."

    class Watching(AgentHooks[Any]):
        async def on_start(self, context: Any, agent: Any) -> None:
            seen.append(("start", agent.name, context.context["name"]))

        async def on_tool_start(self, context: Any, agent: Any, tool: Any) -> None:
            seen.append(("tool_start", tool.name))

        async def on_tool_end(self, context: Any, agent: Any, tool: Any, result: Any) -> None:
            seen.append(("tool_end", tool.name, result))

        async def on_end(self, context: Any, agent: Any, output: Any) -> None:
            seen.append(("end", output))

    model = ScriptedModel([("who", {"order": "o1"}), "done"])
    target = Agent(
        name="desk", model=model, tools=[who], instructions=instructions, hooks=Watching()
    )
    h = declaring(memory_harness, who="read")
    agent = h.wrap(target, id="context", framework_options={"context": {"name": "ada"}})
    result = await agent.run("who asks?", user="u")
    await h.writes.drain()
    assert result.status is RunStatus.SUCCESS and outputs(model) == ["ada asks about o1"]
    assert all(said is not None and said.startswith("You serve ada.\n\n") for said in model.system)
    assert seen == [
        ("start", "desk", "ada"),
        ("tool_start", "who"),
        ("tool_end", "who", "ada asks about o1"),
        ("end", "done"),
    ]
    assert recorded(memory_service) == [("who", "ok")]


async def test_what_a_tool_put_in_the_context_is_there_after_an_approval(harness: Harness) -> None:
    """A tool writes to the run's context object, the SDK's own approval pauses the run, and
    the resumed run's next tool reads what was written."""

    @function_tool
    def put(context: RunContextWrapper[dict[str, Any]], value: str) -> str:
        context.context["value"] = value
        return "put"

    @function_tool
    def get(context: RunContextWrapper[dict[str, Any]]) -> str:
        return str(context.context.get("value"))

    model = ScriptedModel(
        [("put", {"value": "kept"}), ("gated", {"order": "o1"}), ("get", {}), "done"]
    )
    h = declaring(harness, put="read", get="read", gated="read")
    agent = h.wrap(
        Agent(name="desk", model=model, tools=[put, get, gated]),
        id="kept",
        framework_options={"context": {}},
    )
    paused = await agent.run("go", user="u")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
    done = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="r")
    assert done.status is RunStatus.SUCCESS and outputs(model) == ["put", "gated o1", "kept"]


# --------------------------------------------------------------------------- run hooks
class Seen(RunHooks[Any]):
    """The SDK's run hooks, each event kept."""

    def __init__(self) -> None:
        self.events: list[tuple[Any, ...]] = []

    async def on_agent_start(self, context: Any, agent: Any) -> None:
        self.events.append(("agent_start", agent.name))

    async def on_agent_end(self, context: Any, agent: Any, output: Any) -> None:
        self.events.append(("agent_end", agent.name, output))

    async def on_handoff(self, context: Any, from_agent: Any, to_agent: Any) -> None:
        self.events.append(("handoff", from_agent.name, to_agent.name))

    async def on_tool_start(self, context: Any, agent: Any, tool: Any) -> None:
        self.events.append(("tool_start", tool.name))

    async def on_tool_end(self, context: Any, agent: Any, tool: Any, result: Any) -> None:
        self.events.append(("tool_end", tool.name, result))

    async def on_llm_start(
        self, context: Any, agent: Any, system_prompt: Any, input_items: Any
    ) -> None:
        self.events.append(("llm_start", agent.name))

    async def on_llm_end(self, context: Any, agent: Any, response: Any) -> None:
        self.events.append(("llm_end", agent.name))


async def test_the_runs_own_run_hooks_see_the_run_beside_the_harnesss(harness: Harness) -> None:
    """``hooks=`` (the SDK's ``RunHooks``, a framework option) is called with everything the
    SDK reports — agents starting and ending, a handoff, the tool calls, the model calls —
    while the harness's own still count every reply toward the run's budget."""
    seen = Seen()
    clerk = Agent(
        name="clerk",
        model=ScriptedModel([("look", {"order": "o1"}), "o1 shipped"], ids="c"),
        tools=[look],
    )
    desk = Agent(name="desk", model=ScriptedModel([("transfer_to_clerk", {})]), handoffs=[clerk])
    h = declaring(harness, look="read")
    agent = h.wrap(desk, id="hooked", framework_options={"hooks": seen})
    events = [e async for e in agent.stream("where is o1?", user="u")]
    assert events[-1].outcome is RunOutcome.SUCCESS and events[-1].data["result"] == "o1 shipped"
    assert seen.events == [
        ("agent_start", "desk"),
        ("llm_start", "desk"),
        ("llm_end", "desk"),
        ("handoff", "desk", "clerk"),
        ("agent_start", "clerk"),
        ("llm_start", "clerk"),
        ("llm_end", "clerk"),
        ("tool_start", "look"),
        ("tool_end", "look", "o1: shipped"),
        ("llm_start", "clerk"),
        ("llm_end", "clerk"),
        ("agent_end", "clerk", "o1 shipped"),
    ]
    assert events[-1].data["spent"]["model_calls"] == 3  # counted by the harness's hooks


async def test_run_hooks_do_not_stand_in_for_the_harnesss_budget_or_its_last_answer() -> None:
    """With the run's own ``RunHooks``: a token budget still stops the run, and a run at its
    turn limit still ends with one last answer, that call seen by the hooks too."""
    seen = Seen()
    model = ScriptedModel(
        [("look", {"order": "o1"}), "never"], usage=Usage(input_tokens=50, output_tokens=10)
    )
    async with declaring(Harness(config=Settings()), look="read") as h:
        spent = await h.wrap(
            Agent(name="desk", model=model, tools=[look]),
            id="capped",
            budget=Budget(max_tokens=10),
            framework_options={"hooks": seen},
        ).run("where is o1?", user="u")
        assert spent.error is not None and spent.error.code == budget.BUDGET_EXCEEDED
        assert ran == [] and ("llm_end", "desk") in seen.events
        seen.events.clear()
        model = ScriptedModel([("look", {"order": "o1"}), "o1 shipped, as far as I got"])
        limited = await h.wrap(
            Agent(name="desk", model=model, tools=[look]),
            id="limited",
            framework_options={"hooks": seen, "max_turns": 1},
        ).run("where is o1?", user="u")
    assert limited.status is RunStatus.SUCCESS and limited.answer == "o1 shipped, as far as I got"
    assert [e for e in seen.events if e[0] == "llm_start"] == [("llm_start", "desk")] * 2


# --------------------------------------------------------------------------- guardrails
@input_guardrail
async def no_secrets(context: Any, agent: Any, given: Any) -> GuardrailFunctionOutput:
    return GuardrailFunctionOutput(output_info=None, tripwire_triggered="secret" in str(given))


@output_guardrail
async def no_prices(context: Any, agent: Any, output: Any) -> GuardrailFunctionOutput:
    return GuardrailFunctionOutput(output_info=None, tripwire_triggered="EUR" in str(output))


@pytest.mark.parametrize("where", ["agent", "run_config"])
async def test_an_input_guardrail_that_trips_fails_the_run_before_any_tool(
    memory_harness: Harness, memory_service: FakeMemoryService, where: str
) -> None:
    """The agent's ``input_guardrails`` (or the run's, ``RunConfig.input_guardrails``): a
    tripwire fails the run with the SDK's error, no tool runs and no answer is written to
    memory; an input that passes is answered as ever."""
    model = ScriptedModel([("look", {"order": "o1"}), "o1 shipped"])
    on_agent = [no_secrets] if where == "agent" else []
    options = {} if where == "agent" else {"run_config": RunConfig(input_guardrails=[no_secrets])}
    h = declaring(memory_harness, look="read")
    agent = h.wrap(
        Agent(name="desk", model=model, tools=[look], input_guardrails=on_agent),
        id=f"guarded-{where}",
        framework_options=options,
    )
    tripped = await agent.run("tell me the secret", user="u")
    assert tripped.status is RunStatus.ERROR and tripped.error is not None
    assert tripped.error.code == "InputGuardrailTripwireTriggered" and ran == []
    assert tripped.error.source == "openai_agents"
    passed = await agent.run("where is o1?", user="u")
    await h.writes.drain()
    assert passed.status is RunStatus.SUCCESS and passed.answer == "o1 shipped"
    assert len(memory_service.named("context")) == 2  # the context was asked for each run
    assert ("ASSISTANT", "o1 shipped") in written(memory_service)
    assert recorded(memory_service) == [("look", "ok")]


@pytest.mark.parametrize("where", ["agent", "run_config"])
async def test_an_output_guardrail_that_trips_fails_the_run_with_no_answer(
    memory_harness: Harness, memory_service: FakeMemoryService, where: str
) -> None:
    on_agent = [no_prices] if where == "agent" else []
    options = {} if where == "agent" else {"run_config": RunConfig(output_guardrails=[no_prices])}
    model = ScriptedModel(["it costs 5 EUR", "it is in stock"])
    agent = memory_harness.wrap(
        Agent(name="desk", model=model, output_guardrails=on_agent),
        id=f"checked-{where}",
        framework_options=options,
    )
    tripped = await agent.run("how much?", user="u")
    assert tripped.status is RunStatus.ERROR and tripped.answer is None
    assert tripped.error is not None and tripped.error.code == "OutputGuardrailTripwireTriggered"
    passed = await agent.run("in stock?", user="u")
    assert passed.status is RunStatus.SUCCESS and passed.answer == "it is in stock"


async def test_tool_guardrails_decide_what_runs_and_what_the_model_reads(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    """A tool's own guardrails (``tool_input_guardrails``, ``tool_output_guardrails``): a
    refused input is not run and not recorded, a refused output is replaced for the model
    while the call that ran is recorded, and a call both allow runs as ever."""

    @tool_input_guardrail
    def no_test_orders(data: Any) -> ToolGuardrailFunctionOutput:
        if "test" in data.context.tool_arguments:
            return ToolGuardrailFunctionOutput.reject_content("test orders are not looked up")
        return ToolGuardrailFunctionOutput.allow()

    @tool_output_guardrail
    def no_addresses(data: Any) -> ToolGuardrailFunctionOutput:
        if "street" in str(data.output):
            return ToolGuardrailFunctionOutput.reject_content("the address is withheld")
        return ToolGuardrailFunctionOutput.allow()

    @function_tool(tool_input_guardrails=[no_test_orders], tool_output_guardrails=[no_addresses])
    def address(order: str) -> str:
        """Where an order goes."""
        ran.append(order)
        return "1 Main street" if order == "o2" else "a parcel locker"

    model = ScriptedModel(
        [
            ("address", {"order": "test"}),
            ("address", {"order": "o2"}),
            ("address", {"order": "o3"}),
            "done",
        ]
    )
    h = declaring(memory_harness, address="read")
    result = await h.wrap(Agent(name="desk", model=model, tools=[address]), id="tg").run(
        "where do they go?", user="u"
    )
    await h.writes.drain()
    assert result.status is RunStatus.SUCCESS and ran == ["o2", "o3"]
    assert outputs(model) == [
        "test orders are not looked up",
        "the address is withheld",
        "a parcel locker",
    ]
    assert recorded(memory_service) == [("address", "ok"), ("address", "ok")]


async def test_a_tool_guardrail_that_raises_fails_the_run(harness: Harness) -> None:
    @tool_input_guardrail
    def never(data: Any) -> ToolGuardrailFunctionOutput:
        return ToolGuardrailFunctionOutput.raise_exception(output_info="not allowed")

    @function_tool(tool_input_guardrails=[never])
    def wipe(order: str) -> str:
        ran.append(order)
        return "wiped"

    model = ScriptedModel([("wipe", {"order": "o1"}), "never"])
    result = (
        await declaring(harness, wipe="read")
        .wrap(Agent(name="desk", model=model, tools=[wipe]), id="tripped")
        .run("wipe o1", user="u")
    )
    assert result.status is RunStatus.ERROR and result.error is not None and ran == []
    assert result.error.code == "ToolInputGuardrailTripwireTriggered"


# --------------------------------------------------------------------------- answers
class Verdict(BaseModel):
    ok: bool
    why: str


async def test_a_structured_answer_is_the_agents_output_type_after_an_approval_too(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    """``output_type``: the answer is the SDK's parsed object, kept as JSON with the run and
    on the stream — the same after a pause on a call governance asks about."""
    model = ScriptedModel(
        [("send", {"order": "o1"}), '{"ok": true, "why": "sent"}', '{"ok": false, "why": "no"}']
    )
    h = declaring(memory_harness, send="irreversible")
    agent = h.wrap(
        Agent(name="judge", model=model, tools=[send], output_type=Verdict), id="verdict"
    )
    paused = await agent.run("send o1 and judge it", user="u")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None and ran == []
    done = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="r")
    assert done.status is RunStatus.SUCCESS and done.answer == Verdict(ok=True, why="sent")
    record = await h.runs.get(done.run_id)
    assert record is not None and record.output == {"ok": True, "why": "sent"}
    events = [e async for e in agent.stream("judge again", user="u")]
    assert events[-1].data["result"] == {"ok": False, "why": "no"}
    await h.writes.drain()
    assert recorded(memory_service) == [("send", "ok")] and ran == ["send o1"]


async def test_tool_use_behavior_ends_the_run_with_a_tools_result(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    """``tool_use_behavior``: the first tool's result (``stop_on_first_tool``), a named tool's
    (``StopAtTools``) or what a function of the agent's makes of the results is the run's
    answer, the model not asked again — each call governed and recorded all the same."""

    def summed(context: Any, results: list[Any]) -> ToolsToFinalOutputResult:
        said = " + ".join(str(r.output) for r in results)
        return ToolsToFinalOutputResult(is_final_output=True, final_output=said)

    h = declaring(memory_harness, look="read", send="irreversible")
    model = ScriptedModel([("look", {"order": "o1"}), "never"])
    first = Agent(name="a", model=model, tools=[look], tool_use_behavior="stop_on_first_tool")
    result = await h.wrap(first, id="first").run("o1?", user="u")
    assert result.answer == "o1: shipped" and len(model.inputs) == 1
    model = ScriptedModel([("look", {"order": "o1"}), ("send", {"order": "o1"}), "never"])
    named = Agent(
        name="b",
        model=model,
        tools=[look, send],
        tool_use_behavior=StopAtTools(stop_at_tool_names=["send"]),
    )
    agent = h.wrap(named, id="named")
    paused = await agent.run("look then send o1", user="u")
    assert paused.interrupt is not None and ran == ["look o1", "look o1"]  # send asks first
    result = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="r")
    assert result.status is RunStatus.SUCCESS and result.answer == "sent o1"
    assert len(model.inputs) == 2  # the approved call's result ended the run
    model = ScriptedModel([[("look", {"order": "o1"}), ("look", {"order": "o2"})], "never"])
    custom = Agent(name="c", model=model, tools=[look], tool_use_behavior=summed)
    result = await h.wrap(custom, id="custom").run("both?", user="u")
    assert result.answer == "o1: shipped + o2: shipped" and len(model.inputs) == 1
    await h.writes.drain()
    assert (
        recorded(memory_service) == [("look", "ok")] * 2 + [("send", "ok")] + [("look", "ok")] * 2
    )


async def test_the_agents_model_settings_reach_the_model_as_the_sdk_sends_them(
    harness: Harness,
) -> None:
    """``ModelSettings``: ``tool_choice`` is sent, then reset after a tool call unless
    ``reset_tool_choice=False``; ``parallel_tool_calls`` and the sampling settings pass as
    given; the run's own (``RunConfig.model``, ``RunConfig.model_settings``) are over the
    agent's."""
    h = declaring(harness, look="read")
    settings = ModelSettings(tool_choice="required", parallel_tool_calls=False, temperature=0.2)
    model = Offered([("look", {"order": "o1"}), "done"])
    await h.wrap(
        Agent(name="a", model=model, tools=[look], model_settings=settings), id="reset"
    ).run("o1?", user="u")
    assert [s.tool_choice for s in model.settings] == ["required", None]
    assert all(s.parallel_tool_calls is False and s.temperature == 0.2 for s in model.settings)
    model = Offered([("look", {"order": "o1"}), "done"])
    kept = Agent(
        name="b",
        model=model,
        tools=[look],
        reset_tool_choice=False,
        model_settings=ModelSettings(tool_choice="look"),
    )
    await h.wrap(kept, id="kept-choice").run("o1?", user="u")
    assert [s.tool_choice for s in model.settings] == ["look", "look"]
    theirs, runs = Offered(["the agent's"]), Offered(["the run's"])
    config = RunConfig(model=runs, model_settings=ModelSettings(temperature=0.9))
    result = await h.wrap(
        Agent(name="c", model=theirs, model_settings=ModelSettings(temperature=0.1, top_p=0.5)),
        id="over",
        framework_options={"run_config": config},
    ).run("x", user="u")
    assert result.answer == "the run's" and theirs.settings == []
    assert (runs.settings[0].temperature, runs.settings[0].top_p) == (0.9, 0.5)


# --------------------------------------------------------------------------- handoffs
async def test_nested_handoffs_reach_the_last_agent_whose_call_pauses_and_resumes_there(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    """desk hands off to billing, billing to refunds: each is offered the next by its
    ``handoff_description``; refunds' own tool asks a person, and the approval continues the
    SDK's run in refunds (no agent before it is asked again). One run, one turn in memory."""
    refunds_model = ScriptedModel([("send", {"order": "o1"}), "refund sent"], ids="r")
    refunds = Agent(
        name="refunds", model=refunds_model, tools=[send], handoff_description="Sends refunds."
    )
    billing_model = Offered([("transfer_to_refunds", {})], ids="b")
    billing = Agent(
        name="billing", model=billing_model, handoffs=[refunds], handoff_description="Bills."
    )
    desk_model = Offered([("transfer_to_billing", {})], ids="d")
    desk = Agent(name="desk", model=desk_model, handoffs=[billing])
    h = declaring(memory_harness, send="irreversible")
    agent = h.wrap(desk, id="nested")
    paused = await agent.run("refund o1", user="u")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
    assert paused.interrupt.reason is InterruptReason.APPROVAL and ran == []
    assert paused.interrupt.tool_call is not None and paused.interrupt.tool_call.tool == "send"
    assert desk_model.handoffs[0][0][0] == "transfer_to_billing"
    assert desk_model.handoffs[0][0][1].endswith("Bills.")
    assert billing_model.handoffs[0][0][1].endswith("Sends refunds.")
    done = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="r")
    assert done.status is RunStatus.SUCCESS and done.answer == "refund sent"
    assert ran == ["send o1"]
    assert (len(desk_model.inputs), len(billing_model.inputs), len(refunds_model.inputs)) == (
        1,
        1,
        2,
    )
    await h.writes.drain()
    assert recorded(memory_service) == [("send", "ok")]
    asked = memory_service.named("context")[0].body["tools"]["available"]
    assert {"send", "transfer_to_billing", "transfer_to_refunds"} <= set(asked)
    assert written(memory_service)[-1] == ("ASSISTANT", "refund sent")


async def test_a_handoff_objects_filter_callback_and_typed_input_are_the_sdks(
    harness: Harness,
) -> None:
    """``handoff(agent, ...)``: the tool is named and described as overridden, ``on_handoff``
    is given the model's typed input, the ``input_filter`` decides what the next agent reads,
    and a handoff that is not enabled is not offered; the run's own
    ``RunConfig.handoff_input_filter`` filters a handoff that has none."""
    seen: list[Any] = []

    class Reason(BaseModel):
        reason: str

    async def noted(context: Any, given: Reason) -> None:
        seen.append(given.reason)

    def nothing_before(data: Any) -> Any:
        seen.append("filtered")
        return data.clone(input_history=(), pre_handoff_items=(), new_items=())

    def runs_filter(data: Any) -> Any:
        seen.append("the run's filter")
        return data

    clerk_model = ScriptedModel(["clerk here"], ids="c")
    clerk = Agent(name="clerk", model=clerk_model)
    desk_model = Offered([("to_clerk", {"reason": "a refund"})], ids="d")
    desk = Agent(
        name="desk",
        model=desk_model,
        handoffs=[
            handoff(
                clerk,
                tool_name_override="to_clerk",
                tool_description_override="Hand a refund to the clerk.",
                on_handoff=noted,
                input_type=Reason,
                input_filter=nothing_before,
            ),
            handoff(Agent(name="closed"), is_enabled=False),
        ],
    )
    result = await harness.wrap(desk, id="filtered").run("refund o1", user="u")
    assert result.status is RunStatus.SUCCESS and result.answer == "clerk here"
    assert seen == ["a refund", "filtered"]
    assert desk_model.handoffs[0] == [("to_clerk", "Hand a refund to the clerk.")]
    assert clerk_model.inputs[0] == []  # the filter left the next agent nothing to read
    seen.clear()
    plain = Agent(
        name="front",
        model=ScriptedModel([("transfer_to_clerk", {})], ids="f"),
        handoffs=[Agent(name="clerk", model=ScriptedModel(["clerk again"], ids="c2"))],
    )
    config = RunConfig(handoff_input_filter=runs_filter)
    result = await harness.wrap(
        plain, id="run-filter", framework_options={"run_config": config}
    ).run("refund o2", user="u")
    assert result.answer == "clerk again" and seen == ["the run's filter"]


async def test_a_cancel_stops_a_call_under_way_in_the_agent_handed_off_to(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    started = asyncio.Event()

    @function_tool
    async def wait(order: str) -> str:
        started.set()
        await asyncio.sleep(30)
        return "late"

    clerk = Agent(
        name="clerk",
        model=ScriptedModel([("wait", {"order": "o1"}), "never"], ids="c"),
        tools=[wait],
    )
    desk = Agent(name="desk", model=ScriptedModel([("transfer_to_clerk", {})]), handoffs=[clerk])
    agent = declaring(memory_harness, wait="read").wrap(desk, id="cancelled")
    events: list[Any] = []

    async def consume() -> None:
        async for event in agent.stream("wait for o1", user="u"):
            events.append(event)

    task = asyncio.create_task(consume())
    await asyncio.wait_for(started.wait(), 10)
    await agent.cancel(events[0].run_id, reason="enough")
    await asyncio.wait_for(task, 10)
    assert events[-1].outcome is RunOutcome.CANCELLED and events[-1].data["reason"] == "enough"
    [cut] = [e.data for e in events if e.type is RunEventType.TOOL_CALL_RESULT]
    assert (cut["tool"], cut["status"]) == ("wait", bridge.CANCELLED)


# --------------------------------------------------------------------------- agents as tools
def checker_and_desk() -> tuple[ScriptedModel, ScriptedModel, Agent[Any]]:
    """A desk whose ``check`` tool is an agent with a tool behind the SDK's own approval."""
    inner = ScriptedModel([("gated", {"order": "o1"}), "checked o1"], ids="checker_")
    checker = Agent(name="checker", model=inner, tools=[gated])
    outer = ScriptedModel([("check", {"input": "check o1"}), "all good"], ids="desk_")
    return inner, outer, Agent(name="desk", model=outer, tools=[checker.as_tool("check", "Check.")])


@pytest.mark.parametrize("streamed", [False, True], ids=["run", "stream"])
async def test_an_sdk_approval_inside_an_agent_tool_continues_the_agent_inside(
    memory_harness: Harness, memory_service: FakeMemoryService, streamed: bool
) -> None:
    """An agent called as a tool whose own tool has the SDK's ``needs_approval``: the run
    pauses on that call, the agent tool's child run waiting with it; the approval continues
    the SDK's run inside the agent tool — the same child, the tool run once, no model asked
    again — and the calls are recorded once each."""
    inner, outer, desk = checker_and_desk()
    h = declaring(memory_harness, gated="read")
    agent = h.wrap(desk, id="inside")
    if streamed:
        events = [e async for e in agent.stream("check o1", user="u")]
        asked = next(e for e in events if e.type is RunEventType.INTERRUPT)
        interrupt_id, run_id = asked.data["interrupt_id"], events[0].run_id
        [waiting_call] = [e.data for e in events if e.type is RunEventType.TOOL_CALL_RESULT]
        assert (waiting_call["tool"], waiting_call["status"]) == ("check", bridge.PAUSED)
    else:
        paused = await agent.run("check o1", user="u")
        assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
        assert paused.interrupt.tool_call is not None
        assert paused.interrupt.tool_call.tool == "gated"
        assert paused.interrupt.tool_call.args == {"order": "o1"}
        interrupt_id, run_id = paused.interrupt.interrupt_id, paused.run_id
    [waiting] = await children(h, run_id)
    assert waiting.status is RunStatus.RUNNING and ran == []  # it waits with the run
    done = await agent.resume(interrupt_id, "approve", reviewer="r")
    assert done.status is RunStatus.SUCCESS and done.answer == "all good"
    assert ran == ["gated o1"] and (len(outer.inputs), len(inner.inputs)) == (2, 2)
    assert outputs(outer) == ["checked o1"]
    [child] = await children(h, run_id)
    assert child.run_id == waiting.run_id and child.status is RunStatus.SUCCESS
    assert child.output == "checked o1"
    await h.writes.drain()
    assert sorted(recorded(memory_service)) == [("check", "ok"), ("gated", "ok")]


async def test_a_rejected_sdk_approval_inside_an_agent_tool_is_what_the_agent_inside_reads(
    harness: Harness,
) -> None:
    inner, _, desk = checker_and_desk()
    agent = declaring(harness, gated="read").wrap(desk, id="inside-rejected")
    paused = await agent.run("check o1", user="u")
    assert paused.interrupt is not None
    done = await agent.resume(
        paused.interrupt.interrupt_id, "reject", answer="not today", reviewer="r"
    )
    assert done.status is RunStatus.SUCCESS and done.answer == "all good" and ran == []
    assert outputs(inner) == ["not today"]


async def test_an_sdk_approval_inside_an_agent_tool_resumes_in_another_process() -> None:
    """The SDK keeps an agent tool's paused run in its saved ``RunState``: another worker,
    which holds none of the first one's SDK state, continues it from the run's checkpoint."""
    inner, outer, _ = checker_and_desk()

    def build() -> Agent[Any]:
        checker = Agent(name="checker", model=inner, tools=[gated])
        return Agent(name="desk", model=outer, tools=[checker.as_tool("check", "Check.")])

    runs = LocalRuns()  # agent-runs, as two workers share it
    async with declaring(Harness(config=Settings(), runs=runs), gated="read") as first:
        paused = await first.wrap(build(), id="inside").run("check o1", user="u")
    assert paused.interrupt is not None
    # what the SDK holds in the first process of the agent tool's paused run: gone with it
    for held in (
        agent_tool_state._agent_tool_run_results_by_obj,
        agent_tool_state._agent_tool_run_results_by_signature,
        agent_tool_state._agent_tool_run_result_signature_by_obj,
        agent_tool_state._agent_tool_call_refs_by_obj,
    ):
        held.clear()
    async with declaring(Harness(config=Settings(), runs=runs), gated="read") as second:
        done = await second.wrap(build(), id="inside").resume(
            paused.interrupt.interrupt_id, "approve", reviewer="r"
        )
        [child] = await children(second, done.run_id)
    assert done.status is RunStatus.SUCCESS and done.answer == "all good"
    assert ran == ["gated o1"] and (len(outer.inputs), len(inner.inputs)) == (2, 2)
    assert child.status is RunStatus.SUCCESS


async def test_an_agent_tools_own_options_are_the_sdks(harness: Harness) -> None:
    """``Agent.as_tool``'s own: structured ``parameters`` are the tool's schema and the
    agent's input, ``custom_output_extractor`` makes the tool's result, ``on_stream`` gets the
    nested run's events — each call still a child run of the run."""
    streamed: list[str] = []

    class Trip(BaseModel):
        city: str
        days: int

    async def shouted(result: Any) -> str:
        return str(result.final_output).upper()

    async def on_stream(event: Any) -> None:
        streamed.append(type(event["event"]).__name__)

    planner_model = ScriptedModel(["two days in Oslo"], ids="p")
    planner = Agent(name="planner", model=planner_model)
    guide_model = ScriptedModel(["see the fjord"], ids="g")
    guide = Agent(name="guide", model=guide_model)
    model = ScriptedModel(
        [("plan", {"city": "Oslo", "days": 2}), ("guide", {"input": "what to see"}), "done"]
    )
    desk = Agent(
        name="desk",
        model=model,
        tools=[
            planner.as_tool(
                "plan", "Plan a trip.", parameters=Trip, custom_output_extractor=shouted
            ),
            guide.as_tool("guide", "Guide.", on_stream=on_stream),
        ],
    )
    result = await harness.wrap(desk, id="options").run("plan Oslo", user="u")
    assert result.status is RunStatus.SUCCESS
    assert outputs(model) == ["TWO DAYS IN OSLO", "see the fjord"]
    assert planner_model.inputs[0][-1]["content"] == '{"city": "Oslo", "days": 2}'
    assert "RawResponsesStreamEvent" in streamed
    kids = await children(harness, result.run_id)
    assert sorted(k.status for k in kids) == [RunStatus.SUCCESS] * 2


# --------------------------------------------------------------------------- sessions
async def test_the_runs_session_settings_and_callback_shape_what_memorys_thread_gives(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    """With memory keeping the conversation (the SDK's session is the memory service's
    thread): ``RunConfig.session_settings`` limits how many earlier items the SDK reads from
    it, and ``session_input_callback`` decides how they join the new input."""
    model = ScriptedModel(["one", "two", "three"])
    limited = RunConfig(session_settings=SessionSettings(limit=2))
    agent = memory_harness.wrap(
        Agent(name="desk", model=model), id="limited", framework_options={"run_config": limited}
    )
    for question in ("q1", "q2", "q3"):
        await agent.run(question, user="u", thread="t1")
    earlier = [(i["role"], i["content"]) for i in model.inputs[-1][:-1]]
    assert earlier == [("user", "q2"), ("assistant", "two")]  # the newest two, not all four

    def answers_only(history: list[Any], new: list[Any]) -> list[Any]:
        return [*(item for item in history if item.get("role") == "assistant"), *new]

    model = ScriptedModel(["one", "two", "three"])
    agent = memory_harness.wrap(
        Agent(name="desk", model=model),
        id="chosen",
        framework_options={"run_config": RunConfig(session_input_callback=answers_only)},
    )
    for question in ("q1", "q2", "q3"):
        await agent.run(question, user="u", thread="t2")
    earlier = [(i["role"], i["content"]) for i in model.inputs[-1][:-1]]
    assert earlier == [("assistant", "one"), ("assistant", "two")]


# --------------------------------------------------------------------------- the SDK's own tools
def shell_call(command: str) -> Any:
    from openai.types.responses import ResponseFunctionShellToolCall
    from openai.types.responses.response_function_shell_tool_call import Action

    return ResponseFunctionShellToolCall(
        id="sh_1",
        call_id="call_sh",
        status="completed",
        type="shell_call",
        action=Action(commands=[command], max_output_length=None, timeout_ms=None),
    )


async def shell(request: Any) -> str:
    ran.extend(request.data.action.commands)
    return "ran"


@pytest.mark.parametrize("decision", ["approve", "reject"])
async def test_a_shell_tools_own_approval_shows_the_person_the_command(
    harness: Harness, decision: str
) -> None:
    """The SDK's ``ShellTool(needs_approval=True)``: the run pauses on the SDK's approval, the
    person shown the call as the model made it (a shell call has no arguments of that name:
    its commands); approved it runs, rejected it does not."""
    model = ScriptedModel([Items(shell_call("rm -rf build")), "done"])
    target = Agent(name="ops", model=model, tools=[ShellTool(executor=shell, needs_approval=True)])
    agent = harness.wrap(target, id=f"shell-{decision}")
    paused = await agent.run("clean up", user="u")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None and ran == []
    asked = paused.interrupt.tool_call
    assert asked is not None and asked.tool == "shell"
    assert asked.args["action"]["commands"] == ["rm -rf build"]
    done = await agent.resume(paused.interrupt.interrupt_id, decision, reviewer="r")
    assert done.status is RunStatus.SUCCESS and done.answer == "done"
    assert ran == (["rm -rf build"] if decision == "approve" else [])


def test_what_a_person_is_shown_of_a_call_the_sdk_waits_on() -> None:
    """Its arguments; as written when they are not a JSON object; the call itself when it has
    none (the SDK's own tools: a shell's commands), whether the SDK holds it as an object or
    as the JSON its saved state restored."""

    class Waiting:
        def __init__(self, raw: Any, arguments: str | None) -> None:
            self.raw_item, self.arguments = raw, arguments

    assert _asked_about(Waiting(None, '{"order": "o1"}')) == {"order": "o1"}
    assert _asked_about(Waiting(None, "o1, please")) == {"arguments": "o1, please"}
    assert _asked_about(Waiting(None, "")) == {}
    restored = {"type": "shell_call", "action": {"commands": ["ls"]}}
    assert _asked_about(Waiting(restored, None)) == restored
    shown = _asked_about(Waiting(shell_call("ls"), None))
    assert shown["action"] == {"commands": ["ls"]} and shown["call_id"] == "call_sh"


async def test_today_a_tool_the_sdk_runs_itself_with_no_gate_is_not_governed_or_recorded(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    """Today: a shell tool with a local executor and no ``needs_approval`` (and the SDK's other
    tools that run code here: a local shell, a patch editor, a computer) is run by the SDK
    with no call through the harness: nobody is asked about a command that declares nothing,
    and no tool call is recorded. (Its own ``needs_approval`` is the gate that holds it.)"""
    model = ScriptedModel([Items(shell_call("rm -rf build")), "done"])
    target = Agent(name="ops", model=model, tools=[ShellTool(executor=shell)])
    events = [e async for e in memory_harness.wrap(target, id="shell").stream("clean", user="u")]
    await memory_harness.writes.drain()
    assert events[-1].outcome is RunOutcome.SUCCESS and ran == ["rm -rf build"]
    assert recorded(memory_service) == []
    assert not [e for e in events if e.type is RunEventType.TOOL_CALL_START]
    assert events[-1].data["spent"]["tool_calls"] == 0
