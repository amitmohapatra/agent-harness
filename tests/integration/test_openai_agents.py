"""OpenAI Agents SDK: the real ``Runner`` with a scripted ``Model``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from agents import Agent, SQLiteSession, Usage, function_tool

from tests.support.catalog import FakeCatalog
from tests.support.memory import MEMORY_TOOLS, FakeMemoryService
from tests.support.openai_model import ScriptedModel
from trellis import Budget, Deny, Harness, Hooks, Rewrite, Settings, current, tool
from trellis.contracts import InterruptReason, RunEventType, RunStatus, ToolCall
from trellis.harness import budget
from trellis.harness.events import WARNING
from trellis.harness.governance import ApprovalRuleError, Governance
from trellis.harness.governance.catalog import Rule
from trellis.harness.journal import content_key

done: list[str] = []


@tool(side_effects="irreversible")
def ship(order: str) -> str:
    """Ship an order."""
    done.append(order)
    return f"shipped {order}"


@tool(side_effects="read")
async def confirm(order: str) -> Any:
    """Ask the customer to confirm an order."""
    runtime = current()
    assert runtime is not None
    return await runtime.ask(f"Ship {order}?", options=["yes", "no"])


@pytest.fixture(autouse=True)
def _reset() -> None:
    done.clear()


async def test_the_agent_gets_the_harness_tools_next_to_its_own(harness: Harness) -> None:
    @function_tool
    def eta(order: str) -> str:
        return "tomorrow"

    model = ScriptedModel(
        [("eta", {"order": "o1"}), ("ship", {"order": "o1"}), "shipped, arriving tomorrow"]
    )
    target = Agent(name="shipper", instructions="You ship orders.", model=model, tools=[eta])
    agent = harness.wrap(target, id="shipper", tools=[tool(ship.fn, side_effects="write")])
    result = await agent.run("ship o1", user="u1")
    assert result.status is RunStatus.SUCCESS and result.answer == "shipped, arriving tomorrow"
    assert done == ["o1"]
    assert [t.name for t in target.tools] == ["eta"]  # the team's agent is untouched


async def test_an_approval_pauses_and_a_resume_reruns_against_the_journal(harness: Harness) -> None:
    model = ScriptedModel([("ship", {"order": "o1"}), ("ship", {"order": "o1"}), "shipped"])
    agent = harness.wrap(Agent(name="shipper", model=model), id="shipper", tools=[ship])
    paused = await agent.run("ship o1", user="u1")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
    assert paused.interrupt.reason is InterruptReason.APPROVAL and done == []
    finished = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="u1")
    assert finished.status is RunStatus.SUCCESS and done == ["o1"]


async def test_ask_inside_a_tool_returns_the_answer_on_resume(harness: Harness) -> None:
    model = ScriptedModel(
        [("confirm", {"order": "o1"}), ("confirm", {"order": "o1"}), "customer said yes"]
    )
    agent = harness.wrap(Agent(name="c", model=model), id="confirmer", tools=[confirm])
    paused = await agent.run("check o1", user="u1")
    assert paused.interrupt is not None and paused.interrupt.options == ["yes", "no"]
    finished = await agent.resume(
        paused.interrupt.interrupt_id, "answer", answer="yes", reviewer="u1"
    )
    assert finished.answer == "customer said yes"
    tool_output = model.inputs[-1][-1]
    assert tool_output["type"] == "function_call_output" and tool_output["output"] == "yes"


async def test_the_sdks_own_approval_pauses_and_resumes_its_run_state(harness: Harness) -> None:
    shipped: list[str] = []

    @function_tool(needs_approval=True)
    def send(order: str) -> str:
        shipped.append(order)
        return "sent"

    model = ScriptedModel([("send", {"order": "o9"}), "sent it"])
    agent = harness.wrap(Agent(name="s", model=model, tools=[send]), id="sender")
    paused = await agent.run("send o9", user="u1")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
    assert paused.interrupt.tool_call is not None and paused.interrupt.tool_call.args == {
        "order": "o9"
    }
    finished = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="u1")
    assert finished.status is RunStatus.SUCCESS and finished.answer == "sent it"
    assert shipped == ["o9"]
    assert len(model.inputs) == 2  # continued from the SDK's state, not re-run


async def test_a_rejected_sdk_approval_does_not_run_the_tool(harness: Harness) -> None:
    @function_tool(needs_approval=True)
    def send(order: str) -> str:
        raise AssertionError("must not run")

    model = ScriptedModel([("send", {"order": "o9"}), "not sent"])
    agent = harness.wrap(Agent(name="s", model=model, tools=[send]), id="sender")
    paused = await agent.run("send o9", user="u1")
    assert paused.interrupt is not None
    finished = await agent.resume(paused.interrupt.interrupt_id, "reject", reviewer="u1")
    assert finished.answer == "not sent"


async def test_streaming_emits_text_deltas(harness: Harness) -> None:
    agent = harness.wrap(Agent(name="s", model=ScriptedModel(["all good here"])), id="streamer")
    events = [e async for e in agent.stream("hi", user="u1")]
    deltas = [e.data["delta"] for e in events if e.type is RunEventType.TEXT_MESSAGE_CONTENT]
    assert "".join(deltas) == "all good here" and len(deltas) == 3


async def test_the_context_is_a_system_message_and_memory_tools_are_added(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    model = ScriptedModel([("memory_search", {"query": "prefs"}), "email"])
    agent = memory_harness.wrap(Agent(name="m", model=model), id="m")
    assert (await agent.run("how to reach me?", user="u1")).answer == "email"
    assert model.inputs[0][0] == {"role": "system", "content": memory_service.context_text}
    assert memory_service.named("call_agent_tool")[0].path["name"] == "memory_search"


async def test_the_hints_narrow_the_tools_each_turn_and_the_teams_own_stay(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    @function_tool
    def eta(order: str) -> str:
        return "tomorrow"

    def make(i: int) -> Any:
        def lookup(key: str) -> str:
            return f"t{i}"

        return tool(lookup, name=f"t{i}", side_effects="read")

    memory_service.candidates = ["t2"]
    memory_service.candidates_for = {"find t4": ["t4"]}
    model = ScriptedModel([("tool_search", {"task": "find t4"}), ("t4", {"key": "k"}), "done"])
    target = Agent(name="n", model=model, tools=[eta])
    agent = memory_harness.wrap(target, id="n", tools=[make(i) for i in range(6)])
    assert (await agent.run("find t2", user="u1")).answer == "done"
    memory_tools = MEMORY_TOOLS
    assert model.tools[0] == ["eta", "t2", *memory_tools]
    assert model.tools[1] == ["eta", "t2", "t4", *memory_tools]  # tool_search offered t4


async def test_two_sdk_approvals_in_one_turn_are_answered_one_at_a_time(
    harness: Harness,
) -> None:
    shipped: list[str] = []

    @function_tool(needs_approval=True)
    def send(order: str) -> str:
        shipped.append(order)
        return f"sent {order}"

    model = ScriptedModel([[("send", {"order": "a"}), ("send", {"order": "b"})], "both sent"])
    agent = harness.wrap(Agent(name="s", model=model, tools=[send]), id="sender")
    first = await agent.run("send a and b", user="u1")
    assert first.interrupt is not None and first.interrupt.tool_call is not None
    assert first.interrupt.tool_call.args == {"order": "a"}
    # approving the first leaves the second waiting: the run pauses on it next
    second = await agent.resume(first.interrupt.interrupt_id, "approve", reviewer="u1")
    assert second.status is RunStatus.PAUSED and second.interrupt is not None
    assert second.interrupt.tool_call is not None
    assert second.interrupt.tool_call.args == {"order": "b"}
    assert shipped == ["a"]
    done = await agent.resume(second.interrupt.interrupt_id, "approve", reviewer="u1")
    assert done.status is RunStatus.SUCCESS and done.answer == "both sent"
    assert shipped == ["a", "b"]


async def test_a_rejected_sdk_approval_tells_the_model_the_reason(harness: Harness) -> None:
    @function_tool(needs_approval=True)
    def send(order: str) -> str:
        raise AssertionError("must not run")

    model = ScriptedModel([("send", {"order": "o9"}), "not sent"])
    agent = harness.wrap(Agent(name="s", model=model, tools=[send]), id="sender")
    paused = await agent.run("send o9", user="u1")
    assert paused.interrupt is not None
    await agent.resume(paused.interrupt.interrupt_id, "reject", answer="on hold", reviewer="u1")
    output = model.inputs[-1][-1]
    assert output["type"] == "function_call_output" and output["output"] == "on hold"


async def test_an_answered_sdk_approval_is_what_the_model_reads_instead(harness: Harness) -> None:
    @function_tool(needs_approval=True)
    def lookup(order: str) -> str:
        raise AssertionError("must not run")

    model = ScriptedModel([("lookup", {"order": "o9"}), "it ships monday"])
    agent = harness.wrap(Agent(name="s", model=model, tools=[lookup]), id="looker")
    paused = await agent.run("when does o9 ship?", user="u1")
    assert paused.interrupt is not None
    done = await agent.resume(
        paused.interrupt.interrupt_id, "answer", answer="ships monday", reviewer="u1"
    )
    assert done.answer == "it ships monday"
    assert model.inputs[-1][-1]["output"] == "ships monday"


async def test_an_edited_sdk_approval_runs_the_edited_call_once(harness: Harness) -> None:
    shipped: list[str] = []

    @function_tool(needs_approval=True)
    def send(order: str) -> str:
        shipped.append(order)
        return f"sent {order}"

    # the model reads the reviewer's arguments and calls again with them: that call runs
    # without asking again
    model = ScriptedModel([("send", {"order": "o9"}), ("send", {"order": "o10"}), "sent o10"])
    agent = harness.wrap(Agent(name="s", model=model, tools=[send]), id="sender")
    paused = await agent.run("send o9", user="u1")
    assert paused.interrupt is not None
    done = await agent.resume(
        paused.interrupt.interrupt_id, "edit", answer={"order": "o10"}, reviewer="u1"
    )
    assert done.status is RunStatus.SUCCESS and done.answer == "sent o10"
    assert shipped == ["o10"]
    told = model.inputs[1][-1]["output"]
    assert '{"order": "o10"}' in told and "send" in told


async def test_an_edited_call_the_model_changes_again_asks_again(harness: Harness) -> None:
    shipped: list[str] = []

    @function_tool(needs_approval=True)
    def send(order: str) -> str:
        shipped.append(order)
        return f"sent {order}"

    model = ScriptedModel([("send", {"order": "o9"}), ("send", {"order": "o11"}), "sent"])
    agent = harness.wrap(Agent(name="s", model=model, tools=[send]), id="sender")
    paused = await agent.run("send o9", user="u1")
    assert paused.interrupt is not None
    again = await agent.resume(
        paused.interrupt.interrupt_id, "edit", answer={"order": "o10"}, reviewer="u1"
    )
    assert again.status is RunStatus.PAUSED and again.interrupt is not None
    assert again.interrupt.tool_call is not None
    assert again.interrupt.tool_call.args == {"order": "o11"} and shipped == []


async def test_an_edited_sdk_approval_streams_the_continued_run(harness: Harness) -> None:
    shipped: list[str] = []

    @function_tool(needs_approval=True)
    def send(order: str) -> str:
        shipped.append(order)
        return f"sent {order}"

    model = ScriptedModel([("send", {"order": "o9"}), ("send", {"order": "o10"}), "sent o10"])
    agent = harness.wrap(Agent(name="s", model=model, tools=[send]), id="sender")
    paused = [e async for e in agent.stream("send o9", user="u1")][-1]
    interrupt = paused.data["interrupt"]
    record, resolution = await agent._resolution(
        interrupt["interrupt_id"], "edit", {"order": "o10"}, "u1", tenant="default"
    )
    events: list[Any] = []
    result = await agent._continue(record, resolution, listener=events.append)
    assert result.answer == "sent o10" and shipped == ["o10"]
    deltas = [e.data["delta"] for e in events if e.type is RunEventType.TEXT_MESSAGE_CONTENT]
    assert "".join(deltas) == "sent o10"


#: a result over ``tools.results.MAX_CHARS``, and where the run keeps it
LONG = "\n".join(f"row {n}" for n in range(1, 30_001))
KEPT = f"/large_tool_results/{content_key('result', LONG)}"


@tool(side_effects="read")
def dump() -> str:
    """Dump every row."""
    return LONG


def outputs(model: ScriptedModel) -> list[str]:
    """What each call's tool results read, by the model's turn."""
    return [str(item["output"]) for item in model.inputs[-1] if "output" in item]


async def test_a_large_result_is_previewed_and_read_in_pages_across_a_pause(
    harness: Harness,
) -> None:
    @function_tool(needs_approval=True)
    def send(order: str) -> str:
        return "sent"

    model = ScriptedModel(
        [
            ("dump", {}),
            ("send", {"order": "o1"}),
            ("read_file", {"file_path": KEPT, "offset": 10, "limit": 2}),
            "done",
        ]
    )
    agent = harness.wrap(Agent(name="d", model=model, tools=[send]), id="dumper", tools=[dump])
    paused = await agent.run("dump it", user="u1")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
    preview = model.inputs[1][-1]["output"]
    assert f"saved in the filesystem at this path: {KEPT}" in preview and len(preview) < 25_000
    assert "row 30000" in preview and "row 100" not in preview
    assert "read_file" not in model.tools[0] and "read_file" in model.tools[1]
    finished = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="u1")
    assert finished.answer == "done"  # the kept result came back with the run's journal
    assert outputs(model)[-1] == (
        f'11  row 11\n12  row 12\n…[29988 more lines: read_file(file_path="{KEPT}", offset=12)]'
    )


async def test_an_agent_built_with_h_tools_gets_read_file_once(harness: Harness) -> None:
    model = ScriptedModel([("dump", {}), "done"])
    target = Agent(
        name="d", model=model, tools=await harness.tools(dump, framework="openai_agents")
    )
    agent = harness.wrap(target, id="dumper", tools=[ship])
    assert (await agent.run("dump it", user="u1")).answer == "done"
    assert model.tools[1].count("read_file") == 1


async def test_a_session_keeps_the_thread_so_memorys_window_is_not_sent(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    model = ScriptedModel(["one", "two"])
    agent = memory_harness.wrap(Agent(name="s", model=model), id="sess")
    in_ram = SQLiteSession("s1")  # kept in RAM only: memory's window keeps the history
    await agent.run("hi", user="u1", framework_options={"session": in_ram})
    # given none, the run's session is memory's thread, which owns the history the same way
    await agent.run("hi", user="u1")
    windows = [c.body.get("window") for c in memory_service.named("context")]
    assert windows[-2:] == [True, False]


def test_the_sdks_own_tracing_is_left_as_set_and_said_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from agents.tracing import get_trace_provider

    from trellis.harness.adapters.openai_agents import sdk_tracing_said

    provider = get_trace_provider()
    was = getattr(provider, "_disabled", False)
    try:
        provider.set_disabled(False)
        sdk_tracing_said.cache_clear()
        with caplog.at_level("INFO", logger="trellis.harness"):
            assert sdk_tracing_said() is True
        assert "exports to OpenAI" in caplog.text
        provider.set_disabled(True)
        sdk_tracing_said.cache_clear()
        assert sdk_tracing_said() is False
    finally:
        provider.set_disabled(was)
        sdk_tracing_said.cache_clear()


# ---------------------------------------------------------------------- handoffs, budget, history
def _desk(model: ScriptedModel) -> tuple[Agent[Any], Agent[Any], Any]:
    """A triage agent handing off to a billing specialist whose own tool refunds."""

    @function_tool
    def refund(order: str) -> str:
        """Refund an order."""
        done.append(f"refund {order}")
        return f"refunded {order}"

    billing = Agent(name="billing", model=model, tools=[refund])
    triage = Agent(name="triage", model=model, handoffs=[billing])
    billing.handoffs = [triage]  # a handoff that leads back
    return triage, billing, refund


class DenyAll(Hooks):
    async def before_tool(self, call: ToolCall) -> Deny | Rewrite | None:
        return Deny("nothing runs here")


async def test_a_deny_all_rule_blocks_the_tool_of_an_agent_handed_off_to(
    harness: Harness,
) -> None:
    model = ScriptedModel([("transfer_to_billing", {}), ("refund", {"order": "o1"}), "gave up"])
    triage, billing, refund = _desk(model)
    agent = harness.wrap(triage, id="desk", hooks=[DenyAll()])
    result = await agent.run("refund o1", user="u1")
    assert result.answer == "gave up" and done == []  # the specialist's tool did not run
    assert billing.tools == [refund] and triage.handoffs == [billing]  # yours, unchanged


async def test_an_agent_handed_off_to_has_its_own_tools_governed_by_the_catalog() -> None:
    model = ScriptedModel([*[("transfer_to_billing", {}), ("refund", {"order": "o1"})] * 2, "ok"])
    triage, _, _ = _desk(model)
    governance = Governance(FakeCatalog({"refund": Rule(risk="irreversible")}))
    async with Harness(config=Settings(), governance=governance) as h:
        agent = h.wrap(triage, id="desk")
        paused = await agent.run("refund o1", user="u1")
        assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
        assert paused.interrupt.question.startswith("Approve refund?") and done == []
        finished = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="r")
    assert finished.answer == "ok" and done == ["refund o1"]  # once: the journal kept it


async def test_a_handoff_object_is_said_once_and_kept(
    harness: Harness, caplog: pytest.LogCaptureFixture
) -> None:
    from agents import handoff

    model = ScriptedModel(["hi", "hi"])
    target = Agent(name="t", model=model, handoffs=[handoff(Agent(name="x", model=model))])
    agent = harness.wrap(target, id="hidden")
    with caplog.at_level("WARNING", logger="trellis.harness"):
        await agent.run("hi", user="u1")
        await agent.run("hi", user="u1")
    said = [r for r in caplog.records if "Handoff object" in r.getMessage()]
    assert len(said) == 1


@tool(side_effects="read")
def track(order: str) -> str:
    """Track an order."""
    done.append(f"track {order}")
    return "on its way"


async def test_a_token_budget_stops_the_run_without_any_hooks() -> None:
    model = ScriptedModel(
        [("track", {"order": "o1"}), "done"], usage=Usage(input_tokens=50, output_tokens=10)
    )
    async with Harness(config=Settings()) as h:
        agent = h.wrap(
            Agent(name="s", model=model), id="s", tools=[track], budget=Budget(max_tokens=10)
        )
        result = await agent.run("track o1", user="u1")
    assert result.status is not RunStatus.SUCCESS and done == []  # the reply spent it
    assert result.error is not None and result.error.code == budget.BUDGET_EXCEEDED


@pytest.mark.parametrize(
    ("options", "window"),
    [
        ({"session": SQLiteSession("ram")}, True),  # kept in RAM only: memory's window
        ({"conversation_id": "conv_1"}, False),  # kept on OpenAI's servers
        ({"previous_response_id": "resp_1"}, False),
    ],
    ids=["sqlite_in_memory", "conversation_id", "previous_response_id"],
)
async def test_who_keeps_the_thread_decides_memorys_window(
    memory_harness: Harness,
    memory_service: FakeMemoryService,
    options: dict[str, Any],
    window: bool,
) -> None:
    agent = memory_harness.wrap(Agent(name="s", model=ScriptedModel(["one"])), id="thread")
    await agent.run("hi", user="u1", framework_options=options)
    assert [c.body.get("window") for c in memory_service.named("context")][-1] is window


async def test_a_durable_session_keeps_the_thread(
    memory_harness: Harness, memory_service: FakeMemoryService, tmp_path: Path
) -> None:
    own = SQLiteSession("own", tmp_path / "sessions.db")
    agent = memory_harness.wrap(Agent(name="s", model=ScriptedModel(["one"])), id="durable")
    await agent.run("hi", user="u1", framework_options={"session": own})
    assert memory_service.named("context")[-1].body.get("window") is False
    own.close()


async def test_a_rule_that_cannot_be_evaluated_asks_and_is_recorded_on_the_run() -> None:
    model = ScriptedModel([("track", {"order": "o1"}), "done"])
    governance = Governance(FakeCatalog({"track": Rule(risk="read", approve_when="order >>")}))
    async with Harness(config=Settings(), governance=governance) as h:
        agent = h.wrap(Agent(name="s", model=model), id="rule", tools=[track])
        events = [e async for e in agent.stream("track o1", user="u1")]
    assert done == []  # it asked a person instead of running
    assert any(e.type is RunEventType.INTERRUPT for e in events)
    codes = [e.data.get("code") for e in events if e.data.get("name") == WARNING]
    assert codes == [ApprovalRuleError.code]


async def test_the_providers_hosted_tool_calls_are_recorded_with_the_steps_duration(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    """File search, web search, the code interpreter, image generation and a hosted MCP
    server's tool run at the provider, inside the model's step: each call a reply carries is
    recorded as the run's — what it was asked, what it found (never an image's bytes), whether
    it failed, the step's duration — on the stream and in the memory service."""
    from openai.types.responses import (
        ResponseCodeInterpreterToolCall,
        ResponseFileSearchToolCall,
        ResponseFunctionWebSearch,
    )
    from openai.types.responses.mcp_tool_call_error import McpToolExecutionError
    from openai.types.responses.response_file_search_tool_call import Result
    from openai.types.responses.response_function_web_search import ActionSearch
    from openai.types.responses.response_output_item import ImageGenerationCall, McpCall

    from tests.support.openai_model import Items

    hosted = Items(
        ResponseFileSearchToolCall(
            id="fs_1",
            type="file_search_call",
            status="completed",
            queries=["returns policy"],
            results=[Result(filename="policy.pdf", score=0.9)],
        ),
        ResponseFunctionWebSearch(
            id="ws_1",
            type="web_search_call",
            status="completed",
            action=ActionSearch(type="search", query="carrier delays"),
        ),
        ResponseCodeInterpreterToolCall(
            id="ci_1",
            type="code_interpreter_call",
            status="completed",
            code="print(2 + 2)",
            container_id="c1",
            outputs=[],
        ),
        ImageGenerationCall(
            id="ig_1", type="image_generation_call", status="completed", result="aGVsbG8="
        ),
        McpCall(
            id="mcp_1",
            type="mcp_call",
            name="find_order",
            server_label="crm",
            arguments='{"order": "o3"}',
            error=McpToolExecutionError(type="mcp_tool_execution_error", content="refused"),
        ),
    )
    model = ScriptedModel([hosted, "Here is what I found."])
    agent = memory_harness.wrap(Agent(name="research", model=model), id="research")
    events = [e async for e in agent.stream("look it up", user="u1")]
    await memory_harness.writes.drain()
    recorded = {c.body["tool"]: c.body for c in memory_service.calls if c.name == "record_tool"}
    assert set(recorded) == {
        "file_search",
        "web_search",
        "code_interpreter",
        "image_generation",
        "find_order",
    }
    assert recorded["file_search"]["args"] == {"queries": ["returns policy"]}
    assert recorded["web_search"]["args"]["query"] == "carrier delays"
    assert recorded["code_interpreter"]["args"] == {"code": "print(2 + 2)"}
    assert "aGVsbG8=" not in json.dumps(recorded["image_generation"])  # never the bytes
    assert recorded["find_order"]["status"] == "error"
    assert recorded["find_order"]["output"]["content"] == "refused"
    assert recorded["find_order"]["args"] == {"order": "o3"}
    assert all(r["latency_ms"] is not None for r in recorded.values())
    results = [e.data["tool"] for e in events if e.type is RunEventType.TOOL_CALL_RESULT]
    assert sorted(results) == sorted(recorded)
