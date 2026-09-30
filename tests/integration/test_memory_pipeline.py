"""Memory in the pipeline: push (context as a system message), pull (the service's agent
tools), and the background records — against the in-process memory service."""

from __future__ import annotations

from typing import Any

from langchain.agents import create_agent

from tests.support.chat_model import ScriptedChatModel
from tests.support.memory import FakeMemoryService
from tests.support.models import ScriptedChat
from trellis import Harness, ReAct, Runtime, Settings, tool
from trellis.contracts import RunEventType, RunStatus
from trellis.harness.clients.memory import Memory
from trellis.memory.models import GroundingReport, ToolHints


@tool(side_effects="read")
def stock(sku: str) -> int:
    """Units in stock."""
    return 7


async def test_push_injects_the_context_as_a_system_message(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    model = ScriptedChat(["7 units"])
    agent = memory_harness.wrap(
        ReAct(system="You answer stock questions.", model=model), id="stock", memory="read"
    )
    result = await agent.run("how many a?", user="u1", thread="t1")
    assert result.answer == "7 units"
    system = model.requests[0]["messages"][0]
    assert system["role"] == "system"
    assert system["content"] == f"You answer stock questions.\n\n{memory_service.context_text}"
    [call] = memory_service.named("context")
    assert (
        call.scope["user_id"] == "u1"
        and call.scope["thread_id"] == "t1"
        and call.body["query"] == "how many a?"
    )
    await memory_harness.writes.drain()
    assert memory_service.named("message") == []  # read: nothing written


async def test_read_write_records_the_transcript_tools_and_outcome_once(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    model = ScriptedChat([("stock", {"sku": "a"}), "7 units"])
    agent = memory_harness.wrap(
        ReAct(system="s", model=model), id="stock", tools=[stock], memory="read_write"
    )
    await agent.run("how many a?", user="u1")
    await memory_harness.writes.drain()
    messages = [(c.body["role"], c.body["content"]) for c in memory_service.named("message")]
    assert messages == [("USER", "how many a?"), ("ASSISTANT", "7 units")]
    [recorded] = memory_service.named("record_tool")
    assert recorded.body["tool"] == "stock" and recorded.body["task"] == "how many a?"
    [outcome] = memory_service.named("outcome")
    assert outcome.body["success"] is True and outcome.path["run"] == outcome.scope["agent_run_id"]


async def test_pull_adds_the_memory_tools_and_they_call_the_service(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    model = ScriptedChat([("memory_search", {"query": "preferences"}), "email"])
    agent = memory_harness.wrap(ReAct(system="s", model=model), id="prefs", memory="read_write")
    result = await agent.run("how do I like to be contacted?", user="u1")
    assert result.answer == "email"
    offered = [t["function"]["name"] for t in model.requests[0]["tools"]]
    assert offered == ["memory_search", "memory_remember"]
    [call] = memory_service.named("call_agent_tool")
    assert call.path["name"] == "memory_search" and call.body["args"] == {"query": "preferences"}
    await memory_harness.writes.drain()
    assert memory_service.named("record_tool") == []  # the service logs its own tools


async def test_a_reader_gets_only_the_read_only_tools(memory_harness: Harness) -> None:
    model = ScriptedChat(["ok"])
    await memory_harness.wrap(ReAct(system="s", model=model), id="r", memory="read").run(
        "x", user="u"
    )
    assert [t["function"]["name"] for t in model.requests[0]["tools"]] == ["memory_search"]


async def test_tool_hints_ask_for_the_tools_section(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    async def fn(input: str, agent: Runtime) -> Any:
        return await agent.tools.hints("reorder")

    agent = memory_harness.wrap(fn, id="h", tools=[stock], memory="read", tool_hints=True)
    result = await agent.run("reorder a", user="u")
    # the candidates are the run's own tools, never the memory service's pull tools
    assert memory_service.named("context")[0].body["tools"]["available"] == ["stock"]
    assert isinstance(result.answer, ToolHints) and result.answer.candidates[0].name == "stock"
    assert memory_service.named("tool_hints")[0].body["available"] == ["stock"]


async def test_the_tool_search_pull_tool_looks_among_the_runs_tools(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    async def fn(input: str, agent: Runtime) -> Any:
        return await agent.tools.call("tool_search", task="reorder")

    memory_harness.memory.listed = None  # type: ignore[union-attr]
    memory_service.agent_tools.append(
        {"name": "tool_search", "description": "Tools for a task.", "input_schema": {}}
    )
    result = await memory_harness.wrap(fn, id="h", tools=[stock], memory="read").run("x", user="u")
    assert result.answer["candidates"][0]["name"] == "stock"
    assert memory_service.named("tool_hints")[0].body["available"] == ["stock"]
    assert memory_service.named("call_agent_tool") == []


async def test_a_reading_agent_cannot_call_a_memory_tool_that_writes(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    # tools built before the agent existed include the writing ones; the run refuses them
    tools = await memory_harness.tools(framework="langgraph", memory=True)
    model = ScriptedChatModel(turns=[("memory_remember", {"content": "x"}), "could not"])
    graph = create_agent(model, tools=tools)
    result = await memory_harness.wrap(graph, id="r", memory="read").run("x", user="u")
    assert result.answer == "could not"
    assert "only reads" in str(model.seen[1][-1].content)
    assert memory_service.named("call_agent_tool") == []


async def test_an_outcome_the_agent_recorded_is_not_overwritten(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    memory_service.agent_tools.append(
        {"name": "record_outcome", "description": "Say how it went.", "input_schema": {}}
    )
    memory_harness.memory.listed = None  # type: ignore[union-attr]

    async def fn(input: str, agent: Runtime) -> str:
        await agent.tools.call("record_outcome", success=False, note="wrong warehouse")
        return "done"

    result = await memory_harness.wrap(fn, id="o", memory="read_write").run("x", user="u")
    assert result.status is RunStatus.SUCCESS
    await memory_harness.writes.drain()
    assert memory_service.named("outcome") == []  # the agent's own verdict stands


async def test_the_transcript_is_recorded_on_a_pause_and_on_a_failure(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    async def fn(input: str, agent: Runtime) -> str:
        if input == "fail":
            raise RuntimeError("boom")
        return str(await agent.ask("Sure?"))

    agent = memory_harness.wrap(fn, id="t", memory="read_write")
    paused = await agent.run("ask", user="u")
    assert paused.interrupt is not None
    await agent.resume(paused.interrupt.interrupt_id, "answer", answer="yes", reviewer="u")
    failed = await agent.run("fail", user="u")
    assert failed.status is RunStatus.ERROR
    await memory_harness.writes.drain()
    sent = [
        (c.body["role"], c.body["content"], c.idempotency_key)
        for c in memory_service.named("message")
    ]
    run = paused.run_id
    assert sent[:3] == [
        ("USER", "ask", f"{run}:user:0"),  # on the pause
        ("USER", "ask", f"{run}:user:0"),  # again on the resumed attempt: stored once
        ("ASSISTANT", "yes", f"{run}:2:msg:1"),
    ]
    assert sent[3][:2] == ("USER", "fail")  # a failed run's question is kept too
    # a run with no thread is its own thread
    assert {c.scope["thread_id"] for c in memory_service.named("message")} == {
        paused.run_id,
        failed.run_id,
    }


async def test_a_memory_outage_is_a_warning_not_a_failure(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    memory_service.fail = {"context", "message"}

    async def fn(input: str, agent: Runtime) -> str:
        return "fine"

    agent = memory_harness.wrap(fn, id="o", memory="read_write")
    events = [e async for e in agent.stream("x", user="u")]
    warnings = [
        e.data for e in events if e.type is RunEventType.CUSTOM and e.data["name"] == "warning"
    ]
    assert warnings[0]["code"] == "memory_unavailable"
    assert events[-1].type is RunEventType.RUN_FINISHED
    await memory_harness.writes.drain()
    assert memory_harness.writes.failed == 1  # the transcript write, reported and counted


async def test_the_model_key_is_registered_once_per_agent(
    memory_service: FakeMemoryService,
) -> None:
    async with Harness(config=Settings(memory_url="http://m", memory_model_key="sk-mem")) as h:
        h.memory = Memory("http://m", None, client=memory_service.client())

        async def fn(input: str, agent: Runtime) -> str:
            return "ok"

        agent = h.wrap(fn, id="keyed", memory="read")
        await agent.run("a", user="u")
        await agent.run("b", user="u")
        await h.writes.drain()
    [key] = memory_service.named("model_key")
    assert key.body["virtual_key"] == "sk-mem" and key.idempotency_key == "model-key:keyed"
    assert key.scope["agent_id"] == "keyed"


async def test_the_sampled_judge_scores_against_the_context_and_files_feedback(
    memory_service: FakeMemoryService,
) -> None:
    memory_service.report = GroundingReport(supported=2)
    async with Harness(config=Settings(memory_url="http://m", eval_sample=1.0)) as h:
        h.memory = Memory("http://m", None, client=memory_service.client())

        async def fn(input: str, agent: Runtime) -> str:
            return "you prefer email"

        result = await h.wrap(fn, id="judged", memory="read_write").run("contact?", user="u")
        await h.writes.drain()
    [verified] = memory_service.named("verify")
    assert verified.body["answer"] == "you prefer email" and verified.body["items"] == []
    [feedback] = memory_service.named("feedback")
    assert feedback.body["target_kind"] == "answer" and feedback.body["score"] == 1.0
    assert feedback.body["source"] == "judge" and feedback.body["agent_run_id"] == result.run_id


async def test_an_approval_decision_is_feedback_on_the_tool_call(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    @tool(side_effects="irreversible")
    def wipe(disk: str) -> str:
        """Wipe a disk."""
        return "wiped"

    async def fn(input: str, agent: Runtime) -> Any:
        return await agent.tools.call("wipe", disk="d1")

    agent = memory_harness.wrap(fn, id="ops", tools=[wipe], memory="read_write")
    paused = await agent.run("wipe d1", user="u")
    assert paused.interrupt is not None
    await agent.resume(paused.interrupt.interrupt_id, "reject", reviewer="boss")
    await memory_harness.writes.drain()
    [feedback] = memory_service.named("feedback")
    assert feedback.body["target_kind"] == "tool_call" and feedback.body["reviewer"] == "boss"
    assert feedback.body["verdict"] == "reject"
    # what approval patterns are learned from: the tool and the arguments it was asked about
    assert feedback.body["metadata"]["tool"] == "wipe"
    assert feedback.body["metadata"]["args"] == {"disk": "d1"}


async def test_explicit_feedback_on_a_run(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    async def fn(input: str, agent: Runtime) -> str:
        return "12"

    result = await memory_harness.wrap(fn, id="f").run("stock?", user="u")
    feedback = await memory_harness.feedback(result.run_id, "correct", "13")
    assert feedback.correction == "13" and feedback.reviewer == "u"
    [sent] = memory_service.named("feedback")
    assert sent.body["feedback_id"] == feedback.feedback_id and sent.body["verdict"] == "correct"


async def test_local_tool_side_effects_are_published_to_the_catalog(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    async def fn(input: str, agent: Runtime) -> str:
        return "ok"

    await memory_harness.wrap(fn, id="c", tools=[stock]).run("x", user="u")
    await memory_harness.writes.drain()
    [put] = memory_service.named("put_catalog")
    assert put.scope["tenant_id"] == "default" and "user_id" not in put.scope
    assert (
        put.body["tools"][0]["name"] == "stock" and put.body["tools"][0]["side_effects"] == "read"
    )


async def test_status_of_a_run_with_memory_is_unchanged_by_it(memory_harness: Harness) -> None:
    async def fn(input: str, agent: Runtime) -> str:
        assert agent.context is not None
        remembered = await agent.memory.call_agent_tool("memory_search", {"query": input})
        return str(remembered)

    result = await memory_harness.wrap(fn, id="direct", memory="read").run("x", user="u")
    assert result.status is RunStatus.SUCCESS and "memory_search ok" in result.answer
