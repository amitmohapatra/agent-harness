"""Memory in the pipeline: push (context as a system message), pull (the service's agent
tools), and the background records — against the in-process memory service."""

from __future__ import annotations

from typing import Any

from trellis.contracts import FeedbackTargetKind, RunEventType, RunStatus

from tests.support.memory import FakeMemoryService, Report
from tests.support.models import ScriptedChat
from trellis import Harness, ReAct, Runtime, Settings, tool
from trellis.harness.clients.memory import Memory


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
    [(scope, payload)] = memory_service.named("context")
    assert (
        scope["user_id"] == "u1"
        and scope["thread_id"] == "t1"
        and payload["query"] == "how many a?"
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
    messages = [(p["role"], p["content"]) for _, p in memory_service.named("message")]
    assert messages == [("user", "how many a?"), ("assistant", "7 units")]
    [(_, recorded)] = memory_service.named("record_tool")
    assert recorded["tool"] == "stock" and recorded["task"] == "how many a?"
    assert [p for _, p in memory_service.named("outcome")] == [{"success": True, "note": None}]


async def test_pull_adds_the_memory_tools_and_they_call_the_service(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    model = ScriptedChat([("memory_search", {"query": "preferences"}), "email"])
    agent = memory_harness.wrap(ReAct(system="s", model=model), id="prefs", memory="read_write")
    result = await agent.run("how do I like to be contacted?", user="u1")
    assert result.answer == "email"
    offered = [t["function"]["name"] for t in model.requests[0]["tools"]]
    assert offered == ["memory_search", "memory_remember"]
    [(_, call)] = memory_service.named("call_agent_tool")
    assert call == {"name": "memory_search", "args": {"query": "preferences"}}
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
    assert memory_service.named("context")[0][1]["tools"]["available"] == ["stock", "memory_search"]
    assert result.answer == {"candidates": ["stock"]}


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
        h.memory = Memory("http://m", None, client=memory_service)

        async def fn(input: str, agent: Runtime) -> str:
            return "ok"

        agent = h.wrap(fn, id="keyed", memory="read")
        await agent.run("a", user="u")
        await agent.run("b", user="u")
        await h.writes.drain()
    assert [p for _, p in memory_service.named("model_key")] == [
        {"key": "sk-mem", "idempotency_key": "model-key:keyed"}
    ]


async def test_the_sampled_judge_scores_against_the_context_and_files_feedback(
    memory_service: FakeMemoryService,
) -> None:
    memory_service.report = Report(supported=2)
    async with Harness(config=Settings(memory_url="http://m", eval_sample=1.0)) as h:
        h.memory = Memory("http://m", None, client=memory_service)

        async def fn(input: str, agent: Runtime) -> str:
            return "you prefer email"

        result = await h.wrap(fn, id="judged", memory="read").run("contact?", user="u")
        await h.writes.drain()
    [(_, verified)] = memory_service.named("verify")
    assert (
        verified["answer"] == "you prefer email"
        and verified["bundle"].rendered == memory_service.context_text
    )
    [(_, feedback)] = memory_service.named("feedback")
    assert feedback.target_kind is FeedbackTargetKind.ANSWER and feedback.score == 1.0
    assert feedback.agent_run_id == result.run_id


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
    [(_, feedback)] = memory_service.named("feedback")
    assert feedback.target_kind is FeedbackTargetKind.TOOL_CALL and feedback.reviewer == "boss"


async def test_explicit_feedback_on_a_run(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    async def fn(input: str, agent: Runtime) -> str:
        return "12"

    result = await memory_harness.wrap(fn, id="f").run("stock?", user="u")
    feedback = await memory_harness.feedback(result.run_id, "correct", "13")
    assert feedback.correction == "13" and feedback.reviewer == "u"
    assert memory_service.named("feedback")[0][1] is feedback


async def test_local_tool_side_effects_are_published_to_the_catalog(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    async def fn(input: str, agent: Runtime) -> str:
        return "ok"

    await memory_harness.wrap(fn, id="c", tools=[stock]).run("x", user="u")
    await memory_harness.writes.drain()
    [(scope, entries)] = memory_service.named("put_catalog")
    assert scope == {"tenant_id": "default"} and entries[0]["name"] == "stock"


async def test_status_of_a_run_with_memory_is_unchanged_by_it(memory_harness: Harness) -> None:
    async def fn(input: str, agent: Runtime) -> str:
        assert agent.context is not None
        remembered = await agent.memory.call_agent_tool("memory_search", {"query": input})
        return str(remembered)

    result = await memory_harness.wrap(fn, id="direct", memory="read").run("x", user="u")
    assert result.status is RunStatus.SUCCESS and "memory_search ok" in result.answer
