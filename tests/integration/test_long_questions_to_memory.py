"""A user message longer than the memory service takes as a question (``ContextRequest.query`` and
``ToolInvocationIn.task``: at most 4,000 characters, its OpenAPI says) still gets its memory
context and its tool records: the harness sends the service the question cut to fit, as the
service's contract allows, not a request it refuses with 422 (SCN-7, found by the scenario suite:
a 216,000-character message ran with no memory context, and a tool call it made was "failed and
is lost: ValidationError")."""

from __future__ import annotations

from tests.support.arm import arm
from tests.support.memory import FakeMemoryService
from tests.support.models import ScriptedChat
from trellis import Harness, ReAct, tool
from trellis.contracts import RunEventType, RunStatus
from trellis.harness.clients.memory import QUESTION_MAX_CHARS


@tool(side_effects="read")
def lookup(key: str) -> str:
    """Look a key up."""
    return f"value of {key[:10]}"


async def test_a_long_question_gets_its_context_and_its_tool_records(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    model = ScriptedChat([("lookup", {"key": "k"}), "Done."])
    agent = arm(memory_harness.wrap(ReAct(system_prompt="s", model=model), id="long"), [lookup])
    question = "Please read this. " + "lorem ipsum dolor " * 1_000  # ~18,000 characters
    events = [e async for e in agent.stream(question, user="ada", thread="t1")]
    assert events[-1].data.get("outcome") or events[-1].type is RunEventType.RUN_FINISHED
    warned = [
        e.data
        for e in events
        if e.type is RunEventType.CUSTOM and e.data.get("code") == "memory_unavailable"
    ]
    assert warned == [], warned  # the context was loaded, not refused
    await memory_harness.writes.drain()
    assert memory_harness.writes.failed == 0
    [context] = memory_service.named("context")
    assert len(context.body["query"]) <= QUESTION_MAX_CHARS
    [record] = memory_service.named("record_tool")
    assert len(record.body["task"]) <= QUESTION_MAX_CHARS
    assert record.body["task"].startswith("Please read this.")


async def test_a_short_question_is_sent_as_it_is(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    agent = memory_harness.wrap(ReAct(system_prompt="s", model=ScriptedChat(["Done."])), id="short")
    done = await agent.run("What is the locker code?", user="ada", thread="t1")
    assert done.status is RunStatus.SUCCESS
    [context] = memory_service.named("context")
    assert context.body["query"] == "What is the locker code?"
