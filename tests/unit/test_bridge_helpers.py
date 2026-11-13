"""``bridges.helpers``: a native helper's run record, started once and ended once."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command

from trellis.contracts import AgentExecutionContext, RunEvent, RunStart, RunStatus
from trellis.harness.bridges.helpers import HELPER, NOT_CONTINUED, Helpers, Kind, text_of
from trellis.harness.events import RunEvents
from trellis.harness.identity import Identity
from trellis.harness.runs import LocalRuns


class GraphInterrupt(Exception):
    """A pause signal."""


def runtime(runs: LocalRuns, attempt: int = 1) -> Any:
    identity = Identity(tenant="t", user="u", agent_id="desk", run_id="run_p")
    events = RunEvents(AgentExecutionContext.create(tenant_id="t", agent_id="desk"))
    seen: list[RunEvent] = []
    events.listen(seen.append)
    return SimpleNamespace(
        agent=SimpleNamespace(harness=SimpleNamespace(runs=runs, memory=None)),
        events=events,
        seen=seen,
        identity=identity,
        run_id="run_p",
        tenant="t",
        agent_id="desk",
        user="u",
        attempt=attempt,
        run_memory=None,
        pending=None,
        deadline=None,
    )


async def statuses(runs: LocalRuns) -> list[RunStatus]:
    return [c.status async for c in runs.iterate(parent_run_id="run_p")]


async def test_a_helper_started_again_is_the_same_record_and_ends_once() -> None:
    runs = LocalRuns()
    first = runtime(runs)
    helpers = Helpers(first, "langgraph")
    await helpers.start("k", kind=Kind.TASK, name="checker", task="check o1")
    await helpers.end("k", output="fine")
    await helpers.end("k", output="again")  # ended already: no change
    tree = [e.data for e in first.seen]  # on the run's task tree, once each
    assert [(d["name"], d.get("status")) for d in tree] == [
        ("helper_started", None),
        ("helper_finished", "success"),
    ]
    assert tree[0]["framework"] == "langgraph" and tree[0]["kind"] == "task"
    assert tree[1]["agent_id"] == "checker" and tree[1]["seconds"] >= 0
    again = Helpers(runtime(runs, attempt=2), "langgraph")
    await again.start("k2", kind=Kind.TASK, name="checker", task="check o1")  # the same record
    await again.end("k2", output="fine")  # ended before: left as it is
    [child] = [c async for c in runs.iterate(parent_run_id="run_p")]
    record = await runs.get(child.run_id)
    assert record is not None and record.status is RunStatus.SUCCESS and record.output == "fine"
    assert record.metadata[HELPER] == {"framework": "langgraph", "kind": "task", "name": "checker"}


async def test_the_same_task_twice_in_an_attempt_is_two_helpers() -> None:
    runs = LocalRuns()
    helpers = Helpers(runtime(runs), "langgraph")
    for key in ("a", "b"):
        await helpers.start(key, kind=Kind.TASK, name="checker", task="check o1")
    await helpers.closed(None)
    assert await statuses(runs) == [RunStatus.SUCCESS, RunStatus.SUCCESS]


@pytest.mark.parametrize("error", [GraphInterrupt(), KeyboardInterrupt()], ids=["pause", "crash"])
async def test_a_pause_or_a_crash_leaves_the_helper_to_the_next_attempt(
    error: BaseException,
) -> None:
    runs = LocalRuns()
    helpers = Helpers(runtime(runs), "langgraph")
    await helpers.start("k", kind=Kind.TASK, name="checker", task="check o1")
    await helpers.closed(error)
    assert await statuses(runs) == [RunStatus.RUNNING]


async def test_a_failed_run_ends_its_open_helpers_as_errors() -> None:
    runs = LocalRuns()
    helpers = Helpers(runtime(runs), "langgraph")
    await helpers.start("k", kind=Kind.TASK, name="checker", task="check o1")
    await helpers.closed(ValueError("boom"))
    [child] = [c async for c in runs.iterate(parent_run_id="run_p")]
    record = await runs.get(child.run_id)
    assert record is not None and record.error is not None
    assert record.error.message == "the run ended first: ValueError"


async def test_a_helper_an_earlier_attempt_left_open_and_the_next_did_not_take_up_ends() -> None:
    runs = LocalRuns()
    first = Helpers(runtime(runs), "claude_agent_sdk")
    await first.start("k", kind=Kind.SUBAGENT, name="checker", task="check o1")
    await first.closed(GraphInterrupt())  # paused with it open
    await runs.start(  # a sub-agent of the harness's own: it ends itself
        RunStart(
            run_id="run_own",
            tenant_id="t",
            agent_id="x",
            thread_id="x",
            user_id="u",
            parent_run_id="run_p",
            input="x",
        )
    )
    await Helpers(runtime(runs, attempt=2), "claude_agent_sdk").closed(None)
    found = {c.run_id: c.status async for c in runs.iterate(parent_run_id="run_p")}
    assert found.pop("run_own") is RunStatus.RUNNING
    [(run_id, status)] = found.items()
    record = await runs.get(run_id)
    assert status is RunStatus.ERROR and record is not None and record.error is not None
    assert record.error.message == NOT_CONTINUED


async def test_a_call_of_one_of_two_helpers_of_one_name_is_its_own_workers() -> None:
    """Claude Code names the subagent a call comes from by its type (``agent_type``), which
    two subagents at work at once may share, and by its task's id (``agent_id``), which
    ``task_started`` ties to the ``Agent`` call that started it: the call is that helper's,
    not the one of its name started last."""
    runs = LocalRuns()
    helpers = Helpers(runtime(runs), "claude_agent_sdk")
    kind, name = Kind.SUBAGENT, "general-purpose"
    first = await helpers.start("call_1", kind=kind, name=name, task="check o1")
    second = await helpers.start("call_2", kind=kind, name=name, task="check o2")
    helpers.attribute("task_1", "call_1")
    helpers.attribute("task_2", "call_2")
    assert await helpers.of("task_1", name) is first
    assert await helpers.of("task_2", name) is second
    helpers.attribute("task_0", None)  # who works for the run itself
    assert await helpers.of("task_0", name) is None
    await helpers.end("call_1", output="fine")
    assert await helpers.of("task_1", name) is None  # a helper that ended: the run's own


async def test_a_call_waits_to_learn_whose_it_is(monkeypatch: pytest.MonkeyPatch) -> None:
    """The framework says who works for which helper on the stream the run reads, and asks
    about a call in a task of its own: a call that comes before the run read that far waits
    for it. Never told, it is the helper of that name started last; with no helper of that
    name open there is nothing to wait for."""
    from trellis.harness.bridges import helpers as module

    monkeypatch.setattr(module, "ATTRIBUTED_SECONDS", 0.05)
    helpers = Helpers(runtime(LocalRuns()), "claude_agent_sdk")
    kind, name = Kind.SUBAGENT, "general-purpose"
    first = await helpers.start("call_1", kind=kind, name=name, task="check o1")
    second = await helpers.start("call_2", kind=kind, name=name, task="check o2")
    asking = asyncio.create_task(helpers.of("task_1", name))
    again = asyncio.create_task(helpers.of("task_1", name))
    await asyncio.sleep(0)
    helpers.attribute("task_1", "call_1")
    assert await asking is first and await again is first
    assert await helpers.of("task_9", name) is second  # never told
    assert await helpers.of("task_8", "another") is None  # nothing of that name is open


async def test_a_helper_at_work_in_the_background_ends_as_its_work_did() -> None:
    """A helper the framework runs in the background is not ended by its call's result (its
    launch): it ends when the framework says its work ended — with what it said, as an error
    when it failed or was stopped — and, still at work when the attempt ends, as cut off,
    whatever the run answered."""
    from trellis.harness.bridges.helpers import CUT_OFF, FAILED

    runs = LocalRuns()
    at = runtime(runs)
    helpers = Helpers(at, "claude_agent_sdk")
    kind = Kind.SUBAGENT
    for n in range(1, 6):
        await helpers.start(f"call_{n}", kind=kind, name="worker", task=f"task {n}")
        helpers.attribute(f"task_{n}", f"call_{n}", background=n != 5)
    assert helpers.waits("call_1") and not helpers.waits("call_5") and not helpers.waits("gone")
    await helpers.settled("task_1", failed=False, said="found it")
    await helpers.settled("task_2", failed=True)
    await helpers.settled("task_3", failed=False, said="", final=False)  # no more is said
    await helpers.settled("task_5", failed=False, said="not a background one")
    await helpers.settled("task_7", failed=False, said="nobody's")
    assert [c.status async for c in runs.iterate(parent_run_id="run_p", status=RunStatus.RUNNING)]
    await helpers.closed(None)  # the run answered
    found = {}
    async for child in runs.iterate(parent_run_id="run_p"):
        record = await runs.get(child.run_id)
        assert record is not None
        found[record.input] = (record.status, record.output, record.error)
    assert found["task 1"][:2] == (RunStatus.SUCCESS, "found it")
    assert found["task 2"][0] is RunStatus.ERROR and found["task 2"][2].message == FAILED
    assert found["task 3"][:2] == (RunStatus.SUCCESS, None)
    assert found["task 4"][0] is RunStatus.ERROR and found["task 4"][2].message == CUT_OFF
    assert found["task 5"][0] is RunStatus.SUCCESS  # an open helper ends with its run
    ended = [e.data["status"] for e in at.seen if e.data.get("name") == "helper_finished"]
    assert sorted(ended) == ["error", "error", "success", "success", "success"]


async def test_a_background_helper_of_a_run_that_failed_ends_with_why_the_run_did() -> None:
    runs = LocalRuns()
    helpers = Helpers(runtime(runs), "claude_agent_sdk")
    await helpers.start("call_1", kind=Kind.SUBAGENT, name="worker", task="task 1")
    helpers.attribute("task_1", "call_1", background=True)
    await helpers.closed(RuntimeError("boom"))
    [child] = [c async for c in runs.iterate(parent_run_id="run_p")]
    record = await runs.get(child.run_id)
    assert record is not None and record.error is not None
    assert record.error.message == "the run ended first: RuntimeError"


async def test_a_helper_started_twice_at_once_is_one_helper() -> None:
    """A run store that takes time to answer: the second start of a key, come meanwhile,
    is the first's helper, not a second record."""
    runs = LocalRuns()
    start = runs.start

    async def slow(*args: Any, **kwargs: Any) -> Any:
        await asyncio.sleep(0.01)
        return await start(*args, **kwargs)

    runs.start = slow  # type: ignore[method-assign]
    helpers = Helpers(runtime(runs), "claude_agent_sdk")
    kind, name = Kind.SUBAGENT, "checker"
    one, two = await asyncio.gather(
        helpers.start("k", kind=kind, name=name, task="check o1"),
        helpers.start("k", kind=kind, name=name, task="check o1"),
    )
    assert one is two and await statuses(runs) == [RunStatus.RUNNING]


def test_a_helpers_answer_is_its_text() -> None:
    assert text_of("fine") == "fine"
    assert text_of(AIMessage(content="said")) == "said"
    assert text_of(Command(update={"messages": [ToolMessage("done", tool_call_id="c")]})) == "done"
    assert text_of(SimpleNamespace(content="plain")) == "plain"
    assert text_of(SimpleNamespace(content=[1])) == "namespace(content=[1])"


# ---------------------------------------------------------------------- the adapters' hooks
class Recorded:
    """Helpers as the hooks drive them: what started and ended."""

    def __init__(self) -> None:
        self.runtime = SimpleNamespace(task="refund o1")
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    async def start(self, key: str, **given: Any) -> None:
        self.calls.append(("start", key, given))

    async def end(self, key: str, **given: Any) -> None:
        self.calls.append(("end", key, given))

    def attribute(self, worker: str, key: str | None, **given: Any) -> None:
        self.calls.append(("attribute", worker, {"key": key, **given}))

    def waits(self, key: str) -> bool:
        return key == "t3"

    async def settled(self, worker: str, **given: Any) -> None:
        self.calls.append(("settled", worker, given))


async def test_a_claude_subagents_call_starts_it_and_its_result_ends_it() -> None:
    from claude_agent_sdk import (
        AssistantMessage,
        TextBlock,
        ToolResultBlock,
        ToolUseBlock,
        UserMessage,
    )
    from claude_agent_sdk._internal.message_parser import parse_message

    from trellis.harness.adapters.claude import GENERAL_PURPOSE, SUBAGENT_FAILED, _subagents
    from trellis.harness.bridges.helpers import STOPPED

    def task(subtype: str, **more: Any) -> Any:
        frame = {"type": "system", "subtype": subtype, "session_id": "s", "uuid": "u", **more}
        return parse_message(frame)

    helpers = Recorded()
    called = AssistantMessage(
        content=[
            TextBlock(text="checking"),
            ToolUseBlock(id="t1", name="Agent", input={"description": "check o1"}),
            ToolUseBlock(id="t2", name="Read", input={}),
        ],
        model="m",
    )
    await _subagents(helpers, called)  # type: ignore[arg-type]
    await _subagents(helpers, AssistantMessage(content=[], model="m", parent_tool_use_id="t1"))  # type: ignore[arg-type]
    answered = UserMessage(
        content=[
            TextBlock(text="note"),
            ToolResultBlock(tool_use_id="t1", content="fine"),
            ToolResultBlock(tool_use_id="t2", content=None, is_error=True),
            ToolResultBlock(tool_use_id="t3", content="Async agent launched"),
        ]
    )
    await _subagents(helpers, answered)  # type: ignore[arg-type]
    await _subagents(helpers, UserMessage(content="a prompt"))  # type: ignore[arg-type]
    assert helpers.calls == [
        ("start", "t1", {"kind": Kind.SUBAGENT, "name": GENERAL_PURPOSE, "task": "check o1"}),
        ("end", "t1", {"output": "fine"}),
        ("end", "t2", {"error": SUBAGENT_FAILED}),  # t3 works in the background: not ended
    ]
    helpers.calls.clear()
    started = {"task_id": "a1", "description": "d", "task_type": "local_agent"}
    notified = {"task_id": "a1", "output_file": "", "tool_use_id": "t3"}
    frames = [
        task("task_started", **started, tool_use_id="t3", is_backgrounded=True),
        task("task_started", **started),  # one no call started (a shell command's)
        task("task_updated", task_id="a1", patch={"status": "running"}),
        task("task_updated", task_id="a1", patch={"status": "completed"}),
        task("task_updated", task_id="a1", patch={"status": "failed"}),
        task("task_updated", task_id="a1", patch={"status": "killed"}),
        task("task_notification", **notified, status="completed", summary="found it"),
        task("task_notification", **notified, status="failed", summary="no data"),
        task("task_notification", **notified, status="stopped", summary="x"),
    ]
    for frame in frames:
        await _subagents(helpers, frame)  # type: ignore[arg-type]
    assert helpers.calls == [
        ("attribute", "a1", {"key": "t3", "background": True}),
        ("settled", "a1", {"failed": False, "said": "", "final": False}),
        ("settled", "a1", {"failed": True, "said": "", "final": False}),
        ("settled", "a1", {"failed": True, "said": STOPPED, "final": False}),
        ("settled", "a1", {"failed": False, "said": "found it"}),
        ("settled", "a1", {"failed": True, "said": "no data"}),
        ("settled", "a1", {"failed": True, "said": STOPPED}),
    ]


async def test_a_helper_is_briefed_with_its_context_and_its_work_is_its_own() -> None:
    from trellis.harness.bridges.helpers import TASK_HEADING, inside

    runs = LocalRuns()
    parent = runtime(runs)
    helpers = Helpers(parent, "langgraph")
    assert parent.helpers is helpers  # the adapter's hooks find them on the runtime
    helper = await helpers.start("k", kind=Kind.TASK, name="checker", task="check o1")
    assert await helpers.start("k", kind=Kind.TASK, name="x", task="y") is helper  # once
    assert helpers.briefed(helper, "check o1") == "check o1"  # memory off: the task alone
    helper.context = "## Memories\n- o1 was late"
    assert helpers.briefed(helper, "check o1") == f"{helper.context}\n\n{TASK_HEADING}check o1"
    assert helpers.named("checker") is helper and helpers.named("nobody") is None
    assert inside(parent) is None
    with helpers.working(helper):
        assert inside(parent) is helper
        other = SimpleNamespace(run_id="run_other")  # a nested run's code is its own
        assert inside(other) is None  # type: ignore[arg-type]
    assert inside(parent) is None


async def test_a_helpers_context_that_cannot_be_had_is_a_warning() -> None:
    warned: list[tuple[str, str]] = []

    class Down:
        async def context(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("down")

    parent = runtime(LocalRuns())
    parent.uses = lambda feature: True
    parent.document_ids = None
    parent.events = SimpleNamespace(
        warning=lambda code, message: warned.append((code, message)),
        helper=lambda phase, **data: None,
    )
    helpers = Helpers(parent, "langgraph")
    helper = SimpleNamespace(memory=Down(), name="checker")
    assert await helpers._pushed(helper, "check o1") is None  # type: ignore[arg-type]
    assert warned == [("memory_unavailable", "no memory context for checker: down")]
    assert await helpers._pushed(helper, "") is None  # type: ignore[arg-type] # no task


async def test_a_task_whose_tool_answers_an_error_is_a_failed_helper() -> None:
    """``middleware._helped``: a Deep Agents ``task`` call is a helper of the run; one whose
    tool answers with an error message ends it failed; any other call runs as it is."""
    from trellis.harness.middleware import _helped

    runs = LocalRuns()
    parent = runtime(runs)
    Helpers(parent, "langgraph")  # the attempt's helpers, on the runtime
    given: list[dict[str, Any]] = []

    async def ran(args: dict[str, Any]) -> Any:
        given.append(args)
        return ToolMessage("the checker broke", tool_call_id="c1", status="error")

    args = {"description": "check o1", "subagent_type": "checker"}
    done = await _helped(parent, "task", "c1", args, ran)
    assert isinstance(done, ToolMessage) and done.status == "error"
    assert given == [args]  # memory off: the task as it was
    [child] = [c async for c in runs.iterate(parent_run_id="run_p")]
    record = await runs.get(child.run_id)
    assert record is not None and record.status is RunStatus.ERROR
    assert record.error is not None and record.error.message == "the checker broke"
    await _helped(parent, "lookup", "c2", {"sku": "a"}, ran)  # not a task: no helper
    assert len([c async for c in runs.iterate(parent_run_id="run_p")]) == 1


async def test_an_agent_tool_given_no_text_is_a_helper_on_its_arguments() -> None:
    """``tools.convert.openai_agents._helped``: an agent tool called with arguments that are
    not JSON is still the run's helper, its task the arguments as they came."""
    from agents import Agent

    from tests.support.openai_model import ScriptedModel
    from trellis.harness.tools.convert.openai_agents import _helped

    runs = LocalRuns()
    parent = runtime(runs)
    parent.budget = None
    parent.parent = None
    parent.replay = SimpleNamespace(journal=SimpleNamespace(spent={}))
    helpers = Helpers(parent, "openai_agents")
    tool = Agent(name="checker", model=ScriptedModel([])).as_tool("check", "Check.")
    seen: list[str] = []

    async def invoke(context: Any, arguments: str) -> str:
        seen.append(arguments)
        return "checked"

    context = SimpleNamespace(
        tool_call_id="c1",
        usage=SimpleNamespace(input_tokens=0, output_tokens=0, input_tokens_details=None),
    )
    helped = _helped(tool, invoke, helpers)
    assert await helped(context, "not json") == "checked" and seen == ["not json"]
    [child] = [c async for c in runs.iterate(parent_run_id="run_p")]
    record = await runs.get(child.run_id)
    assert record is not None and record.input == "not json"
