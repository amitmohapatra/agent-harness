"""Background sub-agents (``background(*agents)``): the model starts a sub-agent's run that a
worker works while the conversation goes on, checks it, gives it a new instruction at its next
step, cancels it, lists them; its answer reaches the conversation when it ends; and only the
conversation that started a task finds it.

The break-it half: many tasks at once, a cancel in the middle of a step, conflicting
instructions, another user reaching for a task, the turn that started a task ending (or being
cancelled) while it works, and the worker dying under a task."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from tests.support.arm import arm
from tests.support.models import ScriptedChat
from trellis import Harness, ReAct, Runtime, Settings, background, tool
from trellis.contracts import ConfigurationError, RunRecord, RunStatus
from trellis.harness.background import BACKGROUND, NOTE
from trellis.harness.runs import LocalRuns
from trellis.runs import Job

#: How long a test waits for something a worker does, at most (a guard, not a measure).
WAIT = 10.0


class Gate:
    """A tool that holds until released: the task's step in progress."""

    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.released = asyncio.Event()
        self.done: list[str] = []

    def tool(self) -> Any:
        @tool(side_effects="read")
        async def hold(topic: str) -> str:
            """Look a topic up slowly.

            Args:
                topic: what to look up
            """
            self.entered.set()
            await self.released.wait()
            self.done.append(topic)
            return f"facts about {topic}"

        return hold


@asynccontextmanager
async def working(h: Harness, *agents: Any) -> AsyncIterator[None]:
    """A worker of ``h`` working ``agents``' queued runs meanwhile."""
    worker = h.worker(list(agents), concurrency=16)
    task = asyncio.create_task(worker.run())
    try:
        yield
    finally:
        worker.stop()
        worker.stop()
        await asyncio.wait_for(task, WAIT)


async def ended(h: Harness, run_id: str) -> RunRecord:
    async with asyncio.timeout(WAIT):
        while True:
            record = await h.runs.get(run_id)
            assert record is not None
            if record.status.final:
                return record
            await asyncio.sleep(0.01)


async def entered(gate: Gate) -> None:
    await asyncio.wait_for(gate.entered.wait(), WAIT)


def researcher(h: Harness, gate: Gate, *, name: str = "research", model: Any = None) -> Any:
    """A ``ReAct`` sub-agent that holds in its tool, then answers."""
    model = model or ScriptedChat([("hold", {"topic": "tides"}), "the tides are high"])
    return arm(h.wrap(ReAct(system_prompt="You research.", model=model), id=name), [gate.tool()])


def lead(h: Harness, model: ScriptedChat, *children: Any, name: str = "lead") -> Any:
    agent = h.wrap(ReAct(system_prompt="You lead.", model=model), id=name)
    return arm(agent, [background(*children)])


def told(model: ScriptedChat, request: int) -> list[str]:
    """What the tool results of request ``request`` said (an earlier turn's result marked as
    a dated snapshot read as it was: the tools' values are live)."""
    return [
        _undated(str(m["content"]))
        for m in model.requests[request]["messages"]
        if m["role"] == "tool"
    ]


def _undated(text: str) -> str:
    first, _, rest = text.partition("\n")
    return rest if first.startswith("[") and "out of date" in first else text


def said(model: ScriptedChat, request: int) -> str:
    return "\n".join(str(m.get("content")) for m in model.requests[request]["messages"])


def task_id(text: str) -> str:
    return text.rsplit("task_id: ", 1)[1].strip()


async def started(h: Harness, model: ScriptedChat, parent: Any, **run: Any) -> tuple[Any, str]:
    """A turn that starts one research task: its result, and the task's id."""
    model.turns += [
        ("start_async_task", {"description": "tides", "subagent_type": "research"}),
        "started",
    ]
    result = await parent.run("research the tides", user=run.pop("user", "ada"), **run)
    assert result.status is RunStatus.SUCCESS, result.error
    return result, task_id(told(model, len(model.requests) - 1)[-1])


# --------------------------------------------------------------------------- the six
async def test_a_task_works_in_the_background_and_the_answer_does_not_wait(
    harness: Harness,
) -> None:
    gate = Gate()
    child = researcher(harness, gate)
    model = ScriptedChat()
    parent = lead(harness, model, child)
    async with working(harness, child):
        result, task = await started(harness, model, parent, thread="t1")
        await entered(gate)  # the answer came while the task works
        record = await harness.runs.get(task)
        assert record is not None and record.status is RunStatus.RUNNING
        assert record.parent_run_id == result.run_id and record.agent_id == "research"
        assert (record.user_id, record.thread_id) == ("ada", "t1")
        assert record.metadata[BACKGROUND] == await harness.conversation(user="ada", thread="t1")
        gate.released.set()
        assert (await ended(harness, task)).output == "the tides are high"


async def test_its_status_when_asked(harness: Harness) -> None:
    gate = Gate()
    child = researcher(harness, gate)
    model = ScriptedChat()
    parent = lead(harness, model, child)
    async with working(harness, child):
        _, task = await started(harness, model, parent, thread="t1")
        await entered(gate)
        model.turns += [("check_async_task", {"task_id": task}), "still going"]
        await parent.run("how is it going?", user="ada", thread="t1")
        status = json.loads(told(model, -1)[-1])
        assert (status["task_id"], status["status"]) == (task, "running"), status
        gate.released.set()
        await ended(harness, task)
        model.turns += [("list_async_tasks", {"status_filter": "success"}), "done"]
        await parent.run("and now?", user="ada", thread="t1")
        [listed] = json.loads(told(model, -1)[-1])
        assert (listed["status"], listed["result"]) == ("success", "the tides are high")


async def test_a_new_instruction_reaches_the_task_at_its_next_step(harness: Harness) -> None:
    gate = Gate()
    work = ScriptedChat([("hold", {"topic": "tides"}), "tides, and the north too"])
    child = researcher(harness, gate, model=work)
    model = ScriptedChat()
    parent = lead(harness, model, child)
    async with working(harness, child):
        _, task = await started(harness, model, parent, thread="t1")
        await entered(gate)
        model.turns += [
            ("update_async_task", {"task_id": task, "message": "also check the north"}),
            "told it",
        ]
        await parent.run("also check the north", user="ada", thread="t1")
        assert told(model, -1)[-1] == f"Updated async subagent. task_id: {task}"
        gate.released.set()
        await ended(harness, task)
    # its next step read it, after the step it was in finished
    assert "also check the north" in said(work, 1)
    assert "also check the north" not in said(work, 0)


async def test_a_task_stops_when_asked(harness: Harness) -> None:
    gate = Gate()
    child = researcher(harness, gate)
    model = ScriptedChat()
    parent = lead(harness, model, child)
    async with working(harness, child):
        _, task = await started(harness, model, parent, thread="t1")
        await entered(gate)
        model.turns += [("cancel_async_task", {"task_id": task}), "stopped"]
        await parent.run("stop the research", user="ada", thread="t1")
        assert told(model, -1)[-1] == f"Cancelled async subagent task: {task}"
        assert (await ended(harness, task)).status is RunStatus.CANCELLED
        gate.released.set()
        await asyncio.sleep(0.05)
    assert gate.done == [], "the cancelled task's step went on"


async def test_its_answer_reaches_the_conversation(harness: Harness) -> None:
    gate = Gate()
    child = researcher(harness, gate)
    model = ScriptedChat()
    parent = lead(harness, model, child)
    async with working(harness, child):
        _, task = await started(harness, model, parent, thread="t1")
        gate.released.set()
        await ended(harness, task)
        model.turns += ["the research says the tides are high"]
        await parent.run("what did it find?", user="ada", thread="t1")
    context = said(model, -1)
    assert NOTE in context and "the tides are high" in context and task in context


async def test_its_answer_reaches_the_turn_working_when_it_ends(harness: Harness) -> None:
    """A turn working when a task ends reads its answer at its next step."""
    gate, busy = Gate(), Gate()
    child = researcher(harness, gate)
    model = ScriptedChat()
    parent = arm(lead(harness, model, child), [background(child), busy.tool()])
    async with working(harness, child):
        _, task = await started(harness, model, parent, thread="t1")
        model.turns += [("hold", {"topic": "the weather"}), "both done"]
        turn = asyncio.create_task(parent.run("meanwhile, the weather", user="ada", thread="t1"))
        await entered(busy)
        gate.released.set()
        await ended(harness, task)
        busy.released.set()
        assert (await turn).status is RunStatus.SUCCESS
    last = said(model, -1)
    assert f"[background task {task} (research) ended SUCCESS]" in last
    assert "the tides are high" in last


async def test_only_its_user_finds_a_task(harness: Harness) -> None:
    """Another user, on a thread named alike and on another: their check, update, cancel and
    list find no task, and the task works on untouched. Its user finds it from another chat
    (by its id; the list is the chat's)."""
    gate = Gate()
    child = researcher(harness, gate)
    model = ScriptedChat()
    parent = lead(harness, model, child)
    async with working(harness, child):
        _, task = await started(harness, model, parent, thread="t1")
        await entered(gate)
        for user, thread in (("eve", "t1"), ("eve", "t2")):
            model.turns += [
                [
                    ("check_async_task", {"task_id": task}),
                    ("update_async_task", {"task_id": task, "message": "drop it"}),
                    ("cancel_async_task", {"task_id": task}),
                    ("list_async_tasks", {}),
                ],
                "nothing",
            ]
            await parent.run("what tasks are there?", user=user, thread=thread)
            refused = told(model, -1)
            assert refused[:3] == [f"No async subagent task {task!r} of yours."] * 3
            assert refused[3] == "No async subagent tasks."
        record = await harness.runs.get(task)
        assert record is not None and record.status is RunStatus.RUNNING
        model.turns += [("check_async_task", {"task_id": task}), "found"]
        await parent.run("how is the tides research?", user="ada", thread="t2")
        assert json.loads(told(model, -1)[-1])["status"] == "running"
        gate.released.set()
        record = await ended(harness, task)
    assert record.status is RunStatus.SUCCESS
    assert all("drop it" not in str(r) for r in record.input), record.input


# --------------------------------------------------------------------------- break it
async def test_many_tasks_at_once(harness: Harness) -> None:
    async def count(input: str, agent: Runtime) -> str:
        """Count one bin."""
        await asyncio.sleep(0.02)
        return f"counted {input}"

    child = harness.wrap(count, id="research")
    model = ScriptedChat()
    parent = lead(harness, model, child)
    many = 12
    model.turns += [
        [
            ("start_async_task", {"description": f"bin {k}", "subagent_type": "research"})
            for k in range(many)
        ],
        "all started",
    ]
    result = await parent.run("count every bin", user="ada", thread="t1")
    assert result.answer == "all started"
    tasks = [task_id(t) for t in told(model, -1)]
    assert len(set(tasks)) == many
    async with working(harness, child):
        records = [await ended(harness, t) for t in tasks]
    assert sorted(r.output for r in records) == sorted(f"counted bin {k}" for k in range(many))
    model.turns += [("list_async_tasks", {}), "listed"]
    await parent.run("how many?", user="ada", thread="t1")
    assert len(json.loads(told(model, -1)[-1])) == many


async def test_conflicting_instructions_arrive_in_order(harness: Harness) -> None:
    gate = Gate()
    work = ScriptedChat([("hold", {"topic": "tides"}), "tides only"])
    child = researcher(harness, gate, model=work)
    model = ScriptedChat()
    parent = lead(harness, model, child)
    async with working(harness, child):
        _, task = await started(harness, model, parent, thread="t1")
        await entered(gate)
        for message in ("also check the north", "no, do not check the north"):
            model.turns += [("update_async_task", {"task_id": task, "message": message}), "ok"]
            await parent.run(message, user="ada", thread="t1")
        gate.released.set()
        await ended(harness, task)
    users = [m["content"] for m in work.requests[1]["messages"] if m["role"] == "user"]
    assert users[-2:] == ["also check the north", "no, do not check the north"], users


async def test_the_turn_ending_or_cancelled_leaves_the_task_working(harness: Harness) -> None:
    gate, busy = Gate(), Gate()
    child = researcher(harness, gate)
    model = ScriptedChat()
    parent = arm(lead(harness, model, child), [background(child), busy.tool()])
    async with working(harness, child):
        model.turns += [
            ("start_async_task", {"description": "tides", "subagent_type": "research"}),
            ("hold", {"topic": "the weather"}),
        ]
        turn = asyncio.create_task(parent.run("go", user="ada", thread="t1"))
        await entered(busy)
        task = task_id(told(model, -1)[-1])
        [summary] = [s async for s in harness.runs.iterate(agent_id="lead")]
        await parent.cancel(summary.run_id)  # the user stops the turn
        assert (await turn).status is RunStatus.CANCELLED
        await entered(gate)
        gate.released.set()
        assert (await ended(harness, task)).status is RunStatus.SUCCESS


async def test_a_task_whose_worker_dies_is_finished_by_another() -> None:
    """Two worker processes share the run store: the first dies inside the task's step, the
    lease lapses, the second continues the task, and the conversation is told its answer. An
    instruction for a task held by the other process is refused, not lost."""

    class Crash(BaseException):
        """The worker process dies."""

    crashes = [Crash()]

    @tool(side_effects="read")
    async def dies_once(topic: str) -> str:
        """Look a topic up.

        Args:
            topic: what to look up
        """
        if crashes:
            raise crashes.pop()
        return f"facts about {topic}"

    store = LocalRuns()
    first, second = (Harness(config=Settings(), runs=store) for _ in range(2))
    try:
        children = []
        for h in (first, second):
            work = ScriptedChat([("dies_once", {"topic": "tides"}), "the tides are high"])
            children.append(
                arm(h.wrap(ReAct(system_prompt="r", model=work), id="research"), [dies_once])
            )
        model = ScriptedChat()
        parent = lead(first, model, children[0])
        _, task = await started(first, model, parent, thread="t1")
        worker = first.worker([children[0]])
        claimed = await store.claim(worker.worker_id, ["research"])
        assert claimed is not None and claimed.run.run_id == task
        with pytest.raises(Crash):
            await children[0].execute(
                Job(record=claimed.run, worker_id=worker.worker_id, lease_seconds=60, store=store)
            )
        held, _ = store._leases[task]
        store._leases[task] = (held, datetime.now(UTC) - timedelta(seconds=1))
        assert await second.worker([children[1]]).run_once()
        record = await store.get(task)
        assert record is not None and record.status is RunStatus.SUCCESS, record
        assert record.attempt == 2
        model.turns += ["it found the tides are high"]
        await parent.run("what did it find?", user="ada", thread="t1")
        assert "the tides are high" in said(model, -1)
    finally:
        for h in (first, second):
            await h.aclose()


async def test_an_instruction_for_a_task_on_another_worker_is_refused(harness: Harness) -> None:
    gate = Gate()
    child = researcher(harness, gate)
    model = ScriptedChat()
    parent = lead(harness, model, child)
    _, task = await started(harness, model, parent, thread="t1")
    store = harness.runs
    assert isinstance(store, LocalRuns)
    assert await store.claim("another-process", ["research"]) is not None  # held elsewhere
    model.turns += [("update_async_task", {"task_id": task, "message": "the north"}), "ok"]
    await parent.run("also the north", user="ada", thread="t1")
    assert "works on another worker" in told(model, -1)[-1]
    assert not child.turns.given, "an instruction nobody will read was kept"


async def test_a_task_that_asks_is_answered_on_its_own_run(harness: Harness) -> None:
    """A background task's question waits in the inbox on its own run (nobody relays it):
    answering it there continues it."""

    async def asks(input: str, agent: Runtime) -> str:
        """Ask which budget, then answer."""
        budget = await agent.ask("Which budget?", options=["low", "high"])
        return f"{input} on a {budget} budget"

    child = harness.wrap(asks, id="research")
    model = ScriptedChat()
    parent = lead(harness, model, child)
    _, task = await started(harness, model, parent, thread="t1")
    async with working(harness, child):
        async with asyncio.timeout(WAIT):
            while (record := await harness.runs.get(task)) and record.status is not (  # noqa: ASYNC110 - the run store is polled
                RunStatus.PAUSED
            ):
                await asyncio.sleep(0.01)
        assert [s.run_id for s in await harness.inbox()] == [task]
        model.turns += [("check_async_task", {"task_id": task}), "it asks"]
        await parent.run("how is it going?", user="ada", thread="t1")
        assert json.loads(told(model, -1)[-1])["asks"] == "Which budget?"
        await child.resume(task, "answer", answer="low", reviewer="ada")
        assert (await ended(harness, task)).output == "tides on a low budget"


# --------------------------------------------------------------------------- edges
async def test_what_the_tools_say_at_the_edges(harness: Harness) -> None:
    """An unknown agent type, an update of a task that ended or whose agent takes no
    instruction, a cancel of a task that ended, a filter that matches none — each said."""

    async def quick(input: str, agent: Runtime) -> str:
        """Answer at once."""
        return {"answer": input * 3}  # type: ignore[return-value]

    child = harness.wrap(quick, id="research")
    model = ScriptedChat()
    parent = lead(harness, model, child)
    model.turns += [
        ("start_async_task", {"description": "x", "subagent_type": "nobody"}),
        "no",
    ]
    await parent.run("go", user="ada")  # a run with no thread: its tasks are its children
    assert told(model, -1) == ["Unknown subagent_type 'nobody': one of research"]
    _, task = await started(harness, model, parent, thread="t1")
    model.turns += [
        [
            ("update_async_task", {"task_id": task, "message": "more"}),
            ("list_async_tasks", {"status_filter": "cancelled"}),
        ],
        "said",
    ]
    await parent.run("more", user="ada", thread="t1")
    refusal, listed = told(model, -1)
    assert "takes no instruction while it works" in refusal
    assert listed == "No async subagent tasks."
    async with working(harness, child):
        record = await ended(harness, task)
    assert record.output == {"answer": "tidestidestides"}
    model.turns += [
        [
            ("update_async_task", {"task_id": task, "message": "more"}),
            ("cancel_async_task", {"task_id": task}),
            ("check_async_task", {"task_id": task}),
        ],
        "said",
    ]
    await parent.run("more", user="ada", thread="t1")
    update, cancel, check = told(model, -1)
    assert update == f"Task {task} already ended SUCCESS: nothing to update."
    assert cancel == f"Task {task} already ended."
    assert json.loads(json.loads(check)["result"]) == {"answer": "tidestidestides"}


async def test_what_a_turn_is_told_of_its_tasks(harness: Harness) -> None:
    """A task that failed is told with why; a long answer is cut; a run store that cannot say
    is a warning, and the turn goes on."""

    async def fails(input: str, agent: Runtime) -> str:
        """Fail, or answer at length."""
        if input == "fail":
            raise RuntimeError("the archive is down")
        return "word " * 1000

    child = harness.wrap(fails, id="research")
    model = ScriptedChat()
    parent = lead(harness, model, child)
    model.turns += [
        [
            ("start_async_task", {"description": "fail", "subagent_type": "research"}),
            ("start_async_task", {"description": "long", "subagent_type": "research"}),
        ],
        "started",
    ]
    await parent.run("go", user="ada", thread="t1")
    tasks = [task_id(t) for t in told(model, -1)]
    async with working(harness, child):
        for task in tasks:
            await ended(harness, task)
    model.turns += ["told"]
    await parent.run("and?", user="ada", thread="t1")
    context = said(model, -1)
    assert "the archive is down" in context
    assert "(check_async_task: all)" in context and context.count("word") < 1000

    async def broken(**_: Any) -> Any:
        raise RuntimeError("the run store is down")
        yield  # pragma: no cover - an async generator

    store = harness.runs
    original = store.iterate
    store.iterate = broken  # type: ignore[method-assign]
    try:
        model.turns += ["still answers"]
        result = await parent.run("and?", user="ada", thread="t1")
    finally:
        store.iterate = original  # type: ignore[method-assign]
    assert result.answer == "still answers"
    assert NOTE not in said(model, -1)


def test_background_takes_its_agents() -> None:
    with pytest.raises(ConfigurationError, match="takes the agents it starts"):
        background()
