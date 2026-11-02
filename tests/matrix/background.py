"""The matrix's background sub-agent rows (``background(*agents)``), on every adapter: the
agent under test starts a task that a worker works while the conversation goes on (B01), says
how it is going (B02), stops it (B04), is told its answer at a later turn (B05) and finds no
task of another user (B06); and, as the sub-agent, takes a new instruction at its next
step where its framework has a step the harness reaches — else the instruction is refused,
said (B03); and is paused by a person at its next step, given an instruction while it waits,
and resumed, going on from where it was with the instruction (B07).

Each turn is a run of its own (a planned model makes one turn's calls), the task's id read
from the turn that started it. The rows drive the runs themselves: ``run``
mode only, the worker the scenario's own.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any, Final

from tests.matrix.model import NA, Feature
from tests.matrix.world import USER, World
from tests.support.arm import arm
from trellis import Runtime, background, tool
from trellis.contracts import InterruptReason, RunMessageKind, RunRecord, RunStatus

#: How long a scenario waits for what a worker does, at most (a guard, not a measure).
WAIT: Final = 30.0
#: How many steps the sub-agent of B03 takes (each holding until the instruction was sent).
REPEATS: Final = 4
#: What the task answers: told to the conversation's later turn (B05).
ANSWER: Final = "the tides peak at 06:12"
#: Why a graph without the harness's middleware is not paused by a person (B07): no node of
#: the harness's runs in it; between its supersteps is a follow-up (ADR 0009).
PLAIN_GRAPH: Final = (
    "a graph without the harness's middleware (no middleware()) has no node the harness runs: "
    "a pause between its supersteps is a follow-up"
)
DIRECT: Final = "the scenario drives several turns and a worker itself (agent.run): the modes are the other rows'"


def _only_run() -> dict[str, Any]:
    from tests.matrix.model import MODES

    return {m: NA(DIRECT) for m in MODES if m != "run"}


@asynccontextmanager
async def worked(w: World, agent: Any, release: asyncio.Event) -> AsyncIterator[None]:
    """A worker of the cell's deployment working ``agent``'s tasks meanwhile; ``release``d
    as it stops (a task still holding lets go)."""
    worker = w.harness(0).worker([agent])
    task = asyncio.create_task(worker.run())
    try:
        yield
    finally:
        release.set()
        worker.stop()
        worker.stop()
        await asyncio.wait_for(task, WAIT)


class _Task:
    """A research task that holds until released, then answers :data:`ANSWER`."""

    def __init__(self, w: World) -> None:
        self.entered = asyncio.Event()
        self.released = asyncio.Event()
        h = w.harness(0)

        async def research(input: str, agent: Runtime) -> str:
            """Research a topic and report what was found."""
            self.entered.set()
            await self.released.wait()
            return ANSWER

        self.agent = h.agents.get("research") or arm(h.wrap(research, id="research"), [])
        self.worked = lambda: worked(w, self.agent, self.released)


async def _turn(
    w: World, task: _Task, plan: list[Any], *, user: str = USER, thread: str | None = None
) -> Callable[[str], bool]:
    """One more turn of the agent under test with ``background(research)``, following
    ``plan`` — in a chat of its own unless ``thread`` names one (a task is its user's, found
    by its id from any of their chats; a planned model would count an earlier turn's results
    its chat replays as its own); whether a text first reached the models (or the answer) in
    it."""
    before = w.said()
    agent = await w.agent([background(task.agent)], plan)
    thread = thread or f"chat-{len(w.store._runs)}"
    result = await agent.run("about the tides", user=user, thread=thread)
    assert result.status is RunStatus.SUCCESS, (result.status, result.error)
    after = f"{w.said()}\n{result.answer}"  # a function target's model is its code
    return lambda text: after.count(text) > before.count(text)


async def _started(w: World, task: _Task) -> str:
    """A turn that starts the research task: the task's id, read from the answer."""
    start = ("start_async_task", {"description": "the tides", "subagent_type": "research"})
    agent = await w.agent([background(task.agent)], [start])  # the thread's first turn
    answer = str((await agent.run("about the tides", user=USER, thread="t")).answer)
    assert "Launched async subagent. task_id: " in answer, answer
    return answer.rsplit("task_id: ", 1)[1].strip()


async def _record(w: World, run_id: str) -> RunRecord:
    record = await w.store.get(run_id)
    assert record is not None, run_id
    return record


async def _ended(w: World, run_id: str) -> RunRecord:
    async with asyncio.timeout(WAIT):
        while not (record := await _record(w, run_id)).status.final:  # noqa: ASYNC110 - polled
            await asyncio.sleep(0.01)
    return record


# --------------------------------------------------------------------------- the rows
async def starts_in_background(w: World) -> None:
    """The turn that starts a task answers while the task works on."""
    task = _Task(w)
    async with task.worked():
        run_id = await _started(w, task)
        await asyncio.wait_for(task.entered.wait(), WAIT)
        record = await _record(w, run_id)
        assert record.status is RunStatus.RUNNING, record.status
        assert record.parent_run_id is not None and record.user_id == USER
        task.released.set()
        assert (await _ended(w, run_id)).output == ANSWER


async def says_how_it_goes(w: World) -> None:
    task = _Task(w)
    async with task.worked():
        run_id = await _started(w, task)
        await asyncio.wait_for(task.entered.wait(), WAIT)
        seen = await _turn(w, task, [("check_async_task", {"task_id": run_id})])
        assert seen("running"), "the running task's status was not said"
        task.released.set()
        await _ended(w, run_id)
        seen = await _turn(w, task, [("check_async_task", {"task_id": run_id})])
        assert seen("success") and seen(ANSWER), "the ended task's was not said"


async def takes_an_instruction(w: World) -> None:
    """The agent under test is the task: its next step reads the instruction (where its
    framework has a step the harness reaches; else the instruction is refused, said). The
    lead is a function (its code makes the calls): with named models every ``ReAct`` of a
    cell is served the one planned model, which a ``ReAct`` lead's plan would replace."""
    from tests.matrix.kit import Desk

    d = Desk()
    gate = asyncio.Event()

    @tool(side_effects="read")
    async def hold(topic: str) -> str:
        """Look a topic up slowly.

        Args:
            topic: what to look up
        """
        d.started.set()
        await gate.wait()
        return f"facts about {topic}"

    child = await w.agent(
        [hold], [("hold", {"topic": f"tides {k}"}) for k in range(REPEATS)], name="research"
    )
    async with worked(w, child, gate):
        start = ("start_async_task", {"description": "the tides", "subagent_type": "research"})
        lead = await w.agent([background(child)], [start], parent="function")
        answer = str((await lead.run("about the tides", user=USER, thread="t")).answer)
        run_id = answer.rsplit("task_id: ", 1)[1].strip()
        await asyncio.wait_for(d.started.wait(), WAIT)
        update = ("update_async_task", {"task_id": run_id, "message": "mind the north too"})
        lead = await w.agent([background(child)], [update], parent="function")
        updated = await lead.run("and the north", user=USER, thread="another chat")
        answer = str(updated.answer)  # the lead's code's last result
        gate.set()
        record = await _ended(w, run_id)
    assert record.status is RunStatus.SUCCESS, (record.status, record.error)
    if child.adapter.steers(child.target):
        assert f"Updated async subagent. task_id: {run_id}" in answer, "the update was refused"
        added = "".join(p.read_text() for p in w.tmp.glob("cli-*.added"))  # Claude's hooks'
        reached = w.said().replace("mind the north too", "", 1) + added  # less the update call
        assert "mind the north too" in reached, "the instruction never reached the task's model"
    else:
        assert "takes no instruction while it works" in answer, "the refusal was not said"


async def pauses_and_resumes(w: World) -> None:
    """The agent under test is the task: a person pauses it (its inbox, read at its next
    step: LangGraph's own ``interrupt``, Claude's and Kimi's session stopped), sends it an
    instruction while it waits, and resumes it, on the worker: it goes on from where it was
    (no step before the pause made again) and reads the instruction."""
    from tests.matrix.kit import Desk

    d = Desk()
    gate = asyncio.Event()
    looked: list[str] = []

    @tool(side_effects="read")
    async def hold(topic: str) -> str:
        """Look a topic up slowly.

        Args:
            topic: what to look up
        """
        looked.append(topic)
        d.started.set()
        await gate.wait()
        return f"facts about {topic}"

    child = await w.agent(
        [hold], [("hold", {"topic": "first"}), ("hold", {"topic": "second"})], name="research"
    )
    runs = w.harness(0).runs
    async with worked(w, child, gate):
        start = ("start_async_task", {"description": "the tides", "subagent_type": "research"})
        lead = await w.agent([background(child)], [start], parent="function")
        answer = str((await lead.run("about the tides", user=USER, thread="t")).answer)
        run_id = answer.rsplit("task_id: ", 1)[1].strip()
        await asyncio.wait_for(d.started.wait(), WAIT)
        await runs.send_message(run_id, RunMessageKind.PAUSE, message_id="p", sender=USER)
        gate.set()  # the step under way ends; the next one pauses
        async with asyncio.timeout(WAIT):
            while (paused := await _record(w, run_id)).status is not RunStatus.PAUSED:
                assert not paused.status.final, (paused.status, paused.error)
                await asyncio.sleep(0.02)
        assert paused.awaiting is not None and paused.awaiting.reason is InterruptReason.PAUSE
        await runs.send_message(
            run_id,
            RunMessageKind.INSTRUCTION,
            message_id="i",
            text="mind the north too",
            sender=USER,
        )
        await child.resume(run_id, "approve", reviewer=USER)
        record = await _ended(w, run_id)
    assert record.status is RunStatus.SUCCESS, (record.status, record.error)
    assert looked.count("first") == 1, f"a step before the pause was made again: {looked}"
    added = "".join(p.read_text() for p in w.tmp.glob("cli-*.added"))  # Claude's hooks'
    assert "mind the north too" in w.said() + added, "the instruction never reached the model"


async def stops(w: World) -> None:
    task = _Task(w)
    async with task.worked():
        run_id = await _started(w, task)
        await asyncio.wait_for(task.entered.wait(), WAIT)
        seen = await _turn(w, task, [("cancel_async_task", {"task_id": run_id})])
        assert seen(f"Cancelled async subagent task: {run_id}"), "the cancel was not said"
        assert (await _ended(w, run_id)).status is RunStatus.CANCELLED


async def tells_its_answer(w: World) -> None:
    """A later turn of the conversation is told the task's answer."""
    task = _Task(w)
    async with task.worked():
        run_id = await _started(w, task)
        task.released.set()
        await _ended(w, run_id)
        await _turn(w, task, [], thread="t")  # the conversation that started it
    assert ANSWER in w.said(), "the conversation's later turn was not told the task's answer"


async def keeps_to_its_user(w: World) -> None:
    """Another user, on a thread named alike and on another: no task found, none stopped."""
    task = _Task(w)
    async with task.worked():
        run_id = await _started(w, task)
        await asyncio.wait_for(task.entered.wait(), WAIT)
        for user, thread in (("eve", "t"), ("eve", None)):
            seen = await _turn(
                w, task, [("cancel_async_task", {"task_id": run_id})], user=user, thread=thread
            )
            assert seen("No async subagent task"), (user, thread)
        assert (await _record(w, run_id)).status is RunStatus.RUNNING


BACKGROUND_ROWS: Final = [
    Feature(
        "B01",
        "a background sub-agent: started by the model, the answer not waiting for it",
        "BG (task 8)",
        "background(agent): start_async_task, a worker",
        starts_in_background,
        modes=_only_run(),
        way2=NA("the background tasks are the Harness's"),
    ),
    Feature(
        "B02",
        "a background sub-agent's status, when asked",
        "BG (task 8)",
        "background(agent): check_async_task",
        says_how_it_goes,
        modes=_only_run(),
        way2=NA("the background tasks are the Harness's"),
    ),
    Feature(
        "B03",
        "a new instruction for a working background sub-agent, read at its next step",
        "BG (task 8)",
        "background(agent): update_async_task (the framework's step hook, Adapter.steers)",
        takes_an_instruction,
        modes=_only_run(),
        way2=NA("the background tasks are the Harness's"),
    ),
    Feature(
        "B07",
        "a background sub-agent paused by a person and resumed, an instruction sent meanwhile",
        "BG (task 13)",
        "the run's inbox (Runtime.step) at the framework's step: LangGraph interrupt, a "
        "Claude or Kimi session stopped and continued",
        pauses_and_resumes,
        adapters={
            "function": NA("a function has no step the harness reaches: none to pause at"),
            "langgraph": NA(PLAIN_GRAPH),
            "deepagents": NA(PLAIN_GRAPH),
        },
        modes=_only_run(),
        way2=NA("the background tasks are the Harness's"),
    ),
    Feature(
        "B04",
        "a background sub-agent stopped on request",
        "BG (task 8)",
        "background(agent): cancel_async_task (agent.cancel)",
        stops,
        modes=_only_run(),
        way2=NA("the background tasks are the Harness's"),
    ),
    Feature(
        "B05",
        "a background sub-agent's answer told to the conversation",
        "BG (task 8)",
        "automatic: the turn's note of its conversation's tasks",
        tells_its_answer,
        adapters={
            "function": NA("a function target has no model to tell: its code calls the tools")
        },
        modes=_only_run(),
        way2=NA("the background tasks are the Harness's"),
    ),
    Feature(
        "B06",
        "only its own user finds a background sub-agent",
        "BG (task 8)",
        "automatic: a task is its user's, in its tenant",
        keeps_to_its_user,
        modes=_only_run(),
        way2=NA("the background tasks are the Harness's"),
    ),
]
