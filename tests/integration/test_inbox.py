"""A run's inbox (agent-runs' ``/v1/runs/{id}/messages``, ``LocalRuns`` in process), read at
each of the run's steps: an instruction reaches a background task on any worker, and a person
pauses a run (LangGraph's own ``interrupt``, the checkpoint kept) and resumes it later, an
instruction sent while it waited read as it goes on.

The break-it half: a pause asked twice, a pause and a resume racing, an instruction arriving
while paused, the worker that paused the run gone when it resumes, and many runs paused and
resumed at once."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from tests.integration.test_background import (
    WAIT,
    Gate,
    ended,
    entered,
    lead,
    researcher,
    said,
    started,
    told,
)
from tests.support.adapters import BUILDERS, Call
from tests.support.arm import arm
from tests.support.models import ScriptedChat
from trellis import Harness, ReAct, Settings, tool
from trellis.contracts import (
    InterruptReason,
    RunEventType,
    RunMessageKind,
    RunOutcome,
    RunRecord,
    RunStatus,
)
from trellis.harness.runs import LocalRuns
from trellis.runs import ConflictError


@asynccontextmanager
async def working(h: Harness, *agents: Any) -> AsyncIterator[Any]:
    worker = h.worker(list(agents), concurrency=16)
    task = asyncio.create_task(worker.run())
    try:
        yield worker
    finally:
        worker.stop()
        worker.stop()
        await asyncio.wait_for(task, WAIT)


async def status(h: Harness, run_id: str, wanted: RunStatus) -> RunRecord:
    async with asyncio.timeout(WAIT):
        while True:
            record = await h.runs.get(run_id)
            assert record is not None
            if record.status is wanted:
                return record
            await asyncio.sleep(0.01)


async def send(h: Harness, run_id: str, kind: RunMessageKind, text: str | None = None) -> Any:
    sent = len(await h.runs.messages(run_id))
    return await h.runs.send_message(
        run_id, kind, message_id=f"m{sent}", text=text, sender="user:ada"
    )


def stepper(gate: Gate, steps: int, answer: str) -> ScriptedChat:
    """A task that holds in its tool for each of ``steps`` steps, then answers."""
    return ScriptedChat([*(("hold", {"topic": f"t{k}"}) for k in range(steps)), answer])


# --------------------------------------------------------------------------- an instruction
async def test_an_instruction_reaches_a_task_on_another_worker() -> None:
    """Two processes share the run store: the task works on the second one's worker; the
    first one's turn gives it an instruction, which its next step reads."""
    store = LocalRuns()
    first, second = (Harness(config=Settings(), runs=store) for _ in range(2))
    try:
        gate = Gate()
        work = stepper(gate, 1, "tides, and the north")
        idle = ScriptedChat()
        here = researcher(first, Gate(), model=idle)  # this process works nothing
        there = researcher(second, gate, model=work)
        model = ScriptedChat()
        parent = lead(first, model, here)
        async with working(second, there):
            _, task = await started(first, model, parent, thread="t1")
            await entered(gate)
            model.turns += [
                ("update_async_task", {"task_id": task, "message": "also check the north"}),
                "told it",
            ]
            await parent.run("also the north", user="ada", thread="t1")
            assert told(model, -1)[-1] == f"Updated async subagent. task_id: {task}"
            gate.released.set()
            assert (await ended(second, task)).status is RunStatus.SUCCESS
        assert "also check the north" in said(work, 1)
        assert not idle.requests
    finally:
        for h in (first, second):
            await h.aclose()


# --------------------------------------------------------------------------- pause, resume
async def test_a_person_pauses_a_task_and_resumes_it_with_an_instruction(
    harness: Harness,
) -> None:
    gate = Gate()
    work = stepper(gate, 2, "the tides, the north too")
    child = researcher(harness, gate, model=work)
    model = ScriptedChat()
    parent = lead(harness, model, child)
    async with working(harness, child):
        _, task = await started(harness, model, parent, thread="t1")
        await entered(gate)
        await send(harness, task, RunMessageKind.PAUSE)
        gate.released.set()  # the step under way ends; the next one pauses
        paused = await status(harness, task, RunStatus.PAUSED)
        assert paused.awaiting is not None
        assert paused.awaiting.reason is InterruptReason.PAUSE
        asked = len(work.requests)
        await send(harness, task, RunMessageKind.INSTRUCTION, "also check the north")
        await asyncio.sleep(0.05)
        assert len(work.requests) == asked, "a paused task asked its model"
        await child.resume(task, "approve", reviewer="ada")
        record = await ended(harness, task)
    assert record.status is RunStatus.SUCCESS, record.error
    assert record.output == "the tides, the north too"
    # resumed in place from its checkpoint: no step before the pause is asked again (the
    # plan's two steps left, and no more), and the first step after it reads the instruction
    assert len(work.requests) == asked + 2
    assert "also check the north" in said(work, asked)
    assert "also check the north" not in said(work, asked - 1)


async def test_a_pause_asked_twice_pauses_once_and_a_paused_run_takes_no_pause(
    harness: Harness,
) -> None:
    gate = Gate()
    child = researcher(harness, gate, model=stepper(gate, 2, "done"))
    model = ScriptedChat()
    parent = lead(harness, model, child)
    async with working(harness, child):
        _, task = await started(harness, model, parent, thread="t1")
        await entered(gate)
        await send(harness, task, RunMessageKind.PAUSE)
        await send(harness, task, RunMessageKind.PAUSE)
        gate.released.set()
        await status(harness, task, RunStatus.PAUSED)
        with pytest.raises(ConflictError):
            await send(harness, task, RunMessageKind.PAUSE)
        await child.resume(task, "approve", reviewer="ada")
        record = await ended(harness, task)
    assert record.status is RunStatus.SUCCESS, record.error


async def test_a_resume_racing_a_pause_never_leaves_the_task_stuck(harness: Harness) -> None:
    """A resume sent before the task reached its next step is refused (nothing waits yet);
    sent again once it paused, it goes on."""
    gate = Gate()
    child = researcher(harness, gate, model=stepper(gate, 2, "done"))
    model = ScriptedChat()
    parent = lead(harness, model, child)
    async with working(harness, child):
        _, task = await started(harness, model, parent, thread="t1")
        await entered(gate)
        await send(harness, task, RunMessageKind.PAUSE)
        early = await asyncio.gather(
            child.resume(task, "approve", reviewer="ada"), return_exceptions=True
        )
        assert isinstance(early[0], Exception), "a resume of a run not paused yet went through"
        gate.released.set()
        await status(harness, task, RunStatus.PAUSED)
        both = await asyncio.gather(
            child.resume(task, "approve", reviewer="ada"),
            child.resume(task, "approve", reviewer="ada"),
            return_exceptions=True,
        )
        assert sum(not isinstance(r, BaseException) for r in both) >= 1, both
        record = await ended(harness, task)
    assert record.status is RunStatus.SUCCESS, record.error


async def test_a_task_paused_on_one_worker_resumes_on_another() -> None:
    """The worker that paused the task stops (its process restarts); the resume is worked by
    another process's worker, from the task's checkpoint."""
    store = LocalRuns()
    first, second = (Harness(config=Settings(), runs=store) for _ in range(2))
    try:
        gate = Gate()
        work = stepper(gate, 2, "the tides")
        child = researcher(first, gate, model=work)
        model = ScriptedChat()
        parent = lead(first, model, child)
        async with working(first, child):
            _, task = await started(first, model, parent, thread="t1")
            await entered(gate)
            await send(first, task, RunMessageKind.PAUSE)
            gate.released.set()
            await status(first, task, RunStatus.PAUSED)
        await send(first, task, RunMessageKind.INSTRUCTION, "and the north")
        await child.resume(task, "approve", reviewer="ada")
        left = list(work.turns)  # what is left of the task's plan
        later = ScriptedChat(left)
        again = arm(
            second.wrap(ReAct(system_prompt="You research.", model=later), id="research"),
            [gate.tool()],
        )
        async with working(second, again):
            record = await ended(second, task)
        assert record.status is RunStatus.SUCCESS, record.error
        assert len(later.requests) == len(left), "the resumed task asked its earlier steps again"
        # from the checkpoint: the step before the pause is there, and the instruction after
        assert "facts about t0" in said(later, 0)
        assert "and the north" in said(later, 0)
    finally:
        for h in (first, second):
            await h.aclose()


async def test_many_tasks_paused_and_resumed_at_once(harness: Harness) -> None:
    gates = [Gate() for _ in range(6)]
    children = [
        researcher(harness, g, name=f"r{k}", model=stepper(g, 2, f"answer {k}"))
        for k, g in enumerate(gates)
    ]
    model = ScriptedChat()
    parent = lead(harness, model, *children)
    async with working(harness, *children):
        tasks = []
        for k in range(len(children)):
            model.turns += [
                ("start_async_task", {"description": "tides", "subagent_type": f"r{k}"}),
                "started",
            ]
            await parent.run("go", user="ada", thread="t1")
            tasks.append(told(model, -1)[-1].rsplit("task_id: ", 1)[1].strip())
        await asyncio.gather(*(entered(g) for g in gates))
        await asyncio.gather(*(send(harness, t, RunMessageKind.PAUSE) for t in tasks))
        for g in gates:
            g.released.set()
        await asyncio.gather(*(status(harness, t, RunStatus.PAUSED) for t in tasks))
        await asyncio.gather(
            *(c.resume(t, "approve", reviewer="ada") for c, t in zip(children, tasks, strict=True))
        )
        records = await asyncio.gather(*(ended(harness, t) for t in tasks))
    assert [r.output for r in records] == [f"answer {k}" for k in range(len(children))]


# --------------------------------------------------------------------------- every adapter
def _held(gate: Gate) -> Any:
    @tool(side_effects="read")
    async def hold(topic: str) -> str:
        """Look a topic up slowly.

        Args:
            topic: what to look up
        """
        gate.done.append(topic)
        gate.entered.set()
        await gate.released.wait()
        return f"facts about {topic}"

    return hold


async def _in_process(
    harness: Harness, framework: str, tmp_path: Path, gate: Gate
) -> tuple[Any, asyncio.Task[Any], str]:
    plan: list[Call] = [("hold", {"topic": "first"}), ("hold", {"topic": "second"})]
    target, tools = await BUILDERS[framework](harness, [_held(gate)], tmp_path, plan)
    agent = arm(harness.wrap(target, id=f"paused-{framework}"), tools)
    if not agent.adapter.steers(agent.target):
        pytest.skip(f"{framework}: no step the harness reaches (ADR 0009, follow-ups)")
    run = asyncio.create_task(agent.run("look the tides up", user="ada"))
    await entered(gate)
    [run_id] = agent.running
    return agent, run, run_id


@pytest.mark.parametrize("framework", list(BUILDERS))
async def test_a_run_on_every_adapter_pauses_at_its_next_step_and_resumes(
    harness: Harness, framework: str, tmp_path: Path
) -> None:
    gate = Gate()
    agent, run, run_id = await _in_process(harness, framework, tmp_path, gate)
    await send(harness, run_id, RunMessageKind.PAUSE)
    gate.released.set()
    paused = await run
    assert paused.status is RunStatus.PAUSED, (paused.status, paused.error)
    assert paused.interrupt is not None and paused.interrupt.reason is InterruptReason.PAUSE
    await send(harness, run_id, RunMessageKind.INSTRUCTION, "mind the north too")
    done = await agent.resume(run_id, "approve", reviewer="ada")
    assert done.status is RunStatus.SUCCESS, (done.status, done.error)
    assert gate.done.count("first") == 1, gate.done


@pytest.mark.parametrize("framework", list(BUILDERS))
async def test_a_run_on_every_adapter_takes_an_instruction_while_it_works(
    harness: Harness, framework: str, tmp_path: Path
) -> None:
    gate = Gate()
    _, run, run_id = await _in_process(harness, framework, tmp_path, gate)
    await send(harness, run_id, RunMessageKind.INSTRUCTION, "mind the north too")
    gate.released.set()
    done = await run
    assert done.status is RunStatus.SUCCESS, (done.status, done.error)
    assert (await harness.runs.messages(run_id))[0].text == "mind the north too"


# --------------------------------------------------------------------------- a remote inbox
class Remote:
    """A run store whose inbox is in another process (agent-runs): every read counted."""

    remote = True

    def __init__(self, inner: LocalRuns) -> None:
        self.inner = inner
        self.reads = 0
        self.down = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    async def messages(self, run_id: str, **kwargs: Any) -> Any:
        self.reads += 1
        if self.down:
            raise ConnectionError("agent-runs is down")
        return await self.inner.messages(run_id, **kwargs)


async def test_a_remote_inbox_is_read_at_most_every_few_seconds() -> None:
    store = Remote(LocalRuns())
    async with Harness(config=Settings(), runs=store) as h:  # type: ignore[arg-type]
        gate = Gate()
        gate.released.set()
        model = ScriptedChat([*(("hold", {"topic": f"t{k}"}) for k in range(3)), "done"])
        agent = arm(h.wrap(ReAct(system_prompt="r", model=model), id="r"), [gate.tool()])
        result = await agent.run("go", user="ada")
    assert result.status is RunStatus.SUCCESS
    assert len(model.requests) == 4 and store.reads == 1


async def test_an_inbox_that_cannot_be_read_is_said_once_and_the_run_goes_on() -> None:
    store = Remote(LocalRuns())
    store.down = True
    async with Harness(config=Settings(), runs=store) as h:  # type: ignore[arg-type]
        gate = Gate()
        gate.released.set()
        model = ScriptedChat([("hold", {"topic": "t"}), "done"])
        agent = arm(h.wrap(ReAct(system_prompt="r", model=model), id="r"), [gate.tool()])
        with patch("trellis.harness.runtime.INBOX_SECONDS", 0.0):
            seen = [e async for e in agent.stream("go", user="ada")]
    assert seen[-1].type is RunEventType.RUN_FINISHED
    assert seen[-1].outcome is RunOutcome.SUCCESS
    warned = [e for e in seen if e.data.get("code") == "inbox_unread"]
    assert len(warned) == 1 and store.reads == 2


async def test_the_in_process_inbox_keeps_a_message_sent_again_once() -> None:
    store = LocalRuns()

    async def fn(input: Any, agent: Any) -> str:
        return "done"

    async with Harness(config=Settings(), runs=store) as h:
        agent = h.wrap(fn, id="fn")
        await agent.start("x", user="ada")
        [summary] = [s async for s in store.iterate(agent_id="fn")]
        run_id = summary.run_id
        first = await send(h, run_id, RunMessageKind.INSTRUCTION, "north")
        again = await store.send_message(
            run_id, RunMessageKind.INSTRUCTION, message_id="m0", text="north", sender="ada"
        )
        assert again == first and len(await store.messages(run_id)) == 1
        with pytest.raises(ConflictError):
            await store.send_message(
                run_id, RunMessageKind.INSTRUCTION, message_id="m0", text="south", sender="ada"
            )
        await store.cancel(run_id)
        with pytest.raises(ConflictError):
            await send(h, run_id, RunMessageKind.INSTRUCTION, "late")
