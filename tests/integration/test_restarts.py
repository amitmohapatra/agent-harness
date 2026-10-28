"""Restarts: for each kept feature, the worker dies mid-run (a ``BaseException`` from a tool,
nothing more written, the lease left to lapse), another worker — another harness sharing only
the run store, as two worker processes share agent-runs — claims the run and finishes it: the
same answer as an uninterrupted run, no side effect done twice, no question asked twice.

The features covered elsewhere: a function agent's writes and a ``ReAct`` run's model steps
(``test_progress``), scratch files (``test_react_long_tasks``), working time
(``test_run_limits``), sandboxes (``test_sandbox``), the memory spool (``test_writes``)."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
import uuid
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from agents import Agent, function_tool
from deepagents import create_deep_agent
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool as lc_tool
from langgraph.checkpoint.memory import InMemorySaver

from tests.support import resume_worker
from tests.support.arm import arm, native_tools
from tests.support.chat_model import ScriptedChatModel
from tests.support.durable import SharedSaver
from tests.support.openai_model import ScriptedModel
from tests.support.shared_runs import FileRuns
from trellis import Budget, Harness, Hooks, ReAct, Runtime, Settings, current, middleware, tool
from trellis.contracts import RunStatus
from trellis.harness import runtime as runtime_module
from trellis.harness.middleware import GovernTools, RunCheckpointer
from trellis.harness.runs import LocalRuns
from trellis.runs import Job, LeaseLostError

ROOT = Path(__file__).resolve().parents[2]

#: A window with room for the long-task tools.
ROOM = 128_000
STEPS = "Do this step by step: "
done: list[str] = []
crashes: list[BaseException] = []


class Crash(BaseException):
    """The worker process dies: nothing more is written, the lease lapses."""


@pytest.fixture(autouse=True)
def _reset() -> None:
    done.clear()
    crashes[:] = [Crash()]


@tool(side_effects="read")
def boom() -> str:
    """The worker dies here, once."""
    if crashes:
        raise crashes.pop()
    return "fine"


@lc_tool
def lc_boom() -> str:
    """The worker dies here, once."""
    if crashes:
        raise crashes.pop()
    return "fine"


lc_boom.metadata = {"side_effects": "read"}


@tool(side_effects="write")
def ship(order: str) -> str:
    """Ship an order."""
    done.append(f"ship {order}")
    return f"shipped {order}"


@lc_tool
def refund(order: str) -> str:
    """Refund an order."""
    done.append(f"refund {order}")
    return f"refunded {order}"


refund.metadata = {"side_effects": "write"}


@tool(side_effects="irreversible")
def wire(to: str) -> str:
    """Wire money."""
    done.append(f"wire {to}")
    return f"wired {to}"


@pytest.fixture
async def workers() -> AsyncIterator[tuple[Harness, Harness]]:
    """Two worker processes: two harnesses sharing nothing but the run store."""
    store = LocalRuns()
    pair = (Harness(config=Settings(), runs=store), Harness(config=Settings(), runs=store))
    yield pair
    for h in pair:
        await h.aclose()


async def crash_then_finish(
    workers: tuple[Harness, Harness],
    build: Callable[[Harness], Any],
    text: str,
    **wrap: Any,
) -> Any:
    """The run started on the first worker, which dies mid-run; the second finishes it."""
    first, second = workers
    tools = wrap.pop("tools", [])
    dying, taking_over = (
        arm(h.wrap(build(h), id="restart", **wrap), tools) for h in (first, second)
    )
    store = first.runs
    assert isinstance(store, LocalRuns)
    handle = await dying.start(text, user="u", thread="t")
    worker = first.worker([dying])
    claimed = await store.claim(worker.worker_id, [dying.id])
    assert claimed is not None
    with pytest.raises(Crash):
        await dying.execute(
            Job(record=claimed.run, worker_id=worker.worker_id, lease_seconds=60, store=store)
        )
    held, _ = store._leases[handle.id]
    store._leases[handle.id] = (held, datetime.now(UTC) - timedelta(seconds=1))
    assert await second.worker([taking_over]).run_once()
    result = await handle.result(wait=5)
    assert (await handle.status()).attempt == 2
    return taking_over, result


async def test_a_plan_written_before_a_crash_is_the_plan_after_it(
    workers: tuple[Harness, Harness],
) -> None:
    todos = [{"content": "ship o1", "status": "in_progress"}]
    model = ScriptedChatModel(
        turns=[
            ("write_todos", {"todos": todos}),
            ("ship", {"order": "o1"}),
            ("boom", {}),
            ("boom", {}),
            "shipped o1",
        ]
    )
    _, result = await crash_then_finish(
        workers,
        lambda h: ReAct("s", model, context_window=ROOM),
        STEPS + "ship o1",
        tools=[ship, boom],
    )
    assert result.status is RunStatus.SUCCESS and result.answer == "shipped o1", result.error
    assert done == ["ship o1"]
    plan = next(str(m.content) for m in model.seen[-1] if isinstance(m, ToolMessage))
    assert plan.startswith("Updated todo list") and "ship o1" in plan  # on the second worker


async def test_a_budget_counts_what_was_spent_before_the_crash(
    workers: tuple[Harness, Harness],
) -> None:
    async def counter(input: str, agent: Runtime) -> list[Any]:
        first = await agent.tools.call("ship", order="o1")
        await agent.tools.call("boom")
        return [first, await agent.tools.call("ship", order="o2")]

    _, result = await crash_then_finish(
        workers, lambda h: counter, "go", tools=[ship, boom], budget=Budget(max_tool_calls=2)
    )
    assert done == ["ship o1"]  # replayed, not shipped again; o2 is past the budget
    assert result.status is RunStatus.PARTIAL
    assert str(result.answer[1]).startswith("ship was not run")


async def test_a_graphs_own_write_is_not_done_again_on_another_worker(
    workers: tuple[Harness, Harness],
) -> None:
    model = ScriptedChatModel(
        turns=[[("refund", {"order": "o1"}), ("lc_boom", {})], ("lc_boom", {}), "refunded o1"]
    )

    def graph(h: Harness) -> Any:
        return create_agent(
            model,
            tools=[refund, lc_boom],
            middleware=[GovernTools()],
            checkpointer=RunCheckpointer(),
        )

    _, result = await crash_then_finish(workers, graph, "refund o1")
    assert result.status is RunStatus.SUCCESS and result.answer == "refunded o1", result.error
    assert done == ["refund o1"]


async def test_a_framework_approval_given_before_a_crash_is_not_asked_again(
    workers: tuple[Harness, Harness],
) -> None:
    model = ScriptedChatModel(turns=[("wire", {"to": "ada"}), ("boom", {}), ("boom", {}), "wired"])
    first, second = workers
    store = first.runs
    assert isinstance(store, LocalRuns)
    built = {
        h: create_agent(
            model,
            tools=await native_tools(h, wire, boom, framework="langgraph"),
            middleware=[HumanInTheLoopMiddleware(interrupt_on={"wire": True})],
            checkpointer=RunCheckpointer(),
        )
        for h in workers
    }
    dying, taking_over = (h.wrap(built[h], id="restart") for h in workers)
    handle = await dying.start("pay ada", user="u", thread="t")
    assert await first.worker([dying]).run_once()
    paused = await handle.result(wait=5)
    assert paused.interrupt is not None
    await dying.resume(paused.interrupt.interrupt_id, "approve", reviewer="r")
    worker = first.worker([dying])
    claimed = await store.claim(worker.worker_id, [dying.id])
    assert claimed is not None
    with pytest.raises(Crash):
        await dying.execute(
            Job(record=claimed.run, worker_id=worker.worker_id, lease_seconds=60, store=store)
        )
    held, _ = store._leases[handle.id]
    store._leases[handle.id] = (held, datetime.now(UTC) - timedelta(seconds=1))
    assert await second.worker([taking_over]).run_once()
    result = await handle.result(wait=5)
    assert result.status is RunStatus.SUCCESS and result.answer == "wired", result.error
    assert done == ["wire ada"]  # approved once, wired once


async def test_an_sdk_agents_own_tool_is_not_run_again_on_another_worker(
    workers: tuple[Harness, Harness],
) -> None:
    @function_tool
    def deliver(order: str) -> str:
        """Deliver an order."""
        done.append(f"deliver {order}")
        return f"delivered {order}"

    @function_tool
    def die() -> str:
        """The worker dies here, once."""
        if crashes:
            raise crashes.pop()
        return "fine"

    model = ScriptedModel([("deliver", {"order": "o1"}), ("die", {}), ("die", {}), "delivered"])

    def agent(h: Harness) -> Any:
        return Agent(name="s", model=model, tools=[deliver, die])

    _, result = await crash_then_finish(workers, agent, "deliver o1")
    assert result.status is RunStatus.SUCCESS and result.answer == "delivered", result.error
    assert done == ["deliver o1"]


# --------------------------------------------------------------------------- a graph's checkpoint
@lc_tool
def add(item: str) -> str:
    """Add one item to the cart."""
    done.append(f"add {item}")
    return f"the cart has {len(done)}"


add.metadata = {"side_effects": "write"}


@lc_tool
def lc_wire(to: str) -> str:
    """Wire money."""
    done.append(f"wire {to}")
    return f"wired {to}"


lc_wire.metadata = {"side_effects": "irreversible"}


@lc_tool
async def confirm() -> str:
    """Ask the user to confirm before going on."""
    runtime = current()
    assert runtime is not None
    return str(await runtime.ask("Go on?"))


confirm.metadata = {"side_effects": "read"}


def results(model: ScriptedChatModel, call: int) -> list[Any]:
    return [m.content for m in model.seen[call] if isinstance(m, ToolMessage)]


def asked_of(model: ScriptedChatModel, call: int) -> list[Any]:
    return [m.content for m in model.seen[call] if isinstance(m, HumanMessage)]


@pytest.fixture
def every_step_saved(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every checkpoint saved as it is put: a crash leaves the step under way, not an earlier
    one (otherwise a read-only step is saved at most every ``PROGRESS_SECONDS``)."""
    monkeypatch.setattr(runtime_module, "PROGRESS_SECONDS", 0.0)


@pytest.mark.parametrize("build", [create_agent, create_deep_agent])
async def test_a_middleware_graph_without_a_checkpointer_continues_where_it_stopped(
    workers: tuple[Harness, Harness], build: Callable[..., Any]
) -> None:
    model = ScriptedChatModel(turns=[("refund", {"order": "o1"}), ("lc_boom", {}), "refunded o1"])

    def graph(h: Harness) -> Any:
        return build(model, tools=[refund, lc_boom], middleware=middleware())

    _, result = await crash_then_finish(workers, graph, "refund o1")
    assert result.status is RunStatus.SUCCESS and result.answer == "refunded o1", result.error
    assert done == ["refund o1"]
    # continued from its checkpoint: the question once, the refund's result kept, the model
    # not asked again for the steps the checkpoint holds
    assert len(model.seen) == 3
    assert asked_of(model, 2) == ["refund o1"] and results(model, 2) == ["refunded o1"]


async def test_a_later_steps_identical_write_runs_as_a_new_call_after_a_crash(
    workers: tuple[Harness, Harness], every_step_saved: None
) -> None:
    model = ScriptedChatModel(
        turns=[("add", {"item": "apple"}), ("lc_boom", {}), ("add", {"item": "apple"}), "two"]
    )

    def graph(h: Harness) -> Any:
        return create_agent(model, tools=[add, lc_boom], middleware=middleware())

    _, result = await crash_then_finish(workers, graph, "add two apples, one at a time")
    assert result.status is RunStatus.SUCCESS and result.answer == "two", result.error
    # the step under way when the worker died ran once; the later step's add is its own
    assert done == ["add apple", "add apple"]
    assert results(model, 3) == ["the cart has 1", "fine", "the cart has 2"]


@pytest.mark.parametrize("checkpointer", ["the run's", "its own"])
async def test_a_later_steps_identical_call_is_asked_about_again_after_a_pause(
    harness: Harness, checkpointer: str
) -> None:
    model = ScriptedChatModel(
        turns=[("lc_wire", {"to": "ada"}), ("confirm", {}), ("lc_wire", {"to": "ada"}), "twice"]
    )
    own = {"checkpointer": InMemorySaver()} if checkpointer == "its own" else {}
    graph = create_agent(model, tools=[lc_wire, confirm], middleware=middleware(), **own)
    agent = harness.wrap(graph, id="wiring")
    result = await agent.run("wire ada twice", user="u", thread="t")
    asked: list[str] = []
    while result.interrupt is not None:
        asked.append(result.interrupt.question)
        decision = "approve" if result.interrupt.tool_call is not None else "answer"
        result = await agent.resume(
            result.interrupt.interrupt_id, decision, answer="yes", reviewer="r"
        )
    assert result.status is RunStatus.SUCCESS and result.answer == "twice", result.error
    # the first approval is the first call's: the second is asked about, and wired
    wire = "Approve lc_wire? lc_wire is irreversible."
    assert asked == [wire, "Go on?", wire]
    assert done == ["wire ada", "wire ada"]
    assert results(model, 3) == ["wired ada", "yes", "wired ada"]


async def test_an_approval_then_a_crash_continues_without_asking_or_writing_again(
    workers: tuple[Harness, Harness],
) -> None:
    model = ScriptedChatModel(turns=[("lc_wire", {"to": "ada"}), ("lc_boom", {}), "wired"])
    first, second = workers
    store = first.runs
    assert isinstance(store, LocalRuns)
    dying, taking_over = (
        h.wrap(create_agent(model, tools=[lc_wire, lc_boom], middleware=middleware()), id="pay")
        for h in workers
    )
    handle = await dying.start("pay ada", user="u")
    assert await first.worker([dying]).run_once()
    paused = await handle.result(wait=5)
    assert paused.interrupt is not None and paused.interrupt.tool_call is not None
    await dying.resume(paused.interrupt.interrupt_id, "approve", reviewer="r")
    worker = first.worker([dying])
    claimed = await store.claim(worker.worker_id, [dying.id])
    assert claimed is not None
    with pytest.raises(Crash):  # wired, then the worker dies
        await dying.execute(
            Job(record=claimed.run, worker_id=worker.worker_id, lease_seconds=60, store=store)
        )
    assert done == ["wire ada"]
    held, _ = store._leases[handle.id]
    store._leases[handle.id] = (held, datetime.now(UTC) - timedelta(seconds=1))
    assert await second.worker([taking_over]).run_once()
    result = await handle.result(wait=5)
    assert result.status is RunStatus.SUCCESS and result.answer == "wired", result.error
    assert done == ["wire ada"] and (await handle.status()).attempt == 3


async def test_of_two_workers_racing_for_a_lapsed_lease_one_takes_the_run(
    workers: tuple[Harness, Harness],
) -> None:
    """The first worker is slow, not dead: its lease lapses while its write runs, two workers
    race to claim the run, and one does. The slow one's write is not run again (it is known
    to have started), and the slow one writes nothing once its lease is gone."""
    crashes.clear()
    writing, written = asyncio.Event(), asyncio.Event()

    @lc_tool
    async def slow_refund(order: str) -> str:
        """Refund an order (slowly)."""
        writing.set()
        await written.wait()
        done.append(f"refund {order}")
        return f"refunded {order}"

    slow_refund.metadata = {"side_effects": "write"}
    model = ScriptedChatModel(turns=[("slow_refund", {"order": "o1"}), "refunded o1", "late"])
    first, second = workers
    store = first.runs
    assert isinstance(store, LocalRuns)
    third = Harness(config=Settings(), runs=store)
    slow, *racing = (
        h.wrap(create_agent(model, tools=[slow_refund], middleware=middleware()), id="refunds")
        for h in (first, second, third)
    )
    handle = await slow.start("refund o1", user="u")
    worker = first.worker([slow])
    claimed = await store.claim(worker.worker_id, [slow.id])
    assert claimed is not None
    job = Job(record=claimed.run, worker_id=worker.worker_id, lease_seconds=60, store=store)
    running = asyncio.create_task(slow.execute(job))
    await writing.wait()
    held, _ = store._leases[handle.id]
    store._leases[handle.id] = (held, datetime.now(UTC) - timedelta(seconds=1))
    taken = await asyncio.gather(*(store.claim(f"w{n}", [slow.id]) for n in range(2)))
    winner = [n for n, c in enumerate(taken) if c is not None]
    assert len(winner) == 1
    won = taken[winner[0]]
    assert won is not None
    result = await racing[winner[0]].execute(
        Job(record=won.run, worker_id=f"w{winner[0]}", lease_seconds=60, store=store)
    )
    assert result.status is RunStatus.SUCCESS, result.error
    written.set()
    with pytest.raises(LeaseLostError):
        await running
    assert done == ["refund o1"]  # once, by the slow worker; the winner was told so
    assert "may or may not have taken effect" in results(model, 1)[0]
    record = await handle.status()
    assert record.status is RunStatus.SUCCESS and record.output == "refunded o1"
    await third.aclose()


@pytest.mark.parametrize("damage", ["unreadable", "missing"])
async def test_a_damaged_checkpoint_fails_the_run_clearly_and_writes_nothing_again(
    workers: tuple[Harness, Harness], damage: str
) -> None:
    model = ScriptedChatModel(turns=[("refund", {"order": "o1"}), ("lc_boom", {}), "refunded"])
    first, second = workers
    store = first.runs
    assert isinstance(store, LocalRuns)
    dying, taking_over = (
        h.wrap(create_agent(model, tools=[refund, lc_boom], middleware=middleware()), id="r")
        for h in workers
    )
    handle = await dying.start("refund o1", user="u")
    worker = first.worker([dying])
    claimed = await store.claim(worker.worker_id, [dying.id])
    assert claimed is not None
    with pytest.raises(Crash):
        await dying.execute(
            Job(record=claimed.run, worker_id=worker.worker_id, lease_seconds=60, store=store)
        )
    record = store._runs[handle.id]
    assert record.checkpoint is not None
    graph = record.checkpoint["graph"]
    if damage == "unreadable":
        graph[""]["checkpoint"] = ["msgpack", "not base64 !"]
    else:
        graph["tools:somewhere"] = graph.pop("")
    held, _ = store._leases[handle.id]
    store._leases[handle.id] = (held, datetime.now(UTC) - timedelta(seconds=1))
    assert await second.worker([taking_over]).run_once()
    result = await handle.result(wait=5)
    assert result.status is RunStatus.ERROR and result.error is not None
    assert result.error.code == "CHECKPOINT_UNREADABLE" and not result.error.retryable
    assert "would make its calls again" in result.error.message
    assert done == ["refund o1"]  # not refunded again from the start


async def test_a_new_version_continues_a_run_an_old_one_started(
    workers: tuple[Harness, Harness], caplog: pytest.LogCaptureFixture
) -> None:
    """A deploy between the crash and the resume: another prompt, one more tool, a
    different step limit — the run goes on from its checkpoint, said on its events."""

    @lc_tool
    def note(text: str) -> str:
        """Note something."""
        return "noted"

    model = ScriptedChatModel(turns=[("refund", {"order": "o1"}), ("lc_boom", {}), "refunded"])
    first, second = workers
    store = first.runs
    assert isinstance(store, LocalRuns)
    old = first.wrap(
        create_agent(model, tools=[refund, lc_boom], middleware=middleware(), system_prompt="v1"),
        id="refunds",
        version="1",
    )
    new = second.wrap(
        create_agent(
            model,
            tools=[refund, lc_boom, note],
            middleware=middleware(max_steps=20),
            system_prompt="v2",
        ),
        id="refunds",
        version="2",
    )
    handle = await old.start("refund o1", user="u")
    worker = first.worker([old])
    claimed = await store.claim(worker.worker_id, [old.id])
    assert claimed is not None
    with pytest.raises(Crash):
        await old.execute(
            Job(record=claimed.run, worker_id=worker.worker_id, lease_seconds=60, store=store)
        )
    held, _ = store._leases[handle.id]
    store._leases[handle.id] = (held, datetime.now(UTC) - timedelta(seconds=1))
    with caplog.at_level(logging.WARNING, logger="trellis.run"):
        assert await second.worker([new]).run_once()
    result = await handle.result(wait=5)
    assert result.status is RunStatus.SUCCESS and result.answer == "refunded", result.error
    assert done == ["refund o1"]
    assert model.seen[-1][0].content == "v2"  # the new version's prompt, the old checkpoint
    assert f"run {handle.id} was started by refunds 1 and continues on 2" in caplog.text


async def test_a_run_an_old_version_started_without_a_checkpoint_continues_from_its_input(
    workers: tuple[Harness, Harness],
) -> None:
    """Started where its graph kept no checkpoint (a release before the run's checkpoint was
    given to it), continued where it does: from its input, its write replayed by the journal."""
    model = ScriptedChatModel(
        turns=[("refund", {"order": "o1"}), ("lc_boom", {}), ("refund", {"order": "o1"}), "ok"]
    )
    old, new = (
        create_agent(model, tools=[refund, lc_boom], middleware=middleware(), checkpointer=c)
        for c in (False, None)
    )
    _, result = await crash_then_finish(workers, lambda h: old if h is workers[0] else new, "go")
    assert result.status is RunStatus.SUCCESS and result.answer == "ok", result.error
    assert done == ["refund o1"]


async def test_a_graphs_own_checkpointer_continues_the_runs_turn_after_a_crash(
    workers: tuple[Harness, Harness],
) -> None:
    name = uuid.uuid4().hex
    model = ScriptedChatModel(turns=[("refund", {"order": "o1"}), ("lc_boom", {}), "refunded"])

    def graph(h: Harness) -> Any:
        saver = SharedSaver(name)  # a durable checkpointer: every worker reads the same
        return create_agent(
            model, tools=[refund, lc_boom], middleware=middleware(), checkpointer=saver
        )

    _, result = await crash_then_finish(workers, graph, "refund o1")
    assert result.status is RunStatus.SUCCESS and result.answer == "refunded", result.error
    assert done == ["refund o1"]
    # its own checkpoint, continued: the question once, every call answered
    assert asked_of(model, 2) == ["refund o1"] and results(model, 2) == ["refunded o1", "fine"]


async def test_a_graphs_own_checkpointer_starts_a_new_runs_turn_from_its_input(
    workers: tuple[Harness, Harness],
) -> None:
    """The thread's latest checkpoint is an earlier run's: a run that died before its graph
    started begins its own turn, it does not continue the earlier one."""
    name = uuid.uuid4().hex
    model = ScriptedChatModel(turns=["hello", "again"])
    graphs = {
        h: create_agent(model, middleware=middleware(), checkpointer=SharedSaver(name))
        for h in workers
    }
    earlier = await workers[0].wrap(graphs[workers[0]], id="chat").run("hi", user="u", thread="t")
    assert earlier.answer == "hello"

    class Dying(Hooks):
        async def on_run_start(self, run: Any) -> None:
            if crashes:
                raise crashes.pop()

    _, result = await crash_then_finish(workers, lambda h: graphs[h], "and now?", hooks=[Dying()])
    assert result.status is RunStatus.SUCCESS and result.answer == "again", result.error
    assert asked_of(model, 1) == ["hi", "and now?"]


async def test_a_killed_worker_process_leaves_its_write_done_once(tmp_path: Path) -> None:
    """A real worker process, SIGKILLed after its write and before the next model call; a new
    process continues the run: the write ran once, though a model plans it anew each time
    (another request id), and the answer completes."""
    store_path, ledger = tmp_path / "runs", tmp_path / "ledger"
    store = FileRuns(store_path)
    async with Harness(config=Settings(), runs=store) as h:
        caller = h.wrap(placeholder, id=resume_worker.AGENT_ID)
        handle = await caller.start("ship o1", user="u")
    killed = await worker_process(store_path, ledger, tmp_path / "calls-1", die=True)
    assert killed == -signal.SIGKILL
    assert ledger.read_text().splitlines()[0].startswith("ship o1 ")
    store.lapse(handle.id)
    assert await worker_process(store_path, ledger, tmp_path / "calls-2") == 0
    record = await store.get(handle.id)
    assert record is not None and record.status is RunStatus.SUCCESS, record
    assert record.output == "done: shipped o1" and record.attempt == 2
    assert len(ledger.read_text().splitlines()) == 1  # shipped once
    # the new process continued from the checkpoint: one model call, the answer
    assert len((tmp_path / "calls-2").read_text().splitlines()) == 1


async def placeholder(input: str, agent: Runtime) -> str:
    """The caller's handle on the agent the worker processes run."""
    raise AssertionError("this agent runs in a worker process")


async def worker_process(store: Path, ledger: Path, calls: Path, *, die: bool = False) -> int:
    env = {**os.environ, resume_worker.DIE: "1" if die else ""}
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "tests.support.resume_worker",
        str(store),
        str(ledger),
        str(calls),
        cwd=ROOT,
        env=env,
    )
    return await asyncio.wait_for(process.wait(), timeout=90)
