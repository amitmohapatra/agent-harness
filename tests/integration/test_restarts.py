"""Restarts: for each kept feature, the worker dies mid-run (a ``BaseException`` from a tool,
nothing more written, the lease left to lapse), another worker — another harness sharing only
the run store, as two worker processes share agent-runs — claims the run and finishes it: the
same answer as an uninterrupted run, no side effect done twice, no question asked twice.

The features covered elsewhere: a function agent's writes and a ``ReAct`` run's model steps
(``test_progress``), scratch files (``test_react_long_tasks``), working time
(``test_run_limits``), sandboxes (``test_sandbox``), the memory spool (``test_writes``)."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from agents import Agent, function_tool
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool as lc_tool

from tests.support.arm import arm, native_tools
from tests.support.chat_model import ScriptedChatModel
from tests.support.openai_model import ScriptedModel
from trellis import Budget, Harness, ReAct, Runtime, Settings, tool
from trellis.contracts import RunStatus
from trellis.harness.middleware import GovernTools, RunCheckpointer
from trellis.harness.runs import LocalRuns
from trellis.runs import Job

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
