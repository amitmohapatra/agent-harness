"""``ReAct`` with several calls in one step: reads at once, writes one at a time after them in
the model's order, the tool messages in the model's order; a pause inside a batch; replay that
hands each call its own output whatever order they finished in; the bridge safe under calls made
at once from any framework; the step limit, the error streak and a stable tool order."""

from __future__ import annotations

import asyncio
import itertools
from typing import Any

import pytest

from tests.support.models import ScriptedChat
from trellis import Harness, ReAct, Runtime, current, tool
from trellis.contracts import RunEventType, RunStatus
from trellis.harness.adapters.react import LAST_STEP
from trellis.harness.runs import LocalRuns
from trellis.runs import Lease


def told(model: ScriptedChat, request: int) -> list[tuple[str, str]]:
    """The tool messages a request carries: (call id, content), in order."""
    return [
        (m["tool_call_id"], m["content"])
        for m in model.requests[request]["messages"]
        if m["role"] == "tool"
    ]


async def test_reads_run_together_then_writes_one_at_a_time_in_the_models_order(
    harness: Harness,
) -> None:
    happened: list[str] = []
    together = asyncio.Barrier(2)  # each read waits for the other: they must run at once

    @tool(side_effects="read")
    async def price(sku: str) -> str:
        """A price."""
        happened.append(f"price {sku}")
        await asyncio.wait_for(together.wait(), 1)
        happened.append(f"priced {sku}")
        return f"{sku}: 7"

    @tool(side_effects="write")
    async def order(sku: str) -> str:
        """Order a SKU."""
        happened.append(f"order {sku}")
        await asyncio.sleep(0)
        happened.append(f"ordered {sku}")
        return f"ordered {sku}"

    batch = [("order", {"sku": "b"}), ("price", {"sku": "a"}), ("order", {"sku": "c"})]
    model = ScriptedChat([[*batch, ("price", {"sku": "b"})], "done"])
    agent = harness.wrap(ReAct(system="s", model=model), id="shop", tools=[price, order])
    result = await agent.run("x", user="u")
    assert result.status is RunStatus.SUCCESS, result.error
    assert happened[:2] == ["price a", "price b"]  # both reads began before either ended
    assert happened[4:] == ["order b", "ordered b", "order c", "ordered c"]
    assert told(model, 1) == [
        ("call_1", "ordered b"),
        ("call_1_1", "a: 7"),
        ("call_1_2", "ordered c"),
        ("call_1_3", "b: 7"),
    ]


async def test_a_pause_in_a_batch_keeps_what_finished_and_numbers_calls_in_order(
    harness: Harness,
) -> None:
    ran: list[str] = []

    @tool(side_effects="read")
    async def confirm(item: str) -> str:
        """Ask a person whether an item is right."""
        answer = await current().ask(f"Is {item} right?")  # type: ignore[union-attr]
        ran.append(f"confirmed {item}")
        return str(answer)

    @tool(side_effects="read")
    async def look(item: str) -> str:
        """Look an item up (slowly)."""
        await asyncio.sleep(0.05)
        ran.append(f"looked {item}")
        return f"{item} found"

    @tool(side_effects="irreversible")
    def ship(item: str) -> str:
        """Ship an item."""
        ran.append(f"shipped {item}")
        return f"shipped {item}"

    batch = [("ship", {"item": "c"}), ("confirm", {"item": "a"}), ("look", {"item": "b"})]
    model = ScriptedChat([batch, "done"])
    agent = harness.wrap(ReAct(system="s", model=model), id="batch", tools=[confirm, look, ship])
    paused = await agent.run("x", user="u")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
    assert paused.interrupt.question == "Is a right?"
    assert ran == ["looked b"]  # the read beside the pause finished, and is journaled
    answered = await agent.resume(
        paused.interrupt.interrupt_id, "answer", answer="yes", reviewer="r"
    )
    assert answered.status is RunStatus.PAUSED and answered.interrupt is not None
    call = answered.interrupt.tool_call
    assert call is not None and call.tool == "ship"
    assert call.step == 1  # numbered in the model's order, though it runs after the reads
    done = await agent.resume(answered.interrupt.interrupt_id, "approve", reviewer="r")
    assert done.status is RunStatus.SUCCESS and done.answer == "done"
    assert ran == ["looked b", "confirmed a", "shipped c"]  # nothing ran twice
    assert len(model.requests) == 2  # the first step was replayed, twice
    assert told(model, 1) == [
        ("call_1", "shipped c"),
        ("call_1_1", "yes"),
        ("call_1_2", "b found"),
    ]


async def test_a_resumed_run_hands_each_call_its_output_whatever_order_they_finished_in(
    harness: Harness,
) -> None:
    counter = itertools.count()
    keys: list[str | None] = []
    rolls: list[str] = []

    @tool(side_effects="read")
    async def roll(die: str) -> int:
        """Roll a die (a slow one, and a quick one)."""
        keys.append(current().idempotency_key)  # type: ignore[union-attr]
        await asyncio.sleep(0.03 if die == "slow" else 0)
        rolls.append(die)
        return next(counter)

    @tool(side_effects="irreversible")
    def publish(results: str) -> str:
        """Publish the results."""
        return f"published {results}"

    batch = [("roll", {"die": "slow"}), ("roll", {"die": "quick"}), ("roll", {"die": "slow"})]
    model = ScriptedChat([batch, ("publish", {"results": "all"}), "done"])
    agent = harness.wrap(ReAct(system="s", model=model), id="dice", tools=[roll, publish])
    paused = await agent.run("x", user="u")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
    # the quick one finished first; the two identical calls ran one after another
    assert rolls == ["quick", "slow", "slow"]
    first = told(model, 1)
    assert first == [("call_1", "1"), ("call_1_1", "0"), ("call_1_2", "2")]
    assert len(set(keys)) == 3  # each identical call has its own idempotency key
    done = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="r")
    assert done.answer == "done"
    assert len(rolls) == 3  # replayed, not rolled again
    assert told(model, 2)[:3] == first  # the same outputs, call by call


async def test_identical_calls_made_at_once_by_any_framework_take_their_turn(
    harness: Harness,
) -> None:
    keys: list[str | None] = []

    @tool(side_effects="write")
    async def charge(order: str) -> str:
        """Charge an order."""
        keys.append(current().idempotency_key)  # type: ignore[union-attr]
        await asyncio.sleep(0)
        return f"charged {order} #{len(keys)}"

    async def both(input: str, agent: Runtime) -> list[Any]:
        return list(
            await asyncio.gather(
                agent.tools.call("charge", order="o1"), agent.tools.call("charge", order="o1")
            )
        )

    result = await harness.wrap(both, id="both", tools=[charge]).run("x", user="u")
    assert result.answer == ["charged o1 #1", "charged o1 #2"]
    assert keys[0] != keys[1]


class Overlapping(LocalRuns):
    """A run store whose heartbeat is slow, counting the heartbeats sent at once."""

    def __init__(self) -> None:
        super().__init__()
        self.active = 0
        self.most = 0
        self.saved: list[dict[str, Any]] = []

    async def heartbeat(
        self, run_id: str, worker_id: str, *, checkpoint: dict[str, Any] | None = None, **kw: Any
    ) -> Lease:
        self.active += 1
        self.most = max(self.most, self.active)
        await asyncio.sleep(0.01)
        self.active -= 1
        if checkpoint is not None:
            self.saved.append(checkpoint)
        return await super().heartbeat(run_id, worker_id, checkpoint=checkpoint, **kw)


async def test_progress_saved_by_calls_made_at_once_goes_one_save_at_a_time(
    harness: Harness,
) -> None:
    store = harness.runs = Overlapping()

    @tool(side_effects="write")
    async def file(name: str) -> str:
        """File a document."""
        return f"filed {name}"

    async def three(input: str, agent: Runtime) -> list[Any]:
        calls = [agent.tools.call("file", name=n) for n in "abc"]
        return list(await asyncio.gather(*calls))

    agent = harness.wrap(three, id="files", tools=[file])
    handle = await agent.start("x", user="u")
    assert await harness.worker([agent]).run_once()
    assert (await handle.result(timeout=5)).answer == ["filed a", "filed b", "filed c"]
    assert store.most == 1
    assert sum(len(v) for v in store.saved[-1]["calls"].values()) == 3  # the last save has all


async def test_cancelling_the_run_cancels_the_calls_running_beside_each_other(
    harness: Harness,
) -> None:
    started: list[str] = []
    cancelled: list[str] = []
    both = asyncio.Event()

    @tool(side_effects="read")
    async def wait(name: str) -> str:
        """Wait a long time."""
        started.append(name)
        if len(started) == 2:
            both.set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancelled.append(name)
            raise
        return "never"

    model = ScriptedChat([[("wait", {"name": "a"}), ("wait", {"name": "b"})]])
    agent = harness.wrap(ReAct(system="s", model=model), id="waits", tools=[wait])
    running = asyncio.create_task(agent.run("x", user="u"))
    await asyncio.wait_for(both.wait(), 5)
    [record] = [r async for r in harness.runs.iterate(agent_id="waits")]
    await agent.cancel(record.run_id, reason="enough")
    assert (await running).status is RunStatus.CANCELLED
    assert sorted(cancelled) == ["a", "b"]


# --------------------------------------------------------------------------- stopping
async def test_at_the_step_limit_the_model_answers_without_tools(harness: Harness) -> None:
    @tool(side_effects="read")
    def search(query: str) -> str:
        """Search."""
        return f"found {query}"

    model = ScriptedChat([("search", {"query": "a"}), ("search", {"query": "b"}), "partial"])
    agent = harness.wrap(ReAct(system="s", model=model, max_steps=2), id="limit", tools=[search])
    events = [e async for e in agent.stream("x", user="u")]
    finished = events[-1]
    assert finished.data["result"] == "partial"
    last = model.requests[-1]
    assert last["tool_choice"] == "none" and last["tools"]  # the same tools, none to be called
    assert last["messages"][-1] == {"role": "user", "content": LAST_STEP}
    warnings = [e.data for e in events if e.type is RunEventType.CUSTOM]
    assert warnings[0]["code"] == "max_steps"
    assert "stopped at its 2 steps" in warnings[0]["message"]


async def test_a_step_limit_with_no_tools_offered_asks_the_same_way(harness: Harness) -> None:
    model = ScriptedChat([("nothing", {}), "the end"])
    agent = harness.wrap(ReAct(system="s", model=model, max_steps=1), id="bare")
    assert (await agent.run("x", user="u")).answer == "the end"
    assert "tool_choice" not in model.requests[-1] and "tools" not in model.requests[-1]


async def test_steps_in_which_every_call_failed_stop_the_run(harness: Harness) -> None:
    @tool(side_effects="read")
    def flaky(n: int) -> str:
        """Fails for odd numbers."""
        if n % 2:
            raise ValueError("odd")
        return "even"

    model = ScriptedChat(
        [
            ("flaky", {"n": 1}),
            ("flaky", {"n": 2}),  # one success resets the streak
            [("flaky", {"n": 3}), ("nope", {})],
            ("flaky", {"n": 5}),
            ("flaky", {"x": "bad"}),
        ]
    )
    agent = harness.wrap(ReAct(system="s", model=model), id="flaky", tools=[flaky])
    result = await agent.run("x", user="u")
    assert result.status is RunStatus.ERROR and result.error is not None
    assert result.error.message == "ReAct stopped: every tool call failed in 3 consecutive steps"
    assert model.turns == []


async def test_the_tools_are_offered_in_a_stable_order(harness: Harness) -> None:
    tools = [tool(lambda: "z", name="zeta"), tool(lambda: "a", name="alpha")]
    model = ScriptedChat([("zeta", {}), "done"])
    agent = harness.wrap(ReAct(system="s", model=model), id="sorted", tools=tools)
    await agent.run("x", user="u")
    names = [[t["function"]["name"] for t in r["tools"]] for r in model.requests]
    assert names == [["alpha", "zeta"], ["alpha", "zeta"]]


def test_a_context_window_is_a_number_of_tokens() -> None:
    from trellis.contracts import ConfigurationError

    with pytest.raises(ConfigurationError, match="context_window"):
        ReAct(system="s", model="m", context_window=0)
