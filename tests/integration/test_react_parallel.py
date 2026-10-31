"""``ReAct`` with several calls in one step: every call at once (LangChain's own parallel
calls), the writes one at a time in the model's order, the tool messages in the model's order;
pauses inside a step; a resume that hands each call its own output whatever order they finished
in; the bridge safe under calls made at once from any framework; a call repeated in one step
run once; the step limit, the error streak and a stable tool order."""

from __future__ import annotations

import asyncio
import itertools
from typing import Any

from tests.support.arm import arm
from tests.support.models import ScriptedChat
from trellis import Budget, Harness, Hooks, ReAct, Runtime, current, tool
from trellis.contracts import RunEventType, RunStatus, ToolCall, ToolOutcome
from trellis.harness.runs import LocalRuns
from trellis.harness.tools.bridge import SAME_CALL
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
    agent = arm(harness.wrap(ReAct(system_prompt="s", model=model), id="shop"), [price, order])
    result = await agent.run("x", user="u")
    assert result.status is RunStatus.SUCCESS, result.error
    reads = [h for h in happened if h.startswith("price")]
    assert sorted(reads[:2]) == ["price a", "price b"]  # both reads began before either ended
    writes = [h for h in happened if h.startswith("order")]
    assert writes == ["order b", "ordered b", "order c", "ordered c"]  # in the model's order
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
    agent = arm(
        harness.wrap(ReAct(system_prompt="s", model=model), id="batch"), [confirm, look, ship]
    )
    result = await agent.run("x", user="u")
    asked: dict[str, int | None] = {}
    while result.status is RunStatus.PAUSED:
        interrupt = result.interrupt
        assert interrupt is not None
        assert "looked b" in ran  # the read beside the pause finished, and is kept
        call = interrupt.tool_call
        asked[interrupt.question] = call.step if call is not None else None
        decision = "approve" if call is not None else "answer"
        answer = None if call is not None else "yes"
        result = await agent.resume(interrupt.interrupt_id, decision, answer=answer, reviewer="r")
    assert result.status is RunStatus.SUCCESS and result.answer == "done", result.error
    # the two pauses of one step, one after the other; the call numbered in the model's order
    assert asked == {"Is a right?": None, "Approve ship? ship is irreversible.": 1}
    assert sorted(ran) == ["confirmed a", "looked b", "shipped c"]  # nothing ran twice
    assert len(model.requests) == 2  # resumed in place: the first step was not asked again
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
    agent = arm(harness.wrap(ReAct(system_prompt="s", model=model), id="dice"), [roll, publish])
    paused = await agent.run("x", user="u")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
    # the slow die rolled once (its repeat in the step points to it), beside the quick one
    assert sorted(rolls) == ["quick", "slow"]
    first = told(model, 1)
    # the quick one finished first
    assert first == [
        ("call_1", "1"),
        ("call_1_1", "0"),
        ("call_1_2", SAME_CALL.format(first="call_1")),
    ]
    assert len(set(keys)) == 2  # each call that ran has its own idempotency key
    done = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="r")
    assert done.answer == "done"
    assert len(rolls) == 2  # replayed, not rolled again
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

    result = await arm(harness.wrap(both, id="both"), [charge]).run("x", user="u")
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

    agent = arm(harness.wrap(three, id="files"), [file])
    handle = await agent.start("x", user="u")
    assert await harness.worker([agent]).run_once()
    assert (await handle.result(wait=5)).answer == ["filed a", "filed b", "filed c"]
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
    agent = arm(harness.wrap(ReAct(system_prompt="s", model=model), id="waits"), [wait])
    running = asyncio.create_task(agent.run("x", user="u"))
    await asyncio.wait_for(both.wait(), 5)
    [record] = [r async for r in harness.runs.iterate(agent_id="waits")]
    await agent.cancel(record.run_id, reason="enough")
    assert (await running).status is RunStatus.CANCELLED
    assert sorted(cancelled) == ["a", "b"]


# --------------------------------------------------------------------------- repeats
class Counted(Hooks):
    """How often the run's tool hooks saw a call."""

    def __init__(self) -> None:
        self.before = 0
        self.after = 0

    async def before_tool(self, call: ToolCall) -> None:
        self.before += 1

    async def after_tool(self, call: ToolCall, outcome: ToolOutcome) -> ToolOutcome:
        self.after += 1
        return outcome


async def test_a_call_repeated_in_one_step_runs_once_and_each_repeat_is_answered(
    harness: Harness,
) -> None:
    """A small model making one call 14 times in one step: it runs once — one budget charge,
    one pass of the hooks, one announcement — and every call gets its tool message."""
    stocked: list[str] = []

    @tool(side_effects="write")
    def style_stock(style: str, store: str) -> str:
        """Stock a style in a store."""
        stocked.append(style)
        return f"stocked {style}"

    hooks = Counted()
    # the same arguments, in any key order
    same = {"style": "ZZ999999", "store": "S1"}
    step = [("style_stock", same)] * 7 + [("style_stock", dict(reversed(same.items())))] * 7
    model = ScriptedChat([step, "done"])
    react = ReAct(system_prompt="s", model=model)
    # a second charge would reach the budget: the run would end PARTIAL
    wrapped = harness.wrap(react, id="repeats", hooks=[hooks], budget=Budget(max_tool_calls=2))
    events = [e async for e in arm(wrapped, [style_stock]).stream("x", user="u")]
    finished = events[-1]
    assert finished.outcome is not None and finished.outcome.value == "success", finished
    assert finished.data["result"] == "done"
    assert stocked == ["ZZ999999"] and (hooks.before, hooks.after) == (1, 1)
    repeat = SAME_CALL.format(first="call_1")
    assert told(model, 1) == [
        ("call_1", "stocked ZZ999999"),
        *[(f"call_1_{n}", repeat) for n in range(1, 14)],
    ]
    results = [e.data for e in events if e.type is RunEventType.TOOL_CALL_RESULT]
    assert sorted(str(r.get("same_as")) for r in results) == ["None"] + ["call_1"] * 13
    assert {r["status"] for r in results} == {"ok"}
    custom = [e.data["name"] for e in events if e.type is RunEventType.CUSTOM]
    assert custom.count("tool_notice") == 1


async def test_an_irreversible_call_repeated_in_one_step_is_approved_once(
    harness: Harness,
) -> None:
    wired: list[str] = []

    @tool(side_effects="irreversible")
    def wire(to: str) -> str:
        """Wire money."""
        wired.append(to)
        return f"wired {to}"

    model = ScriptedChat([[("wire", {"to": "ada"})] * 3, "done"])
    agent = arm(harness.wrap(ReAct(system_prompt="s", model=model), id="wires"), [wire])
    paused = await agent.run("x", user="u")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None, paused.error
    assert paused.interrupt.question.startswith("Approve wire?") and wired == []
    done = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="r")
    assert done.status is RunStatus.SUCCESS and done.answer == "done", done.error
    assert wired == ["ada"] and len(model.requests) == 2  # resumed in place, asked once
    repeat = SAME_CALL.format(first="call_1")
    assert told(model, 1) == [
        ("call_1", "wired ada"),
        ("call_1_1", repeat),
        ("call_1_2", repeat),
    ]


async def test_calls_with_other_arguments_all_run_and_a_repeat_in_a_later_step_runs_again(
    harness: Harness,
) -> None:
    ran: list[str] = []

    @tool(side_effects="write")
    def note(text: str) -> str:
        """Take a note."""
        ran.append(text)
        return f"noted {text}"

    hooks = Counted()
    model = ScriptedChat(
        [
            [("note", {"text": "a"}), ("note", {"text": "b"}), ("note", {"text": "a"})],
            ("note", {"text": "a"}),  # the next step: StallGuard's, not a repeat
            "done",
        ]
    )
    agent = arm(
        harness.wrap(ReAct(system_prompt="s", model=model), id="notes", hooks=[hooks]), [note]
    )
    events = [e async for e in agent.stream("x", user="u")]
    assert events[-1].data["result"] == "done"
    assert ran == ["a", "b", "a"] and hooks.before == 3
    assert told(model, 1) == [
        ("call_1", "noted a"),
        ("call_1_1", "noted b"),
        ("call_1_2", SAME_CALL.format(first="call_1")),
    ]
    assert told(model, 2)[-1] == ("call_2", "noted a")
    notices = [
        e for e in events if e.type is RunEventType.CUSTOM and e.data["name"] == "tool_notice"
    ]
    assert len(notices) == 3  # each write that ran announced once


async def test_a_failing_call_repeated_in_one_step_still_counts_toward_the_error_streak(
    harness: Harness,
) -> None:
    @tool(side_effects="read")
    def broken(n: int) -> str:
        """Fails."""
        raise ValueError("down")

    turns: list[Any] = [[("broken", {"n": n}), ("broken", {"n": n})] for n in range(3)]
    model = ScriptedChat([*turns, "the tool is down"])
    agent = arm(harness.wrap(ReAct(system_prompt="s", model=model), id="broken"), [broken])
    result = await agent.run("x", user="u")
    # three failing steps: the model is told so and answers with no tool (StallGuard)
    assert result.status is RunStatus.SUCCESS and result.answer == "the tool is down"
    assert model.requests[-1]["tool_choice"] == "none"


# --------------------------------------------------------------------------- stopping
async def test_at_the_step_limit_the_run_ends_with_langchains_message(harness: Harness) -> None:
    @tool(side_effects="read")
    def search(query: str) -> str:
        """Search."""
        return f"found {query}"

    model = ScriptedChat([("search", {"query": "a"}), ("search", {"query": "b"}), "partial"])
    agent = arm(
        harness.wrap(ReAct(system_prompt="s", model=model, max_steps=2), id="limit"), [search]
    )
    events = [e async for e in agent.stream("x", user="u")]
    finished = events[-1]
    # LangChain's ModelCallLimitMiddleware: no call past max_steps, no forced answer
    assert finished.data["result"] == "Model call limits exceeded: thread limit (2/2)"
    assert len(model.requests) == 2


async def test_steps_in_which_every_call_failed_and_no_answer_stop_the_run(
    harness: Harness,
) -> None:
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
            ("flaky", {"n": 7}),  # told every call failed, it calls again: no answer
        ]
    )
    agent = arm(harness.wrap(ReAct(system_prompt="s", model=model), id="flaky"), [flaky])
    result = await agent.run("x", user="u")
    assert result.status is RunStatus.ERROR and result.error is not None
    assert result.error.message == "stopped: every tool call failed in 3 consecutive steps"
    assert model.turns == []


async def test_the_tools_are_offered_in_a_stable_order(harness: Harness) -> None:
    tools = [tool(lambda: "z", name="zeta"), tool(lambda: "a", name="alpha")]
    model = ScriptedChat([("zeta", {}), "done"])
    agent = arm(harness.wrap(ReAct(system_prompt="s", model=model), id="sorted"), tools)
    await agent.run("x", user="u")
    names = [[t["function"]["name"] for t in r["tools"]] for r in model.requests]
    assert names == [["alpha", "zeta"]] * 2
