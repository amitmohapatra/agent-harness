"""A tool the harness runs that says it ``write``\\s asks a person before it runs, as Claude Code
asks: allow once, allow for this conversation, or deny. "This conversation" is every later run
of the same thread for the same user — a resumed run, another worker and a child run included —
never another thread, another user or another tool (an ``irreversible`` tool asks every time:
``test_approvals``).
A schedule set in chat allows the writes its task names when it is confirmed, and its fired runs
honour that; an uncovered write in a fired run pauses in the chat's thread. The frameworks' own
tools keep their own way (``test_govern_framework_tools``, ``test_claude``, ``test_kimi``)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests.support.adapters import BUILDERS
from tests.support.arm import arm
from tests.support.planned import Call
from trellis import Harness, Runtime, Settings, tool
from trellis.contracts import InterruptReason, RunStatus, ToolSpec
from trellis.harness import conversation
from trellis.harness.governance.decision import CONVERSATION, DENY, ONCE
from trellis.harness.runs import LocalRuns
from trellis.harness.schedules import ALLOW, ALLOWED, EACH_TIME, NEITHER
from trellis.harness.tools import bridge
from trellis.harness.tools.convert import text_of

done: list[str] = []
THREAD = "chat-1"
#: what plain chat reads after the question: the three choices
THREE = "Allow once, allow for this conversation, or deny."


@tool(side_effects="write")
def pay(amount: int) -> str:
    """Pay an amount."""
    done.append(f"pay {amount}")
    return f"paid {amount}"


@tool(side_effects="write")
def refund(amount: int) -> str:
    """Refund an amount."""
    done.append(f"refund {amount}")
    return f"refunded {amount}"


@tool(side_effects="irreversible")
def wire(amount: int) -> str:
    """Wire an amount."""
    done.append(f"wire {amount}")
    return f"wired {amount}"


@pytest.fixture(autouse=True)
def _reset() -> None:
    done.clear()


def calling(*calls: tuple[str, int]) -> Any:
    """A function target that makes ``calls`` in order and answers what they returned."""

    async def fn(input: str, agent: Runtime) -> list[str]:
        return [text_of(await agent.tools.call(name, amount=n)) for name, n in calls]

    return fn


async def approved(agent: Any, result: Any, remember: str = "once") -> Any:
    assert result.status is RunStatus.PAUSED and result.interrupt is not None, result
    return await agent.resume(
        result.interrupt.interrupt_id, "approve", reviewer="ada", remember=remember
    )


# --------------------------------------------------------------------------- a write asks


@pytest.mark.parametrize("framework", list(BUILDERS))
async def test_a_write_asks_with_three_choices_and_allow_once_covers_that_call_on_every_adapter(
    harness: Harness, framework: str, tmp_path: Path
) -> None:
    plan: list[Call] = [("pay", {"amount": 5}), ("pay", {"amount": 6})]
    target, tools = await BUILDERS[framework](harness, [pay], tmp_path, plan)
    agent = arm(harness.wrap(target, id=f"payer-{framework}"), tools)
    first = await agent.run("pay twice", user="ada")
    asked = first.interrupt
    assert asked is not None and asked.reason is InterruptReason.APPROVAL, first
    assert done == []  # asked before it ran
    assert asked.question == f"Approve pay? pay is write. {THREE}"
    assert list(asked.options) == [ONCE, CONVERSATION, DENY]
    second = await approved(agent, first)
    assert done == ["pay 5"] and second.status is RunStatus.PAUSED, second  # once: one call
    finished = await approved(agent, second)
    assert finished.status is RunStatus.SUCCESS and done == ["pay 5", "pay 6"], finished


async def test_deny_never_runs_and_a_picked_option_is_the_decision_it_names(
    harness: Harness,
) -> None:
    agent = arm(harness.wrap(calling(("pay", 1), ("pay", 2)), id="picks"), [pay])
    first = await agent.run("go", user="ada", thread=THREAD)
    assert first.interrupt is not None
    denied = await agent.resume(
        first.interrupt.interrupt_id, "answer", answer=DENY.value, reviewer="ada"
    )
    assert denied.interrupt is not None and done == []  # denied: never ran; the next asks
    finished = await agent.resume(
        denied.interrupt.interrupt_id, "answer", answer=CONVERSATION.value, reviewer="ada"
    )
    assert finished.status is RunStatus.SUCCESS and done == ["pay 2"], finished
    assert finished.answer[0] == "pay was not run: the approver rejected it"
    again = await agent.run("go", user="ada", thread=THREAD)  # the pick reached the conversation
    assert again.status is RunStatus.SUCCESS and done == ["pay 2", "pay 1", "pay 2"], again


# --------------------------------------------------------------------------- this conversation


async def test_allow_for_this_conversation_covers_that_tool_in_later_runs_of_that_thread_only(
    harness: Harness,
) -> None:
    payer = arm(harness.wrap(calling(("pay", 1)), id="payer"), [pay, refund])
    allowed = await approved(payer, await payer.run("go", user="ada", thread=THREAD), "run")
    assert allowed.status is RunStatus.SUCCESS and done == ["pay 1"]
    later = await payer.run("go", user="ada", thread=THREAD)
    assert later.status is RunStatus.SUCCESS and done == ["pay 1", "pay 1"], later
    assert (await payer.run("go", user="ada", thread="chat-2")).status is RunStatus.PAUSED
    assert (await payer.run("go", user="bob", thread=THREAD)).status is RunStatus.PAUSED
    refunder = arm(harness.wrap(calling(("refund", 1)), id="refunder"), [pay, refund])
    assert (await refunder.run("go", user="ada", thread=THREAD)).status is RunStatus.PAUSED
    assert done == ["pay 1", "pay 1"]  # another tool, thread or user: asked, not run


async def test_a_resumed_run_another_worker_and_a_child_run_see_the_same_approval() -> None:
    store = LocalRuns()
    async with Harness(config=Settings(), runs=store) as one:
        payer = arm(one.wrap(calling(("pay", 1)), id="payer"), [pay, refund])
        await approved(payer, await payer.run("go", user="ada", thread=THREAD), "run")
        both = arm(one.wrap(calling(("refund", 2), ("pay", 3)), id="both"), [pay, refund])
        paused = await both.run("go", user="ada", thread=THREAD)  # refund asks; pay then not
        resumed = await approved(both, paused)
        assert resumed.status is RunStatus.SUCCESS, resumed
        assert done == ["pay 1", "refund 2", "pay 3"]
    async with Harness(config=Settings(), runs=store) as other:  # another worker, same store
        payer = arm(other.wrap(calling(("pay", 4)), id="payer"), [pay])
        assert (await payer.run("go", user="ada", thread=THREAD)).status is RunStatus.SUCCESS
        child = arm(other.wrap(calling(("pay", 5)), id="child"), [pay])

        async def parent(input: str, agent: Runtime) -> Any:
            return (await child.run("pay", user="ada")).answer

        parenting = other.wrap(parent, id="parent")
        result = await parenting.run("go", user="ada", thread=THREAD)
        assert result.status is RunStatus.SUCCESS, result
        assert done[-2:] == ["pay 4", "pay 5"]


async def test_what_a_conversation_allowed_is_read_page_by_page(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(conversation, "RESOLUTIONS_PAGE", 1)
    both = arm(harness.wrap(calling(("pay", 1), ("refund", 2)), id="both"), [pay, refund])
    first = await both.run("go", user="ada", thread=THREAD)
    await approved(both, await approved(both, first, "run"), "run")  # two answers, two pages
    again = await both.run("go", user="ada", thread=THREAD)
    assert again.status is RunStatus.SUCCESS and len(done) == 4, again


# --------------------------------------------------------------------------- what does not ask


async def test_a_frameworks_own_write_is_announced_as_before_not_asked(harness: Harness) -> None:
    """Claude Code's and Kimi's built-ins go through ``bridge.permitted`` under their own
    permission flow: a ``write`` there is announced, as it was."""
    seen: list[Any] = []

    async def fn(input: str, agent: Runtime) -> Any:
        seen.append(await bridge.permitted(ToolSpec(name="Write", side_effects="write"), {}))
        return "ok"

    result = await harness.wrap(fn, id="claude-like").run("go", user="ada")
    assert result.status is RunStatus.SUCCESS and seen[0].tool == "Write", result


# --------------------------------------------------------------------------- schedules


class Chat:
    """In chat: schedules its task naming the write it will use; a fired run calls ``calls``."""

    def __init__(self, *calls: tuple[str, int]) -> None:
        self.calls = calls

    async def __call__(self, input: str, agent: Runtime) -> Any:
        if input.startswith("Scheduled task:"):
            return [text_of(await agent.tools.call(n, amount=a)) for n, a in self.calls]
        args = {"task": "pay the rent", "when": "0 8 1 * *", "tools": ["pay", "wire", "nope"]}
        return text_of(await agent.tools.call("schedule_task", **args))


async def fired(h: Harness, agent: Any) -> Any:
    [schedule] = h.runs._schedules.values()  # type: ignore[union-attr]
    run_id = (await h.runs.schedules.fire(schedule.schedule_id)).run_id
    assert await h.worker([agent]).run_once() is True
    return await h.runs.get(run_id)


@pytest.mark.parametrize("picked", [ALLOW, EACH_TIME])
async def test_a_schedule_allows_the_writes_its_task_names_when_it_is_confirmed(
    picked: str,
) -> None:
    async with Harness(config=Settings()) as h:
        agent = arm(h.wrap(Chat(("pay", 1), ("refund", 2)), id="desk"), [pay, refund, wire])
        asked = await agent.run("pay my rent monthly", user="ada", thread=THREAD)
        interrupt = asked.interrupt
        assert interrupt is not None, asked
        # only the write it names that asks: not an irreversible one, not an unknown name
        assert "pay" in interrupt.question and "wire" not in interrupt.question
        assert interrupt.option_values == [ALLOW, EACH_TIME, NEITHER]
        made = await agent.resume(interrupt.interrupt_id, "answer", answer=picked, reviewer="ada")
        assert made.status is RunStatus.SUCCESS, made
        [schedule] = h.runs._schedules.values()  # type: ignore[attr-defined]
        assert schedule.metadata.get(ALLOWED, []) == (["pay"] if picked == ALLOW else [])
        run = await fired(h, agent)
        # an uncovered write pauses, in the chat's thread, for its user: never run silently
        assert run.status is RunStatus.PAUSED and run.thread_id == THREAD, run
        assert run.awaiting.tool_call.tool == ("refund" if picked == ALLOW else "pay")
        assert done == (["pay 1"] if picked == ALLOW else [])


async def test_cancelling_the_confirmation_makes_no_schedule() -> None:
    async with Harness(config=Settings()) as h:
        agent = arm(h.wrap(Chat(), id="desk"), [pay])
        asked = await agent.run("pay my rent monthly", user="ada", thread=THREAD)
        assert asked.interrupt is not None
        await agent.resume(asked.interrupt.interrupt_id, "answer", answer=NEITHER, reviewer="ada")
        assert not h.runs._schedules  # type: ignore[attr-defined]
