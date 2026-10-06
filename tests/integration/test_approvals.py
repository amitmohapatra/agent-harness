"""W5 approvals on every adapter, ``ReAct`` and Way 2: an approval rule in code — a
``before_tool`` hook asking by the call's arguments (whose it is, on its own screen), journaled,
beside the tool's ``side_effects`` and the catalog's ``approve_when``; a reviewer's comment, kept
with the decision; an approval remembered for the rest of the run, never for another; an
external tool, whose result comes from outside the run."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from tests.support.adapters import BUILDERS
from tests.support.memory import FakeMemoryService
from tests.support.planned import Call
from trellis import Ask, Harness, Hooks, Runtime, tool
from trellis.contracts import (
    ConfigurationError,
    InterruptReason,
    RunEventType,
    RunStatus,
    ToolCall,
    ToolError,
)
from trellis.harness.events import DECISION
from trellis.harness.governance import Decision, Governance, Rejected, governed
from trellis.testing import Decide, Reviewer

refunded: list[int] = []
#: what was paid, in order (the approval rules in code below)
paid: list[int] = []


@tool(side_effects="irreversible")
def wire(amount: int) -> str:
    """Wire an amount."""
    refunded.append(amount)
    return f"wired {amount}"


@tool(side_effects="write", external=True)
def sign(contract: str) -> str:
    """Have a contract signed: a person signs it in the e-signature system."""
    raise AssertionError("an external tool's function never runs")


@pytest.fixture(autouse=True)
def _reset() -> None:
    refunded.clear()
    paid.clear()


def decisions(events: list[Any]) -> list[dict[str, Any]]:
    return [
        e.data for e in events if e.type is RunEventType.CUSTOM and e.data.get("name") == DECISION
    ]


# --------------------------------------------------------------------------- rules in code: hooks


class Threshold(Hooks):
    """An approval rule in code, by arguments: over 100 goes to finance on its own screen
    (``Ask(assignee=, component=, props=)``); anything else is governance's."""

    def __init__(self) -> None:
        self.ruled: list[int] = []

    async def before_tool(self, call: ToolCall) -> Ask | None:
        amount = call.args["amount"]
        self.ruled.append(amount)
        if amount > 100:
            return Ask(
                f"Pay {amount}?",
                assignee="role:finance",
                component="pay-review",
                props={"amount": amount},
            )
        return None


@tool(side_effects="write")
def pay(amount: int) -> str:
    """Pay an amount."""
    paid.append(amount)
    return f"paid {amount}"


@pytest.mark.parametrize("framework", list(BUILDERS))
async def test_a_hook_and_the_catalog_rule_by_arguments_on_every_adapter(
    memory_harness: Harness, memory_service: FakeMemoryService, framework: str, tmp_path: Path
) -> None:
    """A small amount runs unasked (the tool is ``write``), a middle one is asked by the
    catalog's ``approve_when``, a large one by the hook on finance's screen; the hook's verdict
    is journaled: a re-run reads it."""
    memory_service.catalog = {"pay": {"side_effects": "write", "approve_when": "amount > 20"}}
    rule = Threshold()
    plan: list[Call] = [("pay", {"amount": a}) for a in (5, 50, 500)]
    target, tools = await BUILDERS[framework](memory_harness, [pay], tmp_path, plan)
    agent = memory_harness.wrap(target, id=f"payer-{framework}", tools=tools, hooks=[rule])
    first = await agent.run("pay three", user="ada")
    assert first.interrupt is not None, first
    assert first.interrupt.question == "Approve pay? amount > 20."  # the catalog's rule
    assert first.interrupt.assignee is None and first.interrupt.component is None
    assert paid == [5]  # small: nobody asked
    second = await agent.resume(first.interrupt.interrupt_id, "approve", reviewer="lee")
    asked = second.interrupt
    assert asked is not None and asked.reason is InterruptReason.APPROVAL, second
    assert asked.question == "Approve pay? Pay 500?"
    assert asked.tool_call is not None and asked.tool_call.args == {"amount": 500}
    assert (asked.assignee, asked.component, asked.props) == (
        "role:finance",
        "pay-review",
        {"amount": 500},
    )
    done = await agent.resume(asked.interrupt_id, "approve", reviewer="cfo")
    assert done.status is RunStatus.SUCCESS and done.answer == "Done. paid 500", done
    assert paid == [5, 50, 500]
    assert sorted(rule.ruled) == [5, 50, 500]  # each call ruled once: a re-run reads the journal


class Escalating(Hooks):
    """Every payment is asked about; over 500 of finance."""

    async def before_tool(self, call: ToolCall) -> Ask:
        amount = call.args["amount"]
        return Ask(f"Pay {amount}?", assignee="role:finance" if amount > 500 else None)


@pytest.mark.parametrize("framework", list(BUILDERS))
async def test_remembering_a_hooks_ask_covers_later_asks_of_the_same_assignee_on_every_adapter(
    harness: Harness, framework: str, tmp_path: Path
) -> None:
    plan: list[Call] = [("pay", {"amount": a}) for a in (50, 60, 900, 950)]
    target, tools = await BUILDERS[framework](harness, [pay], tmp_path, plan)
    agent = harness.wrap(target, id=f"remembering-{framework}", tools=tools, hooks=[Escalating()])
    first = await agent.run("pay four", user="ada")
    assert first.interrupt is not None and first.interrupt.assignee is None, first
    finance = await agent.resume(
        first.interrupt.interrupt_id, "approve", reviewer="lee", remember="run"
    )
    asked = finance.interrupt  # 60 was not asked; 900 is finance's: still asked
    assert asked is not None and asked.assignee == "role:finance", finance
    assert asked.question == "Approve pay? Pay 900?" and paid == [50, 60]
    done = await agent.resume(asked.interrupt_id, "approve", reviewer="cfo", remember="run")
    assert done.status is RunStatus.SUCCESS and done.answer == "Done. paid 950", done
    assert paid == [50, 60, 900, 950]  # 950: finance's again, remembered
    other = await agent.run("pay four", user="ada")  # never across runs
    assert other.status is RunStatus.PAUSED


async def test_governed_asks_by_arguments_with_a_hook() -> None:
    """Way 2: ``governed(..., hooks=)`` — a small amount runs unasked, a large one is asked
    on finance's screen (``on_ask`` gets the ``Ask``'s assignee, component and props), and
    an ``irreversible`` tool the hook leaves alone is governance's to ask about."""
    asked: list[Decision] = []

    async def on_ask(decision: Decision) -> bool:
        asked.append(decision)
        return decision.props is None

    def settle(amount: int) -> str:
        return f"settled {amount}"

    gov = Governance()
    small = governed(settle, gov, side_effects="write", on_ask=on_ask, hooks=[Threshold()])
    assert await small(amount=5) == "settled 5"  # nobody asked
    with pytest.raises(Rejected):
        await small(amount=500)  # the hook asked, on finance's screen
    risky = governed(settle, gov, side_effects="irreversible", on_ask=on_ask, hooks=[Threshold()])
    assert await risky(amount=50) == "settled 50"  # governance asked
    assert [(d.question, d.assignee, d.component, d.props) for d in asked] == [
        ("Approve settle? Pay 500?", "role:finance", "pay-review", {"amount": 500}),
        ("Approve settle? settle is irreversible.", None, None, None),
    ]


# --------------------------------------------------------------------------- remember, comment


@pytest.mark.parametrize("framework", list(BUILDERS))
async def test_an_approval_remembered_for_the_run_covers_the_tools_later_calls_only_there(
    harness: Harness, framework: str, tmp_path: Path
) -> None:
    plan: list[Call] = [("wire", {"amount": 50}), ("wire", {"amount": 60})]
    target, tools = await BUILDERS[framework](harness, [wire], tmp_path, plan)
    agent = harness.wrap(target, id=f"wires-{framework}", tools=tools)
    first = await agent.run("wire twice", user="ada")
    assert first.interrupt is not None, first
    resumed = await agent.resume(
        first.interrupt.interrupt_id, "approve", reviewer="lee", remember="run", comment="fine"
    )
    assert resumed.status is RunStatus.SUCCESS and resumed.answer == "Done. wired 60", resumed
    assert refunded == [50, 60]  # the second call was not asked about
    record = await harness.runs.get(first.run_id)
    assert record is not None and record.last_resolution is not None
    assert (record.last_resolution.remember, record.last_resolution.comment) == ("run", "fine")
    other = await agent.run("wire twice", user="ada")  # never across runs
    assert other.status is RunStatus.PAUSED


async def test_a_remembered_approval_and_a_comment_are_events_of_the_next_attempt(
    harness: Harness,
) -> None:
    async def fn(input: str, agent: Runtime) -> Any:
        return [await agent.tools.call("wire", amount=a) for a in (1, 2)]

    agent = harness.wrap(fn, id="wiring", tools=[wire])
    paused = await agent.run("go", user="ada")
    assert paused.interrupt is not None
    seen: list[Any] = []

    async def watch() -> None:
        async for event in agent.events(paused.run_id):
            seen.append(event)

    watcher = asyncio.create_task(watch())
    await asyncio.sleep(0)  # following the run before it goes on
    await agent.resume(
        paused.interrupt.interrupt_id,
        "approve",
        reviewer="lee",
        remember="run",
        comment="ok for today",
    )
    await asyncio.wait_for(watcher, 5)
    said = decisions(seen)
    assert said[0] == {
        "name": DECISION,
        "interrupt_id": paused.interrupt.interrupt_id,
        "decision": "APPROVE",
        "reviewer": "lee",
        "comment": "ok for today",
        "remember": "run",
    }
    assert said[1] == {"name": DECISION, "tool": "wire", "decision": "APPROVE", "remembered": True}
    assert seen[-1].data["result"] == ["wired 1", "wired 2"]


async def test_a_rejection_with_a_comment_tells_the_model_why(harness: Harness) -> None:
    async def fn(input: str, agent: Runtime) -> Any:
        return await agent.tools.call("wire", amount=900)

    agent = harness.wrap(fn, id="rejected", tools=[wire])
    paused = await agent.run("go", user="ada")
    assert paused.interrupt is not None
    done = await agent.resume(
        paused.interrupt.interrupt_id, "reject", reviewer="cfo", comment="over budget"
    )
    assert done.answer == "wire was not run: the approver rejected it (over budget)"


async def test_only_an_approval_of_a_tool_call_is_remembered(harness: Harness) -> None:
    async def fn(input: str, agent: Runtime) -> Any:
        return await agent.ask("Which?")

    agent = harness.wrap(fn, id="asks")
    paused = await agent.run("go", user="ada")
    assert paused.interrupt is not None
    with pytest.raises(ConfigurationError, match="only an approval is remembered"):
        await agent.resume(paused.interrupt.interrupt_id, "answer", reviewer="a", remember="run")
    with pytest.raises(ConfigurationError, match="of a tool call is remembered"):
        await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="a", remember="run")
    with pytest.raises(ConfigurationError, match="name the decision"):
        await agent.resume(paused.interrupt.interrupt_id, reviewer="a")
    with pytest.raises(ConfigurationError, match="pass reviewer="):
        await agent.resume(paused.interrupt.interrupt_id, "answer", answer="x")
    by_run = await agent.resume(paused.run_id, "answer", answer="that", reviewer="a")
    assert by_run.answer == "that"  # the run's id answers what it waits on


# --------------------------------------------------------------------------- external results


@pytest.mark.parametrize("framework", list(BUILDERS))
async def test_an_external_tools_result_comes_from_outside_on_every_adapter(
    harness: Harness, framework: str, tmp_path: Path
) -> None:
    plan: list[Call] = [("sign", {"contract": "c-1"})]
    target, tools = await BUILDERS[framework](harness, [sign], tmp_path, plan)
    agent = harness.wrap(target, id=f"signing-{framework}", tools=tools)
    paused = await agent.run("get c-1 signed", user="ada")
    asked = paused.interrupt
    assert asked is not None and asked.reason is InterruptReason.QUESTION, paused
    assert asked.tool_call is not None and asked.tool_call.tool == "sign"
    assert asked.tool_call.args == {"contract": "c-1"}
    assert asked.expects == {"type": "string"}  # the function's return annotation
    with pytest.raises(ConfigurationError, match="does not fit"):
        await agent.resume(paused.run_id, result=7)
    with pytest.raises(ConfigurationError, match="a result answers the call"):
        await agent.resume(paused.run_id, "approve", result="x")
    done = await agent.resume(paused.run_id, result="signed by ada")
    assert done.status is RunStatus.SUCCESS and done.answer == "Done. signed by ada", done


async def test_a_refused_external_result_is_an_error_the_model_reads(harness: Harness) -> None:
    @tool(external=True)
    def scan(page: int):  # type: ignore[no-untyped-def]  # no annotation: any result
        """Scan a page."""

    async def fn(input: str, agent: Runtime) -> Any:
        return await agent.tools.call("scan", page=1)

    agent = harness.wrap(fn, id="scanning", tools=[scan])
    paused = await agent.run("scan", user="ada")
    assert paused.interrupt is not None and paused.interrupt.expects is None
    done = await agent.resume(paused.interrupt.interrupt_id, "reject", reviewer="ops")
    assert done.answer == "scan failed: scan was not done: its result was refused"


async def test_an_external_tool_called_outside_a_run_says_so() -> None:
    with pytest.raises(ToolError, match="is an external tool"):
        await sign.tool.run({"contract": "c-1"})


# --------------------------------------------------------------------------- a scripted reviewer


async def test_reviewer_answers_approvals_and_questions_by_script(harness: Harness) -> None:
    async def fn(input: str, agent: Runtime) -> Any:
        wired = await agent.tools.call("wire", amount=50)
        plan = await agent.ask("Which plan?", options=["a", "b"])
        return f"{wired}, {plan}"

    agent = harness.wrap(fn, id="scripted", tools=[wire])
    reviewer = Reviewer({"wire": Decide("approve", comment="ok"), "Which plan?": "b"})
    done = await reviewer.run(agent, "go", user="ada")
    assert done.answer == "wired 50, b"
    assert reviewer.answered == [("wire", "APPROVE", None), ("Which plan?", "ANSWER", "b")]
