"""W5 approvals on every adapter, ``ReAct`` and Way 2: a tool's approval function
(``tool(approval=fn)``: ``None``, ``True`` or an ``Ask`` with its own screen), journaled; a
reviewer's comment, kept with the decision; an approval remembered for the rest of the run, never
for another; an external tool, whose result comes from outside the run."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from tests.support.adapters import BUILDERS
from tests.support.planned import Call
from trellis import Ask, Harness, Runtime, tool
from trellis.contracts import (
    ConfigurationError,
    InterruptReason,
    RunEventType,
    RunStatus,
    ToolError,
)
from trellis.harness.events import DECISION
from trellis.harness.governance import Decision, Governance, Rejected, governed
from trellis.testing import Decide, Reviewer

refunded: list[int] = []
ruled: list[int] = []


def refund_rule(args: dict[str, Any]) -> Ask | bool | None:
    """Small refunds are fine, large ones go to finance on their own screen, the rest is
    governance's (an irreversible tool: it asks)."""
    amount = args["amount"]
    ruled.append(amount)
    if amount <= 10:
        return True
    if amount > 100:
        return Ask(
            f"Refund {amount}?",
            assignee="role:finance",
            component="refund-review",
            props={"amount": amount},
        )
    return None


@tool(side_effects="irreversible", approval=refund_rule)
def refund(amount: int) -> str:
    """Refund an amount."""
    refunded.append(amount)
    return f"refunded {amount}"


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
    ruled.clear()


def decisions(events: list[Any]) -> list[dict[str, Any]]:
    return [
        e.data for e in events if e.type is RunEventType.CUSTOM and e.data.get("name") == DECISION
    ]


# --------------------------------------------------------------------------- approval=


@pytest.mark.parametrize("framework", list(BUILDERS))
async def test_an_approval_function_approves_asks_or_leaves_it_to_governance_on_every_adapter(
    harness: Harness, framework: str, tmp_path: Path
) -> None:
    plan: list[Call] = [("refund", {"amount": a}) for a in (5, 50, 500)]
    target, tools = await BUILDERS[framework](harness, [refund], tmp_path, plan)
    agent = harness.wrap(target, id=f"refunds-{framework}", tools=tools)
    first = await agent.run("refund three", user="ada")
    assert first.interrupt is not None, first
    assert first.interrupt.question == "Approve refund? refund is irreversible."
    assert first.interrupt.component is None and refunded == [5]  # 5: approved by the rule
    second = await agent.resume(first.interrupt.interrupt_id, "approve", reviewer="lee")
    asked = second.interrupt
    assert asked is not None and asked.reason is InterruptReason.APPROVAL, second
    assert asked.question == "Approve refund? Refund 500?"
    assert (asked.assignee, asked.component, asked.props) == (
        "role:finance",
        "refund-review",
        {"amount": 500},
    )
    done = await agent.resume(asked.interrupt_id, "approve", reviewer="cfo")
    assert done.status is RunStatus.SUCCESS and done.answer == "Done. refunded 500", done
    assert refunded == [5, 50, 500]
    assert sorted(ruled) == [5, 50, 500]  # each call ruled once: a re-run reads the journal


async def test_an_approval_function_may_be_async_and_must_say_none_true_or_ask(
    harness: Harness,
) -> None:
    async def rule(args: dict[str, Any]) -> Any:
        return args["say"]

    @tool(side_effects="irreversible", approval=rule)
    def act(say: Any) -> str:
        """Act."""
        return "acted"

    async def fn(input: Any, agent: Runtime) -> Any:
        return await agent.tools.call("act", say=input)

    agent = harness.wrap(fn, id="acting", tools=[act])
    assert (await agent.run(True, user="u")).answer == "acted"
    failed = await agent.run("yes", user="u")
    assert failed.status is RunStatus.ERROR and failed.error is not None
    assert "returned 'yes': return None" in failed.error.message


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


# --------------------------------------------------------------------------- Way 2


async def test_governed_takes_the_approval_function_too() -> None:
    asked: list[Decision] = []

    async def on_ask(decision: Decision) -> bool:
        asked.append(decision)
        return decision.props is None

    def pay(amount: int) -> str:
        return f"paid {amount}"

    call = governed(
        pay,
        Governance(),
        side_effects="irreversible",
        on_ask=on_ask,
        approval=refund_rule,
    )
    assert await call(amount=5) == "paid 5"  # approved by the rule: nobody asked
    assert await call(amount=50) == "paid 50"  # governance asked
    with pytest.raises(Rejected):
        await call(amount=500)  # the rule asked, on its screen
    assert [(d.assignee, d.component, d.props) for d in asked] == [
        (None, None, None),
        ("role:finance", "refund-review", {"amount": 500}),
    ]


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


async def test_a_remembered_approval_does_not_answer_a_question_asked_of_someone_else(
    harness: Harness,
) -> None:
    async def fn(input: str, agent: Runtime) -> Any:
        return [await agent.tools.call("refund", amount=a) for a in (50, 60, 900)]

    agent = harness.wrap(fn, id="escalating", tools=[refund])
    paused = await agent.run("go", user="ada")
    assert paused.interrupt is not None and paused.interrupt.assignee is None
    asked = await agent.resume(
        paused.interrupt.interrupt_id, "approve", reviewer="lee", remember="run"
    )
    assert asked.interrupt is not None and asked.interrupt.assignee == "role:finance"
    assert refunded == [50, 60]  # 60 asked of nobody in particular: remembered
    done = await agent.resume(asked.interrupt.interrupt_id, "approve", reviewer="cfo")
    assert done.answer == ["refunded 50", "refunded 60", "refunded 900"]
