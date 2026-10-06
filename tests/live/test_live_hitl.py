"""W5 against agent-runs (and, for one model-driven run, the gateway): an approval rule in code (a
``before_tool`` hook asking by the call's arguments, on its own screen), labelled options with several picks checked by agent-runs itself, a comment
kept in agent-runs' resolution history, an approval remembered for the run, a result from
outside the run (an ``ask`` in the tool), Way 2's ``Question`` paused in agent-runs, and a ``ReAct`` agent through the gateway
pausing inside a tool that waits for a result from outside. What is asserted is the harness's behaviour, not the model's
words."""

from __future__ import annotations

import uuid
from typing import Any

import pytest

import trellis
from tests.live.conftest import MODEL, live_harness, needs_gateway, needs_memory, needs_runs
from trellis import Ask, Hooks, ReAct, Runtime, tool
from trellis.contracts import (
    InterruptDecision,
    InterruptReason,
    InterruptResolution,
    Option,
    RunStart,
    RunStatus,
    ToolCall,
    new_id,
)
from trellis.harness.asking import Question
from trellis.runs import RunsClient, ValidationError
from trellis.testing import Decide, Reviewer

pytestmark = [pytest.mark.live, needs_runs, needs_memory]

wired: list[int] = []


class WireRule(Hooks):
    """Up to 10 runs unasked (the tool is ``write``); over 500 is finance's, on its own screen;
    anything between is asked of whoever answers."""

    async def before_tool(self, call: ToolCall) -> Ask | None:
        if call.tool != "wire" or call.args["amount"] <= 10:
            return None
        if call.args["amount"] > 500:
            return Ask(
                "Wire over 500?",
                assignee="role:finance",
                component="wire-review",
                props=dict(call.args),
            )
        return Ask(f"Wire {call.args['amount']}?")


@tool(side_effects="write")
def wire(amount: int) -> str:
    """Wire an amount."""
    wired.append(amount)
    return f"wired {amount}"


@tool(side_effects="write")
async def sign(contract: str) -> str:
    """Have a contract signed by a person in the e-signature system."""
    runtime = trellis.current()
    assert runtime is not None
    # the run pauses here; what the e-signature system answers is what this tool returns
    return await runtime.ask(
        f"Signed {contract}?",
        expects={"type": "string"},  # what this tool returns
        component="e-signature",
        props={"contract": contract},
    )


async def settle_account(input: str, agent: Runtime) -> Any:
    plans = await agent.ask(
        "Which plans?",
        options=[Option(value="basic", label="Basic"), Option(value="pro", label="Pro")],
        multiple=True,
    )
    wires = [await agent.tools.call("wire", amount=a) for a in (5, 50, 60, 900)]
    signed = await agent.tools.call("sign", contract="c-7")
    return {"plans": plans, "wires": wires, "signed": signed}


@pytest.mark.timeout(120)
async def test_approvals_choices_comments_and_results_from_outside_through_agent_runs() -> None:
    wired.clear()
    async with live_harness() as h:
        assert isinstance(h.runs, RunsClient)
        agent = h.wrap(
            settle_account,
            id=f"live-hitl-{uuid.uuid4().hex[:8]}",
            tools=[wire, sign],
            hooks=[WireRule()],
        )
        paused = await agent.run("settle", user="live-user")
        asked = paused.interrupt
        assert asked is not None and asked.multiple and asked.option_values == ["basic", "pro"]
        record = await h.runs.get(paused.run_id)
        assert record is not None and record.awaiting == asked  # kept as asked, labels and all
        tenant = record.tenant_id
        # agent-runs itself checks the answer: one that picks nothing offered is refused there
        wrong = InterruptResolution(
            interrupt_id=asked.interrupt_id,
            run_id=paused.run_id,
            decision=InterruptDecision.ANSWER,
            answer=["gold"],
        )
        with pytest.raises(ValidationError, match="not among the options"):
            await h.runs.resume(wrong, tenant=tenant)
        reviewer = Reviewer(
            {
                "Which plans?": ["basic", "pro"],
                "wire": Decide("approve", comment="this account only", remember="run"),
            },
            name="live-reviewer",
        )
        result = await reviewer.answer(agent, paused)  # the plans; 5 runs unasked
        assert result.interrupt is not None and result.interrupt.question.startswith(
            "Approve wire?"
        )
        result = await reviewer.answer(agent, result)  # 50, approved for the run: 60 not asked
        screen = result.interrupt
        assert screen is not None and screen.component == "wire-review"  # 900: finance's
        assert screen.props == {"amount": 900} and screen.assignee == "role:finance"
        result = await agent.resume(screen.interrupt_id, "approve", reviewer="cfo", comment="ok")
        outside = result.interrupt
        assert outside is not None and outside.reason is InterruptReason.QUESTION
        assert (outside.component, outside.props) == ("e-signature", {"contract": "c-7"})
        assert outside.expects == {"type": "string"}
        done = await agent.resume(
            result.run_id, "answer", answer="signed by ada", reviewer="e-signature"
        )
        assert done.status is RunStatus.SUCCESS, done.error
        assert done.answer == {
            "plans": ["basic", "pro"],
            "wires": ["wired 5", "wired 50", "wired 60", "wired 900"],
            "signed": "signed by ada",
        }
        assert wired == [5, 50, 60, 900]
        history = (await h.runs.resolutions(done.run_id, tenant=tenant)).items
        said = [
            (e.resolution.decision.value, e.resolution.comment, e.resolution.remember)
            for e in history
        ]
        assert said == [
            ("ANSWER", None, "once"),
            ("APPROVE", "this account only", "run"),
            ("APPROVE", "ok", "once"),
            ("ANSWER", None, "once"),
        ]


@pytest.mark.timeout(60)
async def test_way_2_pauses_with_the_same_question_in_agent_runs() -> None:
    async with live_harness() as h:
        tenant = await h.tenant()
        run = await h.runs.start(
            RunStart(run_id=new_id("run_"), tenant_id=tenant, agent_id="live-way2", input="x")
        )
        question = Question("Where to?", expects={"type": "object", "required": ["zip"]})
        interrupt = question.interrupt(tenant=tenant, run_id=run.run_id)
        await h.runs.pause(interrupt, checkpoint={"step": 1})
        given = Reviewer({"Where to?": {"zip": 10115}}).resolution(interrupt)
        resumed = await h.runs.resume(given, tenant=tenant)
        assert resumed.last_resolution is not None
        assert question.answer(resumed.last_resolution) == {"zip": 10115}
        await h.runs.finish(run.run_id, RunStatus.SUCCESS, tenant=tenant)


@needs_gateway
@pytest.mark.timeout(300)
async def test_a_react_agent_through_the_gateway_reads_a_result_from_outside() -> None:
    async with live_harness() as h:
        target = ReAct(
            system="You get contracts signed. Always call the sign tool with the contract id "
            "the user gives, then reply with what it returned.",
            model=MODEL,
            max_steps=3,
        )
        agent = h.wrap(target, id=f"live-signer-{uuid.uuid4().hex[:8]}", tools=[sign])
        paused = await agent.run("Get contract c-42 signed.", user="live-user")
        if paused.status is not RunStatus.PAUSED:
            pytest.skip(f"the model did not call the tool ({paused.status.value}: {paused.error})")
        asked = paused.interrupt
        assert asked is not None and asked.component == "e-signature"
        assert asked.expects == {"type": "string"}  # what sign returns
        done = await agent.resume(
            paused.run_id, "answer", answer="signed by ada at 10:42", reviewer="e-signature"
        )
        assert done.status is RunStatus.SUCCESS, done.error
        record = await h.runs.get(done.run_id)
        assert record is not None and record.attempt == 2
    assert trellis.current() is None
