"""A run pauses the same way whatever asked, and an answer continues the same run."""

from __future__ import annotations

from typing import Any

import pytest
from trellis.contracts import (
    AgentCancelledError,
    AgentExecutionContext,
    AgentPaused,
    Interrupt,
    InterruptDecision,
    InterruptReason,
    InterruptResolution,
    PolicyDeniedError,
    RunEventType,
    RunOutcome,
)

from trellis.harness import (
    AgentHarness,
    CallablePolicyProvider,
    CollectingEventSink,
    LocalToolClient,
)
from trellis.harness.interrupts import ANSWER, ApprovalRequired

REFUNDS: list[dict[str, Any]] = []
TENANT = "acme"


def refund(order_id: int, amount: int = 100) -> dict[str, Any]:
    REFUNDS.append({"order_id": order_id, "amount": amount})
    return {"refunded": amount}


def _harness(
    sink: CollectingEventSink, *, approval: bool = True, error_mode: str = "raise"
) -> AgentHarness:
    tools = LocalToolClient({"refund": refund})

    async def tool_policy(context, call):
        return "require_approval" if approval and call.tool == "refund" else True

    return AgentHarness(
        memory=None,
        tools=tools,
        policy=CallablePolicyProvider(tool=tool_policy),
        event_sinks=[sink],
        error_mode=error_mode,
    )


async def refund_agent(payload: Any, runtime: Any) -> dict[str, Any]:
    outcome = await runtime.tools.call("refund", order_id=91, amount=240)
    return {"status": str(outcome.status), "output": outcome.output}


def _turn(agent_id: str, thread_id: str) -> AgentExecutionContext:
    """A context that names its turn, so every run of it is the same run (the shape a
    resume needs); a thread context without a turn opens a new run per call."""
    return AgentExecutionContext.create(
        tenant_id=TENANT,
        user_id="u1",
        agent_id=agent_id,
        thread_id=thread_id,
        session_id=f"ses_{thread_id}",
        turn_id=f"trn_{thread_id}",
    )


def _paused(sink: CollectingEventSink, run_id: str) -> Interrupt:
    finished = [e for e in sink.for_run(run_id) if e.type is RunEventType.RUN_FINISHED]
    assert finished and finished[-1].outcome is RunOutcome.INTERRUPT
    return Interrupt.model_validate(finished[-1].data["interrupt"])


def _finishes(sink: CollectingEventSink, run_id: str) -> list[RunOutcome]:
    return [e.outcome for e in sink.for_run(run_id) if e.type is RunEventType.RUN_FINISHED]


@pytest.mark.parametrize(
    ("decision", "expected_status", "expected_amount"),
    [
        (InterruptDecision.APPROVE, "ok", 240),
        (InterruptDecision.EDIT, "ok", 200),
        (InterruptDecision.REJECT, "rejected", None),
    ],
)
async def test_an_approval_pauses_the_run_and_the_answer_continues_it(
    decision: InterruptDecision, expected_status: str, expected_amount: int | None
) -> None:
    REFUNDS.clear()
    sink = CollectingEventSink()
    harness = _harness(sink)
    agent = harness.wrap(refund_agent, agent_id="refund-agent")
    ctx = _turn("refund-agent", f"thr_{decision.value.lower()}")
    with pytest.raises(ApprovalRequired) as paused:
        await agent("refund order 91", context=ctx)
    assert paused.value.tool_call.tool == "refund" and REFUNDS == []
    types = sink.types(ctx.agent_run_id)
    assert types[:2] == [RunEventType.RUN_STARTED, RunEventType.STEP_STARTED]
    assert types[-2:] == [RunEventType.INTERRUPT, RunEventType.RUN_FINISHED]
    interrupt = _paused(sink, ctx.agent_run_id)
    assert interrupt.reason is InterruptReason.APPROVAL and interrupt.tool_call is not None
    assert interrupt.tool_call.idempotency_key == paused.value.tool_call.idempotency_key
    # the harness holds the pause for a surface to answer by id, for the run's own caller
    announced = harness.resolutions.announced(TENANT)
    assert [i.interrupt_id for i in announced] == [interrupt.interrupt_id]

    resolution = InterruptResolution(
        interrupt_id=interrupt.interrupt_id,
        run_id=interrupt.run_id,
        decision=decision,
        payload={"order_id": 91, "amount": 200} if decision is InterruptDecision.EDIT else None,
        reviewer="u1",
    )
    result = await harness.resume(interrupt, resolution, context=ctx, agent=agent)
    assert result is not None and result.data["status"] == expected_status
    if expected_amount is None:
        assert REFUNDS == [] and "rejected" in result.data["output"]
    else:
        assert [{"order_id": 91, "amount": expected_amount}] == REFUNDS
    assert _finishes(sink, ctx.agent_run_id)[-1] is RunOutcome.SUCCESS
    call_id = interrupt.tool_call.idempotency_key
    tool_events = [e.type for e in sink.for_run(ctx.agent_run_id) if e.tool_call_id == call_id]
    # the resumed run's call opens and closes on the stream whether it ran or was refused
    assert tool_events[-4:] == [
        RunEventType.TOOL_CALL_START,
        RunEventType.TOOL_CALL_ARGS,
        RunEventType.TOOL_CALL_END,
        RunEventType.TOOL_CALL_RESULT,
    ]
    results = [e for e in sink.for_run(ctx.agent_run_id) if e.type is RunEventType.TOOL_CALL_RESULT]
    assert results[-1].data["status"] == expected_status
    assert not harness.resolutions.pending(TENANT, ctx.agent_run_id)  # consumed by the run


async def test_cancelling_an_approval_ends_the_run_cancelled() -> None:
    REFUNDS.clear()
    sink = CollectingEventSink()
    harness = _harness(sink)
    agent = harness.wrap(refund_agent, agent_id="refund-agent")
    ctx = _turn("refund-agent", "thr_cancel")
    with pytest.raises(ApprovalRequired):
        await agent("refund order 91", context=ctx)
    interrupt = _paused(sink, ctx.agent_run_id)
    cancel = InterruptResolution(
        interrupt_id=interrupt.interrupt_id,
        run_id=interrupt.run_id,
        decision=InterruptDecision.CANCEL,
    )
    with pytest.raises(AgentCancelledError):
        await harness.resume(interrupt, cancel, context=ctx, agent=agent)
    assert REFUNDS == []
    assert _finishes(sink, ctx.agent_run_id) == [RunOutcome.INTERRUPT, RunOutcome.CANCELLED]


@pytest.mark.parametrize("error_mode", ["return", "raise"])
async def test_cancelling_a_question_ends_the_run_without_running_the_agent(error_mode) -> None:
    sink = CollectingEventSink()
    harness = _harness(sink, approval=False, error_mode=error_mode)
    asked: list[int] = []

    async def asking(payload: Any, runtime: Any) -> str:
        asked.append(1)
        raise AgentPaused("Which region?")

    agent = harness.wrap(asking, agent_id="deploy")
    ctx = _turn("deploy", f"thr_cq_{error_mode}")
    with pytest.raises(AgentPaused):
        await agent("deploy", context=ctx)
    interrupt = _paused(sink, ctx.agent_run_id)
    cancel = InterruptResolution(
        interrupt_id=interrupt.interrupt_id,
        run_id=interrupt.run_id,
        decision=InterruptDecision.CANCEL,
    )
    if error_mode == "raise":
        with pytest.raises(AgentCancelledError):
            await harness.resume(interrupt, cancel, context=ctx, agent=agent)
    else:
        result = await harness.resume(interrupt, cancel, context=ctx, agent=agent)
        assert result is not None and str(result.status) == "CANCELLED"
    assert asked == [1]  # the agent was not run again
    assert _finishes(sink, ctx.agent_run_id) == [RunOutcome.INTERRUPT, RunOutcome.CANCELLED]


async def test_an_approval_answered_a_second_time_for_other_arguments_pauses_again() -> None:
    """An approval binds the arguments the approver saw; a resumed agent that asks for
    something else asks a person again."""
    REFUNDS.clear()
    sink = CollectingEventSink()
    harness = _harness(sink)
    amounts = iter((240, 900))

    async def greedy(payload: Any, runtime: Any) -> Any:
        outcome = await runtime.tools.call("refund", order_id=91, amount=next(amounts))
        return outcome.output

    agent = harness.wrap(greedy, agent_id="refund-agent")
    ctx = _turn("refund-agent", "thr_rebind")
    with pytest.raises(ApprovalRequired):
        await agent("refund", context=ctx)
    interrupt = _paused(sink, ctx.agent_run_id)
    approve = InterruptResolution(
        interrupt_id=interrupt.interrupt_id,
        run_id=interrupt.run_id,
        decision=InterruptDecision.APPROVE,
    )
    with pytest.raises(ApprovalRequired) as again:
        await harness.resume(interrupt, approve, context=ctx, agent=agent)
    assert again.value.tool_call.args["amount"] == 900 and REFUNDS == []


async def test_a_question_is_answered_into_the_resumed_runs_state() -> None:
    sink = CollectingEventSink()
    harness = _harness(sink, approval=False)

    async def choosing_agent(payload: Any, runtime: Any) -> str:
        answer = runtime.state.get("resolutions", {}).get(ANSWER)
        if answer is None:
            raise AgentPaused("Which region?", expects={"type": "string"})
        return f"deploying to {answer.answer}"

    agent = harness.wrap(choosing_agent, agent_id="deploy")
    ctx = _turn("deploy", "thr_q")
    with pytest.raises(AgentPaused):
        await agent("deploy", context=ctx)
    interrupt = _paused(sink, ctx.agent_run_id)
    assert interrupt.question == "Which region?" and interrupt.expects == {"type": "string"}
    answer = InterruptResolution(
        interrupt_id=interrupt.interrupt_id,
        run_id=ctx.agent_run_id,
        decision=InterruptDecision.ANSWER,
        answer="eu",
    )
    with pytest.raises(ValueError, match="is not answered with APPROVE"):
        await harness.resume(
            interrupt,
            answer.model_copy(update={"decision": InterruptDecision.APPROVE}),
            context=ctx,
        )
    result = await harness.resume(interrupt, answer, context=ctx, agent=agent)
    assert result is not None and result.data == "deploying to eu"
    assert result.data == "deploying to eu"
    finished = [e for e in sink.for_run(ctx.agent_run_id) if e.type is RunEventType.RUN_FINISHED]
    assert finished[-1].data["result"] == "deploying to eu"  # the answer rides on the finish
    with pytest.raises(ValueError, match="the resolution answers a different interrupt"):
        await harness.resume(
            interrupt, answer.model_copy(update={"interrupt_id": "int_other"}), context=ctx
        )
    with pytest.raises(ValueError, match="the context is not the paused run's"):
        await harness.resume(
            interrupt, answer, context=ctx.model_copy(update={"agent_run_id": "run_x"})
        )
    with pytest.raises(ValueError, match="the context must name the paused run's turn"):
        await harness.resume(interrupt, answer, context=ctx.model_copy(update={"turn_id": None}))


@pytest.mark.parametrize("error_mode", ["return", "raise"])
async def test_a_denied_tool_is_rejected_in_both_error_modes(error_mode: str) -> None:
    sink = CollectingEventSink()
    tools = LocalToolClient({"refund": refund})
    harness = AgentHarness(
        memory=None,
        tools=tools,
        policy=CallablePolicyProvider(tool=lambda context, call: "refunds are off"),
        event_sinks=[sink],
        error_mode=error_mode,
    )
    agent = harness.wrap(refund_agent, agent_id="refund-agent")
    ctx = _turn("refund-agent", f"thr_deny_{error_mode}")
    if error_mode == "raise":
        with pytest.raises(PolicyDeniedError, match="refunds are off"):
            await agent("refund", context=ctx)
    else:
        result = await agent("refund", context=ctx)
        assert not result.succeeded and "refunds are off" in result.error.message
    finished = [e for e in sink.for_run(ctx.agent_run_id) if e.type is RunEventType.RUN_FINISHED]
    # exactly one finish, and a policy refusal is REJECTED, not ERROR: nothing broke
    assert len(finished) == 1
    assert finished[0].outcome is RunOutcome.REJECTED and finished[0].error is not None
