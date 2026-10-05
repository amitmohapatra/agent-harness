"""Every harness tool call, whichever framework makes it: replay, governance, execution,
record.

1. **replay** — a call the journal already has (a resumed run re-planning the same step)
   returns its recorded output and runs nothing;
2. **governance** — the run's tenant's :class:`~trellis.harness.governance.Governance` decides,
   by the tool's name, as the catalog says at the time of the call: ``run`` runs, ``announce``
   is announced on the run's stream, ``ask`` pauses the run for approval (an approver may edit
   the arguments, or reject);
3. **execution** — inside a span, with ``TOOL_CALL_*`` events around it (the stream gets the
   arguments and the output redacted; the tool and the model get them as they are);
4. **record** — journaled for a later resume (and, in a worker, saved as the run's progress
   checkpoint: at once after a call with side effects), counted, and (memory on) sent,
   redacted, to the memory service's tool records in the background.

Execution is bounded: a call takes at most its tool's ``timeout`` and what is left of the
run's time; a call that only reads (or is idempotent) is tried again after an error that may
pass, within that time; the tool reads its idempotency key from ``trellis.current()``. A call
that does more than read is marked started, and saved, before it runs: one that times out, or
was running when its worker died, has an unknown effect — the model is told so, the journal
keeps what it was told, and it is never run again blind.
"""

from __future__ import annotations

import time
from typing import Any, Final

from trellis.contracts import (
    AgentError,
    ErrorCategory,
    InterruptDecision,
    RunEventType,
    ToolCall,
    ToolError,
    ToolOutcome,
    ToolStatus,
)
from trellis.harness.events import NOTICE
from trellis.harness.governance.decision import Decision
from trellis.harness.journal import OUTCOME, content_key
from trellis.harness.runtime import Paused, RunCancelled, Runtime, answer_of, current, reason_of
from trellis.harness.telemetry import metrics, tool_span
from trellis.harness.telemetry import output as span_output
from trellis.harness.tools.base import Tool, interrupted, retried, retries_of, timed_out

#: The ``ToolOutcome.metadata`` flag of a call whose effect is not known: it does more than
#: read, and timed out or was running when its worker died. Its ``error_class`` says so too
#: (:data:`OUTCOME_UNKNOWN`), which the memory service's tool records keep.
UNKNOWN: Final = "unknown"
OUTCOME_UNKNOWN: Final = "OutcomeUnknown"


async def call(tool: Tool, args: dict[str, Any], *, call_id: str | None = None) -> ToolOutcome:
    """Run ``tool`` for the current run. Pauses propagate; every other failure is an outcome
    the framework shows its model (``ERROR``, or ``TIMEOUT``)."""
    runtime = current()
    if runtime is None:
        raise ToolError(
            f"{tool.name} is a trellis tool: it runs inside a Harness run (agent.run/stream)",
            source="tools",
        )
    key = content_key("call", tool.name, args)
    step = runtime.next_step()
    idempotency_key = f"{runtime.run_id}:{key}:{runtime.replay.occurrence(key)}"
    tool_call = ToolCall(
        tool=tool.name, args=args, task=runtime.task, step=step, idempotency_key=idempotency_key
    )
    ref = call_id or f"{runtime.run_id}:{step}"
    replayed, output = runtime.replay.call(key)
    if replayed:
        outcome = _replayed(tool.name, output)
        _events(runtime, ref, tool_call, outcome)
        return outcome
    if runtime.replay.interrupted(key) and not tool.spec.idempotent:
        # its worker died while it ran: it is not run again blind (an idempotent tool is,
        # with the same key, below)
        return await _interrupted(runtime, ref, tool_call, key)

    decision, tool_call, rejected = await _decided(runtime, tool, tool_call)
    if rejected is not None:
        _events(runtime, ref, tool_call, rejected)
        return rejected
    args = tool_call.args
    runtime.events.tool(RunEventType.TOOL_CALL_START, ref, tool=tool.name)
    runtime.events.tool(RunEventType.TOOL_CALL_ARGS, ref, args=args)
    runtime.used_code_mode |= tool.code_mode
    runtime.used.add(tool.name)
    reads = decision.risk == "read"
    if not reads:
        # saved before it runs: the attempt after a crash knows it was running
        runtime.replay.start(key)
        await runtime.progress(now=True)
    started = time.perf_counter()
    action = decision.action.value
    with tool_span(tool.name, ref, args, source=tool.spec.source, action=action) as span:
        try:
            outcome = await _executed(runtime, tool, args, idempotency_key, reads=reads)
        except (Paused, RunCancelled):
            if not reads:
                runtime.replay.unstart(key)  # it asked a person: it runs again on resume
            raise
        span_output(span, outcome.output, key="gen_ai.tool.call.result")
    outcome.latency_ms = round((time.perf_counter() - started) * 1000, 3)
    if outcome.status in (ToolStatus.OK, ToolStatus.TIMEOUT):
        runtime.replay.record_call(key, _journaled(outcome), tool=tool.name)
    elif not reads:
        runtime.replay.unstart(key)  # it failed: a later attempt runs it again
    # a call with side effects is saved at once: a crash after it does not repeat it
    await runtime.progress(now=not decision.runs)
    runtime.events.tool(RunEventType.TOOL_CALL_END, ref, tool=tool.name)
    runtime.events.tool(
        RunEventType.TOOL_CALL_RESULT,
        ref,
        tool=tool.name,
        status=outcome.status.value,
        output=outcome.output,
    )
    metrics.tool_called(tool.name, outcome.status.value)
    await runtime.agent.record_tool(runtime, tool_call, outcome)
    return outcome


async def _decided(
    runtime: Runtime, tool: Tool, call: ToolCall
) -> tuple[Decision, ToolCall, ToolOutcome | None]:
    """Governance's decision, as the catalog says now, by name (a graph's tools were built
    before), and what came of it: the call (with an approver's edited arguments), announced
    when it is to be, or the outcome of a call the approver rejected."""
    governance = runtime.agent.harness.governance(runtime.tenant)
    decision = await governance.check(tool.name, call.args, side_effects=tool.spec.side_effects)
    if decision.asks:
        resolution = await runtime.approve(call, decision.question)
        answer = answer_of(resolution)  # raises RunCancelled on CANCEL
        if resolution.decision is InterruptDecision.REJECT or answer is False:
            reason = reason_of(resolution)
            rejected = ToolOutcome(
                tool=tool.name,
                status=ToolStatus.REJECTED,
                output=f"{tool.name} was not run: the approver rejected it"
                + (f" ({reason})" if reason else ""),
                error_class="ApprovalRejected",
            )
            return decision, call, rejected
        if resolution.decision is InterruptDecision.EDIT and isinstance(answer, dict):
            call = call.model_copy(update={"args": answer})
    elif decision.announces:
        runtime.events.custom(NOTICE, tool=tool.name, args=call.args, side_effects=decision.risk)
    return decision, call, None


async def _interrupted(runtime: Runtime, ref: str, call: ToolCall, key: str) -> ToolOutcome:
    """A call that was running when its worker died, in an earlier attempt: its effect is
    unknown — what the model reads, journaled (and saved) like an outcome."""
    outcome = ToolOutcome(
        tool=call.tool,
        status=ToolStatus.CANCELLED,
        output=interrupted(call.tool),
        error_class=OUTCOME_UNKNOWN,
        metadata={UNKNOWN: True},
    )
    runtime.replay.record_call(key, _journaled(outcome), tool=call.tool)
    await runtime.progress(now=True)
    _events(runtime, ref, call, outcome)
    await runtime.agent.record_tool(runtime, call, outcome)
    return outcome


async def _executed(
    runtime: Runtime, tool: Tool, args: dict[str, Any], key: str, *, reads: bool
) -> ToolOutcome:
    """One call within its time — the tool's ``timeout`` and what is left of the run's —
    tried again after an error that may pass when it only reads (or is idempotent). A call
    that runs out of time, or fails with a timeout of its own, is a ``TIMEOUT``: for one that
    does more than read, its effect is unknown (``metadata["unknown"]``)."""
    attempts = 0

    async def once() -> Any:
        nonlocal attempts
        attempts += 1
        return await tool.run(args)

    began = time.monotonic()
    try:
        async with runtime.limited(tool.timeout, key=key):
            output = await retried(once, retries=retries_of(tool.spec, reads=reads))
    except (Paused, RunCancelled):
        raise
    except Exception as exc:
        if AgentError.of(exc).category is not ErrorCategory.TIMEOUT:
            return ToolOutcome(
                tool=tool.name,
                status=ToolStatus.ERROR,
                output=f"{tool.name} failed: {exc}",
                error_class=type(exc).__name__,
                attempts=attempts,
            )
        took = time.monotonic() - began
        return ToolOutcome(
            tool=tool.name,
            status=ToolStatus.TIMEOUT,
            output=timed_out(tool.name, took=took, limit=tool.timeout, unknown=not reads),
            error_class=type(exc).__name__ if reads else OUTCOME_UNKNOWN,
            attempts=attempts,
            metadata={} if reads else {UNKNOWN: True},
        )
    return ToolOutcome(tool=tool.name, output=output, attempts=attempts)


def _journaled(outcome: ToolOutcome) -> Any:
    """What the journal keeps of a call: its output, or — a timeout, an unknown effect — the
    outcome itself, so a re-run tells the model the same."""
    if outcome.ok:
        return outcome.output
    return {OUTCOME: outcome.model_dump(mode="json", include={"status", "output", "metadata"})}


def _replayed(tool: str, recorded: Any) -> ToolOutcome:
    if isinstance(recorded, dict) and set(recorded) == {OUTCOME}:
        return ToolOutcome(tool=tool, cached=True, **recorded[OUTCOME])
    return ToolOutcome(tool=tool, output=recorded, cached=True)


def _events(runtime: Runtime, ref: str, call: ToolCall, outcome: ToolOutcome) -> None:
    """A call that did not execute still appears on the stream, so a UI sees every step."""
    runtime.events.tool(RunEventType.TOOL_CALL_START, ref, tool=call.tool)
    runtime.events.tool(RunEventType.TOOL_CALL_ARGS, ref, args=call.args)
    runtime.events.tool(RunEventType.TOOL_CALL_END, ref, tool=call.tool)
    runtime.events.tool(
        RunEventType.TOOL_CALL_RESULT,
        ref,
        tool=call.tool,
        status=outcome.status.value,
        cached=outcome.cached,
        output=outcome.output,
    )
