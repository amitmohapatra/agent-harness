"""Every harness tool call, whichever framework makes it: replay, policy, execution, record.

1. **replay** — a call the journal already has (a resumed run re-planning the same step)
   returns its recorded output and runs nothing;
2. **policy** — the risk tier: ``auto`` runs, ``notify`` is announced on the run's stream,
   ``ask`` pauses the run for approval (an approver may edit the arguments, or reject);
3. **execution** — inside a span, with ``TOOL_CALL_*`` events around it;
4. **record** — journaled for a later resume (and, in a worker, saved as the run's progress
   checkpoint: at once after a call with side effects), counted, and (memory on) sent to the
   memory service's tool records in the background.
"""

from __future__ import annotations

import time
from typing import Any, Final

from trellis.contracts import (
    InterruptDecision,
    RunEventType,
    ToolCall,
    ToolError,
    ToolOutcome,
    ToolStatus,
)
from trellis.harness.events import NOTICE
from trellis.harness.journal import content_key
from trellis.harness.runtime import Paused, RunCancelled, Runtime, answer_of, current, reason_of
from trellis.harness.telemetry import metrics, tool_span
from trellis.harness.telemetry import output as span_output
from trellis.harness.tools.base import Tool
from trellis.harness.tools.policy import Tier, tier

#: How much of a tool result rides on the event stream.
PREVIEW_CHARS: Final = 2000


async def call(tool: Tool, args: dict[str, Any], *, call_id: str | None = None) -> ToolOutcome:
    """Run ``tool`` for the current run. Pauses propagate; every other failure is an ``ERROR``
    outcome the framework shows its model."""
    runtime = current()
    if runtime is None:
        raise ToolError(
            f"{tool.name} is a trellis tool: it runs inside a Harness run (agent.run/stream)",
            source="tools",
        )
    key = content_key("call", tool.name, args)
    step = runtime.next_step()
    tool_call = ToolCall(
        tool=tool.name, args=args, task=runtime.task, step=step, idempotency_key=call_id or key
    )
    ref = call_id or f"{runtime.run_id}:{step}"
    replayed, output = runtime.replay.call(key)
    if replayed:
        _events(runtime, ref, tool_call, ToolOutcome(tool=tool.name, output=output, cached=True))
        return ToolOutcome(tool=tool.name, output=output, cached=True)

    chosen, why = tier(tool, args)
    if chosen is Tier.ASK:
        resolution = await runtime.approve(tool_call, why)
        decision = answer_of(resolution)  # raises RunCancelled on CANCEL
        if resolution.decision is InterruptDecision.REJECT or decision is False:
            reason = reason_of(resolution)
            outcome = ToolOutcome(
                tool=tool.name,
                status=ToolStatus.REJECTED,
                output=f"{tool.name} was not run: the approver rejected it"
                + (f" ({reason})" if reason else ""),
                error_class="ApprovalRejected",
            )
            _events(runtime, ref, tool_call, outcome)
            return outcome
        if resolution.decision is InterruptDecision.EDIT and isinstance(decision, dict):
            args = decision
            tool_call = tool_call.model_copy(update={"args": args})
    elif chosen is Tier.NOTIFY:
        runtime.events.custom(NOTICE, tool=tool.name, args=args, side_effects=tool.side_effects)

    runtime.events.tool(RunEventType.TOOL_CALL_START, ref, tool=tool.name)
    runtime.events.tool(RunEventType.TOOL_CALL_ARGS, ref, args=args)
    runtime.used_code_mode |= tool.code_mode
    runtime.used.add(tool.name)
    started = time.perf_counter()
    with tool_span(tool.name, ref, args, source=tool.spec.source, tier=chosen.value) as span:
        try:
            outcome = ToolOutcome(tool=tool.name, output=await tool.run(args))
        except (Paused, RunCancelled):
            raise
        except Exception as exc:
            outcome = ToolOutcome(
                tool=tool.name,
                status=ToolStatus.ERROR,
                output=f"{tool.name} failed: {exc}",
                error_class=type(exc).__name__,
            )
        span_output(span, outcome.output, key="gen_ai.tool.call.result")
    outcome.latency_ms = round((time.perf_counter() - started) * 1000, 3)
    if outcome.ok:
        runtime.replay.record_call(key, outcome.output, tool=tool.name)
        # a call with side effects is saved at once: a crash after it does not repeat it
        await runtime.progress(now=chosen is not Tier.AUTO)
    runtime.events.tool(RunEventType.TOOL_CALL_END, ref, tool=tool.name)
    runtime.events.tool(
        RunEventType.TOOL_CALL_RESULT,
        ref,
        tool=tool.name,
        status=outcome.status.value,
        output=_preview(outcome.output),
    )
    metrics.tool_called(tool.name, outcome.status.value)
    await runtime.agent.record_tool(runtime, tool_call, outcome)
    return outcome


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
        output=_preview(outcome.output),
    )


def _preview(output: Any) -> Any:
    if output is None or isinstance(output, bool | int | float):
        return output
    text = output if isinstance(output, str) else repr(output)
    if isinstance(output, dict | list) and len(text) <= PREVIEW_CHARS:
        return output
    return text if len(text) <= PREVIEW_CHARS else text[:PREVIEW_CHARS] + "…"
