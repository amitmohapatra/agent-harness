"""``ToolCallBridge``: authorize, announce and record a tool call somebody else executes.

``InstrumentedToolClient`` does five things around every tool call — derives the idempotency
material, asks the policy, opens and closes the call on the run's event stream, records the
invocation in tool memory, and counts it — and then executes the tool itself. A framework
adapter needs the same five things but must **not** execute: Deep Agents' ``wrap_tool_call``,
the OpenAI Agents SDK's tool guardrails and the Claude Agent SDK's ``PreToolUse`` each hand
over a call the framework is about to run itself.

So the five live here, and both callers use them: the client wraps its own execution around
the bridge, and each adapter wraps the framework's handler around it. There is one
implementation of what a ``require_approval`` policy means, one place a resumed run finds its
approver's decision, and one shape of tool event, whoever runs the tool.

    bridge = ToolCallBridge(runtime, policy=policy, source="deepagents")
    call = bridge.prepare("refund", {"order": "o-1"}, call_id=framework_call_id)
    call, rejected = await bridge.authorize(call)     # raises on deny / pause / cancel
    await bridge.opened(call)
    if rejected is not None:
        await bridge.settled(call, rejected, 0.0, str(rejected.status))
        return framework_refusal(rejected)
    ...                                              # the framework runs the tool
    await bridge.settled(call, outcome, watch.ms, str(outcome.status))
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Final

from trellis.contracts import ToolStatus
from trellis.contracts.errors import AgentCancelledError, PolicyDeniedError
from trellis.contracts.events import LifecycleEvent
from trellis.contracts.runs import InterruptDecision, RunEventType
from trellis.contracts.tool import ToolCall, ToolOutcome

from trellis.harness.interrupts.signals import ApprovalRequired
from trellis.harness.policy.outcome import PolicyOutcome, normalize
from trellis.harness.telemetry import names as N
from trellis.harness.telemetry.metrics import TOOL_CALLS, TOOL_LATENCY

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping

    from trellis.harness.runtime.agent_runtime import AgentRuntime

__all__ = [
    "RESULT_PREVIEW_CHARS",
    "ToolCallBridge",
    "brief",
    "call_id_of",
]

#: How much of a tool result rides on the stream; the rest is reachable by reference.
RESULT_PREVIEW_CHARS: Final = 2000


class ToolCallBridge:
    """The instrumentation around one execution's tool calls, minus the execution."""

    def __init__(
        self,
        runtime: AgentRuntime,
        *,
        policy: Any = None,
        events: Any = None,
        record_to_memory: bool = True,
        source: str = "harness",
    ) -> None:
        """``source`` names who runs the tool, for the span and the call summary: the
        harness's own client, or the framework an adapter bridges."""
        self.runtime = runtime
        self.policy = policy
        self.source = source
        self.record_to_memory = record_to_memory
        self._events = events
        self._step = 0

    # ------------------------------------------------------------------ the call
    def prepare(
        self,
        tool: str | ToolCall,
        args: Mapping[str, Any] | None = None,
        /,
        *,
        call_id: str | None = None,
    ) -> ToolCall:
        """Number the call within this execution and give it its idempotency key.

        ``call_id`` is the framework's own id for the call when it has one: it becomes the
        idempotency key, so a pause and its resume agree on which call was approved even
        though the framework re-plans in between.
        """
        call = tool if isinstance(tool, ToolCall) else ToolCall(tool=tool, args=dict(args or {}))
        self._step += 1
        step = call.step if call.step is not None else self._step
        return call.model_copy(
            update={
                "step": step,
                "idempotency_key": call.idempotency_key
                or call_id
                or self.runtime.idempotency_key("tool", call.tool, step),
            }
        )

    async def authorize(self, call: ToolCall) -> tuple[ToolCall, ToolOutcome | None]:
        """The call to run (its arguments possibly edited by an approver), or a rejected
        outcome. A policy that requires approval pauses the run with the call, unless the
        person already answered: a resumed run finds the resolution in ``runtime.state``,
        and it counts only for the arguments the approver saw. Edited arguments go through
        the policy again: an approver may narrow a call, never widen it past a denial."""
        if self.policy is None:
            return call, None
        outcome, reason = normalize(await self.policy.authorize_tool(self.runtime.context, call))
        if outcome is PolicyOutcome.ALLOW:
            return call, None
        if outcome is PolicyOutcome.DENY:
            raise PolicyDeniedError(
                reason or f"tool {call.tool!r} denied by policy", source="policy.tool"
            )
        resolved = self.runtime.state.get("resolutions", {}).get(call.idempotency_key)
        if resolved is None or not _same_call(resolved.interrupt.tool_call, call):
            raise ApprovalRequired(call, reason=reason)
        decision = resolved.decision
        if decision is InterruptDecision.APPROVE:
            return call, None
        if decision is InterruptDecision.EDIT:
            edited = call.model_copy(update={"args": dict(resolved.payload or {})})
            verdict, why = normalize(await self.policy.authorize_tool(self.runtime.context, edited))
            if verdict is PolicyOutcome.DENY:
                raise PolicyDeniedError(
                    why or f"the edited arguments for {call.tool!r} are denied by policy",
                    source="policy.tool",
                )
            return edited, None
        if decision is InterruptDecision.CANCEL:
            # the person abandoned the run, not just the call: it ends CANCELLED
            self.runtime.cancellation.cancel("cancelled by the approver")
            raise AgentCancelledError("cancelled by the approver", source="policy.tool")
        return call, ToolOutcome(
            tool=call.tool,
            status=ToolStatus.REJECTED,
            output=f"the call to {call.tool} was rejected by the approver",
            error_class="ApprovalRejected",
            metadata={"interrupt_id": resolved.interrupt.interrupt_id},
        )

    # ------------------------------------------------------------------ the run's stream
    async def opened(self, call: ToolCall, *, lifecycle: bool = True) -> None:
        """Announce the call: the lifecycle event, then ``TOOL_CALL_START``/``_ARGS``."""
        if lifecycle:
            self.emit(LifecycleEvent.TOOL_START, {"tool": call.tool, "step": call.step})
        await self.runtime.events.emit(
            RunEventType.TOOL_CALL_START, tool_call_id=call_id_of(call), tool=call.tool
        )
        await self.runtime.events.emit(
            RunEventType.TOOL_CALL_ARGS, tool_call_id=call_id_of(call), args=call.args
        )

    async def closed(
        self,
        call: ToolCall,
        *,
        status: str,
        output: Any = None,
        error_class: str | None = None,
        invocation_id: str | None = None,
    ) -> None:
        await self.runtime.events.emit(
            RunEventType.TOOL_CALL_END, tool_call_id=call_id_of(call), tool=call.tool
        )
        await self.runtime.events.emit(
            RunEventType.TOOL_CALL_RESULT,
            tool_call_id=call_id_of(call),
            tool=call.tool,
            status=status,
            output=brief(output),
            error_class=error_class,
            invocation_id=invocation_id,
        )

    # ------------------------------------------------------------------ recording
    def finished(
        self,
        call: ToolCall,
        outcome: ToolOutcome | None,
        ms: float,
        status: str,
        *,
        error: BaseException | None = None,
    ) -> None:
        """Count the call, keep its summary on the runtime, emit ``TOOL_END``.

        Separate from :meth:`settled` because a cancelled call is counted but neither
        remembered nor closed: the run is going away, and tool memory should not learn a
        procedure from a call that never finished.
        """
        runtime = self.runtime
        runtime.tracer.metrics.count(
            TOOL_CALLS, tool=call.tool, status=status, cached=bool(outcome and outcome.cached)
        )
        runtime.tracer.metrics.duration(TOOL_LATENCY, ms, tool=call.tool, status=status)
        runtime.record_tool_call(
            {
                "tool": call.tool,
                "step": call.step,
                "status": status,
                "latency_ms": round(ms, 3),
                "cached": bool(outcome and outcome.cached),
                "attempts": outcome.attempts if outcome else 1,
                "idempotency_key": call.idempotency_key,
                "error_class": type(error).__name__ if error else None,
                "args": sorted(call.args),
                "source": self.source,
            }
        )
        self.emit(
            LifecycleEvent.TOOL_END,
            {"tool": call.tool, "status": status, "latency_ms": ms},
        )

    async def remembered(
        self,
        call: ToolCall,
        outcome: ToolOutcome | None,
        ms: float,
        status: str,
        *,
        error: BaseException | None = None,
    ) -> None:
        """Record the invocation in tool memory so the service can learn procedures (§12)."""
        runtime = self.runtime
        if not self.record_to_memory or not runtime.memory.enabled:
            return
        policy = getattr(runtime.memory, "policy", None)
        include_output = bool(policy and policy.observe_tool_results)
        try:
            await runtime.memory.record_tool_call(
                call.tool,
                call.args,
                output=outcome.output if (outcome and include_output) else None,
                status=status,
                error_class=type(error).__name__ if error else None,
                latency_ms=ms,
                task=task_of(call, runtime),
                step=call.step,
            )
        except Exception:
            runtime.logger.debug("tool memory recording failed", tool=call.tool)

    async def settled(
        self,
        call: ToolCall,
        outcome: ToolOutcome | None,
        ms: float,
        status: str,
        *,
        error: BaseException | None = None,
    ) -> None:
        """The whole close-out: counted, remembered, closed on the stream."""
        self.finished(call, outcome, ms, status, error=error)
        await self.remembered(call, outcome, ms, status, error=error)
        await self.closed(
            call,
            status=status,
            output=outcome.output if outcome is not None else (str(error) if error else None),
            error_class=(outcome.error_class if outcome else None)
            or (type(error).__name__ if error else None),
            invocation_id=outcome.invocation_id if outcome else None,
        )

    # ------------------------------------------------------------------ helpers
    def tool_span(self, call: ToolCall, **attributes: Any) -> Any:
        """The tool span, with the attributes every caller sets; extras are the caller's."""
        return self.runtime.tracer.tool_span(
            call.tool,
            **{
                N.TOOL_SOURCE: self.source,
                N.TOOL_IDEMPOTENCY_KEY: call.idempotency_key,
                N.TOOL_ARGS_SCHEMA: sorted(call.args),
                **attributes,
            },
        )

    def emit(self, event: LifecycleEvent, payload: dict[str, Any]) -> None:
        if self._events is not None:
            self._events.emit(event, {"context": self.runtime.context, **payload})


def _same_call(held: ToolCall | None, call: ToolCall) -> bool:
    """Whether ``call`` is the call a person approved: the same tool with the same arguments
    (a resumed agent re-plans from scratch, and its second attempt may ask for more)."""
    return held is not None and held.tool == call.tool and held.args == call.args


def call_id_of(call: ToolCall) -> str:
    """The id a surface correlates the tool events by: the call's idempotency key, which is
    stable across a pause and its resume, so an approval resolves the same call."""
    return call.idempotency_key or f"call:{call.tool}:{call.step}"


def brief(output: Any) -> Any:
    """The result as the stream carries it: scalars as they are, a structured value as
    JSON carries it when it fits, otherwise the head of its JSON text."""
    if output is None or isinstance(output, bool | int | float):
        return output
    if isinstance(output, str):
        text = output
    else:
        try:
            text = json.dumps(output, default=str)
        except (TypeError, ValueError):
            text = json.dumps(str(output))
        if len(text) <= RESULT_PREVIEW_CHARS:
            return json.loads(text)
    return text if len(text) <= RESULT_PREVIEW_CHARS else text[:RESULT_PREVIEW_CHARS] + "…"


def task_of(call: Any, runtime: Any) -> str:
    """What this tool call was *for*, as the service keys procedures on.

    The service normalises this into a typed-placeholder pattern ("how much stock of
    {entity}?") so two phrasings of the same question mine the same trajectory. Sending it
    ``context.task_id`` — an opaque identifier — either matched only other runs of that same
    id or, far more often, was empty: no task_id, no pattern, no procedure, ever. The turn's
    question is what it wanted.
    """
    explicit = getattr(call, "task", None)
    if explicit:
        return str(explicit)
    request = runtime.state.get("request") if hasattr(runtime, "state") else None
    query = getattr(request, "query", None)
    if query:
        return str(query)
    return runtime.context.task_id or ""
