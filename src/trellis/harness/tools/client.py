"""``InstrumentedToolClient`` — what ``runtime.tools`` is (§12).

Captured per call: tool, version, which arguments were passed (names and types, not
values, unless capture allows), start/end, latency, status, retries, a result reference,
any artifact, the error, the span and the agent run. Idempotency material travels with the
call so a tool that supports it can deduplicate a retried write (§41); the harness never
retries a tool it has not been told is idempotent.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

from trellis.contracts import ToolStatus
from trellis.contracts.artifacts import ArtifactRef
from trellis.contracts.errors import AgentCancelledError, PolicyDeniedError, ToolError
from trellis.contracts.events import LifecycleEvent
from trellis.contracts.runs import InterruptDecision, RunEventType
from trellis.contracts.tool import ToolCall, ToolOutcome, ToolSpec

from trellis.harness.interrupts.signals import ApprovalRequired
from trellis.harness.policy.outcome import PolicyOutcome, normalize
from trellis.harness.telemetry import names as N
from trellis.harness.telemetry.metrics import TOOL_CALLS, TOOL_LATENCY
from trellis.harness.telemetry.tracer import Stopwatch

if TYPE_CHECKING:  # pragma: no cover
    from trellis.harness.runtime.agent_runtime import AgentRuntime


class InstrumentedToolClient:
    """Wraps a :class:`ToolClient` for one execution."""

    def __init__(
        self,
        client: Any,
        *,
        runtime_ref: Any = None,
        timeout: float | None = None,
        policy: Any = None,
        events: Any = None,
        record_to_memory: bool = True,
    ) -> None:
        self._client = client
        self._runtime: AgentRuntime | None = runtime_ref
        self.timeout = timeout
        self._policy = policy
        self._events = events
        self.record_to_memory = record_to_memory
        self._step = 0

    def attach(self, runtime: AgentRuntime) -> None:
        self._runtime = runtime

    @property
    def inner(self) -> Any:
        return self._client

    async def list_tools(self) -> list[ToolSpec]:
        return list(await self._client.list_tools())

    async def call(self, tool: str | ToolCall, /, **args: Any) -> ToolOutcome:
        """Execute a tool with full instrumentation. Returns a normalized outcome."""
        call = tool if isinstance(tool, ToolCall) else ToolCall(tool=tool, args=args)
        runtime = self._runtime
        if runtime is None:
            return await self._execute(call)

        self._step += 1
        call = call.model_copy(
            update={
                "step": call.step if call.step is not None else self._step,
                "idempotency_key": call.idempotency_key
                or runtime.idempotency_key("tool", call.tool, self._step),
            }
        )
        call, rejected = await self._authorize(call)
        if rejected is not None:
            # a refused call is still a call: it opens and closes on the stream like one
            # that ran, and tool memory learns that it was refused
            await self._opened(runtime, call)
            self._finish(call, rejected, 0.0, str(rejected.status))
            await self._record_memory(call, rejected, 0.0, str(rejected.status))
            await self._closed(
                runtime,
                call,
                status=str(rejected.status),
                error_class=rejected.error_class,
                output=rejected.output,
            )
            return rejected
        spec = self._spec(call.tool)
        watch = Stopwatch()
        self._emit(LifecycleEvent.TOOL_START, {"tool": call.tool, "step": call.step})
        await self._opened(runtime, call)
        with runtime.tracer.tool_span(
            call.tool,
            **{
                N.TOOL_VERSION: spec.version if spec else None,
                N.TOOL_SOURCE: spec.source if spec else None,
                N.TOOL_SERVER: spec.server if spec else None,
                N.TOOL_IDEMPOTENCY_KEY: call.idempotency_key,
                N.TOOL_ARGS_SCHEMA: sorted(call.args),
            },
        ) as span:
            span.set_input(call.args, category="tool")
            try:
                outcome = await self._bounded(self._execute(call))
            except asyncio.CancelledError:
                span.error("cancelled", **{N.STATUS: "cancelled"})
                self._finish(call, None, watch.ms, "cancelled")
                raise
            except Exception as exc:
                span.error(exc, **{N.STATUS: "error"})
                self._finish(call, None, watch.ms, "error", error=exc)
                await self._record_memory(call, None, watch.ms, "error", error=exc)
                await self._closed(
                    runtime,
                    call,
                    status="error",
                    error_class=type(exc).__name__,
                    output=str(exc),
                )
                if isinstance(exc, ToolError):
                    raise
                raise ToolError(str(exc), source=f"tool.{call.tool}") from exc
            outcome = outcome.model_copy(update={"latency_ms": watch.ms, "tool": call.tool})
            span.set(
                **{
                    N.STATUS: outcome.status,
                    N.TOOL_CACHED: outcome.cached,
                    N.DURATION_MS: outcome.latency_ms,
                    N.TOOL_RESULT_REF: outcome.invocation_id,
                }
            )
            span.set_output(outcome.output, category="tool")
            span.ok()
        self._finish(call, outcome, watch.ms, outcome.status)
        await self._record_memory(call, outcome, watch.ms, outcome.status)
        await self._closed(
            runtime,
            call,
            status=str(outcome.status),
            output=outcome.output,
            invocation_id=outcome.invocation_id,
        )
        return outcome

    # -- the run's event stream ----------------------------------------------------------
    @staticmethod
    async def _opened(runtime: AgentRuntime, call: ToolCall) -> None:
        await runtime.events.emit(
            RunEventType.TOOL_CALL_START, tool_call_id=_call_id(call), tool=call.tool
        )
        await runtime.events.emit(
            RunEventType.TOOL_CALL_ARGS, tool_call_id=_call_id(call), args=call.args
        )

    @staticmethod
    async def _closed(
        runtime: AgentRuntime,
        call: ToolCall,
        *,
        status: str,
        output: Any = None,
        error_class: str | None = None,
        invocation_id: str | None = None,
    ) -> None:
        await runtime.events.emit(
            RunEventType.TOOL_CALL_END, tool_call_id=_call_id(call), tool=call.tool
        )
        await runtime.events.emit(
            RunEventType.TOOL_CALL_RESULT,
            tool_call_id=_call_id(call),
            tool=call.tool,
            status=status,
            output=_brief(output),
            error_class=error_class,
            invocation_id=invocation_id,
        )

    # -- plumbing --------------------------------------------------------------------
    async def _execute(self, call: ToolCall) -> ToolOutcome:
        result = await self._client.call(call)
        if isinstance(result, ToolOutcome):
            return result
        return ToolOutcome(tool=call.tool, status=ToolStatus.OK, output=result)

    def _spec(self, tool: str) -> ToolSpec | None:
        """Tool clients may expose specs; those that do not simply report less."""
        getter = getattr(self._client, "spec", None)
        if not callable(getter):
            return None
        spec = getter(tool)
        return spec if isinstance(spec, ToolSpec) else None

    async def _authorize(self, call: ToolCall) -> tuple[ToolCall, ToolOutcome | None]:
        """The call to run (its arguments possibly edited by an approver), or a rejected
        outcome. A policy that requires approval pauses the run with the call, unless the
        person already answered: a resumed run finds the resolution in ``runtime.state``,
        and it counts only for the arguments the approver saw. Edited arguments go through
        the policy again: an approver may narrow a call, never widen it past a denial."""
        if self._policy is None or self._runtime is None:
            return call, None
        outcome, reason = normalize(await self._policy.authorize_tool(self._runtime.context, call))
        if outcome is PolicyOutcome.ALLOW:
            return call, None
        if outcome is PolicyOutcome.DENY:
            raise PolicyDeniedError(
                reason or f"tool {call.tool!r} denied by policy", source="policy.tool"
            )
        resolved = self._runtime.state.get("resolutions", {}).get(call.idempotency_key)
        if resolved is None or not _same_call(resolved.interrupt.tool_call, call):
            raise ApprovalRequired(call, reason=reason)
        decision = resolved.decision
        if decision is InterruptDecision.APPROVE:
            return call, None
        if decision is InterruptDecision.EDIT:
            edited = call.model_copy(update={"args": dict(resolved.payload or {})})
            verdict, why = normalize(
                await self._policy.authorize_tool(self._runtime.context, edited)
            )
            if verdict is PolicyOutcome.DENY:
                raise PolicyDeniedError(
                    why or f"the edited arguments for {call.tool!r} are denied by policy",
                    source="policy.tool",
                )
            return edited, None
        if decision is InterruptDecision.CANCEL:
            # the person abandoned the run, not just the call: it ends CANCELLED
            self._runtime.cancellation.cancel("cancelled by the approver")
            raise AgentCancelledError("cancelled by the approver", source="policy.tool")
        return call, ToolOutcome(
            tool=call.tool,
            status=ToolStatus.REJECTED,
            output=f"the call to {call.tool} was rejected by the approver",
            error_class="ApprovalRejected",
            metadata={"interrupt_id": resolved.interrupt.interrupt_id},
        )

    async def _bounded(self, awaitable: Any) -> Any:
        remaining = self._runtime.remaining_seconds if self._runtime else None
        budget = [v for v in (self.timeout, remaining) if v is not None]
        if not budget:
            return await awaitable
        return await asyncio.wait_for(awaitable, min(budget))

    def _finish(
        self,
        call: ToolCall,
        outcome: ToolOutcome | None,
        ms: float,
        status: str,
        *,
        error: BaseException | None = None,
    ) -> None:
        runtime = self._runtime
        if runtime is None:
            return
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
            }
        )
        self._emit(
            LifecycleEvent.TOOL_END,
            {"tool": call.tool, "status": status, "latency_ms": ms},
        )

    async def _record_memory(
        self,
        call: ToolCall,
        outcome: ToolOutcome | None,
        ms: float,
        status: str,
        *,
        error: BaseException | None = None,
    ) -> None:
        """Record the invocation in tool memory so the service can learn procedures (§12)."""
        runtime = self._runtime
        if not self.record_to_memory or runtime is None or not runtime.memory.enabled:
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
                task=_task_of(call, runtime),
                step=call.step,
            )
        except Exception:
            runtime.logger.debug("tool memory recording failed", tool=call.tool)

    def _emit(self, event: LifecycleEvent, payload: dict[str, Any]) -> None:
        if self._events is not None and self._runtime is not None:
            self._events.emit(event, {"context": self._runtime.context, **payload})


def _same_call(held: ToolCall | None, call: ToolCall) -> bool:
    """Whether ``call`` is the call a person approved: the same tool with the same arguments
    (a resumed agent re-plans from scratch, and its second attempt may ask for more)."""
    return held is not None and held.tool == call.tool and held.args == call.args


def _call_id(call: ToolCall) -> str:
    """The id a surface correlates the tool events by: the call's idempotency key, which is
    stable across a pause and its resume, so an approval resolves the same call."""
    return call.idempotency_key or f"call:{call.tool}:{call.step}"


#: How much of a tool result rides on the stream; the rest is reachable by reference.
RESULT_PREVIEW_CHARS = 2000


def _brief(output: Any) -> Any:
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


def outcome_with_artifact(outcome: ToolOutcome, artifact: ArtifactRef) -> ToolOutcome:
    """Replace a large tool output with a reference to a stored artifact (§43)."""
    return outcome.model_copy(
        update={
            "artifacts": [*outcome.artifacts, artifact],
            "output": None,
            "output_summary": outcome.output_summary or f"artifact:{artifact.artifact_id}",
        }
    )


def _task_of(call: Any, runtime: Any) -> str:
    """What this tool call was *for*, as the service keys procedures on.

    The service normalises this into a typed-placeholder pattern ("how much stock of
    {entity}?") so two phrasings of the same question mine the same trajectory. It was being
    sent ``context.task_id`` — an opaque identifier — so the pattern either matched only
    other runs of that same id, or, far more often, was empty: no task_id, no pattern, no
    procedure, ever. The turn's question is what it wanted.
    """
    explicit = getattr(call, "task", None)
    if explicit:
        return str(explicit)
    request = runtime.state.get("request") if hasattr(runtime, "state") else None
    query = getattr(request, "query", None)
    if query:
        return str(query)
    return runtime.context.task_id or ""
