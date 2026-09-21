"""``InstrumentedToolClient`` — what ``runtime.tools`` is (§12).

Captured per call: tool, version, which arguments were passed (names and types, not
values, unless capture allows), start/end, latency, status, retries, a result reference,
any artifact, the error, the span and the agent run. Idempotency material travels with the
call so a tool that supports it can deduplicate a retried write (§41); the harness never
retries a tool it has not been told is idempotent.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from universal_agent_contracts.artifacts import ArtifactRef
from universal_agent_contracts.errors import PolicyDeniedError, ToolError
from universal_agent_contracts.events import LifecycleEvent
from universal_agent_contracts.tool import ToolCall, ToolOutcome, ToolSpec

from universal_agent_harness.telemetry import names as N
from universal_agent_harness.telemetry.metrics import TOOL_CALLS, TOOL_LATENCY
from universal_agent_harness.telemetry.tracer import Stopwatch

if TYPE_CHECKING:  # pragma: no cover
    from universal_agent_harness.runtime.agent_runtime import AgentRuntime


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
        await self._authorize(call)
        spec = self._spec(call.tool)
        watch = Stopwatch()
        self._emit(LifecycleEvent.TOOL_START, {"tool": call.tool, "step": call.step})
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
        return outcome

    # -- plumbing --------------------------------------------------------------------
    async def _execute(self, call: ToolCall) -> ToolOutcome:
        result = await self._client.call(call)
        if isinstance(result, ToolOutcome):
            return result
        return ToolOutcome(tool=call.tool, status="ok", output=result)

    def _spec(self, tool: str) -> ToolSpec | None:
        """Tool clients may expose specs; those that do not simply report less."""
        getter = getattr(self._client, "spec", None)
        if not callable(getter):
            return None
        spec = getter(tool)
        return spec if isinstance(spec, ToolSpec) else None

    async def _authorize(self, call: ToolCall) -> None:
        if self._policy is None or self._runtime is None:
            return
        decision = await self._policy.authorize_tool(self._runtime.context, call)
        if decision is not True:
            raise PolicyDeniedError(
                decision if isinstance(decision, str) else f"tool {call.tool!r} denied by policy",
                source="policy.tool",
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
