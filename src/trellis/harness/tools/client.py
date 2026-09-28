"""``InstrumentedToolClient`` — what ``runtime.tools`` is (§12).

Captured per call: tool, version, which arguments were passed (names and types, not
values, unless capture allows), start/end, latency, status, retries, a result reference,
any artifact, the error, the span and the agent run. Idempotency material travels with the
call so a tool that supports it can deduplicate a retried write (§41); the harness never
retries a tool it has not been told is idempotent.

The policy decision, the event pair and the tool-memory record are
:class:`~trellis.harness.tools.bridge.ToolCallBridge`'s, shared with the framework adapters
(design §8) so "what ``require_approval`` means" has one implementation whoever runs the tool.
This class adds what only an executing client can: the execution itself, the deadline, and
the span attributes that come from the tool's spec.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from trellis.contracts import ToolStatus
from trellis.contracts.artifacts import ArtifactRef
from trellis.contracts.errors import AgentPaused, ToolError
from trellis.contracts.tool import ToolCall, ToolOutcome, ToolSpec

from trellis.harness.telemetry import names as N
from trellis.harness.telemetry.tracer import Stopwatch
from trellis.harness.tools.bridge import ToolCallBridge

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
        self._bridge: ToolCallBridge | None = None
        if runtime_ref is not None:
            self._bridge = self._build_bridge(runtime_ref)

    def attach(self, runtime: AgentRuntime) -> None:
        self._runtime = runtime
        self._bridge = self._build_bridge(runtime)

    def _build_bridge(self, runtime: AgentRuntime) -> ToolCallBridge:
        return ToolCallBridge(
            runtime,
            policy=self._policy,
            events=self._events,
            record_to_memory=self.record_to_memory,
            source="harness",
        )

    @property
    def inner(self) -> Any:
        return self._client

    async def list_tools(self) -> list[ToolSpec]:
        return list(await self._client.list_tools())

    async def call(self, tool: str | ToolCall, /, **args: Any) -> ToolOutcome:
        """Execute a tool with full instrumentation. Returns a normalized outcome."""
        call = tool if isinstance(tool, ToolCall) else ToolCall(tool=tool, args=args)
        runtime, bridge = self._runtime, self._bridge
        if runtime is None or bridge is None:
            return await self._execute(call)

        call = bridge.prepare(call)
        call, rejected = await bridge.authorize(call)
        if rejected is not None:
            # a refused call is still a call: it opens and closes on the stream like one
            # that ran, and tool memory learns that it was refused
            await bridge.opened(call, lifecycle=False)
            await bridge.settled(call, rejected, 0.0, str(rejected.status))
            return rejected
        spec = self._spec(call.tool)
        watch = Stopwatch()
        await bridge.opened(call)
        with bridge.tool_span(
            call,
            **{
                N.TOOL_VERSION: spec.version if spec else None,
                N.TOOL_SOURCE: spec.source if spec else None,
                N.TOOL_SERVER: spec.server if spec else None,
            },
        ) as span:
            span.set_input(call.args, category="tool")
            try:
                outcome = await self._bounded(self._execute(call))
            except asyncio.CancelledError:
                span.error("cancelled", **{N.STATUS: "cancelled"})
                bridge.finished(call, None, watch.ms, "cancelled")
                raise
            except AgentPaused:
                # A tool that needs a person pauses the run; so does a remote agent whose task
                # says ``input-required`` (the A2A tool client). That is not a failed call: the
                # span is fine, and tool memory must not learn a rejection that never happened.
                # The call stays open on the stream and closes when the resumed run completes it,
                # exactly as a call held for approval does.
                span.ok()
                raise
            except Exception as exc:
                span.error(exc, **{N.STATUS: "error"})
                await bridge.settled(call, None, watch.ms, "error", error=exc)
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
        await bridge.settled(call, outcome, watch.ms, str(outcome.status))
        return outcome

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

    async def _bounded(self, awaitable: Any) -> Any:
        remaining = self._runtime.remaining_seconds if self._runtime else None
        budget = [v for v in (self.timeout, remaining) if v is not None]
        if not budget:
            return await awaitable
        return await asyncio.wait_for(awaitable, min(budget))


def outcome_with_artifact(outcome: ToolOutcome, artifact: ArtifactRef) -> ToolOutcome:
    """Replace a large tool output with a reference to a stored artifact (§43)."""
    return outcome.model_copy(
        update={
            "artifacts": [*outcome.artifacts, artifact],
            "output": None,
            "output_summary": outcome.output_summary or f"artifact:{artifact.artifact_id}",
        }
    )
