"""``TrellisRunHooks`` and the tool guardrails: the tool, run-start and run-end moments.

The SDK splits what one hook does elsewhere across two mechanisms, and both are needed:

* ``RunHooks`` observe (``on_agent_start``, ``on_llm_start/end``, ``on_tool_start/end``,
  ``on_agent_end``) — they cannot refuse anything, so they carry the events and tool memory;
  only ``on_agent_start``, ``on_tool_end`` and ``on_agent_end`` are overridden. A call is
  already open by the time ``on_tool_start`` runs (the guardrail opened it, because that is
  the last point a refusal is possible), and the model call is instrumented by
  ``runtime.model``, so overriding ``on_llm_start/end`` would double-count it;
* a ``ToolInputGuardrail`` *can* refuse — it runs before the tool and may allow, reject the
  content, or raise — so it carries the policy decision.

Together they are one tool moment: the guardrail authorizes through
:class:`~trellis.harness.tools.bridge.ToolCallBridge` (the same deny / require-approval /
approver-edit semantics the harness's own loop has) and the hooks open and close the call on
the run's event stream and write it to tool memory.
"""

from __future__ import annotations

from typing import Any, Final

from agents.lifecycle import RunHooks
from agents.tool_guardrails import (
    ToolGuardrailFunctionOutput,
    ToolInputGuardrail,
    ToolInputGuardrailData,
    ToolOutputGuardrail,
    ToolOutputGuardrailData,
)
from trellis.contracts import ToolStatus
from trellis.contracts.artifacts import MemoryObservation
from trellis.contracts.runs import RunEventType
from trellis.contracts.tool import ToolOutcome

from trellis.harness.reasoning.assembler import SUMMARY_KIND
from trellis.harness.runtime.propagation import require_runtime
from trellis.harness.telemetry.tracer import Stopwatch
from trellis.harness.tools.bridge import ToolCallBridge

__all__ = ["FRAMEWORK", "TrellisRunHooks", "tool_input_guardrail", "tool_output_guardrail"]

FRAMEWORK: Final = "openai-agents"
HINT: Final = "run the agent through harness.openai_agents.wrap(...) / .agent(...)"
#: The step an OpenAI Agents run appears as on the event stream.
STEP: Final = "openai_agents.agent"
#: Where the per-execution pieces are kept on ``runtime.state``.
BRIDGE_KEY: Final = "openai_agents.bridge"
CALLS_KEY: Final = "openai_agents.calls"


def bridge_for(
    runtime: Any, *, policy: Any = None, record_to_memory: bool = True
) -> ToolCallBridge:
    """This execution's bridge, created once so its step numbering is per run."""
    bridge = runtime.state.get(BRIDGE_KEY)
    if bridge is None:
        bridge = ToolCallBridge(
            runtime, policy=policy, record_to_memory=record_to_memory, source=FRAMEWORK
        )
        runtime.state[BRIDGE_KEY] = bridge
    return bridge


def _pending(runtime: Any) -> dict[str, Any]:
    """Calls the guardrail authorized, waiting for their hooks to close them."""
    return runtime.state.setdefault(CALLS_KEY, {})


def _call_key(context: Any) -> str | None:
    """The id that ties this call's guardrails and hooks together, or ``None``.

    The SDK's own call id, and nothing else. Falling back to the tool *name* looked harmless
    and was not: the SDK runs a turn's tool calls concurrently, so two calls to the same tool
    would share a key — the second would overwrite the first's pending entry and one call
    would never be closed on the stream or recorded in tool memory. With no id there is
    nothing to correlate, and saying so is better than correlating the wrong things.
    """
    call_id = getattr(context, "tool_call_id", None)
    return str(call_id) if call_id else None


def tool_input_guardrail(
    *, policy: Any = None, record_to_memory: bool = True
) -> ToolInputGuardrail[Any]:
    """The policy decision, as the SDK's own refusal mechanism.

    A denial raises ``PolicyDeniedError`` and a call the policy holds for a person raises
    ``ApprovalRequired``; both leave ``Runner.run``, and the harness turns the second into a
    paused run. An approver's *rejection* comes back as ``reject_content``, which the SDK
    hands to the model as the tool's result — so the agent can plan around it, which is the
    difference between a rejection and a denial.
    """

    async def authorize(data: ToolInputGuardrailData) -> ToolGuardrailFunctionOutput:
        runtime = require_runtime(hint=HINT)
        bridge = bridge_for(runtime, policy=policy, record_to_memory=record_to_memory)
        context = data.context
        call = bridge.prepare(
            str(getattr(context, "tool_name", "") or ""),
            _arguments(context),
            call_id=getattr(context, "tool_call_id", None),
        )
        call, rejected = await bridge.authorize(call)
        await bridge.opened(call)
        if rejected is not None:
            await bridge.settled(call, rejected, 0.0, str(rejected.status))
            return ToolGuardrailFunctionOutput.reject_content(
                message=str(rejected.output), output_info={"tool": call.tool}
            )
        key = _call_key(context)
        if key is not None:
            _pending(runtime)[key] = (call, Stopwatch())
        else:
            # nothing to correlate the result by, so the call is closed here rather than
            # left open forever waiting for a hook that cannot find it
            await bridge.settled(call, None, 0.0, str(ToolStatus.OK))
        return ToolGuardrailFunctionOutput.allow({"tool": call.tool})

    return ToolInputGuardrail(guardrail_function=authorize, name="trellis.policy")


def tool_output_guardrail() -> ToolOutputGuardrail[Any]:
    """Closes the call the input guardrail opened, with what the tool actually returned.

    ``on_tool_end`` would also see the result, but only the guardrail is guaranteed to run
    for every tool the guardrail admitted — so the call that was opened is the call that gets
    closed, and tool memory never learns half a story.
    """

    async def record(data: ToolOutputGuardrailData) -> ToolGuardrailFunctionOutput:
        runtime = require_runtime(hint=HINT)
        key = _call_key(data.context)
        found = _pending(runtime).pop(key, None) if key is not None else None
        if found is None:
            return ToolGuardrailFunctionOutput.allow()
        call, watch = found
        bridge = bridge_for(runtime)
        outcome = ToolOutcome(tool=call.tool, status=ToolStatus.OK, output=data.output)
        await bridge.settled(call, outcome, watch.ms, str(outcome.status))
        return ToolGuardrailFunctionOutput.allow({"tool": call.tool})

    return ToolOutputGuardrail(guardrail_function=record, name="trellis.tool_memory")


def _arguments(context: Any) -> dict[str, Any]:
    """The arguments the model asked for, whatever shape the SDK handed them over in."""
    import json  # noqa: PLC0415 - only on the tool path

    raw = getattr(context, "tool_arguments", None)
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except ValueError:
            return {"arguments": raw}
        return parsed if isinstance(parsed, dict) else {"arguments": parsed}
    return {}


class TrellisRunHooks(RunHooks[Any]):
    """The run's lifecycle on the harness's event stream and in its memory."""

    def __init__(self, *, record_to_memory: bool = True) -> None:
        self.record_to_memory = record_to_memory

    async def on_agent_start(self, context: Any, agent: Any) -> None:
        runtime = require_runtime(hint=HINT)
        await runtime.events.emit(
            RunEventType.STEP_STARTED,
            step=STEP,
            framework=FRAMEWORK,
            agent=getattr(agent, "name", None),
        )

    async def on_tool_end(self, context: Any, agent: Any, tool: Any, result: Any) -> None:
        """Close a call the output guardrail did not (a tool with no guardrails attached)."""
        runtime = require_runtime(hint=HINT)
        key = _call_key(context)
        found = _pending(runtime).pop(key, None) if key is not None else None
        if found is None:
            return
        call, watch = found
        outcome = ToolOutcome(tool=call.tool, status=ToolStatus.OK, output=result)
        await bridge_for(runtime).settled(call, outcome, watch.ms, str(outcome.status))

    async def on_agent_end(self, context: Any, agent: Any, output: Any) -> None:
        """Write the answer where the platform keeps answers, and close the step."""
        runtime = require_runtime(hint=HINT)
        answer = output if isinstance(output, str) else None
        memory = runtime.memory
        if answer and self.record_to_memory and memory.enabled:
            policy = memory.policy
            if policy.record_messages:
                await memory.record_output(answer)
            if policy.observe_output:
                await memory.observe(
                    MemoryObservation(
                        content=answer,
                        kind=SUMMARY_KIND,
                        metadata={"agent_id": runtime.agent_id, "framework": FRAMEWORK},
                    )
                )
        await runtime.events.emit(
            RunEventType.STEP_FINISHED,
            step=STEP,
            framework=FRAMEWORK,
            agent=getattr(agent, "name", None),
        )
