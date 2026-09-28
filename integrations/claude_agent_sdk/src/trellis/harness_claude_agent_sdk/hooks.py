"""``TrellisHooks``: the Claude Agent SDK's hooks, bound to the harness (design §8).

| Moment | Hook | What it does |
|---|---|---|
| context | ``UserPromptSubmit`` | returns the memory bundle as ``additionalContext`` |
| tool | ``PreToolUse`` | policy: ``permissionDecision`` allow / deny, plus ``updatedInput`` |
| tool | ``PostToolUse`` | closes the call on the run's stream and writes tool memory |
| pause | ``can_use_tool`` | the same decision, through the SDK's permission callback |
| compaction | ``PreCompact`` | summarises what was said and remembers it, run-scoped |
| run end | ``Stop`` / ``SubagentStop`` | writes the answer and closes the step |

The policy decision is made **once** per tool call, by :meth:`TrellisHooks.authorize`, and
cached on the runtime under the call's id. The SDK offers two seams onto the same moment —
the ``PreToolUse`` hook and the ``can_use_tool`` callback — and a deployment may have both
installed; one decision serving both is what keeps a single call from being authorized twice,
counted twice, or paused twice.

Every hook here is a plain async callable over dictionaries, so the whole binding is testable
without spawning the ``claude`` CLI — which is the only thing that can actually run a tool.

The SDK types a hook's payload as ``HookInput``, a union of ten TypedDicts, and its answer as
``HookJSONOutput``. A hook that reads ``tool_name`` cannot be typed against the union (nine of
its members have no such key), so each hook narrows its payload once, explicitly, and builds
its answer as a plain dict. That is the shape the CLI sends and expects on the wire.
"""

from __future__ import annotations

import asyncio
from asyncio import Task
from enum import StrEnum
from typing import Any, Final, cast

from claude_agent_sdk import (
    HookContext,
    HookInput,
    HookJSONOutput,
    PermissionResult,
    PermissionResultAllow,
    PermissionResultDeny,
    ToolPermissionContext,
)
from trellis.contracts import ToolStatus
from trellis.contracts.artifacts import MemoryObservation
from trellis.contracts.model import ModelRequest
from trellis.contracts.runs import RunEventType
from trellis.contracts.tool import ToolCall, ToolOutcome

from trellis.harness.interrupts.signals import ApprovalRequired
from trellis.harness.reasoning.assembler import (
    INTERNAL,
    SUMMARY_KIND,
    SUMMARY_PROMPT,
    ContextAssembler,
)
from trellis.harness.runtime.propagation import require_runtime
from trellis.harness.telemetry.tracer import Stopwatch
from trellis.harness.tools.bridge import ToolCallBridge

__all__ = ["FRAMEWORK", "STEP", "TrellisHooks", "Verdict"]

FRAMEWORK: Final = "claude-agent-sdk"
HINT: Final = "run the agent through harness.claude_agent_sdk.wrap(...) / .agent(...)"
#: The step a Claude Agent SDK run appears as on the event stream.
STEP: Final = "claude_agent_sdk.agent"
#: Where the per-execution pieces live on ``runtime.state``.
BRIDGE_KEY: Final = "claude_agent_sdk.bridge"
DECISIONS_KEY: Final = "claude_agent_sdk.decisions"
PENDING_KEY: Final = "claude_agent_sdk.pending_approval"
TRANSCRIPT_KEY: Final = "claude_agent_sdk.transcript"
COMPACTIONS_KEY: Final = "claude_agent_sdk.compactions"


class Verdict(StrEnum):
    """What the policy said about one call. Exactly three answers, so an enum.

    Encoding this as a pair of flags (``paused: bool`` plus ``rejected: ToolOutcome | None``)
    made "paused *and* rejected" expressible, and nothing would have caught a decision in
    that state: ``pre_tool_use`` tests ``paused`` first and ``post_tool_use`` tests both.
    """

    ALLOW = "allow"
    REJECTED = "rejected"
    PAUSED = "paused"


class _Decision:
    """One authorized call: what to run, or why it will not run."""

    __slots__ = ("call", "outcome", "reason", "verdict", "watch")

    def __init__(
        self,
        call: ToolCall,
        verdict: Verdict,
        *,
        outcome: ToolOutcome | None = None,
        reason: str | None = None,
    ) -> None:
        self.call = call
        self.verdict = verdict
        #: The refusal the model is shown, when the verdict is ``REJECTED``.
        self.outcome = outcome
        self.reason = reason
        self.watch = Stopwatch()


class TrellisHooks:
    """The harness's side of a Claude Agent SDK run.

        hooks = harness.claude_agent_sdk.hooks()
        options = harness.claude_agent_sdk.options(system_prompt="…", tools=["Read"])

    Built by :class:`ClaudeAgentSDKHarness`; a test constructs one directly and calls the
    hooks with the payloads the CLI would send.
    """

    def __init__(
        self,
        runtime: Any = None,
        *,
        policy: Any = None,
        skills: Any = (),
        record_to_memory: bool = True,
        inject_context: bool = True,
    ) -> None:
        self._runtime = runtime
        self.policy = policy
        self.skills = list(skills)
        self.record_to_memory = record_to_memory
        self.inject_context = inject_context

    # ------------------------------------------------------------------ per-execution state
    @property
    def runtime(self) -> Any:
        return require_runtime(self._runtime, hint=HINT)

    def bridge(self, runtime: Any) -> ToolCallBridge:
        found = runtime.state.get(BRIDGE_KEY)
        if found is None:
            found = ToolCallBridge(
                runtime,
                policy=self.policy,
                record_to_memory=self.record_to_memory,
                source=FRAMEWORK,
            )
            runtime.state[BRIDGE_KEY] = found
        return found

    # ------------------------------------------------------------------ context
    async def user_prompt_submit(
        self,
        payload: HookInput,
        tool_use_id: str | None = None,
        context: HookContext | None = None,
    ) -> HookJSONOutput:
        """Return the memory bundle as ``additionalContext`` (design §8).

        The SDK has no way to replace the system prompt of a running session, and this is
        the hook it offers instead: whatever is returned here is prepended to the turn the
        person just submitted. Rendered by the core's :class:`ContextAssembler`, so a fact
        reads the same here as in the harness's own loop.
        """
        fields = _fields(payload)
        runtime = self.runtime
        self._remember_said("user", str(fields.get("prompt") or ""))
        if not self.inject_context:
            return {}
        rendered = ContextAssembler(prompt="", skills=self.skills).system_prompt(runtime)
        if not rendered.strip():
            return {}
        return _output(
            {
                "hookEventName": "UserPromptSubmit",
                "additionalContext": rendered,
            }
        )

    # ------------------------------------------------------------------ tool call
    async def authorize(
        self, tool_name: str, tool_input: dict[str, Any], call_id: str | None
    ) -> _Decision:
        """The policy decision for one call, made once and remembered by call id.

        ``PreToolUse`` and ``can_use_tool`` are two seams onto the same moment, and a
        deployment may have both. Deciding once means a call is authorized once, counted
        once and paused once, whichever seam the CLI reaches first.

        The memo holds the **task**, not the result. Caching the result would have left the
        whole policy call between the cache check and the cache write, so two seams arriving
        for the same call before the first finished would each consult the policy, each emit
        ``TOOL_CALL_START``, and each be able to pause the run — the exact thing this method
        exists to prevent. Awaiting a shared task gives every caller one decision.
        """
        runtime = self.runtime
        decisions: dict[str, Task[_Decision]] = runtime.state.setdefault(DECISIONS_KEY, {})
        key = str(call_id or tool_name)
        pending = decisions.get(key)
        if pending is None:
            pending = decisions[key] = asyncio.ensure_future(
                self._decide(runtime, tool_name, tool_input, call_id)
            )
        return await pending

    async def _decide(
        self, runtime: Any, tool_name: str, tool_input: dict[str, Any], call_id: str | None
    ) -> _Decision:
        """Ask the policy about one call and announce it. Runs exactly once per call."""
        bridge = self.bridge(runtime)
        call = bridge.prepare(tool_name, dict(tool_input or {}), call_id=call_id)
        try:
            call, rejected = await bridge.authorize(call)
        except ApprovalRequired as paused:
            # the CLI cannot suspend a tool call and come back to it, so the run is stopped
            # and the harness turns the pause into an Interrupt the platform already knows
            runtime.state[PENDING_KEY] = paused
            await bridge.opened(paused.tool_call)
            return _Decision(paused.tool_call, Verdict.PAUSED, reason="awaiting approval")
        await bridge.opened(call)
        if rejected is not None:
            await bridge.settled(call, rejected, 0.0, str(rejected.status))
            return _Decision(call, Verdict.REJECTED, outcome=rejected)
        return _Decision(call, Verdict.ALLOW)

    async def pre_tool_use(
        self,
        payload: HookInput,
        tool_use_id: str | None = None,
        context: HookContext | None = None,
    ) -> HookJSONOutput:
        """``permissionDecision`` for the call the CLI is about to make."""
        fields = _fields(payload)
        decision = await self.authorize(
            str(fields.get("tool_name") or ""),
            dict(fields.get("tool_input") or {}),
            fields.get("tool_use_id") or tool_use_id,
        )
        output: dict[str, Any] = {"hookEventName": "PreToolUse"}
        if decision.verdict is Verdict.PAUSED:
            output["permissionDecision"] = "deny"
            output["permissionDecisionReason"] = str(decision.reason)
        elif decision.verdict is Verdict.REJECTED:
            output["permissionDecision"] = "deny"
            output["permissionDecisionReason"] = str(
                decision.outcome.output if decision.outcome else "rejected by the approver"
            )
        else:
            output["permissionDecision"] = "allow"
            asked = dict(fields.get("tool_input") or {})
            if decision.call.args != asked:
                # an approver narrowed the arguments: the CLI must run what was approved
                output["updatedInput"] = dict(decision.call.args)
        return _output(output)

    async def can_use_tool(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        context: ToolPermissionContext | None = None,
    ) -> PermissionResult:
        """The SDK's permission callback, over the same single decision."""
        decision = await self.authorize(
            tool_name, tool_input, getattr(context, "tool_use_id", None)
        )
        if decision.verdict is Verdict.PAUSED:
            # ``interrupt`` stops the run rather than letting the model plan around a denial
            # it cannot fix: a person has been asked, and the answer is not here yet
            return PermissionResultDeny(message=str(decision.reason), interrupt=True)
        if decision.verdict is Verdict.REJECTED:
            return PermissionResultDeny(
                message=str(decision.outcome.output if decision.outcome else "rejected"),
                interrupt=False,
            )
        return PermissionResultAllow(updated_input=dict(decision.call.args))

    async def post_tool_use(
        self,
        payload: HookInput,
        tool_use_id: str | None = None,
        context: HookContext | None = None,
    ) -> HookJSONOutput:
        """Close the call on the run's stream and write it to tool memory."""
        fields = _fields(payload)
        runtime = self.runtime
        decisions: dict[str, Task[_Decision]] = runtime.state.setdefault(DECISIONS_KEY, {})
        key = str(fields.get("tool_use_id") or tool_use_id or fields.get("tool_name") or "")
        task = decisions.pop(key, None)
        if task is None:
            return {}
        decision = await task
        if decision.verdict is not Verdict.ALLOW:
            # a refused or paused call never ran, so there is nothing to close here: the
            # decision itself already closed it on the stream
            return {}
        response = fields.get("tool_response")
        failed = isinstance(response, dict) and bool(response.get("is_error"))
        outcome = ToolOutcome(
            tool=decision.call.tool,
            status=ToolStatus.ERROR if failed else ToolStatus.OK,
            output=response,
            error_class="ToolError" if failed else None,
        )
        await self.bridge(runtime).settled(
            decision.call, outcome, decision.watch.ms, str(outcome.status)
        )
        return {}

    # ------------------------------------------------------------------ compaction
    async def pre_compact(
        self,
        payload: HookInput,
        tool_use_id: str | None = None,
        context: HookContext | None = None,
    ) -> HookJSONOutput:
        """Summarise the conversation and remember the summary, run-scoped.

        ``PreCompact`` carries only the trigger (``manual``/``auto``) and any custom
        instructions — never the summary the CLI is about to make, and the SDK has no hook
        that does. So the adapter makes its own, from the turns it has seen on the stream,
        through the harness's model port, and writes it exactly as
        ``ContextAssembler.compact`` does. What is remembered is therefore the harness's
        summary of the same conversation, not the CLI's — the README says so.
        """
        fields = _fields(payload)
        runtime = self.runtime
        memory = runtime.memory
        transcript = "\n".join(
            f"{role}: {text}" for role, text in runtime.state.get(TRANSCRIPT_KEY, ())
        )
        if not transcript.strip() or not memory.enabled:
            return {}
        response = await runtime.model.invoke(
            ModelRequest(
                messages=[
                    {"role": "system", "content": SUMMARY_PROMPT},
                    {"role": "user", "content": transcript},
                ],
                # the harness's own call: no surface shows the summary as an answer
                metadata={INTERNAL: "compaction"},
            )
        )
        summary = (response.text or "").strip()
        if not summary:
            return {}
        compactions = runtime.state.get(COMPACTIONS_KEY, 0) + 1
        runtime.state[COMPACTIONS_KEY] = compactions
        await memory.observe(
            MemoryObservation(
                content=f"Conversation summary: {summary}",
                kind=SUMMARY_KIND,
                # the run's own note: visible to this run and the one that spawned it
                hints={"visibility": "RUN"},
                metadata={
                    "source": "compaction",
                    "framework": FRAMEWORK,
                    "compaction": compactions,
                    "trigger": fields.get("trigger"),
                },
            )
        )
        return {}

    # ------------------------------------------------------------------ run end
    async def stop(
        self,
        payload: HookInput,
        tool_use_id: str | None = None,
        context: HookContext | None = None,
    ) -> HookJSONOutput:
        """Write the answer where the platform keeps answers, and close the step."""
        runtime = self.runtime
        answer = self.said(runtime, "assistant")
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
        await runtime.events.emit(RunEventType.STEP_FINISHED, step=STEP, framework=FRAMEWORK)
        return {}

    async def subagent_stop(
        self,
        payload: HookInput,
        tool_use_id: str | None = None,
        context: HookContext | None = None,
    ) -> HookJSONOutput:
        """A subagent finished: a step on the stream, not the run's answer."""
        await self.runtime.events.emit(
            RunEventType.STEP_FINISHED,
            step=f"{STEP}.subagent",
            framework=FRAMEWORK,
            agent_type=_fields(payload).get("agent_type"),
        )
        return {}

    # ------------------------------------------------------------------ the transcript
    def remember_said(self, role: str, text: str) -> None:
        """Record a turn the adapter saw, for compaction and for the run's answer."""
        self._remember_said(role, text)

    def _remember_said(self, role: str, text: str) -> None:
        if not text.strip():
            return
        runtime = self.runtime
        runtime.state.setdefault(TRANSCRIPT_KEY, []).append((role, text.strip()))

    @staticmethod
    def said(runtime: Any, role: str) -> str | None:
        """The last thing ``role`` said this run."""
        for said_role, text in reversed(runtime.state.get(TRANSCRIPT_KEY, ())):
            if said_role == role:
                return text
        return None


def _fields(payload: HookInput) -> dict[str, Any]:
    """A hook payload as the mapping it is on the wire.

    ``HookInput`` is a union of TypedDicts, so no single member can be indexed for a key only
    some of them declare. The CLI sends JSON objects and the SDK parses them into TypedDicts,
    which *are* dicts, so reading the fields a hook needs is exact — the union is the thing
    that cannot express it.
    """
    return cast("dict[str, Any]", payload)


def _output(specific: dict[str, Any]) -> HookJSONOutput:
    """A hook's answer in the shape the CLI expects."""
    return cast("HookJSONOutput", {"hookSpecificOutput": specific})
