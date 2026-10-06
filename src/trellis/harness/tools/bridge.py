"""Every harness tool call, whichever framework makes it: replay, governance, execution,
record.

1. **replay** — a call the journal already has (a resumed run re-planning the same step)
   returns its recorded output and runs nothing; a tool of a feature the run is without
   (``without=``: a graph's tools are bound when it is built) is an error the model reads;
2. **hooks** — the run's ``before_tool`` hooks (``trellis.harness.hooks``) may deny the call,
   rewrite its arguments or ask a person about it; their decision is journaled;
3. **governance** — the run's tenant's :class:`~trellis.harness.governance.Governance` decides,
   by the tool's name, as the catalog says at the time of the call: ``run`` runs, ``announce``
   is announced on the run's stream, ``ask`` pauses the run for approval (an approver may edit
   the arguments, or reject);
4. **execution** — inside a span, with ``TOOL_CALL_*`` events around it (the stream gets the
   arguments and the output redacted; the tool and the model get them as they are); the
   ``after_tool`` hooks may change the outcome the model reads;
5. **record** — journaled for a later resume (and on the run's trajectory, which evaluators
   read; in a worker, saved as the run's progress checkpoint: at once after a call with side
   effects), counted, and (memory on) sent, redacted, to the memory service's tool records in
   the background.

Execution is bounded: a call takes at most its tool's ``timeout`` and what is left of the
run's time; a call that only reads (or is idempotent) is tried again after an error that may
pass, within that time; the tool reads its idempotency key from ``trellis.current()``. A call
that does more than read is marked started, and saved, before it runs: one that times out, or
was running when its worker died, has an unknown effect — the model is told so, the journal
keeps what it was told, and it is never run again blind (unless its tool is idempotent, or
continues where it was: a sub-agent's run).

Calls may come at once (``ReAct``'s reads, the frameworks that run tools concurrently): their
steps are numbered as they arrive (or as the caller numbered them, ``step=``), identical calls
take their turn (``Replay.exclusive``), and the journal's progress saves go one at a time.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Final

from trellis.contracts import (
    InterruptDecision,
    RunEventType,
    ToolCall,
    ToolError,
    ToolOutcome,
    ToolSpec,
    ToolStatus,
)
from trellis.harness.asking import RunCancelled, answer_of
from trellis.harness.events import DECISION, NOTICE
from trellis.harness.governance.decision import Decision
from trellis.harness.hooks import Ask, Deny, denied, noted, read
from trellis.harness.journal import OUTCOME, content_key
from trellis.harness.runtime import Paused, Runtime, current, reason_of
from trellis.harness.telemetry import metrics, tool_span
from trellis.harness.telemetry import output as span_output
from trellis.harness.tools.base import (
    OUTCOME_UNKNOWN,
    UNKNOWN,
    Tool,
    execute,
    interrupted,
)

#: The ``TOOL_CALL_RESULT`` ``status`` of a call that asked a person, and ended with its
#: attempt (the other calls cut short: ``cancelled``, ``timeout``; docs/observability.md).
PAUSED: Final = "paused"
CANCELLED: Final = ToolStatus.CANCELLED.value
#: What that result's ``output`` says, by its ``status``.
CUT: Final = {
    PAUSED: "{tool} paused: the run waits on a person, and the call runs again when it resumes",
    CANCELLED: "{tool} was cut short: the run was cancelled",
    ToolStatus.TIMEOUT.value: "{tool} was cut short: the run ran out of time",
}


async def call(
    tool: Tool, args: dict[str, Any], *, call_id: str | None = None, step: int | None = None
) -> ToolOutcome:
    """Run ``tool`` for the current run (``step``: the call's number, when the caller numbered
    the calls it makes at once). Pauses propagate; every other failure is an outcome the
    framework shows its model (``ERROR``, or ``TIMEOUT``)."""
    runtime = current()
    if runtime is None:
        raise ToolError(
            f"{tool.name} is a trellis tool: it runs inside a Harness run (agent.run/stream)",
            source="tools",
        )
    key = content_key("call", tool.name, args)
    step = runtime.next_step() if step is None else step
    async with runtime.replay.exclusive(key):
        return await _called(runtime, tool, args, key=key, step=step, call_id=call_id)


async def permitted(spec: ToolSpec, args: dict[str, Any]) -> ToolCall | ToolOutcome:
    """A call its framework runs itself (Claude Code's built-in tools, through its permission
    callback), decided as a harness call is — the run's ``before_tool`` hooks, then governance,
    and a person when it asks — but not run, journaled or recorded here: the call to let
    through (its arguments rewritten or edited, perhaps), or the outcome of one that is not
    (denied, rejected). A pause propagates."""
    runtime = current()
    assert runtime is not None  # a framework asks inside the run it works for
    key = content_key("call", spec.name, args)
    call = ToolCall(tool=spec.name, args=args, task=runtime.task, idempotency_key=key)
    call, verdict = await _hooked(runtime, call, key)
    if isinstance(verdict, Deny):
        return denied(call, verdict)
    _, call, rejected = await _decided(runtime, spec, call, verdict)
    return call if rejected is None else rejected


async def _called(
    runtime: Runtime,
    tool: Tool,
    args: dict[str, Any],
    *,
    key: str,
    step: int,
    call_id: str | None,
) -> ToolOutcome:
    # what the tool hands its service: this run's n-th such call, in every attempt (the
    # call's own key, the same for the same call anywhere, is what feedback is filed under)
    idempotency_key = f"{runtime.run_id}:{key}:{runtime.replay.occurrence(key)}"
    tool_call = ToolCall(
        tool=tool.name, args=args, task=runtime.task, step=step, idempotency_key=call_id or key
    )
    ref = call_id or f"{runtime.run_id}:{step}"
    replayed, output = runtime.replay.call(key)
    if replayed:
        outcome = _replayed(tool.name, output)
        _events(runtime, ref, tool_call, outcome)
        return outcome
    if tool.feature is not None and not runtime.uses(tool.feature):
        # a tool the framework was built with, of a feature this run is without
        off = ToolOutcome(
            tool=tool.name,
            status=ToolStatus.ERROR,
            output=f"{tool.name} is off in this run (without {tool.feature})",
            error_class="FeatureOff",
        )
        _events(runtime, ref, tool_call, off)
        return off
    if runtime.replay.interrupted(key) and not (tool.spec.idempotent or tool.resumable):
        # its worker died while it ran: it is not run again blind (an idempotent tool is,
        # with the same key, below, and so is one that continues where it was)
        return await _interrupted(runtime, ref, tool_call, key)

    tool_call, verdict = await _hooked(runtime, tool_call, key)
    if isinstance(verdict, Deny):
        refused = denied(tool_call, verdict)
        _events(runtime, ref, tool_call, refused)
        return refused
    decision, tool_call, rejected = await _decided(runtime, tool.spec, tool_call, verdict)
    if rejected is not None:
        _events(runtime, ref, tool_call, rejected)
        return rejected
    _started(runtime, ref, tool.name, tool_call.args)
    try:
        outcome = await _ran(
            runtime, tool, tool_call, decision, key=key, ref=ref, idem=idempotency_key
        )
    except asyncio.CancelledError:
        # the run was cancelled, or ran out of time, while the call was under way
        out = runtime.cancelled is None and runtime.remaining() == 0
        _cut(runtime, ref, tool.name, ToolStatus.TIMEOUT.value if out else CANCELLED)
        raise
    except (Paused, RunCancelled) as cut:
        _cut(runtime, ref, tool.name, PAUSED if isinstance(cut, Paused) else CANCELLED)
        raise
    _ended(runtime, ref, tool.name, status=outcome.status.value, output=outcome.output)
    await _record(runtime, tool_call, outcome, decision, key=key)
    return outcome


async def _ran(
    runtime: Runtime,
    tool: Tool,
    tool_call: ToolCall,
    decision: Decision,
    *,
    key: str,
    ref: str,
    idem: str,
) -> ToolOutcome:
    """The call, started on the stream, under way: saved as started (one that does more than
    read), executed (``idem``: the key the tool hands its service), and its outcome as the
    ``after_tool`` hooks leave it. A pause (it asked a person) or a cancellation propagates."""
    args = tool_call.args
    runtime.used_code_mode |= tool.feature == "code_mode"
    runtime.used.add(tool.name)
    reads = decision.risk == "read"
    if not reads:
        # saved before it runs: the attempt after a crash knows it was running
        runtime.replay.start(key)
        await runtime.progress(now=True)
    started = time.perf_counter()
    action = decision.action.value
    with tool_span(tool.name, ref, args, source=tool.spec.source, action=action) as span:
        within = runtime.limited(tool.timeout, key=idem)
        outcome, error = await execute(tool, args, reads=reads, within=within)
        if isinstance(error, Paused | RunCancelled):
            if not reads:
                runtime.replay.unstart(key)  # it asked a person: it runs again on resume
            raise error
        span_output(span, outcome.output, key="gen_ai.tool.call.result")
    hooks = runtime.agent.hooks
    if error is not None:
        await hooks.failed("tool", error)
    outcome = await hooks.done(tool_call, outcome)
    outcome.latency_ms = round((time.perf_counter() - started) * 1000, 3)
    return outcome


async def _record(
    runtime: Runtime,
    call: ToolCall,
    outcome: ToolOutcome,
    decision: Decision,
    *,
    key: str,
) -> None:
    """A call that ran (and ended on the stream): journaled (or, failed, run again by a later
    attempt), saved, counted and recorded."""
    if outcome.status in (ToolStatus.OK, ToolStatus.TIMEOUT):
        runtime.replay.record_call(key, _journaled(outcome), tool=call.tool)
    elif decision.risk != "read":
        runtime.replay.unstart(key)  # it failed: a later attempt runs it again
    runtime.replay.record_step(call, outcome)
    # a call with side effects is saved at once: a crash after it does not repeat it
    await runtime.progress(now=not decision.runs)
    metrics.tool_called(call.tool, outcome.status.value)
    await runtime.agent.record_tool(runtime, call, outcome)


async def _hooked(runtime: Runtime, call: ToolCall, key: str) -> tuple[ToolCall, Deny | Ask | None]:
    """What the run's ``before_tool`` hooks decide about this occurrence of the call: its
    arguments (rewritten, perhaps) and a denial or a question. Journaled: a re-run reads it
    rather than asking the hooks again."""
    hooks = runtime.agent.hooks
    if not hooks:
        return call, None
    journaled = content_key("hooks", key, runtime.replay.occurrence(key))
    replayed, kept = runtime.replay.call(journaled)
    if not replayed:
        hooked, verdict = await hooks.tool(call)
        kept = {"args": hooked.args, "verdict": noted(verdict)}
        runtime.replay.record_call(journaled, kept)
    return call.model_copy(update={"args": kept["args"]}), read(kept["verdict"])


async def _decided(
    runtime: Runtime, spec: ToolSpec, call: ToolCall, asked: Ask | None
) -> tuple[Decision, ToolCall, ToolOutcome | None]:
    """Governance's decision, as the catalog says now, by name (a graph's tools were built
    before) — asking a person whenever a hook ``asked`` (whose it is, on which screen), and
    not when a reviewer approved the tool for the rest of the run (``remember="run"``) — and
    what came of it: the call (with an approver's edited arguments), announced when it is to
    be, or the outcome of a call the approver rejected."""
    governance = runtime.agent.harness.governance(runtime.tenant)
    decision = await governance.check(spec.name, call.args, side_effects=spec.side_effects)
    if asked is not None:
        decision = decision.asking(
            asked.question, assignee=asked.assignee, component=asked.component, props=asked.props
        )
    remembered = runtime.replay.journal.remembered
    reach = _reach(spec.name, decision.assignee)
    if decision.asks and reach in remembered:
        decision = decision.approved(f"{spec.name} was approved for the rest of the run.")
        runtime.events.custom(DECISION, tool=spec.name, decision="APPROVE", remembered=True)
    if decision.asks:
        resolution = await runtime.approve(call, decision)
        answer = answer_of(resolution)  # raises RunCancelled on CANCEL
        if resolution.remember == "run" and reach not in remembered:
            remembered.append(reach)  # in the journal: later calls of it are not asked
        if resolution.decision is InterruptDecision.REJECT or answer is False:
            reason = reason_of(resolution)
            rejected = ToolOutcome(
                tool=spec.name,
                status=ToolStatus.REJECTED,
                output=f"{spec.name} was not run: the approver rejected it"
                + (f" ({reason})" if reason else ""),
                error_class="ApprovalRejected",
            )
            return decision, call, rejected
        if resolution.decision is InterruptDecision.EDIT and isinstance(answer, dict):
            call = call.model_copy(update={"args": answer})
    elif decision.announces:
        runtime.events.custom(NOTICE, tool=spec.name, args=call.args, side_effects=decision.risk)
    return decision, call, None


def _reach(tool: str, assignee: str | None) -> str:
    """What an approval remembered for the run covers: the tool's later calls asked of the
    same person or role (an ``Ask`` naming someone else still asks)."""
    return tool if assignee is None else f"{tool} for {assignee}"


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
    runtime.replay.record_step(call, outcome)
    await runtime.progress(now=True)
    _events(runtime, ref, call, outcome)
    await runtime.agent.record_tool(runtime, call, outcome)
    return outcome


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
    _started(runtime, ref, call.tool, call.args)
    _ended(
        runtime,
        ref,
        call.tool,
        status=outcome.status.value,
        cached=outcome.cached,
        output=outcome.output,
    )


def _cut(runtime: Runtime, ref: str, tool: str, status: str) -> None:
    """A call cut short — it asked a person (``paused``: it runs again on resume), the run was
    ``cancelled``, or ran out of time (``timeout``) — still ends on the stream; it has no
    output of its own: its result says why."""
    _ended(runtime, ref, tool, status=status, output=CUT[status].format(tool=tool))


def unended(runtime: Runtime, *, timed_out: bool) -> None:
    """The calls still under way as the attempt ends, ended on its stream before it does
    (:func:`_cut`): a framework that runs its tools in tasks of its own (an MCP server's
    handler, Claude's) may stop them only later, and their end is not the stream's then."""
    status = ToolStatus.TIMEOUT.value if timed_out else CANCELLED
    for ref, tool in list(runtime.open_calls.items()):
        _cut(runtime, ref, tool, PAUSED if runtime.pending is not None else status)


def _started(runtime: Runtime, ref: str, tool: str, args: dict[str, Any]) -> None:
    """A call's start on the stream: ``TOOL_CALL_START``, then its ``TOOL_CALL_ARGS``."""
    runtime.open_calls[ref] = tool
    runtime.events.tool(RunEventType.TOOL_CALL_START, ref, tool=tool)
    runtime.events.tool(RunEventType.TOOL_CALL_ARGS, ref, args=args)


def _ended(runtime: Runtime, ref: str, tool: str, **result: Any) -> None:
    """A call's end on the stream, once: ``TOOL_CALL_END``, then its ``TOOL_CALL_RESULT``
    (nothing for one the attempt's end has ended already: :func:`unended`)."""
    if runtime.open_calls.pop(ref, None) is None:
        return
    runtime.events.tool(RunEventType.TOOL_CALL_END, ref, tool=tool)
    runtime.events.tool(RunEventType.TOOL_CALL_RESULT, ref, tool=tool, **result)
