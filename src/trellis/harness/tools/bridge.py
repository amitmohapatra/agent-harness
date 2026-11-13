"""Every harness tool call, whichever framework makes it: replay, governance, execution,
record.

1. **replay** — a call the journal already has (a resumed run re-planning the same step)
   returns its recorded output and runs nothing; a tool of a feature the run is without
   (``without=``: a graph's tools are bound when it is built) is an error the model reads, and
   so are arguments that do not fit the tool's schema (``tools.base.arguments_problem``: a
   required field missing, a basic type wrong, an unknown field where none are allowed) —
   whichever framework made the call, the tool does not run and the model can correct it;
2. **hooks** — the run's ``before_tool`` hooks (``trellis.harness.hooks``) may deny the call,
   rewrite its arguments or ask a person about it; their decision is journaled;
3. **governance** — the run's tenant's :class:`~trellis.harness.governance.Governance` decides,
   by the tool's name, as the catalog says at the time of the call: ``run`` runs, ``announce``
   is announced on the run's stream, ``ask`` pauses the run for approval (an approver may edit
   the arguments, or reject);
4. **execution** — inside a span, with ``TOOL_CALL_*`` events around it (the stream gets the
   arguments and a preview of the output; the span masks credentials); the
   ``after_tool`` hooks may change the outcome the model reads;
5. **record** — journaled for a later resume (and on the run's trajectory, which evaluators
   read; in a worker, saved as the run's progress checkpoint: at once after a call with side
   effects), counted, and (memory on) sent to the memory service's tool records in
   the background.

Execution is bounded: a call takes at most its tool's ``timeout`` and what is left of the
run's time; a call that only reads (or is idempotent) is tried again after an error that may
pass, within that time; the tool reads its idempotency key from ``trellis.current()``. A call
that does more than read is marked started, and saved, before it runs: one that times out, or
was running when its worker died, has an unknown effect — the model is told so, the journal
keeps what it was told, and it is never run again blind (unless its tool is idempotent, or
continues where it was: a sub-agent's run).

Calls may come at once (a ``ReAct`` step's, the frameworks that run tools concurrently): their
steps are numbered as they arrive (or as the caller numbered them, ``step=``), identical calls
take their turn (``Replay.exclusive``), and the journal's progress saves go one at a time.

A model step that makes the same call more than once (the same tool, the same arguments: a
small model repeating itself) runs it once: where the harness reads the step's calls
(:func:`firsts`), each repeat is :func:`same` — nothing runs, nothing is asked, charged,
hooked or journaled — and the model reads, under the repeat's own call id, that the result is
the first call's.
"""

from __future__ import annotations

import asyncio
import logging
import time
import weakref
from collections.abc import Hashable, Iterable
from datetime import UTC, datetime
from typing import Any, Final

from opentelemetry.trace import StatusCode

from trellis.contracts import (
    InterruptDecision,
    Option,
    RunEventType,
    ToolCall,
    ToolError,
    ToolOutcome,
    ToolSpec,
    ToolStatus,
)
from trellis.contracts.errors import is_pause_signal
from trellis.harness import budget, snapshots
from trellis.harness.asking import RunCancelled, answer_of
from trellis.harness.bounded import SAID_ONCE, LRUSet
from trellis.harness.clients.memory import Permissions, grants
from trellis.harness.conversation import allowed, result_id
from trellis.harness.events import DECISION, NOTICE
from trellis.harness.governance.decision import Decision, choices, reach
from trellis.harness.hooks import Ask, Deny, denied, noted, read
from trellis.harness.journal import OUTCOME, Made, content_key
from trellis.harness.runtime import Paused, Runtime, current, reason_of
from trellis.harness.telemetry import metrics, tool_span
from trellis.harness.telemetry import output as span_output
from trellis.harness.tools.base import (
    OUTCOME_UNKNOWN,
    UNKNOWN,
    Tool,
    arguments_problem,
    execute,
    interrupted,
    not_run,
)

log = logging.getLogger("trellis.run")
#: The tenants whose tool permissions could not be read, said once each (of the most recent).
_UNREAD: LRUSet[str] = LRUSet(maxsize=SAID_ONCE)

#: Why a person is asked about a call its framework asks about (:func:`asked`).
FRAMEWORK_ASKS: Final = "{framework} asks before it runs this call."
#: The ``error_class`` of a call whose arguments did not fit the tool's schema.
INVALID_ARGUMENTS: Final = "InvalidArguments"
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
#: What the model reads for a call that repeats an earlier one of its model step (:func:`same`).
SAME_CALL: Final = "Same call as {first} above (the same tool, the same arguments): see its result."
#: What a model that stalled — the same call with the same arguments over and over — is told,
#: for its one answer without tools (``middleware.StallGuard``; every adapter's loop).
STALLED: Final = (
    "You called {tool} with the same arguments {times} times in a row and it did not give you "
    "what you need: do not call it again. Do not call tools: answer now with what you have, and "
    "say what went wrong."
)
#: The same, told in a framework's session that the conversation's next turn continues (Kimi
#: Code over ACP, whose own repeat breaker ended the turn with no answer): as
#: ``budget.TURN_LIMIT``, for this turn only.
TURN_STALLED: Final = (
    "This turn was stopped: the same tool call was made over and over and it did not give you "
    "what you need. Do not call tools now: answer with what you have, and say what went wrong. "
    "This is for this turn only: the user's next message starts a new turn, with your tools."
)
#: The times in a row one call (the same tool, the same arguments) is made before it is not run
#: again (:func:`_repeated`, the model reading :data:`STALLED` instead of its result): a model
#: looping on one call, whatever its framework.
MAX_REPEATS: Final = 3
#: The ``error_class`` of that call, and the warning on the run's stream (once).
REPEATED: Final = "Repeated"
#: The error class of a call not run because its start could not be saved (:func:`_unsaved`).
UNSAVED: Final = "StartUnsaved"
#: The outcomes the run's ``after_tool`` hooks changed (:func:`hooked`), by id, while they live.
_HOOKED: weakref.WeakValueDictionary[int, ToolOutcome] = weakref.WeakValueDictionary()


async def call(
    tool: Tool,
    args: dict[str, Any],
    *,
    call_id: str | None = None,
    step: int | None = None,
    decision: Hashable | None = None,
) -> ToolOutcome:
    """Run ``tool`` for the current run (``step``: the call's number, when the caller numbered
    the calls it makes at once; ``decision``: the model decision the call is part of, when the
    caller knows it — :func:`_deciding`). Pauses propagate; every other failure is an outcome
    the framework shows its model (``ERROR``, or ``TIMEOUT``)."""
    runtime = current()
    if runtime is None:
        raise ToolError(
            f"{tool.name} is a trellis tool: it runs inside a Harness run (agent.run/stream)",
            source="tools",
        )
    made = runtime.replay.made(tool.name, args)  # the paused call made again runs as it was
    key = content_key("call", tool.name, made.args)
    step = runtime.next_step() if step is None else step
    async with runtime.replay.exclusive(key):
        outcome = await _called(
            runtime, tool, made, key=key, step=step, call_id=call_id, decision=decision
        )
    runtime.replay.ended(made)
    return outcome


def hooked(outcome: ToolOutcome) -> bool:
    """Whether the run's ``after_tool`` hooks changed this outcome of :func:`call` (what the
    call says, or how it ended). The model then reads the hooks' outcome, whatever its framework
    makes of the call otherwise: a failure a hook turned into a result is that result, and a
    failure it worded otherwise says the hook's words — still delivered as its framework
    delivers a failed call (``middleware.governed_call``, ``tools.convert``)."""
    return _HOOKED.get(id(outcome)) is outcome


def firsts(calls: Iterable[tuple[str, str, Any]]) -> dict[str, str]:
    """The calls of one model step — ``(call id, tool, arguments)``, in the model's order —
    that repeat an earlier one of it (the same tool, the same arguments in any key order), by
    id: the id of the first, the one that runs."""
    first: dict[str, str] = {}
    repeats: dict[str, str] = {}
    for call_id, tool, args in calls:
        earlier = first.setdefault(content_key("call", tool, args), call_id)
        if earlier != call_id:
            repeats[call_id] = earlier
    return repeats


def same(
    runtime: Runtime, tool: str, args: dict[str, Any], *, call_id: str, first: str
) -> ToolOutcome:
    """A call that repeats ``first``, an earlier call of its model step (:func:`firsts`): not
    run — no hooks, governance, budget, journal or record: ``first`` is the call that runs —
    and what the model reads says whose result it is. On the run's stream as every call is,
    its result naming the call it repeats (``same_as``)."""
    outcome = ToolOutcome(tool=tool, output=SAME_CALL.format(first=first))
    _started(runtime, call_id, tool, args)
    _ended(
        runtime, call_id, tool, status=outcome.status.value, output=outcome.output, same_as=first
    )
    return outcome


async def permitted(
    spec: ToolSpec, args: dict[str, Any], *, framework: str | None = None
) -> ToolCall | ToolOutcome:
    """A call its framework runs itself (Claude Code's built-in tools, through its permission
    callback), decided as a harness call is — the run's ``before_tool`` hooks, then governance,
    and a person when it asks — but not run, journaled or recorded here: the call to let
    through (its arguments rewritten or edited, perhaps), or the outcome of one that is not
    (denied, rejected). A pause propagates. ``framework``: the framework asks its own user
    about this call too (:func:`asked`, its name in the question) — asked here once the call
    is let through as it was made, unless a person was just asked about it (governance's
    question, a hook's): one call, one question."""
    runtime = current()
    assert runtime is not None  # a framework asks inside the run it works for
    key = content_key("call", spec.name, args)
    call = ToolCall(tool=spec.name, args=args, task=runtime.task, idempotency_key=key)
    call, verdict = await _hooked(runtime, call, key)
    if isinstance(verdict, Deny):
        return denied(call, verdict)
    decision, call, rejected = await _decided(runtime, spec, call, verdict)
    if rejected is not None:
        return rejected
    if framework is None or decision.asks or call.args != args:
        return call
    return await asked(spec, args, framework)


async def asked(spec: ToolSpec, args: dict[str, Any], framework: str) -> ToolCall | ToolOutcome:
    """A call ``framework`` asks its user about before it runs it (its own permission check:
    Claude Code's ``can_use_tool``, Kimi Code's ``session/request_permission``), once
    :func:`permitted` let it through: a person is asked, as governance asks — answered from the
    journal when the run resumes, not asked again when the tool was approved for the rest of the
    run — and the call to let through (perhaps edited), or the outcome of one rejected. A pause
    propagates."""
    runtime = current()
    assert runtime is not None  # a framework asks inside the run it works for
    key = content_key("call", spec.name, args)
    call = ToolCall(tool=spec.name, args=args, task=runtime.task, idempotency_key=key)
    why = FRAMEWORK_ASKS.format(framework=framework)
    _, call, rejected = await _decided(runtime, spec, call, Ask(why))
    return call if rejected is None else rejected


async def asks_first(spec: ToolSpec, args: dict[str, Any]) -> bool:
    """For a framework that holds a call at its own approval gate before it runs it (OpenAI
    Agents' ``needs_approval``): whether governance asks a person about this call, as it would
    when the call runs — and if so, asked now (the run's pause, governance's question), so the
    framework keeps its run where it paused and its resume runs the call, instead of a new
    attempt that asks the model again. Not for a call governance lets run, one allowed for the
    conversation (or its schedule), or one already approved or answered: the call decides those
    as ever. Only a tool the harness runs is held (never a confined one: ``_gated``)."""
    runtime = current()
    assert runtime is not None  # a framework asks inside the run it works for
    governance = runtime.agent.harness._governance(runtime.tenant)
    decision = await governance.check(
        spec.name, args, side_effects=spec.side_effects, agent_id=runtime.agent_id
    )
    if decision.error is not None:  # a rule that could not be evaluated: recorded on the run
        runtime.events.warning(decision.error.code, decision.error.message)
    journal = runtime.replay.journal
    if (
        not decision.asks
        or await _allowed(runtime, decision, ours=True) is not None
        or content_key("approved", spec.name, args) in journal.approved
        or runtime.replay.answered(content_key("approve", spec.name, args))
    ):
        return False
    key = content_key("call", spec.name, args)
    call = ToolCall(tool=spec.name, args=args, task=runtime.task, idempotency_key=key)
    try:
        await runtime.approve(call, decision, await _offered(runtime, decision))
    except Paused:
        return True
    raise AssertionError("an approval not answered yet pauses")  # pragma: no cover


async def observed(
    runtime: Runtime,
    spec: ToolSpec,
    args: dict[str, Any],
    outcome: ToolOutcome,
    *,
    ref: str,
    started: float,
) -> None:
    """A call its framework ran itself and reported once it ran — a provider's hosted tool (an
    OpenAI agent's file search, web search, code interpreter: run inside the model's step),
    Claude Code's built-ins (after its ``PreToolUse`` decision, ``permitted``): on the run's
    stream, counted, traced with its duration (``started``: ``time.time()`` when it began) and
    recorded in the memory service, as a harness call is. Nothing is run or journaled here."""
    call = ToolCall(
        tool=spec.name, args=args, task=runtime.task, step=runtime.next_step(), idempotency_key=ref
    )
    outcome.latency_ms = round(max(0.0, time.time() - started) * 1000, 3)
    start_time = int(started * 1e9)
    with tool_span(
        spec.name, ref, args, source=spec.source, action="run", start_time=start_time
    ) as span:
        span_output(span, outcome.output, key="gen_ai.tool.call.result")
        if not outcome.ok:
            span.set_status(StatusCode.ERROR, str(outcome.status.value))
    runtime.used.add(spec.name)
    _events(runtime, ref, call, outcome)
    metrics.tool_called(spec.name, outcome.status.value)
    await runtime.agent.record_tool(runtime, call, outcome)


async def _called(
    runtime: Runtime,
    tool: Tool,
    made: Made,
    *,
    key: str,
    step: int,
    call_id: str | None,
    decision: Hashable | None = None,
) -> ToolOutcome:
    args = made.args
    deciding = _deciding(runtime, call_id, decision)
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
    refused = _refused(runtime, tool, args, key, deciding)
    if refused is not None:
        _events(runtime, ref, tool_call, refused)
        return refused
    if runtime.replay.interrupted(key) and not (tool.spec.idempotent or tool.resumable):
        # its worker died while it ran: it is not run again blind (an idempotent tool is,
        # with the same key, below, and so is one that continues where it was)
        return await _interrupted(runtime, ref, tool_call, key)

    tool_call, verdict = await _hooked(runtime, tool_call, key)
    if isinstance(verdict, Deny):
        refused = denied(tool_call, verdict)
        _events(runtime, ref, tool_call, refused)
        return refused
    decision, tool_call, rejected = await _decided(
        runtime, tool.spec, tool_call, verdict, tool=tool
    )
    # the budget again, then counted at once: the calls of one step are decided together
    rejected = rejected or _over_budget(runtime, tool)
    if rejected is not None:
        _events(runtime, ref, tool_call, rejected)
        return rejected
    budget.count_tool_call(runtime)
    _started(runtime, ref, tool.name, tool_call.args)
    try:
        outcome = await _ran(
            runtime, tool, tool_call, decision, key=key, ref=ref, idem=idempotency_key, made=made
        )
    except asyncio.CancelledError:
        # the run was cancelled, or ran out of time, while the call was under way
        out = runtime.cancelled is None and runtime.remaining() == 0
        _cut(runtime, ref, tool.name, ToolStatus.TIMEOUT.value if out else CANCELLED)
        raise
    except Exception as cut:
        # it asked a person (the harness's pause, or LangGraph's), or the run was cancelled
        cancelled = isinstance(cut, RunCancelled)
        if cancelled or is_pause_signal(cut):
            _cut(runtime, ref, tool.name, CANCELLED if cancelled else PAUSED)
        if is_pause_signal(cut):  # not over: made again on the resume, and counted then
            budget.count_tool_call(runtime, times=-1)
        raise
    runtime.fetched[ref] = snapshots.now()  # what a later turn's replay says it is from
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
    made: Made,
) -> ToolOutcome:
    """The call, started on the stream, under way: saved as started (one that does more than
    read), executed (``idem``: the key the tool hands its service; ``made``: the call as the
    model made it, which a pause from within it keeps), and its outcome as the ``after_tool``
    hooks leave it. A pause (it asked a person) or a cancellation propagates."""
    args = tool_call.args
    if tool.feature == "code_mode":
        runtime.code_mode_scripts.append((datetime.now(UTC), tool_call.step))
    runtime.used.add(tool.name)
    reads = decision.risk == "read"
    if not reads:
        # saved before it runs: the attempt after a crash knows it was running
        runtime.replay.start(key)
        if not await runtime.saved_start():
            runtime.replay.unstart(key)
            return _unsaved(tool)
    started = time.perf_counter()
    action = decision.action.value
    with tool_span(tool.name, ref, args, source=tool.spec.source, action=action) as span:
        within = runtime.limited(tool.timeout, key=idem, ref=ref, made=made)
        outcome, error = await execute(tool, args, reads=reads, within=within)
        if isinstance(error, RunCancelled) or (error is not None and is_pause_signal(error)):
            if not reads:
                runtime.replay.unstart(key)  # it asked a person: it runs again on resume
            raise error
        span_output(span, outcome.output, key="gen_ai.tool.call.result")
    hooks = runtime.agent.hooks
    if error is not None:
        await hooks.failed("tool", error)
    before = outcome.model_dump() if hooks else None
    outcome = await hooks.done(tool_call, outcome)
    if before is not None and outcome.model_dump() != before:
        _HOOKED[id(outcome)] = outcome  # the hooks' word on the call (:func:`hooked`)
    outcome.latency_ms = round((time.perf_counter() - started) * 1000, 3)
    return outcome


def _unsaved(tool: Tool) -> ToolOutcome:
    """A call that does more than read, not run: its start could not be saved (agent-runs could
    not be reached), so after a crash another worker would not know it ran and could run it
    again (``Runtime.saved_start``)."""
    return ToolOutcome(
        tool=tool.name,
        status=ToolStatus.ERROR,
        output=(
            f"{tool.name} was not run: its start could not be saved (the run store cannot be "
            "reached), and run unsaved it could be done twice after a crash. Try it again "
            "later."
        ),
        error_class=UNSAVED,
    )


def _over_budget(runtime: Runtime, tool: Tool) -> ToolOutcome | None:
    """A call refused because the run's budget is spent (``budget.py``), as the model reads
    it."""
    spent = budget.reached(runtime)
    if spent is None:
        return None
    budget.stopped(runtime, spent)
    return ToolOutcome(
        tool=tool.name,
        status=ToolStatus.ERROR,
        output=f"{tool.name} was not run: {budget.SPENT} ({spent})",
        error_class=budget.BUDGET_EXCEEDED,
    )


def _deciding(runtime: Runtime, call_id: str | None, given: Hashable | None) -> Hashable | None:
    """The model decision a call is part of — the same each time its framework makes that call
    again, and for each call that one decision makes —: ``given`` by the caller (an OpenAI
    computer call's id, for each of its actions), else the framework's own (``Runtime.deciding``:
    a LangGraph task, the same for each run its ``RetryPolicy`` makes) with the call's id;
    ``None`` when neither says."""
    if given is not None:
        return given
    deciding = runtime.deciding() if runtime.deciding is not None else None
    return None if deciding is None else (deciding, call_id)


def _repeated(
    runtime: Runtime, tool: Tool, key: str, decision: Hashable | None
) -> ToolOutcome | None:
    """The call, refused when it is the same call (the same tool, the same arguments) the run
    made the :data:`MAX_REPEATS` - 1 calls before it: a model looping on one call is told to
    answer with what it has (:data:`STALLED`), whatever its framework — the harness's tools
    are where every adapter's calls meet, so a graph without the harness's middleware, a Deep
    Agent, OpenAI Agents and the Claude CLI are stopped as ``ReAct``'s ``StallGuard`` stops its
    own loop (which acts before its step's calls reach here). Said once on the run's stream.

    A repeat is a decision made again: a call of the decision that made the call before it
    (:func:`_deciding`) — its framework running that decision again (a LangGraph task its
    ``RetryPolicy`` runs again, a tool node's call a wrapper tries again, the same call id in
    the same task), or another action of the same OpenAI computer call (a batch of three
    ``Down`` presses, and the screenshot after them) — is part of it: neither counted nor
    starting a count of its own, and refused when that decision was."""
    last, times, made_by = runtime.repeating
    again = decision is not None and decision == made_by
    if again:
        key = last  # the decision is counted by the call it began with
    times = times if again else times + 1 if key == last else 1
    runtime.repeating = (key, times, decision)
    if times < MAX_REPEATS:
        return None
    if times == MAX_REPEATS and not again:
        said = f"stopped {tool.name}: the same call {times} times in a row (a stall)"
        log.warning("run %s %s", runtime.run_id, said)
        runtime.events.warning("stalled", said)
    return ToolOutcome(
        tool=tool.name,
        status=ToolStatus.ERROR,
        output=STALLED.format(tool=tool.name, times=times),
        error_class=REPEATED,
    )


def stalled(runtime: Runtime) -> bool:
    """Whether the run's latest harness tool call was refused as a loop (:func:`_repeated`): a
    framework that offers its tools per turn offers none until another call is made, so the
    model's reply to :data:`STALLED` is its answer (OpenAI Agents' ``FunctionTool.is_enabled``,
    as ``StallGuard``'s last call is offered no tool)."""
    return runtime.repeating[1] >= MAX_REPEATS


def _refused(
    runtime: Runtime, tool: Tool, args: dict[str, Any], key: str, decision: Hashable | None
) -> ToolOutcome | None:
    """Why a call is not run at all, as the model reads it: the run's budget is spent, the
    same call made :data:`MAX_REPEATS` times in a row, a tool the framework was built with of a
    feature this run is without, or arguments that do not fit its schema (nothing runs,
    nothing is journaled: the model corrects them)."""
    over = _over_budget(runtime, tool) or _repeated(runtime, tool, key, decision)
    if over is not None:
        return over
    if tool.feature is not None and not runtime.uses(tool.feature):
        return ToolOutcome(
            tool=tool.name,
            status=ToolStatus.ERROR,
            output=f"{tool.name} is off in this run (without {tool.feature})",
            error_class="FeatureOff",
        )
    problem = arguments_problem(tool.spec.input_schema or {}, args)
    if problem is None:
        return None
    return ToolOutcome(
        tool=tool.name,
        status=ToolStatus.ERROR,
        output=not_run(tool.name, problem),
        error_class=INVALID_ARGUMENTS,
    )


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
        runtime.replay.record_call(key, _journaled(outcome), tool=call.tool, stepped=True)
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
        runtime.replay.record_call(journaled, kept, stepped=True)
    return call.model_copy(update={"args": kept["args"]}), read(kept["verdict"])


async def _decided(
    runtime: Runtime,
    spec: ToolSpec,
    call: ToolCall,
    asked: Ask | None,
    *,
    tool: Tool | None = None,
) -> tuple[Decision, ToolCall, ToolOutcome | None]:
    """Governance's decision, as the catalog says now, by name (a graph's tools were built
    before) — asking a person whenever a hook ``asked`` (whose it is, on which screen), and not
    when the tool was allowed beyond one call (:func:`_allowed`) — and what came of it: the
    call (with an approver's edited arguments), announced when it is to be, or the outcome of a
    call the approver rejected. ``tool``: a call the harness runs — a write asks (unless the
    tool is ``confined``), and the person picks one of ``governance.decision.choices`` (Always
    allow kept in the memory service: :func:`_kept_always`); none: a call its framework runs
    under its own permission flow — a write is announced, and an approval reaches the rest of
    the run only (``remember="run"``)."""
    governance = runtime.agent.harness._governance(runtime.tenant)
    decision = await governance.check(
        spec.name, call.args, side_effects=spec.side_effects, agent_id=runtime.agent_id
    )
    if tool is None or tool.confined:
        decision = decision.unasked()
    if decision.error is not None:  # a rule that could not be evaluated: recorded on the run
        runtime.events.warning(decision.error.code, decision.error.message)
    if asked is not None:
        decision = decision.asking(
            asked.question, assignee=asked.assignee, component=asked.component, props=asked.props
        )
    ours = tool is not None
    # a call answered already in this run keeps its answer: a denial stays one
    answered = runtime.replay.answered(content_key("approve", spec.name, call.args))
    if (
        decision.asks
        and not answered
        and (why := await _allowed(runtime, decision, ours=ours, hooked=asked is not None))
        is not None
    ):
        decision = decision.approved(why)
        runtime.events.custom(DECISION, tool=spec.name, decision="APPROVE", remembered=True)
    gated = content_key("approved", spec.name, call.args)
    approved = runtime.replay.journal.approved
    if decision.asks and decision.assignee is None and gated in approved:
        approved.remove(gated)  # one approval, one call
        decision = decision.approved(f"{spec.name} was approved in the framework's own gate.")
        runtime.events.custom(DECISION, tool=spec.name, decision="APPROVE", framework=True)
    if decision.asks:
        hooked = asked is not None
        offered = await _offered(runtime, decision, hooked=hooked) if ours else ()
        resolution = await runtime.approve(call, decision, offered)
        answer = answer_of(resolution)  # raises RunCancelled on CANCEL
        remembered = runtime.replay.journal.remembered
        covered = reach(spec.name, decision.assignee)
        always = (
            resolution.remember == "always"
            and ours
            and _keeps(runtime, decision, hooked=hooked)
            and (await _always(runtime)).grants is not False  # unread: kept, or spooled
        )
        beyond = always or (resolution.remember == "run" and decision.risk != "irreversible")
        if beyond and covered not in remembered:
            remembered.append(covered)  # in the journal: later calls of it are not asked
            if always:  # once: a replayed answer finds it remembered
                await _kept_always(runtime, decision, resolution.reviewer)
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


async def _allowed(
    runtime: Runtime, decision: Decision, *, ours: bool, hooked: bool = False
) -> str | None:
    """Why a call that asks runs without asking, or ``None``: its tool was allowed beyond one
    call — in this run (the journal), and, for a call the harness runs (``ours``), by the
    schedule that fired the run (``schedules.ALLOWED``) or in an earlier run of the conversation
    (``conversation.allowed``), or always, by its user (:func:`_always`) unless its
    ``approve_when`` decides the call. Never an ``irreversible`` tool: it asks every call; and
    never a call a hook's ``Ask`` asks about (``hooked``): your rule in code is asked whatever
    was allowed, as a PreToolUse "ask" is in Claude Code."""
    if decision.risk == "irreversible" or hooked:
        return None
    covered = reach(decision.tool, decision.assignee)
    if covered in runtime.replay.journal.remembered:
        reached = "this conversation" if ours else "the rest of the run"
        return f"{decision.tool} was approved for {reached}."
    return await _allowed_beyond(runtime, decision, covered) if ours else None


async def _allowed_beyond(runtime: Runtime, decision: Decision, covered: str) -> str | None:
    """Why a call the harness runs is allowed beyond this run: by its schedule, for its
    conversation, or always."""
    root = runtime
    while root.parent is not None:  # a child run works for its parent's schedule
        root = root.parent
    if root.scheduled is not None and covered in root.scheduled.allowed:
        return f"{decision.tool} was allowed for this schedule."
    if covered in await allowed(runtime):
        return f"{decision.tool} was approved for this conversation."
    if decision.rule is None and covered in (await _always(runtime)).tools:
        return f"{decision.tool} is always allowed for you."
    return None


def _keeps(runtime: Runtime, decision: Decision, *, hooked: bool) -> bool:
    """Whether Always allow applies to the call: a memory service keeps it, the tool is not
    ``irreversible`` and no rule decides the call (its ``approve_when``, or a hook's ``Ask``:
    ``hooked``)."""
    return (
        runtime.run_memory is not None
        and decision.risk != "irreversible"
        and decision.rule is None
        and not hooked
    )


async def _offered(
    runtime: Runtime, decision: Decision, *, hooked: bool = False
) -> tuple[Option, ...]:
    """What a person asked about a call the harness runs picks from: Always allow too where it
    applies (:func:`_keeps`) and a grant would be kept — this attempt read the user's
    permissions, and the key may grant — so a pick is never silently lost."""
    keeps = _keeps(runtime, decision, hooked=hooked) and (await _always(runtime)).grants is True
    return choices(decision.risk, always=keeps)


async def _kept_always(runtime: Runtime, decision: Decision, reviewer: str | None) -> None:
    """The person's Always allow for the call's tool (asked of the same person or role), kept
    in the memory service's tool permissions for the run's user — in the background, spooled
    when it cannot be delivered, and stored once however often it is sent."""
    memory = runtime.run_memory
    assert memory is not None  # offered only with one
    args = {
        "tool": decision.tool,
        "assignee": decision.assignee,
        "allowed_by": reviewer or runtime.user,
    }
    await runtime.agent.harness.writes.submit(
        "memory.tool_permission",
        lambda: memory.allow_always(**args),
        events=runtime.events,
        record=memory.record("allow_always", **args),
    )


async def _always(runtime: Runtime) -> Permissions:
    """What the run's user allowed always (``reach`` of each kept permission) and whether a
    grant is kept: read once per attempt, and only when a call asks, so a revoke counts from
    the next run; none without a memory service, and none while it cannot be read — the call
    asks (failing closed), Always allow is not offered, and a warning is logged once."""
    if runtime.always is None:
        runtime.always = await _always_read(runtime)
    return runtime.always


async def _always_read(runtime: Runtime) -> Permissions:
    memory = runtime.run_memory
    if memory is None:
        return Permissions(grants=False)
    try:
        kept = await memory.always_allowed()
        key = await runtime.agent.harness._key_info()
    except Exception as exc:
        if runtime.tenant not in _UNREAD:
            _UNREAD.add(runtime.tenant)
            log.warning(
                "the tool permissions could not be read (%s): until they can, a tool allowed "
                "always asks again",
                type(exc).__name__,
                extra={"tenant": runtime.tenant, "run_id": runtime.run_id},
            )
        return Permissions()
    _UNREAD.discard(runtime.tenant)  # a later outage is said again
    return Permissions(frozenset(reach(p.tool, p.assignee) for p in kept), grants(key))


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
    runtime.replay.record_call(key, _journaled(outcome), tool=call.tool, stepped=True)
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
    said = result_id(runtime.conversation, ref)  # the id the result has in the transcript
    runtime.events.tool(RunEventType.TOOL_CALL_RESULT, ref, said, tool=tool, **result)
