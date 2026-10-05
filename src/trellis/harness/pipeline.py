"""One attempt of one run: the fixed pipeline every framework goes through.

identity → (the run record, written by the caller) → tools → memory push (with the tool
hints that narrow what the model is offered) → the adapter → the outcome recorded (paused
with its journal as the run's checkpoint, finished with its answer or error) → background
writes (transcript, the run's ``system`` outcome, the sampled grounding check, the sampled
online judges). The adapter is
the only part that knows the framework. Each attempt is one ``invoke_agent`` span in the run's
trace.

The attempt works at most what is left of the run's time (:class:`Budget`: its ``timeout``
less the time it already worked, and its ``deadline``) and then ends ``TIMEOUT``; cancelled
(``agent.cancel``), it ends ``CANCELLED`` with the reason.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from trellis.contracts import (
    AgentError,
    ConfigurationError,
    ErrorCategory,
    Interrupt,
    InterruptReason,
    InterruptResolution,
    RunEvent,
    RunEventType,
    RunOutcome,
    RunRecord,
    RunStatus,
    ToolCall,
)
from trellis.harness.adapters import convert
from trellis.harness.adapters.base import Extracted, Invocation, NativePause, Output, query_of
from trellis.harness.adapters.langgraph import FOREIGN, HITL, holds, is_hitl
from trellis.harness.events import RunEvents
from trellis.harness.identity import Identity
from trellis.harness.journal import Journal, Pending, Replay
from trellis.harness.result import Result
from trellis.harness.runtime import RunCancelled, Runtime, _current, interrupt_id
from trellis.harness.telemetry import RunTrace, agent_span, metrics, output
from trellis.memory.models import PromptContext
from trellis.runs import RELEASED, ConflictError, LeaseLostError

if TYPE_CHECKING:
    from trellis.harness.agent import Agent
    from trellis.harness.tools.base import Tool

log = logging.getLogger("trellis.run")


@dataclass(frozen=True, slots=True)
class Budget:
    """What is left of a run's time as an attempt begins: ``seconds`` (none, or less, past
    it), and the error the run ends ``TIMEOUT`` with when they run out."""

    seconds: float
    error: AgentError

    @classmethod
    def of(
        cls,
        *,
        timeout: float | None,
        worked: float = 0.0,
        deadline: datetime | None,
        remaining: float | None = None,
    ) -> Budget | None:
        """The tighter of the working-time limit (``timeout`` less the seconds ``worked``
        in earlier attempts; or what a worker's lease says is ``remaining``: the run's limit
        or the service's, the lesser) and the ``deadline``; ``None`` when there is neither."""
        found: list[Budget] = []
        if remaining is not None:
            limit = "its time limit" if timeout is None else f"its time limit of {timeout:g}s"
            found.append(cls(remaining, _timed_out("run_timeout", f"the run worked past {limit}")))
        elif timeout is not None:
            message = f"the run worked past its time limit of {timeout:g}s"
            found.append(cls(timeout - worked, _timed_out("run_timeout", message)))
        if deadline is not None:
            left = (deadline - datetime.now(UTC)).total_seconds()
            message = f"the run did not end by its deadline, {deadline.isoformat()}"
            found.append(cls(left, _timed_out("run_deadline", message)))
        return min(found, key=lambda b: b.seconds, default=None)


def _timed_out(code: str, message: str) -> AgentError:
    # the category alone would say retryable: a run out of time is not run again
    return AgentError(code=code, category=ErrorCategory.TIMEOUT, message=message, retryable=False)


async def attempt(
    agent: Agent,
    identity: Identity,
    input: Any,
    *,
    number: int = 1,
    journal: Journal | None = None,
    resolution: InterruptResolution | None = None,
    listener: Callable[[RunEvent], None] | None = None,
    streaming: bool = False,
    worker_id: str | None = None,
    lease_seconds: int | None = None,
    observe: Callable[[PromptContext | None], None] | None = None,
    budget: Budget | None = None,
    started_on: str | None = None,
) -> Result:
    """Run one attempt and record how it ended. The run record must already be RUNNING;
    ``worker_id`` names the worker holding its lease (the store fences its writes), and
    ``lease_seconds`` its length (progress checkpoints extend it). ``observe`` is told the memory
    context the run was given (an offline evaluation's evaluators read it). ``budget`` is what
    is left of the run's time; ``started_on`` the agent version that started the run (a
    resume on another version says so)."""
    journal = journal or Journal()
    unresumable = await _unheld(agent, identity, journal, resolution)
    pending = journal.pending
    replay = _replay(journal, resolution)
    events = RunEvents(identity.context(), number)
    if listener is not None:
        events.listen(listener)
    query = query_of(input)
    runtime = Runtime(
        identity=identity,
        agent=agent,
        events=events,
        replay=replay,
        attempt=number,
        worker_id=worker_id,
        lease_seconds=lease_seconds,
        run_memory=await agent.run_memory(identity),
        writes_memory=await agent.harness.writes_memory(),
        used=set(journal.used),
        task=query,
        started_at=datetime.now(UTC),
        ends_at=None if budget is None else time.monotonic() + budget.seconds,
        running_in=asyncio.current_task(),
    )
    events.emit(RunEventType.RUN_STARTED, data={"agent_id": identity.agent_id})
    _versioned(agent, runtime, started_on)
    extracted: Extracted | None = None
    pushed: PromptContext | None = None
    error: Exception | None = None
    clock = asyncio.timeout(None if budget is None else budget.seconds)
    token = _current.set(runtime)
    agent.running[identity.run_id] = runtime
    try:
        run_trace = RunTrace(
            run_id=identity.run_id,
            agent_id=identity.agent_id,
            tenant=identity.tenant,
            user=identity.user,
            thread=identity.thread,
            framework=agent.adapter.name,
            attempt=number,
            version=agent.version,
        )
        with agent_span(run_trace, query) as span:
            async with clock:
                tools = await agent.tools_for(runtime)
                runtime.toolbox = {t.name: t for t in tools}
                pushed = await agent.push(runtime)
                if observe is not None:
                    observe(pushed)
                if unresumable is not None:
                    raise unresumable
                extracted = await _invoked(
                    agent,
                    runtime,
                    tools,
                    input,
                    pending=pending,
                    resolution=resolution,
                    streaming=streaming,
                )
            output(span, jsonable(extracted.answer))
    except RunCancelled as exc:
        runtime.cancelled = str(exc)
    except asyncio.CancelledError as exc:
        task = asyncio.current_task()
        assert task is not None
        # ``agent.cancel`` ends the run CANCELLED with its reason, and its task goes on;
        # otherwise the caller went away (a closed stream, a lost lease): the run ends here —
        # unless its worker is stopping and released it for another worker to run again
        if runtime.cancelled is None or task.uncancel():
            if RELEASED not in exc.args:
                await _settle_cancelled(agent, identity, events, worker_id)
            raise
    except Exception as exc:
        # a framework may wrap or swallow the pause: the runtime is what says it paused
        if runtime.pending is None:
            error = exc
    finally:
        _current.reset(token)
        agent.running.pop(identity.run_id, None)
    timed_out = budget.error if budget is not None and clock.expired() else None
    return await _concluded(
        agent, runtime, journal, extracted, pushed=pushed, error=error, timed_out=timed_out
    )


async def _concluded(
    agent: Agent,
    runtime: Runtime,
    journal: Journal,
    extracted: Extracted | None,
    *,
    pushed: PromptContext | None,
    error: Exception | None,
    timed_out: AgentError | None,
) -> Result:
    """Record how the attempt ended: cancelled, out of time, paused, failed or answered."""
    if runtime.lease_lost:
        # another worker may hold the run now (a framework may have swallowed the error)
        raise LeaseLostError(
            f"{runtime.worker_id} lost the lease on {runtime.run_id}: nothing written",
            code="LEASE_LOST",
            status=409,
        )
    if runtime.cancelled is not None:
        await _settle_cancelled(
            agent, runtime.identity, runtime.events, runtime.worker_id, reason=runtime.cancelled
        )
        return Result(run_id=runtime.run_id, status=RunStatus.CANCELLED)
    if timed_out is not None:
        log.warning("run %s timed out: %s", runtime.run_id, timed_out.message)
        return await _failed(agent, runtime, timed_out, extracted, status=RunStatus.TIMEOUT)
    paused = _pause(runtime, extracted, error)
    if paused is not None:
        return await _paused(agent, runtime, journal, paused, extracted)
    if error is not None:
        failure = AgentError.of(error, source=agent.adapter.name)
        log.warning("run %s failed: %s", runtime.run_id, failure.message, exc_info=error)
        return await _failed(agent, runtime, failure, extracted)
    assert extracted is not None
    return await _succeeded(agent, runtime, extracted, pushed)


def _versioned(agent: Agent, runtime: Runtime, started_on: str | None) -> None:
    """A run continued by another version of the agent than the one that started it goes
    on, with a warning naming both."""
    if started_on is None or agent.version is None or started_on == agent.version:
        return
    message = (
        f"run {runtime.run_id} was started by {agent.id} {started_on} and continues on "
        f"{agent.version}"
    )
    log.warning("%s", message)
    runtime.events.warning("agent_version", message)


async def _unheld(
    agent: Agent, identity: Identity, journal: Journal, resolution: InterruptResolution | None
) -> ConfigurationError | None:
    """A checkpointed graph's pause resumes in place only where its checkpointer still holds it
    (an ``InMemorySaver`` is the pausing process's). Where it does not, a pause of the harness's
    own (an approval, an ``ask``) is answered from the journal instead — the graph runs again
    from its input, as without a checkpointer — and a graph's own pause (its ``interrupt()``,
    the HITL middleware's), which only the checkpointer can answer, is the error the attempt
    fails with."""
    pending = journal.pending
    if (
        resolution is None
        or pending is None
        or pending.native_id is None
        or pending.native_state is not None  # a serialised run (OpenAI Agents) travels along
        or await holds(agent.target, identity.thread or identity.run_id, pending.native_id)
    ):
        return None
    if pending.key in (FOREIGN, HITL):
        return ConfigurationError(
            f"the graph's checkpointer no longer holds the pause {pending.native_id} of thread "
            f"{identity.thread}: resume it where it paused, or give every process a shared "
            "checkpointer"
        )
    log.info("run %s: its checkpointer lost the pause; answered from the journal", identity.run_id)
    journal.pending = pending.model_copy(update={"native_id": None})
    return None


def _replay(journal: Journal, resolution: InterruptResolution | None) -> Replay:
    """The cursor this attempt reads the journal with, the resolution filed where the
    re-run will ask for it (or left to the framework, when it resumes from its own state)."""
    pending = journal.pending
    if resolution is not None and pending is not None:
        if pending.native_id or pending.native_state:
            journal.pending = None
        else:
            journal.answered(resolution)
    return Replay(journal)


async def _invoked(
    agent: Agent,
    runtime: Runtime,
    tools: list[Tool],
    input: Any,
    *,
    pending: Pending | None,
    resolution: InterruptResolution | None,
    streaming: bool,
) -> Extracted:
    """The framework's part: its input (resumed where it paused), the run's tools in its
    format, and what it produced."""
    adapter, target = agent.adapter, agent.target
    native_input = adapter.prepare_input(target, input, runtime.context)
    if pending is not None and resolution is not None:
        native_input = adapter.resume_input(target, native_input, pending, resolution)
    run_tools = [] if adapter.fixed_tools else tools
    if adapter.narrows == "run":
        run_tools = [t for t in run_tools if runtime.offers(t.name)]
    invocation = Invocation(runtime, run_tools, convert(adapter.tool_format, run_tools))
    produced = await _execute(agent, native_input, invocation, streaming=streaming)
    return adapter.extract(target, produced)


async def _execute(
    agent: Agent, native_input: Any, invocation: Invocation, *, streaming: bool
) -> Any:
    adapter, target = agent.adapter, agent.target
    if not streaming:
        return await adapter.invoke(target, native_input, invocation)
    events = invocation.runtime.events
    message_id = f"{invocation.runtime.run_id}.{invocation.runtime.attempt}.m"
    opened = False
    output: Any = None
    async for item in adapter.stream(target, native_input, invocation):
        if isinstance(item, Output):
            output = item.value
            continue
        if not opened:
            events.text(RunEventType.TEXT_MESSAGE_START, message_id)
            opened = True
        events.text(RunEventType.TEXT_MESSAGE_CONTENT, message_id, delta=item)
    if opened:
        events.text(RunEventType.TEXT_MESSAGE_END, message_id)
    return output


def _pause(
    runtime: Runtime, extracted: Extracted | None, error: BaseException | None
) -> Pending | None:
    native = extracted.pause if extracted is not None else None
    if runtime.pending is not None:
        if native is not None:
            return runtime.pending.model_copy(
                update={"native_id": native.native_id, "native_state": native.state}
            )
        return runtime.pending
    if native is None:
        return None
    return _foreign(runtime, native)


def _foreign(runtime: Runtime, native: NativePause) -> Pending:
    """A pause the framework raised on its own: an SDK approval, or a graph's ``interrupt``."""
    runtime._asked += 1
    ident = interrupt_id(runtime.run_id, runtime.attempt, runtime._asked)
    if native.tool is not None:
        interrupt = Interrupt(
            interrupt_id=ident,
            tenant_id=runtime.tenant,
            run_id=runtime.run_id,
            reason=InterruptReason.APPROVAL,
            question=f"Approve {native.tool}?",
            tool_call=ToolCall(
                tool=native.tool, args=native.args or {}, idempotency_key=native.native_id
            ),
        )
        return Pending(
            key=native.tool,
            interrupt=interrupt,
            native_id=native.native_id,
            native_state=native.state,
        )
    value = native.value
    if is_hitl(value):
        return _middleware_approval(runtime, ident, native)
    question = value.get("question") if isinstance(value, dict) else None
    interrupt = Interrupt(
        interrupt_id=ident,
        tenant_id=runtime.tenant,
        run_id=runtime.run_id,
        question=str(question or value),
        payload=value if isinstance(value, dict) else {"value": value},
    )
    return Pending(key=FOREIGN, interrupt=interrupt, native_id=native.native_id)


def _middleware_approval(runtime: Runtime, ident: str, native: NativePause) -> Pending:
    """LangChain's ``HumanInTheLoopMiddleware`` (Deep Agents' ``interrupt_on``) paused on the
    calls it holds: an approval of the first, the whole request (every call, each one's
    ``allowed_decisions``) in the payload; the resume answers all of them."""
    request = json.loads(json.dumps(native.value, default=str))
    actions = request["action_requests"]
    first, more = actions[0], len(actions) - 1
    question = f"Approve {first['name']}?"
    if more:
        question += f" (and {more} more call{'s' if more > 1 else ''})"
    args = first.get("args")
    interrupt = Interrupt(
        interrupt_id=ident,
        tenant_id=runtime.tenant,
        run_id=runtime.run_id,
        reason=InterruptReason.APPROVAL,
        question=question,
        tool_call=ToolCall(
            tool=first["name"],
            args=args if isinstance(args, dict) else {},
            idempotency_key=native.native_id,
        ),
        payload=request,
    )
    return Pending(key=HITL, interrupt=interrupt, native_id=native.native_id)


async def _paused(
    agent: Agent,
    runtime: Runtime,
    journal: Journal,
    pending: Pending,
    extracted: Extracted | None,
) -> Result:
    journal.pending = pending
    interrupt = pending.interrupt
    checkpoint = await journal.checkpoint(
        agent.harness.runs.artifacts,
        runtime.run_id,
        worker_id=runtime.worker_id,
        tenant=runtime.tenant,
    )
    await _recorded(
        agent,
        runtime.run_id,
        runtime.tenant,
        lambda: agent.harness.runs.pause(
            interrupt, checkpoint=checkpoint, worker_id=runtime.worker_id
        ),
        lambda r: r.awaiting is not None and r.awaiting.interrupt_id == interrupt.interrupt_id,
    )
    runtime.events.emit(RunEventType.INTERRUPT, data=interrupt.awaiting())
    runtime.events.finished(RunOutcome.INTERRUPT, interrupt=interrupt)
    metrics.run_finished(runtime.agent_id, RunOutcome.INTERRUPT.value)
    await agent.recorded_run(runtime, _transcript(runtime, extracted))  # what it said so far
    return Result(run_id=runtime.run_id, status=RunStatus.PAUSED, interrupt=interrupt)


async def _failed(
    agent: Agent,
    runtime: Runtime,
    error: AgentError,
    extracted: Extracted | None,
    *,
    status: RunStatus = RunStatus.ERROR,
) -> Result:
    """The run ended ``ERROR`` — or ``TIMEOUT``, out of time — with ``error``."""
    outcome = RunOutcome.TIMEOUT if status is RunStatus.TIMEOUT else RunOutcome.ERROR
    await _ended(agent, runtime, status, error=error)
    runtime.events.emit(RunEventType.RUN_ERROR, error=error)
    runtime.events.finished(outcome, error=error)
    metrics.run_finished(runtime.agent_id, outcome.value)
    await agent.recorded_run(runtime, _transcript(runtime, extracted))
    await agent.recorded_outcome(runtime, status, error.message)
    return Result(run_id=runtime.run_id, status=status, error=error)


async def _succeeded(
    agent: Agent, runtime: Runtime, extracted: Extracted, pushed: PromptContext | None
) -> Result:
    answer = extracted.answer
    await _ended(agent, runtime, RunStatus.SUCCESS, output=jsonable(answer))
    runtime.events.finished(RunOutcome.SUCCESS, result=jsonable(answer))
    metrics.run_finished(runtime.agent_id, RunOutcome.SUCCESS.value)
    await agent.recorded_run(runtime, _transcript(runtime, extracted))
    await agent.recorded_outcome(runtime, RunStatus.SUCCESS, None)
    await agent.grounded(runtime, answer, pushed)
    await agent.judged(runtime, answer, pushed)
    if runtime.used_code_mode:
        await agent.imported_code_mode_calls(runtime)
    return Result(run_id=runtime.run_id, status=RunStatus.SUCCESS, answer=answer)


async def _ended(agent: Agent, runtime: Runtime, status: RunStatus, **fields: Any) -> None:
    await _recorded(
        agent,
        runtime.run_id,
        runtime.tenant,
        lambda: agent.harness.runs.finish(
            runtime.run_id, status, worker_id=runtime.worker_id, tenant=runtime.tenant, **fields
        ),
        lambda r: r.status is status,
    )


async def _recorded(
    agent: Agent,
    run_id: str,
    tenant: str,
    write: Callable[[], Awaitable[RunRecord]],
    holds: Callable[[RunRecord], bool],
) -> None:
    """Write the pause or the ending. The runs client retries a write whose answer was lost,
    and agent-runs answers a repeat with the stored record; a store that refuses the repeat
    instead (``ConflictError``) is read, and when the run already is what was written, it was
    written — the run is not failed, queued again or executed again because an answer got
    lost on the way. A lost lease (``LeaseLostError``, not a conflict) is never read away."""
    try:
        await write()
    except ConflictError:
        record = await agent.harness.runs.get(run_id, tenant=tenant)
        if record is None or not holds(record):
            raise
        log.info("run %s was already recorded as %s", run_id, record.status.value)


def _transcript(runtime: Runtime, extracted: Extracted | None) -> list[tuple[str, str]]:
    """The attempt's messages: the question, then what the agent said (its answer, when the
    framework reports no transcript of its own)."""
    said: list[tuple[str, str]] = list(extracted.transcript) if extracted is not None else []
    if not said and extracted is not None and isinstance(extracted.answer, str):
        said = [("assistant", extracted.answer)] if extracted.answer else []
    return [("user", runtime.task), *said] if runtime.task else said


async def _settle_cancelled(
    agent: Agent,
    identity: Identity,
    events: RunEvents,
    worker_id: str | None,
    *,
    reason: str | None = None,
) -> None:
    """End the run ``CANCELLED`` (a cancelled run carries no error: the ``reason`` —
    ``agent.cancel``'s, a person's — goes on its ``RUN_FINISHED`` event and in the log)."""
    try:
        await agent.harness.runs.finish(
            identity.run_id, RunStatus.CANCELLED, worker_id=worker_id, tenant=identity.tenant
        )
    except (LeaseLostError, ConflictError):
        log.info("run %s was ended or taken over elsewhere; nothing written", identity.run_id)
    if reason is not None:
        log.info("run %s was cancelled: %s", identity.run_id, reason)
        events.finished(RunOutcome.CANCELLED, reason=reason)
    else:
        events.finished(RunOutcome.CANCELLED)
    metrics.run_finished(identity.agent_id, RunOutcome.CANCELLED.value)


def jsonable(value: Any) -> Any:
    """An answer as a run record stores it."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return str(value)
    return value
