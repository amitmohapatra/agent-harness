"""One attempt of one run: the fixed pipeline every framework goes through.

identity → (the run record, written by the caller) → tools → memory push (with the tool
hints that narrow what the model is offered) → the adapter → the outcome recorded (paused
with its journal as the run's checkpoint, finished with its answer or error) → background
writes (transcript, the run's ``system`` outcome, the sampled grounding check, the sampled
online judges). The adapter is
the only part that knows the framework. Each attempt is one ``invoke_agent`` span in the run's
trace.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Final

from pydantic import BaseModel

from trellis.contracts import (
    AgentError,
    ConfigurationError,
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
from trellis.harness.clients.runs import Conflict, LeaseLost
from trellis.harness.events import RunEvents
from trellis.harness.identity import Identity
from trellis.harness.journal import Journal, Pending, Replay
from trellis.harness.result import Result
from trellis.harness.runtime import RunCancelled, Runtime, _current, interrupt_id
from trellis.harness.telemetry import RunTrace, agent_span, metrics, output
from trellis.memory.models import PromptContext

if TYPE_CHECKING:
    from trellis.harness.agent import Agent

log = logging.getLogger("trellis.run")

#: The message a worker cancels a run with when it stops before the run ends: the run is
#: released, not cancelled — nothing is written, its lease lapses and agent-runs queues it
#: again as its next attempt.
RELEASED: Final = "trellis:released"


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
    lease_seconds: float | None = None,
    observe: Callable[[PromptContext | None], None] | None = None,
) -> Result:
    """Run one attempt and record how it ended. The run record must already be RUNNING;
    ``worker_id`` names the worker holding its lease (the store fences its writes), and
    ``lease_seconds`` its length (progress checkpoints extend it). ``observe`` is told the memory
    context the run was given (an offline evaluation's evaluators read it)."""
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
    )
    events.emit(RunEventType.RUN_STARTED, data={"agent_id": identity.agent_id})
    extracted: Extracted | None = None
    pushed: PromptContext | None = None
    error: Exception | None = None
    cancelled = False
    token = _current.set(runtime)
    try:
        run_trace = RunTrace(
            run_id=identity.run_id,
            agent_id=identity.agent_id,
            tenant=identity.tenant,
            user=identity.user,
            thread=identity.thread,
            framework=agent.adapter.name,
            attempt=number,
        )
        with agent_span(run_trace, query) as span:
            tools = await agent.tools_for(runtime)
            runtime.toolbox = {t.name: t for t in tools}
            pushed = await agent.push(runtime)
            if observe is not None:
                observe(pushed)
            if unresumable is not None:
                raise unresumable
            native_input = agent.adapter.prepare_input(agent.target, input, runtime.context)
            if pending is not None and resolution is not None:
                native_input = agent.adapter.resume_input(
                    agent.target, native_input, pending, resolution
                )
            run_tools = [] if agent.adapter.fixed_tools else tools
            if agent.adapter.narrows == "run":
                run_tools = [t for t in run_tools if runtime.offers(t.name)]
            invocation = Invocation(
                runtime, run_tools, convert(agent.adapter.tool_format, run_tools)
            )
            produced = await _execute(agent, native_input, invocation, streaming=streaming)
            extracted = agent.adapter.extract(agent.target, produced)
            output(span, jsonable(extracted.answer))
    except RunCancelled:
        cancelled = True
    except asyncio.CancelledError as exc:
        # the caller went away (a closed stream, a lost lease): the run ends here — unless
        # its worker is stopping and released it for another worker to run again
        if RELEASED not in exc.args:
            await _settle_cancelled(agent, identity, events, worker_id)
        raise
    except Exception as exc:
        # a framework may wrap or swallow the pause: the runtime is what says it paused
        if runtime.pending is None:
            error = exc
    finally:
        _current.reset(token)
    return await _concluded(
        agent, runtime, journal, extracted, pushed=pushed, error=error, cancelled=cancelled
    )


async def _concluded(
    agent: Agent,
    runtime: Runtime,
    journal: Journal,
    extracted: Extracted | None,
    *,
    pushed: PromptContext | None,
    error: Exception | None,
    cancelled: bool,
) -> Result:
    """Record how the attempt ended: cancelled, paused, failed or answered."""
    if runtime.lease_lost:
        # another worker may hold the run now (a framework may have swallowed the error)
        raise LeaseLost(f"{runtime.worker_id} lost the lease on {runtime.run_id}: nothing written")
    if cancelled:
        await _settle_cancelled(agent, runtime.identity, runtime.events, runtime.worker_id)
        return Result(run_id=runtime.run_id, status=RunStatus.CANCELLED)
    paused = _pause(runtime, extracted, error)
    if paused is not None:
        return await _paused(agent, runtime, journal, paused, extracted)
    if error is not None:
        return await _failed(agent, runtime, error, extracted)
    assert extracted is not None
    return await _succeeded(agent, runtime, extracted, pushed)


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
    await _recorded(
        agent,
        runtime.run_id,
        lambda: agent.harness.runs.paused(
            interrupt, checkpoint=journal.dump(), worker_id=runtime.worker_id
        ),
        lambda r: r.awaiting is not None and r.awaiting.interrupt_id == interrupt.interrupt_id,
    )
    runtime.events.emit(RunEventType.INTERRUPT, data=interrupt.awaiting())
    runtime.events.finished(RunOutcome.INTERRUPT, interrupt=interrupt)
    metrics.run_finished(runtime.agent_id, RunOutcome.INTERRUPT.value)
    await agent.recorded_run(runtime, _transcript(runtime, extracted))  # what it said so far
    return Result(run_id=runtime.run_id, status=RunStatus.PAUSED, interrupt=interrupt)


async def _failed(
    agent: Agent, runtime: Runtime, exc: BaseException, extracted: Extracted | None
) -> Result:
    error = AgentError.of(exc, source=agent.adapter.name)
    log.warning("run %s failed: %s", runtime.run_id, error.message, exc_info=exc)
    await _ended(agent, runtime, RunStatus.ERROR, error=error)
    runtime.events.emit(RunEventType.RUN_ERROR, error=error)
    runtime.events.finished(RunOutcome.ERROR, error=error)
    metrics.run_finished(runtime.agent_id, RunOutcome.ERROR.value)
    await agent.recorded_run(runtime, _transcript(runtime, extracted))
    await agent.recorded_outcome(runtime, RunStatus.ERROR, error.message)
    return Result(run_id=runtime.run_id, status=RunStatus.ERROR, error=error)


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
        lambda: agent.harness.runs.finished(
            runtime.run_id, status, worker_id=runtime.worker_id, **fields
        ),
        lambda r: r.status is status,
    )


async def _recorded(
    agent: Agent,
    run_id: str,
    write: Callable[[], Awaitable[RunRecord]],
    holds: Callable[[RunRecord], bool],
) -> None:
    """Write the pause or the ending. The runs client retries a write whose answer was lost,
    and agent-runs answers a repeat with the stored record; a store that refuses the repeat
    instead (``Conflict``) is read, and when the run already is what was written, it was
    written — the run is not failed, queued again or executed again because an answer got
    lost on the way."""
    try:
        await write()
    except Conflict:
        record = await agent.harness.runs.get(run_id)
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
    agent: Agent, identity: Identity, events: RunEvents, worker_id: str | None
) -> None:
    try:
        await agent.harness.runs.finished(identity.run_id, RunStatus.CANCELLED, worker_id=worker_id)
    except (LeaseLost, Conflict):
        log.info("run %s was taken over by another worker; nothing written", identity.run_id)
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
