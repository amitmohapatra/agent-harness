"""One attempt of one run: the fixed pipeline every framework goes through.

identity → (the run record, written by the caller) → tools → memory push → the adapter →
the outcome recorded (paused with its journal as the run's checkpoint, finished with its
answer or error) → background writes (transcript, outcome, sampled judge). The adapter is
the only part that knows the framework.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from trellis.contracts import (
    AgentError,
    Interrupt,
    InterruptReason,
    InterruptResolution,
    RunEvent,
    RunEventType,
    RunOutcome,
    RunStatus,
    ToolCall,
)
from trellis.harness.adapters import convert
from trellis.harness.adapters.base import Extracted, Invocation, NativePause, Output, query_of
from trellis.harness.adapters.langgraph import FOREIGN
from trellis.harness.clients.runs import LeaseLost
from trellis.harness.events import RunEvents
from trellis.harness.identity import Identity
from trellis.harness.journal import Journal, Pending, Replay
from trellis.harness.result import Result
from trellis.harness.runtime import RunCancelled, Runtime, _current, interrupt_id
from trellis.harness.telemetry import metrics, span
from trellis.memory.models import ContextBundle

if TYPE_CHECKING:
    from trellis.harness.agent import Agent

log = logging.getLogger("trellis.run")


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
) -> Result:
    """Run one attempt and record how it ended. The run record must already be RUNNING;
    ``worker_id`` names the worker holding its lease (the store fences its writes)."""
    journal = journal or Journal()
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
        run_memory=agent.run_memory(identity),
        task=query,
        started_at=datetime.now(UTC),
    )
    events.emit(RunEventType.RUN_STARTED, data={"agent_id": identity.agent_id})
    extracted: Extracted | None = None
    pushed: ContextBundle | None = None
    error: Exception | None = None
    cancelled = False
    token = _current.set(runtime)
    try:
        with span(
            "trellis.run",
            {
                "run.id": identity.run_id,
                "run.attempt": number,
                "agent.id": identity.agent_id,
                "agent.framework": agent.adapter.name,
                "tenant.id": identity.tenant,
            },
        ):
            tools = await agent.tools_for(runtime)
            runtime.toolbox = {t.name: t for t in tools}
            pushed = await agent.push(runtime, runtime.tool_names())
            native_input = agent.adapter.prepare_input(agent.target, input, runtime.context)
            if pending is not None and resolution is not None:
                native_input = agent.adapter.resume_input(
                    agent.target, native_input, pending, resolution
                )
            run_tools = [] if agent.adapter.fixed_tools else tools
            invocation = Invocation(
                runtime, run_tools, convert(agent.adapter.tool_format, run_tools)
            )
            output = await _execute(agent, native_input, invocation, streaming=streaming)
            extracted = agent.adapter.extract(agent.target, output)
    except RunCancelled:
        cancelled = True
    except asyncio.CancelledError:
        # the caller went away (a closed stream, a lost lease): the run ends here
        await _settle_cancelled(agent, identity, events, worker_id)
        raise
    except Exception as exc:
        # a framework may wrap or swallow the pause: the runtime is what says it paused
        if runtime.pending is None:
            error = exc
    finally:
        _current.reset(token)
    if cancelled:
        await _settle_cancelled(agent, identity, events, worker_id)
        return Result(run_id=identity.run_id, status=RunStatus.CANCELLED)
    paused = _pause(runtime, extracted, error)
    if paused is not None:
        return await _paused(agent, runtime, journal, paused, extracted)
    if error is not None:
        return await _failed(agent, runtime, error, extracted)
    assert extracted is not None
    return await _succeeded(agent, runtime, extracted, pushed)


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
    question = value.get("question") if isinstance(value, dict) else None
    interrupt = Interrupt(
        interrupt_id=ident,
        tenant_id=runtime.tenant,
        run_id=runtime.run_id,
        question=str(question or value),
        payload=value if isinstance(value, dict) else {"value": value},
    )
    return Pending(key=FOREIGN, interrupt=interrupt, native_id=native.native_id)


async def _paused(
    agent: Agent,
    runtime: Runtime,
    journal: Journal,
    pending: Pending,
    extracted: Extracted | None,
) -> Result:
    journal.pending = pending
    interrupt = pending.interrupt
    await agent.harness.runs.paused(
        interrupt, checkpoint=journal.dump(), worker_id=runtime.worker_id
    )
    runtime.events.emit(RunEventType.INTERRUPT, data=interrupt.awaiting())
    runtime.events.finished(RunOutcome.INTERRUPT, interrupt=interrupt)
    metrics.run_finished(runtime.agent_id, RunOutcome.INTERRUPT.value)
    agent.recorded_run(runtime, _transcript(runtime, extracted))  # what it said so far
    return Result(run_id=runtime.run_id, status=RunStatus.PAUSED, interrupt=interrupt)


async def _failed(
    agent: Agent, runtime: Runtime, exc: BaseException, extracted: Extracted | None
) -> Result:
    error = AgentError.of(exc, source=agent.adapter.name)
    log.warning("run %s failed: %s", runtime.run_id, error.message, exc_info=exc)
    await agent.harness.runs.finished(
        runtime.run_id, RunStatus.ERROR, error=error, worker_id=runtime.worker_id
    )
    runtime.events.emit(RunEventType.RUN_ERROR, error=error)
    runtime.events.finished(RunOutcome.ERROR, error=error)
    metrics.run_finished(runtime.agent_id, RunOutcome.ERROR.value)
    agent.recorded_run(runtime, _transcript(runtime, extracted))
    agent.recorded_outcome(runtime, success=False, note=error.message)
    return Result(run_id=runtime.run_id, status=RunStatus.ERROR, error=error)


async def _succeeded(
    agent: Agent, runtime: Runtime, extracted: Extracted, pushed: ContextBundle | None
) -> Result:
    answer = extracted.answer
    await agent.harness.runs.finished(
        runtime.run_id, RunStatus.SUCCESS, output=jsonable(answer), worker_id=runtime.worker_id
    )
    runtime.events.finished(RunOutcome.SUCCESS, result=jsonable(answer))
    metrics.run_finished(runtime.agent_id, RunOutcome.SUCCESS.value)
    agent.recorded_run(runtime, _transcript(runtime, extracted))
    agent.recorded_outcome(runtime, success=True, note=None)
    agent.judged(runtime, runtime.task, answer, pushed)
    if runtime.used_code_mode:
        agent.imported_code_mode_calls(runtime)
    return Result(run_id=runtime.run_id, status=RunStatus.SUCCESS, answer=answer)


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
    except LeaseLost:
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
