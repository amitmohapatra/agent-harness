"""Running an agent for an A2A task.

* The A2A task id is the run id; the context id is the thread.
* Identity comes from the deployment (the resolver), never from the message.
* A pause is the platform's one pause: ``input-required`` with the question, and the next
  message on the task is the answer, resumed through the agent (so the run store, feedback and
  the journal see an A2A answer exactly as they see any other).
* A terminal state is sent once, whichever of the stream and a cancel gets there first.
"""

from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from collections.abc import AsyncIterator, Mapping
from typing import TYPE_CHECKING, Any, Final

from a2a.helpers import new_task, new_text_part
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState
from a2a.utils.errors import InvalidRequestError

from trellis.contracts import (
    ConfigurationError,
    InterruptDecision,
    InterruptReason,
    RunEvent,
    RunStatus,
)
from trellis.harness import pipeline
from trellis.harness.a2a.identity import IdentityRefused, UserResolver
from trellis.harness.a2a.tasks import RunTaskStore
from trellis.harness.a2a.translate import (
    RESULT_ARTIFACT,
    TERMINAL_STATES,
    Update,
    text_and_data,
    update_for,
    value_part,
)

if TYPE_CHECKING:
    from trellis.harness.agent import Agent

log = logging.getLogger("trellis.a2a")

#: Words an approval may be answered with, when the caller sends text.
APPROVE_WORDS: Final = frozenset({"approve", "approved", "yes", "ok", "allow"})
REJECT_WORDS: Final = frozenset({"reject", "rejected", "no", "deny", "denied"})
CANCEL_WORDS: Final = frozenset({"cancel", "abort", "stop"})
#: The data-part key a caller names a decision with outright.
DECISION_KEY: Final = "decision"
#: Task ids whose terminal state was sent, remembered at most this many.
MAX_SETTLED: Final = 4096


class RunExecutor(AgentExecutor):
    def __init__(self, agent: Agent, user_of: UserResolver, tasks: RunTaskStore) -> None:
        self.agent = agent
        self.user_of = user_of
        self.tasks = tasks
        self._running: dict[str, asyncio.Task[Any]] = {}
        self._settled: OrderedDict[str, None] = OrderedDict()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        tenant = await self.agent.harness.tenant()  # what an identity header is checked against
        user = self._user(context)
        task = context.current_task
        task_id = str(context.task_id or "")
        context_id = (task.context_id if task is not None else "") or str(
            context.context_id or task_id
        )
        state = task.status.state if task is not None else None
        if state == TaskState.TASK_STATE_INPUT_REQUIRED:
            await self._resume(context, event_queue, user, task_id, context_id, tenant=tenant)
            return
        if state in TERMINAL_STATES:
            raise InvalidRequestError(
                message="this task has ended; send a new message on the same context instead"
            )
        if state == TaskState.TASK_STATE_WORKING:
            raise InvalidRequestError(message="this task is still working; wait, or cancel it")
        if task is None and await self.agent.harness.runs.get(task_id, tenant=tenant) is not None:
            # a task this caller cannot see (another user's): never a second run on its id
            raise InvalidRequestError(message=f"no task {task_id}")
        self.tasks.opening(task_id)
        await event_queue.enqueue_event(
            new_task(
                task_id=task_id,
                context_id=context_id,
                state=TaskState.TASK_STATE_SUBMITTED,
                history=[context.message] if context.message is not None else None,
            )
        )
        payload = _payload(context.message)
        agent = self.agent
        record = await agent._opened(
            payload, user=user, thread=context_id, tenant=tenant, run_id=task_id
        )
        await self._stream(
            event_queue,
            task_id,
            context_id,
            agent._events(lambda listen: pipeline.attempt(agent, record, payload, listener=listen)),
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        self._user(context)
        task_id = str(context.task_id or "")
        task = context.current_task
        context_id = (task.context_id if task is not None else "") or task_id
        running = self._running.pop(task_id, None)
        if running is not None:
            running.cancel()
        else:
            tenant = await self.agent.harness.tenant()
            record = await self.agent.harness.runs.get(task_id, tenant=tenant)
            if record is not None and record.status is RunStatus.PAUSED:
                await self.agent.harness.runs.finish(task_id, RunStatus.CANCELLED, tenant=tenant)
        await self._cancelled(event_queue, task_id, context_id)

    # ------------------------------------------------------------------ answering a pause
    async def _resume(
        self,
        context: RequestContext,
        event_queue: EventQueue,
        user: str,
        task_id: str,
        context_id: str,
        *,
        tenant: str,
    ) -> None:
        record = await self.agent.harness.runs.get(task_id, tenant=tenant)
        if record is None or record.user_id != user or record.awaiting is None:
            await self._ask_again(
                event_queue, task_id, context_id, "this task is not yours to answer"
            )
            return
        interrupt = record.awaiting
        try:
            decision, answer = _decision(interrupt.reason, context.message)
            record, resolution = await self.agent._resolution(
                interrupt.interrupt_id, decision, answer, reviewer=user, tenant=tenant
            )
        except (ValueError, ConfigurationError) as exc:
            await self._ask_again(event_queue, task_id, context_id, str(exc), interrupt.question)
            return
        if resolution.decision is InterruptDecision.CANCEL:
            await self.agent._continue(record, resolution)
            await self._cancelled(event_queue, task_id, context_id)
            return
        agent, resumed = self.agent, record
        await self._stream(
            event_queue,
            task_id,
            context_id,
            agent._events(lambda listen: agent._continue(resumed, resolution, listen)),
        )

    async def _ask_again(
        self,
        event_queue: EventQueue,
        task_id: str,
        context_id: str,
        reason: str,
        question: str | None = None,
    ) -> None:
        """Say why an answer was not acted on and keep waiting: raising would fail the task."""
        updater = TaskUpdater(event_queue, task_id, context_id)
        parts = [new_text_part(reason), *([new_text_part(question)] if question else [])]
        await updater.update_status(
            TaskState.TASK_STATE_INPUT_REQUIRED, message=updater.new_agent_message(parts)
        )

    # ------------------------------------------------------------------ the stream
    async def _stream(
        self,
        event_queue: EventQueue,
        task_id: str,
        context_id: str,
        events: AsyncIterator[RunEvent],
    ) -> None:
        updater = TaskUpdater(event_queue, task_id, context_id)
        self._running[task_id] = asyncio.current_task()  # type: ignore[assignment]
        finished = False
        try:
            await updater.update_status(TaskState.TASK_STATE_WORKING)
            async for event in events:
                update = update_for(event)
                if update is not None:
                    finished = (
                        finished
                        or update.terminal
                        or update.state == TaskState.TASK_STATE_INPUT_REQUIRED
                    )
                    await self._apply(updater, task_id, update)
        except asyncio.CancelledError:
            if task_id not in self._running:  # cancel() took it: the task says so there
                return
            raise
        except Exception as exc:
            log.warning("A2A task %s failed: %s", task_id, exc)
        finally:
            self._running.pop(task_id, None)
        if not finished and self._settle(task_id):
            await updater.update_status(
                TaskState.TASK_STATE_FAILED,
                message=updater.new_agent_message(
                    [new_text_part("the run ended without a result")]
                ),
            )

    async def _apply(self, updater: TaskUpdater, task_id: str, update: Update) -> None:
        if update.terminal and not self._settle(task_id):
            return
        if update.result is not None:
            await updater.add_artifact(
                [value_part(update.result)], name=RESULT_ARTIFACT, last_chunk=True
            )
        message = updater.new_agent_message(list(update.parts)) if update.parts else None
        await updater.update_status(update.state, message=message)

    async def _cancelled(self, event_queue: EventQueue, task_id: str, context_id: str) -> None:
        if self._settle(task_id):
            updater = TaskUpdater(event_queue, task_id, context_id)
            await updater.update_status(
                TaskState.TASK_STATE_CANCELED,
                message=updater.new_agent_message([new_text_part("the run was cancelled")]),
            )

    def _settle(self, task_id: str) -> bool:
        """Claim the right to send this task's terminal state; no ``await`` in between."""
        if task_id in self._settled:
            return False
        self._settled[task_id] = None
        while len(self._settled) > MAX_SETTLED:
            self._settled.popitem(last=False)
        return True

    def _user(self, context: RequestContext) -> str:
        try:
            return self.user_of(context.call_context)
        except IdentityRefused as exc:
            raise InvalidRequestError(message=str(exc)) from exc


def _payload(message: Message | None) -> Any:
    """What the agent is asked: the message's text, or its data when it carries none."""
    text, data = text_and_data(message)
    return text or data or ""


def _decision(reason: InterruptReason, message: Message | None) -> tuple[InterruptDecision, Any]:
    """A caller's answer as a decision and its value. A data part may name the decision
    (``{"decision": "approve"}``); an approval reads approve/reject words, or an object as the
    edited arguments; anything else answers the question."""
    text, data = text_and_data(message)
    word = text.strip().lower()
    named = str(data.get(DECISION_KEY) or "").strip().upper()
    answer: Any = data.get("answer", text or None) if data else (text or None)
    if named:
        decision = InterruptDecision(named)
    elif word in CANCEL_WORDS:
        decision = InterruptDecision.CANCEL
    elif reason is InterruptReason.APPROVAL:
        if word in APPROVE_WORDS:
            decision = InterruptDecision.APPROVE
        elif word in REJECT_WORDS:
            decision = InterruptDecision.REJECT
        elif data:
            decision = InterruptDecision.EDIT
        else:
            raise ValueError(
                "an approval is answered with approve, reject, cancel, or the edited arguments"
            )
    else:
        decision = InterruptDecision.ANSWER
    if decision is InterruptDecision.EDIT:
        payload = data.get("payload")
        answer = (
            dict(payload)
            if isinstance(payload, Mapping)
            else {k: v for k, v in data.items() if k != DECISION_KEY}
        )
    return decision, answer


__all__ = ["RunExecutor"]
