"""Running a harness agent for an A2A task (design §9).

```mermaid
sequenceDiagram
  participant C as Calling agent
  participant X as HarnessAgentExecutor
  participant H as Harness
  participant S as Event sink
  C->>X: SendStreamingMessage
  X->>X: identity from the trusted header (refuse, or place the caller)
  X->>C: Task(submitted)
  X->>H: run(payload, context) with task id as the run id
  H->>S: RunEvents
  S-->>X: subscribed queue
  X->>C: working + text · progress data parts
  alt the agent asked a person
    X->>C: input-required (question + expects)
    C->>X: next message on the same task
    X->>H: resolutions.claim + harness.resume
    X->>C: working ... completed
  else
    X->>C: artifact(result) then completed
  end
```

The four rules that make this more than a bridge:

1. **The A2A task id *is* the harness run id.** One id for one unit of work means a task can be
   rebuilt from the run store, a webhook about the run names the task, and a trace joins both.
2. **Identity comes from the platform, never the message.** Refused before a run starts.
3. **A pause is the platform's one pause.** `input-required` is `AgentPaused` and nothing else, and
   the answer goes back through `harness.resume`, so the run store, the feedback record and tool
   memory see an A2A answer exactly as they see a UI's.
4. **A terminal state is sent once.** The A2A SDK latches terminal states, and a cancel arriving
   while the stream is finishing must not race the stream into an exception.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from collections.abc import Mapping
from typing import Any, Final

from a2a.helpers import new_task
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import Message, Task, TaskState
from a2a.utils.errors import InvalidParamsError, InvalidRequestError
from trellis.contracts.context import AgentExecutionContext
from trellis.contracts.ids import safe_id
from trellis.contracts.runs import (
    Interrupt,
    InterruptDecision,
    InterruptReason,
    InterruptResolution,
    RunEventType,
)

from trellis.harness.events import CollectingEventSink
from trellis.harness.runtime.logging import get_logger
from trellis.harness_a2a.identity import (
    EXTENSION_URI,
    IDENTITY_FIELDS,
    IdentityRefused,
    IdentityResolver,
    check_claimed_tenant,
)
from trellis.harness_a2a.translate import (
    TERMINAL_STATES,
    Update,
    text_part,
    update_for,
    value_part,
)

log = get_logger("trellis.harness_a2a.executor")

#: Words a caller may answer an approval with, when it sends text rather than a decision.
APPROVE_WORDS: Final = frozenset({"approve", "approved", "yes", "ok", "allow"})
REJECT_WORDS: Final = frozenset({"reject", "rejected", "no", "deny", "denied"})
CANCEL_WORDS: Final = frozenset({"cancel", "abort", "stop"})
#: The data-part key by which a caller names a contracts decision outright.
DECISION_KEY: Final = "decision"
#: How many settled task ids are remembered, so a long-lived server does not grow a set per task.
MAX_SETTLED: Final = 4096


class HarnessAgentExecutor(AgentExecutor):
    """One harness agent, served as an A2A agent."""

    def __init__(
        self,
        harness: Any,
        *,
        agent: Any,
        agent_id: str | None = None,
        identity: IdentityResolver | None = None,
        sink: CollectingEventSink | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """``agent`` is a callable the harness can wrap (or an already-wrapped agent).
        ``identity`` is how the deployment places the caller; without one the harness's own
        default tenant is used, which is only ever right for a single-tenant deployment."""
        self.harness = harness
        wrapped = (
            agent
            if hasattr(agent, "descriptor") and hasattr(agent, "harness")
            else harness.wrap(agent, agent_id=agent_id)
        )
        self.agent = wrapped
        self.agent_id: str = wrapped.descriptor.agent_id
        self.identity = identity
        self.sink = sink or _collecting_sink(harness)
        self.base_metadata = dict(metadata or {})
        self._runner = getattr(wrapped, "arun", wrapped)
        self._running: dict[str, asyncio.Task[Any]] = {}
        #: Task ids whose terminal state was sent, bounded like every other memory here.
        self._settled: OrderedDict[str, bool] = OrderedDict()

    # ------------------------------------------------------------------ the SDK's interface
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        identity = self._identity(context)
        task = context.current_task
        task_id = self._run_id(context)
        context_id = self._context_id(context, task)
        state = task.status.state if task is not None else None
        if state is TaskState.TASK_STATE_INPUT_REQUIRED:
            await self._resume(
                context, event_queue, identity, task_id=task_id, context_id=context_id
            )
            return
        if state in TERMINAL_STATES:
            # A2A tasks are immutable once terminal: a refinement is a new task on the same
            # context, which is also what keeps a finished run's record from being rewritten.
            raise InvalidRequestError(
                message="this task has ended; send a new message on the same context instead"
            )
        if state is TaskState.TASK_STATE_WORKING:
            raise InvalidRequestError(
                message="this task is still working; wait for it, or cancel it first"
            )
        await self._start(
            context, event_queue, identity, task, task_id=task_id, context_id=context_id
        )

    @staticmethod
    def _context_id(context: RequestContext, task: Task | None) -> str:
        """The conversation this call belongs to.

        The *task's* own context wins over the request's: a follow-up message that names a task but
        no context makes the SDK generate a fresh context id, and treating that as the thread would
        separate an answer from the run it answers.
        """
        if task is not None and task.context_id:
            return task.context_id
        return str(context.context_id or context.task_id or "")

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel the run behind this task. The caller must be the task's own."""
        self._identity(context)
        task_id = str(context.task_id or "")
        context_id = self._context_id(context, context.current_task) or task_id
        running = self._running.get(task_id)
        if running is not None:
            running.cancel()
        if self._settle(task_id):
            updater = TaskUpdater(event_queue, task_id, context_id)
            await updater.update_status(
                TaskState.TASK_STATE_CANCELED,
                message=updater.new_agent_message([text_part("the run was cancelled")]),
            )

    # ------------------------------------------------------------------ starting a run
    async def _start(
        self,
        context: RequestContext,
        event_queue: EventQueue,
        identity: dict[str, Any],
        task: Task | None,
        *,
        task_id: str,
        context_id: str,
    ) -> None:
        if task is None:
            # The SDK requires the Task itself before any status update, and the run id has to be
            # the task's, so it is built here rather than left to a helper's own id generator.
            await event_queue.enqueue_event(
                new_task(
                    task_id=task_id,
                    context_id=context_id,
                    state=TaskState.TASK_STATE_SUBMITTED,
                    history=[context.message] if context.message is not None else None,
                )
            )
        execution = self._context(task_id, context_id, identity)
        await self._run(execution, _payload(context.message), context, event_queue, context_id)

    def _context(
        self, task_id: str, context_id: str, identity: Mapping[str, Any]
    ) -> AgentExecutionContext:
        """The run is named before it starts: the task id *is* the run id, and the context id is
        the thread, so a conversation between two agents is one thread in memory."""
        overrides: dict[str, Any] = {
            "thread_id": context_id,
            "session_id": safe_id(f"{context_id}-session"),
            "turn_id": safe_id(task_id),
            "agent_run_id": task_id,
            **{k: v for k, v in identity.items() if v},
        }
        return self.harness.context_factory.build(agent_id=self.agent_id, overrides=overrides)

    # ------------------------------------------------------------------ answering a pause
    async def _resume(
        self,
        context: RequestContext,
        event_queue: EventQueue,
        identity: dict[str, Any],
        *,
        task_id: str,
        context_id: str,
    ) -> None:
        interrupt = self._pause_of(str(identity.get("tenant_id") or ""), task_id)
        if interrupt is None:
            await self._ask_again(
                event_queue,
                task_id,
                context_id,
                "this task is not waiting for an answer in this process",
            )
            return
        claimed = self.harness.resolutions.claim(
            interrupt.interrupt_id, thread_id=context_id, **identity
        )
        if claimed is None:
            log.info("a2a.resume_refused", task_id=task_id, reason="not the run's own caller")
            await self._ask_again(
                event_queue, task_id, context_id, "this task is not yours to answer"
            )
            return
        pause, execution = claimed
        try:
            resolution = _resolution(pause, context.message, identity)
            # ``resume`` refuses a decision that does not fit the question (an approval answered
            # with an answer, and so on). That refusal must not end the wait either: claim() has
            # already taken the pause off the registry, so it goes back before we answer the caller.
            await self.harness.resume(pause, resolution, context=execution)
        except ValueError as exc:
            self.harness.resolutions.announce(pause, execution)  # a bad answer is not the end
            await self._ask_again(
                event_queue, task_id, context_id, str(exc), question=pause.question
            )
            return
        if resolution.decision is InterruptDecision.CANCEL:
            # cancelling answers the question without running the agent again
            if self._settle(task_id):
                updater = TaskUpdater(event_queue, task_id, context_id)
                await updater.update_status(
                    TaskState.TASK_STATE_CANCELED,
                    message=updater.new_agent_message([text_part("the run was cancelled")]),
                )
            return
        await self._run(execution, None, context, event_queue, context_id)

    async def _ask_again(
        self,
        event_queue: EventQueue,
        task_id: str,
        context_id: str,
        reason: str,
        *,
        question: str | None = None,
    ) -> None:
        """Say why an answer was not acted on, and keep waiting for a usable one.

        Deliberately not an exception. Raising out of ``execute`` makes the SDK mark the task
        ``failed`` — and a task waiting for input is not terminal, so a malformed or unauthorised
        answer would end a wait that the harness itself keeps open. Asking again is both the
        protocol's own idiom and the only response that leaves the run answerable.
        """
        updater = TaskUpdater(event_queue, task_id, context_id)
        parts = [text_part(reason)]
        if question:
            parts.append(text_part(question))
        await updater.update_status(
            TaskState.TASK_STATE_INPUT_REQUIRED, message=updater.new_agent_message(parts)
        )

    def _pause_of(self, tenant_id: str, task_id: str) -> Interrupt | None:
        """The pause this task is waiting on, as the harness announced it.

        Looked up by the *task* rather than taken from the caller's message: the question of which
        interrupt is being answered is the server's to answer, and one run has one open pause.
        """
        for interrupt in self.harness.resolutions.announced(tenant_id):
            if interrupt.run_id == task_id:
                return interrupt
        return None

    # ------------------------------------------------------------------ the stream
    async def _run(
        self,
        execution: AgentExecutionContext,
        payload: Any,
        context: RequestContext,
        event_queue: EventQueue,
        context_id: str,
    ) -> None:
        task_id = execution.agent_run_id
        updater = TaskUpdater(event_queue, task_id, context_id or task_id)
        queue = self.sink.subscribe(execution.tenant_id, task_id, replay=False)
        run = asyncio.create_task(
            _guarded(self._runner(payload, context=execution, metadata=self._metadata(context))),
            name=f"a2a:{task_id}",
        )
        self._running[task_id] = run
        finished = False
        try:
            await updater.update_status(TaskState.TASK_STATE_WORKING)
            while True:
                event = await _next(queue, run)
                if event is None:
                    break
                finished = event.type is RunEventType.RUN_FINISHED
                update = update_for(event)
                if update is not None:
                    await self._apply(updater, task_id, update)
                if finished:
                    break
            if not finished and self._settle(task_id):
                # the run ended without announcing it (an error before its first event)
                await updater.update_status(
                    TaskState.TASK_STATE_FAILED,
                    message=updater.new_agent_message(
                        [text_part("the run ended without a result")]
                    ),
                )
        finally:
            self.sink.unsubscribe(execution.tenant_id, task_id, queue)
            self._running.pop(task_id, None)
            if not run.done():
                # the SDK cancels the producer when a caller disconnects; an agent left running
                # with nobody consuming its events and no handle to reach it is a leak
                run.cancel()

    async def _apply(self, updater: TaskUpdater, task_id: str, update: Update) -> None:
        """One task update, with the terminal one sent at most once."""
        if update.terminal and not self._settle(task_id):
            return
        if update.artifact is not None:
            name, value = update.artifact
            await updater.add_artifact([value_part(value)], name=name, last_chunk=True)
        message = updater.new_agent_message(list(update.parts)) if update.parts else None
        await updater.update_status(update.state, message=message)

    def _settle(self, task_id: str) -> bool:
        """Claim the right to send this task's terminal state. Second callers get ``False``.

        No ``await`` between the check and the claim, so the stream and a concurrent cancel cannot
        both get through — which the SDK would answer by raising on the second terminal update.
        """
        if task_id in self._settled:
            return False
        self._settled[task_id] = True
        while len(self._settled) > MAX_SETTLED:
            self._settled.popitem(last=False)
        return True

    def _metadata(self, context: RequestContext) -> dict[str, Any]:
        """What the run records about the call. Request metadata, never identity."""
        extensions = sorted(context.requested_extensions or ())
        return {
            **self.base_metadata,
            "a2a": True,
            "a2a_task_id": context.task_id,
            "a2a_context_id": context.context_id,
            "a2a_extensions": extensions,
            "a2a_identity_extension": EXTENSION_URI in extensions,
        }

    # ------------------------------------------------------------------ identity
    def _identity(self, context: RequestContext) -> dict[str, Any]:
        """Who is asking, from the deployment's own authentication.

        The identity is the resolver's, with the harness's own defaults filling only the fields the
        resolver did not assert. That filling matters as much as the resolving: the run this call
        creates gets those defaults too (``ContextFactory`` back-fills user and workspace), so a
        resume's ownership check has to compare the same set of fields or a caller could not answer
        its own pause. Defaults are deployment configuration, never caller input.

        A refusal is an error before anything runs: no task state, no run, no memory.
        """
        defaults = self.harness.context_factory.defaults
        if self.identity is None:
            if not defaults.get("tenant_id"):
                raise InvalidRequestError(
                    message="this deployment cannot place the caller: no identity resolver"
                )
            return {f: defaults[f] for f in IDENTITY_FIELDS if defaults.get(f)}
        try:
            identity = self.identity.resolve(context.call_context)
        except IdentityRefused as exc:
            log.info("a2a.identity_refused", reason=str(exc))
            raise InvalidRequestError(message=str(exc)) from exc
        try:
            check_claimed_tenant(str(identity.get("tenant_id") or ""), _claimed_tenant(context))
        except IdentityRefused as exc:
            log.info("a2a.tenant_mismatch", reason=str(exc))
            raise InvalidRequestError(message=str(exc)) from exc
        for field in IDENTITY_FIELDS[1:]:  # never the tenant: that one is the caller's own
            if not identity.get(field) and defaults.get(field):
                identity[field] = defaults[field]
        return identity

    def _run_id(self, context: RequestContext) -> str:
        """The task id, once it is usable as a run id.

        The harness derives memory scopes and idempotency keys from a run id, so a task id that
        would be rewritten by ``safe_id`` is refused rather than silently renamed — the store
        would then hold a run the task could never be matched to again.
        """
        task_id = str(context.task_id or "")
        if not task_id or safe_id(task_id) != task_id:
            raise InvalidParamsError(message="taskId must be an identifier (letters, digits, -_.:)")
        return task_id


# ---------------------------------------------------------------------------- helpers


def _resolution(
    interrupt: Interrupt, message: Message | None, identity: Mapping[str, Any]
) -> InterruptResolution:
    """The contracts resolution for a caller's answer.

    A data part may name the decision outright (``{"decision": "approve"}``, the same extension
    the AG-UI surface takes); otherwise an approval reads the text as approve/reject/cancel and an
    object as the edited arguments, and a question takes whatever was sent as its answer.
    """
    text, data = _text_and_data(message)
    named = str(data.get(DECISION_KEY) or "").strip().lower() if data else ""
    answer: Any = data.get("answer", text or None) if data else (text or None)
    if named:
        decision = InterruptDecision(named.upper())
    elif text.strip().lower() in CANCEL_WORDS:
        decision = InterruptDecision.CANCEL
    elif interrupt.reason is InterruptReason.APPROVAL:
        decision = _approval_decision(text, data)
    else:
        decision = InterruptDecision.ANSWER
    edited = decision is InterruptDecision.EDIT
    payload = data.get("payload") if data else None
    return InterruptResolution(
        interrupt_id=interrupt.interrupt_id,
        run_id=interrupt.run_id,
        decision=decision,
        answer=None if edited else answer,
        payload=dict(payload or data or {}) if edited else None,
        reviewer=identity.get("user_id"),  # the authenticated caller, never a self-declared name
    )


def _approval_decision(text: str, data: Mapping[str, Any] | None) -> InterruptDecision:
    word = text.strip().lower()
    if word in APPROVE_WORDS:
        return InterruptDecision.APPROVE
    if word in REJECT_WORDS:
        return InterruptDecision.REJECT
    if data:
        return InterruptDecision.EDIT
    raise ValueError(
        "an approval is answered with approve, reject, cancel, or the edited arguments"
    )


def _text_and_data(message: Message | None) -> tuple[str, dict[str, Any]]:
    """The message's text and its merged data parts."""
    if message is None:
        return "", {}
    texts: list[str] = []
    data: dict[str, Any] = {}
    for part in message.parts:
        which = part.WhichOneof("content")
        if which == "text":
            texts.append(part.text)
        elif which == "data":
            value = _plain(part.data)
            if isinstance(value, Mapping):
                data.update(value)
    return "\n".join(texts), data


def _payload(message: Message | None) -> Any:
    """What the agent is asked to do: the message's text, or its data when it carries none."""
    text, data = _text_and_data(message)
    if text:
        return text
    return data or None


def _plain(value: Any) -> Any:
    """A protobuf ``Value`` as plain Python."""
    from google.protobuf.json_format import MessageToDict  # noqa: PLC0415 - proto only here

    try:
        return MessageToDict(value)
    except Exception:  # pragma: no cover - a Value that will not convert is not data we want
        return None


def _claimed_tenant(context: RequestContext) -> str | None:
    """The tenant this *call* names, wherever the SDK found it.

    On JSON-RPC that is the request body's own ``tenant`` field; on the REST routes it is the path
    segment. Both are read for one purpose only — to refuse a call that contradicts the
    authenticated caller (:func:`check_claimed_tenant`). Neither is ever used as identity: the
    trusted header is.
    """
    return str(getattr(context, "tenant", "") or "") or None


async def _next(queue: asyncio.Queue[Any], task: asyncio.Task[Any]) -> Any:
    """The next event, or None when the run's task is over and nothing more is queued."""
    getter = asyncio.ensure_future(queue.get())
    done, _ = await asyncio.wait({getter, task}, return_when=asyncio.FIRST_COMPLETED)
    if getter in done:
        return getter.result()
    getter.cancel()
    return queue.get_nowait() if not queue.empty() else None


async def _guarded(awaitable: Any) -> Any:
    """The run's outcome is on the stream; a pause re-raised or an error already reported must not
    tear the task's stream down."""
    try:
        return await awaitable
    except (Exception, asyncio.CancelledError):
        return None


def _collecting_sink(harness: Any) -> CollectingEventSink:
    for sink in harness.event_sinks:
        if isinstance(sink, CollectingEventSink):
            return sink
    sink = CollectingEventSink()
    harness.event_sinks.append(sink)  # the builder shares the list: every run sees it
    return sink


__all__ = ["HarnessAgentExecutor"]
