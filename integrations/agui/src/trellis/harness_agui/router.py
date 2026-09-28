"""``POST /agui/run``: run an agent, stream its events, take a resume (design §10).

The route runs the harness for the thread the input names; a subscriber on the harness's
collecting sink turns the run's ``RunEvent``s into AG-UI events as they happen. A ``resume``
entry answers an interrupt of an earlier run on the same thread through ``harness.resume``,
so the resumed run finds the answer.

Identity is the deployment's, never the client's: the route sits behind the deployment's
own authentication and ``context_factory(request, body)`` derives the tenant, user and
workspace from it. Without a factory the router's ``tenant_id`` is the whole identity;
``forwardedProps`` reaches the run as metadata and decides nothing about who is calling.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from typing import Any, Final

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from trellis.contracts.context import AgentExecutionContext
from trellis.contracts.ids import new_id, safe_id
from trellis.contracts.runs import (
    Interrupt,
    InterruptDecision,
    InterruptReason,
    InterruptResolution,
    RunEventType,
)
from trellis.harness_agui.events import (
    AGUIEvent,
    AGUIEventType,
    Resume,
    ResumeStatus,
    RunAgentInput,
)
from trellis.harness_agui.sse import MEDIA_TYPE, encode
from trellis.harness_agui.translate import translate

from trellis.harness.events import CollectingEventSink
from trellis.harness.runtime.logging import get_logger

log = get_logger(__name__)

#: A deployment's hook: the identity fields of the caller, from its own authentication.
ContextFactory = Callable[[Request, RunAgentInput], dict[str, Any]]
IDENTITY_FIELDS: Final = ("tenant_id", "user_id", "workspace_id")
#: Error codes on ``RUN_ERROR`` events the route itself emits.
UNKNOWN_INTERRUPT: Final = "UNKNOWN_INTERRUPT"
BAD_RESUME: Final = "BAD_RESUME"
RUN_ABORTED: Final = "RUN_ABORTED"


def agui_router(
    harness: Any,
    *,
    agent: Callable[..., Any],
    agent_id: str,
    prefix: str = "/agui",
    tenant_id: str | None = None,
    context_factory: ContextFactory | None = None,
) -> APIRouter:
    """A router serving ``agent`` (a callable the harness can wrap) at ``POST {prefix}/run``.

    ``context_factory(request, body)`` returns the identity fields (``tenant_id``,
    ``user_id``, ``workspace_id``) the deployment derives from its authentication; without
    it every run belongs to ``tenant_id`` and no user. One of the two is required.
    """
    if context_factory is None and not tenant_id:
        raise ValueError("agui_router needs a context_factory or a tenant_id")
    router = APIRouter(prefix=prefix, tags=["agui"])
    surface = Surface(
        harness, agent=agent, agent_id=agent_id, tenant_id=tenant_id, factory=context_factory
    )

    @router.post("/run", summary="Run the agent for a thread and stream AG-UI events")
    async def run(request: Request, body: RunAgentInput) -> StreamingResponse:
        if body.run_id and safe_id(body.run_id) != body.run_id:
            # echoed on every event and used as the run's id: it has to be an id
            raise HTTPException(422, "runId must be an identifier (letters, digits, -_.:)")
        identity = surface.identity(request, body)
        return StreamingResponse(surface.stream(body, identity), media_type=MEDIA_TYPE)

    return router


class Surface:
    """The route's state: the wrapped agent, the sink it streams from, the identity rule."""

    def __init__(
        self,
        harness: Any,
        *,
        agent: Callable[..., Any],
        agent_id: str,
        tenant_id: str | None,
        factory: ContextFactory | None,
    ) -> None:
        self.harness = harness
        self.tenant_id = tenant_id
        self.factory = factory
        self.sink = _collecting_sink(harness)
        wrapped = harness.wrap(agent, agent_id=agent_id)
        self.runner = getattr(wrapped, "arun", wrapped)
        self.agent_id = wrapped.descriptor.agent_id  # the id the harness runs it under

    def identity(self, request: Request, body: RunAgentInput) -> dict[str, Any]:
        if self.factory is not None:
            fields = {k: v for k, v in self.factory(request, body).items() if k in IDENTITY_FIELDS}
            if not fields.get("tenant_id"):
                raise ValueError("context_factory must return a tenant_id")
            return fields
        return {"tenant_id": self.tenant_id}

    async def stream(self, body: RunAgentInput, identity: dict[str, Any]) -> AsyncIterator[str]:
        metadata: dict[str, Any] = {"agui": True}
        if body.forwarded_props is not None:
            metadata["forwarded_props"] = body.forwarded_props
        if body.tools:
            metadata["frontend_tools"] = [t.model_dump(by_alias=True) for t in body.tools]
        if body.context:
            metadata["context"] = [c.model_dump(by_alias=True) for c in body.context]
        if body.resume:
            context, refusal = await self._resume(body, identity)
            if context is None:
                yield encode(_error(body.thread_id, *refusal, run_id=body.run_id))
                return
        else:
            context = self._context(body, identity)
        async for chunk in self._run(context, body, metadata):
            yield chunk

    def _context(self, body: RunAgentInput, identity: dict[str, Any]) -> AgentExecutionContext:
        """The run is named up front so its id is known before it starts: the client's
        ``runId`` is echoed on every event, and a run without one gets a derived id (the
        tenant is part of the derivation, so two tenants never share one)."""
        overrides: dict[str, Any] = {
            "thread_id": body.thread_id,
            "session_id": safe_id(f"{body.thread_id}-session"),
            "turn_id": safe_id(body.run_id) if body.run_id else new_id("turn_"),
            **{k: v for k, v in identity.items() if v},
        }
        if body.run_id:
            overrides["agent_run_id"] = body.run_id
        if body.parent_run_id:
            overrides["parent_agent_run_id"] = body.parent_run_id
        return self.harness.context_factory.build(agent_id=self.agent_id, overrides=overrides)

    async def _resume(
        self, body: RunAgentInput, identity: dict[str, Any]
    ) -> tuple[AgentExecutionContext | None, tuple[str, str]]:
        """The paused run's context once its interrupt is answered, or why it was not."""
        answer = body.resume[-1]
        claimed = self.harness.resolutions.claim(
            answer.interrupt_id, thread_id=body.thread_id, **identity
        )
        if claimed is None:
            return None, (UNKNOWN_INTERRUPT, "no such interrupt for this thread and caller")
        interrupt, context = claimed
        try:
            resolution = _resolution(interrupt, answer, identity)
            await self.harness.resume(interrupt, resolution, context=context)
        except ValueError as exc:
            # the pause stays announced: a malformed answer is not the end of the run
            self.harness.resolutions.announce(interrupt, context)
            log.info("agui.resume_refused", interrupt_id=interrupt.interrupt_id, reason=str(exc))
            return None, (BAD_RESUME, str(exc))
        return context, ("", "")

    async def _run(
        self, context: AgentExecutionContext, body: RunAgentInput, metadata: dict[str, Any]
    ) -> AsyncIterator[str]:
        # new events only: a resumed run shares its id with the run that paused, whose
        # events the client already has
        queue = self.sink.subscribe(context.tenant_id, context.agent_run_id, replay=False)
        payload = body.latest_user_text() or body.state
        task = asyncio.create_task(
            _guarded(self.runner(payload, context=context, metadata=metadata)),
            name=f"agui:{context.agent_run_id}",
        )
        finished = False
        try:
            while not finished:
                event = await _next(queue, task)
                if event is None:
                    break
                finished = event.type is RunEventType.RUN_FINISHED
                agui = translate(event)
                if agui is not None:
                    yield encode(agui)
            if not finished:
                # the run ended without announcing it (an error before its first event)
                yield encode(
                    _error(
                        body.thread_id,
                        RUN_ABORTED,
                        "the run ended without a result",
                        run_id=context.agent_run_id,
                    )
                )
        finally:
            # a client that went away is a client the webhook notifier serves; the run goes on
            self.sink.unsubscribe(context.tenant_id, context.agent_run_id, queue)


async def _next(queue: asyncio.Queue[Any], task: asyncio.Task[Any]) -> Any:
    """The next event, or None when the run's task is over and nothing more is queued."""
    getter = asyncio.ensure_future(queue.get())
    done, _ = await asyncio.wait({getter, task}, return_when=asyncio.FIRST_COMPLETED)
    if getter in done:
        return getter.result()
    getter.cancel()
    return queue.get_nowait() if not queue.empty() else None


async def _guarded(awaitable: Any) -> Any:
    """The run's own outcome is on the stream; an exception here (a pause re-raised, an
    error the harness already reported) must not tear the response down."""
    try:
        return await awaitable
    except (Exception, asyncio.CancelledError):
        return None


def _resolution(
    interrupt: Interrupt, answer: Resume, identity: dict[str, Any]
) -> InterruptResolution:
    """The contracts resolution for a protocol resume entry: ``cancelled`` abandons the run;
    ``resolved`` answers a question with the payload, and an approval with ``true``
    (approve), ``false`` (reject) or an object (the edited arguments). The ``decision``
    extension names the decision outright."""
    if answer.decision:
        decision = InterruptDecision(answer.decision.upper())
    elif answer.status is ResumeStatus.CANCELLED:
        decision = InterruptDecision.CANCEL
    elif interrupt.reason is InterruptReason.APPROVAL:
        decision = _approval_decision(answer.payload)
    else:
        decision = InterruptDecision.ANSWER
    edited = decision is InterruptDecision.EDIT
    return InterruptResolution(
        interrupt_id=answer.interrupt_id,
        run_id=interrupt.run_id,
        decision=decision,
        answer=None if edited else answer.payload,
        payload=dict(answer.payload) if edited and isinstance(answer.payload, dict) else None,
        reviewer=identity.get("user_id"),  # the authenticated caller, never a self-declared name
    )


def _approval_decision(payload: Any) -> InterruptDecision:
    if payload is True:
        return InterruptDecision.APPROVE
    if payload is False:
        return InterruptDecision.REJECT
    if isinstance(payload, dict):
        return InterruptDecision.EDIT
    raise ValueError("an approval is answered with true, false or the edited arguments")


def _error(thread_id: str, code: str, message: str, *, run_id: str | None = None) -> AGUIEvent:
    return AGUIEvent(
        type=AGUIEventType.RUN_ERROR, thread_id=thread_id, run_id=run_id, message=message, code=code
    )


def _collecting_sink(harness: Any) -> CollectingEventSink:
    for sink in harness.event_sinks:
        if isinstance(sink, CollectingEventSink):
            return sink
    sink = CollectingEventSink()
    harness.event_sinks.append(sink)  # the builder shares the list: every run sees it
    return sink
