"""The AG-UI chat surface: ``agent.serve_chat(app)``.

* ``POST {path}/run`` — run the agent for a thread (or answer the interrupt a paused run of
  it waits on, with a ``resume`` entry) and stream its AG-UI events as SSE. The run executes
  in the background: a client that goes away does not stop it.
* ``GET {path}/runs/{run_id}/events`` — reconnect: the run's events after ``Last-Event-ID``
  (or ``?after=``), then live until it finishes.
* ``GET {path}/runs/{run_id}/artifacts/{artifact_id}`` — data the interrupt a paused run
  waits on carries by reference (``payload_ref``: a large ``ask`` table or diff), read from
  agent-runs.

Identity is the deployment's: ``identity(request)`` returns the user, and the tenant is the
one ``TRELLIS_API_KEY`` speaks for. Nothing the client sends (``forwardedProps``...) decides
who is calling.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import TYPE_CHECKING, Any, Final

from fastapi import APIRouter, FastAPI, Header, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from trellis.contracts import (
    ConfigurationError,
    InterruptDecision,
    InterruptReason,
    RunEvent,
    RunEventType,
    RunStatus,
    safe_id,
)
from trellis.harness import pipeline
from trellis.harness.result import Result
from trellis.harness.runtime import run_of
from trellis.harness.surfaces.agui.events import (
    AGUIEvent,
    AGUIEventType,
    Outcome,
    OutcomeType,
    Resume,
    ResumeStatus,
    RunAgentInput,
)
from trellis.harness.surfaces.agui.hub import Hub, RunBuffer
from trellis.harness.surfaces.agui.sse import MEDIA_TYPE, encode
from trellis.harness.surfaces.agui.translate import translate

if TYPE_CHECKING:
    from trellis.harness.agent import Agent

log = logging.getLogger("trellis.agui")

Identity = Callable[[Request], str | Awaitable[str]]
#: The user every request runs as when the deployment named no ``identity``.
ANONYMOUS: Final = "anonymous"
#: ``RUN_ERROR`` codes the surface itself emits.
BAD_RESUME: Final = "BAD_RESUME"
RUN_ABORTED: Final = "RUN_ABORTED"


def mount(app: FastAPI, agent: Agent, *, path: str, identity: Identity | None) -> None:
    if identity is None:
        log.warning(
            "serve_chat for %s has no identity=: every request runs as user %r",
            agent.id,
            ANONYMOUS,
        )
    app.include_router(_router(_Surface(agent, identity), path))


class _Surface:
    def __init__(self, agent: Agent, identity: Identity | None) -> None:
        self.agent = agent
        self.identity = identity
        self.hub = Hub()
        self._tasks: set[asyncio.Task[None]] = set()

    async def user(self, request: Request) -> str:
        if self.identity is None:
            return ANONYMOUS
        found = self.identity(request)
        user = await found if inspect.isawaitable(found) else found
        if not user:
            raise HTTPException(401, "no user")
        return user

    async def run(self, body: RunAgentInput, user: str) -> tuple[RunBuffer, int]:
        """Start the run (or its resume) in the background; its buffer, and the number of
        the last event the caller already has."""
        agent = self.agent
        tenant = await agent.harness.tenant()
        if body.resume:
            return await self._resume(body, body.resume[-1], user, tenant)
        if body.run_id and (safe_id(body.run_id) != body.run_id or self.hub.get(body.run_id)):
            raise HTTPException(422, "runId must be a new identifier (letters, digits, -_.:)")
        payload = body.latest_user_text() or body.state
        identity = await agent._opened(
            payload, user=user, thread=body.thread_id, tenant=tenant, run_id=body.run_id
        )
        buffer = self.hub.open(identity.run_id, user)
        self._launch(
            buffer,
            body.thread_id,
            identity.run_id,
            lambda listen: pipeline.attempt(
                agent, identity, payload, listener=listen, streaming=True
            ),
        )
        return buffer, -1

    async def _resume(
        self, body: RunAgentInput, answer: Resume, user: str, tenant: str
    ) -> tuple[RunBuffer, int]:
        agent = self.agent
        record = await agent.harness.runs.get(run_of(answer.interrupt_id))
        if (
            record is None
            or record.tenant_id != tenant
            or record.thread_id != body.thread_id
            or record.awaiting is None
        ):
            raise HTTPException(404, "no such interrupt for this thread")
        try:
            decision = _decision(answer, record.awaiting.reason)
            record, resolution = await agent._resolution(
                answer.interrupt_id, decision, answer.payload, user
            )
        except (ConfigurationError, ValueError) as exc:
            raise HTTPException(409, f"{BAD_RESUME}: {exc}") from exc
        buffer = self.hub.open(record.run_id, user)
        after = buffer.published - 1
        self._launch(
            buffer,
            body.thread_id,
            record.run_id,
            lambda listen: agent._continue(record, resolution, listen),
        )
        return buffer, after

    def _launch(
        self,
        buffer: RunBuffer,
        thread_id: str,
        run_id: str,
        execute: Callable[[Callable[[RunEvent], None]], Awaitable[Result]],
    ) -> None:
        def listen(event: RunEvent) -> None:
            translated = translate(event)
            if translated is not None:
                buffer.publish(translated, final=event.type is RunEventType.RUN_FINISHED)

        async def guarded() -> None:
            try:
                result = await execute(listen)
            except Exception as exc:
                log.exception("chat run %s failed", run_id)
                result = None
                message = str(exc)
            else:
                message = "the run ended without a result"
            if not buffer.finished:
                buffer.publish(_ending(result, thread_id, run_id, message), final=True)

        task = asyncio.create_task(guarded(), name=f"agui:{run_id}")
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)


def _router(surface: _Surface, path: str) -> APIRouter:
    router = APIRouter(prefix=path, tags=["agui"])

    @router.post("/run", summary="Run the agent for a thread and stream AG-UI events")
    async def run(request: Request, body: RunAgentInput) -> StreamingResponse:
        user = await surface.user(request)
        buffer, after = await surface.run(body, user)
        return StreamingResponse(_sse(buffer, after), media_type=MEDIA_TYPE)

    @router.get("/runs/{run_id}/events", summary="Reconnect to a run's events")
    async def events(
        request: Request,
        run_id: str,
        after: int = -1,
        last_event_id: int | None = Header(default=None),
    ) -> StreamingResponse:
        user = await surface.user(request)
        buffer = surface.hub.get(run_id)
        if buffer is None or buffer.user != user:
            raise HTTPException(404, f"no run {run_id} here")
        start = last_event_id if last_event_id is not None else after
        return StreamingResponse(_sse(buffer, start), media_type=MEDIA_TYPE)

    @router.get(
        "/runs/{run_id}/artifacts/{artifact_id}",
        summary="Data the interrupt a paused run waits on carries by reference",
    )
    async def artifact(request: Request, run_id: str, artifact_id: str) -> Response:
        """The bytes of the ``payload_ref`` of the interrupt the run waits on (an ``ask``
        table or diff), from agent-runs — so any replica serves them."""
        await surface.user(request)
        agent = surface.agent
        tenant = await agent.harness.tenant()
        record = await agent.harness.runs.get(run_id)
        awaiting = record.awaiting if record is not None else None
        ref = awaiting.payload_ref if awaiting is not None else None
        if (
            record is None
            or record.tenant_id != tenant
            or record.agent_id != agent.id
            or ref is None
            or ref.artifact_id != artifact_id
        ):
            raise HTTPException(404, f"no artifact {artifact_id}")
        data = await agent.harness.runs.artifact(artifact_id, tenant)
        if data is None:
            raise HTTPException(404, f"no artifact {artifact_id}")
        return Response(data, media_type=ref.mime_type or "application/octet-stream")

    return router


async def _sse(buffer: RunBuffer, after: int) -> AsyncIterator[str]:
    async for number, event in buffer.read(after):
        yield encode(event, number)


def _decision(answer: Resume, reason: InterruptReason) -> InterruptDecision:
    """The contracts decision a protocol resume entry means: ``cancelled`` abandons the run;
    an approval is answered ``true`` (approve), ``false`` (reject) or with the edited
    arguments; anything else answers a question. The ``decision`` extension names it."""
    if answer.decision:
        return InterruptDecision(answer.decision.upper())
    if answer.status is ResumeStatus.CANCELLED:
        return InterruptDecision.CANCEL
    if reason is not InterruptReason.APPROVAL:
        return InterruptDecision.ANSWER
    if answer.payload is True:
        return InterruptDecision.APPROVE
    if answer.payload is False:
        return InterruptDecision.REJECT
    if isinstance(answer.payload, dict):
        return InterruptDecision.EDIT
    raise ValueError("an approval is answered with true, false or the edited arguments")


def _ending(result: Any, thread_id: str, run_id: str, message: str) -> AGUIEvent:
    """What a run that ended without announcing it (cancelled while paused, an error before
    its first event) tells its client."""
    if isinstance(result, Result) and result.status is RunStatus.CANCELLED:
        return AGUIEvent(
            type=AGUIEventType.RUN_FINISHED,
            thread_id=thread_id,
            run_id=run_id,
            outcome=Outcome(type=OutcomeType.CANCELLED),
        )
    return AGUIEvent(
        type=AGUIEventType.RUN_ERROR,
        thread_id=thread_id,
        run_id=run_id,
        message=message,
        code=RUN_ABORTED,
    )
