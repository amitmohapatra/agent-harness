"""The AG-UI chat surface: ``agent.serve_chat(app)``.

* ``POST {path}/run`` — run the agent for a thread (or answer the interrupt a paused run of
  it waits on, with a ``resume`` entry) and stream its AG-UI events as SSE. The run executes
  in the background: a client that goes away does not stop it.
* ``GET {path}/runs/{run_id}/events`` — reconnect: the run's events after ``Last-Event-ID``
  (or ``?after=``), then live until it finishes. A run this process did not serve is read
  from agent-runs' event log when ``RUNS_URL`` is set (its ids are then the log's positions):
  any replica serves any run.
* ``GET {path}/runs/{run_id}/artifacts/{artifact_id}`` — data the interrupt a paused run
  waits on carries by reference (``payload_ref``: a large ``ask`` table or diff), read from
  agent-runs.

Identity is the deployment's: ``identity(request)`` returns the user, and the tenant is the
one ``TRELLIS_API_KEY`` speaks for. Nothing the client sends (``forwardedProps``...) decides
who is calling.

The routes are in the app's OpenAPI document (tag ``agui``): the streams as
``text/event-stream``, every refusal as an RFC 9457 problem document (``code``, ``detail``).
Each run's events are buffered in this process (``hub.py``); with agent-runs (``RUNS_URL``)
they are also in the run's log there, so a reconnect that reaches another replica reads them
from it — from the position its ``Last-Event-ID`` names (an id another replica numbered is no
larger than the event's position: a failover may repeat events, never skip one). Without
agent-runs a reconnect must reach the replica that served the run (sticky sessions).

A start or resume agent-runs refuses for the tenant's rate limit, after its SDK's retries, is
``429`` (``RATE_LIMIT``) with ``Retry-After``.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import math
from collections.abc import AsyncIterator, Awaitable, Callable
from importlib import metadata
from typing import TYPE_CHECKING, Any, Final

from fastapi import APIRouter, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

from trellis.contracts import (
    ConfigurationError,
    Interrupt,
    InterruptDecision,
    InterruptReason,
    RunEvent,
    RunEventType,
    RunStatus,
    safe_id,
)
from trellis.harness import pipeline
from trellis.harness.agent import Throttled
from trellis.harness.agui.events import (
    AGUIEvent,
    AGUIEventType,
    Outcome,
    OutcomeType,
    Resume,
    ResumeStatus,
    RunAgentInput,
)
from trellis.harness.agui.hub import Hub, RunBuffer
from trellis.harness.agui.sse import MEDIA_TYPE, encode
from trellis.harness.agui.translate import translate
from trellis.harness.result import Result
from trellis.harness.runlog import event_log
from trellis.harness.runtime import run_of

if TYPE_CHECKING:
    from trellis.harness.agent import Agent

log = logging.getLogger("trellis.agui")

Identity = Callable[[Request], str | Awaitable[str]]
#: The user every request runs as when the deployment named no ``identity``.
ANONYMOUS: Final = "anonymous"
#: ``RUN_ERROR`` codes the surface itself emits.
BAD_RESUME: Final = "BAD_RESUME"
RUN_ABORTED: Final = "RUN_ABORTED"
#: The platform's problem code of each refusal the surface answers.
PROBLEM_CODES: Final = {
    401: "AUTHENTICATION",
    404: "NOT_FOUND",
    409: "CONFLICT",
    422: "VALIDATION",
    429: "RATE_LIMIT",
}
PROBLEM_MEDIA_TYPE: Final = "application/problem+json"
#: What the surface tells an app's OpenAPI document about itself.
TAG: Final = {
    "name": "agui",
    "description": "AG-UI chat surface (serve_chat): run an agent for a thread and stream its "
    "AG-UI events as server-sent events, reconnect to a run, read interrupt artifacts. Run "
    "events are buffered per process: with several replicas, keep a thread on one (sticky "
    "sessions).",
    "externalDocs": {"description": "AG-UI protocol", "url": "https://docs.ag-ui.com"},
}


class Problem(BaseModel):
    """A refusal, as an RFC 9457 problem document (the platform's shape)."""

    type: str
    title: str
    status: int
    detail: str
    instance: str
    code: str
    retryable: bool = False


class EventStream(StreamingResponse):
    """A stream of AG-UI events as server-sent events."""

    media_type = MEDIA_TYPE


_SSE_DOC: Final = {
    "description": "AG-UI events as server-sent events: one `data:` JSON event per message, "
    "each with an `id:` numbered per run (send it back as `Last-Event-ID` to reconnect), "
    "ending with RUN_FINISHED or RUN_ERROR.",
    "content": {MEDIA_TYPE: {"schema": {"type": "string"}}},
}


def _refusals(*statuses: int) -> dict[int | str, dict[str, Any]]:
    why = {
        401: "the deployment's identity(request) named nobody",
        404: "no such run, interrupt or artifact for this caller",
        409: "a resume that cannot be read as a decision, or that the run refuses (BAD_RESUME)",
        422: "a runId that is not a fresh identifier (a problem document), or a body that is "
        "not a RunAgentInput (FastAPI's validation error)",
        429: "agent-runs is rate limiting the tenant: try again after Retry-After seconds",
    }
    problem = {PROBLEM_MEDIA_TYPE: {"schema": Problem.model_json_schema()}}
    validation = {
        "application/json": {"schema": {"$ref": "#/components/schemas/HTTPValidationError"}}
    }
    return {
        status: {
            "description": why[status],
            "content": problem | (validation if status == 422 else {}),
        }
        for status in statuses
    }


def mount(app: FastAPI, agent: Agent, *, path: str, identity: Identity | None) -> None:
    if identity is None:
        log.warning(
            "serve_chat for %s has no identity=: every request runs as user %r",
            agent.id,
            ANONYMOUS,
        )
    app.include_router(_router(_Surface(agent, identity), path))
    _describe(app, agent)


def _describe(app: FastAPI, agent: Agent) -> None:
    """The surface in the app's OpenAPI document: the ``agui`` tag, and a title and
    description for an app that has not named itself."""
    tags = app.openapi_tags or []
    if not any(t.get("name") == TAG["name"] for t in tags):
        app.openapi_tags = [*tags, TAG]
    if app.title == "FastAPI":  # FastAPI's default: nobody named this app
        app.title = f"{agent.id} agent"
        app.description = app.description or (
            f"The {agent.id} agent, served by trellis-harness: AG-UI chat (tag `agui`)."
        )
        if app.version == "0.1.0":  # FastAPI's default too
            app.version = _version()
    app.openapi_schema = None  # built again, with the routes just added


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
        record = await agent._opened(
            payload, user=user, thread=body.thread_id, tenant=tenant, run_id=body.run_id
        )
        buffer = self.hub.open(record.run_id, user)
        self._launch(
            buffer,
            body.thread_id,
            record.run_id,
            lambda listen: pipeline.attempt(agent, record, payload, listener=listen),
        )
        return buffer, -1

    async def _resume(
        self, body: RunAgentInput, answer: Resume, user: str, tenant: str
    ) -> tuple[RunBuffer, int]:
        agent = self.agent
        record = await agent.harness.runs.get(run_of(answer.interrupt_id), tenant=tenant)
        if record is None or record.thread_id != body.thread_id or record.awaiting is None:
            raise HTTPException(404, "no such interrupt for this thread")
        try:
            decision = _decision(answer, record.awaiting)
            record, resolution = await agent._resolution(
                answer.interrupt_id,
                decision,
                answer.payload,
                user,
                tenant=tenant,
                comment=answer.comment,
                remember=answer.remember,
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

    async def artifact(self, request: Request, run_id: str, artifact_id: str) -> Response:
        await self.user(request)
        agent = self.agent
        tenant = await agent.harness.tenant()
        record = await agent.harness.runs.get(run_id, tenant=tenant)
        awaiting = record.awaiting if record is not None else None
        ref = awaiting.payload_ref if awaiting is not None else None
        if (
            record is None
            or record.agent_id != agent.id
            or ref is None
            or ref.artifact_id != artifact_id
        ):
            raise HTTPException(404, f"no artifact {artifact_id}")
        data = await agent.harness.runs.artifacts.download(artifact_id, tenant=tenant)
        if data is None:
            raise HTTPException(404, f"no artifact {artifact_id}")
        return Response(data, media_type=ref.mime_type or "application/octet-stream")

    async def logged(self, run_id: str, user: str, after: int) -> AsyncIterator[str]:
        """A run this process did not serve, from agent-runs' event log: its events past
        position ``after`` as AG-UI events numbered by position, live until it ends. Only the
        run's own user; 404 without agent-runs, or for anyone else's run."""
        agent = self.agent
        store = event_log(agent.harness.runs)
        if store is None:
            raise HTTPException(404, f"no run {run_id} here")
        tenant = await agent.harness.tenant()
        record = await agent.harness.runs.get(run_id, tenant=tenant)
        if record is None or record.agent_id != agent.id or record.user_id != user:
            raise HTTPException(404, f"no run {run_id}")
        entries = store.stream_events(run_id, after=max(after, 0), tenant=tenant)

        async def read() -> AsyncIterator[str]:
            async for entry in entries:
                translated = translate(entry.event)
                if translated is not None:
                    yield encode(translated, entry.position)

        return read()

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


def _version() -> str:
    try:
        return metadata.version("trellis-harness")
    except metadata.PackageNotFoundError:  # a source tree never installed
        return "0"


def _router(surface: _Surface, path: str) -> APIRouter:
    router = APIRouter(prefix=path, tags=["agui"])

    @router.post(
        "/run",
        summary="Run the agent for a thread and stream AG-UI events",
        response_class=EventStream,
        responses={200: _SSE_DOC, **_refusals(401, 404, 409, 422, 429)},
    )
    async def run(request: Request, body: RunAgentInput) -> Response:
        """A new run (the client's `runId`, or one the harness names) executes in the
        background and its events stream back; a `resume` entry answers the interrupt a paused
        run of the thread waits on instead."""
        try:
            user = await surface.user(request)
            buffer, after = await surface.run(body, user)
        except HTTPException as exc:
            return _problem(exc, request)
        except Throttled as exc:
            wait = {"Retry-After": f"{math.ceil(exc.retry_after or 1)}"}
            return _problem(HTTPException(429, str(exc), headers=wait), request)
        return EventStream(_sse(buffer, after))

    @router.get(
        "/runs/{run_id}/events",
        summary="Reconnect to a run's events",
        response_class=EventStream,
        responses={200: _SSE_DOC, **_refusals(401, 404)},
    )
    async def events(
        request: Request,
        run_id: str,
        after: int = -1,
        last_event_id: int | None = Header(default=None),
    ) -> Response:
        """The run's events after `Last-Event-ID` (or `?after=`), then live until it
        finishes. Only the run's own user, on the replica that served it."""
        start = last_event_id if last_event_id is not None else after
        try:
            user = await surface.user(request)
            buffer = surface.hub.get(run_id)
            if buffer is None:
                return EventStream(await surface.logged(run_id, user, start))
            if buffer.user != user:
                raise HTTPException(404, f"no run {run_id} here")
        except HTTPException as exc:
            return _problem(exc, request)
        return EventStream(_sse(buffer, start))

    @router.get(
        "/runs/{run_id}/artifacts/{artifact_id}",
        summary="Data the interrupt a paused run waits on carries by reference",
        response_class=Response,
        responses={
            200: {
                "description": "the artifact's bytes, as the media type it was stored with",
                "content": {"application/json": {}, "application/octet-stream": {}},
            },
            **_refusals(401, 404),
        },
    )
    async def artifact(request: Request, run_id: str, artifact_id: str) -> Response:
        """The bytes of the `payload_ref` of the interrupt the run waits on (an `ask`
        table or diff), from agent-runs — so any replica serves them."""
        try:
            return await surface.artifact(request, run_id, artifact_id)
        except HTTPException as exc:
            return _problem(exc, request)

    return router


def _problem(exc: HTTPException, request: Request) -> JSONResponse:
    code = PROBLEM_CODES.get(exc.status_code, "INTERNAL")
    body = Problem(
        type=f"urn:trellis:problem:{code.lower()}",
        title=code.capitalize(),
        status=exc.status_code,
        detail=str(exc.detail),
        instance=request.url.path,
        code=code,
        retryable=exc.status_code == 429,
    )
    return JSONResponse(
        body.model_dump(),
        status_code=exc.status_code,
        media_type=PROBLEM_MEDIA_TYPE,
        headers=exc.headers,
    )


async def _sse(buffer: RunBuffer, after: int) -> AsyncIterator[str]:
    async for number, event in buffer.read(after):
        yield encode(event, number)


def _decision(answer: Resume, awaiting: Interrupt) -> InterruptDecision:
    """The contracts decision a protocol resume entry means: ``cancelled`` abandons the run;
    an approval is answered ``true`` (approve), ``false`` (reject) or with the edited
    arguments; anything else answers a question (an external tool's result too). The
    ``decision`` extension names it."""
    if answer.decision is not None:
        return answer.decision
    if answer.status is ResumeStatus.CANCELLED:
        return InterruptDecision.CANCEL
    if awaiting.reason is not InterruptReason.APPROVAL:
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
