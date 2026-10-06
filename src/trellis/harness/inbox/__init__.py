"""The reference inbox: ``h.serve_inbox(app)``.

A small page, and the JSON routes it calls, for answering paused runs — a reference to start
from: teams answer from their own screens with the same routes, or with ``h.inbox`` and
``agent.resume``.

* ``GET {path}`` — the page (static HTML and JavaScript, no dependencies). It lists the paused
  runs and renders each question as asked: a choice of the ``options`` (labels, descriptions;
  several picks with ``multiple``), a form built from ``expects`` with the ``ui_schema``
  hints, a table or a diff, an approval with the call's arguments (approve, edit, reject;
  "approve for the rest of this run"), a comment, and cancel. A question naming a
  ``component`` is rendered by your screen when the page has one by that name
  (``window.trellisComponents[name](element, props, interrupt, submit)``), else as above.
* ``GET {path}/runs?assignee=`` — the paused runs of the agents wrapped by the harness
  (``h.inbox``), each with the interrupt it waits on.
* ``POST {path}/runs/{run_id}/resume`` — ``{"interrupt_id", "decision", "answer", "comment",
  "remember"}``: answered as the reviewer ``identity(request)`` names, checked first (an
  answer that does not fit is ``409`` with why, ``BAD_RESUME``), then ``202``: the run goes on
  in the background, as ``agent.resume`` continues it.

``identity(request)`` returns the reviewer, as for ``serve_chat``; without one every answer is
``anonymous``'s (a warning says so). The tenant is the one ``TRELLIS_API_KEY`` speaks for.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from importlib import resources
from typing import TYPE_CHECKING, Any, Final, Literal

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator

from trellis.contracts import ConfigurationError, InterruptDecision

if TYPE_CHECKING:
    from trellis.harness.agui import Identity
    from trellis.harness.harness import Harness

log = logging.getLogger("trellis.inbox")

#: The reviewer every answer is given as when the deployment named no ``identity``.
ANONYMOUS: Final = "anonymous"
#: The page, beside this module.
PAGE: Final = "page.html"
#: The refusal code of an answer that does not fit, or a run that cannot take it.
BAD_RESUME: Final = "BAD_RESUME"


class Answer(BaseModel):
    """An answer the page (or your screen) sends."""

    interrupt_id: str
    decision: InterruptDecision
    answer: Any = None
    comment: str | None = Field(default=None, max_length=4000)
    remember: Literal["once", "run"] = "once"

    @field_validator("decision", mode="before")
    @classmethod
    def _any_case(cls, value: Any) -> Any:
        return value.upper() if isinstance(value, str) else value


def mount(app: FastAPI, harness: Harness, *, path: str, identity: Identity | None) -> None:
    if identity is None:
        log.warning("serve_inbox has no identity=: every answer is %r's", ANONYMOUS)
    page = resources.files(__package__).joinpath(PAGE).read_text(encoding="utf-8")
    router = APIRouter(prefix=path, tags=["inbox"])

    async def reviewer(request: Request) -> str:
        if identity is None:
            return ANONYMOUS
        found = identity(request)
        user = await found if inspect.isawaitable(found) else found
        if not user:
            raise HTTPException(401, "no user")
        return user

    @router.get("", response_class=HTMLResponse, summary="The reference inbox page")
    async def show() -> HTMLResponse:
        return HTMLResponse(page)

    @router.get("/runs", summary="The paused runs of the agents wrapped here")
    async def runs(request: Request, assignee: str | None = None) -> list[dict[str, Any]]:
        await reviewer(request)
        waiting = await harness.inbox(assignee)
        return [
            {
                "run_id": s.run_id,
                "agent_id": s.agent_id,
                "assignee": s.assignee,
                "updated_at": s.updated_at.isoformat(),
                "interrupt": s.awaiting.awaiting(),
            }
            for s in waiting
            if s.agent_id in harness.agents and s.awaiting is not None
        ]

    @router.post("/runs/{run_id}/resume", summary="Answer a paused run")
    async def resume(request: Request, run_id: str, body: Answer) -> JSONResponse:
        try:
            who = await reviewer(request)
            record = await harness.runs.get(run_id, tenant=await harness.tenant())
            agent = harness.agents.get(record.agent_id) if record is not None else None
            if record is None or agent is None:
                raise HTTPException(404, f"no run {run_id} of an agent served here")
            if not body.interrupt_id.startswith(f"{run_id}."):
                raise HTTPException(404, f"{body.interrupt_id} is not a question of {run_id}")
            try:
                record, resolution = await agent._resolution(
                    body.interrupt_id,
                    body.decision,
                    body.answer,
                    who,
                    tenant=record.tenant_id,
                    comment=body.comment,
                    remember=body.remember,
                )
            except ConfigurationError as exc:
                raise HTTPException(409, f"{BAD_RESUME}: {exc}") from exc
        except HTTPException as exc:
            return JSONResponse(
                {"detail": exc.detail, "status": exc.status_code}, status_code=exc.status_code
            )
        # the run goes on in the background (a queued one goes back to the queue): the
        # reviewer is not held for as long as it works
        task = asyncio.create_task(agent._continue(record, resolution), name=f"inbox:{run_id}")
        continuing.add(task)
        task.add_done_callback(_settled)
        return JSONResponse({"run_id": run_id, "interrupt_id": resolution.interrupt_id}, 202)

    continuing: set[asyncio.Task[Any]] = set()

    def _settled(task: asyncio.Task[Any]) -> None:
        continuing.discard(task)
        if not task.cancelled() and task.exception() is not None:
            log.error("an answered run failed to continue", exc_info=task.exception())

    app.include_router(router)


__all__ = ["Answer", "mount"]
