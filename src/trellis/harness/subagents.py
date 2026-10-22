"""``agent.as_tool()``: a wrapped agent as a tool another agent calls — a sub-agent.

    research = h.wrap(ReAct(system="You research.", model=model), id="research", tools=[search])
    writer = h.wrap(ReAct(system="You write.", model=model), id="writer",
                    tools=[research.as_tool()])

Either side may be any target (ReAct, a function, LangGraph, OpenAI Agents, Claude), and any
framework may call the tool (``tools=[...]``, or ``h.tools(..., framework=)``). The model decides
when to delegate, and what to delegate at once. Calling the tool runs a *child run* of the
sub-agent, inside the call:

* its id is derived from the parent's call (the parent run, the call, its occurrence: the
  call's ``idempotency_key``), so the parent's re-run finds it again; its record names its
  parent (``parent_run_id``); it inherits the parent's tenant, user and thread, its deadline
  and what is left of its time (a child never works past its parent), and its spans are in the
  parent's trace; its agent version is its own, and so is its agent's ``without=`` (what the
  parent's run is without, it is without too);
* while it works its journal is kept in the parent's, so the parent's progress saves it: after
  a crash the parent's next attempt continues the child where it was (the tool is
  ``resumable``), repeating none of its side effects;
* a child that finished answers with its output (a re-run reads it from the child's record);
  one that failed is an error the parent's model reads; one that pauses pauses the parent with
  its question — the interrupt's payload names the child (:data:`SUBAGENT`) — and the parent's
  ``resume`` hands the answer to the child and continues both;
* cancelling the parent cancels its children (:func:`cancel_children`).

The tool only reads — and so runs beside the parent's other reads — when every tool the child
declares only reads and it has none the harness does not run (a framework's own tools, whose
effects are not known); otherwise it writes, and runs one at a time. ``side_effects=`` says
otherwise.
"""

from __future__ import annotations

import contextlib
import inspect
from typing import TYPE_CHECKING, Any, Final

from trellis.contracts import (
    Interrupt,
    InterruptResolution,
    RunRecord,
    RunStatus,
    ToolError,
    ToolSpec,
    stable_id,
)
from trellis.harness import pipeline
from trellis.harness.frameworks import BY_NAME
from trellis.harness.invariants import present
from trellis.harness.journal import Journal, content_key
from trellis.harness.result import Result
from trellis.harness.runtime import Runtime, current
from trellis.harness.tools.base import SideEffect, SideEffects, Tool
from trellis.harness.tools.mcp import side_effects_of
from trellis.harness.tools.toolbox import listed
from trellis.runs import ConflictError

if TYPE_CHECKING:
    from trellis.harness.agent import Agent
    from trellis.harness.runs import RunStore

#: The key of a parent's interrupt payload naming the child that asked:
#: ``{"agent_id", "run_id", "interrupt_id"}``.
SUBAGENT: Final = "subagent"
#: What a sub-agent's tool takes: the task, in words.
INPUT: Final = {
    "type": "object",
    "properties": {
        "message": {
            "type": "string",
            "description": "the task, with everything the agent needs to know to do it",
        }
    },
    "required": ["message"],
    "additionalProperties": False,
}
#: What a child's question carries over to its parent's interrupt: every field of the
#: contract's ``Interrupt`` but its identity, its own payload (merged apart) and its time.
ASKED: Final = tuple(
    name
    for name in Interrupt.model_fields
    if name not in {"interrupt_id", "tenant_id", "run_id", "payload", "created_at"}
)


class SubAgent:
    """A wrapped agent as a tool source: what ``Agent.as_tool`` returns."""

    def __init__(
        self,
        agent: Agent,
        *,
        name: str | None,
        description: str | None,
        side_effects: SideEffects | None,
    ) -> None:
        self.agent = agent
        self.name = name or agent.id
        self.description = description or _described(agent)
        self.side_effects = side_effects

    async def resolve(self) -> list[Tool]:
        spec = ToolSpec(
            name=self.name,
            description=self.description,
            input_schema=INPUT,
            source="local",
            side_effects=self.side_effects or await _side_effects(self.agent),
        )
        return [Tool(spec, self._run, resumable=True)]

    async def _run(self, args: dict[str, Any]) -> Any:
        """The child run for this call: started, or found again by its id and continued; its
        output, or its pause as the parent's."""
        parent = present(current(), "the bridge runs a tool inside a run")
        child = self.agent
        start = await child._start(
            args["message"],
            user=parent.user,
            thread=parent.thread,
            tenant=parent.tenant,
            run_id=stable_id(parent.idempotency_key, prefix="run_"),
            record_input=True,
            timeout=parent.remaining(),
            deadline=parent.deadline,
            parent=parent.run_id,
            without=parent.without,
        )
        result = await self._continued(parent, await child.harness.runs.start(start))
        while result.status is RunStatus.PAUSED:
            asked = present(result.interrupt, "a paused run has its interrupt")
            named = {
                "agent_id": child.id,
                "run_id": asked.run_id,
                "interrupt_id": asked.interrupt_id,
            }
            answered = await parent.interrupt(
                content_key("subagent", asked.run_id, asked.interrupt_id),
                payload={**(asked.payload or {}), SUBAGENT: named},
                **{name: getattr(asked, name) for name in ASKED},
            )
            result = await self._answered(parent, asked, answered)
        if result.status is not RunStatus.SUCCESS:
            why = f": {result.error.message}" if result.error is not None else ""
            raise ToolError(
                f"the {child.id} agent's run ended {result.status.value}{why}",
                source="tools",
                retryable=False,
            )
        return pipeline.jsonable(result.answer)

    async def _continued(self, parent: Runtime, record: RunRecord) -> Result:
        """The child as its record says: paused or ended as it is, else run — new, or cut by a
        crash (its journal is the copy the parent's progress saved)."""
        if record.status is RunStatus.PAUSED or record.final:
            return Result.of(record)
        journal = parent.replay.journal.children.get(record.run_id) or Journal()
        return await self._attempt(parent, record, journal, None)

    async def _answered(
        self, parent: Runtime, asked: Interrupt, answer: InterruptResolution
    ) -> Result:
        """The parent's answer to the child's question, the child's own: resumed with it (in
        this process: a child is never queued, and a cancel never reaches it this way — the
        parent's ends the parent, and :func:`cancel_children` the child)."""
        child = self.agent
        record = present(
            await child.harness.runs.get(asked.run_id, tenant=parent.tenant),
            "a child run asking its parent is in the run store",
        )
        resolution = answer.model_copy(
            update={"interrupt_id": asked.interrupt_id, "run_id": asked.run_id}
        )
        journal, resumed = await child._resumed(record, resolution)
        return await self._attempt(parent, resumed, journal, resolution)

    async def _attempt(
        self,
        parent: Runtime,
        record: RunRecord,
        journal: Journal,
        resolution: InterruptResolution | None,
    ) -> Result:
        """One attempt of the child inside the parent's call, its journal kept in the
        parent's meanwhile, its time what is left of the parent's."""
        child = self.agent
        children = parent.replay.journal.children
        children[record.run_id] = journal
        try:
            return await pipeline.attempt(
                child,
                record,
                record.input,
                journal=journal,
                resolution=resolution,
                remaining=parent.remaining(),
                parent=parent,
            )
        finally:
            del children[record.run_id]


def asked_by(interrupt: Interrupt) -> dict[str, Any] | None:
    """The child a parent's interrupt carries the question of, when it does."""
    named = (interrupt.payload or {}).get(SUBAGENT)
    return named if isinstance(named, dict) else None


async def cancel_children(runs: RunStore, run_id: str, *, reason: str | None, tenant: str) -> None:
    """Cancel the children of ``run_id`` that have not ended, and theirs: a child paused for a
    person waits on nobody once its parent is cancelled."""
    async for child in runs.iterate(parent_run_id=run_id, tenant=tenant):
        if child.status.final:
            continue
        with contextlib.suppress(ConflictError):  # it ended meanwhile
            await runs.cancel(child.run_id, reason=reason, tenant=tenant)
        await cancel_children(runs, child.run_id, reason=reason, tenant=tenant)


async def _side_effects(agent: Agent) -> SideEffects:
    """``read`` when every tool ``agent`` declares (its own, and the MCP tools its key allows)
    only reads and none escapes the harness; else ``write``."""
    if _unseen(agent):
        return "write"
    listing = await listed(agent.sources, gateway=agent.harness.gateway)
    effects = [t.spec.side_effects for t in listing.local]
    effects.extend(side_effects_of(d.annotations) for d in listing.defs)
    return "read" if all(effect == SideEffect.READ for effect in effects) else "write"


def _unseen(agent: Agent) -> bool:
    """Whether the target has tools of its framework's own, which the harness does not run:
    Claude's built-in tools, an OpenAI Agents agent's own tools or handoffs, a graph's tools
    not built with ``h.tools``."""
    return BY_NAME[agent.adapter.name].own_tools(agent.target)


def _described(agent: Agent) -> str:
    """A function target's docstring (its first paragraph), else what the tool is for."""
    target = agent.target
    doc = inspect.getdoc(target) if inspect.isfunction(target) else None
    first = (doc or "").split("\n\n")[0]
    return first or (
        f"Ask the {agent.id} agent to do a task: give it the task in `message`; it answers "
        "when it is done, or asks a person first."
    )
