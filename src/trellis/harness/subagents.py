"""A wrapped agent as a tool another agent calls — a sub-agent:

    research = h.wrap(ReAct("You research.", model, tools=[search]), id="research")
    writer = h.wrap(ReAct("You write.", model, subagents=[research]), id="writer")

Either side may be any target (ReAct, a function, LangGraph, OpenAI Agents, Claude): a ``ReAct``
names its sub-agents (``subagents=``), and any agent's code calls another with ``agent.run``,
which is the same child run (:func:`child_run`). The model decides when to delegate, and what
to delegate at once. Calling the tool runs a *child run* of the sub-agent, inside the call:

* its id is derived from the parent's call (the parent run, the call, its occurrence: the
  call's ``idempotency_key``), so the parent's re-run finds it again; its record names its
  parent (``parent_run_id``), and its memory scope is derived from the parent's
  (``MemoryContext.agent``): it reads the parent's hand-off notes (``RUN`` visibility), its
  own go up to the parent, and the runs of one delegation tree share an agent group, named
  by the run at its top (``memory_remember(scope="group")``); it inherits the parent's
  tenant, user and thread, the documents its context draws on (``document_ids=``), its
  deadline and what is left of its time (a child never works past its parent), and its spans
  are in the parent's trace; its agent version is its own, and so is its agent's
  ``without=`` (what the parent's run is without, it is without too);
* while it works its journal is kept in the parent's, so the parent's progress saves it: after
  a crash the parent's next attempt continues the child where it was (the tool is
  ``resumable``), repeating none of its side effects;
* a child that finished answers with its output (a re-run reads it from the child's record);
  one that failed is an error the parent's model reads; one that pauses pauses the parent with
  its question — the interrupt's payload names the child (:data:`SUBAGENT`) — and the parent's
  ``resume`` hands the answer to the child and continues both;
* cancelling the parent cancels its children (:func:`cancel_children`).

``agent.run(...)`` called inside a run — from a tool, a node, a function target's code — is the
same child run (:func:`child_run`): its id derived from where the parent called it (the tool
call, else the run) and how many runs of that agent it started there.

The tool only reads — and so runs beside the parent's other reads — when every tool the child
declares only reads and it has none the harness does not run (a framework's own tools, whose
effects are not known); otherwise it writes, and runs one at a time. Its name is the agent's
id, its description the agent's (``h.wrap(description=)``, else its function's docstring).
"""

from __future__ import annotations

import contextlib
import inspect
from collections.abc import Collection, Mapping, Sequence
from datetime import datetime
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
from trellis.harness import admission, pipeline
from trellis.harness.adapters.langgraph import bound_tools
from trellis.harness.admission import Branch
from trellis.harness.journal import Journal, content_key
from trellis.harness.result import Result
from trellis.harness.runtime import Runtime, current
from trellis.harness.tools.base import SideEffects, Tool
from trellis.harness.tools.mcp import side_effects_of
from trellis.harness.tools.toolbox import listed
from trellis.runs import ConflictError

if TYPE_CHECKING:
    from trellis.harness.agent import Agent
    from trellis.harness.budget import Budget
    from trellis.harness.runs import RunStore

#: The key of a parent's interrupt payload naming the child that asked:
#: ``{"agent_id", "run_id", "interrupt_id"}``.
SUBAGENT: Final = "subagent"
#: What a nested run's id is derived from, besides where its parent is (``child_run``).
CHILD: Final = "child"
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
#: What a child's question carries over to its parent's interrupt.
ASKED: Final = (
    "reason",
    "question",
    "ui",
    "expects",
    "options",
    "multiple",
    "ui_schema",
    "component",
    "props",
    "payload_ref",
    "tool_call",
    "assignee",
    "deadline",
    "escalate_to",
)


class SubAgent:
    """A wrapped agent as a tool source (a ``ReAct``'s ``subagents=``)."""

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
        """The child run for this call (:func:`child_run`, its id the call's): its output, or
        its failure as the tool's error the parent's model reads."""
        parent = current()
        assert parent is not None  # the bridge runs a tool inside a run
        child = self.agent
        result = await child_run(
            child,
            parent,
            args["message"],
            user=parent.user,
            run_id=stable_id(parent.idempotency_key, prefix="run_"),
        )
        if result.status is not RunStatus.SUCCESS:
            why = f": {result.error.message}" if result.error is not None else ""
            raise ToolError(
                f"the {child.id} agent's run ended {result.status.value}{why}",
                source="tools",
                retryable=False,
            )
        return pipeline.jsonable(result.answer)


async def child_run(
    agent: Agent,
    parent: Runtime,
    input: Any,
    *,
    user: str,
    run_id: str | None = None,
    thread: str | None = None,
    tenant: str | None = None,
    timeout: float | None = None,  # noqa: ASYNC109 - the run's limit, kept across attempts
    deadline: datetime | None = None,
    budget: Budget | None = None,
    without: Collection[str] = (),
    framework_options: Mapping[str, Any] | None = None,
    document_ids: Sequence[str] | None = None,
) -> Result:
    """A run of ``agent`` inside ``parent``'s — a sub-agent's call, or ``agent.run`` called
    from the parent's code — as the parent's child: started (its record names the parent and
    takes what it inherits: ``Agent._start``), or found again by its id and continued; it works
    at most what is left of the parent's time (and its own ``timeout``); a question it asks
    pauses the parent, and the parent's answer resumes it. ``run_id``: the call's, else one
    derived from where the parent is (its tool call, or its run) and how many children of this
    agent it started there, so the parent's re-run finds each again. Its result, however it
    ended.

    It is one of the parent's children at once (``admission.branch``: it waits for a slot
    under the adaptive limit, and is on the parent's task tree)."""
    if run_id is None:
        place = parent.idempotency_key or parent.run_id
        run_id = stable_id(
            place, CHILD, agent.id, parent.child_number(place, agent.id), prefix="run_"
        )
    async with admission.branch(parent, run_id, agent.id) as branch:
        left = parent.remaining()
        own = agent.timeout if timeout is None else timeout  # the call's, else the agent's
        limits = [t for t in (own, left) if t is not None]
        start = await agent._start(
            input,
            user=user,
            thread=thread,
            tenant=tenant,
            run_id=run_id,
            record_input=True,
            timeout=min(limits) if limits else None,
            deadline=deadline,
            budget=budget,
            parent=parent,
            without=without,
            framework_options=framework_options,
            document_ids=document_ids,
        )
        result = await _continued(agent, parent, await agent.harness.runs.start(start), branch)
        while result.status is RunStatus.PAUSED:
            assert result.interrupt is not None
            asked = result.interrupt
            named = {
                "agent_id": agent.id,
                "run_id": asked.run_id,
                "interrupt_id": asked.interrupt_id,
            }
            answered = await parent.interrupt(
                content_key("subagent", asked.run_id, asked.interrupt_id),
                payload={**(asked.payload or {}), SUBAGENT: named},
                **{name: getattr(asked, name) for name in ASKED},
            )
            result = await _answered(agent, parent, asked, answered, branch)
        branch.ended(result.status.value.lower(), error=_why(result))
        return result


def _why(result: Result) -> str | None:
    return result.error.message if result.error is not None else None


async def _continued(agent: Agent, parent: Runtime, record: RunRecord, branch: Branch) -> Result:
    """The child as its record says: paused or ended as it is, else run — new, or cut by a
    crash (its journal is the copy the parent's progress saved)."""
    if record.status is RunStatus.PAUSED or record.final:
        return Result.of(record)
    journal = parent.replay.journal.children.get(record.run_id) or Journal()
    return await _attempt(agent, parent, record, journal, None, branch=branch)


async def _answered(
    agent: Agent,
    parent: Runtime,
    asked: Interrupt,
    answer: InterruptResolution,
    branch: Branch,
) -> Result:
    """The parent's answer to the child's question, the child's own: resumed with it (in this
    process: a child is never queued, and a cancel never reaches it this way — the parent's
    ends the parent, and :func:`cancel_children` the child)."""
    record = await agent.harness.runs.get(asked.run_id, tenant=parent.tenant)
    assert record is not None
    resolution = answer.model_copy(
        update={"interrupt_id": asked.interrupt_id, "run_id": asked.run_id}
    )
    journal, resumed = await agent._resumed(record, resolution)
    return await _attempt(agent, parent, resumed, journal, resolution, branch=branch)


async def _attempt(
    agent: Agent,
    parent: Runtime,
    record: RunRecord,
    journal: Journal,
    resolution: InterruptResolution | None,
    *,
    branch: Branch,
) -> Result:
    """One attempt of the child inside the parent's call, its journal kept in the parent's
    meanwhile, its time what is left of the parent's; what it has spent on its ``branch``."""
    children = parent.replay.journal.children
    children[record.run_id] = journal
    try:
        return await pipeline.attempt(
            agent,
            record,
            record.input,
            journal=journal,
            resolution=resolution,
            remaining=_left(parent, record),
            parent=parent,
        )
    finally:
        del children[record.run_id]
        branch.spent = dict(journal.spent)


def _left(parent: Runtime, record: RunRecord) -> float | None:
    """What the child may still work: what is left of the parent's time, and of its own limit
    (``None``: neither limits it)."""
    own = record.timeout_seconds
    limits = [parent.remaining(), None if own is None else own - record.worked_seconds]
    found = [t for t in limits if t is not None]
    return max(0.0, min(found)) if found else None


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
    return "read" if all(effect == "read" for effect in effects) else "write"


def _unseen(agent: Agent) -> bool:
    """Whether the target has tools of its framework's own, whose effects the harness does not
    know: a function target's (its code's calls), Claude's built-in tools, an OpenAI Agents
    agent's own tools or handoffs, a graph's own tools (a ``ReAct``'s are only the harness's
    own read tools)."""
    from trellis.harness.middleware import STATE_TOOLS  # noqa: PLC0415 - react

    target, framework = agent.target, agent.adapter.name
    if framework == "function":  # its code calls what it calls (``agent.tools.call``)
        return not agent.sources
    if framework == "claude_agent_sdk":
        return True
    if framework == "openai_agents":
        return bool(target.tools or target.handoffs)
    harness_own = STATE_TOOLS
    return any(getattr(t, "name", None) not in harness_own for t in bound_tools(target))


def _described(agent: Agent) -> str:
    """What the agent does: its own description (``h.wrap(description=)``), else a function
    target's docstring (its first paragraph), else what the tool is for."""
    if agent.description:
        return agent.description
    target = agent.target
    doc = inspect.getdoc(target) if inspect.isfunction(target) else None
    first = (doc or "").split("\n\n")[0]
    return first or (
        f"Ask the {agent.id} agent to do a task: give it the task in `message`; it answers "
        "when it is done, or asks a person first."
    )
