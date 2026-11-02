"""Sub-agents that work in the background while the conversation goes on:

    research = h.wrap(ReAct("You research.", model, tools=[search]), id="research")
    lead = h.wrap(ReAct("You lead.", model, tools=[background(research)]), id="lead")
    h.worker([research])  # the tasks are queued runs: a worker works them

``background(*agents)`` is a tool source of five tools, Deep Agents' own async sub-agent tools
(``deepagents.middleware.async_subagents``: their names, arguments and usage notes), each
backed by what the harness already has, so every adapter gets them the same way:

* ``start_async_task`` — the sub-agent's run, queued (``agent.start`` inside the run: a child
  run naming its parent, in its tenant, user and thread), and its id returned at once: the
  parent's answer does not wait for it, and a worker works it (a crash continues it there);
* ``check_async_task`` / ``list_async_tasks`` — where it is, from its record in the run store;
* ``update_async_task`` — an instruction given to it at its framework's next step: the hook
  ``mode="steer"`` uses (``Adapter.steers``, ``Runtime.steered``), reached by its run id
  (``Turns.give``);
* ``cancel_async_task`` — ``agent.cancel``, its own children with it.

A task is its user's, in their tenant: a tool given its id finds it from any of the user's
chats, and another user's or another tenant's task is "no task". Its record keeps the
conversation that started it (:data:`BACKGROUND`: the tenant's, the user's and the thread's
key), which lists it and is told of it. When a task ends its answer reaches that
conversation: given to the run working in it at that run's next step (:func:`ended`), and told
to each later turn as it starts (:func:`note`).
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Final

from trellis.contracts import (
    ConfigurationError,
    RunEvent,
    RunEventType,
    RunOutcome,
    RunRecord,
    RunStatus,
    ToolSpec,
    stable_id,
)
from trellis.harness.conversation import Steer
from trellis.harness.runtime import Runtime, current
from trellis.harness.tools.base import Tool
from trellis.runs import ConflictError

if TYPE_CHECKING:
    from trellis.harness.agent import Agent
    from trellis.harness.result import Result

#: The ``RunStart.metadata`` key a background task keeps the conversation that started it under.
BACKGROUND: Final = "background"
#: The most tasks a listing (and a turn's note) reads, newest first.
LISTED: Final = 20
#: The most characters of a task's answer a listing or a note gives (``check_async_task``
#: gives it whole).
ANSWER_CHARS: Final = 2000
#: The statuses ``list_async_tasks(status_filter=)`` takes (Deep Agents' own), and the run
#: statuses each stands for.
FILTERS: Final[dict[str, frozenset[RunStatus]]] = {
    "running": frozenset({RunStatus.QUEUED, RunStatus.RUNNING, RunStatus.PAUSED}),
    "success": frozenset({RunStatus.SUCCESS, RunStatus.PARTIAL}),
    "error": frozenset({RunStatus.ERROR, RunStatus.TIMEOUT, RunStatus.REJECTED}),
    "cancelled": frozenset({RunStatus.CANCELLED}),
}
#: What the start tool says of itself: Deep Agents' ``ASYNC_TASK_TOOL_DESCRIPTION``.
START: Final = (
    "Start an async subagent. The subagent runs in the background and returns a task ID "
    "immediately.\n\nAvailable async agent types:\n{available_agents}\n\n## Usage notes:\n"
    "1. This tool launches a background task and returns immediately with a task ID. Report "
    "the task ID to the user and stop — do NOT immediately check status.\n"
    "2. Use `check_async_task` only when the user asks for a status update or result.\n"
    "3. Use `update_async_task` to send new instructions to a running task.\n"
    "4. Multiple async subagents can run concurrently — launch several and let them run in "
    "the background.\n"
    "5. A task's result is also told to you when it finishes."
)
TASK_ID: Final = {
    "type": "string",
    "description": "The exact task_id string returned by start_async_task. Pass it verbatim.",
}
#: The ``CUSTOM`` event each state of a task is (AG-UI's ``CUSTOM``, an A2A working update):
#: ``{"task_id", "subagent_type", "status", ...}`` — ``queued`` as it starts, ``running`` as it
#: works (``progress``: the step it finished), then how it ended (``success`` with its
#: ``result``, ``error``, ``timeout``, ``cancelled``, ``paused`` asking a person).
TASK_EVENT: Final = "background_task"
#: A task's run's ending as its event says it.
ENDINGS: Final = {RunOutcome.INTERRUPT: "paused"}
#: The heading of what a turn is told of its conversation's tasks (:func:`note`).
NOTE: Final = "## Background tasks of this conversation"
#: What the run working in a conversation reads when one of its tasks ends (:func:`ended`).
FINISHED: Final = "[background task {task_id} ({agent}) ended {status}]\n{answer}"


class Background:
    """``background(*agents)``: the agents as background tasks of the runs that have this source."""

    def __init__(self, agents: Sequence[Agent]) -> None:
        if not agents:
            raise ConfigurationError(
                "background() takes the agents it starts: background(research)"
            )
        self.agents = {agent.id: agent for agent in agents}

    async def resolve(self) -> list[Tool]:
        from trellis.harness.subagents import _described, _side_effects  # noqa: PLC0415

        effects = [await _side_effects(agent) for agent in self.agents.values()]
        acts = "read" if all(e == "read" for e in effects) else "write"
        kinds = "\n".join(f"- {name}: {_described(a)}" for name, a in self.agents.items())
        spec = [
            (
                "start_async_task",
                START.format(available_agents=kinds),
                {
                    "description": {
                        "type": "string",
                        "description": "A detailed description of the task for the async "
                        "subagent to perform.",
                    },
                    "subagent_type": {
                        "type": "string",
                        "enum": list(self.agents),
                        "description": "The type of async subagent to use. Must be one of the "
                        "available types listed in the tool description.",
                    },
                },
                acts,
                self._start,
            ),
            (
                "check_async_task",
                "Check the status of an async subagent task. Returns its status and, once it "
                "finished, its result (or why it failed, or the question it waits on).",
                {"task_id": TASK_ID},
                "read",
                self._check,
            ),
            (
                "update_async_task",
                "Send updated instructions to a running async subagent: it reads them at its "
                "next step, its work so far kept. The task_id remains the same.",
                {
                    "task_id": TASK_ID,
                    "message": {
                        "type": "string",
                        "description": "Follow-up instructions or context to send to the subagent.",
                    },
                },
                acts,
                self._update,
            ),
            (
                "cancel_async_task",
                "Cancel a running async subagent task: it stops at once.",
                {"task_id": TASK_ID},
                "write",
                self._cancel,
            ),
            (
                "list_async_tasks",
                "List the async subagent tasks of this conversation with their statuses: the "
                "way to find a task_id again, or how all of them are doing.",
                {
                    "status_filter": {
                        "type": "string",
                        "enum": [*FILTERS, "all"],
                        "description": "Filter tasks by status. One of: 'running', 'success', "
                        "'error', 'cancelled', 'all'. Defaults to 'all'.",
                    }
                },
                "read",
                self._list,
            ),
        ]
        return [
            Tool(
                ToolSpec(
                    name=name,
                    description=description,
                    input_schema={
                        "type": "object",
                        "properties": properties,
                        "required": [p for p in properties if p != "status_filter"],
                        "additionalProperties": False,
                    },
                    source="local",
                    side_effects=side_effects,
                ),
                run,
            )
            for name, description, properties, side_effects, run in spec
        ]

    # ------------------------------------------------------------------ the tools
    async def _start(self, args: dict[str, Any]) -> str:
        """The sub-agent's run, queued as this call's child (its id the call's, so the parent's
        re-run after a crash finds it again rather than starting a second one)."""
        from trellis.harness.agent import admitted  # noqa: PLC0415 - the agent imports the sources

        runtime = _runtime()
        agent = self.agents.get(args["subagent_type"])
        if agent is None:
            known = ", ".join(self.agents)
            return f"Unknown subagent_type {args['subagent_type']!r}: one of {known}"
        start = await agent._start(
            args["description"],
            user=runtime.user,
            thread=None,
            tenant=None,
            run_id=stable_id(runtime.idempotency_key, prefix="run_"),
            parent=runtime,
        )
        start = start.model_copy(
            update={"metadata": {**start.metadata, BACKGROUND: runtime.conversation}}
        )
        await admitted(agent.harness.runs.start(start, queue=True), "start")
        said = {"task_id": start.run_id, "subagent_type": agent.id}
        runtime.events.custom(TASK_EVENT, **said, status="queued")
        return f"Launched async subagent. task_id: {start.run_id}"

    async def _check(self, args: dict[str, Any]) -> str:
        found = await self._owned(_runtime(), args["task_id"])
        if isinstance(found, str):
            return found
        return json.dumps(described(found, whole=True), default=str)

    async def _update(self, args: dict[str, Any]) -> str:
        """The instruction given to the task at its framework's next step (``Turns.give``):
        now, when it works in this process; when it starts, while it is queued."""
        runtime = _runtime()
        found = await self._owned(runtime, args["task_id"])
        if isinstance(found, str):
            return found
        if found.status.final:
            return f"Task {found.run_id} already ended {found.status.value}: nothing to update."
        agent = self.agents[found.agent_id]
        if not agent.adapter.steers(agent.target):
            return (
                f"The {agent.id} agent takes no instruction while it works (its framework has "
                "no step the harness reaches): cancel the task and start it again with the "
                "new instructions."
            )
        if found.status is RunStatus.RUNNING and found.run_id not in agent.running:
            return (
                f"Task {found.run_id} works on another worker, which this instruction cannot "
                "reach: cancel the task and start it again with the new instructions."
            )
        # sent as this call (its message's id is its own: ``conversation.question_id``)
        sent_as = runtime.idempotency_key or runtime.run_id
        agent.turns.give(found.run_id, Steer(args["message"], sent_as))
        return f"Updated async subagent. task_id: {found.run_id}"

    async def _cancel(self, args: dict[str, Any]) -> str:
        runtime = _runtime()
        found = await self._owned(runtime, args["task_id"])
        if isinstance(found, str):
            return found
        agent = self.agents[found.agent_id]
        try:
            await agent.cancel(
                found.run_id, reason=f"cancelled by {runtime.user}", tenant=runtime.tenant
            )
        except ConflictError:
            return f"Task {found.run_id} already ended."
        agent.turns.given.pop(found.run_id, None)
        return f"Cancelled async subagent task: {found.run_id}"

    async def _list(self, args: dict[str, Any]) -> str:
        wanted = FILTERS.get(args.get("status_filter") or "all")
        tasks = [
            described(r)
            for r in await tasks_of(_runtime(), self.agents)
            if wanted is None or r.status in wanted
        ]
        return json.dumps(tasks, default=str) if tasks else "No async subagent tasks."

    async def _owned(self, runtime: Runtime, task_id: str) -> RunRecord | str:
        """The task ``task_id`` when it is one of this source's agents' and the run's user's,
        in its tenant (the run store reads the tenant's runs only), from whichever of their
        chats; else what the model reads (the same for a task that is not there and one that
        is someone else's)."""
        record = await runtime.agent.harness.runs.get(task_id, tenant=runtime.tenant)
        if (
            record is None
            or record.user_id != runtime.user
            or record.agent_id not in self.agents
            or BACKGROUND not in record.metadata
        ):
            return f"No async subagent task {task_id!r} of yours."
        return record


def background(*agents: Agent) -> Background:
    """``agents`` as background tasks: a tool source (``ReAct(tools=[background(research)])``,
    ``agent.tools.use(...)``) of Deep Agents' five async sub-agent tools — start one (it works
    on a worker while the conversation goes on), check it, give it a new instruction, cancel
    it, list them. Each task is a run of its agent in the run store, the child of the run that
    started it, and its conversation's only."""
    return Background(agents)


def _runtime() -> Runtime:
    runtime = current()
    assert runtime is not None  # the bridge runs a tool inside a run
    return runtime


def _of(record: RunRecord, conversation: str, agents: Any) -> bool:
    return record.agent_id in agents and record.metadata.get(BACKGROUND) == conversation


async def tasks_of(runtime: Runtime, agents: Any) -> list[RunRecord]:
    """The tasks of ``agents`` started in the run's conversation, newest first, at most
    :data:`LISTED`: the conversation's thread's runs of those agents (a run with no thread: its
    own children), each read whole at once."""
    runs, tenant = runtime.agent.harness.runs, runtime.tenant
    where: dict[str, Any] = (
        {"thread_id": runtime.thread} if runtime.thread else {"parent_run_id": runtime.run_id}
    )
    found = []
    for agent_id in agents:
        found.extend(
            [
                s
                async for s in runs.iterate(
                    agent_id=agent_id, tenant=tenant, limit=LISTED, max_pages=1, **where
                )
            ]
        )
    found.sort(key=lambda s: s.updated_at, reverse=True)
    records = await asyncio.gather(*(runs.get(s.run_id, tenant=tenant) for s in found[:LISTED]))
    return [r for r in records if r is not None and _of(r, runtime.conversation, agents)]


def described(record: RunRecord, *, whole: bool = False) -> dict[str, Any]:
    """A task as the model reads it: its id, agent and status, and — as it stands — its
    answer (cut to :data:`ANSWER_CHARS` unless ``whole``), why it failed, or its question."""
    said: dict[str, Any] = {
        "task_id": record.run_id,
        "subagent_type": record.agent_id,
        "status": record.status.value.lower(),
        "started_at": record.created_at.isoformat(),
        "updated_at": record.updated_at.isoformat(),
    }
    if record.status in FILTERS["success"]:
        answer = record.output if isinstance(record.output, str) else json.dumps(record.output)
        said["result"] = answer if whole else _cut(answer)
    elif record.error is not None:
        said["error"] = record.error.message
    elif record.status is RunStatus.PAUSED and record.awaiting is not None:
        said["asks"] = record.awaiting.question
    return said


def _cut(text: str) -> str:
    return text if len(text) <= ANSWER_CHARS else f"{text[:ANSWER_CHARS]}… (check_async_task: all)"


def sources(agent: Agent) -> list[Background]:
    """The background sources of ``agent`` (none: it starts no task)."""
    return [s for s in agent.sources if isinstance(s, Background)]


def looks(runtime: Runtime) -> bool:
    """Whether a run is told of its conversation's background tasks (:func:`note`): a
    top-level run of an agent that starts them, but a function's (no model to tell: its code
    calls the tools)."""
    return (
        runtime.parent is None
        and runtime.agent.adapter.name != "function"
        and bool(sources(runtime.agent))
    )


async def note(runtime: Runtime) -> str | None:
    """What a turn of a conversation is told of its background tasks as it starts: each one's
    status, and the answer of each that ended; ``None`` when it has none, or the run store
    cannot say (a warning). Asked only of a top-level run whose agent has a background
    source."""
    agents = {name: a for s in sources(runtime.agent) for name, a in s.agents.items()}
    try:
        tasks = await tasks_of(runtime, agents)
    except Exception as exc:
        runtime.events.warning("background", f"the conversation's background tasks: {exc}")
        return None
    for task in tasks:  # the conversation's stream starts with each task as it stands
        runtime.events.custom(TASK_EVENT, **described(task))
    if not tasks:
        return None
    return "\n".join([NOTE, *(json.dumps(described(t), default=str) for t in tasks)])


def told(runtime: Runtime, said: str | None) -> None:
    """The run told ``said`` (:func:`note`), after the rest of its context."""
    if said is not None:
        runtime.context = "\n\n".join(p for p in (runtime.pushed_context, said) if p)


def relay(agent: Agent, record: RunRecord) -> Callable[[RunEvent], None]:
    """A task's run's events as :data:`TASK_EVENT` events of the run working in the conversation
    that started it, when one works in this process: its start, each step it finished, its
    end."""
    said = {"task_id": record.run_id, "subagent_type": agent.id}

    def heard(event: RunEvent) -> None:
        if event.type is RunEventType.RUN_STARTED:
            state: dict[str, Any] = {"status": "running"}
        elif event.type is RunEventType.TOOL_CALL_RESULT:
            state = {"status": "running", "progress": {"tool": event.data.get("tool")}}
        elif event.type is RunEventType.RUN_FINISHED:
            outcome = event.outcome or RunOutcome.SUCCESS
            state = {"status": ENDINGS.get(outcome, outcome.value)}
            if event.error is not None:
                state["error"] = event.error.message
            elif outcome is RunOutcome.SUCCESS:
                state["result"] = event.data.get("result")
        else:
            return
        live = _live(agent, record.metadata.get(BACKGROUND))
        if live is not None:
            live.events.custom(TASK_EVENT, **said, **state)

    return heard


def _live(agent: Agent, conversation: Any) -> Runtime | None:
    """The run working in ``conversation`` in this process, if one does."""
    for parent in agent.harness.agents.values():
        live = parent.turns.live(conversation) if isinstance(conversation, str) else None
        if live is not None and (runtime := parent.running.get(live.run_id)) is not None:
            return runtime
    return None


def ended(agent: Agent, record: RunRecord, result: Result) -> None:
    """A background task that ended: its answer given to the run working in the conversation
    that started it, at that run's next step, when one works in this process (else the
    conversation's next turn is told: :func:`note`)."""
    conversation = record.metadata.get(BACKGROUND)
    if not isinstance(conversation, str) or not result.status.final:
        return
    agent.turns.given.pop(record.run_id, None)
    if result.status in FILTERS["success"]:
        answer = result.answer
    else:
        answer = result.error.message if result.error is not None else ""
    text = FINISHED.format(
        task_id=record.run_id,
        agent=agent.id,
        status=result.status.value,
        answer=answer if isinstance(answer, str) else json.dumps(answer, default=str),
    )
    for parent in agent.harness.agents.values():
        live = parent.turns.live(conversation)
        if live is not None and live.steers:
            live.inbox.append(Steer(text, f"{record.run_id}:ended"))
            return


__all__ = [
    "BACKGROUND",
    "TASK_EVENT",
    "Background",
    "background",
    "described",
    "ended",
    "looks",
    "note",
    "relay",
    "sources",
    "tasks_of",
    "told",
]
