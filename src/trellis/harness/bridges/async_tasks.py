"""Deep Agents' async sub-agent tasks (``AsyncSubAgentMiddleware``: a task on an Agent Protocol
server of yours) as child runs. Each task a ``start_async_task`` call starts is a native helper
of the turn that started it (``bridges.helpers``, :attr:`Kind.ASYNC_TASK`) that outlives the
turn, as the task does — kept as a Claude subagent working in the background is
(``bridges.claude_live.Background``), with no process of the harness's at work in it:

* **its record** — started when the start's result names the task: its input the task's text,
  its metadata the task's id, its server's ``url`` and its ``graph_id`` (never the spec's
  headers); its id derived from the agent and the task (:func:`run_of`), so whoever learns
  how the task ended finds it, on another server that shares the run store too. A follow-up (``update_async_task``) to a task at work goes on as its
  run; one to a task that ended runs it again on its server: a run of its own here too, the
  child of the turn that sent it;
* **held and read** — handed on past the turn's attempt (``Helpers.detached``) and held in the
  run store from this process (``holding``) while the task runs: a process that dies with it
  leaves a run the store ends (cut off). Every :data:`WATCH_SECONDS` this process reads the
  task (:meth:`AsyncTasks._watch`): ended on its server (Deep Agents' own ``check_async_task``),
  its run ends and the chat is told how — a turn of the chat's own when it is idle, as a
  harness task's outcome is (``background.delivered``); its run ended elsewhere (its turn
  cancelled from another server), the task is cancelled on its server;
* **its end** — when Deep Agents' own tools say the task ended: a later check or list
  (``check_async_task``, ``list_async_tasks``) reads its final status from its server, or the
  model cancels it (``cancel_async_task``) — or a read of it, above. Its run ends ``SUCCESS``
  with the task's result, ``ERROR`` saying why it failed, or ``CANCELLED``, on the stream of the
  run that learned it;
* **cancelled with its turn** — as a cancelled run's other children are: the turn's attempt
  cancelled while it runs (``Runtime.cancelled``: :meth:`AsyncTasks.cancelled`), or the turn
  cancelled where it waits for a person (``subagents.cancel_children``:
  :meth:`AsyncTasks.stopped`). The task is cancelled on its server by Deep Agents' own
  ``cancel_async_task`` (the Agent Protocol's cancel of the task's run of now), and its run
  ends ``CANCELLED``.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import json
import logging
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

from trellis.contracts import AgentError, RunStatus, stable_id
from trellis.harness.bounded import MAXSIZE
from trellis.harness.bridges.helpers import HELPER, UNFINISHED, Helper, Kind, text_of
from trellis.harness.history import Said, Speaker
from trellis.runs import ConflictError, LeaseLostError

if TYPE_CHECKING:
    from trellis.harness.agent import Agent
    from trellis.harness.runtime import Runtime

log = logging.getLogger("trellis.harness")

#: Deep Agents' tools that start a task and cancel one (``middleware.ASYNC_TASKS`` has all five).
START: Final = "start_async_task"
UPDATE: Final = "update_async_task"
CHECK: Final = "check_async_task"
CANCEL: Final = "cancel_async_task"
#: How often the process holding a task's run reads the task: how it goes on its server (Deep
#: Agents' own ``check_async_task``) and whether its run was ended elsewhere.
WATCH_SECONDS: Final = 10.0
#: The statuses Deep Agents takes for a task that has ended (its ``_TERMINAL_STATUSES``), and
#: how the task's run ends for each.
ENDED: Final[Mapping[str, RunStatus]] = {
    "success": RunStatus.SUCCESS,
    "cancelled": RunStatus.CANCELLED,
    "error": RunStatus.ERROR,
    "timeout": RunStatus.ERROR,
    "interrupted": RunStatus.ERROR,
}
#: Why a task's run ended when its server said how and not why (a list), or it was cancelled.
ENDED_THERE: Final = "the task ended on its server: {status}"
#: Why the run of a task its turn's cancel reached ended.
CANCELLED_WITH: Final = "the run that started the task was cancelled"
#: The most tasks one agent keeps held in this process: past it the oldest is let go, and the
#: run store ends its run once its hold lapses (cut off), as a process that stopped leaves it.
KEPT: Final = MAXSIZE


def run_of(tenant: str, agent_id: str, task_id: str, again: str | None = None) -> str:
    """The id of the run of the task ``task_id`` an agent's run started: the same wherever it
    is derived (the task's id is its thread's on its server) — and, ``again``, of a follow-up
    that ran it again after it ended (its run on the server)."""
    parts = (task_id,) if again is None else (task_id, again)
    return stable_id(tenant, agent_id, HELPER, Kind.ASYNC_TASK, *parts, prefix="run_")


@dataclass(slots=True)
class Kept:
    """A task this process holds the run of: its helper, the agent whose run started it, and
    Deep Agents' record of it (``AsyncTask``: its thread and its run of now)."""

    helper: Helper
    agent: Agent
    tenant: str
    task: dict[str, Any]
    #: its reads (:meth:`AsyncTasks._watch`)
    watching: asyncio.Task[None] | None = None


class AsyncTasks:
    """The tasks an agent's runs started that this process holds the runs of, by run id."""

    def __init__(self) -> None:
        self._kept: OrderedDict[str, Kept] = OrderedDict()

    async def said(
        self, runtime: Runtime, tool: Any, *, args: Mapping[str, Any], done: Any, told: str
    ) -> None:
        """What a call of Deep Agents' task tool ``tool`` (made with ``args``) returned:
        ``done``, whose message says ``told``. A task it started is a child run of
        ``runtime``'s run from here; Deep Agents' record of every task it names is kept, and a
        task it found ended ends its run."""
        from langgraph.types import Command

        update = done.update if isinstance(done, Command) else None
        found = update.get("async_tasks") if isinstance(update, Mapping) else None
        for task_id, task in (found or {}).items():
            run_id = self._held(task_id) or await _run(runtime, task)
            if tool.name == START:
                await self._started(runtime, tool, run_id, dict(task), str(args["description"]))
            elif tool.name == UPDATE and await _ended_or_none(runtime, run_id):
                # a follow-up runs the ended task again on its server: a run of the turn that
                # sent it (one at work goes on as its run)
                again = run_of(runtime.tenant, runtime.agent_id, task_id, task["run_id"])
                await self._started(runtime, tool, again, dict(task), str(args["message"]))
                run_id = again
            kept = self._kept.get(run_id)
            if kept is not None:
                kept.task = dict(task)
            status = ENDED.get(str(task.get("status")))
            if status is not None:
                await self._ended(runtime, run_id, status, told)

    async def _started(
        self, runtime: Runtime, tool: Any, run_id: str, task: dict[str, Any], text: str
    ) -> None:
        """The task's run, started as a helper of the run and handed on to this process."""
        helpers = runtime.helpers
        assert helpers is not None  # a graph's attempt has its helpers (``adapters.langgraph``)
        spec = _specs(tool).get(task["agent_name"]) or {}
        about = {"task_id": task["task_id"]}
        about.update({k: str(spec[k]) for k in ("url", "graph_id") if spec.get(k)})
        key, name = task["task_id"], task["agent_name"]
        helper = await helpers.start(
            key, kind=Kind.ASYNC_TASK, name=name, task=text, run_id=run_id, about=about
        )
        helpers.detached(key)
        runtime.agent.harness.holds.hold(run_id, runtime.tenant)
        kept = Kept(helper, runtime.agent, runtime.tenant, task)
        self._kept[run_id] = kept
        kept.watching = asyncio.get_running_loop().create_task(
            self._watch(run_id), context=contextvars.Context()
        )
        while len(self._kept) > KEPT:
            _, oldest = self._kept.popitem(last=False)
            self._released(oldest)

    def _held(self, task_id: str) -> str | None:
        """The run of the task ``task_id`` this process holds, if it holds one."""
        return next((r for r, k in self._kept.items() if k.task["task_id"] == task_id), None)

    @staticmethod
    def _released(kept: Kept) -> None:
        """``kept`` held no more here, and read no more."""
        kept.agent.harness.holds.let_go(kept.helper.run_id)
        if kept.watching is not None and kept.watching is not asyncio.current_task():
            kept.watching.cancel()

    async def _watch(self, run_id: str) -> None:
        """Every :data:`WATCH_SECONDS` while this process holds ``run_id``: its run ended
        elsewhere (cancelled with its turn from another server) cancels the task on its server;
        the task ended on its server ends its run, and the chat is told how
        (``background.delivered``: a turn of its own when it is idle)."""
        while (kept := self._kept.get(run_id)) is not None:
            await asyncio.sleep(WATCH_SECONDS)
            if self._kept.get(run_id) is not kept:
                return
            runs = kept.agent.harness.runs
            record = await runs.get(run_id, tenant=kept.tenant)
            if record is None or record.status.final:
                stopped = record is not None and record.status is RunStatus.CANCELLED
                await (self.stopped(run_id, CANCELLED_WITH) if stopped else self._let(run_id))
                return
            with contextlib.suppress(Exception):  # the server out of reach: read again later
                await self._read(run_id, kept)

    async def _read(self, run_id: str, kept: Kept) -> None:
        """How the task goes on its server, as Deep Agents' own check reads it: ended, its run
        ends and the chat is told."""
        from trellis.harness import background

        task_id = kept.task["task_id"]
        done = await _tool(kept.agent.target, CHECK).ainvoke(
            {"task_id": task_id, "runtime": _tool_runtime(kept.task, CHECK)}
        )
        update = getattr(done, "update", None)
        if not isinstance(update, Mapping):
            return  # Deep Agents' word of a check that failed: read again later
        found = update.get("async_tasks", {}).get(task_id)
        status = ENDED.get(str((found or {}).get("status")))
        if found is None or status is None:
            return
        told = text_of(update["messages"][-1])
        said = _checked(told)
        await self._closed(run_id, status, str(said.get("error") or said.get("result") or ""))
        event = {k: found[k] for k in ("task_id", "agent_name", "status", "run_id")}
        if "result" in said:
            event["result"] = said["result"]
        elif status is not RunStatus.SUCCESS and status is not RunStatus.CANCELLED:
            event["error"] = background.FAILED
        record = await kept.agent.harness.runs.get(run_id, tenant=kept.tenant)
        if record is not None:
            await background.delivered(kept.agent, record, update=event)

    async def _let(self, run_id: str) -> None:
        kept = self._kept.pop(run_id, None)
        if kept is not None:
            self._released(kept)

    async def _closed(self, run_id: str, status: RunStatus, said: str) -> None:
        """The task's run held here ended ``status`` outside any attempt (``said``: its result,
        or why it ended), held and read no more."""
        kept = self._kept.pop(run_id, None)
        if kept is None:
            return
        self._released(kept)
        runs, tenant = kept.agent.harness.runs, kept.tenant
        with contextlib.suppress(ConflictError, LeaseLostError):  # ended meanwhile
            if status is RunStatus.CANCELLED:
                await runs.cancel(run_id, reason=said or None, tenant=tenant)
            elif status is RunStatus.SUCCESS:
                await runs.finish(run_id, status, output=said or None, tenant=tenant)
            else:
                why = said or ENDED_THERE.format(status=status.value.lower())
                error = AgentError(code=UNFINISHED, message=why)
                await runs.finish(run_id, status, error=error, tenant=tenant)

    async def _ended(self, runtime: Runtime, run_id: str, status: RunStatus, told: str) -> None:
        """The task's run ended as its server says (``told``: the check's result, if it is
        one), on the stream of ``runtime``'s run — one this process holds, or one it finds in
        the run store not ended (held elsewhere, or by nobody now)."""
        kept = self._kept.pop(run_id, None)
        if kept is not None:
            self._released(kept)
        helper = kept.helper if kept is not None else await _found(runtime, run_id)
        if helper is None:
            return
        helpers = runtime.helpers
        assert helpers is not None
        said = _checked(told)
        why = str(said.get("error") or ENDED_THERE.format(status=status.value.lower()))
        if status is RunStatus.SUCCESS:
            await helpers.finished(helper, output=str(said["result"]) if "result" in said else None)
        else:
            await helpers.finished(helper, error=why, cancelled=status is RunStatus.CANCELLED)

    async def cancelled(self, runtime: Runtime) -> None:
        """``runtime``'s attempt was cancelled (``Runtime.cancelled``): the tasks its run started
        are stopped with it (:meth:`stopped`)."""
        if runtime.cancelled is None:
            return
        for run_id, kept in list(self._kept.items()):
            if kept.helper.parent_run_id == runtime.run_id:
                await self.stopped(run_id, runtime.cancelled)

    async def stopped(self, run_id: str, reason: str | None) -> None:
        """The run ``run_id`` was cancelled with the run that started it (``reason``: why):
        when it is a task's this process holds, the task is cancelled on its server with Deep
        Agents' own ``cancel_async_task`` and its run ends ``CANCELLED``, held no more. A
        server that refuses the cancel is said in the log: the run ends all the same."""
        kept = self._kept.pop(run_id, None)
        if kept is None:
            return
        agent = kept.agent
        self._released(kept)
        task_id = kept.task["task_id"]
        said = await _tool(agent.target, CANCEL).ainvoke(
            {"task_id": task_id, "runtime": _tool_runtime(kept.task, CANCEL)}
        )
        if isinstance(said, str):  # Deep Agents' word of a cancel that failed
            log.warning("async task %s not cancelled on its server: %s", task_id, said)
        with contextlib.suppress(ConflictError):  # cancelled already, with its parent
            await agent.harness.runs.cancel(
                run_id, reason=reason or CANCELLED_WITH, tenant=kept.tenant
            )


def _tool_runtime(task: Mapping[str, Any], tool: str) -> Any:
    """What Deep Agents' task tool ``tool`` is run with outside a graph: the task's record as
    the state, nothing else of a run."""
    from langchain.tools import ToolRuntime

    task_id = task["task_id"]
    return ToolRuntime(
        state={"async_tasks": {task_id: dict(task)}},
        context=None,
        config={},
        stream_writer=lambda _: None,
        tool_call_id=f"{tool}:{task_id}",
        store=None,
    )


async def _run(runtime: Runtime, task: Mapping[str, Any]) -> str:
    """The run of ``task`` that a process other than this one holds, or nobody: a follow-up's
    after the task ended (:func:`run_of` ``again``), when one was recorded, else the task's."""
    first = run_of(runtime.tenant, runtime.agent_id, task["task_id"])
    again = run_of(runtime.tenant, runtime.agent_id, task["task_id"], task["run_id"])
    found = await runtime.agent.harness.runs.get(again, tenant=runtime.tenant)
    return again if found is not None else first


async def _ended_or_none(runtime: Runtime, run_id: str) -> bool:
    """Whether the run ``run_id`` ended, or was never recorded."""
    found = await runtime.agent.harness.runs.get(run_id, tenant=runtime.tenant)
    return found is None or found.status.final


async def _found(runtime: Runtime, run_id: str) -> Helper | None:
    """The task's run as the run store has it, when it has not ended: a helper to end
    (``None``: none, or ended already)."""
    record = await runtime.agent.harness.runs.get(run_id, tenant=runtime.tenant)
    if record is None or record.status.final:
        return None
    name = str((record.metadata.get(HELPER) or {}).get("name", ""))
    said = [Said(Speaker.USER, str(record.input))]
    return Helper(run_id, record.parent_run_id or "", name, said, kind=Kind.ASYNC_TASK)


def _checked(told: str) -> dict[str, Any]:
    """What a check of a task said (``check_async_task``: its status, and its ``result`` or
    ``error``); ``{}`` for any other tool's word."""
    with contextlib.suppress(ValueError):
        said = json.loads(told)
        return said if isinstance(said, dict) else {}
    return {}


def _specs(start: Any) -> dict[str, Mapping[str, Any]]:
    """The async sub-agents ``start_async_task`` starts tasks of, as its code holds them (their
    specs by name: ``url``, ``graph_id``); ``{}`` when they cannot be read (a release laid out
    otherwise: the task's run then records its id alone)."""
    cells = getattr(getattr(start, "coroutine", None), "__closure__", None) or ()
    return next((c.cell_contents for c in cells if _are_specs(c.cell_contents)), {})


def _are_specs(value: Any) -> bool:
    """Whether ``value`` is async sub-agents' specs by name (Deep Agents' ``agent_map``)."""
    return (
        isinstance(value, dict)
        and bool(value)
        and all(isinstance(v, Mapping) and "graph_id" in v for v in value.values())
    )


def _tool(graph: Any, name: str) -> Any:
    """Deep Agents' own tool ``name`` of ``graph`` (its sub-agents' graphs too): Deep Agents
    gives a graph the five task tools together."""
    from trellis.harness.adapters.langgraph import graph_tools
    from trellis.harness.middleware import framework_made

    return next(t for t in graph_tools(graph) if t.name == name and framework_made(t))
