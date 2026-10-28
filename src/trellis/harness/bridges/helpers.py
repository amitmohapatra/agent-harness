"""Native helpers: the agents a framework spawns itself inside a run — a Deep Agent's ``task``
sub-agents, an OpenAI agent called as a tool (``Agent.as_tool``), Claude's
``agents=`` subagents — each a full child run of the run, reached through the framework's own
hooks (the spawning stays the framework's):

* **lineage** — a run record of its own in the run store, its parent the run
  (``parent_run_id``), its metadata naming it (:data:`HELPER`) and the run's agent group
  (:data:`~trellis.harness.identity.AGENT_GROUP`): started when the framework
  starts it, ended with its answer or its error. Its id is derived from the run, the helper and
  its task (and how many times the attempt started that one), so the attempt after a pause or a
  crash finds the same record again and ends it;
* **its own memory** — its scope derived from the run's as a sub-agent's is (it reads the
  run's hand-off notes; ``scope="group"`` binds to the run's group), in it: the memory context
  for its task, pushed into what it is given (:meth:`Helpers.briefed`: the task the framework
  hands it, the context before it); the tool calls made while it works
  (:meth:`Helpers.working`, :func:`inside`), recorded under its run; its turns — the task and
  its answer — under its run and its own thread (the conversation of the run is not the
  helper's); and how it ended, as its run's ``system`` outcome;
* **the task tree** — on the run's stream as every child run is (``admission.py``):
  ``helper_started`` when the framework starts it, ``helper_finished`` (its status and
  seconds) when it ends; its parallelism stays the framework's own;
* **governance and budget** — its tool calls and its model replies are the run's: they go
  through the same governance (the catalog's rules) and count toward the run's budget, by the
  adapter's hooks that reach the helper (:mod:`trellis.harness.adapters`).

A helper still open when the attempt ends is ended with it — ``SUCCESS`` when the run's answer
came, ``ERROR`` when the run failed — except on a pause or a crash: the next attempt, which runs
the helper again, ends it.
"""

from __future__ import annotations

import contextlib
import logging
import time
from collections import Counter
from collections.abc import Iterator
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Final

from trellis.contracts import AgentError, RunStart, RunStatus, stable_id
from trellis.contracts.errors import is_pause_signal
from trellis.harness.history import Said, Speaker, recorded
from trellis.harness.identity import AGENT_GROUP

if TYPE_CHECKING:
    from trellis.harness.clients.memory import RunMemory
    from trellis.harness.runtime import Runtime

log = logging.getLogger("trellis.harness")

#: The metadata key of a helper's run record: ``{"framework", "kind", "name"}``.
HELPER: Final = "native_helper"
#: The error code of a helper whose run failed before it ended.
UNFINISHED: Final = "HelperUnfinished"
#: What comes between a helper's memory context and its task, in what it is given.
TASK_HEADING: Final = "## Your task\n"
#: Why a helper an earlier attempt started, and the next did not take up again, ended.
NOT_CONTINUED: Final = "cut short by a pause or a crash, and not taken up again: the run went on"


class Kind(StrEnum):
    """How the framework spawned the helper."""

    TASK = "task"  # Deep Agents' ``task`` tool
    AGENT_TOOL = "agent_tool"  # an OpenAI agent called as a tool (``Agent.as_tool``)
    SUBAGENT = "subagent"  # Claude's ``agents=`` (its ``Agent``/``Task`` tool)


@dataclass
class Helper:
    """One open helper: its run, whose run it works in, its name, what it was told and said
    so far, its memory (its scope's: ``None`` with memory off for the run) and the memory
    context pushed for its task."""

    run_id: str
    parent_run_id: str
    name: str
    said: list[Said] = field(default_factory=list)
    memory: RunMemory | None = None
    context: str | None = None
    #: when it started (``time.monotonic``)
    began: float = field(default_factory=time.monotonic)


#: The helper whose work is running in this task (:meth:`Helpers.working`).
_INSIDE: ContextVar[Helper | None] = ContextVar("trellis_helper", default=None)


def inside(runtime: Runtime) -> Helper | None:
    """The helper of ``runtime``'s run whose work the calling code is part of, if any: what it
    does is recorded under its run (a nested run's code is its own run's, not the helper's)."""
    helper = _INSIDE.get()
    return helper if helper is not None and helper.parent_run_id == runtime.run_id else None


class Helpers:
    """The helpers of one attempt of a run, by the framework's key for each (a tool call id,
    an agent's name): each started once and ended once. Holds the open ones only."""

    def __init__(self, runtime: Runtime, framework: str) -> None:
        self.runtime = runtime
        self.framework = framework
        self._open: dict[str, Helper] = {}
        #: how many times this attempt started each helper on each task
        self._started: Counter[tuple[str, str]] = Counter()
        runtime.helpers = self  # the adapter's hooks find the attempt's helpers here

    async def start(self, key: str, *, kind: Kind, name: str, task: str) -> Helper:
        """The helper ``key`` started on ``task``: its run record started (found again on a
        later attempt), its memory bound, and the memory context for its task fetched
        (:meth:`briefed` hands it over). A key already open is that helper."""
        found = self._open.get(key)
        if found is not None:
            return found
        parent = self.runtime
        occurrence = self._started[(name, task)]
        self._started[(name, task)] += 1
        run_id = stable_id(parent.run_id, HELPER, kind, name, task, occurrence, prefix="run_")
        await parent.agent.harness.runs.start(
            RunStart(
                run_id=run_id,
                tenant_id=parent.tenant,
                agent_id=parent.agent_id,
                parent_run_id=parent.run_id,
                thread_id=run_id,  # its own conversation, not the run's
                user_id=parent.user,
                workspace_id=parent.identity.workspace,
                input=task,
                metadata={
                    HELPER: {"framework": self.framework, "kind": kind, "name": name},
                    AGENT_GROUP: parent.identity.own_group,
                },
            )
        )
        helper = Helper(run_id, parent.run_id, name, [Said(Speaker.USER, task)])
        helper.memory = self._memory(run_id)
        helper.context = await self._pushed(helper, task)
        self._open[key] = helper
        parent.events.helper(
            "started", run_id=run_id, agent_id=name, framework=self.framework, kind=kind.value
        )
        log.info(
            "helper started",
            extra={"run_id": parent.run_id, "helper_run_id": run_id, "helper": name},
        )
        return helper

    @staticmethod
    def briefed(helper: Helper, task: str) -> str:
        """What the helper is given for ``task``: the memory context pushed for it, then the
        task (the task alone without one)."""
        return f"{helper.context}\n\n{TASK_HEADING}{task}" if helper.context else task

    @contextlib.contextmanager
    def working(self, helper: Helper | None) -> Iterator[None]:
        """The code inside is ``helper``'s work (``None``: the run's own): the tool calls it
        makes are recorded under the helper's run (:func:`inside`)."""
        token = _INSIDE.set(helper)
        try:
            yield
        finally:
            _INSIDE.reset(token)

    def named(self, name: str) -> Helper | None:
        """The open helper called ``name`` started last (a framework that names the helper
        a call comes from, not the call that started it: Claude's ``agent_type``)."""
        found = [h for h in self._open.values() if h.name == name]
        return found[-1] if found else None

    async def end(self, key: str, *, output: str | None = None, error: str | None = None) -> None:
        """The helper ``key`` ended with ``output``, or failed (``error``): its run record
        ended, its turns recorded. A key not open (ended already) is no change, and so is
        any while the run is pausing (a framework that hid the pause in a tool's result still
        hands back): the next attempt takes the helper up again."""
        if self.runtime.pending is not None:
            return
        helper = self._open.pop(key, None)
        if helper is None:
            return
        if output:
            helper.said.append(Said(Speaker.ASSISTANT, output))
        runs = self.runtime.agent.harness.runs
        record = await runs.get(helper.run_id, tenant=self.runtime.tenant)
        # a helper a re-run took up again after it ended (before a crash) is ended already
        if record is not None and not record.status.final:
            await runs.finish(
                helper.run_id,
                RunStatus.SUCCESS if error is None else RunStatus.ERROR,
                output=output,
                error=None if error is None else AgentError(code=UNFINISHED, message=error),
                tenant=self.runtime.tenant,
            )
        self.runtime.events.helper(
            "finished",
            run_id=helper.run_id,
            agent_id=helper.name,
            framework=self.framework,
            status=(RunStatus.SUCCESS if error is None else RunStatus.ERROR).value.lower(),
            seconds=round(time.monotonic() - helper.began, 3),
        )
        await self._recorded(helper)
        await self._outcome(helper, error)
        log.info(
            "helper ended",
            extra={
                "run_id": self.runtime.run_id,
                "helper_run_id": helper.run_id,
                "status": "ok" if error is None else "error",
            },
        )

    async def closed(self, error: BaseException | None) -> None:
        """The attempt ended (``error``: how, ``None`` with its answer): the helpers still open
        end with it — except on a pause or a crash, after which the next attempt ends them;
        and, after an earlier attempt, the helpers it left open that this one did not take up
        again (the framework went on without them) end ``ERROR`` (:data:`NOT_CONTINUED`)."""
        paused = self.runtime.pending is not None or (
            error is not None and (not isinstance(error, Exception) or is_pause_signal(error))
        )
        if paused:
            return
        why = None if error is None else f"the run ended first: {type(error).__name__}"
        for key in list(self._open):
            await self.end(key, error=why)
        if self.runtime.attempt > 1:
            await self._left_open()

    async def _left_open(self) -> None:
        parent, runs = self.runtime, self.runtime.agent.harness.runs
        left = [
            child.run_id
            async for child in runs.iterate(
                parent_run_id=parent.run_id, status=RunStatus.RUNNING, tenant=parent.tenant
            )
        ]
        for run_id in left:
            record = await runs.get(run_id, tenant=parent.tenant)
            if record is None or HELPER not in record.metadata:
                continue  # a sub-agent's child run (``subagents``) ends itself
            await runs.finish(
                run_id,
                RunStatus.ERROR,
                error=AgentError(code=UNFINISHED, message=NOT_CONTINUED),
                tenant=parent.tenant,
            )
            name = str((record.metadata.get(HELPER) or {}).get("name", ""))
            left = Helper(run_id, parent.run_id, name, [Said(Speaker.USER, str(record.input))])
            left.memory = self._memory(run_id)
            await self._recorded(left)
            await self._outcome(left, NOT_CONTINUED)

    def _memory(self, run_id: str) -> RunMemory | None:
        """The helper's memory: its scope derived from the run's (``None`` with memory off for
        the run)."""
        parent = self.runtime
        memory = parent.agent.harness.memory
        if memory is None or parent.run_memory is None:
            return None
        identity = parent.identity.model_copy(
            update={
                "run_id": run_id,
                "thread": run_id,
                "parent_run": parent.run_id,
                "group": parent.identity.own_group,
            }
        )
        return memory.bind(identity)

    async def _pushed(self, helper: Helper, task: str) -> str | None:
        """The memory context for the helper's task, in its scope, as the run's is pushed (the
        documents the run draws on; no window: the helper's conversation is its task). A
        failure is a warning: the helper works without it."""
        parent = self.runtime
        if helper.memory is None or not task or not parent.uses("memory_push"):
            return None
        from trellis.harness.clients.memory import Window

        try:
            pushed = await helper.memory.context(
                task,
                tools=None,
                window=Window.NONE,
                hints=False,
                document_ids=parent.document_ids,
            )
        except Exception as exc:
            parent.events.warning(
                "memory_unavailable", f"no memory context for {helper.name}: {exc}"
            )
            return None
        return pushed.rendered or None

    async def _recorded(self, helper: Helper) -> None:
        """The helper's turns, in the memory service under its own run (the run's records
        feature, as a run's transcript)."""
        parent = self.runtime
        scoped = helper.memory
        if scoped is None or not parent.uses("records"):
            return
        messages = list(helper.said)
        await parent.agent.harness.writes.submit(
            "memory.transcript",
            lambda: scoped.record_messages(messages, helper.run_id, 1),
            events=parent.events,
            record=scoped.record(
                "record_messages",
                messages=[recorded(m) for m in messages],
                run_id=helper.run_id,
                attempt=1,
            ),
        )

    async def _outcome(self, helper: Helper, error: str | None) -> None:
        """How the helper ended, as its run's ``system`` outcome (``confirm``: it answered;
        ``reject``: it failed), as a run's is."""
        parent = self.runtime
        scoped = helper.memory
        if scoped is None or not parent.uses("records"):
            return
        feedback = {
            "verdict": "confirm" if error is None else "reject",
            "source": "system",
            "comment": error,
            "key": f"{helper.run_id}:outcome",
        }
        await parent.agent.harness.writes.submit(
            "memory.outcome",
            lambda: scoped.run_feedback(**feedback),
            events=parent.events,
            record=scoped.record("run_feedback", **feedback),
        )


def text_of(output: Any) -> str:
    """A helper's answer as text: a string, a message's text, or what a tool's ``Command``
    carries back (its last message's)."""
    if isinstance(output, str):
        return output
    update = getattr(output, "update", None)
    if isinstance(update, dict) and update.get("messages"):
        return text_of(update["messages"][-1])
    text = getattr(output, "text", None)
    if isinstance(text, str):
        return text
    content = getattr(output, "content", None)
    return content if isinstance(content, str) else str(output)
