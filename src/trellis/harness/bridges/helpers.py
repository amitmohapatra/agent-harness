"""Native helpers: the agents a framework spawns itself inside a run — a Deep Agent's ``task``
sub-agents, an OpenAI agent handed off to or called as a tool (``Agent.as_tool``), Claude's
``agents=`` subagents — each a child of the run, reached through the framework's own hooks
(the spawning stays the framework's):

* **lineage** — a run record of its own in the run store, its parent the run
  (``parent_run_id``), its metadata naming it (:data:`HELPER`): started when the framework
  starts it, ended with its answer or its error. Its id is derived from the run, the helper and
  its task (and how many times the attempt started that one), so the attempt after a pause or a
  crash finds the same record again and ends it;
* **its turns in memory** — the task it was given and what it answered, recorded in the memory
  service under its own run (and its own thread: the conversation of the run is not the
  helper's), as a run's transcript is;
* **governance and budget** — its tool calls and its model replies are the run's: they go
  through the same governance (the catalog's rules) and count toward the run's budget, by the
  adapter's hooks that reach the helper (:mod:`trellis.harness.adapters`).

A helper still open when the attempt ends is ended with it — ``SUCCESS`` when the run's answer
came, ``ERROR`` when the run failed — except on a pause or a crash: the next attempt, which runs
the helper again, ends it.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Final

from trellis.contracts import AgentError, RunStart, RunStatus, stable_id
from trellis.contracts.errors import is_pause_signal

if TYPE_CHECKING:
    from trellis.harness.runtime import Runtime

log = logging.getLogger("trellis.harness")

#: The metadata key of a helper's run record: ``{"framework", "kind", "name"}``.
HELPER: Final = "native_helper"
#: The error code of a helper whose run failed before it ended.
UNFINISHED: Final = "HelperUnfinished"
#: Why a helper an earlier attempt started, and the next did not take up again, ended.
NOT_CONTINUED: Final = "cut short by a pause or a crash, and not taken up again: the run went on"


class Kind(StrEnum):
    """How the framework spawned the helper."""

    TASK = "task"  # Deep Agents' ``task`` tool
    HANDOFF = "handoff"  # an OpenAI agent handed off to
    AGENT_TOOL = "agent_tool"  # an OpenAI agent called as a tool (``Agent.as_tool``)
    SUBAGENT = "subagent"  # Claude's ``agents=`` (its ``Agent``/``Task`` tool)


@dataclass
class Helper:
    """One open helper: its run, and what it was told and said so far."""

    run_id: str
    said: list[tuple[str, str]] = field(default_factory=list)


class Helpers:
    """The helpers of one attempt of a run, by the framework's key for each (a tool call id,
    an agent's name): each started once and ended once. Holds the open ones only."""

    def __init__(self, runtime: Runtime, framework: str) -> None:
        self.runtime = runtime
        self.framework = framework
        self._open: dict[str, Helper] = {}
        #: how many times this attempt started each helper on each task
        self._started: Counter[tuple[str, str]] = Counter()

    async def start(self, key: str, *, kind: Kind, name: str, task: str) -> None:
        """The helper ``key`` started on ``task``: its run record started (found again on a
        later attempt)."""
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
                metadata={HELPER: {"framework": self.framework, "kind": kind, "name": name}},
            )
        )
        self._open[key] = Helper(run_id, [("user", task)])
        log.info(
            "helper started",
            extra={"run_id": parent.run_id, "helper_run_id": run_id, "helper": name},
        )

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
            helper.said.append(("assistant", output))
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
        await self._recorded(helper)
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
                continue  # a sub-agent of the harness's own (``Agent.as_tool``) ends itself
            await runs.finish(
                run_id,
                RunStatus.ERROR,
                error=AgentError(code=UNFINISHED, message=NOT_CONTINUED),
                tenant=parent.tenant,
            )
            await self._recorded(Helper(run_id, [("user", str(record.input))]))

    async def _recorded(self, helper: Helper) -> None:
        """The helper's turns, in the memory service under its own run (the run's records
        feature, as a run's transcript)."""
        parent = self.runtime
        memory = parent.agent.harness.memory
        if memory is None or parent.run_memory is None or not parent.uses("records"):
            return
        identity = parent.identity.model_copy(
            update={"run_id": helper.run_id, "thread": helper.run_id}
        )
        scoped, messages = memory.bind(identity), list(helper.said)
        await parent.agent.harness.writes.submit(
            "memory.transcript",
            lambda: scoped.record_messages(messages, helper.run_id, 1),
            events=parent.events,
            record=scoped.record(
                "record_messages",
                messages=[list(m) for m in messages],
                run_id=helper.run_id,
                attempt=1,
            ),
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
