"""``trellis.current()``: the run a tool or a node is executing in.

The runtime is set for the duration of one attempt of one run (a ``ContextVar``, so every
task the framework spawns inherits it). It carries the run's identity, its memory scope, its
tools and the one way to pause: :meth:`Runtime.ask`.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Final

from trellis.contracts import (
    AgentPaused,
    ConfigurationError,
    Interrupt,
    InterruptDecision,
    InterruptReason,
    InterruptResolution,
    ToolCall,
    ToolError,
)
from trellis.harness.clients.runs import LeaseLost
from trellis.harness.events import LOG, RunEvents
from trellis.harness.identity import Identity
from trellis.harness.journal import Pending, Replay, content_key

if TYPE_CHECKING:
    from trellis.harness.agent import Agent
    from trellis.harness.clients.memory import RunMemory
    from trellis.harness.tools.base import Tool
    from trellis.memory import MemoryContext
    from trellis.memory.models import ToolHints

log = logging.getLogger("trellis.run")

#: A table or diff larger than this (compact JSON) travels by reference: stored as a run
#: artifact in agent-runs (``payload_ref``), not inside the question.
INLINE_PAYLOAD_BYTES: Final = 16 * 1024

#: The most a progress checkpoint may be, as compact JSON (agent-runs' ``MAX_CHECKPOINT_BYTES``:
#: a larger one is refused with ``413``, so it is not sent).
MAX_CHECKPOINT_BYTES: Final = 1024 * 1024
#: How often the journal is saved as progress after calls that only read (or model steps);
#: after a side-effecting call it is saved at once.
PROGRESS_SECONDS: Final = 20.0

_current: ContextVar[Runtime | None] = ContextVar("trellis_runtime", default=None)


def current() -> Runtime | None:
    """The run this code executes in, or ``None`` outside a harness run."""
    return _current.get()


class Paused(AgentPaused):
    """Raised by :meth:`Runtime.ask` to stop the run until a person answers. Frameworks that
    swallow it do not keep the run going: the harness checks the runtime, not the exception."""

    def __init__(self, interrupt: Interrupt) -> None:
        super().__init__(interrupt.question, expects=interrupt.expects, payload=interrupt.payload)
        self.interrupt = interrupt


class RunCancelled(Exception):
    """A person cancelled the run while answering it."""


Suspend = Callable[[dict[str, Any]], Any]


@dataclass(eq=False)
class Runtime:
    """One attempt of one run."""

    identity: Identity
    agent: Agent
    events: RunEvents
    replay: Replay
    attempt: int = 1
    run_memory: RunMemory | None = None
    #: the pushed memory context, as the framework's system message carries it
    context: str | None = None
    #: the question the run is working on (memory queries, tool-call tasks)
    task: str = ""
    #: this run's tools by name (the agent's own, the MCP tools, the memory pull tools)
    toolbox: dict[str, Tool] = field(default_factory=dict)
    #: the tools offered to the model (``None``: all of them) — the tool hints' candidates,
    #: the memory tools, and every tool this run has used (``used``)
    offered: set[str] | None = None
    #: the tools this run has called, in any attempt (the journal keeps them across a pause)
    used: set[str] = field(default_factory=set)
    #: set by an adapter whose framework can suspend itself (LangGraph ``interrupt``)
    suspend: Suspend | None = None
    #: the pause this attempt ended on
    pending: Pending | None = None
    #: a Code Mode script ran: its nested calls are read back from the gateway's log
    used_code_mode: bool = False
    #: the worker holding the run's lease, when a worker runs it, and that lease's length
    #: (what a progress checkpoint's heartbeat extends it by)
    worker_id: str | None = None
    lease_seconds: float | None = None
    #: the lease was lost while saving progress: the run stops and writes nothing more
    lease_lost: bool = False
    #: the run's transcript, tool calls and outcome are recorded in the memory service
    writes_memory: bool = False
    started_at: datetime | None = None
    _asked: int = 0
    _steps: int = 0
    _saved_at: float | None = None
    _too_large: bool = False

    # ------------------------------------------------------------------ identity
    @property
    def run_id(self) -> str:
        return self.identity.run_id

    @property
    def agent_id(self) -> str:
        return self.identity.agent_id

    @property
    def user(self) -> str:
        return self.identity.user

    @property
    def thread(self) -> str | None:
        return self.identity.thread

    @property
    def tenant(self) -> str:
        return self.identity.tenant

    # ------------------------------------------------------------------ services
    @property
    def memory(self) -> MemoryContext:
        """The memory service in this run's scope (the SDK's verbs)."""
        if self.run_memory is None:
            raise ConfigurationError("memory is off in this deployment: set MEMORY_URL")
        return self.run_memory.ctx

    @property
    def tools(self) -> Tools:
        return Tools(self)

    def log(self, message: str, **fields: Any) -> None:
        """A line in the run's log and on its event stream."""
        log.info("%s", message, extra={"run_id": self.run_id, **fields})
        self.events.custom(LOG, message=message, **fields)

    def tool_names(self) -> list[str]:
        """The run's own tools (the memory service's pull tools are not candidates)."""
        return [name for name, found in self.toolbox.items() if found.spec.source != "memory"]

    def offers(self, name: str) -> bool:
        """Whether the model is offered ``name`` now (per turn where the framework allows)."""
        if self.offered is None or name in self.offered or name in self.used:
            return True
        found = self.toolbox.get(name)
        return found is not None and found.spec.source == "memory"

    def offer(self, names: Sequence[str]) -> None:
        """Offer more tools (a ``tool_search`` found them)."""
        if self.offered is not None:
            self.offered.update(n for n in names if n in self.toolbox)

    def next_step(self) -> int:
        self._steps += 1
        return self._steps

    # ------------------------------------------------------------------ progress
    async def progress(self, *, now: bool) -> None:
        """Save the journal as the run's progress checkpoint, on a heartbeat of the worker's
        lease (a worker's run only: nobody resumes an in-process run after its process died).
        ``now`` after a side-effecting call — the attempt after a crash replays it instead of
        running it again; otherwise at most every :data:`PROGRESS_SECONDS`. A checkpoint over
        :data:`MAX_CHECKPOINT_BYTES` is not sent and a refused save is a warning (the run goes
        on; the next save tries again); a lost lease stops the run (:class:`LeaseLost`)."""
        if self.worker_id is None or self.lease_seconds is None:
            return
        clock = time.monotonic()
        if not now and self._saved_at is not None and clock - self._saved_at < PROGRESS_SECONDS:
            return
        checkpoint = self.replay.journal.dump()
        size = len(json.dumps(checkpoint, default=str, separators=(",", ":")).encode())
        if size > MAX_CHECKPOINT_BYTES:
            if not self._too_large:
                self._too_large = True
                self._unsaved(f"its journal ({size} bytes) is too large to save as progress")
            return
        try:
            await self.agent.harness.runs.heartbeat(
                self.run_id, self.worker_id, self.lease_seconds, checkpoint=checkpoint
            )
        except LeaseLost:
            self.lease_lost = True
            raise
        except Exception as exc:
            self._unsaved(f"{type(exc).__name__}: {exc}")
            return
        self._saved_at = clock

    def _unsaved(self, why: str) -> None:
        message = f"progress of run {self.run_id} not saved: {why}"
        log.warning("%s", message)
        self.events.warning("progress_unsaved", message)

    # ------------------------------------------------------------------ pausing
    async def ask(
        self,
        question: str,
        *,
        expects: dict[str, Any] | None = None,
        table: Sequence[dict[str, Any]] | None = None,
        diff: tuple[Any, Any] | None = None,
        options: Sequence[str] | None = None,
        assignee: str | None = None,
        deadline: datetime | None = None,
        escalate_to: str | None = None,
    ) -> Any:
        """Ask a person and wait for the answer: the run pauses here, and on resume this call
        returns what they answered (the edited value for a review). A cancellation ends the
        run.

        What the person sees follows from what is asked: ``options`` a choice, ``table`` a
        table, ``diff=(before, after)`` a diff, otherwise a form (``expects`` its schema). A
        table or diff with ``expects`` is a review. It is the run's user's to answer unless
        ``assignee`` names someone else (``user:…``, ``role:…``)."""
        ui = "choice" if options else "diff" if diff else "table" if table is not None else "form"
        reason = (
            InterruptReason.CHOICE
            if options
            else InterruptReason.REVIEW
            if ui in ("diff", "table") and expects is not None
            else InterruptReason.QUESTION
        )
        payload: dict[str, Any] | None = None
        if table is not None:
            payload = {"table": list(table)}
        elif diff is not None:
            payload = {"diff": {"before": diff[0], "after": diff[1]}}
        resolution = await self.interrupt(
            content_key("ask", question, ui, list(options or [])),
            payload=payload,
            reason=reason,
            question=question,
            ui=ui,
            expects=expects,
            options=list(options or []),
            assignee=assignee or principal(self.user),
            deadline=deadline,
            escalate_to=escalate_to,
        )
        return answer_of(resolution)

    async def approve(self, call: ToolCall, question: str) -> InterruptResolution:
        """Ask for approval of a tool call (the bridge's pause): ``question`` is governance's
        (``Decision.question``)."""
        return await self.interrupt(
            content_key("approve", call.tool, call.args),
            reason=InterruptReason.APPROVAL,
            question=question,
            ui="approve",
            tool_call=call,
        )

    async def interrupt(
        self, key: str, *, payload: dict[str, Any] | None = None, **fields: Any
    ) -> InterruptResolution:
        """The one pause: an answer already given (a re-run), the framework's own suspension,
        or a :class:`Paused` that ends this attempt. A ``payload`` larger than
        :data:`INLINE_PAYLOAD_BYTES` is stored as a run artifact and travels as
        ``payload_ref``."""
        answered = self.replay.answer(key)
        if answered is not None:
            return answered
        self._asked += 1
        ident = interrupt_id(self.run_id, self.attempt, self._asked)
        if payload is not None:
            data = json.dumps(payload, default=str, separators=(",", ":")).encode()
            if len(data) <= INLINE_PAYLOAD_BYTES:
                fields["payload"] = payload
            else:
                fields["payload_ref"] = await self.agent.harness.runs.put_artifact(
                    self.run_id, data, worker_id=self.worker_id
                )
        interrupt = Interrupt(
            interrupt_id=ident, tenant_id=self.tenant, run_id=self.run_id, **fields
        )
        if self.pending is None:
            self.pending = Pending(key=key, interrupt=interrupt)
        if self.suspend is None:
            raise Paused(self.pending.interrupt)
        value = self.suspend({MARKER: True, **interrupt.awaiting()})
        resolution = InterruptResolution.model_validate(value)
        self.pending = None
        self.replay.record_answer(key, resolution)
        return resolution


#: The key an ``ask`` marks its LangGraph interrupt value with, telling it apart from a
#: graph's own ``interrupt(...)``.
MARKER: Final = "trellis_interrupt"


def interrupt_id(run_id: str, attempt: int, n: int) -> str:
    """Unique per run and attempt, and names its run (``resume`` needs nothing else)."""
    return f"{run_id}.{attempt}.{n}"


def principal(user: str) -> str:
    """A run's user as an assignee (``user:<id>``); one already naming a kind is kept."""
    return user if ":" in user else f"user:{user}"


def run_of(interrupt_id_: str) -> str:
    return interrupt_id_.rsplit(".", 2)[0]


def answer_of(resolution: InterruptResolution) -> Any:
    decision = resolution.decision
    if decision is InterruptDecision.CANCEL:
        raise RunCancelled(f"cancelled by {resolution.reviewer or 'the reviewer'}")
    if decision is InterruptDecision.APPROVE:
        return True
    if decision is InterruptDecision.REJECT:
        return False
    if decision is InterruptDecision.EDIT:
        return resolution.payload
    return resolution.answer


def reason_of(resolution: InterruptResolution) -> str | None:
    """The reviewer's reason given with a decision (``resume(..., "reject", answer="why")``):
    what the model is told about a rejected call."""
    answer = resolution.answer
    return answer.strip() if isinstance(answer, str) and answer.strip() else None


@dataclass(frozen=True, slots=True)
class Tools:
    """``trellis.current().tools``: call this run's tools by name, and ask for hints."""

    runtime: Runtime

    async def call(self, name: str, /, **args: Any) -> Any:
        from trellis.harness.tools.bridge import call  # noqa: PLC0415 - bridge imports runtime

        found = self.runtime.toolbox.get(name)
        if found is None:
            raise ToolError(f"no tool {name!r} in this run", source="tools")
        return (await call(found, args)).output

    async def hints(self, task: str) -> ToolHints:
        """What the memory service suggests for ``task`` among this run's tools; the
        candidates are offered to the model from then on."""
        if self.runtime.run_memory is None:
            raise ConfigurationError("tool hints come from the memory service: set MEMORY_URL")
        hints = await self.runtime.run_memory.tool_hints(task, self.runtime.tool_names())
        self.runtime.offer([t.name for t in hints.tools])
        return hints  # type: ignore[no-any-return]
