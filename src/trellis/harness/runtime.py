"""``trellis.current()``: the run a tool or a node is executing in.

The runtime is set for the duration of one attempt of one run (a ``ContextVar``, so every
task the framework spawns inherits it). It carries the run's identity, its memory scope, its
tools and the one way to pause: :meth:`Runtime.ask`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Final, Literal

from trellis.contracts import (
    AgentPaused,
    ArtifactRef,
    ConfigurationError,
    Interrupt,
    InterruptDecision,
    InterruptReason,
    InterruptResolution,
    ToolCall,
    ToolError,
)
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

#: A table this long travels by reference (``payload_ref``), not inside the question.
TABLE_INLINE_ROWS: Final = 50

UI = Literal["approve", "form", "table", "diff", "choice"]

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
    #: this run's tools by name (the agent's own sources and the memory pull tools)
    toolbox: dict[str, Tool] = field(default_factory=dict)
    #: set by an adapter whose framework can suspend itself (LangGraph ``interrupt``)
    suspend: Suspend | None = None
    #: the pause this attempt ended on
    pending: Pending | None = None
    #: a Code Mode script ran: its nested calls are read back from the gateway's log
    used_code_mode: bool = False
    #: the worker holding the run's lease, when a worker runs it
    worker_id: str | None = None
    started_at: datetime | None = None
    _asked: int = 0
    _steps: int = 0

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
            raise ConfigurationError(
                f"memory is off for agent {self.agent_id!r} (wrap it with memory='read' or "
                "'read_write', and set MEMORY_URL)"
            )
        return self.run_memory.ctx

    @property
    def tools(self) -> Tools:
        return Tools(self)

    def log(self, message: str, **fields: Any) -> None:
        """A line in the run's log and on its event stream."""
        log.info("%s", message, extra={"run_id": self.run_id, **fields})
        self.events.custom(LOG, message=message, **fields)

    def next_step(self) -> int:
        self._steps += 1
        return self._steps

    # ------------------------------------------------------------------ pausing
    async def ask(
        self,
        question: str,
        *,
        ui: UI | None = None,
        expects: dict[str, Any] | None = None,
        table: Sequence[dict[str, Any]] | None = None,
        options: Sequence[str] | None = None,
        assignee: str | None = None,
        deadline: datetime | None = None,
        escalate_to: str | None = None,
    ) -> Any:
        """Ask a person and wait for the answer: the run pauses here, and on resume this call
        returns what they answered (``True``/``False`` for an approval, the edited value for
        an edit). A cancellation ends the run."""
        chosen: UI = ui or ("choice" if options else "table" if table else "form")
        reason = (
            InterruptReason.CHOICE
            if options
            else InterruptReason.REVIEW
            if chosen in ("diff", "table") and expects is not None
            else InterruptReason.QUESTION
        )
        payload, payload_ref = await self._table(table)
        resolution = await self.interrupt(
            content_key("ask", question, chosen, list(options or [])),
            reason=reason,
            question=question,
            ui=chosen,
            expects=expects,
            options=list(options or []),
            payload=payload,
            payload_ref=payload_ref,
            assignee=assignee,
            deadline=deadline,
            escalate_to=escalate_to,
        )
        return answer_of(resolution)

    async def approve(self, call: ToolCall, reason: str) -> InterruptResolution:
        """Ask for approval of a tool call (the bridge's pause)."""
        return await self.interrupt(
            content_key("approve", call.tool, call.args),
            reason=InterruptReason.APPROVAL,
            question=f"Approve {call.tool}? {reason}",
            ui="approve",
            tool_call=call,
        )

    async def interrupt(self, key: str, **fields: Any) -> InterruptResolution:
        """The one pause: an answer already given (a re-run), the framework's own suspension,
        or a :class:`Paused` that ends this attempt."""
        answered = self.replay.answer(key)
        if answered is not None:
            return answered
        self._asked += 1
        interrupt = Interrupt(
            interrupt_id=interrupt_id(self.run_id, self.attempt, self._asked),
            tenant_id=self.tenant,
            run_id=self.run_id,
            **fields,
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

    async def _table(
        self, table: Sequence[dict[str, Any]] | None
    ) -> tuple[dict[str, Any] | None, ArtifactRef | None]:
        if table is None:
            return None, None
        rows = list(table)
        if len(rows) <= TABLE_INLINE_ROWS:
            return {"table": rows}, None
        return None, await self.agent.harness.artifacts.put_json(rows)


#: The key an ``ask`` marks its LangGraph interrupt value with, telling it apart from a
#: graph's own ``interrupt(...)``.
MARKER: Final = "trellis_interrupt"


def interrupt_id(run_id: str, attempt: int, n: int) -> str:
    """Unique per run and attempt, and names its run (``resume`` needs nothing else)."""
    return f"{run_id}.{attempt}.{n}"


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
        """What the memory service suggests for ``task`` among this run's tools."""
        if self.runtime.run_memory is None:
            raise ConfigurationError("tool hints need memory='read' or 'read_write'")
        return await self.runtime.run_memory.tool_hints(task, list(self.runtime.toolbox))
