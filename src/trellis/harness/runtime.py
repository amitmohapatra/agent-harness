"""``trellis.current()``: the run a tool or a node is executing in.

The runtime is set for the duration of one attempt of one run (a ``ContextVar``, so every
task the framework spawns inherits it). It carries the run's identity, its memory scope, its
tools, its time (:meth:`Runtime.remaining`; inside a tool call, that call's
:attr:`~Runtime.idempotency_key`) and the one way to pause: :meth:`Runtime.ask`.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from collections import Counter
from collections.abc import AsyncIterator, Callable, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Final

from pydantic import BaseModel

from trellis.contracts import (
    AgentPaused,
    ConfigurationError,
    ErrorCategory,
    Interrupt,
    InterruptReason,
    InterruptResolution,
    ModelError,
    Option,
    ToolCall,
    ToolError,
    ToolSpec,
)
from trellis.harness import telemetry
from trellis.harness.asking import Question
from trellis.harness.events import LOG, RunEvents
from trellis.harness.features import Feature
from trellis.harness.history import Said
from trellis.harness.identity import Identity
from trellis.harness.journal import Pending, Replay, content_key
from trellis.runs import LeaseLostError

if TYPE_CHECKING:
    from trellis.harness.agent import Agent
    from trellis.harness.bridges.helpers import Helpers
    from trellis.harness.budget import Budget
    from trellis.harness.clients.memory import RunMemory
    from trellis.harness.governance.decision import Decision
    from trellis.harness.prompts import ResolvedPrompt
    from trellis.harness.sandbox.base import Sandbox
    from trellis.harness.tools.base import Tool
    from trellis.memory import MemoryContext
    from trellis.memory.models import ToolHints

log = logging.getLogger("trellis.run")

#: A table or diff larger than this (compact JSON) travels by reference: stored as a run
#: artifact in agent-runs (``payload_ref``), not inside the question.
INLINE_PAYLOAD_BYTES: Final = 16 * 1024

#: The tag of a framework's own tool that hands the work to another agent
#: (``adapters.base.HELPER``, kept here: the runtime imports no adapter).
HELPER: Final = "helper"
#: The ``RunStart.metadata`` key of a ``delegate`` job's run (``trellis.harness.delegate``):
#: how deep in delegation it is. Such a run only reads.
DELEGATED: Final = "delegated"
#: How often the journal is saved as progress after calls that only read (or model steps);
#: after a side-effecting call it is saved at once.
PROGRESS_SECONDS: Final = 20.0

#: The most one model call — a step, the provider's hosted tools it runs included — may go
#: without an answer when nothing tighter bounds it (a ``call_timeout``, the run's time left):
#: past it the step has stalled, and the run ends with a :class:`ModelTimeout`.
STALL_SECONDS: Final = 600.0
#: The message a stalled step's watchdog cancels the attempt with.
STALLED: Final = "trellis: the model call stalled"


class ModelTimeout(ModelError):
    """A model call (a step) got no answer within its bound: ``min(call timeout, the run's
    time left)``, else :data:`STALL_SECONDS`. A ``TIMEOUT`` error, retryable (a stalled
    provider may answer the next time); ``details``: the ``step`` and its ``seconds``."""

    code = "MODEL_TIMEOUT"
    category = ErrorCategory.TIMEOUT
    retryable = True


_current: ContextVar[Runtime | None] = ContextVar("trellis_runtime", default=None)


@dataclass(frozen=True, slots=True)
class _Call:
    """The tool or model call the code running now is in: when it must end (``time.monotonic``)
    and, for a tool call, its idempotency key."""

    ends: float | None
    key: str | None


_call: ContextVar[_Call | None] = ContextVar("trellis_call", default=None)


def current() -> Runtime | None:
    """The run this code executes in, or ``None`` outside a harness run."""
    return _current.get()


class Paused(AgentPaused):
    """Raised by :meth:`Runtime.ask` to stop the run until a person answers. Frameworks that
    swallow it do not keep the run going: the harness checks the runtime, not the exception."""

    def __init__(self, interrupt: Interrupt) -> None:
        super().__init__(interrupt.question, expects=interrupt.expects, payload=interrupt.payload)
        self.interrupt = interrupt


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
    #: the pushed memory context, as the framework's system message carries it (the run's
    #: own code reads it as :attr:`context`, the harness as :attr:`pushed_context`)
    _context: str | None = field(default=None, repr=False)
    #: whether the run's own code read :attr:`context` (a function target, a graph's node)
    context_read: bool = field(default=False, repr=False)
    #: where memory keeps the conversation and the framework replays it as messages: the
    #: thread's recent messages (``[]``: none yet); ``None`` where it does not
    history: list[Said] | None = None
    #: how far memory covered the question (its ``evidence_status``: ``COMPLETE``,
    #: ``INCOMPLETE``, ``INSUFFICIENT``), when a context was pushed
    evidence: str | None = None
    #: what the pushed memory context showed — its ``bundle_id`` and the ids of the memories
    #: in it (``memory_ids``) — kept with the run's outcome record so a person's verdict on
    #: the run cites them (``RunMemory.cited``); empty when nothing was pushed
    pushed: dict[str, Any] = field(default_factory=dict)
    #: the question the run is working on (memory queries, tool-call tasks)
    task: str = ""
    #: the documents the run's memory context draws its passages from (``document_ids=``;
    #: ``None``: every one its scope may read)
    document_ids: list[str] | None = None
    #: this run's tools by name (the agent's own, the MCP tools, the memory pull tools)
    toolbox: dict[str, Tool] = field(default_factory=dict)
    #: the tools the framework gives the model itself, beside the harness's (an OpenAI agent's
    #: own and hosted tools, its MCP servers' and handoffs', Claude's built-ins and its MCP
    #: servers', a graph's tool nodes', a Deep Agent's sub-agents'), by name: with the
    #: toolbox they are the run's tool set (:meth:`tool_set`)
    native: dict[str, ToolSpec] = field(default_factory=dict)
    #: the sources the run's code added (``agent.tools.use``), beside its agent's
    added: list[Any] = field(default_factory=list)
    #: what each object given to ``agent.tools.use`` added, by its id (the object held, so the
    #: id stays its own): a tool called again in the run joins it once
    joined: dict[int, tuple[Any, list[str]]] = field(default_factory=dict)
    #: the tools offered to the model (``None``: all of them) — the tool hints' candidates,
    #: the memory tools, and every tool this run has used (``used``)
    offered: set[str] | None = None
    #: the tools this run has called, in any attempt (the journal keeps them across a pause)
    used: set[str] = field(default_factory=set)
    #: the run's budget is spent (``budget.reached``: which limit): no new tool call starts,
    #: and the run ends ``PARTIAL``
    budget_spent: str | None = None
    #: the stored prompts this attempt pinned (``Prompts.pin``: a ``ReAct``'s
    #: ``system_prompt=stored(...)``), by their journal key
    prompts: dict[str, ResolvedPrompt] = field(default_factory=dict)
    #: the sandbox this attempt's sandbox tools work in, once one of them ran (``sandbox``)
    sandbox: Sandbox | None = None
    #: set by an adapter whose framework can suspend itself (LangGraph ``interrupt``)
    suspend: Suspend | None = None
    #: the pause this attempt ended on
    pending: Pending | None = None
    #: the Code Mode scripts this attempt ran, each as (when it started, its step): their
    #: nested calls are read back from the gateway's log and recorded at their script's step
    code_mode_scripts: list[tuple[datetime, int | None]] = field(default_factory=list)
    #: the tool calls started on this attempt's stream and not ended there yet, by reference
    #: (``tools.bridge``: each ends once, the attempt's end closing any still under way)
    open_calls: dict[str, str] = field(default_factory=dict)
    #: the latest model reply's calls repeating an earlier one of it (the same tool, the same
    #: arguments), where its framework runs the reply's calls and the harness reads the reply
    #: (OpenAI Agents: ``hooks.openai_agents``), by what each call's tool is handed — its id,
    #: its tool, its arguments as the model wrote them: the earlier call's id, taken once by
    #: the repeat, which does not run (``tools.bridge.same``)
    repeats: dict[tuple[str, str, str], str] = field(default_factory=dict)
    #: the worker holding the run's lease, when a worker runs it, and that lease's length
    #: (what a progress checkpoint's heartbeat extends it by)
    worker_id: str | None = None
    lease_seconds: int | None = None
    #: the lease was lost while saving progress: the run stops and writes nothing more
    lease_lost: bool = False
    #: what this run turned off itself (``agent.run(without=)``, kept with its record; its
    #: sub-agents' runs inherit it) — its agent's are ``agent.without``
    without: frozenset[Feature] = frozenset()
    #: the framework's own run options, as its adapter hands them to its run call: the agent's
    #: (``h.wrap(framework_options=)``) with the run's own over them (``agent.run(
    #: framework_options=)``, kept with its record)
    framework_options: dict[str, Any] = field(default_factory=dict)
    started_at: datetime | None = None
    #: when the run must stop working (``time.monotonic``): what was left of its time limit
    #: and its deadline when the attempt began (``None``: neither)
    ends_at: float | None = None
    #: when the run must have ended (its ``deadline``), which a sub-agent's run inherits
    deadline: datetime | None = None
    #: the run whose tool call this run works in (a sub-agent's run, a nested run): its
    #: journal is kept in the parent's, and saved with the parent's progress
    parent: Runtime | None = None
    #: the most this run may spend: its own (``run(budget=)``, kept with its record), else its
    #: agent's (``h.wrap(budget=)``); ``None``: no cap
    budget: Budget | None = None
    #: how deep this run is in a ``ReAct``'s delegation (a ``delegate`` job's run: 1; 0: not
    #: one): it only reads — a call of a tool that does more is refused (``tools.bridge``)
    delegated: int = 0
    #: the helpers the framework spawns in this attempt (``bridges.helpers``), once its
    #: adapter made them
    helpers: Helpers | None = None
    #: the task the attempt runs in (``agent.cancel`` cancels it), and why it was cancelled
    running_in: asyncio.Task[Any] | None = None
    cancelled: str | None = None
    _asked: int = 0
    _steps: int = 0
    #: the child runs this attempt started, by where it started them and their agent
    _children: Counter[tuple[str, str]] = field(default_factory=Counter, repr=False)
    _saved_at: float | None = None
    #: one progress save at a time, so a later save never lands before an earlier one
    _saving: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    #: the model calls under way, each with the watchdog that ends the attempt when it stalls
    _watched: dict[str, asyncio.TimerHandle] = field(default_factory=dict, repr=False)
    #: the step that stalled (its watchdog cancelled the attempt): the attempt's error
    stalled: ModelTimeout | None = None

    # ------------------------------------------------------------------ the memory context
    @property
    def context(self) -> str | None:
        """The memory context pushed for this run, as the framework's system message carries
        it: a function target whose input is text (and a graph's node, whose input has no
        ``messages``) reads it here, where the harness could not put it into the input."""
        self.context_read = True
        return self._context

    @context.setter
    def context(self, value: str | None) -> None:
        self._context = value

    @property
    def pushed_context(self) -> str | None:
        """:attr:`context` as the harness reads it (not the run's own code reading it)."""
        return self._context

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

    @property
    def trace_run(self) -> str:
        """The run whose trace this run's spans and scores are in: a sub-agent's run is in its
        parent's."""
        return self.run_id if self.parent is None else self.parent.trace_run

    # ------------------------------------------------------------------ services
    @property
    def memory(self) -> MemoryContext:
        """The memory service in this run's scope (the SDK's verbs)."""
        if self.run_memory is None:
            raise ConfigurationError(
                "memory is off for this run: set MEMORY_URL (and leave it out of without=)"
            )
        return self.run_memory.ctx

    def uses(self, feature: Feature) -> bool:
        """Whether ``feature`` is on for this run: neither the run nor its agent turned it off
        (``without=``)."""
        return feature not in self.without and feature not in self.agent.without

    @property
    def tools(self) -> Tools:
        return Tools(self)

    def log(self, message: str, **fields: Any) -> None:
        """A line in the run's log and on its event stream."""
        log.info("%s", message, extra={"run_id": self.run_id, **fields})
        self.events.custom(LOG, message=message, **fields)

    def tool_names(self) -> list[str]:
        """The run's own harness tools (the memory service's pull tools are not candidates)."""
        return [name for name, found in self.toolbox.items() if found.spec.source != "memory"]

    def tool_set(self) -> list[str]:
        """The run's tool set: its harness tools (:meth:`tool_names`) and the framework's own
        (:attr:`native`), each once — what the tool hints, ``tool_search`` and the narrowing
        choose among."""
        return list(dict.fromkeys([*self.tool_names(), *self.native]))

    def offers(self, name: str) -> bool:
        """Whether the model is offered ``name`` now (per turn where the framework allows): a
        tool of the run's tool set the narrowing left out is not; a memory tool always is, and
        so is a tool outside the set (a graph's own state tools, ``read_file``)."""
        if self.offered is None or name in self.offered or name in self.used:
            return True
        found = self.toolbox.get(name)
        if found is not None:
            return found.spec.source == "memory"
        own = self.native.get(name)
        return own is None or HELPER in own.tags  # a helper: never narrowed away

    def offer(self, names: Sequence[str]) -> None:
        """Offer more tools of the run's tool set (a ``tool_search`` found them)."""
        if self.offered is not None:
            self.offered.update(n for n in names if n in self.toolbox or n in self.native)

    def child_number(self, place: str, agent_id: str) -> int:
        """How many runs of ``agent_id`` this attempt started at ``place`` (a tool call's key,
        or the run's id) before this one: the same in every attempt that makes the same
        calls, so a re-run finds each child again (``subagents.child_run``)."""
        n = self._children[(place, agent_id)]
        self._children[(place, agent_id)] = n + 1
        return n

    def next_step(self, count: int = 1) -> int:
        """The number of the run's next call — the first of ``count`` calls made at once,
        numbered in their order."""
        first = self._steps + 1
        self._steps += count
        return first

    # ------------------------------------------------------------------ time
    @property
    def idempotency_key(self) -> str | None:
        """Inside a tool call: its idempotency key — the same for that call in every attempt
        of the run (re-run after a pause or a crash), different for every other call (another
        run's, a second identical call of this run). Hand it to the service the tool calls,
        which then performs a repeated request once (OpenAPI tools send it as
        ``Idempotency-Key``). ``None`` outside a tool call."""
        call = _call.get()
        return call.key if call is not None else None

    def remaining(self) -> float | None:
        """The seconds the code running now may still take: what is left of the run's time
        (its ``timeout`` and ``deadline``) and of the tool or model call it is in; ``None``
        when nothing limits it."""
        call = _call.get()
        ends = [t for t in (self.ends_at, call.ends if call else None) if t is not None]
        if not ends:
            return None
        return max(0.0, min(ends) - time.monotonic())

    @contextlib.asynccontextmanager
    async def limited(
        self, seconds: float | None, *, key: str | None = None
    ) -> AsyncIterator[None]:
        """The code inside — a tool call (``key``: its idempotency key) or a model call —
        takes at most ``seconds`` (``None``: no limit of its own), or raises ``TimeoutError``;
        the run's own time bounds it as well (the pipeline stops the run)."""
        own = None if seconds is None else time.monotonic() + seconds
        outer = _call.get()
        ends = [t for t in (own, outer.ends if outer else None) if t is not None]
        token = _call.set(_Call(min(ends) if ends else None, key))
        try:
            async with asyncio.timeout(seconds):
                yield
        finally:
            _call.reset(token)

    # ------------------------------------------------------------------ stalls
    def model_started(self, key: str, seconds: float | None = None) -> None:
        """A model call (``key``: the framework's id of it) began: its watchdog ends the
        attempt with a :class:`ModelTimeout` unless :meth:`model_ended` comes within
        ``seconds`` (else :data:`STALL_SECONDS`). The run's own time left bounds it too: the
        run ends ``TIMEOUT`` when that comes first. A key already watched is watched anew."""
        self.model_ended(key)
        bound = STALL_SECONDS if seconds is None else seconds
        loop = asyncio.get_running_loop()
        self._watched[key] = loop.call_later(bound, self._stall, key, bound)

    def model_ended(self, key: str) -> None:
        """The model call ``key`` answered (or failed): its watchdog is off."""
        found = self._watched.pop(key, None)
        if found is not None:
            found.cancel()

    def unwatched(self) -> None:
        """The attempt ended: every watchdog still on is off."""
        for handle in self._watched.values():
            handle.cancel()
        self._watched.clear()

    def _stall(self, key: str, seconds: float) -> None:
        self._watched.pop(key, None)
        task = self.running_in
        if self.stalled is not None or task is None or task.done():
            return
        self.stalled = ModelTimeout(
            f"the model did not answer within {seconds:g}s: the step stalled",
            details={"step": key, "seconds": seconds},
            source=self.agent.adapter.name,
        )
        log.warning("run %s: %s", self.run_id, self.stalled.message)
        self.events.warning(ModelTimeout.code.lower(), self.stalled.message)
        attributes = {"trellis.step": key, "trellis.step.seconds": seconds}
        telemetry.failure(
            self.trace_run, self.run_id, "model_timeout", self.stalled.message, attributes
        )
        task.cancel(STALLED)

    # ------------------------------------------------------------------ progress
    async def progress(self, *, now: bool) -> None:
        """Save the journal as the run's progress checkpoint, on a heartbeat of the worker's
        lease (a worker's run only: nobody resumes an in-process run after its process died).
        ``now`` after a side-effecting call — the attempt after a crash replays it instead of
        running it again; otherwise at most every :data:`PROGRESS_SECONDS`. A journal too large
        for a checkpoint travels as a run artifact (``Journal.checkpoint``). A refused save is a
        warning (the run goes on; the next save tries again); a lost lease stops the run
        (``LeaseLostError``). A sub-agent's run saves its parent's progress, which holds its
        journal. Saves made at once (tools running together) go one at a time."""
        try:
            if self.parent is not None:
                await self.parent.progress(now=now)
                return
            async with self._saving:
                await self._saved(now=now)
        except LeaseLostError:
            self.lease_lost = True
            raise

    async def _saved(self, *, now: bool) -> None:
        if self.worker_id is None or self.lease_seconds is None:
            return
        clock = time.monotonic()
        if not now and self._saved_at is not None and clock - self._saved_at < PROGRESS_SECONDS:
            return
        runs = self.agent.harness.runs
        try:
            checkpoint = await self.replay.journal.checkpoint(
                runs.artifacts, self.run_id, worker_id=self.worker_id, tenant=self.tenant
            )
            await runs.heartbeat(
                self.run_id,
                self.worker_id,
                lease_seconds=self.lease_seconds,
                checkpoint=checkpoint,
                tenant=self.tenant,
            )
        except LeaseLostError:
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
        form: type[BaseModel] | None = None,
        table: Sequence[dict[str, Any]] | None = None,
        diff: tuple[Any, Any] | None = None,
        options: Sequence[Option | str] | None = None,
        multiple: bool = False,
        ui_schema: dict[str, Any] | None = None,
        component: str | None = None,
        props: dict[str, Any] | None = None,
        assignee: str | None = None,
        deadline: datetime | None = None,
        escalate_to: str | None = None,
    ) -> Any:
        """Ask a person and wait for the answer: the run pauses here, and on resume this call
        returns what they answered (the edited value for a review; a list of values with
        ``multiple``; an instance of ``form``, a pydantic model whose JSON Schema is
        ``expects``). A cancellation ends the run.

        What the person sees follows from what is asked (``trellis.harness.asking.Question``):
        ``options`` (strings or ``Option(value, label, description)``) a choice, ``table`` a
        table, ``diff=(before, after)`` a diff, otherwise a form (``expects`` or ``form`` its
        schema, ``ui_schema`` its widget hints). ``component`` names your own screen and
        ``props`` its data: a surface that has it renders it, any other the control above. A
        table or diff with ``expects`` is a review. It is the run's user's to answer unless
        ``assignee`` names someone else (``user:…``, ``role:…``). A question that cannot be
        asked as given (an ``expects`` that is not a JSON Schema, ``props`` without a
        ``component``...) is refused here (``ConfigurationError``, saying why); an answer
        must fit it (``Agent.resume``)."""
        asked = Question(
            question,
            expects=expects,
            form=form,
            table=table,
            diff=diff,
            options=list(options or []),
            multiple=multiple,
            ui_schema=ui_schema,
            component=component,
            props=props,
            assignee=assignee or principal(self.user),
            deadline=deadline,
            escalate_to=escalate_to,
        )
        resolution = await self.interrupt(asked.key, payload=asked.payload, **asked.fields())
        return asked.answer(resolution)

    async def approve(self, call: ToolCall, decision: Decision) -> InterruptResolution:
        """Ask for approval of a tool call (the bridge's pause): the question is governance's
        (``Decision.question``), whose it is and the screen it is reviewed on a hook's ``Ask``
        (else anyone's, the approval control)."""
        return await self.interrupt(
            content_key("approve", call.tool, call.args),
            reason=InterruptReason.APPROVAL,
            question=decision.question,
            ui="approve",
            tool_call=call,
            assignee=decision.assignee,
            component=decision.component,
            props=None if decision.props is None else dict(decision.props),
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
                fields["payload_ref"] = await self.agent.harness.runs.artifacts.upload(
                    self.run_id, data, worker_id=self.worker_id, tenant=self.tenant
                )
        interrupt = Interrupt(
            interrupt_id=ident, tenant_id=self.tenant, run_id=self.run_id, **fields
        )
        # calls made at once may each ask: the run pauses on the first, the others ask again
        # when it resumes
        if self.pending is None:
            self.pending = Pending(key=key, interrupt=interrupt)
        if self.suspend is None:
            raise Paused(self.pending.interrupt)
        # the framework's own pause (LangGraph's ``interrupt``): its resume is answered from the
        # journal, above (the attempt files the answer there first). LangGraph also hands a
        # call's answers back by their position in it, and a call asks only what is still open:
        # the earlier questions' answers it hands back are passed over until it pauses
        filed = {
            a.get("interrupt_id")
            for answers in self.replay.journal.answers.values()
            for a in answers
        }
        while True:
            value = self.suspend({MARKER: True, **interrupt.awaiting()})
            handed = InterruptResolution.model_validate(value)
            assert handed.interrupt_id in filed, "a resume's answer is filed in the journal"


#: The key an ``ask`` marks its LangGraph interrupt value with, telling it apart from a
#: graph's own ``interrupt(...)``.
MARKER: Final = "trellis_interrupt"
#: The key of the LangGraph interrupt a call waiting its turn ends its task with when an
#: earlier call of the same model step paused (``middleware.HarnessTools``): no question —
#: the call runs, in its turn, when the run resumes.
DEFERRED: Final = "trellis_deferred"


def delegated_of(metadata: dict[str, Any]) -> int:
    """How deep in delegation a run is, from its record's metadata (:data:`DELEGATED`)."""
    depth = metadata.get(DELEGATED)
    return depth if isinstance(depth, int) and not isinstance(depth, bool) and depth > 0 else 0


def interrupt_id(run_id: str, attempt: int, n: int) -> str:
    """Unique per run and attempt, and names its run (``resume`` needs nothing else)."""
    return f"{run_id}.{attempt}.{n}"


def principal(user: str) -> str:
    """A run's user as an assignee (``user:<id>``); one already naming a kind is kept."""
    return user if ":" in user else f"user:{user}"


def run_of(interrupt_id_: str) -> str:
    return interrupt_id_.rsplit(".", 2)[0]


def reason_of(resolution: InterruptResolution) -> str | None:
    """The reviewer's reason given with a decision (``resume(..., "reject", answer="why")``,
    else the decision's ``comment``): what the model is told about a rejected call."""
    for given in (resolution.answer, resolution.comment):
        if isinstance(given, str) and given.strip():
            return given.strip()
    return None


@dataclass(frozen=True, slots=True)
class Tools:
    """``trellis.current().tools`` (a function target's ``agent.tools``): call this run's tools,
    add more to it, and ask for hints."""

    runtime: Runtime

    async def call(self, tool: Any, /, **args: Any) -> Any:
        """Call one of the run's tools with ``args`` — governed, journaled, recorded, as every
        call is (``tools.bridge``) — and return what it answered: a tool by name (the run's
        toolbox: its builder's tools, the MCP tools the virtual key allows, the memory tools,
        what :meth:`use` added), or a tool itself (``tool(fn)``, which joins the run first)."""
        from trellis.harness.tools.bridge import call  # noqa: PLC0415 - bridge imports runtime

        name = tool if isinstance(tool, str) else (await self.use(tool))[0]
        found = self.runtime.toolbox.get(name)
        if found is None:
            raise ToolError(f"no tool {name!r} in this run", source="tools")
        return (await call(found, args)).output

    async def use(self, *sources: Any) -> list[str]:
        """Add the tools of ``sources`` — ``tool(fn)`` (a function is one), ``mcp(...)``,
        ``openapi(...)``, ``sandbox().tools()``, ``a2a(url)``, a wrapped agent (a sub-agent) —
        to this run's toolbox, for :meth:`call`: the names they add, in order (a function
        target's own tools; a builder's are declared on it)."""
        from trellis.harness.tools.sources import as_source  # noqa: PLC0415 - imports runtime

        names: list[str] = []
        for given in sources:
            known = self.runtime.joined.get(id(given))
            if known is None:
                source = as_source(given)
                self.runtime.added.append(source)
                resolved = list(await source.resolve())
                for found in resolved:
                    self.runtime.toolbox.setdefault(found.spec.name, found)
                known = self.runtime.joined[id(given)] = (given, [t.spec.name for t in resolved])
            names.extend(known[1])
        return names

    async def hints(self, task: str) -> ToolHints:
        """What the memory service suggests for ``task`` among this run's tool set (its harness
        tools and the framework's own); the candidates are offered to the model from then
        on."""
        if self.runtime.run_memory is None:
            raise ConfigurationError("tool hints come from the memory service: set MEMORY_URL")
        hints = await self.runtime.run_memory.tool_hints(task, self.runtime.tool_set())
        self.runtime.offer([t.name for t in hints.tools])
        return hints  # type: ignore[no-any-return]
