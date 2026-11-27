"""What a paused run needs to continue when its framework cannot keep the state itself.

A LangGraph graph with a checkpointer resumes where it stopped. Every other target is run
again from its input, and the journal makes that re-run land in the same place: a question
already answered returns its answer, and a tool call already made returns its recorded
output instead of running twice. Entries are keyed by *content* (the question, or the tool and
its arguments) and consumed in order, so the n-th identical call gets the n-th recorded
result, and a re-planned call the person never saw is asked about again rather than matched
to someone else's approval.

A call that does more than read is marked *started* before it runs (and the journal saved), so
the attempt after a crash knows which call was running when the worker died: it was neither
finished nor recorded, and whether it took effect is not known (``tools/bridge.py``). A call
whose outcome is not an output — it timed out, its effect is unknown — is recorded as that
outcome, so a re-run tells the model the same thing and runs nothing: every call's record is
kept in an envelope (:data:`RECORD`), so an output is never read as one.

The journal is the run's checkpoint (``RunRecord.checkpoint``): the run store keeps it with the
pause and hands it to whichever worker resumes the run, so a resume on another machine repeats
no question and no side effect. It is saved as segments (agent-runs' payload objects,
``trellis.runs.payloads``): each save uploads only what changed since the last one the run
store took (:func:`delta`), nothing at all when nothing did, and the checkpoint lists the
segments in order (:meth:`Journal.save`); a resume reads them all at once and applies them in
order (:meth:`Journal.read`). Every :data:`MAX_SEGMENTS` saves, one segment holds the whole
journal again, so a resume reads a bounded number. A
sub-agent's run working inside one of the run's tool calls keeps its journal in the run's
(``children``), so the run's progress saves the child's too; the run's sandbox is named here as
soon as it exists (``sandbox``), so every later attempt works in that one
(``trellis.harness.sandbox``); a graph that checkpoints into the run
(``middleware.RunCheckpointer``) keeps its checkpoints here (``graph``), and a large tool
result the model read a preview of is kept whole here (``results``), for ``read_file``.

Every call that ran is also on the run's trajectory (``trajectory``), in order: the calls and
outcomes its tool records are made of, kept across its attempts for the evaluators
(``EvalCase.trajectory``), each output as the event stream previews it (``events.preview``).

A graph that keeps a checkpoint (``middleware.RunCheckpointer``, or a checkpointer of its own)
continues from it rather than from its input, so an earlier step's records are not what a call
of a later step repeats: each tool call's record, and each answer a call read, is marked with
the step it was made in — the graph checkpoint the step started from (``steps``,
:attr:`Replay.where`). An attempt passes over the records of the steps before the checkpoint it
resumed from (:attr:`Replay.resumed`), so a later step's identical call runs, and is asked about,
as a new one; and as a graph that checkpoints into its run moves past a step, that step's
records are dropped (:meth:`Replay.passed`): its journal holds the graph's checkpoints and
the records of the step under way, however long the run.

Calls made at once (a framework running several tools together) take their occurrences in the
order they asked: identical calls run one after another (:meth:`Replay.exclusive`), so each has
its own occurrence and a re-run hands each the output it got, whatever order they finished in.

A model asked for a step again rarely repeats its call word for word, and a framework that does
not resume in place has it make the paused call again: run again from its input (OpenAI Agents,
a graph with no checkpointer), or told to repeat it in the session it kept (Claude, an ACP
agent). The call a pause came from within (a tool's own question, asked while it ran) is
therefore kept as the model made it, with its place among the run's tool calls (:class:`Made`,
``Pending.inside``), and the call of that tool made at that place when the run resumes runs as
it (:meth:`Replay.made`): the question asked is the one the person answered, and their answer
is for the arguments they saw. Nothing else is matched by place: any other call with other
arguments is a new call, and an approval is asked about again.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
from collections import defaultdict
from collections.abc import Callable, Iterator, Sequence
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, Final

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from trellis.contracts import (
    HarnessError,
    Interrupt,
    InterruptDecision,
    InterruptResolution,
    ToolCall,
    ToolOutcome,
)
from trellis.harness.events import preview
from trellis.runs import NotFoundError
from trellis.runs.payloads import SEGMENTS, PayloadRef, adecode, aencode

if TYPE_CHECKING:
    from trellis.harness.runs import RunArtifacts

#: Segments a checkpoint lists before a save writes the whole journal as one again: a resume
#: reads at most this many, in one request. Measured on a 50-step run of 1.5 MiB: 16 resumes
#: in 27 ms (64: 45 ms) for 0.4 MB stored (64: 0.1 MB; a whole journal per save: 23 MB).
MAX_SEGMENTS: Final = 16
#: The key a call's record is filed under, whatever its outcome: ``{RECORD: {"status": ...,
#: "output": ..., "metadata": ...}}`` (``tools.bridge.journal_record``; read back by its
#: ``journal_outcome``) — an envelope around every output, so one a tool shapes like a record
#: is still its output.
RECORD: Final = "trellis_record"
#: The key the release before filed an outcome that was not an output under, its outputs kept
#: bare: read still (a run paused before an upgrade), never written.
OUTCOME: Final = "trellis_outcome"


def delta(old: Any, new: Any) -> dict[str, Any] | None:
    """What turns ``old`` into ``new`` (``None``: nothing): ``{"$r": value}`` replaces,
    ``{"$a": items}`` appends to a list that kept its start, ``{"$d": {key: delta},
    "$x": [key, ...]}`` changes an object's keys and drops some. A journal's lists grow at
    their end and its objects gain keys, so a save's delta is about what the step added."""
    if old == new:
        return None
    if isinstance(old, dict) and isinstance(new, dict):
        changed = {k: d for k, v in new.items() if (d := delta(old.get(k, _ABSENT), v))}
        node: dict[str, Any] = {"$d": changed}
        if gone := [k for k in old if k not in new]:
            node["$x"] = gone
        return node
    if isinstance(old, list) and isinstance(new, list) and new[: len(old)] == old:
        return {"$a": new[len(old) :]}
    return {"$r": new}


def apply(base: Any, node: dict[str, Any]) -> Any:
    """``base`` changed by a :func:`delta`."""
    if "$r" in node:
        return node["$r"]
    if "$a" in node:
        return [*base, *node["$a"]]
    changed = dict(base) if isinstance(base, dict) else {}
    for key in node.get("$x", ()):
        changed.pop(key, None)
    for key, sub in node["$d"].items():
        changed[key] = apply(changed.get(key), sub)
    return changed


def _spending(dumped: dict[str, Any], journal: Journal) -> dict[str, Any] | None:
    """What turns ``dumped`` (a dump of the journal, as a save left it) into it with the
    journal's spending now — its ``spent``, and its sub-agents' runs' (one ``dumped`` does
    not hold yet: a journal of its spending alone) — and nothing else of it (``None``: the
    spending is the same): a :func:`delta`, made in O(the run's sub-agents' runs)."""
    changed: dict[str, Any] = {}
    if dumped.get("spent", {}) != journal.spent:
        changed["spent"] = {"$r": dict(journal.spent)}
    held = dumped.get("children") or {}
    children: dict[str, Any] = {}
    for run_id, child in journal.children.items():
        if run_id in held:
            if (node := _spending(held[run_id], child)) is not None:
                children[run_id] = node
        else:
            children[run_id] = {"$r": _spent_alone(child).dump()}
    if children:
        changed["children"] = {"$d": children}
    return {"$d": changed} if changed else None


def _spent_alone(journal: Journal) -> Journal:
    """A journal of ``journal``'s spending alone, its sub-agents' runs' included."""
    return Journal(
        spent=dict(journal.spent),
        children={run_id: _spent_alone(child) for run_id, child in journal.children.items()},
    )


class _Absent:
    """A key an object did not have (unequal to any JSON value)."""


_ABSENT: Final = _Absent()


class Save(BaseModel):
    """A save made ready: the checkpoint to send, and what the journal was then."""

    model_config = ConfigDict(frozen=True)

    checkpoint: dict[str, Any]
    segments: list[PayloadRef]
    dumped: dict[str, Any]


#: The tool calls whose approval a person answered, read in this context while
#: :func:`approvals_read` watches.
_APPROVED: ContextVar[list[ToolCall] | None] = ContextVar("trellis_approvals_read", default=None)
#: The call the code running now decides or runs (its key): :meth:`Replay.calling`.
_CALLING: ContextVar[str | None] = ContextVar("trellis_calling", default=None)
#: That call's identity, as its framework names it (its call id): :meth:`Replay.calling`.
_IDENT: ContextVar[str | None] = ContextVar("trellis_call_ident", default=None)


def calling_ident() -> str | None:
    """The framework's id of the call the code running now decides or runs, if it named one
    (:meth:`Replay.calling`): the call a pause asked about (``Pending.ident``)."""
    return _IDENT.get()


@contextlib.contextmanager
def approvals_read() -> Iterator[list[ToolCall]]:
    """The tool calls whose approval (``Runtime.approve``: governance's question about a call)
    the code it holds read a person's recorded answer to, tasks it starts included: whether a
    person approved — or denied — the call that code decided, whatever made governance ask (a
    catalog rule, a hook's ``Ask``, a rule about the arguments a hook rewrote), the call as
    asked about. Any other question answered meanwhile (a hook's own ``ask``, ``ask_user``, a
    form) is no answer about a call, and is not among them."""
    read: list[ToolCall] = []
    token = _APPROVED.set(read)
    try:
        yield read
    finally:
        _APPROVED.reset(token)


def approval_read(call: ToolCall) -> None:
    """A person's recorded answer to the approval of ``call`` was read: for
    :func:`approvals_read`."""
    watching = _APPROVED.get()
    if watching is not None:
        watching.append(call)


def content_key(kind: str, *parts: Any) -> str:
    """A stable key for a question or a call: same content, same key, across processes."""
    text = json.dumps([kind, *parts], sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.blake2b(text.encode(), digest_size=12).hexdigest()


def call_key(tool: str, args: Any) -> str:
    """What a tool call is — ``tool`` called with ``args`` — as one key, wherever it is made:
    the journal files the call's record under it, a pending question names its call by it,
    and the call's id and idempotency key are the run's id, this key and which time the run
    makes the call (``tools.bridge``)."""
    return content_key("call", tool, args)


class Made(BaseModel):
    """A tool call as the model made it: its arguments, and its place among the run's tool
    calls, as a run from its input counts them (0: the first)."""

    model_config = ConfigDict(frozen=True, extra="allow")

    tool: str
    args: dict[str, Any]
    at: int


class Pending(BaseModel):
    """The interrupt the run is waiting on, and the key its answer is filed under."""

    model_config = ConfigDict(frozen=True, extra="allow")

    key: str
    interrupt: Interrupt
    #: The tool call the pause came from within (a question the tool asked while it ran), on a
    #: framework that does not resume in place: its model makes the call again.
    inside: Made | None = None

    #: The framework's own handle on the pause (LangGraph's interrupt id), when it has one.
    native_id: str | None = None
    #: The framework's id of the call the pause asked about, when it named one: whose answer
    #: it is (``Journal.answered_by``).
    ident: str | None = None
    #: The framework's serialised run, when it resumes from one (an OpenAI Agents
    #: ``RunState`` paused on a ``needs_approval`` tool).
    native_state: dict[str, Any] | None = None

    @property
    def call(self) -> ToolCall | None:
        """The tool call the pause is about: the one waiting for an approval, else the one it
        came from within — what a session resumed from the pause is told to call again."""
        if self.interrupt.tool_call is not None or self.inside is None:
            return self.interrupt.tool_call
        return ToolCall(tool=self.inside.tool, args=self.inside.args)


def governance_held(pending: Pending) -> bool:
    """Whether ``pending`` is governance's own approval of a call that a framework's gate held
    (OpenAI Agents' ``needs_approval``, ``bridge.asks_first``): its answer is filed as a harness
    pause's, read where the call asks again when the framework runs it."""
    call = pending.interrupt.tool_call
    return (
        pending.native_state is not None
        and call is not None
        and pending.key == content_key("approve", call.tool, call.args)
    )


class Journal(BaseModel):
    """Answers and tool outputs recorded so far, plus the pause currently open.

    A field it does not know — written by a newer server of a rolling upgrade — is kept and
    saved again as it was (``extra="allow"``, here and in :class:`Pending` and :class:`Made`),
    never refused and never dropped: servers on two versions share one run store."""

    model_config = ConfigDict(extra="allow")

    answers: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    calls: dict[str, list[Any]] = Field(default_factory=dict)
    #: the tools the run has called (they stay offered to the model after a pause)
    used: list[str] = Field(default_factory=list)
    #: the run's trajectory: every harness tool call that ran, in order across its attempts,
    #: with its outcome — the call and outcome its tool record is made of (what evaluators read:
    #: ``EvalCase.trajectory``); a call replayed from the journal is in it once
    trajectory: list[tuple[ToolCall, ToolOutcome]] = Field(default_factory=list)
    #: the calls that did more than read (``content_key("call", tool, args)``), each with the
    #: graph step it was made in (the checkpoint the step started from; ``None``: no
    #: checkpointed graph's step), in the order made: what a rewind would make again
    effects: list[tuple[str | None, str]] = Field(default_factory=list)
    #: in a run rewound from another (:meth:`rewound`): the calls that one had made past the
    #: checkpoint this one continues from — each asks a person before it is made again,
    #: whatever was allowed before (``tools.bridge``), and leaves the list once asked
    repeats: list[str] = Field(default_factory=list)
    #: the tools a reviewer approved for the rest of the run (``remember="run"``; ``"<tool> for
    #: <assignee>"`` when the approval was asked of someone): their later calls asked of the
    #: same are not asked about again in this run
    remembered: list[str] = Field(default_factory=list)
    #: the calls a framework's own approval gate approved (LangChain's
    #: ``HumanInTheLoopMiddleware``, Deep Agents' ``interrupt_on``, an OpenAI Agents
    #: ``needs_approval`` tool), by content: governance counts that approval, once each, and
    #: does not ask again (``approved_in_framework``)
    approved: list[str] = Field(default_factory=list)
    #: a run in its framework's plan mode (Claude Code's ``permission_mode="plan"``) whose plan
    #: was let through (``ExitPlanMode`` ran: a person approved it, as Claude Code asks): its
    #: calls are governed as ever from then on, in any attempt (``adapters.claude``)
    plan_approved: bool = False
    #: the approval keys of the plans it presented (``content_key("approve", ...)``): a person's
    #: approval of any of them ends plan mode, however the framework presents it again
    plans: list[str] = Field(default_factory=list)
    #: the calls a person rejected at a framework's own approval gate (LangChain's
    #: ``HumanInTheLoopMiddleware``), by content: each is that person's decision — the failed
    #: message the gate answers it with is no failed call of the run (:meth:`rejected_by_person`)
    rejected: list[str] = Field(default_factory=list)
    #: the tool calls the run's answered pauses came from within, which a later attempt's
    #: model makes again, each at its place (:meth:`Replay.made`)
    again: list[Made] = Field(default_factory=list)
    #: how many tool calls the run has made to their end: the place of the next, where an
    #: attempt goes on from the framework's own state (its session, its serialised run)
    place: int = 0
    #: how far the calls made before the call the run paused on read each record and answer
    #: (``"c:<key>"``, ``"a:<key>"``: its cursor), where an attempt goes on from the
    #: framework's own state: it does not make those calls again, so the call it makes again
    #: reads its own records, not an earlier identical call's (:meth:`Replay.paused`)
    cursors: dict[str, int] = Field(default_factory=dict)
    #: per call record (``calls``, in its order), the answers and records the call read
    #: besides its own (``"a:<key>"``: its approval...), by cursor: a re-run that replays the
    #: call reads past them too, so an identical call after it reads its own (:meth:`Replay.call`)
    reads: dict[str, list[dict[str, int]]] = Field(default_factory=dict)
    #: per answer (``answers``, in its order), the framework's id of the call it answered, when
    #: it named one: a rejection is that call's — an attempt that goes on from the framework's
    #: own state gives it to that call only (the paused call made again under its id, or still
    #: waiting), and a new call, even an identical one, is asked (:meth:`Replay.answer`)
    answered_by: dict[str, list[str | None]] = Field(default_factory=dict)
    #: per call record (``calls``, in its order), the framework's id of the call it is, when it
    #: named one: in the step its framework makes again, or an attempt that goes on from the
    #: framework's own state, the record is that call's — an identical call under another id
    #: (the model's reply made it twice) is another call, not a replay (:meth:`Replay.occurrence`)
    called_by: dict[str, list[str | None]] = Field(default_factory=dict)
    #: per call, how many of its occurrences started: one more than ``calls`` records was
    #: running when the attempt ended without recording it (a crash)
    started: dict[str, int] = Field(default_factory=dict)
    #: the journals of the sub-agents' runs working inside the run's tool calls now, by run id
    children: dict[str, Journal] = Field(default_factory=dict)
    #: the runs of the helpers handed on past the attempt that started them (a Claude subagent
    #: its conversation's live session keeps at work in the background:
    #: ``bridges.helpers.Helpers.detached``): whoever keeps each going ends it, never a later
    #: attempt — on any worker
    handed_on: list[str] = Field(default_factory=list)
    #: the runs the conversation's earlier turns handed on (their ``handed_on``) whose end the
    #: session this turn continues was not told yet: carried with the session from turn to
    #: turn, on any worker (``pipeline._session_only``), so a turn the session's process did
    #: not serve tells what each that ended came to (``adapters.claude``), once
    in_background: list[str] = Field(default_factory=list)
    #: the tool records queued for the memory service and not known to have landed yet, by
    #: call, as the write spool keeps them (``Agent.record_tool``): saved with the call, so
    #: the attempt after a crash sends them again (``Agent.resent_records``) — the service
    #: stores a record once per run, step, tool and arguments, and these are the same
    unrecorded: dict[str, dict[str, Any]] = Field(default_factory=dict)
    #: the framework's own conversation, when it keeps one the next attempt continues (a
    #: Claude Agent SDK session): what it already did — its built-in tools — is not done again
    session: str | None = None
    #: the reads a framework ran itself and reported (Claude Code's built-ins: ``bridge.observed``)
    #: journaled since the run's last write, by call key: the write drops their records — a
    #: read is not answered across a write with what it returned before it (``Replay.wrote``)
    reread: list[str] = Field(default_factory=list)
    #: that session's running cost, as the last of its results this run was charged carried
    #: it (a Claude Agent SDK result's cost is its session's, the turns and the attempts
    #: before it included): its next result is charged what the cost grew by
    session_cost: float | None = None
    #: the transcripts of the framework's own sessions, by session, as its session store
    #: mirrors them (``bridges.claude.JournalSessions``): a resume on any machine continues them
    transcripts: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    #: the sandbox the run's sandbox tools work in, as its reference (``sandbox.SandboxRef``):
    #: recorded as soon as it exists, so every later attempt works in the same one
    sandbox: dict[str, Any] | None = None
    #: the graph's latest checkpoint and its pending writes, by namespace, with the earlier
    #: ones (its ``history``), when the graph keeps them in the run
    #: (``middleware.RunCheckpointer``): a resume continues where it stopped
    graph: dict[str, Any] | None = None
    #: the step each tool call record (``"c:<key>"``) and each answer a call read
    #: (``"a:<key>"``) was made in, in their order: the graph checkpoint the step started from
    #: (``None``: not made in a checkpointed graph's step)
    steps: dict[str, list[str | None]] = Field(default_factory=dict)
    #: what the run has spent so far, its sub-agents' runs included (``budget.py``): tokens,
    #: cached tokens, cost, model calls, tool calls — a resume goes on from it
    spent: dict[str, float] = Field(default_factory=dict)
    #: the large tool results the model read a preview of, whole, by the path it reads them at
    #: (``tools.results``: OpenAI Agents and Claude, which have no cut of their own)
    results: dict[str, str] = Field(default_factory=dict)
    #: the files the run kept for its user (``Runtime.save_file``): name, size, media type and
    #: what made each — their bytes are in the file store
    files: list[dict[str, Any]] = Field(default_factory=list)
    #: the files its framework's own file tools wrote directly in the folder it works in (a
    #: bare name there: what a user names without a place) and that are not kept for its user
    #: yet: each by its whole path, with the SHA-256 of what the call left there (empty: in
    #: the run's own sandbox) — kept once, when the attempt pauses or the run succeeds
    #: (``Runtime.wrote_file``)
    written: dict[str, str] = Field(default_factory=dict)
    pending: Pending | None = None
    #: the segments the run store holds, and the journal they make: what a save adds to
    #: (a constant default: a default factory costs pydantic a signature look-up per journal)
    _segments: tuple[PayloadRef, ...] = PrivateAttr(default=())
    _saved: dict[str, Any] | None = PrivateAttr(default=None)
    #: the segments and journal of the last whole save while a spending save's segment
    #: follows them (:meth:`save_spending`), which the next spending save replaces
    _unspent: tuple[tuple[PayloadRef, ...], dict[str, Any] | None] | None = PrivateAttr(
        default=None
    )

    # ------------------------------------------------------------------ persistence
    @classmethod
    async def read(
        cls, checkpoint: dict[str, Any] | None, artifacts: RunArtifacts, *, tenant: str
    ) -> Journal:
        """The journal a run record's checkpoint lists (an empty one for a fresh run): its
        segments, read in one request (the store reads them in parallel) and applied in
        order."""
        if not checkpoint:
            return cls()
        if SEGMENTS not in checkpoint:
            raise HarnessError("the run's checkpoint lists no journal segments")
        segments = [PayloadRef.model_validate(ref) for ref in checkpoint[SEGMENTS]]
        try:
            stored = await artifacts.get_payloads(segments, tenant=tenant)
        except NotFoundError as exc:
            raise HarnessError(f"the run's journal is gone: {exc}") from exc
        dumped: Any = None
        for data in stored:
            dumped = apply(dumped, await adecode(data))
        journal = cls.model_validate(dumped)
        # what the next save compares with: a copy, never the objects the journal holds
        journal._segments, journal._saved = tuple(segments), journal.dump()
        return journal

    def dump(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)

    def asked_again(self, key: str | None) -> None:
        """A person was asked about the call ``key`` (``None``: a call that is none of
        :attr:`repeats`): the next call of it is decided as any call is."""
        if key in self.repeats:
            self.repeats.remove(key)

    def rewound(self, graph: dict[str, Any], checkpoint_id: str) -> Journal:
        """The journal of a run that continues this one's graph from its checkpoint
        ``checkpoint_id`` (``graph``: its checkpoints up to there,
        ``RunCheckpointer.rewound``): none of this run's call records — the graph makes its
        steps past the checkpoint anew — but what they did stays known: every call that did
        more than read past the checkpoint is one the new run asks a person about before
        making it again (:attr:`repeats`); what this run spent stays spent, and the large
        results its history names stay readable."""
        past = {key for step, key in self.effects if step is None or step >= checkpoint_id}
        return Journal(
            graph=graph,
            effects=list(self.effects),
            repeats=sorted(past | set(self.repeats)),
            spent=dict(self.spent),
            results=dict(self.results),
        )

    async def save(
        self, artifacts: RunArtifacts, run_id: str, *, worker_id: str | None, tenant: str
    ) -> Save | None:
        """Upload what changed since the last save the run store took, as one segment
        (named by ``worker_id``, as the store fences a worker's writes): the save to send
        (and then :meth:`saved`), or ``None`` when nothing changed. O(what the step added)
        sent, the journal compared in memory."""
        dumped = self.dump()
        node = delta(self._saved, dumped)
        if node is None and self._segments:
            return None
        segments: list[PayloadRef] = list(self._segments)
        if node is None or len(segments) >= MAX_SEGMENTS:
            node, segments = {"$r": dumped}, []
        data = await aencode(node)
        ref = await artifacts.put_payload(run_id, data, worker_id=worker_id, tenant=tenant)
        segments = [*segments, ref]
        return Save(
            checkpoint={SEGMENTS: [r.model_dump() for r in segments]},
            segments=segments,
            dumped=dumped,
        )

    def spends_alone(self) -> bool:
        """Whether a spending save (:meth:`save_spending`) has segments to add to: a save
        was made, and its segments are fewer than :data:`MAX_SEGMENTS` (else a whole save,
        :meth:`save`, which starts them again)."""
        segments = (self._unspent or (self._segments, None))[0]
        return 0 < len(segments) < MAX_SEGMENTS

    async def save_spending(
        self, artifacts: RunArtifacts, run_id: str, *, worker_id: str | None, tenant: str
    ) -> Save | None:
        """A spending save: what the run and its sub-agents' runs have spent now, uploaded as
        one small segment after the segments the last whole save left — a spending save's
        own replaced, so they grow by one at most — the checkpoint every save sends (its
        segments alone: agent-runs keeps any other whole, letting the segments go); then
        :meth:`spending_saved`. ``None`` when the spending is what the last save took; the
        last whole save's segments alone, nothing uploaded, when it is what that one took
        (a tool call counted no more). Only when :meth:`spends_alone`."""
        assert self._saved is not None and self.spends_alone()
        if _spending(self._saved, self) is None:
            return None
        segments, dumped = self._unspent or (self._segments, self._saved)
        assert dumped is not None
        node = _spending(dumped, self)
        if node is None:  # the whole save's own spending again: its segments alone
            return Save(
                checkpoint={SEGMENTS: [r.model_dump() for r in segments]},
                segments=list(segments),
                dumped=dumped,
            )
        ref = await artifacts.put_payload(
            run_id, await aencode(node), worker_id=worker_id, tenant=tenant
        )
        return Save(
            checkpoint={SEGMENTS: [r.model_dump() for r in [*segments, ref]]},
            segments=[*segments, ref],
            dumped=apply(dumped, node),
        )

    def spending_saved(self, save: Save) -> None:
        """The run store took a spending save: the next one replaces its segment, a whole
        save adds to it."""
        if self._unspent is None:
            self._unspent = (self._segments, self._saved)
        self._segments, self._saved = tuple(save.segments), save.dumped

    def saved(self, save: Save) -> None:
        """The run store took ``save``: the next one adds to it."""
        self._segments, self._saved = tuple(save.segments), save.dumped
        self._unspent = None

    async def checkpoint(
        self, artifacts: RunArtifacts, run_id: str, *, worker_id: str | None, tenant: str
    ) -> dict[str, Any]:
        """The run's checkpoint for a pause: what changed saved (:meth:`save`), and every
        segment listed."""
        save = await self.save(artifacts, run_id, worker_id=worker_id, tenant=tenant)
        if save is None:
            return {SEGMENTS: [r.model_dump() for r in self._segments]}
        self.saved(save)
        return save.checkpoint

    def approved_in_framework(
        self, resolution: InterruptResolution, gated: Sequence[tuple[str, Any]] | None = None
    ) -> None:
        """The calls the open pause — a framework's own approval gate — approved by
        ``resolution``, filed for governance (:attr:`approved`), so a person answers once for
        a call whichever gate asks. ``gated``: the calls a gate of several runs as decided,
        each as approved or edited (LangChain's ``HumanInTheLoopMiddleware``, read by its
        adapter); else the one call of the pause's own approval, with its edited arguments on
        an edit. Called with the pause open: its answer is what resolves it."""
        assert self.pending is not None  # the attempt files an open pause's answer
        interrupt = self.pending.interrupt
        if gated is not None:
            calls = list(gated)
        elif resolution.decision not in (InterruptDecision.APPROVE, InterruptDecision.EDIT):
            return
        elif self.pending.native_state is not None and interrupt.tool_call is not None:
            if governance_held(self.pending) and resolution.decision is InterruptDecision.APPROVE:
                return  # governance's own question: its answer is read where the call asks
            edited = resolution.payload if resolution.decision is InterruptDecision.EDIT else None
            args = edited if isinstance(edited, dict) else interrupt.tool_call.args
            calls = [(interrupt.tool_call.tool, args)]
        else:
            return  # the harness's own question: governance asked it
        self.approved += [content_key("approved", name, args) for name, args in calls]

    def rejected_in_framework(self, calls: Sequence[tuple[str, Any]]) -> None:
        """The calls a person rejected at a framework's own approval gate (:attr:`rejected`)."""
        self.rejected += [content_key("rejected", name, args) for name, args in calls]

    def rejected_by_person(self, tool: str, args: Any) -> bool:
        """Whether a call a wrapper of the graph refused is one a person rejected at the
        framework's gate (:attr:`rejected`) — taken: one rejection, one call."""
        key = content_key("rejected", tool, args)
        if key not in self.rejected:
            return False
        self.rejected.remove(key)
        return True

    def answered(self, resolution: InterruptResolution) -> None:
        """File the answer to the open pause, and close it: the tool call it came from within
        is one a later attempt's model makes again (:attr:`again`: one call a place)."""
        if self.pending is None:
            return
        answers = self.answers.setdefault(self.pending.key, [])
        answers.append(resolution.model_dump(mode="json"))
        if self.pending.ident is not None:  # whose answer it is
            by = self.answered_by.setdefault(self.pending.key, [])
            by.extend(None for _ in range(len(answers) - 1 - len(by)))
            by.append(self.pending.ident)
        inside = self.pending.inside
        if inside is not None:
            self.again = [m for m in self.again if m.at != inside.at] + [inside]
            self.place = min(self.place, inside.at)  # not ended, whatever call after it did
        self.pending = None


class Replay:
    """One attempt's cursor over a journal: what the n-th occurrence of a key already got."""

    __slots__ = (
        "_announced",
        "_began",
        "_goes_on",
        "_made",
        "_seeded",
        "_seen",
        "_session",
        "_turns",
        "journal",
        "resumed",
        "where",
    )

    def __init__(self, journal: Journal, *, goes_on: bool = False) -> None:
        self.journal = journal
        self._seen: dict[str, int] = defaultdict(int)
        self._turns: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        #: the place of the attempt's next tool call (``None``: it has made none yet)
        self._made: int | None = None
        #: the attempt goes on from the framework's serialised run (``goes_on``), or from the
        #: session the journal holds now, while the framework keeps to it: the calls made
        #: before are not made again
        self._goes_on = goes_on
        self._session = journal.session
        #: the cursors this attempt has started where the calls made before left them
        #: (:meth:`_index`), and where each call's own reads began (:meth:`calling`)
        self._seeded: set[str] = set()
        self._began: dict[str, dict[str, int]] = {}
        #: the framework's ids of the calls it said it is making, by call (:meth:`announced`)
        self._announced: dict[str, list[str]] = defaultdict(list)
        #: the step a call is being made in now (its graph's checkpoint id), for a graph that
        #: keeps a checkpoint (``adapters.langgraph``); ``None`` elsewhere
        self.where: Callable[[], str | None] | None = None
        #: the checkpoint the attempt's graph resumed from: the records of the steps before it
        #: are passed over (:meth:`_index`)
        self.resumed: str | None = None

    def _continues(self) -> bool:
        """Whether the attempt goes on from the framework's own state — its serialised run, or
        the session the journal holds while the framework keeps to it: the calls made before
        are not made again."""
        kept = self._session is not None and self.journal.session == self._session
        return self._goes_on or kept

    @contextlib.contextmanager
    def calling(self, key: str, *, again: bool = False, ident: str | None = None) -> Iterator[None]:
        """The code inside decides or runs one call of ``key`` (a harness tool's call, a
        framework's own call its permission check asks about): what it reads of the journal
        is that call's (``again``: the same call's check goes on — Claude Code's own question
        after the ``PreToolUse`` decision), read again when the run pauses on it and an
        attempt that goes on makes it again (:meth:`paused`). ``ident``: the framework's id of
        the call, whose answers are its own (``Journal.answered_by``)."""
        if not again or key not in self._began:
            self._began[key] = {}
        token, named = _CALLING.set(key), _IDENT.set(ident)
        try:
            yield
        finally:
            _IDENT.reset(named)
            _CALLING.reset(token)

    @contextlib.contextmanager
    def by_content(self) -> Iterator[None]:
        """What the code inside reads and records of the journal is filed by the call's content
        and order, not by its framework's id of the call: a framework may give the call it makes
        again after a resume another id (Kimi Code does), and a record of what depends on the
        call's content alone — a hook's verdict on it — is the same call's whatever its id."""
        token = _IDENT.set(None)
        try:
            yield
        finally:
            _IDENT.reset(token)

    def announced(self, key: str, ident: str) -> None:
        """The framework makes call ``key`` under its id ``ident`` (a call it hands the harness
        later, through its MCP server, where the id is not given): :meth:`claimed`."""
        self._announced[key].append(ident)

    def claimed(self, key: str) -> str | None:
        """The framework's id of call ``key`` it made, the first it said and not yet taken."""
        idents = self._announced.get(key)
        return idents.pop(0) if idents else None

    def cursor(self) -> dict[str, int]:
        """How far this attempt has read the journal now: each cursor read so far."""
        return {name: self._seen[name] for name in self._seeded}

    def paused(self, *, since: dict[str, int] | None = None) -> None:
        """The run pauses on the call the code running now decides or runs: how far the calls
        made before it read each record and answer is kept (``Journal.cursors``) — the cursors as
        they are, the paused call's own put back where it began — so an attempt that goes on
        from the framework's own state, which makes only the paused call again, starts there
        (:meth:`_index`). ``since``: the cursors when the work the framework begins again
        started (a subagent paused in its work: ``Helpers.begun_again``), all of whose reads
        are put back. An attempt run again from its input makes every call again: it keeps
        nothing of an earlier attempt's."""
        kept = dict(self.journal.cursors) if self._continues() else {}
        kept.update(self._seen)
        calling = _CALLING.get()
        if calling is not None:
            kept.update(self._began.get(calling, {}))
        if since is not None:
            kept.update({name: since.get(name, self._seed(name)) for name in self._seeded})
        self.journal.cursors = {name: n for name, n in kept.items() if n}

    def _seed(self, name: str) -> int:
        """Where cursor ``name`` starts in this attempt: past the reads of the calls made
        before, where the attempt goes on from the framework's own state."""
        return self.journal.cursors.get(name, 0) if self._continues() else 0

    def _moved(self, name: str) -> None:
        """Cursor ``name`` moves on: where it was, kept for the call reading it now."""
        calling = _CALLING.get()
        if calling is not None:
            self._began.setdefault(calling, {}).setdefault(name, self._seen[name])

    def exclusive(self, key: str) -> asyncio.Lock:
        """What one call of ``key`` holds from its replay to its record: identical calls made
        at once run one after another, in the order they asked."""
        return self._turns[key]

    def made(self, tool: str, args: dict[str, Any]) -> Made:
        """The call of ``tool`` the attempt makes now, as it runs: with ``args`` — or, at the
        place of a call of ``tool`` an answered pause came from within (``Journal.again``), as
        that call: the model made it again, in other words. A call the journal holds a record
        of is itself. The attempt's first call is the run's first, unless the attempt goes on
        from the framework's own state: then it is the next after the calls already made
        (``Journal.place``)."""
        if self._made is None:
            if not self._continues():
                self.journal.place = 0  # run again from its input
            self._made = self.journal.place
        at, self._made = self._made, self._made + 1
        key = call_key(tool, args)
        paused = next((m for m in self.journal.again if m.at == at and m.tool == tool), None)
        if paused is None or self.occurrence(key) < len(self.journal.calls.get(key, ())):
            return Made(tool=tool, args=args, at=at)
        return paused

    def ended(self, made: Made) -> None:
        """``made`` ran to its end (it did not pause): the framework's own state holds it."""
        self.journal.place = max(self.journal.place, made.at + 1)

    def answer(self, key: str) -> InterruptResolution | None:
        """The recorded answer for this occurrence of ``key``, advancing the cursor."""
        recorded = self.journal.answers.get(key, [])
        index = self._answer_at(key)
        if index < len(recorded):
            self._moved(f"a:{key}")
            self._seen[f"a:{key}"] = max(self._seen[f"a:{key}"], index + 1)
            self._mark("a", key, index, self.here())
            return InterruptResolution.model_validate(recorded[index])
        if index > self._seen[f"a:{key}"]:  # past the other calls' answers
            self._moved(f"a:{key}")
            self._seen[f"a:{key}"] = index
        return None

    def answered(self, key: str) -> bool:
        """Whether this occurrence of ``key`` has a recorded answer (the cursor stays)."""
        return self._answer_at(key) < len(self.journal.answers.get(key, ()))

    def _answer_at(self, key: str) -> int:
        """Where this occurrence of ``key``'s answer is: the answer to the call its framework
        names, in a step its framework makes again by their ids (:meth:`here`); else the next
        answer that is not another call's (:meth:`_anothers`) — past the last when none is."""
        given = len(self.journal.answers.get(key, ()))
        index = self._index("a", key, given)
        ident = _IDENT.get()
        if ident is not None and self.here() is not None:
            by = self.journal.answered_by.get(key, [])
            own = next((n for n, named in enumerate(by) if named == ident), None)
            if own is not None:
                return own
        while index < given and self._anothers(key, index):
            index += 1
        return index

    def _anothers(self, key: str, index: int) -> bool:
        """Whether the ``index``-th answer of ``key`` is another call's: in a step its
        framework makes again by their ids (:meth:`here`: one model reply's identical calls),
        any answer filed for a call it named (``Journal.answered_by``) is that call's; where
        the attempt goes on from the framework's own state, a rejection is — and a call of
        another id, a new one, identical or not, is asked again."""
        by = self.journal.answered_by.get(key, [])
        owner = by[index] if index < len(by) else None
        ident = _IDENT.get()
        if owner is None or ident is None or owner == ident:
            return False
        if self.here() is not None:
            return True
        if not self._continues():
            return False
        decision = self.journal.answers[key][index].get("decision")
        return decision == InterruptDecision.REJECT.value

    def occurrence(self, key: str) -> int:
        """Which occurrence of ``key`` the next call is (0 for the first): the record of the
        call its framework names (``called_by``), where that names it; else the next record no
        other call named claims (:meth:`_named`), or a new one."""
        recorded = len(self.journal.calls.get(key, ()))
        index = self._index("c", key, recorded)
        ident, by = _IDENT.get(), self.journal.called_by.get(key, [])
        if ident is None:
            return index
        own = next(
            (n for n, named in enumerate(by) if named == ident and self._named(key, n)), None
        )
        if own is not None:
            return own
        while index < min(recorded, len(by)) and by[index] is not None and self._named(key, index):
            index += 1
        return index

    def _named(self, key: str, index: int) -> bool:
        """Whether the ``index``-th record of ``key`` belongs to the call its framework named it
        for: the framework makes its calls again by their ids — the attempt goes on from its own
        state, or the record was made in the step it makes again (:meth:`here`)."""
        if self._continues():
            return True
        here, tags = self.here(), self.journal.steps.get(f"c:{key}", [])
        return here is not None and index < len(tags) and tags[index] == here

    def interrupted(self, key: str) -> bool:
        """Whether this occurrence of ``key`` started in an earlier attempt and never ended."""
        return self.occurrence(key) < self.journal.started.get(key, 0)

    def start(self, key: str) -> None:
        """Mark this occurrence of ``key`` as running (undone by :meth:`unstart`)."""
        self.journal.started[key] = self.occurrence(key) + 1

    def unstart(self, key: str) -> None:
        """This occurrence of ``key`` ended without a record (it failed, or paused): it runs
        again in a later attempt, as if it never started."""
        if self.occurrence(key):
            self.journal.started[key] = self.occurrence(key)
        else:
            self.journal.started.pop(key, None)

    def call(self, key: str) -> tuple[bool, Any]:
        """``(True, output)`` when this occurrence of the call already ran — and what it read
        besides is read past (``Journal.reads``: its approval's answer, which the next
        identical call does not take for its own)."""
        index = self.occurrence(key)
        recorded = self.journal.calls.get(key, [])
        if index >= len(recorded):
            return False, None
        self._moved(f"c:{key}")
        self._seen[f"c:{key}"] = max(self._seen[f"c:{key}"], index + 1)
        reads = self.journal.reads.get(key, [])
        for name, n in (reads[index] if index < len(reads) else {}).items():
            kind, read = name.split(":", 1)
            self._index(kind, read, 0)  # seeded where the attempt starts it
            self._moved(name)
            self._seen[name] += n
        tags = self.journal.steps.get(f"c:{key}")
        if tags is not None and index < len(tags) and tags[index] is not None:
            self._mark("c", key, index, self.here())  # a step's record: now this step's
        return True, recorded[index]

    def record_call(
        self,
        key: str,
        output: Any,
        *,
        tool: str | None = None,
        stepped: bool = False,
        scope: str | None = None,
    ) -> None:
        """Record what this occurrence of ``key`` produced: a tool's output (``tool`` names
        it, and it stays offered after a pause), or a model step of a loop the harness runs.
        ``stepped``: a record of the step it is made in (a tool call's) — dropped once the
        run's graph has moved past that step (:meth:`passed`); else it is the run's (a stored
        prompt's version, pinned for the run)."""
        recorded = self.journal.calls.setdefault(key, [])
        index = self.occurrence(key)
        recorded.append(output)
        self._moved(f"c:{key}")
        self._seen[f"c:{key}"] = max(self._seen[f"c:{key}"], index + 1)
        ident = _IDENT.get()
        if ident is not None or key in self.journal.called_by:
            by = self.journal.called_by.setdefault(key, [])
            by.extend(None for _ in range(len(recorded) - 1 - len(by)))
            by.append(ident)
        # what the call read besides its own record, when it is the call being made (the
        # code inside its ``calling``, or ``scope``: its key, after its framework ran it)
        began = self._began.get(scope or key, {}) if (scope or _CALLING.get()) == key else {}
        read = {
            name: self._seen[name] - n
            for name, n in began.items()
            if name.startswith("a:") and self._seen[name] > n
        }
        if read or key in self.journal.reads:
            reads = self.journal.reads.setdefault(key, [])
            reads.extend({} for _ in range(len(recorded) - 1 - len(reads)))
            reads.append(read)
        if stepped:
            self._mark("c", key, len(recorded) - 1, self.here())
        if tool is not None and tool not in self.journal.used:
            self.journal.used.append(tool)

    def effected(self, key: str) -> None:
        """The call ``key`` — one that does more than read — is made in the step under way
        (``Journal.effects``)."""
        self.journal.effects.append((self.here(), key))

    def read(self, key: str) -> None:
        """The record just made of ``key`` is a read's its framework ran: answered from the
        journal when made again only until a write of the run (:meth:`wrote`)."""
        if key not in self.journal.reread:
            self.journal.reread.append(key)

    def wrote(self) -> None:
        """A write of the run ran. The reads journaled before it (:meth:`read`) may have read
        what it changed: their records are dropped, so one made again — by a query run again
        from its prompt, a subagent started again — runs again; the write's own record
        stands, and it runs once."""
        for key in self.journal.reread:
            self.journal.calls.pop(key, None)
            self.journal.reads.pop(key, None)
        self.journal.reread.clear()

    def record_step(self, call: ToolCall, outcome: ToolOutcome) -> None:
        """A call that ran, and its outcome, on the run's trajectory (the call without the
        run's task, which every call repeats; the outcome's output as the stream previews
        it)."""
        kept = outcome.model_copy(update={"output": preview(outcome.output)})
        self.journal.trajectory.append((call.model_copy(update={"task": None}), kept))

    def passed(self, step: str | None) -> None:
        """The run's graph moved past ``step`` (a checkpoint id) and the steps before it: their
        records are dropped — the graph does not make their calls again — and the cursor and
        the calls started move back by as many."""
        if step is None:
            return
        steps = self.journal.steps
        for marked in list(steps):
            kind, key = marked.split(":", 1)
            tags = steps[marked]
            gone = next((n for n, tag in enumerate(tags) if tag is None or tag > step), len(tags))
            if not gone:
                continue
            kept = self.journal.calls if kind == "c" else self.journal.answers
            entries = kept.get(key, [])
            del entries[:gone], tags[:gone]
            if not entries:
                kept.pop(key, None)
            beside = (
                (self.journal.reads, self.journal.called_by)
                if kind == "c"
                else (self.journal.answered_by,)
            )
            for parallel in beside:
                if key in parallel:
                    del parallel[key][:gone]
                    if not entries:
                        del parallel[key]
            if not tags:
                del steps[marked]
            self._seen[marked] = max(0, self._seen[marked] - gone)
            if kind == "c" and key in self.journal.started:
                left = self.journal.started[key] - gone
                if left > 0:
                    self.journal.started[key] = left
                else:
                    del self.journal.started[key]

    def here(self) -> str | None:
        """The step a call is being made in now (:attr:`where`); ``None`` where the framework
        keeps no checkpoint, or outside one of its steps."""
        return None if self.where is None else self.where()

    def _index(self, kind: str, key: str, length: int) -> int:
        """The cursor of ``key``, past the records of a step before the one the graph resumed
        from (made in an earlier attempt: the graph does not make that call again, so this one
        is a later step's)."""
        seen = f"{kind}:{key}"
        if seen not in self._seeded:
            self._seeded.add(seen)  # the calls made before are not made again: past theirs
            self._seen[seen] = max(self._seen[seen], self._seed(seen))
        index = self._seen[seen]
        tags = self.journal.steps.get(seen)
        if tags is None or self.resumed is None:
            return index
        here = self.here()
        while (
            index < min(length, len(tags))
            and (tag := tags[index]) is not None
            and here is not None
            and tag != here
            and tag <= self.resumed
        ):
            index += 1
        self._seen[seen] = index
        return index

    def _mark(self, kind: str, key: str, index: int, step: str | None) -> None:
        """Mark the ``index``-th record of ``key`` as made in ``step`` (none: left as it is)."""
        if step is None:
            return
        tags = self.journal.steps.setdefault(f"{kind}:{key}", [])
        tags.extend([None] * (index + 1 - len(tags)))
        tags[index] = step
