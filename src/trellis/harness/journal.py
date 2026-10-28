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
outcome (:data:`OUTCOME`), so a re-run tells the model the same thing and runs nothing.

The journal is the run's checkpoint (``RunRecord.checkpoint``): the run store keeps it with the
pause and hands it to whichever worker resumes the run, so a resume on another machine repeats
no question and no side effect. A journal larger than a checkpoint may be is stored as a run
artifact, and the checkpoint names it (:meth:`Journal.checkpoint`, :meth:`Journal.read`). A
sub-agent's run working inside one of the run's tool calls keeps its journal in the run's
(``children``), so the run's progress saves the child's too; the run's sandbox is named here as
soon as it exists (``sandbox``), so every later attempt works in that one
(``trellis.harness.sandbox``); a graph that checkpoints into the run
(``middleware.RunCheckpointer``) keeps its latest checkpoint here (``graph``), and a large tool
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
records are dropped (:meth:`Replay.passed`): its journal holds the graph's latest checkpoint and
the records of the step under way, however long the run.

Calls made at once (a framework running several tools together) take their occurrences in the
order they asked: identical calls run one after another (:meth:`Replay.exclusive`), so each has
its own occurrence and a re-run hands each the output it got, whatever order they finished in.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections import defaultdict
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Final

from pydantic import BaseModel, ConfigDict, Field

from trellis.contracts import (
    ArtifactRef,
    HarnessError,
    Interrupt,
    InterruptDecision,
    InterruptResolution,
    ToolCall,
    ToolOutcome,
)
from trellis.harness.events import preview

if TYPE_CHECKING:
    from trellis.harness.runs import RunArtifacts

#: The most a checkpoint may be, as compact JSON (agent-runs' ``MAX_CHECKPOINT_BYTES``: a
#: larger one is refused with ``413``). A larger journal is stored as a run artifact, and the
#: checkpoint holds only its reference, under :data:`JOURNAL_REF`.
MAX_CHECKPOINT_BYTES: Final = 1024 * 1024
JOURNAL_REF: Final = "journal_ref"
#: The key a call's recorded outcome is filed under when it is not an output (a timeout, an
#: unknown effect): ``{OUTCOME: {"status": ..., "output": ..., "metadata": ...}}``.
OUTCOME: Final = "trellis_outcome"


def content_key(kind: str, *parts: Any) -> str:
    """A stable key for a question or a call: same content, same key, across processes."""
    text = json.dumps([kind, *parts], sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.blake2b(text.encode(), digest_size=12).hexdigest()


class Pending(BaseModel):
    """The interrupt the run is waiting on, and the key its answer is filed under."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    key: str
    interrupt: Interrupt
    #: The framework's own handle on the pause (LangGraph's interrupt id), when it has one.
    native_id: str | None = None
    #: The framework's serialised run, when it resumes from one (an OpenAI Agents
    #: ``RunState`` paused on a ``needs_approval`` tool).
    native_state: dict[str, Any] | None = None


class Journal(BaseModel):
    """Answers and tool outputs recorded so far, plus the pause currently open."""

    model_config = ConfigDict(extra="forbid")

    answers: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    calls: dict[str, list[Any]] = Field(default_factory=dict)
    #: the tools the run has called (they stay offered to the model after a pause)
    used: list[str] = Field(default_factory=list)
    #: the run's trajectory: every harness tool call that ran, in order across its attempts,
    #: with its outcome — the call and outcome its tool record is made of (what evaluators read:
    #: ``EvalCase.trajectory``); a call replayed from the journal is in it once
    trajectory: list[tuple[ToolCall, ToolOutcome]] = Field(default_factory=list)
    #: the tools a reviewer approved for the rest of the run (``remember="run"``; ``"<tool> for
    #: <assignee>"`` when the approval was asked of someone): their later calls asked of the
    #: same are not asked about again in this run
    remembered: list[str] = Field(default_factory=list)
    #: every person's decision on a tool call of the run, in order: the pause's
    #: ``interrupt_id``, the tool, the ``decision``, whom it was asked of (``assignee``), who
    #: decided (``reviewer``), their ``comment`` and when (``resolved_at``) — what the run's
    #: record says of its approvals (``trellis.harness.explain``)
    approvals: list[dict[str, Any]] = Field(default_factory=list)
    #: the calls a framework's own approval gate approved (LangChain's
    #: ``HumanInTheLoopMiddleware``, Deep Agents' ``interrupt_on``, an OpenAI Agents
    #: ``needs_approval`` tool), by content: governance counts that approval, once each, and
    #: does not ask again (``approved_in_framework``)
    approved: list[str] = Field(default_factory=list)
    #: per call, how many of its occurrences started: one more than ``calls`` records was
    #: running when the attempt ended without recording it (a crash)
    started: dict[str, int] = Field(default_factory=dict)
    #: the journals of the sub-agents' runs working inside the run's tool calls now, by run id
    children: dict[str, Journal] = Field(default_factory=dict)
    #: the framework's own conversation, when it keeps one the next attempt continues (a
    #: Claude Agent SDK session): what it already did — its built-in tools — is not done again
    session: str | None = None
    #: the transcripts of the framework's own sessions, by session, as its session store
    #: mirrors them (``bridges.claude.JournalSessions``): a resume on any machine continues them
    transcripts: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    #: the sandbox the run's sandbox tools work in, as its reference (``sandbox.SandboxRef``):
    #: recorded as soon as it exists, so every later attempt works in the same one
    sandbox: dict[str, Any] | None = None
    #: the graph's latest checkpoint and its pending writes, by namespace, when the graph keeps
    #: them in the run (``middleware.RunCheckpointer``): a resume continues where it stopped
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
    pending: Pending | None = None

    # ------------------------------------------------------------------ persistence
    @classmethod
    def of(cls, checkpoint: dict[str, Any] | None) -> Journal:
        """The journal a run record's checkpoint holds (an empty one for a fresh run)."""
        return cls() if checkpoint is None else cls.model_validate(checkpoint)

    @classmethod
    async def read(
        cls, checkpoint: dict[str, Any] | None, artifacts: RunArtifacts, *, tenant: str
    ) -> Journal:
        """The journal a run record's checkpoint holds, or names: one stored as a run artifact
        (:meth:`checkpoint`) is read back from it."""
        ref = (checkpoint or {}).get(JOURNAL_REF)
        if ref is None:
            return cls.of(checkpoint)
        artifact_id = ArtifactRef.model_validate(ref).artifact_id
        data = await artifacts.download(artifact_id, tenant=tenant)
        if data is None:
            raise HarnessError(f"the run's journal (run artifact {artifact_id}) is gone")
        return cls.of(json.loads(data))

    def dump(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)

    async def checkpoint(
        self, artifacts: RunArtifacts, run_id: str, *, worker_id: str | None, tenant: str
    ) -> dict[str, Any]:
        """The run's checkpoint for a pause or a progress save: the journal itself, or — over
        :data:`MAX_CHECKPOINT_BYTES` — a reference to it, uploaded as a run artifact (named by
        ``worker_id``, as the store fences a worker's writes)."""
        dumped = self.dump()
        data = json.dumps(dumped, default=str, separators=(",", ":")).encode()
        if len(data) <= MAX_CHECKPOINT_BYTES:
            return dumped
        ref = await artifacts.upload(run_id, data, worker_id=worker_id, tenant=tenant)
        return {JOURNAL_REF: ref.model_dump(mode="json", exclude_none=True)}

    def approved_in_framework(self, resolution: InterruptResolution) -> None:
        """The calls the open pause — a framework's own approval gate — approved by
        ``resolution``, filed for governance (:attr:`approved`): every call it holds on an
        approval, the first with its edited arguments on an edit."""
        if self.pending is None or resolution.decision not in (
            InterruptDecision.APPROVE,
            InterruptDecision.EDIT,
        ):
            return
        interrupt = self.pending.interrupt
        payload = interrupt.payload or {}
        actions = payload.get("action_requests")
        if isinstance(actions, list) and payload.get("review_configs") is not None:
            calls = [(a.get("name"), a.get("args") or {}) for a in actions]
        elif self.pending.native_state is not None and interrupt.tool_call is not None:
            calls = [(interrupt.tool_call.tool, interrupt.tool_call.args)]
        else:
            return  # the harness's own question: governance asked it
        edited = resolution.payload if resolution.decision is InterruptDecision.EDIT else None
        if isinstance(edited, dict) and calls:
            calls[0] = (calls[0][0], edited)
        self.approved += [content_key("approved", name, args) for name, args in calls]

    def decided(self, resolution: InterruptResolution) -> None:
        """A person's ``resolution`` of the open pause, filed in :attr:`approvals` when the pause
        is a tool call's (an approval: the harness's, or a framework's own gate)."""
        interrupt = self.pending.interrupt if self.pending is not None else None
        if interrupt is None or interrupt.tool_call is None:
            return
        self.approvals.append(
            {
                "interrupt_id": resolution.interrupt_id,
                "tool": interrupt.tool_call.tool,
                "decision": resolution.decision.value,
                "assignee": interrupt.assignee,
                "reviewer": resolution.reviewer,
                "comment": resolution.comment,
                "resolved_at": resolution.resolved_at.isoformat(),
            }
        )

    def answered(self, resolution: InterruptResolution) -> None:
        """File the answer to the open pause, and close it."""
        if self.pending is None:
            return
        self.answers.setdefault(self.pending.key, []).append(resolution.model_dump(mode="json"))
        self.pending = None


class Replay:
    """One attempt's cursor over a journal: what the n-th occurrence of a key already got."""

    __slots__ = ("_seen", "_turns", "journal", "resumed", "where")

    def __init__(self, journal: Journal) -> None:
        self.journal = journal
        self._seen: dict[str, int] = defaultdict(int)
        self._turns: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        #: the step a call is being made in now (its graph's checkpoint id), for a graph that
        #: keeps a checkpoint (``adapters.langgraph``); ``None`` elsewhere
        self.where: Callable[[], str | None] | None = None
        #: the checkpoint the attempt's graph resumed from: the records of the steps before it
        #: are passed over (:meth:`_index`)
        self.resumed: str | None = None

    def exclusive(self, key: str) -> asyncio.Lock:
        """What one call of ``key`` holds from its replay to its record: identical calls made
        at once run one after another, in the order they asked."""
        return self._turns[key]

    def answer(self, key: str) -> InterruptResolution | None:
        """The recorded answer for this occurrence of ``key``, advancing the cursor."""
        index = self._index("a", key, len(self.journal.answers.get(key, ())))
        recorded = self.journal.answers.get(key, [])
        if index < len(recorded):
            self._seen[f"a:{key}"] = index + 1
            self._mark("a", key, index, self._here())
            return InterruptResolution.model_validate(recorded[index])
        return None

    def occurrence(self, key: str) -> int:
        """Which occurrence of ``key`` the next call is (0 for the first)."""
        return self._index("c", key, len(self.journal.calls.get(key, ())))

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
        """``(True, output)`` when this occurrence of the call already ran."""
        index = self.occurrence(key)
        recorded = self.journal.calls.get(key, [])
        if index >= len(recorded):
            return False, None
        self._seen[f"c:{key}"] = index + 1
        tags = self.journal.steps.get(f"c:{key}")
        if tags is not None and index < len(tags) and tags[index] is not None:
            self._mark("c", key, index, self._here())  # a step's record: now this step's
        return True, recorded[index]

    def record_call(
        self, key: str, output: Any, *, tool: str | None = None, stepped: bool = False
    ) -> None:
        """Record what this occurrence of ``key`` produced: a tool's output (``tool`` names
        it, and it stays offered after a pause), or a model step of a loop the harness runs.
        ``stepped``: a record of the step it is made in (a tool call's) — dropped once the
        run's graph has moved past that step (:meth:`passed`); else it is the run's (a stored
        prompt's version, pinned for the run)."""
        recorded = self.journal.calls.setdefault(key, [])
        index = self.occurrence(key)
        recorded.append(output)
        self._seen[f"c:{key}"] = index + 1
        if stepped:
            self._mark("c", key, len(recorded) - 1, self._here())
        if tool is not None and tool not in self.journal.used:
            self.journal.used.append(tool)

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
            if not tags:
                del steps[marked]
            self._seen[marked] = max(0, self._seen[marked] - gone)
            if kind == "c" and key in self.journal.started:
                left = self.journal.started[key] - gone
                if left > 0:
                    self.journal.started[key] = left
                else:
                    del self.journal.started[key]

    def _here(self) -> str | None:
        return None if self.where is None else self.where()

    def _index(self, kind: str, key: str, length: int) -> int:
        """The cursor of ``key``, past the records of a step before the one the graph resumed
        from (made in an earlier attempt: the graph does not make that call again, so this one
        is a later step's)."""
        seen = f"{kind}:{key}"
        index = self._seen[seen]
        tags = self.journal.steps.get(seen)
        if tags is None or self.resumed is None:
            return index
        here = self._here()
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
