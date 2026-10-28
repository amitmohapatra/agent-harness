"""Explanations: what a run did to give its answer, kept as the run's record, and what each
audience is shown of it.

The record of a run (:class:`Record`) is made of what the harness keeps already: the run's
trajectory (``Journal.trajectory``: every tool call that ran, with its outcome — and, for a tool
that says where its result came from, the sources: ``tools.base.Citation``), the people's
decisions on its calls (``Journal.approvals``), the memories its pushed context showed and how
far they covered the question (``Runtime.pushed``, ``Runtime.evidence``) and what it spent
(``Journal.spent``). The run store keeps the rest: the question, the answer, how the run ended
and how long it took (``RunRecord``). The journal is the run's checkpoint, so the record
survives a pause and a crash with it; when the run ends it is kept (:func:`keep`) — with
agent-runs in the run's event log (a :data:`RECORD` event, read by any replica), else in the
agent, as long as the in-process store keeps finished runs (``runs.KEEP_FINISHED``).

A follow-up about an earlier answer is given what it needs one of two ways, each measured
against the other and against neither (``benchmark.cost``'s ``explain`` suite), and neither on
until the measurements decide (:func:`measured`; ``docs/explanations.md#the-two-arms``):

* ``sized`` — the conversation window asked of the memory service at a share of the model's
  window, capped (:func:`conversation_budget`), and the request fitted to the model by
  LangChain's ``trim_messages`` (``middleware.FitWindow``), the older tool results cleared by
  LangChain's ``ContextEditingMiddleware`` as always;
* ``lookup`` — the window as the service cuts it by default (whole turns), and the record of an
  earlier answer read on demand with one tool, :data:`EARLIER` (:func:`earlier_tool`):
  read-only, governed as a read, offered in every run of a conversation and scoped to it — its
  tenant, user, agent and thread, never a child run's.

What :data:`EARLIER` returns is rendered here from the record, never written by the model, for
the audience the harness's key is (:func:`audience`):

* ``client`` — a service key, acting for the end user: the steps in plain words (each tool's
  own description, its first sentence), the sources with their name, date and link (only
  those the run's tools returned), how far the user's saved memories covered the question and
  how fresh the sources are, the answer tied to the steps it rests on, the honest limits, the
  user's own memories the run was shown (another person's never), the time taken, and the
  approvals by role. Never a tool's name or arguments, a raw output, an instruction, a
  governance rule, a model, tokens, cost, an id, an error trace or a score
  (:func:`client_view`);
* ``admin`` — an ``admin`` or ``platform`` key (agent-runs' administering roles): all of it,
  as JSON (:func:`admin_view`).
"""

from __future__ import annotations

import contextlib
import json
import re
from collections.abc import Iterator, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from trellis.contracts import RunEventType, RunRecord, RunStatus, ToolSpec
from trellis.harness.adapters.base import query_of
from trellis.harness.history import Speaker
from trellis.harness.runlog import event_log
from trellis.harness.runtime import EARLIER, Runtime, current
from trellis.harness.tools.base import SOURCES, Citation, Tool, citations
from trellis.harness.tools.convert import text_of
from trellis.memory.errors import MemoryError as MemoryServiceError

if TYPE_CHECKING:
    from trellis.harness.agent import Agent
    from trellis.memory.models import MemoryResult

#: The custom event a run's record is kept as in agent-runs' event log.
RECORD: Final = "record"
#: The key roles that administer a tenant (agent-runs' ``ADMINISTERING_ROLES``): the
#: administrator's view; any other key acts for end users, who get the client's.
ADMINISTERING: Final = frozenset({"admin", "platform"})
Audience = Literal["client", "admin"]
#: How a step reads in plain words when its tool says nothing of what it does, by its risk.
DOES: Final = {
    "read": "Looked something up",
    "write": "Made a change",
    "irreversible": "Made a change that cannot be undone",
}
#: How far the user's saved memories covered the question (the context's ``evidence_status``).
COVERED: Final = {
    "COMPLETE": "your saved notes and history covered the question",
    "INCOMPLETE": "your saved notes and history covered only part of the question",
    "INSUFFICIENT": "your saved notes and history held nothing about the question",
}
#: What a person's decision on a step is called, in plain words.
DECIDED: Final = {
    "APPROVE": "approved",
    "EDIT": "approved with changes",
    "REJECT": "rejected",
    "CANCEL": "cancelled",
}
#: What the model reads of :data:`EARLIER`.
EARLIER_DESCRIPTION: Final = (
    "Your record of an answer you gave earlier in this conversation: your steps, its sources "
    "(name, date, link), the user's notes it used, approvals, time taken. Call it when asked "
    "about an earlier answer (where from, how you know, why, how sure, what you did, how "
    "long). What it does not say, you do not know."
)
_WORD_CHARS: Final = r"[\w.:-]"


# --------------------------------------------------------------------------- the two arms
Arm = Literal["sized", "lookup"]


@dataclass(frozen=True, slots=True)
class Measured:
    """The arm agents wrapped (and graphs built) inside :func:`measured` take: for ``sized``,
    the ``share`` of the model's window the conversation window may take, at most ``cap``
    tokens."""

    arm: Arm
    share: float = 0.0
    cap: int = 0


_MEASURED: ContextVar[Measured | None] = ContextVar("trellis_explain_arm", default=None)


@contextlib.contextmanager
def measured(arm: Arm, *, share: float = 0.0, cap: int = 0) -> Iterator[Measured]:
    """Agents wrapped and ``ReAct`` graphs built inside take ``arm`` (the module says what each
    is; ``sized`` takes the window's ``share``, at most ``cap`` tokens): a measurement of what
    each is worth, as ``react.defaults_on`` — not an option of ``h.wrap``: until the
    measurements decide, an agent takes neither."""
    if arm == "sized" and not (0 < share <= 1 and cap > 0):
        raise ValueError("the sized arm takes a share of the window (0 to 1) and a cap over 0")
    taken = Measured(arm, share, cap)
    token = _MEASURED.set(taken)
    try:
        yield taken
    finally:
        _MEASURED.reset(token)


def measuring() -> Arm | None:
    """The arm agents wrapped now take (:func:`measured`); ``None``: neither."""
    taken = _MEASURED.get()
    return None if taken is None else taken.arm


def conversation_budget(window: int | None) -> int | None:
    """The conversation window's budget for a model reading ``window`` tokens: in the ``sized``
    arm its share, at most its cap; else (or the window unknown) ``None``, the service's own."""
    taken = _MEASURED.get()
    if taken is None or taken.arm != "sized" or not window:
        return None
    return min(taken.cap, int(window * taken.share))


class Step(BaseModel):
    """One tool call of a run, as its record keeps it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    status: str
    #: what the step did, in plain words: its tool's description (its first sentence)
    said: str
    #: what it returned, as the run's trajectory keeps it (a preview)
    output: Any = None
    error_class: str | None = None
    sources: list[Citation] = Field(default_factory=list)
    seconds: float | None = None


class Record(BaseModel):
    """What a run did to give its answer (the module says what each part is)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str
    steps: list[Step] = Field(default_factory=list)
    approvals: list[dict[str, Any]] = Field(default_factory=list)
    memories: list[str] = Field(default_factory=list)
    evidence: str | None = None
    spent: dict[str, float] = Field(default_factory=dict)

    @property
    def sources(self) -> list[Citation]:
        """Every source the run's steps returned, each once, in the order they came."""
        return list(dict.fromkeys(c for step in self.steps for c in step.sources))


# --------------------------------------------------------------------------- the record
def record_of(runtime: Runtime) -> Record:
    """The run's record as its attempt ends (the module says what it is made of)."""
    journal = runtime.replay.journal
    names = set(runtime.toolbox) | set(runtime.native) | {c.tool for c, _ in journal.trajectory}
    steps = [
        Step(
            tool=call.tool,
            args=call.args,
            status=outcome.status.value,
            said=_said(_spec(runtime, call.tool), names),
            output=outcome.output,
            error_class=outcome.error_class,
            sources=citations(outcome.metadata.get(SOURCES)),
            seconds=None if outcome.latency_ms is None else round(outcome.latency_ms / 1000, 3),
        )
        for call, outcome in journal.trajectory
    ]
    return Record(
        run_id=runtime.run_id,
        steps=steps,
        approvals=list(journal.approvals),
        memories=list(runtime.pushed.get("memory_ids") or ()),
        evidence=runtime.evidence,
        spent=dict(journal.spent),
    )


def _spec(runtime: Runtime, name: str) -> ToolSpec | None:
    found = runtime.toolbox.get(name)
    return found.spec if found is not None else runtime.native.get(name)


def _said(spec: ToolSpec | None, names: set[str]) -> str:
    """A step in plain words: its tool's description, the first sentence, without the name of
    any tool of the run; what its risk says when the description says nothing."""
    text = (spec.description if spec is not None else "") or ""
    first = _without(re.split(r"(?<=[.!?。।])\s|\n", text.strip(), maxsplit=1)[0], names)
    if len(first) >= 3:
        return first
    risk = spec.side_effects if spec is not None else "write"
    return DOES.get(risk, DOES["write"])


def _without(text: str, names: set[str]) -> str:
    """``text`` without any of ``names`` (tool names) that is an identifier (``stock_quote``,
    ``getQuote``, ``crm.lookup``) as a whole word, a longer name first: a tool named with a
    plain word (``order``) reads as that word."""
    for name in sorted(filter(_identifier, names), key=len, reverse=True):
        # a whole name: not part of a longer one (``a.quote``, ``quote_v2``), a sentence's
        # full stop after it allowed
        text = re.sub(rf"(?<!{_WORD_CHARS}){re.escape(name)}(?![\w:-]|\.\w)", "", text)
    return re.sub(r"[ \t]{2,}", " ", text).strip(" ,;:")


def _identifier(name: str) -> bool:
    """Whether a tool's name is no plain word: it has a character a word of prose has not."""
    return not re.fullmatch(r"[^\W\d_]+", name) or (name[1:] != name[1:].lower())


def keep(agent: Agent, runtime: Runtime) -> Record:
    """The run's record, kept as its attempt ends: in agent-runs' event log (any replica reads
    it), else in the agent while the in-process store keeps the run."""
    record = record_of(runtime)
    if event_log(agent.harness.runs) is None:
        agent.records[runtime.run_id] = record
    else:
        runtime.events.custom(RECORD, record=record.model_dump(mode="json"))
    return record


async def recorded(agent: Agent, run_id: str, *, tenant: str) -> Record | None:
    """A run's record as it was kept (:func:`keep`): the last one, the run's latest attempt's;
    ``None`` when none was (a run older than records, or forgotten)."""
    log = event_log(agent.harness.runs)
    if log is None:
        return agent.records.get(run_id)
    found: Record | None = None
    async for entry in log.stream_events(run_id, tenant=tenant):
        event = entry.event
        data = event.data or {}
        if event.type is RunEventType.CUSTOM and data.get("name") == RECORD:
            found = Record.model_validate(data["record"])
    return found


# --------------------------------------------------------------------------- the lookup
def earlier_tool() -> Tool:
    """:data:`EARLIER`: the record of an earlier answer of the run's conversation, for the
    audience the harness's key is — read-only (governed as a read)."""
    spec = ToolSpec(
        name=EARLIER,
        description=EARLIER_DESCRIPTION,
        input_schema={
            "type": "object",
            "properties": {
                "back": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Which answer: 1 the latest before this question, 2 the one "
                    "before it, and so on",
                }
            },
        },
        side_effects="read",
    )

    async def run(args: dict[str, Any]) -> str:
        runtime = current()
        assert runtime is not None  # the bridge runs it inside its run
        return await earlier(runtime, back=max(1, int(args.get("back") or 1)))

    return Tool(spec, run)


async def earlier(runtime: Runtime, *, back: int = 1) -> str:
    """The ``back``-th answer before this run in its conversation — a run of the same agent,
    tenant, user and thread that ended, never a child run — rendered for :func:`audience`."""
    runs = runtime.agent.harness.runs
    seen = 0
    async for summary in runs.iterate(
        thread_id=runtime.thread, agent_id=runtime.agent_id, tenant=runtime.tenant
    ):
        if summary.run_id == runtime.run_id or not summary.status.final:
            continue
        run = await runs.get(summary.run_id, tenant=runtime.tenant)
        if run is None or run.user_id != runtime.user or run.parent_run_id is not None:
            continue
        seen += 1
        if seen == back:
            return await explained(runtime, run)
    if not seen:
        return "There is no earlier answer in this conversation."
    return f"There are only {seen} earlier answer(s) in this conversation."


async def explained(runtime: Runtime, run: RunRecord) -> str:
    """``run`` (an earlier run of ``runtime``'s conversation) as ``runtime``'s audience sees
    it: its record, and the memories it was shown, read in ``runtime``'s scope."""
    record = await recorded(runtime.agent, run.run_id, tenant=run.tenant_id)
    found = await _memories(runtime, record.memories if record is not None else [])
    # what the run's conversation window already carries is not said twice: the earlier
    # question, its answer and its tools' results, when that turn is in the window
    asked = query_of(run.input)
    said = any(s.role is Speaker.USER and s.content == asked for s in runtime.history or ())
    if await audience(runtime.agent) == "admin":
        return admin_view(run, record, found, in_window=said)
    own = [m for m in found if _principal(m.owner_principal or "") == _principal(runtime.user)]
    return client_view(run, record, [m.content for m in own], in_window=said)


async def audience(agent: Agent) -> Audience:
    """Whom the harness's key acts for: an ``admin`` or ``platform`` key administers the
    tenant (the administrator's view); any other key acts for end users (the client's)."""
    role = (await agent.harness._key_info()).role
    return "admin" if role in ADMINISTERING else "client"


async def _memories(runtime: Runtime, ids: Sequence[str]) -> list[MemoryResult]:
    """The memories ``ids`` names that ``runtime``'s scope may read (the memory service
    refuses the others); none without memory."""
    memory = runtime.run_memory
    if memory is None:
        return []
    found = []
    for memory_id in ids:
        try:
            found.append(await memory.ctx.advanced.memories.get(memory_id))
        except MemoryServiceError:
            continue  # gone, or not this scope's to read
    return found


def _principal(name: str) -> str:
    """``name`` as ``kind:id``: a bare id is a user's (agent-runs' ``as_principal``)."""
    return name if ":" in name else f"user:{name}"


# --------------------------------------------------------------------------- the views
def client_view(
    run: RunRecord, record: Record | None, memories: Sequence[str], *, in_window: bool = False
) -> str:
    """What the end user is shown of an earlier answer (the module says what it holds and
    never holds): plain text, built from the record alone — the question and the answer only
    when the conversation window does not carry them already (``in_window``)."""
    said = [
        f"Earlier question: {_quoted(query_of(run.input))}",
        f"Answer given: {_answer(run, record)}",
    ]
    lines = [
        *([] if in_window else said),
        f"When: {run.created_at.astimezone(UTC):%Y-%m-%d %H:%M} UTC. {_took(run)}",
    ]
    if record is None:
        lines.append("I have no record of the steps behind that answer, so I do not know them.")
        return "\n".join(lines)
    sources = record.sources
    lines.append(_steps(record))
    lines.append(_sources(sources, run))
    lines.append(_basis(record))
    if memories:
        lines.append("Your saved notes I used:\n" + "\n".join(f"- {m}" for m in memories))
    else:
        lines.append("Your saved notes I used: none.")
    if record.evidence in COVERED:
        lines.append(f"Memory: {COVERED[record.evidence]}.")
    lines.append(_approvals(record))
    lines.append(_limits(record, sources))
    return "\n".join(lines)


def admin_view(
    run: RunRecord,
    record: Record | None,
    memories: Sequence[MemoryResult],
    *,
    in_window: bool = False,
) -> str:
    """What an administrator is shown of an earlier answer: everything, as JSON — but what the
    conversation window carries already (``in_window``: the input, the output and each step's
    output), which is not said twice."""
    carried = {"checkpoint", "input", "output"} if in_window else {"checkpoint"}
    steps = {"steps": {"__all__": {"output"}}} if in_window else {}
    seen = {
        "run": run.model_dump(mode="json", exclude=carried),
        "record": record.model_dump(mode="json", exclude=steps) if record is not None else None,
        "memories": [m.model_dump(mode="json") for m in memories],
    }
    return json.dumps(seen, ensure_ascii=False, default=str)


def _quoted(text: str) -> str:
    return json.dumps(text, ensure_ascii=False)


def _answer(run: RunRecord, record: Record | None) -> str:
    """The answer the run gave, without the name of any tool it called."""
    said = _without(text_of(run.output), {s.tool for s in record.steps} if record else set())
    if run.status is RunStatus.SUCCESS:
        return _quoted(said)
    if run.status is RunStatus.PARTIAL:
        return f"{_quoted(said)} (cut short: it reached its spending limit)"
    return f"none: that request ended {run.status.value.lower()} before an answer"


def _took(run: RunRecord) -> str:
    """How long the run took: its working time, and in all when it waited (for a person)."""
    worked = run.worked_seconds
    wall = (run.updated_at - run.created_at).total_seconds()
    said = f"It took {_duration(worked)} of work"
    if wall - worked >= 1:
        said += f", {_duration(wall)} in all (waiting included)"
    return said + "."


def _duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    minutes, rest = divmod(round(seconds), 60)
    return f"{minutes} min {rest} s"


def _steps(record: Record) -> str:
    if not record.steps:
        return "What I did: I answered without looking anything up (0 steps)."
    lines = [f"What I did ({len(record.steps)} step(s)):"]
    for n, step in enumerate(record.steps, start=1):
        line = f"{n}. {step.said}"
        if step.sources:
            line += " — from " + "; ".join(_cited(c) for c in step.sources)
        if step.status != "ok":
            line += f" — this step did not complete ({_ended(step.status)})"
        lines.append(line)
    return "\n".join(lines)


def _ended(status: str) -> str:
    return {
        "timeout": "it ran out of time",
        "rejected": "it was not allowed",
        "cancelled": "it was stopped",
    }.get(status, "it failed")


def _cited(citation: Citation) -> str:
    return ", ".join(part for part in (citation.name, citation.date, citation.link) if part)


def _sources(sources: Sequence[Citation], run: RunRecord) -> str:
    if not sources:
        return "Sources: no step returned a source, so I cannot name one."
    lines = ["Sources:"]
    for citation in sources:
        age = _age(citation.date, run.updated_at)
        lines.append(f"- {_cited(citation)}" + (f" ({age})" if age else ""))
    return "\n".join(lines)


def _age(date: str | None, answered: datetime) -> str | None:
    """How long before the answer a source's date was, when the date is an ISO date."""
    try:
        when = datetime.fromisoformat(date or "")
    except ValueError:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    days = (answered - when).days
    if days < 0:
        return None
    return "the same day as the answer" if days == 0 else f"{days} day(s) before the answer"


def _basis(record: Record) -> str:
    """The answer tied to the steps it rests on (the ones that returned something)."""
    done = [n for n, s in enumerate(record.steps, start=1) if s.status == "ok"]
    if not done:
        return "Basis: no step returned anything, so the answer rests on the conversation alone."
    cited = [n for n in done if record.steps[n - 1].sources]
    said = f"Basis: the answer rests on what step(s) {_numbers(done)} returned"
    if cited:
        said += f"; step(s) {_numbers(cited)} named their source"
    return said + "."


def _numbers(numbers: Sequence[int]) -> str:
    return ", ".join(map(str, numbers))


def _approvals(record: Record) -> str:
    if not record.approvals:
        return "Approvals: none was needed."
    lines = ["Approvals:"]
    for approval in record.approvals:
        who = _role(approval.get("assignee"))
        decided = DECIDED.get(str(approval.get("decision")), "decided on")
        step = _step_of(record, approval)
        lines.append(f"- {who} {decided} {step}.")
    return "\n".join(lines)


def _role(assignee: Any) -> str:
    """Who decided, by role only: a ``role:`` assignee's role, else "a reviewer"."""
    kind, _, name = str(assignee or "").partition(":")
    return f"a reviewer ({name})" if kind == "role" and name else "a reviewer"


def _step_of(record: Record, approval: Mapping[str, Any]) -> str:
    """The step an approval was of: its tool's first call in the record, else "a step"."""
    for n, step in enumerate(record.steps, start=1):
        if step.tool == approval.get("tool"):
            return f"step {n}"
    return "a step that did not run"


def _limits(record: Record, sources: Sequence[Citation]) -> str:
    """What could be wrong with the answer, as the record shows it."""
    limits = []
    failed = [n for n, s in enumerate(record.steps, start=1) if s.status != "ok"]
    if failed:
        limits.append(f"step(s) {_numbers(failed)} did not complete")
    if not sources:
        limits.append("no source was recorded, so it could not be checked against one")
    elif len(sources) == 1:
        limits.append("it rests on a single source")
    undated = [c.name for c in sources if not c.date]
    if undated:
        limits.append(f"{', '.join(undated)} gave no date, so how current it is is unknown")
    if record.evidence in ("INCOMPLETE", "INSUFFICIENT"):
        limits.append(COVERED[record.evidence])
    if not limits:
        return "Limits: none that the record shows."
    return "Limits: " + "; ".join(limits) + "."


__all__ = [
    "EARLIER",
    "RECORD",
    "Measured",
    "Record",
    "Step",
    "admin_view",
    "audience",
    "client_view",
    "conversation_budget",
    "earlier",
    "earlier_tool",
    "keep",
    "measured",
    "measuring",
    "record_of",
    "recorded",
]
