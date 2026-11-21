"""A conversation across runs: its earlier turns, a message sent while it is busy, and the id
each of its messages keeps.

A conversation is a user's thread (``identity.conversation_key``: its tenant's, its user's
and its thread's); each run with ``thread=`` is a turn of it. Who keeps its history across
runs is ADR 0003's decision (point 6), unchanged: the framework's own thread when the target
keeps one, else the memory service's window. Where neither does — memory off, a target that
keeps no thread — the run store already holds every turn (its question, its answer):
:func:`earlier` reads them, so a follow-up still has the conversation (ADR 0005).

A message sent to a conversation one of whose runs works does what its ``mode`` says
(:class:`Mode`): it waits its turn (``queue``, the default — what a second message always
did), it cancels the working run and continues the conversation from there (``interrupt``,
wherever that run works), or it is given to the working run at its framework's next step
(``steer``: the target's ``Adapter.steers``), and that run's answer answers it. A target
with no hook between two of its steps steers nothing, and nothing reaches a run working in
another process: the message waits its turn instead, said once per agent. The turns are taken
in the order of the run store's line (``line``), on whichever process each message arrives.

Every message has an id (:func:`message_id`): a digest of the conversation and the message's
own name in it — the run's question (``<run>:input``), its answer (``<run>:answer``, the id
its text streams under), a tool call's result (``result:<call>``), an assistant message's
calls (``call:<first call>``), anything else the run said by what it said
(``<run>:said:...``, :func:`said_id`) — so the same message keeps its id across a resume, a
replay and a cleared tool result, and two conversations never share one (:func:`identified`).
A paused attempt has not answered: its last message is what it said, not the run's answer,
so the answer the resumed attempt gives takes the answer's id (the memory service keeps the
first message of an id it has seen, so a draft under it would stand for the answer).
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import functools
import hashlib
import itertools
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Final

from trellis.contracts import Interrupt, InterruptDecision, InterruptResolution, RunStatus
from trellis.harness.bounded import LRU, SAID_ONCE
from trellis.harness.governance.decision import CONVERSATION, reach
from trellis.harness.history import Said, Speaker, notification_in
from trellis.harness.line import Here

if TYPE_CHECKING:
    from trellis.harness.line import Cache, Line
    from trellis.harness.runtime import Runtime
    from trellis.runs import ResolutionEntry

log = logging.getLogger("trellis.harness")

#: The most earlier runs of a conversation :func:`earlier` reads from the run store.
EARLIER_RUNS: Final = 10
#: How many of the store's newest runs of a thread and an ending :func:`earlier` lists (one
#: listing an ending: other users' threads named alike are among them).
EARLIER_PAGE: Final = 50
#: A run's answered interrupts read at once, for what the person allowed (:func:`allowed`).
RESOLUTIONS_PAGE: Final = 50
#: The endings whose question and answer are a turn of the conversation; a cancelled run's
#: question is one too (an ``interrupt`` continues from it), with no answer.
ANSWERED: Final = frozenset({RunStatus.SUCCESS, RunStatus.PARTIAL})
ASKED: Final = ANSWERED | {RunStatus.CANCELLED}
#: The ``RunStart.metadata`` key a run sent with ``mode="steer"`` keeps the run it was given
#: to under (its answer is that run's: :func:`earlier` keeps it once).
STEERED_INTO: Final = "steered_into"
#: The reason a run cancelled by a message sent with ``mode="interrupt"`` ends with.
INTERRUPTED: Final = "interrupted by a newer message of the conversation"
#: The context's line for what an earlier turn of the conversation still waits on a person for
#: (:func:`waiting`): each paused run's question, and the call it holds back.
WAITING: Final = (
    "An earlier turn of this conversation still waits for the user, so it has not happened "
    "yet: {waiting}. The user's new message is not their answer to it (they answer it where "
    "it was asked): do not do it again, and do not say it was done."
)


class Mode(StrEnum):
    """What a message sent to a conversation one of whose runs works in this process does."""

    #: it waits for that run to end, then runs (the default)
    QUEUE = "queue"
    #: that run is cancelled; the message continues the conversation from there
    INTERRUPT = "interrupt"
    #: it is given to that run at its framework's next step, and that run's answer answers
    #: it (a target that cannot take it: ``QUEUE``)
    STEER = "steer"


# --------------------------------------------------------------------------- message ids
def message_id(conversation: str, native: str) -> str:
    """The harness's id of the message named ``native`` in ``conversation``."""
    digest = hashlib.sha256(f"{conversation}\0{native}".encode()).hexdigest()
    return f"msg_{digest[:32]}"


def question_id(conversation: str, run_id: str) -> str:
    """The id of the question run ``run_id`` was asked (a message steered into another run
    keeps its own run's)."""
    return message_id(conversation, f"{run_id}:input")


def answer_id(conversation: str, run_id: str) -> str:
    """The id of run ``run_id``'s answer: the one its text streams under, and the one it is
    written to memory with."""
    return message_id(conversation, f"{run_id}:answer")


def result_id(conversation: str, call_id: str) -> str:
    """The id of the result of tool call ``call_id``: its ``TOOL_CALL_RESULT``'s message."""
    return message_id(conversation, f"result:{call_id}")


def said_id(conversation: str, run_id: str, said: Said, occurrence: int) -> str:
    """The id of a message run ``run_id`` said that names nothing else (an assistant's words
    on the way, a result with no call id): by who said what, and which time it said it
    (``occurrence``, from 0) — the same on every attempt that says it again (a resumed graph
    sends its whole transcript), and another for anything said differently."""
    digest = hashlib.sha256(f"{said.role.value}\0{said.content}".encode()).hexdigest()[:16]
    return message_id(conversation, f"{run_id}:said:{digest}:{occurrence}")


def identified(said: Sequence[Said], runtime: Runtime, *, answered: bool = True) -> list[Said]:
    """An attempt's messages (its question first, when it has one) each with its id: one
    named already keeps it (a steered message, an earlier turn); the question is the run's
    input; a tool result the call it answers; an assistant message that calls tools its
    first call; the last thing the assistant said without one, the run's answer — when the
    attempt ``answered`` (a paused one has not: it answers on its resume); anything else
    what it said (:func:`said_id`)."""
    conversation, run_id = runtime.conversation, runtime.run_id
    last = max(
        (n for n, s in enumerate(said) if s.role is Speaker.ASSISTANT and not s.tool_calls),
        default=-1,
    )
    seen: dict[tuple[Speaker, str], int] = {}
    out: list[Said] = []
    for n, s in enumerate(said):
        if s.id is not None:
            found = s.id
        elif s.role is Speaker.USER and n == 0:
            found = question_id(conversation, run_id)
        elif s.role is Speaker.TOOL and s.tool_call_id:
            found = result_id(conversation, s.tool_call_id)
        elif s.tool_calls:
            found = message_id(conversation, f"call:{s.tool_calls[0].id}")
        elif n == last and answered:
            found = answer_id(conversation, run_id)
        else:
            occurrence = seen.get((s.role, s.content), 0)
            seen[(s.role, s.content)] = occurrence + 1
            found = said_id(conversation, run_id, s, occurrence)
        out.append(dataclasses.replace(s, id=found))
    return out


# --------------------------------------------------------------------------- earlier turns
async def allowed(runtime: Runtime) -> frozenset[str]:
    """What the person allowed for this conversation: the tools (``governance.decision.reach``)
    whose approval offered :data:`~trellis.harness.governance.decision.CONVERSATION` and was
    answered with it, in any run of this tenant, user and thread — whichever agent or worker
    ran it, a child run's and a resumed run's included — as the run store keeps every answered
    interrupt (``resolutions``, append-only). Read once per attempt, and only when a call
    asks; a run with no conversation of its own has none."""
    if runtime.allowed is None:
        runtime.allowed = await _allowed(runtime)
    return runtime.allowed


async def _allowed(runtime: Runtime) -> frozenset[str]:
    thread = runtime.thread
    if thread is None or thread == runtime.run_id:
        return frozenset()
    runs, tenant = runtime.agent.harness.runs, runtime.tenant
    found: set[str] = set()
    async for summary in runs.iterate(thread_id=thread, tenant=tenant):
        record = await runs.get(summary.run_id, tenant=tenant)
        if record is None or record.user_id != runtime.user or record.last_resolution is None:
            continue
        cursor: str | None = None
        while True:
            page = await runs.resolutions(
                record.run_id, cursor=cursor, limit=RESOLUTIONS_PAGE, tenant=tenant
            )
            found.update(r for r in map(_allowing, page.items) if r is not None)
            if (cursor := page.next_cursor) is None:
                break
    return frozenset(found)


def _allowing(entry: ResolutionEntry) -> str | None:
    """What one answered interrupt allowed for the conversation, if anything."""
    interrupt, resolution = entry.interrupt, entry.resolution
    call = interrupt.tool_call
    if (
        call is None
        or not remembered(interrupt, resolution)
        or resolution.decision is not InterruptDecision.APPROVE
        or resolution.remember != "run"
    ):
        return None
    return reach(call.tool, interrupt.assignee)


def remembered(interrupt: Interrupt, resolution: InterruptResolution) -> bool:
    """Whether the answer to a call's approval covers the tool's later calls (``remember``
    beyond ``"once"``): where the question offered it
    (:data:`~trellis.harness.governance.decision.CONVERSATION`), or offered no choices of its
    own (a framework's own approval, such as the OpenAI Agents SDK's ``needs_approval``: its
    ``always_approve``)."""
    if resolution.remember == "once" or interrupt.tool_call is None:
        return False
    return not interrupt.options or CONVERSATION.value in interrupt.option_values


async def earlier(
    runtime: Runtime,
    *,
    whole: bool = False,
    budget: int | None = None,
    held: Set[str] = frozenset(),
) -> list[Said]:
    """The conversation's earlier turns, oldest first, as the run store keeps them: each
    turn's question and answer (a cancelled turn's question alone), of this agent, tenant,
    user and thread — the run's own and other users' left out — at most
    :data:`EARLIER_RUNS` turns. ``whole``: a turn this process kept the transcript of
    (``Agent.results``) is that transcript — its tool calls and their results too, the
    figures behind its answer — for a follow-up ("why was that?", "redo it with X").
    ``budget``: the tokens they may take (``Agent.conversation_budget``), spent as the memory
    service spends its window's (:func:`_fitted`). ``held``: the message ids the caller has
    of the conversation already (memory's window: ``Agent.caught_up``) — only the turns after
    the newest one among them are read (one run each) and given, none when the newest is, or
    when none of the listed turns is among them (nothing says where they would go). A run
    with no conversation of its own, or a child run, has none."""
    thread = runtime.thread
    if runtime.parent is not None or thread is None or thread == runtime.run_id:
        return []
    found = sorted(await _asked(runtime, thread), key=lambda r: r.created_at)[-EARLIER_RUNS:]
    conversation = runtime.conversation
    if held:  # newest first: the turns before the first one the caller holds are the newer
        newer = list(
            itertools.takewhile(
                lambda turn: _lacking(held, conversation, turn.run_id), reversed(found)
            )
        )
        found = newer[::-1] if len(newer) < len(found) else []
    records = found
    kept = runtime.agent.results.turns(conversation) if whole else {}
    turns: list[list[Said]] = []
    previous: Any = None
    for record in records:
        # a run steered into another ran no attempt of its own: it has no transcript kept
        transcript = kept.get(record.run_id) if record.status in ANSWERED else None
        if transcript:
            turns.append(list(transcript))
            previous = record
            continue
        # a turn of the chat's own for a background task's outcome asked its notification
        told = notification_in(record.input)
        asked = dataclasses.replace(
            told or Said(Speaker.USER, _text(record.input)),
            id=question_id(conversation, record.run_id),
        )
        answer = record.output if record.status in ANSWERED else None
        if (
            previous is not None
            and record.steered_into == previous.run_id
            and answer == previous.output
        ):
            # given to that run: its answer is this one's
            turns[-1].insert(len(turns[-1]) - 1, asked)
            continue
        turns.append([asked])
        if answer not in (None, ""):
            turns[-1].append(
                Said(Speaker.ASSISTANT, _text(answer), id=answer_id(conversation, record.run_id))
            )
        previous = record if answer not in (None, "") else None
    return [s for turn in (turns if budget is None else _fitted(turns, budget)) for s in turn]


async def waiting(runtime: Runtime) -> str | None:
    """The context's line (:data:`WAITING`) for what the conversation's earlier turns still
    wait on a person for: each paused run of this agent, tenant, user and thread (not the
    run's own, nor a child's: its parent's run asks it), by the question its record awaits
    and the call it holds back — ``None`` when none waits. A paused run leaves the line, so a
    new message is a new run, never the answer (that is ``agent.resume``'s): told nothing, its
    model did the waiting action again. Listed only for a conversation a turn of which paused
    in this process (``Agent.waiting_in``), so a turn with nothing waiting costs the run store
    nothing more; a run paused in another process is not known here."""
    thread, waits = runtime.thread, runtime.agent.waiting_in
    if runtime.parent is not None or thread is None or runtime.conversation not in waits:
        return None
    runs, tenant = runtime.agent.harness.runs, runtime.tenant
    asked: list[str] = []
    async for summary in runs.iterate(
        agent_id=runtime.agent_id,
        thread_id=thread,
        status=RunStatus.PAUSED,
        tenant=tenant,
        limit=EARLIER_PAGE,
        max_pages=1,
    ):
        if summary.awaiting is None or summary.run_id == runtime.run_id:
            continue
        record = await runs.get(summary.run_id, tenant=tenant)
        if record is not None and record.user_id == runtime.user and not record.parent_run_id:
            asked.append(_awaited(summary.awaiting))
    if not asked:
        waits.discard(runtime.conversation)  # answered or ended: nothing to list next turn
        return None
    return WAITING.format(waiting="; ".join(asked))


def _awaited(interrupt: Interrupt) -> str:
    """A paused run's question, as asked, and the call it holds back (``tool(args)``)."""
    call = interrupt.tool_call
    if call is None:
        return interrupt.question
    args = json.dumps(call.args, ensure_ascii=False, default=str)
    return f"{interrupt.question} {call.tool}({args})"


@dataclass(frozen=True, slots=True)
class Turn:
    """An ended turn of a conversation as :func:`earlier` keeps it (``Agent.earlier_turns``):
    what it reads of the run's record, and no more (its checkpoint, its error, its metadata
    are not kept)."""

    run_id: str
    created_at: datetime
    status: RunStatus
    #: whose turn it is (another user's thread named alike is not this conversation's), and
    #: whether it is a child run (a sub-agent's: no turn)
    user: str | None
    child: bool
    input: Any
    output: Any
    steered_into: str | None


async def _asked(runtime: Runtime, thread: str) -> list[Turn]:
    """The conversation's turns that ended asked (:data:`ASKED`), the newest
    :data:`EARLIER_RUNS` of each ending: one listing an ending, of the store's newest
    :data:`EARLIER_PAGE` — the messages sent after this one, waiting their turn, are never
    listed — and each ended run read once (``Agent.earlier_turns``)."""
    agent, tenant = runtime.agent, runtime.tenant
    found: list[Turn] = []
    for status in sorted(ASKED):
        mine = 0
        async for summary in agent.harness.runs.iterate(
            agent_id=runtime.agent_id,
            thread_id=thread,
            status=status,
            tenant=tenant,
            limit=EARLIER_PAGE,
            max_pages=1,
        ):
            turn = agent.earlier_turns.get(summary.run_id)
            if turn is None:
                record = await agent.harness.runs.get(summary.run_id, tenant=tenant)
                if record is None:
                    continue
                turn = agent.earlier_turns[record.run_id] = Turn(
                    run_id=record.run_id,
                    created_at=record.created_at,
                    status=record.status,
                    user=record.user_id,
                    child=bool(record.parent_run_id),
                    input=record.input,
                    output=record.output,
                    steered_into=record.metadata.get(STEERED_INTO),
                )
            if turn.user == runtime.user and not turn.child and turn.run_id != runtime.run_id:
                found.append(turn)
                mine += 1
                if mine == EARLIER_RUNS:
                    break
    return found


def _lacking(held: Set[str], conversation: str, run_id: str) -> bool:
    """Whether ``held`` has neither the question nor the answer of the turn ``run_id``."""
    return held.isdisjoint((question_id(conversation, run_id), answer_id(conversation, run_id)))


def _fitted(turns: Sequence[Sequence[Said]], budget: int) -> list[Sequence[Said]]:
    """``turns`` (oldest first) within ``budget`` tokens (LangChain's estimate, as the budget
    and ``FitWindow`` count), spent as the memory service spends its window's (its
    ``replay_window``): each turn's text — its questions and what the assistant said, without
    the calls and their results — from the newest back while it fits (the newest always, a
    window has no gaps), then each of those whole, from the newest back, while it fits."""
    texts = [[dataclasses.replace(s, tool_calls=()) for s in turn if _spoken(s)] for turn in turns]
    shown: list[Sequence[Said]] = []
    used = 0
    for text in reversed(texts):
        cost = _tokens(text)
        if shown and used + cost > budget:
            break
        shown.append(text)
        used += cost
    for n, text in enumerate(shown):  # newest first
        turn = turns[len(turns) - 1 - n]
        more = _tokens(turn) - _tokens(text)
        if used + more > budget:
            break
        shown[n] = turn
        used += more
    return shown[::-1]


def _spoken(said: Said) -> bool:
    """Whether ``said`` is a turn's text: a question, or something the assistant said."""
    return said.role is Speaker.USER or (said.role is Speaker.ASSISTANT and bool(said.content))


def _tokens(said: Sequence[Said]) -> int:
    from langchain_core.messages.utils import count_tokens_approximately  # noqa: PLC0415

    from trellis.harness.adapters.langgraph import langchain_message  # noqa: PLC0415

    return count_tokens_approximately([langchain_message(s) for s in said])


def _text(value: Any) -> str:
    from trellis.harness.adapters.base import query_of  # noqa: PLC0415 - it imports the runtime

    return value if isinstance(value, str) else query_of(value) or str(value)


def rendered(history: Sequence[Said]) -> str:
    """Earlier turns as text, for a framework that takes them in its context."""
    lines = [f"{s.role.value}: {s.content}" for s in history if s.content]
    return "## Earlier in this conversation\n" + "\n".join(lines)


#: The heading of the latest turns a rendered window does not hold yet (:func:`latest`).
LATEST: Final = "## The latest turns of this conversation (after those above)"


def latest(history: Sequence[Said]) -> str:
    """The conversation's latest turns, which memory's window does not hold yet, as text after
    it (``Agent.unwritten_said``)."""
    lines = [f"{s.role.value}: {s.content}" for s in history if s.content]
    return f"{LATEST}\n" + "\n".join(lines)


# --------------------------------------------------------------------------- busy conversations
@dataclass(eq=False, slots=True)
class Steer:
    """A message given to a working run: its text, and the run it was sent as."""

    text: str
    run_id: str
    #: the run's framework took it in (at one of its steps)
    taken: bool = False


@dataclass(eq=False, slots=True)
class Live:
    """The run holding a conversation's turn in this process."""

    run_id: str
    #: its target takes a message between two of its steps (``Adapter.steers``)
    steers: bool
    inbox: list[Steer] = field(default_factory=list)
    #: set as it ends, with how it ended
    ended: asyncio.Event = field(default_factory=asyncio.Event)
    #: what waits for it to let go of the conversation's turn: each run then, in order, by the
    #: run's own task (``background.delivered``: an outcome it was too late to read)
    later: list[Callable[[], Awaitable[None]]] = field(default_factory=list)
    result: Any = None
    #: the conversation's line it took its turn in (none: a run outside any)
    line: Line | None = None

    async def steered(self, text: str, run_id: str) -> bool:
        """``text`` (sent as run ``run_id``) given to this run; once it ends, whether its
        framework took it in."""
        steer = Steer(text, run_id)
        self.inbox.append(steer)
        await self.ended.wait()
        return steer.taken


class Turns:
    """The conversations' line in this process (``line``: the run store keeps the order): the
    messages of a conversation recorded in the order they were sent (:meth:`accepting`), the
    runs waiting their turn or holding it here (:meth:`taken`), and the run holding each
    conversation's turn here, :class:`Live`, which a message sent with ``mode="interrupt"``
    cancels and one sent with ``mode="steer"`` reaches."""

    def __init__(self) -> None:
        self._live: dict[str, Live] = {}
        #: the runs in their conversation's line here, waiting their turn or holding it
        self._here: dict[str, Here] = {}
        #: the messages of a conversation being recorded here, one at a time (:meth:`accepting`)
        self._accepting: dict[str, tuple[asyncio.Lock, int]] = {}
        #: what the lines of this process read of the runs in them (each run read once)
        self.seen: Cache = LRU()

    @contextlib.asynccontextmanager
    async def accepting(self, key: str) -> AsyncIterator[None]:
        """A message of the conversation ``key`` recorded in the run store while no other of
        it is, in the order they were sent here (a lock is first come, first served): the
        store places each after the one before, and their turns follow that order (``line``)
        however long each start takes to answer."""
        lock, waiting = self._accepting.get(key) or (asyncio.Lock(), 0)
        self._accepting[key] = (lock, waiting + 1)
        try:
            async with lock:
                yield
        finally:
            lock, waiting = self._accepting[key]
            if waiting == 1:
                del self._accepting[key]
            else:
                self._accepting[key] = (lock, waiting - 1)

    @contextlib.asynccontextmanager
    async def taken(
        self, key: str, run_id: str, *, steers: bool = False, line: Line | None = None
    ) -> AsyncIterator[Live | None]:
        """Run ``run_id`` holding the turn of its conversation ``key`` while the block runs,
        once its ``line`` says it is its turn (a run with none — started by a release before the
        line, or outside a conversation — at once). ``None``: it takes no turn (cancelled while
        it waited, :meth:`cancel`, or ended in the run store)."""
        mine = self._here[run_id] = Here()
        try:
            if line is not None and not await line.turn(self._here, mine):
                yield None
                return
            mine.holding = True
            live = self._live[key] = Live(run_id, steers, line=line)
            try:
                yield live
            finally:
                if self._live.get(key) is live:  # (runs with no line may hold a turn together)
                    del self._live[key]
                live.ended.set()
                for then in live.later:
                    await then()
        finally:
            del self._here[run_id]
            mine.left.set_result(None)

    def cancel(self, run_id: str, reason: str) -> asyncio.Future[None] | None:
        """Run ``run_id``, waiting its turn here, is cancelled for ``reason``: it takes no
        turn. What is done once it left the line; ``None`` when it does not wait here. One
        that holds its turn and whose attempt has not begun yet (``Agent.running`` names it
        not) is marked: that attempt ends it ``CANCELLED`` as it begins (:meth:`cancelled`),
        and nothing waits for it — it may wait on the very caller (a slow memory read)."""
        mine = self._here.get(run_id)
        if mine is None:
            return None
        mine.cancelled = reason
        if mine.holding:
            return None
        mine.again.set()
        return mine.left

    async def interrupted(self, run_id: str) -> bool:
        """Whether run ``run_id`` holds its conversation's turn here and a newer message of the
        conversation, sent to another process with ``mode="interrupt"``, asks for it
        (``Line.interrupted``)."""
        for live in self._live.values():
            if live.run_id == run_id and live.line is not None:
                return await live.line.interrupted()
        return False

    def cancelled(self, run_id: str) -> str | None:
        """Why run ``run_id``, in line here, was cancelled while it waited or before its
        attempt began, if it was (``None`` too for a run not in a line here)."""
        mine = self._here.get(run_id)
        return None if mine is None else mine.cancelled

    def live(self, key: str) -> Live | None:
        """The run holding ``key``'s turn in this process, if one does."""
        return self._live.get(key)

    def steered(self, key: str, run_id: str) -> list[Steer]:
        """The messages given to run ``run_id`` since it last looked, now taken."""
        live = self._live.get(key)
        if live is None or live.run_id != run_id:
            return []
        fresh = [s for s in live.inbox if not s.taken]
        for s in fresh:
            s.taken = True
        return fresh


def steering(agent_id: str, framework: str) -> None:
    """A message sent with ``mode="steer"`` to a target that takes none waits its turn: said
    once per agent."""
    _said_unsteered(agent_id, framework)


def steering_elsewhere(agent_id: str) -> None:
    """A message sent with ``mode="steer"`` while the conversation's turn is held in another
    process waits its turn: nothing carries a message to a run working elsewhere. Said once per
    agent."""
    _said_elsewhere(agent_id)


@functools.lru_cache(maxsize=SAID_ONCE)  # said once per agent, of the most recent
def _said_elsewhere(agent_id: str) -> None:
    log.warning(
        "%s: the conversation's run works in another process, which no message reaches "
        "between its steps, so a message sent with mode='steer' waits for it to end, as "
        "mode='queue' does",
        agent_id,
        extra={"agent_id": agent_id},
    )


@functools.lru_cache(maxsize=SAID_ONCE)  # said once per agent, of the most recent
def _said_unsteered(agent_id: str, framework: str) -> None:
    log.warning(
        "%s: a %s target takes no message between its steps, so a message sent with "
        "mode='steer' waits for the working run to end, as mode='queue' does",
        agent_id,
        framework,
        extra={"agent_id": agent_id, "framework": framework},
    )


__all__ = [
    "ANSWERED",
    "ASKED",
    "EARLIER_RUNS",
    "INTERRUPTED",
    "STEERED_INTO",
    "WAITING",
    "Live",
    "Mode",
    "Steer",
    "Turns",
    "answer_id",
    "earlier",
    "identified",
    "message_id",
    "question_id",
    "rendered",
    "result_id",
    "steering",
    "steering_elsewhere",
    "waiting",
]
