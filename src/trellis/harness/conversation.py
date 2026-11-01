"""A conversation across runs: its earlier turns, a message sent while it is busy, and the id
each of its messages keeps.

A conversation is a user's thread (``identity.conversation_key``: its tenant's, its user's
and its thread's); each run with ``thread=`` is a turn of it. Who keeps its history across
runs is ADR 0003's decision (point 6), unchanged: the framework's own thread when the target
keeps one, else the memory service's window. Where neither does — memory off, a target that
keeps no thread — the run store already holds every turn (its question, its answer):
:func:`earlier` reads them, so a follow-up still has the conversation (ADR 0005).

A message sent to a conversation one of whose runs works in this process does what its
``mode`` says (:class:`Mode`): it waits its turn (``queue``, the default — what a second
message always did), it cancels the working run and continues the conversation from there
(``interrupt``), or it is given to the working run at its framework's next step
(``steer``: the target's ``Adapter.steers``), and that run's answer answers it. A target
with no hook between two of its steps steers nothing: the message waits its turn instead,
said once per agent.

Every message has an id (:func:`message_id`): a digest of the conversation and the message's
own name in it — the run's question (``<run>:input``), its answer (``<run>:answer``, the id
its text streams under), a tool call's result (``result:<call>``), an assistant message's
calls (``call:<first call>``) — so the same message keeps its id across a resume, a replay
and a cleared tool result, and two conversations never share one (:func:`identified`).
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import functools
import hashlib
import logging
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Final

from trellis.contracts import RunStatus
from trellis.harness.bounded import SAID_ONCE
from trellis.harness.history import Said, Speaker

if TYPE_CHECKING:
    from trellis.harness.runtime import Runtime

log = logging.getLogger("trellis.harness")

#: The most earlier runs of a conversation :func:`earlier` reads from the run store.
EARLIER_RUNS: Final = 10
#: The endings whose question and answer are a turn of the conversation; a cancelled run's
#: question is one too (an ``interrupt`` continues from it), with no answer.
ANSWERED: Final = frozenset({RunStatus.SUCCESS, RunStatus.PARTIAL})
ASKED: Final = ANSWERED | {RunStatus.CANCELLED}
#: The ``RunStart.metadata`` key a run sent with ``mode="steer"`` keeps the run it was given
#: to under (its answer is that run's: :func:`earlier` keeps it once).
STEERED_INTO: Final = "steered_into"


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


def identified(said: Sequence[Said], runtime: Runtime) -> list[Said]:
    """An attempt's messages (its question first, when it has one) each with its id: one
    named already keeps it (a steered message, an earlier turn); the question is the run's
    input; a tool result the call it answers; an assistant message that calls tools its
    first call; the last thing the assistant said without one, the run's answer; anything
    else its run, attempt and place."""
    conversation, run_id = runtime.conversation, runtime.run_id
    last = max(
        (n for n, s in enumerate(said) if s.role is Speaker.ASSISTANT and not s.tool_calls),
        default=-1,
    )
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
        elif n == last:
            found = answer_id(conversation, run_id)
        else:
            found = message_id(conversation, f"{run_id}:{runtime.attempt}:{n}")
        out.append(dataclasses.replace(s, id=found))
    return out


# --------------------------------------------------------------------------- earlier turns
async def earlier(runtime: Runtime) -> list[Said]:
    """The conversation's earlier turns, oldest first, as the run store keeps them: each
    turn's question and answer (a cancelled turn's question alone), of this agent, tenant,
    user and thread — the run's own and other users' left out — at most
    :data:`EARLIER_RUNS` turns. A run with no conversation of its own, or a child run, has
    none."""
    thread = runtime.thread
    if runtime.parent is not None or thread is None or thread == runtime.run_id:
        return []
    runs, tenant = runtime.agent.harness.runs, runtime.tenant
    records = []
    async for summary in runs.iterate(
        agent_id=runtime.agent_id,
        thread_id=thread,
        tenant=tenant,
        limit=EARLIER_RUNS,
        max_pages=1,
    ):
        if summary.run_id == runtime.run_id or summary.status not in ASKED:
            continue
        record = await runs.get(summary.run_id, tenant=tenant)
        if record is not None and record.user_id == runtime.user and not record.parent_run_id:
            records.append(record)
    records.sort(key=lambda r: r.created_at)
    conversation = runtime.conversation
    history: list[Said] = []
    previous: Any = None
    for record in records:
        asked = Said(Speaker.USER, _text(record.input), id=question_id(conversation, record.run_id))
        answer = record.output if record.status in ANSWERED else None
        if (
            previous is not None
            and record.metadata.get(STEERED_INTO) == previous.run_id
            and answer == previous.output
        ):
            history.insert(len(history) - 1, asked)  # given to that run: its answer is this one's
            continue
        history.append(asked)
        if answer not in (None, ""):
            history.append(
                Said(Speaker.ASSISTANT, _text(answer), id=answer_id(conversation, record.run_id))
            )
        previous = record if answer not in (None, "") else None
    return history


def _text(value: Any) -> str:
    from trellis.harness.adapters.base import query_of  # noqa: PLC0415 - it imports the runtime

    return value if isinstance(value, str) else query_of(value) or str(value)


def rendered(history: Sequence[Said]) -> str:
    """Earlier turns as text, for a framework that takes them in its context."""
    lines = [f"{s.role.value}: {s.content}" for s in history if s.content]
    return "## Earlier in this conversation\n" + "\n".join(lines)


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
    result: Any = None

    async def steered(self, text: str, run_id: str) -> bool:
        """``text`` (sent as run ``run_id``) given to this run; once it ends, whether its
        framework took it in."""
        steer = Steer(text, run_id)
        self.inbox.append(steer)
        await self.ended.wait()
        return steer.taken


class Turns:
    """One run of a conversation at a time in this process: the others wait their turn, in
    order (``pipeline.attempt``). The run holding each turn is :class:`Live`, which a message
    sent with ``mode="interrupt"`` cancels and one sent with ``mode="steer"`` reaches."""

    def __init__(self) -> None:
        self._held: dict[str, tuple[asyncio.Lock, int]] = {}
        self._live: dict[str, Live] = {}

    @contextlib.asynccontextmanager
    async def taken(
        self, key: str, run_id: str | None = None, *, steers: bool = False
    ) -> AsyncIterator[Live | None]:
        lock, waiting = self._held.get(key) or (asyncio.Lock(), 0)
        self._held[key] = (lock, waiting + 1)
        try:
            async with lock:
                if run_id is None:
                    yield None
                    return
                live = self._live[key] = Live(run_id, steers)
                try:
                    yield live
                finally:
                    del self._live[key]
                    live.ended.set()
        finally:
            lock, waiting = self._held[key]
            if waiting == 1:
                del self._held[key]
            else:
                self._held[key] = (lock, waiting - 1)

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
    "STEERED_INTO",
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
]
