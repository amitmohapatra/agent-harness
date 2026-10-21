"""The OpenAI Agents SDK's conversation (``Session``), kept in the memory service's thread.

An agent run without a session of its own (``framework_options={"session": ...}``) is given a
:class:`MemorySession` while memory is on for the run (``bridges.history``: the policy's
``Thread.BRIDGED``): the SDK reads the conversation's earlier
items from it and adds each turn's to it, as with its own sessions — and they are the messages
of the run's thread in the memory service (``thread=``; a run with no conversation is its own),
which every agent of the user's learns from. The memory service then leaves the recent
conversation out of the pushed context (the session gives it), and the harness does not write
the transcript a second time (the session wrote it).

Each item is one message of the thread: a message as its text (``USER``, ``ASSISTANT``,
``SYSTEM``), any other item — a tool call, its output, reasoning — as its JSON, an ``INTERNAL``
message; each says which run and attempt added it. The memory context the harness puts first in
the input is never added. The thread only grows, so removing is recorded: ``pop_item`` and
``clear_session`` add an ``EVENT`` that later reads honour. A run run again from its input (after
a pause the SDK does not keep, a crash) adds its input again: what its earlier attempts added is
left out of what the SDK reads — one run, one copy of its turn; an attempt continuing the SDK's
own saved run (an approval of a ``needs_approval`` tool) continues its earlier attempt's items.

A read returns at most :data:`HISTORY_LIMIT` messages of the thread (the newest), the
service's page. The calls are the run's: made in its memory scope, they fail as a memory call
fails, and that failure is the SDK's (the run fails).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Final

from trellis.memory.models import Message

if TYPE_CHECKING:
    from trellis.harness.runtime import Runtime

#: The newest messages a read of the thread returns (the service's largest page).
HISTORY_LIMIT: Final = 500
#: The ``custom_metadata`` keys a session message carries.
KIND: Final = "trellis_session"
RUN: Final = "trellis_run"
ATTEMPT: Final = "trellis_attempt"
FRESH: Final = "trellis_fresh"
FORMAT: Final = "trellis_format"
TARGET: Final = "trellis_target"
THROUGH: Final = "trellis_through"


class Entry(StrEnum):
    """What a session message of the thread is (its :data:`KIND`)."""

    ITEM = "item"
    POP = "pop"
    CLEAR = "clear"


class Format(StrEnum):
    """How a session item is kept in its message (its :data:`FORMAT`)."""

    TEXT = "text"
    JSON = "json"


#: Where a session message came from (its ``source_system``).
SOURCE: Final = "trellis-session"
ROLES: Final = {"user": "USER", "assistant": "ASSISTANT", "system": "SYSTEM"}
TEXT_PARTS: Final = frozenset({"input_text", "output_text", "text"})


def bridgeable(runtime: Runtime) -> bool:
    """Whether a run's conversation can be a :class:`MemorySession`: memory is on for the run —
    its context (the push) and its records both. (Whether it is, the history policy says:
    ``bridges.history``.)"""
    return (
        runtime.run_memory is not None and runtime.uses("memory_push") and runtime.uses("records")
    )


@dataclass(frozen=True, slots=True)
class _Kept:
    """A session item as the thread holds it."""

    message_id: str
    sequence: int
    run: str
    attempt: int
    fresh: bool
    item: dict[str, Any]


class MemorySession:
    """The SDK's ``Session`` for one attempt of one run. ``fresh``: the attempt runs from the
    run's input (not from the SDK's saved run)."""

    session_settings = None

    def __init__(self, runtime: Runtime, *, fresh: bool) -> None:
        assert runtime.run_memory is not None
        self.runtime = runtime
        self.fresh = fresh
        self.session_id = runtime.thread or runtime.run_id
        self._added = 0

    @property
    def _ctx(self) -> Any:
        assert self.runtime.run_memory is not None
        return self.runtime.run_memory.ctx

    # ------------------------------------------------------------------ the protocol
    async def get_items(self, limit: int | None = None) -> list[Any]:
        items = [kept.item for kept in await self._read()]
        return items if limit is None else items[-limit:] if limit > 0 else []

    async def add_items(self, items: list[Any]) -> None:
        context = self.runtime.context
        kept = [
            item
            for item in items
            if not (
                item.get("role") == "system"
                and context is not None
                and item.get("content") == context
            )
        ]
        if not kept:
            return
        await self._append([_message(item, self._lineage()) for item in kept])

    async def pop_item(self) -> Any | None:
        kept = await self._read()
        if not kept:
            return None
        last = kept[-1]
        await self._append(
            [_event("a session item was removed", {KIND: Entry.POP, TARGET: last.message_id})]
        )
        return last.item

    async def clear_session(self) -> None:
        messages = await self._messages()
        if not messages:
            return
        through = max(m.sequence for m in messages)
        cleared = {KIND: Entry.CLEAR, THROUGH: through}
        await self._append([_event("the session was cleared", cleared)])

    # ------------------------------------------------------------------ internals
    def _lineage(self) -> dict[str, Any]:
        return {
            KIND: Entry.ITEM,
            RUN: self.runtime.run_id,
            ATTEMPT: self.runtime.attempt,
            FRESH: self.fresh,
        }

    async def _append(self, messages: list[Message]) -> None:
        self._added += 1
        run, attempt = self.runtime.run_id, self.runtime.attempt
        key = f"{run}:{attempt}:session:{self._added}"
        await self._ctx.history.add(messages, idempotency_key=key)

    async def _messages(self) -> list[Any]:
        return list(await self._ctx.history(limit=HISTORY_LIMIT, include_internal=True))

    async def _read(self) -> list[_Kept]:
        """The items the SDK reads: the thread's session items, less those removed and those
        an attempt run again from its input replaces."""
        popped: set[str] = set()
        cleared = 0
        kept: list[_Kept] = []
        for message in await self._messages():
            meta = (message.model_extra or {}).get("custom_metadata") or {}
            kind = meta.get(KIND)
            if kind == Entry.POP:
                popped.add(str(meta.get(TARGET)))
            elif kind == Entry.CLEAR:
                cleared = max(cleared, int(meta.get(THROUGH) or 0))
            elif kind == Entry.ITEM:
                item = _item(message.content, meta, message.role)
                kept.append(
                    _Kept(
                        message.message_id,
                        message.sequence,
                        str(meta.get(RUN)),
                        int(meta.get(ATTEMPT) or 1),
                        bool(meta.get(FRESH, True)),
                        item,
                    )
                )
        current = self.runtime.run_id, self.runtime.attempt, self.fresh
        latest = _latest_fresh(kept, current)
        return [
            k
            for k in kept
            if k.message_id not in popped
            and k.sequence > cleared
            and k.attempt >= latest.get(k.run, 0)
        ]


def _latest_fresh(kept: list[_Kept], current: tuple[str, int, bool]) -> dict[str, int]:
    """Per run, the last attempt that ran from the run's input: what it added replaces what its
    earlier attempts did (the current attempt counts, for its own run)."""
    latest: dict[str, int] = {}
    for k in kept:
        if k.fresh:
            latest[k.run] = max(latest.get(k.run, 0), k.attempt)
    run, attempt, fresh = current
    if fresh:
        latest[run] = attempt
    return latest


def _text(item: dict[str, Any]) -> str | None:
    """A message item's text, when it is only text (``None`` otherwise)."""
    if item.get("role") not in ROLES or item.get("type", "message") != "message":
        return None
    content = item.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list) and all(
        isinstance(p, dict) and p.get("type") in TEXT_PARTS and isinstance(p.get("text"), str)
        for p in content
    ):
        return "".join(p["text"] for p in content)
    return None


def _message(given: dict[str, Any], lineage: dict[str, Any]) -> Message:
    """An item of the SDK's (an input item: a JSON object) as a message of the thread."""
    text = _text(given)
    if text is not None:
        role = ROLES[given["role"]]
        return Message(
            role=role,  # type: ignore[arg-type]
            content=text,
            custom_metadata={**lineage, FORMAT: Format.TEXT},
            source_system=SOURCE,
        )
    output = str(given.get("type", "")).endswith("_output")
    return Message(
        role="TOOL" if output else "AGENT",
        kind="INTERNAL",
        content=json.dumps(given, default=str, separators=(",", ":")),
        custom_metadata={**lineage, FORMAT: Format.JSON},
        source_system=SOURCE,
    )


def _item(content: str, meta: dict[str, Any], role: str) -> dict[str, Any]:
    if meta.get(FORMAT) == Format.JSON:
        found = json.loads(content)
        return found if isinstance(found, dict) else {"content": found}
    named = {v: k for k, v in ROLES.items()}
    return {"role": named.get(role, "user"), "content": content}


def _event(text: str, meta: dict[str, Any]) -> Message:
    return Message(
        role="EVENT", kind="INTERNAL", content=text, custom_metadata=meta, source_system=SOURCE
    )


__all__ = ["HISTORY_LIMIT", "MemorySession", "bridgeable"]
