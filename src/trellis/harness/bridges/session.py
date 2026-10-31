"""The OpenAI Agents SDK's conversation (``Session``), read from the memory service's thread.

An agent run without a session of its own (``framework_options={"session": ...}``, or a
conversation OpenAI's servers keep) is given a :class:`MemorySession` while memory is on for
the run: the SDK reads the conversation's earlier turns from it, as from its own sessions —
and they are the messages of the run's thread in the memory service, which every agent of the
user's learns from. Memory owns the history (``history.MEMORY_THREAD``) and the session is
how the SDK reads it, so the memory service leaves the recent conversation out of the pushed
context, and each run's turns are written before it returns: the next run reads them.

The memory service is the source of truth: every turn is recorded there by the harness (the
run's transcript), so what the SDK adds to the session is not written a second time, and the
turns of the run itself (an earlier attempt's, before a pause) are not read back — the SDK
holds them. The thread only grows: removing an item through the session is refused
(:class:`ThreadReadOnly`). A read returns at most :data:`HISTORY_LIMIT` messages, the newest.
The calls are the run's: made in its memory scope, they fail as a memory call fails, and that
failure is the SDK's (the run fails).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Final

from trellis.contracts import ErrorCategory, HarnessError
from trellis.harness import snapshots
from trellis.harness.history import Said, Speaker, answered, said_of
from trellis.harness.tools.base import max_age
from trellis.memory.errors import NotFoundError

if TYPE_CHECKING:
    from trellis.harness.runtime import Runtime

#: The newest messages a read of the thread returns (the service's largest page).
HISTORY_LIMIT: Final = 500
#: The thread's roles the SDK reads: the chat, and the results of the tool calls made in it.
ROLES: Final = frozenset({"USER", "ASSISTANT", "TOOL"})


class ThreadReadOnly(HarnessError):
    """An item was to be removed from a conversation the memory service keeps: its thread is
    the record of what was said, and only grows."""

    code = "THREAD_READ_ONLY"
    category = ErrorCategory.MEMORY


class MemorySession:
    """The SDK's ``Session`` for one attempt of one run."""

    session_settings = None

    def __init__(self, runtime: Runtime) -> None:
        assert runtime.run_memory is not None
        self.runtime = runtime
        self.session_id = runtime.conversation

    async def get_items(self, limit: int | None = None) -> list[Any]:
        """The thread's earlier turns, oldest first (``limit``: the newest that many), as the
        SDK's input items: the messages with the tool calls made in them and their results, a
        call and its result never parted (``history.answered``). A thread the service does not
        have yet (``404``: nothing was said in it) has none."""
        assert self.runtime.run_memory is not None
        ctx = self.runtime.run_memory.ctx
        try:
            messages = await ctx.history(limit=HISTORY_LIMIT, include_internal=True)
        except NotFoundError:
            return []
        run_id = self.runtime.run_id
        said = [
            said_of({"role": m.role, "content": m.content, **(m.model_extra or {})})
            for m in messages
            if m.role in ROLES
            and (m.kind == "VISIBLE" or (m.model_extra or {}).get("tool_call_id"))
            and (m.model_extra or {}).get("agent_run_id") != run_id
        ]
        # an earlier value of a live tool is out of date, as the memory window's replay says,
        # and its tool stays offered: the model fetches it again
        said, stale = snapshots.out_of_date(said, self._max_age)
        if stale and self.runtime.offered is not None:
            self.runtime.offered |= set(stale)
        items = [item for s in answered(said) for item in input_items(s)]
        return items if limit is None else items[-limit:] if limit > 0 else []

    def _max_age(self, name: str) -> float | None:
        """How long a value of the run's tool ``name`` stays current (``None``: not said)."""
        found = self.runtime.toolbox.get(name)
        spec = found.spec if found is not None else self.runtime.native.get(name)
        return None if spec is None else max_age(spec)

    async def add_items(self, items: list[Any]) -> None:
        """Nothing: the harness records the run's turns in the thread itself."""

    async def pop_item(self) -> Any | None:
        raise ThreadReadOnly("the conversation is the memory service's thread: it only grows")

    async def clear_session(self) -> None:
        raise ThreadReadOnly("the conversation is the memory service's thread: it only grows")


def input_items(said: Said) -> list[dict[str, Any]]:
    """``said`` as the SDK's input items: a message, then one ``function_call`` per call it
    made; a tool's result as its ``function_call_output``."""
    if said.role is Speaker.TOOL:
        return [
            {"type": "function_call_output", "call_id": said.tool_call_id, "output": said.content}
        ]
    items: list[dict[str, Any]] = (
        [{"role": said.role.value, "content": said.content}]
        if said.content or not said.tool_calls
        else []
    )
    items += [
        {
            "type": "function_call",
            "call_id": c.id,
            "name": c.name,
            "arguments": json.dumps(dict(c.arguments), default=str),
        }
        for c in said.tool_calls
    ]
    return items


__all__ = [
    "HISTORY_LIMIT",
    "MemorySession",
    "ThreadReadOnly",
    "input_items",
]
