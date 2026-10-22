"""The OpenAI Agents SDK's conversation (``Session``), read from the memory service's thread.

An agent run without a session of its own (``framework_options={"session": ...}``, or a
conversation OpenAI's servers keep) is given a :class:`MemorySession` while memory is on for
the run: the SDK reads the conversation's earlier turns from it, as from its own sessions —
and they are the messages of the run's thread in the memory service, which every agent of the
user's learns from. The session owns the history (``history.owner``), so the memory service
leaves the recent conversation out of the pushed context.

The memory service is the source of truth: every turn is recorded there by the harness (the
run's transcript), so what the SDK adds to the session is not written a second time, and the
turns of the run itself (an earlier attempt's, before a pause) are not read back — the SDK
holds them. The thread only grows: removing an item through the session is refused
(:class:`ThreadReadOnly`). A read returns at most :data:`HISTORY_LIMIT` messages, the newest.
The calls are the run's: made in its memory scope, they fail as a memory call fails, and that
failure is the SDK's (the run fails).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

from trellis.contracts import ErrorCategory, HarnessError

if TYPE_CHECKING:
    from trellis.harness.runtime import Runtime

#: The newest messages a read of the thread returns (the service's largest page).
HISTORY_LIMIT: Final = 500
#: The thread's roles the SDK reads, as the SDK names them.
ROLES: Final = {"USER": "user", "ASSISTANT": "assistant"}


class ThreadReadOnly(HarnessError):
    """An item was to be removed from a conversation the memory service keeps: its thread is
    the record of what was said, and only grows."""

    code = "THREAD_READ_ONLY"
    category = ErrorCategory.MEMORY


class MemoryThread:
    """The thread store of a run whose session is the memory service's thread
    (``history.owner``: durable, so memory's window is left out)."""


#: The one :class:`MemoryThread`: what the adapter's ``thread_store`` names.
MEMORY_THREAD: Final = MemoryThread()


class MemorySession:
    """The SDK's ``Session`` for one attempt of one run."""

    session_settings = None

    def __init__(self, runtime: Runtime) -> None:
        assert runtime.run_memory is not None
        self.runtime = runtime
        self.session_id = runtime.thread or runtime.run_id

    async def get_items(self, limit: int | None = None) -> list[Any]:
        """The thread's earlier turns, oldest first (``limit``: the newest that many)."""
        assert self.runtime.run_memory is not None
        messages = await self.runtime.run_memory.ctx.history(limit=HISTORY_LIMIT)
        run_id = self.runtime.run_id
        items = [
            {"role": ROLES[m.role], "content": m.content}
            for m in messages
            if m.role in ROLES and (m.model_extra or {}).get("agent_run_id") != run_id
        ]
        return items if limit is None else items[-limit:] if limit > 0 else []

    async def add_items(self, items: list[Any]) -> None:
        """Nothing: the harness records the run's turns in the thread itself."""

    async def pop_item(self) -> Any | None:
        raise ThreadReadOnly("the conversation is the memory service's thread: it only grows")

    async def clear_session(self) -> None:
        raise ThreadReadOnly("the conversation is the memory service's thread: it only grows")


__all__ = ["HISTORY_LIMIT", "MEMORY_THREAD", "MemorySession", "MemoryThread", "ThreadReadOnly"]
