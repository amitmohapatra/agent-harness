"""The Claude Agent SDK's own slots: the memory context as a ``UserPromptSubmit`` hook, and the
session kept by a ``SessionStore`` in the run's journal.

* **Context.** The memory context reaches Claude the way Claude Code adds context to a prompt:
  a ``UserPromptSubmit`` hook answering ``additionalContext`` (:func:`with_context`), beside
  the options' own hooks. Every prompt of the run gets it — the first, and the one a resumed
  session is sent.
* **Session.** Claude keeps its conversation in a session the CLI writes to its own disk; the
  SDK mirrors it to a ``SessionStore`` and, resuming, writes it back for the CLI to continue
  (``resume=``). :class:`JournalSessions` is that store in the run's journal, which agent-runs
  keeps with the run: a run paused (or whose worker died) on one machine continues its session
  on any other, its built-in tools not run again. Options with a store of their own keep it
  (said once); options asking for file checkpointing, which the SDK keeps on the local disk
  only, keep that.

The memory service has no part in Claude's built-in memory (its ``CLAUDE.md`` files and
sub-agents' ``memory`` scopes are files the CLI reads): the SDK exposes no memory-tool backend
to put it behind.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Final

from trellis.harness.bridges import said
from trellis.harness.bridges.history import DURABLE_INSTEAD, in_ram
from trellis.harness.journal import Journal

if TYPE_CHECKING:
    from claude_agent_sdk import HookContext, HookInput
    from claude_agent_sdk.types import SyncHookJSONOutput

#: The hook event the context is added at.
PROMPT_EVENT: Final = "UserPromptSubmit"


def with_context(hooks: Mapping[str, Sequence[Any]] | None, context: str) -> dict[str, Any]:
    """The options' hooks with one more ``UserPromptSubmit`` matcher, whose hook adds
    ``context`` to every prompt (the options' own hooks run as they did)."""
    from claude_agent_sdk import HookMatcher

    async def add_context(
        given: HookInput, tool_use_id: str | None, hook: HookContext
    ) -> SyncHookJSONOutput:
        return {"hookSpecificOutput": {"hookEventName": PROMPT_EVENT, "additionalContext": context}}

    return with_hook(hooks, PROMPT_EVENT, HookMatcher(hooks=[add_context]))


def with_hook(
    hooks: Mapping[str, Sequence[Any]] | None, event: str, matcher: Any
) -> dict[str, Any]:
    """The options' hooks with ``matcher`` added at ``event``, after theirs."""
    own = dict(hooks or {})
    own[event] = [*own.get(event, ()), matcher]
    return own


def session_store(options: Any, journal: Journal, agent_id: str) -> Any | None:
    """The store a query's session is kept in — the run's per-step state, the journal's: the
    run's journal (:class:`JournalSessions`), unless the options keep it themselves (``None``):
    their own store — durable unless it is one kept in RAM (``bridges.history.in_ram``, said
    once) — or file checkpointing, which the SDK keeps on the local disk only (said once)."""
    own = options.session_store
    if own is not None:
        if in_ram(own):
            said(
                f"claude_session:{agent_id}",
                "%s: its %s keeps Claude's session in this process's memory only: a resume on "
                "another worker runs the query again from its prompt; give it %s",
                agent_id,
                type(own).__name__,
                DURABLE_INSTEAD["claude_agent_sdk"],
                extra={"agent_id": agent_id, "store": type(own).__name__},
            )
        return None
    if not options.enable_file_checkpointing:
        return JournalSessions(journal)
    said(
        f"claude_session:{agent_id}",
        "%s: the options checkpoint Claude's files on this machine's disk, so the session is "
        "kept there and not in the run's journal: a resume elsewhere runs the query again from "
        "its prompt; leave enable_file_checkpointing off to keep it in the journal",
        agent_id,
        extra={"agent_id": agent_id},
    )
    return None


class JournalSessions:
    """A ``SessionStore`` in the run's journal (``Journal.transcripts``): each session's
    transcript entries, by session (and sub-agent ``subpath``) — the run's own sessions, so the
    directory the CLI ran in (``project_key``) does not part them: a worker elsewhere runs in
    another one. Entries are kept once each by their ``uuid`` (the SDK retries a batch)."""

    def __init__(self, journal: Journal) -> None:
        self.journal = journal
        #: each session's entry uuids, read from the journal the first time it is appended to
        self._seen: dict[str, set[str]] = {}

    async def append(self, key: Mapping[str, Any], entries: list[dict[str, Any]]) -> None:
        name = _name(key)
        kept = self.journal.transcripts.setdefault(name, [])
        seen = self._seen.get(name)
        if seen is None:
            seen = self._seen[name] = {e["uuid"] for e in kept if "uuid" in e}
        for entry in entries:
            uuid = entry.get("uuid")
            if uuid is None or uuid not in seen:
                kept.append(entry)
                if uuid is not None:
                    seen.add(uuid)

    async def load(self, key: Mapping[str, Any]) -> list[dict[str, Any]] | None:
        kept = self.journal.transcripts.get(_name(key))
        return list(kept) if kept else None


def _name(key: Mapping[str, Any]) -> str:
    subpath = key.get("subpath")
    return f"{key['session_id']}/{subpath}" if subpath else str(key["session_id"])


__all__ = ["PROMPT_EVENT", "JournalSessions", "session_store", "with_context", "with_hook"]
