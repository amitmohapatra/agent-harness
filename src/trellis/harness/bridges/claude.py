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

import functools
import logging
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Final

from trellis.harness.bounded import SAID_ONCE
from trellis.harness.history import in_ram
from trellis.harness.journal import Journal

if TYPE_CHECKING:
    from claude_agent_sdk import HookContext, HookInput
    from claude_agent_sdk.types import SyncHookJSONOutput

log = logging.getLogger("trellis.harness")

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

    own = dict(hooks or {})
    own[PROMPT_EVENT] = [*own.get(PROMPT_EVENT, ()), HookMatcher(hooks=[add_context])]
    return own


def session_store(options: Any, journal: Journal, agent_id: str) -> Any | None:
    """The store a query's session is kept in: the run's journal (:class:`JournalSessions`),
    unless the options keep it themselves (``None``) — their own store (one kept in RAM only,
    ``history.in_ram``, is said once), or file checkpointing, which the SDK keeps on the local
    disk only (said once)."""
    own = options.session_store
    if own is not None:
        if in_ram(own):
            _said_local(agent_id, type(own).__name__)
        return None
    if not options.enable_file_checkpointing:
        return JournalSessions(journal)
    _said_local(agent_id, "enable_file_checkpointing")
    return None


@functools.lru_cache(maxsize=SAID_ONCE)  # said once per agent and store, of the most recent
def _said_local(agent_id: str, store: str) -> None:
    log.warning(
        "Claude's session is kept on this machine only, not in the run's journal: a resume on "
        "another worker runs the query again from its prompt (give the options no "
        "session_store and no enable_file_checkpointing to keep it in the journal)",
        extra={"agent_id": agent_id, "store": store},
    )


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


__all__ = ["PROMPT_EVENT", "JournalSessions", "session_store", "with_context"]
