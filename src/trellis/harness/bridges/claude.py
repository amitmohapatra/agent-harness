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
  only, or to continue the CLI's latest session in its folder (``continue_conversation``),
  keep that.

The memory service has no part in Claude's built-in memory (its ``CLAUDE.md`` files and
sub-agents' ``memory`` scopes are files the CLI reads): the SDK exposes no memory-tool backend
to put it behind.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import functools
import logging
import weakref
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Final

from trellis.harness.bounded import SAID_ONCE
from trellis.harness.history import in_ram
from trellis.harness.journal import Journal

if TYPE_CHECKING:
    from claude_agent_sdk import HookContext, HookInput, HookMatcher
    from claude_agent_sdk.types import SyncHookJSONOutput

log = logging.getLogger("trellis.harness")

#: The hook event the context is added at.
PROMPT_EVENT: Final = "UserPromptSubmit"
#: The harness's own hook callbacks (:func:`harness_matcher`), told apart from a team's.
_OURS: weakref.WeakSet[Any] = weakref.WeakSet()


def harness_matcher(hook: Any) -> HookMatcher:
    """A hook matcher of the harness's own, with ``hook`` (:func:`ours` says so): a live
    session tells the CLI to wait for it as long as a person may take to answer about a
    background subagent's call (``claude_live.HOOK_SECONDS``); a team's own matchers keep
    the timeout they set."""
    from claude_agent_sdk import HookMatcher

    _OURS.add(hook)
    return HookMatcher(hooks=[hook])


def deciding_matcher(hook: Any) -> HookMatcher:
    """A :func:`harness_matcher` whose ``hook`` decides whether a call runs: Claude Code
    blocks the call when the hook fails (the SDK's ``on_failure="block"``; unset, a failed
    hook decides nothing and the call goes on to Claude Code's permission mode), and the SDK
    starts no query on a Claude Code too old to block."""
    return dataclasses.replace(harness_matcher(hook), on_failure="block")


def ours(matcher: Any) -> bool:
    """Whether a hook matcher is the harness's own (:func:`harness_matcher`)."""
    return bool(matcher.hooks) and all(hook in _OURS for hook in matcher.hooks)


#: The ``task_type`` of a dynamic workflow's task (it names no ``is_backgrounded``: it is
#: always in the background).
WORKFLOW_TASK: Final = "local_workflow"


@dataclasses.dataclass(frozen=True, slots=True)
class TaskStarted:
    """A task Claude Code started (its ``task_started``): its id — the ``agent_id`` its
    worker's calls name —, the tool call that started it, how the call described it, and
    whether it works in the background of that call (a subagent that says so,
    ``is_backgrounded``; a dynamic workflow, always)."""

    task_id: str
    call_id: str | None
    described: str
    background: bool


@dataclasses.dataclass(frozen=True, slots=True)
class TaskEnded:
    """A task that ended: its status and — ``final``, its ``task_notification`` — its
    summary; a ``task_updated`` to a terminal status, which may be all that comes, has
    none and may be followed by the notification."""

    task_id: str
    status: str
    summary: str
    final: bool


def task_said(message: Any) -> TaskStarted | TaskEnded | None:
    """What a message of Claude's stream says of a task (``None``: nothing): the one reader
    of its task messages, for a query's helpers and a live session's background runs."""
    from claude_agent_sdk import (
        TERMINAL_TASK_STATUSES,
        TaskNotificationMessage,
        TaskStartedMessage,
        TaskUpdatedMessage,
    )

    if isinstance(message, TaskStartedMessage):
        data = message.data
        background = bool(data.get("is_backgrounded")) or data.get("task_type") == WORKFLOW_TASK
        return TaskStarted(message.task_id, message.tool_use_id, message.description, background)
    if isinstance(message, TaskNotificationMessage):
        return TaskEnded(message.task_id, str(message.status), message.summary or "", True)
    if isinstance(message, TaskUpdatedMessage) and message.status in TERMINAL_TASK_STATUSES:
        return TaskEnded(message.task_id, str(message.status), "", False)
    return None


#: How long Claude Code waits for a hook whose matcher sets no ``timeout`` (seconds).
HOOK_SECONDS: Final = 60.0
#: The events at which a team's hook may put something in the place of what the call's other
#: hooks were given: the keys of its answer that do, and the key of the hook's input they
#: replace — a call's arguments before it runs, its result after.
REPLACES: Final[dict[str, tuple[tuple[str, ...], str]]] = {
    "PreToolUse": (("updatedInput",), "tool_input"),
    "PostToolUse": (("updatedToolOutput", "updatedMCPToolOutput"), "tool_response"),
}


class _Theirs:
    """The team's own hooks asked about one call at one event: when each has answered (and
    how long Claude Code waits for it), and what they said, in the order the answers came."""

    __slots__ = ("asked", "said")

    def __init__(self) -> None:
        self.asked: list[tuple[asyncio.Future[None], float]] = []
        self.said: list[Any] = []


def after_theirs(hooks: Mapping[str, Sequence[Any]]) -> dict[str, list[Any]]:
    """``hooks`` with the harness's own hooks of a tool call (:func:`ours`) answering after
    the team's. Claude Code asks every hook of an event at once, in the order they are
    registered — the team's first — and takes the arguments (``updatedInput``) of the one
    that answers last (seen with the bundled CLI, 2.1.325): asked beside them, the harness
    decided and recorded the call as the model made it, not as the team's hook left it. Here
    a harness hook waits for the team's hooks asked about the same call (no longer than
    Claude Code waits for them), reads the call as they left it — its arguments before it
    runs, its result after (:data:`REPLACES`) — and says nothing of a call one of them
    refused: Claude Code refuses it. Kept per call until the harness's hook read it."""
    calls: dict[tuple[str, str], _Theirs] = {}

    def theirs(event: str, callback: Any, seconds: float) -> Any:
        async def relay(given: Any, tool_use_id: str | None, hook: Any) -> Any:
            ref = tool_use_id or given.get("tool_use_id")
            answered: asyncio.Future[None] = asyncio.get_running_loop().create_future()
            found = calls.setdefault((event, ref), _Theirs()) if ref else _Theirs()
            found.asked.append((answered, seconds))
            try:
                answer = await callback(given, tool_use_id, hook)
                found.said.append(answer)
                return answer
            finally:
                answered.set_result(None)

        return relay

    def own(event: str, callback: Any) -> Any:
        keys, replaced = REPLACES[event]

        async def relay(given: Any, tool_use_id: str | None, hook: Any) -> Any:
            ref = tool_use_id or given.get("tool_use_id")
            found = calls.pop((event, ref), None) if ref else None
            for answered, seconds in found.asked if found is not None else ():
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(asyncio.shield(answered), seconds)
            said = [a for a in (found.said if found is not None else ()) if isinstance(a, dict)]
            if any(_refuses(answer) for answer in said):
                return {}
            specific = [answer.get("hookSpecificOutput") or {} for answer in said]
            left = [out[key] for out in specific for key in keys if key in out]
            if left:
                given = {**given, replaced: left[-1]}
            return await callback(given, tool_use_id, hook)

        _OURS.add(relay)
        return relay

    def ordered(event: str, matcher: Any) -> Any:
        if ours(matcher):
            return dataclasses.replace(matcher, hooks=[own(event, c) for c in matcher.hooks])
        seconds = matcher.timeout or HOOK_SECONDS
        return dataclasses.replace(
            matcher, hooks=[theirs(event, c, seconds) for c in matcher.hooks]
        )

    return {
        event: [ordered(event, m) if event in REPLACES else m for m in matchers]
        for event, matchers in hooks.items()
    }


def _refuses(answer: Mapping[str, Any]) -> bool:
    """Whether a hook's answer about a call refuses it, or stops the turn."""
    decided = (answer.get("hookSpecificOutput") or {}).get("permissionDecision")
    return decided == "deny" or answer.get("continue_") is False


def with_context(hooks: Mapping[str, Sequence[Any]] | None, context: str) -> dict[str, Any]:
    """The options' hooks with one more ``UserPromptSubmit`` matcher, whose hook adds
    ``context`` to every prompt — nothing, when there is none — (the options' own hooks run
    as they did)."""

    async def add_context(
        given: HookInput, tool_use_id: str | None, hook: HookContext
    ) -> SyncHookJSONOutput:
        if not context:
            return {}
        return {"hookSpecificOutput": {"hookEventName": PROMPT_EVENT, "additionalContext": context}}

    own = dict(hooks or {})
    own[PROMPT_EVENT] = [*own.get(PROMPT_EVENT, ()), harness_matcher(add_context)]
    return own


def session_store(options: Any, journal: Journal, agent_id: str) -> Any | None:
    """The store a query's session is kept in: the run's journal (:class:`JournalSessions`),
    unless the options keep it themselves (``None``) — their own store (one kept in RAM only,
    ``history.in_ram``, is said once), or file checkpointing or ``continue_conversation``,
    whose session the SDK keeps on (and finds on) the local disk only (said once)."""
    own = options.session_store
    if own is not None:
        if in_ram(own):
            _said_local(agent_id, type(own).__name__)
        return None
    # what the SDK keeps on the local disk only: file checkpoints, and the session
    # ``continue_conversation`` continues (the CLI's latest in its folder: given a store,
    # the SDK looks for it there, and a run's journal starts with none)
    local = [
        name
        for name in ("enable_file_checkpointing", "continue_conversation")
        if getattr(options, name)
    ]
    if not local:
        return JournalSessions(journal)
    _said_local(agent_id, local[0])
    return None


@functools.lru_cache(maxsize=SAID_ONCE)  # said once per agent and store, of the most recent
def _said_local(agent_id: str, store: str) -> None:
    log.warning(
        "Claude's session is kept on this machine only, not in the run's journal: a resume on "
        "another worker runs the query again from its prompt (give the options no "
        "session_store, enable_file_checkpointing or continue_conversation to keep it in "
        "the journal)",
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

    async def list_subkeys(self, key: Mapping[str, Any]) -> list[str]:
        """The ``subpath`` of each transcript the session's subagents keep beside its own: a
        resume restores them with it (the SDK asks when the store has this)."""
        under = f"{key['session_id']}/"
        return [
            name.removeprefix(under) for name in self.journal.transcripts if name.startswith(under)
        ]


def _name(key: Mapping[str, Any]) -> str:
    subpath = key.get("subpath")
    return f"{key['session_id']}/{subpath}" if subpath else str(key["session_id"])


__all__ = ["PROMPT_EVENT", "JournalSessions", "session_store", "with_context"]
