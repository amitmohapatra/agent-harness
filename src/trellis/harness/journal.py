"""What a paused run needs to continue when its framework cannot keep the state itself.

A LangGraph graph with a checkpointer resumes where it stopped. Every other target is run
again from its input, and the journal makes that re-run land in the same place: a question
already answered returns its answer, and a tool call already made returns its recorded
output instead of running twice. Entries are keyed by *content* (the question, or the tool and
its arguments) and consumed in order, so the n-th identical call gets the n-th recorded
result, and a re-planned call the person never saw is asked about again rather than matched
to someone else's approval.

The journal travels with the run record (``RunRecord.metadata[JOURNAL_KEY]``), so a worker on
another machine resumes from it.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field

from trellis.contracts import Interrupt, InterruptResolution

#: Where the journal lives on a run record's metadata.
JOURNAL_KEY: Final = "trellis_journal"


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
    pending: Pending | None = None

    # ------------------------------------------------------------------ persistence
    @classmethod
    def of(cls, metadata: dict[str, Any] | None) -> Journal:
        raw = (metadata or {}).get(JOURNAL_KEY)
        return cls.model_validate(raw) if isinstance(raw, dict) else cls()

    def dump(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)

    def answered(self, resolution: InterruptResolution) -> None:
        """File the answer to the open pause, and close it."""
        if self.pending is None:
            return
        self.answers.setdefault(self.pending.key, []).append(resolution.model_dump(mode="json"))
        self.pending = None


class Replay:
    """One attempt's cursor over a journal: what the n-th occurrence of a key already got.

    ``orphan`` is an answer whose journal did not travel with the run (a store that keeps
    only the last resolution): it answers the question its interrupt id names and nothing
    else — a question it does not name is asked again, never answered wrongly."""

    __slots__ = ("_seen", "journal", "orphan")

    def __init__(self, journal: Journal, orphan: InterruptResolution | None = None) -> None:
        self.journal = journal
        self.orphan = orphan
        self._seen: dict[str, int] = defaultdict(int)

    def answer(self, key: str) -> InterruptResolution | None:
        """The recorded answer for this occurrence of ``key``, advancing the cursor."""
        index = self._seen[f"a:{key}"]
        recorded = self.journal.answers.get(key, [])
        if index < len(recorded):
            self._seen[f"a:{key}"] = index + 1
            return InterruptResolution.model_validate(recorded[index])
        orphan = self.orphan
        if orphan is not None and key.startswith(orphan.interrupt_id.rsplit(".", 1)[-1]):
            self.orphan = None
            self.record_answer(key, orphan)
            return orphan
        return None

    def call(self, key: str) -> tuple[bool, Any]:
        """``(True, output)`` when this occurrence of the call already ran."""
        index = self._seen[f"c:{key}"]
        recorded = self.journal.calls.get(key, [])
        if index >= len(recorded):
            return False, None
        self._seen[f"c:{key}"] = index + 1
        return True, recorded[index]

    def record_call(self, key: str, output: Any) -> None:
        self.journal.calls.setdefault(key, []).append(output)
        self._seen[f"c:{key}"] += 1

    def record_answer(self, key: str, resolution: InterruptResolution) -> None:
        self.journal.answers.setdefault(key, []).append(resolution.model_dump(mode="json"))
        self._seen[f"a:{key}"] += 1
