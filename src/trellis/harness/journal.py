"""What a paused run needs to continue when its framework cannot keep the state itself.

A LangGraph graph with a checkpointer resumes where it stopped. Every other target is run
again from its input, and the journal makes that re-run land in the same place: a question
already answered returns its answer, and a tool call already made returns its recorded
output instead of running twice. Entries are keyed by *content* (the question, or the tool and
its arguments) and consumed in order, so the n-th identical call gets the n-th recorded
result, and a re-planned call the person never saw is asked about again rather than matched
to someone else's approval.

The journal is the run's checkpoint (``RunRecord.checkpoint``): the run store keeps it with the
pause and hands it to whichever worker resumes the run, so a resume on another machine repeats
no question and no side effect.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from trellis.contracts import Interrupt, InterruptResolution


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
    #: the tools the run has called (they stay offered to the model after a pause)
    used: list[str] = Field(default_factory=list)
    pending: Pending | None = None

    # ------------------------------------------------------------------ persistence
    @classmethod
    def of(cls, checkpoint: dict[str, Any] | None) -> Journal:
        """The journal a run record's checkpoint holds (an empty one for a fresh run)."""
        return cls() if checkpoint is None else cls.model_validate(checkpoint)

    def dump(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)

    def answered(self, resolution: InterruptResolution) -> None:
        """File the answer to the open pause, and close it."""
        if self.pending is None:
            return
        self.answers.setdefault(self.pending.key, []).append(resolution.model_dump(mode="json"))
        self.pending = None


class Replay:
    """One attempt's cursor over a journal: what the n-th occurrence of a key already got."""

    __slots__ = ("_seen", "journal")

    def __init__(self, journal: Journal) -> None:
        self.journal = journal
        self._seen: dict[str, int] = defaultdict(int)

    def answer(self, key: str) -> InterruptResolution | None:
        """The recorded answer for this occurrence of ``key``, advancing the cursor."""
        index = self._seen[f"a:{key}"]
        recorded = self.journal.answers.get(key, [])
        if index < len(recorded):
            self._seen[f"a:{key}"] = index + 1
            return InterruptResolution.model_validate(recorded[index])
        return None

    def call(self, key: str) -> tuple[bool, Any]:
        """``(True, output)`` when this occurrence of the call already ran."""
        index = self._seen[f"c:{key}"]
        recorded = self.journal.calls.get(key, [])
        if index >= len(recorded):
            return False, None
        self._seen[f"c:{key}"] = index + 1
        return True, recorded[index]

    def record_call(self, key: str, output: Any, *, tool: str) -> None:
        self.journal.calls.setdefault(key, []).append(output)
        self._seen[f"c:{key}"] += 1
        if tool not in self.journal.used:
            self.journal.used.append(tool)

    def record_answer(self, key: str, resolution: InterruptResolution) -> None:
        self.journal.answers.setdefault(key, []).append(resolution.model_dump(mode="json"))
        self._seen[f"a:{key}"] += 1
