"""Offline datasets, assembled from what already happened: the runs, and what people said.

A correction says what the right answer was; a confirmation anchors an answer that was
right; a rejection with no correction still says "not this". **Judge feedback is excluded by
default**: a judge's verdicts becoming the ground truth it is later measured against is a
circle that always closes.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Final, Protocol

from pydantic import BaseModel, ConfigDict, Field

from trellis.contracts import (
    Feedback,
    FeedbackSource,
    FeedbackTargetKind,
    FeedbackVerdict,
    RunRecord,
)

#: Whose judgement may become ground truth.
GROUND_TRUTH_SOURCES: Final = frozenset({FeedbackSource.HUMAN, FeedbackSource.INTERRUPT})


class DatasetItem(BaseModel):
    """One example: what was asked, what should come back, and the evidence it rests on."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    input: Any
    expected_output: Any = None
    #: passages the answer should be grounded in (shown to the judge's rubric)
    evidence: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def negative(self) -> bool:
        """An example of what *not* to answer: rejected, with nobody saying what was right."""
        return bool(self.metadata.get("rejected")) and self.expected_output is None


class Dataset(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    items: list[DatasetItem] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.items)

    def write(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.model_dump(mode="json"), indent=2) + "\n")
        return target

    @classmethod
    def read(cls, path: str | Path) -> Dataset:
        return cls.model_validate(json.loads(Path(path).read_text()))


class RunReader(Protocol):
    """Where runs are read back from (a run store)."""

    async def get(self, run_id: str) -> RunRecord | None: ...


class FeedbackReader(Protocol):
    """Where feedback is read back from (the memory service)."""

    async def list_for(
        self, target_kind: FeedbackTargetKind, target_id: str
    ) -> Sequence[Feedback]: ...


class DatasetBuilder:
    """Turns (run, feedback) pairs into examples. Deterministic and duplicate-free."""

    def __init__(
        self,
        name: str,
        *,
        include_sources: Iterable[FeedbackSource] = GROUND_TRUTH_SOURCES,
        min_score: float | None = None,
    ) -> None:
        self.name = name
        self.include_sources = frozenset(include_sources)
        self.min_score = min_score
        self._items: dict[str, DatasetItem] = {}

    def __len__(self) -> int:
        return len(self._items)

    def add(
        self, record: RunRecord, feedback: Sequence[Feedback] = (), *, evidence: Sequence[str] = ()
    ) -> int:
        """Add what this run and its feedback say. Keyed on the feedback id, so assembling an
        overlapping page twice adds nothing twice. Returns how many examples are new."""
        before = len(self._items)
        for judgement in feedback:
            if judgement.source not in self.include_sources:
                continue
            if (
                self.min_score is not None
                and judgement.score is not None
                and judgement.score < self.min_score
            ):
                continue
            item = _item(record, judgement, list(evidence))
            if item is not None:
                self._items[judgement.feedback_id] = item
        return len(self._items) - before

    def build(self) -> Dataset:
        return Dataset(
            name=self.name,
            items=list(self._items.values()),
            metadata={"sources": sorted(s.value for s in self.include_sources)},
        )

    async def from_store(
        self,
        runs: RunReader,
        feedback: FeedbackReader,
        *,
        run_ids: Sequence[str],
        target_kind: FeedbackTargetKind = FeedbackTargetKind.RUN,
    ) -> Dataset:
        """Assemble from a run store and the feedback store. Missing runs are skipped."""
        for run_id in run_ids:
            record = await runs.get(run_id)
            if record is not None:
                self.add(record, list(await feedback.list_for(target_kind, run_id)))
        return self.build()


def _item(record: RunRecord, feedback: Feedback, evidence: list[str]) -> DatasetItem | None:
    base = {
        "run_id": record.run_id,
        "agent_id": record.agent_id,
        "feedback_id": feedback.feedback_id,
        "verdict": feedback.verdict.value,
        "source": feedback.source.value,
        "reviewer": feedback.reviewer,
    }
    evidence = evidence or [e.citation for e in feedback.evidence_refs if e.citation]
    if feedback.verdict in (FeedbackVerdict.CORRECT, FeedbackVerdict.EDIT):
        return DatasetItem(
            input=record.input,
            expected_output=feedback.correction,
            evidence=evidence,
            metadata=base,
        )
    if feedback.verdict in (FeedbackVerdict.CONFIRM, FeedbackVerdict.APPROVE):
        if record.output is None:
            return None
        return DatasetItem(
            input=record.input, expected_output=record.output, evidence=evidence, metadata=base
        )
    return DatasetItem(
        input=record.input,
        evidence=evidence,
        metadata={**base, "rejected": True, "rejected_output": record.output},
    )
