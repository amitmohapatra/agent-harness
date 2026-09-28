"""Offline datasets, assembled from what already happened (design §11, §12).

A dataset here is not written by hand. It is what the system already knows: the runs it made
and what people said about them. A correction says what the right answer was; a confirmation
anchors an answer that was right; a rejection with no correction still says "not this", which
is a test too.

One rule is load-bearing: **judge feedback is excluded by default**. A judge's own verdicts
becoming the ground truth it is later measured against is a circle that always closes — the
numbers improve and nothing got better. Human and interrupt decisions are the ground truth;
the judge is what is being measured.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from trellis.contracts.feedback import Feedback, FeedbackSource, FeedbackVerdict
from trellis.contracts.runs import RunRecord

from trellis.harness.runtime.logging import get_logger

log = get_logger("trellis.harness.datasets")

#: Whose judgement may become ground truth.
GROUND_TRUTH_SOURCES: frozenset[FeedbackSource] = frozenset(
    {FeedbackSource.HUMAN, FeedbackSource.INTERRUPT}
)


class DatasetItem(BaseModel):
    """One example: what was asked, what should come back, and where it came from."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    input: Any
    expected_output: Any = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def negative(self) -> bool:
        """An example of what *not* to answer: rejected, with nobody saying what was right."""
        return bool(self.metadata.get("rejected")) and self.expected_output is None


class Dataset(BaseModel):
    """A named set of examples. Portable: Langfuse when configured, JSON when not."""

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

    async def publish(self, provider: Any) -> int:
        """Submit every item to a tracing backend's dataset (Langfuse's, through the
        ``EvaluationProvider`` port). Returns how many landed; a backend that refuses one
        item must not lose the rest."""
        written = 0
        for item in self.items:
            try:
                await provider.submit_dataset_item(self.name, item.model_dump(mode="json"))
                written += 1
            except Exception as exc:
                log.warning("dataset.item_rejected", dataset=self.name, error=str(exc))
        return written


class DatasetBuilder:
    """Turns (run, feedback) pairs into examples. Deterministic and duplicate-free."""

    def __init__(
        self,
        name: str,
        *,
        include_sources: Iterable[FeedbackSource] = GROUND_TRUTH_SOURCES,
        min_score: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.include_sources = frozenset(include_sources)
        #: Ignore feedback that scored below this, when it carries a score at all.
        self.min_score = min_score
        self.metadata = dict(metadata or {})
        self._items: dict[str, DatasetItem] = {}

    def __len__(self) -> int:
        return len(self._items)

    def add(self, record: RunRecord, feedback: Sequence[Feedback] = ()) -> int:
        """Add what this run and its feedback say. Returns the number of new examples.

        Keyed on (run, feedback) so assembling the same page twice adds nothing twice — a
        dataset built from a paginated read must not double-count its overlap.
        """
        before = len(self._items)
        for record_feedback in feedback:
            if not self._admits(record_feedback):
                continue
            item = self._item(record, record_feedback)
            if item is not None:
                self._items[record_feedback.feedback_id] = item
        return len(self._items) - before

    def build(self) -> Dataset:
        return Dataset(
            name=self.name,
            items=list(self._items.values()),
            metadata={**self.metadata, "sources": sorted(s.value for s in self.include_sources)},
        )

    async def from_store(
        self,
        runs: Any,
        feedback: Any,
        *,
        run_ids: Sequence[str],
        target_kind: str = "run",
    ) -> Dataset:
        """Assemble from the ports: a ``RunStore`` for the runs, a ``FeedbackStore`` for what
        people said. Runs the store cannot find are skipped, not invented."""
        from trellis.contracts.feedback import FeedbackTargetKind  # noqa: PLC0415 - one use

        kind = FeedbackTargetKind(target_kind)
        for run_id in run_ids:
            record = await runs.get(run_id)
            if record is None:
                continue
            said = await feedback.list_for(kind, run_id)
            self.add(record, list(said))
        return self.build()

    # ------------------------------------------------------------------ internals
    def _admits(self, feedback: Feedback) -> bool:
        if feedback.source not in self.include_sources:
            return False
        if self.min_score is None or feedback.score is None:
            return True
        return feedback.score >= self.min_score

    def _item(self, record: RunRecord, feedback: Feedback) -> DatasetItem | None:
        """The example one judgement makes, or ``None`` when it makes none."""
        base = {
            "run_id": record.run_id,
            "agent_id": record.agent_id,
            "tenant_id": record.tenant_id,
            "feedback_id": feedback.feedback_id,
            "verdict": feedback.verdict.value,
            "source": feedback.source.value,
            "reviewer": feedback.reviewer,
        }
        if feedback.verdict in (FeedbackVerdict.CORRECT, FeedbackVerdict.EDIT):
            return DatasetItem(
                input=record.input, expected_output=feedback.correction, metadata=base
            )
        if feedback.verdict in (FeedbackVerdict.CONFIRM, FeedbackVerdict.APPROVE):
            if record.output is None:
                return None
            return DatasetItem(input=record.input, expected_output=record.output, metadata=base)
        if feedback.verdict is FeedbackVerdict.REJECT:
            return DatasetItem(
                input=record.input,
                expected_output=None,
                metadata={**base, "rejected": True, "rejected_output": record.output},
            )
        return None  # pragma: no cover - the vocabulary is closed and covered above


__all__ = ["GROUND_TRUTH_SOURCES", "Dataset", "DatasetBuilder", "DatasetItem"]
