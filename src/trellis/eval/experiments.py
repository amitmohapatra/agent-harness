"""Offline experiments: one dataset, one agent version, one number.

The same judge that samples production scores every item, with the item's reference answer
and evidence in front of it — so "did this version get better?" is asked with the ruler that
answered "was that answer good?". Results are written to a JSON file, which is what the
regression gate reads.
"""

from __future__ import annotations

import json
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from trellis.contracts import AgentEvalEvent, JudgeVerdict, new_id, now
from trellis.eval.datasets import Dataset, DatasetItem
from trellis.eval.judge import GroundedJudge

Target = Callable[[Any], Awaitable[Any]]


class ItemResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    input: Any
    output: Any = None
    expected_output: Any = None
    score: float | None = None
    method: str | None = None
    label: str | None = None
    rationale: str | None = None
    cost_usd: float | None = None
    error: str | None = None
    latency_ms: float = 0.0


class ExperimentResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    dataset: str
    agent_id: str
    agent_version: str
    items: list[ItemResult] = Field(default_factory=list)
    created_at: AwareDatetime = Field(default_factory=now)

    @property
    def judged(self) -> int:
        return sum(1 for i in self.items if i.score is not None)

    @property
    def abstained(self) -> int:
        return sum(1 for i in self.items if i.score is None and i.error is None)

    @property
    def failed(self) -> int:
        return sum(1 for i in self.items if i.error is not None)

    @property
    def mean_score(self) -> float | None:
        """The mean over judged items; ``None`` (not zero) when nothing was judged."""
        scored = [i.score for i in self.items if i.score is not None]
        return sum(scored) / len(scored) if scored else None

    def summary(self) -> dict[str, Any]:
        """The shape the regression gate reads."""
        return {
            "name": self.name,
            "dataset": self.dataset,
            "agent_id": self.agent_id,
            "agent_version": self.agent_version,
            "items": len(self.items),
            "judged": self.judged,
            "abstained": self.abstained,
            "failed": self.failed,
            "mean_score": self.mean_score,
            "cost_usd": round(sum(i.cost_usd or 0.0 for i in self.items), 6),
            "created_at": self.created_at.isoformat(),
        }

    def write(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {**self.model_dump(mode="json"), "summary": self.summary()}
        target.write_text(json.dumps(payload, indent=2) + "\n")
        return target


class ExperimentRunner:
    """Replays a dataset through a candidate (``async (input) -> answer``) and judges it.

    An agent is a candidate as ``lambda x: agent.run(x, user="eval")``: a returned ``Result``
    is read for its answer."""

    def __init__(self, judge: GroundedJudge, *, output_dir: str | Path | None = None) -> None:
        self.judge = judge
        self.output_dir = Path(output_dir) if output_dir else None

    async def run(
        self,
        dataset: Dataset,
        target: Target,
        *,
        agent_id: str,
        agent_version: str,
        name: str | None = None,
    ) -> ExperimentResult:
        """Every item, in order. A failing item is recorded, never fatal."""
        result = ExperimentResult(
            name=name or f"{dataset.name}@{agent_version}",
            dataset=dataset.name,
            agent_id=agent_id,
            agent_version=agent_version,
            items=[await self._one(item, target, agent_id) for item in dataset.items],
        )
        if self.output_dir is not None:
            result.write(self.output_dir / f"{_slug(result.name)}.json")
        return result

    async def _one(self, item: DatasetItem, target: Target, agent_id: str) -> ItemResult:
        started = time.perf_counter()
        try:
            produced = _answer(await target(item.input))
        except Exception as exc:
            return ItemResult(
                input=item.input,
                expected_output=item.expected_output,
                error=f"{type(exc).__name__}: {exc}",
                latency_ms=_ms(started),
            )
        latency = _ms(started)
        verdict = await self._score(item, produced, agent_id)
        return ItemResult(
            input=item.input,
            output=produced,
            expected_output=item.expected_output,
            score=None if verdict is None else float(verdict.score),
            method=None if verdict is None else verdict.method.value,
            label=None if verdict is None else verdict.label,
            rationale=None if verdict is None else verdict.rationale,
            cost_usd=None if verdict is None else verdict.cost_usd,
            latency_ms=latency,
        )

    async def _score(self, item: DatasetItem, produced: Any, agent_id: str) -> JudgeVerdict | None:
        answer = produced if isinstance(produced, str) else json.dumps(produced, default=str)
        event = AgentEvalEvent(
            agent_id=agent_id, agent_run_id=new_id("exp_"), tenant_id="experiment"
        )
        return await self.judge.verdict(
            event,
            question=item.input if isinstance(item.input, str) else json.dumps(item.input),
            answer=answer,
            evidence="\n".join(item.evidence),
            expected=item.expected_output,
        )


def _answer(produced: Any) -> Any:
    """A harness ``Result`` is read for its answer; anything else is the answer."""
    from trellis.harness.result import Result  # noqa: PLC0415 - eval works without a harness

    return produced.answer if isinstance(produced, Result) else produced


def _ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-._" else "-" for c in text).strip("-") or "experiment"
