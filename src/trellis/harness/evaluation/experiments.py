"""Offline experiments: one dataset, one agent version, one number (design §11).

An experiment is the offline half of the same judge. Where the online judge samples live
traffic, this replays a dataset against a candidate and scores every item with the *same*
``Judge`` port — so "did this version get better?" is asked with the same ruler that answered
"was that answer good?" in production.

Results go to Langfuse when a provider is configured **and** to a local JSON file always. The
local file is not a fallback that nobody tests: it is the artifact a CI gate reads, and a
deployment with no tracing backend still gets its number.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field
from trellis.contracts.evaluation import JudgeVerdict
from trellis.contracts.events import AgentEvalEvent
from trellis.contracts.ids import new_id, now
from trellis.contracts.messages import AgentResponse

from trellis.harness.evaluation.datasets import Dataset, DatasetItem
from trellis.harness.runtime.logging import get_logger

log = get_logger("trellis.harness.experiments")


class ItemResult(BaseModel):
    """What one example produced, and what the judge made of it."""

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
    """One run of one dataset against one agent version."""

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
        """The mean over *judged* items. ``None`` when the judge settled nothing — which is a
        different statement from zero, and must never be reported as one."""
        scored = [i.score for i in self.items if i.score is not None]
        return sum(scored) / len(scored) if scored else None

    @property
    def cost_usd(self) -> float:
        return sum(i.cost_usd or 0.0 for i in self.items)

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
            "cost_usd": round(self.cost_usd, 6),
            "created_at": self.created_at.isoformat(),
        }

    def write(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {**self.model_dump(mode="json"), "summary": self.summary()}
        target.write_text(json.dumps(payload, indent=2) + "\n")
        return target


class ExperimentRunner:
    """Replays a dataset through a candidate and scores it with the injected judge."""

    def __init__(
        self,
        judge: Any,
        *,
        provider: Any = None,
        output_dir: str | Path | None = None,
    ) -> None:
        self.judge = judge
        #: Where scores go when a tracing backend is configured. Optional, always.
        self.provider = provider
        self.output_dir = Path(output_dir) if output_dir else None

    async def run(
        self,
        dataset: Dataset,
        target: Callable[[Any], Any],
        *,
        agent_id: str,
        agent_version: str,
        name: str | None = None,
        tenant_id: str = "experiment",
    ) -> ExperimentResult:
        """Run every item. One item failing is recorded, never fatal: an experiment that
        stopped at the first exception would report the score of a prefix."""
        results: list[ItemResult] = []
        for index, item in enumerate(dataset.items):
            results.append(
                await self._one(item, target, index=index, agent_id=agent_id, tenant_id=tenant_id)
            )
        result = ExperimentResult(
            name=name or f"{dataset.name}@{agent_version}",
            dataset=dataset.name,
            agent_id=agent_id,
            agent_version=agent_version,
            items=results,
        )
        if self.output_dir is not None:
            result.write(self.output_dir / f"{_slug(result.name)}.json")
        await self._publish(result)
        return result

    # ------------------------------------------------------------------ internals
    async def _one(
        self,
        item: DatasetItem,
        target: Callable[[Any], Any],
        *,
        index: int,
        agent_id: str,
        tenant_id: str,
    ) -> ItemResult:
        started = time.perf_counter()
        try:
            produced = await _call(target, item.input)
        except Exception as exc:
            return ItemResult(
                input=item.input,
                expected_output=item.expected_output,
                error=f"{type(exc).__name__}: {exc}",
                latency_ms=round((time.perf_counter() - started) * 1000, 3),
            )
        latency = round((time.perf_counter() - started) * 1000, 3)
        response = _as_response(produced)
        verdict = await self._score(
            item, response, index=index, agent_id=agent_id, tenant=tenant_id
        )
        return ItemResult(
            input=item.input,
            output=response.data,
            expected_output=item.expected_output,
            score=None if verdict is None else float(verdict.score),
            method=None if verdict is None else verdict.method.value,
            label=None if verdict is None else verdict.label,
            rationale=None if verdict is None else verdict.rationale,
            cost_usd=None if verdict is None else verdict.cost_usd,
            latency_ms=latency,
        )

    async def _score(
        self,
        item: DatasetItem,
        response: AgentResponse,
        *,
        index: int,
        agent_id: str,
        tenant: str,
    ) -> JudgeVerdict | None:
        event = AgentEvalEvent(
            agent_id=agent_id,
            agent_run_id=f"exp_{index}_{new_id('')}",
            tenant_id=tenant,
            status=response.status,
            metadata={
                "experiment": True,
                "expected_output": item.expected_output,
                **dict(item.metadata),
            },
        )
        judge: Any = self.judge
        bind = getattr(judge, "bound", None)
        if callable(bind):
            # Offline there is no memory scope to verify against; the judge is told so
            # explicitly rather than silently reusing whatever a previous run bound.
            judge = bind(question=_question(item))
        try:
            return await judge.judge(event, response=response)
        except Exception as exc:  # pragma: no cover - a judge that raises is a missing score
            log.warning("experiment.judge_failed", error=str(exc))
            return None

    async def _publish(self, result: ExperimentResult) -> None:
        if self.provider is None:
            return
        mean = result.mean_score
        if mean is None:
            return
        try:
            await self.provider.score(
                "experiment_mean_score",
                mean,
                data_type="NUMERIC",
                comment=f"{result.judged}/{len(result.items)} judged",
                agent_id=result.agent_id,
                agent_version=result.agent_version,
                dataset=result.dataset,
                experiment=result.name,
            )
        except Exception as exc:  # pragma: no cover
            log.warning("experiment.publish_failed", error=str(exc))


async def _call(target: Callable[[Any], Any], payload: Any) -> Any:
    produced = target(payload)
    if hasattr(produced, "__await__"):
        return await produced
    return produced


def _as_response(produced: Any) -> AgentResponse:
    return produced if isinstance(produced, AgentResponse) else AgentResponse.ok(produced)


def _question(item: DatasetItem) -> str | None:
    return item.input if isinstance(item.input, str) else None


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-._" else "-" for c in text).strip("-") or "experiment"


__all__ = ["ExperimentResult", "ExperimentRunner", "ItemResult"]
