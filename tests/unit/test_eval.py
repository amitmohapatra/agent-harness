from __future__ import annotations

import itertools
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tests.support.models import ScriptedChat
from trellis.contracts import (
    AgentEvalEvent,
    AgentResponse,
    FeedbackSource,
    FeedbackTargetKind,
    FeedbackVerdict,
    Judge,
    JudgeMethod,
    RunRecord,
    RunStart,
    RunStatus,
)
from trellis.eval import (
    Dataset,
    DatasetBuilder,
    DatasetItem,
    ExperimentRunner,
    GateThresholds,
    GroundedJudge,
    JudgeBudget,
    RegressionGate,
)
from trellis.eval.budget import sampled
from trellis.eval.gate import main as gate_main
from trellis.harness.result import Result
from trellis.memory.models import Feedback
from trellis.memory.models import GroundingReport as Report


def event(run_id: str = "run_1", **metadata: object) -> AgentEvalEvent:
    return AgentEvalEvent(agent_id="a", agent_run_id=run_id, tenant_id="t", metadata=metadata)


class Verifier:
    def __init__(self, report: Report | None) -> None:
        self.report = report
        self.calls = 0

    async def verify(self, answer: str, bundle: object) -> Report | None:
        self.calls += 1
        return self.report


def verdict_reply(score: float) -> str:
    return json.dumps({"score": score, "label": "ok", "rationale": "because"})


# --------------------------------------------------------------------------- budget
def test_sampling_is_deterministic_in_the_run_id() -> None:
    assert sampled("run_x", 1.0) and not sampled("run_x", 0.0)
    assert sampled("run_x", 0.5) == sampled("run_x", 0.5)
    share = sum(sampled(f"run_{i}", 0.3) for i in range(2000)) / 2000
    assert 0.25 < share < 0.35


def test_the_hourly_count_holds_and_rolls_over() -> None:
    clock = [0.0]
    budget = JudgeBudget(1.0, max_per_hour=2, clock=lambda: clock[0])
    assert budget.reserve("a", "r1") and budget.reserve("a", "r2")
    assert budget.reserve("a", "r3").reason == "over_max_per_hour"
    assert budget.reserve("b", "r1")  # per agent
    clock[0] = 3601
    assert budget.reserve("a", "r4")


def test_spend_stops_the_judge_and_never_refunds() -> None:
    budget = JudgeBudget(1.0, max_usd_per_hour=0.5)
    budget.spend("a", 0.6)
    budget.spend("a", -10)
    assert budget.admit("a", "r").reason == "over_budget"


# --------------------------------------------------------------------------- judge
async def test_it_implements_the_contracts_judge() -> None:
    assert isinstance(GroundedJudge(budget=JudgeBudget(1.0)), Judge)


async def test_a_grounded_report_decides_without_a_model() -> None:
    model = ScriptedChat([])
    judge = GroundedJudge(budget=JudgeBudget(1.0), model=model)
    verdict = await judge.verdict(
        event(), question="q", answer="a", verifier=Verifier(Report(supported=3)), bundle=object()
    )
    assert verdict is not None and verdict.score == 1.0 and verdict.method is JudgeMethod.GROUNDED
    contradicted = await judge.verdict(
        event("run_2"),
        question="q",
        answer="a",
        verifier=Verifier(Report(supported=1, contradicted=1)),
        bundle=object(),
    )
    assert contradicted is not None and contradicted.score == 0.5
    assert model.requests == []


async def test_what_grounding_cannot_settle_goes_to_the_rubric_with_the_evidence() -> None:
    model = ScriptedChat([verdict_reply(0.7)])
    judge = GroundedJudge(budget=JudgeBudget(1.0), model=model)
    verdict = await judge.verdict(
        event(),
        question="how many?",
        answer="12",
        verifier=Verifier(Report(supported=1, unsupported=1)),
        bundle=object(),
        evidence="stock is 12",
    )
    assert verdict is not None and verdict.score == 0.7 and verdict.method is JudgeMethod.LLM
    prompt = model.requests[0]["messages"][0]["content"]
    assert "stock is 12" in prompt and "how many?" in prompt
    assert model.requests[0]["response_format"]["type"] == "json_schema"


async def test_no_second_opinion_when_the_service_already_consulted_its_judge() -> None:
    model = ScriptedChat([verdict_reply(0.1)])
    judge = GroundedJudge(budget=JudgeBudget(1.0), model=model)
    report = Report(supported=1, borderline=1, judge_consulted=1)
    assert (
        await judge.verdict(
            event(), question="q", answer="a", verifier=Verifier(report), bundle=object()
        )
        is None
    )
    assert model.requests == []


async def test_the_judge_abstains_when_unsampled_empty_or_broken() -> None:
    judge = GroundedJudge(budget=JudgeBudget(0.0), model=ScriptedChat([verdict_reply(1)]))
    assert await judge.verdict(event(), question="q", answer="a") is None
    broken = GroundedJudge(budget=JudgeBudget(1.0), model=ScriptedChat([]))  # pops from empty
    assert await broken.verdict(event(), question="q", answer="a") is None
    assert await broken.verdict(event(), question="q", answer="  ") is None
    garbled = GroundedJudge(budget=JudgeBudget(1.0), model=ScriptedChat(["not json"]))
    assert await garbled.verdict(event(), question="q", answer="a") is None


async def test_the_port_scores_a_response_against_its_reference() -> None:
    judge = GroundedJudge(budget=JudgeBudget(1.0))
    verdict = await judge.judge(
        event(expected_output="Paris"), response=AgentResponse.ok(" paris ")
    )
    assert verdict is not None and verdict.label == "matches_reference"
    assert await judge.judge(event(), response=AgentResponse.ok({"not": "text"})) is None


# --------------------------------------------------------------------------- datasets
def record(output: object = "12") -> RunRecord:
    start = RunStart(run_id="run_1", tenant_id="t", agent_id="a", input="how many?")
    return RunRecord.model_validate(
        {**RunRecord.from_start(start).model_dump(), "status": RunStatus.SUCCESS, "output": output}
    )


_feedback_ids = itertools.count()


def feedback(
    verdict: FeedbackVerdict, source: FeedbackSource = FeedbackSource.HUMAN, **fields: object
) -> Feedback:
    """A record as the memory service returns it: verdict and source are plain strings."""
    return Feedback.model_validate(
        {
            "feedback_id": f"fb_{next(_feedback_ids)}",
            "tenant_id": "t",
            "target_kind": "run",
            "target_id": "run_1",
            "verdict": verdict.value,
            "source": source.value,
            "created_at": datetime.now(UTC),
            **fields,
        }
    )


def test_feedback_becomes_examples_and_the_judge_is_not_ground_truth() -> None:
    builder = DatasetBuilder("stock")
    added = builder.add(
        record(),
        [
            feedback(FeedbackVerdict.CORRECT, correction="13"),
            feedback(FeedbackVerdict.CONFIRM),
            feedback(FeedbackVerdict.REJECT),
            feedback(FeedbackVerdict.CONFIRM, source=FeedbackSource.JUDGE),
        ],
        evidence=["stock is 13"],
    )
    assert added == 3
    items = builder.build().items
    assert [i.expected_output for i in items] == ["13", "12", None]
    assert items[0].evidence == ["stock is 13"]
    assert items[2].negative


def test_datasets_round_trip_through_json(tmp_path: Path) -> None:
    dataset = Dataset(name="d", items=[DatasetItem(input="q", expected_output="a", evidence=["e"])])
    assert Dataset.read(dataset.write(tmp_path / "d.json")) == dataset


class Store:
    async def get(self, run_id: str) -> RunRecord | None:
        return record() if run_id == "run_1" else None


class FeedbackReader:
    async def list_for(self, target_kind: FeedbackTargetKind, target_id: str) -> list[Feedback]:
        return [feedback(FeedbackVerdict.CORRECT, correction="13")]


async def test_a_dataset_is_assembled_from_the_run_store_and_the_feedback_store() -> None:
    dataset = await DatasetBuilder("d").from_store(
        Store(), FeedbackReader(), run_ids=["run_1", "missing"]
    )
    assert [i.expected_output for i in dataset.items] == ["13"]


# --------------------------------------------------------------------------- experiments
async def test_an_experiment_scores_with_the_reference_and_the_evidence(tmp_path: Path) -> None:
    model = ScriptedChat([verdict_reply(0.4)])
    runner = ExperimentRunner(
        GroundedJudge(budget=JudgeBudget(1.0), model=model), output_dir=tmp_path
    )
    dataset = Dataset(
        name="stock",
        items=[
            DatasetItem(input="capital of France?", expected_output="Paris"),
            DatasetItem(input="stock of a?", expected_output="13", evidence=["a: 13 units"]),
            DatasetItem(input="explode"),
        ],
    )

    async def candidate(question: object) -> object:
        if question == "explode":
            raise RuntimeError("boom")
        if question == "stock of a?":
            return Result(run_id="r", status=RunStatus.SUCCESS, answer="12")
        return "Paris"

    result = await runner.run(dataset, candidate, agent_id="a", agent_version="v2")
    assert [i.score for i in result.items] == [1.0, 0.4, None]
    assert result.failed == 1 and result.judged == 2
    prompt = model.requests[0]["messages"][0]["content"]
    assert "a: 13 units" in prompt and "13" in prompt
    written = json.loads((tmp_path / "stock-v2.json").read_text())
    assert written["summary"]["mean_score"] == pytest.approx(0.7)


# --------------------------------------------------------------------------- gate
def test_the_gate_fails_on_a_latency_regression_or_a_score_drop() -> None:
    gate = RegressionGate(GateThresholds(max_latency_regression_pct=10, max_score_drop=0.05))
    report = gate.evaluate(
        baseline={"results": {"run": {"p50": 10.0}}},
        current={"results": {"run": {"p50": 12.0}}},
        baseline_judge={"summary": {"mean_score": 0.9}},
        current_judge={"summary": {"mean_score": 0.8}},
    )
    assert not report.passed
    assert {f.metric for f in report.failures} == {"latency.run.p50", "judge.mean_score"}


def test_a_missing_baseline_fails_unless_allowed() -> None:
    gate = RegressionGate()
    assert not gate.evaluate(baseline=None, current={"results": {}}).passed
    assert gate.evaluate(baseline=None, current={"results": {}}, allow_missing_baseline=True).passed


def test_the_cli_reads_files_and_exits_nonzero_on_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    base, cur = tmp_path / "b.json", tmp_path / "c.json"
    base.write_text(json.dumps({"results": {"run": {"p50": 10.0}}}))
    cur.write_text(json.dumps({"results": {"run": {"p50": 10.5}}}))
    assert gate_main(["--baseline", str(base), "--current", str(cur)]) == 0
    cur.write_text(json.dumps({"results": {"run": {"p50": 50.0}}}))
    assert gate_main(["--baseline", str(base), "--current", str(cur)]) == 1
    assert "FAILED" in capsys.readouterr().out
