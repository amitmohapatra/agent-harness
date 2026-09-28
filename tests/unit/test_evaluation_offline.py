"""The offline half: datasets out of what happened, experiments, and the gate in CI.

Nothing here spends anything — the judge is scripted, the "agent" is a function — which is the
same claim the online tests make from the other side: the decision about whether to spend is
separable from the spending.
"""

from __future__ import annotations

import json

import pytest
from trellis.contracts.evaluation import JudgeMethod, JudgeVerdict
from trellis.contracts.feedback import (
    Feedback,
    FeedbackSource,
    FeedbackTargetKind,
    FeedbackVerdict,
)
from trellis.contracts.messages import AgentResponse
from trellis.contracts.runs import RunRecord, RunStatus

from trellis.harness.evaluation.datasets import (
    GROUND_TRUTH_SOURCES,
    Dataset,
    DatasetBuilder,
    DatasetItem,
)
from trellis.harness.evaluation.experiments import ExperimentRunner
from trellis.harness.evaluation.gate import (
    GateThresholds,
    RegressionGate,
    main,
)

TENANT = "acme"


def run(run_id: str = "run_1", *, output: str | None = "issued on Tuesday") -> RunRecord:
    return RunRecord(
        run_id=run_id,
        tenant_id=TENANT,
        agent_id="refunds",
        status=RunStatus.SUCCESS,
        input="when was the refund issued?",
        output=output,
    )


def said(
    verdict: FeedbackVerdict,
    *,
    source: FeedbackSource = FeedbackSource.HUMAN,
    correction: object = None,
    score: float | None = None,
    run_id: str = "run_1",
) -> Feedback:
    return Feedback(
        tenant_id=TENANT,
        agent_id="refunds",
        agent_run_id=run_id,
        target_kind=FeedbackTargetKind.ANSWER,
        target_id=run_id,
        verdict=verdict,
        source=source,
        correction=correction,
        score=score,
        reviewer="alex",
    )


# --------------------------------------------------------------------------- datasets


def test_a_correction_becomes_the_expected_answer() -> None:
    builder = DatasetBuilder("refunds-goldens")
    added = builder.add(run(), [said(FeedbackVerdict.CORRECT, correction="issued on Monday")])
    assert added == 1
    item = builder.build().items[0]
    assert item.input == "when was the refund issued?"
    assert item.expected_output == "issued on Monday"
    assert item.metadata["verdict"] == "correct" and item.metadata["reviewer"] == "alex"


def test_a_confirmation_anchors_the_answer_the_run_gave() -> None:
    builder = DatasetBuilder("refunds-goldens")
    builder.add(run(), [said(FeedbackVerdict.CONFIRM)])
    assert builder.build().items[0].expected_output == "issued on Tuesday"


def test_a_confirmation_of_a_run_with_no_output_anchors_nothing() -> None:
    builder = DatasetBuilder("refunds-goldens")
    assert builder.add(run(output=None), [said(FeedbackVerdict.CONFIRM)]) == 0


def test_a_rejection_with_no_correction_is_still_a_test() -> None:
    builder = DatasetBuilder("refunds-goldens")
    builder.add(run(), [said(FeedbackVerdict.REJECT)])
    item = builder.build().items[0]
    assert item.expected_output is None
    assert item.negative, '"do not answer this again" is an example too'
    assert item.metadata["rejected_output"] == "issued on Tuesday"


def test_an_edit_carries_its_edited_arguments() -> None:
    builder = DatasetBuilder("tool-goldens")
    builder.add(run(), [said(FeedbackVerdict.EDIT, correction={"order": "A-1029"})])
    assert builder.build().items[0].expected_output == {"order": "A-1029"}


def test_judge_feedback_is_excluded_by_default() -> None:
    """A judge's own verdicts becoming the ground truth it is later measured against is a
    circle that always closes: the numbers improve and nothing got better."""
    assert FeedbackSource.JUDGE not in GROUND_TRUTH_SOURCES
    builder = DatasetBuilder("refunds-goldens")
    assert builder.add(run(), [said(FeedbackVerdict.CONFIRM, source=FeedbackSource.JUDGE)]) == 0

    deliberate = DatasetBuilder("audit", include_sources=[FeedbackSource.JUDGE])
    assert deliberate.add(run(), [said(FeedbackVerdict.CONFIRM, source=FeedbackSource.JUDGE)]) == 1


def test_an_interrupt_decision_is_ground_truth() -> None:
    builder = DatasetBuilder("approvals")
    added = builder.add(run(), [said(FeedbackVerdict.APPROVE, source=FeedbackSource.INTERRUPT)])
    assert added == 1


def test_low_scored_feedback_can_be_filtered_out() -> None:
    builder = DatasetBuilder("refunds-goldens", min_score=0.8)
    assert builder.add(run(), [said(FeedbackVerdict.CONFIRM, score=0.4)]) == 0
    assert builder.add(run(), [said(FeedbackVerdict.CONFIRM, score=0.9)]) == 1
    assert builder.add(run(), [said(FeedbackVerdict.CONFIRM)]) == 1, "no score is not a low score"


def test_the_same_page_read_twice_adds_nothing_twice() -> None:
    builder = DatasetBuilder("refunds-goldens")
    one = said(FeedbackVerdict.CORRECT, correction="Monday")
    assert builder.add(run(), [one]) == 1
    assert builder.add(run(), [one]) == 0, "a paginated read must not double-count its overlap"
    assert len(builder) == 1


def test_a_dataset_round_trips_through_json(tmp_path) -> None:
    builder = DatasetBuilder("refunds-goldens", metadata={"built_by": "test"})
    builder.add(run(), [said(FeedbackVerdict.CORRECT, correction="Monday")])
    builder.add(run("run_2"), [said(FeedbackVerdict.REJECT, run_id="run_2")])
    dataset = builder.build()
    assert dataset.metadata["sources"] == ["human", "interrupt"]

    path = dataset.write(tmp_path / "nested" / "goldens.json")
    assert Dataset.read(path) == dataset
    assert json.loads(path.read_text())["name"] == "refunds-goldens"


async def test_a_dataset_is_assembled_from_the_ports() -> None:
    class Runs:
        async def get(self, run_id: str):
            return run(run_id) if run_id != "run_missing" else None

    class Store:
        def __init__(self) -> None:
            self.asked: list[tuple[str, str]] = []

        async def list_for(self, kind, target_id, *, limit: int = 100):
            self.asked.append((kind.value, target_id))
            return [said(FeedbackVerdict.CORRECT, correction="Monday", run_id=target_id)]

    store = Store()
    dataset = await DatasetBuilder("from-store").from_store(
        Runs(), store, run_ids=["run_1", "run_missing", "run_2"]
    )
    assert len(dataset) == 2, "a run the store cannot find is skipped, not invented"
    assert store.asked == [("run", "run_1"), ("run", "run_2")]

    answers = Store()
    await DatasetBuilder("from-store").from_store(
        Runs(), answers, run_ids=["run_1"], target_kind="answer"
    )
    assert answers.asked == [("answer", "run_1")], "which id space was judged is the caller's"


async def test_publishing_keeps_going_when_a_backend_refuses_one_item() -> None:
    class Provider:
        def __init__(self) -> None:
            self.items: list[dict] = []

        async def submit_dataset_item(self, dataset: str, item) -> None:
            if item["input"] == "bad":
                raise RuntimeError("rejected")
            self.items.append(dict(item))

    provider = Provider()
    dataset = Dataset(
        name="d",
        items=[DatasetItem(input="good"), DatasetItem(input="bad"), DatasetItem(input="also")],
    )
    assert await dataset.publish(provider) == 2
    assert [i["input"] for i in provider.items] == ["good", "also"]


# --------------------------------------------------------------------------- experiments


class ScriptedJudge:
    """Scores by whether the output matches the expectation. Costs nothing, ever."""

    def __init__(self, *, abstain_on: str | None = None) -> None:
        self.abstain_on = abstain_on
        self.events: list = []

    def bound(self, **kwargs):
        self.bound_with = kwargs
        return self

    async def judge(self, event, /, *, response=None):
        self.events.append(event)
        answer = None if response is None else response.data
        if answer == self.abstain_on:
            return None
        expected = event.metadata.get("expected_output")
        return JudgeVerdict(
            score=1.0 if answer == expected else 0.0,
            method=JudgeMethod.GROUNDED,
            label="match" if answer == expected else "mismatch",
            cost_usd=0.0,
        )


def two_items() -> Dataset:
    return Dataset(
        name="refunds-goldens",
        items=[
            DatasetItem(input="q1", expected_output="a1"),
            DatasetItem(input="q2", expected_output="a2"),
        ],
    )


async def test_an_experiment_scores_every_item_and_writes_its_artifact(tmp_path) -> None:
    judge = ScriptedJudge()
    runner = ExperimentRunner(judge, output_dir=tmp_path)
    result = await runner.run(
        two_items(),
        lambda q: "a1" if q == "q1" else "wrong",
        agent_id="refunds",
        agent_version="0.4.0",
    )
    assert result.name == "refunds-goldens@0.4.0"
    assert result.judged == 2 and result.abstained == 0 and result.failed == 0
    assert result.mean_score == pytest.approx(0.5)
    assert result.cost_usd == 0.0

    written = json.loads((tmp_path / "refunds-goldens-0.4.0.json").read_text())
    assert written["summary"]["mean_score"] == pytest.approx(0.5)
    assert written["summary"]["judged"] == 2
    assert [i["label"] for i in written["items"]] == ["match", "mismatch"]
    assert judge.bound_with == {"question": "q2"}, "offline there is no memory scope to verify"


async def test_an_async_target_and_a_full_response_both_work(tmp_path) -> None:
    async def target(question: str) -> AgentResponse:
        return AgentResponse.ok("a1" if question == "q1" else "a2")

    result = await ExperimentRunner(ScriptedJudge(), output_dir=tmp_path).run(
        two_items(), target, agent_id="refunds", agent_version="0.4.0"
    )
    assert result.mean_score == 1.0


async def test_one_item_failing_does_not_report_the_score_of_a_prefix(tmp_path) -> None:
    def target(question: str) -> str:
        if question == "q1":
            raise RuntimeError("the agent fell over")
        return "a2"

    result = await ExperimentRunner(ScriptedJudge(), output_dir=tmp_path).run(
        two_items(), target, agent_id="refunds", agent_version="0.4.0"
    )
    assert result.failed == 1 and result.judged == 1
    assert result.mean_score == 1.0, "the mean is over what was judged"
    assert "RuntimeError" in (result.items[0].error or "")


async def test_a_judge_that_settles_nothing_reports_no_score_rather_than_zero(tmp_path) -> None:
    judge = ScriptedJudge(abstain_on="a1")
    result = await ExperimentRunner(judge, output_dir=tmp_path).run(
        Dataset(name="one", items=[DatasetItem(input="q1", expected_output="a1")]),
        lambda q: "a1",
        agent_id="refunds",
        agent_version="0.4.0",
    )
    assert result.judged == 0 and result.abstained == 1
    assert result.mean_score is None, "no score is a different statement from zero"
    assert result.summary()["mean_score"] is None


async def test_a_local_artifact_is_written_with_no_langfuse_at_all(tmp_path) -> None:
    result = await ExperimentRunner(ScriptedJudge(), output_dir=tmp_path).run(
        two_items(), lambda q: "a1", agent_id="refunds", agent_version="0.4.0", name="local only"
    )
    assert (tmp_path / "local-only.json").exists()
    assert result.summary()["items"] == 2


async def test_a_provider_gets_the_experiment_mean_when_one_is_configured(tmp_path) -> None:
    class Provider:
        def __init__(self) -> None:
            self.scores: list[tuple[str, float, dict]] = []

        async def score(self, name, value, /, **metadata):
            self.scores.append((name, value, metadata))

    provider = Provider()
    await ExperimentRunner(ScriptedJudge(), provider=provider, output_dir=tmp_path).run(
        two_items(), lambda q: "a1", agent_id="refunds", agent_version="0.4.0"
    )
    name, value, metadata = provider.scores[0]
    assert name == "experiment_mean_score" and value == pytest.approx(0.5)
    assert metadata["agent_version"] == "0.4.0" and metadata["dataset"] == "refunds-goldens"


# --------------------------------------------------------------------------- the gate


def benchmark(p50: float, p95: float = 1.0) -> dict:
    return {
        "iterations": 100,
        "results": {
            "telemetry_enabled": {"p50": p50, "p90": p50, "p95": p95, "p99": p95, "mean": p50},
            "context_creation": {"p50": 0.02, "mean": 0.02},
        },
    }


def experiment(mean: float | None) -> dict:
    return {"summary": {"name": "e", "mean_score": mean, "judged": 20}}


def test_equal_artifacts_pass() -> None:
    report = RegressionGate().evaluate(baseline=benchmark(10.0), current=benchmark(10.0))
    assert report.passed and report.exit_code == 0
    assert "latency.telemetry_enabled.p50" in report.render()


def test_a_latency_regression_beyond_the_delta_fails() -> None:
    gate = RegressionGate(GateThresholds(max_latency_regression_pct=20.0))
    assert gate.evaluate(baseline=benchmark(10.0), current=benchmark(11.9)).passed
    failed = gate.evaluate(baseline=benchmark(10.0), current=benchmark(13.0))
    assert not failed.passed
    assert [f.metric for f in failed.failures] == [
        "latency.telemetry_enabled.p50",
        "latency.telemetry_enabled.p90",
        "latency.telemetry_enabled.mean",
    ]


def test_sub_millisecond_numbers_are_reported_but_not_gated() -> None:
    """A percent change on 0.02 ms is noise, and a gate that fails on noise gets turned off."""
    report = RegressionGate().evaluate(baseline=benchmark(10.0), current=benchmark(10.0))
    floor = next(f for f in report.findings if f.metric == "latency.context_creation.p50")
    assert floor.passed and floor.note is not None and "floor" in floor.note


def test_a_judge_score_drop_beyond_the_delta_fails() -> None:
    gate = RegressionGate(GateThresholds(max_score_drop=0.05))
    assert gate.evaluate(
        baseline=None, current=None, baseline_judge=experiment(0.9), current_judge=experiment(0.86)
    ).passed
    dropped = gate.evaluate(
        baseline=None, current=None, baseline_judge=experiment(0.9), current_judge=experiment(0.7)
    )
    assert not dropped.passed
    assert dropped.failures[0].metric == "judge.mean_score"


def test_a_floor_the_score_must_clear_whatever_the_baseline_was() -> None:
    gate = RegressionGate(GateThresholds(min_score=0.8))
    report = gate.evaluate(
        baseline=None, current=None, baseline_judge=experiment(0.5), current_judge=experiment(0.7)
    )
    assert not report.passed
    assert "judge.min_score" in [f.metric for f in report.failures]


def test_a_missing_baseline_fails_unless_it_is_allowed() -> None:
    """Silence is not a pass."""
    gate = RegressionGate()
    assert not gate.evaluate(baseline=None, current=benchmark(10.0)).passed
    assert gate.evaluate(baseline=None, current=benchmark(10.0), allow_missing_baseline=True).passed
    assert not gate.evaluate(baseline=None, current=None, current_judge=experiment(0.9)).passed


def test_a_judge_that_abstained_on_everything_is_not_a_failure() -> None:
    report = RegressionGate().evaluate(
        baseline=None, current=None, baseline_judge=experiment(0.9), current_judge=experiment(None)
    )
    assert report.passed
    assert report.findings[0].note is not None and "abstained" in report.findings[0].note


def test_a_gate_that_compared_nothing_fails() -> None:
    report = RegressionGate().evaluate(baseline=None, current=None)
    assert not report.passed
    assert report.findings[0].note == "nothing was compared"


def test_an_artifact_cannot_widen_the_gate_that_is_checking_it() -> None:
    hostile = {
        **benchmark(30.0),
        "max_latency_regression_pct": 1000.0,
        "thresholds": {"max_score_drop": 1.0},
    }
    assert not RegressionGate().evaluate(baseline=benchmark(10.0), current=hostile).passed


def test_a_scenario_the_baseline_never_measured_is_not_compared() -> None:
    current = benchmark(10.0)
    current["results"]["brand_new"] = {"p50": 99.0}
    report = RegressionGate().evaluate(baseline=benchmark(10.0), current=current)
    assert report.passed
    assert not any(f.metric.startswith("latency.brand_new") for f in report.findings)


def test_the_cli_exits_non_zero_on_failure_and_zero_on_a_pass(tmp_path, capsys) -> None:
    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"
    baseline.write_text(json.dumps(benchmark(10.0)))
    current.write_text(json.dumps(benchmark(10.5)))

    assert main(["--baseline", str(baseline), "--current", str(current)]) == 0
    assert "regression gate: passed" in capsys.readouterr().out

    current.write_text(json.dumps(benchmark(50.0)))
    assert main(["--baseline", str(baseline), "--current", str(current)]) == 1
    assert "FAILED" in capsys.readouterr().out


def test_the_cli_reads_the_judge_artifacts_and_the_thresholds_from_the_command_line(
    tmp_path, capsys
) -> None:
    before = tmp_path / "before.json"
    after = tmp_path / "after.json"
    before.write_text(json.dumps(experiment(0.9)))
    after.write_text(json.dumps(experiment(0.8)))

    argv = ["--baseline", str(tmp_path / "none.json"), "--judge", str(after)]
    assert main([*argv, "--baseline-judge", str(before), "--max-score-drop", "0.2"]) == 0
    assert main([*argv, "--baseline-judge", str(before), "--max-score-drop", "0.05"]) == 1
    assert "judge.mean_score" in capsys.readouterr().out

    assert main([*argv, "--allow-missing-baseline"]) == 0
