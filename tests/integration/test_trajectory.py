"""Trajectory evaluations: an evaluator sees which harness tools a run called, in order, with
their arguments and outcomes (``EvalCase.trajectory``, the run's journal) — offline
(``h.evaluate``), online (the judges, across a pause), and for any code that returns its own
(``EvalOutput.trajectory``) — on every adapter; and the two built-in trajectory evaluators."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

import pytest

from tests.support.adapters import BUILDERS
from tests.support.planned import Call
from trellis import Harness, Settings, tool
from trellis.contracts import RunStatus, ToolCall, ToolOutcome, ToolStatus
from trellis.harness.evals import (
    EvalCase,
    EvalOutput,
    EvalScore,
    EvalServices,
    called,
    evaluate,
    tool_sequence,
)

PLAN: Final[list[Call]] = [("lookup", {"topic": "tides"}), ("note", {"text": "high"})]


@tool(side_effects="read")
async def lookup(topic: str) -> str:
    """Look a topic up."""
    return f"facts about {topic}"


@tool(side_effects="write")
async def note(text: str) -> str:
    """Write a note down."""
    return f"noted {text}"


@tool(side_effects="irreversible")
async def refund(order: str) -> str:
    """Refund an order."""
    return f"refunded {order}"


class Seen:
    """An evaluator that keeps the cases it was given."""

    name = "seen"

    def __init__(self) -> None:
        self.cases: list[EvalCase] = []

    async def __call__(self, case: EvalCase) -> EvalScore | None:
        self.cases.append(case)
        return None


def steps(case: EvalCase) -> list[tuple[str, dict[str, Any], ToolStatus]]:
    assert case.trajectory is not None
    return [(c.tool, c.args, o.status) for c, o in case.trajectory]


@pytest.mark.parametrize("framework", list(BUILDERS))
async def test_h_evaluate_gives_every_adapters_trajectory_to_its_evaluators(
    harness: Harness, framework: str, tmp_path: Path
) -> None:
    target, tools = await BUILDERS[framework](harness, [lookup, note], tmp_path, PLAN)
    agent = harness.wrap(target, id=f"traj-{framework}", tools=tools)
    seen = Seen()
    evaluators = [
        seen,
        called("lookup", before="note"),
        called("note", args={"text": "high"}),
        called("refund"),
        tool_sequence(["lookup", "note"], exact=True),
    ]
    report = await harness.evaluate(agent, [{"input": "tides?"}], evaluators)
    [item] = report.items
    assert item.status == "success", item.error
    assert {s.name: s.value for s in item.scores} == {
        "called:lookup": True,
        "called:note": True,
        "called:refund": False,
        "tool_sequence": True,
    }
    assert steps(seen.cases[0]) == [
        ("lookup", {"topic": "tides"}, ToolStatus.OK),
        ("note", {"text": "high"}, ToolStatus.OK),
    ]
    first, outcome = seen.cases[0].trajectory[0]  # type: ignore[index]
    assert first.task is None and outcome.output == "facts about tides"


@pytest.mark.parametrize("framework", list(BUILDERS))
async def test_an_online_judge_sees_the_whole_trajectory_of_a_resumed_run(
    framework: str, tmp_path: Path
) -> None:
    """The call before the pause ran in the first attempt, the approved one in the second: the
    judge of the run reads both, once each, in order — whether the framework re-ran the run
    from its input (the first call replayed from the journal) or went on where it stopped."""
    seen = Seen()
    async with Harness(config=Settings(judge_sample=1.0), judges=[seen]) as h:
        plan: list[Call] = [("lookup", {"topic": "o1"}), ("refund", {"order": "o1"})]
        target, tools = await BUILDERS[framework](h, [lookup, refund], tmp_path, plan)
        agent = h.wrap(target, id=f"resumed-{framework}", tools=tools)
        paused = await agent.run("refund o1", user="u")
        assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
        done = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="ops")
        await h.writes.drain()
    assert done.status is RunStatus.SUCCESS, done.error
    assert [steps(c) for c in seen.cases] == [
        [("lookup", {"topic": "o1"}, ToolStatus.OK), ("refund", {"order": "o1"}, ToolStatus.OK)]
    ]


async def test_any_code_gives_its_own_trajectory_or_gets_no_trajectory_score() -> None:
    trajectory = [
        (ToolCall(tool="lookup", args={"topic": "a"}), ToolOutcome(tool="lookup", output="x")),
        (ToolCall(tool="note", args={"text": "b"}), ToolOutcome(tool="note", output="y")),
    ]

    async def with_steps(input: str) -> EvalOutput:
        return EvalOutput("done", trajectory=trajectory)

    async def bare(input: str) -> str:
        return "done"

    evaluators = [called("note", before="lookup"), tool_sequence(["note"])]
    async with EvalServices() as services:
        own = await evaluate(with_steps, [{"input": "q"}], evaluators, services=services)
        none = await evaluate(bare, [{"input": "q"}], evaluators, services=services)
    assert [(s.name, s.value, s.comment) for s in own.items[0].scores] == [
        ("called:note", False, "note came after lookup: the calls were lookup, note"),
        ("tool_sequence", True, None),
    ]
    assert none.items[0].scores == [] and none.summary["tool_sequence"].count == 0


@pytest.mark.parametrize(
    ("evaluator", "steps_made", "value", "comment"),
    [
        (called("a"), [], False, "a was not called: no tool was called"),
        (called("a", args={"x": 2}), [("a", {"x": 1})], False, "a was not called with those"),
        (called("a", args={"x": 1}), [("a", {"x": 1, "y": 0})], True, None),
        (called("b", args={"x": 1}), [("a", {"x": 1})], False, "b was not called: the calls"),
        (called("a", before="b"), [("a", {})], False, "b was not called: the calls were a"),
        (called("a", before="b"), [("b", {}), ("a", {}), ("b", {})], False, "a came after b"),
        (called("a", before="b"), [("c", {}), ("a", {}), ("b", {})], True, None),
        (tool_sequence(["a", "b"]), [("b", {}), ("a", {})], False, "the calls were b, a"),
        (tool_sequence(["a", "b"]), [("a", {}), ("c", {}), ("b", {})], True, None),
        (tool_sequence(["a", "b"], exact=True), [("a", {}), ("c", {}), ("b", {})], False, "a, c"),
        (tool_sequence([], exact=True), [], True, None),
    ],
)
async def test_the_trajectory_evaluators(
    evaluator: Any, steps_made: list[tuple[str, dict[str, Any]]], value: bool, comment: str | None
) -> None:
    trajectory = [(ToolCall(tool=t, args=a), ToolOutcome(tool=t)) for t, a in steps_made]
    score = await evaluator(EvalCase(input="q", output="a", trajectory=trajectory))
    assert score is not None and score.value is value
    assert (score.comment is None) if comment is None else (comment in (score.comment or ""))
    assert await evaluator(EvalCase(input="q", output="a")) is None  # no trajectory, no score


def test_a_trajectory_evaluator_takes_a_name() -> None:
    assert called("refund", name="refunded").name == "refunded"
    assert tool_sequence(["a"], name="order").name == "order"
