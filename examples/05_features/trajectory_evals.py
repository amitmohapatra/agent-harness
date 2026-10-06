"""Offline evaluation of a wrapped agent: ``h.evaluate`` runs it over a dataset and scores every
answer — what it said (exact match, contains, a judge) and what it did (its trajectory: which
tools it called, in what order, with what arguments).

* ``exact_match()``, ``contains()`` — against each item's ``expected``;
* ``llm_judge(criteria)`` — a judge model grades the answer (offline a scripted judge; online
  ``TRELLIS_JUDGE_MODEL``, a stronger model than the agent's);
* ``called("lookup")``, ``tool_sequence(["lookup"], exact=True)`` — against the run's
  tool calls (``EvalCase.trajectory``); your own evaluator can read it too.

The dataset is a local list; with Langfuse configured pass a Langfuse dataset's name instead and
every run is linked to the dataset run there. ``concurrency=1`` keeps the scripted model's
answers in the dataset's order. (Code that is not wrapped: ``examples/03_way2_evals``.)

    python -m examples.05_features.trajectory_evals
"""

from __future__ import annotations

import asyncio

from examples._support.offline import judge_offline, react_model

from trellis import Harness, ReAct, tool
from trellis.harness.evals import (
    EvalCase,
    EvalReport,
    EvalScore,
    called,
    contains,
    exact_match,
    llm_judge,
    tool_sequence,
)

CAPITALS = {"France": "Paris", "Japan": "Tokyo", "Australia": "Canberra"}
DATASET = [
    {"input": "What is the capital of France?", "expected": "Paris"},
    {"input": "What is the capital of Japan?", "expected": "Tokyo"},
    {"input": "What is the capital of Australia?", "expected": "Canberra"},
]


@tool(side_effects="read")
def lookup(country: str) -> str:
    """The capital of a country."""
    return CAPITALS.get(country, "unknown")


@tool(side_effects="read")
def guess(country: str) -> str:
    """A guess at the capital, from the biggest city."""
    return {"Australia": "Sydney"}.get(country, "unknown")


async def few_calls(case: EvalCase) -> EvalScore | None:
    """Your own evaluator: at most two tool calls per answer."""
    if case.trajectory is None:
        return None
    return EvalScore("few_calls", len(case.trajectory) <= 2, f"{len(case.trajectory)} calls")


def show(report: EvalReport) -> None:
    print(report)
    for item in report.items:
        scores = ", ".join(f"{s.name}={s.value}" for s in item.scores)
        print(f"{item.status:<8} {item.input} -> {item.output} [{scores}]")


async def main() -> None:
    model = react_model(
        [
            ("lookup", {"country": "France"}),
            "Paris",
            ("lookup", {"country": "Japan"}),
            "Tokyo",
            ("guess", {"country": "Australia"}),  # the wrong tool: the trajectory says so
            "Sydney",
        ]
    )
    async with Harness() as h:
        judge_offline(h)
        target = ReAct(system="Answer with the name of the city only.", model=model)
        agent = h.wrap(target, id="capitals", tools=[lookup, guess])
        report = await h.evaluate(
            agent,
            DATASET,
            [
                exact_match(),
                contains(),
                llm_judge("The answer names the correct capital city."),
                called("lookup"),
                tool_sequence(["lookup"], exact=True),
                few_calls,
            ],
            run_name="capitals-offline",
            concurrency=1,
        )
        show(report)


if __name__ == "__main__":
    asyncio.run(main())
