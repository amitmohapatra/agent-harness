"""Way 1, wrapped — offline evaluation: run a wrapped agent over a dataset with ``h.evaluate``,
score every answer, read the report. (Your own code, not wrapped: ``blocks_evaluate.py``.)

    .venv/bin/python examples/evaluate_offline.py

The dataset here is a local list; with Langfuse configured (the OTLP settings,
``docs/evaluation.md``) pass a Langfuse dataset's name instead and every run is linked to the
dataset run there. Offline the judge is a scripted one; online set ``TRELLIS_JUDGE_MODEL`` to
another, stronger model than the agent's (unset, the judge falls back to the agent's own gateway
model, with a warning: a model grading its own answers is biased).
"""

from __future__ import annotations

import asyncio

from _offline import answering_model, judge_offline

from trellis import Harness, ReAct
from trellis.harness.evals import EvalReport, contains, exact_match, llm_judge

ANSWERS = {
    "What is the capital of France?": "Paris",
    "What is the capital of Japan?": "Tokyo",
    "What is the capital of Australia?": "Sydney",
}
DATASET = [
    {"input": "What is the capital of France?", "expected": "Paris"},
    {"input": "What is the capital of Japan?", "expected": "Tokyo"},
    {"input": "What is the capital of Australia?", "expected": "Canberra"},
]


def show(report: EvalReport) -> None:
    print(report)
    for item in report.items:
        scores = ", ".join(f"{s.name}={s.value}" for s in item.scores)
        print(f"{item.status:<8} {item.input} -> {item.output} [{scores}]")


async def main() -> None:
    # each item runs through the harness pipeline (memory, tools, approvals)
    async with Harness() as h:
        judge_offline(h)
        target = ReAct(
            system="Answer with the name of the city only.", model=answering_model(ANSWERS)
        )
        agent = h.wrap(target, id="capitals")
        show(
            await h.evaluate(
                agent,
                DATASET,
                [
                    exact_match(),
                    contains(),
                    llm_judge("The answer names the correct capital city."),
                ],
                run_name="capitals-offline",
            )
        )


if __name__ == "__main__":
    asyncio.run(main())
