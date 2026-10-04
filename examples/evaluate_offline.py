"""Offline evaluation: run an agent over a dataset, score every answer, read the report — a
wrapped agent with ``h.evaluate``, and your own code, unwrapped, with ``evaluate``.

    .venv/bin/python examples/evaluate_offline.py

The dataset here is a local list; with Langfuse configured (the OTLP settings,
``docs/evaluation.md``) pass a Langfuse dataset's name instead and every run is linked to the
dataset run there. Offline the judge is the scripted model (``TRELLIS_JUDGE_MODEL`` unset: the
judge shares the agent's model, which is logged); online set ``TRELLIS_JUDGE_MODEL`` to a
stronger model than the agent's.
"""

from __future__ import annotations

import asyncio

from _offline import answering_model

from trellis import Harness, ReAct
from trellis.harness.evals import EvalReport, contains, evaluate, exact_match, llm_judge

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


async def capitals(question: str) -> str:
    """Your own agent, in any framework: any ``async (input) -> answer``."""
    return ANSWERS.get(question, "I don't know")


def show(report: EvalReport) -> None:
    print(report)
    for item in report.items:
        scores = ", ".join(f"{s.name}={s.value}" for s in item.scores)
        print(f"{item.status:<8} {item.input} -> {item.output} [{scores}]")


async def main() -> None:
    # Way 1, wrapped: each item runs through the harness pipeline (memory, tools, approvals)
    async with Harness() as h:
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

    # Way 2, pluggable: your code as it is; Langfuse and the judge come from the environment
    show(await evaluate(capitals, DATASET, [exact_match(), contains()], run_name="capitals-own"))


if __name__ == "__main__":
    asyncio.run(main())
