"""Offline evaluation: run an agent over a dataset, score every answer, read the report.

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

from trellis import Harness, ReAct, contains, exact_match, llm_judge

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


async def main() -> None:
    async with Harness() as h:
        target = ReAct(
            system="Answer with the name of the city only.", model=answering_model(ANSWERS)
        )
        agent = h.wrap(target, id="capitals")
        report = await h.evaluate(
            agent,
            DATASET,
            [exact_match(), contains(), llm_judge("The answer names the correct capital city.")],
            run_name="capitals-offline",
        )
        print(report)
        for item in report.items:
            scores = ", ".join(f"{s.name}={s.value}" for s in item.scores)
            print(f"{item.status:<8} {item.input} -> {item.output} [{scores}]")


if __name__ == "__main__":
    asyncio.run(main())
