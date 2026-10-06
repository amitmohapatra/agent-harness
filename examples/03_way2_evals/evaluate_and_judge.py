"""Way 2, pluggable blocks: evaluation of your own code, not wrapped (``trellis.harness.evals``).

* Offline: ``evaluate`` runs an ``async (input) -> answer`` of yours over a dataset — each item
  one call, in a span of its own, an item of a Langfuse experiment — and scores every answer.
* Online: ``judge`` scores one run your code just finished, on that run's trace, with a judge
  of your own and a judge model; ``sample=`` judges a stable share of runs.

What evaluation reaches is ``EvalServices.from_env()``: Langfuse (the OTLP settings) and the
judge (``BIFROST_URL``, ``TRELLIS_JUDGE_MODEL``). Offline the judge is a scripted model; the
dataset is a local list (with Langfuse configured, pass a Langfuse dataset's name instead).

    python -m examples.03_way2_evals.evaluate_and_judge
"""

from __future__ import annotations

import asyncio

from examples._support.offline import judge_services

from trellis.harness.evals import (
    EvalCase,
    EvalScore,
    contains,
    evaluate,
    exact_match,
    judge,
    llm_judge,
)

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


async def concise(case: EvalCase) -> EvalScore:
    """A judge of your own: any async function of the case."""
    words = len(str(case.output).split())
    return EvalScore("concise", words <= 5, f"{words} words")


async def main() -> None:
    async with judge_services() as services:
        # offline: every item of the dataset, scored
        report = await evaluate(
            capitals,
            DATASET,
            [exact_match(), contains(), llm_judge("The answer names the correct capital city.")],
            services=services,
            run_name="capitals-own",
        )
        print(report)
        for item in report.items:
            scores = ", ".join(f"{s.name}={s.value}" for s in item.scores)
            print(f"{item.status:<8} {item.input} -> {item.output} [{scores}]")

        # online: one run of yours, judged on its trace (or on your own trace: trace_id=)
        question = "What is the capital of Japan?"
        answer = await capitals(question)
        case = EvalCase(input=question, output=answer, run_id="run_capitals_1")
        scores, failed = await judge(
            case, [concise, llm_judge("Answers the question correctly.")], services=services
        )
        print("judged:", [(s.name, s.value) for s in scores], failed)


if __name__ == "__main__":
    asyncio.run(main())
