"""Way 1, wrapped — online judging: every sampled successful run is scored in the background by
the agent's judges (``h.wrap(judges=[...])``) — never on the request path; the scores land on
each run's trace. (Your own code, not wrapped, judges its runs with ``judge()``:
``examples/03_way2_evals``.)

Each judge scores the share of runs its ``sample`` says, chosen by the run id
(``llm_judge(sample=)``; a judge of your own may say it too; else a tenth). This example judges
every run.

    python -m examples.05_features.online_judges
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from examples._support.offline import answering_model, judge_offline

from trellis import Harness, ReAct, llm_judge
from trellis.harness.evals import EvalCase, EvalScore


@dataclass
class Concise:
    """A judge of your own: any async callable of the case (``sample``: the share it scores)."""

    name: str = "concise"
    sample: float = 1.0

    async def __call__(self, case: EvalCase) -> EvalScore:
        words = len(str(case.output).split())
        print(f"judged {case.run_id}: {words} word(s)")
        return EvalScore("concise", words <= 5, f"{words} words")


async def main() -> None:
    judges = [Concise(), llm_judge("Answers the question correctly.", name="correct", sample=1.0)]
    async with Harness() as h:
        judge_offline(h)
        model = answering_model({"What is the capital of Japan?": "Tokyo"})
        target = ReAct(system_prompt="Answer briefly.", model=model)
        agent = h.wrap(target, id="concierge", judges=judges)
        result = await agent.run("What is the capital of Japan?", user="ada")
        print(result.status.value, result.answer)  # returned before any judge ran
        await h.writes.drain()  # the judges ran in the background writes queue


if __name__ == "__main__":
    asyncio.run(main())
