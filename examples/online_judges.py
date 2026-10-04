"""Online judging: every sampled successful run is scored in the background by the judges the
harness was given — never on the request path; the scores land on each run's trace.

    .venv/bin/python examples/online_judges.py

``TRELLIS_JUDGE_SAMPLE`` is the share of runs judged (0.1 by default with judges); this example
judges every run unless the environment says otherwise.
"""

from __future__ import annotations

import asyncio
import os

from _offline import answering_model

from trellis import EvalCase, EvalScore, Harness, ReAct, Settings, llm_judge


async def concise(case: EvalCase) -> EvalScore:
    """A custom judge: any async function of the case."""
    words = len(str(case.output).split())
    print(f"judged {case.run_id}: {words} word(s)")
    return EvalScore("concise", words <= 5, f"{words} words")


async def main() -> None:
    environment = {"TRELLIS_JUDGE_SAMPLE": "1", **os.environ}
    judges = [concise, llm_judge("Answers the question correctly.", name="correct")]
    async with Harness(config=Settings.from_env(environment), judges=judges) as h:
        model = answering_model({"What is the capital of Japan?": "Tokyo"})
        agent = h.wrap(ReAct(system="Answer briefly.", model=model), id="concierge")
        result = await agent.run("What is the capital of Japan?", user="ada")
        print(result.status.value, result.answer)  # returned before any judge ran
        await h.writes.drain()  # the judges ran in the background writes queue


if __name__ == "__main__":
    asyncio.run(main())
