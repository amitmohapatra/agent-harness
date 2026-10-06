"""Prompts and skills from wherever they are kept: a folder of ``.md`` prompts, a folder of
Agent Skills (``<name>/SKILL.md``), and a prompt and a skill in code — read by a ``ReAct`` and by
a plain function, the same way every framework reads them.

A deployment names its folders with ``PROMPTS_DIR`` and ``SKILLS_DIR`` (and Langfuse with its
keys, the gateway with ``BIFROST_URL``); this example passes them to ``Harness`` instead, so it
runs from anywhere.

    python -m examples.05_features.skills_and_prompts
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from examples._support.offline import react_model

from trellis import Harness, ReAct, Runtime
from trellis.harness.prompts import Prompt, prompts_dir
from trellis.harness.skills import Skill, skills_dir

HERE = Path(__file__).resolve().parent
TONE = Skill("tone", "How we write review comments.", "One line per finding. No blame.")


async def main() -> None:
    async with Harness(
        prompts=[prompts_dir(HERE / "prompts"), Prompt("greet", "Greet {{name}} in one line.")],
        skills=[skills_dir(HERE / "skills")],
    ) as h:
        # a ReAct: the folder's prompt is its instructions; skills of both sources, one section
        model = react_model(
            [
                ("load_skill", {"name": "sql-review"}),
                ("read_skill_file", {"name": "sql-review", "path": "rules.md"}),
                "It breaks rule 1: name the columns instead of SELECT *.",
            ]
        )
        reviewer = h.wrap(
            ReAct(system="", model=model, prompt="review@1", prompt_vars={"team": "data"}),
            id="sql-reviewer",
            skills=["sql-review", TONE],
        )
        result = await reviewer.run("Review: SELECT * FROM orders", user="ada")
        print(result.status.value, result.answer)

        # any other framework (here a function): h.prompt, pinned for the run
        async def greeter(name: str, agent: Runtime) -> str:
            return await h.prompt("greet", name=name)

        greeted = await h.wrap(greeter, id="greeter").run("Ada", user="ada")
        print(greeted.status.value, greeted.answer)


if __name__ == "__main__":
    asyncio.run(main())
