"""The answer checks of ``examples/06_scenarios/tool_choice.py``: what reads as a plain
"cannot" and as "there is no such record"."""

from __future__ import annotations

import importlib

from trellis.harness.evals import EvalCase

scenario = importlib.import_module("examples.06_scenarios.tool_choice")


def _case(output: str, **metadata: object) -> EvalCase:
    return EvalCase(input="q", output=output, metadata=metadata)


async def test_a_capability_answer_reads_no_with_any_of_the_cannot_words() -> None:
    no = "I'm sorry, I don't have the capability to accept image uploads."
    yes = "Yes, I can look up a store's stock."
    for output, expected, passed in ((no, "no", True), (no, "yes", False), (yes, "yes", True)):
        score = await scenario.plain_answer(_case(output, capability=expected))
        assert score is not None and score.value is passed, output


async def test_no_record_is_read_from_its_plain_wordings_and_never_with_a_figure() -> None:
    for output, passed in (
        ("The system does not have data on style ZZ999999.", True),
        ("There is no record of style ZZ999999.", True),
        ("Style ZZ999999 has 12 units.", False),
        ("There might be no record of ZZ999999.", False),
    ):
        score = await scenario.says_no_record(_case(output, no_record="ZZ999999"))
        assert score is not None and score.value is passed, output
