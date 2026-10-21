"""The thread-history benchmark's report: recall, history after the restart and tokens per turn
for each owner, and the decision its rule makes."""

from __future__ import annotations

from typing import Any

import pytest

from benchmark.thread.__main__ import (
    CONVERSATIONS,
    RESTART_AFTER,
    TARGETS,
    decision,
    markdown,
    summary,
)


def rows(variant: str, *, tokens: int = 100, recalled: bool = True, history: bool = True) -> Any:
    return [
        {
            "variant": variant,
            "turn": t,
            "after_restart": t >= RESTART_AFTER,
            "status": "SUCCESS",
            "input_tokens": tokens * (t + 1),
            "seconds": 1.0,
            "history": history,
        }
        | ({"recalled": recalled or t == 3} if t >= 3 else {})
        for t in range(5)
    ]


def test_the_report_gives_recall_history_and_tokens_per_turn_for_each_owner() -> None:
    found = summary(rows("framework") + rows("memory", recalled=False))
    assert found["framework"]["recall"] == 1.0 and found["memory"]["recall"] == 0.5
    assert found["memory"]["input_tokens_by_turn"][4] == 500
    assert found["framework"]["history_after_restart"] == 1.0
    report = {"model": "m", "repeats": 1, "summary": found, "decision": decision(found)}
    table = markdown(report)
    assert (
        "| framework | 1.00 (n=2) | 1.00 | 0 | 300 | 100 / 200 / 300 / 400 / 500 | 1.0 |" in table
    )
    assert "Decision: the memory window saves 0% of the input tokens" in table
    assert all(len(turns) == 5 and expected for turns, expected in CONVERSATIONS)
    assert set(TARGETS) == {"framework", "framework-ram", "memory"}


@pytest.mark.parametrize(
    ("memory", "owner"),
    [
        ({"tokens": 79}, "the owner is memory"),  # 21% fewer, recall equal
        ({"tokens": 81}, "the owner is framework"),  # 19% fewer: not enough
        ({"tokens": 50, "recalled": False}, "the owner is framework"),  # fewer, worse recall
    ],
)
def test_the_memory_window_wins_only_with_equal_recall_and_a_fifth_fewer_tokens(
    memory: dict[str, Any], owner: str
) -> None:
    found = summary(rows("framework") + rows("memory", **memory))
    assert decision(found).endswith(owner)


def test_a_variant_that_lost_the_history_after_the_restart_loses_outright() -> None:
    found = summary(
        rows("framework-ram", tokens=10, history=False) + rows("framework") + rows("memory")
    )
    said = decision(found)
    assert said.startswith("framework-ram loses outright") and "the owner is" in said
    lost = summary(rows("framework", history=False) + rows("memory", tokens=500))
    assert decision(lost).endswith("the owner is memory")
    gone = summary(rows("framework", history=False) + rows("memory", history=False))
    assert decision(gone).endswith("no variant stands")
