"""The thread-history benchmark's report: recall and tokens per turn for each owner."""

from __future__ import annotations

from benchmark.thread.__main__ import CONVERSATIONS, markdown, summary


def test_the_report_gives_recall_and_tokens_per_turn_for_each_owner() -> None:
    rows = [
        {
            "variant": v,
            "turn": t,
            "status": "SUCCESS",
            "input_tokens": 100 * (t + 1),
            "seconds": 1.0,
        }
        | ({"recalled": v == "framework" or t == 3} if t >= 3 else {})
        for v in ("framework", "memory")
        for t in range(5)
    ]
    found = summary(rows)
    assert found["framework"]["recall"] == 1.0 and found["memory"]["recall"] == 0.5
    assert found["memory"]["input_tokens_by_turn"][4] == 500
    table = markdown({"model": "m", "repeats": 1, "summary": found})
    assert "| framework | 1.00 (n=2) | 0 | 300 | 100 / 200 / 300 / 400 / 500 | 1.0 |" in table
    assert all(len(turns) == 5 and expected for turns, expected in CONVERSATIONS)
