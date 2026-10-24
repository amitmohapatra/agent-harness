"""The thread-history benchmark's report: recall, tokens, and the overhead past the model per
turn for each owner."""

from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver

from benchmark.thread.__main__ import (
    CONVERSATIONS,
    RESTART_BEFORE_TURN,
    Variant,
    markdown,
    own_thread,
    percentile,
    summary,
)
from trellis.harness.history import Owner, owner


def test_the_report_gives_recall_tokens_and_overhead_per_turn_for_each_owner() -> None:
    rows = [
        {
            "variant": v,
            "turn": t,
            "status": "SUCCESS",
            "input_tokens": 100 * (t + 1),
            "output_tokens": 10,
            "seconds": 1.0,
            "model_seconds": 0.9,
            "overhead_ms": 10.0 * (t + 1),
            "context_ms": 5.0,
            "drain_ms": 0.0,
        }
        | ({"recalled": v == "framework" or t == 3} if t >= 3 else {})
        for v in ("framework", "memory")
        for t in range(5)
    ]
    found = summary(rows)
    assert found["framework"]["recall"] == 1.0 and found["memory"]["recall"] == 0.5
    assert found["memory"]["input_tokens_by_turn"][4] == 500
    assert found["memory"]["overhead_ms_p50"] == 30.0
    assert found["memory"]["overhead_ms_p95"] == 48.0
    table = markdown({"model": "m", "repeats": 1, "summary": found})
    assert (
        "| framework | 1.00 (n=2) | 0 | 300 | 100 / 200 / 300 / 400 / 500 | 10 | 1.0 | 0.9 "
        "| 30 / 48 | 5 |" in table
    )
    assert all(len(turns) == 5 and expected for turns, expected in CONVERSATIONS)
    # the restart falls between the facts said and the facts asked back
    assert all(min(expected) >= RESTART_BEFORE_TURN for _, expected in CONVERSATIONS)


def test_a_percentile_is_read_between_ranks() -> None:
    assert percentile([], 95) == 0.0
    assert percentile([5.0], 95) == 5.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5


def test_the_variants_say_who_owns_the_thread_and_whether_they_restart() -> None:
    assert Variant.FRAMEWORK.framework_thread and not Variant.FRAMEWORK.restarts
    assert Variant.MEMORY_RESTART.restarts and not Variant.MEMORY_RESTART.framework_thread
    assert Variant.FRAMEWORK_RESTART.restarts and Variant.FRAMEWORK_RESTART.framework_thread


async def test_the_framework_variant_owns_its_thread_and_a_new_one_starts_empty() -> None:
    saver = own_thread()
    assert owner(saver, agent_id="b") is Owner.FRAMEWORK
    assert owner(InMemorySaver(), agent_id="b") is Owner.MEMORY
    config = {"configurable": {"thread_id": "t", "checkpoint_ns": ""}}
    checkpoint = {"v": 1, "id": "c1", "ts": "", "channel_values": {}, "channel_versions": {}}
    await saver.aput(config, checkpoint, {}, {})
    assert (await saver.aget_tuple(config)).checkpoint["id"] == "c1"
    assert [c async for c in saver.alist(config)]
    assert await own_thread().aget_tuple(config) is None  # a restart's new saver
