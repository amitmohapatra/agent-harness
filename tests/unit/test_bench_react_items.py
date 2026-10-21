"""The ReAct items' measurement on the framework benchmark: the ``C0`` and ``C+<item>`` arms,
the item cells (structured output, remembered rules, an image), the rubric given to a ReAct
arm, the counters of a run, and the keep-or-drop decision."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from benchmark.cost import catalog
from benchmark.cost.frameworks import arms as built
from benchmark.cost.frameworks import counters, items, runner
from tests.unit import test_bench_frameworks as fw
from trellis.harness import telemetry
from trellis.harness.telemetry import LoopEvent


# --------------------------------------------------------------------------- the arms
def test_c0_is_react_without_every_adaptive_item_and_c_plus_adds_one() -> None:
    assert built.ARMS["react/C0"].without == set(built.ADAPTIVE)
    one = built.ARMS["react/C+think"]
    assert one.without == set(built.ADAPTIVE) - {"think"} and one.kind == "C"
    assert built.ARMS["react/C"].without == set()
    assert built.ARMS["react/C-planning"].without == {"planning"}


def test_every_adaptive_item_is_a_feature_and_has_a_target() -> None:
    from typing import get_args

    from trellis.harness.features import COVERS, Feature

    assert set(built.ADAPTIVE) <= set(get_args(Feature))
    assert set(COVERS["react_adaptive"]) == set(built.ADAPTIVE)
    assert set(items.TARGETS) == set(built.ADAPTIVE)


# --------------------------------------------------------------------------- the cells
def test_the_item_cells_plan_their_own_tasks() -> None:
    output = runner.plan("output", 5, 0)
    assert {p.taskset.output for p in output} == {
        catalog.StyleFigure,
        catalog.TicketList,
        catalog.PendingClaim,
    }
    rules = runner.plan("rules", 4, 0)
    assert all(p.task["metadata"]["length"] == "rules" for p in rules)
    [image] = runner.plan("image", 1, 0)
    assert image.task["input"][0]["content"][1]["type"] == "image_url"
    assert all(t["metadata"]["length"] != "output" for s in catalog.task_sets() for t in s.tasks)


def test_every_long_task_has_a_rubric() -> None:
    long = [t["metadata"]["task"] for s in catalog.task_sets() for t in s.tasks]
    assert {n for n in long if n in catalog.RUBRICS} == set(catalog.RUBRICS)
    assert len(catalog.RUBRICS) == len(runner.plan("long", 3, 0))


def test_a_react_arm_gets_the_rubric_and_no_other_arm_does() -> None:
    [planned] = runner.plan("long", 1, 0)
    task = planned.task
    graded = runner.graded(built.ARMS["react/C0"], task)
    assert graded["input"]["rubric"] == catalog.RUBRICS[task["metadata"]["task"]]
    assert graded["input"]["messages"] == [{"role": "user", "content": task["input"]}]
    assert runner.graded(built.ARMS["langchain/B"], task) is task
    [short] = runner.plan("short", 1, 0)
    assert runner.graded(built.ARMS["react/C0"], short.task) is short.task
    [image] = runner.plan("image", 1, 0)
    assert runner.graded(built.ARMS["react/C0"], image.task) is image.task


# --------------------------------------------------------------------------- the counters
def test_a_runs_item_counters() -> None:
    used = {"input_tokens": 200, "cached_tokens": 50, "cache_reported": True, "refused_calls": 2}
    loop = {"malformed_args": 3, "repaired": 1, "strict_on": 1}
    assert runner.item_counters(used, loop, 1) == {
        "malformed_args": 3,
        "repairs": 1,
        "strict_on": 1,
        "context_overflow": 2,
        "cache_ratio": 0.25,
        "answer_tool_calls": 1,
    }
    quiet = runner.item_counters({"input_tokens": 0, "cache_reported": False}, {}, None)
    assert quiet["cache_ratio"] is None and quiet["context_overflow"] == 0


def test_the_answer_tools_calls_are_counted() -> None:
    def reply(*names: str) -> LLMResult:
        calls = [{"name": n, "args": {}, "id": str(i)} for i, n in enumerate(names)]
        return LLMResult(generations=[[ChatGeneration(message=AIMessage("", tool_calls=calls))]])

    with counters.answer_turns(catalog.StyleFigure) as turns:
        turns.on_llm_end(reply("StyleFigure", "excess_stock"))
        turns.on_llm_end(reply("excess_stock"))
    assert turns.calls == 1
    with counters.answer_turns(None) as none:
        none.on_llm_end(reply("StyleFigure"))
    assert none.calls == 0


def test_the_loop_events_are_counted_while_open_and_the_counter_restored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    own = vars(telemetry.metrics).get("loop")
    with counters.loop_counts() as loop:
        before = loop.now()
        telemetry.metrics.loop(LoopEvent.REPAIRED)
        telemetry.metrics.loop(LoopEvent.REPAIRED)
        telemetry.metrics.loop(LoopEvent.STRICT_ON)
        since = counters.LoopCounts.since(before, loop.now())
    assert since["repaired"] == 2 and since["strict_on"] == 1 and since["malformed_args"] == 0
    assert vars(telemetry.metrics).get("loop") is own  # the counter as it was
    seen: list[LoopEvent] = []
    monkeypatch.setattr(telemetry.metrics, "loop", seen.append)
    with counters.loop_counts() as loop:
        telemetry.metrics.loop(LoopEvent.REPAIRED)
    assert seen == [LoopEvent.REPAIRED] and loop.now()["repaired"] == 1
    assert telemetry.metrics.loop == seen.append  # a test's own counter restored


async def test_an_item_arm_runs_the_item_cells_with_its_counters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    every = [*catalog.task_sets(), *catalog.item_sets()]

    def needed(text: str) -> list[str]:
        for taskset in every:
            for task in taskset.tasks:
                if task["input"] == text or (
                    isinstance(task["input"], list) and text in json.dumps(task["input"])
                ):
                    return list(task["metadata"]["needed_tools"])
        return []

    monkeypatch.setattr(fw, "_needed", needed)
    for name in ("MEMORY_URL", "RUNS_URL", "TRELLIS_API_KEY", "OTEL_EXPORTER_OTLP_ENDPOINT"):
        monkeypatch.delenv(name, raising=False)
    with fw.serving(fw.Scripted()) as root:
        monkeypatch.setenv("BIFROST_URL", f"{root}/v1")
        monkeypatch.setenv("BIFROST_VIRTUAL_KEY", "vk")
        options = fw._options(
            root, "react/C+tool_repair", tmp_path / "rows.jsonl", ("output", "rules"), 2
        )
        rows = await runner.run_arm(options, progress=lambda _: None)
    output, _, rule, _ = rows
    assert output["cell"] == "output" and output["window"] == 8192
    assert output["answer_tool_calls"] == 0 and rule["answer_tool_calls"] is None
    assert {"malformed_args", "repairs", "strict_on", "context_overflow", "cache_ratio"} <= set(
        output
    )
    assert rule["success"] is True  # the rule forbids the change: nothing called


# --------------------------------------------------------------------------- the decision
def _row(arm: str, index: int, **values: Any) -> dict[str, Any]:
    row = {
        "arm": arm,
        "cell": "short",
        "index": index,
        "window": 8192,
        "success": True,
        "input_tokens": 1000,
        "output_tokens": 100,
        "model_calls": 3,
        "cpu_seconds": {"arm": 1.0},
        "peak_rss_mb": 300.0,
    }
    return row | values


def _pairs(item: str, base: list[dict[str, Any]], mine: list[dict[str, Any]]) -> list[dict]:
    rows = [_row("react/C0", n, **v) for n, v in enumerate(base)]
    return rows + [_row(f"react/C+{item}", n, **v) for n, v in enumerate(mine)]


def test_an_item_that_improves_its_target_within_the_margins_is_kept() -> None:
    rows = _pairs("tool_repair", [{"model_calls": 3}] * 5, [{"model_calls": 2}] * 4 + [{}])
    decided = items.decide(rows, "tool_repair")
    assert decided["kept"] and decided["why"] == "kept" and decided["pairs"] == 5


@pytest.mark.parametrize(
    ("mine", "why"),
    [
        ([{"success": False, "model_calls": 2}] * 5, "task success fell"),
        ([{"input_tokens": 1500, "model_calls": 2}] * 5, "tokens per run rose"),
        ([{"model_calls": 3}] * 5, "model_calls not lower"),
        ([{"model_calls": 4}, {"model_calls": 2}] * 2 + [{}], "model_calls not lower"),
    ],
)
def test_an_item_that_fails_a_rule_is_dropped_saying_which(
    mine: list[dict[str, Any]], why: str
) -> None:
    decided = items.decide(_pairs("tool_repair", [{}] * 5, mine), "tool_repair")
    assert not decided["kept"] and decided["why"].startswith(why)


def test_a_higher_is_better_target_and_an_unmeasured_item() -> None:
    rows = _pairs("rubric", [{"success": False}] * 4 + [{}], [{}] * 5)
    for row in rows:
        row["cell"] = "long"
    assert items.decide(rows, "rubric")["kept"]
    assert items.decide(rows, "think") == {
        "item": "think",
        "kept": False,
        "why": "not measured",
        "pairs": 0,
    }


def test_the_decisions_table_and_the_command(tmp_path: Path, capsys: Any) -> None:
    rows = _pairs("tool_repair", [{}] * 5, [{"model_calls": 2}] * 5)
    path = tmp_path / "rows.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    out = tmp_path / "decisions.json"
    assert items.main([str(path), "--json", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "| tool_repair | 5 | short |" in printed and "**kept**" in printed
    assert "| think | 0 | — |" in printed
    decided = {d["item"]: d for d in json.loads(out.read_text())}
    assert decided["tool_repair"]["kept"] and not decided["rubric"]["kept"]


def test_the_multi_cell_scores_the_combined_answer_and_the_serial_arm_runs_one_helper() -> None:
    planned = runner.plan("multi", 3, 0)
    assert all(p.task["metadata"]["needed_tools"] == [] and p.task["expected"] for p in planned)
    serial = built.ARMS["react/C+subagents-serial"]
    assert serial.helpers_at_once == 1 and serial.without == set(built.ADAPTIVE) - {"subagents"}
    assert len(catalog.every_set()) == len(catalog.task_sets()) + 5 + 3


async def test_the_serial_arm_sets_one_helper_at_a_time(monkeypatch: pytest.MonkeyPatch) -> None:
    from trellis.harness import adaptive

    monkeypatch.setattr(adaptive, "HELPERS_AT_ONCE", adaptive.HELPERS_AT_ONCE)
    setup = built.Setup.on("http://p", "granite/g", "vk")
    monkeypatch.setenv("BIFROST_URL", "http://p/v1")
    await built.wrapped(built.ARMS["react/C+subagents-serial"], setup, None, "s", [])
    assert adaptive.HELPERS_AT_ONCE == 1
