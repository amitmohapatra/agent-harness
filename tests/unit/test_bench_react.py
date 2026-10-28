"""The ReAct loop's benchmark: its task classes (``benchmark.cost.classes``), the fault a task
injects, the wrong calls a run is charged, the verdict's rules (``benchmark.cost.analyze``) and
the loop's arms (``benchmark.cost.variants``)."""

from __future__ import annotations

import collections
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from benchmark.cost import analyze, catalog, classes, run, variants
from benchmark.cost.variants import VARIANTS, Setup
from tests.support.chat_model import ScriptedChatModel
from trellis.contracts import ToolCall, ToolOutcome, ToolStatus


# --------------------------------------------------------------------------- the classes
def test_twenty_distinct_tasks_per_class_on_their_own_tools() -> None:
    sets = classes.react_sets()
    tasks = [t for s in sets for t in s.tasks]
    counted = collections.Counter(t["metadata"]["length"] for t in tasks)
    assert counted == {**dict.fromkeys(classes.CLASSES, 20), "helpers": 10, "reasoning": 40}
    assert len({t["metadata"]["task"] for t in tasks}) == len(tasks) == 170
    for taskset in sets:
        own = {t.spec.name for t in taskset.tools}
        for task in taskset.tasks:
            meta = task["metadata"]
            assert set(meta["needed_tools"]) | set(meta["avoid_tools"]) <= own, meta["task"]


def test_the_error_class_fails_a_read_ten_five_and_five_ways() -> None:
    tasks = [t for s in classes.react_sets() for t in s.tasks]
    faulted = [t for t in tasks if t["metadata"]["length"] == "error"]
    kinds = collections.Counter(t["metadata"][classes.INJECT]["kind"] for t in faulted)
    assert kinds == {"error": 10, "timeout": 5, "schema": 5}
    changes = {"create_stock_rule", "reset_password", "submit_expense"}
    for task in faulted:
        fault = classes.fault_of(task)
        assert fault is not None and fault.tool in task["metadata"]["needed_tools"]
        assert not changes & set(task["metadata"]["needed_tools"])  # it would pause first
    assert classes.fault_of(tasks[0]) is None
    assert classes.Fault("t", "timeout").outcome({})[0] is ToolStatus.TIMEOUT
    status, said = classes.Fault("t", "schema").outcome({"store": "S1"})
    assert status is ToolStatus.ERROR and "fields sent: store" in said
    assert "409 Conflict" in classes.Fault("t", "error").outcome({})[1]


def test_the_memory_and_ambiguous_classes_are_scored_as_they_say() -> None:
    tasks = [t for s in classes.react_sets() for t in s.tasks]
    memory = [t for t in tasks if t["metadata"]["length"] == "memory"]
    assert sum(bool(t["metadata"]["no_record"]) for t in memory) == 5
    assert sum(t.get("expected") is not None for t in memory) == 15
    ambiguous = [t for t in tasks if t["metadata"]["length"] == "ambiguous"]
    asked = [t for t in ambiguous if t["metadata"].get("ambiguous")]
    assert len(asked) == 10 and all(t["metadata"]["avoid_tools"] for t in asked)
    controls = [t for t in ambiguous if not t["metadata"].get("ambiguous")]
    assert all(len(t["metadata"]["needed_tools"]) == 1 for t in controls)
    helpers = [t for t in tasks if t["metadata"]["length"] == "helpers"]
    assert min(len(t["expected"]) for t in helpers) >= 8


async def test_a_change_missing_a_value_is_asked_for_not_made() -> None:
    meta = {"ambiguous": True, "avoid_tools": ["reset_password"]}

    def case(output: str, made: list[str]) -> Any:
        trajectory = [(SimpleNamespace(tool=t), None) for t in made]
        return SimpleNamespace(metadata=meta, output=output, trajectory=trajectory)

    for said, made, right in (
        ("Whose password?", [], True),
        ("Done.", [], False),
        ("Whose?", ["reset_password"], False),
    ):
        score = await classes.asks_back(case(said, made))
        assert score is not None and score.value is right
    plain = SimpleNamespace(metadata={}, output="x", trajectory=[])
    assert await classes.asks_back(plain) is None
    assert classes.asks_back.scores_paused  # type: ignore[attr-defined]


async def test_the_notes_are_added_for_the_benchmarks_user() -> None:
    added: list[Any] = []

    async def add_document(file: Any, **kwargs: Any) -> None:
        added.append((file, kwargs))

    await classes.seed(SimpleNamespace(add_document=add_document), "u")
    [(file, kwargs)] = added
    assert file[0] == "store-notes.md" and b"Priya Rao" in file[1]
    assert kwargs == {"user": "u", "title": classes.NOTES_TITLE}


# --------------------------------------------------------------------------- the runner
async def test_a_fault_fails_its_tools_first_call_only() -> None:
    faults = run.Faults()
    task = {"metadata": {classes.INJECT: {"tool": "look", "kind": "error"}}}
    faults.arm(task)
    call = ToolCall(tool="look", args={"x": 1})
    ok = ToolOutcome(tool="look", output="fine")
    other = await faults.after_tool(ToolCall(tool="else", args={}), ok)
    assert other is ok
    failed = await faults.after_tool(call, ok)
    assert failed.status is ToolStatus.ERROR and failed.error_class == "error"
    assert (await faults.after_tool(call, ok)) is ok  # the next call passes
    faults.arm({"metadata": {}})
    assert (await faults.after_tool(call, ok)) is ok


def test_wrong_calls_are_calls_the_task_does_not_need() -> None:
    needs = {"metadata": {"needed_tools": ["a"], "avoid_tools": []}}
    assert run.wrong_calls(needs, ["a", "a", "b", "memory_search", "read_result"]) == 1
    asks = {"metadata": {"needed_tools": [], "avoid_tools": ["reset_password"]}}
    assert run.wrong_calls(asks, ["memory_search", "reset_password"]) == 1


def test_a_report_names_the_trellis_it_measured() -> None:
    found = run.source()
    assert Path(found["path"]).name == "harness"
    assert found["commit"] is None or len(found["commit"]) == 40


# --------------------------------------------------------------------------- the arms
def _setup() -> Setup:
    return Setup(
        system="s", model=ScriptedChatModel(turns=[]), window=16_384, max_steps=6, tools=()
    )


@pytest.mark.parametrize(
    ("variant", "repeats"),
    [("all", True), ("limitsA", False), ("limitsB", True), ("errorsA", False)],
)
def test_a_loop_arm_has_the_pieces_it_names(variant: str, repeats: bool) -> None:
    graph, _ = VARIANTS[variant].build(_setup())
    nodes = set(graph.nodes)
    assert ("StallGuard.after_model" in nodes) is repeats
    assert "ModelCallLimitMiddleware.before_model" in nodes


def test_a_loop_arm_is_refused_on_a_source_without_the_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trellis.harness import react

    monkeypatch.delattr(react, "limits")
    with pytest.raises(RuntimeError, match="today's ReAct is the main arm"):
        VARIANTS["limitsA"].build(_setup())


def test_the_helper_arms_give_react_a_sub_agent_admitted_or_not() -> None:
    seen: list[dict[str, Any]] = []
    real = variants.main

    def spy(setup: Setup, **kwargs: Any) -> Any:
        seen.append(kwargs)
        return real(setup, **kwargs)

    tools = tuple(catalog.task_sets()[0].tools)
    setup = Setup(
        system="s", model=ScriptedChatModel(turns=[]), window=None, max_steps=4, tools=tools
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(variants, "main", spy)
        variants.helpers(setup)
        variants.helpers(setup, admitted=True)
    [plain, admitted] = seen
    [researcher] = plain["subagents"]
    assert researcher["name"] == "researcher" and len(researcher["tools"]) == len(tools)
    assert plain["middleware"] == () and isinstance(
        admitted["middleware"][0], variants.AdmittedTasks
    )


def test_task_only_holds_delegate_back_by_its_middlewares_name() -> None:
    assert variants.HELPERS == {"delegate": "Delegation"}
    graph, _ = VARIANTS["task-only"].build(_setup())
    assert graph is not None


def test_the_native_context_arm_replaces_fit_window_and_read_result() -> None:
    graph, _ = VARIANTS["context-native"].build(_setup())
    nodes = set(graph.nodes)
    assert "SummarizationMiddleware.before_model" in nodes
    assert not any(n.startswith(("FitWindow", "ReadTools")) for n in nodes)
    with pytest.raises(RuntimeError, match="window"):
        VARIANTS["context-native"].build(
            Setup(system="s", model=ScriptedChatModel(turns=[]), window=None, max_steps=4)
        )


# --------------------------------------------------------------------------- the verdict
def _rows(
    variant: str, by_class: dict[str, list[tuple[bool, int]]], repeats: int = 3
) -> list[dict[str, Any]]:
    """Runs of ``variant``: per class, per task, (success, tokens) the same in every repeat."""
    rows = []
    for kind, tasks in by_class.items():
        for n, (ok, tokens) in enumerate(tasks):
            for repeat in range(repeats):
                rows.append(
                    {
                        "variant": variant,
                        "size": 10,
                        "repeat": repeat,
                        "task": f"{kind} {n}",
                        "length": kind,
                        "success": ok,
                        "tool_coverage": 1.0 if ok else 0.0,
                        "wrong_calls": 0,
                        "errored": False,
                        "timed_out": False,
                        "model_calls": 2,
                        "failed_model_calls": 0,
                        "input_tokens": tokens,
                        "output_tokens": 10,
                        "cached_tokens": None,
                        "seconds": tokens / 100,
                        "cpu_seconds": {"harness": 0.5},
                    }
                )
    return rows


def _arms(base: list[Any], candidate: list[Any]) -> dict[str, Any]:
    report = {"processes": {"harness": {"peak_rss_mb": 300.0}}}
    for rows in (base, candidate):
        analyze.priced(rows, "claude-sonnet-5-5")
    return {"main": (report, base), "new": (report, candidate)}


SAME = [(True, 1000)] * 10 + [(False, 1000)] * 10


def test_a_change_better_on_its_target_and_worse_nowhere_is_kept() -> None:
    base = _rows("main", {"error": [(n % 2 == 0, 1000) for n in range(20)], "simple": SAME})
    candidate = _rows("new", {"error": [(True, 1000)] * 20, "simple": SAME})
    found = analyze.compare("new", "main", ["error"], _arms(base, candidate), {})
    assert found.decision == "KEEP", found.reasons
    error = next(c for c in found.classes if c.name == "error")
    assert error.deltas["success"].delta == pytest.approx(0.5) and error.deltas["success"].better


def test_a_change_worse_on_any_class_is_reverted() -> None:
    base = _rows("main", {"error": SAME, "simple": [(True, 1000)] * 20})
    candidate = _rows("new", {"error": [(True, 1000)] * 20, "simple": [(True, 1500)] * 20})
    found = analyze.compare("new", "main", ["error"], _arms(base, candidate), {})
    assert found.decision == "REVERT"
    assert "simple: tokens worse" in found.reasons


def test_arms_within_the_interval_are_a_tie_with_the_classes_to_repeat() -> None:
    base = _rows("main", {"wide": SAME, "simple": SAME})
    candidate = _rows("new", {"wide": SAME, "simple": SAME})
    found = analyze.compare("new", "main", ["wide"], _arms(base, candidate), {})
    assert found.decision == "TIE" and set(found.more_repeats) == {"wide", "simple"}


def test_a_gain_short_of_the_keep_rule_is_reverted() -> None:
    # tokens down 5% for certain: measurably better, short of the 10% the rule asks
    base = _rows("main", {"wide": [(True, 1000)] * 20})
    candidate = _rows("new", {"wide": [(True, 950)] * 20})
    found = analyze.compare("new", "main", ["wide"], _arms(base, candidate), {})
    assert found.decision == "REVERT"
    assert found.reasons == ["wide (target): no gain reaches the keep rule"]


def test_latency_is_the_serial_passes_when_given() -> None:
    base = _rows("main", {"simple": SAME})
    candidate = _rows("new", {"simple": SAME})
    timed = {
        "main": _rows("main", {"simple": [(True, 1000)] * 20}),
        "new": _rows("new", {"simple": [(True, 500)] * 20}),
    }
    found = analyze.compare("new", "main", [], _arms(base, candidate), timed)
    p50 = found.classes[0].deltas["p50"]
    assert p50.delta == pytest.approx(-0.5) and p50.note is None and p50.better
    concurrent = analyze.compare("new", "main", [], _arms(base, candidate), {})
    assert concurrent.classes[0].deltas["p50"].note == "from concurrent runs: not latency"


def test_the_command_writes_the_verdicts(tmp_path: Path) -> None:
    rows = [
        *_rows("main", {"error": SAME}),
        *_rows("all", {"error": [(True, 800)] * 20}),
    ]
    report = tmp_path / "r.json"
    report.write_text(json.dumps({"runs": rows, "processes": {}}))
    out = tmp_path / "verdict"
    argv = ["--arm", f"main={report}", "--arm", f"all={report}", "--compare", "all:main:error"]
    assert analyze.main([*argv, "--out", str(out)]) == 0
    [found] = json.loads(out.with_suffix(".json").read_text())
    assert found["decision"] == "KEEP"
    assert "## all against main: **KEEP**" in out.with_suffix(".md").read_text()
    with pytest.raises(SystemExit):
        analyze.main(["--arm", f"main={report}", "--compare", "x:main:error"])
    with pytest.raises(SystemExit):
        analyze.arguments(["--arm", "nolabel", "--compare", "a:b"])
