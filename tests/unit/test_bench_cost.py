"""The cost benchmark (``benchmark/cost``) without a model: its catalogues are what it says
(the sizes, realistic definitions the check of a tool set passes), its statistics and prices
are right, and a whole cell runs through the harness with a scripted model — every run's
tokens, calls, time, coverage and success recorded, and the report written."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jsonschema
import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult, LLMResult

from benchmark.cost import __main__ as cli
from benchmark.cost import catalog, prices, run, stats
from benchmark.cost.variants import VARIANTS
from tests.support.chat_model import ScriptedChatModel
from trellis import Settings
from trellis.harness.tools.lint import lint

#: What the realistic definitions must never be (the set's size, and two list tools of one
#: server sharing most words, are what a large catalogue is).
DEFECTS = {"invalid_name", "no_description", "short_description", "undescribed_parameter"}


# --------------------------------------------------------------------------- catalogues
def test_there_are_500_distinct_distractors_in_a_fixed_order() -> None:
    specs = catalog.distractor_specs()
    assert len(specs) == 500 == len({s.name for s in specs})
    assert [s.name for s in specs] == [s.name for s in catalog.distractor_specs()]
    assert {s.server for s in specs} == {s.name for s in catalog.SERVERS}
    assert {s.side_effects for s in specs} == {"read", "write", "irreversible"}


def test_every_distractor_is_a_valid_described_mcp_style_definition() -> None:
    specs = catalog.distractor_specs()
    for spec in specs:
        assert spec.name.startswith(f"{spec.server}-")
        assert spec.input_schema is not None
        jsonschema.Draft202012Validator.check_schema(spec.input_schema)
    assert not [f for f in lint(specs) if f.code in DEFECTS]


@pytest.mark.parametrize("size", catalog.SIZES)
async def test_each_catalogue_is_its_size_and_holds_the_sets_own_tools(size: int) -> None:
    for taskset in catalog.task_sets():
        tools = catalog.catalogue(taskset.tools, size)
        resolved = [t for source in tools for t in await source.resolve()]
        assert len(resolved) == max(size, len(taskset.tools))
        names = {t.spec.name for t in resolved}
        for task in taskset.tasks:
            assert set(task["metadata"]["needed_tools"]) <= names


async def test_a_distractor_answers_a_small_result() -> None:
    tools = {t.spec.name: t for t in await catalog.Distractors(500).resolve()}
    listed, got = tools["crm-list_deals"], tools["crm-get_deal"]
    assert await listed.run({}) == {"items": [], "next_cursor": None}
    found = await got.run({"deal_id": "DL-1", "other": 2})
    assert found == {"ok": True, "tool": "crm-get_deal", "deal_id": "DL-1"}


def test_the_export_is_past_what_react_pages_and_oldest_first() -> None:
    export = catalog.ticket_export.fn("network")
    assert len(export) > 80_000
    assert export.splitlines()[1].startswith(catalog.OLDEST + ",")


def test_ten_tasks_in_three_sets_each_named() -> None:
    sets = catalog.task_sets()
    tasks = [t for s in sets for t in s.tasks]
    assert [s.name for s in sets] == ["store", "help desk", "travel"]
    assert len(tasks) == 10 == len({t["metadata"]["task"] for t in tasks})
    lengths = [t["metadata"]["length"] for t in tasks]
    assert (lengths.count("short"), lengths.count("long"), lengths.count("large")) == (6, 3, 1)


def test_the_class_suite_has_ten_distinct_tasks_per_class_on_their_own_tools() -> None:
    sets = catalog.class_sets()
    tasks = [t for s in sets for t in s.tasks]
    kinds = [t["metadata"]["length"] for t in tasks]
    assert {k: kinds.count(k) for k in catalog.CLASSES} == dict.fromkeys(catalog.CLASSES, 10)
    assert len({t["metadata"]["task"] for t in tasks}) == 40
    assert sum(bool(t["metadata"]["mix"]) for t in tasks) == 10
    for taskset in sets:
        own = {t.spec.name for t in taskset.tools}
        assert all(set(t["metadata"]["needed_tools"]) <= own for t in taskset.tasks)
    large = [t for t in tasks if t["metadata"]["length"] == "large"]
    assert {t["expected"] for t in large} >= {"NET-0001", "DES-1100"}


async def test_the_command_picks_a_suites_classes_and_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}

    async def cell(h: Any, variant: str, size: int, **kwargs: Any) -> list[Any]:
        seen["tasks"] = [t["metadata"]["task"] for s in kwargs["sets"] for t in s.tasks]
        seen["memory"] = h.memory is not None
        return []

    monkeypatch.setattr(run.Settings, "from_env", classmethod(lambda cls: Settings()))
    monkeypatch.setattr(cli.run, "run_cell", cell)
    out = str(tmp_path / "b")
    common = ["--model", "m", "--sizes", "50", "--variants", "main", "--out", out]
    await cli.bench(cli.arguments([*common, "--suite", "classes", "--classes", "mix"]))
    assert len(seen["tasks"]) == 10 and seen["memory"] is False
    await cli.bench(cli.arguments([*common, "--suite", "classes", "--classes", "large"]))
    assert len(seen["tasks"]) == 10 and all("export" in t for t in seen["tasks"])
    await cli.bench(
        cli.arguments(
            [*common, "--suite", "react", "--classes", "reasoning,simple", "--split", "test"]
        )
    )
    assert len(seen["tasks"]) == 40 and not any("maths 0" in t for t in seen["tasks"])  # dev
    await cli.bench(
        cli.arguments([*common, "--suite", "react", "--classes", "simple,wide", "--per-class", "5"])
    )
    assert len(seen["tasks"]) == 10


# --------------------------------------------------------------------------- statistics
def test_the_interval_is_students_t() -> None:
    assert stats.interval([]) == {"mean": None, "ci95": None, "n": 0}
    assert stats.interval([4.0]) == {"mean": 4.0, "ci95": None, "n": 1}
    found = stats.interval([1.0, 2.0, 3.0])
    assert found["mean"] == 2.0 and found["n"] == 3
    assert found["ci95"] == pytest.approx(4.303 * 1.0 / 3**0.5)
    assert stats.t975(30) == 2.042
    assert stats.t975(120) == pytest.approx(1.980, abs=0.003)


def test_a_cell_is_shown_as_mean_and_half_width() -> None:
    assert stats.shown(None) == "—"
    assert stats.shown({"mean": None}) == "—"
    assert stats.shown({"mean": 1234.5, "ci95": None}) == "1,234"
    assert stats.shown({"mean": 0.5, "ci95": 0.25}, 2) == "0.50 ± 0.25"


# --------------------------------------------------------------------------- prices
def test_prices_are_per_million_tokens_with_cached_input_apart() -> None:
    price = prices.Price(input=2.0, cached_input=0.2, output=10.0, cache_write=2.5, source="x")
    assert prices.dollars(price, input_tokens=1_000_000, output_tokens=0) == 2.0
    found = prices.dollars(
        price, input_tokens=1_000_000, output_tokens=100_000, cached=500_000, written=100_000
    )
    assert found == pytest.approx((400_000 * 2.0 + 500_000 * 0.2 + 100_000 * 2.5) / 1e6 + 1.0)
    no_write = prices.Price(1.0, 0.1, 1.0, None, "x")
    assert prices.dollars(no_write, input_tokens=10, output_tokens=0, written=10) == 1e-5


def test_a_model_is_priced_by_its_gateway_names_last_part() -> None:
    assert prices.price_of("anthropic/claude-sonnet-5-5") is prices.PRICES["claude-sonnet-5-5"]
    assert prices.price_of("openrouter/openai/gpt-4.1-nano") is prices.PRICES["gpt-4.1-nano"]
    assert prices.price_of("somewhere/unknown-model") is None
    assert all(p.source for p in prices.PRICES.values())
    assert set(prices.REFERENCE) <= set(prices.PRICES)


# --------------------------------------------------------------------------- the meter
def test_the_meter_counts_every_call_its_tokens_and_failures() -> None:
    meter = run.Meter()
    usage = {
        "input_tokens": 100,
        "output_tokens": 7,
        "total_tokens": 107,
        "input_token_details": {"cache_read": 60, "cache_creation": 10},
    }
    reply = AIMessage(content="hi", usage_metadata=usage)  # type: ignore[arg-type]
    meter.on_llm_end(LLMResult(generations=[[ChatGeneration(message=reply)]]))
    meter.on_llm_end(LLMResult(generations=[[ChatGeneration(message=AIMessage(content=""))]]))
    meter.on_llm_error(RuntimeError("context exceeded"))
    meter.on_llm_error(
        RuntimeError("request (71121 tokens) exceeds the available context size (8192 tokens)")
    )
    assert (meter.calls, meter.failed, meter.refused_tokens) == (4, 2, 71121)
    assert (meter.input_tokens, meter.output_tokens) == (100, 7)
    assert (meter.cached_tokens, meter.written_tokens, meter.cache_reported) == (60, 10, True)


# --------------------------------------------------------------------------- a whole cell
class Counted(ScriptedChatModel):
    """The script, each reply reporting its usage as a provider does."""

    def _generate(
        self, messages: Any, stop: Any = None, run_manager: Any = None, **kw: Any
    ) -> ChatResult:
        result = super()._generate(messages, stop, run_manager, **kw)
        message = result.generations[0].message
        assert isinstance(message, AIMessage)
        message.usage_metadata = {
            "input_tokens": 50 * len(messages),
            "output_tokens": 5,
            "total_tokens": 50 * len(messages) + 5,
        }
        return result


#: The travel set, scripted: its three tasks — two parts, a read, four parts.
TRAVEL_SCRIPT: list[Any] = [
    [
        ("flight_bookings", {"employee": "e-401"}),
        ("expense_claims", {"employee": "e-401", "status": "pending"}),
    ],
    "Flight QX7P2 on 2026-11-03; claim X-55 is pending.",
    "Meals: 45 EUR a day.",  # the read that sounds like a change, answered without a tool
    [
        ("trip_summary", {"employee": "e-401"}),
        ("flight_bookings", {"employee": "e-401"}),
        ("expense_claims", {"employee": "e-401", "status": "pending"}),
        ("travel_policy", {"topic": "meal allowance"}),
    ],
    "6 trips; QX7P2; X-55 pending; meals up to 45 EUR a day.",
]


#: The arms that need a service a unit test has not: a sandbox, the memory service, Anthropic's
#: client (``test_bench_reasoning`` covers them).
UNSCRIPTED = {"code", "examples", "anthropic-cache"}


@pytest.mark.parametrize("variant", [v for v in VARIANTS if v not in UNSCRIPTED])
async def test_a_cell_runs_every_task_and_records_each_run(
    variant: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(run.Settings, "from_env", classmethod(lambda cls: Settings()))
    travel = [s for s in catalog.task_sets() if s.name == "travel"]
    model = Counted(turns=list(TRAVEL_SCRIPT))
    setup = {"model": model, "window": 8192, "max_steps": 4}
    seen: list[str] = []
    async with run.harness() as h:
        rows = await run.run_cell(
            h,
            variant,
            10,
            setup=setup,
            repeats=1,
            sets=travel,
            run_seconds=30,
            progress=seen.append,
        )
    assert [r["task"] for r in rows] == [t["metadata"]["task"] for t in travel[0].tasks]
    assert len(seen) == 3
    first, second, third = rows
    assert [r["length"] for r in rows] == ["short", "short", "long"]
    assert first["status"] == "success" and first["tool_coverage"] == 1.0 and first["success"]
    assert sorted(first["called"]) == ["expense_claims", "flight_bookings"]  # read at once
    assert first["model_calls"] == 2 and first["input_tokens"] > 0 and first["output_tokens"] == 10
    assert first["cached_tokens"] is None  # the script reports no cache
    # the policy question answered without its tool: no coverage, no success
    assert second["tool_coverage"] == 0.0 and not second["success"] and not second["errored"]
    assert third["tool_coverage"] == 1.0 and third["success"] and len(third["called"]) == 4
    # what the model was offered: the set's 5 tools and 5 distractors (and, in ReAct, its
    # own read tools once they have something to read)
    offered = {name for call in model.requests for name in call.get("tools", [])}
    assert len(offered & {t.spec.name for t in travel[0].tools}) == 5
    assert len([n for n in offered if "-" in n]) == 5

    report = run.report_of(
        rows,
        model="granite/granite-3.3-2b",
        sizes=[10],
        variants=[variant],
        repeats=1,
        sets=travel,
        setup=setup,
        priced_as=["claude-sonnet-5-5"],
        notes=["a note"],
    )
    cell = report["cells"][0]
    assert cell["stats"]["success"]["mean"] == pytest.approx(2 / 3)
    assert cell["stats"]["cached_tokens"]["n"] == 0
    assert set(cell["dollars"]) == {"granite-3.3-2b", "claude-sonnet-5-5"}
    assert cell["dollars"]["granite-3.3-2b"]["mean"] == 0.0
    assert cell["dollars"]["claude-sonnet-5-5"]["mean"] > 0
    assert "model" not in report["setup"]
    assert report["environment"]["vendor_keys"].keys() == {"ANTHROPIC_API_KEY", "OPENAI_API_KEY"}
    text = stats.markdown(report)
    assert "| 10 | " + variant in text and "a note" in text and "as claude-sonnet-5-5" in text
    assert f"| 10 | {variant} | long |" in text
    assert cell["by_length"]["long"]["stats"]["success"]["mean"] == 1.0
    json.dumps(report, default=str)


def test_an_errored_run_is_scored_from_the_calls_it_made() -> None:
    result = SimpleNamespace(
        status="error",
        scores=[],
        error="the request exceeds the available context size",
        output=None,
    )
    task = {"metadata": {"task": "t", "needed_tools": ["a", "b"]}}
    row = run.row_of(result, task, run.Meter(), ["a"], 1.0)
    assert row["tool_coverage"] == 0.5 and row["errored"] and not row["success"]
    assert run.coverage_of([], []) is None


# --------------------------------------------------------------------------- the command
def test_the_command_needs_a_model_and_known_variants_and_prices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TRELLIS_BENCH_MODEL", raising=False)
    monkeypatch.delenv("TRELLIS_LIVE_MODEL", raising=False)
    for argv in (
        [],
        ["--model", "m", "--variants", "nope"],
        ["--model", "m", "--price-as", "nope"],
        ["--model", "m", "--watch", "server"],
        ["--model", "m", "--watch", "=12"],
    ):
        with pytest.raises(SystemExit):
            cli.arguments(argv)
    monkeypatch.setenv("TRELLIS_LIVE_MODEL", "granite/granite-3.3-2b")
    assert cli.arguments([]).model == "granite/granite-3.3-2b"
    assert cli.arguments(["--watch", "server=12"]).watch == {"server": 12}


async def test_the_command_writes_the_json_and_the_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(run.Settings, "from_env", classmethod(lambda cls: Settings()))
    monkeypatch.setattr(cli.run, "run_cell", _one_row)
    out = tmp_path / "bench"
    args = cli.arguments(
        ["--model", "m", "--sizes", "10", "--sets", "travel", "--out", str(out), "--watch", "me=1"]
    )
    report = await cli.bench(args)
    assert len(report["cells"]) == 2  # both variants
    assert json.loads(out.with_suffix(".json").read_text())["model"] == "m"
    assert "| 10 | baseline |" in out.with_suffix(".md").read_text()
    assert report["cells"][0]["dollars"].keys() == set(prices.REFERENCE)  # m is not priced
    assert (
        report["setup"]["window_source"] == "unknown"
    )  # nothing said what m's window is: its features off
    assert "not priced" in stats.markdown(report)
    assert set(report["processes"]) == {"harness", "me"}
    table = stats.markdown(report)
    assert "| 10 | main | 2 | 0.20 | 0.29 | 0.50 | — |" in table
    assert "* harness: " in table and "* me: " in table


async def _one_row(h: Any, variant: str, size: int, **kwargs: Any) -> list[Any]:
    row = {
        "size": size,
        "variant": variant,
        "set": "travel",
        "repeat": 0,
        "task": "t",
        "input_tokens": 10,
        "output_tokens": 1,
        "cached_tokens": None,
        "cache_written_tokens": 0,
        "model_calls": 1,
        "seconds": 0.1,
        "tool_coverage": 1.0,
        "success": True,
        "errored": False,
        "length": "short",
        "model_call_seconds": [0.1, 0.3],
        "cpu_seconds": {"harness": 0.5, "me": None},
    }
    if run.key(row) in kwargs["done"]:
        return []
    kwargs["on_row"](row)
    return [row]


@pytest.mark.parametrize(("variant", "paged"), [("deep", True), ("deep-paging", False)])
async def test_a_main_piece_is_measured_on_its_own(
    variant: str, paged: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(run.Settings, "from_env", classmethod(lambda cls: Settings()))
    desk = next(s for s in catalog.task_sets() if s.name == "help desk")
    export = [t for t in desk.tasks if t["metadata"]["needed_tools"] == ["ticket_export"]]
    only = catalog.TaskSet(desk.name, desk.system, desk.tools, export)
    model = Counted(
        turns=[("ticket_export", {"team": "network"}), f"The oldest: {catalog.OLDEST}."]
    )
    setup = {"model": model, "window": 128_000, "max_steps": 4}
    async with run.harness() as h:
        [row] = await run.run_cell(
            h, variant, 10, setup=setup, repeats=1, sets=[only], run_seconds=30, progress=str
        )
    assert row["success"] and row["checks"] == {"contains": True}
    result = model.seen[-1][-1]  # what the model read of the export
    assert (len(str(result.content)) < 80_000) is paged


def test_only_the_measured_models_own_price_counts_its_cache() -> None:
    row = {"input_tokens": 1_000_000, "output_tokens": 0, "cached_tokens": 900_000}
    row |= {"cache_written_tokens": 0, "size": 10, "variant": "main"}
    found = run.summary([row], "anthropic/claude-sonnet-5-5", ["gpt-6-sol"])["dollars"]
    assert found["claude-sonnet-5-5"]["mean"] == pytest.approx(0.1 * 2.0 + 0.9 * 0.10)
    assert found["gpt-6-sol"]["mean"] == pytest.approx(2.0)  # every input token in full


async def test_a_resumed_run_keeps_the_runs_measured_and_runs_only_the_others(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(run.Settings, "from_env", classmethod(lambda cls: Settings()))
    monkeypatch.setattr(cli.run, "run_cell", _one_row)
    out = tmp_path / "bench"
    argv = ["--model", "m", "--sizes", "10", "--variants", "main", "--out", str(out)]
    first = await cli.bench(cli.arguments(argv))
    assert len(first["runs"]) == 1
    again = await cli.bench(cli.arguments([*argv, "--sizes", "50", "--resume"]))
    assert [(r["size"], r["variant"]) for r in again["runs"]] == [(10, "main"), (50, "main")]
    assert [(c["size"], c["runs"]) for c in again["cells"]] == [(10, 1), (50, 1)]
    other = await cli.bench(cli.arguments([*argv, "--model", "n", "--resume"]))
    assert len(other["runs"]) == 1  # another model's runs are not kept


def test_a_paused_run_succeeds_only_when_its_task_asks_for_a_change() -> None:
    paused = {"status": "interrupted", "tool_coverage": 1.0, "coverage_comment": None}
    paused |= {"checks": {}}
    assert run.succeeded(paused, approval=True)
    assert not run.succeeded(paused, approval=False)  # it paused at a change nobody asked for
    assert not run.succeeded(paused | {"checks": {"contains": False}}, approval=True)
    broader = paused | {"coverage_comment": "x called though not needed"}
    assert not run.succeeded(broader | {"status": "success"}, approval=False)
    assert run.succeeded(paused | {"status": "success", "tool_coverage": None}, approval=False)


def test_the_meter_times_each_call_from_its_start() -> None:
    meter = run.Meter()
    meter.on_chat_model_start({}, [], run_id="a")
    meter.on_chat_model_start({}, [], run_id="b")
    meter.on_llm_end(LLMResult(generations=[]), run_id="a")
    meter.on_llm_error(RuntimeError("x"), run_id="b")
    meter.on_llm_end(LLMResult(generations=[]), run_id="never started")
    assert len(meter.seconds) == 2 and all(s >= 0 for s in meter.seconds)


def test_a_process_is_measured_from_proc() -> None:
    from benchmark import procstat

    used = procstat.cpu_seconds()
    assert used is not None and used > 0
    assert (procstat.rss_mb() or 0) > 0 and (procstat.peak_rss_mb() or 0) >= (
        procstat.rss_mb() or 0
    )
    assert procstat.cpu_seconds(2**22 + 1) is None and procstat.rss_mb(2**22 + 1) is None
    assert procstat._status(None, "NoSuchKey") is None
    assert procstat.percentile([], 50) is None and procstat.percentile([3.0], 95) == 3.0
    assert procstat.percentile([1.0, 2.0, 3.0, 4.0, 5.0], 50) == 3.0
    assert run._used({"me": None}, {"me": None}) == {"me": None}


@pytest.mark.parametrize(
    ("answer", "store"),
    [
        # the fragile case: the right conclusion after the other store's figures
        ("S12 has 3 weeks of cover, S40 0.2: S40 first.", "S40"),
        ("Store S12 should get stock of ST-101 first, as it has a lower ratio than S40.", "S12"),
        ("S40 runs out sooner than S12, so it should get stock first.", "S40"),
        ("First, the figures: S12 has 3 weeks and S40 0.2, so S40 should get it.", "S40"),
        ("S12: 3 weeks of cover. S40: 0.2 weeks.", "S40"),  # no conclusion: the last named
        ("No store data.", None),
    ],
)
def test_picks_scores_the_store_the_answer_concludes_on(answer: str, store: str | None) -> None:
    """S6: ``picks`` scored the first store an answer named, so a right answer giving the other
    store's figures first failed it."""
    assert run.scenario.concluded(answer) == store


async def test_picks_passes_a_right_answer_that_names_the_other_store_first() -> None:
    from trellis.harness.evals import EvalCase

    answer = "S12 has 3 weeks of cover, S40 0.2: S40 first."
    scored = await run.scenario.picks(EvalCase(input="which first?", output=answer, expected="S40"))
    assert scored is not None and scored.value is True
