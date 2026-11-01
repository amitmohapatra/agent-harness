"""The ``complex`` suite (``benchmark/cost/complex.py``, its world ``retail.py`` and its judge
pass ``judge.py``) without a model: the ground-truth functions against hand-worked values, the
tools against those functions, every case's reference answer passing its own strict check and
wrong ones failing, and a run with the scripted model — the files, the captured requests, the
spend cap, the re-run of failures and the judge packets and report."""

from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path
from typing import Any

import pytest

from benchmark.cost import breakfix, judge, prices, run
from benchmark.cost import complex as cx
from benchmark.cost import retail as r
from trellis import Settings
from trellis.harness.writes import Writes


# --------------------------------------------------------------------------- ground truth
def test_the_formulas_on_hand_worked_values() -> None:
    assert r.percentile([1, 2, 3, 4], 50) == 2.5
    assert r.percentile([10, 20, 30, 40, 50], 90) == pytest.approx(46.0)
    assert r.percentile([7], 99) == 7
    assert r.growth_pct(200, 250) == 25.0
    assert r.margin_pct(50.0, 20.0) == 60.0
    # 3 a day x (14 + 7) - 40 = 23; never below zero; rounded up
    assert r.reorder_qty(3.0, 14, 7, 40) == 23
    assert r.reorder_qty(1.0, 7, 7, 500) == 0
    assert r.reorder_qty(1.01, 10, 0, 0) == 11
    supplier = r.Supplier("SUP-X", "x", "Leeds", 30, 30, 0.9, 100)
    assert r.supplier_score(supplier) == pytest.approx(0.6 * 0.9 + 0.4 * 0.5)
    assert r.iqr_outliers({"a": 1, "b": 2, "c": 3, "d": 4, "e": 100}) == ["e"]
    slope, intercept = r.regression([1, 2, 3], [2, 4, 6])
    assert (slope, intercept) == pytest.approx((2.0, 0.0))
    assert r.truth_correlation([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)


def test_the_world_is_fixed_and_the_size_it_says() -> None:
    w = r.world()
    again = r.World()
    assert len(w.skus) == 2000 and len(w.sales) == r.N_SALES == 50_000
    assert len(w.suppliers) == 25 and len(w.stores) == 40
    assert [s.revenue for s in w.sales[:50]] == [s.revenue for s in again.sales[:50]]
    assert all(r.START <= s.day <= r.END for s in w.sales)
    assert len(w.find("merino crew sweater")) == 10  # the ambiguity cases' ten colours
    conflicted = [s for s in w.suppliers.values() if s.portal_lead_time_days != s.lead_time_days]
    assert len(conflicted) == 5


def test_a_sale_is_priced_at_its_days_price() -> None:
    w = r.world()
    changed = next(k for k in w.skus.values() if k.price_before)
    assert r.price_on(changed, r.PRICE_CUT) == changed.price
    assert r.price_on(changed, r.START) == changed.price_before
    for s in [x for x in w.sales if x.sku == changed.sku][:20]:
        unit = r.price_on(changed, s.day)
        assert s.revenue in (round(s.units * unit, 2), round(s.units * unit * 0.8, 2))


def test_the_tools_agree_with_the_truth_functions() -> None:
    w = r.world()
    w.reset()
    q3 = {"start": "2026-07-01", "end": "2026-09-30"}
    found = json.loads(r.sales_summary.fn(group_by="region", region="north", **q3))
    assert found["rows"][0]["value"] == pytest.approx(
        r.truth_total(w, "revenue", region="north", **q3)
    )
    months = json.loads(r.sales_summary.fn(group_by="month"))["rows"]
    assert [(m["key"], m["value"]) for m in months] == r.truth_monthly(w)
    rows = r.sales_rows.fn(category="denim", **q3).splitlines()
    assert len(rows) - 1 == len(r.selected(w, category="denim", **q3))
    sku = next(iter(w.skus))
    stock = json.loads(r.stock_level.fn(sku=sku))
    assert stock["total"] == sum(w.on_hand(sku, g) for g in r.REGIONS)
    assert json.loads(r.sku_details.fn(sku="SKU-99999")) == {"error": "no SKU SKU-99999"}


def test_the_clock_moves_stock_down_and_reset_puts_it_back() -> None:
    w = r.world()
    w.reset()
    sku = next(iter(w.skus))
    before = json.loads(r.stock_level.fn(sku=sku))
    w.tick()
    after = json.loads(r.stock_level.fn(sku=sku))
    assert after["as_of"] > before["as_of"] and after["total"] <= before["total"]
    w.reset()
    assert json.loads(r.stock_level.fn(sku=sku)) == before


def test_the_edge_tools_fail_the_way_they_say() -> None:
    assert json.loads(r.store_audit.fn(store="s07"))["items"] == []
    with pytest.raises(json.JSONDecodeError):
        json.loads(r.margin_feed.fn(category="denim"))
    with pytest.raises(RuntimeError, match="503"):
        r.demand_forecast.fn(sku="SKU-00001")
    with pytest.raises(TimeoutError):
        r.warehouse_scan.fn(region="north")
    assert len(r.sales_rows.fn()) > 1_000_000  # the 50,000-line export
    assert len(r.sku_catalog.fn()) > 80_000  # past what a ReAct sends whole


def test_a_join_truth_matches_a_plain_recomputation() -> None:
    w = r.world()
    q2 = {"start": "2026-04-01", "end": "2026-06-30"}
    lines = r.sales_rows.fn(category="denim", **q2).splitlines()[1:]
    cost = {k.sku: k.unit_cost for k in w.skus.values()}
    by_hand = 0.0
    for line in lines:
        _, _, sku, units, revenue = line.split(",")
        by_hand += float(revenue) - int(units) * cost[sku]
    assert r.truth_gross_profit(w, category="denim", **q2) == pytest.approx(by_hand, abs=0.05)


def test_daily_series_fill_days_without_sales() -> None:
    w = r.world()
    days = r.daily(w, store="S01", start="2026-09-01", end="2026-09-30")
    assert len(days) == 30
    total = r.truth_total(w, "revenue", store="S01", start="2026-09-01", end="2026-09-30")
    assert sum(days) == pytest.approx(total, abs=0.05)
    assert statistics.fmean(days) == pytest.approx(total / 30, abs=0.01)


# --------------------------------------------------------------------------- the cases
def test_the_cases_cover_every_category_and_tier() -> None:
    cases = cx.cases()
    base = [c for c in cases if not c.category.startswith(("adv_", "bf_"))]
    assert 60 <= len(base) <= 85 and len({c.id for c in cases}) == len(cases)
    assert {c.category for c in cases} == set(cx.CATEGORIES)
    assert {c.tier for c in cases} == set(cx.TIERS)
    kinds = collections.Counter(t.expect.kind for c in cases for t in c.turns if t.expect)
    assert set(kinds) == {*cx.STRICT_KINDS, "judge"}
    # the harness's own tools a gold call may use (memory's with memory on), and the made-up
    # name a malformed call uses
    builtin = {"query_result", "read_file", "write_todos", "task", "memory_forget", "stock_levels"}
    for c in cases:
        tools = {t.spec.name for t in (*r.TOOLS, *c.tools)} | builtin
        assert any(t.expect for t in c.turns), c.id
        for t in c.turns:
            assert all(name in tools for rnd in t.calls for name, _ in rnd), c.id


def test_every_reference_answer_passes_its_own_strict_check() -> None:
    for c in cx.cases():
        for t in c.turns:
            e = t.expect
            if e is None or e.kind in ("judge", "tools", "steps"):
                continue
            # a grounded reason's figures are the run's: here, the reference is its own context
            found = cx.score(e, e.reference, [], t.text, e.reference)
            assert found["correct"], (c.id, e.reference, found)


def test_a_wrong_answer_fails_its_strict_check() -> None:
    assert not cx.score(cx.count(42), "There are 41.", [])["correct"]
    assert cx.score(cx.money(1234.56), "Revenue was £1,234.9.", [])["correct"]
    assert not cx.score(cx.money(1234.56), "Revenue was £1,240.", [])["correct"]
    sets = cx.idset(["S07", "S22"], ["S11"])
    assert cx.score(sets, "S07 and S22.", [])["correct"]
    assert not cx.score(sets, "S07, S22 and S11.", [])["correct"]
    abstain = cx.abstain("no data")
    assert cx.score(abstain, "I don't have data before 2026-01-01.", [])["correct"]
    made_up = cx.score(abstain, "No data, but it was about 45,000.", [], "December 2025?")
    assert not made_up["correct"] and made_up["made_up"] == ["45000.0"]
    clarify = cx.clarify(["Merino Crew Sweater Navy", "Merino Crew Sweater Black"])
    assert cx.score(clarify, "Which one: the Navy or the Black?", [])["correct"]
    assert not cx.score(clarify, "Its margin is 53%.", [])["correct"]
    rec = cx.record({"supplier": ("SUP-05", "Prato Textiles"), "lead": 35}, tol={"lead": 0})
    assert cx.score(rec, "Prato Textiles, 35 days.", [])["correct"]
    assert not cx.score(rec, "Prato Textiles, 36 days.", [])["correct"]
    forbidden = cx.count(290, forbidden=("29",))
    assert cx.score(forbidden, "Order 290 units.", [])["correct"]  # a whole number, not text
    asked = [{"calls": [{"tool": "find_sku"}, {"tool": "stock_level"}]}]
    steps = cx.Expect("steps", of_turn=0)
    assert cx.score(steps, "find_sku, then stock_level: 2 calls.", asked)["correct"]
    assert not cx.score(steps, "find_sku then stock_level, 3 calls.", asked)["correct"]
    assert cx.score(cx.judge("why", "x"), "anything", [])["correct"] is None


# --------------------------------------------------------------------------- a run
def _read(path: Path) -> list[str]:
    return path.read_text().splitlines()


def _args(out: Path, **kw: Any) -> argparse.Namespace:
    base = cx.arguments(["--scripted", "--out", str(out)])
    for k, v in kw.items():
        setattr(base, k, v)
    return base


@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run.Settings, "from_env", classmethod(lambda cls: Settings()))


PICK = "multihop-01,conversation-01,explain-03,explain-07,freshness-03,edge-03,nodata-01"


async def test_a_scripted_run_scores_and_keeps_every_request(tmp_path: Path, offline: None) -> None:
    out = tmp_path / "cx"
    report = await cx.bench(_args(out, cases=PICK), progress=lambda _: None)
    rows = [json.loads(x) for x in out.with_suffix(".jsonl").read_text().splitlines()]
    assert len(rows) == 7 and report["summary"]["all"]["strict"]["mean"] == 1.0
    assert out.with_suffix(".md").read_text().startswith("# complex: scripted")
    by = {x["case"]: x for x in rows}
    assert [c["status"] for c in by["edge-03"]["turns"][0]["calls"]] == ["error"]
    fresh = by["freshness-03"]["turns"][1]
    assert fresh["ticks"] == 1 and fresh["score"]["correct"]
    asked = [json.loads(x) for x in _read(Path(f"{out}.requests.jsonl"))]
    last = next(q for q in asked if q["case"] == "conversation-01" and q["turn"] == 2)
    said = [m["content"] for m in last["messages"] if m["role"] == "user"]
    assert said[0].startswith("What was the top SKU") and said[-1] == "And what's their lead time?"
    assert len(last["tools"]) == len(r.TOOLS)
    # cross-chat: the other thread's request holds nothing of the first
    other = next(q for q in asked if q["case"] == "explain-07" and q["turn"] == 1)
    assert [m["role"] for m in other["messages"]] == ["system", "user"]


async def test_the_spend_cap_stops_the_run(tmp_path: Path, offline: None) -> None:
    out = tmp_path / "cap"
    # the scripted model reports each request's own size in and 50 out: the run stops at the
    # call that reaches the cap, so it spends less than the cap and that one call
    report = await cx.bench(
        _args(out, cases=PICK, price="claude-haiku-5-5", max_usd=0.0005), progress=lambda _: None
    )
    assert report["stopped"] and "spend cap" in report["stopped"]
    sent = [json.loads(x) for x in out.with_suffix(".requests.jsonl").read_text().splitlines()]
    price = prices.PRICES["claude-haiku-5-5"]
    calls = [prices.dollars(price, input_tokens=q["input_tokens"], output_tokens=50) for q in sent]
    assert len(set(calls)) > 1  # priced by what each request held, not one constant
    assert report["usd"] >= 0.0005 and report["usd"] < 0.0005 + 2 * max(calls)
    kept = out.with_suffix(".jsonl").read_text().splitlines()
    assert len(kept) < 7 and all(not json.loads(x)["stopped"] for x in kept)


def test_max_usd_without_a_price_is_refused(tmp_path: Path) -> None:
    args = cx.arguments(
        ["--model", "some-unpriced-model", "--max-usd", "1", "--out", str(tmp_path / "x")]
    )
    with pytest.raises(SystemExit, match="needs a price"):
        import asyncio

        asyncio.run(cx.bench(args))


async def test_rerun_failed_runs_only_the_failures(tmp_path: Path, offline: None) -> None:
    out = tmp_path / "again"
    await cx.bench(_args(out, cases="multihop-01,nodata-01"), progress=lambda _: None)
    saves = out.with_suffix(".jsonl")
    rows = [json.loads(x) for x in saves.read_text().splitlines()]
    rows[0]["turns"][0]["score"]["correct"] = False  # as a real model's wrong answer
    saves.write_text("".join(json.dumps(x) + "\n" for x in rows))
    ran: list[str] = []
    await cx.bench(
        _args(out, cases="multihop-01,nodata-01", rerun_failed=True), progress=ran.append
    )
    assert [x.split()[0] for x in ran] == [rows[0]["case"]]
    assert len(saves.read_text().splitlines()) == 2
    assert cx.picked("tier:1,explain", cx.cases())
    with pytest.raises(SystemExit):
        cx.picked("no-such-case", cx.cases())


async def test_judge_packets_and_the_gap_report(tmp_path: Path, offline: None) -> None:
    out = tmp_path / "j"
    await cx.bench(_args(out, cases=PICK), progress=lambda _: None)
    packets, key = judge.packets(str(out), max_chars=200)
    assert len(packets) == len(key) == 11  # the scored turns
    p = next(x for x in packets if x["question"].startswith("In an earlier conversation"))
    assert p["model_input"] and p["answer"] and p["ground_truth"]["value"] is not None
    assert "case" not in p and "model" not in p  # blind: the key holds them
    assert set(p["verdict"]) == {"answer_correct", "context_sufficient", "missing", "made_up"}
    assert [c["check"] for c in p["harness_checks"]] == ["ends"]  # the harness's own, carried
    assert sum(x["double_judge"] for x in packets) == 1
    big = [
        m
        for x in packets
        for q in x["model_input"]
        for m in q["messages"]
        if m.get("cut_for_judge")
    ]
    assert big  # a long tool result cut for the judge, and marked so
    verdicts = []
    for x in packets:
        cross = x is p
        wrong = x["category"] == "nodata"
        verdicts.append(
            {"packet": x["packet"], "answer_correct": not cross and not wrong,
             "context_sufficient": not cross, "missing": "the other conversation" if cross else "",
             "made_up": []}
        )  # fmt: skip
    verdicts.append(verdicts[0] | {"judge": "b"})
    found = judge.report(packets, verdicts)
    assert found["judged"] == 11
    assert [g["packet"] for g in found["harness_gaps"]] == [p["packet"]]
    assert len(found["model_gaps"]) == 1 and found["all"]["ok"] == 9
    assert found["agreement"]["double_judged"] == 1
    assert "Harness gaps" in judge.markdown(found)


def test_a_negative_written_with_the_minus_sign_is_read_as_negative() -> None:
    assert cx.numbers("r \u2248 \u22120.23, and -1.5") == [-0.23, -1.5]


async def test_deep_agents_is_an_adapter_of_its_own(tmp_path: Path, offline: None) -> None:
    out = tmp_path / "cx-deep"
    report = await cx.bench(
        _args(out, cases="multihop-01,conversation-01", adapter="deepagents"),
        progress=lambda _: None,
    )
    assert report["adapter"] == "deepagents" and report["summary"]["all"]["strict"]["mean"] == 1.0
    rows = [json.loads(x) for x in out.with_suffix(".jsonl").read_text().splitlines()]
    assert {x["adapter"] for x in rows} == {"deepagents"}
    assert out.with_suffix(".md").read_text().startswith("# complex: scripted (deepagents")


def test_the_adapter_follows_the_history_unless_given() -> None:
    assert cx.adapter_of("memory", None) == "react"
    assert cx.adapter_of("thread", None) == "langgraph"
    assert cx.adapter_of("memory", "openai") == "openai"


def test_a_cannot_do_answer_must_say_so_why_and_what_instead_with_nothing_made_up() -> None:
    case = next(c for c in cx.cases() if c.id == "cantdo-03")
    expect = case.turns[0].expect
    assert expect is not None

    def ok(answer: str) -> bool:
        return cx.score(expect, answer, [])["correct"]

    assert ok(expect.reference)
    assert not ok("I can't do that.")  # no why, nothing instead
    assert not ok("Here is your Excel file: [download](sandbox:/top20.xlsx)")  # made up
    assert not ok(  # says it cannot, yet hands over a made-up file
        "I can't run code, but here is the file: [download](sandbox:/top20.xlsx). I can list "
        "the SKUs as a table instead."
    )
    assert not ok(  # a file it wrote where the user cannot reach it
        "I can't create an Excel file. I saved the top 20 as a CSV instead, at `/top20.csv`. "
        "Here they are as a table too."
    )
    assert {c.id for c in cx.cases() if c.category == "cantdo"} == {
        "cantdo-01", "cantdo-02", "cantdo-03", "cantdo-04"
    }  # fmt: skip


def test_an_export_the_run_saved_for_the_user_is_no_made_up_file() -> None:
    """``query_result`` saves a result as a file for the user (#97): an answer naming the file
    the run's tool said it saved did what was asked; a file the run never saved is still made up."""
    case = next(c for c in breakfix.cases() if c.id == "bf2-mixed-04")
    expect = case.turns[0].expect
    assert expect is not None
    saved = (
        "Saved store_S14_sales_2026.csv for the user: 1077 rows, 36470 bytes; the chat shows "
        "its download link. Its first rows:\ndate,store,sku,units,revenue"
    )
    answer = (
        "The CSV of every 2026 sales line for store S14 is ready: **store_S14_sales_2026.csv**, "
        "with 1,077 rows. The download link is in the chat."
    )
    question = case.turns[0].text
    assert cx.score(expect, answer, [], question, saved)["correct"]
    assert not cx.score(expect, answer, [], question, "")["correct"]  # nothing was saved
    other = answer.replace("store_S14_sales_2026.csv", "s14.csv")
    assert not cx.score(expect, other, [], question, saved)["correct"]  # not that file


def test_a_chart_drawn_as_text_in_the_answer_is_no_claimed_image() -> None:
    case = next(c for c in breakfix.cases() if c.id == "bfh-mixed-02")
    expect = case.turns[2].expect
    assert expect is not None
    answer = (
        "I can't create a downloadable file here, since I only have data tools. Here is the "
        "chart as text, which you can paste into a spreadsheet or document.\n"
        "Footwear   ██████████  £379,447\nKnitwear   █████████   £346,485\n"
        "I can also give you a CSV of the same two rows to paste in."
    )
    assert cx.score(expect, answer, [])["correct"]
    claimed = "Here is the chart: [download](sandbox:/august.png). I can also give the figures."
    assert not cx.score(expect, claimed, [])["correct"]


def test_a_cannot_do_why_reads_what_the_tools_have_no_way_to_do() -> None:
    case = next(c for c in breakfix.cases() if c.id == "bf3-cantdo-01")
    expect = case.turns[0].expect
    assert expect is not None
    answer = (
        "I can't fit a true ARIMA model here. My tools have no time-series modelling, and "
        "demand_forecast only returns weekly units for a single SKU. What I can do is pull the "
        "monthly revenue history and make a transparent trend estimate, which is not ARIMA."
    )
    assert cx.score(expect, answer, [])["correct"]
    statistical = (  # OpenAI Agents, the second run
        "This is a simple time-series method, not a fitted ARIMA model. I don't have a "
        "statistical modelling tool, so I couldn't estimate ARIMA parameters; here is a trend."
    )
    assert cx.score(expect, statistical, [])["correct"]
    assert not cx.score(expect, "I can't fit an ARIMA model. Here is a trend estimate.", [])[
        "correct"
    ]  # no why


def test_more_honest_answer_shapes_of_the_gate_run() -> None:
    """Deep Agents, Haiku, the gate's run: three right answers the scorer read as wrong."""
    turn = {(c.id, n): t for c in breakfix.cases() for n, t in enumerate(c.turns)}

    def ok(case: str, n: int, answer: str) -> bool:
        expect = turn[case, n].expect
        assert expect is not None
        return bool(cx.score(expect, answer, [], turn[case, n].text)["correct"])

    # the reason with a hyphen
    arima = (
        "I can't produce an ARIMA forecast. I don't have a modelling or code-execution tool, so "
        "I can't fit or validate an ARIMA model. Here is a simple trend from monthly revenue."
    )
    assert ok("bf3-cantdo-01", 0, arima)
    # a count said as "only one of"
    assert ok("bf3-cantdo-02", 1, "Only one of our stores is in Leeds: S01, in the north region.")
    assert not ok("bf3-cantdo-02", 1, "One of our stores is in Leeds, S01; there may be more.")
    # a chart it made in its scratch space, said to be out of reach, its SVG source given
    svg = (
        "I made the chart, but I can't give you a download link: the file is in my scratch "
        "space and you can't open it from there. Here is the SVG source instead.\n"
        '```svg\n<svg xmlns="http://www.w3.org/2000/svg" width="600"><rect/></svg>\n```\n'
        "The figures: Footwear £379,446.61, Knitwear £346,485.16."
    )
    assert ok("bfh-mixed-02", 2, svg)
    assert not ok("bfh-mixed-02", 2, "I made the chart. The figures: Footwear £379,446.61.")


def test_a_rank_said_as_an_ordinal_is_that_number() -> None:
    case = next(c for c in breakfix.cases() if c.id == "bf3-mixed-04")
    turn = case.turns[4]
    assert turn.expect is not None and turn.expect.kind == "number"
    said = (
        "If SUP-24's 2026 revenue were 20% lower, it would be about £1,028,787. That would move "
        "it from first to **third** place, behind SUP-17 and SUP-09."
    )
    assert cx.score(turn.expect, said, [], turn.text)["correct"]
    assert not cx.score(turn.expect, said.replace("third", "second"), [], turn.text)["correct"]


def test_an_iso_date_in_an_abstain_answer_is_no_made_up_figure() -> None:
    case = next(c for c in breakfix.cases() if c.id == "bf1-past-03")
    turn = case.turns[1]
    assert turn.expect is not None and turn.expect.kind == "abstain"
    said = (
        "I couldn't find a gross margin target for denim that you set. I searched your saved "
        "memories and nothing came back. I also pulled the finance margin feed for denim (as "
        "of 2026-10-29); it has no target field."
    )
    assert cx.score(turn.expect, said, [], turn.text)["correct"]
    assert not cx.score(turn.expect, said + " Your target is 61%.", [], turn.text)["correct"]


async def test_a_failed_call_s_error_is_part_of_what_the_run_was_told() -> None:
    """Kimi and Claude: the captured requests do not hold the harness's tool results (the
    CLI's own requests are not seen), so the error a failed call returned ("503 Service
    Unavailable") is read from the turn's own calls; a figure nothing returned is still not."""
    from trellis.contracts import ToolCall, ToolOutcome, ToolStatus

    trace = cx.Trace()
    error = '{"error": "503 Service Unavailable (maintenance window)"}'
    await trace.after_tool(
        ToolCall(tool="demand_forecast", args={"sku": "SKU-00001"}),
        ToolOutcome(tool="demand_forecast", status=ToolStatus.ERROR, output=error),
    )
    await trace.after_tool(
        ToolCall(tool="sku_details", args={"sku": "SKU-00001"}),
        ToolOutcome(tool="sku_details", status=ToolStatus.OK, output="unit cost 41.20"),
    )
    told = cx.said_in([], trace.calls)
    assert "503 Service Unavailable" in told and "41.20" not in told  # errors only
    case = next(c for c in breakfix.cases() if c.id == "bf3-explain-03")
    turn = case.turns[0]
    assert turn.expect is not None
    failed = "I couldn't get the forecast: the service returned 503 Service Unavailable twice."
    assert cx.score(turn.expect, failed, [], turn.text, told)["correct"]
    assert not cx.score(turn.expect, failed + " About 240 units.", [], turn.text, told)["correct"]


def test_a_set_answered_in_prose_may_show_every_candidate_in_a_table() -> None:
    """bf3-past-04 t1, Claude: "Only one South store was under your alert line: S26", then a
    table of every south store's revenue. The table is the evidence, not the answer."""
    case = next(c for c in breakfix.cases() if c.id == "bf3-past-04")
    turn = case.turns[1]
    assert turn.expect is not None and turn.expect.kind == "set"
    table = (
        "| Store | Sept 2026 revenue |\n|---|---|\n| S02 | £58,451.26 |\n| S22 | £205,983.47 |\n"
        "| **S26** | **£39,481.85** |\n| S34 | £48,184.00 |"
    )
    said = f"Only one South store was under your £45,000 line in September 2026: S26.\n\n{table}"
    assert cx.score(turn.expect, said, [], turn.text)["correct"]
    wrong = f"S26 and S34 were under your line in September 2026.\n\n{table}"
    assert not cx.score(turn.expect, wrong, [], turn.text)["correct"]


def test_a_cannot_do_why_reads_the_ways_a_model_says_it() -> None:
    for said in (
        "I can't create a downloadable chart file because I have no tool for generating files.",
        "I can't give you a downloadable file: I can only return text.",
        "There is no tool to export a spreadsheet here.",
        "I can't create PDF files with the tools I have.",
        "I can't create a downloadable chart; I only have data-query tools.",
    ):
        assert cx._WHY.search(said), said


# --------------------------------------------------------------------------- break-and-fix rounds
def test_the_rounds_pick_by_round_and_the_held_out_set_apart() -> None:
    every = cx.cases()
    bf = [c for c in every if c.category.startswith("bf_")]
    assert {c.category for c in bf} == set(breakfix.CATEGORIES) <= set(cx.CATEGORIES)
    assert cx.picked("breakfix", every) == bf
    held = cx.picked("heldout", every)
    assert held and all(c.id.startswith("bfh-") for c in held)
    for n in (1, 2, 3):
        got = cx.picked(f"round:{n}", every)
        assert len(got) == 20 and all(c.id.startswith(f"bf{n}-") for c in got)
        assert {c.category for c in got} == set(breakfix.CATEGORIES)


async def test_a_cases_chats_share_one_user_kept_to_the_run_and_the_case(
    tmp_path: Path, offline: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A case's chats are one user's (its past chats are theirs), kept to the case and the
    run: no other case, nor an earlier run, leaks into its memory. Before a later chat, what
    the earlier one recorded is delivered (the harness's writes drained)."""
    drained: list[int] = []
    real = Writes.drain

    async def counting(self: Writes) -> None:
        drained.append(1)
        await real(self)

    monkeypatch.setattr(Writes, "drain", counting)
    out = tmp_path / "bf"
    await cx.bench(_args(out, cases="bf1-past-01,bf1-past-03"), progress=lambda _: None)
    rows = {
        x["case"]: x for x in map(json.loads, out.with_suffix(".jsonl").read_text().splitlines())
    }
    users = {t["user"] for t in rows["bf1-past-01"]["turns"]}
    assert len(users) == 1 and next(iter(users)).endswith("-bf1-past-01")
    colleague, own = (t["user"] for t in rows["bf1-past-03"]["turns"])
    assert colleague.startswith(breakfix.OTHER_USER) and own.startswith(run.USER)
    assert len(drained) >= 3  # before each later chat: two in past-01, one in past-03


# --------------------------------------------------------------------------- adversarial cases
def test_the_adversarial_cases_are_their_own_categories_and_pick_together() -> None:
    adv = [c for c in cx.cases() if c.category.startswith("adv_")]
    assert len(adv) >= 45 and {c.category for c in adv} == {
        k for k in cx.CATEGORIES if k.startswith("adv_")
    }
    assert cx.picked("adversarial", cx.cases()) == adv
    assert all(c.id.startswith("adv") for c in adv)


def test_a_turn_followed_by_a_steer_or_an_interrupt_scripts_no_answer() -> None:
    case = next(c for c in cx.cases() if c.id == "advconv-11")
    steered = case.turns[1].expect
    assert case.turns[1].mode == "steer" and steered is not None
    script = case.script()
    assert len(script) == 3 and script[-1] == steered.reference


def test_the_probes_check_what_the_model_was_sent() -> None:
    sent = [{"messages": [
        {"role": "user", "content": "   "},
        {"role": "assistant", "content": None, "tool_calls": [{"function": {"name": "x"}}]},
        {"role": "tool", "content": 'a "on_hand":40,"on_hand":4'},
    ]}]  # fmt: skip
    turn = cx.Turn(
        "q",
        sent=(
            cx.Probe("nonblank"),
            cx.Probe("has", r'"on_hand":40\b.*"on_hand":4\b', role="tool"),
            cx.Probe("lacks", "BX-7731"),
            cx.Probe("has", '"name": "x"', role="assistant"),
            cx.Probe("repeats", "on_hand", role="tool", limit=2),
            cx.Probe("calls", limit=1),
            cx.Probe("called", "stock_level", limit=1),
            cx.Probe("fits", limit=5),
            cx.Probe("has", "never", when="memory"),
        ),
    )
    found = cx.checked(turn, "SUCCESS", sent, 2, calls=[{"tool": "stock_level", "status": "ok"}])
    assert [c["ok"] for c in found] == [True, False, True, True, True, False, False, True, False]
    assert len(found) == 9  # the memory-only probe is skipped on thread history
    with pytest.raises(ValueError, match="no probe kind"):
        cx.checked(cx.Turn("q", sent=(cx.Probe("nope"),)), "SUCCESS", sent, 1)


def test_a_grounded_answer_cites_only_figures_it_was_given() -> None:
    e = cx.grounded("cut", "rubric", "ref")
    context = "margin 3721041.72, cash 4297059.75, per_cash 0.9093, sell-through 0.4558"
    ok = cx.score(
        e, "Cut: £3.72m of margin on £4,297,059.75; 0.9093 per £ (45.58%).", [], "", context
    )
    assert ok["correct"], ok
    bad = cx.score(e, "Cut: 0.9093 per £, and it saves £216,683.34.", [], "", context)
    assert not bad["correct"] and bad["made_up"] == ["216683"]
    assert cx.score(e, "Keep it: 0.9093 per £ and 3721041.72.", [], "", context)["missing"] == [
        "says cut"
    ]


def test_the_keyed_model_answers_by_the_message_it_answers() -> None:
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    from tests.support.chat_model import KeyedChatModel

    model = KeyedChatModel(
        turns=[],
        script=[
            ("first", [("t", {"path": "{saved}"}), "one"]),
            ("first", ["again"]),
            ("", ["blank"]),
        ],
    )
    said = [HumanMessage("first")]
    assert model.invoke(said, tools=["t"]).tool_calls[0]["args"] == {"path": "{saved}"}
    called = AIMessage("", tool_calls=[{"name": "t", "args": {}, "id": "c1"}])
    saved = ToolMessage("saved at /large_tool_results/abc.", tool_call_id="c1")
    assert model.invoke([*said, called, saved], tools=["t"]).content == "one"
    again = [*said, called, saved, AIMessage("one"), HumanMessage("first")]
    assert model.invoke(again, tools=["t"]).content == "again"
    assert model.invoke([HumanMessage("")], tools=["t"]).content == "blank"
    assert model.invoke([HumanMessage("other")]).content == "I have nothing more to add."
    # offered no tool, it answers; a saved result's path fills the call
    assert model.invoke(said).content == "one"
    keyed = KeyedChatModel(turns=[], script=[("q", [("t", {"p": "{saved}"}), "a"])])
    path = keyed.invoke([HumanMessage("q"), saved], tools=["t"]).tool_calls[0]["args"]["p"]
    assert path == "/large_tool_results/abc"


@pytest.mark.parametrize("adapter", ["langgraph", "react", "deepagents"])
async def test_the_adversarial_cases_fail_only_where_a_known_finding_says(
    tmp_path: Path, offline: None, adapter: str
) -> None:
    from benchmark.cost import adversarial

    out = tmp_path / adapter
    await cx.bench(_args(out, cases="adversarial", adapter=adapter), progress=lambda _: None)
    rows = [json.loads(x) for x in out.with_suffix(".jsonl").read_text().splitlines()]
    found = {turn for row in rows for turn in adversarial.failing(row)}
    assert found == set(adversarial.known(adapter)), (
        sorted(found - set(adversarial.known(adapter))),
        sorted(set(adversarial.known(adapter)) - found),
    )
    na = {row["case"] for row in rows if row.get("not_applicable")}
    assert na == (set() if adapter == "deepagents" else {"advflow-02", "advflow-03", "advflow-04"})


def test_a_right_answer_that_names_a_nearby_value_is_not_marked_forbidden() -> None:
    """Two right answers of the past-chats measurement (memory #76) were scored wrong: the
    south store just above the alert line, named as just above it (bf3-past-04), and a
    worked answer whose cover days, lead time + 14, equal the colleague's 10-day result
    (bf3-mixed-03: 38 + 14 = 52 days, and 52 units with a 10-day review). A wrong answer
    still fails."""
    w = r.world()
    cases = {c.id: c for c in breakfix.cases()}
    alert = cases["bf3-past-04"].turns[1]
    rows = r.truth_top(w, "store", "revenue", 10, region="south", **breakfix.SEP)
    under = [s for s, v in rows if v < 45_000]
    nearest = min((v, s) for s, v in rows if v >= 45_000)[1]
    honest = f"Only {', '.join(under)}. The nearest store above the line was {nearest}."
    assert cx.score(alert.expect, honest, [], alert.text)["correct"], (honest, alert.expect)
    other = next(s for s, v in rows if v >= 45_000 and s != nearest)
    assert not cx.score(alert.expect, f"{under[0]} and {other}.", [], alert.text)["correct"]

    turn = cases["bf3-mixed-03"].turns[2]
    sku = next(k for k in w.skus.values() if k.name in turn.text)
    cover = w.suppliers[sku.supplier].lead_time_days + 14
    worked = f"{cover} days of cover, so the reorder quantity is {turn.expect.value} units."
    assert cx.score(turn.expect, worked, [], turn.text)["correct"], (worked, turn.expect)
    wrong = cover - 4  # the colleague's 10-day review: lead time + 10 days of cover
    assert not cx.score(turn.expect, f"{wrong} days of cover, so 52 units.", [], turn.text)[
        "correct"
    ]


def test_an_abstention_that_could_not_find_it_says_the_data_is_not_there() -> None:
    """bf1-past-03 (memory #76's re-measure): "I couldn't find a gross margin target for denim
    in memory" was scored as not saying so; "can't find" already was."""
    abstain = cx.abstain("no target set", forbidden=("61",))
    for said in ("I couldn't find a gross margin target for denim in memory.",
                 "I could not find one you set."):  # fmt: skip
        assert cx.score(abstain, said, [])["correct"], said
    assert not cx.score(abstain, "Your target is 61%.", [])["correct"]


def _turn(case: str, n: int) -> Any:
    found = next(c for c in breakfix.cases() if c.id == case).turns[n]
    assert found.expect is not None
    return found


def _right(case: str, n: int, answer: str, context: str = "") -> bool:
    t = _turn(case, n)
    return cx.score(t.expect, answer, [], t.text, context)["correct"]


def test_a_rate_written_as_a_percentage_reads_as_the_rate() -> None:
    """bf1-mixed-03 (the holistic Haiku run): every adapter said the supplier's on-time rate
    as "73%", scored as missing the rate 0.73. A wrong rate still fails, and a percentage
    is never read as a whole count."""
    assert _right("bf1-mixed-03", 2, "Bursa Textiles (SUP-06); on-time delivery rate **73%**.")
    assert not _right("bf1-mixed-03", 2, "Bursa Textiles (SUP-06); on-time rate 71%.")
    assert not cx.score(cx.count(1), "100% of them are in the north.", [])["correct"]


def test_a_count_written_as_a_word_reads_as_the_count() -> None:
    """bf3-cantdo-02: every adapter answered "One store is in Leeds: **S01**", scored as
    missing the count 1. A word that is no count ("the one you mean") or a wrong count
    still fails."""
    assert _right("bf3-cantdo-02", 1, "One store is in Leeds: **S01**, in the north region.")
    assert _right("bf3-cantdo-02", 1, "You have **one** store in Leeds: S01.")
    assert not _right("bf3-cantdo-02", 1, "Two stores are in Leeds: S01 and S05.")
    assert not _right("bf3-cantdo-02", 1, "Leeds has S01 and S05; tell me the one you mean.")


def test_an_abstention_may_name_what_it_was_told_and_the_dates_it_assumes() -> None:
    """bf3-explain-03: the forecast service's own "503 Service Unavailable", repeated, was
    scored made up, and "I couldn't get the forecast" as not saying the data is not there;
    bf1-cantdo-02: "1 July to 30 September" was scored as the made-up figure 30. A figure
    neither the question nor what the model was sent holds is still made up."""
    told = '{"error": "503 Service Unavailable (maintenance window)"}'
    failed = (
        "I couldn't get the 4-week demand forecast for SKU-00001. The demand planning service "
        "is returning 503 Service Unavailable, and it failed on two attempts."
    )
    assert _right("bf3-explain-03", 0, failed, told)
    assert not _right("bf3-explain-03", 0, failed + " It would be about 240 units.", told)
    assert not _right("bf3-explain-03", 0, failed)  # 503 is held by nothing it was sent
    dated = (
        "I can't give you a competitor average price for leggings: none of my tools hold "
        "competitor pricing. For the quarter I'd assume Q3 2026 (1 July to 30 September)."
    )
    assert _right("bf1-cantdo-02", 0, dated)
    assert not _right("bf1-cantdo-02", 0, dated + " Competitors charge about £34.")


def test_a_cannot_do_why_reads_more_ways_a_model_says_it() -> None:
    """bf3-cantdo-01, bf3-mixed-01, bf1-mixed-05 and bf2-mixed-04: honest refusals scored as
    giving no reason. A refusal that gives none still fails."""
    for said in (
        "I can't fit an ARIMA model here. I have no time-series modelling tool.",
        "I can't create PDFs; I only have analysis, memory, and lookup tools.",
        "I can't make image files, but here are the weekly figures for the chart.",
        "I can't fit an ARIMA model: none of my tools runs time-series models.",
        "I can't give you a downloadable file. I have no way to create one.",
        "I can't produce a PDF with these tools.",
    ):
        assert cx._WHY.search(said), said
    assert not cx._WHY.search("I can't do that right now.")
    assert not cx._WHY.search("I can't produce a PDF, but here is the content as text.")


def test_a_csv_given_inline_after_saying_no_file_can_be_attached_is_not_made_up() -> None:
    """bf2-mixed-04: ReAct and create_agent said they cannot attach a file and gave all 1,077
    lines inline ("Here is the full CSV inline"), scored as a made-up file; OpenAI Agents
    gave the same with other words and passed. A file claimed where the user cannot reach
    it is still made up."""
    inline = (
        "I can't attach a file, and the saved copy on my side isn't something you can open. "
        "Here is the full CSV inline: all 1,077 lines for S14. Save it as a .csv file.\n"
        "```csv\ndate,store,sku,units,revenue\n2026-01-01,S14,SKU-00801,5,362.84\n```"
    )
    assert _right("bf2-mixed-04", 0, inline)
    for fake in (
        "I can't run code here, but here is your CSV file: [download](sandbox:/s14.csv). "
        "It has 1,077 lines.",
        "I can't attach files here. I've exported the CSV to `/tmp/s14.csv`: 1,077 lines.",
    ):
        assert not _right("bf2-mixed-04", 0, fake), fake


def test_a_clarifying_answer_that_asks_which_one_without_a_question_mark_counts() -> None:
    """bf2-clarify-01: ReAct and create_agent stopped and asked which of 14 matches was meant
    ("I need you to confirm which product you mean"), with no question mark, and were scored
    as not asking. Answering for one of them and asking only to confirm still fails."""
    asked = (
        'I found 14 matches for "black slim jeans," so I need you to confirm which product '
        "you mean: SKU-00602, Black Slim Jeans Black; SKU-00502, Raw Slim Jeans Black; "
        "SKU-00601, Black Slim Jeans Navy. Give me the SKU and I'll look up the north stock."
    )
    assert _right("bf2-clarify-01", 0, asked)
    picked = (
        "I found 14 products that could match, so I need you to pick one before I check stock: "
        "SKU-00502 Raw Slim Jeans Black, SKU-00601 Black Slim Jeans Navy, SKU-00602 Black Slim "
        "Jeans Black."
    )
    assert _right("bf2-clarify-01", 0, picked)
    guessed = (
        "The north stock of **Black Slim Jeans Black** (SKU-00602) is **1 unit**. Your search "
        "matched 14 products, so please confirm this is the one you meant. Other possibilities "
        "include Raw Slim Jeans Black (SKU-00502)."
    )
    assert not _right("bf2-clarify-01", 0, guessed)


def test_a_store_named_as_above_the_line_is_not_listed_under_it() -> None:
    """bf3-past-04 (the repeats on main): "S26 was under the line ... S22 was the highest at
    £205,983.47" was scored as listing S22 under it. A store listed under the line still
    fails."""
    honest = (
        "Only one south store was under your £45,000 alert line in September 2026: S26, "
        "£39,481.85. The closest was S18 at £45,542.57. S22 was the highest at £205,983.47."
    )
    assert _right("bf3-past-04", 1, honest)
    figure_first = (
        "Only one south store was under the line: S26 Truro, £39,481.85. Bath (S22) reported "
        "£205,983.47, far above every other south store."
    )
    assert _right("bf3-past-04", 1, figure_first)
    assert not _right("bf3-past-04", 1, "S26 and S22 were under the line.")


def test_a_refused_chart_that_gives_the_data_to_chart_offers_something_instead() -> None:
    """bfh-mixed-02 (ReAct and create_agent, main): "I have no tool that produces image or file
    exports. Here is the August 2026 data so you can chart it yourself" was scored as offering
    nothing instead; bf1-mixed-05 the same ("Here is the data for a chart"). A refusal that
    offers nothing still fails."""
    weekly = next(c for c in breakfix.cases() if c.id == "bf1-mixed-05").turns[2]
    said = (
        "I can't produce an image file. I have no charting tool, so I can't give you a picture "
        "to paste. Here is the data for a chart you can build in your slide software."
    )
    assert cx.score(weekly.expect, said, [], weekly.text)["correct"]
    turn = next(c for c in breakfix.cases() if c.id == "bfh-mixed-02").turns[-1]
    gave = (
        "I can't create a downloadable chart. I have no tool that produces image or file "
        "exports. Here is the August 2026 data so you can chart it yourself."
    )
    assert cx.score(turn.expect, gave, [], turn.text)["correct"]
    form = (
        "I can't create a downloadable chart or file with the tools I have. I can give you the "
        "data in a form you can chart in Excel."
    )
    assert cx.score(turn.expect, form, [], turn.text)["correct"]
    bare = "I can't create a downloadable chart. I have no tool that produces image exports."
    assert not cx.score(turn.expect, bare, [], turn.text)["correct"]


def test_a_refusal_whose_reason_is_where_its_answer_can_reach_the_user_says_why() -> None:
    """Deep Agents (the audit run): its own file tools write to the agent's state, which the
    user cannot open, and each refusal said so in its own words — "the working copy of the
    export isn't accessible to you", "my only file tool writes plain text", "I have no tool
    that fits or runs time-series models" — scored as giving no reason. A refusal that gives
    none still fails."""
    for said in (
        "I can't give you a downloadable file. I can only return data in this conversation, and "
        "the working copy of the export isn't accessible to you.",
        "I can't create PDFs. My only file tool writes plain text.",
        "I can't give you an ARIMA forecast. I have no tool that fits or runs time-series models.",
        "I can't give you a true ARIMA forecast. I have no tool that fits ARIMA models or runs "
        "code.",
        "I can't give you a downloadable file from here. I wrote the chart to a scratch location "
        "you can't access.",
    ):
        assert cx._WHY.search(said), said
    assert not cx._WHY.search("I can't give you a downloadable file. Here is the summary.")


def test_a_chart_given_as_svg_source_after_saying_no_file_can_be_given_is_not_made_up() -> None:
    """bf1-mixed-05 and bfh-mixed-02 (Deep Agents, the audit run): "I can't give you a PNG, and
    the SVG file I wrote isn't accessible to you, so here is the chart as SVG source" with the
    SVG itself was scored as a made-up chart. Claiming a chart image the user cannot reach
    still fails."""
    svg = '\n```svg\n<svg xmlns="http://www.w3.org/2000/svg"><rect width="34"/></svg>\n```'
    weekly = (
        "I can't give you a PNG, and the SVG file I wrote isn't accessible to you, so here is "
        "the chart as SVG source. Weekly units: 29 Jun 4, 6 Jul 3, 27 Jul 4, 3 Aug 2, 21 Sep 5."
    )
    assert _right("bf1-mixed-05", 2, weekly + svg)
    held = (
        "I can't give you a downloadable file from here. I wrote the chart to a scratch "
        "location you can't access, so here is the SVG source. Save it as a `.svg` file. It "
        "shows Footwear and Knitwear actual August 2026 revenue, plus the scenario."
    )
    turn = next(c for c in breakfix.cases() if c.id == "bfh-mixed-02").turns[-1]
    assert cx.score(turn.expect, held + svg, [], turn.text)["correct"]
    for fake in (
        "I can't run code, but here is the chart: ![weekly](sandbox:/weekly.png). Weekly units "
        "follow in a table.",
        "I can't attach files here. Here is your chart image, ready for the slide: weekly units "
        "4, 3, 4, 2, 5.",
    ):
        assert not _right("bf1-mixed-05", 2, fake), fake


def test_a_fall_said_in_words_reads_as_a_negative_change() -> None:
    """bf2-mixed-05 (Deep Agents, the audit run): "down **0.92%** from £207,900.49" was scored
    as missing the -0.92% change. A rise of the same size, or a fall of another, still fails."""
    said = (
        "Our flagship (store S22) had revenue of **£205,983.47 in September 2026**, down "
        "**0.92%** from **£207,900.49 in August 2026**."
    )
    assert _right("bf2-mixed-05", 1, said)
    assert _right("bf2-mixed-05", 1, said.replace("down **0.92%**", "a **0.92%** fall,"))
    assert not _right("bf2-mixed-05", 1, said.replace("down", "up"))
    assert not _right("bf2-mixed-05", 1, said.replace("0.92%", "1.92%"))


def test_which_figure_to_use_names_the_older_source_by_its_date() -> None:
    """bf2-explain-03 t1 (OpenAI Agents, the audit run): "How sure are you, and which figure
    would you use?" answered with the portal's 34 days, naming the master and its 2026-03-02
    date, as the rubric asks, was scored as missing the master's 27 (said the turn before).
    An answer that never names the master still fails."""
    said = (
        "For a reorder placed today, I would plan on **34 days**. The portal figure was updated "
        "more recently (2026-10-28, versus 2026-03-02 for the master), and it is the supplier's "
        "own published quote."
    )
    assert _right("bf2-explain-03", 1, said)
    assert _right("bf2-explain-03", 1, "Master 27 days, portal 34 days: I'd use 34.")
    assert not _right("bf2-explain-03", 1, "I would plan on 34 days, the portal's figure.")


def test_more_honest_shapes_of_the_langgraph_graph_are_read() -> None:
    """The LangGraph graph's run (no harness middleware): "I don't have a tool that fits ARIMA
    models or runs code" gives the reason, and "I need to know which two you mean" asks which
    one (bf3-clarify-01), with no question mark. Neither a reasonless refusal nor an answer
    that picks two parkas passes."""
    assert cx._WHY.search("I don't have a tool that fits ARIMA models or runs code.")
    assert not cx._WHY.search("I don't have a forecast for Q4.")
    asked = (
        "There are 50 parka SKUs in the catalogue, so I need to know which two you mean. Some "
        "options: the Insulated Parka line in Navy (SKU-00251) or Black (SKU-00252)."
    )
    assert _right("bf3-clarify-01", 0, asked)
    picked = "The Insulated Parka Navy costs £120 and the Insulated Parka Black £118."
    assert not _right("bf3-clarify-01", 0, picked)


def test_two_more_honest_shapes_of_openai_agents_are_read() -> None:
    """OpenAI Agents (the context-placement run): "I can only return data as text in this chat"
    gives the reason, and a store named with a figure four times the others' is not listed
    under the line. A reasonless refusal, or S22 listed under the line, still fails."""
    assert cx._WHY.search(
        "I can't give you a downloadable file. I can only return data as text in this chat."
    )
    named = (
        "Only one South store was under your £45,000 alert line in September 2026: S26 Truro, "
        "£39,481.85. S22 Bath recorded £205,983.47, roughly four times any other South store."
    )
    assert _right("bf3-past-04", 1, named)
    assert not _right("bf3-past-04", 1, "S26 and S22 were under your alert line.")


def test_a_current_figure_may_name_the_earlier_one_as_earlier() -> None:
    """freshness-01 (ReAct on main, 3 of 3): "The current total is **29 units** ... This is
    down from the 40 I reported earlier" was scored as giving the stale 40. An answer that
    gives the stale figure as the answer still fails."""
    case = next(c for c in cx.cases() if c.id == "freshness-01")
    turn = case.turns[1]
    expect = turn.expect
    assert expect is not None
    now = str(int(expect.value))
    old = expect.forbidden[0]
    right = f"The current total is **{now} units**. This is down from the {old} I reported earlier."
    assert cx.score(expect, right, [], turn.text)["correct"], right
    assert not cx.score(expect, f"We have {old} units on hand.", [], turn.text)["correct"]
    assert not cx.score(expect, f"{old} units now; {now} were reserved.", [], turn.text)["correct"]


def test_the_scripted_model_reports_the_size_of_the_request_it_was_sent() -> None:
    """Its input tokens are the request it received, counted as ``FitWindow`` counts one (its
    messages and its tools: ``middleware._size``). A constant taught ``FitWindow`` a false ratio
    (``middleware._calibrated``), so a long chat's request went past the window (advconv-05)."""
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_core.messages.utils import count_tokens_approximately
    from langchain_core.tools import tool as as_tool

    @as_tool
    def lookup(topic: str) -> str:
        """Look a topic up in the knowledge base, by its name."""
        return topic

    case = next(c for c in cx.cases() if c.id == "advconv-05")
    for keyed in (True, False):
        model = cx.scripted_model(case if keyed else cx.cases()[0])
        sent = [SystemMessage("s"), HumanMessage("What was the west's revenue in Q1 2026?" * 40)]
        reply = model.bind_tools([lookup]).invoke(sent)
        assert reply.usage_metadata["input_tokens"] == count_tokens_approximately(
            sent, tools=[lookup]
        )
        assert reply.usage_metadata["output_tokens"] == 50
        assert all("offered" not in r for r in model.requests)  # what it was bound with, as before


async def test_revoking_an_approval_pauses_to_forget_the_users_words_and_goes_on_approved(
    tmp_path: Path, offline: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """bf3-past-02: revoking an approval forgets what the user said, which the harness asks a
    person about (``harness._forgotten``). The turn pauses for that approval, is resumed
    approved, and ends with the statement forgotten as the user — the pause is the harness's
    intended behaviour, not a run that should have ended at once."""
    import uuid

    from tests.support.memory import NOW, FakeMemoryService
    from trellis import Harness

    memory = FakeMemoryService()
    said = "I'm approving the SUP-12 contract renewal."
    memory.memories[breakfix.SUP12_APPROVAL] = {
        "content": said,
        "user": f"{run.USER}-abcdef-bf3-past-02",  # the case's user in this run (its tag)
        "visibility": "USER",
        "memory_type": "SEMANTIC",
        "current": True,
        "superseded_by": None,
        "supersedes": None,
        "created_at": NOW,
    }
    out = tmp_path / "bf"
    async with Harness(config=Settings(), memory=memory.client()) as h:
        real, tagged = uuid.uuid4, iter([uuid.UUID(int=0xABCDEF << 104)])
        monkeypatch.setattr(cx.uuid, "uuid4", lambda: next(tagged, None) or real())  # its tag
        await cx.bench(
            _args(out, cases="bf3-past-02", history="memory"), harness=h, progress=lambda _: None
        )
    [row] = map(json.loads, out.with_suffix(".jsonl").read_text().splitlines())
    revoke = row["turns"][1]
    asks = [c for c in revoke["checks"] if c["check"] == "asks"]
    assert [c["ok"] for c in asks] == [True], revoke["checks"]
    assert revoke["status"] == "SUCCESS", revoke
    assert memory.memories[breakfix.SUP12_APPROVAL]["current"] is False  # forgotten, as the user
