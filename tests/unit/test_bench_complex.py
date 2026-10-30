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

from benchmark.cost import complex as cx
from benchmark.cost import judge, run
from benchmark.cost import retail as r
from trellis import Settings


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
    assert 60 <= len(cases) <= 85 and len({c.id for c in cases}) == len(cases)
    assert {c.category for c in cases} == set(cx.CATEGORIES)
    assert {c.tier for c in cases} == set(cx.TIERS)
    kinds = collections.Counter(t.expect.kind for c in cases for t in c.turns if t.expect)
    assert set(kinds) == {*cx.STRICT_KINDS, "judge"}
    tools = {t.spec.name for t in r.TOOLS}
    for c in cases:
        assert any(t.expect for t in c.turns), c.id
        for t in c.turns:
            assert all(name in tools for rnd in t.calls for name, _ in rnd), c.id


def test_every_reference_answer_passes_its_own_strict_check() -> None:
    for c in cx.cases():
        for t in c.turns:
            e = t.expect
            if e is None or e.kind in ("judge", "tools", "steps"):
                continue
            found = cx.score(e, e.reference, [], t.text)
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
    # the scripted model reports 1,000 in and 50 out a call: $0.000125 at this price
    report = await cx.bench(
        _args(out, cases=PICK, price="claude-haiku-5-5", max_usd=0.0005), progress=lambda _: None
    )
    assert report["stopped"] and "spend cap" in report["stopped"]
    assert report["usd"] >= 0.0005 and report["usd"] < 0.0005 + 0.000125 * 2
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
