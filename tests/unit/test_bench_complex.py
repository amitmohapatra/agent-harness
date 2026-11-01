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


def test_a_rate_given_as_a_percentage_matches_its_fraction() -> None:
    """A rate the world keeps as a fraction (an on-time rate of 0.73) is right when the answer
    gives it as a percentage ("73%"), as a person and a model write it; a bare 73 is not."""
    expect = cx.record({"supplier": "SUP-06", "on_time": 0.73}, tol={"on_time": 0.005})
    right = "Bursa Textiles (SUP-06). Its on-time delivery rate is **73%**."
    assert cx.score(expect, right, [], "")["correct"] is True
    assert cx.score(expect, "SUP-06, on time 0.73.", [], "")["correct"] is True
    assert cx.score(expect, "SUP-06, on time 81%.", [], "")["correct"] is False
    assert cx.score(expect, "SUP-06 shipped 73 orders.", [], "")["correct"] is False


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


async def test_a_langgraph_graph_without_the_harness_middleware_is_an_adapter_of_its_own(
    tmp_path: Path, offline: None
) -> None:
    """``graph``: ``create_agent`` as a LangGraph team builds it — its tools bound when it is
    built (the retail tools, and memory's own from ``h.memory_tools()``), its checkpointer, and
    no harness middleware (the matrix's ``langgraph`` adapter); its step limit is LangGraph's own
    ``recursion_limit``: a model call and a tool step for each of ``max_steps``."""
    out = tmp_path / "cx-graph"
    report = await cx.bench(
        _args(out, cases="multihop-01,conversation-01", adapter="graph"),
        progress=lambda _: None,
    )
    assert report["adapter"] == "graph" and report["summary"]["all"]["strict"]["mean"] == 1.0
    graph, options = cx.target_of("m", "memory", None, 6, adapter="graph", tools=r.TOOLS[:2])
    assert options == {"framework_options": {"recursion_limit": 13}}
    assert sorted(graph.nodes["tools"].bound.tools_by_name) == sorted(
        t.spec.name for t in r.TOOLS[:2]
    )
    assert [n for n in graph.nodes if "." in n] == []  # no middleware node
    assert getattr(graph, "trellis_tools", None) is None  # bound, not declared to the harness


def test_the_adapter_follows_the_history_unless_given() -> None:
    assert cx.adapter_of("memory", None) == "react"
    assert cx.adapter_of("thread", None) == "langgraph"
    assert cx.adapter_of("memory", "openai") == "openai"


def test_the_claude_agent_reads_no_settings_of_the_checkout_it_runs_in() -> None:
    """The SDK loads every filesystem setting by default, the CLAUDE.md of the directory the
    CLI runs in among them: run from this checkout, the agent under test read the repository's
    own instructions to coding agents (and tried a shell command they name)."""
    options, _ = cx.target_of("m", "memory", None, 6, adapter="claude")
    assert options.setting_sources == [] and options.max_turns == 12


def test_kimi_is_an_adapter_with_its_instructions_in_its_own_agents_md(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Kimi Code over ACP: no system prompt crosses ACP, so the suite's instructions are the
    AGENTS.md of a working directory of its own (Kimi's slot for a project's instructions); its
    model is the endpoint's (Anthropic's own API: Kimi's ``anthropic`` provider)."""
    from trellis.harness.adapters.acp import ACPAgent

    monkeypatch.setenv("KIMI_RUN_ROOT", str(tmp_path))
    monkeypatch.setenv("LLM_BASE_URL", "https://api.anthropic.com/v1/")
    monkeypatch.setenv("LLM_API_KEY", "k")
    agent, options = cx.target_of("claude-haiku-5-5", "memory", None, 6, adapter="kimi")
    assert isinstance(agent, ACPAgent) and options == {}
    assert Path(agent.cwd or "", "AGENTS.md").read_text() == r.SYSTEM
    assert agent.env["KIMI_MODEL_PROVIDER_TYPE"] == "anthropic"
    assert agent.env["KIMI_MODEL_BASE_URL"] == "https://api.anthropic.com"
    assert agent.env["KIMI_MODEL_NAME"] == "claude-haiku-5-5"
    assert agent.env["KIMI_CODE_HOME"] == str(tmp_path / "home")
    monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:8080/v1")
    agent, _ = cx.target_of("granite", "memory", None, 6, adapter="kimi")
    assert agent.env["KIMI_MODEL_PROVIDER_TYPE"] == "openai"
    assert agent.env["KIMI_MODEL_BASE_URL"] == "http://127.0.0.1:8080/v1"


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
