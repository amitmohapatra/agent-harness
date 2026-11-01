"""The tool-chain benchmark (``benchmark/tool_chains``) without a model: every case's gold
calls, run on the world's tools, give its expected answer; no question names an id; the
failing and the large tools behave as the cases say; the scorer reads figures, not an id's
digits; and every offline arm runs the cases through the harness with a scripted model."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from benchmark.tool_chains import __main__ as cli
from benchmark.tool_chains import cases as C
from benchmark.tool_chains import world as w
from trellis.harness.tools import results

TOOLS = {t.spec.name: t for t in w.TOOLS}
IDS = re.compile(r"\b(SKU|STY|SUP|WH|ST|AL|SH|LOC|GEN)-\d")


def test_42_cases_two_train_and_one_heldout_per_shape() -> None:
    every = C.cases()
    assert len(every) == 42 == len({c.id for c in every})
    shapes = {c.shape for c in every}
    assert len(shapes) == 14
    for shape in shapes:
        splits = sorted(c.split for c in every if c.shape == shape)
        assert splits == ["heldout", "train", "train"], shape


def test_no_question_names_an_id() -> None:
    for case in C.cases():
        for turn in case.turns:
            assert not IDS.search(turn.text), (case.id, turn.text)


@pytest.mark.parametrize("case", C.cases(), ids=lambda c: c.id)
def test_the_gold_calls_give_the_expected_answer(case: C.Case) -> None:
    w.reset()
    for turn in case.turns:
        failed = []
        for rnd in turn.rounds:
            for name, args in rnd:
                try:
                    TOOLS[name].fn(**args)
                except (RuntimeError, ConnectionError) as exc:
                    failed.append(str(exc))
        assert C.scored(turn, turn.reference)["correct"], (case.id, turn.reference)
        assert bool(failed) == case.shape.startswith("error-"), (case.id, failed)


def test_a_flaky_lookup_fails_once_per_case_and_a_down_one_always() -> None:
    flaky, down = sorted(w.FLAKY)[0], sorted(w.DOWN)[0]
    for _ in range(2):
        w.reset()
        with pytest.raises(ConnectionError):
            w.shipment_status.fn(flaky)
        assert w.shipment_status.fn(flaky)["allocation_id"] == flaky
    with pytest.raises(RuntimeError):
        w.shipment_status.fn(down)
    assert not w.FLAKY & w.DOWN


def test_an_assortment_is_kept_and_paged_not_sent_and_holds_the_local_items() -> None:
    for store, items in w.LOCAL_ITEMS.items():
        answer = w.store_assortment.fn(store)
        assert len(json.dumps(answer)) > results.MAX_CHARS
        held = {i["item_name"]: i["sku_id"] for i in answer["items"]}
        assert all(held[name] == sku for name, sku in items.items())
        with pytest.raises(ValueError):
            w.find_product.fn(next(iter(items)))  # only the assortment has it


def test_the_scorer_reads_figures_not_an_ids_or_a_dates_digits() -> None:
    assert C.numbers("SKU-1041 on 2026-11-04: 14 units, $1,053.50") == [14.0, 1053.5]
    turn = C.Turn("q", (), "r", expect=(14.0, "in transit"))
    assert C.scored(turn, "In_transit, 14 units.")["correct"]
    assert not C.scored(turn, "SKU-14 is in transit")["correct"]
    report = C.Turn("q", (), "r", forbidden=(C.DATE,), failed=True)
    assert C.scored(report, "The tracking system is down: no date.")["correct"]
    assert not C.scored(report, "It arrives 2026-11-04.")["correct"]
    assert C.scored(report, "I have no information.")["failure_unsaid"]
    assert not C.scored(report, None)["correct"]


@pytest.mark.parametrize("arm", ["react", "langgraph", "deep"])
def test_every_offline_arm_runs_the_cases_with_a_scripted_model(arm: str, tmp_path: Path) -> None:
    chosen = "large-id-h1,error-retry-h1,followup-move-h1,fanin-reorder-h1"
    out = tmp_path / "chains"
    assert cli.main(["--scripted", "--arms", arm, "--cases", chosen, "--out", str(out)]) == 0
    report = json.loads(Path(f"{out}.json").read_text())
    cell = report["arms"][arm]["all"]
    assert (cell["cases"], cell["correct"]) == (4, 4), report
    rows = [json.loads(line) for line in Path(f"{out}-{arm}.jsonl").read_text().splitlines()]
    retried = next(r for r in rows if r["case"] == "error-retry-h1")
    shipments = [
        c["status"] for c in retried["turns"][0]["calls"] if c["tool"] == "shipment_status"
    ]
    assert shipments == ["error", "ok"]
    followup = next(r for r in rows if r["case"] == "followup-move-h1")
    assert len(followup["turns"]) == 2 and followup["model_calls"] == 6
    assert all(r["narrowed_away"] == [] and r["missed_tools"] == [] for r in rows)
    assert "| " + arm + " | heldout | 4/4 |" in Path(f"{out}.md").read_text()


def test_narrowed_to_one_tool_a_single_path_chain_stays_offered_and_the_gaps_are_named() -> None:
    """The check without a model: a chain that is one path (each call takes the one before's
    output) stays whole whichever of its tools the hints keep; where a call needs two
    results (fan-in) or takes a value under another name (a date, ``from_store_id``), the
    links do not keep the other branch — the report names each such pair."""
    found = cli.narrowing(C.cases())
    single_path = {"d2-category", "d2-region", "d3-lead", "d3-contact", "large-id"}
    assert not single_path & set(found["by_shape"]), found["by_shape"]
    assert ("warehouse_info", ("fiscal_week",)) in found["by_shape"]["d5-truck"]
    assert 0 < found["dropped"] < found["pairs"]
