"""The ``explain`` suite (``benchmark.cost.explain``), offline: a scripted model, memory on (the
memory fake), two conversations — the record read from turn 1's journal, the strict checks and
the made-up and leak counts against it, every request kept, a rerun skipping what is saved,
and the report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from benchmark.cost import explain
from benchmark.cost.__main__ import arguments, main
from tests.support.memory import FakeMemoryService
from tests.support.models import Script, ScriptedChat
from trellis import Harness, Settings

FIRST_CALLS: dict[str, Any] = {
    explain.FIRSTS[0].question: ("stock_quote", {"symbol": "ACME"}),
    explain.FIRSTS[1].question: ("quarterly_filing", {"company": "ACME"}),
    explain.FIRSTS[2].question: [
        ("tape_page", {"symbol": "ACME", "page": n}) for n in range(1, 13)
    ],
}
ANSWERS = {
    "Where did that come from? Give me the citation.": (
        "From the Nasdaq Last Sale feed, 2026-10-27, via stock_quote."
    ),
    "What steps did you take to answer that?": (
        "I made 1 tool call: quarterly_filing. Bloomberg agreed. <tool_response>"
    ),
}


class Reader(Script):
    """Turn 1: the calls its question needs, then an answer; turn 2: a fixed answer."""

    def next(self, body: dict[str, Any]) -> Any:
        messages = body["messages"]
        asked = [m["content"] for m in messages if m["role"] == "user"][-1]
        if isinstance(asked, list):  # the memory context comes first, then the question
            asked = asked[-1]["text"]
        if messages[-1]["role"] == "tool":
            return "ACME closed at 123.45 USD."
        return FIRST_CALLS.get(asked) or ANSWERS.get(asked, "I do not know.")


def _lines(path: str) -> list[str]:
    return Path(path).read_text().splitlines()


async def _bench(h: Harness, out: str, pick: list[tuple[str, str]]) -> dict[str, Any]:
    return await explain.bench(
        model=ScriptedChat(script=Reader([])), model_name="scripted", window=32_000, out=out,
        arm="t", pick=pick, harness=h, progress=lambda _: None,
    )  # fmt: skip


@pytest.mark.asyncio
async def test_two_conversations_are_scored_against_turn_ones_record(tmp_path: Any) -> None:
    fake = FakeMemoryService()
    out = str(tmp_path / "smoke")
    pick = [("short", "citation"), ("long", "steps")]
    async with Harness(config=Settings(), memory=fake.client()) as h:
        report = await _bench(h, out, pick)
    short, long = report["conversations"]
    assert short["record"]["steps"][0]["tool"] == "stock_quote"
    assert short["record"]["steps"][0]["sources"][0]["name"] == explain.NASDAQ.name
    assert short["record"]["status"] == "SUCCESS" and short["record"]["seconds"] > 0
    assert short["rows"][1]["correct"] is True and short["rows"][1]["made_up"] == []
    # the step it took, rightly counted - and a source no call returned, and a leaked tag
    assert long["rows"][1]["correct"] is True
    assert long["rows"][1]["made_up"] == ["source:Bloomberg"]
    assert long["rows"][1]["leaks"] == ["<tool_response>"]
    # every request kept, turn 2's with turn 1 replayed from memory
    requests = [json.loads(line) for line in _lines(f"{out}.requests.jsonl")]
    second = [r for r in requests if r["first"] == "short" and r["turn"] == 2]
    said = [m.get("content") for m in second[0]["messages"]]
    assert explain.FIRSTS[0].question in said and "input_tokens" in second[0]
    arm = report["arms"]["t"]
    assert arm["explain"]["correct"]["mean"] == 1.0 and arm["explain"]["made_up"]["mean"] == 0.5
    assert arm["stalls"] == 0
    assert explain.markdown(report).startswith("# explain: scripted")
    # a rerun runs nothing saved
    async with Harness(config=Settings(), memory=fake.client()) as h:
        again = await _bench(h, out, pick)
    assert [c["rows"] for c in again["conversations"]] == [short["rows"], long["rows"]]
    assert len(_lines(f"{out}.jsonl")) == 2


def test_the_checks_are_strict_against_the_record() -> None:
    record = explain.Record(
        status="SUCCESS",
        seconds=10.0,
        steps=[
            explain.Step("stock_quote", {"symbol": "ACME"}, "OK", 0.1, [{"name": "Nasdaq Last Sale feed", "date": "2026-10-27"}]),
            explain.Step("memory_search", {"q": "x"}, "OK", 0.2),
        ],
    )  # fmt: skip
    check = {k.name: k.check for k in explain.KINDS}
    assert check["steps"]("Two steps: stock_quote, then memory_search.", record, [])
    assert not check["steps"]("Two steps: I looked up the quote.", record, [])
    assert check["citation"]("Nasdaq Last Sale feed (2026-10-27)", record, [])
    assert not check["citation"]("the Nasdaq Last Sale feed", record, [])
    assert check["how_long"]("about 11 seconds", record, [])
    assert not check["how_long"]("about 2 seconds", record, [])
    assert check["how_long"]("0.2 minutes", record, [])
    assert check["approved"]("No approval was needed.", record, [])
    assert not check["approved"]("Yes, the analyst approved it.", record, [])
    assert check["how_many"]("I made 2 tool calls.", record, [])
    assert check["which_tools"]("stock_quote and memory_search", record, [])
    assert not check["which_tools"]("the quote tool", record, [])
    assert check["about_me"]("You hold 40 ACME shares.", record, [explain.OWN_NOTE])
    assert not check["about_me"]("You hold 40 shares; Kofi holds ZETA.", record, [])
    assert check["unknown"]("I do not know which analyst.", record, [])
    assert check["sure"]("I am fairly sure.", explain.Record(), []) is False
    assert explain.made_up("I made 3 calls to tape_page, from Reuters", record) == [
        "tool:tape_page",
        "source:Reuters",
        "count:3",
    ]
    assert explain.made_up("stock_quote, from the Nasdaq Last Sale feed: 2 calls", record) == []
    assert explain.leaks("[price_range_low] to {time_taken} <|end_of_role|>") == [
        "[price_range_low]",
        "{time_taken}",
        "<|end_of_role|>",
    ]
    assert explain.score("control", "short", "ZETA closed at 61.2.", record, [])["correct"]
    assert explain.score("control", "many", "Yes.", record, [])["correct"] is None


def test_the_pick_names_conversations() -> None:
    assert len(explain.picked("pilot")) == 12
    assert explain.picked("short:steps, long:control") == [("short", "steps"), ("long", "control")]
    assert len(explain.picked("all")) == 36
    with pytest.raises(ValueError, match="first:kind"):
        explain.picked("tiny:steps")


def test_the_cli_takes_the_suite(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    args = arguments(["--model", "m", "--suite", "explain", "--arm", "c", "--pick", "short:steps"])
    assert args.suite == "explain" and args.arm == "c" and args.pick == "short:steps"
    seen: dict[str, Any] = {}

    async def bench(**kw: Any) -> dict[str, Any]:
        seen.update(kw)
        return {"model": "m", "arms": {}}

    monkeypatch.setattr(explain, "bench", bench)
    out = tmp_path / "x"
    assert (
        main(["--model", "m", "--suite", "explain", "--pick", "long:sure", "--out", str(out)]) == 0
    )
    assert (
        seen["pick"] == [("long", "sure")]
        and seen["repeats"] == 3
        and seen["render"] is None
        and seen["run_seconds"] == 600
    )
    assert json.loads(out.with_suffix(".json").read_text())["model"] == "m"


def test_a_count_said_in_markdown_is_read() -> None:
    """A small model writes ``12 `tape_page` calls`` or ``**12** calls``: still twelve."""
    assert explain.counts("I made **12 `tape_page` calls**, one per page.") == [12]
    assert explain.counts("I made **12** tool calls.") == [12]
    assert explain.counts("on October 30, 2 calls") == [2]
    assert explain.counts("fetched at 21:01:16 UTC\n\nThe tool output said") == []  # a time
