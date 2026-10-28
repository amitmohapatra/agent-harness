"""The ``explain`` suite (``benchmark.cost.explain``), offline: a scripted model that reads the
earlier answer's record when it is offered and answers from it, in each arm, as the end user and
as the administrator — the rows, the scores against the record and the report."""

from __future__ import annotations

import json
from typing import Any

import pytest

from benchmark.cost import explain
from benchmark.cost.__main__ import arguments, main
from tests.support.memory import FakeMemoryService
from tests.support.models import Script, ScriptedChat
from trellis import Harness, Settings
from trellis.harness.explain import EARLIER, Record, Step

#: Each turn 1's calls.
FIRST_CALLS: dict[str, Any] = {
    explain.FIRSTS[0].question: ("stock_quote", {"symbol": "ACME"}),
    explain.FIRSTS[1].question: ("quarterly_filing", {"company": "ACME"}),
    explain.FIRSTS[2].question: [
        ("tape_page", {"symbol": "ACME", "page": n}) for n in range(1, 13)
    ],
}


class Reader(Script):
    """Turn 1: the calls its question needs; turn 2: the record when offered and asked about
    the answer (then its text as the answer), else a plain answer."""

    def next(self, body: dict[str, Any]) -> Any:
        messages = body["messages"]
        last = messages[-1]
        offered = [t["function"]["name"] for t in body.get("tools") or []]
        asked = [m["content"] for m in messages if m["role"] == "user"][-1]
        if last["role"] == "tool":
            called = [c["function"]["name"] for c in messages[-2].get("tool_calls") or []]
            if called == [EARLIER]:
                return last["content"]  # the record, as the answer
            return "ACME closed at 123.45 USD, per the Nasdaq Last Sale feed."
        first = FIRST_CALLS.get(asked)
        if first is not None:
            return first
        if EARLIER in offered and asked not in explain.CONTROLS.values():
            return (EARLIER, {})
        return "From the Nasdaq Last Sale feed; I do not know more."


def reader() -> ScriptedChat:
    return ScriptedChat(script=Reader([]))


@pytest.mark.asyncio
async def test_each_arm_is_scored_against_the_earlier_runs_record() -> None:
    kinds = [k for k in explain.KINDS if k.name in ("citation", "how_many", "unknown")]
    async with Harness(config=Settings()) as h:
        report = await explain.bench(
            model=reader(), model_name="scripted", window=32_000, harness=h,
            firsts=explain.FIRSTS, kinds=kinds, progress=lambda _: None,
        )  # fmt: skip
    assert report["audience"] == "client" and report["tier"] == "large"
    arms = report["arms"]
    assert set(arms) == {"main", "sized", "lookup"}
    looked = arms["lookup"]["explain"]
    assert looked["looked_up"]["mean"] == 1.0 and looked["forbidden_items"] == 0
    assert (
        looked["by_kind"]["citation"]["mean"] == 1.0
        and looked["by_kind"]["how_many"]["mean"] == 1.0
    )
    assert arms["main"]["explain"]["looked_up"]["mean"] == 0.0
    assert arms["lookup"]["controls"]["looked_up"]["mean"] == 0.0
    many = [c for c in report["conversations"] if c["first"] == "many" and c["arm"] == "lookup"]
    assert all(c["rows"][1]["steps"] == 12 for c in many)
    assert all(c["rows"][1]["sources"] == ["Consolidated tape"] for c in many)
    assert "| lookup | 0.67 ± 0.38 | 1.00 ± 0.00 | 0 | 0.00" in explain.markdown(report)


@pytest.mark.asyncio
async def test_as_the_administrator_the_record_is_shown_whole_and_not_checked_for_leaks() -> None:
    fake = FakeMemoryService(role="admin")
    kinds = [k for k in explain.KINDS if k.name == "which_tools"]
    async with Harness(config=Settings(), memory=fake.client()) as h:
        report = await explain.bench(
            model=reader(), model_name="scripted", window=32_000, harness=h, arms=["lookup"],
            firsts=explain.FIRSTS[:1], kinds=kinds, progress=lambda _: None,
        )  # fmt: skip
    assert report["audience"] == "admin"
    [asked, _control] = report["conversations"]
    assert asked["rows"][1]["looked_up"] and asked["rows"][1]["forbidden"] == []
    assert asked["rows"][1]["answer"].startswith('{"run": ')  # the administrator's view, whole
    assert [m["content"] for m in fake.memories.values()] == [explain.OWN_NOTE, explain.OTHER_NOTE]


def test_the_checks_read_the_record() -> None:
    record = Record(
        run_id="r",
        steps=[Step(tool="stock_quote", status="ok", said="x", sources=[explain.NASDAQ])],
        approvals=[{"tool": "stock_quote", "decision": "APPROVE"}],
    )
    by = {k.name: k.check for k in explain.KINDS}
    assert by["steps"]("1 step, from the Nasdaq Last Sale feed", record, [])
    assert not by["steps"]("two steps", record, [])
    assert by["how_long"]("about 2.5 seconds", record, []) and not by["how_long"](
        "fast", record, []
    )
    assert by["approved"]("Yes, a reviewer approved it.", record, [])
    assert not by["approved"]("No approval was needed.", record, [])
    assert by["approved"]("No approval was needed.", Record(run_id="r"), [])
    assert by["about_me"]("You hold 40 ACME shares.", record, [explain.OWN_NOTE])
    assert by["about_me"]("I do not know anything about you.", record, [])
    assert not by["citation"]("It is accurate.", Record(run_id="r"), [])
    assert explain.forbidden('stock_quote {"symbol": "A"} feed-7731 Kofi') == [
        "stock_quote",
        '"symbol"',
        "feed-7731",
        "another person's note",
    ]
    assert explain.forbidden(f"{EARLIER} said", other=False) == [EARLIER]


def test_the_cli_takes_the_suite_and_its_arms(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    args = arguments(
        ["--model", "m", "--suite", "explain", "--arms", "main,lookup", "--cap", "8000"]
    )
    assert args.suite == "explain" and args.arms == "main,lookup" and args.cap == 8000
    seen: dict[str, Any] = {}

    async def bench(**kw: Any) -> dict[str, Any]:
        seen.update(kw)
        return {"model": "m", "tier": "small", "audience": "client", "arms": {}}

    monkeypatch.setattr(explain, "bench", bench)
    out = tmp_path / "x"
    assert main(["--model", "m", "--suite", "explain", "--arms", "lookup", "--out", str(out)]) == 0
    assert seen["arms"] == ["lookup"] and seen["share"] == explain.SHARE
    assert json.loads(out.with_suffix(".json").read_text())["model"] == "m"


def test_the_administrators_harness_needs_the_admin_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TRELLIS_ADMIN_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="TRELLIS_ADMIN_API_KEY"):
        explain._harness(memory=True, admin=True)
