"""The skills benchmark (``benchmark/cost/skills.py``) without a model: its cases and their
truths against the retail world, its checks on chats made here, the skills a run read as each
framework's trace says them, each adapter given the skills in its own slot (``complex.target_of``)
and a scripted run scored end to end."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from benchmark.cost import complex as cx
from benchmark.cost import retail as r
from benchmark.cost import skills as b
from tests.support.chat_model import ScriptedChatModel
from trellis import Settings, skills
from trellis.contracts import RunEvent, RunEventType
from trellis.harness.skills import MAX_BODY, SKILLS_ROOT, ClaudeSkills, SkillTools, valid_name

NOW = datetime(2026, 11, 3, 8, 30, tzinfo=UTC)
SEP = {"start": "2026-09-01", "end": "2026-09-30"}


def chat(answer: str = "Done.", **given: Any) -> b.Chat:
    """A one-turn chat that answered ``answer`` and ended well, with ``given`` over it."""
    base: dict[str, Any] = {
        "answers": [answer], "calls": [[]], "statuses": ["SUCCESS"], "read": [[]],
        "pinned": [{}], "warnings": [[]],
    }  # fmt: skip
    return b.Chat(now=NOW, **(base | given))


def _kept(path: Path) -> list[dict[str, Any]]:
    """What the bench wrote to ``path``: its JSON lines, or its one JSON document."""
    text = path.read_text()
    return [json.loads(x) for x in text.splitlines()] if path.suffix == ".jsonl" else [
        json.loads(text)
    ]  # fmt: skip


@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(b.Settings, "from_env", classmethod(lambda cls: Settings()))


# --------------------------------------------------------------------------- the cases
def test_the_set_has_every_category_and_about_thirty_cases() -> None:
    cases = b.cases()
    assert {c.category for c in cases} == set(b.CATEGORIES)
    assert len({c.id for c in cases}) == len(cases) and 28 <= len(cases) <= 34
    per = {k: sum(c.category == k for c in cases) for k in b.CATEGORIES}
    assert min(per.values()) >= 5, per


def test_every_case_serves_skills_a_store_can_hold() -> None:
    for case in b.cases():
        assert all(valid_name(s.name) for s in case.skills), case.id
        assert all(s.description and "\n" not in s.description for s in case.skills), case.id
    picks = [c for c in b.cases() if c.category == "pick"]
    for case in picks:  # the right one among six to ten of the same tenant, alike in words
        names = [s.name for s in case.skills]
        assert 6 <= len(names) <= 10 and len(set(names)) == len(names), case.id
    families = {n.split("-")[0] for n in (s.name for s in b.LIBRARY)}
    assert {"stock", "supplier", "reorder"} <= families  # three alike of each


def test_the_truths_are_the_worlds_by_the_rules_only_the_skills_say() -> None:
    w = r.world()
    sku = "SKU-01517"
    units = r.truth_total(w, "units", sku=sku, **SEP)
    on_hand = sum(w.on_hand(sku, g) for g in r.REGIONS)
    assert b.cover(sku) == pytest.approx(on_hand / (units / 30))
    assert b.cover_line(sku, 14) == "COVER SKU-01517 21.5 OK @14"
    assert b.cover_line(sku, 25) == "COVER SKU-01517 21.5 LOW @25"
    # 5.933 a day x (27 + 21) - 201 = 83.8 -> 84, under SUP-08's minimum of 150
    assert b.reorder("SKU-00858") == 150
    supplier = w.suppliers[w.skus["SKU-00964"].supplier]
    avg = r.truth_total(w, "units", sku="SKU-00964", **SEP) / 30
    stock = sum(w.on_hand("SKU-00964", g) for g in r.REGIONS)
    assert b.reorder("SKU-00964") == math.ceil(avg * (supplier.lead_time_days + 21) - stock)
    assert b.grade("SUP-07") == ("C", pytest.approx(0.7 * 0.7 + 0.3 * (1 - 35 / 45)))
    assert b.grade("SUP-02")[0] == "A" and b.grade("SUP-09")[0] == "B"
    assert b.promise("SUP-03") == 49 + 4 and b.promise("SUP-02") == 13 + 4
    assert b.markdown("SKU-01517") == pytest.approx(15.77 * 0.65)  # a 63.5% margin: 35% off
    assert b.markdown("SKU-00964") == pytest.approx(60.89 * 0.85)  # 43.2%: 15% off


def test_the_rules_are_in_the_skills_text_and_one_body_is_over_the_cap() -> None:
    by = {s.name: s for s in b.LIBRARY}
    assert "21" in by["reorder-quantity"].body and "0.7" in by["supplier-scorecard"].body
    assert "14" in by["stock-cover"].body
    assert "4" in dict(by["supplier-lead-time"].files)["rules.md"]
    assert "rules.md" in by["supplier-lead-time"].body and "4 " not in by["supplier-lead-time"].body
    assert "35%" in dict(by["markdown-price"].files)["tiers.md"]
    over = next(c for c in b.cases() if c.id == "oversized-body")
    body = next(s.body for s in over.skills if s.name == "stock-cover")
    assert len(body) > MAX_BODY and "@40" in body[MAX_BODY:] and "@40" not in body[:MAX_BODY]


# --------------------------------------------------------------------------- the checks
def test_a_fixed_line_a_figure_and_what_was_read() -> None:
    line = b.line("COVER SKU-01517 21.5 OK @14")
    assert line(chat("Cover is fine.\n**COVER  SKU-01517 21.5 OK @14**")) == []
    assert line(chat("COVER SKU-01517 21.5 LOW @25")) == [
        "answer lacks the line 'COVER SKU-01517 21.5 OK @14'"
    ]
    assert b.says(1523104.74)(chat("Revenue was £1,523,104.74.")) == []
    assert b.says(0.5567, 0.00006)(chat("score 0.55667")) == []
    assert b.says(53)(chat("about 49 days")) == ["answer lacks 53"]
    read = chat(read=[[("supplier-lead-time", None), ("supplier-lead-time", "rules.md")]])
    assert b.reads("supplier-lead-time", files=("rules.md",))(read) == []
    assert b.reads("supplier-scorecard")(read) == ["did not read supplier-scorecard"]
    assert b.reads("supplier-lead-time", files=("tiers.md",))(read) == [
        "did not read supplier-lead-time/tiers.md"
    ]
    assert b.reads_none()(chat()) == []
    assert b.reads_none()(read) == ["read a skill: supplier-lead-time"]
    assert b.lacks("COVER")(chat("965 units")) == []
    assert b.lacks("COVER")(chat("COVER SKU-1 2.0 OK @14")) == ["answer has 'COVER'"]


def test_the_calls_their_order_and_their_arguments() -> None:
    made = [("stock_level", {"sku": "S"}, ""), ("sales_summary", {}, ""), ("sku_details", {}, ""),
            ("supplier_info", {}, ""), ("reorder", {"quantity": 150, "supplier": "SUP-08"}, "")]  # fmt: skip
    ordered = b.in_order("stock_level", "sales_summary", "sku_details", "supplier_info")
    assert ordered(chat(calls=[made])) == []
    assert ordered(chat(calls=[[made[1], made[0], *made[2:]]])) == [
        "called sales_summary, stock_level, sku_details, supplier_info: not in the skill's order"
    ]
    assert ordered(chat(calls=[made[:2]])) == [
        "did not call sku_details",
        "did not call supplier_info",
    ]
    assert b.called("reorder", quantity=150, supplier="sup-08")(chat(calls=[made])) == []
    assert b.called("reorder", quantity=84)(chat(calls=[made])) == [
        "no reorder call with quantity=84 (made: [{'quantity': 150, 'supplier': 'SUP-08'}])"
    ]


def test_the_versions_a_run_pinned_a_run_that_went_on_and_a_change_only_once_approved() -> None:
    pinned = chat(pinned=[{"stock-cover": "u1"}])
    assert b.pins("stock-cover", "u1")(pinned) == [] and b.pins("stock-watch", None)(pinned) == []
    assert b.pins("stock-cover", "t1")(pinned) == ["pinned stock-cover u1, not t1"]
    assert b.ended()(chat()) == [] and b.ended()(chat(statuses=["ERROR"])) == ["turn ended ERROR"]
    assert b.never_ran("issue_refund")(chat()) == []
    ran = chat(ran=[("issue_refund", {"order": "ORD-7781", "amount": 950})])
    assert b.never_ran("issue_refund")(ran) == ["issue_refund ran without anyone approving it"]
    asked = chat(decided=[("issue_refund", {"amount": 950}, "reject")])
    assert b.never_ran("issue_refund")(asked) == []  # asked for, refused: it never ran
    approved = chat(
        decided=[("set_price", {"sku": "SKU-01517", "price": 10.25}, "approve")],
        ran=[("set_price", {"sku": "SKU-01517", "price": 10.25})],
    )
    assert b.approved("set_price", price=10.2505)(approved) == []
    assert b.approved("set_price", price=15.77)(approved) == [
        "no set_price call with price=15.77 asked for approval "
        "(asked: [{'sku': 'SKU-01517', 'price': 10.25}])"
    ]
    unasked = chat(ran=[("set_price", {"sku": "SKU-01517", "price": 10.25})])
    assert b.approved("set_price", price=10.25)(unasked)[0].startswith("no set_price call")
    twice = chat(decided=approved.decided, ran=approved.ran * 2)
    assert b.approved("set_price", price=10.25)(twice) == ["set_price ran 2 times, not once"]
    both = b.every(b.ended(), b.says(965))
    assert both(chat("965")) == [] and both(chat("12")) == ["answer lacks 965"]


def test_a_check_on_a_turn_that_did_not_run_says_so() -> None:
    assert b.line("X", turn=1)(chat()) == ["turn 1 did not run"]
    assert b.reads("stock-cover", turn=1)(chat()) == ["turn 1 did not run"]


# --------------------------------------------------------------------------- the trace
def test_the_calls_a_model_made_are_read_from_its_later_requests_each_once() -> None:
    def call(cid: str, name: str, **args: Any) -> dict[str, Any]:
        return {"id": cid, "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)}}  # fmt: skip

    first = {"messages": [
        {"role": "user", "content": "go"},
        {"role": "assistant", "tool_calls": [call("c1", "read_file", file_path="/skills/a/SKILL.md")]},
        {"role": "tool", "tool_call_id": "c1", "content": "..."},
    ]}  # fmt: skip
    second = {"messages": [*first["messages"],
                           {"role": "assistant", "tool_calls": [call("c2", "stock_level", sku="S")]}]}  # fmt: skip
    seen: set[str] = set()
    assert b.model_calls([first, second], seen) == [
        ("read_file", {"file_path": "/skills/a/SKILL.md"}),
        ("stock_level", {"sku": "S"}),
    ]
    assert b.model_calls([second], seen) == []  # a later turn's requests carry them again
    assert b.model_calls([{"messages": None}, {}], seen) == []


def test_a_skill_read_is_found_as_each_framework_reads_one() -> None:
    names = {"stock-cover", "supplier-lead-time"}
    calls = [
        ("read_file", {"file_path": f"{SKILLS_ROOT}stock-cover/SKILL.md", "limit": 100}),  # Deep Agents
        ("load_skill", {"name": "supplier-lead-time"}),  # the harness's tools
        ("load_skill", {"name": "supplier-lead-time", "path": "rules.md"}),
        ("load_skill", {"skill_name": "stock-cover"}),  # the OpenAI Agents SDK's
        ("exec_command", {"cmd": "cat .agents/stock-cover/SKILL.md .agents/supplier-lead-time/rules.md"}),
        ("Skill", {"skill": "trellis:stock-cover"}),  # Claude Code's
        ("Read", {"file_path": "/tmp/trellis-skills-x/skills/supplier-lead-time/rules.md"}),
        ("Skill", {"name": "stock-cover"}),  # Kimi Code's
        ("ls", {"path": SKILLS_ROOT}),  # a listing reads none
        ("load_skill", {"name": "no-such"}),
        ("stock_level", {"sku": "SKU-00012"}),
    ]  # fmt: skip
    assert b.skills_read(calls, names) == [
        ("stock-cover", None),
        ("supplier-lead-time", None),
        ("supplier-lead-time", "rules.md"),
        ("stock-cover", None),
        ("stock-cover", None),
        ("supplier-lead-time", "rules.md"),
        ("stock-cover", None),
        ("supplier-lead-time", "rules.md"),
        ("stock-cover", None),
    ]


def test_a_runs_events_say_its_own_tools_calls_its_pins_and_its_warnings() -> None:
    def event(kind: RunEventType, call: str | None = None, **data: Any) -> RunEvent:
        return RunEvent(run_id="r", tenant_id="t", type=kind, sequence=0, tool_call_id=call,
                        data=data)  # fmt: skip

    seen = b.Seen()
    seen.events = [
        event(RunEventType.CUSTOM, name="skills", versions={"stock-cover": "t1"}),
        event(RunEventType.TOOL_CALL_START, "c1", tool="Skill"),
        event(RunEventType.TOOL_CALL_ARGS, "c1", args={"skill": "trellis:stock-cover"}),
        event(RunEventType.CUSTOM, name="warning", code="skills_unavailable", message="gone"),
        event(RunEventType.CUSTOM, name="warning", code="other", message="x"),
    ]
    assert seen.made() == [("Skill", {"skill": "trellis:stock-cover"})]
    assert seen.pinned() == {"stock-cover": "t1"} and seen.warnings() == ["gone"]


# --------------------------------------------------------------------------- the store
async def test_the_store_serves_each_level_to_its_owner_and_changes_once_listed() -> None:
    shelf = b.Shelf(tenant="acme", agent="bot", users={"ada": "u-ada", "bob": "u-bob"})
    low = b.Skill("stock-cover", "Cover.", "tenant's")
    for skill in (
        low,
        b.Skill("stock-cover", "Cover.", "ada's", level="user", owner="ada"),
        b.Skill("stock-cover", "Cover.", "the agent's", level="agent"),
    ):
        b.put(skill)(shelf)
    src = skills.combine(shelf)
    scope = skills.Scope("acme", user="u-bob", agent="other")
    assert (await src.pinned(scope))["stock-cover"].version == "t1"
    mine = skills.Scope("acme", user="u-ada", agent="other")
    assert (await src.pinned(mine))["stock-cover"].level == "user"
    assert (await src.pinned(skills.Scope("acme", user="u-ada", agent="bot")))[
        "stock-cover"
    ].level == "agent"
    shelf.once_listed = [b.delete("stock-cover")]  # gone once this listing is given
    fresh = skills.Scope("acme", user="u-carol")
    assert "stock-cover" in await src.pinned(fresh)  # listed, as it was
    with pytest.raises(Exception, match="cannot be read"):
        await src.document("stock-cover", fresh)
    b.down(shelf)
    assert shelf.down


# --------------------------------------------------------------------------- each adapter's slot
def test_each_adapter_is_given_the_skills_in_its_own_slot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agents.sandbox import SandboxAgent

    monkeypatch.setenv("KIMI_RUN_ROOT", str(tmp_path / "kimi"))
    monkeypatch.setenv("CLAUDE_RUN_ROOT", str(tmp_path / "claude"))
    src = skills.combine(b.Shelf(tenant="t", agent="a", users={}))
    model = ScriptedChatModel(turns=[])

    def built(adapter: str, **kw: Any) -> tuple[Any, dict[str, Any]]:
        given = "m" if adapter in ("openai", "claude", "kimi") else model
        return cx.target_of(given, "thread", None, 6, adapter=adapter, skills=src, **kw)

    for adapter, variant in (("react", "main"), ("react", "deep"), ("deepagents", "main")):
        graph, _ = built(adapter, variant=variant)  # Deep Agents' own: its skills middleware
        assert "read_file" in graph.nodes["tools"].bound.tools_by_name, adapter
        assert any("Skills" in n for n in graph.nodes), (adapter, list(graph.nodes))
    langgraph, _ = built("langgraph")  # no slot: the harness's two tools, declared
    assert [type(t) for t in langgraph.trellis_tools if isinstance(t, SkillTools)] == [SkillTools]
    assert not any("Skills" in n for n in langgraph.nodes)
    agent, options = built("openai")  # the SDK's Skills capability, on a sandbox agent
    assert isinstance(agent, SandboxAgent)
    assert [type(c).__name__ for c in agent.capabilities] == ["Shell", "TrellisSkills"]
    assert options["framework_options"]["run_config"].sandbox is not None
    claude, _ = built("claude")
    assert isinstance(claude.plugins[0], ClaudeSkills) and claude.skills == "all"
    kimi, _ = built("kimi")
    assert kimi.skills is src
    plain, _ = cx.target_of("m", "thread", None, 6, adapter="claude")  # none given: as before
    assert plain.plugins == [] and plain.skills is None


# --------------------------------------------------------------------------- a scripted run
def scripted(case_id: str, turns: list[Any]) -> tuple[b.Case, ScriptedChatModel]:
    case = next(c for c in b.cases() if c.id == case_id)
    return case, ScriptedChatModel(turns=turns, spends=(1000, 50))


async def test_a_scripted_run_that_reads_the_skill_and_answers_by_it_passes(offline: None) -> None:
    grade, score = b.grade("SUP-07")
    case, model = scripted("pick-named", [
        ("read_file", {"file_path": f"{SKILLS_ROOT}supplier-scorecard/SKILL.md"}),
        ("supplier_info", {"supplier": "SUP-07"}),
        f"GRADE SUP-07 {grade} {score:.4f}",
    ])  # fmt: skip
    row = await b.run_case(case, model, "t")
    assert row["passed"], row["problems"]
    assert row["read"] == [[["supplier-scorecard", None]]]
    assert set(row["pinned"][0]) == {s.name for s in b.LIBRARY}
    assert row["calls"][0] == [["supplier_info", {"supplier": "SUP-07"}]]
    assert row["model_calls"] == 3 and row["input_tokens"] == 3000 and row["category"] == "pick"
    system = str(model.seen[0][0].content)
    assert "supplier-scorecard" in system and "supplier-lead-time" in system  # Deep Agents' listing


async def test_a_scripted_run_that_answers_without_the_skill_fails(offline: None) -> None:
    case, model = scripted("pick-named", [
        ("supplier_info", {"supplier": "SUP-07"}),
        "SUP-07 scores 0.5667 by the usual weighting: grade B.",
    ])  # fmt: skip
    row = await b.run_case(case, model, "t")
    assert not row["passed"]
    assert row["problems"] == [
        "did not read supplier-scorecard",
        f"answer lacks the line 'GRADE SUP-07 C {b.grade('SUP-07')[1]:.4f}'",
    ]


async def test_an_adapter_with_no_slot_loads_it_with_the_harness_tools(offline: None) -> None:
    case = next(c for c in b.cases() if c.id == "follow-rules-file")
    for adapter in ("langgraph", "graph"):
        model = ScriptedChatModel(turns=[
            ("load_skill", {"name": "supplier-lead-time"}),
            ("load_skill", {"name": "supplier-lead-time", "path": "rules.md"}),
            [("supplier_info", {"supplier": "SUP-13"}), ("supplier_portal", {"supplier": "SUP-13"})],
            f"PROMISE SUP-13 {b.promise('SUP-13')}",
        ], spends=(1000, 50))  # fmt: skip
        row = await b.run_case(case, model, "t", adapter=adapter)
        assert row["passed"], (adapter, row["problems"])
        assert ["supplier-lead-time", "rules.md"] in row["read"][0], adapter


async def test_a_change_the_skill_asks_for_waits_for_its_approval_and_an_edit_meanwhile_is_not_read(
    offline: None,
) -> None:
    """The run pauses at ``set_price``; the skill is edited while it waits; approved, the run
    reads the receipt file of the version it pinned, and the next run reads the edit."""
    case = next(c for c in b.cases() if c.id == "edit-resume-pinned")
    price = round(b.markdown("SKU-01517"), 2)
    receipt = ("read_file", {"file_path": f"{SKILLS_ROOT}markdown-price/receipt.md"})
    model = ScriptedChatModel(turns=[
        ("read_file", {"file_path": f"{SKILLS_ROOT}markdown-price/SKILL.md"}),
        ("sku_details", {"sku": "SKU-01517"}),
        ("set_price", {"sku": "SKU-01517", "price": price}),
        receipt,
        f"MARKDOWN SKU-01517 {price}\nRECEIPT {b.RECEIPTS[0]}",
    ], spends=(1000, 50))  # fmt: skip
    row = await b.run_case(case, model, "t")
    assert row["passed"], row["problems"]
    assert row["decided"] == [["set_price", {"sku": "SKU-01517", "price": price}, "approve"]]
    assert row["ran"] == [["set_price", {"sku": "SKU-01517", "price": price}]]
    told = "\n".join(str(m.content) for call in model.seen for m in call if m.type == "tool")
    assert f"RECEIPT {b.RECEIPTS[0]}" in told and b.RECEIPTS[1] not in told
    assert row["pinned"] == [{**row["pinned"][0], "markdown-price": "t1"}]
    wrong = ScriptedChatModel(turns=[  # the price never asked about: the change did not wait
        ("read_file", {"file_path": f"{SKILLS_ROOT}markdown-price/SKILL.md"}),
        f"MARKDOWN SKU-01517 {price}\nRECEIPT {b.RECEIPTS[1]}",
    ], spends=(1000, 50))  # fmt: skip
    failed = await b.run_case(case, wrong, "t2")
    assert failed["problems"] == [
        f"no set_price call with sku=SKU-01517, price={b.markdown('SKU-01517')} asked for "
        "approval (asked: [])",
        f"answer lacks the line 'RECEIPT {b.RECEIPTS[0]}'",
        f"answer has '{b.RECEIPTS[1]}'",
    ]


async def test_a_skill_deleted_once_pinned_and_a_store_that_is_down_leave_the_run_going(
    offline: None,
) -> None:
    w = r.world()
    stock = sum(w.on_hand("SKU-01517", g) for g in r.REGIONS)
    gone = next(c for c in b.cases() if c.id == "deleted-mid-run")
    model = ScriptedChatModel(turns=[
        ("read_file", {"file_path": f"{SKILLS_ROOT}stock-cover/SKILL.md"}),
        ("stock_level", {"sku": "SKU-01517"}),
        f"I have no rule for low stock here; it has {stock} units on hand.",
    ], spends=(1000, 50))  # fmt: skip
    row = await b.run_case(gone, model, "t")
    assert row["passed"], row["problems"]
    assert row["pinned"][0]["stock-cover"] == "t1"  # pinned, then deleted
    assert any("stock-cover" in x and "cannot be read" in x for x in row["warnings"][0])
    never = next(c for c in b.cases() if c.id == "down-never")
    model = ScriptedChatModel(turns=[("stock_level", {"sku": "SKU-01517"}), f"{stock} units."],
                              spends=(1000, 50))  # fmt: skip
    row = await b.run_case(never, model, "t")
    assert row["passed"], row["problems"]
    assert row["pinned"] == [{}] and any("lists nothing" in x for x in row["warnings"][0])


async def test_the_bench_keeps_each_row_and_stops_at_the_cap(
    tmp_path: Path, offline: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def ran(case: b.Case, model: str, tag: str, **kw: Any) -> dict[str, Any]:
        return {"case": case.id, "category": case.category, "passed": case.id != "pick-none-price",
                "problems": [], "model_calls": 2, "input_tokens": 4000, "output_tokens": 100,
                "seconds": 1.0, "usd": 0.4, "adapter": kw["adapter"], "variant": kw["variant"]}  # fmt: skip

    monkeypatch.setattr(b, "run_case", ran)
    out = tmp_path / "sk"
    args = b.arguments(["--model", "m", "--cases", "pick", "--max-usd", "1.0", "--out", str(out),
                        "--adapter", "claude"])  # fmt: skip
    report = await b.bench(args)
    rows = _kept(Path(f"{out}.jsonl"))
    assert len(rows) == 3 and report["cases"] == 3 and report["usd"] == pytest.approx(1.2)
    assert report["by_category"] == {"pick": "3/3"} and report["adapter"] == "claude"
    assert report["input_tokens_per_case"] == 4000 and _kept(Path(f"{out}.json")) == [report]


def test_the_set_measures_any_adapter_the_complex_suite_wraps_and_react_on_deep_agents() -> None:
    assert b.arguments(["--model", "m"]).adapter == "react"
    for adapter in cx.ADAPTERS:
        assert b.arguments(["--model", "m", "--adapter", adapter]).adapter == adapter
    args = b.arguments(["--model", "m", "--variant", "deep"])
    assert (args.adapter, args.variant) == ("react", "deep")
    with pytest.raises(SystemExit):
        b.arguments(["--model", "m", "--adapter", "nope"])
    with pytest.raises(SystemExit):
        b.arguments([])
