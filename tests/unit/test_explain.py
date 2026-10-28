"""The views of a run's record (``trellis.harness.explain``): the end user's is built from the
record by the harness, and never holds what is not theirs to see — a tool's name or arguments,
a raw output, an id, a reviewer, a governance rule, tokens, cost, an error trace or a score —
whatever the record holds; the administrator's holds all of it."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from trellis.contracts import ConfigurationError, RunRecord, RunStart, RunStatus, ToolSpec
from trellis.harness import explain
from trellis.harness.explain import (
    Record,
    Step,
    admin_view,
    client_view,
    offered,
    without_lookup,
)
from trellis.harness.tools.base import Citation, citations
from trellis.harness.tools.sources import tool

ASKED = datetime(2026, 10, 28, 12, 0, tzinfo=UTC)
#: what the record holds that the end user never sees
FORBIDDEN = (
    "wire_funds",
    "stock_quote",
    "acct-991",
    '"account"',
    "RAW-PAYLOAD-xyz",
    "TimeoutError",
    "run_secret_1",
    "acme-tenant",
    "priya",
    "only managers approve wires",
    "mem_42",
    "intr_7",
    "4711",  # tokens
    "0.0931",  # cost
    "claude-opus",
)


def run(
    status: RunStatus = RunStatus.SUCCESS, output: Any = "ACME closed at 123.45 USD.", **kw: Any
) -> RunRecord:
    start = RunStart(
        run_id="run_secret_1",
        tenant_id="acme-tenant",
        agent_id="stock",
        thread_id="t",
        user_id="u1",
        input={"messages": [{"role": "user", "content": "What did ACME close at?"}]},
    )
    fields = {
        "status": status,
        "output": output,
        "worked_seconds": kw.pop("worked", 2.5),
        "created_at": ASKED,
        "updated_at": ASKED + timedelta(seconds=kw.pop("wall", 2.5)),
        **kw,
    }
    return RunRecord.from_start(start).model_copy(update=fields)


def step(tool: str = "stock_quote", status: str = "ok", **kw: Any) -> Step:
    return Step(
        tool=tool,
        args={"account": "acct-991"},
        status=status,
        said=kw.pop("said", "The latest closing quote of a stock."),
        output="RAW-PAYLOAD-xyz",
        error_class="TimeoutError" if status != "ok" else None,
        **kw,
    )


NASDAQ = Citation(name="Nasdaq Last Sale feed", date="2026-10-27", link="https://nasdaq.test")


def full() -> Record:
    return Record(
        run_id="run_secret_1",
        steps=[
            step(sources=[NASDAQ]),
            step("wire_funds", "timeout", said="Make a change"),
            step("wire_funds", "rejected"),
            step("stock_quote", "cancelled"),
            step("stock_quote", "error"),
        ],
        approvals=[
            {
                "interrupt_id": "intr_7",
                "tool": "wire_funds",
                "decision": "EDIT",
                "assignee": "role:finance",
                "reviewer": "priya",
                "comment": "only managers approve wires",
            },
            {"tool": "wire_funds", "decision": "APPROVE", "assignee": None, "reviewer": "priya"},
        ],
        memories=["mem_42"],
        evidence="INCOMPLETE",
        spent={"tokens": 4711, "cost": 0.0931},
    )


def test_the_clients_view_never_holds_what_is_not_theirs_whatever_the_record_holds() -> None:
    seen = client_view(run(output="stock_quote said 123.45"), full(), ["I hold ACME."])
    for leak in FORBIDDEN:
        assert leak not in seen, leak
    assert (
        "1. The latest closing quote of a stock. — from Nasdaq Last Sale feed, 2026-10-27" in seen
    )
    assert "2. Make a change — this step did not complete (it ran out of time)" in seen
    assert "(it was not allowed)" in seen and "(it was stopped)" in seen and "(it failed)" in seen
    assert "- a reviewer (finance) approved with changes step 2." in seen
    assert "- a reviewer approved step 2." in seen
    assert "Memory: your saved notes and history covered only part of the question." in seen
    assert "Your saved notes I used:\n- I hold ACME." in seen
    assert "Limits: step(s) 2, 3, 4, 5 did not complete; it rests on a single source;" in seen
    assert 'Answer given: "said 123.45"' in seen  # the tool's name taken out of it


def test_the_administrators_view_holds_everything() -> None:
    seen = admin_view(run(), full(), [])
    shown = json.loads(seen)
    for kept in ("wire_funds", "acct-991", "RAW-PAYLOAD-xyz", "priya", "intr_7", "mem_42"):
        assert kept in seen, kept
    assert shown["record"]["spent"] == {"tokens": 4711, "cost": 0.0931}
    assert shown["run"]["run_id"] == "run_secret_1"
    assert shown["record"]["steps"][0]["sources"] == [NASDAQ.model_dump()]
    assert shown["run"]["output"] and shown["record"]["steps"][0]["output"] == "RAW-PAYLOAD-xyz"
    windowed = json.loads(admin_view(run(), full(), [], in_window=True))
    assert "output" not in windowed["run"] and "output" not in windowed["record"]["steps"][0]
    assert json.loads(admin_view(run(), None, [], in_window=True))["record"] is None


def test_what_the_window_carries_is_not_said_twice() -> None:
    seen = client_view(run(), full(), [], in_window=True)
    assert "Earlier question" not in seen and "Answer given" not in seen
    assert seen.startswith("When: 2026-10-28 12:00 UTC.")


def test_a_run_with_no_steps_no_source_and_a_long_wait() -> None:
    seen = client_view(run(worked=75, wall=600), Record(run_id="r"), [])
    assert "It took 1 min 15 s of work, 10 min 0 s in all (waiting included)." in seen
    assert "What I did: I answered without looking anything up (0 steps)." in seen
    assert "Sources: no step returned a source, so I cannot name one." in seen
    assert (
        "Basis: no step returned anything, so the answer rests on the conversation alone." in seen
    )
    assert "Approvals: none was needed." in seen and "Memory:" not in seen
    assert "Limits: no source was recorded, so it could not be checked against one." in seen


def test_sources_and_their_freshness() -> None:
    sources = [
        Citation(name="Same day", date="2026-10-28T09:00:00+00:00"),
        Citation(name="Naive", date="2026-10-20"),
        Citation(name="Future", date="2026-11-30"),
        Citation(name="Words", date="last Tuesday"),
        Citation(name="Undated"),
    ]
    record = Record(run_id="r", steps=[step(sources=sources)], evidence="COMPLETE")
    seen = client_view(run(), record, [])
    assert "- Same day, 2026-10-28T09:00:00+00:00 (the same day as the answer)" in seen
    assert "- Naive, 2026-10-20 (8 day(s) before the answer)" in seen
    assert "- Future, 2026-11-30\n" in seen and "- Words, last Tuesday\n" in seen
    assert "Limits: Undated gave no date, so how current it is is unknown." in seen
    assert (
        "Basis: the answer rests on what step(s) 1 returned; step(s) 1 named their source." in seen
    )


@pytest.mark.parametrize(
    ("status", "said"),
    [
        (RunStatus.PARTIAL, 'Answer given: "half" (cut short: it reached its spending limit)'),
        (RunStatus.ERROR, "Answer given: none: that request ended error before an answer"),
    ],
)
def test_an_answer_cut_short_or_never_given(status: RunStatus, said: str) -> None:
    assert said in client_view(run(status, "half"), Record(run_id="r"), [])


def test_insufficient_memory_is_a_limit() -> None:
    seen = client_view(run(), Record(run_id="r", evidence="INSUFFICIENT"), [])
    assert "Memory: your saved notes and history held nothing about the question." in seen
    assert seen.endswith("your saved notes and history held nothing about the question.")


def test_a_step_in_plain_words_is_its_tools_description_without_any_tools_name() -> None:
    names = {"quote", "quote_v2", "getQuote", "crm.wire", "wire"}
    spec = ToolSpec(name="quote_v2", description="Call quote_v2 or getQuote, then crm.wire.")
    assert explain._said(spec, names) == "Call or , then ."
    plain = ToolSpec(name="quote", description="Get a quote for a stock.")
    assert explain._said(plain, names) == "Get a quote for a stock."  # a plain word stays
    bare = ToolSpec(name="crm.wire", description="crm.wire", side_effects="irreversible")
    assert explain._said(bare, names) == "Made a change that cannot be undone"
    assert explain._said(None, names) == "Made a change"
    hindi = ToolSpec(name="bhav", description="किसी शेयर का भाव। और।", side_effects="read")
    assert explain._said(hindi, names) == "किसी शेयर का भाव।"


def test_who_decided_is_said_by_role_only() -> None:
    assert explain._role("role:finance") == "a reviewer (finance)"
    assert explain._role("user:priya") == "a reviewer"
    assert explain._role(None) == "a reviewer"
    assert explain._principal("u1") == "user:u1" and explain._principal("agent:x") == "agent:x"


def test_the_lookup_is_offered_unless_measured_without() -> None:
    assert offered()
    with without_lookup():
        assert not offered()
    assert offered()


def test_an_artifact_names_sources_only_as_citations() -> None:
    assert citations(NASDAQ) == [NASDAQ]
    assert citations([{"name": "a"}, {"name": "b", "link": "x"}]) == [
        Citation(name="a"),
        Citation(name="b", link="x"),
    ]
    assert citations({"rows": 1}) == [] and citations(None) == [] and citations([]) == []


def test_a_tools_response_format_is_langchains() -> None:
    with pytest.raises(ConfigurationError, match="content_and_artifact"):
        tool(lambda: "x", response_format="artifact")  # type: ignore[arg-type]
    assert tool(lambda: "x").response_format == "content"


def test_the_earlier_answers_tool_is_a_read() -> None:
    made = explain.earlier_tool()
    assert made.spec.side_effects == "read" and made.name == explain.EARLIER


def test_two_dated_sources_and_every_step_done_show_no_limit() -> None:
    other = Citation(name="Exchange tape", date="2026-10-27")
    record = Record(run_id="r", steps=[step(sources=[NASDAQ]), step(sources=[other])])
    assert client_view(run(), record, []).endswith("Limits: none that the record shows.")
    unsourced = client_view(run(), Record(run_id="r", steps=[step()]), [])
    assert "Basis: the answer rests on what step(s) 1 returned.\n" in unsourced
