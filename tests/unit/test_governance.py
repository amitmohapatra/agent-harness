"""Governance: the one place a tool call's action is decided (run, announce, ask), the tool
catalog it reads — kept fresh, failing closed — and what it publishes and learns; usable
inside ``h.wrap`` and from any code."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict

from tests.support.catalog import FakeCatalog
from tests.support.memory import FakeMemoryService
from trellis import Harness, Settings
from trellis.contracts import ConfigurationError, HarnessError, ToolCall, ToolSpec
from trellis.harness import writes as writes_module
from trellis.harness.clients.memory import Memory
from trellis.harness.governance import (
    Action,
    ApprovalRuleError,
    Decision,
    Denied,
    Governance,
    Rejected,
    governed,
)
from trellis.harness.governance import catalog as catalog_module
from trellis.harness.governance.catalog import (
    CATALOG_UNREAD,
    GOVERNANCE_STALE_SECONDS,
    GOVERNANCE_TTL_SECONDS,
    TOP_RISK,
    MemoryCatalog,
    Rule,
    entry,
)
from trellis.harness.governance.decision import decide
from trellis.harness.hooks import Deny, Hooks
from trellis.harness.writes import SPOOL_FILE, Writes
from trellis.memory.errors import NotFoundError

ENV = {"MEMORY_URL": "http://memory.test", "TRELLIS_API_KEY": "test"}


# --------------------------------------------------------------------------- the decision


@pytest.mark.parametrize(
    ("risk", "expected"),
    [
        ("read", Action.RUN),
        ("write", Action.ANNOUNCE),
        ("irreversible", Action.ASK),
        ("unknown", Action.ANNOUNCE),
    ],
)
def test_the_action_follows_the_risk(risk: str, expected: Action) -> None:
    decision = decide("erp-create_po", risk, None, {})
    assert decision.action is expected and decision.reason == f"erp-create_po is {risk}."
    assert (decision.runs, decision.announces, decision.asks) == (
        expected is Action.RUN,
        expected is Action.ANNOUNCE,
        expected is Action.ASK,
    )


def test_a_decision_says_what_a_person_is_asked() -> None:
    decision = decide("refund", "irreversible", None, {"amount": 5})
    assert decision == Decision(
        "refund", {"amount": 5}, Action.ASK, "refund is irreversible.", "irreversible"
    )
    assert decision.question == "Approve refund? refund is irreversible."


def test_approve_when_asks_exactly_when_its_condition_holds() -> None:
    rule = 'amount > 10000 and currency == "EUR"'
    held = decide("po", "write", rule, {"amount": 20000, "currency": "EUR"})
    assert held.asks and held.question == f"Approve po? {rule}." and held.rule == rule
    assert decide("po", "write", rule, {"amount": 20000, "currency": "USD"}).announces
    assert decide("po", "read", rule, {"amount": 5, "currency": "EUR"}).runs


def test_approve_when_overrides_an_irreversible_risk_when_it_does_not_hold() -> None:
    assert decide("po", "irreversible", "amount > 10000", {"amount": 1}).announces


def test_true_always_asks() -> None:
    assert decide("po", "read", "true", {}).asks


@pytest.mark.parametrize(
    ("source", "args", "asks"),
    [
        ('sku in ["a", "b"]', {"sku": "a"}, True),
        ('sku in ["b"]', {"sku": "a"}, False),
        ("not urgent", {"urgent": False}, True),
        ("0 < qty", {"qty": 5}, True),
        ('shape == "amount:num:1e4"', {"amount": 20000}, True),
        ("amount > 10", {"qty": 3}, False),  # a missing argument is false, not an error
    ],
)
def test_the_rules_are_the_memory_services_expression_language(
    source: str, args: dict, asks: bool
) -> None:
    """One implementation of ``approve_when``: the service writes and validates these rules,
    ``trellis.memory.approval`` evaluates them (``shape`` is the call's argument shape)."""
    assert decide("po", "write", source, args).asks is asks


@pytest.mark.parametrize("source", ["__import__('os')", "lambda: 1", "amount >", "("])
def test_a_rule_that_cannot_be_read_asks_on_every_call(source: str) -> None:
    """An administrator's rule is data, not code: one that does not parse fails closed."""
    decision = decide("po", "read", source, {"amount": 1})
    assert decision.asks and "could not be checked" in decision.reason


def test_a_rule_that_cannot_be_evaluated_never_allows_is_typed_and_logged_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("ERROR", logger="trellis.harness"):
        first = decide("wire", "read", "amount >>> 1", {"amount": 1})
        again = decide("wire", "read", "amount >>> 1", {"amount": 2})
    assert first.asks and again.asks  # fails closed, even for a tool that only reads
    assert isinstance(first.error, ApprovalRuleError) and isinstance(first.error, HarnessError)
    assert first.error.code == "APPROVAL_RULE_INVALID"
    assert first.error.details == {"tool": "wire", "rule": "amount >>> 1", "reason": "ValueError"}
    said = [r for r in caplog.records if "approval rule could not be evaluated" in r.getMessage()]
    assert len(said) == 1  # once per rule


def test_an_unread_catalog_asks() -> None:
    decision = decide("po", "write", CATALOG_UNREAD, {})
    assert decision.asks and "could not be read" in decision.reason


# --------------------------------------------------------------------------- checking calls


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> Clock:
    clock = Clock()
    monkeypatch.setattr(catalog_module, "_now", clock)
    return clock


async def test_without_a_catalog_the_tools_own_risk_decides() -> None:
    governance = Governance()
    assert (await governance.check("po", {}, side_effects="irreversible")).asks
    assert (await governance.check("po", {})).announces  # an undeclared tool writes
    assert await governance.rules(["po"]) == {"po": None}


async def test_the_catalog_overrides_the_risk_and_sets_the_approval_rule() -> None:
    catalog = FakeCatalog(
        {
            "erp-get_stock": Rule(risk="irreversible"),
            "erp-create_po": Rule(risk="write", approve_when="amount > 10000"),
        }
    )
    governance = Governance(catalog)
    stock = await governance.check("erp-get_stock", {}, side_effects="read")
    assert stock.asks and stock.risk == "irreversible"
    assert (await governance.check("erp-create_po", {"amount": 20000})).asks
    assert (await governance.check("erp-create_po", {"amount": 5})).announces
    assert (await governance.check("other", {}, side_effects="read")).runs  # not in it
    assert await governance.rules(["erp-get_stock", "other"]) == {
        "erp-get_stock": Rule(risk="irreversible"),
        "other": None,
    }


async def test_rules_are_read_again_within_seconds_conditionally(clock: Clock) -> None:
    catalog = FakeCatalog({"po": Rule(risk="write")})
    governance = Governance(catalog)
    assert (await governance.check("po", {"amount": 50})).announces
    clock.now += GOVERNANCE_TTL_SECONDS / 2
    await governance.check("po", {"amount": 50})
    assert len(catalog.asked) == 1  # fresh: not asked again
    clock.now += GOVERNANCE_TTL_SECONDS
    assert (await governance.check("po", {"amount": 50})).announces  # a 304: the same rules
    assert catalog.etags_sent == [None, '"v1"']
    # an administrator adds a rule: the next read (seconds later, not minutes) has it
    catalog.rules["po"] = Rule(risk="write", approve_when="amount > 10")
    catalog.version = 2
    clock.now += GOVERNANCE_TTL_SECONDS + 1
    assert (await governance.check("po", {"amount": 50})).asks


async def test_a_tool_asked_about_for_the_first_time_is_read_at_once_with_the_others(
    clock: Clock,
) -> None:
    catalog = FakeCatalog({"pay": Rule(risk="write", approve_when="amount > 100")})
    governance = Governance(catalog)
    await governance.rules(["stock"])
    assert (await governance.check("pay", {"amount": 500})).asks
    # a new name: a full read (no ETag) of every name asked about
    assert catalog.asked == [["stock"], ["stock", "pay"]] and catalog.etags_sent[-1] is None


async def test_concurrent_calls_share_one_read() -> None:
    catalog = FakeCatalog()
    governance = Governance(catalog)
    await asyncio.gather(*(governance.check("po", {}) for _ in range(10)))
    assert len(catalog.asked) == 1


async def test_an_unreadable_catalog_makes_every_tool_that_does_more_than_read_ask(
    clock: Clock, caplog: pytest.LogCaptureFixture
) -> None:
    catalog = FakeCatalog()
    catalog.down = True
    governance = Governance(catalog)
    with caplog.at_level("WARNING", logger="trellis.governance"):
        assert (await governance.check("erp-get", {}, side_effects="read")).runs
        clock.now += GOVERNANCE_TTL_SECONDS + 1
        for effects in ("write", "irreversible"):  # still down: asked again, warned once
            decision = await governance.check("erp-po", {}, side_effects=effects)
            assert decision.asks and "could not be read" in decision.reason
    assert caplog.text.count("the tool catalog could not be read") == 1
    assert len(catalog.asked) == 2  # the second name was new: read at once
    assert await governance.rules(["erp-po"]) == {"erp-po": None}
    catalog.down = False
    clock.now += GOVERNANCE_TTL_SECONDS + 1
    with caplog.at_level("INFO", logger="trellis.governance"):
        assert (await governance.check("erp-po", {})).announces  # back to its own risk
    assert "can be read again" in caplog.text


async def test_rules_read_earlier_stand_for_a_while_when_the_catalog_goes_down(
    clock: Clock,
) -> None:
    catalog = FakeCatalog({"po": Rule(risk="write", approve_when="amount > 10")})
    governance = Governance(catalog)
    await governance.check("po", {"amount": 1})
    catalog.down = True
    clock.now += GOVERNANCE_TTL_SECONDS + 1
    still = await governance.check("po", {"amount": 1})
    assert still.announces and still.rule == "amount > 10"  # the rule read 31 s ago stands
    clock.now += GOVERNANCE_STALE_SECONDS
    unread = await governance.check("po", {"amount": 1})
    assert unread.asks and unread.rule == CATALOG_UNREAD  # too old to trust: it asks
    catalog.down = False
    clock.now += GOVERNANCE_TTL_SECONDS + 1
    back = await governance.check("po", {"amount": 1})
    assert back.rule == "amount > 10" and catalog.etags_sent[-1] is None


async def test_no_names_ask_the_catalog_nothing() -> None:
    catalog = FakeCatalog()
    assert await Governance(catalog).rules([]) == {}
    assert catalog.asked == []


# --------------------------------------------------------------------------- learned suggestions


async def test_an_accepted_suggestion_stops_a_repeated_pause(
    clock: Clock, caplog: pytest.LogCaptureFixture
) -> None:
    catalog = FakeCatalog({"po": Rule(risk="write", approve_when="amount > 10")})
    governance = Governance(catalog)
    # a person approves, again...
    assert (await governance.check("po", {"amount": 50}, agent_id="buyer")).asks
    catalog.suggest("po", "amount > 1000", agent="buyer")  # ...and the service learns from it
    clock.now += GOVERNANCE_TTL_SECONDS + 1
    with caplog.at_level("INFO", logger="trellis.governance"):
        decision = await governance.check("po", {"amount": 50}, agent_id="buyer")
    assert decision.announces and decision.rule == "amount > 1000"  # no pause any more
    assert catalog.accepted == ["sug_1"]
    [said] = [r for r in caplog.records if "accepted automatically" in r.getMessage()]
    assert said.tool == "po" and said.suggestion_id == "sug_1"  # type: ignore[attr-defined]
    assert said.agent_id == "buyer"  # type: ignore[attr-defined]
    clock.now += GOVERNANCE_TTL_SECONDS + 1
    await governance.check("po", {"amount": 50}, agent_id="buyer")
    assert catalog.accepted == ["sug_1"]  # accepted once


async def test_suggestions_are_read_and_accepted_for_the_agent_they_were_learned_from(
    clock: Clock,
) -> None:
    catalog = FakeCatalog(
        {
            "po": Rule(risk="write", approve_when="amount > 10"),
            "ship": Rule(risk="write", approve_when="true"),
        }
    )
    catalog.suggest("po", "amount > 1000", agent="buyer")
    catalog.suggest("ship", "false", agent="clerk")
    governance = Governance(catalog)
    # with no agent named (a tenant-wide scope) there is nobody's to read: none is accepted
    assert (await governance.check("po", {"amount": 50})).asks
    assert catalog.suggestions_asked == [] and catalog.accepted == []
    # a new agent's are read at once — its own only
    assert (await governance.check("po", {"amount": 50}, agent_id="buyer")).announces
    assert catalog.suggestions_asked == ["buyer"] and catalog.accepted == ["sug_1"]
    # the clerk's rule is the clerk's: read and accepted when the clerk asks
    assert (await governance.check("ship", {}, agent_id="clerk")).announces
    assert catalog.accepted == ["sug_1", "sug_2"]
    # every refresh checks each agent asked about
    clock.now += GOVERNANCE_TTL_SECONDS + 1
    catalog.suggestions_asked.clear()
    await governance.check("po", {"amount": 1}, agent_id="buyer")
    assert sorted(catalog.suggestions_asked) == ["buyer", "clerk"]


async def test_governance_for_one_agent_reads_that_agents_suggestions() -> None:
    catalog = FakeCatalog({"po": Rule(risk="write", approve_when="amount > 10")})
    catalog.suggest("po", "amount > 1000", agent="buyer")
    governance = Governance(catalog, tenant="acme", agent_id="buyer")  # Way 2: one agent's
    assert (await governance.check("po", {"amount": 50})).announces
    assert (await governance.rules(["po"]))["po"] == Rule("write", "amount > 1000")


async def test_the_agents_checked_are_bounded_the_least_recently_asked_forgotten(
    clock: Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("trellis.harness.governance.catalog.AGENTS_KEPT", 2)
    catalog = FakeCatalog({"po": Rule(risk="write")})
    governance = Governance(catalog)
    for agent in ("a", "b", "c"):
        await governance.check("po", {}, agent_id=agent)
    clock.now += GOVERNANCE_TTL_SECONDS + 1
    catalog.suggestions_asked.clear()
    await governance.check("po", {}, agent_id="c")
    assert sorted(catalog.suggestions_asked) == ["b", "c"]


async def test_a_top_risk_tool_is_not_relaxed_automatically(clock: Clock) -> None:
    catalog = FakeCatalog({"wire": Rule(risk=TOP_RISK)})
    catalog.suggest("wire", "amount > 1000")
    catalog.suggest("unasked", "x > 1")  # a tool governance was not asked about
    governance = Governance(catalog)
    assert (await governance.check("wire", {"amount": 5}, agent_id="buyer")).asks
    assert catalog.accepted == []  # a person accepts it, or not


async def test_a_learned_rule_that_stops_asking_never_lifts_a_hooks_deny(clock: Clock) -> None:
    """A learned rule only changes when governance asks (``approve_when``); a ``before_tool``
    hook's ``Deny`` still stops the call, unrun, whatever the rule learned."""
    catalog = FakeCatalog({"po": Rule(risk="write", approve_when="amount > 10")})
    catalog.suggest("po", "amount > 1000", agent="buyer")
    governance = Governance(catalog, tenant="acme", agent_id="buyer")
    assert (await governance.check("po", {"amount": 50})).announces, "the learned rule holds"

    class Frozen(Hooks):
        async def before_tool(self, call: ToolCall) -> Deny:
            return Deny("purchase orders are frozen")

    ran: list[int] = []

    def po(amount: int) -> str:
        ran.append(amount)
        return "ordered"

    call = governed(po, governance, on_ask=lambda decision: True, hooks=[Frozen()])
    with pytest.raises(Denied, match="frozen"):
        await call(amount=50)
    assert ran == []


async def test_suggestions_that_cannot_be_read_or_accepted_wait_for_the_next_refresh(
    clock: Clock, caplog: pytest.LogCaptureFixture
) -> None:
    catalog = FakeCatalog({"po": Rule(risk="write", approve_when="amount > 10")})
    catalog.suggest("po", "amount > 1000")
    catalog.suggestions_fail = {"read"}
    governance = Governance(catalog, agent_id="buyer")
    with caplog.at_level("WARNING", logger="trellis.governance"):
        for _ in range(2):
            assert (await governance.check("po", {"amount": 50})).asks
            clock.now += GOVERNANCE_TTL_SECONDS + 1
        catalog.suggestions_fail = {"accept"}
        assert (await governance.check("po", {"amount": 50})).asks
    unread = [r for r in caplog.records if "could not be read" in r.getMessage()]
    assert len(unread) == 1  # said once while they cannot be read
    assert any("could not be accepted" in r.getMessage() for r in caplog.records)
    catalog.suggestions_fail = set()
    clock.now += GOVERNANCE_TTL_SECONDS + 1
    assert (await governance.check("po", {"amount": 50})).announces


async def test_the_memory_catalog_lists_and_accepts_suggestions() -> None:
    suggestion = {
        "id": "sug_1",
        "tool": "po",
        "arg_shape": "{}",
        "suggestion": "auto_approve",
        "approvals": 5,
        "rejections": 0,
        "edits": 0,
        "support": 5,
        "approve_rate": 1.0,
        "accepted": False,
        "agent_id": "buyer",
        "rule": "amount > 1000",
    }
    service = FakeMemoryService(
        catalog={"po": {"risk": "write", "approve_when": "amount > 10"}}, suggestions=[suggestion]
    )
    # tenant-wide: the scope the harness's governance is built with
    catalog = MemoryCatalog(service.client().bind(tenant_id="acme"))
    assert await catalog.suggestions("clerk") == []  # another agent's decisions taught nothing
    with pytest.raises(NotFoundError):  # and it cannot accept the buyer's
        await catalog.accept("sug_1", agent_id="clerk")
    [found] = await catalog.suggestions("buyer")
    assert found.id == "sug_1" and not found.accepted
    accepted = await catalog.accept("sug_1", agent_id="buyer")
    assert accepted == Rule(risk="write", approve_when="amount > 1000")
    assert [s.accepted for s in await catalog.suggestions("buyer")] == [True]
    asked = [c for c in service.calls if c.name in ("suggestions", "accept")]
    assert {c.scope.get("agent_id") for c in asked} == {"clerk", "buyer"}
    # a catalog already in the agent's scope is used as it is
    own = MemoryCatalog(service.client().bind(tenant_id="acme", agent_id="buyer"))
    assert own._agent("buyer") is own.ctx


# --------------------------------------------------------------------------- the memory service


async def test_the_memory_catalog_says_what_it_knows_conditionally() -> None:
    service = FakeMemoryService(
        catalog={"erp-get_stock": {"risk": "read", "approve_when": "qty > 5"}}
    )
    catalog = MemoryCatalog(service.client().bind(tenant_id="acme"))
    found, etag = await catalog.read(["erp-get_stock", "missing"])
    assert found == {"erp-get_stock": Rule(risk="read", approve_when="qty > 5")}
    assert etag is not None
    # asked again with that ETag: nothing changed, nothing sent back
    assert await catalog.read(["erp-get_stock", "missing"], etag=etag) == (None, etag)
    service.catalog["erp-get_stock"]["approve_when"] = "qty > 9"
    changed, newer = await catalog.read(["erp-get_stock", "missing"], etag=etag)
    assert changed == {"erp-get_stock": Rule(risk="read", approve_when="qty > 9")}
    assert newer not in (None, etag)
    service.etags = False  # a service that sends no ETag is read in full every time
    assert (await catalog.read(["erp-get_stock"], etag=newer))[1] is None


async def test_the_memory_catalog_says_whether_tool_search_finds_the_tools() -> None:
    """B13: what the service lists as not indexed yet is waited for; a service that does not
    say (``indexed`` absent) is taken as ready, and so is a tool it does not list."""
    service = FakeMemoryService(catalog={"a": {"indexed": True}, "b": {"indexed": False}})
    catalog = MemoryCatalog(service.client().bind(tenant_id="acme"))
    assert await catalog.indexed(["a", "missing"]) is True
    assert await catalog.indexed(["a", "b"]) is False
    service.catalog["b"] = {}
    assert await catalog.indexed(["a", "b"]) is True


async def test_governance_says_indexed_once_nothing_is_still_being_published() -> None:
    assert await Governance().indexed(["refund"]) is True, "no catalog: nothing to wait for"
    catalog, sends = FakeCatalog(), []

    async def submit(entries: list[dict[str, object]], send: Any) -> None:
        sends.append(send)  # held: still being published

    governance = Governance(catalog, submit=submit)
    await governance.publish(SPECS)
    assert await governance.indexed(["refund"]) is False and catalog.index_asked == []
    await sends[0]()
    assert await governance.indexed(["refund"]) is True and catalog.index_asked == [["refund"]]


async def test_catalog_entries_carry_what_is_known_and_no_more() -> None:
    service = FakeMemoryService()
    await MemoryCatalog(service.client().bind(tenant_id="t")).publish(
        [
            entry(ToolSpec(name="refund", side_effects="irreversible", source="local"), None),
            entry(
                ToolSpec(name="erp-get", source="mcp", server="erp", side_effects="write"),
                {"readOnlyHint": True},
            ),
        ]
    )
    refund, mcp_tool = service.named("put_catalog")[0].body["tools"]
    assert refund["side_effects"] == "irreversible" and "annotations" not in refund
    assert mcp_tool["annotations"] == {"readOnlyHint": True} and "side_effects" not in mcp_tool


# --------------------------------------------------------------------------- publishing


SPECS = [ToolSpec(name="refund", side_effects="irreversible", source="local")]


async def test_a_tool_is_published_once_per_content() -> None:
    catalog = FakeCatalog()
    governance = Governance(catalog)
    await governance.publish(SPECS)
    await governance.publish(SPECS)  # the same content: not sent again
    changed = [SPECS[0].model_copy(update={"description": "Refund an order."})]
    await governance.publish(changed)
    assert [[e["name"] for e in sent] for sent in catalog.published] == [["refund"], ["refund"]]
    assert catalog.published[1][0]["description"] == "Refund an order."


async def test_a_failed_publish_raises_and_is_sent_again_next_time() -> None:
    catalog = FakeCatalog()
    catalog.refuse_publish = 1
    governance = Governance(catalog)
    with pytest.raises(Exception, match="not stored"):
        await governance.publish(SPECS)
    await governance.publish(SPECS)
    assert len(catalog.published) == 1


async def test_a_publish_goes_through_submit_when_given() -> None:
    catalog, submitted = FakeCatalog(), []

    async def submit(entries: list[dict[str, object]], send: Any) -> None:
        submitted.append(entries)
        await send()

    governance = Governance(catalog, submit=submit)
    await governance.publish(SPECS, annotations={"refund": {"destructiveHint": True}})
    assert submitted == catalog.published
    assert submitted[0][0]["annotations"] == {"destructiveHint": True}


async def test_without_a_catalog_nothing_is_published_or_learned() -> None:
    governance = Governance()
    await governance.publish(SPECS)
    await governance.decided(
        decide("refund", "irreversible", None, {}), "approve", reviewer="u", run_id="r", user="u"
    )
    await governance.aclose()


# --------------------------------------------------------------------------- governed


def irreversible_catalog() -> Governance:
    return Governance(FakeCatalog({"create_po": Rule(risk="irreversible")}))


async def create_po(sku: str, qty: int) -> str:
    return f"{qty} x {sku}"


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync-on-ask", "async-on-ask"])
@pytest.mark.parametrize(
    ("answer", "expected"),
    [(True, "5 x A-1"), ({"sku": "A-1", "qty": 2}, "2 x A-1")],
    ids=["approve", "edit"],
)
async def test_a_governed_call_that_asks_runs_as_the_person_decides(
    asynchronous: bool, answer: Any, expected: str
) -> None:
    asked: list[Decision] = []

    def sync(decision: Decision) -> Any:
        asked.append(decision)
        return answer

    async def async_(decision: Decision) -> Any:
        return sync(decision)

    call = governed(create_po, irreversible_catalog(), on_ask=async_ if asynchronous else sync)
    assert await call(sku="A-1", qty=5) == expected
    [decision] = asked
    assert decision.tool == "create_po" and decision.args == {"sku": "A-1", "qty": 5}
    assert call.__name__ == "create_po"


async def test_a_rejected_call_raises_and_does_not_run() -> None:
    ran: list[int] = []

    def po(qty: int) -> int:
        ran.append(qty)
        return qty

    call = governed(po, irreversible_catalog(), name="create_po", on_ask=lambda d: False)
    with pytest.raises(Rejected, match="create_po was not run") as rejected:
        await call(qty=5)
    assert rejected.value.decision.asks and ran == []


async def test_an_answer_that_is_no_decision_is_refused() -> None:
    call = governed(create_po, irreversible_catalog(), on_ask=lambda d: "yes")
    with pytest.raises(TypeError, match="on_ask answered 'yes'"):
        await call(sku="A-1", qty=5)


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_a_governed_call_that_is_announced_tells_on_announce_first(
    asynchronous: bool,
) -> None:
    heard: list[str] = []

    def sync(decision: Decision) -> None:
        heard.append(decision.tool)

    async def async_(decision: Decision) -> None:
        sync(decision)

    def note(text: str) -> str:  # a sync tool
        heard.append("ran")
        return text

    governance = Governance()
    on_announce = async_ if asynchronous else sync
    call = governed(note, governance, on_ask=lambda d: True, on_announce=on_announce)
    assert await call(text="hi") == "hi" and heard == ["note", "ran"]
    quiet = governed(note, governance, on_ask=lambda d: True)  # no on_announce: it just runs
    assert await quiet(text="hi") == "hi"
    read = governed(note, governance, side_effects="read", on_ask=lambda d: True)
    assert await read(text="x") == "x" and heard == ["note", "ran", "ran", "ran"]


class State(TypedDict):
    sku: str
    result: str


async def test_a_langgraph_node_pauses_a_call_that_asks_with_its_own_interrupt() -> None:
    """The pluggable recipe: ``on_ask`` is LangGraph's ``interrupt``, so the graph's
    checkpointer keeps the pause and ``Command(resume=...)`` is the person's decision."""
    governance = irreversible_catalog()
    call = governed(create_po, governance, on_ask=lambda d: interrupt(d.question))

    async def order(state: State) -> dict[str, str]:
        return {"result": await call(sku=state["sku"], qty=5)}

    builder = StateGraph(State)
    builder.add_node("order", order)
    builder.add_edge(START, "order")
    builder.add_edge("order", END)
    graph = builder.compile(checkpointer=InMemorySaver())
    config: Any = {"configurable": {"thread_id": "t1"}}
    paused = await graph.ainvoke({"sku": "A-1", "result": ""}, config)
    [asked] = paused["__interrupt__"]
    assert asked.value == "Approve create_po? create_po is irreversible."
    done = await graph.ainvoke(Command(resume=True), config)
    assert done["result"] == "5 x A-1"


# --------------------------------------------------------------------------- decided


async def test_a_decision_is_feedback_the_memory_service_learns_from() -> None:
    service = FakeMemoryService()
    governance = Governance(
        MemoryCatalog(service.client().bind(tenant_id="acme")), tenant="acme", agent_id="buyer"
    )
    decision = await governance.check("create_po", {"qty": 50}, side_effects="irreversible")
    for _ in range(2):  # a retried send is stored once
        await governance.decided(
            decision, "edit", reviewer="user:lead", run_id="run_1", user="ada", edited={"qty": 5}
        )
    sent = service.named("feedback")
    assert len(sent) == 2 and len(service.stored_feedback) == 1
    body = sent[0].body
    assert (body["target_kind"], body["verdict"], body["source"]) == (
        "tool_call",
        "edit",
        "interrupt",
    )
    assert (body["tenant_id"], body["agent_id"], body["agent_run_id"], body["user_id"]) == (
        "acme",
        "buyer",
        "run_1",
        "ada",
    )
    assert body["reviewer"] == "user:lead" and body["correction"] == {"qty": 5}
    assert body["metadata"]["tool"] == "create_po" and body["metadata"]["args"] == {"qty": 50}
    assert sent[0].idempotency_key == body["feedback_id"]


async def test_a_decision_needs_a_verdict_an_agent_and_a_tenant() -> None:
    decision = decide("po", "irreversible", None, {})
    governance = Governance(FakeCatalog())
    with pytest.raises(ValueError, match="approve, reject or edit"):
        await governance.decided(decision, "answer", reviewer="u", run_id="r", user="u")  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="agent_id= and tenant="):
        await governance.decided(decision, "approve", reviewer="u", run_id="r", user="u")
    catalog = FakeCatalog()
    named = Governance(catalog, tenant="acme", agent_id="buyer")
    await named.decided(decision, "reject", reviewer="user:lead", run_id="run_1", user="ada")
    [record] = catalog.feedback_sent
    assert record.verdict == "reject" and record.correction is None


# --------------------------------------------------------------------------- from the environment


async def test_from_an_environment_without_memory_the_tools_own_risks_decide() -> None:
    governance = Governance.from_env(agent_id="buyer", environ={})
    assert governance.catalog is None and governance.agent_id == "buyer"
    assert (await governance.check("po", {}, side_effects="irreversible")).asks
    await governance.aclose()


async def test_from_an_environment_with_memory_the_catalog_decides() -> None:
    governance = Governance.from_env(agent_id="buyer", tenant="acme", environ=ENV)
    assert isinstance(governance.catalog, MemoryCatalog)
    assert governance.catalog.ctx.scope.tenant_id == "acme" and governance.tenant == "acme"
    keyed = Governance.from_env(environ=ENV)  # the key's own tenant
    assert isinstance(keyed.catalog, MemoryCatalog) and keyed.catalog.ctx.scope.tenant_id is None
    await governance.aclose()
    await keyed.aclose()


def test_memory_without_a_key_is_a_configuration_error() -> None:
    with pytest.raises(ConfigurationError, match="TRELLIS_API_KEY"):
        Governance.from_env(environ={"MEMORY_URL": "http://memory.test"})


# --------------------------------------------------------------------------- inside the harness


async def test_the_harness_keeps_one_governance_per_tenant(harness: Harness) -> None:
    acme = harness._governance("acme")
    assert harness._governance("acme") is acme and harness._governance("globex") is not acme
    assert acme.catalog is None and acme.tenant == "acme"  # memory off: the tools' own risks


async def test_with_memory_a_tenants_catalog_publishes_through_the_background_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(writes_module, "WRITE_BACKOFF_SECONDS", 0.0)
    service = FakeMemoryService()
    service.fail.add("put_catalog")  # unavailable: the write is kept in the spool
    settings = Settings(spool_dir=str(tmp_path))
    async with Harness(config=settings, memory=service.client()) as h:
        governance = h._governance("acme")
        assert isinstance(governance.catalog, MemoryCatalog)
        assert governance.catalog.ctx.scope.tenant_id == "acme"
        await governance.publish(SPECS)
        await h.writes.drain()
    [kept] = [json.loads(line) for line in (tmp_path / SPOOL_FILE).read_text().splitlines()]
    assert (kept["label"], kept["op"], kept["scope"]) == (
        "memory.tool_catalog",
        "publish_catalog",
        {"tenant_id": "acme", "custom_metadata": {}},
    )
    service.fail.clear()
    replay = Memory(service.client()).replay
    writes = Writes(spool=tmp_path, replay=replay)
    writes.start()  # the next start replays the spooled publish
    await writes.drain()
    [put] = service.named("put_catalog")
    assert put.body["tools"][0]["name"] == "refund"
    await writes.aclose()
