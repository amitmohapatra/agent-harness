from __future__ import annotations

from types import SimpleNamespace

import pytest

from trellis.contracts import ToolSpec
from trellis.harness.tools.base import Tool
from trellis.harness.tools.policy import Tier, side_effects_of, tier


async def _run(args: dict) -> None:
    return None


def made(side_effects: str = "write", approve_when: str | None = None) -> Tool:
    spec = ToolSpec(name="erp-create_po", side_effects=side_effects)
    return Tool(spec, _run, approve_when=approve_when)


@pytest.mark.parametrize(
    ("side_effects", "expected"),
    [
        ("read", Tier.AUTO),
        ("write", Tier.NOTIFY),
        ("irreversible", Tier.ASK),
        ("unknown", Tier.NOTIFY),
    ],
)
def test_the_tier_follows_the_side_effects(side_effects: str, expected: Tier) -> None:
    assert tier(made(side_effects), {})[0] is expected


@pytest.mark.parametrize(
    ("hints", "expected"),
    [
        (None, "write"),
        ({"read_only_hint": True}, "read"),
        ({"read_only_hint": True, "destructive_hint": True}, "read"),
        ({"destructive_hint": True}, "irreversible"),
        ({"destructive_hint": False, "idempotent_hint": True}, "write"),
    ],
)
def test_annotations_give_an_mcp_tool_its_side_effects(hints: dict | None, expected: str) -> None:
    annotations = None
    if hints is not None:
        annotations = SimpleNamespace(read_only_hint=None, destructive_hint=None)
        for name, value in hints.items():
            setattr(annotations, name, value)
    assert side_effects_of(annotations) == expected


def test_approve_when_asks_exactly_when_its_condition_holds() -> None:
    rule = 'amount > 10000 and currency == "EUR"'
    assert tier(made(approve_when=rule), {"amount": 20000, "currency": "EUR"})[0] is Tier.ASK
    assert tier(made(approve_when=rule), {"amount": 20000, "currency": "USD"})[0] is Tier.NOTIFY
    read = made("read", rule)
    assert tier(read, {"amount": 5, "currency": "EUR"})[0] is Tier.AUTO


def test_approve_when_overrides_an_irreversible_tier_when_it_does_not_hold() -> None:
    assert tier(made("irreversible", "amount > 10000"), {"amount": 1})[0] is Tier.NOTIFY


def test_true_always_asks() -> None:
    assert tier(made("read", "true"), {})[0] is Tier.ASK


@pytest.mark.parametrize(
    ("source", "args", "expected"),
    [
        ('sku in ["a", "b"]', {"sku": "a"}, Tier.ASK),
        ('sku in ["b"]', {"sku": "a"}, Tier.NOTIFY),
        ("not urgent", {"urgent": False}, Tier.ASK),
        ("0 < qty", {"qty": 5}, Tier.ASK),
        ('shape == "amount:num:1e4"', {"amount": 20000}, Tier.ASK),
        ("amount > 10", {"qty": 3}, Tier.NOTIFY),  # a missing argument is false, not an error
    ],
)
def test_the_rules_are_the_memory_services_expression_language(
    source: str, args: dict, expected: Tier
) -> None:
    """One implementation of ``approve_when``: the service writes and validates these rules,
    ``trellis.memory.approval`` evaluates them (``shape`` is the call's argument shape)."""
    assert tier(made(approve_when=source), args)[0] is expected


@pytest.mark.parametrize("source", ["__import__('os')", "lambda: 1", "amount >", "("])
def test_a_rule_that_cannot_be_read_asks_on_every_call(source: str) -> None:
    """An administrator's rule is data, not code: one that does not parse fails closed."""
    chosen, why = tier(made("read", source), {"amount": 1})
    assert chosen is Tier.ASK and "could not be checked" in why
