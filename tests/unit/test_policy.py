from __future__ import annotations

import pytest

from trellis.contracts import ConfigurationError, ToolSpec
from trellis.harness.tools.policy import Condition, Policy, Tier


def spec(name: str = "erp-create_po", side_effects: str = "write") -> ToolSpec:
    return ToolSpec(name=name, side_effects=side_effects)


@pytest.mark.parametrize(
    ("side_effects", "tier"),
    [
        ("read", Tier.AUTO),
        ("write", Tier.NOTIFY),
        ("irreversible", Tier.ASK),
        ("unknown", Tier.NOTIFY),
    ],
)
def test_the_tier_follows_the_side_effects(side_effects: str, tier: Tier) -> None:
    assert Policy().tier(spec(side_effects=side_effects), {})[0] is tier


def test_a_rule_asks_exactly_when_its_condition_holds() -> None:
    policy = Policy({"erp-create_po": "amount > 10000 and currency == 'EUR'"})
    assert policy.tier(spec(), {"amount": 20000, "currency": "EUR"})[0] is Tier.ASK
    assert policy.tier(spec(), {"amount": 20000, "currency": "USD"})[0] is Tier.NOTIFY
    assert policy.tier(spec(side_effects="read"), {"amount": 5, "currency": "EUR"})[0] is Tier.AUTO


def test_a_rule_overrides_an_irreversible_tier_when_it_does_not_hold() -> None:
    policy = Policy({"erp-create_po": "amount > 10000"})
    assert policy.tier(spec(side_effects="irreversible"), {"amount": 1})[0] is Tier.NOTIFY


def test_true_always_asks_and_false_never_does() -> None:
    assert Policy({"erp-create_po": True}).tier(spec(side_effects="read"), {})[0] is Tier.ASK
    assert (
        Policy({"erp-create_po": False}).tier(spec(side_effects="irreversible"), {})[0]
        is Tier.NOTIFY
    )


def test_a_rule_that_cannot_be_checked_asks() -> None:
    tier, why = Policy({"erp-create_po": "amount > 10"}).tier(spec(), {"qty": 3})
    assert tier is Tier.ASK
    assert "could not be checked" in why


@pytest.mark.parametrize(
    ("source", "args", "expected"),
    [
        ("sku in ['a', 'b']", {"sku": "a"}, True),
        ("sku not in ('a',)", {"sku": "a"}, False),
        ("qty * price >= 100", {"qty": 10, "price": 10}, True),
        ("not urgent", {"urgent": False}, True),
        ("-delta < 0", {"delta": 3}, True),
        ("0 < qty <= 5", {"qty": 5}, True),
        ("total % 2 == 1 or flag", {"total": 4, "flag": False}, False),
    ],
)
def test_conditions_evaluate(source: str, args: dict, expected: bool) -> None:
    assert Condition(source).holds(args) is expected


@pytest.mark.parametrize(
    "source", ["__import__('os')", "a.b > 1", "x[0] == 1", "lambda: 1", "amount >"]
)
def test_unsafe_or_broken_rules_fail_when_the_agent_is_wrapped(source: str) -> None:
    with pytest.raises(ConfigurationError):
        Policy({"t": source})
