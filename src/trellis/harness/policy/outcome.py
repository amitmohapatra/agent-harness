"""What a policy answer means. Providers keep returning ``True``, ``False`` or a denial
reason; the third answer, ``REQUIRE_APPROVAL``, pauses the run for a person (design §7)."""

from __future__ import annotations

from enum import StrEnum
from typing import Any


class PolicyOutcome(StrEnum):
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"


def normalize(decision: Any) -> tuple[PolicyOutcome, str | None]:
    """``(outcome, reason)`` for any of the shapes a provider may return: a boolean, a
    denial reason, a ``PolicyOutcome`` (or its value), or an ``(outcome, reason)`` pair."""
    if isinstance(decision, tuple) and len(decision) == 2:
        outcome, reason = decision
        return PolicyOutcome(outcome), reason
    if isinstance(decision, bool | type(None)):
        return (PolicyOutcome.ALLOW if decision else PolicyOutcome.DENY), None
    if isinstance(decision, PolicyOutcome):
        return decision, None
    if isinstance(decision, str) and decision in PolicyOutcome.__members__.values():
        return PolicyOutcome(decision), None
    return PolicyOutcome.DENY, str(decision)
