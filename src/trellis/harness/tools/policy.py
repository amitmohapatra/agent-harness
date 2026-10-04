"""Which calls run, which are announced, and which wait for a person.

A tool's risk tier comes from what it does: its MCP server's annotations (``readOnlyHint`` →
read, ``destructiveHint`` → irreversible, anything else → write) or a local tool's declaration,
and — over both — the tool catalog's ``risk``, which the memory service derives from what an
administrator set and what the harness published. ``read`` runs, ``write`` runs and is
announced (a ``tool_notice`` event), ``irreversible`` asks a person.

The catalog's ``approve_when`` (an administrator's rule, or an accepted approval suggestion)
overrides the tier for its tool: the call asks exactly when the expression holds on its
arguments. The expression language is the memory service's own
(:mod:`trellis.memory.approval`, which writes and validates these rules) — there is one
implementation of it, and the harness evaluates rather than re-invents it. A rule that cannot
be read, or cannot be evaluated on a call, asks: the policy fails closed. So does a tool whose
catalog entry could not be read at all (the memory service unreachable): every tool that does
more than read asks (:data:`CATALOG_UNREAD`) until the catalog answers again.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from typing import TYPE_CHECKING, Any, Final

from trellis.memory.approval import evaluate

if TYPE_CHECKING:
    from trellis.harness.tools.base import Tool


class Tier(Enum):
    AUTO = "auto"
    NOTIFY = "notify"
    ASK = "ask"


#: The approval rule of a tool that does more than read when the catalog could not be read:
#: whether an administrator wants its calls approved is unknown, so every call asks.
CATALOG_UNREAD: Final = "<the tool catalog could not be read>"

TIERS: Final[dict[str, Tier]] = {
    "read": Tier.AUTO,
    "write": Tier.NOTIFY,
    "irreversible": Tier.ASK,
}


def side_effects_of(annotations: Any) -> str:
    """An MCP tool's tier from its server's annotations (bifrost-sdk ``ToolAnnotations``).
    Hints, not guarantees: the catalog has the last word."""
    if annotations is not None and annotations.read_only_hint:
        return "read"
    if annotations is not None and annotations.destructive_hint:
        return "irreversible"
    return "write"


def tier(tool: Tool, args: Mapping[str, Any]) -> tuple[Tier, str]:
    """The tier of this call, and why (the question an approver reads)."""
    effects = tool.spec.side_effects
    rule = tool.approve_when
    if rule is None:
        return TIERS.get(effects, Tier.NOTIFY), f"{tool.name} is {effects}."
    if rule == CATALOG_UNREAD:
        return Tier.ASK, (
            f"{tool.name} is {effects}, and the tool catalog that says when it needs approval "
            "could not be read."
        )
    try:
        if evaluate(rule, args):
            return Tier.ASK, f"{rule}."
    except Exception:
        return Tier.ASK, f"the rule {rule!r} could not be checked on this call."
    # the rule decided no approval: the call runs, announced unless it only reads
    return (Tier.AUTO if effects == "read" else Tier.NOTIFY), ""
