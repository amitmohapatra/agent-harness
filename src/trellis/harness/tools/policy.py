"""Which calls run, which are announced, and which wait for a person.

Risk tiers come from a tool's side effects: ``read`` runs, ``write`` runs and is announced
(a ``tool_notice`` event), ``irreversible`` asks for approval. An ``approve`` rule overrides
the tier for its tool: the call asks exactly when the rule's condition holds on its
arguments (``"amount > 10000"``), and a rule of ``True`` always asks.

Conditions are a small, safe expression language (comparisons, ``and``/``or``/``not``,
arithmetic, ``in``, literals and argument names), parsed when the agent is wrapped so a typo
fails at startup. A condition that cannot be evaluated on a call — a missing argument, a
type error — asks: the policy fails closed.
"""

from __future__ import annotations

import ast
import operator
from collections.abc import Callable, Mapping
from enum import Enum
from typing import Any, Final

from trellis.contracts import ConfigurationError, ToolSpec


class Tier(Enum):
    AUTO = "auto"
    NOTIFY = "notify"
    ASK = "ask"


TIERS: Final[dict[str, Tier]] = {
    "read": Tier.AUTO,
    "write": Tier.NOTIFY,
    "irreversible": Tier.ASK,
}

Rule = bool | str

_COMPARE: Final[dict[type[ast.cmpop], Callable[[Any, Any], bool]]] = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
}
_ARITHMETIC: Final[dict[type[ast.operator], Callable[[Any, Any], Any]]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
}


class Condition:
    """A parsed rule condition, evaluated against a call's arguments."""

    __slots__ = ("source", "tree")

    def __init__(self, source: str) -> None:
        try:
            tree = ast.parse(source, mode="eval").body
        except SyntaxError as exc:
            raise ConfigurationError(f"approve rule {source!r} is not an expression") from exc
        _check(tree, source)
        self.source = source
        self.tree = tree

    def holds(self, args: Mapping[str, Any]) -> bool:
        return bool(_eval(self.tree, args))


class Policy:
    """One agent's approve rules and the tiers."""

    __slots__ = ("rules",)

    def __init__(self, approve: Mapping[str, Rule] | None = None) -> None:
        self.rules: dict[str, Condition | bool] = {
            name: rule if isinstance(rule, bool) else Condition(rule)
            for name, rule in (approve or {}).items()
        }

    def tier(self, spec: ToolSpec, args: Mapping[str, Any]) -> tuple[Tier, str]:
        """The tier of this call, and why (the question an approver reads)."""
        rule = self.rules.get(spec.name)
        if rule is None:
            return TIERS.get(spec.side_effects, Tier.NOTIFY), f"{spec.name} is {spec.side_effects}."
        if rule is True:
            return Tier.ASK, "every call needs approval."
        if isinstance(rule, Condition):
            try:
                if rule.holds(args):
                    return Tier.ASK, f"{rule.source}."
            except Exception:
                return Tier.ASK, f"the rule {rule.source!r} could not be checked on this call."
        # the rule decided no approval: the call runs, announced unless it only reads
        return (Tier.AUTO if spec.side_effects == "read" else Tier.NOTIFY), ""


def _check(node: ast.AST, source: str) -> None:
    allowed = (
        ast.BoolOp,
        ast.And,
        ast.Or,
        ast.UnaryOp,
        ast.Not,
        ast.USub,
        ast.Compare,
        ast.BinOp,
        ast.Name,
        ast.Load,
        ast.Constant,
        ast.List,
        ast.Tuple,
        *_COMPARE,
        *_ARITHMETIC,
    )
    for child in ast.walk(node):
        if not isinstance(child, allowed):
            raise ConfigurationError(
                f"approve rule {source!r}: {type(child).__name__} is not allowed "
                "(comparisons, and/or/not, arithmetic, in, literals and argument names)"
            )


def _eval(node: ast.AST, args: Mapping[str, Any]) -> Any:  # noqa: PLR0911 - one case per node
    match node:
        case ast.Constant(value=value):
            return value
        case ast.Name(id=name):
            return args[name]
        case ast.List(elts=items) | ast.Tuple(elts=items):
            return [_eval(item, args) for item in items]
        case ast.UnaryOp(op=ast.Not(), operand=operand):
            return not _eval(operand, args)
        case ast.UnaryOp(op=ast.USub(), operand=operand):
            return -_eval(operand, args)
        case ast.BoolOp(op=ast.And(), values=values):
            return all(_eval(v, args) for v in values)
        case ast.BoolOp(op=ast.Or(), values=values):
            return any(_eval(v, args) for v in values)
        case ast.BinOp(left=left, op=op, right=right):
            return _ARITHMETIC[type(op)](_eval(left, args), _eval(right, args))
        case ast.Compare(left=left, ops=ops, comparators=comparators):
            current = _eval(left, args)
            for op, comparator in zip(ops, comparators, strict=True):
                right = _eval(comparator, args)
                if not _COMPARE[type(op)](current, right):
                    return False
                current = right
            return True
    raise ValueError(f"cannot evaluate {type(node).__name__}")
