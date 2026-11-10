"""What happens to one tool call: it runs, it runs and is announced, or it waits for a person.

The action follows from the tool's risk — what it does: ``read`` runs, ``write`` and
``irreversible`` ask, and so does a tool that has not said what it does (any other risk: nothing
known is never a reason to run) — unless an approval rule (``approve_when``: an
administrator's, or an accepted approval suggestion) governs the tool: then the call asks
exactly when the rule holds on its arguments. The rule language is the memory service's own
(:mod:`trellis.memory.approval`, which writes and validates these rules): there is one
implementation of it, and governance evaluates rather than re-invents it. A rule that cannot be
evaluated on a call asks, and so does :data:`~trellis.harness.governance.catalog.CATALOG_UNREAD`
— the decision fails closed. A rule that cannot be evaluated is an :class:`ApprovalRuleError`
too: logged once per rule, and carried on the decision (``error``) for the run to record.

A write that asks only because it writes is announced instead where something else already
answers for it (:meth:`Decision.unasked`): a framework's own tool under its own permission flow,
or a harness built-in that keeps to what the run owns. A person asked about a call of a tool the
harness runs picks one of :data:`CHOICES`, as in Claude Code — allow once, allow for this
conversation, deny — and about an ``irreversible`` one of :data:`EVERY_TIME`: it asks every
call (:func:`choices`, :func:`chosen`).
"""

from __future__ import annotations

import dataclasses
import functools
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final

from trellis.contracts import (
    ErrorCategory,
    HarnessError,
    InterruptDecision,
    InterruptRemember,
    Option,
)
from trellis.harness.bounded import SAID_ONCE
from trellis.harness.governance.catalog import CATALOG_UNREAD
from trellis.memory.approval import evaluate

log = logging.getLogger("trellis.harness")

#: Why a call of a tool that has not said what it does asks, after the tool's name.
UNSAID: Final = "has not declared its side effects."


class ApprovalRuleError(HarnessError):
    """A tool's approval rule (``approve_when``) could not be evaluated on a call: the call
    asks a person — the rule never lets a call through it could not check."""

    code = "APPROVAL_RULE_INVALID"
    category = ErrorCategory.POLICY


class Action(StrEnum):
    RUN = "run"
    ANNOUNCE = "announce"
    ASK = "ask"


#: The action of each risk when no rule governs the tool.
ACTIONS: Final[dict[str, Action]] = {
    "read": Action.RUN,
    "write": Action.ASK,
    "irreversible": Action.ASK,
}

#: The choices of a person asked about a call the harness runs. A value is what an answer that
#: picks it carries; "for this conversation" is the contracts' ``remember="run"``, which covers
#: the tool's later calls in every later run of the same thread for the same user.
ONCE: Final = Option(value="once", label="Allow once")
CONVERSATION: Final = Option(value="run", label="Allow for this conversation")
DENY: Final = Option(value="reject", label="Deny")
CHOICES: Final = (ONCE, CONVERSATION, DENY)
#: An ``irreversible`` tool's: never allowed beyond the call asked about.
EVERY_TIME: Final = (ONCE, DENY)
#: What plain chat reads after the question, for each set of choices.
SAID: Final[dict[tuple[Option, ...], str]] = {
    CHOICES: "Allow once, allow for this conversation, or deny.",
    EVERY_TIME: "Allow once, or deny.",
}


def choices(risk: str) -> tuple[Option, ...]:
    """The choices offered on a call of a tool with ``risk``."""
    return EVERY_TIME if risk == "irreversible" else CHOICES


def reach(tool: str, assignee: str | None) -> str:
    """What an approval allowed beyond its call covers: the tool's later calls asked of the
    same person or role (an ``Ask`` naming someone else still asks)."""
    return tool if assignee is None else f"{tool} for {assignee}"


def chosen(value: Any) -> tuple[InterruptDecision, InterruptRemember] | None:
    """The decision a picked choice names, and how far it reaches; ``None``: no choice's
    value."""
    picks: dict[Any, tuple[InterruptDecision, InterruptRemember]] = {
        ONCE.value: (InterruptDecision.APPROVE, "once"),
        CONVERSATION.value: (InterruptDecision.APPROVE, "run"),
        DENY.value: (InterruptDecision.REJECT, "once"),
    }
    return picks.get(value) if isinstance(value, str) else None


def action_of(risk: str) -> Action:
    """The action of ``risk`` when no rule governs the tool: any risk that is none of
    :data:`ACTIONS` — a tool that has not said what it does — asks."""
    return ACTIONS.get(risk, Action.ASK)


@dataclass(frozen=True, slots=True)
class Decision:
    """One call's action, and why (``reason``: what an approver reads). ``risk`` is the tool's
    as governance saw it (the catalog's over the tool's own) and ``rule`` its approval rule,
    if any. A call your code asks about (a ``before_tool`` hook's ``Ask``) also
    says whose it is (``assignee``) and the screen it is reviewed on (``component``,
    ``props``)."""

    tool: str
    args: Mapping[str, Any]
    action: Action
    reason: str
    risk: str
    rule: str | None = None
    assignee: str | None = None
    component: str | None = None
    props: Mapping[str, Any] | None = None
    #: why the rule could not be evaluated, when it could not (the call then asks)
    error: ApprovalRuleError | None = None

    @property
    def asks(self) -> bool:
        return self.action is Action.ASK

    @property
    def announces(self) -> bool:
        return self.action is Action.ANNOUNCE

    @property
    def runs(self) -> bool:
        return self.action is Action.RUN

    @property
    def question(self) -> str:
        """What a person is asked when the call waits for approval."""
        return f"Approve {self.tool}? {self.reason}"

    def asking(
        self,
        reason: str,
        *,
        assignee: str | None = None,
        component: str | None = None,
        props: Mapping[str, Any] | None = None,
    ) -> Decision:
        """This call waiting for a person, for ``reason`` (a hook's ``Ask``: whose it is, and
        the screen it is reviewed on)."""
        return dataclasses.replace(
            self,
            action=Action.ASK,
            reason=reason,
            assignee=assignee,
            component=component,
            props=props,
        )

    def unasked(self) -> Decision:
        """This call as a write announced rather than asked, when only its risk asks (no rule):
        for a framework's own tool, under its own permission flow, and a harness built-in that
        keeps to what the run owns (``Tool.confined``)."""
        if self.asks and self.risk == "write" and self.rule is None:
            return dataclasses.replace(self, action=Action.ANNOUNCE)
        return self

    def approved(self, why: str) -> Decision:
        """This call approved without asking, for ``why`` (a person allowed the tool for this
        conversation, or a schedule's runs): it runs, announced unless it only reads."""
        action = Action.RUN if self.risk == "read" else Action.ANNOUNCE
        return dataclasses.replace(self, action=action, reason=why)


def decide(tool: str, risk: str, rule: str | None, args: Mapping[str, Any]) -> Decision:
    """The action for a call of ``tool`` with ``args``, given its ``risk`` and approval
    ``rule``."""

    def decided(action: Action, reason: str) -> Decision:
        return Decision(tool, args, action, reason, risk, rule)

    if rule is None:
        known = f"{tool} is {risk}."
        return decided(action_of(risk), known if risk in ACTIONS else f"{tool} {UNSAID}")
    if rule == CATALOG_UNREAD:
        return decided(
            Action.ASK,
            f"{tool} is {risk}, and the tool catalog that says when it needs approval could "
            "not be read.",
        )
    try:
        if evaluate(rule, args):
            return decided(Action.ASK, f"{rule}.")
    except Exception as exc:  # fails closed: the call asks, whatever went wrong
        _said_unevaluable(tool, rule, type(exc).__name__)
        error = ApprovalRuleError(
            f"{tool}'s approval rule could not be evaluated: the call asks a person",
            details={"tool": tool, "rule": rule, "reason": type(exc).__name__},
        )
        reason = f"the rule {rule!r} could not be checked on this call."
        return dataclasses.replace(decided(Action.ASK, reason), error=error)
    # the rule decided no approval: the call runs, announced unless it only reads
    return decided(Action.RUN if risk == "read" else Action.ANNOUNCE, "")


@functools.lru_cache(maxsize=SAID_ONCE)  # said once per rule, of the most recent
def _said_unevaluable(tool: str, rule: str, reason: str) -> None:
    log.error(
        "an approval rule could not be evaluated: every call it governs asks a person until "
        "it is fixed",
        extra={"tool": tool, "rule": rule, "reason": reason},
    )
