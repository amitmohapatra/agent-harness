"""What happens to one tool call: it runs, it runs and is announced, or it waits for a person.

The action follows from the tool's risk — what it does: ``read`` runs, ``write`` runs and is
announced, ``irreversible`` asks — unless an approval rule (``approve_when``: an
administrator's, or an accepted approval suggestion) governs the tool: then the call asks
exactly when the rule holds on its arguments. The rule language is the memory service's own
(:mod:`trellis.memory.approval`, which writes and validates these rules): there is one
implementation of it, and governance evaluates rather than re-invents it. A rule that cannot be
evaluated on a call asks, and so does :data:`~trellis.harness.governance.catalog.CATALOG_UNREAD`
— the decision fails closed.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final

from trellis.harness.governance.catalog import CATALOG_UNREAD
from trellis.memory.approval import evaluate


class Action(StrEnum):
    RUN = "run"
    ANNOUNCE = "announce"
    ASK = "ask"


#: The action of each risk when no rule governs the tool (an unknown risk is announced).
ACTIONS: Final[dict[str, Action]] = {
    "read": Action.RUN,
    "write": Action.ANNOUNCE,
    "irreversible": Action.ASK,
}


@dataclass(frozen=True, slots=True)
class Decision:
    """One call's action, and why (``reason``: what an approver reads). ``risk`` is the tool's
    as governance saw it (the catalog's over the tool's own) and ``rule`` its approval rule,
    if any. A call your code asks about (a hook's or an approval function's ``Ask``) also
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
        """This call waiting for a person, for ``reason`` (a hook's or an approval function's
        ``Ask``: whose it is, and the screen it is reviewed on)."""
        return dataclasses.replace(
            self,
            action=Action.ASK,
            reason=reason,
            assignee=assignee,
            component=component,
            props=props,
        )

    def approved(self, why: str | None = None) -> Decision:
        """This call approved without asking — by its tool's approval function
        (``tool(approval=)``), or ``why`` (a reviewer approved the tool for the rest of the
        run) —: it runs, announced unless it only reads."""
        action = Action.RUN if self.risk == "read" else Action.ANNOUNCE
        reason = why or f"{self.tool} was approved by its approval function."
        return dataclasses.replace(self, action=action, reason=reason)


def decide(tool: str, risk: str, rule: str | None, args: Mapping[str, Any]) -> Decision:
    """The action for a call of ``tool`` with ``args``, given its ``risk`` and approval
    ``rule``."""

    def decided(action: Action, reason: str) -> Decision:
        return Decision(tool, args, action, reason, risk, rule)

    if rule is None:
        return decided(ACTIONS.get(risk, Action.ANNOUNCE), f"{tool} is {risk}.")
    if rule == CATALOG_UNREAD:
        return decided(
            Action.ASK,
            f"{tool} is {risk}, and the tool catalog that says when it needs approval could "
            "not be read.",
        )
    try:
        if evaluate(rule, args):
            return decided(Action.ASK, f"{rule}.")
    except Exception:
        return decided(Action.ASK, f"the rule {rule!r} could not be checked on this call.")
    # the rule decided no approval: the call runs, announced unless it only reads
    return decided(Action.RUN if risk == "read" else Action.ANNOUNCE, "")
