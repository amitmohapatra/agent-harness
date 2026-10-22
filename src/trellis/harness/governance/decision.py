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
import functools
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final

from trellis.contracts import PolicyDeniedError
from trellis.harness.bounded import SAID_ONCE
from trellis.harness.governance.catalog import CATALOG_UNREAD
from trellis.harness.tools.base import SideEffect
from trellis.memory.approval import evaluate

log = logging.getLogger("trellis.harness")


class RuleError(PolicyDeniedError):
    """An approval rule (``approve_when``) that cannot be evaluated on a call: the rule is
    broken, so the call asks a person (governance fails closed, never open). Kept on the
    :class:`Decision` (``error``) and, in a run, recorded on its event stream as a warning
    with this ``code``."""

    code = "APPROVAL_RULE_INVALID"


class Action(StrEnum):
    RUN = "run"
    ANNOUNCE = "announce"
    ASK = "ask"


#: The action of each risk when no rule governs the tool (an unknown risk is announced).
ACTIONS: Final[dict[str, Action]] = {
    SideEffect.READ: Action.RUN,
    SideEffect.WRITE: Action.ANNOUNCE,
    SideEffect.IRREVERSIBLE: Action.ASK,
}


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
    #: why the approval rule could not decide (the call asks)
    error: RuleError | None = None

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

    def approved(self, why: str) -> Decision:
        """This call approved without asking, for ``why`` (a reviewer approved the tool for
        the rest of the run): it runs, announced unless it only reads."""
        action = Action.RUN if self.risk == SideEffect.READ else Action.ANNOUNCE
        return dataclasses.replace(self, action=action, reason=why)


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
        holds = evaluate(rule, args)
    except (ValueError, RecursionError) as exc:  # the evaluator's own: a rule that is broken
        error = RuleError(
            f"the approval rule {rule!r} of {tool} could not be evaluated ({exc}): every call "
            "of it asks for approval until the rule is fixed in the tool catalog",
            details={"tool": tool, "rule": rule},
        )
        _broken(tool, rule, error.message)
        _recorded(error)
        reason = f"the rule {rule!r} could not be checked on this call."
        return dataclasses.replace(decided(Action.ASK, reason), error=error)
    if holds:
        return decided(Action.ASK, f"{rule}.")
    # the rule decided no approval: the call runs, announced unless it only reads
    return decided(Action.RUN if risk == SideEffect.READ else Action.ANNOUNCE, "")


@functools.lru_cache(maxsize=SAID_ONCE)  # said once per tool and rule
def _broken(tool: str, rule: str, message: str) -> None:
    log.error("%s", message, extra={"tool": tool, "rule": rule, "code": RuleError.code})


def _recorded(error: RuleError) -> None:
    """The broken rule on the event stream of the run deciding the call, when there is one."""
    from trellis.harness.runtime import current  # noqa: PLC0415 - runtime imports governance

    runtime = current()
    if runtime is not None:
        runtime.events.warning(error.code, error.message)
