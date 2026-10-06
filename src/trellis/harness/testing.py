"""Scripted people for tests: :class:`Reviewer` answers a run's questions and approvals from a
script, the same way every time (``trellis.testing.Reviewer``)::

    reviewer = Reviewer({"refund": "approve", "Which plans?": ["a", "b"]})
    result = await reviewer.run(agent, "refund order 7", user="ada")    # to its end
    assert result.status is RunStatus.SUCCESS
    assert reviewer.answered[0] == ("refund", "APPROVE", None)

A script entry is found for an interrupt by, in order: the tool of the call it asks about (an
approval), the ``component`` it names, its question (exactly), then ``"*"``. What the entry
says:

* for an approval: ``"approve"``, ``"reject"``, ``"cancel"``, ``True``/``False``, or a dict —
  the edited arguments (an ``edit``);
* for anything else: the answer itself (a list of option values with ``multiple``; a result
  from outside the run that an ``ask`` in a tool waits for);
* :class:`Decide` for full control (``Decide("approve", remember="run", comment="ok")``,
  ``Decide("cancel")``), or a function of the ``Interrupt`` returning any of these.

An interrupt the script does not cover raises ``LookupError`` naming it. Way 2:
:meth:`Reviewer.resolution` is the ``InterruptResolution`` to hand ``RunsClient.resume``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Literal

from trellis.contracts import (
    Interrupt,
    InterruptDecision,
    InterruptReason,
    InterruptResolution,
    RunStatus,
)

if TYPE_CHECKING:
    from trellis.harness.agent import Agent
    from trellis.harness.result import Result

#: The entry that answers what no other entry does.
ANY: Final = "*"
#: The most pauses :meth:`Reviewer.settle` answers in one run: a run that asks more is a loop.
MAX_ANSWERS: Final = 50
#: The words an approval's entry may say.
_WORDS: Final = frozenset({"approve", "reject", "cancel"})


@dataclass(frozen=True, slots=True)
class Decide:
    """A decision spelled out: ``decision`` (answer, approve, reject, edit, cancel), its
    ``answer`` (the edited arguments of an edit), a ``comment``, and ``remember="run"`` for an
    approval that covers the tool's later calls in the run."""

    decision: str
    answer: Any = None
    comment: str | None = None
    remember: Literal["once", "run"] = "once"


class Reviewer:
    """Answers interrupts from ``script`` as ``name`` (the reviewer the decisions carry)."""

    def __init__(self, script: Mapping[str, Any], *, name: str = "reviewer") -> None:
        self.script = dict(script)
        self.name = name
        #: what was answered, in order: (the entry's key, the decision, the answer)
        self.answered: list[tuple[str, str, Any]] = []

    def decide(self, interrupt: Interrupt) -> tuple[str, Decide]:
        """The script's entry for ``interrupt`` and the decision it means."""
        call = interrupt.tool_call
        for key in (call.tool if call else None, interrupt.component, interrupt.question, ANY):
            if key is not None and key in self.script:
                return key, self._meant(interrupt, self.script[key])
        raise LookupError(
            f"no scripted answer for {interrupt.question!r} (tool "
            f"{call.tool if call else None}, component {interrupt.component}): the script "
            f"answers {sorted(self.script)}"
        )

    def resolution(self, interrupt: Interrupt) -> InterruptResolution:
        """The resolution of ``interrupt`` the script says (Way 2: ``runs.resume`` it)."""
        key, decided = self.decide(interrupt)
        chosen = InterruptDecision(decided.decision.upper())
        edited = chosen is InterruptDecision.EDIT
        self.answered.append((key, chosen.value, decided.answer))
        return InterruptResolution(
            interrupt_id=interrupt.interrupt_id,
            run_id=interrupt.run_id,
            decision=chosen,
            answer=None if edited else decided.answer,
            payload=decided.answer if edited else None,
            reviewer=self.name,
            comment=decided.comment,
            remember=decided.remember,
        )

    async def settle(self, agent: Agent, result: Result) -> Result:
        """Answer ``result``'s pauses until the run ends (or goes back to the queue, for a
        worker to continue: ``QUEUED``)."""
        for _ in range(MAX_ANSWERS):
            if result.status is not RunStatus.PAUSED:
                return result
            result = await self.answer(agent, result)
        raise RuntimeError(f"run {result.run_id} asked more than {MAX_ANSWERS} times")

    async def answer(self, agent: Agent, result: Result) -> Result:
        """Answer the one pause ``result`` is (``agent.resume``), as the script says."""
        assert result.interrupt is not None, f"run {result.run_id} is not paused"
        given = self.resolution(result.interrupt)
        return await agent.resume(
            given.interrupt_id,
            given.decision,
            answer=given.payload if given.decision is InterruptDecision.EDIT else given.answer,
            reviewer=self.name,
            comment=given.comment,
            remember=given.remember,
        )

    async def run(self, agent: Agent, input: Any, **kwargs: Any) -> Result:
        """``agent.run(input, **kwargs)``, its pauses answered to its end."""
        return await self.settle(agent, await agent.run(input, **kwargs))

    def _meant(self, interrupt: Interrupt, entry: Any) -> Decide:
        if isinstance(entry, Decide):
            return entry
        if callable(entry) and not isinstance(entry, str | bytes):
            given: Callable[[Interrupt], Any] = entry
            return self._meant(interrupt, given(interrupt))
        if interrupt.reason is not InterruptReason.APPROVAL:
            return Decide("answer", entry)
        if isinstance(entry, bool):
            return Decide("approve" if entry else "reject")
        if isinstance(entry, dict):
            return Decide("edit", entry)
        word = str(entry).lower()
        if word not in _WORDS:
            raise ValueError(
                f"an approval is answered approve, reject, cancel, True/False or the edited "
                f"arguments, not {entry!r}"
            )
        return Decide(word)


__all__ = ["ANY", "Decide", "Reviewer"]
