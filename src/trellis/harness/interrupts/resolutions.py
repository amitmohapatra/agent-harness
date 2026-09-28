"""Pauses announced and answers waiting, per run.

The coordinator announces every interrupt here with the context of the run that paused (the
unredacted record: what leaves the process on the stream is redacted, and a resumed tool
call is checked against the arguments the approver actually saw). ``harness.resume`` records
a person's answer before the agent runs again; the runtime built for that run (the same run
id, derived from the thread and turn) picks it up, so a tool call the policy held for
approval finds its decision and an agent that asked a question finds the answer in
``runtime.state["resolutions"]``. In-process and short-lived by design: the durable record
is the run store and the feedback the answer became.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Final

from trellis.contracts.context import AgentExecutionContext
from trellis.contracts.runs import Interrupt, InterruptResolution

#: The key an answer to a question (not a tool approval) is filed under.
ANSWER: Final = "answer"
#: How many announced, unanswered interrupts a harness remembers.
MAX_ANNOUNCED: Final = 1000
#: The identity a claim must match: the run's tenant, and its user and workspace.
IDENTITY_FIELDS: Final = ("tenant_id", "user_id", "workspace_id")


class ResolutionRegistry:
    """Keyed by tenant and run: a run id is caller-chosen material and never enough."""

    def __init__(self, *, capacity: int = MAX_ANNOUNCED) -> None:
        self._by_run: dict[tuple[str, str], dict[str, Resolved]] = {}
        self._announced: OrderedDict[str, tuple[Interrupt, AgentExecutionContext]] = OrderedDict()
        self.capacity = capacity

    # -- pauses --------------------------------------------------------------------------
    def announce(self, interrupt: Interrupt, context: AgentExecutionContext) -> None:
        """A run paused: remember what it asked and whose run it is, bounded."""
        self._announced[interrupt.interrupt_id] = (interrupt, context)
        while len(self._announced) > self.capacity:
            self._announced.popitem(last=False)

    def claim(
        self,
        interrupt_id: str,
        *,
        tenant_id: str,
        user_id: str | None = None,
        workspace_id: str | None = None,
        thread_id: str | None = None,
    ) -> tuple[Interrupt, AgentExecutionContext] | None:
        """The interrupt and the context of the run that asked, removed, when the claim
        comes from the run's own tenant, user and workspace (and thread, when named);
        ``None`` otherwise, and the pause stays unanswered."""
        found = self._announced.get(interrupt_id)
        if found is None:
            return None
        _interrupt, context = found
        identity = {"tenant_id": tenant_id, "user_id": user_id, "workspace_id": workspace_id}
        if any(identity[name] != getattr(context, name) for name in IDENTITY_FIELDS):
            return None
        if thread_id is not None and context.thread_id != thread_id:
            return None
        del self._announced[interrupt_id]
        return found

    def announced(self, tenant_id: str) -> list[Interrupt]:
        """The unanswered interrupts of one tenant, oldest first."""
        return [i for i, c in self._announced.values() if c.tenant_id == tenant_id]

    # -- answers -------------------------------------------------------------------------
    def record(self, interrupt: Interrupt, resolution: InterruptResolution) -> None:
        key = (
            interrupt.tool_call.idempotency_key
            if interrupt.tool_call is not None and interrupt.tool_call.idempotency_key
            else ANSWER
        )
        self._by_run.setdefault((interrupt.tenant_id, interrupt.run_id), {})[key] = Resolved(
            interrupt, resolution
        )

    def for_run(self, tenant_id: str, run_id: str) -> dict[str, Any]:
        """The answers a resumed run finds; consumed once, so a later run asks afresh."""
        return self._by_run.pop((tenant_id, run_id), {})

    def pending(self, tenant_id: str, run_id: str) -> bool:
        return (tenant_id, run_id) in self._by_run


class Resolved:
    """An answer together with the question it answered, so a tool call held for approval
    is checked against the arguments the approver saw."""

    __slots__ = ("interrupt", "resolution")

    def __init__(self, interrupt: Interrupt, resolution: InterruptResolution) -> None:
        self.interrupt = interrupt
        self.resolution = resolution

    @property
    def decision(self) -> Any:
        return self.resolution.decision

    @property
    def answer(self) -> Any:
        return self.resolution.answer

    @property
    def payload(self) -> Any:
        return self.resolution.payload
