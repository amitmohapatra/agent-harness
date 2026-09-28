"""The run state machine a Temporal workflow holds.

Deliberately free of ``temporalio``: the transitions are the interesting part and they are
testable without a server, a worker or a task queue. :mod:`trellis.harness_temporal.workflow`
is a thin deterministic shell over this class, which is also why a Temporal outage never
changes what a paused run *means* — the meaning lives in ``trellis-contracts``.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict
from trellis.contracts.errors import AgentError
from trellis.contracts.ids import now
from trellis.contracts.runs import (
    Interrupt,
    InterruptResolution,
    RunRecord,
    RunStart,
    RunStatus,
)


class RunEnding(BaseModel):
    """How a run ended, as the ``finish`` signal carries it.

    A signal payload of this adapter, not a platform contract: the platform's vocabulary is
    :class:`~trellis.contracts.runs.RunStatus` plus the output and the error, and this is the
    one envelope they travel in so adding a field later does not change the signal's arity.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: RunStatus
    output: Any = None
    error: AgentError | None = None


class RunTransitionError(ValueError):
    """An illegal or out-of-order transition. Raised by the state machine, *caught* by the
    workflow: a signal handler that raised would fail the workflow task and retry forever,
    so a bad signal is recorded and answerable by query instead of wedging the run."""


class RunState:
    """One run's record, and the four moves that may change it.

    Every move returns the new record. The record is frozen (contracts), so a move replaces
    it rather than mutating it, which is what makes a query handler safe: whatever it read
    cannot change underneath it.
    """

    __slots__ = ("_record", "_refusals")

    def __init__(self, start: RunStart, *, run_id: str | None = None) -> None:
        # The workflow id *is* the run id for a Temporal-backed run: one workflow per run,
        # started under the run's own id. A schedule firing is the case that matters — the
        # server appends the nominal time to the action's id, so the firing's workflow id is
        # the run id of that firing and the schedule's template id is not.
        if run_id and run_id != start.run_id:
            start = start.model_copy(update={"run_id": run_id})
        self._record = RunRecord.from_start(start)
        self._refusals: list[str] = []

    @property
    def record(self) -> RunRecord:
        return self._record

    @property
    def refusals(self) -> list[str]:
        """Signals this run declined, newest last. Queryable, because a resume that arrived
        for the wrong interrupt is an operational fact somebody has to be able to see."""
        return list(self._refusals)

    @property
    def done(self) -> bool:
        return self._record.status.final

    # ------------------------------------------------------------------ transitions
    def pause(self, interrupt: Interrupt) -> RunRecord:
        """The run is waiting on something outside it. ``awaiting`` is the question."""
        self._belongs(interrupt.run_id, interrupt.tenant_id, "interrupt")
        if self._record.status.final:
            raise RunTransitionError(f"a {self._record.status.value} run cannot pause")
        self._record = self._touch(status=RunStatus.PAUSED, awaiting=interrupt)
        return self._record

    def resume(self, resolution: InterruptResolution) -> RunRecord:
        """The answer arrived: back to ``RUNNING``, same run id, next attempt."""
        awaiting = self._record.awaiting
        if awaiting is None:
            raise RunTransitionError(f"a {self._record.status.value} run is not waiting")
        if not resolution.resolves(awaiting):
            raise RunTransitionError("the resolution answers a different interrupt or run")
        self._record = self._touch(
            status=RunStatus.RUNNING,
            awaiting=None,
            last_resolution=resolution,
            attempt=self._record.attempt + 1,
        )
        return self._record

    def finish(
        self, status: RunStatus, *, output: Any = None, error: AgentError | None = None
    ) -> RunRecord:
        """End the run, once. A second ending is the same ending, not an error: a retried
        transition must not turn a recorded success into a failure."""
        if not status.final:
            raise RunTransitionError(f"{status.value} is not an ending")
        if self._record.status.final:
            return self._record
        self._record = self._touch(
            status=status,
            output=output,
            error=error if status in ENDINGS_WITH_ERROR else None,
            awaiting=None,
        )
        return self._record

    def end(self, ending: RunEnding) -> RunRecord:
        return self.finish(ending.status, output=ending.output, error=ending.error)

    # ------------------------------------------------------------------ internals
    def _belongs(self, run_id: str, tenant_id: str, what: str) -> None:
        if run_id != self._record.run_id or tenant_id != self._record.tenant_id:
            raise RunTransitionError(f"the {what} belongs to another run")

    def _touch(self, **fields: Any) -> RunRecord:
        """The next record, re-validated.

        ``model_copy`` would be cheaper and would skip the validator — which is exactly the
        one that says a PAUSED run is the only run with an ``awaiting`` and that a succeeded
        run carries no error. A state machine that can build a record the contract forbids is
        not a state machine.
        """
        data = {**self._record.model_dump(), **fields, "updated_at": now()}
        return RunRecord.model_validate(data)

    def refuse(self, error: Exception) -> None:
        """Record a declined signal. Bounded: a run that is signalled wrongly forever must
        not grow its own history without limit."""
        self._refusals.append(str(error))
        if len(self._refusals) > MAX_REFUSALS:
            del self._refusals[: len(self._refusals) - MAX_REFUSALS]


#: How many declined signals one run remembers.
MAX_REFUSALS = 32

#: Endings that carry an error; the others must not, and ``RunRecord`` refuses a record that
#: gets it wrong. Spelled once here and imported by the run store, because two copies of "which
#: endings carry an error" drift silently and the first symptom is a validation error on a
#: record this adapter built itself. ``test_state_and_cadence`` derives the truth from the
#: contract by construction, so a new failing status in contracts fails a test rather than a run.
ENDINGS_WITH_ERROR = frozenset({RunStatus.ERROR, RunStatus.TIMEOUT, RunStatus.REJECTED})


__all__ = [
    "ENDINGS_WITH_ERROR",
    "MAX_REFUSALS",
    "RunEnding",
    "RunState",
    "RunTransitionError",
]
