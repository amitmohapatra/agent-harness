"""One workflow per run (design §10 "Long-running / cowork").

A run outlives a process, a deploy and a framework swap. Temporal's durable execution is the
strongest way to say that, so the run *is* a workflow: its state machine is the workflow's
state, ``paused``/``resumed``/``finished`` are signals, and ``get``/``list_paused`` are
queries. The agent itself does not run here — the harness runs the agent — which is why this
workflow has no activities: it is the record, kept alive by the cluster instead of by a
process, and it is the same record ``trellis-contracts`` defines.

```mermaid
stateDiagram-v2
  [*] --> RUNNING: start_workflow(id=run_id, RunStart)
  RUNNING --> PAUSED: signal pause(Interrupt)
  PAUSED --> RUNNING: signal resume(InterruptResolution)<br/>attempt + 1
  RUNNING --> SUCCESS: signal finish(RunEnding)
  RUNNING --> ERROR: signal finish(RunEnding)
  PAUSED --> CANCELLED: signal finish(RunEnding)
  SUCCESS --> [*]
```
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import timedelta

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from trellis.contracts.runs import Interrupt, InterruptResolution, RunRecord, RunStart

    from trellis.harness_temporal.state import RunEnding, RunState, RunTransitionError

#: The workflow type name. Stable: it is what a visibility query filters on, so renaming the
#: class must not rename the type.
RUN_WORKFLOW = "TrellisAgentRun"

#: Signal and query names, spelled once so the adapter and the workflow cannot disagree.
PAUSE_SIGNAL = "pause"
RESUME_SIGNAL = "resume"
FINISH_SIGNAL = "finish"
RECORD_QUERY = "record"
REFUSALS_QUERY = "refusals"

#: How long a signal handler waits for the run method to install the state. Signals are
#: delivered to handlers that exist before ``run`` has executed, so the handler waits rather
#: than dropping the pause of a run that was signalled in the same workflow task it started.
_READY_TIMEOUT = timedelta(minutes=1)


@workflow.defn(name=RUN_WORKFLOW)
class AgentRunWorkflow:
    """The run record, durable. Deterministic: no clocks, no I/O, no randomness."""

    def __init__(self) -> None:
        self._state: RunState | None = None

    @workflow.run
    async def run(self, start: RunStart) -> RunRecord:
        self._state = RunState(start, run_id=workflow.info().workflow_id)
        await workflow.wait_condition(lambda: self._state is not None and self._state.done)
        return self._state.record

    # ------------------------------------------------------------------ signals
    @workflow.signal(name=PAUSE_SIGNAL)
    async def pause(self, interrupt: Interrupt) -> None:
        state = await self._ready()
        self._apply(lambda: state.pause(interrupt))

    @workflow.signal(name=RESUME_SIGNAL)
    async def resume(self, resolution: InterruptResolution) -> None:
        state = await self._ready()
        self._apply(lambda: state.resume(resolution))

    @workflow.signal(name=FINISH_SIGNAL)
    async def finish(self, ending: RunEnding) -> None:
        state = await self._ready()
        self._apply(lambda: state.end(ending))

    # ------------------------------------------------------------------ queries
    @workflow.query(name=RECORD_QUERY)
    def record(self) -> RunRecord | None:
        """The run as it stands, or ``None`` before the run method has started — which is
        exactly what ``RunStore.get`` means by "no such run"."""
        return self._state.record if self._state is not None else None

    @workflow.query(name=REFUSALS_QUERY)
    def refusals(self) -> list[str]:
        """Signals this run declined. An answer to "why is it still RUNNING?"."""
        return self._state.refusals if self._state is not None else []

    # ------------------------------------------------------------------ internals
    async def _ready(self) -> RunState:
        await workflow.wait_condition(lambda: self._state is not None, timeout=_READY_TIMEOUT)
        if self._state is None:  # pragma: no cover - the wait raises on timeout
            raise RunTransitionError("the run never started")
        return self._state

    def _apply(self, move: Callable[[], RunRecord]) -> None:
        """Run one transition, keeping a refused signal off the workflow's failure path.

        A signal handler that raises fails the *workflow task*, which Temporal then retries
        forever: an answer to the wrong question would stop the run responding to the right
        one. So a refusal is recorded on the run and readable by query.
        """
        state = self._state
        if state is None:  # pragma: no cover - callers pass through _ready first
            return
        try:
            move()
        except RunTransitionError as exc:
            state.refuse(exc)
            workflow.logger.warning("trellis.run.signal_refused: %s", exc)


__all__ = [
    "FINISH_SIGNAL",
    "PAUSE_SIGNAL",
    "RECORD_QUERY",
    "REFUSALS_QUERY",
    "RESUME_SIGNAL",
    "RUN_WORKFLOW",
    "AgentRunWorkflow",
]
