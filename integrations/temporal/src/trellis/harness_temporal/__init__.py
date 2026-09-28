"""trellis-harness-temporal: Temporal behind the run and schedule ports.

    from temporalio.worker import Worker
    from trellis.harness_temporal import AgentRunWorkflow, TemporalRunStore, TemporalScheduler

    runs = TemporalRunStore("localhost:7233", task_queue="trellis-runs")
    harness = AgentHarness(memory=memory, runs=runs)

    # somewhere in the deployment, one worker keeps the run workflows alive
    Worker(await runs.connection.client(), task_queue="trellis-runs",
           workflows=[AgentRunWorkflow])

The harness core imports none of this; a deployment chooses its durability engine with
``runs.engine`` (an enum) and nothing about an agent changes.

One thing here is not decoration. Registering ``AgentRunWorkflow`` makes Temporal's workflow
sandbox re-import the module it lives in, and importing a submodule runs *this* file first —
which would drag the adapters, and through them ``httpx``, into a sandbox that refuses it (the
restriction is real: HTTP in a workflow is exactly the non-determinism the sandbox exists to
catch). The adapters are never called *from* workflow code, so they are passed through instead
of re-imported. That is what makes ``workflows=[AgentRunWorkflow]`` work with no sandbox
configuration in the deployment.
"""

from temporalio import workflow as _workflow

with _workflow.unsafe.imports_passed_through():
    from trellis.harness_temporal.cadence import Cadence, is_manual, parse_cadence
    from trellis.harness_temporal.client import TemporalConnection
    from trellis.harness_temporal.runs import TemporalRunStore, TemporalRunsUnavailable
    from trellis.harness_temporal.scheduler import TemporalScheduler
    from trellis.harness_temporal.state import (
        ENDINGS_WITH_ERROR,
        RunEnding,
        RunState,
        RunTransitionError,
    )

# Deliberately *not* passed through: the workflow itself is the one module the sandbox should
# check, and it declares its own passthrough for the contracts it holds.
from trellis.harness_temporal.workflow import RUN_WORKFLOW, AgentRunWorkflow

__version__ = "0.1.0"

__all__ = [
    "ENDINGS_WITH_ERROR",
    "RUN_WORKFLOW",
    "AgentRunWorkflow",
    "Cadence",
    "RunEnding",
    "RunState",
    "RunTransitionError",
    "TemporalConnection",
    "TemporalRunStore",
    "TemporalRunsUnavailable",
    "TemporalScheduler",
    "__version__",
    "is_manual",
    "parse_cadence",
    "temporalio_version",
]


def temporalio_version() -> str | None:
    """The installed ``temporalio``, for the compatibility matrix."""
    from importlib.metadata import version  # noqa: PLC0415 - one call, at report time

    try:
        return version("temporalio")
    except Exception:  # pragma: no cover - not installed
        return None
