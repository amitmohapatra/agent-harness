"""What a run call returns: the run's state after the call."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from trellis.contracts import AgentError, Interrupt, RunStatus


class Result(BaseModel):
    """``SUCCESS`` with the ``answer``; ``PAUSED`` with the ``interrupt`` a person answers
    (``agent.resume``); ``ERROR`` with the ``error``; ``QUEUED`` when a worker continues it;
    ``CANCELLED``."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    run_id: str
    status: RunStatus
    answer: Any = None
    interrupt: Interrupt | None = None
    error: AgentError | None = None
