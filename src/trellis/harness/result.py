"""What a run call returns: the run's state after the call."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from trellis.contracts import AgentError, Interrupt, RunRecord, RunStatus
from trellis.harness.tools.base import Citation


class Result(BaseModel):
    """``SUCCESS`` with the ``answer``; ``PAUSED`` with the ``interrupt`` a person answers
    (``agent.resume``); ``ERROR`` with the ``error``; ``QUEUED`` when a worker continues it;
    ``CANCELLED``. An answer given in process carries its ``sources``: those its tools returned
    (``tool(response_format="content_and_artifact")``), each once — never one the model named
    itself."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    run_id: str
    status: RunStatus
    answer: Any = None
    interrupt: Interrupt | None = None
    error: AgentError | None = None
    sources: list[Citation] = Field(default_factory=list)

    @classmethod
    def of(cls, record: RunRecord) -> Result:
        """A run as its record says it is now."""
        return cls(
            run_id=record.run_id,
            status=record.status,
            answer=record.output,
            interrupt=record.awaiting,
            error=record.error,
        )
