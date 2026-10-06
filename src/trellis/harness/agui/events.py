"""The AG-UI vocabulary, spelled as the protocol spells it (camelCase on the wire; the
schemas are ``ag-ui-protocol/ag-ui`` ``sdks/typescript/packages/core/src/generated/schemas.ts``)."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic.alias_generators import to_camel

from trellis.contracts import InterruptDecision, InterruptRemember


class AGUIEventType(StrEnum):
    TEXT_MESSAGE_START = "TEXT_MESSAGE_START"
    TEXT_MESSAGE_CONTENT = "TEXT_MESSAGE_CONTENT"
    TEXT_MESSAGE_END = "TEXT_MESSAGE_END"
    TOOL_CALL_START = "TOOL_CALL_START"
    TOOL_CALL_ARGS = "TOOL_CALL_ARGS"
    TOOL_CALL_END = "TOOL_CALL_END"
    TOOL_CALL_RESULT = "TOOL_CALL_RESULT"
    STATE_SNAPSHOT = "STATE_SNAPSHOT"
    STATE_DELTA = "STATE_DELTA"
    MESSAGES_SNAPSHOT = "MESSAGES_SNAPSHOT"
    RAW = "RAW"
    CUSTOM = "CUSTOM"
    RUN_STARTED = "RUN_STARTED"
    RUN_FINISHED = "RUN_FINISHED"
    RUN_ERROR = "RUN_ERROR"
    STEP_STARTED = "STEP_STARTED"
    STEP_FINISHED = "STEP_FINISHED"


class OutcomeType(StrEnum):
    """``RunFinished.outcome.type``: the three endings the protocol knows."""

    SUCCESS = "success"
    INTERRUPT = "interrupt"
    CANCELLED = "cancelled"


class Role(StrEnum):
    """``Message.role``: who said a message of the thread."""

    DEVELOPER = "developer"
    SYSTEM = "system"
    ASSISTANT = "assistant"
    USER = "user"
    TOOL = "tool"
    ACTIVITY = "activity"
    REASONING = "reasoning"


class ResumeStatus(StrEnum):
    """``ResumeEntry.status``: the interrupt was answered, or the run is abandoned."""

    RESOLVED = "resolved"
    CANCELLED = "cancelled"


class _Wire(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="allow")


class InterruptEntry(_Wire):
    """One entry of ``RunFinished.outcome.interrupts``."""

    id: str
    reason: str
    message: str | None = None
    tool_call_id: str | None = None
    response_schema: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None


class Outcome(_Wire):
    """``RunFinished.outcome``: a discriminated object, never a bare string."""

    type: OutcomeType
    interrupts: list[InterruptEntry] | None = None
    pending_tool_call_ids: list[str] | None = None


class AGUIEvent(_Wire):
    """One event on the wire. Only the members a type uses are set."""

    type: AGUIEventType
    timestamp: int | None = None
    thread_id: str | None = None
    run_id: str | None = None
    step_name: str | None = None
    message_id: str | None = None
    role: str | None = None
    delta: Any = None
    tool_call_id: str | None = None
    tool_call_name: str | None = None
    parent_message_id: str | None = None
    content: str | None = None
    snapshot: Any = None
    messages: list[dict[str, Any]] | None = None
    name: str | None = None
    value: Any = None
    event: Any = None
    source: str | None = None
    outcome: Outcome | None = None
    result: Any = None
    message: str | None = None
    code: str | None = None

    def wire(self) -> dict[str, Any]:
        return self.model_dump(by_alias=True, exclude_none=True)


class InputMessage(_Wire):
    id: str | None = None
    role: Role
    content: str | None = None


class Resume(_Wire):
    """``ResumeEntry``: the answer to an interrupt of an earlier run on the same thread.
    ``payload`` is the answer (a question's value; ``true``/``false`` or edited arguments for
    an approval); ``decision`` is the harness's optional extension naming the contracts
    decision outright. The reviewer is the authenticated caller, so no field for it."""

    interrupt_id: str
    status: ResumeStatus = ResumeStatus.RESOLVED
    payload: Any = None
    metadata: dict[str, Any] | None = None
    decision: InterruptDecision | None = Field(
        default=None,
        description="the contracts decision outright (any case): answer, approve, reject, "
        "edit or cancel",
    )
    comment: str | None = Field(
        default=None, max_length=4000, description="the reviewer's remark on the decision"
    )
    remember: InterruptRemember = Field(
        default="once",
        description="run: an approval of a tool call approves that tool's later calls in the "
        "run without asking",
    )

    @field_validator("decision", mode="before")
    @classmethod
    def _any_case(cls, value: Any) -> Any:
        return value.upper() if isinstance(value, str) else value


class RunAgentInput(_Wire):
    """``RunAgentInput`` as the protocol defines it. ``runId`` is echoed on every event of the
    run; without one the harness names the run itself."""

    thread_id: str
    run_id: str | None = None
    parent_run_id: str | None = None
    messages: list[InputMessage] = Field(default_factory=list)
    tools: list[dict[str, Any]] = Field(default_factory=list)
    context: list[dict[str, Any]] = Field(default_factory=list)
    state: Any = None
    forwarded_props: Any = None
    resume: list[Resume] = Field(default_factory=list)

    def latest_user_text(self) -> str | None:
        for message in reversed(self.messages):
            if message.role is Role.USER and message.content:
                return message.content
        return None
