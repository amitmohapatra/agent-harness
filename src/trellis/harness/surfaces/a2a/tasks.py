"""Where an A2A task lives: the SDK's in-process store, with the run store behind it.

The task id *is* the run id, so a paused or finished task this process does not hold (a
restart, another replica, a caller returning tomorrow) is rebuilt from the run record — its
state, input, result and open question — when the caller is the run's own user. No run
transitions are written here: the pipeline records the run.
"""

from __future__ import annotations

from typing import Final

from a2a.server.context import ServerCallContext
from a2a.server.tasks import InMemoryTaskStore, TaskStore
from a2a.types import (
    ListTasksRequest,
    ListTasksResponse,
    Message,
    Role,
    Task,
    TaskState,
    TaskStatus,
)

from trellis.contracts import RunRecord, RunStatus
from trellis.harness.clients.runs import Runs
from trellis.harness.surfaces.a2a.identity import IdentityRefused, UserResolver
from trellis.harness.surfaces.a2a.translate import RESULT_ARTIFACT, asked, value_part

#: The owner a refused call is scoped to: no real caller can be it.
REFUSED_OWNER: Final = "\x00refused"

TASK_STATES: Final[dict[RunStatus, TaskState]] = {
    RunStatus.QUEUED: TaskState.TASK_STATE_SUBMITTED,
    RunStatus.RUNNING: TaskState.TASK_STATE_WORKING,
    RunStatus.PAUSED: TaskState.TASK_STATE_INPUT_REQUIRED,
    RunStatus.SUCCESS: TaskState.TASK_STATE_COMPLETED,
    RunStatus.PARTIAL: TaskState.TASK_STATE_COMPLETED,
    RunStatus.ERROR: TaskState.TASK_STATE_FAILED,
    RunStatus.TIMEOUT: TaskState.TASK_STATE_FAILED,
    RunStatus.CANCELLED: TaskState.TASK_STATE_CANCELED,
    RunStatus.REJECTED: TaskState.TASK_STATE_REJECTED,
}


def owner(user_of: UserResolver) -> UserResolver:
    """The SDK owner scope: the caller's user, so one caller never reads another's tasks."""

    def resolve(context: ServerCallContext) -> str:
        try:
            return user_of(context)
        except IdentityRefused:
            return REFUSED_OWNER

    return resolve


class RunTaskStore(TaskStore):
    def __init__(self, runs: Runs, *, agent_id: str, user_of: UserResolver) -> None:
        self._runs = runs
        self._agent_id = agent_id
        self._owner = owner(user_of)
        self._store = InMemoryTaskStore(owner_resolver=self._owner)
        #: tasks this process is creating: their run is being opened, not rebuilt
        self._opening: set[str] = set()

    def opening(self, task_id: str) -> None:
        """This process is starting ``task_id``: until its first save, a lookup finds nothing
        (the run exists already, but the task is the new one the executor announces)."""
        self._opening.add(task_id)

    async def save(self, task: Task, context: ServerCallContext) -> None:
        self._opening.discard(task.id)
        await self._store.save(task, context)

    async def get(self, task_id: str, context: ServerCallContext) -> Task | None:
        found = await self._store.get(task_id, context)
        if found is not None or task_id in self._opening:
            return found
        record = await self._runs.get(task_id)
        if (
            record is None
            or record.agent_id != self._agent_id
            or record.user_id != self._owner(context)
            or not (record.final or record.status is RunStatus.PAUSED)
        ):
            # a live run is this process's own task (or another replica's): nothing to rebuild
            return None
        rebuilt = task_from_run(record)
        await self._store.save(rebuilt, context)
        return rebuilt

    async def list(self, params: ListTasksRequest, context: ServerCallContext) -> ListTasksResponse:
        return await self._store.list(params, context)

    async def delete(self, task_id: str, context: ServerCallContext) -> None:
        await self._store.delete(task_id, context)


def task_from_run(record: RunRecord) -> Task:
    """The A2A view of a run record."""
    context_id = record.thread_id or record.run_id
    task = Task(
        id=record.run_id,
        context_id=context_id,
        status=TaskStatus(state=TASK_STATES.get(record.status, TaskState.TASK_STATE_UNSPECIFIED)),
    )
    if record.input is not None:
        task.history.append(
            Message(
                message_id=f"{record.run_id}-input",
                context_id=context_id,
                task_id=record.run_id,
                role=Role.ROLE_USER,
                parts=[value_part(record.input)],
            )
        )
    if record.awaiting is not None:
        task.status.message.CopyFrom(
            Message(
                message_id=record.awaiting.interrupt_id,
                context_id=context_id,
                task_id=record.run_id,
                role=Role.ROLE_AGENT,
                parts=[
                    value_part(record.awaiting.question),
                    value_part(asked(record.awaiting.awaiting())),
                ],
            )
        )
    elif record.error is not None:
        task.status.message.CopyFrom(
            Message(
                message_id=f"{record.run_id}-error",
                context_id=context_id,
                task_id=record.run_id,
                role=Role.ROLE_AGENT,
                parts=[value_part(record.error.message)],
            )
        )
    if record.output is not None:
        artifact = task.artifacts.add()
        artifact.artifact_id = f"{record.run_id}-{RESULT_ARTIFACT}"
        artifact.name = RESULT_ARTIFACT
        artifact.parts.append(value_part(record.output))
    return task
