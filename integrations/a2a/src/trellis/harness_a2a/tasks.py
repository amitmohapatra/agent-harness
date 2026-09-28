"""Where an A2A task lives: in this process, and behind it the harness's run store.

An A2A ``Task`` and a harness run are the same thing seen from two sides — the task is the
protocol's view of the run that is answering it, which is why the A2A task id *is* the run id
here. The consequence worth designing for: a task has to be answerable after the process that
served it is gone (a restart, another replica, a client that comes back tomorrow with a task id),
and the thing that survives is the **run**.

```mermaid
flowchart LR
  A["GetTask / a follow-up message"] --> B{in this process?}
  B -- yes --> C["the A2A view: status, history, artifacts"]
  B -- no --> D["RunStore.get(run_id)"]
  D -- "the caller's own run" --> E["rebuilt Task (status, input, output, its open question)"]
  D -- "another tenant's, or unknown" --> F["None: a 404, never a peek"]
```

So this store *composes* the SDK's in-memory store (owner scoping, filtering, paging and copying
are already solved there) and adds one thing: the rebuild from the run store on a miss. It
deliberately writes no run transitions — the harness's own ``RunRecorder`` already records
started, paused, resumed and finished from the run's lifecycle, and a second writer would mean
two records of one truth.

Ownership is not an afterthought: tasks are partitioned by the *trusted* identity (the tenant and
user the platform authenticated), so one tenant naming another tenant's task id gets ``None``.
"""

from __future__ import annotations

from typing import Any, Final

from a2a.server.context import ServerCallContext
from a2a.server.owner_resolver import resolve_user_scope
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
from google.protobuf.timestamp_pb2 import Timestamp
from trellis.contracts.runs import RunRecord, RunStatus

from trellis.harness.runtime.logging import get_logger
from trellis.harness_a2a.identity import IdentityRefused, IdentityResolver
from trellis.harness_a2a.translate import asked, data_part, text_part, value_part

log = get_logger("trellis.harness_a2a.tasks")

#: An owner no caller can be: what a refused or unidentifiable call is scoped to, so it can never
#: read a real caller's tasks (and ``""``, the SDK's unauthenticated scope, is not reused for it).
REFUSED_OWNER: Final = "\x00refused"

#: How a run's status shows up as a task state. ``PAUSED`` is the interesting one: A2A's
#: ``INPUT_REQUIRED`` is not terminal, which is what lets the next message resume the same task.
TASK_STATES: Final[dict[RunStatus, TaskState]] = {
    RunStatus.RUNNING: TaskState.TASK_STATE_WORKING,
    RunStatus.PAUSED: TaskState.TASK_STATE_INPUT_REQUIRED,
    RunStatus.SUCCESS: TaskState.TASK_STATE_COMPLETED,
    RunStatus.PARTIAL: TaskState.TASK_STATE_COMPLETED,
    RunStatus.ERROR: TaskState.TASK_STATE_FAILED,
    RunStatus.TIMEOUT: TaskState.TASK_STATE_FAILED,
    RunStatus.CANCELLED: TaskState.TASK_STATE_CANCELED,
    RunStatus.REJECTED: TaskState.TASK_STATE_REJECTED,
}


def task_owner(identity: IdentityResolver | None) -> Any:
    """An owner resolver for the SDK's store: the authenticated tenant and user.

    Without an identity resolver this falls back to the SDK's own (the authenticated principal's
    name), so a deployment that put authentication in front of the server still gets per-caller
    partitioning rather than one shared bucket.
    """

    def resolve(context: ServerCallContext) -> str:
        if identity is None:
            return resolve_user_scope(context)
        try:
            fields = identity.resolve(context)
        except IdentityRefused:
            return REFUSED_OWNER
        # \x00 rather than "/": an id may contain a slash, and "a/b"+"c" must not scope the same
        # as "a"+"b/c" — that would put two callers' tasks in one partition.
        return "\x00".join((str(fields.get("tenant_id") or ""), str(fields.get("user_id") or "")))

    return resolve


class HarnessTaskStore(TaskStore):
    """The SDK's task store for this process, with the harness's run store behind it."""

    def __init__(
        self,
        runs: Any = None,
        *,
        identity: IdentityResolver | None = None,
        store: TaskStore | None = None,
    ) -> None:
        """``runs`` is the harness's run store (``harness.runs``); a no-op store, or none at all,
        leaves this purely in-memory, which is the documented fallback. ``identity`` is the
        deployment's resolver, used to partition tasks by the caller the platform authenticated."""
        self._store = store or InMemoryTaskStore(owner_resolver=task_owner(identity))
        self._identity = identity
        self._runs = runs if runs is not None and getattr(runs, "name", "") != "noop" else None

    @property
    def durable(self) -> bool:
        """Whether a task can outlive this process (a real run store is configured)."""
        return self._runs is not None

    async def save(self, task: Task, context: ServerCallContext) -> None:
        await self._store.save(task, context)

    async def get(self, task_id: str, context: ServerCallContext) -> Task | None:
        found = await self._store.get(task_id, context)
        if found is not None:
            return found
        rebuilt = await self._from_run(task_id, context)
        if rebuilt is not None:
            # cache it so the rest of this process's handling sees one task, not two shapes
            await self._store.save(rebuilt, context)
        return rebuilt

    async def list(self, params: ListTasksRequest, context: ServerCallContext) -> ListTasksResponse:
        """This process's tasks. Deliberately not a run-store query: ``list_paused`` answers a
        different question (every paused run of a tenant, for a human's inbox) and paging over two
        stores with one page token would report a different page per replica."""
        return await self._store.list(params, context)

    async def delete(self, task_id: str, context: ServerCallContext) -> None:
        await self._store.delete(task_id, context)

    # ------------------------------------------------------------------ the run store behind it
    async def _from_run(self, task_id: str, context: ServerCallContext) -> Task | None:
        if self._runs is None:
            return None
        try:
            record = await self._runs.get(task_id)
        except Exception as exc:  # a run store outage is a miss, never an error to the caller
            log.warning("a2a.task_rebuild_failed", task_id=task_id, error=str(exc))
            return None
        if record is None or not self._is_callers(record, context):
            return None
        log.info("a2a.task_rebuilt_from_run", task_id=task_id, status=str(record.status))
        return task_from_run(record)

    def _is_callers(self, record: RunRecord, context: ServerCallContext) -> bool:
        """Whether this run belongs to the caller the platform authenticated.

        Without an identity resolver the deployment has told us it serves one caller, and the
        store's own owner partitioning is the boundary; with one, the run's tenant (and its user,
        when the caller has one) must match.
        """
        if self._identity is None:
            return True
        try:
            fields = self._identity.resolve(context)
        except IdentityRefused:
            return False
        if record.tenant_id != fields.get("tenant_id"):
            return False
        user = fields.get("user_id")
        return not (user and record.user_id and record.user_id != user)


def task_from_run(record: RunRecord) -> Task:
    """The A2A view of a run record: its state, what went in, what came out, what it is asking."""
    task = Task(
        id=record.run_id,
        context_id=record.thread_id or record.run_id,
        status=TaskStatus(state=TASK_STATES.get(record.status, TaskState.TASK_STATE_UNSPECIFIED)),
        metadata={"agent_id": record.agent_id, "attempt": record.attempt},
    )
    task.status.timestamp.CopyFrom(_timestamp(record))
    if record.input is not None:
        task.history.append(
            Message(
                message_id=f"{record.run_id}-input",
                context_id=task.context_id,
                task_id=task.id,
                role=Role.ROLE_USER,
                parts=[value_part(record.input)],
            )
        )
    if record.awaiting is not None:
        # ``Interrupt.awaiting()`` is the unredacted record (it carries the tool call's arguments),
        # so it is pruned to what a caller needs — the same four keys the live stream sends.
        task.status.message.CopyFrom(
            Message(
                message_id=record.awaiting.interrupt_id,
                context_id=task.context_id,
                task_id=task.id,
                role=Role.ROLE_AGENT,
                parts=[
                    text_part(record.awaiting.question),
                    data_part(asked(record.awaiting.awaiting())),
                ],
            )
        )
    elif record.error is not None:
        task.status.message.CopyFrom(
            Message(
                message_id=f"{record.run_id}-error",
                context_id=task.context_id,
                task_id=task.id,
                role=Role.ROLE_AGENT,
                parts=[text_part(record.error.message), data_part({"code": record.error.code})],
            )
        )
    if record.output is not None:
        artifact = task.artifacts.add()
        artifact.artifact_id = f"{record.run_id}-result"
        artifact.name = "result"
        artifact.parts.append(value_part(record.output))
    return task


def _timestamp(record: RunRecord) -> Timestamp:
    stamp = Timestamp()
    stamp.FromDatetime(record.updated_at)
    return stamp


__all__ = ["REFUSED_OWNER", "TASK_STATES", "HarnessTaskStore", "task_from_run", "task_owner"]
