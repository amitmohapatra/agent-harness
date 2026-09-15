"""``AgentExecutionContext``: the immutable identity and lineage of one agent execution.

Every side effect the harness performs — a memory read, a memory write, a span, a metric,
an artifact, an evaluation event — is attributed with this context. It is created once per
execution and never mutated: a nested agent gets a *child* context via :meth:`for_agent`,
which inherits the trusted parent identity (tenant, user, thread, trace) and replaces only
the agent-run fields.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field

from universal_agent_harness.contracts.ids import new_id, safe_id, stable_id

#: Fields a child execution inherits verbatim from its parent. Anything outside this set is
#: either re-derived (agent run lineage) or explicitly passed.
INHERITED_FIELDS = (
    "tenant_id",
    "workspace_id",
    "user_id",
    "group_ids",
    "thread_id",
    "session_id",
    "turn_id",
    "work_id",
    "agent_group_id",
    "request_id",
    "correlation_id",
    "trace_id",
    "deadline",
)

#: Fields :meth:`AgentExecutionContext.for_agent` sets itself rather than inheriting.
_EXPLICIT = frozenset({"deadline", "agent_group_id"})


class AgentExecutionContext(BaseModel):
    """Immutable execution identity. Construct with :meth:`create` unless every id is known."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # -- tenancy / principal
    tenant_id: str
    workspace_id: str | None = None
    user_id: str | None = None
    group_ids: tuple[str, ...] = ()

    # -- conversation
    thread_id: str | None = None
    session_id: str | None = None
    turn_id: str | None = None

    # -- unit of work
    work_id: str | None = None
    task_id: str | None = None

    # -- agent identity
    agent_id: str
    agent_group_id: str | None = None
    agent_run_id: str
    parent_agent_run_id: str | None = None

    # -- correlation
    request_id: str
    correlation_id: str
    causation_id: str | None = None
    trace_id: str

    # -- deadline (absolute, timezone-aware)
    deadline: datetime | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)

    # ------------------------------------------------------------------ construction
    @classmethod
    def create(
        cls,
        *,
        tenant_id: str,
        agent_id: str = "agent",
        agent_run_id: str | None = None,
        request_id: str | None = None,
        correlation_id: str | None = None,
        trace_id: str | None = None,
        timeout_seconds: float | None = None,
        **fields: Any,
    ) -> AgentExecutionContext:
        """Create a root context, filling the ids an application should not have to invent."""
        req = request_id or new_id("req_")
        deadline = fields.pop("deadline", None)
        if deadline is None and timeout_seconds is not None:
            deadline = datetime.now(UTC) + timedelta(seconds=timeout_seconds)
        if "group_ids" in fields and fields["group_ids"] is not None:
            fields["group_ids"] = tuple(fields["group_ids"])
        return cls(
            tenant_id=tenant_id,
            agent_id=safe_id(agent_id),
            agent_run_id=agent_run_id or new_id("run_"),
            request_id=req,
            correlation_id=correlation_id or req,
            trace_id=trace_id or new_id(),
            deadline=deadline,
            **fields,
        )

    # ------------------------------------------------------------------ derivation
    def for_agent(
        self,
        agent_id: str,
        *,
        agent_run_id: str | None = None,
        agent_group_id: str | None = None,
        task_id: str | None = None,
        deadline: datetime | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Self:
        """A child context for a nested agent run.

        Trusted identity (tenant, workspace, principal, thread, work, request, trace) is
        inherited; the executing agent becomes ``agent_id``, this context's run becomes the
        parent run, and ``causation_id`` records which run caused the child.
        """
        child_deadline = _earliest(self.deadline, deadline)
        return type(self)(
            **{f: getattr(self, f) for f in INHERITED_FIELDS if f not in _EXPLICIT},
            deadline=child_deadline,
            agent_id=safe_id(agent_id),
            agent_run_id=agent_run_id or new_id("run_"),
            parent_agent_run_id=self.agent_run_id,
            agent_group_id=agent_group_id or self.agent_group_id,
            task_id=task_id if task_id is not None else self.task_id,
            causation_id=self.agent_run_id,
            metadata={**self.metadata, **(metadata or {})},
        )

    def with_fields(self, **changes: Any) -> Self:
        """A copy with ``changes`` applied. The context stays immutable; this returns a new one."""
        if "group_ids" in changes and changes["group_ids"] is not None:
            changes["group_ids"] = tuple(changes["group_ids"])
        return self.model_copy(update=changes)

    def with_deadline(self, deadline: datetime | None) -> Self:
        """Tighten the deadline. A child deadline never outlives the parent's (§38)."""
        return self.model_copy(update={"deadline": _earliest(self.deadline, deadline)})

    # ------------------------------------------------------------------ derived values
    @property
    def remaining_seconds(self) -> float | None:
        """Seconds left before :attr:`deadline` (never negative), or ``None`` if unbounded."""
        if self.deadline is None:
            return None
        return max(0.0, (self.deadline - datetime.now(UTC)).total_seconds())

    @property
    def expired(self) -> bool:
        return self.deadline is not None and self.remaining_seconds == 0.0

    def idempotency_key(self, *parts: object, prefix: str = "uah") -> str:
        """A key that is stable across retries of *this logical execution*.

        Derived from the durable lineage (tenant, thread, turn, task, agent, run) plus the
        caller's ``parts`` — never from a wall clock or a fresh uuid, so a framework that
        replays a step produces the same key and the write deduplicates (§42).
        """
        return stable_id(
            self.tenant_id,
            self.thread_id,
            self.turn_id,
            self.work_id,
            self.task_id,
            self.agent_id,
            self.agent_run_id,
            *parts,
            prefix=f"{prefix}-",
        )

    def scope_fields(self) -> dict[str, Any]:
        """The Memory Service ``Scope`` keyword arguments for this execution."""
        out: dict[str, Any] = {
            "tenant_id": self.tenant_id,
            "workspace_id": self.workspace_id,
            "user_id": self.user_id,
            "group_ids": list(self.group_ids),
            "thread_id": self.thread_id,
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "work_id": self.work_id,
            "task_id": self.task_id,
            "agent_id": self.agent_id,
            "agent_group_id": self.agent_group_id,
            "agent_run_id": self.agent_run_id,
            "parent_agent_run_id": self.parent_agent_run_id,
            "trace_id": self.trace_id,
            "correlation_id": self.correlation_id,
        }
        return {k: v for k, v in out.items() if v not in (None, [])}

    def log_fields(self) -> dict[str, str]:
        """The identifiers every structured log line carries (§36)."""
        return {
            k: str(v)
            for k, v in (
                ("trace_id", self.trace_id),
                ("request_id", self.request_id),
                ("correlation_id", self.correlation_id),
                ("tenant_id", self.tenant_id),
                ("agent_id", self.agent_id),
                ("agent_run_id", self.agent_run_id),
                ("task_id", self.task_id),
                ("thread_id", self.thread_id),
                ("turn_id", self.turn_id),
            )
            if v
        }


def _earliest(a: datetime | None, b: datetime | None) -> datetime | None:
    if a is None:
        return b
    if b is None:
        return a
    return min(a, b)
