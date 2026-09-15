"""Where a LangGraph execution *is*, expressed as harness identity (§51).

Only documented ``RunnableConfig`` keys are read:

* ``configurable.thread_id`` — the conversation, and therefore the memory thread;
* ``configurable.checkpoint_ns`` — ``"node:task"`` segments joined by ``|``, one per graph
  level, so the enclosing subgraph invocations are the agent lineage and the last segment
  is the executing node;
* ``configurable.checkpoint_id`` and ``metadata.langgraph_step`` / ``langgraph_node`` — the
  position in the run;
* ``configurable.harness`` — an explicit escape hatch for applications that carry identity
  in the config (tenant, user, workspace, agent group, turn...).

LangGraph derives task ids deterministically from the checkpoint, the step and the node, so
they are stable across retries of the same superstep. That is what makes a replayed
superstep produce the same ``agent_run_id`` and therefore the same idempotency keys (§55).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from universal_agent_harness.contracts.ids import safe_id, stable_id

#: Fields an application may set under ``configurable.harness``.
IDENTITY_FIELDS = (
    "tenant_id",
    "workspace_id",
    "user_id",
    "group_ids",
    "thread_id",
    "session_id",
    "turn_id",
    "work_id",
    "task_id",
    "agent_group_id",
    "agent_id",
    "agent_run_id",
    "correlation_id",
    "metadata",
)


@dataclass(frozen=True, slots=True)
class Segment:
    node: str
    task_id: str

    @classmethod
    def parse(cls, raw: str) -> Segment:
        node, _, task = raw.partition(":")
        return cls(node=node or raw, task_id=task or raw)


@dataclass(frozen=True, slots=True)
class Lineage:
    thread_id: str | None = None
    segments: tuple[Segment, ...] = ()
    checkpoint_id: str | None = None
    checkpoint_ns: str = ""
    run_id: str | None = None
    step: int | None = None
    node: str | None = None
    overrides: Mapping[str, Any] = field(default_factory=dict)

    @property
    def subgraphs(self) -> tuple[Segment, ...]:
        """The enclosing subgraph invocations (every segment but the executing node's)."""
        return self.segments[:-1]

    @property
    def task(self) -> Segment | None:
        return self.segments[-1] if self.segments else None

    @property
    def path(self) -> str:
        return "|".join(f"{s.node}:{s.task_id}" for s in self.segments)

    def run_id_for(self, agent_id: str) -> str:
        """A run id that is stable across checkpoint retries of this superstep."""
        anchor = self.task.task_id if self.task else (self.checkpoint_id or self.run_id or "")
        return stable_id(self.thread_id, self.checkpoint_ns, anchor, agent_id, prefix="run_lg_")

    def parent_run_id(self) -> str | None:
        parents = self.subgraphs
        if not parents:
            return None
        return stable_id(
            self.thread_id, self.checkpoint_ns, parents[-1].task_id, parents[-1].node,
            prefix="run_lg_",
        )

    def turn_id(self) -> str | None:
        if not self.thread_id:
            return None
        return safe_id(f"{self.thread_id}-step{self.step if self.step is not None else 0}")


def lineage_from_config(config: Mapping[str, Any] | None) -> Lineage:
    """Parse a ``RunnableConfig``. Unknown/missing keys degrade to ``None``, never raise."""
    conf = dict((config or {}).get("configurable") or {})
    meta = dict((config or {}).get("metadata") or {})
    ns = str(conf.get("checkpoint_ns") or "")
    segments = tuple(Segment.parse(s) for s in ns.split("|") if s)
    node = meta.get("langgraph_node") or (segments[-1].node if segments else None)
    step = meta.get("langgraph_step")
    overrides = {
        k: v for k, v in dict(conf.get("harness") or {}).items() if k in IDENTITY_FIELDS
    }
    return Lineage(
        thread_id=conf.get("thread_id"),
        segments=segments,
        checkpoint_id=conf.get("checkpoint_id"),
        checkpoint_ns=ns,
        run_id=str((config or {}).get("run_id") or "") or None,
        step=int(step) if step is not None else None,
        node=str(node) if node else None,
        overrides=overrides,
    )


def context_fields(lineage: Lineage, *, thread_prefix: str = "") -> dict[str, Any]:
    """Identity fields for :class:`AgentExecutionContext`, derived from the graph position."""
    fields: dict[str, Any] = {}
    if lineage.thread_id:
        thread = safe_id(f"{thread_prefix}{lineage.thread_id}")
        fields["thread_id"] = thread
        fields["session_id"] = safe_id(f"{thread}-session")
        fields["turn_id"] = lineage.turn_id()
    if lineage.task is not None:
        fields["task_id"] = safe_id(lineage.task.task_id)
    if lineage.run_id:
        fields["correlation_id"] = safe_id(lineage.run_id)
    fields.update(lineage.overrides)
    return {k: v for k, v in fields.items() if v is not None}
