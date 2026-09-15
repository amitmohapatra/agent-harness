"""Where an :class:`AgentExecutionContext` comes from (§5).

Resolution order, most specific first:

1. an explicit ``context=`` argument;
2. a context bound in this async task (a parent agent run) — the new run becomes its child;
3. the harness defaults (tenant, workspace, user...) — for a top-level execution.

Run ids are **derived**, not random, whenever the execution has a durable position (thread
+ turn/task + agent). That is what makes a framework replay of the same step produce the
same run id, and therefore the same idempotency keys (§42/§55).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from universal_agent_harness.contracts.context import AgentExecutionContext
from universal_agent_harness.contracts.ids import new_id, safe_id, stable_id
from universal_agent_harness.runtime.propagation import current_context


class ContextFactory:
    """Builds execution contexts from defaults plus whatever the call site supplies."""

    def __init__(
        self,
        defaults: Mapping[str, Any] | None = None,
        *,
        deterministic_run_ids: bool = True,
    ) -> None:
        self.defaults = dict(defaults or {})
        self.deterministic_run_ids = deterministic_run_ids

    def with_defaults(self, **defaults: Any) -> ContextFactory:
        return ContextFactory(
            {**self.defaults, **defaults}, deterministic_run_ids=self.deterministic_run_ids
        )

    def build(
        self,
        *,
        agent_id: str,
        context: AgentExecutionContext | None = None,
        parent: AgentExecutionContext | None = None,
        overrides: Mapping[str, Any] | None = None,
        timeout_seconds: float | None = None,
    ) -> AgentExecutionContext:
        fields = {k: v for k, v in (overrides or {}).items() if v is not None}
        explicit_run_id = fields.pop("agent_run_id", None)

        if context is not None:
            # An explicit context wins; it is only re-pointed at this agent when it names a
            # different one, so a caller can hand the same context to a nested agent safely.
            if context.agent_id == safe_id(agent_id) and not fields:
                return context
            if context.agent_id == safe_id(agent_id):
                return context.with_fields(**fields)
            return context.for_agent(
                agent_id,
                agent_run_id=explicit_run_id or self._run_id(agent_id, context, fields),
                **{k: v for k, v in fields.items() if k in ("agent_group_id", "task_id")},
            ).with_fields(
                **{k: v for k, v in fields.items() if k not in ("agent_group_id", "task_id")}
            )

        base = parent if parent is not None else current_context()
        if base is not None:
            child = base.for_agent(
                agent_id,
                agent_run_id=explicit_run_id or self._run_id(agent_id, base, fields),
                agent_group_id=fields.pop("agent_group_id", None),
                task_id=fields.pop("task_id", None),
            )
            return child.with_fields(**fields) if fields else child

        merged: dict[str, Any] = {**self.defaults, **fields}
        tenant_id = merged.pop("tenant_id", None)
        if not tenant_id:
            raise ValueError(
                "no tenant_id available: pass context=..., or set tenant_id on the harness "
                "(AgentHarness(defaults={'tenant_id': ...}))"
            )
        merged.pop("agent_id", None)
        run_id = explicit_run_id or self._derive(agent_id, merged)
        return AgentExecutionContext.create(
            tenant_id=tenant_id,
            agent_id=agent_id,
            agent_run_id=run_id,
            timeout_seconds=timeout_seconds,
            **merged,
        )

    # -- run ids ---------------------------------------------------------------------
    def _run_id(
        self, agent_id: str, base: AgentExecutionContext, fields: Mapping[str, Any]
    ) -> str | None:
        if not self.deterministic_run_ids:
            return None
        anchor = fields.get("task_id") or base.task_id or base.turn_id
        if not (base.thread_id and anchor):
            return None
        return stable_id(base.thread_id, anchor, base.agent_run_id, agent_id, prefix="run_")

    def _derive(self, agent_id: str, fields: Mapping[str, Any]) -> str:
        if not self.deterministic_run_ids:
            return new_id("run_")
        thread = fields.get("thread_id")
        anchor = fields.get("task_id") or fields.get("turn_id")
        if thread and anchor:
            return stable_id(thread, anchor, agent_id, prefix="run_")
        return new_id("run_")
