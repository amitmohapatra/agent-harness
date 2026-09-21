"""Execution context: immutability, inheritance, deterministic keys (§5, §42)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from universal_agent_harness import AgentExecutionContext


def test_create_fills_ids():
    ctx = AgentExecutionContext.create(tenant_id="acme", agent_id="inv")
    assert ctx.request_id.startswith("req_")
    assert ctx.agent_run_id.startswith("run_")
    assert ctx.correlation_id == ctx.request_id
    assert ctx.trace_id


def test_context_is_immutable():
    ctx = AgentExecutionContext.create(tenant_id="acme")
    with pytest.raises(ValidationError):
        ctx.tenant_id = "other"  # type: ignore[misc]


def test_child_inherits_trusted_identity_and_records_lineage():
    parent = AgentExecutionContext.create(
        tenant_id="acme",
        agent_id="planner",
        user_id="u1",
        workspace_id="w1",
        thread_id="t1",
        turn_id="turn-1",
        agent_group_id="crew",
        group_ids=("g1",),
    )
    child = parent.for_agent("inventory")

    for field in (
        "tenant_id",
        "user_id",
        "workspace_id",
        "thread_id",
        "turn_id",
        "trace_id",
        "request_id",
        "correlation_id",
        "group_ids",
    ):
        assert getattr(child, field) == getattr(parent, field), field
    assert child.agent_id == "inventory"
    assert child.parent_agent_run_id == parent.agent_run_id
    assert child.causation_id == parent.agent_run_id
    assert child.agent_run_id != parent.agent_run_id
    assert child.agent_group_id == "crew"


def test_child_deadline_never_outlives_parent():
    now = datetime.now(UTC)
    parent = AgentExecutionContext.create(tenant_id="acme", deadline=now + timedelta(seconds=5))
    later = parent.for_agent("child", deadline=now + timedelta(seconds=60))
    earlier = parent.for_agent("child", deadline=now + timedelta(seconds=1))
    assert later.deadline == parent.deadline
    assert earlier.deadline == now + timedelta(seconds=1)


def test_idempotency_key_is_stable_across_equivalent_contexts():
    fields = {
        "tenant_id": "acme",
        "agent_id": "inv",
        "thread_id": "t1",
        "turn_id": "turn-1",
        "agent_run_id": "run_fixed",
    }
    a = AgentExecutionContext.create(**fields)
    b = AgentExecutionContext.create(**fields)  # different request/trace ids
    assert a.request_id != b.request_id
    assert a.idempotency_key("obs", "hello") == b.idempotency_key("obs", "hello")
    assert a.idempotency_key("obs", "hello") != a.idempotency_key("obs", "other")


def test_scope_fields_drop_empties_and_map_to_memory_scope():
    ctx = AgentExecutionContext.create(tenant_id="acme", agent_id="inv", thread_id="t1")
    scope = ctx.scope_fields()
    assert scope["tenant_id"] == "acme"
    assert scope["agent_id"] == "inv"
    assert "workspace_id" not in scope
    assert "user_id" not in scope


def test_remaining_seconds_and_expiry():
    ctx = AgentExecutionContext.create(tenant_id="acme", timeout_seconds=0.0)
    assert ctx.remaining_seconds == 0.0
    assert ctx.expired
    unbounded = AgentExecutionContext.create(tenant_id="acme")
    assert unbounded.remaining_seconds is None
    assert not unbounded.expired


def test_agent_id_is_sanitised():
    ctx = AgentExecutionContext.create(tenant_id="acme", agent_id="my agent/v2")
    assert ctx.agent_id == "my-agent-v2"
