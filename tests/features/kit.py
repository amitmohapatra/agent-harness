"""The feature suite's tools, each side effect booked in a ledger (a file when ``FEAT_LEDGER``
names one, so a worker process books too), and helpers reading back what the services keep."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

from tests.features.conftest import ADMIN_KEY, Built, build, memory_sql, runs_http
from tests.support.arm import arm
from trellis import Harness, tool
from trellis.contracts import RunRecord, RunStatus
from trellis.harness.agent import Agent
from trellis.harness.result import Result

LEDGER: list[str] = []


def book(entry: str) -> None:
    LEDGER.append(entry)
    path = os.environ.get("FEAT_LEDGER")
    if path:
        with Path(path).open("a") as out:
            out.write(entry + "\n")


def ledger() -> list[str]:
    path = os.environ.get("FEAT_LEDGER")
    if path and Path(path).exists():
        return Path(path).read_text().splitlines()
    return list(LEDGER)


@tool(side_effects="read")
def lookup(sku: str) -> str:
    """Look a SKU's stock up."""
    book(f"lookup:{sku}")
    return f"stock of {sku} is 3"


@tool(side_effects="write")
def restock(sku: str, qty: int) -> str:
    """Order more of a SKU (a write: asks once a conversation)."""
    book(f"restock:{sku}:{qty}")
    return f"ordered {qty} of {sku}"


@tool(side_effects="irreversible")
def wire(amount: int, to: str, meta: dict[str, Any] | None = None) -> str:
    """Wire money (irreversible: asks at every call)."""
    book(f"wire:{amount}:{to}:{json.dumps(meta or {}, sort_keys=True)}")
    return f"wired {amount} to {to}"


@tool(side_effects="read")
def big_report(rows: int) -> str:
    """A large report: ``rows`` lines."""
    book(f"big_report:{rows}")
    return "\n".join(f"row {i:06d} sku-{i % 97:03d} qty {i % 13}" for i in range(rows))


TOOLS: Final = [lookup, restock, wire, big_report]


async def armed(
    adapter: str,
    h: Harness,
    tmp: Path,
    plan: list[Any],
    *,
    id: str,
    tools: Sequence[Any] = TOOLS,
    **wrap: Any,
) -> tuple[Agent, Built]:
    """``adapter`` following ``plan``, wrapped by ``h`` as ``id`` with ``tools``."""
    built = await build(adapter, h, list(tools), tmp, plan)
    return arm(h.wrap(built.target, id=id, **wrap), built.tools), built


async def approve_all(agent: Agent, result: Result, *, reviewer: str = "lee", **kw: Any) -> Result:
    """``result`` with every question it pauses on approved (each once), to its end."""
    for _ in range(10):
        if result.interrupt is None:
            return result
        result = await agent.resume(
            result.interrupt.interrupt_id, "approve", reviewer=reviewer, **kw
        )
    raise AssertionError(f"still paused after ten approvals: {result}")


async def record(run_id: str, *, key: str | None = None) -> RunRecord:
    """The run as agent-runs keeps it, read over its API and parsed as the contract."""
    async with runs_http(key) as c:
        got = await c.get(f"/v1/runs/{run_id}")
        got.raise_for_status()
        return RunRecord.model_validate(got.json())


def ended(record: RunRecord) -> bool:
    return record.status not in (RunStatus.QUEUED, RunStatus.RUNNING, RunStatus.PAUSED)


def messages_of_run(run_id: str) -> list[dict[str, Any]]:
    """The transcript rows the memory service keeps for a run, oldest first."""
    return memory_sql(
        "select m.role, m.content, m.tool_call_id, m.tool_calls, m.source_message_id,"
        " m.thread_id, m.content_bytes from messages m where m.agent_run_id = :1"
        " and m.deleted_at is null order by m.sequence",
        run_id,
    )


def tool_records(run_id: str) -> list[dict[str, Any]]:
    return memory_sql(
        "select tool_name, args_redacted, status, output_summary, length(output_summary) as size,"
        " user_id, agent_id, idempotency_key from tool_invocations where run_id = :1"
        " order by occurred_at",
        run_id,
    )


def feedback_rows(run_id: str) -> list[dict[str, Any]]:
    return memory_sql(
        "select target_kind, target_id, verdict, source, metadata, review_state, evidence_refs,"
        " user_id, agent_id from feedback where agent_run_id = :1 order by created_at",
        run_id,
    )


ADMIN: Final = ADMIN_KEY


def user_key(user: str, tenant: str = "feat") -> str:
    """The visibility key of a user's own memories (USER visibility)."""
    return f"user:{tenant}/{user}"


def memories_of(user: str, *, tenant: str = "feat", live: bool = True) -> list[dict[str, Any]]:
    """The memories a user may read as their own, from the store: every scope (a thread's
    memories carry the user's key, not ``user_id``)."""
    pattern = f'%"{user_key(user, tenant)}"%'
    rows = memory_sql(
        "select memory_id, tenant_id, user_id, scope_level, memory_type, content, predicate,"
        " normalized_hash, visibility, visibility_keys, superseded_by, valid_to,"
        " temporal_status, reinforcement_count, deleted_at, expires_at, created_at, indexed_at"
        " from memories where visibility_keys::text like :1",
        pattern,
    )
    if live:
        rows = [r for r in rows if r["deleted_at"] is None and r["superseded_by"] is None]
    return rows


async def recall_text(
    h: Harness, user: str, query: str, kinds: Sequence[str] = ("memory", "message")
) -> str:
    """Everything the memory service's recall answers ``user`` for ``query``, as text."""
    assert h.memory is not None
    ctx = h.memory.client.bind(tenant_id=await h._tenant(), user_id=user)
    found = await ctx.search(query, kinds=list(kinds), limit=50)
    return json.dumps(
        found.model_dump(mode="json")
        if hasattr(found, "model_dump")
        else [f.model_dump(mode="json") for f in found],
        default=str,
    )


async def inventory_text(h: Harness, user: str) -> str:
    assert h.memory is not None
    ctx = h.memory.client.bind(tenant_id=await h._tenant(), user_id=user)
    page = await ctx.advanced.memories.page(limit=100)
    return json.dumps(
        page.model_dump(mode="json")
        if hasattr(page, "model_dump")
        else [p.model_dump(mode="json") for p in page],
        default=str,
    )


async def scope_of(h: Harness, user: str, *, agent: str | None = None, thread: str | None = None):
    """The memory SDK bound to a user (and an agent, a conversation) of ``h``'s tenant."""
    from tests.features.conftest import scope

    return await scope(h, user=user, agent=agent, thread=thread)
