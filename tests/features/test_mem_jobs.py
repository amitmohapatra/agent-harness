"""MEM-06 and MEM-15 against the real memory service and its job worker:

* retention: a tenant whose policy keeps memories 1 day; a memory created 3 days ago is
  forgotten by the retention sweep (gone from recall, marked deleted in the store), one made
  today is kept — and another tenant's old memory is not touched;
* the archive read path: a conversation's messages are archived into a verified segment; once
  the purge grace has passed the large ones are emptied in the hot store, and reading the
  thread back serves every message whole, from the segment.

The two sweeps are periodic jobs (daily, hourly): each is run once on demand
(:func:`run_periodic_now`), and a row's age is moved back where the policy counts it
(:func:`move_clock`)."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.features.conftest import (
    MEMORY_DSN,
    MEMORY_URL,
    PLATFORM_KEY,
    eventually,
    harness,
    memory_sql,
    move_clock,
    run_periodic_now,
)
from tests.features.kit import recall_text
from trellis.memory import MemoryClient

pytestmark = pytest.mark.timeout(300)


@pytest.mark.plan("MEM-06")
async def test_retention_forgets_what_is_older_than_the_tenants_policy(uid: str) -> None:
    tenant = f"featret{uid}"
    async with MemoryClient(MEMORY_URL, api_key=PLATFORM_KEY, timeout=60) as platform:
        created = await platform.admin.create_tenant(
            f"Ret {uid}", tenant_id=tenant, retention_days=1
        )
    async with MemoryClient(MEMORY_URL, api_key=created.admin_key.token, timeout=60) as admin:
        service = await admin.administer(tenant).keys.issue("service", "ret")
    user = f"ola-{uid}"
    async with harness(key=service.token) as h:
        ctx = h.memory.client.bind(tenant_id=tenant, user_id=user)  # type: ignore[union-attr]
        old = await ctx.remember(f"My old badge was OLD{uid}.")
        new = await ctx.remember(f"My new badge is NEW{uid}.")
    async with harness() as feat:
        keep = await feat.memory.client.bind(tenant_id="feat", user_id=user).remember(  # type: ignore[union-attr]
            f"Elsewhere my badge was FEAT{uid}."
        )
    old_id, new_id, keep_id = (m.memory_id for m in (old, new, keep))
    for memory_id in (old_id, keep_id):
        move_clock(
            MEMORY_DSN,
            "update memories set created_at = now() - interval '3 days' where memory_id = :1",
            memory_id,
        )
    run_periodic_now("periodic.retention", "reconcile")

    async def forgotten() -> bool:
        rows = memory_sql("select deleted_at from memories where memory_id = :1", old_id)
        return bool(rows and rows[0]["deleted_at"])

    assert await eventually(forgotten, within=90)
    rows = {
        r["memory_id"]: r["deleted_at"]
        for r in memory_sql(
            "select memory_id, deleted_at from memories where memory_id in (:1, :2, :3)",
            old_id,
            new_id,
            keep_id,
        )
    }
    assert rows[new_id] is None  # today's is kept
    assert rows[keep_id] is None  # another tenant's policy (none) is its own
    async with harness(key=service.token) as h:
        found = await recall_text(h, user, "badge", ["memory"])
    assert f"OLD{uid}" not in found and f"NEW{uid}" in found


@pytest.mark.plan("MEM-15")
@pytest.mark.timeout(400)
async def test_a_purged_message_is_read_back_whole_from_its_archive_segment(
    tmp_path: Path, uid: str
) -> None:
    user = f"pia-{uid}"
    big = [
        f"Line {n} of my inventory notes for {uid}: " + "pallet bay row " * 400 for n in range(3)
    ]
    async with harness() as h:
        thread = await h.conversation(user=user, thread="notes")
        assert h.memory is not None
        ctx = h.memory.client.bind(
            tenant_id=await h._tenant(),
            user_id=user,
            agent_id=f"arch-{uid}",
            thread_id=thread,
        )
        await ctx.history.add([("USER", text) for text in big])

    async def archived() -> list[dict]:
        rows = memory_sql(
            "select message_id, archive_status, archive_segment_id from messages"
            " where thread_id = :1",
            thread,
        )
        return rows if rows and all(r["archive_status"] == "ARCHIVED" for r in rows) else []

    rows = await eventually(archived, within=180, every=3)
    assert rows, memory_sql("select archive_status from messages where thread_id = :1", thread)
    segments = memory_sql(
        "select status, checksum_sha256, message_count from archive_segments where thread_id = :1",
        thread,
    )
    assert segments and all(s["status"] == "VERIFIED" for s in segments), segments
    move_clock(
        MEMORY_DSN,
        "update messages set archived_at = now() - interval '2 days' where thread_id = :1",
        thread,
    )
    run_periodic_now("periodic.archive_purge", "archive")

    async def purged() -> bool:
        got = memory_sql(
            "select count(*) as n from messages where thread_id = :1 and archive_status = 'PURGED'",
            thread,
        )
        return got[0]["n"] == len(big)

    assert await eventually(purged, within=120, every=3), memory_sql(
        "select archive_status, content is null as empty from messages where thread_id = :1", thread
    )
    assert all(
        r["empty"]
        for r in memory_sql(
            "select content is null as empty from messages where thread_id = :1", thread
        )
    )
    async with harness() as h:
        assert h.memory is not None
        ctx = h.memory.client.bind(
            tenant_id=await h._tenant(),
            user_id=user,
            agent_id=f"arch-{uid}",
            thread_id=thread,
        )
        history = await ctx.history(limit=50)
    assert [m.content for m in history] == big  # whole, from the segment
