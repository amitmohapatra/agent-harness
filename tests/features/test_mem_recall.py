"""MEM-02, MEM-03 on every adapter, against the real memory service: a fact a user tells the
agent in one chat is what the agent knows in the user's next chat — pushed into what its model
is sent, and found by its ``memory_search`` tool — and it never reaches another user of the
same agent, nor the same user id in another tenant. The memory database is read too: the
fact is stored once, as the user's, in their tenant."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.features.conftest import (
    ADAPTERS,
    OTHER_KEY,
    eventually,
    harness,
    memory_http,
    memory_sql,
)
from tests.features.kit import armed
from trellis.contracts import RunStatus

pytestmark = pytest.mark.timeout(300)

FACT = "My favourite warehouse is in Rotterdam."
QUESTION = "Where is my favourite warehouse?"
SEARCH = ("memory_search", {"query": "favourite warehouse", "kinds": ["memory", "message"]})


def facts_of(user: str) -> list[dict]:
    return memory_sql(
        "select tenant_id, user_id, memory_type, content, visibility, visibility_keys,"
        " superseded_by, deleted_at from memories where user_id = :1 and deleted_at is null",
        user,
    )


@pytest.mark.plan("MEM-02", "MEM-03", "INV-5")
@pytest.mark.parametrize("adapter", ADAPTERS)
async def test_a_fact_told_in_one_chat_is_known_in_the_next_and_to_no_one_else(
    adapter: str, tmp_path: Path, uid: str
) -> None:
    user, other = f"ada-{uid}", f"bob-{uid}"
    agent_id = f"recall-{adapter}-{uid}"
    pulls = adapter != "function"  # a function target is given no memory tools (matrix F42)
    async with harness() as h:
        agent, _ = await armed(adapter, h, tmp_path, [], id=agent_id)
        told = await agent.run(FACT, user=user, thread="chat-1")
        assert told.status is RunStatus.SUCCESS, told
        await h.writes.drain()

    async def extracted() -> list[dict]:
        rows = facts_of(user)
        return rows if any(r["memory_type"] != "OBSERVATION" for r in rows) else []

    rows = await eventually(extracted, within=90)
    assert rows, facts_of(user)
    assert {r["tenant_id"] for r in rows} == {"feat"}
    assert all("Rotterdam" in r["content"] for r in rows if r["memory_type"] != "OBSERVATION")
    assert all(user in " ".join(r["visibility_keys"]) for r in rows), rows  # the user's alone

    # the user's next chat, in another process: pushed to the model, and found by its tool
    async with harness() as h:
        plan = [SEARCH] if pulls else []
        agent, built = await armed(adapter, h, tmp_path, plan, id=agent_id)
        asked = await agent.run(QUESTION, user=user, thread="chat-2")
        assert asked.status is RunStatus.SUCCESS, asked
        assert "Rotterdam" in built.said(), built.said()[-2000:]  # pushed
        if pulls:
            assert "Rotterdam" in str(asked.answer), asked.answer  # pulled by memory_search

        # another user of the same agent, asking the same: nothing of the first user's
        agent2, built2 = await armed(adapter, h, tmp_path, plan, id=f"{agent_id}-b")
        theirs = await agent2.run(QUESTION, user=other, thread="chat-1")
        assert theirs.status is RunStatus.SUCCESS, theirs
        assert "Rotterdam" not in built2.said()
        assert "Rotterdam" not in str(theirs.answer), theirs.answer

    # the same user id in another tenant: nothing either
    async with harness(key=OTHER_KEY) as elsewhere:
        plan = [SEARCH] if pulls else []
        agent3, built3 = await armed(adapter, elsewhere, tmp_path, plan, id=agent_id)
        foreign = await agent3.run(QUESTION, user=user, thread="chat-2")
        assert foreign.status is RunStatus.SUCCESS, foreign
        assert "Rotterdam" not in built3.said()
        assert "Rotterdam" not in str(foreign.answer)


@pytest.mark.plan("MEM-03", "INV-5")
async def test_recall_and_the_inventory_never_cross_a_user_or_a_tenant(
    tmp_path: Path, uid: str
) -> None:
    """The service's own reads, asked directly as another user and as another tenant: none
    returns a memory of the user's, and the database holds them under the user alone."""
    user = f"cara-{uid}"
    async with harness() as h:
        agent, _ = await armed("react", h, tmp_path, [], id=f"iso-{uid}")
        await agent.run("My locker code is 4417 and I work in Lisbon.", user=user, thread="t")
        await h.writes.drain()
    await eventually(lambda: _any(user), within=90)
    query = {"query": "locker code Lisbon", "kinds": ["memory", "message"], "limit": 50}
    async with memory_http() as mine:
        got = await mine.post("/v1/recall", json={**query, "scope": {"user_id": user}})
        assert got.status_code == 200, got.text
        assert "4417" in got.text  # the user's own recall finds it
        stranger = await mine.post(
            "/v1/recall", json={**query, "scope": {"user_id": f"dan-{uid}"}}
        )
        assert stranger.status_code == 200 and "4417" not in stranger.text
        listed = await mine.get("/v1/memories", params={"user_id": f"dan-{uid}", "limit": 100})
        assert listed.status_code == 200 and "4417" not in listed.text
    async with memory_http(OTHER_KEY) as foreign:
        got = await foreign.post("/v1/recall", json={**query, "scope": {"user_id": user}})
        assert got.status_code == 200 and "4417" not in got.text
        listed = await foreign.get("/v1/memories", params={"user_id": user, "limit": 100})
        assert "4417" not in listed.text
    owners = memory_sql(
        "select distinct tenant_id, user_id from memories where content like '%4417%'"
        " and deleted_at is null"
    )
    assert {(o["tenant_id"], o["user_id"]) for o in owners} <= {("feat", user)}, owners
    said = memory_sql(
        "select distinct t.tenant_id, t.user_id from messages m join threads t"
        " on t.thread_id = m.thread_id and t.tenant_id = m.tenant_id"
        " where m.content like '%4417%'"
    )
    assert {(o["tenant_id"], o["user_id"]) for o in said} == {("feat", user)}, said


async def _any(user: str) -> list[dict]:
    return [r for r in facts_of(user) if r["memory_type"] != "OBSERVATION"]
