"""MEM-02, MEM-03 on every adapter, against the real memory service: a fact a user tells the
agent in one chat is what the agent knows in the user's next chat — pushed into what its model
is sent, and found by its ``memory_search`` tool — and it never reaches another user of the
same agent, nor the same user id in another tenant. The memory database is read too: the
fact is stored once, as the user's, in their tenant."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.features.conftest import (
    ADAPTERS,
    OTHER_KEY,
    eventually,
    harness,
    memory_sql,
)
from tests.features.kit import armed, inventory_text, memories_of, recall_text, user_key
from trellis.contracts import RunStatus

pytestmark = pytest.mark.timeout(300)

FACT = "My favourite warehouse is in Rotterdam."
QUESTION = "Where is my favourite warehouse?"
SEARCH = ("memory_search", {"query": "favourite warehouse", "kinds": ["memory", "message"]})


def facts_of(user: str) -> list[dict]:
    return memories_of(user)


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
        ready = [r for r in rows if r["memory_type"] != "OBSERVATION" and r["indexed_at"]]
        return rows if ready else []  # extracted, and indexed: what recall searches

    rows = await eventually(extracted, within=90)
    assert rows, facts_of(user)
    assert {r["tenant_id"] for r in rows} == {"feat"}
    assert all("Rotterdam" in r["content"] for r in rows if r["memory_type"] != "OBSERVATION")
    assert all(user_key(user) in r["visibility_keys"] for r in rows), rows  # the user's alone

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
    code = str(int(uid, 16) % 900_000 + 100_000)  # this test's own secret
    async with harness() as h:
        agent, _ = await armed("react", h, tmp_path, [], id=f"iso-{uid}")
        await agent.run(f"My locker code is {code} and I work in Lisbon.", user=user, thread="t")
        await h.writes.drain()
    rows = await eventually(lambda: _any(user), within=90)
    assert rows, memories_of(user)
    query = "locker code Lisbon"
    async with harness() as h:
        assert code in await recall_text(h, user, query)  # the user's own recall finds it
        assert code not in await recall_text(h, f"dan-{uid}", query)
        assert code not in await inventory_text(h, f"dan-{uid}")
    async with harness(key=OTHER_KEY) as foreign:
        assert code not in await recall_text(foreign, user, query)
        assert code not in await inventory_text(foreign, user)
    owners = memory_sql(
        "select distinct tenant_id, visibility_keys::text as keys from memories"
        " where content like '%' || :1 || '%' and deleted_at is null",
        code,
    )
    assert owners and {o["tenant_id"] for o in owners} == {"feat"}, owners
    for o in owners:  # readable by the user (and the agent acting for them), nobody else
        keys = json.loads(o["keys"])
        assert user_key(user) in keys and all(user in k for k in keys), keys
    said = memory_sql(
        "select distinct t.tenant_id, t.owner_user_id as user_id from messages m join threads t"
        " on t.thread_id = m.thread_id and t.tenant_id = m.tenant_id"
        " where m.content like '%' || :1 || '%'",
        code,
    )
    assert {(o["tenant_id"], o["user_id"]) for o in said} == {("feat", user)}, said


async def _any(user: str) -> list[dict]:
    return [r for r in memories_of(user) if r["memory_type"] != "OBSERVATION"]


@pytest.mark.plan("MEM-03", "MEM-14", "INV-5")
async def test_two_users_adding_the_same_file_each_get_a_document_of_their_own(
    tmp_path: Path, uid: str
) -> None:
    """The same bytes under the same name, added for two users of one tenant: two documents,
    each its user's, each cited by its user's next context. (The SDK keyed the upload on the
    bytes alone: the second user got the first's document id back and could not read it.)"""
    doc = tmp_path / "terms.md"
    doc.write_text(f"# Terms {uid}\n\nPallets are delivered on Mondays by Uline.\n")
    async with harness() as h:
        first = await h.add_document(doc, user=f"ivy-{uid}", wait=120)
        second = await h.add_document(doc, user=f"jon-{uid}", wait=120)
    assert first.document_id != second.document_id
    owners = memory_sql(
        "select document_id, owner_user_id from documents where document_id in (:1, :2)",
        first.document_id,
        second.document_id,
    )
    assert {(o["document_id"], o["owner_user_id"]) for o in owners} == {
        (first.document_id, f"ivy-{uid}"),
        (second.document_id, f"jon-{uid}"),
    }
