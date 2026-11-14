"""MEM-04, MEM-05, MEM-07, MEM-13 against the real memory service: what memory makes of what
a user says over time.

* extraction and consolidation: a fact said in three chats is one fact (reinforced, never a
  duplicate), and a correction replaces it — the old value closed, the new one served;
* forget: the agent's ``memory_forget`` of the user's own words asks a person first, then the
  memory is gone from recall and marked deleted in the store; erasing a user (their memories
  and their conversations) leaves nothing any read returns;
* the profile an agent edits is in the user's next context, in every chat;
* the memory budget the model is given: 8,000 tokens, a quarter of a known small window,
  never under 2,000;
* the memories a run cites are capped (50) on its outcome;
* two chats of one user writing at once lose nothing and duplicate nothing.

The memory database is read for every one."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from tests.features.conftest import (
    ADAPTERS,
    ADMIN_KEY,
    MEMORY_URL,
    Traffic,
    eventually,
    harness,
    memory_sql,
)
from tests.features.kit import (
    approve_all,
    armed,
    inventory_text,
    memories_of,
    recall_text,
    scope_of,
)
from tests.support.planned import PlannedChat
from trellis import ReAct
from trellis.contracts import RunStatus
from trellis.memory import MemoryClient
from trellis.memory.errors import NotFoundError

pytestmark = pytest.mark.timeout(300)

FACT = "My favourite warehouse is in Rotterdam."


def live_facts(user: str, predicate: str | None = None) -> list[dict]:
    """The user's current memories (not deleted, not superseded), every scope."""
    rows = memories_of(user)
    return [r for r in rows if predicate is None or r["predicate"] == predicate]


def _no_duplicates(user: str) -> None:
    """No two live memories of the user say the same, among the readings and among the
    verbatim turns (a turn is kept beside its reading in the same words, by design: ADR 0036)."""
    for turns in (True, False):
        rows = [r for r in live_facts(user) if (r["memory_type"] == "OBSERVATION") is turns]
        hashes = [r["normalized_hash"] for r in rows]
        assert len(hashes) == len(set(hashes)), rows


async def tell(  # noqa: PLR0917 - a chat turn: who says what, where
    adapter: str, tmp: Path, agent_id: str, user: str, thread: str, text: str
) -> None:
    async with harness() as h:
        agent, _ = await armed(adapter, h, tmp, [], id=agent_id)
        done = await agent.run(text, user=user, thread=thread)
        assert done.status is RunStatus.SUCCESS, done
        await h.writes.drain()


@pytest.mark.plan("MEM-04")
async def test_a_fact_said_in_three_chats_is_one_fact_and_a_correction_replaces_it(
    tmp_path: Path, uid: str
) -> None:
    user, agent_id = f"eve-{uid}", f"facts-{uid}"
    for n, text in enumerate([FACT, FACT, "my favourite warehouse is in Rotterdam"], 1):
        await tell("react", tmp_path, agent_id, user, f"chat-{n}", text)

    async def settled() -> list[dict]:
        rows = live_facts(user, "favourite_warehouse")
        return rows if rows else []

    facts = await eventually(settled, within=90)
    assert len(facts) == 1, facts  # one fact, however often it was said
    assert "Rotterdam" in facts[0]["content"]
    _no_duplicates(user)

    await tell(
        "react", tmp_path, agent_id, user, "chat-4", "No, my favourite warehouse is now in Hamburg."
    )

    async def corrected() -> list[dict]:
        rows = live_facts(user, "favourite_warehouse")
        return rows if rows and all("Hamburg" in r["content"] for r in rows) else []

    now = await eventually(corrected, within=90)
    assert now and len(now) == 1, live_facts(user, "favourite_warehouse")
    old = memory_sql(
        "select superseded_by, valid_to, temporal_status from memories where memory_id = :1",
        facts[0]["memory_id"],
    )[0]
    assert old["superseded_by"] == now[0]["memory_id"] or old["valid_to"] is not None, old
    # the next chat is told the new value as the fact
    async with harness() as h:
        agent, built = await armed("react", h, tmp_path, [], id=agent_id)
        await agent.run("Where is my favourite warehouse?", user=user, thread="chat-5")
    said = built.said()
    assert "Hamburg" in said, said[-3000:]
    memories = said.split("## Memories", 1)[-1].split("##", 1)[0]
    assert "Rotterdam" not in memories or "earlier" in memories, memories


#: BUG-M1 (agent-memory-service): a fact the rules read out of a user's own turn, relayed by
#: the harness, is owned by ``agent:<user>/<agent>`` and ``said_by_a_person``
#: (modules/memory/pipeline.py:358) only knows a ``user:`` owner or a ``said`` turn, so
#: ``_require_owner`` (modules/memory/service.py:276) lets the agent forget or rewrite it:
#: ``memory_forget`` runs unasked, against docs/memory.md ("what the user said ... after a
#: person approves it").
BUG_M1 = (
    "BUG-M1: the user's extracted facts are not recognised as their own words "
    "(memory-service pipeline.py:358); the agent forgets them without asking"
)
#: BUG-M2 (agent-memory-service, fixed on its branch claude/cool-maxwell-w5avo8-features): the
#: user an agent acts for was refused forgetting what that agent relayed for them —
#: ``_require_owner`` (modules/memory/service.py:288) accepted only the owner principal or
#: ``user:<id>``, the owner being ``agent:<user>/<agent>`` (USAGE §4: "the memory's owner, the
#: user an agent acts for, or a tenant admin"). These tests need that fix in the service.
#: BUG-M3 (agent-memory-service): the inventory (GET /v1/memories, "what do we hold about
#: this user?") lists memories anchored at the user's scope only, while ADR 0036 anchors a
#: user's statements at their thread with USER visibility: list_memories
#: (modules/memory/service.py:322-396) never lists them without the thread's id.
BUG_M3 = (
    "BUG-M3: the user's inventory misses every statement anchored in a thread "
    "(memory-service service.py:340)"
)


async def _forget_by_the_agent(
    adapter: str, tmp: Path, uid: str, pick: str
) -> tuple[str, str, str]:
    """A user tells a fact; the agent's ``memory_forget`` of one of its memories (``pick``:
    ``fact`` the reading, ``turn`` the verbatim turn) must ask that user, then forget it."""
    user, agent_id = f"fay-{uid}", f"forget-{adapter}-{uid}"
    # a reading that is not lasting (a fact, not a preference): its turn is kept beside it
    await tell(
        adapter, tmp, agent_id, user, "chat-1", "My locker code is 4417 and I work in Lisbon."
    )
    rows = await eventually(lambda: _facts(user), within=90)
    assert rows, live_facts(user)
    if pick == "turn":
        rows = [r for r in live_facts(user) if r["predicate"] == "said"]
    memory_id = rows[0]["memory_id"]
    async with harness() as h:
        plan = [("memory_forget", {"id": memory_id})]
        agent, _ = await armed(adapter, h, tmp, plan, id=agent_id)
        paused = await agent.run("Forget my locker code.", user=user, thread="c2")
        assert paused.interrupt is not None, paused  # what the user said: a person approves
        assert paused.interrupt.tool_call is not None
        assert paused.interrupt.tool_call.tool == "memory_forget"
        done = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer=user)
        assert done.status is RunStatus.SUCCESS, done
        assert done.interrupt is None  # asked once
        assert "forgotten" in str(done.answer), done.answer
    row = memory_sql("select deleted_at from memories where memory_id = :1", memory_id)[0]
    assert row["deleted_at"] is not None  # gone from the store
    async with harness() as h:
        assert memory_id not in await recall_text(h, user, "locker code")
        ctx = await scope_of(h, user)
        with pytest.raises(NotFoundError):  # gone: a read by its id finds nothing
            await ctx.advanced.memories.get(memory_id)
    return user, agent_id, memory_id


@pytest.mark.plan("MEM-05", "INV-3", "INV-4")
@pytest.mark.parametrize("adapter", [a for a in ADAPTERS if a != "function"])
async def test_forgetting_the_users_words_asks_then_they_are_gone(
    adapter: str, tmp_path: Path, uid: str, invariants: Any
) -> None:
    """``memory_forget`` of the user's turn as they said it: the service refuses the agent,
    the run pauses on one approval, and approved it is forgotten as the user — gone from
    recall and marked deleted in the store (before BUG-M2's fix the approved forget was
    refused too: the documented flow never forgot)."""
    invariants.allow("INV-4", "the first memory_forget is the service's refusal: no effect")
    await _forget_by_the_agent(adapter, tmp_path, uid, "turn")


@pytest.mark.plan("MEM-05")
@pytest.mark.xfail(strict=True, reason=BUG_M1)
async def test_forgetting_a_fact_read_from_the_users_words_asks_first(
    tmp_path: Path, uid: str
) -> None:
    await _forget_by_the_agent("react", tmp_path, uid, "fact")


@pytest.mark.plan("MEM-05")
async def test_the_user_may_forget_what_an_agent_relayed_for_them(tmp_path: Path, uid: str) -> None:
    user, agent_id = f"gus-{uid}", f"own-{uid}"
    await tell("react", tmp_path, agent_id, user, "chat-1", FACT)
    rows = await eventually(lambda: _facts(user), within=90)
    async with harness() as h:
        ctx = await scope_of(h, user)  # the user, no agent: the person themselves
        await ctx.forget(rows[0]["memory_id"])
    assert memory_sql("select deleted_at from memories where memory_id = :1", rows[0]["memory_id"])[
        0
    ]["deleted_at"]


@pytest.mark.plan("MEM-03")
@pytest.mark.xfail(strict=True, reason=BUG_M3)
async def test_the_users_inventory_lists_what_they_said_in_a_chat(tmp_path: Path, uid: str) -> None:
    user = f"hep-{uid}"
    await tell("react", tmp_path, f"inv-{uid}", user, "chat-1", "My locker code is 5528.")
    await eventually(lambda: _with(user, "5528"), within=90)
    async with harness() as h:
        assert "5528" in await recall_text(h, user, "locker code")  # recall finds it
        assert "5528" in await inventory_text(h, user)  # the audit must too


async def _facts(user: str) -> list[dict]:
    return [r for r in live_facts(user) if r["memory_type"] != "OBSERVATION"]


#: BUG-M4 (agent-memory-service): the tenant's admin key cannot forget a user's memory:
#: ``is_tenant_admin`` (modules/authz/service.py:170) checks only an authorization tuple of the
#: acting principal (``user:<id>`` admin of the tenant, written by ``grant_membership``), never
#: the key's own ``admin`` role, so with BUG-M2 nothing but the relaying agent itself can erase
#: what a user said through the harness (USAGE §4: "or a tenant admin").
BUG_M4 = (
    "BUG-M4: the tenant's admin key is refused forgetting a user's memory "
    "(memory-service authz/service.py:170)"
)


@pytest.mark.plan("MEM-05", "INV-5")
@pytest.mark.xfail(strict=True, reason=BUG_M4)
async def test_erasing_a_user_leaves_nothing_any_read_returns(tmp_path: Path, uid: str) -> None:
    """The tenant's administrator erases a user — every memory forgotten, the conversation
    deleted: no recall (of any kind), no history and no next context holds what they said,
    and the store holds no live row of it."""
    user, agent_id = f"gil-{uid}", f"erase-{uid}"
    secret = f"My badge number is B{uid}."
    await tell("react", tmp_path, agent_id, user, "chat-1", secret)
    await eventually(lambda: _with(user, uid), within=90)
    async with harness() as h:
        tenant = await h._tenant()
        conversation = await h.conversation(user=user, thread="chat-1")
    async with MemoryClient(MEMORY_URL, api_key=ADMIN_KEY, timeout=60) as admin:
        ctx = admin.bind(tenant_id=tenant, user_id=user)
        for row in memories_of(user):
            await ctx.forget(row["memory_id"])
        await admin.bind(tenant_id=tenant, user_id=user, thread_id=conversation).history.delete()
    async with harness() as h:
        for kinds in (["memory"], ["message"], ["episode"], ["summary"], ["memory", "chunk"]):
            assert f"B{uid}" not in await recall_text(h, user, "badge number", kinds), kinds
        thread = await scope_of(h, user, agent=agent_id, thread="chat-1")
        try:
            held = json.dumps([m.model_dump(mode="json") for m in await thread.history()])
        except NotFoundError:
            held = ""
        assert f"B{uid}" not in held
    assert not memories_of(user)
    rows = memory_sql(
        "select count(*) as n from messages where tenant_id = :1 and content like :2"
        " and deleted_at is null",
        tenant,
        f"%B{uid}%",
    )
    assert rows[0]["n"] == 0
    async with harness() as h:
        agent, built = await armed("react", h, tmp_path, [], id=agent_id)
        await agent.run("What is my badge number?", user=user, thread="chat-2")
    assert f"B{uid}" not in built.said()


async def _with(user: str, token: str) -> list[dict]:
    return [r for r in live_facts(user) if token in r["content"]]


@pytest.mark.plan("MEM-07")
@pytest.mark.parametrize("adapter", [a for a in ADAPTERS if a != "function"])
async def test_a_profile_the_agent_edits_is_in_every_next_chat(
    adapter: str, tmp_path: Path, uid: str
) -> None:
    user, agent_id = f"hal-{uid}", f"profile-{adapter}-{uid}"
    note = f"Prefers metric units, ticket {uid}."
    async with harness() as h:
        plan = [("profile_edit", {"block": "user.units", "new": note})]
        agent, _ = await armed(adapter, h, tmp_path, plan, id=agent_id)
        done = await approve_all(
            agent, await agent.run("Remember I use metric.", user=user, thread="c1"), reviewer=user
        )
        assert done.status is RunStatus.SUCCESS, done
    rows = memory_sql(
        "select block, text from profile_blocks where block = 'user.units' and text like :1",
        f"%{uid}%",
    )
    assert len(rows) == 1, rows
    async with harness() as h:
        agent, built = await armed(adapter, h, tmp_path, [], id=agent_id)
        await agent.run("Convert 3 miles for me.", user=user, thread="c2")
    assert note in built.said()
    async with harness() as h:  # and not another user's
        agent, built = await armed(adapter, h, tmp_path, [], id=agent_id)
        await agent.run("Convert 3 miles for me.", user=f"ivy-{uid}", thread="c1")
    assert uid not in built.said()


@pytest.mark.plan("MEM-07")
@pytest.mark.parametrize(("window", "expected"), [(None, 8000), (24_000, 6000), (6000, 2000)])
async def test_the_memory_budget_follows_the_models_window(
    traffic: Traffic, tmp_path: Path, uid: str, window: int | None, expected: int
) -> None:
    """The context the model is given is asked for 8,000 tokens; a model whose window is known
    to be under 32,000 gets a quarter of it, never under 2,000 — and the service holds the
    answer within it."""
    async with harness() as h:
        model = PlannedChat([])
        target = ReAct(system_prompt="You work.", model=model, context_window=window)
        agent = h.wrap(target, id=f"budget-{uid}")
        await agent.run("What do I like?", user=f"jo-{uid}", thread="t")
    asked = traffic.of("memory", "POST", "/v1/context")
    assert asked, [x.path for x in traffic.seen]
    budgets = {x.body()["token_budget"] for x in asked}
    assert budgets == {expected}, budgets
    for x in asked:
        assert x.response is not None and x.response.status_code == 200
        assert x.response.json()["token_estimate"] <= expected


@pytest.mark.plan("MEM-07")
async def test_the_memories_a_run_cites_are_capped(tmp_path: Path, uid: str) -> None:
    """60 memories that all bear on the question: the run's outcome cites at most 50."""
    user, agent_id = f"kai-{uid}", f"cap-{uid}"
    async with harness() as h:
        scope_ = h.memory.client.bind(tenant_id=await h._tenant(), user_id=user)  # type: ignore[union-attr]
        await asyncio.gather(
            *(
                scope_.remember(f"Supplier S{n:02d} delivers pallets of steel bolts on day {n}.")
                for n in range(60)
            )
        )
        agent, _ = await armed("react", h, tmp_path, [], id=agent_id)
        done = await agent.run(
            "Which suppliers deliver pallets of steel bolts, and on which days?",
            user=user,
            thread="t",
        )
        await h.writes.drain()

    async def outcome() -> list[dict]:
        return memory_sql(
            "select metadata from feedback where agent_run_id = :1 and source = 'system'",
            done.run_id,
        )

    (row,) = await eventually(outcome, within=60)
    cited = row["metadata"]["pushed_memory_ids"]
    assert 0 < len(cited) <= 50, len(cited)


@pytest.mark.plan("MEM-13", "INV-2")
async def test_two_chats_of_one_user_writing_at_once_lose_nothing(tmp_path: Path, uid: str) -> None:
    user, agent_id = f"lu-{uid}", f"busy-{uid}"
    texts = [f"My office number {n} is room {uid}{n}." for n in range(6)]

    async def chat(n: int) -> str:
        async with harness() as h:
            agent, _ = await armed("react", h, tmp_path, [], id=agent_id)
            done = await agent.run(texts[n], user=user, thread=f"chat-{n % 2}")
            await h.writes.drain()
            return done.run_id

    runs = await asyncio.gather(*(chat(n) for n in range(6)))

    async def stored() -> list[dict]:
        rows = memory_sql(
            "select m.content, m.agent_run_id from messages m where m.role = 'USER'"
            " and m.content like :1 and m.deleted_at is null",
            f"%room {uid}%",
        )
        return rows if len(rows) >= 6 else []

    rows = await eventually(stored, within=60)
    assert sorted(r["content"] for r in rows) == sorted(texts), rows  # each turn once
    assert {r["agent_run_id"] for r in rows} == set(runs)
    _no_duplicates(user)
