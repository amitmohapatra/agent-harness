"""Scenario 2 — one user, three chats open at once, messages interleaved: the chats run in
parallel, no chat's history reaches another, and what the user said in one chat is recalled in
another through user-level memory (``memory_search``), as designed, while each thread's history
stays its own.

Scenario 3 — two users, A and B, on the same thread id at once: their conversations, memory,
approvals (inbox), files, schedules and run records stay apart, and a key of another tenant
reads none of it."""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest

from tests.scenarios.brains import ADAPTERS, command, parse, token, wrapped
from tests.scenarios.conftest import OTHER_KEY, harness
from tests.scenarios.kit import TOOLS, answered, eventually, history, overlapped, user_scope
from trellis import Harness
from trellis.contracts import RunStatus

pytestmark = pytest.mark.timeout(900)


@pytest.mark.parametrize("adapter", ADAPTERS)
async def test_three_chats_at_once_stay_apart(
    h: Harness, adapter: str, uid: str, tmp_path: Path
) -> None:
    agent, _ = await wrapped(
        h,
        adapter,
        TOOLS,
        tmp_path,
        id=f"s2-{adapter}-{uid}",
        claude=[{"tool": "wait", "args": {"seconds": 1.5}}],
    )
    user = f"ada-{uid}"
    chats = [f"c{n}{uid[:4]}" for n in range(3)]
    sent: dict[str, list[str]] = {c: [] for c in chats}

    async def chat(c: str) -> None:
        for n in range(1, 4):
            tok = token(c, n)
            sent[c].append(tok)
            slow = (
                f" {command('wait', seconds=0.5)}" if adapter not in ("claude_agent_sdk",) else ""
            )
            done = await agent.run(f"{tok} in {c}{slow}", user=user, thread=f"t-{c}")
            assert done.status is RunStatus.SUCCESS, (c, n, done.error)
            said = parse(done.answer)
            assert said.mine == tok, (c, done.answer)
            assert set(sent[c][:-1]) <= said.seen, ("this chat's turns missing", c, done.answer)
            foreign = {t for other in chats if other != c for t in sent[other]}
            # another chat's turns are never this chat's history (they may be recalled as the
            # user's long-term memories: ``mem``, by design)
            assert not said.seen & foreign, (
                "another chat's history leaked in",
                c,
                said.seen & foreign,
            )

    await asyncio.gather(*(chat(c) for c in chats))
    for c in chats:
        kept = await history(h, user=user, agent_id=agent.id, thread=f"t-{c}")
        assert answered(kept) == sent[c], (c, kept)  # each thread: its own turns only, in order
    # the chats' slow calls ran at the same time
    assert overlapped({f"t-{c}" for c in chats}), "the chats did not run in parallel"


@pytest.mark.parametrize("adapter", ADAPTERS)
async def test_recall_across_chats_is_user_level(
    h: Harness, adapter: str, uid: str, tmp_path: Path
) -> None:
    """A fact told in chat 1 is in the user's memory, which a search from chat 2 finds (the
    memory tools are the user's); chat 2's own history does not hold chat 1's turns."""
    if adapter == "function":
        pytest.skip("a function target has no model to call memory_search (matrix F42, reason 31)")
    agent, _ = await wrapped(
        h,
        adapter,
        TOOLS,
        tmp_path,
        id=f"s2r-{adapter}-{uid}",
        claude=[{"tool": "memory_search", "args": {"query": f"locker {uid}"}}],
    )
    user = f"ada-{uid}"
    fact = f"locker code for {uid} is 4242"
    told = token(f"a{uid[:4]}", 1)
    done = await agent.run(f"{told} remember: my {fact}", user=user, thread=f"one-{uid}")
    assert done.status is RunStatus.SUCCESS, done.error
    await h.writes.drain()
    scope = await user_scope(h, user=user, agent_id=agent.id)

    async def indexed() -> bool:  # the transcript is the user's memory once indexed
        found = await scope.search(f"locker code {uid}", kinds=["message"])
        return any("4242" in item.text for item in found)

    assert await eventually(indexed, within=90), "the first chat's message never became searchable"
    asking = token(f"b{uid[:4]}", 1)
    search = command("memory_search", query=f"locker code {uid}", kinds=["message"])
    second = await agent.run(
        f"{asking} what is my locker code? {search}", user=user, thread=f"two-{uid}"
    )
    assert second.status is RunStatus.SUCCESS, second.error
    assert "4242" in second.answer, ("user-level recall failed", second.answer)
    kept = await history(h, user=user, agent_id=agent.id, thread=f"two-{uid}")
    assert told not in " ".join(c for r, c in kept if r.upper() == "USER"), kept


# --------------------------------------------------------------------------- scenario 3
@pytest.mark.parametrize("adapter", ADAPTERS)
async def test_two_users_on_one_thread_id(
    h: Harness, adapter: str, uid: str, tmp_path: Path
) -> None:
    agent, _ = await wrapped(h, adapter, TOOLS, tmp_path, id=f"s3-{adapter}-{uid}")
    thread = f"shared-{uid}"
    users = {"a": f"alice-{uid}", "b": f"bob-{uid}"}
    sent: dict[str, list[str]] = {"a": [], "b": []}

    async def talk(who: str) -> None:
        for n in range(1, 4):
            tok = token(f"{who}{uid[:4]}", n)
            sent[who].append(tok)
            done = await agent.run(f"{tok} from {who}", user=users[who], thread=thread)
            assert done.status is RunStatus.SUCCESS, (who, done.error)
            said = parse(done.answer)
            other = sent["b" if who == "a" else "a"]
            assert said.mine == tok and not (said.seen | said.mem) & set(other), (who, done.answer)
            assert set(sent[who][:-1]) <= said.seen, (who, done.answer)

    await asyncio.gather(talk("a"), talk("b"))
    for who in "ab":
        kept = await history(h, user=users[who], agent_id=agent.id, thread=thread)
        assert answered(kept) == sent[who], (who, kept)
    # the run records: each user's runs are theirs, under conversation keys of their own
    keys = {who: await h.conversation(user=users[who], thread=thread) for who in "ab"}
    assert keys["a"] != keys["b"]


async def test_two_users_approvals_files_schedules_and_runs_stay_apart(
    h: Harness, uid: str, tmp_path: Path
) -> None:
    """Alice and Bob on one thread id: each pauses for an approval; each sees only their own in
    the inbox by assignee, can resume only their own over AG-UI, and their schedules made in
    chat are listed only to themselves; another tenant's key sees none of their runs."""
    from fastapi import FastAPI, Request

    from trellis.harness.agui.sse import decode

    agent, _ = await wrapped(h, "react", TOOLS, tmp_path, id=f"s3b-{uid}")
    thread = f"shared-{uid}"
    alice, bob = f"alice-{uid}", f"bob-{uid}"
    paused = {}
    for who in (alice, bob):
        r = await agent.run(
            f"{token('x' + uid[:4], len(paused) + 1)} pay {command('transfer', amount=5, to=who)}",
            user=who,
            thread=thread,
        )
        assert r.status is RunStatus.PAUSED, r.error
        paused[who] = r
    for who in (alice, bob):
        mine = [r.run_id for r in await h.inbox(f"user:{who}")]
        assert paused[who].run_id in mine
        other = bob if who == alice else alice
        assert paused[other].run_id not in mine, "another user's approval in this user's inbox"
    # over AG-UI, Bob cannot answer Alice's pause (404), and Alice can
    app = FastAPI()

    def user_of(request: Request) -> str:
        return request.headers["x-user"]

    agent.serve_chat(app, identity=user_of)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://agui"
    ) as client:
        entry = paused[alice].interrupt
        assert entry is not None
        stolen = await client.post(
            "/agui/run",
            headers={"x-user": bob},
            json={
                "threadId": thread,
                "resume": [{"interruptId": entry.interrupt_id, "payload": True}],
            },
        )
        assert stolen.status_code == 404, stolen.text
        reconnect = await client.get(
            f"/agui/runs/{paused[alice].run_id}/events", headers={"x-user": bob}
        )
        assert reconnect.status_code == 404
        cancel = await client.post(
            f"/agui/runs/{paused[alice].run_id}/cancel", headers={"x-user": bob}
        )
        assert cancel.status_code == 404
        mine = await client.post(
            "/agui/run",
            headers={"x-user": alice},
            json={
                "threadId": thread,
                "resume": [{"interruptId": entry.interrupt_id, "payload": True}],
            },
        )
        assert mine.status_code == 200 and decode(mine.text)[-1][1]["outcome"] == {
            "type": "success"
        }
    await agent.resume(paused[bob].interrupt.interrupt_id, "reject", answer="no", reviewer=bob)
    # schedules set in chat: each user's own
    for who in (alice, bob):
        r = await agent.run(
            f"{token('s' + uid[:4], 1)} {command('schedule_task', task=f'report for {who}', when='0 9 * * 1')}",
            user=who,
            thread=thread,
        )
        assert r.status is RunStatus.SUCCESS, r.error
    for who in (alice, bob):
        r = await agent.run(
            f"{token('l' + uid[:4], 1)} {command('list_schedules')}", user=who, thread=thread
        )
        other = bob if who == alice else alice
        assert f"report for {who}" in r.answer and f"report for {other}" not in r.answer, r.answer
    # the tenant boundary: another tenant's key reads none of these runs
    if OTHER_KEY is None:
        pytest.skip("no second tenant's key (TRELLIS_API_KEY_OTHER)")
    async with harness(key=OTHER_KEY) as foreign:
        assert await foreign.runs.get(paused[alice].run_id) is None
        assert all(
            r.run_id not in (paused[alice].run_id, paused[bob].run_id)
            for r in await foreign.inbox()
        )
        theirs = await foreign.conversation(user=alice, thread=thread)
        assert theirs != await h.conversation(user=alice, thread=thread)
