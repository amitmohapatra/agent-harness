"""More of one conversation (CONV-03, CONV-06, CONV-09, CONV-12, CONV-14):

* one chat's messages landing on two servers at once (a load balancer): still one turn at a
  time, in order — limitation L1, being fixed on branch chat-order: ``xfail(strict=True)``;
* a steer that arrives after the working turn's last step runs as its own turn, never lost;
* a new message to a chat whose turn waits for an approval: answered, and the approval still
  resumes its own turn;
* a queued turn (``start``) and a direct one (``run``) in one chat: one at a time;
* a stress run: many chats of many users, several messages each, on three servers at once,
  its interleaving seeded (``SCEN_SEED``; ``SCEN_STRESS_CHATS``, default 100).
"""

from __future__ import annotations

import asyncio
import os
import random
from pathlib import Path

import httpx
import pytest

from tests.scenarios.brains import STEERS, command, parse, token, wrapped
from tests.scenarios.conftest import harness
from tests.scenarios.kit import TOOLS, Served, answered, asked, asked_id, free_port, history
from trellis import Harness
from trellis.contracts import RunStatus
from trellis.harness.agui.sse import decode

pytestmark = pytest.mark.timeout(1200)


async def _post(c: httpx.AsyncClient, agent_id: str, thread: str, text: str) -> dict:
    sent = await c.post(
        f"/agui/{agent_id}/run",
        json={"threadId": thread, "messages": [{"id": text[:16], "role": "user", "content": text}]},
    )
    assert sent.status_code == 200, sent.text
    return decode(sent.text)[-1][1]


@pytest.mark.plan("CONV-03")
@pytest.mark.xfail(
    strict=True,
    reason="pending branch chat-order: limitation L1, one chat on two servers can run two turns at once",
)
async def test_one_chat_on_two_servers_is_one_turn_at_a_time(uid: str, tmp_path: Path) -> None:
    agent_id = f"s14s-{uid}"
    spec = {"mode": "agui", "hold": 6, "agents": [{"adapter": "react", "id": agent_id}]}
    one = await Served({**spec, "port": free_port()}, tmp_path, "one").started()
    two = await Served({**spec, "port": free_port()}, tmp_path, "two").started()
    user, thread = f"ada-{uid}", f"chat-{uid}"
    try:
        async with httpx.AsyncClient(headers={"x-user": user}, timeout=300) as c:
            first = asyncio.create_task(
                c.post(
                    f"http://127.0.0.1:{one.port}/agui/{agent_id}/run",
                    json={
                        "threadId": thread,
                        "messages": [
                            {
                                "id": "m1",
                                "role": "user",
                                "content": f"{token(uid, 1)} {command('wait', seconds=5)}",
                            }
                        ],
                    },
                )
            )
            await asyncio.sleep(2)
            second = await c.post(
                f"http://127.0.0.1:{two.port}/agui/{agent_id}/run",
                json={
                    "threadId": thread,
                    "messages": [
                        {"id": "m2", "role": "user", "content": f"{token(uid, 2)} second"}
                    ],
                },
            )
            await first
        said = parse(decode(second.text)[-1][1]["result"])
        assert token(uid, 1) in said.seen, ("the second turn ran beside the first", said)
        async with harness() as h:
            kept = await history(h, user=user, agent_id=agent_id, thread=thread)
            assert answered(kept) == [token(uid, 1), token(uid, 2)], kept
    finally:
        one.stop()
        two.stop()


@pytest.mark.plan("CONV-06")
@pytest.mark.parametrize("adapter", sorted(STEERS))
async def test_a_steer_after_the_last_step_is_its_own_turn(
    h: Harness, adapter: str, uid: str, tmp_path: Path
) -> None:
    """The working turn is in its last model call (stalled ``slow:``, no tool left to call): the
    steered message comes too late to be taken, and runs as its own turn."""
    agent, _ = await wrapped(
        h, adapter, TOOLS, tmp_path, id=f"s14t-{adapter}-{uid}", claude=[{"stall": 4}]
    )
    user, thread = f"ada-{uid}", f"chat-{uid}"
    first = asyncio.create_task(agent.run(f"{token(uid, 1)} slow:4", user=user, thread=thread))
    await asyncio.sleep(2)
    second = await agent.run(f"{token(uid, 2)} also this", user=user, thread=thread, mode="steer")
    done = await first
    assert done.status is RunStatus.SUCCESS and second.status is RunStatus.SUCCESS, (
        done.error,
        second.error,
    )
    kept = await history(h, user=user, agent_id=agent.id, thread=thread)
    assert asked(kept, [token(uid, 1), token(uid, 2)]) == [token(uid, 1), token(uid, 2)], kept
    assert token(uid, 2) in answered(kept) or token(uid, 2) in second.answer.lower(), (
        "the steer was lost",
        kept,
    )


@pytest.mark.plan("CONV-09")
@pytest.mark.parametrize(
    "adapter", ["react", "langgraph", "openai_agents", "claude_agent_sdk", "kimi", "function"]
)
async def test_a_new_message_while_the_chat_waits_for_an_approval(
    h: Harness, adapter: str, uid: str, tmp_path: Path
) -> None:
    claude = [{"tool": "transfer", "args": {"amount": 4, "to": "x"}}]
    agent, _ = await wrapped(h, adapter, TOOLS, tmp_path, id=f"s14p-{adapter}-{uid}", claude=claude)
    user, thread = f"ada-{uid}", f"chat-{uid}"
    paused = await agent.run(
        f"{token(uid, 1)} {command('transfer', amount=4, to='x')}", user=user, thread=thread
    )
    assert paused.status is RunStatus.PAUSED, paused.error
    if adapter == "claude_agent_sdk":  # its script calls transfer every turn: it pauses too
        meanwhile = await agent.run(f"{token(uid, 2)} what's up?", user=user, thread=thread)
        assert meanwhile.status is RunStatus.PAUSED
        await agent.resume(asked_id(meanwhile), "reject", answer="not this one", reviewer=user)
    else:
        meanwhile = await agent.run(f"{token(uid, 2)} what's up?", user=user, thread=thread)
        assert meanwhile.status is RunStatus.SUCCESS and parse(meanwhile.answer).mine == token(
            uid, 2
        ), meanwhile.error
    done = await agent.resume(asked_id(paused), "approve", reviewer=user)
    assert done.status is RunStatus.SUCCESS and parse(done.answer).mine == token(uid, 1), done.error


@pytest.mark.plan("CONV-12")
async def test_a_queued_turn_and_a_direct_one_in_one_chat(uid: str, tmp_path: Path) -> None:
    agent_id = f"s14q-{uid}"
    spec = {
        "mode": "worker",
        "lease": 6,
        "hold": 6,
        "agents": [{"adapter": "react", "id": agent_id}],
    }
    async with harness() as h:
        agent, _ = await wrapped(h, "react", TOOLS, tmp_path, id=agent_id)
        user, thread = f"ada-{uid}", f"chat-{uid}"
        worker = await Served(spec, tmp_path, "w").started()
        try:
            handle = await agent.start(
                f"{token(uid, 1)} {command('wait', seconds=4)}", user=user, thread=thread
            )
            await asyncio.sleep(1.5)
            direct = await agent.run(
                f"{token(uid, 2)} {command('wait', seconds=1)}", user=user, thread=thread
            )
            queued = await handle.result(wait=120)
        finally:
            worker.stop()
        assert queued.status is RunStatus.SUCCESS and direct.status is RunStatus.SUCCESS
        # one at a time: the turns' slow calls never overlapped (the worker's are in its process;
        # the direct turn saw the queued one's answer in its history if it came after)
        said = parse(direct.answer)
        assert token(uid, 1) in said.seen, ("the direct turn ran beside the queued one", said)


@pytest.mark.plan("CONV-14", "CONV-07", "CONV-08")
async def test_stress_many_chats_on_three_servers(uid: str, tmp_path: Path) -> None:
    seed = int(os.environ.get("SCEN_SEED", "1234"))
    chats = int(os.environ.get("SCEN_STRESS_CHATS", "100"))
    rng = random.Random(seed)
    agent_id = f"s14x-{uid}"
    spec = {"mode": "agui", "hold": 10, "agents": [{"adapter": "function", "id": agent_id}]}
    servers = [
        await Served({**spec, "port": free_port()}, tmp_path, f"s{n}").started() for n in range(3)
    ]
    plan = [(f"u{n % 20}-{uid}", f"c{n}-{uid}") for n in range(chats)]
    sent: dict[str, list[str]] = {}
    try:
        async with httpx.AsyncClient(timeout=600) as c:

            async def chat(user: str, thread: str) -> None:
                server = servers[
                    rng.randrange(3)
                ]  # a thread sticks to one server (sticky sessions)
                for n in range(1, 6):
                    tok = token(thread.split("-", maxsplit=1)[0], n)
                    sent.setdefault(thread, []).append(tok)
                    c.headers["x-user"] = user
                    done = await c.post(
                        f"http://127.0.0.1:{server.port}/agui/{agent_id}/run",
                        headers={"x-user": user},
                        json={
                            "threadId": thread,
                            "messages": [
                                {"id": f"{thread}-{n}", "role": "user", "content": f"{tok} hi"}
                            ],
                        },
                    )
                    finished = decode(done.text)[-1][1]
                    assert finished.get("outcome") == {"type": "success"}, (thread, n, finished)
                    said = parse(finished["result"])
                    assert said.mine == tok and set(sent[thread][:-1]) <= said.seen, (thread, said)
                    others = {t for k, v in sent.items() if k != thread for t in v}
                    assert not said.seen & others, ("another chat's history", thread)

            order = list(plan)
            rng.shuffle(order)
            await asyncio.gather(*(chat(u, t) for u, t in order))
    finally:
        for s in servers:
            s.stop()
    async with harness() as h:
        for user, thread in rng.sample(plan, 10):
            kept = await history(h, user=user, agent_id=agent_id, thread=thread)
            assert answered(kept) == sent[thread], (thread, kept)
