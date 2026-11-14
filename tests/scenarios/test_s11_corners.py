"""Scenario 11 — corner cases (CONV-11, CONV-13, REL): empty, huge and Unicode messages; tool
arguments at their limits; a cancel at every phase of a run (queued, waiting its turn, running,
paused, resuming); the same message submitted twice; a resume of a pause already answered."""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI, Request

from tests.scenarios.brains import ADAPTERS, command, parse, token, wrapped
from tests.scenarios.kit import TOOLS, Served, history, ledger, record
from trellis import Harness
from trellis.contracts import ConfigurationError, RunStatus

pytestmark = pytest.mark.timeout(900)


def _app(agent: object) -> FastAPI:
    app = FastAPI()

    def user_of(request: Request) -> str:
        return request.headers["x-user"]

    agent.serve_chat(app, identity=user_of)  # type: ignore[attr-defined]
    return app


@pytest.mark.plan("REL")
@pytest.mark.parametrize("blank", ["", "   ", "\n\t"])
async def test_a_blank_message_opens_no_run(
    h: Harness, blank: str, uid: str, tmp_path: Path
) -> None:
    agent, _ = await wrapped(h, "react", TOOLS, tmp_path, id=f"s11b-{uid}")
    with pytest.raises(ConfigurationError):
        await agent.run(blank, user=f"ada-{uid}", thread=f"t-{uid}")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=_app(agent)), base_url="http://agui"
    ) as c:
        sent = await c.post(
            "/agui/run",
            headers={"x-user": f"ada-{uid}"},
            json={
                "threadId": f"t-{uid}",
                "messages": [{"id": "m1", "role": "user", "content": blank}],
            },
        )
        assert sent.status_code == 422, sent.text
    assert [r async for r in h.runs.iterate(agent_id=agent.id)] == []


@pytest.mark.plan("REL")
@pytest.mark.parametrize("adapter", ADAPTERS)
async def test_huge_and_unicode_messages(
    h: Harness, adapter: str, uid: str, tmp_path: Path
) -> None:
    agent, _ = await wrapped(h, adapter, TOOLS, tmp_path, id=f"s11u-{adapter}-{uid}")
    user, thread = f"ada-{uid}", f"t-{uid}"
    unicode = f"{token(uid, 1)} héllo 世界 🚀👩‍👩‍👧‍👦 ﷽ ½ — tab\tnew\nline"
    done = await agent.run(unicode, user=user, thread=thread)
    assert done.status is RunStatus.SUCCESS and parse(done.answer).mine == token(uid, 1), done.error
    huge = f"{token(uid, 2)} " + ("lorem ipsum dolor " * 12_000)  # ~216,000 characters
    big = await agent.run(huge, user=user, thread=thread)
    assert big.status is RunStatus.SUCCESS, big.error
    assert parse(big.answer).mine == token(uid, 2), big.answer[:200]
    kept = await history(h, user=user, agent_id=agent.id, thread=thread)
    users = [c for r, c in kept if r.upper() == "USER"]
    assert unicode in users, users[:1]  # kept as it was said, emoji and all
    assert sum(token(uid, 2) in c for c in users) == 1


@pytest.mark.plan("APR-07", "REL")
@pytest.mark.parametrize("adapter", ADAPTERS)
async def test_tool_arguments_at_their_limits(
    h: Harness, adapter: str, uid: str, tmp_path: Path
) -> None:
    deep: dict = {"leaf": "🚀"}
    for n in range(30):
        deep = {f"l{n}": deep, "list": [n, None, True, 1.5]}
    long_key = "k" * 20_000
    claude = [
        {"tool": "lookup", "args": {"key": long_key}},
        {"tool": "note", "args": {"text": "x"}},
    ]
    agent, _ = await wrapped(h, adapter, TOOLS, tmp_path, id=f"s11a-{adapter}-{uid}", claude=claude)
    done = await agent.run(
        f"{token(uid, 1)} {command('lookup', key=long_key)}", user=f"ada-{uid}", thread=f"t-{uid}"
    )
    assert done.status is RunStatus.SUCCESS, done.error
    assert parse(done.answer).last.endswith(long_key[-50:]) or long_key[:50] in done.answer
    if adapter == "claude_agent_sdk":
        return
    before = len(ledger())
    pay = await agent.run(
        f"{token(uid, 2)} {command('transfer', amount=1, to='x', meta=deep)}",
        user=f"ada-{uid}",
        thread=f"t-{uid}",
    )
    assert (
        pay.status is RunStatus.PAUSED
        and pay.interrupt is not None
        and pay.interrupt.tool_call is not None
    ), pay.error
    assert pay.interrupt.tool_call.args["meta"] == deep  # nested 30 deep, kept structured
    ok = await agent.resume(pay.interrupt.interrupt_id, "approve", reviewer=f"ada-{uid}")
    assert ok.status is RunStatus.SUCCESS and len(ledger()) == before + 1, ok.error


# --------------------------------------------------------------------------- cancel at every phase
@pytest.mark.plan("CONV-13", "REL")
async def test_cancel_a_queued_run(h: Harness, uid: str, tmp_path: Path) -> None:
    agent, _ = await wrapped(h, "react", TOOLS, tmp_path, id=f"s11q-{uid}")
    handle = await agent.start(f"{token(uid, 1)} later", user=f"ada-{uid}", thread=f"t-{uid}")
    assert (await handle.status()).status is RunStatus.QUEUED  # no worker serves this agent
    cancelled = await agent.cancel(handle.id, reason="changed my mind")
    assert cancelled.status is RunStatus.CANCELLED


@pytest.mark.plan("CONV-13")
@pytest.mark.parametrize("adapter", ADAPTERS)
async def test_cancel_a_message_waiting_its_turn_and_a_running_one(
    h: Harness, adapter: str, uid: str, tmp_path: Path
) -> None:
    claude = [{"tool": "wait", "args": {"seconds": 6}}]
    agent, _ = await wrapped(h, adapter, TOOLS, tmp_path, id=f"s11c-{adapter}-{uid}", claude=claude)
    user, thread = f"ada-{uid}", f"t-{uid}"
    first = asyncio.create_task(
        agent.run(f"{token(uid, 1)} {command('wait', seconds=6)}", user=user, thread=thread)
    )
    await asyncio.sleep(2)
    second = asyncio.create_task(
        agent.run(f"{token(uid, 2)} waits its turn", user=user, thread=thread)
    )
    await asyncio.sleep(1)
    running = [r async for r in h.runs.iterate(agent_id=agent.id)]
    by_input = {}
    for r in running:
        rec = await record(h, r.run_id)
        by_input[str(rec.input)[: len(token(uid, 2))]] = rec
    waiting = by_input[token(uid, 2)]
    await agent.cancel(waiting.run_id, reason="never mind")  # waiting for the turn
    working = by_input[token(uid, 1)]
    await agent.cancel(working.run_id, reason="stop")  # in its slow call
    one, two = await asyncio.gather(first, second)
    assert one.status is RunStatus.CANCELLED, one
    assert two.status is RunStatus.CANCELLED, two
    after = await agent.run(f"{token(uid, 3)} still here?", user=user, thread=thread)
    assert after.status is RunStatus.SUCCESS, after.error  # the chat is not stuck


@pytest.mark.plan("CONV-13", "REL")
@pytest.mark.parametrize("adapter", ADAPTERS)
async def test_cancel_a_paused_run_and_a_resuming_one(
    adapter: str, uid: str, tmp_path: Path
) -> None:
    from tests.scenarios.conftest import harness

    claude = [{"tool": "transfer", "args": {"amount": 2, "to": "x"}}]
    async with harness() as h:
        agent, _ = await wrapped(
            h, adapter, TOOLS, tmp_path, id=f"s11p-{adapter}-{uid}", claude=claude
        )
        before = len(ledger())
        pay = command("transfer", amount=2, to="x")
        paused = await agent.run(f"{token(uid, 1)} {pay}", user=f"ada-{uid}", thread=f"t-{uid}")
        assert paused.status is RunStatus.PAUSED, paused.error
        assert (await agent.cancel(paused.run_id, reason="no")).status is RunStatus.CANCELLED
        assert len(ledger()) == before
        # resuming: a queued run answered, cancelled before any worker continues it
        handle = await agent.start(f"{token(uid, 2)} {pay}", user=f"ada-{uid}", thread=f"q-{uid}")
        spec = {
            "mode": "worker",
            "agents": [{"adapter": adapter, "id": agent.id, "claude": claude}],
            "lease": 6,
            "hold": 6,
        }
        worker = await Served(spec, tmp_path, "w").started()
        try:
            waiting = await handle.result(wait=180)
        finally:
            worker.stop()
        assert waiting.status is RunStatus.PAUSED, waiting
        resumed = await agent.resume(
            waiting.interrupt.interrupt_id, "approve", reviewer=f"ada-{uid}"
        )
        assert resumed.status is RunStatus.QUEUED
        assert (await agent.cancel(handle.id, reason="too late")).status is RunStatus.CANCELLED
        assert not any(line.startswith("transfer:2:x") for line in ledger()[before:])


@pytest.mark.plan("CONV-11")
async def test_the_same_message_submitted_twice_runs_once(
    h: Harness, uid: str, tmp_path: Path
) -> None:
    agent, _ = await wrapped(h, "react", TOOLS, tmp_path, id=f"s11d-{uid}")
    user, thread = f"ada-{uid}", f"t-{uid}"
    body = {
        "threadId": thread,
        "runId": f"run_{uid}dup",
        "messages": [
            {
                "id": f"m-{uid}",
                "role": "user",
                "content": f"{token(uid, 1)} {command('wait', seconds=2)}",
            }
        ],
    }
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=_app(agent)), base_url="http://agui", timeout=120
    ) as c:
        one, two = await asyncio.gather(
            c.post("/agui/run", headers={"x-user": user}, json=body),
            c.post("/agui/run", headers={"x-user": user}, json=body),
        )
    codes = sorted([one.status_code, two.status_code])
    assert codes == [200, 422], codes  # a client's retry of the same message: one run
    runs = [r async for r in h.runs.iterate(agent_id=agent.id)]
    assert len(runs) == 1, runs
    kept = await history(h, user=user, agent_id=agent.id, thread=thread)
    assert sum(token(uid, 1) in c for r, c in kept if r.upper() == "USER") == 1


@pytest.mark.plan("REL")
async def test_a_resume_of_a_pause_already_answered_is_refused(
    h: Harness, uid: str, tmp_path: Path
) -> None:
    agent, _ = await wrapped(h, "react", TOOLS, tmp_path, id=f"s11r-{uid}")
    before = len(ledger())
    paused = await agent.run(
        f"{token(uid, 1)} {command('transfer', amount=3, to='y')}",
        user=f"ada-{uid}",
        thread=f"t-{uid}",
    )
    asked = paused.interrupt
    assert asked is not None
    done = await agent.resume(asked.interrupt_id, "approve", reviewer=f"ada-{uid}")
    assert done.status is RunStatus.SUCCESS
    with pytest.raises(ConfigurationError):  # a stale answer: the run is not paused any more
        await agent.resume(asked.interrupt_id, "approve", reviewer=f"ada-{uid}")
    assert len(ledger()) == before + 1  # the call ran once
    assert (await record(h, done.run_id)).status is RunStatus.SUCCESS
