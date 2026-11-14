"""Scenario 4 — approvals under concurrency: a turn pauses for an approval while another chat of
the same user works; approve, reject or edit; resume on another worker process; "always allow"
learned in one chat applies in another; one question per call; the decision is fed back to the
memory service once, its nested arguments kept structured."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from tests.scenarios.brains import ADAPTERS, command, parse, token, wrapped
from tests.scenarios.kit import TOOLS, Served, eventually, ledger, record, thread_scope
from trellis import Harness
from trellis.contracts import InterruptReason, RunStatus

pytestmark = pytest.mark.timeout(900)
META = {"reason": "rent", "split": {"ada": [1, 2], "bob": {"share": 0.5}}}


def _transfer(amount: int, to: str) -> tuple[str, list[dict[str, Any]]]:
    args = {"amount": amount, "to": to, "meta": META}
    return command("transfer", **args), [{"tool": "transfer", "args": args}]


async def _fed_back(h: Harness, user: str, agent_id: str, thread: str, call_key: str) -> list[Any]:
    scope = await thread_scope(h, user=user, agent_id=agent_id, thread=thread)
    await h.writes.drain()
    return (await scope.feedback.page_for("tool_call", call_key)).items


@pytest.mark.parametrize("decision", ["approve", "reject", "edit"])
@pytest.mark.parametrize("adapter", ADAPTERS)
async def test_a_pause_while_another_chat_works(
    h: Harness,
    adapter: str,
    decision: str,
    uid: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SCEN_LEDGER", raising=False)
    said, claude = _transfer(40, f"bob-{uid}")
    agent, _ = await wrapped(h, adapter, TOOLS, tmp_path, id=f"s4-{adapter}-{uid}", claude=claude)
    user, paying, chatting = f"ada-{uid}", f"pay-{uid}", f"talk-{uid}"
    before = len(ledger())
    pause = asyncio.create_task(
        agent.run(f"{token('p' + uid[:4], 1)} pay {said}", user=user, thread=paying)
    )
    talk = (
        [
            await agent.run(f"{token('t' + uid[:4], n)} hello", user=user, thread=chatting)
            for n in range(1, 3)
        ]
        if adapter == "claude_agent_sdk"
        else None
    )
    paused = await pause
    if talk is None:  # the other chat works while this one waits for the person
        talk = [
            await agent.run(f"{token('t' + uid[:4], n)} hello", user=user, thread=chatting)
            for n in range(1, 3)
        ]
    assert all(t.status is RunStatus.SUCCESS for t in talk), [
        (t.status, t.error, t.answer, t.interrupt) for t in talk
    ]
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None, paused.error
    asked = paused.interrupt
    assert asked.reason is InterruptReason.APPROVAL and asked.tool_call is not None
    assert asked.tool_call.tool == "transfer" and asked.tool_call.args["meta"] == META
    assert len(ledger()) == before  # nothing ran before the person decided
    assert paused.run_id in [r.run_id for r in await h.inbox()]
    answer: Any = {
        "reject": "not this month",
        "edit": {"amount": 25, "to": f"bob-{uid}", "meta": META},
    }.get(decision)
    done = await agent.resume(asked.interrupt_id, decision, answer=answer, reviewer=user)
    assert done.status is RunStatus.SUCCESS, done.error
    ran = ledger()[before:]
    if decision == "approve":
        assert ran == [f"transfer:40:bob-{uid}:{json.dumps(META, sort_keys=True)}"], ran
    elif decision == "edit":
        assert ran == [f"transfer:25:bob-{uid}:{json.dumps(META, sort_keys=True)}"], ran
    else:
        assert ran == [] and "not this month" in done.answer, (ran, done.answer)
    assert (await record(h, done.run_id)).status is RunStatus.SUCCESS
    # fed back once, the nested arguments structured as they were asked
    key = asked.tool_call.idempotency_key or asked.interrupt_id

    async def once() -> bool:
        return len(await _fed_back(h, user, agent.id, paying, key)) >= 1

    assert await eventually(once, within=30)
    fed = await _fed_back(h, user, agent.id, paying, key)
    assert len(fed) == 1, fed
    verdict = {"approve": "approve", "reject": "reject", "edit": "edit"}[decision]
    assert str(fed[0].verdict).lower().endswith(verdict), fed[0]
    assert fed[0].metadata["args"]["meta"] == META, fed[0].metadata  # structured, not a string
    if decision == "edit":
        assert fed[0].correction == answer


@pytest.mark.parametrize("adapter", ADAPTERS)
async def test_always_allow_in_one_chat_applies_in_another(
    h: Harness, adapter: str, uid: str, tmp_path: Path
) -> None:
    claude = [{"tool": "note", "args": {"text": f"n-{uid}"}}]
    agent, _ = await wrapped(h, adapter, TOOLS, tmp_path, id=f"s4a-{adapter}-{uid}", claude=claude)
    user = f"ada-{uid}"
    write = command("note", text=f"n-{uid}")
    first = await agent.run(f"{token('a' + uid[:4], 1)} {write}", user=user, thread=f"one-{uid}")
    assert first.status is RunStatus.PAUSED and first.interrupt is not None, first.error
    done = await agent.resume(
        first.interrupt.interrupt_id, "approve", reviewer=user, remember="always"
    )
    assert done.status is RunStatus.SUCCESS, done.error
    await h.writes.drain()

    async def unasked() -> bool:  # the user's standing approval, in another conversation
        r = await agent.run(
            f"{token('b' + uid[:4], 1)} {write}",
            user=user,
            thread=f"two-{uid}-{asyncio.get_running_loop().time()}",
        )
        if r.status is RunStatus.PAUSED and r.interrupt is not None:
            await agent.resume(r.interrupt.interrupt_id, "reject", answer="probe", reviewer=user)
            return False
        assert r.status is RunStatus.SUCCESS, r.error
        return True

    assert await eventually(unasked, within=30, every=2), (
        "always-allow did not reach the other chat"
    )
    other = await agent.run(
        f"{token('c' + uid[:4], 1)} {write}", user=f"eve-{uid}", thread=f"one-{uid}"
    )
    assert other.status is RunStatus.PAUSED, "another user's write ran on this user's approval"
    await agent.resume(other.interrupt.interrupt_id, "reject", answer="no", reviewer=f"eve-{uid}")


@pytest.mark.parametrize("adapter", ADAPTERS)
async def test_one_question_per_call_under_concurrency(
    h: Harness, adapter: str, uid: str, tmp_path: Path
) -> None:
    """Five chats of one user pause on one irreversible call each at once: five questions, one
    each; approved, each call runs once and its run asks nothing more."""
    said, claude = _transfer(1, f"x-{uid}")
    agent, _ = await wrapped(h, adapter, TOOLS, tmp_path, id=f"s4q-{adapter}-{uid}", claude=claude)
    user = f"ada-{uid}"
    before = len(ledger())
    paused = await asyncio.gather(
        *(
            agent.run(f"{token('q' + uid[:4], n)} {said}", user=user, thread=f"q{n}-{uid}")
            for n in range(5)
        )
    )
    assert all(p.status is RunStatus.PAUSED for p in paused), [p.error for p in paused]
    inbox = [r.run_id for r in await h.inbox()]
    for p in paused:
        assert inbox.count(p.run_id) == 1
    done = await asyncio.gather(
        *(agent.resume(p.interrupt.interrupt_id, "approve", reviewer=user) for p in paused)
    )
    assert all(d.status is RunStatus.SUCCESS for d in done), [(d.status, d.error) for d in done]
    assert len(ledger()) - before == 5


@pytest.mark.parametrize("decision", ["approve", "reject", "edit"])
@pytest.mark.parametrize("adapter", ADAPTERS)
async def test_resumed_on_another_worker(
    adapter: str, decision: str, uid: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A queued run paused on worker 1; worker 1 is gone; the decision is made in this
    process; worker 2 continues it and the call runs once (or never)."""
    from tests.scenarios.conftest import harness

    book = tmp_path / "ledger"
    monkeypatch.setenv("SCEN_LEDGER", str(book))
    said, claude = _transfer(7, f"w-{uid}")
    agent_id = f"s4w-{adapter}-{uid}"
    spec = {
        "mode": "worker",
        "agents": [{"adapter": adapter, "id": agent_id, "claude": claude}],
        "lease": 6,
        "hold": 6,
    }
    async with harness() as h:
        agent, _ = await wrapped(h, adapter, TOOLS, tmp_path, id=agent_id, claude=claude)
        one = await Served(spec, tmp_path, "w1").started()
        try:
            handle = await agent.start(
                f"{token('w' + uid[:4], 1)} {said}", user=f"ada-{uid}", thread=f"w-{uid}"
            )
            paused = await handle.result(wait=180)
            assert paused.status is RunStatus.PAUSED, (
                paused.status,
                paused.error,
                one.log.read_text()[-2000:],
            )
            assert not book.exists()
        finally:
            one.stop()
        assert one.violations() == []
        answer: Any = {"reject": "no", "edit": {"amount": 3, "to": f"w-{uid}", "meta": META}}.get(
            decision
        )
        resumed = await agent.resume(
            paused.interrupt.interrupt_id, decision, answer=answer, reviewer=f"ada-{uid}"
        )
        assert resumed.status is RunStatus.QUEUED, resumed  # back on the queue, for any worker
        two = await Served(spec, tmp_path, "w2").started()
        try:
            done = await handle.result(wait=180)
        finally:
            two.stop()
        assert done.status is RunStatus.SUCCESS, (done.error, two.log.read_text()[-2000:])
        lines = book.read_text().splitlines() if book.exists() else []
        expect = {
            "approve": [f"transfer:7:w-{uid}"],
            "edit": [f"transfer:3:w-{uid}"],
            "reject": [],
        }[decision]
        assert [line.split(":{")[0] for line in lines] == expect, lines
        assert two.violations() == []
        assert parse(done.answer).mine == token("w" + uid[:4], 1), done.answer
