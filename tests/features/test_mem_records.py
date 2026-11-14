"""MEM-01, MEM-08, APR-07 on every adapter, against the real memory service: what a run
records is recorded once — the user's turn, each tool call, the answer — even when the run
paused and an approval resumed it (the resumed attempt sends the first attempt's messages
again); the person's decision on the paused call is one ``TOOL_CALL`` feedback on that run, its
nested arguments kept structured; the run's outcome is the system's ``confirm`` on that run.
Every assertion reads the memory database's own rows."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.features.conftest import ADAPTERS, eventually, harness, memory_sql
from tests.features.kit import (
    armed,
    feedback_rows,
    ledger,
    messages_of_run,
    record,
    tool_records,
)
from trellis.contracts import RunStatus

pytestmark = pytest.mark.timeout(300)

META = {"po": {"number": 17, "lines": [{"sku": "A-1", "qty": 2}]}, "urgent": True}


@pytest.mark.plan("MEM-01", "MEM-08", "APR-07", "INV-2", "INV-4")
@pytest.mark.parametrize("adapter", ADAPTERS)
async def test_a_paused_and_approved_run_is_recorded_once(
    adapter: str, tmp_path: Path, uid: str
) -> None:
    user, thread = f"rec-{uid}", f"chat-{uid}"
    question = f"Check A-1 and pay acme for order {uid}."
    plan = [("lookup", {"sku": "A-1"}), ("wire", {"amount": 5, "to": "acme", "meta": META})]
    async with harness() as h:
        agent, _ = await armed(adapter, h, tmp_path, plan, id=f"rec-{adapter}-{uid}")
        paused = await agent.run(question, user=user, thread=thread)
        assert paused.interrupt is not None, paused
        assert paused.interrupt.tool_call is not None
        assert paused.interrupt.tool_call.args == {"amount": 5, "to": "acme", "meta": META}
        done = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="lee")
        assert done.status is RunStatus.SUCCESS, done
        assert done.answer == "Done. wired 5 to acme", done
        await h.writes.drain()
    run_id = done.run_id
    # the call approved ran once, the read once (the resumed attempt replayed it)
    assert [e for e in ledger() if e.startswith("wire:5:acme")].count(
        'wire:5:acme:{"po": {"lines": [{"qty": 2, "sku": "A-1"}], "number": 17}, "urgent": true}'
    ) == 1

    async def transcript() -> list[dict]:
        rows = messages_of_run(run_id)
        return (
            rows
            if any(r["role"] == "ASSISTANT" and "wired 5" in r["content"] for r in rows)
            else []
        )

    rows = await eventually(transcript, within=60)
    assert rows, messages_of_run(run_id)
    users = [r for r in rows if r["role"] == "USER"]
    assert [r["content"] for r in users] == [question], rows  # the turn, once
    answers = [r for r in rows if r["role"] == "ASSISTANT" and r["content"] == done.answer]
    assert len(answers) == 1, rows  # the answer, once
    ids = [r["source_message_id"] for r in rows]
    assert len(ids) == len(set(ids)), rows  # no message stored twice under one harness id
    assert len({r["thread_id"] for r in rows}) == 1  # one conversation

    async def tools() -> list[dict]:
        got = tool_records(run_id)
        return got if len(got) >= 2 else []

    records = await eventually(tools, within=60)
    names = sorted(r["tool_name"] for r in records)
    assert names == ["lookup", "wire"], records  # each call that ran: one record
    (wired,) = [r for r in records if r["tool_name"] == "wire"]
    assert wired["args_redacted"]["meta"] == META, wired  # nested arguments kept structured
    assert wired["user_id"] == user

    async def decided() -> list[dict]:
        rows = feedback_rows(run_id)
        kinds = {r["target_kind"] for r in rows}
        return rows if {"tool_call", "run"} <= kinds else []

    feedback = await eventually(decided, within=60)
    calls = [r for r in feedback if r["target_kind"] == "tool_call"]
    assert len(calls) == 1, feedback  # one decision, once, on this run
    assert calls[0]["verdict"] == "approve", calls
    assert calls[0]["source"] == "interrupt" and calls[0]["metadata"]["tool"] == "wire", calls
    assert calls[0]["metadata"]["args"]["meta"] == META, calls  # nested arguments, structured
    outcome = [r for r in feedback if r["target_kind"] == "run" and r["source"] == "system"]
    assert [r["verdict"] for r in outcome] == ["confirm"], feedback  # the system's, once
    # and agent-runs keeps the same run, ended well, with the resolution it was resumed by
    stored = await record(run_id)
    assert stored.status is RunStatus.SUCCESS
    assert stored.last_resolution is not None and stored.last_resolution.decision.value == "APPROVE"


@pytest.mark.plan("MEM-01")
async def test_a_large_tool_output_is_recorded_bounded(tmp_path: Path, uid: str) -> None:
    """A 2 MB result: the run answers from it, and memory keeps a bounded record of it —
    the tool record and the transcript's tool message each a small fraction of it."""
    plan = [("big_report", {"rows": 50_000})]
    async with harness() as h:
        agent, _ = await armed("react", h, tmp_path, plan, id=f"big-{uid}")
        done = await agent.run("Report everything.", user=f"big-{uid}", thread="t")
        assert done.status is RunStatus.SUCCESS, done
        await h.writes.drain()
    raw = 50_000 * len("row 000000 sku-000 qty 0")

    async def kept() -> list[dict]:
        return tool_records(done.run_id)

    (rec,) = await eventually(kept, within=60)
    assert rec["size"] is not None and rec["size"] < raw // 20, rec["size"]
    rows = await eventually(lambda: _messages(done.run_id), within=60)
    tool_messages = [r for r in rows if r["role"] == "TOOL"]
    assert tool_messages, rows
    assert all(r["content_bytes"] < raw // 20 for r in tool_messages), [
        r["content_bytes"] for r in tool_messages
    ]


async def _messages(run_id: str) -> list[dict]:
    rows = messages_of_run(run_id)
    return rows if any(r["role"] == "ASSISTANT" for r in rows) else []


@pytest.mark.plan("MEM-01", "INV-2")
async def test_a_turn_retried_by_a_new_process_is_stored_once(tmp_path: Path, uid: str) -> None:
    """The same transcript written twice — a process that died after writing it and its
    spool replayed, or the service's answer lost — is one set of messages: every message
    carries its harness id, and the service keeps one per id."""
    user, thread = f"dup-{uid}", "t"
    async with harness() as h:
        agent, _ = await armed("react", h, tmp_path, [], id=f"dup-{uid}")
        done = await agent.run("Hello there, I keep pallets in Gdansk.", user=user, thread=thread)
        await h.writes.drain()
        before = await eventually(lambda: _messages(done.run_id), within=60)
        scope_ = h.memory.client.bind(  # type: ignore[union-attr]
            tenant_id=await h._tenant(),
            user_id=user,
            agent_id=f"dup-{uid}",
            thread_id=await h.conversation(user=user, thread=thread),
        )
        # the very batch again, as a replay sends it: the same harness ids
        again = [
            {
                "role": r["role"],
                "content": r["content"],
                "source_system": "trellis-harness",
                "source_message_id": r["source_message_id"],
            }
            for r in before
        ]
        await scope_.history.add(again)
    after = messages_of_run(done.run_id)
    assert len(after) == len(before), (before, after)
    count = memory_sql(
        "select count(*) as n from messages where thread_id = :1 and deleted_at is null",
        before[0]["thread_id"],
    )[0]["n"]
    assert count == len(before)
