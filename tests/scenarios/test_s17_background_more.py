"""More background work with the Claude Agent SDK (the scripted CLI; BG-03, BG-04, BG-06, BG-12,
APR-06), against the real services: an approval a background subagent's call needs reaches the
person in the parent chat, one question, and the call runs once; two subagents at once, one
cancelled alone while the other and the chat go on; a result that arrives while the chat is busy
with a turn is not mixed into that turn's answer and is there for the next."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from tests.scenarios.kit import ClaudeChat, asked_id, background, eventually, ledger, record
from trellis import Harness
from trellis.contracts import RunStatus

pytestmark = pytest.mark.timeout(600)


@pytest.mark.plan("BG-03", "APR-06")
async def test_an_approval_inside_a_background_task_reaches_the_parent_chat(
    h: Harness, uid: str, tmp_path: Path
) -> None:
    chat = ClaudeChat(
        h, tmp_path, f"s17a-{uid}", [[background(), {"text": "started it"}], [{"text": "ok"}]]
    )
    user, thread = f"ada-{uid}", f"t-{uid}"
    before = len(ledger())
    first = await chat.agent.run("research it in the background", user=user, thread=thread)
    [child] = (await chat.children(first.run_id)).values()
    await chat.call({"tool": "transfer", "args": {"amount": 6, "to": f"bg-{uid}"}})

    async def asked() -> bool:
        return (await record(h, child.run_id)).status is RunStatus.PAUSED

    assert await eventually(asked, within=60), await record(h, child.run_id)
    assert len(ledger()) == before  # nothing ran before the person decided
    turn = await chat.agent.run("how is it going?", user=user, thread=thread)
    assert turn.status is RunStatus.PAUSED and turn.interrupt is not None, turn  # asked in the chat
    assert turn.interrupt.tool_call is not None and turn.interrupt.tool_call.tool == "transfer"
    done = await chat.agent.resume(asked_id(turn), "approve", reviewer=user)
    assert done.status is RunStatus.SUCCESS and done.answer == "ok", done
    assert (await chat.called())["ok"] is True
    assert [x.split(":{")[0] for x in ledger()[before:]] == [f"transfer:6:bg-{uid}"]  # once
    ended = await chat.finish(child.run_id, "done")
    assert ended.status is RunStatus.SUCCESS, ended


@pytest.mark.plan("BG-04", "BG-06")
async def test_two_background_tasks_one_cancelled_alone(
    h: Harness, uid: str, tmp_path: Path
) -> None:
    turns = [
        [background("t1", "research"), background("t2", "write it up"), {"text": "started both"}],
        [{"text": "ok"}],
    ]
    chat = ClaudeChat(h, tmp_path, f"s17b-{uid}", turns)
    user, thread = f"ada-{uid}", f"t-{uid}"
    first = await chat.agent.run("research and write", user=user, thread=thread)
    children = await chat.children(first.run_id)
    assert set(children) == {"research", "write it up"}, children  # a run each
    assert all(c.status is RunStatus.RUNNING for c in children.values())
    research, writing = children["research"].run_id, children["write it up"].run_id
    cancelled = await chat.agent.cancel(research, reason="not needed")
    assert cancelled.status is RunStatus.CANCELLED

    async def stopped() -> bool:
        return (chat.tasks / "t1.stopped").exists()

    assert await eventually(stopped, within=30), "the cancelled task was never stopped"
    assert (await record(h, writing)).status is RunStatus.RUNNING  # the other goes on
    turn = await chat.agent.run("and?", user=user, thread=thread)
    assert turn.status is RunStatus.SUCCESS and turn.answer == "ok", turn  # and so does the chat
    ended = await chat.finish(writing, "written up", task="t2")
    assert ended.status is RunStatus.SUCCESS and ended.output == "written up", ended


@pytest.mark.plan("BG-12")
async def test_a_result_that_arrives_while_the_chat_is_busy(
    h: Harness, uid: str, tmp_path: Path
) -> None:
    turns = [
        [background(), {"text": "started it"}],
        [{"stall": 4}, {"text": "busy answer"}],
        [{"text": "ANS {heard}"}],
    ]
    chat = ClaudeChat(h, tmp_path, f"s17c-{uid}", turns)
    user, thread = f"ada-{uid}", f"t-{uid}"
    first = await chat.agent.run("research it in the background", user=user, thread=thread)
    [child] = (await chat.children(first.run_id)).values()
    busy = asyncio.create_task(chat.agent.run("a long question", user=user, thread=thread))
    ended = await chat.finish(child.run_id, f"found {uid}")  # while the turn works
    assert ended.status is RunStatus.SUCCESS, ended
    answer = await busy
    assert answer.status is RunStatus.SUCCESS and answer.answer == "busy answer", answer
    later = await chat.agent.run("what did it find?", user=user, thread=thread)
    assert later.status is RunStatus.SUCCESS and f"found {uid}" in later.answer, later.answer
