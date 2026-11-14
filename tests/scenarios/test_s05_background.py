"""Scenario 5 — background work while the user keeps chatting (BG-01, BG-02, BG-10, BG-12).

* Claude: a subagent started in the background (the scripted CLI's ``background`` step); the
  user chats on in the same conversation; the result arrives and a later turn has it; the
  subagent's child run in agent-runs is ``RUNNING`` while it works and ``SUCCESS`` with its
  result after.
* Deep Agents' async sub-agents (``start_async_task`` …) against a stand-in Agent Protocol
  server (``tests/integration/test_react_async_subagent.py``'s), on ``ReAct`` and on
  ``create_deep_agent``: the turn replies at once, the next turn is answered while the task
  runs, a later turn checks it and reads its result.
* What is still on an unmerged branch is ``xfail(strict=True)``: Kimi's background work across
  turns (bg-kimi), ``ReAct``'s own sub-agents in the background with no server
  (react-background), and Claude Workflows (bg-claude-da). Each flips to a failure — and must be
  unmarked — the day its branch lands.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any, cast

import pytest
from claude_agent_sdk import ClaudeAgentOptions
from deepagents import create_deep_agent

from tests.integration.test_react_async_subagent import RESULT, StandIn
from tests.scenarios.brains import (
    CLI,
    SYSTEM,
    BrainChat,
    BrainChatModel,
    command,
    parse,
    token,
    wrapped,
)
from tests.scenarios.kit import TOOLS, eventually, history, record
from tests.support.arm import arm
from trellis import Harness, ReAct
from trellis.contracts import RunStatus

pytestmark = [pytest.mark.timeout(600)]
UUID = re.compile(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}")


def plan(*ids: str) -> pytest.MarkDecorator:
    return pytest.mark.plan(*ids)


async def _children(h: Harness, run_id: str) -> list[Any]:
    return [r async for r in h.runs.iterate(parent_run_id=run_id)]


@plan("BG-01", "BG-02", "BG-10")
async def test_claude_background_subagent_while_the_user_chats(
    h: Harness, uid: str, tmp_path: Path
) -> None:
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    turns = [
        [
            {"background": {"id": "t1", "agent": "researcher", "prompt": f"research {uid}"}},
            {"text": "started it"},
        ],
        [{"text": "ANS {heard}"}],
        [{"text": "ANS {heard}"}],
    ]
    options = ClaudeAgentOptions(
        cli_path=CLI,
        system_prompt=SYSTEM,
        env={
            "FAKE_CLAUDE_TURNS": json.dumps(turns),
            "FAKE_CLAUDE_RECORD": str(tmp_path / "cli.json"),
            "FAKE_CLAUDE_SESSIONS": str(tmp_path),
            "FAKE_CLAUDE_TASKS": str(tasks),
        },
    )
    agent = arm(h.wrap(options, id=f"s5c-{uid}"), TOOLS)
    user, thread = f"ada-{uid}", f"chat-{uid}"
    first = await agent.run(
        f"{token(uid, 1)} research it in the background", user=user, thread=thread
    )
    assert first.status is RunStatus.SUCCESS and first.answer == "started it", first.error
    [child] = await _children(h, first.run_id)
    assert (
        await record(h, child.run_id)
    ).status is RunStatus.RUNNING  # working, not done at launch
    second = await agent.run(f"{token(uid, 2)} meanwhile, hello", user=user, thread=thread)
    assert second.status is RunStatus.SUCCESS and token(uid, 2) in second.answer, second.error
    assert (await record(h, child.run_id)).status is RunStatus.RUNNING  # answered while it works
    (tasks / "t1.done").write_text(f"found {uid}")

    async def ended() -> bool:
        return (await record(h, child.run_id)).status is RunStatus.SUCCESS

    assert await eventually(ended, within=60), await record(h, child.run_id)
    assert f"found {uid}" in str((await record(h, child.run_id)).output)
    third = await agent.run(f"{token(uid, 3)} what did it find?", user=user, thread=thread)
    assert third.status is RunStatus.SUCCESS and f"found {uid}" in third.answer, third.answer
    kept = await history(h, user=user, agent_id=agent.id, thread=thread)
    assert sum(token(uid, 3) in c for r, c in kept if r.upper() == "USER") == 1


def _async_spec(server: StandIn) -> dict[str, Any]:
    return {
        "name": "optimizer",
        "description": "Runs the slow optimization, in the background.",
        "graph_id": "optimizer",
        "url": server.url,
    }


@plan("BG-01", "BG-09", "BG-12")
@pytest.mark.parametrize("adapter", ["react", "deepagents"])
async def test_deep_agents_async_task_while_the_user_chats(
    h: Harness, adapter: str, uid: str, tmp_path: Path
) -> None:
    server = StandIn(seconds=3600)
    server.start()
    try:
        if adapter == "react":
            target: Any = ReAct(SYSTEM, BrainChat(), subagents=[_async_spec(server)])
        else:
            target = create_deep_agent(
                model=BrainChatModel(),
                subagents=cast(Any, [_async_spec(server)]),
                system_prompt=SYSTEM,
            )
        agent = arm(h.wrap(target, id=f"s5d-{adapter}-{uid}"), TOOLS)
        user, thread = f"ada-{uid}", f"chat-{uid}"
        start = command(
            "start_async_task", description="Find the optimum.", subagent_type="optimizer"
        )
        first = await agent.run(f"{token(uid, 1)} {start}", user=user, thread=thread)
        assert first.status is RunStatus.SUCCESS, first.error
        [task] = list(server.runs)
        assert task in parse(first.answer).last, first.answer  # the turn replied with the task id
        second = await agent.run(f"{token(uid, 2)} meanwhile, hello", user=user, thread=thread)
        assert second.status is RunStatus.SUCCESS and parse(second.answer).mine == token(uid, 2)
        assert server.status(task) == "running"  # answered while the task works
        server.release()
        check = command("check_async_task", task_id=task)
        third = await agent.run(f"{token(uid, 3)} {check}", user=user, thread=thread)
        assert third.status is RunStatus.SUCCESS, third.error
        assert RESULT in third.answer, third.answer
        assert [c for c in server.calls if c[0] == "POST" and c[1].endswith("/runs")] == [
            ("POST", f"/threads/{task}/runs")
        ], "the task was started more than once"
    finally:
        server.stop()


@plan("BG-07", "BG-10")
@pytest.mark.xfail(
    strict=True, reason="pending branch bg-kimi: Kimi's background work ends with its process"
)
async def test_kimi_background_shell_result_reaches_a_later_turn(
    h: Harness, uid: str, tmp_path: Path
) -> None:
    agent, built = await wrapped(h, "kimi", TOOLS, tmp_path, id=f"s5k-{uid}")
    user, thread = f"ada-{uid}", f"chat-{uid}"
    shell = command("Bash", command=f"sleep 8; echo BGDONE-$((40+2))-{uid}", run_in_background=True)
    first = await agent.run(f"{token(uid, 1)} {shell}", user=user, thread=thread)
    assert first.status is RunStatus.SUCCESS, first.error
    meanwhile = await agent.run(
        f"{token(uid, 2)} meanwhile", user=user, thread=thread
    )  # still working
    assert meanwhile.status is RunStatus.SUCCESS, meanwhile.error
    await asyncio.sleep(10)  # the job ends after the turn that started it, and its process, ended
    later = await agent.run(f"{token(uid, 3)} is it done?", user=user, thread=thread)
    assert later.status is RunStatus.SUCCESS, later.error
    assert f"BGDONE-42-{uid}" in built.model.said(), "the background output never reached Kimi"


@plan("BG-01")
@pytest.mark.xfail(
    strict=True, reason="pending branch react-background: ReAct's own sub-agents in the background"
)
async def test_react_own_subagent_in_the_background_over_agui(h: Harness, uid: str) -> None:
    import httpx
    from fastapi import FastAPI, Request

    from trellis.harness.agui.sse import decode

    research = h.wrap(ReAct("You research.", BrainChat()), id=f"s5r-research-{uid}")
    chat = h.wrap(ReAct(SYSTEM, BrainChat(), subagents=[research]), id=f"s5r-chat-{uid}")
    app = FastAPI()

    def user_of(request: Request) -> str:
        return f"ada-{uid}"

    chat.serve_chat(app, identity=user_of)
    start = command(
        "start_async_task", description=f"{token(uid, 9)} research", subagent_type=research.id
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://agui"
    ) as client:
        sent = await client.post(
            "/agui/run",
            json={
                "threadId": f"t-{uid}",
                "messages": [{"id": "m1", "role": "user", "content": f"{token(uid, 1)} {start}"}],
            },
        )
        events = decode(sent.text)
        assert events[-1][1]["outcome"] == {"type": "success"}, events[-1]
        assert UUID.search(json.dumps(events)), "no task was started"


@plan("BG-10", "WF-08")
@pytest.mark.xfail(
    strict=True,
    reason="pending branch bg-claude-da: Claude Workflows (one run per workflow, or refused)",
)
async def test_claude_workflow_is_one_child_run(h: Harness, uid: str, tmp_path: Path) -> None:
    claude = [
        {
            "workflow": {
                "id": "w1",
                "agents": [{"agent": "a", "prompt": "one"}, {"agent": "b", "prompt": "two"}],
            }
        }
    ]
    agent, _ = await wrapped(h, "claude_agent_sdk", TOOLS, tmp_path, id=f"s5w-{uid}", claude=claude)
    done = await agent.run(
        f"{token(uid, 1)} run the workflow", user=f"ada-{uid}", thread=f"t-{uid}"
    )
    assert done.status is RunStatus.SUCCESS, done.error
    children = await _children(h, done.run_id)
    assert len(children) == 1, children  # one run for the workflow
