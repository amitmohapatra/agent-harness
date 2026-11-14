"""Scenario 6 — failure and recovery (REL, CONV-10, MEM-10, LR-03).

* A harness server is SIGKILLed mid-turn: agent-runs ends the run ``ERROR`` (``holder_lost``)
  once its hold lapses, and the chat goes on on another server.
* The memory service goes down mid-chat: the turn still answers (the documented fallback: no
  context, a warning, the memory tools left out), and once it is back the turn's transcript is
  delivered (the harness's background writes retry): nothing lost.
* agent-runs is slow (its process stopped for seconds) or briefly down (stopped and started
  again) while a turn works: the turn still ends well.
* A model error, a run past its time limit, a step limit and a budget limit each end the run
  with the documented status.

The services are this suite's own stack (``SC_STACK``: ``scripts/scenarios/stack.sh``); the
tests that stop one of them skip without it.
"""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
from pathlib import Path

import httpx
import pytest

from tests.scenarios.brains import ADAPTERS, command, parse, token, wrapped
from tests.scenarios.kit import TOOLS, Served, eventually, free_port, history, record
from trellis import Budget, Harness
from trellis.contracts import RunStatus
from trellis.harness.agui.sse import decode

pytestmark = pytest.mark.timeout(900)
STACK = os.environ.get("SC_STACK")
needs_stack = pytest.mark.skipif(
    not STACK, reason="needs the suite's own stack (SC_STACK) to stop a service"
)


def _stack(*args: str) -> None:
    assert STACK is not None
    subprocess.run([STACK, *args], check=True, capture_output=True, timeout=600)


def _pid(name: str) -> int:
    return int(
        (Path(os.environ.get("SC", "/var/tmp/cons/scen-tmp")) / "run" / f"{name}.pid").read_text()
    )


async def _up(url: str, within: float = 300) -> None:
    async def live() -> bool:
        try:
            async with httpx.AsyncClient() as c:
                return (await c.get(url + "/health/ready", timeout=3)).status_code == 200
        except httpx.HTTPError:
            return False

    assert await eventually(live, within=within, every=1), f"{url} never came back"


# --------------------------------------------------------------------------- a killed server
@pytest.mark.plan("CONV-10", "REL", "LR-05")
@pytest.mark.parametrize("adapter", ADAPTERS)
async def test_a_server_killed_mid_turn(
    adapter: str, uid: str, tmp_path: Path, invariants: object
) -> None:
    from tests.scenarios.conftest import harness

    agent_id = f"s6k-{adapter}-{uid}"
    claude = [{"tool": "wait", "args": {"seconds": 120}}]
    spec = {
        "mode": "agui",
        "agents": [{"adapter": adapter, "id": agent_id, "claude": claude}],
        "hold": 6,
    }
    user, thread = f"ada-{uid}", f"chat-{uid}"
    one = await Served({**spec, "port": free_port()}, tmp_path, "one").started()
    run_id = f"run_{uid}killed"
    async with httpx.AsyncClient(
        base_url=f"http://127.0.0.1:{one.port}", headers={"x-user": user}, timeout=300
    ) as c:
        body = {
            "threadId": thread,
            "runId": run_id,
            "messages": [
                {
                    "id": "m1",
                    "role": "user",
                    "content": f"{token(uid, 1)} {command('wait', seconds=120)}",
                }
            ],
        }
        posting = asyncio.create_task(c.post(f"/agui/{agent_id}/run", json=body))
        async with harness() as h:

            async def working() -> bool:
                got = await h.runs.get(run_id)
                return got is not None and got.status is RunStatus.RUNNING

            assert await eventually(working, within=120), one.log.read_text()[-2000:]
            await asyncio.sleep(2)  # in its slow call
            one.kill()
            posting.cancel()

            async def lost() -> bool:
                got = await h.runs.get(run_id)
                return got is not None and got.status is RunStatus.ERROR

            assert await eventually(lost, within=60), await h.runs.get(run_id)
            ended = await record(h, run_id)
            assert ended.error is not None and ended.error.code == "holder_lost", ended.error
            two = await Served({**spec, "port": free_port()}, tmp_path, "two").started()
            try:
                async with httpx.AsyncClient(
                    base_url=f"http://127.0.0.1:{two.port}", headers={"x-user": user}, timeout=300
                ) as c2:
                    nxt = await c2.post(
                        f"/agui/{agent_id}/run",
                        json={
                            "threadId": thread,
                            "messages": [
                                {
                                    "id": "m2",
                                    "role": "user",
                                    "content": f"{token(uid, 2)} are you there?",
                                }
                            ],
                        },
                    )
                    finished = decode(nxt.text)[-1][1]
                    assert finished.get("outcome") == {"type": "success"}, finished
                    assert parse(finished["result"]).mine == token(uid, 2), finished
            finally:
                two.stop()
            assert two.violations() == [] and one.violations() == []


# --------------------------------------------------------------------------- memory down
@needs_stack
@pytest.mark.plan("MEM-10", "REL")
async def test_memory_down_mid_chat_then_back(uid: str, tmp_path: Path, invariants: object) -> None:
    from tests.scenarios.conftest import harness

    adapters = ["function", "react", "langgraph", "openai_agents", "claude_agent_sdk"]
    async with harness() as h:
        agents = {
            a: (await wrapped(h, a, TOOLS, tmp_path, id=f"s6m-{a}-{uid}"))[0] for a in adapters
        }
        user = f"ada-{uid}"
        for a, agent in agents.items():
            ok = await agent.run(
                f"{token(a[:3] + uid[:4], 1)} before", user=user, thread=f"t-{a}-{uid}"
            )
            assert ok.status is RunStatus.SUCCESS, (a, ok.error)
        await h.writes.drain()
        invariants.allow("INV-7", "the memory service is stopped on purpose: refused connections")  # type: ignore[attr-defined]
        _stack("stop-one", "memory")
        try:
            for a, agent in agents.items():
                down = await agent.run(
                    f"{token(a[:3] + uid[:4], 2)} while memory is down",
                    user=user,
                    thread=f"t-{a}-{uid}",
                )
                assert down.status is RunStatus.SUCCESS, (a, down.error)  # answered without memory
                assert parse(down.answer).mine == token(a[:3] + uid[:4], 2), (a, down.answer)
        finally:
            _stack("start-one", "memory")
        await _up(os.environ["MEMORY_URL"])
        await h.writes.drain()
        for a, agent in agents.items():
            kept = await history(h, user=user, agent_id=agent.id, thread=f"t-{a}-{uid}")
            told = " ".join(c for r, c in kept if r.upper() == "USER")
            assert token(a[:3] + uid[:4], 2) in told, (
                a,
                "the turn made while memory was down was lost",
                kept,
            )
            assert h.writes.failed == 0


# --------------------------------------------------------------------------- runs slow / down
@needs_stack
@pytest.mark.plan("REL")
@pytest.mark.parametrize("fault", ["slow", "down"])
@pytest.mark.parametrize("adapter", ["react", "openai_agents", "claude_agent_sdk"])
async def test_runs_service_slow_or_briefly_down_mid_turn(  # noqa: PLR0917 - pytest's fixtures
    h: Harness, adapter: str, fault: str, uid: str, tmp_path: Path, invariants: object
) -> None:
    claude = [{"tool": "wait", "args": {"seconds": 6}}]
    agent, _ = await wrapped(h, adapter, TOOLS, tmp_path, id=f"s6r-{adapter}-{uid}", claude=claude)
    turn = asyncio.create_task(
        agent.run(
            f"{token(uid, 1)} {command('wait', seconds=6)}", user=f"ada-{uid}", thread=f"t-{uid}"
        )
    )
    await asyncio.sleep(2)
    pid = _pid("runs-api")
    if fault == "slow":
        os.kill(pid, signal.SIGSTOP)
        await asyncio.sleep(8)
        os.kill(pid, signal.SIGCONT)
    else:
        invariants.allow("INV-7", "agent-runs is stopped on purpose")  # type: ignore[attr-defined]
        _stack("stop-one", "runs-api")
        await asyncio.sleep(3)
        _stack("start-one", "runs-api")
        await _up(os.environ["RUNS_URL"], within=60)
    done = await turn
    assert done.status is RunStatus.SUCCESS, (fault, done.status, done.error)
    assert (await record(h, done.run_id)).status is RunStatus.SUCCESS


# --------------------------------------------------------------------------- model, time, steps, budget
@pytest.mark.plan("REL", "LR-03")
@pytest.mark.parametrize("adapter", [a for a in ADAPTERS if a != "claude_agent_sdk"])
async def test_a_model_error_ends_the_run_error(
    h: Harness, adapter: str, uid: str, tmp_path: Path
) -> None:
    """(The scripted Claude CLI's model cannot fail on a message's word: its script is fixed.)"""
    agent, _ = await wrapped(h, adapter, TOOLS, tmp_path, id=f"s6e-{adapter}-{uid}")
    done = await agent.run(f"{token(uid, 1)} fail:model", user=f"ada-{uid}", thread=f"t-{uid}")
    assert done.status is RunStatus.ERROR and done.error is not None, (done.status, done.answer)
    after = await agent.run(f"{token(uid, 2)} and now?", user=f"ada-{uid}", thread=f"t-{uid}")
    assert after.status is RunStatus.SUCCESS, after.error  # the chat goes on


@pytest.mark.plan("REL", "LR-03")
@pytest.mark.parametrize("adapter", ADAPTERS)
async def test_past_its_time_limit_the_run_ends_timeout(
    h: Harness, adapter: str, uid: str, tmp_path: Path
) -> None:
    claude = [{"tool": "wait", "args": {"seconds": 60}}]
    agent, _ = await wrapped(h, adapter, TOOLS, tmp_path, id=f"s6t-{adapter}-{uid}", claude=claude)
    done = await agent.run(
        f"{token(uid, 1)} {command('wait', seconds=60)}",
        user=f"ada-{uid}",
        thread=f"t-{uid}",
        run_timeout=4,
    )
    assert done.status is RunStatus.TIMEOUT, (done.status, done.error)
    assert (await record(h, done.run_id)).status is RunStatus.TIMEOUT


@pytest.mark.plan("REL", "LR-02", "LR-03")
@pytest.mark.parametrize("adapter", [a for a in ADAPTERS if a not in ("function",)])
async def test_a_budget_limit_ends_the_run_partial(
    h: Harness, adapter: str, uid: str, tmp_path: Path
) -> None:
    claude = [{"tool": "lookup", "args": {"key": f"k{i}"}} for i in range(4)]
    agent, _ = await wrapped(h, adapter, TOOLS, tmp_path, id=f"s6b-{adapter}-{uid}", claude=claude)
    done = await agent.run(
        f"{token(uid, 1)} loop:4:lookup",
        user=f"ada-{uid}",
        thread=f"t-{uid}",
        budget=Budget(max_tool_calls=2),
    )
    assert done.status is RunStatus.PARTIAL, (done.status, done.error, done.answer)
    assert done.error is not None and "BUDGET" in done.error.code.upper(), done.error


@pytest.mark.plan("LR-02", "LR-03")
@pytest.mark.parametrize("adapter", ["react", "react_da", "middleware"])
async def test_a_step_limit_stops_a_long_task_and_it_still_answers(
    h: Harness, adapter: str, uid: str, tmp_path: Path
) -> None:
    from langchain.agents import create_agent

    from tests.scenarios.brains import SYSTEM, BrainChat, BrainChatModel
    from tests.support.arm import arm
    from trellis import ReAct, middleware
    from trellis.harness.automatic import JournalBackend
    from trellis.harness.middleware import RunCheckpointer

    if adapter == "middleware":
        target = create_agent(
            BrainChatModel(),
            system_prompt=SYSTEM,
            middleware=middleware(max_steps=5),
            checkpointer=RunCheckpointer(),
        )
    else:
        target = ReAct(
            SYSTEM,
            BrainChat(),
            max_steps=5,
            backend=JournalBackend() if adapter == "react_da" else None,
        )
    agent = arm(h.wrap(target, id=f"s6s-{adapter}-{uid}"), TOOLS)
    done = await agent.run(f"{token(uid, 1)} loop:60:lookup", user=f"ada-{uid}", thread=f"t-{uid}")
    assert done.status in (RunStatus.SUCCESS, RunStatus.PARTIAL), (done.status, done.error)
    assert parse(done.answer).mine == token(uid, 1), done.answer  # told to answer, it answered
    events = [e async for e in agent.events(done.run_id)]
    calls = [e for e in events if e.type.value == "TOOL_CALL_START"]
    assert len(calls) <= 5, len(calls)
