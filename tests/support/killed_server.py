"""A server process for the killed-server tests: ``python -m tests.support.killed_server <store>
<folder> <scenario>`` starts the work of ``scenario`` with the run store in the file ``store``
(``shared_runs.FileRuns``) and, once that work is under way — :data:`RUNNING` runs of it
``RUNNING`` in the store, for longer than a heartbeat's rhythm — writes their ids to
``folder/ready`` and waits, to be killed (``SIGKILL``: nothing closed, as a crashed or restarted
server). Each hold lasts :data:`HOLD_SECONDS`.

* ``turn``: an ordinary turn (a function agent) whose work takes for ever — a long tool call;
* ``subagent``: a turn that called a sub-agent (``agent.run`` inside it: a child run), which
  works for ever;
* ``helper``: a Claude agent's turn whose subagent works in the foreground for ever (the
  scripted stand-in CLI, ``tests/support/fake_claude_cli.py``, its files in ``folder``);
* ``background``: a Claude agent's turn that started a subagent in the background and ended;
  the subagent works on in the conversation's live session;
* ``idle``: a Claude agent's turn that ended with nothing in the background (its live session
  kept, idle);
* ``react_background``: a ``ReAct`` chat turn (a front end's, as ``serve_chat`` starts it) that
  started a task of its sub-agent in the background (``trellis.harness.background``) and
  ended; the task works on, for ever;
* ``asked``: as ``background``, then the subagent's call asks a person (its run pauses) and the
  conversation's next turn asks it (that turn pauses on it): ``ready`` holds, as JSON, the
  subagent's run, the turn's run and the turn's interrupt.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Final

from claude_agent_sdk import ClaudeAgentOptions

from tests.support.arm import arm
from tests.support.chat_model import ScriptedChatModel
from tests.support.shared_runs import FileRuns
from trellis import Harness, ReAct, Runtime, Settings, tool
from trellis.contracts import RunStatus
from trellis.harness import holding, pipeline
from trellis.harness.frontend import Frontend

CLI: Final = str(Path(__file__).resolve().parent / "fake_claude_cli.py")
AGENT_ID: Final = "chat"
#: How long one heartbeat holds a run here (renewed each third of it).
HOLD_SECONDS: Final = 1
#: How many of the scenario's runs are ``RUNNING`` once its work is under way.
RUNNING: Final = {
    "turn": 1,
    "subagent": 2,
    "helper": 2,
    "background": 1,
    "idle": 0,
    "react_background": 1,
}
BACKGROUND: Final = {"background": {"id": "t1", "agent": "researcher", "prompt": "research"}}
#: a subagent in the foreground whose tool call never ends
FOREGROUND: Final = {
    "agent": "researcher",
    "prompt": "research",
    "steps": [{"tool": "look_up", "args": {}}, {"text": "found"}],
}


@tool(side_effects="irreversible")
def refund(order: str) -> str:
    """Refund an order (asks a person first)."""
    return f"refunded {order}"


@tool(side_effects="read")
async def look_up() -> str:
    """Look something up (it takes for ever)."""
    await asyncio.Event().wait()
    return "never"


async def forever(input: str, agent: Runtime) -> str:
    """A run's work that never ends (a long tool call, a slow model)."""
    await asyncio.Event().wait()
    return "never"


def claude(folder: Path, *scripts: list[dict[str, Any]]) -> ClaudeAgentOptions:
    (folder / "tasks").mkdir(exist_ok=True)
    env = {
        "FAKE_CLAUDE_TURNS": json.dumps(list(scripts)),
        "FAKE_CLAUDE_RECORD": str(folder / "cli.json"),
        "FAKE_CLAUDE_SESSIONS": str(folder),
        "FAKE_CLAUDE_TASKS": str(folder / "tasks"),
        "FAKE_CLAUDE_BUILTINS": str(folder / "builtins.jsonl"),
    }
    return ClaudeAgentOptions(cli_path=CLI, env=env)


def ready(folder: Path, found: list[str]) -> None:
    (folder / "ready").write_text("\n".join(found))


def calls(folder: Path, step: dict[str, Any]) -> None:
    """Task ``t1`` makes the call ``step`` (``fake_claude_cli``'s task files)."""
    (folder / "tasks" / "t1.call").write_text(json.dumps(step))


async def asked(harness: Harness, store: FileRuns, folder: Path) -> None:
    """The ``asked`` scenario, up to the turn that asks the subagent's question."""
    target = claude(folder, [BACKGROUND, {"text": "started it"}], [{"text": "ok"}])
    agent = arm(harness.wrap(target, id=AGENT_ID), [refund])
    first = await agent.run("research the market", user="u1", thread="t1")
    [child] = await running(store)
    calls(folder, {"tool": "refund", "args": {"order": "o9"}})
    while (await store.get(child)).status is not RunStatus.PAUSED:  # type: ignore[union-attr]  # noqa: ASYNC110
        await asyncio.sleep(0.05)
    second = await agent.run("and now?", user="u1", thread="t1")
    assert first.status is RunStatus.SUCCESS and second.interrupt is not None
    said = {"child": child, "turn": second.run_id, "interrupt": second.interrupt.interrupt_id}
    ready(folder, [json.dumps(said)])


async def turn(agent: Any, scenario: str) -> Any:
    """The scenario's turn: a chat turn as ``serve_chat`` starts one (a front end's) for a
    ``ReAct`` whose sub-agents work in the background, else ``agent.run``."""
    if scenario != "react_background":
        return await agent.run("research the market", user="u1", thread="t1")
    record = await agent._opened("dig", user="u1", thread="t1", tenant=None, app=Frontend())
    return await pipeline.attempt(agent, record, "dig")


async def running(store: FileRuns) -> list[str]:
    with store.synced():
        return [r.run_id async for r in store.iterate(status=RunStatus.RUNNING)]


async def main(store_path: Path, folder: Path, scenario: str) -> None:
    setattr(holding, "HOLD_SECONDS", HOLD_SECONDS)  # noqa: B010 - a Final, for this process
    store = FileRuns(store_path)
    async with Harness(config=Settings(), runs=store) as harness:
        if scenario == "asked":
            await asked(harness, store, folder)
            await asyncio.Event().wait()  # until killed
        if scenario == "turn":
            agent = harness.wrap(forever, id=AGENT_ID)
        elif scenario == "subagent":
            worker = harness.wrap(forever, id="worker")

            async def boss(input: str, agent: Runtime) -> str:
                return str(await worker.run("dig", user="u1"))

            agent = harness.wrap(boss, id=AGENT_ID)
        elif scenario == "helper":
            target = claude(folder, [FOREGROUND, {"text": "done"}])
            agent = arm(harness.wrap(target, id=AGENT_ID), [look_up])
        elif scenario == "idle":
            agent = harness.wrap(claude(folder, [{"text": "hello"}]), id=AGENT_ID)
        elif scenario == "react_background":
            worker = harness.wrap(forever, id="worker")
            start = ("start_async_task", {"description": "dig", "subagent_type": "worker"})
            model = ScriptedChatModel(turns=[start, "started it"])
            agent = harness.wrap(ReAct("You help.", model, subagents=[worker]), id=AGENT_ID)
        else:
            script = [BACKGROUND, {"text": "started it"}]
            agent = harness.wrap(claude(folder, script), id=AGENT_ID)
        work = asyncio.create_task(turn(agent, scenario))
        while len(found := await running(store)) < RUNNING[scenario] or (  # noqa: ASYNC110
            scenario in ("background", "idle", "react_background") and not work.done()
        ):
            await asyncio.sleep(0.05)
        await asyncio.sleep(HOLD_SECONDS)  # a heartbeat's rhythm, and more
        ready(folder, found)
        await asyncio.Event().wait()  # until killed


if __name__ == "__main__":
    asyncio.run(main(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]))
