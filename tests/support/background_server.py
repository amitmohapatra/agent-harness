"""A server process for the background-run restart test: ``python -m
tests.support.background_server <store> <folder>`` serves one turn of a Claude agent (the
scripted stand-in CLI, ``tests/support/fake_claude_cli.py``, its files in ``folder``) that starts
a subagent in the background, with the run store in the file ``store``
(``shared_runs.FileRuns``). Once the session has held the subagent's child run in the store, it
writes the run's id to ``folder/held`` and waits — to be killed (``SIGKILL``: nothing closed,
as a crashed or restarted server). Each hold lasts :data:`HOLD_SECONDS`."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Final

from claude_agent_sdk import ClaudeAgentOptions

from tests.support.shared_runs import FileRuns
from trellis import Harness, Settings
from trellis.harness.bridges import claude_live
from trellis.runs import Lease

CLI: Final = str(Path(__file__).resolve().parent / "fake_claude_cli.py")
AGENT_ID: Final = "chat"
#: How long one heartbeat of the session holds its background run (renewed each third of it).
HOLD_SECONDS: Final = 1
BACKGROUND: Final = {"background": {"id": "t1", "agent": "researcher", "prompt": "research"}}


class Holding(FileRuns):
    """The shared store, writing the id of each run it was asked to hold to :attr:`held`."""

    #: where (set by :func:`main`; on the class, as what is on the store goes in its file)
    held = Path()

    async def heartbeat(self, run_id: str, worker_id: str, **given: Any) -> Lease:
        lease = await super().heartbeat(run_id, worker_id, **given)
        self._said(run_id)
        return lease

    def _said(self, run_id: str) -> None:
        type(self).held.write_text(run_id)


async def main(store_path: Path, folder: Path) -> None:
    Holding.held = folder / "held"
    setattr(claude_live, "HOLD_SECONDS", HOLD_SECONDS)  # noqa: B010 - a Final, for this process
    (folder / "tasks").mkdir(exist_ok=True)
    env = {
        "FAKE_CLAUDE_TURNS": json.dumps([[BACKGROUND, {"text": "started it"}]]),
        "FAKE_CLAUDE_RECORD": str(folder / "cli.json"),
        "FAKE_CLAUDE_SESSIONS": str(folder),
        "FAKE_CLAUDE_TASKS": str(folder / "tasks"),
    }
    async with Harness(config=Settings(), runs=Holding(store_path)) as harness:
        agent = harness.wrap(ClaudeAgentOptions(cli_path=CLI, env=env), id=AGENT_ID)
        await agent.run("research the market", user="u1", thread="t1")
        await asyncio.Event().wait()  # until killed


if __name__ == "__main__":
    asyncio.run(main(Path(sys.argv[1]), Path(sys.argv[2])))
