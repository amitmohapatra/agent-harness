"""A server process for the double-delivery test: ``python -m tests.support.notified_twice
<store> <folder> <name>`` builds the chat agent on the run store in the file ``store``
(``shared_runs.FileRuns``), waits for ``folder/go``, then tells the chat the outcome of the task
whose run ``folder/task`` names (``background.notify``: a turn of the chat's own), as a server
that found the chat idle does, and writes how many model calls it made to ``folder/<name>``."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

from tests.support.chat_model import ScriptedChatModel
from tests.support.shared_runs import FileRuns
from trellis import Harness, ReAct, Settings
from trellis.harness import background


class FarRuns(FileRuns):
    """The store across the network: a read answers after a while, so both servers read that
    the chat's own turn is not made yet before either makes it."""

    async def get(self, run_id: str, *, tenant: str | None = None) -> Any:
        found = await super().get(run_id, tenant=tenant)
        await asyncio.sleep(0.3)
        return found


async def found(input: str, agent: object) -> str:
    """Research a topic."""
    return "FOUND"


async def main(store_path: Path, folder: Path, name: str) -> None:
    store = FarRuns(store_path)
    async with Harness(config=Settings(), runs=store) as harness:
        research = harness.wrap(found, id="research")
        model = ScriptedChatModel(turns=["Your research is back."])
        agent = harness.wrap(ReAct("You help.", model, subagents=[research]), id="chat")
        while not (folder / "go").exists():  # noqa: ASYNC110 - another process
            await asyncio.sleep(0.01)
        task = await store.get((folder / "task").read_text(), tenant="default")
        assert task is not None
        origin = task.metadata[background.BACKGROUND]
        update = background.update_of({"task_id": task.thread_id, "agent_name": "research"}, task)
        assert update is not None
        await background.notify(agent, task, update, user="u1", thread="t1", origin=origin)
        while harness.background._at_work:  # noqa: ASYNC110 - its own turn's attempt
            await asyncio.sleep(0.02)
        (folder / name).write_text(str(len(model.seen)))


if __name__ == "__main__":
    asyncio.run(main(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]))
