"""A worker process for the restart tests: ``python -m tests.support.resume_worker <store>
<ledger> <calls>`` claims one run of :data:`AGENT_ID` from the run store in the file ``store``
(``shared_runs.FileRuns``) and executes it. The agent is a ``create_agent`` graph built with
``trellis.middleware()`` and compiled without a checkpointer, as a team builds one; its model
plans the write (``ship``, a request id of its own each time it plans it, as a model's
arguments are not the same twice) and then answers with its result. ``ship`` appends a line to
``ledger``; each model call appends one to ``calls``. With :data:`DIE` set, the process kills
itself (``SIGKILL``: nothing more written) as the model is called after the write ran."""

from __future__ import annotations

import asyncio
import os
import signal
import sys
import uuid
from pathlib import Path
from typing import Any, Final

from langchain.agents import create_agent
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool

from tests.support.chat_model import ScriptedChatModel
from tests.support.shared_runs import FileRuns
from trellis import Harness, Settings, middleware

AGENT_ID: Final = "shipping"
#: The environment variable that makes the process die after the write.
DIE: Final = "TRELLIS_TEST_DIE"


class Planner(ScriptedChatModel):
    """Plans the write until it has a result, then answers with it (logging every call)."""

    calls: str

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        with open(self.calls, "a") as log:
            log.write(f"{len(messages)}\n")
        results = [m for m in messages if isinstance(m, ToolMessage)]
        if not results:
            planned = {"order": "o1", "request": uuid.uuid4().hex}
            message = AIMessage(
                content="",
                tool_calls=[{"name": "ship", "args": planned, "id": f"call_{uuid.uuid4().hex}"}],
            )
        else:
            if os.environ.get(DIE):
                os.kill(os.getpid(), signal.SIGKILL)
            message = AIMessage(content=f"done: {results[-1].content}")
        return ChatResult(generations=[ChatGeneration(message=message)])


def agent_graph(ledger: Path, calls: Path) -> Any:
    @tool
    def ship(order: str, request: str) -> str:
        """Ship an order."""
        with open(ledger, "a") as kept:
            kept.write(f"ship {order} {request}\n")
        return f"shipped {order}"

    ship.metadata = {"side_effects": "write"}
    return create_agent(Planner(turns=[], calls=str(calls)), tools=[ship], middleware=middleware())


async def main(store: Path, ledger: Path, calls: Path) -> bool:
    async with Harness(config=Settings(), runs=FileRuns(store)) as h:
        agent = h.wrap(agent_graph(ledger, calls), id=AGENT_ID)
        return await h.worker([agent]).run_once()


if __name__ == "__main__":
    claimed = asyncio.run(main(*map(Path, sys.argv[1:4])))
    sys.exit(0 if claimed else 3)
