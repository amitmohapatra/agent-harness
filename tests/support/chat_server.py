"""A server process for the cross-process conversation tests (``tests/live/test_live_chat_order
.py``): ``python -m tests.support.chat_server <runs_url> <api_key> <port> <folder>`` serves one
agent per adapter that can run here, on agent-runs at ``runs_url`` with memory off — a
conversation's earlier turns come from the run store, whichever server ran them — and a worker
that claims their queued runs. Each agent (``chat-<adapter>``, the same id on every server) calls
``step`` once and answers (``tests.support.adapters.BUILDERS``); the function agent's turn asked
``refund <order>`` calls ``refund`` instead, which a person approves first.

``step`` logs the turn to ``folder/turns.jsonl`` (``{"event": "start", "server", "adapter",
"question", "history", "at"}``: ``history`` the earlier turns it was given, ``Runtime.history``,
``null`` where the framework continued its own session instead; then ``"end"``, or
``"cancelled"``) and works while ``folder/hold-<question>`` exists. A hold lasts
``HOLD_SECONDS`` from the environment when set: a killed server's holds lapse that soon.

Routes (JSON): ``POST /run {adapter, input, user, thread, mode, stream}`` runs a turn
(``agent.run``, or ``agent.stream`` read to its end) and answers its result; ``POST /resume
{adapter, interrupt_id}`` approves; ``POST /start {adapter, input, user, thread}`` queues one;
``POST /close`` ends what the agents keep between turns; ``GET /ready``; and the function agent's AG-UI routes (``serve_chat``) at ``/agui``.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Final

import uvicorn
from fastapi import FastAPI

from tests.support.adapters import BUILDERS
from tests.support.arm import arm
from tests.support.invariants import record_in_process
from trellis import Harness, Runtime, Settings, current, tool
from trellis.contracts import InterruptDecision
from trellis.harness import holding
from trellis.harness.agent import Agent
from trellis.runs import RunsClient

#: The adapters a server runs (Kimi where its program is installed).
ADAPTERS: Final = [a for a in BUILDERS if a != "kimi"]


def logged(folder: Path, **entry: Any) -> None:
    with (folder / "turns.jsonl").open("a") as out:
        out.write(json.dumps({**entry, "at": time.time()}) + "\n")


def serving(runs_url: str, api_key: str, folder: Path) -> tuple[FastAPI, Harness]:
    server = f"pid-{os.getpid()}"
    harness = Harness(config=Settings(), memory=False, runs=RunsClient(runs_url, api_key=api_key))
    agents: dict[str, Agent] = {}

    @tool(side_effects="read")
    async def step() -> str:
        """Do the work of this turn."""
        runtime = current()
        assert runtime is not None
        question = str(runtime.task)
        said = {"server": server, "adapter": runtime.agent_id, "question": question}
        history = None if runtime.history is None else [s.content for s in runtime.history]
        logged(folder, event="start", history=history, **said)
        try:
            while (folder / f"hold-{question}").exists():  # noqa: ASYNC110 - the test's file
                await asyncio.sleep(0.05)
        except asyncio.CancelledError:
            logged(folder, event="cancelled", **said)
            raise
        logged(folder, event="end", **said)
        return f"worked on {question}"

    @tool(side_effects="irreversible")
    def refund(order: str) -> str:
        """Refund an order (a person approves it first)."""
        logged(folder, event="refunded", server=server, question=order)
        return f"refunded {order}"

    async def function(input: str, agent: Runtime) -> str:
        if input.startswith("refund "):
            await agent.tools.call("refund", order=input.removeprefix("refund "))
        else:
            await agent.tools.call("step")
        return f"re: {input}"

    app = FastAPI()

    @app.on_event("startup")
    async def built() -> None:
        agents["function"] = arm(harness.wrap(function, id="chat-function"), [step, refund])
        agents["function"].serve_chat(app)
        names = [*ADAPTERS, "kimi"] if os.environ.get("CHAT_KIMI") else ADAPTERS
        for name in names:
            if name == "function":
                continue
            target, tools = await BUILDERS[name](harness, [step], folder, [("step", {})])
            agents[name] = arm(harness.wrap(target, id=f"chat-{name}"), tools)
        app.state.worker = asyncio.create_task(harness.worker(list(agents.values())).run())

    @app.get("/ready")
    async def ready() -> dict[str, Any]:
        return {"server": server, "adapters": sorted(agents)}

    @app.post("/run")
    async def run(body: dict[str, Any]) -> dict[str, Any]:
        agent = agents[body["adapter"]]
        given = {k: body[k] for k in ("user", "thread", "mode") if body.get(k)}
        if body.get("stream"):  # its record says how it ended, as run's result does
            events = [e async for e in agent.stream(body["input"], **given)]
            record = await harness.runs.get(events[-1].run_id)
            assert record is not None
            return {"run_id": record.run_id, "status": record.status, "answer": record.output}
        result = await agent.run(body["input"], **given)
        return result.model_dump(mode="json")

    @app.post("/resume")
    async def resume(body: dict[str, Any]) -> dict[str, Any]:
        agent = agents[body["adapter"]]
        result = await agent.resume(
            body["interrupt_id"], InterruptDecision.APPROVE, reviewer="lead"
        )
        return result.model_dump(mode="json")

    @app.post("/close")
    async def close() -> dict[str, Any]:
        """What the agents keep between turns (a conversation's Claude process) ended."""
        for agent in agents.values():
            await agent.aclose()
        return {}

    @app.post("/start")
    async def start(body: dict[str, Any]) -> dict[str, Any]:
        agent = agents[body["adapter"]]
        handle = await agent.start(body["input"], user=body["user"], thread=body["thread"])
        return {"run_id": handle.id}

    return app, harness


def main(runs_url: str, api_key: str, port: str, folder: str) -> None:
    record_in_process()  # its exchanges with agent-runs, for the test's invariants
    if seconds := os.environ.get("HOLD_SECONDS"):
        holding.HOLD_SECONDS = int(seconds)
    app, _ = serving(runs_url, api_key, Path(folder))
    uvicorn.run(app, host="127.0.0.1", port=int(port), log_level="warning")


if __name__ == "__main__":
    main(*sys.argv[1:])
