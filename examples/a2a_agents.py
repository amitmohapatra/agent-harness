"""A2A both ways: one agent published with ``serve_a2a`` (its own FastAPI app, here served by
uvicorn on a local port), and another that uses it as a tool, ``a2a(url)``. The remote agent's
question becomes the caller's own pause: answered with ``agent.resume``, it goes back to the
remote agent, whose answer is the tool's result. Then plain code with no harness calls the same
agent with ``remote()``: once answering its question with ``on_input``, once with ``reply``.

    .venv/bin/python examples/a2a_agents.py
"""

from __future__ import annotations

import asyncio
import socket
import threading
from typing import Any

import uvicorn
from fastapi import FastAPI

from trellis import Harness, Runtime, a2a
from trellis.harness.a2a import InputRequired, remote


async def planner(input: str, agent: Runtime) -> str:
    """The published agent: it asks where before it plans."""
    region = await agent.ask("Which region?", options=["eu", "us"])
    return f"deploy {input} to {region} at 02:00"


async def caller(input: str, agent: Runtime) -> Any:
    """The consuming agent: the remote planner is one of its tools."""
    plan = await agent.tools.call("planner", message=input)
    return f"Scheduled: {plan}"


def serve(app: FastAPI, port: int) -> uvicorn.Server:
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    while not server.started:
        threading.Event().wait(0.05)
    return server


async def main() -> None:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}/a2a"

    published = Harness()  # the published agent's deployment (its own process, in production)
    app = FastAPI()
    published.wrap(planner, id="planner").serve_a2a(app, url)
    server = serve(app, port)
    try:
        async with Harness() as h:
            agent = h.wrap(caller, id="caller", tools=[a2a(url, name="planner")])
            result = await agent.run("the shop", user="ada")
            while result.interrupt is not None:  # the planner's question, asked here
                print("the remote agent asks:", result.interrupt.question)
                result = await agent.resume(
                    result.interrupt.interrupt_id, "answer", answer="eu", reviewer="ada"
                )
            print(result.status.value, result.answer)

        # plain code: the tenant is the one the published agent's key speaks for
        tenant = await published.tenant()
        async with remote(url, tenant=tenant, user="ada", on_input=lambda q: "us") as planner_agent:
            print(planner_agent.card.name, "answers:", await planner_agent("the warehouse"))
        async with remote(url, tenant=tenant, user="ada") as planner_agent:
            try:
                print(await planner_agent("the office"))
            except InputRequired as asked:
                print("asked:", asked.question)
                print("answers:", await planner_agent.reply(asked.task_id, "eu"))
    finally:
        server.should_exit = True


if __name__ == "__main__":
    asyncio.run(main())
