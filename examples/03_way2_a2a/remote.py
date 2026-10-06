"""Way 2, A2A: call any A2A agent from plain code with ``remote()`` — no harness on the calling
side. The remote agent's question comes back to your code: answered at once by ``on_input``,
or raised as ``InputRequired`` and answered later with ``reply``.

The agent called here is published by a harness (``serve_a2a``, its own FastAPI app served by
uvicorn on a local port), but any A2A server is called the same way. A wrapped agent calls one
as a tool instead: ``a2a(url)`` (``examples/05_features/serve_agui_and_a2a.py``).

    python -m examples.03_way2_a2a.remote
"""

from __future__ import annotations

import asyncio
import socket
import threading

import uvicorn
from fastapi import FastAPI

from trellis import Harness, Runtime
from trellis.harness.a2a import InputRequired, remote


async def planner(input: str, agent: Runtime) -> str:
    """The published agent: it asks where before it plans."""
    region = await agent.ask("Which region?", options=["eu", "us"])
    return f"deploy {input} to {region} at 02:00"


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
