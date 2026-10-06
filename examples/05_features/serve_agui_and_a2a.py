"""Serving: one agent behind AG-UI (a chat UI) and A2A (other agents) on one FastAPI app, and
another wrapped agent that calls it as a tool, ``a2a(url)``.

* ``agent.serve_chat(app, identity=...)`` — AG-UI over SSE: a run's events as they happen, a
  question as an interrupt the UI answers, a reconnect that replays from ``Last-Event-ID``;
* ``agent.serve_a2a(app, url)`` — the agent card and A2A JSON-RPC at ``url``;
* ``a2a(url)`` in another agent's ``tools=[...]`` — the remote agent as one tool: its question
  becomes the caller's own pause, answered with ``agent.resume``, and goes back to it.

``identity=`` is how your deployment says who is calling (here: a header your auth proxy sets);
without it every chat caller is ``anonymous``.

    uvicorn examples.05_features.serve_agui_and_a2a:app --port 8000   # serve it
    python -m examples.05_features.serve_agui_and_a2a                  # exercise it and exit
"""

from __future__ import annotations

import asyncio
import socket
import threading
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, Request

from trellis import Harness, Runtime, a2a

h = Harness()
app = FastAPI()


async def concierge(input: str, agent: Runtime) -> str:
    if "book" in input:
        when = await agent.ask("Which evening?", options=["friday", "saturday"])
        return f"Booked for {when}."
    return "Ask me to book a table."


def user(request: Request) -> str:
    return request.headers.get("x-user", "anonymous")


agent = h.wrap(concierge, id="concierge")
agent.serve_chat(app, path="/agui", identity=user)


async def assistant(input: str, runtime: Runtime) -> Any:
    """Another agent: the concierge is one of its tools."""
    return f"Done: {await runtime.tools.call('concierge', message=input)}"


async def main() -> None:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    agent.serve_a2a(app, f"{base}/a2a")
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    while not server.started:  # noqa: ASYNC110 - uvicorn says so with a flag, not an event
        await asyncio.sleep(0.05)
    try:
        async with httpx.AsyncClient(base_url=base) as client:
            card = await client.get("/a2a/.well-known/agent-card.json")
            print("A2A card:", card.json()["name"])
            run = {
                "threadId": "t1",
                "runId": "r1",
                "messages": [{"id": "m1", "role": "user", "content": "book a table"}],
            }
            async with client.stream(
                "POST", "/agui/run", json=run, headers={"x-user": "ada"}
            ) as sse:
                events = [line async for line in sse.aiter_lines() if line.startswith("data:")]
            print("AG-UI events:", len(events), "last:", events[-1][:100])

        async with Harness() as caller:
            helper = caller.wrap(
                assistant, id="assistant", tools=[a2a(f"{base}/a2a", name="concierge")]
            )
            result = await helper.run("book a table", user="ada")
            while result.interrupt is not None:  # the concierge's question, asked here
                print("the remote agent asks:", result.interrupt.question)
                result = await helper.resume(
                    result.run_id, "answer", answer="friday", reviewer="ada"
                )
            print(result.status.value, result.answer)
    finally:
        server.should_exit = True
        await h.aclose()


if __name__ == "__main__":
    asyncio.run(main())
