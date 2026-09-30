"""Serving: one agent behind AG-UI (a chat UI) and A2A (other agents), on one FastAPI app.

    uvicorn examples.serve_chat:app --port 8000      # serve it
    .venv/bin/python examples/serve_chat.py          # exercise it once in process and exit

``identity=`` is how your deployment says who is calling (here: a header set by your auth
proxy). Without it every request runs as "anonymous".
"""

from __future__ import annotations

import asyncio

import httpx
from fastapi import FastAPI, Request

from trellis import Harness, Runtime

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
agent.serve_a2a(app, "http://localhost:8000/a2a")


async def main() -> None:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        card = await client.get("/a2a/.well-known/agent-card.json")
        print("A2A card:", card.json()["name"])
        run = {
            "threadId": "t1",
            "runId": "r1",
            "messages": [{"id": "m1", "role": "user", "content": "book a table"}],
        }
        async with client.stream(
            "POST", "/agui/run", json=run, headers={"x-user": "ada"}
        ) as stream:
            events = [line async for line in stream.aiter_lines() if line.startswith("data:")]
        print("AG-UI events:", len(events), "last:", events[-1][:120])
    await h.aclose()


if __name__ == "__main__":
    asyncio.run(main())
