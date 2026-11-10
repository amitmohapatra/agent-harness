"""``ReAct`` on Deep Agents on a very small window: a long chat whose tool results are dense
JSON (more tokens per character than the estimate counts) never sends a request over the
window and never ends in an error for its size — the window's server counting its own way."""

from __future__ import annotations

import json
from typing import Any

import httpx

from tests.support.arm import arm
from tests.support.models import Script, ScriptedChat
from trellis import Harness, ReAct, tool
from trellis.contracts import RunStatus
from trellis.harness.automatic import JournalBackend

WINDOW = 6_000
#: the server's tokens: dense JSON is about 3 characters a token, not the estimate's 4
CHARS_PER_TOKEN = 3


def tokens(body: dict[str, Any]) -> int:
    """A request's size as the window's server counts it."""
    return len(json.dumps([body.get("messages"), body.get("tools")])) // CHARS_PER_TOKEN


class Server(Script):
    """A model server with a ``WINDOW``-token window: each turn's question answered with one
    catalog call then a reply, a request to summarize with a summary, and a request over the
    window refused with its count, as Anthropic's is."""

    def __init__(self) -> None:
        super().__init__([])
        self.sizes: list[int] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        size = tokens(json.loads(request.content))
        self.sizes.append(size)
        if size > WINDOW:
            message = f"prompt is too long: {size} tokens > {WINDOW} maximum"
            error = {"type": "invalid_request_error", "message": message}
            return httpx.Response(400, json={"error": error})
        return super().handle(request)

    def reply(self, body: dict[str, Any]) -> dict[str, Any]:
        reply = super().reply(body)
        reply["usage"] = {"prompt_tokens": tokens(body), "completion_tokens": 5, "total_tokens": 1}
        return reply

    def next(self, body: dict[str, Any]) -> Any:
        last = body["messages"][-1]
        if "Messages to summarize" in str(last.get("content")):
            return "SESSION INTENT: count SKUs. The west's Q1 revenue was 41,234."
        if last.get("role") == "tool":
            return "12 of them."
        return ("catalog", {"category": str(last.get("content"))[-12:]})


@tool(side_effects="read")
def catalog(category: str) -> str:
    """The SKUs of a category, with their prices."""
    rows = [
        {"sku": f"SKU-{n:05d}", "name": f"Item {n}", "price": 100 + n * 7, "cat": category}
        for n in range(30)
    ]
    return json.dumps(rows, separators=(",", ":"))


async def test_a_long_chat_on_a_small_window_never_goes_over_it(harness: Harness) -> None:
    server = Server()
    model = ScriptedChat(script=server)
    target = ReAct(
        system_prompt="You answer questions about the catalog.",
        model=model,
        context_window=WINDOW,
        backend=JournalBackend(),
    )
    agent = arm(harness.wrap(target, id="small-window"), [catalog])
    for turn in range(14):
        done = await agent.run(f"How many SKUs priced over 100 in cat-{turn:02d}?", user="u",
                               thread="t1")  # fmt: skip
        assert done.status is RunStatus.SUCCESS, (turn, done.error)
    assert max(server.sizes) <= WINDOW, server.sizes
