"""A scripted chat-completions endpoint behind a real ``ChatOpenAI``: what ``ReAct`` asks a
gateway model with, the openai client's requests and replies included (streamed or not), and
what the judge asks (``complete``)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import httpx
from langchain_openai import ChatOpenAI
from pydantic import Field

Call = tuple[str, dict[str, Any]]
Turn = str | Call | list[Call] | dict[str, Any]
#: Where the scripted endpoint pretends to be (nothing is sent anywhere).
BASE_URL = "http://scripted.test/v1"
#: The tools every ``ReAct`` graph offers of its own: a large result's file, a cleared result.
REACT_TOOLS = ["read_file", "read_result"]


class Script:
    """The turns a model answers with, one per request, and every request it was sent."""

    def __init__(self, turns: Sequence[Turn]) -> None:
        self.turns: list[Turn] = list(turns)
        #: each request's body, as the openai client sent it
        self.requests: list[dict[str, Any]] = []
        #: each request's headers
        self.headers: list[dict[str, str]] = []

    def next(self, body: dict[str, Any]) -> Turn:
        return self.turns.pop(0)

    def handle(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.headers.append(dict(request.headers))
        reply = self.reply(body)
        if not body.get("stream"):
            return httpx.Response(200, json=reply)
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=_streamed(reply)
        )

    def reply(self, body: dict[str, Any]) -> dict[str, Any]:
        self.requests.append(body)
        turn = self.next(body)
        if isinstance(turn, str):
            message: dict[str, Any] = {"role": "assistant", "content": turn}
        elif isinstance(turn, tuple | list):
            calls = turn if isinstance(turn, list) else [turn]
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"call_{len(self.requests)}" + (f"_{n}" if n else ""),
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(args)},
                    }
                    for n, (name, args) in enumerate(calls)
                ],
            }
        else:
            message = turn
        finish = "tool_calls" if message.get("tool_calls") else "stop"
        return {
            "id": f"chatcmpl-{len(self.requests)}",
            "object": "chat.completion",
            "created": 0,
            "model": "scripted",
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }


def _streamed(reply: dict[str, Any]) -> bytes:
    """The reply as the chunks of a streamed completion (server-sent events)."""
    [choice] = reply["choices"]
    message = choice["message"]
    delta: dict[str, Any] = {"role": "assistant", "content": message.get("content") or ""}
    if message.get("tool_calls"):
        delta["tool_calls"] = [{"index": n, **c} for n, c in enumerate(message["tool_calls"])]
    base = {key: reply[key] for key in ("id", "created", "model")}
    chunks = [
        {**base, "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": delta}]},
        {
            **base,
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {}, "finish_reason": choice["finish_reason"]}],
        },
        {**base, "object": "chat.completion.chunk", "choices": [], "usage": reply["usage"]},
    ]
    events = [f"data: {json.dumps(c)}\n\n" for c in chunks] + ["data: [DONE]\n\n"]
    return "".join(events).encode()


class ScriptedChat(ChatOpenAI):
    """A ``ChatOpenAI`` whose endpoint answers each request with the next turn: text,
    ``(tool, args)``, several calls in one message (``[(tool, args), ...]``), or a raw
    message."""

    script: Any = Field(default=None, exclude=True)

    def __init__(self, turns: Sequence[Turn] = (), **kwargs: Any) -> None:
        script = kwargs.pop("script", None) or Script(turns)
        client = httpx.AsyncClient(transport=httpx.MockTransport(script.handle))
        super().__init__(  # type: ignore[call-arg]
            model=kwargs.pop("model", "scripted"),
            api_key="scripted",  # type: ignore[arg-type]
            base_url=BASE_URL,
            max_retries=0,
            http_async_client=client,
            script=script,  # type: ignore[call-arg]
            **kwargs,
        )

    @property
    def turns(self) -> list[Turn]:
        return self.script.turns

    @turns.setter
    def turns(self, turns: list[Turn]) -> None:
        self.script.turns = turns

    @property
    def requests(self) -> list[dict[str, Any]]:
        return self.script.requests

    @property
    def headers(self) -> list[dict[str, str]]:
        return self.script.headers

    async def complete(self, messages: list[dict[str, Any]], **body: Any) -> dict[str, Any]:
        """The judge's way in: a chat-completions request body, its response object."""
        return self.script.reply({"messages": [dict(m) for m in messages], **body})
