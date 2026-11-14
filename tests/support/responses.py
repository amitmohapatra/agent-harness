"""Stand-in OpenAI Responses endpoints behind the real ``openai`` client.

* :class:`BackgroundResponses` (an ``httpx2.MockTransport``): ``POST /v1/responses`` answers as
  OpenAI does to a ``background: true`` request, at once, ``queued`` and with no output; any
  other request (the ``GET /v1/responses/{id}`` a caller polls, a cancel) is not found. Each
  request is recorded (:attr:`BackgroundResponses.asked`).
* :class:`ScriptedResponses`: a gateway's ``POST /v1/responses`` answering scripted turns
  (``tests.support.gateway.FakeGateway`` routes it there)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx2

#: A response OpenAI has queued to run in the background: no output yet.
QUEUED: dict[str, Any] = {
    "id": "resp_bg_1",
    "object": "response",
    "created_at": 0,
    "model": "gpt-test",
    "status": "queued",
    "background": True,
    "output": [],
    "parallel_tool_calls": True,
    "tool_choice": "auto",
    "tools": [],
}


@dataclass
class BackgroundResponses:
    asked: list[tuple[str, str]] = field(default_factory=list)  #: (method, path) per request

    def client(self) -> Any:
        import openai

        return openai.AsyncOpenAI(
            base_url="https://responses.test/v1",
            api_key="not-a-key",
            http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(self.handle)),
            max_retries=0,
        )

    def handle(self, request: httpx2.Request) -> httpx2.Response:
        self.asked.append((request.method, request.url.path))
        if request.method == "POST" and request.url.path == "/v1/responses":
            return httpx2.Response(200, json=QUEUED)
        return httpx2.Response(404, json={"error": {"message": "not served"}})


@dataclass
class ScriptedResponses:
    """A stand-in for a gateway's ``POST /v1/responses`` (Bifrost serves it for every provider
    it routes): each request recorded (:attr:`requests`, its body) and answered, not streamed
    or streamed as server-sent events, with the next turn — text, ``(tool, args)`` for a
    function call, or a list of calls — in the shape the Responses API answers with."""

    turns: list[Any] = field(default_factory=list)
    requests: list[dict[str, Any]] = field(default_factory=list)
    #: each request's headers
    headers: list[dict[str, str]] = field(default_factory=list)

    def handle(self, request: Any) -> Any:
        import json

        import httpx

        body = json.loads(request.content)
        self.requests.append(body)
        self.headers.append(dict(request.headers))
        reply = self.reply(self.turns.pop(0))
        if not body.get("stream"):
            return httpx.Response(200, json=reply)
        events = [
            {"type": "response.created", "sequence_number": 0, "response": {**reply, "output": []}},
            *(
                {
                    "type": "response.output_text.delta",
                    "sequence_number": 1,
                    "item_id": item["id"],
                    "output_index": n,
                    "content_index": 0,
                    "delta": item["content"][0]["text"],
                    "logprobs": [],
                }
                for n, item in enumerate(reply["output"])
                if item["type"] == "message"
            ),
            {"type": "response.completed", "sequence_number": 2, "response": reply},
        ]
        said = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=said.encode()
        )

    def reply(self, turn: Any) -> dict[str, Any]:
        import json

        n = len(self.requests)
        if isinstance(turn, str):
            output: list[dict[str, Any]] = [
                {
                    "id": f"msg_{n}",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": turn, "annotations": []}],
                }
            ]
        else:
            calls = turn if isinstance(turn, list) else [turn]
            output = [
                {
                    "id": f"fc_{n}_{i}",
                    "type": "function_call",
                    "call_id": f"call_{n}_{i}",
                    "name": name,
                    "arguments": json.dumps(args),
                    "status": "completed",
                }
                for i, (name, args) in enumerate(calls)
            ]
        return {
            "id": f"resp_{n}",
            "object": "response",
            "created_at": 0,
            "model": "scripted",
            "status": "completed",
            "output": output,
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
            },
        }
