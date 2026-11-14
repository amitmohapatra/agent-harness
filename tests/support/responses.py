"""A stand-in OpenAI Responses endpoint in background mode, behind the real ``openai`` client
(an ``httpx2.MockTransport``): ``POST /v1/responses`` answers as OpenAI does to a
``background: true`` request, at once, ``queued`` and with no output; any other request (the
``GET /v1/responses/{id}`` a caller polls, a cancel) is not found. Each request is recorded
(:attr:`BackgroundResponses.asked`)."""

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
