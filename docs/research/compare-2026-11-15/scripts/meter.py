"""A recording reverse proxy in front of api.anthropic.com that only forwards and records:
each response's usage, as the provider sent it, and how long the provider took, appended to a
JSONL file (path, model, status, usage, seconds; never a header or a body). It renames no
model and answers nothing itself: it is not a gateway.

Usage: python meter.py <port> <log.jsonl>
Point a client at http://127.0.0.1:<port> (Anthropic SDK / CLI: ANTHROPIC_BASE_URL; the
OpenAI-compatible route: LLM_BASE_URL=http://127.0.0.1:<port>/v1/).
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse
from starlette.routing import Route

UPSTREAM = "https://api.anthropic.com"
LOG = sys.argv[2]
DROP = {"host", "content-length", "accept-encoding", "connection"}
client = httpx.AsyncClient(base_url=UPSTREAM, timeout=900)


def record(path: str, status: int, model: Any, usage: Any, *, stream: bool, began: float) -> None:
    now = time.time()
    with open(LOG, "a") as fh:
        fh.write(
            json.dumps(
                {
                    "t": round(now, 3),
                    "seconds": round(now - began, 3),
                    "path": path,
                    "status": status,
                    "model": model,
                    "stream": stream,
                    "usage": usage,
                }
            )
            + "\n"
        )


def merged(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        if v is not None:
            out[k] = v
    return out


async def forward(request: Request) -> Response:
    began = time.time()
    body = await request.body()
    headers = {k: v for k, v in request.headers.items() if k.lower() not in DROP}
    path = request.url.path + (f"?{request.url.query}" if request.url.query else "")
    try:
        sent = json.loads(body) if body else {}
    except ValueError:
        sent = {}
    stream = bool(sent.get("stream")) if isinstance(sent, dict) else False
    upstream = client.build_request(request.method, path, headers=headers, content=body)
    reply = await client.send(upstream, stream=True)
    out_headers = {
        k: v
        for k, v in reply.headers.items()
        if k.lower() not in ("content-length", "content-encoding", "transfer-encoding")
    }
    if not stream:
        data = await reply.aread()
        await reply.aclose()
        try:
            parsed = json.loads(data)
            record(
                request.url.path, reply.status_code, parsed.get("model"), parsed.get("usage"),
                stream=False, began=began,
            )  # fmt: skip
        except (ValueError, AttributeError):
            record(request.url.path, reply.status_code, None, None, stream=False, began=began)
        return Response(data, status_code=reply.status_code, headers=out_headers)

    async def relay() -> AsyncIterator[bytes]:
        usage: dict[str, Any] = {}
        model = None
        buffer = b""
        try:
            async for chunk in reply.aiter_bytes():
                yield chunk
                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    if not line.startswith(b"data:"):
                        continue
                    try:
                        event = json.loads(line[5:].strip() or b"{}")
                    except ValueError:
                        continue
                    if not isinstance(event, dict):
                        continue
                    if event.get("type") == "message_start":  # Anthropic messages
                        message = event.get("message") or {}
                        model = message.get("model")
                        usage = merged(usage, message.get("usage") or {})
                    elif event.get("type") == "message_delta":
                        usage = merged(usage, event.get("usage") or {})
                    elif event.get("usage"):  # OpenAI-compatible chunks
                        model = event.get("model") or model
                        usage = merged(usage, event["usage"])
        finally:
            await reply.aclose()
            record(
                request.url.path, reply.status_code, model, usage or None, stream=True, began=began
            )

    return StreamingResponse(relay(), status_code=reply.status_code, headers=out_headers)


app = Starlette(routes=[Route("/{path:path}", forward, methods=["GET", "POST", "PUT", "DELETE"])])

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(sys.argv[1]), log_level="warning")
