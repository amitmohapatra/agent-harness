"""The metering proxy every arm's model calls go through, between the framework and Bifrost:
the one place that counts every framework's calls the same way — whatever its client (the
OpenAI SDK, LangChain, LiteLLM, the Claude Code CLI) — and that holds the sampling the same.

    python -m benchmark.cost.frameworks.meter --upstream http://127.0.0.1:8180 --port 8291

* **Counts** each model call (a ``POST`` to ``…/chat/completions`` or ``…/messages``): the
  input, output and cached tokens its reply reports (a JSON body or a stream of server-sent
  events, OpenAI's or Anthropic's shape), whether it failed, the prompt tokens of a call the
  model server refused as too long, and its seconds. ``GET /__meter`` reads the counts; an arm
  reads them before and after each run (one run at a time) and keeps the difference.
* **Holds the sampling the same** for every framework: the seed of the current run
  (``POST /__meter?seed=N``; OpenAI's shape only — Anthropic's has no seed), one temperature,
  at most :data:`MAX_TOKENS` output tokens, and the usage asked for on an OpenAI stream (the
  only way it reports one). Frameworks send different defaults; only the framework may vary.

Everything else is passed through as it is, both ways.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections.abc import Awaitable, Callable, MutableMapping
from dataclasses import asdict, dataclass, field
from typing import Any, Final
from urllib.parse import parse_qs

import httpx

#: The output tokens a call may ask for at most (the model server's own ``-n``).
MAX_TOKENS: Final = 512
#: The temperature every call is sent with (the model server's own default).
TEMPERATURE: Final = 0.1
#: A call refused for its length, as llama.cpp's server says so (through the gateway).
REFUSED: Final = re.compile(r"request \((\d+) tokens\) exceeds the available context size")
#: The paths that are model calls (``…/messages/count_tokens`` is not).
CALLS: Final = ("/chat/completions", "/messages")
#: Request headers not passed on: the proxy's own address, a length it may change, and
#: compression (the reply is read to count its tokens).
DROPPED: Final = frozenset({"host", "content-length", "accept-encoding", "connection"})
#: Reply headers not passed back (the server sets its own framing).
HOP: Final = frozenset({"content-length", "transfer-encoding", "connection", "content-encoding"})

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]


@dataclass
class Counts:
    """What the model calls made so far reported."""

    calls: int = 0
    failed: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    cache_reported: bool = False
    refused_tokens: int = 0
    #: the calls the model server refused as too long (a context overflow)
    refused_calls: int = 0
    seconds: list[float] = field(default_factory=list)


@dataclass
class Usage:
    """One reply's tokens."""

    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int | None = None

    def add(self, usage: dict[str, Any] | None) -> None:
        """Take one usage object, OpenAI's (``prompt_tokens``) or Anthropic's
        (``input_tokens``); Anthropic's ``message_delta`` carries the output so far."""
        if not isinstance(usage, dict):
            return
        if "prompt_tokens" in usage or "completion_tokens" in usage:
            self.input_tokens = int(usage.get("prompt_tokens") or 0)
            self.output_tokens = int(usage.get("completion_tokens") or 0)
            details = usage.get("prompt_tokens_details") or {}
            if isinstance(details, dict) and "cached_tokens" in details:
                self.cached_tokens = int(details.get("cached_tokens") or 0)
            return
        if usage.get("input_tokens"):
            self.input_tokens = int(usage["input_tokens"])
        if "output_tokens" in usage:
            self.output_tokens = int(usage.get("output_tokens") or 0)
        if "cache_read_input_tokens" in usage:
            self.cached_tokens = int(usage.get("cache_read_input_tokens") or 0)


def usage_of(body: bytes) -> Usage:
    """The tokens a reply reports: a JSON body, or a stream of ``data:`` lines (each event's
    usage taken in turn: the last one of each kind holds)."""
    found = Usage()
    text = body.decode("utf-8", "replace").strip()
    if not text:
        return found
    if text.startswith("{"):
        try:
            data = json.loads(text)
        except ValueError:
            return found
        found.add(data.get("usage") if isinstance(data, dict) else None)
        return found
    for line in text.splitlines():
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload.startswith("{"):
            continue  # [DONE]
        try:
            event = json.loads(payload)
        except ValueError:
            continue
        if not isinstance(event, dict):
            continue
        found.add(event.get("usage"))
        message = event.get("message")
        if isinstance(message, dict):  # Anthropic's message_start
            found.add(message.get("usage"))
    return found


def is_call(method: str, path: str) -> bool:
    return method == "POST" and path.rstrip("/").endswith(CALLS)


def normalized(body: bytes, path: str, seed: int | None) -> bytes:
    """A model call's body with the sampling every arm shares: the run's seed (OpenAI's
    shape), :data:`TEMPERATURE`, at most :data:`MAX_TOKENS` output tokens, and a stream's
    usage asked for. A body that is not a JSON object is passed on as it is."""
    try:
        data = json.loads(body)
    except ValueError:
        return body
    if not isinstance(data, dict):
        return body
    data["temperature"] = TEMPERATURE
    if path.rstrip("/").endswith("/chat/completions"):
        if seed is not None:
            data["seed"] = seed
        for key in ("max_tokens", "max_completion_tokens"):
            if key in data and data[key] is not None:
                data[key] = min(int(data[key]), MAX_TOKENS)
        if data.get("max_tokens") is None and data.get("max_completion_tokens") is None:
            data["max_tokens"] = MAX_TOKENS
        if data.get("stream"):
            options = data.get("stream_options") or {}
            data["stream_options"] = {**options, "include_usage": True}
    else:
        data["max_tokens"] = min(int(data.get("max_tokens") or MAX_TOKENS), MAX_TOKENS)
    return json.dumps(data).encode()


class Meter:
    """The proxy, an ASGI app: ``upstream`` is the gateway's address (no path)."""

    def __init__(self, upstream: str, *, client: httpx.AsyncClient | None = None) -> None:
        self.upstream = upstream.rstrip("/")
        self.counts = Counts()
        self.seed: int | None = None
        self._client = client

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(900.0, connect=10.0))
        return self._client

    def record(self, status: int, body: bytes, seconds: float) -> None:
        counts = self.counts
        counts.calls += 1
        counts.seconds.append(round(seconds, 3))
        if status >= 400:
            counts.failed += 1
            refused = REFUSED.search(body.decode("utf-8", "replace"))
            if refused:
                counts.refused_tokens += int(refused.group(1))
                counts.refused_calls += 1
            return
        usage = usage_of(body)
        counts.input_tokens += usage.input_tokens
        counts.output_tokens += usage.output_tokens
        if usage.cached_tokens is not None:
            counts.cache_reported = True
            counts.cached_tokens += usage.cached_tokens

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "lifespan":
            await self._lifespan(receive, send)
            return
        if scope["type"] != "http":  # pragma: no cover - websockets are not proxied
            return
        body = await _body(receive)
        path, method = scope["path"], scope["method"]
        if path == "/__meter":
            await self._control(scope, send)
            return
        query = scope.get("query_string", b"").decode()
        url = f"{self.upstream}{path}" + (f"?{query}" if query else "")
        headers = [
            (k.decode(), v.decode())
            for k, v in scope["headers"]
            if k.decode().lower() not in DROPPED
        ]
        call = is_call(method, path)
        if call:
            body = normalized(body, path, self.seed)
        began = time.monotonic()
        request = self.client.build_request(method, url, headers=headers, content=body)
        try:
            reply = await self.client.send(request, stream=True)
        except httpx.HTTPError as exc:
            if call:
                self.record(502, str(exc).encode(), time.monotonic() - began)
            await _respond(send, 502, {"error": f"upstream: {exc}"})
            return
        await send(
            {
                "type": "http.response.start",
                "status": reply.status_code,
                "headers": [
                    (k.encode(), v.encode())
                    for k, v in reply.headers.items()
                    if k.lower() not in HOP
                ],
            }
        )
        kept = bytearray()
        try:
            async for chunk in reply.aiter_bytes():
                if call:
                    kept.extend(chunk)
                await send({"type": "http.response.body", "body": chunk, "more_body": True})
        finally:
            await reply.aclose()
            if call:
                self.record(reply.status_code, bytes(kept), time.monotonic() - began)
        await send({"type": "http.response.body", "body": b"", "more_body": False})

    async def _control(self, scope: Scope, send: Send) -> None:
        if scope["method"] == "POST":
            asked = parse_qs(scope.get("query_string", b"").decode())
            seed = (asked.get("seed") or [""])[0]
            self.seed = int(seed) if seed.lstrip("-").isdigit() else None
        await _respond(send, 200, {**asdict(self.counts), "seed": self.seed})

    async def _lifespan(self, receive: Receive, send: Send) -> None:
        while True:
            message = await receive()
            if message["type"] == "lifespan.startup":
                await send({"type": "lifespan.startup.complete"})
            elif message["type"] == "lifespan.shutdown":
                if self._client is not None:
                    await self._client.aclose()
                await send({"type": "lifespan.shutdown.complete"})
                return


async def _body(receive: Receive) -> bytes:
    parts = []
    while True:
        message = await receive()
        parts.append(message.get("body", b""))
        if not message.get("more_body"):
            return b"".join(parts)


async def _respond(send: Send, status: int, data: dict[str, Any]) -> None:
    body = json.dumps(data).encode()
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [(b"content-type", b"application/json")],
        }
    )
    await send({"type": "http.response.body", "body": body})


# --------------------------------------------------------------------------- a client's side
def snapshot(url: str, *, seed: int | None = None) -> dict[str, Any]:
    """The proxy's counts (and, with ``seed``, the next calls' seed set first)."""
    with httpx.Client(timeout=10.0) as client:
        if seed is not None:
            return client.post(f"{url}/__meter", params={"seed": seed}).json()
        return client.get(f"{url}/__meter").json()


def used(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """What the calls between two snapshots used."""
    return {
        "calls": after["calls"] - before["calls"],
        "failed": after["failed"] - before["failed"],
        "input_tokens": after["input_tokens"] - before["input_tokens"],
        "output_tokens": after["output_tokens"] - before["output_tokens"],
        "cached_tokens": after["cached_tokens"] - before["cached_tokens"],
        "cache_reported": bool(after["cache_reported"]),
        "refused_tokens": after["refused_tokens"] - before["refused_tokens"],
        "refused_calls": after["refused_calls"] - before["refused_calls"],
        "seconds": list(after["seconds"][len(before["seconds"]) :]),
    }


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - a server
    import uvicorn

    parser = argparse.ArgumentParser(prog="python -m benchmark.cost.frameworks.meter")
    parser.add_argument("--upstream", required=True, help="the gateway's address, no path")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args(argv)
    uvicorn.run(Meter(args.upstream), host="127.0.0.1", port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
