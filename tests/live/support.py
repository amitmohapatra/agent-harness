"""Helpers of the live suite."""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Final
from urllib.parse import parse_qs, urlparse

import httpx
import uvicorn
from langchain.agents.middleware import AgentMiddleware

from trellis import Harness
from trellis.memory import MemoryContext

#: How long a background effect of the services (a projection, a learning job) may take.
SETTLE_SECONDS: Final = 60.0


async def eventually(
    check: Callable[[], Awaitable[bool]], *, within: float = SETTLE_SECONDS, every: float = 1.0
) -> bool:
    """Whether ``check`` holds within ``within`` seconds."""
    deadline = time.monotonic() + within
    while True:
        if await check():
            return True
        if time.monotonic() >= deadline:
            return False
        await asyncio.sleep(every)


async def seed(scope: MemoryContext, content: str, **options: Any) -> None:
    """Remember ``content`` for a test and wait until it is findable. A memory is stored at
    once and searchable when its index job lands (a fraction of a second; longer on a busy
    worker): a run started in between is pushed a context without it, which is not what a test
    that seeds a memory is about."""
    await scope.remember(content, **options)

    async def found() -> bool:
        return any(item.text == content for item in await scope.search(content))

    assert await eventually(found, every=0.2), f"{content!r} never became findable"


async def memory_scope(
    h: Harness, *, user: str, agent_id: str, thread: str | None = None
) -> MemoryContext:
    """The memory service in a user's (and thread's) scope, read as the agent reads it."""
    assert h.memory is not None
    scope = {"tenant_id": await h.tenant(), "user_id": user, "agent_id": agent_id}
    if thread is not None:
        scope["thread_id"] = thread
    return h.memory.client.bind(**scope)


class StubLangfuse:
    """A local Langfuse: one dataset, and every POST recorded (a dataset-run link is answered
    with the dataset run :data:`DATASET_RUN`, as Langfuse answers it).

        with StubLangfuse("golden", items) as langfuse:
            h = live_harness(**langfuse.settings())
    """

    DATASET_RUN: Final = "dsr-live"

    def __init__(self, dataset: str = "none", items: list[dict[str, Any]] | None = None) -> None:
        items = items or []
        self.received: list[tuple[str, Any]] = []
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                url = urlparse(self.path)
                if url.path == f"/api/public/v2/datasets/{dataset}":
                    self._answer({"id": "ds-live", "name": dataset})
                elif url.path == "/api/public/dataset-items":
                    assert parse_qs(url.query)["datasetName"] == [dataset]
                    meta = {"page": 1, "limit": 50, "totalItems": len(items), "totalPages": 1}
                    self._answer({"data": items, "meta": meta})
                else:
                    self._answer({"message": "not found"}, status=404)

            def do_POST(self) -> None:
                body = self.rfile.read(int(self.headers.get("content-length", 0)))
                stub.received.append((self.path, json.loads(body)))
                if self.path == "/api/public/dataset-run-items":
                    self._answer({"id": "dri-live", "datasetRunId": stub.DATASET_RUN})
                else:
                    self._answer({"id": "ok"})

            def _answer(self, body: Any, status: int = 200) -> None:
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(body).encode())

            def log_message(self, format: str, *args: Any) -> None:
                return None

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def __enter__(self) -> StubLangfuse:
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.server.shutdown()
        self.server.server_close()

    def posted(self, path: str) -> list[Any]:
        return [body for p, body in self.received if p == path]

    def settings(self) -> dict[str, Any]:
        """Langfuse reached through the OTLP headers (no exporter: spans stay in process)."""
        headers = {"authorization": "Basic cGs6c2s=", "x-langfuse-host": self.url}
        return {"otlp_endpoint": None, "otlp_headers": headers}

    def environ(self) -> dict[str, str]:
        """The same, as the environment a deployment sets (``EvalServices.from_env``)."""
        headers = f"authorization=Basic%20cGs6c2s%3D,x-langfuse-host={self.url}"
        return {"OTEL_EXPORTER_OTLP_HEADERS": headers}


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@contextmanager
def serving(app: Any, port: int) -> Iterator[None]:
    """``app`` served by uvicorn on ``port`` (a real socket) in a thread, until the block ends."""
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        threading.Event().wait(0.05)
    try:
        yield
    finally:
        server.should_exit = True
        thread.join(timeout=10)


class Sent:
    """What the live model was sent: each chat request's body and headers, as the gateway got
    them (an ``httpx`` request hook on the model client)."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.headers: list[dict[str, str]] = []

    async def __call__(self, request: httpx.Request) -> None:
        if request.url.path.endswith("/chat/completions"):
            self.requests.append(json.loads(request.content))
            self.headers.append(dict(request.headers))


def gateway_model(h: Harness, sent: Sent, *, wait: float = 600.0, **kwargs: Any) -> Any:
    """The live model as a ``ReAct`` builds it for a model name — the gateway, the virtual
    key, the deny-all MCP scope — with ``sent`` recording what it is sent, short answers, and
    as long a wait as the shared local model takes."""
    from bifrost_sdk import NO_GATEWAY_TOOLS
    from langchain_openai import ChatOpenAI

    from tests.live.conftest import BIFROST_URL, MODEL

    client = httpx.AsyncClient(timeout=wait, event_hooks={"request": [sent]})
    return ChatOpenAI(
        model=MODEL,
        base_url=BIFROST_URL,
        api_key=h.settings.bifrost_virtual_key or "none",  # type: ignore[arg-type]
        default_headers=dict(NO_GATEWAY_TOOLS),
        max_tokens=160,  # type: ignore[call-arg]
        http_async_client=client,
        **kwargs,
    )


class Forcing(AgentMiddleware):
    """The model's tool choice forced at each step (``None``: answer, no tool): a small local
    model is slow and unreliable at choosing tools, so a test that needs a particular call
    picks it — the model still writes the call, the harness still runs it."""

    def __init__(self, choices: list[str | None]) -> None:
        super().__init__()
        self.choices = list(choices)

    async def awrap_model_call(self, request: Any, handler: Any) -> Any:
        choice = self.choices.pop(0) if self.choices else None
        forced = {"type": "function", "function": {"name": choice}} if choice else "none"
        return await handler(request.override(tool_choice=forced))
