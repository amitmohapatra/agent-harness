"""Helpers of the live suite."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections.abc import Awaitable, Callable
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Final
from urllib.parse import parse_qs, urlparse

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
