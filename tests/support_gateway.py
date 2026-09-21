"""A real OpenAI-compatible gateway on a real socket.

Bifrost is an HTTP service, so the tests talk HTTP: a threaded server on an ephemeral port,
driven by a script of responses. Nothing about the client is patched — retries, the circuit
breaker and streaming are exercised through actual sockets, which is the only way those three
can be believed.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

#: what a scripted turn may return: a body dict, or an (status, body) pair
Turn = dict[str, Any] | tuple[int, Any]


def completion(text: str, *, tool_calls: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": text}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {
        "id": "chatcmpl-test",
        "model": "gateway-model",
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
    }


class FakeGateway:
    """Serves ``/chat/completions`` and ``/models`` from a script."""

    def __init__(self, script: list[Turn] | Callable[[dict[str, Any]], Turn]) -> None:
        self.script = script
        self.requests: list[dict[str, Any]] = []
        self._calls = 0
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_: Any) -> None:  # keep the test output readable
                return

            def do_GET(self) -> None:
                outer._send(self, 200, {"data": [{"id": "gateway-model"}]})

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                outer.requests.append(body)
                turn = outer._next(body)
                status, payload = turn if isinstance(turn, tuple) else (200, turn)
                if body.get("stream"):
                    outer._send_sse(self, payload)
                else:
                    outer._send(self, status, payload)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    # -- lifecycle ----------------------------------------------------------------
    def __enter__(self) -> FakeGateway:
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)

    @property
    def url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    @property
    def call_count(self) -> int:
        return self._calls

    # -- internals ----------------------------------------------------------------
    def _next(self, body: dict[str, Any]) -> Turn:
        self._calls += 1
        if callable(self.script):
            return self.script(body)
        index = min(self._calls - 1, len(self.script) - 1)
        return self.script[index]

    @staticmethod
    def _send(handler: BaseHTTPRequestHandler, status: int, payload: Any) -> None:
        raw = payload.encode() if isinstance(payload, str) else json.dumps(payload).encode()
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(raw)))
        handler.end_headers()
        handler.wfile.write(raw)

    @staticmethod
    def _send_sse(handler: BaseHTTPRequestHandler, payload: Any) -> None:
        text = payload["choices"][0]["message"]["content"]
        frames = [
            "data: " + json.dumps({"choices": [{"delta": {"content": piece}}]}) + "\n\n"
            for piece in text.split(" ")
        ]
        raw = ("".join(frames) + "data: [DONE]\n\n").encode()
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.send_header("Content-Length", str(len(raw)))
        handler.end_headers()
        handler.wfile.write(raw)
