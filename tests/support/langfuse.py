"""A fake of Langfuse's prompt management API, in the shapes Langfuse answers with
(``GET /api/public/v2/prompts/{name}``, its public API): the unit tests answer through it with
respx (:meth:`LangfusePrompts.handle`), the live test serves it over HTTP on this machine
(:meth:`LangfusePrompts.serve`).

Each prompt is a list of versions (1, 2, ...): a text, or chat messages; labels point at
versions (``production`` at the latest unless moved; ``latest`` always at the latest). A
request without the project's keys (basic auth) is ``401``; a name, version or label it does
not have is ``404`` with Langfuse's ``LangfuseNotFoundError`` body."""

from __future__ import annotations

import base64
import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Final
from urllib.parse import parse_qs, unquote, urlparse

import httpx

HOST: Final = "http://langfuse.test"
PUBLIC: Final = "pk-lf-test"
SECRET: Final = "sk-lf-test"
PATH: Final = "/api/public/v2/prompts/"


@dataclass
class LangfusePrompts:
    #: each prompt's versions, by name: a text, or a list of chat messages
    prompts: dict[str, list[str | list[dict[str, Any]]]] = field(default_factory=dict)
    #: labels moved off the latest version: (name, label) -> version
    labels: dict[tuple[str, str], int] = field(default_factory=dict)
    #: the config of each prompt
    configs: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: every request answered: (name, query)
    asked: list[tuple[str, dict[str, str]]] = field(default_factory=list)

    def answer(
        self, path: str, query: dict[str, str], authorization: str | None
    ) -> tuple[int, dict[str, Any]]:
        token = base64.b64encode(f"{PUBLIC}:{SECRET}".encode()).decode()
        if authorization != f"Basic {token}":
            return 401, {"message": "Invalid credentials", "error": "UnauthorizedError"}
        name = unquote(path.removeprefix(PATH))
        self.asked.append((name, query))
        versions = self.prompts.get(name, [])
        label = query.get("label", "production")
        if "version" in query:
            number = int(query["version"])
        elif label == "latest":
            number = len(versions)
        else:
            number = self.labels.get((name, label), len(versions) if label == "production" else 0)
        if not 1 <= number <= len(versions):
            return 404, {"message": f"Prompt not found: '{name}'", "error": "LangfuseNotFoundError"}
        body = versions[number - 1]
        labels = [label for (n, label), v in self.labels.items() if n == name and v == number]
        return 200, {
            "id": f"lf-{name}-{number}",
            "projectId": "project",
            "name": name,
            "version": number,
            "type": "text" if isinstance(body, str) else "chat",
            "prompt": body,
            "config": self.configs.get(name, {}),
            "labels": labels + (["latest"] if number == len(versions) else []),
            "tags": [],
            "commitMessage": None,
            "createdAt": "2026-10-05T00:00:00.000Z",
            "updatedAt": "2026-10-05T00:00:00.000Z",
            "createdBy": "API",
            "isActive": None,
            "resolutionGraph": None,
        }

    def handle(self, request: httpx.Request) -> httpx.Response:
        """A respx side effect."""
        query = dict(request.url.params)
        status, body = self.answer(request.url.path, query, request.headers.get("authorization"))
        return httpx.Response(status, json=body)

    @contextmanager
    def serve(self) -> Iterator[str]:
        """This fake over HTTP on this machine, for as long as the block runs: its base URL."""
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                url = urlparse(self.path)
                query = {k: v[0] for k, v in parse_qs(url.query).items()}
                status, body = fake.answer(url.path, query, self.headers.get("authorization"))
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, format: str, *args: Any) -> None:
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_address[1]}"
        finally:
            server.shutdown()
            server.server_close()
