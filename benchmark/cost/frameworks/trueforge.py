"""TrueForge, TrueFoundry's open-source agent harness (MIT, ``@truefoundry/trueforge`` on npm),
as a native arm: its local mode (one Node process, SQLite, no account) started by the arm's
process, the model a ``custom`` OpenAI-compatible provider on the metering proxy, and the task
set's tools — the same tool code — served to it as a remote MCP server (streamable HTTP) by the
arm's own process, so every call is recorded as a native arm's are.

TrueForge's context saving is **deferred tool loading**: by default (``preload: false``) the
model sees each MCP server's name and description and four meta tools (``list_tools``,
``get_tool_info``, ``get_tool_output_schema``, ``call_tool``) instead of every tool's schema.
``trueforge/A`` runs that default; ``trueforge/A+preload`` loads every schema up front (the
feature off). Code Mode and large-result offloading need a sandbox provider (Daytona, an
account): off here.

The install is ``FWBENCH_TRUEFORGE_DIR`` (an ``npm install @truefoundry/trueforge@<version>``).
"""

from __future__ import annotations

import atexit
import json
import os
import re
import socket
import subprocess
import tempfile
import threading
import time
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any, Final

import httpx

#: Where the install is, and the CLI inside it.
DIR: Final = "FWBENCH_TRUEFORGE_DIR"
CLI: Final = "node_modules/@truefoundry/trueforge/dist/cli.js"
#: The provider and model names the agent spec refers to.
PROVIDER: Final = "bench"
MODEL: Final = "granite"
#: The output tokens a call may ask for (the proxy caps every call the same).
MAX_OUTPUT: Final = 512


def install_problem(env: dict[str, str] | None = None) -> str | None:
    """``None`` when TrueForge is installed where ``FWBENCH_TRUEFORGE_DIR`` says; else why
    not."""
    found = (env if env is not None else os.environ).get(DIR, "")
    if not found:
        return f"no TrueForge install ({DIR}): npm install @truefoundry/trueforge into a folder"
    if not (Path(found) / CLI).exists():
        return f"no TrueForge CLI at {Path(found) / CLI}"
    return None


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def resource_name(text: str) -> str:
    """A TrueForge resource name (lowercase letters, digits and dashes) for ``text``."""
    name = re.sub(r"[^a-z0-9-]+", "-", text.lower()).strip("-")
    return (name if name[:1].isalpha() else f"s-{name}")[:63].rstrip("-") or "server"


# --------------------------------------------------------------------------- the tools as MCP
def mcp_app(tools: Sequence[Any], call: Callable[[str, dict[str, Any]], Awaitable[str]]) -> Any:
    """A streamable-HTTP MCP server (at ``/mcp``) listing ``tools`` with their own schemas and
    descriptions, each call handed to ``call(name, arguments)``."""
    import mcp_types as types
    from mcp.server.lowlevel import Server

    listed = [
        types.Tool(
            name=t.name,
            description=t.spec.description or t.name,
            input_schema=_schema(t),
        )
        for t in tools
    ]

    async def list_tools(ctx: Any, params: Any) -> Any:
        return types.ListToolsResult(tools=listed)

    async def call_tool(ctx: Any, params: Any) -> Any:
        text = await call(params.name, dict(params.arguments or {}))
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=text)],
            is_error=text.startswith("Error:"),
        )

    server = Server("bench-tools", on_list_tools=list_tools, on_call_tool=call_tool)
    return server.streamable_http_app(stateless_http=True, json_response=True)


def _schema(tool: Any) -> dict[str, Any]:
    schema = dict(tool.spec.input_schema or {})
    return {**schema, "type": "object", "properties": schema.get("properties") or {}}


class Served:
    """An ASGI app served by uvicorn in a thread, on a free port."""

    def __init__(self, app: Any) -> None:
        import uvicorn

        self.port = free_port()
        config = uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="error")
        self.server = uvicorn.Server(config)
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.thread.start()
        deadline = time.monotonic() + 20
        while not self.server.started and time.monotonic() < deadline:
            time.sleep(0.05)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/mcp"


# --------------------------------------------------------------------------- the server
class TrueForge:
    """One local-mode TrueForge server for this process (started on first use, stopped at
    exit), and its pid (its CPU and memory count as the arm's)."""

    _current: TrueForge | None = None

    def __init__(self, install: Path, *, port: int | None = None) -> None:
        self.port = port or free_port()
        self.data = Path(tempfile.mkdtemp(prefix="fwbench-trueforge-"))
        env = {
            **os.environ,
            "PORT": str(self.port),
            "SQLITE_PATH": str(self.data / "db.sqlite"),
            # the model (the proxy) and the tools are on this machine
            "OUTBOUND_URL_ALLOWED_HOSTS": json.dumps(["127.0.0.1", "localhost"]),
            "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost",
        }
        self.log = (self.data / "server.log").open("w")
        self.process = subprocess.Popen(
            ["node", str(install / CLI)], cwd=self.data, env=env, stdout=self.log, stderr=self.log
        )
        self.url = f"http://127.0.0.1:{self.port}"
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            try:
                if httpx.get(f"{self.url}/api/v1/capabilities", timeout=2).status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.5)
        else:
            self.stop()
            raise RuntimeError(f"TrueForge did not start: see {self.data / 'server.log'}")
        atexit.register(self.stop)

    @classmethod
    def current(cls) -> TrueForge:
        if cls._current is None:
            problem = install_problem()
            if problem:
                raise RuntimeError(problem)
            cls._current = cls(Path(os.environ[DIR]))
        return cls._current

    @property
    def pid(self) -> int:
        return self.process.pid

    def stop(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.process.kill()

    def put(self, path: str, manifest: dict[str, Any]) -> None:
        reply = httpx.put(f"{self.url}{path}", json={"manifest": manifest}, timeout=30)
        if reply.status_code >= 400:
            raise RuntimeError(f"TrueForge {path}: {reply.status_code} {reply.text[:300]}")

    def provider(self, base_url: str, key: str, model: str, window: int) -> str:
        """The proxy as a ``custom`` provider; the agent spec's model name."""
        self.put(
            "/api/v1/settings/model-providers",
            {
                "type": "custom",
                "name": PROVIDER,
                "base_url": base_url,
                "auth": {"api_key": key or "none"},
                "models": [
                    {
                        "name": MODEL,
                        "model_id": model,
                        "properties": {"context_length": window, "max_output_tokens": MAX_OUTPUT},
                    }
                ],
            },
        )
        return f"{PROVIDER}/{MODEL}"

    def mcp_server(self, name: str, url: str, description: str) -> str:
        self.put(
            "/api/v1/settings/mcp-servers",
            {"type": "remote", "name": name, "url": url, "description": description},
        )
        return name


def spec(model: str, system: str, server: str, *, preload: bool) -> dict[str, Any]:
    """The agent: TrueForge's defaults, with no person in the loop (no approval gate, no
    clarifying question — as every native arm runs its tools) and no sandbox (none is
    configured: that takes a Daytona account)."""
    return {
        "model": {"name": model},
        "instructions": system,
        "mcp_servers": [
            {
                "name": server,
                "enable_tools": ["@all"],
                "require_approval_for_tools": [],
                "preload": preload,
            }
        ],
        "config": {"ask_user_questions": {"enabled": False}, "sandbox": {"enabled": False}},
    }


def text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b.get("text", "") for b in content if isinstance(b, dict))
    return ""


async def turn(base: str, agent: dict[str, Any], task: str, *, seconds: float) -> str:
    """One session, one turn: TrueForge's answer (its ``turn.done`` output, else the last
    model message). A turn that ends in an error raises it."""
    timeout = httpx.Timeout(seconds, connect=10.0)
    async with httpx.AsyncClient(base_url=base, timeout=timeout) as client:
        made = await client.post("/api/v1/sessions", json={"agent": {"spec": agent}})
        made.raise_for_status()
        session = made.json()["data"]["id"]
        body = {
            "input": [{"type": "user.message", "content": task}],
            "previous_turn_id": "none",
            "stream": True,
        }
        last, answer, state = "", "", None
        headers = {"Accept": "text/event-stream"}
        async with client.stream(
            "POST", f"/api/v1/sessions/{session}/turns", json=body, headers=headers
        ) as reply:
            reply.raise_for_status()
            async for line in reply.aiter_lines():
                if not line.startswith("data:"):
                    continue
                try:
                    event = json.loads(line[5:].strip())
                except ValueError:
                    continue
                kind = event.get("type", "")
                if kind == "model.message":
                    said = text_of(event.get("content"))
                    last = said if said.strip() else last
                elif kind == "turn.done":
                    state = event.get("state") or {}
                    output = state.get("output") or {}
                    answer = text_of(output.get("content") if isinstance(output, dict) else None)
                    break
        if state is not None and state.get("status") == "error":
            raise RuntimeError(f"TrueForge turn failed: {json.dumps(state.get('error'))[:300]}")
        return answer or last
