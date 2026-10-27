"""Harness tools as one MCP server (the official SDK's low-level ``Server``), for a framework
that reaches tools only over MCP: an ACP agent (``adapters.acp``), which is handed MCP servers
when its session starts.

The adapter serves it for one attempt (:func:`served`: streamable HTTP on the loopback
interface, every request carrying a bearer token of the attempt's own), inside the attempt's
task, so each call runs as the run's (``trellis.current()``). Every call goes to the bridge, as
every framework's does; a call that pauses the run tells the agent it waits (:data:`WAITING`).
A result longer than ``tools.results.MAX_CHARS`` is kept in the run and the model reads a
preview naming where; the server's ``read_file`` pages it (listed from the start: an agent
lists a server's tools once, when its session starts).
"""

from __future__ import annotations

import asyncio
import contextlib
import secrets
from collections.abc import AsyncIterator, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Final

import mcp_types as types
from mcp.server.lowlevel import Server

from trellis.harness.runtime import Paused
from trellis.harness.tools import bridge, results
from trellis.harness.tools.base import Tool

#: The server's name, and so the prefix an agent names its tools with (``mcp__trellis__``).
SERVER: Final = "trellis"
#: What the agent is told about a call that paused the run.
WAITING: Final = "Waiting for a person's approval."
#: Where the server answers, under its origin.
PATH: Final = "/mcp"
#: How long the server waits for its socket at the start of an attempt, at most.
START_SECONDS: Final = 10.0
#: How long, at the attempt's end, a call still under way has before it is cancelled.
STOP_SECONDS: Final = 0.1


def convert(tools: Sequence[Tool]) -> Server[Any]:
    by_name = {t.name: t for t in tools}

    async def list_tools(ctx: Any, params: Any) -> types.ListToolsResult:
        listed = [_listed(t) for t in tools]
        listed.append(
            types.Tool(
                name=results.READ_FILE,
                description=results.READ_FILE_DESCRIPTION,
                input_schema=results.READ_FILE_SCHEMA,
            )
        )
        return types.ListToolsResult(tools=listed)

    async def call_tool(ctx: Any, params: types.CallToolRequestParams) -> types.CallToolResult:
        args = dict(params.arguments or {})
        if params.name == results.READ_FILE:
            return _result(results.read(args), error=False)
        tool = by_name.get(params.name)
        if tool is None:
            return _result(f"no tool named {params.name!r}", error=True)
        try:
            outcome = await bridge.call(tool, args)
        except Paused:  # the run stops here (the adapter reads the pause from the runtime)
            return _result(WAITING, error=True)
        return _result(results.readable(outcome.output), error=not outcome.ok)

    return Server(SERVER, on_list_tools=list_tools, on_call_tool=call_tool)


def _listed(tool: Tool) -> types.Tool:
    schema = tool.spec.input_schema or {}
    return types.Tool(
        name=tool.name,
        description=tool.spec.description or tool.name,
        input_schema={**schema, "type": "object", "properties": schema.get("properties") or {}},
    )


def _result(text: str, *, error: bool) -> types.CallToolResult:
    return types.CallToolResult(content=[types.TextContent(text=text)], is_error=error)


@dataclass(frozen=True, slots=True)
class Served:
    """Where an attempt's server answers, and the header every request must carry."""

    url: str
    token: str

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"}


def _quiet_server() -> type:
    import uvicorn

    class Quiet(uvicorn.Server):
        """A server that leaves the process's signal handlers alone (``uvicorn.Server`` takes
        SIGINT and SIGTERM while it serves): the application's — a worker's graceful stop —
        stay its own."""

        @contextlib.contextmanager
        def capture_signals(self) -> Iterator[None]:
            yield

    return Quiet


@contextlib.asynccontextmanager
async def served(server: Server[Any]) -> AsyncIterator[Served]:
    """``server`` over streamable HTTP on ``127.0.0.1`` (a port the system picks) for as long as
    the block runs, in this task's context: its calls run in the run that serves it. Each
    request needs the bearer token made for it (another process on the machine has none)."""
    import uvicorn
    from starlette.responses import Response

    _Server = _quiet_server()

    token = secrets.token_urlsafe(24)
    app = server.streamable_http_app(
        streamable_http_path=PATH, stateless_http=True, json_response=True
    )
    expected = f"Bearer {token}".encode()

    async def guarded(scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] == "http" and dict(scope["headers"]).get(b"authorization") != expected:
            await Response(status_code=401)(scope, receive, send)
            return
        await app(scope, receive, send)

    config = uvicorn.Config(
        guarded,
        host="127.0.0.1",
        port=0,
        log_level="warning",
        lifespan="on",
        # the attempt is over: a call still under way is cut short now (the run's end says
        # why), not waited for
        timeout_graceful_shutdown=STOP_SECONDS,
    )
    web = _Server(config)
    task = asyncio.create_task(web.serve())
    try:
        async with asyncio.timeout(START_SECONDS):
            while not web.started:  # noqa: ASYNC110 - uvicorn says so with a flag only
                await asyncio.sleep(0.01)
        port = web.servers[0].sockets[0].getsockname()[1]
        yield Served(f"http://127.0.0.1:{port}{PATH}", token)
    finally:
        web.should_exit = True
        with contextlib.suppress(asyncio.CancelledError):
            await task


__all__ = ["PATH", "SERVER", "WAITING", "Served", "convert", "served"]
