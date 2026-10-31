"""Harness tools as one MCP server (the official SDK's low-level ``Server``), for a framework
that reaches tools only over MCP: an ACP agent (``adapters.acp``), which is handed MCP servers
when its session starts.

The adapter serves it for one attempt (:func:`served`: streamable HTTP on the loopback
interface, every request carrying a bearer token of the attempt's own), inside the attempt's
task, so each call runs as the run's (``trellis.current()``). Every call goes to the bridge, as
every framework's does; a call that pauses the run tells the agent it waits (:data:`WAITING`).
A result longer than ``tools.results.MAX_CHARS`` is kept in the run and the model reads a
preview naming where; the server's ``read_file`` pages it (listed from the start: an agent
lists a server's tools once, when its session starts). A call still under way when the attempt
ends (the run cancelled, out of time, or paused at another call) is cancelled there, as an
in-process tool is: it never runs on past the run's end.
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
#: What the agent is told about a call the attempt's end cut short.
STOPPED: Final = "The run ended before this tool finished."
#: How long, at the attempt's end, the server has to answer what it still owes (its calls are
#: cancelled first).
STOP_SECONDS: Final = 1


class Tools(Server[Any]):
    """The harness's tools as an MCP server, and the calls it has under way: each runs in a
    task of its own, which the attempt's end cancels (:func:`served`)."""

    def __init__(self, calls: set[asyncio.Task[Any]], **handlers: Any) -> None:
        super().__init__(SERVER, **handlers)
        self.calls = calls

    async def stopped(self) -> None:
        """Every call still under way cancelled, and its end waited for."""
        under_way = list(self.calls)
        for call in under_way:
            call.cancel()
        await asyncio.gather(*under_way, return_exceptions=True)


def convert(tools: Sequence[Tool]) -> Tools:
    by_name = {t.name: t for t in tools}
    calls: set[asyncio.Task[Any]] = set()
    readers = results.readers()  # the run's: converted in it

    async def list_tools(ctx: Any, params: Any) -> types.ListToolsResult:
        listed = [_listed(t) for t in tools]
        listed.extend(
            types.Tool(name=name, description=description, input_schema=schema)
            for name, description, schema, _ in readers
        )
        return types.ListToolsResult(tools=listed)

    async def call_tool(ctx: Any, params: types.CallToolRequestParams) -> types.CallToolResult:
        args = dict(params.arguments or {})
        reader = next((r for r in readers if r[0] == params.name), None)
        if reader is not None:
            return _result(await reader[3](args), error=False)
        tool = by_name.get(params.name)
        if tool is None:
            return _result(f"no tool named {params.name!r}", error=True)
        # a task of its own (in this context: the run's), which the attempt's end can cancel
        call = asyncio.ensure_future(bridge.call(tool, args))
        calls.add(call)
        call.add_done_callback(calls.discard)
        try:
            await asyncio.wait({call})
        finally:
            call.cancel()  # this request cancelled: its call with it (nothing, once it ended)
        if call.cancelled():
            return _result(STOPPED, error=True)
        try:
            outcome = call.result()
        except Paused:  # the run stops here (the adapter reads the pause from the runtime)
            return _result(WAITING, error=True)
        return _result(results.readable(outcome.output, tool.name), error=not outcome.ok)

    return Tools(calls, on_list_tools=list_tools, on_call_tool=call_tool)


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
async def served(server: Tools) -> AsyncIterator[Served]:
    """``server`` over streamable HTTP on ``127.0.0.1`` (a port the system picks) for as long as
    the block runs, in this task's context: its calls run in the run that serves it. Each
    request needs the bearer token made for it (another process on the machine has none).
    When the block ends, its calls still under way are cancelled before the server stops."""
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
        await server.stopped()
        web.should_exit = True
        with contextlib.suppress(asyncio.CancelledError):
            await task


__all__ = ["PATH", "SERVER", "STOPPED", "WAITING", "Served", "Tools", "convert", "served"]
