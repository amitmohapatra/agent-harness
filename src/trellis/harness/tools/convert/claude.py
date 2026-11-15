"""Harness tools as one in-process MCP server for the Claude Agent SDK.

Claude names them ``mcp__trellis__<tool>``; the harness's ``can_use_tool`` lets them through
(the bridge is their permission check). A call that pauses the run tells the CLI it waits
(:data:`WAITING`): the run stops there, and the resumed session calls the tool again. A call
that the process's own end stops (a ``BaseException`` that is no cancellation: ``SystemExit``,
``KeyboardInterrupt``) ends the attempt with it (``Runtime.stopped_by``), as it ends any
framework's: the SDK runs the call in its tool server's task group, which it would otherwise
end for the rest of the process's life while the CLI went on, every later call failing.

A result longer than ``tools.results.MAX_CHARS`` is kept in the run and the model reads a
preview naming where; the server's ``read_file`` pages it. The CLI lists a server's tools once,
when its session starts (the SDK carries no ``list_changed``), so ``read_file`` is listed from
the start: before a result was kept it finds nothing to read.

The server's config says ``alwaysLoad``: Claude Code otherwise defers every MCP server's tools
behind its ``ToolSearch`` (the model is told their names only, and spends a step loading them
before it can call one). Which tools a run offers is the harness's choice (the tool hints'
narrowing, memory's ``tool_search``), made before the query: they are the tools to load. A
team's own servers keep the CLI's deferral.

A CLI process that serves several turns of a conversation (``bridges.claude_live``) lists the
server once, for all of them, while each turn has its own tools (bound to its run): that
process is given :func:`relayed`, a server that lists what the first turn's lists and hands
each call to the turn being served (:func:`handler`: that turn's own tool of the name).
"""

from __future__ import annotations

import asyncio
import json
import weakref
from collections.abc import Awaitable, Callable, Sequence
from contextvars import ContextVar
from typing import Any, Final, cast

from claude_agent_sdk import McpSdkServerConfig, create_sdk_mcp_server
from claude_agent_sdk import tool as sdk_tool

from trellis.harness.runtime import Paused, current
from trellis.harness.tools import bridge, results
from trellis.harness.tools.base import Tool

#: The in-process server's name, and so the prefix Claude sees.
SERVER: Final = "trellis"
#: How Claude names a tool of that server: this before the tool's own name.
PREFIX: Final = f"mcp__{SERVER}__"
#: The server config's key (the CLI's ``--mcp-config``, which the SDK passes on with all but the
#: server's instance) that loads its tools from the first step instead of deferring them.
ALWAYS_LOAD: Final = "alwaysLoad"
#: What the CLI is told about a call that paused the run (a harness tool's, or a built-in's
#: the permission check asked a person about).
WAITING: Final = "Waiting for a person's approval."
#: What the CLI is told about a call the process's own end stopped (the attempt ends with it).
STOPPED: Final = "The worker running this call stopped before it finished; it did not complete."


#: The key of a ``tools/call`` request's ``_meta`` under which Claude Code names the tool call
#: it makes (its ``tool_use_id``, the one its hooks and permission requests carry).
CALL_META: Final = "claudecode/toolUseId"
#: The tool call the request being served is (:func:`named_call`); ``None``: not said.
_CALL: ContextVar[str | None] = ContextVar("claude_tool_call", default=None)


async def named_call(ctx: Any, call_next: Callable[[Any], Awaitable[Any]]) -> Any:
    """The MCP server middleware of a server made here (``mcp.server.Server.middleware``): the
    request is served knowing which of Claude's tool calls it is (:func:`call_id`), which
    the SDK's tool handlers are not given."""
    token = _CALL.set((ctx.meta or {}).get(CALL_META))
    try:
        return await call_next(ctx)
    finally:
        _CALL.reset(token)


def named(tool: str) -> str:
    """The harness tool ``tool`` as Claude names it."""
    return f"{PREFIX}{tool}"


def bare(name: str) -> str | None:
    """The harness tool's own name, of Claude's name for a tool; ``None``: not a harness
    tool (a built-in, another server's)."""
    return name.removeprefix(PREFIX) if name.startswith(PREFIX) else None


def call_id() -> str | None:
    """Claude's id of the tool call a handler is serving now (:func:`named_call`)."""
    return _CALL.get()


#: A tool's handler, as the SDK calls it: its arguments in, an MCP tool result out.
Handler = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]
#: The SDK tools each server made here serves, by the server's instance (the config itself
#: goes to the CLI as JSON, less the instance: it carries nothing else).
_SERVED: weakref.WeakKeyDictionary[Any, list[Any]] = weakref.WeakKeyDictionary()


def convert(tools: Sequence[Tool]) -> McpSdkServerConfig:
    readers = [_reader(*r) for r in results.readers()]
    return _server([*(_one(t) for t in tools), *readers])


def _server(served: list[Any]) -> McpSdkServerConfig:
    server = create_sdk_mcp_server(SERVER, tools=served)
    server["instance"].middleware.append(named_call)
    _SERVED[server["instance"]] = served
    return cast("McpSdkServerConfig", {**server, ALWAYS_LOAD: True})


def listing(server: McpSdkServerConfig) -> tuple[tuple[str, str], ...]:
    """What ``server`` (one :func:`convert` made) offers: each tool's name and input schema
    (as JSON) — what a CLI process that lists it once must have for good to serve a later
    turn. Not the descriptions: a tool's wording may change from one turn to the next (the
    hour ``schedule_task`` says it is), and the process keeps the wording it listed."""
    return tuple(
        (t.name, json.dumps(t.input_schema, sort_keys=True, default=str))
        for t in _SERVED[server["instance"]]
    )


def handler(server: McpSdkServerConfig, name: str) -> Handler:
    """The handler of ``server``'s tool ``name``."""
    [found] = [t.handler for t in _SERVED[server["instance"]] if t.name == name]
    return cast("Handler", found)


def lists(server: McpSdkServerConfig | None, name: str) -> bool:
    """Whether ``server`` (none: no harness tool) lists the tool ``name``."""
    return server is not None and any(t.name == name for t in _SERVED[server["instance"]])


def tool_handler(tool: Tool) -> Handler:
    """The handler a server :func:`convert` makes gives ``tool``, made alone."""
    return cast("Handler", _one(tool).handler)


def relayed(
    server: McpSdkServerConfig, call: Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]
) -> McpSdkServerConfig:
    """A server that lists what ``server`` lists and hands each call to ``call`` (the tool's
    name, its arguments): the one a CLI process serving several turns is given."""

    def relay(tool: Any) -> Any:
        async def relayed_call(args: dict[str, Any]) -> dict[str, Any]:
            return await call(tool.name, args)

        return sdk_tool(tool.name, tool.description, tool.input_schema)(relayed_call)

    return _server([relay(t) for t in _SERVED[server["instance"]]])


def _one(tool: Tool) -> Any:
    async def handler(args: dict[str, Any]) -> dict[str, Any]:
        try:
            outcome = await bridge.call(tool, dict(args))
        except Paused:  # the run stops here (the harness reads the pause from the runtime)
            return {"content": [{"type": "text", "text": WAITING}], "is_error": True}
        except BaseException as exc:
            runtime = current()
            if isinstance(exc, Exception | asyncio.CancelledError) or runtime is None:
                raise
            runtime.stopped_by = exc  # the attempt ends with it (``ClaudeAdapter``)
            return {"content": [{"type": "text", "text": STOPPED}], "is_error": True}
        content = [{"type": "text", "text": results.readable(outcome.output, tool.name)}]
        return {"content": content, "is_error": not outcome.ok}

    return sdk_tool(tool.name, tool.spec.description or tool.name, _schema(tool))(handler)


def _reader(
    name: str,
    description: str,
    schema: dict[str, Any],
    answer: Callable[[dict[str, Any]], Awaitable[str]],
) -> Any:
    """A tool reading a large result the run keeps (``tools.results.readers``: ``read_file``
    its pages, ``query_result`` SQL over its rows — saving them as a file in a run with no tool
    that runs code)."""

    async def handler(args: dict[str, Any]) -> dict[str, Any]:
        return {"content": [{"type": "text", "text": await answer(dict(args))}]}

    return sdk_tool(name, description, schema)(handler)


def _schema(tool: Tool) -> dict[str, Any]:
    """The tool's input schema as the SDK takes a JSON schema: with a ``type`` and
    ``properties`` — any other dict it reads as a map of argument names to Python types, and
    an MCP tool without arguments declares only ``{"type": "object"}``."""
    schema = tool.spec.input_schema or {}
    return {**schema, "type": "object", "properties": schema.get("properties") or {}}
