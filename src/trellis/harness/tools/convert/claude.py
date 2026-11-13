"""Harness tools as one in-process MCP server for the Claude Agent SDK.

Claude names them ``mcp__trellis__<tool>``; the harness's ``can_use_tool`` lets them through
(the bridge is their permission check). A call that pauses the run tells the CLI it waits
(:data:`WAITING`): the run stops there, and the resumed session calls the tool again.

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

import json
import weakref
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, Final, cast

from claude_agent_sdk import McpSdkServerConfig, create_sdk_mcp_server
from claude_agent_sdk import tool as sdk_tool

from trellis.harness.runtime import Paused
from trellis.harness.tools import bridge, results
from trellis.harness.tools.base import Tool

#: The in-process server's name, and so the prefix Claude sees.
SERVER: Final = "trellis"
#: The server config's key (the CLI's ``--mcp-config``, which the SDK passes on with all but the
#: server's instance) that loads its tools from the first step instead of deferring them.
ALWAYS_LOAD: Final = "alwaysLoad"
#: What the CLI is told about a call that paused the run (a harness tool's, or a built-in's
#: the permission check asked a person about).
WAITING: Final = "Waiting for a person's approval."


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
    _SERVED[server["instance"]] = served
    return cast("McpSdkServerConfig", {**server, ALWAYS_LOAD: True})


def listing(server: McpSdkServerConfig) -> tuple[tuple[str, str, str], ...]:
    """What ``server`` (one :func:`convert` made) lists: each tool's name, description and
    input schema (as JSON) — what a CLI process that lists it once has for good."""
    return tuple(
        (t.name, t.description, json.dumps(t.input_schema, sort_keys=True, default=str))
        for t in _SERVED[server["instance"]]
    )


def handler(server: McpSdkServerConfig, name: str) -> Handler:
    """The handler of ``server``'s tool ``name``."""
    [found] = [t.handler for t in _SERVED[server["instance"]] if t.name == name]
    return cast("Handler", found)


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
