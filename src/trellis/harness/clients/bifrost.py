"""The Bifrost gateway, as the harness uses it: MCP tools (normal and Code Mode), their
execution log, and chat completions for the ``ReAct`` target.

Agent Mode (the gateway running tools itself) is never used: every call comes back to the
harness so governance and approval sit in front of it.
"""

from __future__ import annotations

import asyncio
import functools
import json
import time
from collections.abc import Sequence
from datetime import datetime
from typing import Any, Final

from bifrost_sdk import Bifrost, MCPLog, Options, ToolDef

from trellis.contracts import ToolError, ToolSpec
from trellis.harness.runtime import current
from trellis.harness.tools.base import Tool

#: Page size when reading the MCP execution log back.
LOG_PAGE: Final = 500
#: The gateway writes its MCP log a few seconds behind: a script's nested calls are read
#: every :data:`LOG_POLL_SECONDS` until two reads agree, for at most :data:`LOG_SETTLE_SECONDS`.
LOG_POLL_SECONDS: Final = 2.0
LOG_SETTLE_SECONDS: Final = 20.0

#: Bifrost's Code Mode meta-tools. The gateway injects these into completions itself and
#: publishes no schema for them; these are the arguments its executor checks for.
CODE_MODE_TOOLS: Final[tuple[ToolSpec, ...]] = (
    ToolSpec(
        name="listToolFiles",
        description="List the virtual declaration files of the MCP servers available to code.",
        input_schema={"type": "object", "properties": {}},
        source="mcp",
        side_effects="read",
    ),
    ToolSpec(
        name="readToolFile",
        description="Read a virtual declaration file (optionally a line range).",
        input_schema={
            "type": "object",
            "properties": {
                "fileName": {"type": "string"},
                "startLine": {"type": "integer"},
                "endLine": {"type": "integer"},
            },
            "required": ["fileName"],
        },
        source="mcp",
        side_effects="read",
    ),
    ToolSpec(
        name="getToolDocs",
        description="The documentation of one tool of one server.",
        input_schema={
            "type": "object",
            "properties": {"server": {"type": "string"}, "tool": {"type": "string"}},
            "required": ["server", "tool"],
        },
        source="mcp",
        side_effects="read",
    ),
    ToolSpec(
        name="executeToolCode",
        description=(
            "Run a Starlark (Python-like) script that calls the servers' tools as "
            "server.tool(param=value); returns what it prints."
        ),
        input_schema={
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
        },
        source="mcp",
        side_effects="read",
    ),
)


class Gateway:
    """One Bifrost gateway (``BIFROST_URL`` is its OpenAI-compatible ``/v1`` base)."""

    def __init__(self, url: str, virtual_key: str | None, *, client: Bifrost | None = None) -> None:
        self.client = client or Bifrost(url, api_key=virtual_key)

    async def tools(self) -> list[ToolDef]:
        """Every MCP tool the virtual key allows: the gateway's own `/mcp` listing asked with
        the key (never `/api`, which admin auth closes to it), Code Mode clients included."""
        return await self.client.tools()

    async def execute(
        self,
        name: str,
        args: dict[str, Any],
        *,
        clients: Sequence[str],
        parent_request_id: str | None = None,
    ) -> Any:
        """Run one call; the tool's result, or :class:`ToolError` when the tool failed."""
        runtime = current()
        call = {
            "id": f"call_{runtime.run_id if runtime else 'direct'}_{name}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        }
        options = Options(mcp_clients=list(clients), parent_request_id=parent_request_id)
        turn = await self.client.execute_tool(call, options=options)
        content = turn.get("content")
        if turn.get("is_error") or turn.get("isError"):
            raise ToolError(str(content), source=f"mcp.{name}")
        if isinstance(content, str):
            try:
                return json.loads(content)
            except ValueError:
                return content
        return content

    async def code_mode_calls(self, parent_request_id: str, since: datetime) -> list[MCPLog]:
        """The nested calls the Code Mode scripts of one run made, from the gateway's log,
        once it has caught up (two reads agree, or :data:`LOG_SETTLE_SECONDS` passed)."""
        deadline = time.monotonic() + LOG_SETTLE_SECONDS
        seen: list[MCPLog] | None = None
        while True:
            found = await self.client.mcp_logs(since, LOG_PAGE, parent_request_id=parent_request_id)
            settled = seen is not None and found and [e.id for e in found] == [e.id for e in seen]
            if settled or time.monotonic() >= deadline:
                return found
            seen = found
            await asyncio.sleep(LOG_POLL_SECONDS)

    async def complete(self, messages: list[dict[str, Any]], **body: Any) -> dict[str, Any]:
        return await self.client.complete(messages, **body)

    async def aclose(self) -> None:
        await self.client.aclose()


def code_mode_tools(gateway: Gateway, servers: Sequence[str]) -> list[Tool]:
    """The meta-tools, scoped to ``servers``. Scripts run under the run's id, so their nested
    calls can be read back from the log and recorded."""

    async def run(name: str, args: dict[str, Any]) -> Any:
        runtime = current()
        return await gateway.execute(
            name,
            args,
            clients=servers,
            parent_request_id=runtime.run_id if runtime is not None else None,
        )

    return [
        Tool(spec, functools.partial(run, spec.name), code_mode=True) for spec in CODE_MODE_TOOLS
    ]
