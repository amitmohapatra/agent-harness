"""A scripted Bifrost gateway, in process: the MCP tools the examples use when ``BIFROST_URL`` is
unset.

The tools are plain Python functions, served the way the gateway serves an MCP server's:
listed on ``/mcp`` with their annotations (``readOnlyHint``, ``destructiveHint``), executed on
``/v1/mcp/tool/execute``. A client named in ``code_mode`` is a Code Mode client: its tools are
not listed one by one but declared through the Code Mode meta-tools, and ``executeToolCode``
runs a script's ``server.tool(arg=...)`` calls, logging each as the gateway's MCP log does (the
harness reads them back). Its Prompt Repository holds :attr:`ScriptedGateway.prompts` (each a
list of committed versions, their messages). Everything else of the gateway (models) is not
here.

    gateway = ScriptedGateway({"erp-stock": McpTool(stock, read_only=True)})
    async with Harness(gateway=gateway.gateway()) as h: ...
"""

from __future__ import annotations

import ast
import itertools
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Final

import httpx
from bifrost_sdk import Bifrost
from bifrost_sdk.admin import Admin

from trellis import tool as local_tool
from trellis.harness.clients.bifrost import Gateway

URL: Final = "http://gateway.offline/v1"
META_TOOLS: Final = ("listToolFiles", "readToolFile", "getToolDocs", "executeToolCode")
CALL: Final = re.compile(r"(\w+)\.(\w+)\(([^()]*)\)")


@dataclass(frozen=True)
class McpTool:
    """One tool of an MCP server: the function that answers it, and its annotations."""

    fn: Callable[..., Any]
    read_only: bool = False
    destructive: bool = False

    @property
    def description(self) -> str:
        """The docstring without its ``Args:`` (as the harness reads a local tool's)."""
        return local_tool(self.fn).spec.description

    def schema(self) -> dict[str, Any]:
        """The arguments' JSON schema, each described by the docstring's ``Args:``."""
        return dict(local_tool(self.fn).spec.input_schema or {})


@dataclass
class ScriptedGateway:
    #: the tools, by their gateway name ``<client>-<tool>``
    tools: dict[str, McpTool]
    #: the clients that are Code Mode clients
    code_mode: frozenset[str] = frozenset()
    #: the stored prompts, by name: each a list of committed versions (their chat messages)
    prompts: dict[str, list[list[dict[str, Any]]]] = field(default_factory=dict)
    #: every tool the gateway executed, in order (a Code Mode script's nested calls included)
    executed: list[str] = field(default_factory=list)
    _log: list[dict[str, Any]] = field(default_factory=list)
    _ids: itertools.count[int] = field(default_factory=itertools.count)

    def gateway(self) -> Gateway:
        """The harness's gateway client, talking to this gateway."""
        transport = httpx.MockTransport(self.handle)
        http = httpx.AsyncClient(transport=transport, base_url=URL)
        api = httpx.AsyncClient(transport=transport, base_url=URL.removesuffix("/v1"))
        bifrost = Bifrost(URL, api_key="offline", client=http, admin_client=api, max_retries=0)
        return Gateway(URL, "offline", client=bifrost, admin=Admin(URL, client=api))

    # ------------------------------------------------------------------ the routes
    def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/mcp":
            return self._rpc(json.loads(request.content))
        if path == "/v1/mcp/tool/execute":
            call = json.loads(request.content)
            function = call["function"]
            parent = request.headers.get("x-bf-parent-request-id")
            content = self._execute(function["name"], json.loads(function["arguments"]), parent)
            return httpx.Response(
                200, json={"role": "tool", "content": content, "tool_call_id": call.get("id")}
            )
        if path == "/api/mcp-logs":
            parent = request.url.params.get("llm_request_ids")
            logs = [e for e in self._log if e["llm_request_id"] == parent]
            return httpx.Response(200, json={"logs": logs})
        if path == "/api/mcp/clients":
            return httpx.Response(200, json={"clients": [], "count": 0})
        if path.startswith("/api/prompt-repo/prompts"):
            return self._prompts(path)
        return httpx.Response(404, json={"error": {"message": f"no route {path}"}})

    def _prompts(self, path: str) -> httpx.Response:
        """The Prompt Repository: the prompts, or one prompt's versions."""
        if path == "/api/prompt-repo/prompts":
            return httpx.Response(200, json={"prompts": [self._prompt(n) for n in self.prompts]})
        name = path.removeprefix("/api/prompt-repo/prompts/p-").removesuffix("/versions")
        rows = [self._version(name, n) for n in range(1, len(self.prompts[name]) + 1)]
        return httpx.Response(200, json={"versions": rows})

    def _prompt(self, name: str) -> dict[str, Any]:
        latest = self._version(name, len(self.prompts[name]))
        return {"id": f"p-{name}", "name": name, "latest_version": latest}

    def _version(self, name: str, number: int) -> dict[str, Any]:
        messages = self.prompts[name][number - 1]
        return {
            "id": number,
            "prompt_id": f"p-{name}",
            "version_number": number,
            "messages": [{"order_index": i, "message": m} for i, m in enumerate(messages)],
            "provider": "offline",
            "model": "offline",
        }

    def _listed(self) -> list[dict[str, Any]]:
        listed: list[dict[str, Any]] = [{"name": n, "inputSchema": {}} for n in META_TOOLS]
        listed = listed if self.code_mode else []
        for name, tool in self.tools.items():
            if name.split("-", 1)[0] in self.code_mode:
                continue  # a Code Mode client's tools are declared, not listed
            hints = {"readOnlyHint": tool.read_only, "destructiveHint": tool.destructive}
            listed.append(
                {
                    "name": name,
                    "description": tool.description,
                    "inputSchema": tool.schema(),
                    "annotations": hints,
                }
            )
        return listed

    def _rpc(self, message: dict[str, Any]) -> httpx.Response:
        if message["method"] == "tools/list":
            result: dict[str, Any] = {"tools": self._listed()}
        else:
            params = message["params"]
            text = self._meta(params["name"], params.get("arguments") or {})
            result = {"content": [{"type": "text", "text": text}]}
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": message["id"], "result": result})

    def _meta(self, name: str, args: dict[str, Any]) -> str:
        """The Code Mode meta-tools that declare the clients' tools."""
        if name == "listToolFiles":
            return "\n".join(f"servers/{c}.pyi" for c in sorted(self.code_mode))
        if name == "readToolFile":
            client = args["fileName"].removeprefix("servers/").removesuffix(".pyi")
            lines = []
            for full, tool in self.tools.items():
                owner, _, short = full.partition("-")
                if owner == client:
                    params = ", ".join(f"{p}: str" for p in tool.schema()["properties"])
                    lines.append(f"def {short}({params}) -> str:  # {tool.description}")
            return "\n".join(lines)
        return f"{args.get('server')}.{args.get('tool')}: see its declaration"

    def _execute(self, name: str, args: dict[str, Any], parent: str | None) -> str:
        if name in META_TOOLS and name != "executeToolCode":
            return self._meta(name, args)
        if name == "executeToolCode":
            return self._script(str(args.get("code", "")), parent)
        self.executed.append(name)
        return json.dumps(self.tools[name].fn(**args))

    def _script(self, code: str, parent: str | None) -> str:
        """A Code Mode script: each ``server.tool(arg="...")`` call it makes, run and logged
        under the run (``parent``); what it prints is what the calls returned."""
        printed = []
        for client, short, arguments in CALL.findall(code):
            kwargs = {
                k.arg: ast.literal_eval(k.value)
                for k in ast.parse(f"f({arguments})", mode="eval").body.keywords  # type: ignore[attr-defined]
            }
            name = f"{client}-{short}"
            self.executed.append(name)
            result = self.tools[name].fn(**kwargs)
            printed.append(str(result))
            self._log.append(
                {
                    "id": f"log_{next(self._ids)}",
                    "timestamp": datetime.now(UTC).isoformat(),
                    "server_label": client,
                    "tool_name": short,
                    "status": "success",
                    "llm_request_id": parent,
                    "arguments": kwargs,
                    "result": result,
                    "latency": 1.0,
                }
            )
        return "\n".join(printed)
