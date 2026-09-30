#!/usr/bin/env python3
"""A stand-in for the Claude Code CLI, speaking its stream-json protocol on stdin/stdout.

Point ``ClaudeAgentOptions(cli_path=...)`` here and the real Claude Agent SDK drives it: the
control handshake, the prompt, and tool calls into the SDK's in-process MCP servers (the
``mcp_message`` control requests the real CLI sends). What the "model" does is scripted in
``FAKE_CLAUDE_SCRIPT`` (JSON): ``{"tool": "<server tool name>", "args": {...}}`` calls a tool of
the ``trellis`` server, ``{"text": "..."}`` answers. ``FAKE_CLAUDE_RECORD`` names a file the CLI
writes what it was started with (system prompt, allowed tools, prompt) to.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

SERVER = "trellis"


def send(message: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def read() -> dict[str, Any] | None:
    line = sys.stdin.readline()
    return json.loads(line) if line else None


def argument(name: str) -> str | None:
    args = sys.argv[1:]
    return args[args.index(name) + 1] if name in args else None


class Cli:
    def __init__(self) -> None:
        self.requests = 0
        self.mcp_ready = False

    def control(self, request: dict[str, Any]) -> dict[str, Any]:
        """Send a control request to the SDK and wait for its answer."""
        self.requests += 1
        request_id = f"cli_{self.requests}"
        send({"type": "control_request", "request_id": request_id, "request": request})
        while (message := read()) is not None:
            if message.get("type") == "control_response":
                response = message["response"]
                if response.get("request_id") == request_id:
                    if response.get("subtype") == "error":
                        raise RuntimeError(response.get("error"))
                    return response.get("response") or {}
        raise RuntimeError("stdin closed while waiting for the SDK")

    def mcp(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self.requests += 1
        message = {"jsonrpc": "2.0", "id": self.requests, "method": method, "params": params}
        answer = self.control({"subtype": "mcp_message", "server_name": SERVER, "message": message})
        return answer["mcp_response"]

    def call_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if not self.mcp_ready:
            self.mcp(
                "initialize",
                {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "fake-claude", "version": "1"},
                },
            )
            self.control(
                {
                    "subtype": "mcp_message",
                    "server_name": SERVER,
                    "message": {"jsonrpc": "2.0", "method": "notifications/initialized"},
                }
            )
            self.mcp_ready = True
        return self.mcp("tools/call", {"name": name, "arguments": args})

    def turn(self, prompt: Any) -> None:
        script = json.loads(os.environ.get("FAKE_CLAUDE_SCRIPT", '[{"text": "ok"}]'))
        record = os.environ.get("FAKE_CLAUDE_RECORD")
        if record:
            with open(record, "w") as handle:
                json.dump(
                    {
                        "system_prompt": argument("--system-prompt"),
                        "allowed_tools": argument("--allowedTools"),
                        "mcp_config": argument("--mcp-config"),
                        "prompt": prompt,
                    },
                    handle,
                )
        last = ""
        for step, item in enumerate(script):
            if "tool" in item:
                tool_id = f"toolu_{step}"
                send(
                    assistant(
                        [
                            {
                                "type": "tool_use",
                                "id": tool_id,
                                "name": f"mcp__{SERVER}__{item['tool']}",
                                "input": item.get("args", {}),
                            }
                        ]
                    )
                )
                result = self.call_tool(item["tool"], item.get("args", {}))
                content = (result.get("result") or {}).get("content") or [
                    {"type": "text", "text": json.dumps(result.get("error"))}
                ]
                send(
                    {
                        "type": "user",
                        "message": {
                            "role": "user",
                            "content": [
                                {"type": "tool_result", "tool_use_id": tool_id, "content": content}
                            ],
                        },
                        "parent_tool_use_id": None,
                        "session_id": "fake",
                    }
                )
            else:
                last = item["text"]
                send(assistant([{"type": "text", "text": last}]))
        send(
            {
                "type": "result",
                "subtype": "success",
                "duration_ms": 1,
                "duration_api_ms": 1,
                "is_error": False,
                "num_turns": len(script),
                "session_id": "fake",
                "result": last,
                "total_cost_usd": 0.0,
            }
        )


def assistant(content: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "type": "assistant",
        "message": {"role": "assistant", "model": "fake", "content": content},
        "parent_tool_use_id": None,
        "session_id": "fake",
    }


def main() -> None:
    if "-v" in sys.argv or "--version" in sys.argv:
        print("9.9.9 (Claude Code)")
        return
    cli = Cli()
    while (message := read()) is not None:
        kind = message.get("type")
        if kind == "control_request":
            request = message["request"]
            send(
                {
                    "type": "control_response",
                    "response": {
                        "subtype": "success",
                        "request_id": message["request_id"],
                        "response": {"commands": []}
                        if request.get("subtype") == "initialize"
                        else {},
                    },
                }
            )
        elif kind == "user":
            cli.turn(message["message"]["content"])


if __name__ == "__main__":
    main()
