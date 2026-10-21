#!/usr/bin/env python3
"""A stand-in for the Claude Code CLI, speaking its stream-json protocol on stdin/stdout.

Point ``ClaudeAgentOptions(cli_path=...)`` here and the real Claude Agent SDK drives it: the
control handshake, the prompt, the permission requests (``can_use_tool``) and tool calls into the
SDK's in-process MCP servers (the ``mcp_message`` control requests the real CLI sends). What the
"model" does is scripted in ``FAKE_CLAUDE_SCRIPT`` (JSON): ``{"tool": "<server tool name>",
"args": {...}}`` calls a tool of the ``trellis`` server, ``{"builtin": "Bash", "args": {...}}``
one of Claude Code's own (run here: a line in the ``FAKE_CLAUDE_BUILTINS`` file), ``{"text":
"..."}`` answers (``{last}`` in it is the text of the last tool result, so an answer shows what
the "model" read). ``FAKE_CLAUDE_RECORD`` names a file the CLI writes what it was started with
(system prompt, allowed tools, the session it resumes, prompt, its
``ANTHROPIC_CUSTOM_HEADERS``, the context its ``UserPromptSubmit`` hooks added) to.

Hooks: the ``UserPromptSubmit`` hooks the SDK registered (its ``initialize`` request) are called
before a turn, as the real CLI calls them, and the ``additionalContext`` they answer is recorded
(``context``) and is what ``{context}`` in a scripted text stands for.

A query runs in a session (its id on every message), kept in ``FAKE_CLAUDE_SESSIONS`` (else
the temporary directory): the steps done so far — a tool call that failed or paused is not done.
``--resume=<id>`` goes on after them, as the real CLI continues a conversation; a session it
does not hold is the real CLI's error result. With ``--session-mirror`` (the SDK has a
``SessionStore``) every step is mirrored as the real CLI mirrors its transcript (a
``transcript_mirror`` frame), and a resume the SDK materialised (``CLAUDE_CONFIG_DIR``) goes on
from that transcript — whatever this machine's own sessions hold. ``--max-turns`` stops a query after that many
steps, with the real CLI's ``error_max_turns`` result.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

SERVER = os.environ.get("FAKE_CLAUDE_SERVER", "trellis")
#: What the harness's tools answer when the run paused (``tools.convert.claude.WAITING``).
WAITING = "Waiting for a person's approval."
SESSIONS = Path(os.environ.get("FAKE_CLAUDE_SESSIONS") or tempfile.gettempdir()) / "fake-claude"


def send(message: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def read() -> dict[str, Any] | None:
    line = sys.stdin.readline()
    return json.loads(line) if line else None


def argument(name: str) -> str | None:
    args = sys.argv[1:]
    joined = [a.split("=", 1)[1] for a in args if a.startswith(f"{name}=")]
    return joined[0] if joined else args[args.index(name) + 1] if name in args else None


class Cli:
    def __init__(self) -> None:
        self.requests = 0
        self.mcp_ready = False
        #: the hooks the SDK registered, by event (its ``initialize`` request)
        self.hooks: dict[str, Any] = {}

    def prompt_context(self, session: str, prompt: Any) -> str:
        """What the ``UserPromptSubmit`` hooks add to ``prompt`` (their ``additionalContext``)."""
        added = []
        for matcher in self.hooks.get("UserPromptSubmit") or []:
            for callback in matcher.get("hookCallbackIds") or []:
                answer = self.control(
                    {
                        "subtype": "hook_callback",
                        "callback_id": callback,
                        "input": {
                            "session_id": session,
                            "transcript_path": "",
                            "cwd": os.getcwd(),
                            "hook_event_name": "UserPromptSubmit",
                            "prompt": prompt,
                        },
                        "tool_use_id": None,
                    }
                )
                context = (answer.get("hookSpecificOutput") or {}).get("additionalContext")
                if context:
                    added.append(context)
        return "\n".join(added)

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

    def listed(self) -> list[str]:
        """The tools the ``trellis`` server lists."""
        self.ready()
        return [t["name"] for t in self.mcp("tools/list", {})["result"]["tools"]]

    def call_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        self.ready()
        return self.mcp("tools/call", {"name": name, "arguments": args})

    def ready(self) -> None:
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

    def permission(self, name: str, args: dict[str, Any], tool_id: str) -> dict[str, Any]:
        """What the SDK's ``can_use_tool`` says about a call (allowed when it is not asked)."""
        allowed = (argument("--allowedTools") or "").split(",")
        if argument("--permission-prompt-tool") != "stdio" or name in allowed:
            return {"behavior": "allow"}
        return self.control(
            {
                "subtype": "can_use_tool",
                "tool_name": name,
                "input": args,
                "permission_suggestions": None,
                "tool_use_id": tool_id,
            }
        )

    def turn(self, prompt: Any) -> None:
        script = json.loads(os.environ.get("FAKE_CLAUDE_SCRIPT", '[{"text": "ok"}]'))
        resumed = argument("--resume")
        session = resumed or str(uuid.uuid4())
        context = self.prompt_context(session, prompt)
        record = os.environ.get("FAKE_CLAUDE_RECORD")
        if record:
            servers = json.loads(argument("--mcp-config") or "{}").get("mcpServers", {})
            with open(record, "w") as handle:
                json.dump(
                    {
                        "system_prompt": argument("--system-prompt"),
                        "allowed_tools": argument("--allowedTools"),
                        "mcp_config": argument("--mcp-config"),
                        "tools": self.listed() if SERVER in servers else [],
                        "permission_prompt_tool": argument("--permission-prompt-tool"),
                        "resume": resumed,
                        "prompt": prompt,
                        "custom_headers": os.environ.get("ANTHROPIC_CUSTOM_HEADERS"),
                        "context": context,
                        "materialised": mirrored(session) is not None,
                    },
                    handle,
                )
        kept = SESSIONS / f"{session}.json"
        restored = mirrored(session) if resumed is not None else None
        if restored is not None:  # the SDK wrote the session back from its store
            SESSIONS.mkdir(parents=True, exist_ok=True)
            kept.write_text(json.dumps({"done": restored}))
        if resumed is not None and not kept.exists():
            send(result(session, f"No conversation found with session ID: {session}"))
            return
        if resumed is None:  # a new conversation: kept from its first message
            SESSIONS.mkdir(parents=True, exist_ok=True)
            kept.write_text(json.dumps({"done": 0}))
            mirror(session, 0)
        done = json.loads(kept.read_text())["done"]
        last = read = ""
        limit = argument("--max-turns")
        for turns, (step, item) in enumerate(enumerate(script[done:], start=done)):
            if limit is not None and turns >= int(limit):  # what the real CLI ends with
                send(result(session, "Reached maximum number of turns", subtype="error_max_turns"))
                return
            if "text" in item:
                last = item["text"].replace("{last}", read).replace("{context}", context)
                send(assistant(session, [{"type": "text", "text": last}]))
            else:
                read, completed, interrupted = self.tool(session, step, item)
                if not completed:
                    if interrupted:
                        break
                    continue
            kept.write_text(json.dumps({"done": step + 1}))
            mirror(session, step + 1)
        send(result(session, last=last, turns=len(script)))

    def tool(self, session: str, step: int, item: dict[str, Any]) -> tuple[str, bool, bool]:
        """One tool call: the text of its result, whether it is done (it ran, without an
        error), and whether the permission check stopped the turn."""
        tool_id = f"toolu_{step}"
        builtin = item.get("builtin")
        name = builtin or f"mcp__{SERVER}__{item['tool']}"
        args = item.get("args", {})
        send(assistant(session, [{"type": "tool_use", "id": tool_id, "name": name, "input": args}]))
        permission = self.permission(name, args, tool_id)
        allowed = permission.get("behavior", "allow") == "allow"
        error = not allowed
        if allowed and builtin:
            ran = permission.get("updatedInput", args)
            with open(os.environ["FAKE_CLAUDE_BUILTINS"], "a") as log:
                log.write(json.dumps({"tool": builtin, "args": ran}) + "\n")
            content = [{"type": "text", "text": f"{builtin} ran"}]
        elif allowed:
            called = self.call_tool(item["tool"], permission.get("updatedInput", args))
            answered = called.get("result") or {}
            error = bool(answered.get("isError")) or "error" in called
            content = answered.get("content") or [
                {"type": "text", "text": json.dumps(called.get("error"))}
            ]
        else:
            content = [{"type": "text", "text": permission.get("message", "denied")}]
        read = "".join(c.get("text", "") for c in content if isinstance(c, dict))
        waits = read == WAITING  # the run paused: the SDK stops reading, the process ends
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
                "session_id": session,
            }
        )
        return read, not error, waits or bool(permission.get("interrupt"))


def projects() -> Path:
    """Where the CLI keeps its transcripts (as the real CLI: under ``CLAUDE_CONFIG_DIR``)."""
    home = os.environ.get("CLAUDE_CONFIG_DIR") or os.path.expanduser("~/.claude")
    return Path(home) / "projects"


def mirror(session: str, done: int) -> None:
    """With ``--session-mirror``: one transcript entry (the steps done) mirrored to the SDK."""
    if "--session-mirror" not in sys.argv:
        return
    entry = {"type": "progress", "uuid": f"{session}-{done}", "done": done}
    path = projects() / "fake-project" / f"{session}.jsonl"
    send({"type": "transcript_mirror", "filePath": str(path), "entries": [entry]})


def mirrored(session: str) -> int | None:
    """The steps done, as a transcript the SDK materialised says (``None``: there is none)."""
    if not os.environ.get("CLAUDE_CONFIG_DIR"):
        return None
    for found in projects().glob(f"*/{session}.jsonl"):
        lines = [json.loads(line) for line in found.read_text().splitlines() if line.strip()]
        return max(int(e.get("done", 0)) for e in lines)
    return None


def assistant(session: str, content: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "type": "assistant",
        "message": {"role": "assistant", "model": "fake", "content": content},
        "parent_tool_use_id": None,
        "session_id": session,
    }


def result(
    session: str,
    error: str | None = None,
    *,
    last: str = "",
    turns: int = 0,
    subtype: str = "error_during_execution",
) -> dict[str, Any]:
    """The query's end: its answer, or — ``error`` — what the real CLI says when it cannot run
    it (``num_turns`` 0), or stops it (``subtype``: ``error_max_turns`` past ``--max-turns``)."""
    ended: dict[str, Any] = {
        "type": "result",
        "subtype": "success" if error is None else subtype,
        "duration_ms": 1,
        "duration_api_ms": 1,
        "is_error": error is not None,
        "num_turns": turns,
        "session_id": session,
        "result": last,
        "total_cost_usd": 0.0,
    }
    if error is not None:
        ended["errors"] = [error]
    return ended


def main() -> None:
    if "-v" in sys.argv or "--version" in sys.argv:
        print("9.9.9 (Claude Code)")
        return
    cli = Cli()
    while (message := read()) is not None:
        kind = message.get("type")
        if kind == "control_request":
            request = message["request"]
            if request.get("subtype") == "initialize":
                cli.hooks = request.get("hooks") or {}
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
