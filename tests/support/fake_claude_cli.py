#!/usr/bin/env python3
"""A stand-in for the Claude Code CLI, speaking its stream-json protocol on stdin/stdout.

Point ``ClaudeAgentOptions(cli_path=...)`` here and the real Claude Agent SDK drives it: the
control handshake, the prompt, the permission requests (``can_use_tool``) and tool calls into the
SDK's in-process MCP servers (the ``mcp_message`` control requests the real CLI sends). What the
"model" does is scripted in ``FAKE_CLAUDE_SCRIPT`` (JSON): ``{"tool": "<server tool name>",
"args": {...}}`` calls a tool of the ``trellis`` server (``"server": "<name>"``: of another
in-process server), ``{"builtin": "Bash", "args": {...}}``
one of Claude Code's own (run here: a line in the ``FAKE_CLAUDE_BUILTINS`` file), ``{"text":
"..."}`` answers (``{last}`` in it is the text of the last tool result, so an answer shows what
the "model" read). ``FAKE_CLAUDE_RECORD`` names a file the CLI writes what it was started with
(system prompt, allowed tools, the session it resumes, prompt, its
``ANTHROPIC_CUSTOM_HEADERS``, the context its ``UserPromptSubmit`` hooks added, the files of
each ``--plugin-dir``) to.

Hooks: the ``UserPromptSubmit`` hooks the SDK registered (its ``initialize`` request) are called
before a turn, as the real CLI calls them, and the ``additionalContext`` they answer is recorded
(``context``) and is what ``{context}`` in a scripted text stands for. The ``PreToolUse`` hooks
are called before every tool call, before any permission rule: ``allow`` runs it (with their
``updatedInput``), ``deny`` refuses it, ``continue: false`` stops the turn; with no decision the
permission rules go on (``--allowedTools``, then ``can_use_tool``). After a call that ran, the
``PostToolUse`` hooks are called with its result, or the ``PostToolUseFailure`` hooks with its
error (``{"builtin": ..., "fails": "why"}`` scripts a built-in that fails). ``{"stall":
seconds}`` is a step on which the "model" sends nothing for that long (then goes on).

Spending: every result carries usage (:data:`STEP_TOKENS` a step) and a cost
(``FAKE_CLAUDE_COST`` a step); past ``--max-budget-usd`` the query stops with the real CLI's
``error_max_budget_usd`` result. ``--include-partial-messages`` sends each text as word deltas
(``stream_event``) before its message.

A query runs in a session (its id on every message), kept in ``FAKE_CLAUDE_SESSIONS`` (else
the temporary directory): the steps done so far — a tool call that failed or paused is not done.
``--resume=<id>`` goes on after them, as the real CLI continues a conversation; a session it
does not hold is the real CLI's error result. With ``--session-mirror`` (the SDK has a
``SessionStore``) every step is mirrored as the real CLI mirrors its transcript (a
``transcript_mirror`` frame), and a resume the SDK materialised (``CLAUDE_CONFIG_DIR``) goes on
from that transcript — whatever this machine's own sessions hold. ``--max-turns`` stops a query after that many
steps, with the real CLI's ``error_max_turns`` result. A resumed session sent a prompt of its own
(not the harness's "This task was paused" — the conversation's next turn) runs the script anew as
the session's next turn; the record's ``earlier`` is the prompts its transcript holds. The
``additionalContext`` the ``PostToolUse`` hooks answer is appended to ``<record>.added`` (one
JSON string a line): what the "model" is told after a call.
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

SERVER = os.environ.get("FAKE_CLAUDE_SERVER", "trellis")
#: What the harness's tools answer when the run paused (``tools.convert.claude.WAITING``).
WAITING = "Waiting for a person's approval."
SESSIONS = Path(os.environ.get("FAKE_CLAUDE_SESSIONS") or tempfile.gettempdir()) / "fake-claude"
#: How the harness begins a paused task's resumed prompt (``adapters.claude.RESUMED*``).
PAUSED_PROMPT = "This task was paused"
#: The tools ``--permission-mode acceptEdits`` lets run without asking.
EDITS = frozenset({"Edit", "MultiEdit", "Write", "NotebookEdit"})
#: The input and output tokens each step of a query reports.
STEP_TOKENS = (10, 5)


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


def plugins() -> dict[str, dict[str, str]]:
    """Each ``--plugin-dir`` the CLI was given, its files by relative path (as it found them
    at the start: the harness removes a run's plugin when the query ends)."""
    args = sys.argv[1:]
    found: dict[str, dict[str, str]] = {}
    for at, arg in enumerate(args):
        if arg == "--plugin-dir":
            root = Path(args[at + 1])
            files = sorted(p for p in root.rglob("*") if p.is_file()) if root.is_dir() else []
            found[str(root)] = {str(p.relative_to(root)): p.read_text() for p in files}
    return found


class Cli:
    def __init__(self) -> None:
        self.requests = 0
        #: the in-process servers initialized (``initialize``, then its notification)
        self.mcp_ready: set[str] = set()
        #: the hooks the SDK registered, by event (its ``initialize`` request)
        self.hooks: dict[str, Any] = {}
        #: the subagent working now (its ``agent_id`` and ``agent_type``), as the real CLI
        #: names it in the hooks of the tool calls it makes
        self.working: dict[str, str] = {}

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

    def pre_tool_use(self, session: str, name: str, args: dict[str, Any], tool_id: str) -> dict:
        """What the ``PreToolUse`` hooks matching ``name`` decide (the first decision)."""
        for matcher in self.hooks.get("PreToolUse") or []:
            pattern = matcher.get("matcher")
            if pattern not in (None, "", "*") and not re.fullmatch(pattern, name):
                continue
            for callback in matcher.get("hookCallbackIds") or []:
                answer = self.control(
                    {
                        "subtype": "hook_callback",
                        "callback_id": callback,
                        "input": {
                            "session_id": session,
                            "transcript_path": "",
                            "cwd": os.getcwd(),
                            "hook_event_name": "PreToolUse",
                            "tool_name": name,
                            "tool_input": args,
                            "tool_use_id": tool_id,
                            **self.working,
                        },
                        "tool_use_id": tool_id,
                    }
                )
                said = answer.get("hookSpecificOutput") or {}
                if said.get("permissionDecision") or answer.get("continue") is False:
                    return answer
        return {}

    def post_tool_use(
        self,
        session: str,
        name: str,
        args: dict[str, Any],
        tool_id: str,
        *,
        content: list[dict[str, Any]],
        error: str | None,
    ) -> None:
        """A call that ran, to the ``PostToolUse`` hooks matching ``name`` (its result) or —
        it failed — the ``PostToolUseFailure`` ones (its error), as the real CLI reports it."""
        event = "PostToolUse" if error is None else "PostToolUseFailure"
        given: dict[str, Any] = {
            "session_id": session,
            "transcript_path": "",
            "cwd": os.getcwd(),
            "hook_event_name": event,
            "tool_name": name,
            "tool_input": args,
            "tool_use_id": tool_id,
            **self.working,
        }
        if error is None:
            given["tool_response"] = content
        else:
            given["error"] = error
        for matcher in self.hooks.get(event) or []:
            pattern = matcher.get("matcher")
            if pattern not in (None, "", "*") and not re.fullmatch(pattern, name):
                continue
            for callback in matcher.get("hookCallbackIds") or []:
                answer = self.control(
                    {
                        "subtype": "hook_callback",
                        "callback_id": callback,
                        "input": given,
                        "tool_use_id": tool_id,
                    }
                )
                added = (answer.get("hookSpecificOutput") or {}).get("additionalContext")
                record = os.environ.get("FAKE_CLAUDE_RECORD")
                if added and record:
                    with open(f"{record}.added", "a") as handle:
                        handle.write(json.dumps(added) + "\n")

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

    def mcp(self, method: str, params: dict[str, Any], server: str = SERVER) -> dict[str, Any]:
        self.requests += 1
        message = {"jsonrpc": "2.0", "id": self.requests, "method": method, "params": params}
        answer = self.control({"subtype": "mcp_message", "server_name": server, "message": message})
        return answer["mcp_response"]

    def listed(self) -> list[str]:
        """The tools the ``trellis`` server lists."""
        self.ready()
        return [t["name"] for t in self.mcp("tools/list", {})["result"]["tools"]]

    def call_tool(self, name: str, args: dict[str, Any], server: str = SERVER) -> dict[str, Any]:
        self.ready(server)
        return self.mcp("tools/call", {"name": name, "arguments": args}, server)

    def ready(self, server: str = SERVER) -> None:
        if server not in self.mcp_ready:
            self.mcp(
                "initialize",
                {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "fake-claude", "version": "1"},
                },
                server,
            )
            self.control(
                {
                    "subtype": "mcp_message",
                    "server_name": server,
                    "message": {"jsonrpc": "2.0", "method": "notifications/initialized"},
                }
            )
            self.mcp_ready.add(server)

    def permission(self, name: str, args: dict[str, Any], tool_id: str) -> dict[str, Any]:
        """What the SDK's ``can_use_tool`` says about a call — not asked, as the real CLI does
        not ask it, for a tool in ``--allowedTools``, any tool under ``bypassPermissions`` and
        an edit under ``acceptEdits`` (allowed then)."""
        allowed = (argument("--allowedTools") or "").split(",")
        mode = argument("--permission-mode")
        if (
            argument("--permission-prompt-tool") != "stdio"
            or name in allowed
            or mode == "bypassPermissions"
            or (mode == "acceptEdits" and name in EDITS)
        ):
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
                        "earlier": prompts(session),
                        "disallowed_tools": argument("--disallowedTools"),
                        "plugins": plugins(),
                    },
                    handle,
                )
        kept = SESSIONS / f"{session}.json"
        restored = mirrored(session) if resumed is not None else None
        if restored is not None:  # the SDK wrote the session back from its store
            SESSIONS.mkdir(parents=True, exist_ok=True)
            kept.write_text(json.dumps({"done": restored, "turn": mirrored_turn(session)}))
        if resumed is not None and not kept.exists():
            send(result(session, f"No conversation found with session ID: {session}"))
            return
        if resumed is None:  # a new conversation: kept from its first message
            SESSIONS.mkdir(parents=True, exist_ok=True)
            kept.write_text(json.dumps({"done": 0, "turn": 0}))
            mirror(session, 0, prompt=str(prompt))
        state = json.loads(kept.read_text())
        done, turn = state["done"], state.get("turn", 0)
        if resumed is not None and not str(prompt).startswith(PAUSED_PROMPT):
            done, turn = 0, turn + 1  # the conversation's next turn, in the same session
            kept.write_text(json.dumps({"done": 0, "turn": turn}))
            mirror(session, 0, turn, prompt=str(prompt))
        last = read = ""
        limit = argument("--max-turns")
        cap = argument("--max-budget-usd")
        steps = 0
        for turns, (step, item) in enumerate(enumerate(script[done:], start=done)):
            ended = stopped(turns, steps, limit, cap)
            if ended is not None:  # what the real CLI ends with
                send(result(session, ended[0], subtype=ended[1], steps=steps))
                return
            steps += 1
            said, read, completed, interrupted, inner = self.step(
                session, step, item, read, context
            )
            steps += inner
            last = said if said is not None else last
            if not completed:
                if interrupted:
                    break
                continue
            kept.write_text(json.dumps({"done": step + 1, "turn": turn}))
            mirror(session, step + 1, turn)
        send(result(session, last=last, turns=len(script), steps=steps))

    def step(
        self, session: str, step: int, item: dict[str, Any], read: str, context: str
    ) -> tuple[str | None, str, bool, bool, int]:
        """One step of the script: a text, a subagent or a tool call. What it said (``None``:
        no text), the text of its result, whether it is done, whether a check stopped the
        turn, and how many steps of a subagent's own it took."""
        if "stall" in item:  # the model says nothing for a while, then goes on
            time.sleep(float(item["stall"]))
            return None, read, True, False, 0
        if "text" in item:
            said = item["text"].replace("{last}", read).replace("{context}", context)
            if "--include-partial-messages" in sys.argv:
                send(stream_event(session, {"type": "message_start"}))
                for n, word in enumerate(said.split(" ")):
                    send(delta(session, word if n == 0 else f" {word}"))
            send(assistant(session, [{"type": "text", "text": said}]))
            return said, read, True, False, 0
        if "agent" in item:
            answer, completed, interrupted, inner = self.subagent(session, step, item)
            return None, answer, completed, interrupted, inner
        read, completed, interrupted = self.tool(session, step, item)
        return None, read, completed, interrupted, 0

    def subagent(
        self, session: str, step: int, item: dict[str, Any]
    ) -> tuple[str, bool, bool, int]:
        """One subagent (``{"agent", "prompt", "steps"}``), as the real CLI runs one of
        ``agents=``: its ``Agent`` tool call (through the hooks and the permission rules, as
        any call), then its own steps — their messages naming the call
        (``parent_tool_use_id``) — then the call's result, the subagent's last text. The
        text of the result, whether it is done, whether a check stopped the turn, and how
        many of its own steps it took."""
        tool_id = f"toolu_{step}"
        args = {
            "description": item["prompt"],
            "prompt": item["prompt"],
            "subagent_type": item["agent"],
        }
        send(
            assistant(
                session, [{"type": "tool_use", "id": tool_id, "name": "Agent", "input": args}]
            )
        )
        permission = self.allowed(session, "Agent", args, tool_id)
        if permission.get("behavior", "allow") != "allow":
            self.result(session, tool_id, [{"type": "text", "text": "denied"}])
            return "denied", False, False, 0
        briefs = os.environ.get("FAKE_CLAUDE_BRIEFS")
        if briefs:  # what the subagent was given, the hooks' updates included
            given = permission.get("updatedInput", args)
            with open(briefs, "a") as log:
                log.write(json.dumps({"agent": item["agent"], "prompt": given["prompt"]}) + "\n")
        self.working = {"agent_id": f"agent_{tool_id}", "agent_type": item["agent"]}
        try:
            return self._worked(session, step, item, tool_id)
        finally:
            self.working = {}

    def _worked(
        self, session: str, step: int, item: dict[str, Any], tool_id: str
    ) -> tuple[str, bool, bool, int]:
        """The subagent's own steps, then its ``Agent`` call's result."""
        last, taken = "", 0
        for n, inner in enumerate(item["steps"]):
            taken += 1
            if "text" in inner:
                last = inner["text"]
                send(assistant(session, [{"type": "text", "text": last}], parent=tool_id))
                continue
            _, _, interrupted = self.tool(
                session, step, inner, tool_id=f"{tool_id}_{n}", parent=tool_id
            )
            if interrupted:
                return "", False, True, taken
        self.result(session, tool_id, [{"type": "text", "text": last}])
        return last, True, False, taken

    def allowed(
        self, session: str, name: str, args: dict[str, Any], tool_id: str
    ) -> dict[str, Any]:
        """What the ``PreToolUse`` hooks, then the permission rules, decide about a call."""
        hooked = self.pre_tool_use(session, name, args, tool_id)
        decided = hooked.get("hookSpecificOutput") or {}
        if decided.get("permissionDecision") == "deny" or hooked.get("continue") is False:
            return {
                "behavior": "deny",
                "message": decided.get("permissionDecisionReason", "denied"),
                "interrupt": hooked.get("continue") is False,
            }
        if decided.get("permissionDecision") == "allow":
            return {"behavior": "allow", "updatedInput": decided.get("updatedInput", args)}
        return self.permission(name, args, tool_id)

    @staticmethod
    def result(
        session: str, tool_id: str, content: list[dict[str, Any]], parent: str | None = None
    ) -> None:
        """A tool call's result, as the user message the CLI sends."""
        send(
            {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": tool_id, "content": content}
                    ],
                },
                "parent_tool_use_id": parent,
                "session_id": session,
            }
        )

    def tool(
        self,
        session: str,
        step: int,
        item: dict[str, Any],
        *,
        tool_id: str | None = None,
        parent: str | None = None,
    ) -> tuple[str, bool, bool]:
        """One tool call (``parent``: of a subagent, its ``Agent`` call): the text of its
        result, whether it is done (it ran, without an error), and whether the permission
        check stopped the turn."""
        tool_id = tool_id or f"toolu_{step}"
        builtin = item.get("builtin")
        server = item.get("server", SERVER)
        name = builtin or f"mcp__{server}__{item['tool']}"
        args = item.get("args", {})
        call = [{"type": "tool_use", "id": tool_id, "name": name, "input": args}]
        send(assistant(session, call, parent=parent))
        permission = self.allowed(session, name, args, tool_id)
        allowed = permission.get("behavior", "allow") == "allow"
        error = not allowed
        ran = permission.get("updatedInput", args)
        if allowed and builtin:
            with open(os.environ["FAKE_CLAUDE_BUILTINS"], "a") as log:
                log.write(json.dumps({"tool": builtin, "args": ran}) + "\n")
            failure = item.get("fails")
            error = failure is not None
            text = f"{builtin} failed: {failure}" if error else f"{builtin} ran"
            content = [{"type": "text", "text": text}]
            self.post_tool_use(session, name, ran, tool_id, content=content, error=failure)
        elif allowed:
            called = self.call_tool(item["tool"], ran, server)
            answered = called.get("result") or {}
            error = bool(answered.get("isError")) or "error" in called
            content = answered.get("content") or [
                {"type": "text", "text": json.dumps(called.get("error"))}
            ]
            failed = "".join(c.get("text", "") for c in content) if error else None
            self.post_tool_use(session, name, ran, tool_id, content=content, error=failed)
        else:
            content = [{"type": "text", "text": permission.get("message", "denied")}]
        read = "".join(c.get("text", "") for c in content if isinstance(c, dict))
        waits = read == WAITING  # the run paused: the SDK stops reading, the process ends
        self.result(session, tool_id, content, parent)
        return read, not error, waits or bool(permission.get("interrupt"))


def projects() -> Path:
    """Where the CLI keeps its transcripts (as the real CLI: under ``CLAUDE_CONFIG_DIR``)."""
    home = os.environ.get("CLAUDE_CONFIG_DIR") or os.path.expanduser("~/.claude")
    return Path(home) / "projects"


def mirror(session: str, done: int, turn: int = 0, *, prompt: str | None = None) -> None:
    """With ``--session-mirror``: one transcript entry (the steps done in the session's
    ``turn``, the prompt it began with) mirrored to the SDK."""
    if "--session-mirror" not in sys.argv:
        return
    uuid_ = f"{session}-{done}" if not turn else f"{session}-{turn}-{done}"
    entry: dict[str, Any] = {"type": "progress", "uuid": uuid_, "done": done, "turn": turn}
    if prompt is not None:
        entry["prompt"] = prompt
    path = projects() / "fake-project" / f"{session}.jsonl"
    send({"type": "transcript_mirror", "filePath": str(path), "entries": [entry]})


def materialised(session: str) -> list[dict[str, Any]] | None:
    """The entries of a transcript the SDK materialised (``None``: there is none)."""
    if not os.environ.get("CLAUDE_CONFIG_DIR"):
        return None
    for found in projects().glob(f"*/{session}.jsonl"):
        return [json.loads(line) for line in found.read_text().splitlines() if line.strip()]
    return None


def mirrored_turn(session: str) -> int:
    """The session's latest turn, as a transcript the SDK materialised says."""
    return max((int(e.get("turn", 0)) for e in materialised(session) or ()), default=0)


def mirrored(session: str) -> int | None:
    """The steps done in the session's latest turn, as a transcript the SDK materialised says
    (``None``: there is none)."""
    lines = materialised(session)
    if lines is None:
        return None
    turn = mirrored_turn(session)
    return max(int(e.get("done", 0)) for e in lines if int(e.get("turn", 0)) == turn)


def prompts(session: str) -> list[str]:
    """The prompts a transcript the SDK materialised holds: the session's earlier turns."""
    return [e["prompt"] for e in materialised(session) or () if "prompt" in e]


def assistant(
    session: str, content: list[dict[str, Any]], parent: str | None = None
) -> dict[str, Any]:
    """An assistant message (``parent``: a subagent's, the ``Agent`` call that runs it)."""
    return {
        "type": "assistant",
        "message": {"role": "assistant", "model": "fake", "content": content},
        "parent_tool_use_id": parent,
        "session_id": session,
    }


def stopped(turns: int, steps: int, limit: str | None, cap: str | None) -> tuple[str, str] | None:
    """Why the query stops before its next step, and the result's subtype (``None``: it goes
    on): ``--max-turns`` reached, or ``--max-budget-usd`` spent."""
    if limit is not None and turns >= int(limit):
        return "Reached maximum number of turns", "error_max_turns"
    if cap is not None and spent(steps) > float(cap):
        return "Reached maximum budget", "error_max_budget_usd"
    return None


def spent(steps: int) -> float:
    """What ``steps`` steps of a query cost (``FAKE_CLAUDE_COST`` a step)."""
    return steps * float(os.environ.get("FAKE_CLAUDE_COST") or 0)


def delta(session: str, text: str) -> dict[str, Any]:
    """One text delta, as the real CLI streams it with ``--include-partial-messages``."""
    event = {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": text},
    }
    return stream_event(session, event)


def stream_event(session: str, event: dict[str, Any]) -> dict[str, Any]:
    """One raw event of the model's stream (``--include-partial-messages``)."""
    return {
        "type": "stream_event",
        "uuid": str(uuid.uuid4()),
        "session_id": session,
        "event": event,
        "parent_tool_use_id": None,
    }


def result(
    session: str,
    error: str | None = None,
    *,
    last: str = "",
    turns: int = 0,
    subtype: str = "error_during_execution",
    steps: int = 0,
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
        "total_cost_usd": spent(steps),
        "usage": {"input_tokens": STEP_TOKENS[0] * steps, "output_tokens": STEP_TOKENS[1] * steps},
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
