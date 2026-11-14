#!/usr/bin/env python3
"""A stand-in for an ACP agent (Kimi Code's ``kimi acp``), speaking ACP over stdio through the
official SDK's agent side — what the adapter's tests drive without Node.

What the "model" does is scripted in ``FAKE_ACP_SCRIPT`` (JSON), one step after another; a
step that did not run (the run paused on it) runs again in the next prompt, a step a prompt
told it was rejected or denied is done (``{last}``: what the prompt said), and an edited one
runs again with the prompt's arguments — what Kimi's model does, told so:

* ``{"text": "..."}`` answers (``{last}``: the last result it read);
* ``{"tool": "<harness tool>", "args": {...}}`` calls a tool of the ``trellis`` MCP server the
  session was given (asking permission first, as Kimi does in its default mode);
* ``{"builtin": "Bash", "args": {...}, "ask": true}`` one of its own tools (``ask``: asking
  permission first; ``"fails": "why"`` it fails; ``"output"`` what it returns); with the
  client's terminal capability, ``Bash`` runs in the client's terminal, and ``Write`` writes
  through the client's ``fs`` (``Read``: reads through it);
* ``{"agent": {...}, "result": "...", "fails": false}`` a sub-agent (``Agent``, its own call
  asking nothing; ``"asks": {...builtin step}`` a call it makes asking permission, which shows
  no update); ``{"swarm": {...}, "result": "..."}`` a swarm (``AgentSwarm``);
* ``{"agent": {..., "run_in_background": true}, "background": {...}}`` a sub-agent in the
  background, and ``{"builtin": "Bash", "args": {..., "run_in_background": true}, "background":
  {...}}`` a command: the call answers at once with the task's id (and the sub-agent's), as
  Kimi's does, and the task works on — ``"after"`` seconds, and until the file ``"gate"``
  names is there (a test's word that what it waits for is over) — then (a sub-agent's) ``"usage"``
  in its log and ``"asks"`` (a call it asks permission for, its request in its log first:
  ``interaction.request``; ``{"tool": ...}`` a harness tool's, called once allowed —
  ``"call_after"`` seconds later), ``"calls"`` (a harness tool's call it asks nothing for),
  then it ends (``"result"``, or ``"fails": "why"``) in the main
  agent's log (``task.started``, ``subagent.completed``, ``task.terminated``) — and then Kimi
  runs a turn of its own for the task's end (``turn.prompt`` whose origin is the task, then
  ``turn.ended``; ``"own_turn"`` seconds long; ``"notified": {...builtin step}`` a call it asks
  permission for). A prompt that comes while that turn runs is answered ``end_turn`` at once
  and recorded (``queued``), as Kimi queues it behind its turn;
* ``{"question": "...", "options": [...]}`` its question to the user (``AskUserQuestion``:
  two options or more, as Kimi's takes), and ``{"questions": [{"question", "header",
  "options": [{"label", "description"}], "multi_select"}, ...]}`` several in one call: asked in
  one form of a client that takes forms (``FAKE_ACP_FORMS=0``: it sends none), else the first
  only, as a permission request — as Kimi does;
  ``{"review": {"plan": "...", "options": [...]}}`` its plan, presented for review
  (``ExitPlanMode``) as Kimi asks: an option an approach (two or more; else ``Approve``),
  ``Revise`` and ``Reject and Exit``;
* ``{"usage": {...}, "agent": "main"}`` a model request's usage, in its usage log under
  ``KIMI_CODE_HOME`` (Kimi's ``usage.record`` line); ``{"bound": {"agent": "agent-0",
  "profile": "explore"}}`` the profile a sub-agent was started with, in its log (Kimi's
  ``profile.bind`` line);
* ``{"stall": seconds}`` sends nothing for that long; ``{"crash": "why"}`` the process ends;
  ``{"stop": "refusal"}`` ends the turn so; ``{"step_limit": n}`` ends it at Kimi's step limit (no
  answer, the reason only in its log's ``turn.ended``); ``{"prompt_usage": {...}}`` puts ACP's own usage on
  the prompt's answer; ``{"plan": true}`` a plan update; ``{"wait": seconds}`` a tool call that
  runs that long (a cancel stops it); ``{"updates": true}`` every other update ACP has (a
  thought, a message chunk that is not text, its commands, its mode, its usage, the session's
  title), an extension notification and an extension request (its answer, or the client's
  refusal, recorded).

``FAKE_ACP_RECORD`` names a file it writes what it was started and asked with to (its
environment's ``KIMI_*`` and ``HOME``, its working directory, the client's capabilities, each
session's MCP servers and how it began, the prompts, the permission answers).
``FAKE_ACP_VERSION`` is the version it reports (``2.6.2``); ``FAKE_ACP_RESUME=0`` offers no
``session/resume`` (only ``session/load``, which replays the session's text) and no
``session/fork`` (which copies a session into a new one, as Kimi's does).
``FAKE_ACP_CONFIG`` (JSON) is what its ``config.toml`` starts every session in, as Kimi's does
(``default_permission_mode``: ``auto`` or ``yolo``, in which it asks about no call;
``default_plan_mode``): in its log (``permission.set_mode``, ``plan_mode.enter``) while ACP
goes on saying ``default``. ``session/set_mode`` sets both (logged so), and fails for the plan
mode of a session already in it, as Kimi's does — ``FAKE_ACP_MODE_STUCK=1``: it answers and
changes nothing.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import uuid
from pathlib import Path
from typing import Any, cast

from acp import PROTOCOL_VERSION, RequestError, run_agent, text_block
from acp.schema import (
    AgentCapabilities,
    AgentMessageChunk,
    AgentPlanUpdate,
    AgentThoughtChunk,
    AvailableCommand,
    AvailableCommandsUpdate,
    ContentToolCallContent,
    CurrentModeUpdate,
    ElicitationFormSessionMode,
    ElicitationSchema,
    EnvVariable,
    ForkSessionResponse,
    Implementation,
    InitializeResponse,
    LoadSessionResponse,
    NewSessionResponse,
    PermissionOption,
    PlanEntry,
    PromptResponse,
    ResourceContentBlock,
    ResumeSessionResponse,
    SessionInfoUpdate,
    SessionMode,
    SessionModeState,
    SetSessionModeResponse,
    ToolCallProgress,
    ToolCallStart,
    ToolCallUpdate,
    Usage,
    UsageUpdate,
)

SCRIPT: list[dict[str, Any]] = json.loads(os.environ.get("FAKE_ACP_SCRIPT", "[]"))
RECORD = os.environ.get("FAKE_ACP_RECORD")
VERSION = os.environ.get("FAKE_ACP_VERSION", "2.6.2")
HOME = Path(os.environ.get("KIMI_CODE_HOME") or Path.home() / ".kimi-code")
CANONICAL = [
    PermissionOption(option_id="approve_once", name="Approve once", kind="allow_once"),
    PermissionOption(
        option_id="approve_always", name="Approve for this session", kind="allow_always"
    ),
    PermissionOption(option_id="reject", name="Reject", kind="reject_once"),
]
CONFIG: dict[str, Any] = json.loads(os.environ.get("FAKE_ACP_CONFIG", "{}"))
#: Kimi's session modes over ACP, and what each sets (``acpModeToToggles``): plan mode, and the
#: permission mode.
MODES = {
    "default": (False, "manual"),
    "plan": (True, "manual"),
    "auto": (False, "auto"),
    "yolo": (False, "yolo"),
}
EDITED = re.compile(r"Call (?P<tool>\S+) again now with exactly these arguments: (?P<args>\{.*\})$")

#: This process, in its calls' ids.
PROCESS = uuid.uuid4().hex[:8]


def written(path: Path, data: dict[str, Any]) -> None:
    """Replaces ``path`` whole, so a reader (the test, the next agent) never sees half of it."""
    part = path.with_suffix(".part")
    part.write_text(json.dumps(data))
    os.replace(part, path)


def record(key: str, value: Any) -> None:
    if not RECORD:
        return
    path = Path(RECORD)
    data = json.loads(path.read_text()) if path.exists() else {}
    data.setdefault(key, []).append(value)
    written(path, data)


def session_dir(session_id: str) -> Path:
    return HOME / "sessions" / "wd_fake" / session_id


def state(session_id: str) -> dict[str, Any]:
    path = session_dir(session_id) / "state.json"
    return json.loads(path.read_text()) if path.exists() else {"done": 0, "last": "", "said": []}


def save(session_id: str, data: dict[str, Any]) -> None:
    path = session_dir(session_id) / "state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    written(path, data)


def logged_usage(session_id: str, agent: str, usage: dict[str, Any]) -> None:
    path = session_dir(session_id) / "agents" / agent / "wire.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    line = {"type": "usage.record", "agentId": agent, "model": "__kimi_env_model__", "usage": usage}
    with path.open("a") as file:
        file.write("not json\n" if usage.get("garbled") else json.dumps(line) + "\n")


def logged(session_id: str, agent: str, line: dict[str, Any]) -> None:
    """One line of an agent's log (``agents/<agent>/wire.jsonl``), as Kimi writes it."""
    path = session_dir(session_id) / "agents" / agent / "wire.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as file:
        file.write(json.dumps(line) + "\n")


def modes() -> SessionModeState:
    """What Kimi says of a session's mode when it opens: ``default``, whatever its config
    started the session in."""
    return SessionModeState(
        current_mode_id="default",
        available_modes=[SessionMode(id=mode, name=mode.title()) for mode in MODES],
    )


def mode_of(session_id: str) -> tuple[bool, str]:
    """Whether the session is in plan mode, and its permission mode, by its log."""
    path = session_dir(session_id) / "agents" / "main" / "wire.jsonl"
    planning, permission = False, "manual"
    for line in path.read_text().splitlines() if path.exists() else ():
        entry = json.loads(line) if line.startswith("{") else {}
        if entry.get("type") == "permission.set_mode":
            permission = entry["mode"]
        elif str(entry.get("type")).startswith("plan_mode."):
            planning = entry["type"] == "plan_mode.enter"
    return planning, permission


def logged_end(session_id: str, reason: str, limit: int | None = None) -> None:
    """The turn's end, as Kimi 2.1 logs every turn's after its usage (its durable
    ``TurnEnded``: ``reason`` completed, cancelled or failed; at its step limit failed, with
    ``loop.max_steps_exceeded``)."""
    line: dict[str, Any] = {"type": "turn.ended", "agentId": "main", "turnId": 0, "reason": reason}
    if limit is not None:
        line["error"] = {"code": "loop.max_steps_exceeded", "details": {"maxSteps": limit}}
    logged(session_id, "main", line)


class Stopped(Exception):
    """The turn stops here: the step did not run (a pause, a rejection, an edit)."""


class FakeAgent:
    def __init__(self) -> None:
        self.conn: Any = None
        self.client: Any = None
        self.servers: dict[str, list[Any]] = {}
        self.cancelled: dict[str, asyncio.Event] = {}
        self.calls = 0
        #: its background tasks at work
        self.tasks: set[asyncio.Task[None]] = set()
        #: a turn of its own is under way (a prompt now is queued behind it)
        self.busy = False
        #: its main agent's turns so far
        self.turns = 0

    def on_connect(self, conn: Any) -> None:
        self.conn = conn

    async def initialize(
        self, protocol_version: int, client_capabilities: Any = None, **kw: Any
    ) -> Any:
        self.client = client_capabilities
        if os.environ.get("FAKE_ACP_CRASH") == "initialize":
            sys.stderr.write("cannot start\n")
            sys.stderr.flush()
            os._exit(4)
        record(
            "initialize",
            client_capabilities.model_dump(by_alias=True) if client_capabilities else None,
        )
        record(
            "started",
            {
                "cwd": os.getcwd(),
                "env": {
                    k: v
                    for k, v in os.environ.items()
                    if k.startswith(("KIMI_", "HOME", "FAKE_SECRET"))
                },
            },
        )
        caps: dict[str, Any] = {"loadSession": True}
        if os.environ.get("FAKE_ACP_RESUME", "1") == "1":
            caps["sessionCapabilities"] = {"resume": {}, "fork": {}}
        return InitializeResponse(
            protocol_version=PROTOCOL_VERSION,
            agent_capabilities=AgentCapabilities.model_validate(caps),
            agent_info=Implementation(name="fake", version=VERSION) if VERSION else None,
        )

    async def new_session(self, cwd: str, mcp_servers: list[Any] | None = None, **kw: Any) -> Any:
        if os.environ.get("FAKE_ACP_AUTH") == "missing":
            raise RequestError.auth_required({"why": "log in first"})
        session = f"session_{uuid.uuid4().hex[:8]}"
        self.servers[session] = list(mcp_servers or [])
        save(session, state(session))
        if CONFIG.get("default_permission_mode"):
            mode = CONFIG["default_permission_mode"]
            logged(
                session, "main", {"type": "permission.set_mode", "agentId": "main", "mode": mode}
            )
        if CONFIG.get("default_plan_mode"):
            logged(session, "main", {"type": "plan_mode.enter", "agentId": "main", "id": "p0"})
        record(
            "sessions",
            {
                "how": "new",
                "id": session,
                "servers": [s.model_dump(by_alias=True) for s in mcp_servers or []],
            },
        )
        return NewSessionResponse(session_id=session, modes=modes())

    async def resume_session(
        self, cwd: str, session_id: str, mcp_servers: list[Any] | None = None, **kw: Any
    ) -> Any:
        if not (session_dir(session_id) / "state.json").exists():
            raise RequestError.resource_not_found(session_id)
        self.servers[session_id] = list(mcp_servers or [])
        record("sessions", {"how": "resume", "id": session_id})
        return ResumeSessionResponse(modes=modes())

    async def fork_session(
        self, cwd: str, session_id: str, mcp_servers: list[Any] | None = None, **kw: Any
    ) -> Any:
        if not (session_dir(session_id) / "state.json").exists():
            raise RequestError.resource_not_found(session_id)
        forked = f"session_{uuid.uuid4().hex[:8]}"
        self.servers[forked] = list(mcp_servers or [])
        save(forked, state(session_id))
        record("sessions", {"how": "fork", "id": forked, "from": session_id})
        return ForkSessionResponse(session_id=forked, modes=modes())

    async def load_session(
        self, cwd: str, session_id: str, mcp_servers: list[Any] | None = None, **kw: Any
    ) -> Any:
        if not (session_dir(session_id) / "state.json").exists():
            raise RequestError.resource_not_found(session_id)
        self.servers[session_id] = list(mcp_servers or [])
        for said in state(session_id)["said"]:  # the history, replayed
            await self.conn.session_update(
                session_id,
                AgentMessageChunk(session_update="agent_message_chunk", content=text_block(said)),
            )
        record("sessions", {"how": "load", "id": session_id})
        return LoadSessionResponse(modes=modes())

    async def set_session_mode(self, mode_id: str, session_id: str, **kw: Any) -> Any:
        record("modes", mode_id)
        if os.environ.get("FAKE_ACP_MODE_STUCK") != "1":
            plan, permission = MODES[mode_id]
            if plan and mode_of(session_id)[0]:  # as Kimi: its permission mode left as it was
                raise RequestError.internal_error({"why": "Already in plan mode"})
            kind = "plan_mode.enter" if plan else "plan_mode.cancel"
            logged(session_id, "main", {"type": kind, "agentId": "main"})
            line = {"type": "permission.set_mode", "agentId": "main", "mode": permission}
            logged(session_id, "main", line)
        return SetSessionModeResponse()

    async def cancel(self, session_id: str, **kw: Any) -> None:
        record("cancels", session_id)
        self.cancelled.setdefault(session_id, asyncio.Event()).set()

    async def prompt(self, prompt: list[Any], session_id: str, **kw: Any) -> Any:
        if self.busy:  # queued behind its own turn: no turn of the prompt's own, as Kimi does
            record("queued", "\n".join(getattr(b, "text", "") for b in prompt))
            return PromptResponse(stop_reason="end_turn")
        self.limit: int | None = None
        reason = "failed"
        self.turns += 1
        logged(session_id, "main", {"type": "turn.prompt", "origin": {"kind": "user"}})
        try:
            answered = await self._prompt(prompt, session_id)
            stopped = answered.stop_reason == "cancelled"
            reason = "cancelled" if stopped else "failed" if self.limit else "completed"
            return answered
        finally:
            logged_end(session_id, reason, self.limit)

    async def _prompt(self, prompt: list[Any], session_id: str) -> Any:
        text = "\n".join(getattr(b, "text", "") for b in prompt)
        record("prompts", text)
        self.cancelled[session_id] = asyncio.Event()
        data = state(session_id)
        edited = EDITED.search(text)
        told = ("rejected" in (data.get("stopped") or "") and "was not run" in text) or (
            "denied it" in text
        )
        if data.get("stopped") and not edited and data["done"] < len(SCRIPT) and told:
            data["done"] += 1  # the step was rejected: what the prompt says is its result
            data["last"] = text
        data["stopped"] = None
        override = json.loads(edited["args"]) if edited else None
        usage = None
        try:
            while data["done"] < len(SCRIPT):
                step = SCRIPT[data["done"]]
                if "stop" in step:
                    data["done"] += 1
                    save(session_id, data)
                    return PromptResponse(stop_reason=step["stop"])
                if "step_limit" in step:  # a finished turn with no answer, over ACP
                    data["done"] += 1
                    save(session_id, data)
                    self.limit = int(step["step_limit"])
                    return PromptResponse(stop_reason="end_turn")
                if "prompt_usage" in step:
                    usage = Usage.model_validate(step["prompt_usage"])
                    data["done"] += 1
                    continue
                result = await self.step(session_id, step, data, override)
                override = None
                data["done"] += 1
                if result is not None:
                    data["last"] = result
                save(session_id, data)
        except Stopped as stop:
            data["stopped"] = str(stop)
            save(session_id, data)
            await self.cancelled[session_id].wait()
            return PromptResponse(stop_reason="cancelled")
        except asyncio.CancelledError:
            raise
        except Exception:
            import traceback

            record("errors", traceback.format_exc())
            raise
        save(session_id, data)
        return PromptResponse(stop_reason="end_turn", usage=usage)

    async def updates(self, session_id: str) -> None:
        """Every update ACP has beside text, plans and tool calls, then the extension points:
        a notification, and a request whose answer (or refusal) is recorded."""
        link = ResourceContentBlock(type="resource_link", name="notes", uri="file:///notes.md")
        sent: list[Any] = [
            AgentThoughtChunk(
                session_update="agent_thought_chunk", content=text_block("thinking it over")
            ),
            AgentMessageChunk(session_update="agent_message_chunk", content=link),
            AvailableCommandsUpdate(
                session_update="available_commands_update",
                available_commands=[AvailableCommand(name="init", description="Start.")],
            ),
            CurrentModeUpdate(session_update="current_mode_update", current_mode_id="default"),
            UsageUpdate(session_update="usage_update", used=10, size=100),
            SessionInfoUpdate(session_update="session_info_update", title="A session"),
        ]
        for update in sent:
            await self.conn.session_update(session_id, update)
        await self.conn.ext_notification("fake/progress", {"done": 1})
        try:
            record("extensions", await self.conn.ext_method("fake/ask", {"q": 1}))
        except RequestError as refused:
            record("extensions", {"refused": refused.code})

    async def say(self, session_id: str, text: str) -> None:
        await self.conn.session_update(
            session_id,
            AgentMessageChunk(session_update="agent_message_chunk", content=text_block(text)),
        )

    async def step(  # noqa: PLR0911, PLR0912 - a branch and a return a kind of step
        self, session_id: str, step: dict[str, Any], data: dict[str, Any], override: Any
    ) -> str | None:
        if "text" in step:
            said = step["text"].replace("{last}", data["last"])
            data["said"].append(said)
            words = said.split(" ")
            for n, word in enumerate(words):
                await self.say(session_id, word if n == len(words) - 1 else word + " ")
            return None
        if "usage" in step:
            logged_usage(session_id, step.get("agent", "main"), step["usage"])
            return None
        if "bound" in step:
            agent, profile = step["bound"]["agent"], step["bound"]["profile"]
            logged(session_id, agent, {"type": "metadata", "protocol_version": "1.5"})
            logged(session_id, agent, {"type": "profile.bind", "profileName": profile})
            return None
        if "stall" in step:
            await asyncio.sleep(step["stall"])
            return None
        if "crash" in step:
            sys.stderr.write(step["crash"] + "\n")
            sys.stderr.flush()
            os._exit(3)
        if "skills" in step:  # what its skills folder holds, as Kimi reads it
            folder = HOME / "skills"
            found = {
                str(p.relative_to(folder)): p.read_text()
                for p in sorted(folder.rglob("*"))
                if p.is_file()
            }
            linked = sorted(p.name for p in HOME.iterdir() if p.is_symlink())  # noqa: ASYNC240
            record("skills", {"files": found, "linked": linked})
            return None
        if "error" in step:
            raise RequestError.internal_error({"why": step["error"]})
        if "orphan" in step:  # an update of a call it never said it started
            await self.progress(session_id, "0:unknown", status="completed", raw_input={"x": 1})
            return None
        if "parallel" in step:
            return await self.parallel(session_id, step["parallel"])
        if "plan" in step:
            await self.conn.session_update(
                session_id,
                AgentPlanUpdate(
                    session_update="plan",
                    entries=[PlanEntry(content="look", priority="high", status="pending")],
                ),
            )
            return None
        if "updates" in step:
            await self.updates(session_id)
            return None
        if "question" in step or "questions" in step:
            return await self.question(session_id, step)
        if "review" in step:
            return await self.review(session_id, step["review"])
        if "tool" in step:
            args = override if override is not None else step["args"]
            return await self.mcp_call(session_id, step["tool"], args)
        if "agent" in step:
            return await self.subagent(session_id, step)
        if "swarm" in step:
            return await self.own(
                session_id,
                "AgentSwarm",
                step["swarm"],
                {**step, "ask": False, "output": step["result"]},
            )
        return await self.own(
            session_id, step["builtin"], override if override is not None else step["args"], step
        )

    def call_id(self) -> str:
        """A call's id, unique in the session whichever of its processes makes it, as the
        model's call ids are."""
        self.calls += 1
        return f"0:call_{self.calls}_{PROCESS}"

    async def permitted(
        self, session_id: str, call_id: str, title: str, text: str, options: list[Any] | None = None
    ) -> str:
        """The client's answer: the option chosen, or ``cancelled`` — its own, in a permission
        mode that asks about nothing."""
        if mode_of(session_id)[1] != "manual":
            record("permissions", {"title": title, "answer": "unasked"})
            return "approve_once"
        answer = await self.conn.request_permission(
            options=options or CANONICAL,
            session_id=session_id,
            tool_call=ToolCallUpdate(
                tool_call_id=call_id,
                title=title,
                content=[ContentToolCallContent(type="content", content=text_block(text))],
            ),
        )
        chosen = answer.outcome.option_id if answer.outcome.outcome == "selected" else "cancelled"
        record("permissions", {"title": title, "answer": chosen})
        return chosen

    async def unseen(self, session_id: str, agent: str, call_id: str, asked: dict[str, Any]) -> str:
        """The permission request for a call ACP shows no update of (a sub-agent's, one of
        Kimi's own turn), as Kimi makes it: in ``agent``'s log first, with what Kimi shows its
        own user of the call (``interaction.request``, its ``display``: a command whole, a
        file's path and content); over ACP only the tool's name and a line that cuts a command
        at 50 characters. The client's answer."""
        title = f"mcp__trellis__{asked['tool']}" if "tool" in asked else asked["builtin"]
        args = asked["args"]
        if title == "Bash":
            command = args["command"]
            display = {
                "kind": "command",
                "command": command,
                "cwd": os.getcwd(),
                "language": "bash",
            }
            action = "Running: " + (command if len(command) <= 50 else command[:50] + "…")
        elif title == "Write":
            display = {"kind": "file_io", "operation": "write", **args}
            action = f"Writing {args['path']}"
        else:
            display = {"kind": "generic", "summary": title}
            action = f"Calling {title}"
        request = {"toolName": title, "action": action, "display": display}
        line = {"type": "interaction.request", "agentId": agent, "toolCallId": call_id}
        logged(session_id, agent, {**line, "request": request})
        return await self.permitted(
            session_id, f"0:{call_id}", title, f"Requesting approval to {action}"
        )

    async def started(
        self, session_id: str, call_id: str, title: str, args: dict[str, Any]
    ) -> None:
        await self.conn.session_update(
            session_id,
            ToolCallStart(
                session_update="tool_call",
                tool_call_id=call_id,
                title=title,
                kind="other",
                status="pending",
                content=[
                    ContentToolCallContent(type="content", content=text_block(json.dumps(args)))
                ],
            ),
        )

    async def progress(self, session_id: str, call_id: str, **fields: Any) -> None:
        await self.conn.session_update(
            session_id,
            ToolCallProgress(session_update="tool_call_update", tool_call_id=call_id, **fields),
        )

    async def decided(
        self, session_id: str, call_id: str, title: str, args: dict[str, Any]
    ) -> None:
        """Its own call asked about, unless the session's approvals cover it: Kimi keeps a
        call approved "for this session" (its literal rule: the tool with this command, or
        this path) and asks about the same one no more, in this process or a later one."""
        kept = session_dir(session_id) / "approved.json"
        approved = json.loads(kept.read_text()) if kept.exists() else []
        rule = f"{title}({args.get('command') or args.get('path') or ''})"
        if rule in approved:
            record("permissions", {"title": title, "answer": "approved for this session"})
            return
        chosen = await self.permitted(session_id, call_id, title, f"Requesting approval to {title}")
        if chosen == "approve_always":
            kept.write_text(json.dumps([*approved, rule]))
        if chosen == "cancelled":
            await self.progress(
                session_id,
                call_id,
                status="failed",
                raw_output=f'Tool "{title}" was not run because the approval request was cancelled.',
            )
            raise Stopped("paused")
        if chosen == "reject":
            await self.progress(
                session_id, call_id, status="failed", raw_output=f'Tool "{title}" was rejected.'
            )
            raise Stopped("rejected")

    async def own(
        self, session_id: str, name: str, args: dict[str, Any], step: dict[str, Any]
    ) -> str:
        call_id = self.call_id()
        await self.started(session_id, call_id, name, args)
        if step.get("ask"):
            await self.decided(session_id, call_id, name, args)
        await self.progress(
            session_id, call_id, status="in_progress", title=f"Running {name}", raw_input=args
        )
        if "wait" in step:
            await asyncio.sleep(step["wait"])
        output = step.get("output", f"{name} ran")
        if "background" in step:  # a command run in the background: launched
            output = self.launch(session_id, "process", step["background"], call_id)
        if self.client is not None and self.client.terminal and name == "Bash":
            # run in the client's terminal: Kimi's last update names it and carries no output
            output = await self.terminal(session_id, call_id, args["command"], step)
            await self.progress(session_id, call_id, status="completed")
            return output
        if self.client is not None and self.client.fs.write_text_file and name == "Write":
            await self.conn.write_text_file(
                session_id=session_id, path=args["path"], content=args["content"]
            )
            output = f"Wrote {len(args['content'])} bytes to {args['path']}"
        if self.client is not None and self.client.fs.read_text_file and name == "Read":
            try:
                read = await self.conn.read_text_file(
                    session_id=session_id,
                    path=args["path"],
                    line=args.get("line"),
                    limit=args.get("limit"),
                )
            except RequestError as exc:  # no such file: the call fails, as Kimi's does
                step = {**step, "fails": f"{args['path']} does not exist ({exc})"}
            else:
                output = read.content
        if "fails" in step:
            await self.progress(session_id, call_id, status="failed", raw_output=step["fails"])
            return step["fails"]
        await self.progress(session_id, call_id, status="completed", raw_output=output)
        return output

    def launch(self, session_id: str, kind: str, task: dict[str, Any], call_id: str) -> str:
        """A task the call ``call_id`` started in the background (``kind``: an ``agent``, a
        ``process``), as Kimi logs it (its call by its id less its turn), at work on its own:
        what its call answers (Kimi's words)."""
        number = len(self.tasks)
        task_id, agent = f"{kind[:5]}-{PROCESS}{number}", f"agent-{PROCESS}{number}"
        info = {"taskId": task_id, "kind": kind, "status": "running", "detached": True}
        info["parentToolCallId"] = call_id.partition(":")[2]
        if kind == "agent":
            info["agentId"] = agent
        logged(session_id, "main", {"type": "task.started", "agentId": "main", "info": info})
        work = asyncio.ensure_future(self.background(session_id, info, task))
        self.tasks.add(work)
        work.add_done_callback(self.tasks.discard)
        said = f"task_id: {task_id}\nstatus: running\n"
        if kind == "agent":
            said += f"agent_id: {agent}\nactual_subagent_type: coder\n"
        return said + "automatic_notification: true\n"

    async def background(self, session_id: str, info: dict[str, Any], task: dict[str, Any]) -> None:
        """A background task's work, its end, and the turn Kimi runs of its own for it."""
        await asyncio.sleep(task.get("after", 0))
        while "gate" in task and not os.path.exists(task["gate"]):  # noqa: ASYNC110, ASYNC240 - a file
            await asyncio.sleep(0.01)
        agent = info.get("agentId", "")
        if "usage" in task:
            logged_usage(session_id, agent, task["usage"])
        if "asks" in task:
            asked = task["asks"]
            chosen = await self.unseen(session_id, agent, f"call_bg_{info['taskId']}", asked)
            record("background", {"task": info["taskId"], "answer": chosen})
            if "tool" in asked and chosen == "approve_once":  # its harness tool's call
                await asyncio.sleep(task.get("call_after", 0))
                said = await self.called(session_id, asked["tool"], asked["args"])
                record("background", {"task": info["taskId"], "result": said})
        if "calls" in task:  # a harness tool's call it asks no permission for
            said = await self.called(session_id, task["calls"]["tool"], task["calls"]["args"])
            record("background", {"task": info["taskId"], "result": said})
        status = task.get("status", "failed" if "fails" in task else "completed")
        if status == "completed" and agent:
            line = {"type": "subagent.completed", "subagentId": agent}
            logged(session_id, "main", {**line, "resultSummary": task.get("result", "")})
        ended = {**info, "status": status, "stopReason": task.get("fails")}
        logged(session_id, "main", {"type": "task.terminated", "agentId": "main", "info": ended})
        self.busy = True  # Kimi's own turn for the task's end: the client sees none of it
        logged(session_id, "main", {"type": "turn.prompt", "origin": {"kind": "task"}})
        try:
            if "notified" in task:
                call_id = f"call_own_{info['taskId']}"
                chosen = await self.unseen(session_id, "main", call_id, task["notified"])
                record("own", {"task": info["taskId"], "answer": chosen})
            await asyncio.sleep(task.get("own_turn", 0))
        finally:
            self.busy = False
            logged_end(session_id, "completed")

    async def parallel(self, session_id: str, names: list[str]) -> str:
        """Its own read calls made together: each starts and runs, then each ends."""
        ids = [self.call_id() for _ in names]
        for call_id, name in zip(ids, names, strict=True):
            await self.started(session_id, call_id, name, {"path": name})
            await self.progress(session_id, call_id, status="in_progress", raw_input={"path": name})
        for call_id, name in zip(ids, names, strict=True):
            await self.progress(session_id, call_id, status="completed", raw_output={"read": name})
        return ", ".join(names)

    async def terminal(
        self, session_id: str, call_id: str, command: str, step: dict[str, Any]
    ) -> str:
        raw = os.environ.get("FAKE_ACP_RAW") == "1"  # the command as it is (a scripted sandbox's)
        created = await self.conn.create_terminal(
            session_id=session_id,
            command=command if raw else "/bin/bash",
            args=[] if raw else ["-c", f"cd '{os.getcwd()}' && {command}"],
            env=[EnvVariable(name="FAKE", value="1")],
            cwd=os.getcwd(),
            output_byte_limit=step.get("limit"),
        )
        terminal = created.terminal_id
        await self.progress(
            session_id, call_id, content=[{"type": "terminal", "terminalId": terminal}]
        )
        early = await self.conn.terminal_output(session_id=session_id, terminal_id=terminal)
        record("terminal", {"early": early.output})
        if step.get("kill"):
            await self.conn.kill_terminal(session_id=session_id, terminal_id=terminal)
        exited = await self.conn.wait_for_terminal_exit(session_id=session_id, terminal_id=terminal)
        output = await self.conn.terminal_output(session_id=session_id, terminal_id=terminal)
        await self.conn.release_terminal(session_id=session_id, terminal_id=terminal)
        record(
            "terminal",
            {
                "exit": exited.exit_code,
                "signal": exited.signal,
                "output": output.output,
                "truncated": output.truncated,
            },
        )
        return output.output

    async def mcp_call(self, session_id: str, tool: str, args: dict[str, Any]) -> str:
        name = f"mcp__trellis__{tool}"
        call_id = self.call_id()
        await self.started(session_id, call_id, name, args)
        await self.decided(session_id, call_id, name, args)
        await self.progress(session_id, call_id, status="in_progress", raw_input=args)
        text, failed = await self.called(session_id, tool, args, listed=True)
        status = "failed" if failed else "completed"
        await self.progress(session_id, call_id, status=status, raw_output=text)
        if text == "Waiting for a person's approval.":
            raise Stopped("paused")
        return text

    async def called(
        self, session_id: str, tool: str, args: dict[str, Any], *, listed: bool = False
    ) -> Any:
        """A call of the harness's MCP server (``trellis``), as the session was handed it: its
        text and whether it failed (``listed``: the tools it lists recorded first); a
        background sub-agent's: its text."""
        from mcp import Client
        from mcp.client.streamable_http import streamable_http_client
        from mcp.shared._httpx_utils import create_mcp_http_client

        server = next(s for s in self.servers[session_id] if s.name == "trellis")
        http = create_mcp_http_client(headers={h.name: h.value for h in server.headers})
        async with http, Client(streamable_http_client(server.url, http_client=http)) as client:
            if listed:
                found = await client.list_tools()
                record("listed", [t.name for t in found.tools])
            result = await client.call_tool(tool, args)
        text = "".join(getattr(c, "text", "") for c in result.content)
        return (text, result.is_error) if listed else text

    async def subagent(self, session_id: str, step: dict[str, Any]) -> str:
        args = step["agent"]
        call_id = self.call_id()
        await self.started(session_id, call_id, "Agent", args)
        await self.progress(
            session_id, call_id, status="in_progress", title="Launching agent", raw_input=args
        )
        if "background" in step:  # launched: the call answers at once, the task works on
            # launched: Kimi sends no update of the call's result, which only its model reads
            launched = self.launch(session_id, "agent", step["background"], call_id)
            if step["background"].get("first"):  # it ends before its launch is read
                await asyncio.wait(self.tasks)
            return launched
        if "asks" in step:  # a call of the sub-agent's own: no update, only its permission
            chosen = await self.unseen(session_id, "agent-0", f"call_sub_{call_id}", step["asks"])
            if chosen == "cancelled":
                await self.progress(
                    session_id,
                    call_id,
                    status="failed",
                    raw_output="the sub-agent's call was cancelled",
                )
                raise Stopped("paused")
        if step.get("fails"):
            await self.progress(session_id, call_id, status="failed", raw_output=step.get("result"))
            return step.get("result") or ""
        await self.progress(session_id, call_id, status="completed", raw_output=step["result"])
        return step["result"]

    async def question(self, session_id: str, step: dict[str, Any]) -> str:
        """Its questions to the user (Kimi's ``AskUserQuestion``), asked as Kimi asks them: a
        client that takes forms (``elicitation.form``) is sent every question in one form, a
        field each; any other the first question only, as a permission request. What the call
        answers: the answers by question (the one answer, of ``"question"``)."""
        questions = step.get("questions") or [
            {"question": step["question"], "options": [{"label": o} for o in step["options"]]}
        ]
        call_id = self.call_id()
        await self.started(session_id, call_id, "AskUserQuestion", {"questions": questions})
        forms = self.client is not None and self.client.elicitation is not None
        if forms and os.environ.get("FAKE_ACP_FORMS", "1") == "1":
            answers = await self.formed(session_id, call_id, questions)
        else:
            answers = await self.first(session_id, call_id, questions[0])
        output = json.dumps({"answers": answers}) if answers else "dismissed"
        await self.progress(session_id, call_id, status="completed", raw_output=output)
        if "questions" not in step and answers:
            return next(iter(answers.values()))
        return output

    async def formed(
        self, session_id: str, call_id: str, questions: list[dict[str, Any]]
    ) -> dict[str, str] | None:
        """Kimi's form of ``questions`` (``questionRequestToElicitationParams``), and the
        answers it reads of the client's (``elicitationResponseToQuestionAnswers``)."""
        properties: dict[str, Any] = {}
        for n, asked in enumerate(questions):
            picks = [
                {"const": o["label"], "title": o["label"], "description": o.get("description")}
                for o in asked["options"]
            ]
            field = {"title": asked.get("header") or asked["question"]}
            if asked.get("multi_select"):
                field |= {"type": "array", "minItems": 1, "items": {"anyOf": picks}}
            else:
                field |= {"type": "string", "oneOf": picks}
            properties[f"q{n}"] = field
        schema = {"type": "object", "properties": properties, "required": list(properties)}
        answer = await self.conn.create_elicitation(
            "\n".join(asked["question"] for asked in questions),
            ElicitationFormSessionMode(
                session_id=session_id,
                tool_call_id=call_id,
                requested_schema=ElicitationSchema.model_validate(schema),
            ),
        )
        record("forms", {"schema": schema, "answer": answer.model_dump(exclude_none=True)})
        if answer.action == "cancel":
            raise Stopped("paused")
        content = getattr(answer, "content", None) or {}
        answers = {}
        for n, asked in enumerate(questions):
            value = content.get(f"q{n}")
            if value:
                answers[asked["question"]] = ", ".join(value) if isinstance(value, list) else value
        return answers or None

    async def first(
        self, session_id: str, call_id: str, asked: dict[str, Any]
    ) -> dict[str, str] | None:
        """Kimi's fallback for a client that takes no form: the first question, its options
        and ``Skip``, as a permission request."""
        labels = [o["label"] for o in asked["options"]]
        options = [
            PermissionOption(option_id=f"q0_opt_{n}", name=label, kind="allow_once")
            for n, label in enumerate(labels)
        ]
        options.append(PermissionOption(option_id="q0_skip", name="Skip", kind="reject_once"))
        chosen = await self.permitted(
            session_id, call_id, "AskUserQuestion", asked["question"], options
        )
        if chosen == "cancelled":
            raise Stopped("paused")
        if chosen == "q0_skip":
            return None
        return {asked["question"]: labels[int(chosen.rsplit("_", 1)[1])]}

    async def review(self, session_id: str, plan: dict[str, Any]) -> str:
        """Its plan presented for review (Kimi's ``ExitPlanMode``): what the call answers."""
        labels = plan.get("options") or []
        call_id = self.call_id()
        args = {"options": [{"label": label} for label in labels]} if labels else {}
        await self.started(session_id, call_id, "ExitPlanMode", args)
        approve = [
            PermissionOption(option_id=f"plan_opt_{n}", name=label, kind="allow_once")
            for n, label in enumerate(labels)
        ] or [PermissionOption(option_id="plan_approve", name="Approve", kind="allow_once")]
        options = [
            *approve,
            PermissionOption(option_id="plan_revise", name="Revise", kind="reject_once"),
            PermissionOption(
                option_id="plan_reject_and_exit", name="Reject and Exit", kind="reject_once"
            ),
        ]
        said = f"{plan['plan']}\nRequesting approval to Presenting plan and exiting plan mode"
        chosen = await self.permitted(session_id, call_id, "ExitPlanMode", said, options)
        if chosen == "cancelled":
            output = "Plan approval dismissed. Plan mode remains active."
            await self.progress(session_id, call_id, status="completed", raw_output=output)
            raise Stopped("paused")
        if chosen == "plan_revise":
            output = "User requested revisions. Plan mode remains active."
        elif chosen == "plan_reject_and_exit":
            logged(session_id, "main", {"type": "plan_mode.exit", "agentId": "main"})
            output = "Plan rejected by user. Plan mode deactivated."
        else:
            logged(session_id, "main", {"type": "plan_mode.exit", "agentId": "main"})
            picked = next((o.name for o in options if o.option_id == chosen), "")
            output = "Exited plan mode. " + (f"Selected approach: {picked}" if labels else "")
        await self.progress(session_id, call_id, status="completed", raw_output=output)
        return output


if __name__ == "__main__":
    asyncio.run(run_agent(cast(Any, FakeAgent()), use_unstable_protocol=True))
