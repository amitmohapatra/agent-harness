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
* ``{"question": "...", "options": [...]}`` its question to the user (``AskUserQuestion``);
* ``{"usage": {...}, "agent": "main"}`` a model request's usage, in its usage log under
  ``KIMI_CODE_HOME`` (Kimi's ``usage.record`` line);
* ``{"stall": seconds}`` sends nothing for that long; ``{"crash": "why"}`` the process ends;
  ``{"stop": "refusal"}`` ends the turn so; ``{"prompt_usage": {...}}`` puts ACP's own usage on
  the prompt's answer; ``{"plan": true}`` a plan update; ``{"wait": seconds}`` a tool call that
  runs that long (a cancel stops it).

``FAKE_ACP_RECORD`` names a file it writes what it was started and asked with to (its
environment's ``KIMI_*`` and ``HOME``, its working directory, the client's capabilities, each
session's MCP servers and how it began, the prompts, the permission answers).
``FAKE_ACP_VERSION`` is the version it reports (``2.1.1``); ``FAKE_ACP_RESUME=0`` offers no
``session/resume`` (only ``session/load``, which replays the session's text).
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
    ContentToolCallContent,
    EnvVariable,
    Implementation,
    InitializeResponse,
    LoadSessionResponse,
    NewSessionResponse,
    PermissionOption,
    PlanEntry,
    PromptResponse,
    ResumeSessionResponse,
    SetSessionModeResponse,
    ToolCallProgress,
    ToolCallStart,
    ToolCallUpdate,
    Usage,
)

SCRIPT: list[dict[str, Any]] = json.loads(os.environ.get("FAKE_ACP_SCRIPT", "[]"))
RECORD = os.environ.get("FAKE_ACP_RECORD")
VERSION = os.environ.get("FAKE_ACP_VERSION", "2.1.1")
HOME = Path(os.environ.get("KIMI_CODE_HOME") or Path.home() / ".kimi-code")
CANONICAL = [
    PermissionOption(option_id="approve_once", name="Approve once", kind="allow_once"),
    PermissionOption(
        option_id="approve_always", name="Approve for this session", kind="allow_always"
    ),
    PermissionOption(option_id="reject", name="Reject", kind="reject_once"),
]
EDITED = re.compile(r"Call (?P<tool>\S+) again now with exactly these arguments: (?P<args>\{.*\})$")


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


def logged_end(session_id: str) -> None:
    """The turn's end, as Kimi logs it after the turn's usage (``turn.ended``)."""
    path = session_dir(session_id) / "agents" / "main" / "wire.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as file:
        file.write(json.dumps({"type": "turn.ended", "agentId": "main"}) + "\n")


class Stopped(Exception):
    """The turn stops here: the step did not run (a pause, a rejection, an edit)."""


class FakeAgent:
    def __init__(self) -> None:
        self.conn: Any = None
        self.client: Any = None
        self.servers: dict[str, list[Any]] = {}
        self.cancelled: dict[str, asyncio.Event] = {}
        self.calls = 0

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
            caps["sessionCapabilities"] = {"resume": {}}
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
        record(
            "sessions",
            {
                "how": "new",
                "id": session,
                "servers": [s.model_dump(by_alias=True) for s in mcp_servers or []],
            },
        )
        return NewSessionResponse(session_id=session)

    async def resume_session(
        self, cwd: str, session_id: str, mcp_servers: list[Any] | None = None, **kw: Any
    ) -> Any:
        if not (session_dir(session_id) / "state.json").exists():
            raise RequestError.resource_not_found(session_id)
        self.servers[session_id] = list(mcp_servers or [])
        record("sessions", {"how": "resume", "id": session_id})
        return ResumeSessionResponse()

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
        return LoadSessionResponse()

    async def set_session_mode(self, mode_id: str, session_id: str, **kw: Any) -> Any:
        record("modes", mode_id)
        return SetSessionModeResponse()

    async def cancel(self, session_id: str, **kw: Any) -> None:
        record("cancels", session_id)
        self.cancelled.setdefault(session_id, asyncio.Event()).set()

    async def prompt(self, prompt: list[Any], session_id: str, **kw: Any) -> Any:
        try:
            return await self._prompt(prompt, session_id)
        finally:
            logged_end(session_id)

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
        if "question" in step:
            return await self.question(session_id, step)
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
        self.calls += 1
        return f"0:call_{self.calls}"

    async def permitted(
        self, session_id: str, call_id: str, title: str, text: str, options: list[Any] | None = None
    ) -> str:
        """The client's answer: the option chosen, or ``cancelled``."""
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
        chosen = await self.permitted(session_id, call_id, title, f"Requesting approval to {title}")
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
        if self.client is not None and self.client.terminal and name == "Bash":
            output = await self.terminal(session_id, call_id, args["command"], step)
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
        from mcp import Client
        from mcp.client.streamable_http import streamable_http_client
        from mcp.shared._httpx_utils import create_mcp_http_client

        name = f"mcp__trellis__{tool}"
        call_id = self.call_id()
        await self.started(session_id, call_id, name, args)
        await self.decided(session_id, call_id, name, args)
        await self.progress(session_id, call_id, status="in_progress", raw_input=args)
        server = next(s for s in self.servers[session_id] if s.name == "trellis")
        http = create_mcp_http_client(headers={h.name: h.value for h in server.headers})
        async with http, Client(streamable_http_client(server.url, http_client=http)) as client:
            listed = await client.list_tools()
            record("listed", [t.name for t in listed.tools])
            result = await client.call_tool(tool, args)
        text = "".join(getattr(c, "text", "") for c in result.content)
        status = "failed" if result.is_error else "completed"
        await self.progress(session_id, call_id, status=status, raw_output=text)
        if text == "Waiting for a person's approval.":
            raise Stopped("paused")
        return text

    async def subagent(self, session_id: str, step: dict[str, Any]) -> str:
        args = step["agent"]
        call_id = self.call_id()
        await self.started(session_id, call_id, "Agent", args)
        await self.progress(
            session_id, call_id, status="in_progress", title="Launching agent", raw_input=args
        )
        if "asks" in step:  # a call of the sub-agent's own: no update, only its permission
            asked = step["asks"]
            chosen = await self.permitted(
                session_id,
                "0:call_sub",
                asked["builtin"],
                f"Requesting approval to Running: {json.dumps(asked['args'])}",
            )
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
        options = [
            PermissionOption(option_id=f"q0_opt_{n}", name=label, kind="allow_once")
            for n, label in enumerate(step["options"])
        ]
        options.append(PermissionOption(option_id="q0_skip", name="Skip", kind="reject_once"))
        chosen = await self.permitted(
            session_id, "ask-user", "AskUserQuestion", step["question"], options
        )
        if chosen == "cancelled":
            raise Stopped("paused")
        if chosen == "q0_skip":
            return "dismissed"
        return step["options"][int(chosen.rsplit("_", 1)[1])]


if __name__ == "__main__":
    asyncio.run(run_agent(cast(Any, FakeAgent()), use_unstable_protocol=True))
