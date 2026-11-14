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
``updatedInput``), ``deny`` refuses it, ``continue: false`` stops the turn, ``ask`` asks
``can_use_tool`` whatever the rules (with their ``updatedInput``); with no decision the
permission rules go on (``--allowedTools``, then ``can_use_tool``). After a call that ran, the
``PostToolUse`` hooks are called with its result, or the ``PostToolUseFailure`` hooks with its
error (``{"builtin": ..., "fails": "why"}`` scripts a built-in that fails). ``{"stall":
seconds}`` is a step on which the "model" sends nothing for that long (then goes on). A call's
``"again": {...}`` is the arguments the "model" makes it with when a resumed session has it
make that call again (a real model rarely repeats a call word for word).

Spending: every result carries usage (:data:`STEP_TOKENS` a step) and a cost
(``FAKE_CLAUDE_COST`` a step); past ``--max-budget-usd`` the query stops with the real CLI's
``error_max_budget_usd`` result. ``{"cut": true}`` is a step whose reply reached the model's
output limit with no answer: the query ends with the result the real CLI gives once its own
recovery gave up (``success``, no text, ``stop_reason`` ``max_tokens``).
``--include-partial-messages`` sends each text as word deltas
(``stream_event``) before its message.

A query runs in a session (its id on every message), kept in ``FAKE_CLAUDE_SESSIONS`` (else
the temporary directory): the steps done so far — a tool call the run paused at is not done
(one denied or failed is: the "model" read why).
``--resume=<id>`` goes on after them, as the real CLI continues a conversation; a session it
does not hold is the real CLI's error result. Stdin closed (the SDK's process died) with no
subagent at work in the background, the process ends; with one it goes on to that work's end
(a call it makes has nobody to decide it: it does not run), then ends, as the bundled CLI does. With ``--session-mirror`` (the SDK has a
``SessionStore``) every step is mirrored as the real CLI mirrors its transcript (a
``transcript_mirror`` frame), and a resume the SDK materialised (``CLAUDE_CONFIG_DIR``) goes on
from that transcript — whatever this machine's own sessions hold. ``--max-turns`` stops a query after that many
steps, with the real CLI's ``error_max_turns`` result. A resumed session sent a prompt of its own
(not the harness's "This task was paused" — the conversation's next turn) runs the script anew as
the session's next turn; the record's ``earlier`` is the prompts its transcript holds. The
``additionalContext`` the ``PostToolUse`` hooks answer is appended to ``<record>.added`` (one
JSON string a line): what the "model" is told after a call.

One process, many turns (the SDK's ``ClaudeSDKClient``: stdin stays open). Every prompt a
process is sent after its first is its session's next turn, in the same session:
``FAKE_CLAUDE_TURNS`` (a JSON list of scripts) scripts each turn of the session by its number
(the last one for any later turn), else ``FAKE_CLAUDE_SCRIPT`` scripts them all. Each turn's
record is also a line of ``<record>.turns`` with the process's ``pid``, so a test sees which
process answered and whether it was started with ``--resume``; ``heard`` (``{heard}`` in a
scripted text) is every prompt the session's "model" has been given, injected ones included.
A result's cost is the process's running total and its usage the turn's own — the main
thread's: a subagent's steps are in the cost, and in the usage of its own replies only — as
the bundled CLI (2.1.283) reports them when driven against a scripted model.

Background work, as far as the SDK's message types say its shape (``task_started``,
``task_notification``, a turn of the session's own whose result has ``origin: {"kind":
"task-notification"}``):
``{"background": {"id": "t1", "agent": "researcher", "prompt": "..."}}`` is an ``Agent`` call
with ``run_in_background`` (``"shell": "<command>"`` instead: a ``Bash`` one) that returns at
once and leaves task ``t1`` running. Files in ``FAKE_CLAUDE_TASKS`` drive a running task from
the test: ``t1.call`` (a JSON step) is a tool call the task makes, as a subagent's — its
hooks and its permission request name the subagent by the task's id (``agent_id``), and its
result is written to ``t1.called``; ``t1.done`` (a summary) ends the task: the subagent's last
reply, a ``task_updated`` and a ``task_notification``, then — after the turn under way, if any
— a turn the session injects,
scripted by ``FAKE_CLAUDE_INJECTED`` (``{summary}``: the task's). ``t1.failed`` (why) ends it
``failed``, and ``t1.stop`` stops it as the "model"'s own ``TaskStop`` does; a ``stop_task``
control request stops the task it names (named in ``t1.stopped``), leaving the others: either
way a ``task_updated`` ``killed``, a ``task_notification`` ``stopped`` and an injected turn,
as the bundled CLI sends them. An ``interrupt`` control
request ends the turn under way with the result of an interrupted turn — and stops every
background task, as the bundled CLI does (``task_updated`` ``killed``, ``task_notification``
``stopped``); a hook's ``continue: false`` and a denied permission's ``interrupt`` end the turn
and leave the tasks, as it does too. ``FAKE_CLAUDE_STARTS_AFTER`` (seconds) is a CLI slow to
start, and a step's ``"quit": true`` one that ends of itself after the turn's result; its pid
is in the file ``FAKE_CLAUDE_PID`` names from its first moment. The frames and these rules
were seen from the bundled CLI driven against a scripted model; no real model was called.
"""

from __future__ import annotations

import atexit
import fcntl
import json
import os
import re
import select
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
#: How the harness begins the prompt that asks for one last answer at the turn limit
#: (``budget.TURN_LIMIT``): the "model" answers it with ``FAKE_CLAUDE_AT_LIMIT``, calling no tool
#: — unless ``FAKE_CLAUDE_AT_LIMIT_CALLS`` is a step it makes first (its answer: that call's result).
LIMIT_PROMPT = "This turn has reached its limit"
#: The tool the "model" hands a structured answer to (``--json-schema``): the CLI runs it
#: without asking, and the turn's result carries what it was given (``structured_output``).
STRUCTURED = "StructuredOutput"
#: The slash command that compacts the conversation (its ``PreCompact`` hooks run).
COMPACT = "/compact"
#: The tools ``--permission-mode acceptEdits`` lets run without asking.
EDITS = frozenset({"Edit", "MultiEdit", "Write", "NotebookEdit"})
#: The input and output tokens each step of a query reports.
STEP_TOKENS = (10, 5)


#: ``"orphaned"``: the SDK is gone (stdin closed while background work runs) — nothing is
#: sent, nobody answers.
_SDK = {"orphaned": False}


def send(message: dict[str, Any]) -> None:
    if _SDK["orphaned"]:
        return
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def orphaned(cli: Cli) -> None:
    """Stdin closed (the SDK's process died) while a background subagent works: as the bundled
    CLI (2.1.283) does, its work goes on to its end — a call it makes has nobody to decide it
    and does not run — and then the process ends (its parent ``SIGKILL``-ed, driven against a
    scripted model on this machine). A background shell command ends with the process, as
    before (not probed)."""
    _SDK["orphaned"] = True
    while any(task["agent"] != "shell" for task in cli.tasks.values()):
        cli.triggers()
        time.sleep(0.02)


#: What :func:`read` answers when nothing came within its ``timeout``.
NOTHING: dict[str, Any] = {"type": "nothing"}
_unread = bytearray()


def read(timeout: float | None = None) -> dict[str, Any] | None:
    """The next message on stdin (``None``: it closed; :data:`NOTHING`: none came within
    ``timeout`` seconds)."""
    while b"\n" not in _unread:
        if timeout is not None and not select.select([0], [], [], timeout)[0]:
            return NOTHING
        chunk = os.read(0, 1 << 20)
        if not chunk:
            return None
        _unread.extend(chunk)
    line, _, rest = bytes(_unread).partition(b"\n")
    _unread[:] = rest
    return json.loads(line)


class Interrupted(Exception):
    """The SDK interrupted the turn under way."""


def argument(name: str) -> str | None:
    args = sys.argv[1:]
    joined = [a.split("=", 1)[1] for a in args if a.startswith(f"{name}=")]
    return joined[0] if joined else args[args.index(name) + 1] if name in args else None


def replaced(path: str, data: Any) -> None:
    """``path`` holds ``data`` (JSON) whole: written beside it, then put in its place — a
    test, or another process of the stand-in, never reads it half written."""
    part = f"{path}.{os.getpid()}.part"
    with open(part, "w") as handle:
        json.dump(data, handle)
    os.replace(part, path)


def appended(path: str, data: Any) -> None:
    """``data`` (JSON) a line of ``path``, under a lock on it: the stand-in's processes that
    run at once (a matrix cell's runs) each write whole lines."""
    with open(path, "a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            handle.write(json.dumps(data) + "\n")
            handle.flush()
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


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
        #: the hooks the SDK registered, by event (its ``initialize`` request), and the rest
        #: of that request (``agents``, ``skills``...: what the SDK sends there, not in flags)
        self.hooks: dict[str, Any] = {}
        self.initialized: dict[str, Any] = {}
        #: the subagent working now (its ``agent_id`` and ``agent_type``), as the real CLI
        #: names it in the hooks of the tool calls it makes
        self.working: dict[str, str] = {}
        #: the session this process runs (one for all its turns), once its first turn began
        self.session: str | None = None
        #: whether the last tool call ended without running, or failed (not paused): done
        self.denied = False
        #: every prompt the session's "model" was given: the user's and the injected ones
        self.heard: list[str] = []
        #: the prompts sent while a turn was under way, and the turns the session owes itself
        #: (a finished task's notification): run in order once the turn ends
        self.queued: list[Any] = []
        self.owed: list[str] = []
        #: the background tasks under way, by id: their call's id and who works in them
        self.tasks: dict[str, dict[str, str]] = {}
        #: the SDK interrupted the turn under way
        self.stopping = False
        #: the steps this process took, in every turn (a result's running cost), and how many
        #: it had taken when the turn under way began (the rest are the turn's own: its usage)
        self.steps = 0
        self.began = 0
        self.in_turn = False
        #: how many of those steps subagents took: a result's usage is the main thread's own
        self.helped = 0
        #: what the process exits with when its input ends: 1 once a turn ended at a limit
        self.status = 0
        #: the structured answer the turn under way gave (:data:`STRUCTURED`)
        self.structured: dict[str, Any] | None = None
        #: the steps the session took before this process resumed it: a result's cost is
        #: the session's (the bundled CLI's resumed query reports the cost of the turns
        #: before it too), while ``--max-budget-usd`` caps what this process spends
        self.before = 0

    def aside(self, message: dict[str, Any]) -> None:
        """A message that came while another was waited for: a prompt is queued, a control
        request answered (an ``interrupt`` stops the turn under way)."""
        if message.get("type") == "user":
            self.queued.append(message["message"]["content"])
        elif message.get("type") == "control_request":
            if message["request"].get("subtype") == "interrupt":
                self.stopping = self.in_turn
                self.stopped()
            self.asked_to_stop(message)
            acknowledge(message)

    def asked_to_stop(self, message: dict[str, Any]) -> None:
        """A ``stop_task`` control request stops the task it names, and no other."""
        request = message["request"]
        if request.get("subtype") == "stop_task" and request["task_id"] in self.tasks:
            folder = os.environ.get("FAKE_CLAUDE_TASKS")
            if folder:
                (Path(folder) / f"{request['task_id']}.stopped").write_text("stop_task")
            self.stopped(request["task_id"])
            self.owed.append("stopped")

    def stopped(self, only: str | None = None) -> None:
        """An interrupt stops what runs in the background, as the real CLI's does (``only``:
        the one task a ``stop_task`` names)."""
        for task_id, task in list(self.tasks.items()):
            if only not in (None, task_id):
                continue
            del self.tasks[task_id]
            frame = {"type": "system", "task_id": task_id, "session_id": self.session}
            send({**frame, "subtype": "task_updated", "patch": {"status": "killed"}})
            send(
                {
                    **frame,
                    "subtype": "task_notification",
                    "tool_use_id": task["tool_id"],
                    "status": "stopped",
                    "output_file": "",
                    "summary": "stopped",
                    "uuid": str(uuid.uuid4()),
                }
            )

    def wait(self, seconds: float) -> None:
        """Nothing is sent for ``seconds``, unless the turn is interrupted; a running task's
        triggers are served meanwhile."""
        until = time.monotonic() + seconds
        while time.monotonic() < until:
            message = read(timeout=0.02)
            if message is None:
                sys.exit(0)
            if message is not NOTHING:
                self.aside(message)
            if self.stopping:
                raise Interrupted
            self.triggers()

    def triggers(self) -> None:
        """What the test asked of a running task (``FAKE_CLAUDE_TASKS``): a tool call it
        makes (``<id>.call``), or its end (``<id>.done``)."""
        folder = os.environ.get("FAKE_CLAUDE_TASKS")
        if not folder or not self.tasks or self.session is None:
            return
        for task_id, task in list(self.tasks.items()):
            call = Path(folder) / f"{task_id}.call"
            if call.exists() and _SDK["orphaned"]:  # nobody to decide it: it does not run
                call.unlink()
                unanswered = {"result": "no answer from the SDK: the call did not run", "ok": False}
                (Path(folder) / f"{task_id}.called").write_text(json.dumps(unanswered))
            elif call.exists():
                item = json.loads(call.read_text())
                call.unlink()
                task["calls"] = str(int(task.get("calls", "0")) + 1)
                was, self.working = (
                    self.working,
                    {"agent_id": task_id, "agent_type": task["agent"]},
                )
                try:
                    said, ok, _ = self.tool(
                        self.session,
                        0,
                        item,
                        tool_id=f"{task['tool_id']}_{task['calls']}",
                        parent=task["tool_id"],
                    )
                finally:
                    self.working = was
                self.steps += 1
                self.helped += 1
                (Path(folder) / f"{task_id}.called").write_text(
                    json.dumps({"result": said, "ok": ok})
                )
            if task_id not in self.tasks:  # stopped while its call waited
                continue
            if (Path(folder) / f"{task_id}.stop").exists():
                (Path(folder) / f"{task_id}.stop").unlink()
                self.stopped(task_id)
                self.owed.append("stopped")
                continue
            done, failed = Path(folder) / f"{task_id}.done", Path(folder) / f"{task_id}.failed"
            ended = done if done.exists() else failed if failed.exists() else None
            if ended is not None:
                summary = ended.read_text()
                ended.unlink()
                del self.tasks[task_id]
                self.steps += 1  # what the task's own model steps spent
                self.helped += 1
                status = "completed" if ended is done else "failed"
                frame = {"type": "system", "task_id": task_id, "session_id": self.session}
                if task["agent"] != "shell":  # the subagent's last reply
                    said = [{"type": "text", "text": summary}]
                    send(assistant(self.session, said, parent=task["tool_id"]))
                send({**frame, "subtype": "task_updated", "patch": {"status": status}})
                send(
                    {
                        **frame,
                        "subtype": "task_notification",
                        "tool_use_id": task["tool_id"],
                        "status": status,
                        "output_file": f"/tmp/fake-claude/{task_id}.output",
                        "summary": summary,
                        "usage": {"total_tokens": 15, "tool_uses": 0, "duration_ms": 1},
                        "uuid": str(uuid.uuid4()),
                    }
                )
                self.owed.append(summary)

    def injected(self, summary: str) -> None:
        """The turn the session runs by itself once a background task ended: the "model"'s
        reply to its notification (``FAKE_CLAUDE_INJECTED``) and that turn's result, whose
        origin says whose turn it was. No prompt is sent for it: the bundled CLI sends none."""
        assert self.session is not None
        origin = {"kind": "task-notification"}
        prompt = f"<task-notification>{summary}</task-notification>"
        self.heard.append(prompt)
        state = json.loads((SESSIONS / f"{self.session}.json").read_text())
        mirror(self.session, state["done"], state.get("turn", 0), prompt=prompt, key=summary)
        script = json.loads(
            os.environ.get("FAKE_CLAUDE_INJECTED") or '[{"text": "Noted: {summary}"}]'
        )
        last = read_ = ""
        self.began = self.steps - self.helped
        self.in_turn = True
        try:
            for n, scripted in enumerate(script):
                self.steps += 1
                item = {**scripted, "id": f"toolu_injected_{len(self.heard)}_{n}"}
                if "text" in item:
                    item["text"] = item["text"].replace("{summary}", summary)
                said, read_, _, interrupted, _ = self.step(self.session, n, item, read_, "")
                last = said if said is not None else last
                if interrupted:
                    break
        except Interrupted:
            pass
        finally:
            self.in_turn = self.stopping = False
        send({**self.ended(self.session, last=last, turns=len(script)), "origin": origin})

    def background(self, session: str, step: int, item: dict[str, Any]) -> tuple[str, bool, bool]:
        """A call that leaves a task running in the background: an ``Agent`` call (or, with
        ``shell``, a ``Bash`` one) with ``run_in_background``, through the hooks and the
        permission rules as any call; then the task's start, and the call's result at once."""
        spec = item["background"]
        tool_id = f"toolu_bg_{spec['id']}"
        if "shell" in spec:
            name, kind = "Bash", "local_bash"
            args = {"command": spec["shell"], "run_in_background": True}
        else:
            name, kind = "Agent", "local_agent"
            args = {
                "description": spec["prompt"],
                "prompt": spec["prompt"],
                "subagent_type": spec["agent"],
                "run_in_background": True,
            }
        send(assistant(session, [{"type": "tool_use", "id": tool_id, "name": name, "input": args}]))
        permission = self.allowed(session, name, args, tool_id)
        if permission.get("behavior", "allow") != "allow":
            content = [{"type": "text", "text": permission.get("message", "denied")}]
            self.result(session, tool_id, content)
            waits = content[0]["text"] == WAITING
            return content[0]["text"], False, waits or bool(permission.get("interrupt"))
        ran = permission.get("updatedInput", args)
        self.tasks[spec["id"]] = {"tool_id": tool_id, "agent": spec.get("agent", "shell")}
        send(
            {
                "type": "system",
                "subtype": "task_started",
                "task_id": spec["id"],
                "tool_use_id": tool_id,
                "description": spec.get("prompt") or spec.get("shell"),
                "task_type": kind,
                "is_backgrounded": True,
                "uuid": str(uuid.uuid4()),
                "session_id": session,
            }
        )
        content = [{"type": "text", "text": f"Async task launched: {spec['id']}"}]
        self.post_tool_use(session, name, ran, tool_id, content=content, error=None)
        self.result(session, tool_id, content)
        return content[0]["text"], True, False

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

    def hooked(self, event: str, session: str | None, **fields: Any) -> list[dict[str, Any]]:
        """The hooks registered for ``event`` called with ``fields`` (``Stop``,
        ``SubagentStart``, ``SubagentStop``, ``PermissionRequest``, ``PreCompact``: when and
        with what the bundled CLI was seen calling each): what each answered."""
        given = {
            "session_id": session,
            "transcript_path": "",
            "cwd": os.getcwd(),
            "hook_event_name": event,
            **fields,
        }
        return [
            self.control(
                {
                    "subtype": "hook_callback",
                    "callback_id": callback,
                    "input": given,
                    "tool_use_id": None,
                }
            )
            for matcher in self.hooks.get(event) or []
            for callback in matcher.get("hookCallbackIds") or []
        ]

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
                    appended(f"{record}.added", added)

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
            else:
                self.aside(message)
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

    def permission(
        self, name: str, args: dict[str, Any], tool_id: str, *, ask: bool = False
    ) -> dict[str, Any]:
        """What the SDK's ``can_use_tool`` says about a call — not asked, as the real CLI does
        not ask it, for a tool in ``--allowedTools``, any tool under ``bypassPermissions`` and
        an edit under ``acceptEdits`` (allowed then), unless a hook said ``ask``."""
        allowed = (argument("--allowedTools") or "").split(",")
        mode = argument("--permission-mode")
        if argument("--permission-prompt-tool") != "stdio" or (
            not ask
            and (
                name in allowed
                or name == STRUCTURED
                or mode == "bypassPermissions"
                or (mode == "acceptEdits" and name in EDITS)
            )
        ):
            return {"behavior": "allow"}
        suggested = [
            {
                "type": "addRules",
                "rules": [{"toolName": name, "ruleContent": None}],
                "behavior": "allow",
                "destination": "session",
            }
        ]
        self.hooked(
            "PermissionRequest",
            self.session,
            tool_name=name,
            tool_input=args,
            permission_suggestions=suggested,
            **self.working,
        )
        answered = self.control(
            {
                "subtype": "can_use_tool",
                "tool_name": name,
                "input": args,
                "permission_suggestions": suggested,
                "tool_use_id": tool_id,
                **{k: v for k, v in self.working.items() if k == "agent_id"},
            }
        )
        record = os.environ.get("FAKE_CLAUDE_RECORD")
        if record:  # what the SDK answered, as the CLI reads it
            appended(f"{record}.permissions", {"tool": name, "answer": answered})
        return answered

    def resumes(self) -> str | None:
        """The session this process was started to go on with: the one ``--resume`` names,
        or with ``--continue`` the one this machine holds that was written last; with
        ``--fork-session`` a copy of it under a new id (``--session-id``, else its own)."""
        found = argument("--resume")
        if found is None and "--continue" in sys.argv:
            kept = sorted(SESSIONS.glob("*.json"), key=lambda p: p.stat().st_mtime)
            found = kept[-1].stem if kept else None
        if found is None or "--fork-session" not in sys.argv:
            return found
        forked = argument("--session-id") or str(uuid.uuid4())
        if (SESSIONS / f"{found}.json").exists():
            (SESSIONS / f"{forked}.json").write_text((SESSIONS / f"{found}.json").read_text())
        return forked

    def turn(self, prompt: Any) -> None:
        script = json.loads(os.environ.get("FAKE_CLAUDE_SCRIPT", '[{"text": "ok"}]'))
        # this process's first turn resumes the session it was started with, or begins one;
        # every later prompt is that session's next turn
        first = self.session is None
        self.began = self.steps - self.helped
        resumed = (self.resumes() if first else self.session) or None
        session = self.session = resumed or argument("--session-id") or str(uuid.uuid4())
        if str(prompt).strip() == COMPACT:  # a slash command: no prompt hook, no model turn
            self.hooked("PreCompact", session, trigger="manual", custom_instructions=None)
            send(self.ended(session, last="", turns=0))
            return
        if first:
            self.heard = prompts(session)
            self.before = spent_before(session) if resumed is not None else 0
        earlier = list(self.heard)
        if not str(prompt).startswith((PAUSED_PROMPT, LIMIT_PROMPT)):
            self.heard.append(str(prompt))
        context = self.prompt_context(session, prompt)
        self.recorded(session, prompt, context, earlier, first=first)
        kept = SESSIONS / f"{session}.json"
        restored = mirrored(session) if resumed is not None and first else None
        if restored is None and resumed is not None and first and "--fork-session" in sys.argv:
            restored = mirrored(str(argument("--resume")))  # the transcript it was forked from
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
        if str(prompt).startswith(LIMIT_PROMPT):  # one last answer, no tool
            self.at_limit(session, turn, context)
            return
        if resumed is not None and not str(prompt).startswith(PAUSED_PROMPT):
            done, turn = 0, turn + 1  # the conversation's next turn, in the same session
            kept.write_text(json.dumps({"done": 0, "turn": turn}))
            mirror(session, 0, turn, prompt=str(prompt))
        by_turn = json.loads(os.environ.get("FAKE_CLAUDE_TURNS") or "[]")
        if by_turn:
            script = by_turn[min(turn, len(by_turn) - 1)]
        script = made_again(script, done) if str(prompt).startswith(PAUSED_PROMPT) else script
        self.in_turn = True
        try:
            self.played(session, script, done, turn, context)
        except Interrupted:
            send(self.ended(session, "Request interrupted by user"))
        finally:
            self.in_turn = self.stopping = False

    def at_limit(self, session: str, turn: int, context: str) -> None:
        """The one last answer at the turn limit: ``FAKE_CLAUDE_AT_LIMIT``, no tool — or a
        "model" that calls ``FAKE_CLAUDE_AT_LIMIT_CALLS`` all the same, then says what it
        was told."""
        called = os.environ.get("FAKE_CLAUDE_AT_LIMIT_CALLS")
        if called:
            self.played(session, [json.loads(called), {"text": "{last}"}], 0, turn, context)
            return
        said = os.environ.get("FAKE_CLAUDE_AT_LIMIT", "What I found before the limit.")
        self.steps += 1
        send(assistant(session, [{"type": "text", "text": said}]))
        send(self.ended(session, last=said, turns=1))

    def ended(self, session: str, error: str | None = None, **fields: Any) -> dict[str, Any]:
        """The result of the turn under way: its own steps' usage, the process's cost."""
        own = self.steps - self.helped - self.began
        self.kept()
        if "--session-mirror" in sys.argv:  # the transcript holds what the session cost
            entry = {"type": "progress", "uuid": f"{session}-spent-{self.before + self.steps}"}
            path = projects() / "fake-project" / f"{session}.jsonl"
            mirrored_ = {**entry, "spent": self.before + self.steps}
            send({"type": "transcript_mirror", "filePath": str(path), "entries": [mirrored_]})
        structured, self.structured = self.structured, None
        if structured is not None and error is None:
            fields = {**fields, "last": json.dumps(structured), "structured": structured}
        return result(session, error, steps=own, total=self.before + self.steps, **fields)

    def kept(self) -> None:
        """What the session cost so far, kept with it (also as the process ends: a query
        that paused has no result)."""
        if self.session is not None:
            spent_file(self.session).write_text(str(self.before + self.steps))

    def recorded(
        self, session: str, prompt: Any, context: str, earlier: list[str], *, first: bool
    ) -> None:
        """What the CLI was started with and this turn's prompt, in ``FAKE_CLAUDE_RECORD``
        (the latest turn's) and as a line of ``<record>.turns``."""
        record = os.environ.get("FAKE_CLAUDE_RECORD")
        if record:
            servers = json.loads(argument("--mcp-config") or "{}").get("mcpServers", {})
            said = {
                "pid": os.getpid(),
                "session": session,
                "started_resuming": argument("--resume"),
                "system_prompt": argument("--system-prompt"),
                "allowed_tools": argument("--allowedTools"),
                "mcp_config": argument("--mcp-config"),
                "tools": self.listed() if SERVER in servers else [],
                "permission_prompt_tool": argument("--permission-prompt-tool"),
                "resume": argument("--resume") if first else None,
                "prompt": prompt,
                "custom_headers": os.environ.get("ANTHROPIC_CUSTOM_HEADERS"),
                "max_mcp_output_tokens": os.environ.get("MAX_MCP_OUTPUT_TOKENS"),
                "context": context,
                "materialised": mirrored(session) is not None,
                "earlier": earlier,
                "disallowed_tools": argument("--disallowedTools"),
                "builtins": argument("--tools"),
                "max_turns": argument("--max-turns"),
                "plugins": plugins(),
                "argv": sys.argv[1:],
                "cwd": os.getcwd(),
                "team_env": {k: v for k, v in os.environ.items() if k.startswith("TEAM_")},
                "initialize": self.initialized,
            }
            replaced(record, said)
            appended(f"{record}.turns", said)

    def played(
        self, session: str, script: list[dict[str, Any]], done: int, turn: int, context: str
    ) -> None:
        """The turn's script from step ``done``, then its result — or the result the real CLI
        stops it with (its turn limit, its budget, a reply cut at the output limit)."""
        kept = SESSIONS / f"{session}.json"
        last = read = ""
        limit = argument("--max-turns")
        cap = argument("--max-budget-usd")
        for turns, (step, item) in enumerate(enumerate(script[done:], start=done)):
            ended = stopped(turns, self.steps, limit, cap)
            if ended is not None:  # what the real CLI ends with
                send(self.ended(session, ended[0], subtype=ended[1]))
                # and exits 1 once its input ends, as it does; a process kept for more turns
                # (the SDK's client: its input stays open) goes on, its next prompt answered (the
                # bundled CLI, driven against a scripted model)
                self.status = 1
                return
            self.steps += 1
            self.denied = False
            if "cut" in item:
                send(self.ended(session, turns=turns + 1, stop_reason="max_tokens"))
                return
            said, read, completed, interrupted, inner = self.step(
                session, step, item, read, context
            )
            self.steps += inner
            self.helped += inner
            last = said if said is not None else last
            if self.stopping:
                raise Interrupted
            if not completed:
                if interrupted:
                    break
                if not self.denied:
                    continue
            kept.write_text(json.dumps({"done": step + 1, "turn": turn}))
            mirror(session, step + 1, turn)
        self.hooked("Stop", session, stop_hook_active=False)
        send(self.ended(session, last=last, turns=len(script)))
        if any(item.get("quit") for item in script):  # a CLI that ends of itself, cleanly
            sys.exit(0)

    def step(
        self, session: str, step: int, item: dict[str, Any], read: str, context: str
    ) -> tuple[str | None, str, bool, bool, int]:
        """One step of the script: a text, a subagent or a tool call. What it said (``None``:
        no text), the text of its result, whether it is done, whether a check stopped the
        turn, and how many steps of a subagent's own it took."""
        if "stall" in item:  # the model says nothing for a while, then goes on
            self.wait(float(item["stall"]))
            return None, read, True, False, 0
        if "background" in item:
            read, completed, interrupted = self.background(session, step, item)
            return None, read, completed, interrupted, 0
        if "text" in item:
            said = item["text"].replace("{last}", read).replace("{context}", context)
            said = said.replace("{heard}", " | ".join(self.heard))
            if "--include-partial-messages" in sys.argv:
                send(stream_event(session, {"type": "message_start"}))
                for n, word in enumerate(said.split(" ")):
                    send(delta(session, word if n == 0 else f" {word}"))
            send(assistant(session, [{"type": "text", "text": said}]))
            return said, read, True, False, 0
        if "agent" in item or "agents" in item:
            answer, completed, interrupted, inner = self.subagent(session, step, item)
            return None, answer, completed, interrupted, inner
        read, completed, interrupted = self.tool(session, step, item)
        return None, read, completed, interrupted, 0

    def subagent(
        self, session: str, step: int, item: dict[str, Any]
    ) -> tuple[str, bool, bool, int]:
        """One subagent (``{"agent", "prompt", "steps"}``) — or several at work at once
        (``{"agents": [...]}``) — as the bundled CLI runs one of ``agents=`` whose call says
        ``run_in_background: false``: its ``Agent`` tool call (through the hooks and the
        permission rules, as any call), its task's start (``task_started``, not
        backgrounded: the task's id is the ``agent_id`` of the subagent's hooks and
        permission requests) and its ``SubagentStart`` hooks, then its own steps — their
        messages naming the call (``parent_tool_use_id``), several subagents' steps in turn
        — then its ``SubagentStop`` hooks, its task's end (``task_updated``,
        ``task_notification``) and the call's result, the subagent's last text. The text of
        the (last) result, whether it is done, whether a check stopped the turn, and how
        many of their own steps the subagents took."""
        several = "agents" in item
        subs: list[dict[str, Any]] = []
        for n, one in enumerate(item["agents"] if several else [item]):
            tool_id = f"toolu_{step}{'abcdefgh'[n]}" if several else f"toolu_{step}"
            args = {
                "description": one["prompt"],
                "prompt": one["prompt"],
                "subagent_type": one["agent"],
                "run_in_background": False,
            }
            call = [{"type": "tool_use", "id": tool_id, "name": "Agent", "input": args}]
            send(assistant(session, call))
            permission = self.allowed(session, "Agent", args, tool_id)
            if permission.get("behavior", "allow") != "allow":
                self.result(session, tool_id, [{"type": "text", "text": "denied"}])
                continue
            given = permission.get("updatedInput", args)
            briefs = os.environ.get("FAKE_CLAUDE_BRIEFS")
            if briefs:  # what the subagent was given, the hooks' updates included
                appended(briefs, {"agent": one["agent"], "prompt": given["prompt"]})
            who = {"agent_id": f"agent_{tool_id}", "agent_type": one["agent"]}
            subs.append({"item": one, "tool_id": tool_id, "who": who, "args": given, "last": ""})
        if not subs:
            return "denied", False, False, 0
        for sub in subs:
            send(
                {
                    "type": "system",
                    "subtype": "task_started",
                    "task_id": sub["who"]["agent_id"],
                    "tool_use_id": sub["tool_id"],
                    "description": sub["item"]["prompt"],
                    "task_type": "local_agent",
                    "subagent_type": sub["item"]["agent"],
                    "is_backgrounded": False,
                    "uuid": str(uuid.uuid4()),
                    "session_id": session,
                }
            )
            self.hooked("SubagentStart", session, **sub["who"])
        try:
            interrupted, taken = self._worked(session, step, subs)
        finally:
            self.working = {}
        if interrupted:
            return "", False, True, taken
        for sub in subs:
            self._finished(session, sub)
        return subs[-1]["last"], True, False, taken

    def _worked(self, session: str, step: int, subs: list[dict[str, Any]]) -> tuple[bool, int]:
        """The subagents' own steps, one of each in turn: whether a check stopped the turn,
        and how many steps they took."""
        taken = 0
        for n in range(max(len(sub["item"]["steps"]) for sub in subs)):
            for sub in subs:
                if n >= len(sub["item"]["steps"]):
                    continue
                inner, tool_id = sub["item"]["steps"][n], sub["tool_id"]
                self.working = sub["who"]
                taken += 1
                if "text" in inner:
                    sub["last"] = inner["text"]
                    said = [{"type": "text", "text": inner["text"]}]
                    send(assistant(session, said, parent=tool_id))
                    continue
                _, _, interrupted = self.tool(
                    session, step, inner, tool_id=f"{tool_id}_{n}", parent=tool_id
                )
                if interrupted:
                    return True, taken
        return False, taken

    def _finished(self, session: str, sub: dict[str, Any]) -> None:
        """A subagent's end: its ``SubagentStop`` hooks, its task's end, its ``Agent``
        call's ``PostToolUse`` hooks and that call's result."""
        tool_id, last = sub["tool_id"], sub["last"]
        self.hooked(
            "SubagentStop", session, stop_hook_active=False, agent_transcript_path="", **sub["who"]
        )
        frame = {"type": "system", "task_id": sub["who"]["agent_id"], "session_id": session}
        send({**frame, "subtype": "task_updated", "patch": {"status": "completed"}})
        send(
            {
                **frame,
                "subtype": "task_notification",
                "tool_use_id": tool_id,
                "status": "completed",
                "output_file": "",
                "summary": last,
                "uuid": str(uuid.uuid4()),
            }
        )
        content = [{"type": "text", "text": last}]
        self.post_tool_use(session, "Agent", sub["args"], tool_id, content=content, error=None)
        self.result(session, tool_id, content)

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
        given = decided.get("updatedInput", args)
        asked = self.permission(
            name, given, tool_id, ask=decided.get("permissionDecision") == "ask"
        )
        return {"updatedInput": given, **asked}

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
        tool_id = tool_id or str(item.get("id") or f"toolu_{step}")
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
        if allowed and builtin == STRUCTURED:
            self.structured = ran
        if allowed and builtin:
            appended(os.environ["FAKE_CLAUDE_BUILTINS"], {"tool": builtin, "args": ran})
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
        stops = waits or bool(permission.get("interrupt"))
        # a call that did not run, or failed, is done all the same: the "model" read why, as
        # the real CLI's transcript keeps it, and a resumed session goes on after it
        # (``played``); only a call the run paused at is made again
        self.denied = not stops
        return read, not error, stops


def made_again(script: list[dict[str, Any]], done: int) -> list[dict[str, Any]]:
    """``script`` as a session told its task was paused goes on with it: the call it goes on at
    made with its ``again`` arguments, when it has them."""
    worded = [
        {**item, "args": item["again"]} for item in script[done : done + 1] if "again" in item
    ]
    return [*script[:done], *worded, *script[done + len(worded) :]]


def spent_file(session: str) -> Path:
    """Where the steps a session took so far are kept (with the session, on this machine)."""
    SESSIONS.mkdir(parents=True, exist_ok=True)
    return SESSIONS / f"{session}.spent"


def spent_before(session: str) -> int:
    """The steps the session took before this process: as this machine keeps them, else as
    the transcript the SDK materialised says."""
    kept = spent_file(session)
    if kept.exists():
        return int(kept.read_text())
    return max((int(e.get("spent", 0)) for e in materialised(session) or ()), default=0)


def projects() -> Path:
    """Where the CLI keeps its transcripts (as the real CLI: under ``CLAUDE_CONFIG_DIR``)."""
    home = os.environ.get("CLAUDE_CONFIG_DIR") or os.path.expanduser("~/.claude")
    return Path(home) / "projects"


def mirror(
    session: str, done: int, turn: int = 0, *, prompt: str | None = None, key: str = ""
) -> None:
    """With ``--session-mirror``: one transcript entry (the steps done in the session's
    ``turn``, the prompt it began with; ``key``: what tells an injected turn's entry from the
    turn's own) mirrored to the SDK."""
    if "--session-mirror" not in sys.argv:
        return
    uuid_ = f"{session}-{done}" if not turn else f"{session}-{turn}-{done}"
    uuid_ = f"{uuid_}-{key}" if key else uuid_
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
    """An assistant message (``parent``: a subagent's, the ``Agent`` call that runs it): one
    step's reply, with its id and usage as the real CLI sends them."""
    return {
        "type": "assistant",
        "message": {
            "id": f"msg_{uuid.uuid4().hex}",
            "role": "assistant",
            "model": "fake",
            "content": content,
            "usage": {"input_tokens": STEP_TOKENS[0], "output_tokens": STEP_TOKENS[1]},
        },
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
    stop_reason: str = "end_turn",
    total: int | None = None,
    structured: dict[str, Any] | None = None,
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
        "stop_reason": stop_reason,
        "total_cost_usd": spent(steps if total is None else total),
        "usage": {"input_tokens": STEP_TOKENS[0] * steps, "output_tokens": STEP_TOKENS[1] * steps},
    }
    if error is not None:
        ended["errors"] = [error]
    if structured is not None:
        ended["structured_output"] = structured
    return ended


def acknowledge(message: dict[str, Any]) -> None:
    """A control request of the SDK's, answered."""
    initialize = message["request"].get("subtype") == "initialize"
    send(
        {
            "type": "control_response",
            "response": {
                "subtype": "success",
                "request_id": message["request_id"],
                "response": {"commands": []} if initialize else {},
            },
        }
    )


def refused() -> None:
    """The flags the bundled CLI refuses to start with, in its own words (and its exit)."""
    named, forks = argument("--session-id"), "--fork-session" in sys.argv
    if named and not forks and (argument("--resume") or "--continue" in sys.argv):
        # the bundled CLI's own words, and its exit
        print(
            "Error: --session-id can only be used with --continue or --resume if "
            "--fork-session is also specified.",
            file=sys.stderr,
        )
        sys.exit(1)
    if named and not forks and (SESSIONS / f"{named}.json").exists():
        print(f"Error: Session ID {named} is already in use.", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    if "-v" in sys.argv or "--version" in sys.argv:
        print("9.9.9 (Claude Code)")
        return
    refused()
    if os.environ.get("FAKE_CLAUDE_STDERR"):  # what the options' ``stderr`` callback reads
        print(os.environ["FAKE_CLAUDE_STDERR"], file=sys.stderr, flush=True)
    cli = Cli()
    atexit.register(cli.kept)
    if os.environ.get("FAKE_CLAUDE_PID"):  # which process this is, before any turn
        Path(os.environ["FAKE_CLAUDE_PID"]).write_text(str(os.getpid()))
    time.sleep(float(os.environ.get("FAKE_CLAUDE_STARTS_AFTER") or 0))  # a slow start
    while True:
        if cli.queued:
            cli.turn(cli.queued.pop(0))
            continue
        if cli.owed:
            cli.injected(cli.owed.pop(0))
            continue
        message = read(timeout=0.02 if cli.tasks else None)
        if message is None:  # the SDK closed stdin, or its process died
            orphaned(cli)
            sys.exit(cli.status)
        if message is NOTHING:
            cli.triggers()
            continue
        kind = message.get("type")
        if kind == "control_request":
            if message["request"].get("subtype") == "initialize":
                cli.hooks = message["request"].get("hooks") or {}
                cli.initialized = {
                    k: (sorted(v or {}) if k == "hooks" else v)
                    for k, v in message["request"].items()
                    if k != "subtype"
                }
            cli.asked_to_stop(message)
            acknowledge(message)
        elif kind == "user":
            cli.turn(message["message"]["content"])


if __name__ == "__main__":
    main()
