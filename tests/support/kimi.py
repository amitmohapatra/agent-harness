"""Kimi Code for the tests: the real ``kimi acp`` (``@moonshot-ai/kimi-code``, the version
``compat.BINARIES`` pins) driving a planned model behind a local OpenAI-compatible endpoint.

:class:`PlannedEndpoint` answers Kimi's chat completions (streamed, as Kimi asks for them) the
way ``tests.support.planned`` models do: it counts the tool results the conversation holds and
makes the plan's next call, or answers ``final`` with ``{last}`` the last result it read. A
call Kimi did not run (the run paused on it, or stopped the turn) is no result; what the
harness tells Kimi in a prompt of its own after a rejection is that call's result, and a
reviewer's edit is followed (the call made again with the edited arguments) — what a model
told so does. A harness tool is called as Kimi names it (``mcp__trellis__<tool>``); Kimi's
own tools (``Bash``, ``Agent``...) by their names. ``turns`` answers by the request instead,
for a test that scripts a sub-agent's requests too.

:func:`kimi_target` is the target: Kimi in a home and a working directory of the test's own,
its model the endpoint. Without the ``kimi`` program the tests that need it are skipped — in CI
it is installed, and ``CI`` set makes its absence a failure.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import shutil
import weakref
from collections.abc import AsyncIterator, Callable, Sequence
from pathlib import Path
from typing import Any, Final

import pytest
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

from tests.support.planned import FINAL, Call
from trellis.harness.adapters.acp import KIMI_BUILTINS, ACPAgent, kimi

#: The ``kimi`` program, when it is installed.
KIMI_BIN: Final = shutil.which("kimi")
needs_kimi = pytest.mark.skipif(
    KIMI_BIN is None and not os.environ.get("CI"),
    reason="the kimi program is not installed (npm install -g @moonshot-ai/kimi-code@2.1.1)",
)
#: The tokens every reply reports, as Kimi's usage log keeps them.
REPLY_USAGE: Final = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
#: What Kimi writes before a tool's result, and around an MCP result's extras.
_WALL_TIME = re.compile(r"^Wall time: [0-9.]+ seconds\n")
_EXTRAS = re.compile(r"\n?<mcp-result-extras>.*</mcp-result-extras>\s*$", re.DOTALL)
#: What Kimi puts before a failed tool's result.
_FAILED = re.compile(r"^<system>ERROR: [^<]*</system>\n?")
#: A tool result of a call that did not run: the run paused on it, or stopped the turn.
NOT_RUN: Final = ("approval request was cancelled", "Waiting for a person's approval.")
#: The harness's prompts after a rejection and an edit (``adapters.acp``) and a denied resume.
_REJECTED = re.compile(r"^Your call to \S+ was not run: ")
_DENIED = "the person denied it"
#: The harness's prompt at a step limit (``budget.TURN_LIMIT``): a model told so answers.
_LIMIT = "This turn has reached its limit"
_EDITED = re.compile(
    r"Call (?P<tool>\S+) again now with exactly these arguments: (?P<args>\{.*\})$"
)

Turn = str | Call | list[Call] | dict[str, Any]


def text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") for p in content or () if isinstance(p, dict))


def result_text(content: Any) -> str:
    """A tool result as the model reads it, less Kimi's wall time and MCP extras."""
    return _FAILED.sub("", _EXTRAS.sub("", _WALL_TIME.sub("", text_of(content)))).strip()


def kimi_name(tool: str) -> str:
    """A tool as Kimi names it: its own by name, the harness's through the ``trellis`` server."""
    return tool if tool in KIMI_BUILTINS or tool.startswith("mcp__") else f"mcp__trellis__{tool}"


class PlannedEndpoint:
    """A chat completions endpoint following ``plan`` (or ``turns``: a function of the request's
    body to its turn), every request kept (``requests``, ``headers``)."""

    def __init__(
        self,
        plan: Sequence[Any] = (),
        *,
        final: str = FINAL,
        turns: Callable[[dict[str, Any]], Turn] | None = None,
        stall: float = 0.0,
    ) -> None:
        #: each step a call, or the calls made in one message
        self.plan: list[Any] = list(plan)
        self.final = final
        self.turns = turns
        self.stall = stall
        self.requests: list[dict[str, Any]] = []
        self.headers: list[dict[str, str]] = []
        self.url = ""

    def said(self) -> str:
        """Everything the model was sent, as one text."""
        return "\n".join(
            text_of(m.get("content")) for body in self.requests for m in body["messages"]
        )

    def offered(self) -> list[list[str]]:
        return [[t["function"]["name"] for t in b.get("tools") or ()] for b in self.requests]

    def next(self, body: dict[str, Any]) -> Turn:
        if self.turns is not None:
            return self.turns(body)
        results: list[str] = []
        edited: Call | None = None
        users = 0
        for message in body["messages"]:
            role, text = message.get("role"), text_of(message.get("content"))
            if role == "tool":
                found = result_text(message.get("content"))
                if not any(mark in found for mark in NOT_RUN):
                    results.append(found)
            elif role == "user":
                users += 1
                said = _system_reminders(text)
                matched = _EDITED.search(said)
                if matched:
                    edited = (matched["tool"], json.loads(matched["args"]))
                elif users > 1 and (_REJECTED.match(said) or _DENIED in said):
                    results.append(said)
        if _told_limit(body):  # told its step limit is reached: it answers, calling nothing
            return self.final.replace("{last}", results[-1] if results else "")
        done = 0
        for step in self.plan:  # a step is a call, or several made in one message
            if len(results) <= done:
                if edited is not None:
                    return edited
                if isinstance(step, list):
                    return [(kimi_name(n), a) for n, a in step]
                return (kimi_name(step[0]), step[1])
            done += len(step) if isinstance(step, list) else 1
        return self.final.replace("{last}", results[-1] if results else "")

    def reply(self, body: dict[str, Any]) -> dict[str, Any]:
        self.requests.append(body)
        turn = self.next(body)
        if isinstance(turn, str):
            message: dict[str, Any] = {"role": "assistant", "content": turn}
        elif isinstance(turn, tuple | list):
            calls = turn if isinstance(turn, list) else [turn]
            n = len(self.requests)
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"call_{n}_{i}",
                        "type": "function",
                        "function": {"name": kimi_name(name), "arguments": json.dumps(args)},
                    }
                    for i, (name, args) in enumerate(calls)
                ],
            }
        else:
            message = turn
        finish = "tool_calls" if message.get("tool_calls") else "stop"
        return {
            "id": f"chatcmpl-{len(self.requests)}",
            "created": 0,
            "model": "scripted",
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": REPLY_USAGE,
        }

    async def _chat(self, request: Request) -> Any:
        body = await request.json()
        self.headers.append(dict(request.headers))
        if self.stall:
            await asyncio.sleep(self.stall)
        reply = self.reply(body)
        if not body.get("stream"):
            return JSONResponse({**reply, "object": "chat.completion"})
        return StreamingResponse(_chunks(reply), media_type="text/event-stream")

    async def attached(self) -> PlannedEndpoint:
        """This endpoint on the event loop's shared server (made the first time, stopped at the
        test's end: :func:`detached`): for a target built where nothing can close a server of
        its own."""
        loop = asyncio.get_running_loop()
        shared = _SHARED.get(loop)
        if shared is None:
            shared = _SHARED[loop] = _Shared()
            await shared.start()
        self.url = shared.add(self)
        return self

    @contextlib.asynccontextmanager
    async def serving(self) -> AsyncIterator[PlannedEndpoint]:
        app = Starlette(routes=[Route("/v1/chat/completions", self._chat, methods=["POST"])])
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error"))
        server.capture_signals = contextlib.nullcontext  # type: ignore[method-assign]
        task = asyncio.create_task(server.serve())
        while not server.started:  # noqa: ASYNC110 - uvicorn says so with a flag only
            await asyncio.sleep(0.01)
        self.url = f"http://127.0.0.1:{server.servers[0].sockets[0].getsockname()[1]}/v1"
        try:
            yield self
        finally:
            server.should_exit = True
            await task


class _Shared:
    """One server per event loop, each endpoint under a path of its own."""

    def __init__(self) -> None:
        self.endpoints: dict[str, PlannedEndpoint] = {}
        self.origin = ""

    async def _chat(self, request: Request) -> Any:
        return await self.endpoints[request.path_params["name"]]._chat(request)

    async def start(self) -> None:
        app = Starlette(routes=[Route("/{name}/v1/chat/completions", self._chat, methods=["POST"])])
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error"))
        server.capture_signals = contextlib.nullcontext  # type: ignore[method-assign]
        self.server = server
        self.task = asyncio.create_task(server.serve())
        while not server.started:  # noqa: ASYNC110 - uvicorn says so with a flag only
            await asyncio.sleep(0.01)
        self.origin = f"http://127.0.0.1:{server.servers[0].sockets[0].getsockname()[1]}"

    async def stop(self) -> None:
        self.server.should_exit = True
        await self.task

    def add(self, endpoint: PlannedEndpoint) -> str:
        name = f"e{len(self.endpoints)}"
        self.endpoints[name] = endpoint
        return f"{self.origin}/{name}/v1"


#: The shared servers, by event loop (each test has its own loop).
_SHARED: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, _Shared] = weakref.WeakKeyDictionary()


async def detached() -> None:
    """This loop's shared server stopped, if a test started one: at the test's end, before its
    loop closes (a server left serving there is cancelled mid-lifespan)."""
    shared = _SHARED.pop(asyncio.get_running_loop(), None)
    if shared is not None:
        await shared.stop()


def attached_endpoints() -> list[PlannedEndpoint]:
    """The endpoints on this loop's shared server: what every Kimi of the test was sent."""
    shared = _SHARED.get(asyncio.get_running_loop())
    return list(shared.endpoints.values()) if shared is not None else []


def _system_reminders(text: str) -> str:
    """A user message less the reminders Kimi puts before it."""
    return re.sub(r"<system-reminder>.*?</system-reminder>\s*", "", text, flags=re.DOTALL).strip()


async def _chunks(reply: dict[str, Any]) -> AsyncIterator[str]:
    [choice] = reply["choices"]
    message = choice["message"]
    delta: dict[str, Any] = {"role": "assistant", "content": message.get("content") or ""}
    if message.get("tool_calls"):
        delta["tool_calls"] = [{"index": n, **c} for n, c in enumerate(message["tool_calls"])]
    base = {key: reply[key] for key in ("id", "created", "model")}
    chunk = "chat.completion.chunk"
    for part in (
        {**base, "object": chunk, "choices": [{"index": 0, "delta": delta}]},
        {
            **base,
            "object": chunk,
            "choices": [{"index": 0, "delta": {}, "finish_reason": choice["finish_reason"]}],
        },
        {**base, "object": chunk, "choices": [], "usage": reply["usage"]},
    ):
        yield f"data: {json.dumps(part)}\n\n"
    yield "data: [DONE]\n\n"


def kimi_env(endpoint: PlannedEndpoint, home: Path) -> dict[str, str]:
    """Kimi in ``home`` (its sessions, its usage log), its model the endpoint."""
    return {
        "KIMI_CODE_HOME": str(home),
        "KIMI_MODEL_PROVIDER_TYPE": "openai",
        "KIMI_MODEL_BASE_URL": endpoint.url,
        "KIMI_MODEL_API_KEY": "scripted",
        "KIMI_MODEL_NAME": "scripted",
        "KIMI_DISABLE_TELEMETRY": "1",
        "KIMI_CODE_NO_AUTO_UPDATE": "1",
    }


def _told_limit(body: dict[str, Any]) -> bool:
    """Whether a later user message tells the model its step limit is reached."""
    users = [m for m in body["messages"] if m.get("role") == "user"]
    return any(_system_reminders(text_of(m.get("content"))).startswith(_LIMIT) for m in users[1:])


def kimi_target(endpoint: PlannedEndpoint, tmp: Path, **fields: Any) -> ACPAgent:
    """Kimi Code working in ``tmp/work``, its home ``tmp/home``, its model ``endpoint``."""
    home, work = tmp / "kimi-home", tmp / "kimi-work"
    home.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    env = {**kimi_env(endpoint, home), **fields.pop("env", {})}
    return kimi(env=env, cwd=str(work), **fields)
