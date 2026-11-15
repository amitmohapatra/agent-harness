"""Claude Code for the tests: the CLI the Claude Agent SDK bundles (2.1.325), driving a scripted
model behind a local Anthropic Messages endpoint (no paid model call).

:class:`ScriptedClaude` answers the CLI's ``POST /v1/messages`` (streamed, as the CLI asks):
it counts the tool results the conversation holds and makes the script's next step — a tool
call (``{"tool": "Bash", "input": {...}}``, ``{last}`` in its input the last tool result's
text) or an answer (``{"text": "..."}``, ``{last}`` the
last tool result's text). It keeps every tool result it was sent (:attr:`results`), as the
model read them. A request with no tools (the CLI's own side requests: a title) is answered
with a word. :func:`claude_target` is the target: the CLI's model the endpoint, nothing of this
machine's account in its way (its own config folder, no non-essential traffic)."""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any

import claude_agent_sdk
import uvicorn
from claude_agent_sdk import ClaudeAgentOptions
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

MODEL = "claude-haiku-5-5"
#: The CLI the SDK bundles (what it runs when no ``cli_path`` is given), given by its path.
BUNDLED = str(Path(claude_agent_sdk.__file__).parent / "_bundled" / "claude")


def _event(kind: str, **data: Any) -> str:
    return f"event: {kind}\ndata: {json.dumps({'type': kind, **data})}\n\n"


def _streamed(block: dict[str, Any]) -> str:
    """One reply of one block, as the Messages API streams it."""
    message = {
        "id": "msg_scripted",
        "type": "message",
        "role": "assistant",
        "model": MODEL,
        "content": [],
        "stop_reason": None,
        "usage": {"input_tokens": 10, "output_tokens": 1},
    }
    out = [_event("message_start", message=message)]
    if block["type"] == "text":
        out.append(_event("content_block_start", index=0, content_block={"type": "text", "text": ""}))
        delta = {"type": "text_delta", "text": block["text"]}
        stop = "end_turn"
    else:
        start = {"type": "tool_use", "id": block["id"], "name": block["name"], "input": {}}
        out.append(_event("content_block_start", index=0, content_block=start))
        delta = {"type": "input_json_delta", "partial_json": json.dumps(block["input"])}
        stop = "tool_use"
    out.append(_event("content_block_delta", index=0, delta=delta))
    out.append(_event("content_block_stop", index=0))
    out.append(_event("message_delta", delta={"stop_reason": stop}, usage={"output_tokens": 5}))
    out.append(_event("message_stop"))
    return "".join(out)


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") for b in content or [] if isinstance(b, dict))


class ScriptedClaude:
    """The scripted model (the module's rules), served while :meth:`serving` is open."""

    def __init__(self, script: Sequence[dict[str, Any]]) -> None:
        self.script = list(script)
        self.url = ""
        #: the tool results the last request held, as the model read them
        self.results: list[dict[str, Any]] = []
        #: each request as sent (its system prompt, its messages): for what a test looks for
        #: in it, never shown — the CLI's own context is in it
        self.sent: list[str] = []

    async def _messages(self, request: Request) -> Any:
        body = await request.json()
        if not body.get("tools"):
            return StreamingResponse(
                iter([_streamed({"type": "text", "text": "a title"})]),
                media_type="text/event-stream",
            )
        results = [
            block
            for message in body.get("messages", [])
            if message["role"] == "user" and isinstance(message["content"], list)
            for block in message["content"]
            if block.get("type") == "tool_result"
        ]
        self.results = [
            {"text": _text(r.get("content")), "is_error": bool(r.get("is_error"))} for r in results
        ]
        self.sent.append(json.dumps(body))
        n = len(results)
        step = self.script[n] if n < len(self.script) else {"text": "done"}
        if "text" in step:
            last = self.results[-1]["text"] if self.results else ""
            block: dict[str, Any] = {"type": "text", "text": step["text"].replace("{last}", last)}
        else:
            last = json.dumps(self.results[-1]["text"] if self.results else "")[1:-1]
            made = json.loads(json.dumps(step["input"]).replace("{last}", last))
            block = {"type": "tool_use", "id": f"toolu_{n}", "name": step["tool"], "input": made}
        return StreamingResponse(iter([_streamed(block)]), media_type="text/event-stream")

    async def _other(self, request: Request) -> Any:
        return JSONResponse({})

    @contextlib.asynccontextmanager
    async def serving(self) -> AsyncIterator[ScriptedClaude]:
        routes = [
            Route("/v1/messages", self._messages, methods=["POST"]),
            Route("/{rest:path}", self._other, methods=["GET", "POST"]),
        ]
        server = uvicorn.Server(
            uvicorn.Config(Starlette(routes=routes), host="127.0.0.1", port=0, log_level="error")
        )
        server.capture_signals = contextlib.nullcontext  # type: ignore[method-assign]
        task = asyncio.create_task(server.serve())
        while not server.started:  # noqa: ASYNC110 - uvicorn says so with a flag only
            await asyncio.sleep(0.01)
        self.url = f"http://127.0.0.1:{server.servers[0].sockets[0].getsockname()[1]}"
        try:
            yield self
        finally:
            server.should_exit = True
            await task


def claude_target(model: ScriptedClaude, tmp: Path, **fields: Any) -> ClaudeAgentOptions:
    """The bundled CLI, its model ``model``, its config folder ``tmp/claude-config``."""
    config = tmp / "claude-config"
    config.mkdir(parents=True, exist_ok=True)
    env = {
        "ANTHROPIC_BASE_URL": model.url,
        "ANTHROPIC_API_KEY": "sk-ant-scripted",
        "CLAUDE_CONFIG_DIR": str(config),
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "DISABLE_AUTOUPDATER": "1",
        # nothing of this machine's own git identity reaches the CLI's prompt
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        **fields.pop("env", {}),
    }
    return ClaudeAgentOptions(model=MODEL, env=env, cli_path=BUNDLED, **fields)
