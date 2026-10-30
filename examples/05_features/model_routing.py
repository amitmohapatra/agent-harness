"""Bifrost, any OpenAI-compatible endpoint, or a provider's own SDK: a model name is routed one
way for every framework, chosen from the settings (``LLM_GATEWAY`` names the route) — never a
chain of fallbacks.

* ``LLM_BASE_URL`` with ``LLM_GATEWAY=bifrost``: every model call goes through Bifrost on
  ``LLM_API_KEY``, the name as its ``provider/model`` (and the endpoint wins when a provider
  key is set too); the MCP tools, Code Mode, stored prompts and dollar budgets come with it;
* ``LLM_BASE_URL`` alone (``openai-compatible``): every model call goes to that endpoint —
  OpenAI, OpenRouter, a LiteLLM proxy, Ollama, vLLM, llama.cpp — the name as given; models
  only;
* no ``LLM_BASE_URL``: a name goes to its provider's own SDK, which reads its own key
  (``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``...); models only. A name must say its provider
  (``openai/...``).

The same ``ReAct`` runs on each route. Offline all three are served by a stand-in
OpenAI-compatible endpoint on a local port, which answers saying which name and key reached
it; with Bifrost set the first run is the real Bifrost's, and with ``OPENAI_API_KEY`` set the
last is OpenAI's.

    python -m examples.05_features.model_routing
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from examples._support.gateway import ScriptedGateway
from examples._support.offline import MODEL, offline_blocks, online

from trellis import Harness, ReAct, Settings
from trellis.contracts import RunStatus

#: the name every run asks for: its provider, then the provider's own name for it
NAME = MODEL if online() else "openai/gpt-4.1-mini"


class StandIn(BaseHTTPRequestHandler):
    """An OpenAI-compatible chat completions endpoint: it answers with the key it was given."""

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        if not self.path.endswith("/chat/completions"):
            self.send_error(404)
            return
        key = self.headers.get("authorization", "").removeprefix("Bearer ")
        text = f"served {body['model']} on key {key}"
        if body.get("stream"):
            self._stream(body["model"], text)
            return
        answer = {
            "id": "c1",
            "object": "chat.completion",
            "created": 0,
            "model": body["model"],
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        self._send("application/json", json.dumps(answer).encode())

    def _stream(self, model: str, text: str) -> None:
        def chunk(choices: list[dict[str, Any]], **more: Any) -> str:
            data = {"id": "c1", "object": "chat.completion.chunk", "created": 0, "model": model}
            return f"data: {json.dumps(data | {'choices': choices} | more)}\n\n"

        usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        events = [
            chunk([{"index": 0, "delta": {"role": "assistant", "content": text}}]),
            chunk([{"index": 0, "delta": {}, "finish_reason": "stop"}]),
            chunk([], usage=usage),  # the usage, as stream_options.include_usage asks
            "data: [DONE]\n\n",
        ]
        self._send("text/event-stream", "".join(events).encode())

    def _send(self, kind: str, payload: bytes) -> None:
        self.send_response(200)
        self.send_header("content-type", kind)
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: Any) -> None:  # quiet
        pass


@contextmanager
def stand_in() -> Iterator[str]:
    """The stand-in endpoint on a free local port: its base URL."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), StandIn)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    finally:
        server.shutdown()


async def ask(h: Harness, route: str) -> None:
    agent = h.wrap(ReAct(system_prompt="Say who served you.", model=NAME), id="router")
    result = await agent.run("Who served this?", user="ada")
    assert result.status is RunStatus.SUCCESS, result.error
    print(f"{route}: {h.model_gateway.name} ({h.model_gateway.title}) -> {result.answer}")


async def main() -> None:
    with stand_in() as url:
        # 1. Bifrost: the deployment's (LLM_GATEWAY=bifrost), else the stand-in in its place
        bifrost = (
            Settings.from_env()
            if online()
            else Settings(llm_gateway="bifrost", llm_base_url=url, llm_api_key="vk-support")
        )
        # offline the gateway's MCP side is the scripted one (no tools); online, Bifrost's own
        blocks = offline_blocks()
        if not online():
            blocks["gateway"] = ScriptedGateway({}).gateway()
        async with Harness(config=bifrost, **blocks) as h:
            await ask(h, "LLM_GATEWAY=bifrost")

        # 2. any OpenAI-compatible endpoint: the name as given, on LLM_API_KEY
        compatible = Settings(llm_base_url=url, llm_api_key="sk-endpoint")
        async with Harness(config=compatible, **offline_blocks()) as h:
            await ask(h, "LLM_BASE_URL alone")

        # 3. no endpoint: the provider's own SDK, with its own key and endpoint
        if online() and not os.environ.get("OPENAI_API_KEY"):
            print("no LLM_BASE_URL: skipped (set OPENAI_API_KEY to run it against OpenAI)")
            return
        if not os.environ.get("OPENAI_API_KEY"):
            os.environ.update(OPENAI_API_KEY="sk-provider", OPENAI_BASE_URL=url)
        async with Harness(config=Settings(), **offline_blocks()) as h:
            await ask(h, "no LLM_BASE_URL")


if __name__ == "__main__":
    asyncio.run(main())
