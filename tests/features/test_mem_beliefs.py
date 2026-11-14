"""MEM-04, beliefs: the memory service folds a person's facts into beliefs with a model — here
a scripted one (no model is paid for), registered as a tenant's model endpoint the way an
operator registers a real one, allowed only the ``reflection`` use.

What the user tells the agent becomes facts; 60 s later (the debounce) the consolidation job
asks the model, checks every line it returns against the facts it cites, and keeps the beliefs:
one line per facet, citing the facts it rests on, the person's own. A fact repeated is kept by
the belief it already supports (no second line), and a forgotten fact takes its belief with it.
The scripted model follows the service's own instructions: it adds a facet for a new fact,
keeps a belief a fact repeats, and checks every claim supported."""

from __future__ import annotations

import json
import re
import socket
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from tests.features.conftest import MEMORY_URL, PLATFORM_KEY, eventually, harness, memory_sql
from tests.features.kit import armed
from trellis.contracts import RunStatus
from trellis.memory import MemoryClient

pytestmark = pytest.mark.timeout(600)

FACT = re.compile(r"^F(\d+) \[[^\]]*\] \([^)]*\): (.*)$")
BELIEF = re.compile(r"^B(\d+) \[([a-z]+)\] (.*)$")
CLAIM = re.compile(r"^L(\d+): (.*)$")


class ScriptedReflection:
    """An OpenAI-compatible endpoint answering the consolidation job's three requests."""

    def __init__(self) -> None:
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        self.port = sock.getsockname()[1]
        sock.close()
        self.requests: list[dict[str, Any]] = []

    def facet(self, text: str) -> str:
        lowered = text.lower()
        if any(w in lowered for w in ("work", "nurse", "job")):
            return "work"
        if any(w in lowered for w in ("live", "porto", "city")):
            return "places"
        return "other"

    def answer(self, body: dict[str, Any]) -> dict[str, Any]:
        system = body["messages"][0]["content"]
        user = body["messages"][-1]["content"]
        lines = user.splitlines()
        if system.startswith("You keep an assistant's beliefs"):
            beliefs = {m[3]: m[1] for m in map(BELIEF.match, lines) if m}
            entries = []
            for m in filter(None, map(FACT.match, lines)):
                n, text = m[1], m[2].strip()
                if text in beliefs:
                    entries.append(
                        {
                            "op": "keep",
                            "facts": [f"F{n}"],
                            "belief": f"B{beliefs[text]}",
                            "text": "",
                            "facet": "",
                        }
                    )
                else:
                    entries.append(
                        {
                            "op": "add",
                            "facts": [f"F{n}"],
                            "belief": "",
                            "text": text,
                            "facet": self.facet(text),
                        }
                    )
            return {"entries": entries}
        if system.startswith("Check each claim"):
            return {
                "verdicts": [
                    {"line": f"L{m[1]}", "label": "SUPPORTED"} for m in map(CLAIM.match, lines) if m
                ]
            }
        if system.startswith("Rewrite each claim"):
            return {
                "rewrites": [
                    {"line": f"L{m[1]}", "text": m[2]} for m in map(CLAIM.match, lines) if m
                ]
            }
        return {}

    def app(self) -> Starlette:
        async def models(request: Request) -> JSONResponse:
            return JSONResponse({"object": "list", "data": [{"id": "scripted", "object": "model"}]})

        async def chat(request: Request) -> JSONResponse:
            body = await request.json()
            self.requests.append(body)
            content = json.dumps(self.answer(body))
            return JSONResponse(
                {
                    "id": f"chatcmpl-{len(self.requests)}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": "scripted",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": content},
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
                }
            )

        return Starlette(
            routes=[
                Route("/v1/models", models),
                Route("/v1/chat/completions", chat, methods=["POST"]),
            ]
        )


@pytest.fixture
def reflection() -> Iterator[ScriptedReflection]:
    model = ScriptedReflection()
    server = uvicorn.Server(
        uvicorn.Config(model.app(), host="127.0.0.1", port=model.port, log_level="error")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.05)
    yield model
    server.should_exit = True
    thread.join(timeout=10)


def beliefs_of(tenant: str, user: str) -> list[dict]:
    return memory_sql(
        "select b.belief_id, b.text, b.facet, b.proof_count, b.person_key, b.visibility_keys,"
        " b.superseded_by, b.deleted_at, b.stale_since,"
        " (select json_agg(s.memory_id) from belief_sources s where s.belief_id = b.belief_id)"
        " as sources from beliefs b where b.tenant_id = :1 and b.person_key like :2",
        tenant,
        f"%{user}%",
    )


@pytest.mark.plan("MEM-04")
async def test_facts_are_consolidated_into_cited_beliefs_once(
    reflection: ScriptedReflection, tmp_path: Path, uid: str
) -> None:
    tenant = f"featllm{uid}"
    async with MemoryClient(MEMORY_URL, api_key=PLATFORM_KEY, timeout=60) as platform:
        created = await platform.admin.create_tenant(f"Llm {uid}", tenant_id=tenant)
    async with MemoryClient(MEMORY_URL, api_key=created.admin_key.token, timeout=60) as admin:
        t = admin.administer(tenant)
        service = await t.keys.issue("service", "beliefs")
        await t.set_model_key(
            {
                "base_url": f"http://127.0.0.1:{reflection.port}/v1",
                "gateway": "openai-compatible",
                "auth": "bearer",
                "model": "scripted",
                "api_key": "scripted",
            }
        )
        await t.set_model_policy(["reflection"], read_assist=False)
    user = f"rita-{uid}"
    async with harness(key=service.token) as h:
        agent, _ = await armed("react", h, tmp_path, [], id="pages")
        for n, text in enumerate(["I live in Porto.", "I work as a nurse."]):
            done = await agent.run(text, user=user, thread=f"c{n}")
            assert done.status is RunStatus.SUCCESS
        await h.writes.drain()

    async def kept() -> list[dict]:
        rows = [b for b in beliefs_of(tenant, user) if b["deleted_at"] is None]
        return rows if len(rows) >= 2 else []

    rows = await eventually(kept, within=240, every=5)
    assert rows, (beliefs_of(tenant, user), len(reflection.requests))
    texts = sorted(b["text"] for b in rows)
    assert texts == ["I live in Porto.", "I work as a nurse."], rows
    assert {b["facet"] for b in rows} == {"places", "work"}
    facts = {
        r["memory_id"]: r["content"]
        for r in memory_sql(
            "select memory_id, content from memories where tenant_id = :1 and deleted_at is null",
            tenant,
        )
    }
    for b in rows:  # each belief cites the facts it rests on, the person's own
        assert b["sources"] and all(facts.get(s) for s in b["sources"]), (b, facts)
        assert user in b["person_key"]
    systems = [r["messages"][0]["content"][:30] for r in reflection.requests]
    assert any(s.startswith("You keep") for s in systems) and any(
        s.startswith("Check each") for s in systems
    ), systems

    # the same fact again: kept by its belief, no second line
    async with harness(key=service.token) as h:
        agent, built = await armed("react", h, tmp_path, [], id="pages")
        await agent.run("I live in Porto.", user=user, thread="c9")
        await h.writes.drain()
    asked = len(reflection.requests)
    await eventually(lambda: _more(reflection, asked), within=240, every=5)
    live = [
        b for b in beliefs_of(tenant, user) if b["deleted_at"] is None and not b["superseded_by"]
    ]
    assert sorted(b["text"] for b in live) == texts, live  # no duplicate belief
    # another user of the tenant has no belief of rita's
    assert not beliefs_of(tenant, f"other-{uid}")
    # and the beliefs reach the user's next context
    async with harness(key=service.token) as h:
        agent, built = await armed("react", h, tmp_path, [], id="pages")
        await agent.run("Where do I live and what do I do?", user=user, thread="c10")
    assert "Porto" in built.said() and "nurse" in built.said()


async def _more(model: ScriptedReflection, before: int) -> bool:
    return len(model.requests) > before
