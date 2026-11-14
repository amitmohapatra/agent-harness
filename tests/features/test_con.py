"""CON: what the harness sends the memory service and agent-runs, and what they answer, against
their committed OpenAPI documents — on the real services, for every route the harness's docs
say it calls (docs/runs.md "agent-runs wire", docs/memory.md). One conversation drives them
all: a document added, a queued run a worker claims, heartbeats with its journal's segments,
pauses on an approval and its question, resumed and finished, its events appended, a
schedule made; memory pushed, pulled and recorded, a decision and an outcome fed back, the
catalog read and published, an "always" kept, the inbox read. Every exchange is checked here
(request and answer), and the routes hit must cover the documented list."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

import pytest

from tests.features.conftest import Traffic, harness
from tests.features.kit import armed, record
from tests.support.openapi import MEMORY_OPENAPI, RUNS_OPENAPI, OpenAPI
from trellis import ask_user, tool
from trellis.contracts import RunStatus

pytestmark = pytest.mark.timeout(400)

#: what docs/runs.md says the harness calls on agent-runs (by route)
RUNS_ROUTES = {
    ("POST", "/v1/runs"),
    ("POST", "/v1/runs/claim"),
    ("POST", "/v1/runs/{run_id}/heartbeat"),
    ("POST", "/v1/runs/{run_id}/pause"),
    ("POST", "/v1/runs/{run_id}/resume"),
    ("POST", "/v1/runs/{run_id}/finish"),
    ("GET", "/v1/runs/{run_id}"),
    ("GET", "/v1/runs"),
    ("POST", "/v1/runs/{run_id}/payloads"),
    ("POST", "/v1/runs/{run_id}/events"),
    ("GET", "/v1/runs/{run_id}/resolutions"),
    ("POST", "/v1/schedules"),
}
#: what docs/memory.md and docs/governance.md say the harness calls on the memory service
MEMORY_ROUTES = {
    ("POST", "/v1/context"),
    ("GET", "/v1/agent-tools"),
    ("POST", "/v1/agent-tools/{name}"),
    ("POST", "/v1/messages"),
    ("POST", "/v1/tools/invocations"),
    ("POST", "/v1/feedback"),
    ("GET", "/v1/tools"),
    ("PUT", "/v1/tools/catalog"),
    ("GET", "/v1/keys/self"),
    ("POST", "/v1/documents"),
    ("GET", "/v1/documents/{document_id}"),
    ("PUT", "/v1/tool-permissions"),
    ("GET", "/v1/tool-permissions"),
}


def template(api: OpenAPI, method: str, path: str) -> str | None:
    for verb, pattern, tpl in api.operations:
        if verb == method.lower() and pattern.match(path):
            return tpl
    return None


@pytest.mark.plan("CON", "INV-7")
async def test_every_route_the_harness_calls_speaks_its_openapi_document(
    traffic: Traffic, tmp_path: Path, uid: str
) -> None:
    user, agent_id, thread = f"con-{uid}", f"con-{uid}", "c"

    def order(sku: str, qty: int) -> str:
        """Order more of a SKU."""
        return f"ordered {qty} of {sku}"

    write = tool(order, name=f"con_order_{uid}", side_effects="write")
    plan = [
        ("memory_search", {"query": "pallets", "kinds": ["memory", "message"]}),
        (f"con_order_{uid}", {"sku": "A-1", "qty": 2}),
        ("ask_user", {"question": "Which day?", "kind": "date"}),
    ]
    doc = tmp_path / "terms.md"
    doc.write_text("# Terms\n\nPallets are delivered on Mondays by Uline.\n")
    async with harness() as h:
        added = await h.add_document(doc, user=user, wait=120)
        agent, _ = await armed("react", h, tmp_path, plan, id=agent_id,
                               tools=[write, ask_user()])
        handle = await agent.start("Order two A-1 for me.", user=user, thread=thread,
                                   document_ids=[added.document_id])
        worker = h.worker([agent], concurrency=1)
        assert await worker.run_once()
        first = await record(handle.id)
        assert first.status is RunStatus.PAUSED and first.awaiting is not None
        await h.inbox(f"user:{user}")
        await agent.resume(first.awaiting.interrupt_id, "approve", reviewer=user,
                           remember="always")
        assert await worker.run_once()
        second = await record(handle.id)
        assert second.status is RunStatus.PAUSED, second  # the date question
        await agent.resume(second.awaiting.interrupt_id, "answer", answer="2026-11-20",
                           reviewer=user)
        assert await worker.run_once()
        assert (await record(handle.id)).status is RunStatus.SUCCESS
        await agent.schedule("0 3 * * *", user=user, input="nightly")
        await h.writes.drain()
        await asyncio.sleep(1)
    memory = OpenAPI.load(MEMORY_OPENAPI)
    runs = OpenAPI.load(RUNS_OPENAPI, strict_query=True)
    hit: dict[str, set[tuple[str, str]]] = {"memory": set(), "runs": set()}
    violations: list[str] = []
    for x in traffic.seen:
        api = memory if x.service == "memory" else runs
        tpl = template(api, x.method, x.path)
        assert tpl is not None, f"{x.service}: {x.method} {x.path} is in no document"
        hit[x.service].add((x.method, tpl))
        violations += [f"{x.service}: {v}" for v in api.request(x.request)]
        if x.response is not None:
            violations += [f"{x.service}: {v}" for v in api.response(x.request, x.response)]
    assert not violations, "\n".join(dict.fromkeys(violations))
    assert hit["runs"] >= RUNS_ROUTES, sorted(RUNS_ROUTES - hit["runs"])
    assert hit["memory"] >= MEMORY_ROUTES, sorted(MEMORY_ROUTES - hit["memory"])
    for x in traffic.seen:  # every request named its tenant's key, never another credential
        assert x.request.headers.get("x-api-key"), (x.method, x.path)


def test_the_documented_route_lists_name_real_routes() -> None:
    """The two lists above are routes of the documents (a renamed route fails here first)."""
    for api, routes in ((OpenAPI.load(MEMORY_OPENAPI), MEMORY_ROUTES),
                        (OpenAPI.load(RUNS_OPENAPI), RUNS_ROUTES)):
        known = {(v.upper(), tpl) for v, _, tpl in api.operations}
        assert routes <= known, sorted(routes - known)
    assert re.search(r"POST /v1/runs/\{id\}/payloads", Path("docs/runs.md").read_text())
