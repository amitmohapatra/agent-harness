"""MEM-10 and MEM-11 against the real memory service, and the memory service's ``504``.

* memory down mid-chat: the turn still answers (a ``memory_unavailable`` warning), the chat's
  earlier turn comes from the run store, and the writes the outage refused are spooled and
  delivered — once — when memory is back;
* a ``504`` (the service's own request deadline, ``api/deadline.py``, ``retryable: true``):
  a read (``/v1/context``) is retried and the context still reaches the model; one that stays
  ``504`` is a warning and the run answers; a write the service committed but answered ``504``
  is sent again and stored once (transcript, tool record, decision feedback);
* every route but the probes refuses a missing key; the platform's routes refuse a tenant's
  keys and the tenant administrator's routes refuse its service key.

Each fault goes through :class:`FaultProxy`, a proxy in front of the real service."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from tests.features.conftest import (
    ADMIN_KEY,
    API_KEY,
    MEMORY_URL,
    PLATFORM_KEY,
    FaultProxy,
    eventually,
    harness,
)
from tests.features.kit import armed, feedback_rows, messages_of_run, tool_records
from tests.support.openapi import MEMORY_OPENAPI
from trellis.contracts import ConfigurationError, RunEventType, RunStatus

pytestmark = pytest.mark.timeout(400)


async def streamed(agent: Any, question: str, **kw: Any) -> tuple[list[Any], Any]:
    events = [e async for e in agent.stream(question, **kw)]
    return events, events


def warnings(events: list[Any]) -> list[str]:
    return [
        str((e.data or {}).get("code") or (e.data or {}).get("name") or e.data)
        for e in events
        if e.type is RunEventType.CUSTOM and "warning" in json.dumps(e.data, default=str)
    ]


@pytest.mark.plan("MEM-10", "INV-2")
async def test_memory_down_mid_chat_the_turn_answers_and_its_writes_arrive_once_it_is_back(
    memory_proxy: FaultProxy, tmp_path: Path, uid: str, invariants: Any
) -> None:
    """One process serving a chat: memory goes away between two turns. The second turn still
    answers, says memory was unavailable, reads the first turn from the run store; its
    transcript, refused while memory is away, is spooled and delivered once when it is back."""
    user, agent_id, spool = f"mo-{uid}", f"down-{uid}", tmp_path / "spool"
    spool.mkdir()
    invariants.allow("INV-2", "the outage is injected: the turn is delivered after it")
    async with harness(memory_url=memory_proxy.url, spool_dir=str(spool)) as h:
        agent, built = await armed("react", h, tmp_path, [], id=agent_id)
        first = await agent.run("I store pallets in bay 12.", user=user, thread="c")
        assert first.status is RunStatus.SUCCESS
        await h.writes.drain()
        memory_proxy.down()  # memory unreachable from here on
        events = [e async for e in agent.stream("Which bay was it?", user=user, thread="c")]
        ended = [e for e in events if e.type is RunEventType.RUN_FINISHED]
        assert ended and ended[-1].data.get("result") == "Done. ", [e.type for e in events]
        run_id = events[0].run_id
        said = json.dumps([e.data for e in events], default=str)
        assert "memory_unavailable" in said, said[:2000]  # said, not silent
        assert "bay 12" in built.said()  # the earlier turn, from the run store
    spooled = [p for p in spool.rglob("*.jsonl") if p.read_text().strip()]
    assert spooled, list(spool.rglob("*"))
    memory_proxy.up()
    async with harness(memory_url=memory_proxy.url, spool_dir=str(spool)) as h:
        agent, _ = await armed("react", h, tmp_path, [], id=agent_id)
        await agent.run("Thanks.", user=user, thread="c")  # its first write replays the spool
        await h.writes.drain()

    async def delivered() -> list[dict]:
        rows = messages_of_run(run_id)
        return rows if any(r["role"] == "ASSISTANT" for r in rows) else []

    rows = await eventually(delivered, within=90)
    assert [r["content"] for r in rows if r["role"] == "USER"] == ["Which bay was it?"], rows
    assert not any(p.read_text().strip() for p in spool.rglob("*.jsonl"))  # nothing left


@pytest.mark.plan("MEM-10")
async def test_a_process_started_while_memory_is_down_runs_nothing(
    memory_proxy: FaultProxy, tmp_path: Path, uid: str
) -> None:
    """A limitation, documented here: the tenant a key speaks for is the memory service's word
    (``GET /v1/keys/self``), so a harness process started while memory is away cannot run at
    all — a restarted worker or server waits for memory, whatever the run needs."""
    memory_proxy.down()
    async with harness(memory_url=memory_proxy.url) as h:
        agent, _ = await armed("react", h, tmp_path, [], id=f"cold-{uid}")
        with pytest.raises(ConfigurationError, match="could not be reached"):
            await agent.run("Hello?", user=f"np-{uid}", thread="c")


@pytest.mark.plan("MEM-10")
async def test_a_context_answered_504_once_is_retried_and_reaches_the_model(
    memory_proxy: FaultProxy, tmp_path: Path, uid: str
) -> None:
    user = f"np-{uid}"
    async with harness() as h:  # a fact to find
        ctx = h.memory.client.bind(tenant_id=await h._tenant(), user_id=user)  # type: ignore[union-attr]
        await ctx.remember(f"My forklift licence number is FL-{uid}.")
    memory_proxy.rules[("POST", "/v1/context")] = ("504", 1)
    async with harness(memory_url=memory_proxy.url) as h:
        agent, built = await armed("react", h, tmp_path, [], id=f"ctx504-{uid}")
        done = await agent.run("What is my forklift licence number?", user=user, thread="c")
    assert done.status is RunStatus.SUCCESS
    statuses = [s for m, p, s in memory_proxy.hits if p == "/v1/context"]
    assert statuses[:2] == [504, 200], statuses  # retried, once
    assert f"FL-{uid}" in built.said()


@pytest.mark.plan("MEM-10")
async def test_a_context_that_stays_504_is_a_warning_and_the_run_answers(
    memory_proxy: FaultProxy, tmp_path: Path, uid: str
) -> None:
    memory_proxy.rules[("POST", "/v1/context")] = "504"
    async with harness(memory_url=memory_proxy.url) as h:
        agent, _ = await armed("react", h, tmp_path, [], id=f"ctx504b-{uid}")
        events = [e async for e in agent.stream("Hello?", user=f"oq-{uid}", thread="c")]
    finished = [e for e in events if e.type is RunEventType.RUN_FINISHED]
    assert finished and finished[-1].data.get("result") == "Done. ", events
    assert not [e for e in events if e.type is RunEventType.RUN_ERROR]
    assert "memory_unavailable" in json.dumps([e.data for e in events], default=str)


@pytest.mark.plan("MEM-10", "MEM-01", "MEM-08", "INV-2")
@pytest.mark.parametrize(
    "path", ["/v1/messages", "/v1/tools/invocations", "/v1/feedback"], ids=["turn", "tool", "fb"]
)
async def test_a_write_committed_but_answered_504_is_stored_once(
    memory_proxy: FaultProxy, tmp_path: Path, uid: str, path: str
) -> None:
    memory_proxy.rules[("POST", path)] = ("504-after", 1)
    plan = [("lookup", {"sku": "A-1"}), ("wire", {"amount": 7, "to": "acme"})]
    async with harness(memory_url=memory_proxy.url) as h:
        agent, _ = await armed("react", h, tmp_path, plan, id=f"w504-{uid}")
        paused = await agent.run("Check and pay.", user=f"pr-{uid}", thread="c")
        assert paused.interrupt is not None
        done = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="lee")
        assert done.status is RunStatus.SUCCESS
        await h.writes.drain()
    assert any(p == path and s == 504 for _, p, s in memory_proxy.hits), memory_proxy.hits
    run_id = done.run_id

    async def settled() -> bool:
        rows = messages_of_run(run_id)
        fb = feedback_rows(run_id)
        return (
            any(r["role"] == "ASSISTANT" for r in rows)
            and len(tool_records(run_id)) >= 2
            and any(r["target_kind"] == "tool_call" for r in fb)
        )

    assert await eventually(settled, within=90), (
        messages_of_run(run_id),
        tool_records(run_id),
        feedback_rows(run_id),
    )
    rows = messages_of_run(run_id)
    assert [r["content"] for r in rows if r["role"] == "USER"] == ["Check and pay."]
    assert len([r for r in rows if r["role"] == "ASSISTANT" and r["content"] == done.answer]) == 1
    assert sorted(r["tool_name"] for r in tool_records(run_id)) == ["lookup", "wire"]
    fb = feedback_rows(run_id)
    assert len([r for r in fb if r["target_kind"] == "tool_call"]) == 1, fb
    assert len([r for r in fb if r["target_kind"] == "run" and r["source"] == "system"]) == 1, fb


# --------------------------------------------------------------------------- MEM-11
OPEN = {"/health/live", "/health/ready", "/metrics", "/version"}
PLATFORM = {"/v1/admin/tenants", "/v1/admin/tenants/{tenant_id}"}
TENANT_ADMIN = {
    ("POST", "/v1/keys"),
    ("GET", "/v1/keys"),
    ("PATCH", "/v1/keys/{key_id}"),
    ("DELETE", "/v1/keys/{key_id}"),
    ("POST", "/v1/workspaces"),
    ("DELETE", "/v1/workspaces/{workspace_id}"),
    ("PUT", "/v1/workspaces/{workspace_id}/members/{principal_ref}"),
    ("DELETE", "/v1/workspaces/{workspace_id}/members/{principal_ref}"),
    ("PUT", "/v1/model-key"),
    ("DELETE", "/v1/model-key"),
    ("PUT", "/v1/model-key/policy"),
    ("GET", "/v1/reads"),
    ("GET", "/v1/feedback/pending"),
    ("POST", "/v1/feedback/{feedback_id}/approve"),
    ("POST", "/v1/feedback/{feedback_id}/dismiss"),
}


def operations() -> list[tuple[str, str]]:
    doc = json.loads(MEMORY_OPENAPI.read_text())
    return [
        (m.upper(), p)
        for p, item in doc["paths"].items()
        for m in item
        if m in ("get", "post", "put", "patch", "delete")
    ]


def concrete(path: str) -> str:
    return path.replace("{tenant_id}", "feat").replace("{", "x-").replace("}", "")


@pytest.mark.plan("MEM-11", "SEC-06")
async def test_every_route_but_the_probes_refuses_a_missing_or_unknown_key(invariants: Any) -> None:
    invariants.allow("INV-7", "requests made malformed on purpose: only the key is probed")
    async with httpx.AsyncClient(base_url=MEMORY_URL or "", timeout=30) as c:
        for method, path in operations():
            if path in OPEN:
                continue
            for headers in ({}, {"X-API-Key": "not-a-key"}):
                got = await c.request(method, concrete(path), headers=headers, json={})
                assert got.status_code == 401, (method, path, headers, got.status_code, got.text)


@pytest.mark.plan("MEM-11", "SEC-06")
async def test_the_platform_and_admin_routes_refuse_lesser_keys(invariants: Any) -> None:
    invariants.allow("INV-7", "requests made malformed on purpose: only the role is probed")
    async with httpx.AsyncClient(base_url=MEMORY_URL or "", timeout=30) as c:
        for method, path in operations():
            if path in PLATFORM:
                for key in (API_KEY, ADMIN_KEY):
                    got = await c.request(
                        method, concrete(path), headers={"X-API-Key": key or ""}, json={}
                    )
                    assert got.status_code == 403, (method, path, got.status_code, got.text)
            elif (method, path) in TENANT_ADMIN:
                got = await c.request(
                    method, concrete(path), headers={"X-API-Key": API_KEY or ""}, json={}
                )
                assert got.status_code == 403, (method, path, got.status_code, got.text)
        # and what each may do, it does: the platform lists tenants, the admin its keys
        listed = await c.get("/v1/admin/tenants", headers={"X-API-Key": PLATFORM_KEY or ""})
        assert listed.status_code == 200 and "feat" in listed.text
        keys = await c.get("/v1/keys", headers={"X-API-Key": ADMIN_KEY or ""})
        assert keys.status_code == 200, keys.text
        assert API_KEY not in keys.text and (ADMIN_KEY or "x") not in keys.text  # no secret
