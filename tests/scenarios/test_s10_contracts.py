"""Scenario 10 — contracts (CON, INV-7). Every scenario's traffic with the memory service and
agent-runs is checked as it happens (``tests/support/invariants.py``): each request and answer
against the services' committed OpenAPI documents, and the runs, records, events, pauses,
resumes, schedules and feedback against agent-contracts' own models. This test drives one user's
day through every surface the harness talks to the services from — a chat turn with memory and a
tool, a pause and its resume, a queued run a worker claims, a schedule, a person's feedback — and
asserts the checker saw each operation it names, with no violation."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.scenarios.brains import command, token, wrapped
from tests.scenarios.kit import TOOLS, Served, asked_id
from tests.support.invariants import Invariants
from trellis import Harness
from trellis.contracts import RunStatus

pytestmark = [pytest.mark.timeout(600), pytest.mark.plan("CON", "INV-7")]

#: (service, method, path prefix) each of which the day must have exercised
EXPECTED = [
    ("runs", "POST", "/v1/runs"),
    ("runs", "POST", "/v1/runs/{id}/events"),
    ("runs", "POST", "/v1/runs/{id}/heartbeat"),
    ("runs", "POST", "/v1/runs/{id}/pause"),
    ("runs", "POST", "/v1/runs/{id}/resume"),
    ("runs", "POST", "/v1/runs/{id}/finish"),
    ("runs", "GET", "/v1/runs/{id}"),
    ("runs", "POST", "/v1/schedules"),
    ("memory", "POST", "/v1/context"),
    ("memory", "POST", "/v1/messages"),
    ("memory", "POST", "/v1/tools/invocations"),
    ("memory", "POST", "/v1/feedback"),
    ("memory", "PUT", "/v1/tools/catalog"),
]


def _shape(path: str) -> str:
    parts = path.split("/")
    return "/".join("{id}" if p.startswith(("run_", "sch_")) else p for p in parts)


async def test_a_users_day_validates_against_every_contract(
    h: Harness, uid: str, tmp_path: Path, invariants: Invariants
) -> None:
    agent, _ = await wrapped(h, "react", TOOLS, tmp_path, id=f"s10-{uid}")
    user, thread = f"ada-{uid}", f"t-{uid}"
    turn = await agent.run(
        f"{token(uid, 1)} {command('lookup', key='k')}", user=user, thread=thread
    )
    assert turn.status is RunStatus.SUCCESS, turn.error
    paused = await agent.run(
        f"{token(uid, 2)} {command('transfer', amount=1, to='x')}", user=user, thread=thread
    )
    done = await agent.resume(asked_id(paused), "approve", reviewer=user)
    assert done.status is RunStatus.SUCCESS
    await h.feedback(done.run_id, "confirm")
    made = await agent.schedule("0 9 * * 1", user=user, input=f"{token(uid, 3)} weekly")
    handle = await agent.start(f"{token(uid, 4)} queued", user=user, thread=thread)
    worker = await Served(
        {
            "mode": "worker",
            "agents": [{"adapter": "react", "id": agent.id}],
            "lease": 10,
            "hold": 10,
        },
        tmp_path,
        "w",
    ).started()
    try:
        queued = await handle.result(wait=120)
    finally:
        worker.stop()
        await h.runs.schedules.delete(made.schedule_id)
    assert queued.status is RunStatus.SUCCESS, queued.error
    await h.writes.drain()
    invariants.load_files()
    seen = {(e.service, e.method, _shape(e.path)) for e in invariants.traffic.exchanges}
    missing = [x for x in EXPECTED if x not in seen]
    assert not missing, missing
    assert not invariants.traffic.violations, invariants.traffic.violations
