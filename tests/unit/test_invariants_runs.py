"""The invariants checker (``tests/support/invariants.py``) finds every run a test's processes
made or claimed: a run created with ``POST /v1/runs`` (agent-runs answers the record) and one a
worker claimed (``POST /v1/runs/claim`` answers ``{"run": <record>, "lease": ...}``), such as a
schedule's fire, which no process of the test created."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tests.support.invariants import Exchange, Invariants


def _exchange(path: str, response: Any, status: int = 200) -> Exchange:
    return Exchange(
        service="runs",
        method="POST",
        path=path,
        query={},
        request={},
        status=status,
        response=response,
    )


def test_a_run_a_worker_claimed_is_one_the_test_touched(tmp_path: Path) -> None:
    inv = Invariants(tmp_path)
    claimed = _exchange(
        "/v1/runs/claim",
        {"run": {"run_id": "run_fired", "status": "RUNNING"}, "lease": {"worker_id": "w"}},
    )
    created = _exchange("/v1/runs", {"run_id": "run_made", "status": "QUEUED"}, status=201)
    inv.traffic.exchanges += [claimed, created]
    found = inv.runs()
    assert found == {"run_fired": claimed, "run_made": created}


def test_an_empty_claim_or_a_refused_create_names_no_run(tmp_path: Path) -> None:
    inv = Invariants(tmp_path)
    inv.traffic.exchanges += [
        _exchange("/v1/runs/claim", None, status=204),
        _exchange("/v1/runs", {"code": "CONFLICT"}, status=409),
        _exchange("/v1/runs/claim", {"lease": {}}),
    ]
    assert inv.runs() == {}
