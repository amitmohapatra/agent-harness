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


def _tool_events(tool: str, args: dict[str, Any], status: str, output: str) -> list[dict[str, Any]]:
    ref = "run_a:k:0"
    return [
        {"event": {"type": "TOOL_CALL_ARGS", "tool_call_id": ref, "data": {"args": args}}},
        {
            "event": {
                "type": "TOOL_CALL_RESULT",
                "tool_call_id": ref,
                "data": {"tool": tool, "status": status, "output": output},
            }
        },
    ]


def _with_memory(inv: Invariants) -> Invariants:
    inv.traffic.exchanges.append(
        Exchange(
            service="memory",
            method="GET",
            path="/v1/keys/self",
            query={},
            request=None,
            status=200,
            response={},
        )
    )
    return inv


def test_a_call_the_runs_end_cut_short_owes_no_tool_record(tmp_path: Path) -> None:
    """INV-2: a call the run's time limit cut short (``tools.bridge._cut``: "<tool> was cut
    short: the run ran out of time") has no outcome of its own, as a cancelled one has none;
    a call that timed out on its own limit is an outcome, and owes its record."""
    inv = _with_memory(Invariants(tmp_path))
    cut = _tool_events("wait", {"s": 60}, "timeout", "wait was cut short: the run ran out of time")
    assert inv._inv2_tools({"run_a": cut}, {"run_a": {}}) == []
    own = _tool_events("wait", {"s": 60}, "timeout", "wait took longer than its 5 s limit")
    assert inv._inv2_tools({"run_a": own}, {"run_a": {}}) == [
        "INV-2 run run_a: calls that ran have no tool record: ['wait:{\"s\": 60}']"
    ]
