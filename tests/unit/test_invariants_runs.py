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


def _tried(*statuses: str | None) -> list[dict[str, Any]]:
    """One attempt's events of the call ``run_a:k:0`` begun once per status — its result's
    status, or ``None``: begun with no result (cut short) — then the run's end."""
    ref, n = "run_a:k:0", 0
    events: list[dict[str, Any]] = [{"type": "RUN_STARTED", "sequence": 0}]
    for status in statuses:
        events.append({"type": "TOOL_CALL_START", "tool_call_id": ref, "sequence": n + 1})
        events.append({"type": "TOOL_CALL_END", "tool_call_id": ref, "sequence": n + 2})
        n += 2
        if status is not None:
            n += 1
            result = {"tool": "flaky", "status": status, "output": "x"}
            events.append(
                {"type": "TOOL_CALL_RESULT", "tool_call_id": ref, "sequence": n, "data": result}
            )
    events.append({"type": "RUN_FINISHED", "sequence": n + 1})
    return [{"event": e} for e in events]


def test_a_call_its_graph_tries_again_after_its_error_started_once(tmp_path: Path) -> None:
    """INV-10: a node's ``RetryPolicy`` runs a failed read again (``tools/convert/langchain``:
    the try ended and recorded, the node's policy tries it again, the harness not on top) — the
    same call, by the same id, begun again after its error: one start, not two (WF-07, a read
    run exactly twice). A call begun again with no error before it is still a second start."""
    inv = Invariants(tmp_path)
    assert inv._inv10({"run_a": _tried("error", "ok")}) == []
    assert inv._inv10({"run_a": _tried("error", "error", "ok")}) == []
    twice = ["INV-10 run run_a: tool calls started more than once: ['run_a:k:0']"]
    assert inv._inv10({"run_a": _tried("ok", "ok")}) == twice
    assert inv._inv10({"run_a": _tried(None, "ok")}) == twice


def _memory(path: str, response: Any, key: str, request: Any = None) -> Exchange:
    return Exchange(
        service="memory",
        method="GET" if request is None else "POST",
        path=path,
        query={},
        request=request,
        status=200,
        response=response,
        auth={"X-API-Key": key},
    )


def test_a_second_tenants_key_reading_its_own_tenant_crosses_no_boundary(tmp_path: Path) -> None:
    """INV-5: a test that uses a second tenant's key on purpose (s02: another tenant's key reads
    none of the runs) has that key's harness ask memory whose key it is (``GET /v1/keys/self``),
    and memory answers the key's own tenant: no boundary crossed. A memory answer naming a
    tenant other than its key's own still is one."""
    inv = Invariants(tmp_path)
    inv.traffic.exchanges += [
        _memory("/v1/keys/self", {"tenant_id": "scen"}, "key-a"),
        _memory("/v1/context", {"items": []}, "key-a", {"scope": {"tenant_id": "scen"}}),
        _memory("/v1/keys/self", {"tenant_id": "scen2"}, "key-b"),
        _memory("/v1/threads", {"threads": [{"tenant_id": "scen2"}]}, "key-b"),
    ]
    assert inv._inv5() == []
    inv.traffic.exchanges.append(_memory("/v1/threads", {"tenant_id": "scen2"}, "key-a"))
    assert inv._inv5() == [
        "INV-5 memory GET /v1/threads answered tenant scen2 to a key of tenant scen"
    ]
