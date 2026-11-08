"""``query_result(results=[...])``: past tool results, named by the reference an earlier
conversation's steps list gives (``result msg_...``), loaded as tables ``t1``, ``t2``… beside
the run's own, read through the memory service's message read (its own audience rules) — a
join across chats with exact numbers; another user's refused; one deleted said; the limits;
and where each table came from, dated, for the explanation."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest

from tests.support.memory import FakeMemoryService
from trellis.harness import runtime as runtime_module
from trellis.harness.clients.memory import RunMemory
from trellis.harness.journal import Journal
from trellis.harness.tools import query, results

#: Monday's list (an earlier chat's whole result) and today's stock (this run's).
MONDAY = "sku,units\n" + "\n".join(f"S{n},{100 - n}" for n in range(30))
TODAY = "sku,stock\n" + "\n".join(f"S{n},{n * 2}" for n in range(30))


class Run:
    """A run in progress, as ``tools.query`` reads it: its journal and its memory."""

    def __init__(self, journal: Journal, memory: RunMemory) -> None:
        self.replay = SimpleNamespace(journal=journal)
        self.toolbox: dict[str, object] = {}
        self.native: dict[str, object] = {}
        self.run_memory = memory


@pytest.fixture
def service() -> FakeMemoryService:
    return FakeMemoryService()


def _said(service: FakeMemoryService, user: str, content: str) -> str:
    """A tool result said in ``user``'s earlier chat: its message id."""
    ctx = service.client().bind(tenant_id="acme", user_id=user, thread_id=f"thr_{user}")
    message = {"role": "TOOL", "kind": "INTERNAL", "content": content, "tool_call_id": "c1"}
    asyncio.run(ctx.history.add([message]))
    return next(
        m["message_id"] for t in service.threads.values() for m in t if m["content"] == content
    )


@pytest.fixture
def run(service: FakeMemoryService) -> Iterator[Run]:
    ctx = service.client().bind(tenant_id="acme", user_id="ann", thread_id="thr_new")
    made = Run(Journal(), RunMemory(None, ctx))  # type: ignore[arg-type]
    token = runtime_module._current.set(made)  # type: ignore[arg-type]
    yield made
    runtime_module._current.reset(token)


def ask(**args: Any) -> str:
    return asyncio.run(query.query(args))


def test_a_past_result_joins_this_runs_with_exact_numbers(
    run: Run, service: FakeMemoryService
) -> None:
    ref = _said(service, "ann", MONDAY)
    path = results.keep(TODAY)
    assert path is not None
    sql = (
        "SELECT t1.sku, t1.units, result.stock, t1.units - result.stock AS gap FROM t1 "
        "JOIN result USING (sku) ORDER BY t1.units DESC LIMIT 2"
    )
    said = ask(file_path=path, sql=sql, results=[ref])
    assert said.endswith("sku,units,stock,gap\nS0,100,0,100\nS1,99,2,97"), said
    # where t1 came from, dated: what an explanation names
    assert said.startswith(f"t1 is past result {ref}, said 2026-"), said


def test_past_results_alone_need_no_file(run: Run, service: FakeMemoryService) -> None:
    """A new chat keeps no result of its own: the past one is the table, ``result`` too."""
    ref = _said(service, "ann", MONDAY)
    sql = "SELECT sku, units FROM t1 ORDER BY units DESC LIMIT 1 OFFSET 6"
    assert ask(sql=sql, results=[ref]).endswith("sku,units\nS6,94")
    assert ask(sql="SELECT count(*) AS n FROM result", results=[ref]).endswith("n\n30")


def test_another_users_result_is_refused(run: Run, service: FakeMemoryService) -> None:
    ref = _said(service, "bob", MONDAY)
    said = ask(sql="SELECT * FROM t1", results=[ref])
    assert said == (
        f"Error: past result {ref} is refused: it is not this user's to read. Nothing was run."
    )


def test_a_deleted_result_is_said_plainly(run: Run) -> None:
    said = ask(sql="SELECT * FROM t1", results=["msg_404"])
    assert said == (
        "Error: past result msg_404 is no longer kept (deleted, or past its retention): "
        "call the tool again. Nothing was run."
    )


def test_the_limits_hold_for_past_results(
    run: Run, service: FakeMemoryService, monkeypatch: pytest.MonkeyPatch
) -> None:
    ref = _said(service, "ann", MONDAY)
    many = [ref] * (query.PAST_RESULTS + 1)
    assert "results names at most" in ask(sql="SELECT 1", results=many)
    assert ask(sql="SELECT 1") == query.not_run(
        query.QUERY_RESULT, "name file_path, results, or both"
    )
    monkeypatch.setattr(query, "LOAD_CHARS", len(MONDAY) - 1)
    assert "more than a query here loads" in ask(sql="SELECT 1", results=[ref])


def test_a_ref_to_what_is_no_tool_result_is_said(run: Run, service: FakeMemoryService) -> None:
    ctx = service.client().bind(tenant_id="acme", user_id="ann", thread_id="thr_ann")
    asyncio.run(ctx.history.add([{"role": "ASSISTANT", "content": "S0 leads."}]))
    [said] = [m["message_id"] for t in service.threads.values() for m in t]
    assert ask(sql="SELECT 1", results=[said]) == (
        f"Error: {said} is not a tool's result. Nothing was run."
    )


def test_no_memory_no_past_results(run: Run) -> None:
    run.run_memory = None  # type: ignore[assignment]
    said = ask(sql="SELECT 1", results=["msg_1"])
    assert said == "Error: past results are read from memory, which this run does not have."


def test_the_refs_a_context_names_are_found() -> None:
    """Read from the steps lists the memory context renders: the service's own format."""
    text = '  1. sales_summary(group_by="sku"): ok, result msg_01ABC\n  2. x(): result msg_7'
    assert query.past_refs(text) == ("msg_01ABC", "msg_7")
    assert query.past_refs("no steps here") == ()


def test_a_path_this_run_keeps_given_as_a_result_is_that_table(
    run: Run, service: FakeMemoryService
) -> None:
    """A model that names this run's own kept path in ``results`` gets that result as the
    table, beside a past one: never sent to memory, never refused."""
    ref = _said(service, "ann", MONDAY)
    path = results.keep(TODAY)
    assert path is not None
    sql = "SELECT t1.units - t2.stock AS gap FROM t1 JOIN t2 USING (sku) WHERE sku = 'S1'"
    said = ask(sql=sql, results=[ref, path])
    assert said.endswith("gap\n97"), said
    assert f"t2 is this run's result {path}" in said
