"""Harness features on every adapter, end to end against the real memory service and agent-runs
(the matrix proves them in process with fakes; here the run store, the inbox and memory are the
real services):

* HAR-ASK ``ask_user``: the run is PAUSED in agent-runs with the structured question (in the
  inbox of the run's user), the answer resumes it from another process, the model reads it;
* HAR-SUB a sub-agent: the child is a run of its own in agent-runs naming its parent, for its
  parent's user; cancelling a parent while its child waits cancels the child too;
* HAR-BUDGET a budget: past ``max_tool_calls`` the run ends ``PARTIAL`` (``BUDGET_EXCEEDED``)
  and agent-runs keeps the spending with the record;
* HAR-DATE the date line: the model is told today's date;
* HAR-OBS redaction: a credential a tool is given never reaches a span, the rest does;
* HAR-LARGE large results: kept, read to the last page (``read_file``) and counted over every
  row (``query_result``) where the adapter has them (matrix F26r, F26q).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from tests.features.conftest import ADAPTERS, eventually, harness, runs_http, runs_sql
from tests.features.kit import armed, ledger, lookup, record
from tests.support.arm import arm
from trellis import Budget, Runtime, ask_user, tool
from trellis.contracts import RunStatus
from trellis.harness import telemetry
from trellis.harness.journal import content_key

pytestmark = pytest.mark.timeout(300)
#: the adapters that keep a large result and page it, and query it (matrix F26r, F26q)
PAGED = ("react", "middleware", "openai_agents", "claude_agent_sdk", "kimi")
SECRET = "sk-live-0123456789abcdef0123456789abcdef"


@pytest.mark.plan("HAR-ASK", "F36u", "RUN-07")
@pytest.mark.parametrize("adapter", ADAPTERS)
async def test_ask_user_pauses_in_agent_runs_and_the_answer_resumes_elsewhere(
    adapter: str, tmp_path: Path, uid: str
) -> None:
    user, agent_id = f"aki-{uid}", f"ask-{adapter}-{uid}"
    plan = [("ask_user", {"question": "Which day?", "kind": "date"})]
    async with harness() as h:
        agent, _ = await armed(adapter, h, tmp_path, plan, id=agent_id, tools=[ask_user()])
        paused = await agent.run("Book the delivery.", user=user, thread="c")
        assert paused.status is RunStatus.PAUSED and paused.interrupt is not None, paused
        assert paused.interrupt.component == "ask_user"
        assert paused.interrupt.expects == {"type": "string", "format": "date"}
        inbox = await h.inbox(f"user:{user}")
        assert [s.run_id for s in inbox] == [paused.run_id]  # waits on the run's user
    stored = await record(paused.run_id)
    assert stored.status is RunStatus.PAUSED and stored.awaiting.question.startswith("Which day?")
    async with harness() as h:  # another process answers
        agent, _ = await armed(adapter, h, tmp_path, plan, id=agent_id, tools=[ask_user()])
        bad = None
        try:
            await agent.resume(paused.interrupt.interrupt_id, "answer", answer="next week",
                               reviewer=user)
        except Exception as exc:
            bad = exc
        assert bad is not None  # an answer that does not fit the question is refused
        done = await agent.resume(paused.interrupt.interrupt_id, "answer", answer="2026-11-20",
                                  reviewer=user)
    assert done.status is RunStatus.SUCCESS and done.answer == "Done. 2026-11-20", done
    assert (await record(done.run_id)).last_resolution.answer == "2026-11-20"


@pytest.mark.plan("HAR-SUB", "RUN-10", "F30p", "X10")
@pytest.mark.parametrize("adapter", ADAPTERS)
async def test_a_sub_agent_is_a_child_run_and_cancelling_the_parent_cancels_it(
    adapter: str, tmp_path: Path, uid: str
) -> None:
    user = f"bea-{uid}"
    async with harness() as h:
        async def research(input: str, agent: Runtime) -> str:
            """Research a topic and report what was found."""
            return str(await agent.tools.call("lookup", sku=str(input)))

        child = arm(h.wrap(research, id=f"research-{adapter}-{uid}"), [lookup])
        plan = [(child.id, {"message": "A-1"})]
        agent, _ = await armed(adapter, h, tmp_path, plan, id=f"parent-{adapter}-{uid}",
                               tools=[child])
        done = await agent.run("Research A-1.", user=user, thread="c")
        assert done.status is RunStatus.SUCCESS, done
        assert "stock of A-1 is 3" in str(done.answer), done.answer
    kids = runs_sql("select run_id, agent_id, status, user_id, parent_run_id from agent_runs"
                    " where parent_run_id = :1", done.run_id)
    assert [(k["agent_id"], k["status"], k["user_id"]) for k in kids] == [
        (f"research-{adapter}-{uid}", "SUCCESS", user)
    ], kids
    async with runs_http() as c:
        listed = await c.get("/v1/runs", params={"parent_run_id": done.run_id})
        assert [r["run_id"] for r in listed.json()] == [kids[0]["run_id"]]

    # a child that waits for a person; the parent cancelled: both end CANCELLED
    async with harness() as h:
        asking = arm(h.wrap(_asking_child(), id=f"asker-{adapter}-{uid}"), [ask_user()])
        plan = [(asking.id, {"message": "which day"})]
        agent, _ = await armed(adapter, h, tmp_path, plan, id=f"parent2-{adapter}-{uid}",
                               tools=[asking])
        paused = await agent.run("Ask the helper.", user=user, thread="c2")
        assert paused.status is RunStatus.PAUSED, paused
        await agent.cancel(paused.run_id, reason="the user changed their mind")
    parent = await record(paused.run_id)
    assert parent.status is RunStatus.CANCELLED

    async def children_cancelled() -> list[dict]:
        rows = runs_sql("select status from agent_runs where parent_run_id = :1", paused.run_id)
        return rows if rows and all(r["status"] == "CANCELLED" for r in rows) else []

    assert await eventually(children_cancelled, within=60), runs_sql(
        "select status from agent_runs where parent_run_id = :1", paused.run_id)


def _asking_child() -> Any:
    async def asker(input: str, agent: Runtime) -> str:
        """Ask the user which day."""
        return str(await agent.tools.call("ask_user", question="Which day?", kind="date"))

    return asker


@pytest.mark.plan("HAR-BUDGET", "LR-03")
@pytest.mark.parametrize("adapter", ADAPTERS)
async def test_a_budget_ends_the_run_partial_and_is_kept_with_its_record(
    adapter: str, tmp_path: Path, uid: str
) -> None:
    plan = [("lookup", {"sku": "A-1"}), ("lookup", {"sku": "B-2"}), ("lookup", {"sku": "C-3"})]
    async with harness() as h:
        agent, _ = await armed(adapter, h, tmp_path, plan, id=f"budget-{adapter}-{uid}")
        done = await agent.run("Check three.", user=f"cy-{uid}", thread="c",
                               budget=Budget(max_tool_calls=2))
    assert done.status is RunStatus.PARTIAL, done
    assert done.error is not None and done.error.code == "BUDGET_EXCEEDED", done.error
    assert ledger().count("lookup:C-3") == 0  # the third call never ran
    stored = await record(done.run_id)
    assert stored.status is RunStatus.PARTIAL
    assert stored.metadata["budget"]["max_tool_calls"] == 2, stored.metadata


@pytest.mark.plan("HAR-DATE")
@pytest.mark.parametrize("adapter", [a for a in ADAPTERS if a != "function"])
async def test_the_model_is_told_todays_date(adapter: str, tmp_path: Path, uid: str) -> None:
    async with harness() as h:
        agent, built = await armed(adapter, h, tmp_path, [], id=f"date-{adapter}-{uid}")
        await agent.run("What day is it?", user=f"di-{uid}", thread="c")
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    assert today in built.said(), built.said()[:1500]


@pytest.fixture
def spans(monkeypatch: pytest.MonkeyPatch) -> InMemorySpanExporter:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(telemetry, "_tracer", provider.get_tracer("features"))
    return exporter


@tool(side_effects="read")
def notify(email: str, api_key: str) -> str:
    """Notify someone (the key authenticates the mail service)."""
    return f"notified {email}"


@pytest.mark.plan("HAR-OBS", "F61", "INV-8")
@pytest.mark.parametrize("adapter", ADAPTERS)
async def test_a_credential_never_reaches_a_trace(
    adapter: str, tmp_path: Path, uid: str, spans: InMemorySpanExporter
) -> None:
    email = f"ops-{uid}@example.com"
    plan = [("notify", {"email": email, "api_key": SECRET})]
    async with harness() as h:
        agent, _ = await armed(adapter, h, tmp_path, plan, id=f"obs-{adapter}-{uid}",
                               tools=[notify])
        done = await agent.run("Tell ops.", user=f"ed-{uid}", thread="c")
    assert done.status is RunStatus.SUCCESS, done
    traced = json.dumps([dict(s.attributes or {}) for s in spans.get_finished_spans()],
                        default=str)
    assert spans.get_finished_spans(), "no span"
    assert SECRET not in traced
    assert email in traced  # the rest goes as it is


@pytest.mark.plan("HAR-LARGE", "FS", "F26r", "F26q")
@pytest.mark.parametrize("adapter", PAGED)
async def test_a_large_result_is_kept_paged_to_its_end_and_counted_over_every_row(
    adapter: str, tmp_path: Path, uid: str
) -> None:
    rows = 50_000
    text = "\n".join(f"row {i:06d} sku-{i % 97:03d} qty {i % 13}" for i in range(rows))
    path = f"/large_tool_results/{content_key('result', text)}"
    plan = [
        ("big_report", {"rows": rows}),
        ("read_file", {"file_path": path, "offset": rows - 3}),
        ("query_result", {"file_path": path,
                          "sql": "SELECT count(*) AS n, max(line) AS last FROM result"}),
    ]
    async with harness() as h:
        agent, built = await armed(adapter, h, tmp_path, plan, id=f"large-{adapter}-{uid}")
        done = await agent.run("Count the report.", user=f"fa-{uid}", thread="c")
    assert done.status is RunStatus.SUCCESS, done
    assert f"n,last\n{rows},row {rows - 1:06d}" in str(done.answer), str(done.answer)[:500]
    said = built.said()
    assert f"row {rows - 2:06d}" in said  # the last page was read
    assert len(said) < len(text)  # never the whole result
