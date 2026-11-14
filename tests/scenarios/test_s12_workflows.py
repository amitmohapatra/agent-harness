"""Scenario 12 — workflows, each as its framework builds them (``tests/scenarios/workflows.py``):
a hand-built LangGraph graph (branches, ``Send`` fan-out, a subgraph, an approval and LangGraph's
own ``interrupt()`` mid-graph; checkpointed and not), Deep Agents planning with ``task``
sub-agents, an OpenAI Agents handoff chain (with a cycle) and an agent called as a tool, and a
``ReAct`` multi-step task. Each is checked step by step with an approval inside and resumed on
another worker process, with its worker SIGKILLed mid-workflow, cancelled mid-workflow, and its
records and spending exact. (Claude Workflows: ``test_s05_background.py``, pending bg-claude-da.)
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from tests.scenarios import workflows
from tests.scenarios.brains import command, parse, token, wrapped
from tests.scenarios.conftest import harness
from tests.scenarios.kit import TOOLS, Served, eventually, ledger, record
from trellis import Harness
from trellis.contracts import RunEventType, RunStatus

#: about what a workflow does, not whether its writes ask: a write runs announced (``writes_unasked``)
pytestmark = [pytest.mark.timeout(900), pytest.mark.usefixtures("writes_unasked")]
GRAPHS = ["wf_graph", "wf_graph_cp"]


def _message(
    adapter: str, tok: str, *, pay: bool = True, slowly: bool = False, confirm: bool = False
) -> str:
    """What the user asks each workflow: its steps (the graphs' are built in; the others' said)."""
    words = (
        ("pay " if pay else "") + ("slowly " if slowly else "") + ("confirm " if confirm else "")
    )
    wait = 8 if slowly else 0.2
    if adapter in GRAPHS:
        return f"{tok} {words}"
    if adapter == "wf_deepagents":
        todo = command(
            "write_todos",
            todos=[
                {"content": "fetch", "status": "in_progress"},
                {"content": "pay", "status": "pending"},
            ],
        )
        sub = f"{tok.replace('tok_', 'tok_s', 1)} {command('note', text=f'step1 {tok}')} {command('wait', seconds=wait)}"
        task = command("task", description=sub, subagent_type="helper")
        steps = [todo, task] + (
            [command("transfer", amount=9, to="wf", meta={"step": "pay"})] if pay else []
        )
        return f"{tok} {' '.join(steps)} {command('note', text=f'final {tok}')}"
    if adapter == "wf_handoffs":
        steps = [
            command("note", text=f"step1 {tok}"),
            command("wait", seconds=wait),
            command("transfer_to_billing"),
        ]
        steps += [command("transfer", amount=9, to="wf", meta={"step": "pay"})] if pay else []
        steps += [command("transfer_to_triage"), command("note", text=f"final {tok}")]
        return f"{tok} {' '.join(steps)}"
    if adapter == "wf_tools":
        asked = f"{tok.replace('tok_', 'tok_x', 1)} {command('lookup', key='k0')}"
        steps = [
            command("note", text=f"step1 {tok}"),
            command("wait", seconds=wait),
            command("specialist", input=asked),
        ]
        steps += [command("transfer", amount=9, to="wf", meta={"step": "pay"})] if pay else []
        return f"{tok} {' '.join(steps)} {command('note', text=f'final {tok}')}"
    # ReAct: a multi-step task, its reads at once
    fetch = command("batch", calls=[["lookup", {"key": f"k{n}"}] for n in range(3)])
    steps = [fetch, command("note", text=f"step1 {tok}"), command("wait", seconds=wait)]
    steps += [command("transfer", amount=9, to="wf", meta={"step": "pay"})] if pay else []
    return f"{tok} {' '.join(steps)} {command('note', text=f'final {tok}')}"


ALL = [*GRAPHS, "wf_deepagents", "wf_handoffs", "wf_tools", "react"]


def _worker(adapter: str, agent_id: str, **more: Any) -> dict[str, Any]:
    return {
        "mode": "worker",
        "agents": [{"adapter": adapter, "id": agent_id}],
        "lease": 6,
        "hold": 6,
        **more,
    }


async def _starts(agent: Any, run_id: str) -> list[str]:
    return [
        e.data.get("tool")
        for e in [x async for x in agent.events(run_id)]
        if e.type is RunEventType.TOOL_CALL_START
    ]


def _book(tok: str, book: Path) -> list[str]:
    lines = book.read_text().splitlines() if book.exists() else []
    return [line for line in lines if tok in line or line.startswith("transfer:9:wf")]


@pytest.mark.plan("WF-01", "WF-02", "WF-05", "WF-06")
@pytest.mark.parametrize("decision", ["approve", "reject"])
@pytest.mark.parametrize("adapter", ALL)
async def test_an_approval_mid_workflow_resumed_on_another_worker(
    adapter: str, decision: str, uid: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    book = tmp_path / "ledger"
    monkeypatch.setenv("SCEN_LEDGER", str(book))
    agent_id = f"s12a-{adapter}-{uid}"
    tok = token(uid, 1)
    async with harness() as h:
        agent, _ = await wrapped(h, adapter, TOOLS, tmp_path, id=agent_id)
        handle = await agent.start(_message(adapter, tok), user=f"ada-{uid}", thread=f"t-{uid}")
        one = await Served(_worker(adapter, agent_id), tmp_path, "w1").started()
        try:
            paused = await handle.result(wait=240)
        finally:
            one.stop()
        assert paused.status is RunStatus.PAUSED and paused.interrupt is not None, (
            paused,
            one.log.read_text()[-3000:],
        )
        assert (
            paused.interrupt.tool_call is not None and paused.interrupt.tool_call.tool == "transfer"
        )
        assert [x for x in _book(tok, book) if x.startswith("note:step1")] == [
            f"note:step1 {tok}"
        ]  # done before the pause
        await agent.resume(
            paused.interrupt.interrupt_id,
            decision,
            answer="not now" if decision == "reject" else None,
            reviewer=f"ada-{uid}",
        )
        two = await Served(_worker(adapter, agent_id), tmp_path, "w2").started()
        try:
            done = await handle.result(wait=240)
        finally:
            two.stop()
        assert done.status is RunStatus.SUCCESS, (done.error, two.log.read_text()[-3000:])
        assert parse(done.answer).mine == tok, done.answer
        ran = _book(tok, book)
        assert ran.count(f"note:step1 {tok}") == 1, ran  # the step before the pause: not run again
        assert ran.count(f"note:final {tok}") == 1, ran
        payments = [x for x in ran if x.startswith("transfer:9:wf")]
        assert len(payments) == (1 if decision == "approve" else 0), ran
        # the spending and the records: every call once, the run's tool calls those its events show
        starts = await _starts(agent, done.run_id)
        assert starts.count("transfer") == (1 if decision == "approve" else 0), starts
        if adapter in GRAPHS:
            assert len(starts) == workflows.GRAPH_CALLS - (0 if decision == "approve" else 1), (
                starts
            )
        events = [e async for e in agent.events(done.run_id)]
        spent = [e.data["spent"] for e in events if e.type is RunEventType.RUN_FINISHED][-1]
        assert int(spent["tool_calls"]) >= len(set(starts)), (spent, starts)


@pytest.mark.plan("WF-02")
@pytest.mark.parametrize("adapter", ["wf_graph", "wf_graph_cp"])
async def test_langgraphs_own_interrupt_mid_graph(
    h: Harness, adapter: str, uid: str, tmp_path: Path
) -> None:
    agent, _ = await wrapped(h, adapter, TOOLS, tmp_path, id=f"s12i-{adapter}-{uid}")
    tok = token(uid, 1)
    before = len(ledger())
    asked = await agent.run(
        _message(adapter, tok, confirm=True), user=f"ada-{uid}", thread=f"t-{uid}"
    )
    assert asked.status is RunStatus.PAUSED and asked.interrupt is not None, asked.error
    went = await agent.resume(
        asked.interrupt.interrupt_id, "answer", answer="yes", reviewer=f"ada-{uid}"
    )
    assert went.status is RunStatus.PAUSED and went.interrupt is not None, (
        went
    )  # then the payment asks
    assert went.interrupt.tool_call is not None and went.interrupt.tool_call.tool == "transfer"
    done = await agent.resume(went.interrupt.interrupt_id, "approve", reviewer=f"ada-{uid}")
    assert done.status is RunStatus.SUCCESS, done.error
    assert "confirmed:yes" in done.answer
    ran = ledger()[before:]
    assert (
        ran.count(f"note:step1 {tok}") == 1 and sum(x.startswith("transfer:9:wf") for x in ran) == 1
    ), ran


@pytest.mark.plan("WF-03")
@pytest.mark.parametrize("adapter", ALL)
async def test_a_worker_killed_mid_workflow(
    adapter: str, uid: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    book = tmp_path / "ledger"
    monkeypatch.setenv("SCEN_LEDGER", str(book))
    agent_id = f"s12k-{adapter}-{uid}"
    tok = token(uid, 1)
    async with harness() as h:
        agent, _ = await wrapped(h, adapter, TOOLS, tmp_path, id=agent_id)
        handle = await agent.start(
            _message(adapter, tok, pay=False, slowly=True), user=f"ada-{uid}", thread=f"t-{uid}"
        )
        one = await Served(_worker(adapter, agent_id), tmp_path, "w1").started()

        async def in_its_slow_step() -> bool:
            return any(x.startswith("note:step1") for x in _book(tok, book))

        assert await eventually(in_its_slow_step, within=180), one.log.read_text()[-3000:]
        await asyncio.sleep(2)
        one.kill()
        two = await Served(_worker(adapter, agent_id), tmp_path, "w2").started()
        try:
            done = await handle.result(wait=300)
        finally:
            two.stop()
        assert done.status is RunStatus.SUCCESS, (
            done.status,
            done.error,
            two.log.read_text()[-3000:],
        )
        ran = _book(tok, book)
        assert ran.count(f"note:step1 {tok}") == 1, ("a write done before the crash ran again", ran)
        assert ran.count(f"note:final {tok}") == 1, ran
        assert (await record(h, done.run_id)).attempt >= 2


@pytest.mark.plan("WF-04")
@pytest.mark.parametrize("adapter", ALL)
async def test_cancel_mid_workflow(
    adapter: str, uid: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    book = tmp_path / "ledger"
    monkeypatch.setenv("SCEN_LEDGER", str(book))
    agent_id = f"s12c-{adapter}-{uid}"
    tok = token(uid, 1)
    async with harness() as h:
        agent, _ = await wrapped(h, adapter, TOOLS, tmp_path, id=agent_id)
        handle = await agent.start(
            _message(adapter, tok, pay=False, slowly=True), user=f"ada-{uid}", thread=f"t-{uid}"
        )
        one = await Served(_worker(adapter, agent_id), tmp_path, "w1").started()
        try:

            async def in_its_slow_step() -> bool:
                return any(x.startswith("note:step1") for x in _book(tok, book))

            assert await eventually(in_its_slow_step, within=180), one.log.read_text()[-3000:]
            cancelled = await agent.cancel(handle.id, reason="stop the workflow")
            assert cancelled.status is RunStatus.CANCELLED

            async def stopped() -> bool:
                return (await record(h, handle.id)).status is RunStatus.CANCELLED

            assert await eventually(stopped, within=60)
            await asyncio.sleep(10)  # past the slow step: nothing after the cancel runs
        finally:
            one.stop()
        ran = _book(tok, book)
        assert f"note:final {tok}" not in ran, ("a step after the cancel ran", ran)
        children = [r async for r in h.runs.iterate(parent_run_id=handle.id)]
        for child in children:  # the sub-agents' runs end with it
            assert (await record(h, child.run_id)).status in (
                RunStatus.CANCELLED,
                RunStatus.SUCCESS,
                RunStatus.ERROR,
            ), child
