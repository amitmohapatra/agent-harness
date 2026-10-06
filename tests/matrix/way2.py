"""Way 2: the blocks without a Harness, each used the way the docs show (``docs/blocks``,
``examples/blocks_*.py``): ``governed`` around a tool, the run store and ``trellis.runs.Worker``,
the memory SDK, ``judge``/``grounding``, ``remote()``, the redactor. A block is framework-neutral
— your code calls it whatever the framework — so these run as the ``function`` adapter (the
framework recipes are the ``W2R`` row)."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import httpx
import pytest
from fastapi import FastAPI

from tests.matrix.kit import EMAIL, SECRET, UNKNOWN, Desk
from tests.support.catalog import FakeCatalog
from trellis import Harness, Runtime, Settings
from trellis.contracts import (
    Interrupt,
    InterruptDecision,
    InterruptReason,
    InterruptResolution,
    RunStart,
    RunStatus,
    ScheduleSpec,
    new_id,
)
from trellis.harness.a2a import remote
from trellis.harness.evals import EvalCase, EvalServices, judge
from trellis.harness.evals import grounding as grounding_block
from trellis.harness.governance import Decision, Governance, Rejected, governed
from trellis.harness.governance.catalog import Rule
from trellis.harness.redaction import DEFAULT as REDACTOR
from trellis.harness.tools.base import ToolTimeout
from trellis.runs import Job
from trellis.runs import Worker as RunsWorker
from trellis.runs import worker as claim_loop

if TYPE_CHECKING:
    from tests.matrix.world import World

TENANT = "acme"
AGENT = "blocks"
USER = "ada"


async def _call(tool: Any) -> Any:
    """A tool made by the matrix's ``Desk``, called as plain code (Way 2 has no bridge)."""
    return await tool.fn(**tool.args) if hasattr(tool, "args") else None


def _fn(harness_tool: Any) -> Any:
    """The function behind a ``@tool`` (Way 2 wraps your own function with ``governed``)."""
    return harness_tool.fn


# --------------------------------------------------------------------------- governance
async def tiers(w: World) -> None:
    d, gov = Desk(), Governance()
    announced: list[Decision] = []
    asked: list[Decision] = []

    def approve(decision: Decision) -> bool:
        asked.append(decision)
        return True

    lookup = governed(_fn(d.lookup()), gov, side_effects="read", on_ask=approve)
    note = governed(
        _fn(d.note()), gov, side_effects="write", on_ask=approve, on_announce=announced.append
    )
    refund = governed(_fn(d.refund()), gov, side_effects="irreversible", on_ask=approve)
    assert await lookup(topic="o1") == "facts about o1"
    assert await note(text="x") == "noted x"
    assert await refund(order="o1") == "refunded o1"
    assert [a.tool for a in asked] == ["refund"] and [a.tool for a in announced] == ["note"]


async def approve_when(w: World) -> None:
    d = Desk()
    gov = Governance(FakeCatalog({"note": Rule("write", 'text == "big"')}))
    asked: list[Decision] = []
    note = governed(
        _fn(d.note()), gov, side_effects="write", on_ask=lambda x: asked.append(x) or True
    )
    await note(text="small")
    await note(text="big")
    assert [dict(a.args) for a in asked] == [{"text": "big"}], asked


async def reject(w: World) -> None:
    d = Desk()
    refund = governed(
        _fn(d.refund()), Governance(), side_effects="irreversible", on_ask=lambda _: False
    )
    with pytest.raises(Rejected):
        await refund(order="o1")
    assert d.ran("refund") == []


async def ask(w: World) -> None:
    """A question by hand: pause the run with an Interrupt, the reviewer's resolution resumes
    it (Way 2: your code asks, the run store keeps the pause)."""
    store = w.store
    run_id = new_id("run_")
    await store.start(
        RunStart(run_id=run_id, tenant_id=TENANT, agent_id=AGENT, thread_id=run_id, user_id=USER)
    )
    interrupt = Interrupt(
        interrupt_id=f"{run_id}.1.1",
        tenant_id=TENANT,
        run_id=run_id,
        reason=InterruptReason.CHOICE,
        question="Which size?",
        options=["S", "L"],
    )
    await store.pause(interrupt)
    resolution = InterruptResolution(
        interrupt_id=interrupt.interrupt_id,
        run_id=run_id,
        decision=InterruptDecision.ANSWER,
        answer="L",
        reviewer=USER,
    )
    resumed = await store.resume(resolution, tenant=TENANT)
    assert resumed.last_resolution is not None and resumed.last_resolution.answer == "L"
    done = await store.finish(run_id, RunStatus.SUCCESS, output="size L", tenant=TENANT)
    assert done.status is RunStatus.SUCCESS


async def decided(w: World) -> None:
    catalog = FakeCatalog()
    gov = Governance(catalog, tenant=TENANT, agent_id=AGENT)
    d = Desk()
    decisions: list[Decision] = []
    refund = governed(
        _fn(d.refund()),
        gov,
        side_effects="irreversible",
        on_ask=lambda x: decisions.append(x) or True,
    )
    await refund(order="o1")
    await gov.decided(decisions[0], "approve", reviewer="cfo", run_id="run_1", user=USER)
    assert [f.verdict.value for f in catalog.feedback_sent] == ["approve"]


# --------------------------------------------------------------------------- reliability
async def read_timeout(w: World) -> None:
    slow = governed(
        _fn(Desk().slow()), Governance(), side_effects="read", timeout=0.05, on_ask=bool
    )
    with pytest.raises(ToolTimeout) as raised:
        await slow(sku="A")
    assert not raised.value.unknown


async def unknown_outcome(w: World) -> None:
    d = Desk()
    transfer = governed(
        _fn(d.transfer()), Governance(), side_effects="write", timeout=0.05, on_ask=lambda _: True
    )
    with pytest.raises(ToolTimeout) as raised:
        await transfer(amount=5)
    assert raised.value.unknown and str(raised.value) == UNKNOWN
    assert d.ran("transfer") == [{"amount": 5}]


async def retries(w: World) -> None:
    d = Desk()
    quote = governed(_fn(d.quote()), Governance(), side_effects="read", on_ask=bool)
    assert await quote(sku="A-1") == "A-1 costs 7" and d.quotes == 3


# --------------------------------------------------------------------------- runs
async def _queued(w: World, agent: str = AGENT) -> str:
    run_id = new_id("run_")
    start = RunStart(
        run_id=run_id, tenant_id=TENANT, agent_id=agent, thread_id=run_id, user_id=USER, input="x"
    )
    await w.store.start(start, queue=True)
    return run_id


async def durable(w: World) -> None:
    """``trellis.runs.Worker`` around your own handler; ``elsewhere``: a pause, then another
    worker continues it from the resolution."""
    seen: list[tuple[str, Any]] = []

    async def handle(job: Job) -> None:
        resolution = job.record.last_resolution
        seen.append((job.worker_id, resolution.answer if resolution else None))
        if w.mode == "elsewhere" and resolution is None:
            interrupt = Interrupt(
                interrupt_id=f"{job.record.run_id}.1.1",
                tenant_id=TENANT,
                run_id=job.record.run_id,
                question="Go on?",
            )
            await job.pause(interrupt)
            return
        await job.finish(RunStatus.SUCCESS, output="done")

    run_id = await _queued(w)
    assert await RunsWorker(w.store, handle, [AGENT], worker_id="w1").run_once()
    if w.mode == "elsewhere":
        record = await w.store.get(run_id)
        assert record is not None and record.awaiting is not None
        resolution = InterruptResolution(
            interrupt_id=record.awaiting.interrupt_id,
            run_id=run_id,
            decision=InterruptDecision.ANSWER,
            answer="yes",
            reviewer=USER,
        )
        await w.store.resume(resolution, tenant=TENANT)
        assert await RunsWorker(w.store, handle, [AGENT], worker_id="w2").run_once()
    record = await w.store.get(run_id)
    assert record is not None and record.status is RunStatus.SUCCESS, record
    if w.mode == "elsewhere":
        assert seen == [("w1", None), ("w2", "yes")], seen


async def cancel(w: World) -> None:
    w.monkeypatch.setattr(claim_loop, "_sleep", lambda seconds: asyncio.sleep(0.01))
    started = asyncio.Event()

    async def handle(job: Job) -> None:
        started.set()
        await asyncio.sleep(10)

    run_id = await _queued(w)
    working = asyncio.create_task(RunsWorker(w.store, handle, [AGENT]).run_once())
    await asyncio.wait_for(started.wait(), 5)
    await w.store.cancel(run_id, reason="a duplicate", tenant=TENANT)
    assert await asyncio.wait_for(working, 5)
    record = await w.store.get(run_id)
    assert record is not None and record.status is RunStatus.CANCELLED


async def scheduled(w: World) -> None:
    ran: list[str | None] = []

    async def handle(job: Job) -> None:
        ran.append(job.record.on_behalf_of)
        await job.finish(RunStatus.SUCCESS, output="briefed")

    spec = ScheduleSpec(
        tenant_id=TENANT,
        agent_id=AGENT,
        name="briefing",
        cadence="0 0 1 1 *",
        timezone="UTC",
        on_behalf_of=USER,
        input="inbox",
    )
    schedule = await w.store.schedules.create(spec)
    w.store._schedules[schedule.schedule_id] = schedule.model_copy(
        update={"next_fire_at": datetime.now(UTC) - timedelta(seconds=1)}
    )
    assert await RunsWorker(w.store, handle, [AGENT]).run_once()
    assert ran == [USER]


# --------------------------------------------------------------------------- memory
async def push(w: World) -> None:
    client = w.memory_service.client()
    try:
        scope = client.bind(tenant_id=TENANT, user_id=USER).agent(AGENT, agent_run_id="run_1")
        pushed = await scope.context("what does the user prefer?", window=False)
    finally:
        await client.aclose()
    assert w.memory_service.context_text in pushed.rendered


async def records(w: World) -> None:
    client = w.memory_service.client()
    try:
        scope = client.bind(tenant_id=TENANT, user_id=USER).agent(AGENT, agent_run_id="run_1")
        await scope.record_tool("lookup", {"topic": "a"}, output="facts about a")
    finally:
        await client.aclose()
    assert [c.body["tool"] for c in w.memory_service.named("record_tool")] == ["lookup"]


# --------------------------------------------------------------------------- gateway
async def key_tools(w: World) -> None:
    from tests.matrix.world import KEY_TOOL

    gateway = w.fake_gateway.gateway()
    try:
        names = [t.name for t in await gateway.client.tools()]
    finally:
        await gateway.aclose()
    assert names == [KEY_TOOL], names


async def virtual_mcps(w: World) -> None:
    w.fake_gateway.bundles["billing"] = ["billing-invoice"]
    gateway = w.fake_gateway.gateway()
    try:
        names = [t.name for t in await gateway.client.tools(slug="billing")]
    finally:
        await gateway.aclose()
    assert names == ["billing-invoice"], names


# --------------------------------------------------------------------------- evaluation
async def judges(w: World) -> None:
    seen: list[EvalCase] = []

    async def polite(case: EvalCase) -> Any:
        seen.append(case)
        from trellis.harness.evals import EvalScore

        return EvalScore("polite", 1.0)

    case = EvalCase(input="hi", output="Hello!", run_id="run_1")
    scores, failed = await judge(case, [polite], services=EvalServices())
    assert failed == {} and [s.value for s in scores] == [1.0] and seen == [case]


async def grounding(w: World) -> None:
    client = w.memory_service.client()
    try:
        scope = client.bind(tenant_id=TENANT, user_id=USER).agent(AGENT, agent_run_id="run_1")
        pushed = await scope.context("what does the user prefer?", window=False)
        case = EvalCase(
            input="q", output="Email.", run_id="run_1", bundle_id=pushed.bundle_id, memory=scope
        )
        score = await grounding_block()(case)
    finally:
        await client.aclose()
    assert score is not None and score.value == 0.8, score


async def redaction(w: World) -> None:
    shown = REDACTOR.redact_input({"email": EMAIL, "api_key": SECRET})
    assert EMAIL not in json.dumps(shown) and SECRET not in json.dumps(shown), shown


async def remote_agent(w: World) -> None:
    async def greeter(input: Any, agent: Runtime) -> str:
        """Greets."""
        return f"hello {input}"

    url = "http://a2a.remote/agents/greeter"
    served = Harness(config=Settings())
    app = FastAPI()
    served.wrap(greeter, id="greeter").serve_a2a(app, url)
    http = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://a2a.remote")
    try:
        async with remote(url, tenant="default", user=USER, client=http) as agent:
            assert await agent("world") == "hello world"
    finally:
        await http.aclose()
        await served.aclose()
