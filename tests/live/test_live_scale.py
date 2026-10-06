"""W3 against agent-runs: a worker's run's events in agent-runs' event log, read by another
harness (another replica) through ``agent.events`` and an AG-UI reconnect; a queued run's
priority and conversation key kept by agent-runs, and a conversation's second run claimed only
after its first; a schedule carrying the agent's limit and version."""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from tests.live.conftest import live_harness, needs_memory, needs_runs
from trellis import Runtime, tool
from trellis.contracts import RunEventType, RunOutcome, RunStatus
from trellis.harness.agui.sse import decode
from trellis.runs import RunsClient

pytestmark = [pytest.mark.live, needs_runs, needs_memory]


@tool(side_effects="write")
def charge(order: str) -> str:
    """Charge an order."""
    return f"charged {order}"


async def billing(input: str, agent: Runtime) -> Any:
    receipt = await agent.tools.call("charge", order=input)
    size = await agent.ask("Which size?", options=["S", "L"])
    return f"{receipt}; {size}"


@pytest.mark.timeout(120)
async def test_a_workers_run_is_streamed_by_another_replica_from_agent_runs() -> None:
    agent_id = f"live-scale-{uuid.uuid4().hex[:8]}"
    async with live_harness() as worker_side, live_harness() as reader_side:
        assert isinstance(worker_side.runs, RunsClient)
        working = worker_side.wrap(billing, id=agent_id, tools=[charge])
        reading = reader_side.wrap(billing, id=agent_id, tools=[charge])  # another replica
        handle = await working.start("o-1", user="live-user", thread=f"chat-{agent_id}", priority=7)
        record = await handle.status()
        assert (record.priority, record.concurrency_key) == (7, f"thread:chat-{agent_id}")
        worker = worker_side.worker([working], concurrency=1)
        assert await worker.run_once()
        paused = await handle.result(timeout=30)
        assert paused.interrupt is not None
        follower = asyncio.create_task(_collect(reading.events(handle.run_id)))
        await working.resume(paused.interrupt.interrupt_id, "answer", answer="L", reviewer="lee")
        assert await worker.run_once()
        events = await asyncio.wait_for(follower, 30)
        finished = [e.outcome for e in events if e.type is RunEventType.RUN_FINISHED]
        assert finished == [RunOutcome.INTERRUPT, RunOutcome.SUCCESS]
        assert {e.attempt for e in events} == {1, 2}
        assert any(e.type is RunEventType.TOOL_CALL_RESULT for e in events)
        decided = [e.data for e in events if e.data.get("name") == "decision"]
        assert decided and decided[0]["reviewer"] == "lee"
        # an AG-UI reconnect on the replica that served nothing reads the same log
        app = FastAPI()
        reading.serve_chat(app, identity=lambda r: "live-user")
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://replica") as http:
            response = await http.get(f"/agui/runs/{handle.run_id}/events")
        replayed = decode(response.text)
        assert replayed[0][0] == 1 and replayed[-1][1]["type"] == "RUN_FINISHED"
        assert replayed[-1][1]["outcome"]["type"] == "success"


async def _collect(events: Any) -> list[Any]:
    return [e async for e in events]


@pytest.mark.timeout(120)
async def test_a_conversations_second_run_is_claimed_after_its_first() -> None:
    agent_id = f"live-turns-{uuid.uuid4().hex[:8]}"
    async with live_harness() as h:

        async def note(input: str, agent: Runtime) -> str:
            return input

        agent = h.wrap(note, id=agent_id)
        thread = f"chat-{agent_id}"
        first = await agent.start("first", user="live-user", thread=thread)
        second = await agent.start("second", user="live-user", thread=thread)
        held = await h.runs.claim("live-w1", [agent_id], lease_seconds=30)
        assert held is not None and held.run.run_id == first.run_id
        assert await h.runs.claim("live-w2", [agent_id], lease_seconds=30) is None  # it waits
        await h.runs.finish(
            first.run_id, RunStatus.SUCCESS, worker_id="live-w1", tenant=held.run.tenant_id
        )
        after = await h.runs.claim("live-w2", [agent_id], lease_seconds=30)
        assert after is not None and after.run.run_id == second.run_id
        await h.runs.finish(
            second.run_id, RunStatus.SUCCESS, worker_id="live-w2", tenant=after.run.tenant_id
        )


@pytest.mark.timeout(60)
async def test_a_schedule_carries_the_agents_limit_and_version() -> None:
    async with live_harness() as h:

        async def brief(input: str, agent: Runtime) -> str:
            return input

        agent = h.wrap(brief, id=f"live-brief-{uuid.uuid4().hex[:8]}", timeout=90, version="w5w3")
        schedule = await agent.schedule("manual", "brief", on_behalf_of="live-user")
        assert isinstance(h.runs, RunsClient)
        try:
            assert (schedule.timeout_seconds, schedule.agent_version) == (90, "w5w3")
            fired = await h.runs.schedules.fire(schedule.schedule_id)
            run = await h.runs.get(fired.run_id)
            assert run is not None and (run.timeout_seconds, run.agent_version) == (90, "w5w3")
            await h.runs.cancel(fired.run_id)
        finally:
            await h.runs.schedules.delete(schedule.schedule_id)
