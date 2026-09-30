"""What the harness decides on its own, against the real services: who the key says the
deployment is, approval rules from the tool catalog, tool hints narrowing a large toolbox, a
large ``ask`` payload stored as a run artifact and served back, the inbox by default assignee,
and the sampled grounding check."""

from __future__ import annotations

import json
import uuid
from typing import Any

import pytest

from tests.live.conftest import live_harness, needs_memory, needs_runs
from tests.live.support import memory_scope
from trellis import Runtime, tool
from trellis.contracts import RunStatus
from trellis.harness import agent as agent_module
from trellis.harness.clients.memory import read_only

pytestmark = [pytest.mark.live, needs_memory]


async def test_the_key_names_the_tenant_and_its_role() -> None:
    async with live_harness() as h:
        key = await h.key()
        assert key.tenant_id is not None and await h.tenant() == key.tenant_id
        assert await h.writes_memory() is not read_only(key)


async def test_an_approval_rule_in_the_catalog_decides_which_calls_wait() -> None:
    suffix = uuid.uuid4().hex[:8]
    name = f"pay_{suffix}"

    @tool(name=name, side_effects="write")
    def pay(amount: int) -> str:
        """Pay an invoice."""
        return f"paid {amount}"

    async def payer(input: int, agent: Runtime) -> Any:
        return await agent.tools.call(name, amount=input)

    async with live_harness() as h:
        agent = h.wrap(payer, id=f"live-payer-{suffix}", tools=[pay])
        scope = await memory_scope(h, user="live-ada", agent_id=agent.id)
        # an admin's rule: payments over 100 wait for a person
        await scope.advanced.tools.put_catalog(
            [{"name": name, "side_effects": "write", "approve_when": "amount > 100"}]
        )
        assert (await agent.run(50, user="live-ada")).answer == "paid 50"
        paused = await agent.run(500, user="live-ada")
        assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
        assert "amount > 100" in paused.interrupt.question
        await h.writes.drain()
        [entry] = await scope.advanced.tools.catalog(names=[name])
        assert entry.approve_when == "amount > 100"  # the harness's publication kept it


async def test_hints_narrow_a_large_toolbox_to_the_task() -> None:
    suffix = uuid.uuid4().hex[:8]
    names = [f"weather_{suffix}", f"stock_{suffix}", f"invoice_{suffix}"]
    names += [f"shipping_{suffix}", f"payroll_{suffix}", f"calendar_{suffix}"]
    descriptions = [
        "Today's weather in a city.",
        "Units of a SKU in the warehouse.",
        "Create an invoice for a customer.",
        "Track a shipment by its number.",
        "Run the monthly payroll.",
        "Book a meeting room.",
    ]

    def make(n: str, d: str) -> Any:
        def fn(value: str) -> str:
            return f"{n}:{value}"

        return tool(fn, name=n, description=d, side_effects="read")

    async def planner(input: str, agent: Runtime) -> Any:
        return [c.name for c in (await agent.tools.hints(input)).candidates]

    async with live_harness() as h:
        agent = h.wrap(
            planner,
            id=f"live-hints-{suffix}",
            tools=[make(n, d) for n, d in zip(names, descriptions, strict=True)],
        )
        await agent.run("warm up the catalog", user="live-ada")  # publishes the tools
        await h.writes.drain()
        result = await agent.run("How many units of SKU A-1 are in the warehouse?", user="live-ada")
        assert result.status is RunStatus.SUCCESS, result.error
        # the candidates the model would be offered: the warehouse tool first
        assert result.answer and result.answer[0] == f"stock_{suffix}", result.answer


@needs_runs
async def test_a_large_table_is_a_run_artifact_served_back() -> None:
    rows = [{"sku": f"S-{i}", "note": "x" * 200} for i in range(200)]

    async def reviewer(input: str, agent: Runtime) -> Any:
        return await agent.ask("Check the order lines", table=rows)

    async with live_harness() as h:
        agent = h.wrap(reviewer, id=f"live-review-{uuid.uuid4().hex[:8]}")
        paused = await agent.run("review", user="live-ada")
        assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
        ref = paused.interrupt.payload_ref
        assert ref is not None and paused.interrupt.payload is None
        stored = await h.runs.artifact(ref.artifact_id, await h.tenant())
        assert stored is not None and json.loads(stored) == {"table": rows}
        record = await h.runs.get(paused.run_id)
        assert record is not None and len(json.dumps(record.checkpoint)) < 16 * 1024
        # the question is the run's user's by default: it is in their inbox
        inbox = await h.inbox("user:live-ada")
        assert paused.run_id in [r.run_id for r in inbox]
        done = await agent.resume(
            paused.interrupt.interrupt_id, "answer", answer="fine", reviewer="live-ada"
        )
        assert done.answer == "fine"


async def test_a_sampled_answer_is_verified_against_its_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(agent_module, "GROUNDING_SAMPLE", 1.0)
    suffix = uuid.uuid4().hex[:8]
    user = f"live-user-{suffix}"

    async def answer(input: str, agent: Runtime) -> str:
        return f"Your warehouse is in Berlin ({suffix})."

    async with live_harness() as h:
        agent = h.wrap(answer, id=f"live-grounded-{suffix}")
        scope = await memory_scope(h, user=user, agent_id=agent.id)
        await scope.remember(f"The warehouse of {user} is in Berlin.", visibility="USER")
        result = await agent.run("Where is my warehouse?", user=user)
        await h.writes.drain()
        assert h.writes.failed == 0
        judged = await scope.feedback.list_for("run", result.run_id)
        assert any(f.source == "judge" for f in judged), judged
