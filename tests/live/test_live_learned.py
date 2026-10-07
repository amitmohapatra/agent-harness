"""What memory learned, back in an agent's hands, against the running memory service:

* **past conversations** - a run in a new thread finds what the user said in an earlier one
  through the ``memory_search`` tool (``kinds=["message"]``), and never what another user said;
* **learned skills** - two successful runs of an agent make its learned skill, offered in the
  context of its next run with nobody publishing anything: to the user whose runs taught it,
  to the agent's other users once a second user's run agreed, never to another agent, and no
  longer once the tenant's administrator dismisses it.

Every step is the harness's own path (tools through the bridge, the context pushed at run start);
nothing depends on what a model chooses."""

from __future__ import annotations

import os
import uuid
from typing import Any

import pytest

from tests.live.conftest import live_harness, needs_memory
from tests.live.support import eventually
from trellis import Runtime, tool
from trellis.contracts import RunStatus
from trellis.memory import MemoryClient

pytestmark = [pytest.mark.live, needs_memory]


def _texts(found: Any) -> list[tuple[str, str | None]]:
    items = found if isinstance(found, list) else []
    return [(str(i.get("text")), i.get("thread_id")) for i in items if isinstance(i, dict)]


async def test_a_run_finds_what_its_user_said_in_an_earlier_conversation() -> None:
    suffix = uuid.uuid4().hex[:8]
    user, stranger = f"live-ann-{suffix}", f"live-bob-{suffix}"
    old_thread = f"live-old-{suffix}"
    mine, theirs = f"locker {suffix} code is 3141", f"locker {suffix} code is 2718"

    async def noted(input: str, agent: Runtime) -> str:
        return "Noted."

    found: dict[str, Any] = {}

    async def recall(input: str, agent: Runtime) -> str:
        query = f"locker {suffix} code"
        found["all"] = await agent.tools.call("memory_search", query=query, kinds=["message"])
        return "found"

    async with live_harness() as h:
        teller = h.wrap(noted, id=f"live-teller-{suffix}")
        for who, said, thread in ((user, mine, old_thread), (stranger, theirs, f"t-{suffix}")):
            told = await teller.run(f"My {said}.", user=who, thread=thread)
            assert told.status is RunStatus.SUCCESS
        await h.writes.drain()
        assert h.writes.failed == 0

        asker = h.wrap(recall, id=f"live-asker-{suffix}")
        asked = await asker.run("What is my locker code?", user=user, thread=f"live-new-{suffix}")
        assert asked.status is RunStatus.SUCCESS, asked

    every = _texts(found["all"])
    assert (f"USER: My {mine}.", old_thread) in every, every
    assert not any(theirs in text for text, _ in every), "another user's chat is never read"


# the learning job is waited for after the runs it learns from
@pytest.mark.timeout(300)
async def test_an_agent_learns_a_skill_from_its_runs_and_its_next_runs_are_offered_it() -> None:
    admin_key = os.environ.get("TRELLIS_ADMIN_KEY")
    if not admin_key:
        pytest.skip("TRELLIS_ADMIN_KEY (the tenant administrator's memory key) is not set")
    suffix = uuid.uuid4().hex[:8]
    find_name, refund_name = f"find_order_{suffix}", f"refund_{suffix}"
    agent_id = f"live-refunds-{suffix}"
    ann, bob, cy = (f"live-{who}-{suffix}" for who in ("ann", "bob", "cy"))
    learned = f"{find_name} -> {refund_name}"

    @tool(name=find_name, side_effects="read")
    def find_order(order: str) -> str:
        """The order's payment id."""
        return f"pay-{order}"

    @tool(name=refund_name, side_effects="write")
    def refund(payment: str) -> str:
        """Refund a payment."""
        return f"refunded {payment}"

    seen: dict[str, str] = {}

    async def refunds(input: str, agent: Runtime) -> str:
        seen[str(agent.user)] = agent.context or ""
        payment = await agent.tools.call(find_name, order="O-1")
        return await agent.tools.call(refund_name, payment=payment)

    agents: dict[str, Any] = {}

    async def run(h: Any, user: str, n: int, agent: str = agent_id) -> None:
        if agent not in agents:
            agents[agent] = h.wrap(refunds, id=agent, tools=[find_order, refund])
        done = await agents[agent].run("Refund order O-1", user=user, thread=f"live-r-{suffix}-{n}")
        assert done.status is RunStatus.SUCCESS and done.answer == "refunded pay-O-1", done
        await h.writes.drain()
        assert h.writes.failed == 0

    async with live_harness() as h:
        tenant = await h.tenant()
        async with MemoryClient(os.environ["MEMORY_URL"], api_key=admin_key) as client:
            admin = client.bind(tenant_id=tenant)

            async def skills() -> list[Any]:
                return await admin.advanced.skills.list(agent=agent_id)

            async def users(n: int) -> bool:
                return any(s.status == "active" and s.users >= n for s in await skills())

            await run(h, ann, 0)
            await run(h, ann, 1)
            assert await eventually(lambda: users(1), within=120, every=3)
            [skill] = await skills()
            assert skill.steps == [find_name, refund_name] and skill.runs == 2

            await run(h, ann, 2)  # Ann's next run: offered what her runs taught
            assert "## Learned skills for this task" in seen[ann] and learned in seen[ann]
            await run(h, bob, 3)  # Bob is not offered Ann's wording ...
            assert learned not in seen[bob]
            assert await eventually(lambda: users(2), within=120, every=3)
            await run(h, cy, 4)  # ... and once Bob's run agreed, every user of the agent is
            assert learned in seen[cy]

            await run(h, cy, 5, agent=f"live-other-{suffix}")
            assert learned not in seen[cy], "another agent never learns this one's skills"

            dismissed = await admin.advanced.skills.dismiss(skill.id)
            assert dismissed.status == "dismissed"
            await run(h, cy, 6)
            assert learned not in seen[cy], "a dismissed skill is offered no more"
