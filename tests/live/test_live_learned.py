"""What memory learned, back in an agent's hands, against the running memory service:

* **past conversations** - a run in a new thread finds what the user said in an earlier one
  through the ``memory_search`` tool (``threads="all"``), and never what another user said;
* **learned skills** - two successful runs make a procedure, the tenant's administrator
  publishes its draft as an Agent Skill, and a later run loads that skill by name like any
  other: from the folder the memory service publishes to (``SKILLS_DIR``, given to this suite
  as ``TRELLIS_LIVE_SKILLS_DIR``) or from the gateway's skills repository.

Every step is the harness's own path (tools through the bridge, skills pinned at run start);
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
from trellis.harness.skills import LOAD_SKILL
from trellis.memory import MemoryClient

pytestmark = [pytest.mark.live, needs_memory]

#: The folder the memory service publishes learned skills to, when it does (``SKILLS_DIR``).
SKILLS_DIR = os.environ.get("TRELLIS_LIVE_SKILLS_DIR")


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
        found["all"] = await agent.tools.call(
            "memory_search", query=query, kinds=["message"], threads="all"
        )
        found["here"] = await agent.tools.call("memory_search", query=query, kinds=["message"])
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
    assert not any(mine in text for text, _ in _texts(found["here"])), "this thread only"


# the learning job is waited for up to 120 s after the runs it learns from
@pytest.mark.timeout(300)
async def test_a_learned_procedure_published_as_a_skill_is_loaded_by_a_later_run() -> None:
    admin_key = os.environ.get("TRELLIS_ADMIN_KEY")
    if not admin_key:
        pytest.skip("TRELLIS_ADMIN_KEY (the tenant administrator's memory key) is not set")
    suffix = uuid.uuid4().hex[:8]
    find_name, refund_name = f"find_order_{suffix}", f"refund_{suffix}"
    skill = f"live-learned-{suffix}"

    @tool(name=find_name, side_effects="read")
    def find_order(order: str) -> str:
        """The order's payment id."""
        return f"pay-{order}"

    @tool(name=refund_name, side_effects="write")
    def refund(payment: str) -> str:
        """Refund a payment."""
        return f"refunded {payment}"

    async def refunds(input: str, agent: Runtime) -> str:
        payment = await agent.tools.call(find_name, order="O-1")
        return await agent.tools.call(refund_name, payment=payment)

    async with live_harness() as h:
        agent = h.wrap(refunds, id=f"live-refunds-{suffix}", tools=[find_order, refund])
        for n in range(2):
            done = await agent.run(
                "Refund order O-1", user=f"live-u-{suffix}", thread=f"live-r-{suffix}-{n}"
            )
            assert done.status is RunStatus.SUCCESS and done.answer == "refunded pay-O-1"
        await h.writes.drain()
        tenant = await h.tenant()
        url = os.environ["MEMORY_URL"]

    async with MemoryClient(url, api_key=admin_key) as client:
        admin = client.bind(tenant_id=tenant)

        async def drafted() -> bool:
            drafts = await admin.advanced.tools.skill_drafts()
            return any(find_name in d.body and refund_name in d.body for d in drafts)

        assert await eventually(drafted, within=120, every=3)
        [draft] = [d for d in await admin.advanced.tools.skill_drafts() if find_name in d.body]
        assert draft.state == "new" and draft.support == 2
        assert draft.body.index(find_name) < draft.body.index(refund_name), "steps in order"
        decision = await admin.advanced.tools.publish_skill(draft.id, name=skill)
        assert (decision.state, decision.name, decision.version) == ("published", skill, "1.0.0")
        assert all(d.id != draft.id for d in await admin.advanced.tools.skill_drafts())

    folder = None
    if decision.destination == "skills_dir":
        if not SKILLS_DIR:
            pytest.skip("the memory service publishes to a folder: set TRELLIS_LIVE_SKILLS_DIR")
        folder = SKILLS_DIR
    else:
        assert decision.destination == "bifrost"
    loaded: dict[str, str] = {}

    async def follows(input: str, agent: Runtime) -> str:
        loaded["context"] = agent.context or ""
        loaded["skill"] = await agent.tools.call(LOAD_SKILL, name=skill)
        return "followed"

    async with live_harness(skills_dir=folder) if folder else live_harness() as h:
        run = await h.wrap(follows, id=f"live-follows-{suffix}", skills=[skill]).run(
            "Refund order O-2", user=f"live-u-{suffix}"
        )
    assert run.status is RunStatus.SUCCESS, run
    assert f"- {skill}: " in loaded["context"]
    assert loaded["skill"].startswith(f"# {skill} (version 1.0.0)")
    assert f"`{find_name}`" in loaded["skill"] and f"`{refund_name}`" in loaded["skill"]
