"""What the memory service offers, used on every run, against a running memory service
(``MEMORY_URL``, ``TRELLIS_API_KEY``): a sub-agent reads its parent's hand-off notes and writes
to its delegation tree's group; a sampled run is verified, and the judge's verdict labels it; a
person's verdict cites what the answer rested on; a run's context draws on the
documents it names. No model: function agents, so each assertion is about the service."""

from __future__ import annotations

import uuid
from typing import Any

import pytest

from tests.live.conftest import live_harness, needs_memory
from trellis import Harness, Runtime
from trellis.contracts import RunStatus

pytestmark = [pytest.mark.live, needs_memory]


async def test_a_sub_agent_reads_its_parents_notes_and_writes_to_its_group(
    harness: Harness,
) -> None:
    marker = uuid.uuid4().hex[:8]
    note = f"The hand-off code for this task is {marker}."
    found: dict[str, Any] = {}

    async def child(input: str, agent: Runtime) -> str:
        hits = await agent.tools.call("memory_search", query=f"hand-off code {marker}")
        found["child"] = hits
        stored = await agent.tools.call(
            "memory_remember", content=f"The group finding is {marker}.", scope="group"
        )
        found["group"] = stored
        return "done"

    async def parent(input: str, agent: Runtime) -> Any:
        await agent.tools.call("memory_remember", content=note, scope="run")
        return await agent.tools.call("helper", message="use the hand-off code")

    user = f"live-{marker}"
    helper = harness.wrap(child, id=f"helper-{marker}")
    lead = harness.wrap(parent, id=f"lead-{marker}", tools=[helper.as_tool()])
    result = await lead.run("start the task", user=user)
    assert result.status is RunStatus.SUCCESS, result.error
    assert marker in str(found["child"]), "the child reads the note its parent kept for its run"
    assert "id" in str(found["group"]), "a group note binds to the tree's group"

    # another run of the same agent for the same user is not the parent: it cannot read it
    async def outsider(input: str, agent: Runtime) -> Any:
        return await agent.tools.call("memory_search", query=f"hand-off code {marker}")

    other = await harness.wrap(outsider, id=f"helper-{marker}").run("look", user=user)
    assert note not in str(other.answer)


async def test_a_sampled_run_is_labelled_by_its_check_and_cited_by_a_person() -> None:
    async with live_harness(eval_sample=1.0) as harness:
        await _labelled_and_cited(harness)


async def _labelled_and_cited(harness: Harness) -> None:
    marker = uuid.uuid4().hex[:8]
    user = f"live-{marker}"

    async def remember(input: str, agent: Runtime) -> Any:
        return await agent.tools.call(
            "memory_remember", content=f"The user's locker code is {marker}.", scope="user"
        )

    async def answer(input: str, agent: Runtime) -> str:
        return f"Your locker code is {marker}."

    await harness.wrap(remember, id=f"setup-{marker}").run("remember it", user=user)
    await harness.writes.drain()
    result = await harness.wrap(answer, id=f"answer-{marker}").run(
        "What is my locker code?", user=user
    )
    await harness.writes.drain()
    assert harness.memory is not None
    scope = harness.memory.client.bind(
        user_id=user, agent_id=f"answer-{marker}", agent_run_id=result.run_id
    )
    page = await scope.feedback.page_for("run", result.run_id)
    verdicts = {f.source: f for f in page.items}
    assert "judge" in verdicts, "a sampled run is checked"
    judge = verdicts["judge"]
    assert judge.projection is not None and judge.projection.action == "run_labelled"
    stored = await harness.feedback(result.run_id, "confirm")
    assert stored is not None
    assert {r.source_id for r in stored.evidence_refs} == {r.source_id for r in judge.evidence_refs}


async def test_a_runs_context_draws_on_the_documents_it_names(harness: Harness) -> None:
    marker = uuid.uuid4().hex[:8]
    user = f"live-{marker}"
    kept = await harness.add_document(
        (
            f"returns-{marker}.txt",
            f"Returns are accepted for 30 days ({marker}).".encode(),
            "text/plain",
        ),
        user=user,
    )
    other = await harness.add_document(
        (f"shipping-{marker}.txt", f"Shipping takes 4 days ({marker}).".encode(), "text/plain"),
        user=user,
    )
    seen: list[str | None] = []

    async def reads(input: str, agent: Runtime) -> str:
        seen.append(agent.context)
        return "ok"

    agent = harness.wrap(reads, id=f"docs-{marker}")
    await agent.run(
        f"How long do returns and shipping take ({marker})?",
        user=user,
        document_ids=[kept.document_id],
    )
    assert seen[0] is not None and "30 days" in seen[0] and "4 days" not in seen[0]
    assert other.document_id != kept.document_id
