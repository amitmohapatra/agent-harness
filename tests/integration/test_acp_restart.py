"""An ACP agent's run whose worker died mid-turn, taken over by another worker sharing the run
store: the next attempt resumes the agent's session (the journal holds it) and, no person
having been asked since, tells it only that the task goes on — to call again a call that has
no result yet, or whose result says the run ended before it finished
(``adapters.claude.RESUMED``): the call the worker died in is made again, once; the write a
person approved before is not made again."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from tests.integration.test_acp import fake, recorded
from tests.support.arm import arm
from trellis import Harness, Settings, tool
from trellis.contracts import RunStatus
from trellis.harness.adapters.claude import RESUMED
from trellis.harness.runs import LocalRuns
from trellis.harness.worker import Job


class Crash(BaseException):
    """The worker process dies: nothing more is written, the lease lapses."""


crashes: list[BaseException] = []
looked: list[str] = []


@tool(side_effects="read")
def look(topic: str) -> str:
    """Look a topic up (the worker dies here, once)."""
    looked.append(topic)
    if crashes:
        raise crashes.pop()
    return f"facts about {topic}"


noted: list[str] = []


@tool(side_effects="write")
def note(text: str) -> str:
    """Note a text down (a write: the journal is kept before it runs)."""
    noted.append(text)
    return "noted"


@pytest.fixture
async def workers() -> AsyncIterator[tuple[Harness, Harness]]:
    store = LocalRuns()
    # Kimi Code runs its shell in a folder of this machine: this deployment allows it
    here = Settings(host_execution="allow")
    pair = (Harness(config=here, runs=store), Harness(config=here, runs=store))
    yield pair
    for h in pair:
        await h.aclose()


async def test_a_turn_taken_over_after_its_worker_died_resumes_the_session_told_it_goes_on(
    workers: tuple[Harness, Harness], tmp_path: Path
) -> None:
    crashes[:], looked[:], noted[:] = [Crash()], [], []
    script = [
        {"tool": "note", "args": {"text": "tea"}},
        {"tool": "look", "args": {"topic": "tea"}},
        {"text": "found: {last}"},
    ]
    first, second = workers
    dying, taking_over = (
        arm(h.wrap(fake(tmp_path, script), id="kimi"), [note, look]) for h in workers
    )
    store = first.runs
    assert isinstance(store, LocalRuns)
    handle = await dying.start("tell me about tea", user="u", thread="t")
    worker = first.worker([dying])
    assert await worker.run_once()  # the write asks a person first: paused
    paused = await store.get(handle.id)
    assert paused is not None and paused.awaiting is not None
    queued = await dying.resume(paused.awaiting.interrupt_id, "approve", reviewer="u")
    assert queued.status is RunStatus.QUEUED
    claimed = await store.claim(worker.worker_id, [dying.id])
    assert claimed is not None
    with pytest.raises(Crash):
        await dying.execute(
            Job(record=claimed.run, worker_id=worker.worker_id, lease_seconds=60, store=store)
        )
    held, _ = store._leases[handle.id]
    store._leases[handle.id] = (held, datetime.now(UTC) - timedelta(seconds=1))
    assert await second.worker([taking_over]).run_once()
    result = await handle.result(wait=10)
    assert (await handle.status()).attempt == 3
    assert result.status is RunStatus.SUCCESS, result.error
    record = recorded(tmp_path)
    assert [s["how"] for s in record["sessions"]][-1] == "resume"  # the journal's session
    assert record["prompts"][-1] == RESUMED  # no person asked: it only goes on
    assert noted == ["tea"]  # the approved write, done once
    # the read the worker died in was cut short (``convert.mcp.STOPPED``, in the session): made
    # again after the takeover, its result the answer's — not the cut-short call's error
    assert looked == ["tea", "tea"], looked
    assert result.answer == "found: facts about tea", result.answer
