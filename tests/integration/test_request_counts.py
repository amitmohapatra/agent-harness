"""What a turn costs, counted — requests to the run store and bytes in the journal, never time
(the performance review's PERF-6 and PERF-8): neither grows with the conversation's length or
with the size of what a framework's own tools returned."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from typing import Any
from unittest.mock import Mock

import pytest

from trellis import Harness, Runtime, Settings
from trellis.contracts import RunStatus, ToolOutcome, ToolSpec
from trellis.harness import pipeline
from trellis.harness.identity import Identity
from trellis.harness.runs import LocalRuns
from trellis.harness.tools import bridge

TURNS = 500


async def echo(input: Any, agent: Runtime) -> str:
    return str(input)


async def test_a_500_turn_conversation_is_checked_for_a_turn_served_elsewhere_in_2_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PERF-6: at a turn's start, whether another server served a turn of the conversation
    since this one's last: the read of that last turn and one page of the listing — not one
    read for each of the conversation's turns (it was up to 50). A turn another server did
    serve since is still found."""
    store = LocalRuns()
    async with Harness(config=Settings(), runs=store) as h:
        agent = h.wrap(echo, id="long")
        turns = [await agent.run(f"m{n}", user="ada", thread="t1") for n in range(TURNS)]
        mine = turns[-1].run_id
        now = Identity(tenant="default", user="ada", agent_id="long", run_id="run_now", thread="t1")
        read, listed = Mock(wraps=store.get), Mock(wraps=store.iterate)
        monkeypatch.setattr(store, "get", read)
        monkeypatch.setattr(store, "iterate", listed)
        assert not await pipeline._served_elsewhere(agent, now, mine, datetime.now(UTC))
        assert read.call_count + listed.call_count == 2
        monkeypatch.undo()
        elsewhere = await agent.run("on another server", user="ada", thread="t1")
        assert elsewhere.status is RunStatus.SUCCESS
        assert await pipeline._served_elsewhere(agent, now, mine, datetime.now(UTC))


async def test_a_page_of_another_users_turns_on_a_thread_named_alike_is_not_served_elsewhere() -> (
    None
):
    """The listing is the thread name's, every user's: a whole page of another user's turns
    on a thread named alike, since this one's last, is read and passed over — none is a turn
    of this conversation, so none was served elsewhere — though the page ends before the last
    turn served here is reached."""
    from trellis.harness.runs import PAGE_LIMIT

    store = LocalRuns()
    async with Harness(config=Settings(), runs=store) as h:
        agent = h.wrap(echo, id="long")
        mine = (await agent.run("mine", user="ada", thread="t1")).run_id
        for n in range(PAGE_LIMIT):
            await agent.run(f"bob {n}", user="bob", thread="t1")
        now = Identity(tenant="default", user="ada", agent_id="long", run_id="run_now", thread="t1")
        assert not await pipeline._served_elsewhere(agent, now, mine, datetime.now(UTC))


async def test_200_observed_calls_of_10_kb_keep_the_journal_under_half_a_megabyte() -> None:
    """PERF-8: a call its framework ran itself (Claude Code's ``Read``) is journaled so that
    made again it is not run twice — as the stream previews its result (2,000 characters),
    not whole: its transcript has the rest. 200 calls of 10 KB kept 2 MB; they keep under
    0.5 MB, and each still says it ran."""
    kept: list[dict[str, list[Any]]] = []

    async def reads(input: Any, agent: Runtime) -> str:
        spec = ToolSpec(name="Read", description="Read a file.", side_effects="read")
        for n in range(200):
            outcome = ToolOutcome(tool="Read", output=f"{n}:" + "x" * 10_000)
            await bridge.observed(
                agent, spec, {"file": n}, outcome, ref=f"c{n}", started=time.time()
            )
        kept.append(agent.replay.journal.calls)
        return "read"

    async with Harness(config=Settings()) as h:
        done = await h.wrap(reads, id="reader").run("go", user="ada")
        assert done.status is RunStatus.SUCCESS, done
    [calls] = kept
    assert len(calls) == 200
    assert len(json.dumps(calls)) < 500_000
    [first] = calls[next(iter(calls))]
    again = bridge.journal_outcome("Read", first)
    assert again.ok and again.output.startswith("0:xxx") and len(again.output) == 2001
