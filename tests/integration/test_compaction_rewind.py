"""A ``ReAct`` run's history: the turns a summary replaces saved to memory first
(``compaction_memory``), the model's own ``compact_conversation`` on a long task, and rewind —
the graph's last checkpoints kept in the run's journal, and ``agent.fork`` starting a new run
from one of them, the calls already made replayed, never made again."""

from __future__ import annotations

import logging
from typing import Any

import pytest

from tests.integration.test_react_context import Summarizing
from tests.support.memory import FakeMemoryService
from tests.support.models import ScriptedChat
from trellis import Harness, ReAct, Settings, tool
from trellis.contracts import ConfigurationError, RunStatus
from trellis.harness import automatic
from trellis.harness.journal import OUTCOME, Journal
from trellis.harness.middleware import CHECKPOINT_HISTORY, RunCheckpointer


@tool(side_effects="read")
def note(text: str) -> str:
    """Take a note."""
    noted.append(text[:1])
    return "noted"


noted: list[str] = []


@tool(side_effects="irreversible")
def publish(name: str) -> str:
    """Publish a report."""
    published.append(name)
    return f"published {name}"


published: list[str] = []


def notes(n: int = 5) -> list[Any]:
    return [("note", {"text": f"{k} " + "n" * 2400}) for k in range(n)]


def compacted(memory: FakeMemoryService) -> list[dict[str, Any]]:
    return [
        m
        for call in memory.named("messages")
        for m in call.body["messages"]
        if ":compacted:" in str(m.get("source_message_id"))
    ]


@pytest.fixture(autouse=True)
def _fresh() -> None:
    automatic._unsaved.cache_clear()
    published.clear()
    noted.clear()


# --------------------------------------------------------------------------- compaction
async def test_the_turns_a_summary_replaces_are_saved_to_memory_first(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    script = Summarizing([*notes(), "done"])
    agent = memory_harness.wrap(
        ReAct("s", ScriptedChat(script=script), context_window=4000), id="n", tools=[note]
    )
    result = await agent.run("take five notes", user="u")
    await memory_harness.writes.drain()
    assert result.answer == "done" and script.summaries >= 1
    saved = compacted(memory_service)
    assert saved and {m["role"] for m in saved} <= {"ASSISTANT", "TOOL"}
    assert any('note({"text": "0 ' in m["content"] for m in saved)
    assert not any(m["content"] == "take five notes" for m in saved)  # the transcript's


async def test_without_compaction_memory_it_summarizes_without_saving(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    script = Summarizing([*notes(), "done"])
    agent = memory_harness.wrap(
        ReAct("s", ScriptedChat(script=script), context_window=4000), id="n", tools=[note]
    )
    result = await agent.run("take notes", user="u", without={"compaction_memory"})
    await memory_harness.writes.drain()
    assert result.answer == "done" and script.summaries >= 1 and compacted(memory_service) == []


async def test_with_memory_off_it_summarizes_and_says_so_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    script = Summarizing([*notes(), "done", *notes(), "again"])
    async with Harness(config=Settings()) as h:
        agent = h.wrap(
            ReAct("s", ScriptedChat(script=script), context_window=4000), id="n", tools=[note]
        )
        with caplog.at_level(logging.WARNING, logger="trellis.run"):
            assert (await agent.run("take notes", user="u")).answer == "done"
            assert (await agent.run("take notes", user="u")).answer == "again"
    said = [r for r in caplog.records if "compaction is not saved to memory" in r.message]
    assert len(said) == 1 and script.summaries >= 2


async def test_nothing_but_the_question_and_the_context_is_not_saved() -> None:
    from types import SimpleNamespace

    from langchain_core.messages import HumanMessage, SystemMessage

    runtime: Any = SimpleNamespace(task="q", run_memory=object())  # no write is made
    await automatic._saved(runtime, [HumanMessage("q"), SystemMessage("remembered")])


async def test_a_failed_memory_write_does_not_stop_the_compaction(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    memory_service.fail.add("messages")
    script = Summarizing([*notes(), "done"])
    agent = memory_harness.wrap(
        ReAct("s", ScriptedChat(script=script), context_window=4000), id="n", tools=[note]
    )
    result = await agent.run("take notes", user="u")
    await memory_harness.writes.drain()
    assert result.answer == "done" and script.summaries >= 1  # the write is the spool's


async def test_compaction_twice_and_across_a_pause_saves_each_once(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    script = Summarizing([*notes(4), ("publish", {"name": "a"}), *notes(4), "done"])
    agent = memory_harness.wrap(
        ReAct("s", ScriptedChat(script=script), context_window=4000),
        id="n",
        tools=[note, publish],
    )
    paused = await agent.run("take notes, publish, take more", user="u")
    assert paused.interrupt is not None, paused.error
    done = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="r")
    await memory_harness.writes.drain()
    assert done.answer == "done" and script.summaries >= 2
    saved = compacted(memory_service)
    # a turn is one message in memory, however many summaries it is in
    turns = {(m["role"], m["content"]) for m in saved}
    assert len({m["source_message_id"] for m in saved}) == len(turns)


async def test_the_model_is_offered_compact_conversation_on_a_long_task_only() -> None:
    from tests.support.chat_model import ScriptedChatModel

    model = ScriptedChatModel(turns=["short", "long", "long, without files"])
    async with Harness(config=Settings()) as h:
        agent = h.wrap(ReAct("s", model, context_window=64_000), id="c", tools=[note])
        await agent.run("hi", user="u")
        await agent.run("Do this step by step: note a.", user="u")
        await agent.run("Do this step by step: note a.", user="u", without={"files"})
    offered = [automatic.COMPACT in r.get("tools", []) for r in model.requests]
    assert offered == [False, True, False]


# --------------------------------------------------------------------------- rewind
async def test_a_fork_starts_where_the_run_was_and_replays_what_it_did(
    harness: Harness,
) -> None:
    model = ScriptedChat([("note", {"text": "a"}), ("publish", {"name": "x"})])
    agent = harness.wrap(ReAct("s", model), id="f", tools=[note, publish])
    paused = await agent.run("note, then publish", user="u")
    assert paused.interrupt is not None
    kept = await agent.checkpoints(paused.run_id)
    assert 2 <= len(kept) <= CHECKPOINT_HISTORY
    # from the start again, the model choosing otherwise: the note is the journal's
    model.turns = [("note", {"text": "a"}), ("publish", {"name": "y"}), "forked"]
    forked = await agent.fork(paused.run_id, kept[-1])
    assert forked.status is RunStatus.PAUSED and forked.run_id != paused.run_id
    assert forked.interrupt is not None and forked.interrupt.tool_call is not None
    assert forked.interrupt.tool_call.args == {"name": "y"}
    assert noted == ["a"]  # replayed, not made again
    # a fork of the fork, while it waits
    model.turns = [("note", {"text": "a"}), "a fork of a fork", "forked"]
    twice = await agent.fork(forked.run_id, (await agent.checkpoints(forked.run_id))[-1])
    assert twice.answer == "a fork of a fork" and noted == ["a"]
    done = await agent.resume(forked.interrupt.interrupt_id, "approve", reviewer="r")
    assert done.answer == "forked" and published == ["y"]


async def test_a_fork_with_another_model() -> None:
    async with Harness(config=Settings()) as h:
        first = ScriptedChat([("note", {"text": "a"}), ("publish", {"name": "x"})])
        agent = h.wrap(ReAct("s", first), id="f", tools=[note, publish])
        paused = await agent.run("note, then publish", user="u")
        assert paused.interrupt is not None
        kept = await agent.checkpoints(paused.run_id)
        async with Harness(config=Settings(), runs=h.runs) as other_h:
            other = other_h.wrap(ReAct("s", ScriptedChat(["by another model"])), id="f")
            assert (await other.fork(paused.run_id, kept[-1])).answer == "by another model"


async def test_a_fork_is_refused_with_its_reason(harness: Harness) -> None:
    model = ScriptedChat([("publish", {"name": "x"})])
    agent = harness.wrap(ReAct("s", model), id="f", tools=[publish])
    paused = await agent.run("publish", user="u")
    assert paused.interrupt is not None
    with pytest.raises(ConfigurationError, match="keeps no checkpoint nope"):
        await agent.fork(paused.run_id, "nope")
    with pytest.raises(ConfigurationError, match="no run run_missing"):
        await agent.fork("run_missing", "c")
    with pytest.raises(ConfigurationError, match="no run run_missing"):
        await agent.checkpoints("run_missing")


def test_a_run_holding_a_change_of_unknown_effect_is_not_forked() -> None:
    started = Journal(started={"k": 1})
    with pytest.raises(ConfigurationError, match="whose effect is unknown"):
        started.forked("run_1", "c")
    timed_out = Journal(calls={"k": [{OUTCOME: {"status": "timeout"}}]}, started={"k": 1})
    with pytest.raises(ConfigurationError, match="whose effect is unknown"):
        timed_out.forked("run_1", "c")


def test_the_checkpointer_keeps_the_last_ones_only() -> None:
    from langgraph.checkpoint.base import empty_checkpoint

    saver = RunCheckpointer()
    config: Any = {"configurable": {"thread_id": "t", "checkpoint_ns": ""}}
    ids = []
    for n in range(CHECKPOINT_HISTORY + 5):
        checkpoint = empty_checkpoint()
        checkpoint["id"] = f"{n:04d}"
        ids.append(checkpoint["id"])
        config = saver.put(config, checkpoint, {"step": n}, {})  # type: ignore[arg-type]
    kept = [t.checkpoint["id"] for t in saver.list({"configurable": {"thread_id": "t"}})]
    assert kept == list(reversed(ids))[:CHECKPOINT_HISTORY]  # the oldest pruned
    old: Any = {"configurable": {"thread_id": "t", "checkpoint_ns": "", "checkpoint_id": "0010"}}
    found = saver.get_tuple(old)
    assert found is not None and found.metadata.get("step") == 10
    gone: Any = {"configurable": {"thread_id": "t", "checkpoint_ns": "", "checkpoint_id": "0000"}}
    assert saver.get_tuple(gone) is None
