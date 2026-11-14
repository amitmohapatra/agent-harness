"""Time travel on ``RunCheckpointer``: it keeps every checkpoint, as LangGraph's own savers do
(``InMemorySaver`` and the Postgres saver prune none), so LangGraph's own ``get_state_history``,
``update_state`` and a run continued from a ``checkpoint_id`` work on a ``ReAct`` graph as on
any saver; in a harness run the history is the run's (its journal), read with the same API, and
a run is never forked from an earlier checkpoint behind governance."""

from __future__ import annotations

import json
from typing import Any

import pytest
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool as lc_tool
from langgraph.checkpoint.memory import InMemorySaver

from tests.support.arm import arm
from tests.support.chat_model import ScriptedChatModel
from tests.support.models import ScriptedChat
from trellis import Harness, ReAct, tool
from trellis.contracts import RunStatus
from trellis.harness.middleware import RunCheckpointer
from trellis.harness.runtime import current

TURNS: list[Any] = [("lookup", {"key": "a"}), ("lookup", {"key": "b"}), "a and b"]


@lc_tool
def lookup(key: str) -> str:
    """Look a key up."""
    return f"value of {key}"


def said(state: Any) -> list[str]:
    return [str(m.content) for m in state.values.get("messages", [])]


async def history(saver: Any) -> tuple[Any, Any, list[Any]]:
    """A two-call run on ``saver``: its graph, its thread's config and its states, newest first."""
    graph = create_agent(ScriptedChatModel(turns=list(TURNS)), tools=[lookup], checkpointer=saver)
    config: Any = {"configurable": {"thread_id": "t"}}
    await graph.ainvoke({"messages": [HumanMessage("look a and b up")]}, config)
    return graph, config, [s async for s in graph.aget_state_history(config)]


async def test_every_checkpoint_is_kept_as_langgraphs_own_saver_keeps_them() -> None:
    _, _, ours = await history(RunCheckpointer())
    _, _, theirs = await history(InMemorySaver())
    assert len(ours) == len(theirs) > 5
    assert [said(s) for s in ours] == [said(s) for s in theirs]
    assert [s.next for s in ours] == [s.next for s in theirs]
    assert [s.metadata["step"] for s in ours] == [s.metadata["step"] for s in theirs]
    ids = [s.config["configurable"]["checkpoint_id"] for s in ours]
    assert ids == sorted(ids, reverse=True)  # newest first
    parents = [s.parent_config["configurable"]["checkpoint_id"] for s in ours[:-1]]
    assert parents == ids[1:] and ours[-1].parent_config is None


async def test_the_history_is_listed_before_a_checkpoint_and_up_to_a_limit() -> None:
    saver = RunCheckpointer()
    _, config, states = await history(saver)
    third = states[2].config
    before = [c async for c in saver.alist(config, before=third)]
    assert len(before) == len(states) - 3
    assert len([c async for c in saver.alist(config, limit=2)]) == 2
    assert [c async for c in saver.alist(config, filter={"source": "nope"})] == []


@pytest.mark.parametrize("saver", [RunCheckpointer, InMemorySaver])
async def test_a_thread_continues_from_an_earlier_checkpoint(saver: Any) -> None:
    graph, _, states = await history(saver())
    # the checkpoint after the first lookup's result: the model is asked again from there
    earlier = next(s for s in states if s.next == ("model",) and len(said(s)) == 3)
    again = create_agent(
        ScriptedChatModel(turns=["only a"]), tools=[lookup], checkpointer=graph.checkpointer
    )
    result = await again.ainvoke(None, earlier.config)
    assert [str(m.content) for m in result["messages"]][-2:] == ["value of a", "only a"]
    latest = await again.aget_state({"configurable": {"thread_id": "t"}})
    assert said(latest)[-1] == "only a"  # the fork is the thread's latest
    kept = [s async for s in again.aget_state_history({"configurable": {"thread_id": "t"}})]
    assert len(kept) > len(states)  # and the checkpoints it left behind are still there


@pytest.mark.parametrize("saver", [RunCheckpointer, InMemorySaver])
async def test_update_state_forks_from_an_earlier_checkpoint(saver: Any) -> None:
    graph, config, states = await history(saver())
    earlier = next(s for s in states if s.next == ("model",) and len(said(s)) == 3)
    forked = await graph.aupdate_state(
        earlier.config, {"messages": [AIMessage("stopped")]}, as_node="model"
    )
    state = await graph.aget_state(forked)
    assert said(state)[-2:] == ["value of a", "stopped"]
    assert said(await graph.aget_state(config))[-1] == "stopped"


async def test_a_deep_agents_delta_channels_come_back_whole_at_an_earlier_checkpoint() -> None:
    from deepagents import create_deep_agent

    turns: list[Any] = [
        ("write_file", {"file_path": f"/n{n}.txt", "content": f"note {n}"}) for n in range(60)
    ]
    graph = create_deep_agent(
        ScriptedChatModel(turns=[*turns, "done"]), checkpointer=RunCheckpointer()
    )
    config: Any = {"configurable": {"thread_id": "t"}, "recursion_limit": 1000}
    await graph.ainvoke({"messages": [HumanMessage("go")]}, config)
    states = [s async for s in graph.aget_state_history(config)]
    # past a snapshot (every 50 updates) and before one: each rebuilt from what was kept
    for files in (55, 20):
        state = next(s for s in states if len(s.values.get("files") or {}) == files)
        assert sorted(state.values["files"]) == sorted(f"/n{n}.txt" for n in range(files))
        assert len(state.values["messages"]) in (2 * files + 1, 2 * files + 2)


async def test_a_record_an_earlier_release_kept_is_continued_and_its_history_starts_there() -> None:
    from deepagents import create_deep_agent

    saver = RunCheckpointer()
    model = ScriptedChatModel(
        turns=[("write_file", {"file_path": "/a.txt", "content": "a"}), "one", "two"]
    )
    graph = create_deep_agent(model, checkpointer=saver)
    config: Any = {"configurable": {"thread_id": "t"}}
    await graph.ainvoke({"messages": [HumanMessage("go")]}, config)
    del saver._threads["t"][""]["history"]  # the latest checkpoint only, as it was kept before
    done = await graph.ainvoke({"messages": [HumanMessage("more")]}, config)
    assert [str(m.content) for m in done["messages"]][-3:] == ["one", "more", "two"]
    states = [s async for s in graph.aget_state_history(config)]
    assert states[-1].parent_config is not None  # the first kept: what came before it is not
    assert all(sorted(s.values["files"]) == ["/a.txt"] for s in states)
    assert said(states[-1])[-1] == "one" and said(states[0])[-1] == "two"


# --------------------------------------------------------------------------- in a harness run
async def test_a_runs_history_is_in_its_journal_and_read_with_langgraphs_api(
    harness: Harness,
) -> None:
    seen: dict[str, Any] = {}

    @tool(side_effects="read")
    async def audit(note: str) -> str:
        """Note what the run did so far."""
        run = current()
        assert run is not None
        config: Any = {"configurable": {"thread_id": run.run_id}}
        seen["states"] = [s async for s in graph.aget_state_history(config)]
        seen["journal"] = json.dumps(run.replay.journal.graph)
        return "noted"

    model = ScriptedChat([("stock", {"sku": "a"}), ("audit", {"note": "n"}), "done"])
    graph = ReAct("s", model)

    @tool(side_effects="read")
    def stock(sku: str) -> int:
        """Units in stock."""
        return 7

    agent = arm(harness.wrap(graph, id="audited"), [stock, audit])
    result = await agent.run("how many a?", user="u")
    assert result.status is RunStatus.SUCCESS
    states = seen["states"]
    assert len(states) > 5 and states[-1].parent_config is None
    # an earlier state of the run: before the model's first call, only its question
    assert any(said(s) == ["how many a?"] for s in states)
    assert len(json.loads(seen["journal"])[""]["history"]) == len(states) - 1


async def test_a_run_is_not_forked_from_an_earlier_checkpoint(harness: Harness) -> None:
    refused: list[str] = []

    @tool(side_effects="read")
    async def rewind(note: str) -> str:
        """Try to rewind the run."""
        run = current()
        assert run is not None
        config: Any = {"configurable": {"thread_id": run.run_id}}
        states = [s async for s in graph.aget_state_history(config)]
        try:
            await graph.aupdate_state(states[-2].config, {"messages": [HumanMessage("again")]})
        except Exception as exc:
            refused.append(str(exc))
        return "tried"

    model = ScriptedChat([("rewind", {"note": "n"}), "done"])
    graph = ReAct("s", model)
    agent = arm(harness.wrap(graph, id="rewound"), [rewind])
    result = await agent.run("go", user="u")
    assert result.answer == "done"
    assert len(refused) == 1 and "earlier checkpoint" in refused[0]
