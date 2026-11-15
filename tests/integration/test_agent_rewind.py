"""``Agent.checkpoints`` and ``Agent.rewind``: a paused ``ReAct`` run's checkpoint history read
from the run store, and its conversation continued from an earlier checkpoint as a new run the
run store records (who rewound what, from where) — the state LangGraph's own time travel
gives, each call that did more than read asked about again before it is made a second time,
the run rewound left as it was."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from agents import Agent as OpenAIAgent
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool as lc_tool
from langgraph.checkpoint.memory import InMemorySaver

from tests.support.arm import arm
from tests.support.chat_model import ScriptedChatModel
from tests.support.journals import journal_of
from tests.support.models import ScriptedChat
from trellis import Harness, ReAct, Settings, middleware, tool
from trellis.contracts import ConfigurationError, RunStatus
from trellis.harness.bridges.rewind import CHECKPOINTS_PAGE
from trellis.harness.middleware import RunCheckpointer

done: list[str] = []


@tool(side_effects="read")
def order(order_id: str) -> str:
    """An order's state."""
    return f"{order_id}: paid"


@tool(side_effects="write")
def refund(order_id: str) -> str:
    """Refund an order."""
    done.append(f"refund {order_id}")
    return f"refunded {order_id}"


@tool(side_effects="irreversible")
def close(order_id: str) -> str:
    """Close an order for good."""
    done.append(f"close {order_id}")
    return f"closed {order_id}"


@pytest.fixture(autouse=True)
def nothing_done() -> None:
    done.clear()


LOOKUPS: list[Any] = [("order", {"order_id": f"o{n}"}) for n in (1, 2, 3)]
REFUND = ("refund", {"order_id": "o1"})
CLOSE = ("close", {"order_id": "o1"})


def refunds(harness: Harness, turns: list[Any]) -> tuple[Any, ScriptedChat]:
    model = ScriptedChat(turns)
    graph = ReAct("You handle refunds.", model)
    return arm(harness.wrap(graph, id="refunds"), [order, refund, close]), model


async def paused_after_a_refund(harness: Harness, more: list[Any]) -> tuple[Any, Any, str]:
    """A run that looked three orders up, refunded one — the refund allowed for the
    conversation — and waits for the approval of closing it; the agent, its model (``more``:
    its turns after) and the run's id."""
    agent, model = refunds(harness, [*LOOKUPS, REFUND, CLOSE, *more])
    asked = await agent.run("refund and close o1", user="u", thread="t")
    assert asked.interrupt is not None
    paused = await agent.resume(
        asked.interrupt.interrupt_id, "approve", reviewer="u", remember="run"
    )
    assert paused.status is RunStatus.PAUSED and done == ["refund o1"]
    return agent, model, paused.run_id


def said(state: Any) -> list[str]:
    return [str(m.content) for m in state.values.get("messages", [])]


def id_of(state: Any) -> str:
    return str(state.config["configurable"]["checkpoint_id"])


async def history(agent: Any, run_id: str) -> list[Any]:
    return [state async for state in agent.checkpoints(run_id, reader="u")]


async def before_the_refund(agent: Any, run_id: str) -> str:
    """The run's checkpoint whose state ends with the last lookup: the earliest such."""
    return id_of([s for s in await history(agent, run_id) if said(s)[-1:] == ["o3: paid"]][-1])


# --------------------------------------------------------------------------- the history
async def test_a_paused_runs_history_is_listed_newest_first_page_after_page(
    harness: Harness,
) -> None:
    agent, _, run_id = await paused_after_a_refund(harness, [])
    states = await history(agent, run_id)
    ids = [id_of(s) for s in states]
    assert len(ids) > CHECKPOINTS_PAGE and len(set(ids)) == len(ids)  # more than one page
    assert ids == sorted(ids, reverse=True) and states[-1].parent_config is None
    assert said(states[0])[-2] == "refunded o1" and said(states[-1]) == []
    assert all(s.config["configurable"]["thread_id"] == run_id for s in states)


async def test_a_run_that_has_not_worked_yet_has_no_history(harness: Harness) -> None:
    agent, _ = refunds(harness, [])
    queued = await agent.start("refund o1", user="u")
    assert await history(agent, queued.id) == []


async def test_an_ended_run_says_the_run_store_keeps_none_of_its_checkpoints(
    harness: Harness,
) -> None:
    agent, _ = refunds(harness, [LOOKUPS[0], "paid"])
    ended = await agent.run("is o1 paid?", user="u")
    assert ended.status is RunStatus.SUCCESS
    with pytest.raises(ConfigurationError, match=r"has ended \(success\).*rewindable"):
        await history(agent, ended.run_id)
    with pytest.raises(ConfigurationError, match=r"has ended \(success\)"):
        await agent.rewind(ended.run_id, "any", reviewer="u")


async def test_an_agent_wrapped_rewindable_rewinds_a_run_that_ended(harness: Harness) -> None:
    """``h.wrap(rewindable=True)``: the run store keeps an ended run's checkpoint history (its
    last state included), so the run is read and rewound as a paused one is — its write past
    the checkpoint asks a person before it is made again."""
    model = ScriptedChat([*LOOKUPS, REFUND, "refunded", REFUND, "refunded again"])
    graph = ReAct("You handle refunds.", model)
    agent = arm(harness.wrap(graph, id="refunds-kept", rewindable=True), [order, refund, close])
    first = await agent.run("refund o1", user="u", thread="t")
    asked = first
    if first.interrupt is not None:  # the refund asks first: approved for the conversation
        asked = await agent.resume(
            first.interrupt.interrupt_id, "approve", reviewer="u", remember="run"
        )
    assert asked.status is RunStatus.SUCCESS and asked.answer == "refunded", asked.error
    states = await history(agent, asked.run_id)
    assert states and said(states[0])[-1] == "refunded"  # its end is in the history
    before = [s for s in states if said(s)[-1:] == ["o3: paid"]][-1]
    rewound = await agent.rewind(asked.run_id, id_of(before), reviewer="u")
    assert rewound.interrupt is not None  # the refund it made past the checkpoint asks again
    again = await agent.resume(rewound.interrupt.interrupt_id, "approve", reviewer="u")
    assert again.status is RunStatus.SUCCESS and again.answer == "refunded again", again.error


# --------------------------------------------------------------------------- the rewind
async def test_a_rewound_run_sees_the_state_langgraphs_time_travel_gives(harness: Harness) -> None:
    @lc_tool
    def lookup(key: str) -> str:
        """Look a key up."""
        return f"value of {key}"

    @lc_tool
    def publish(key: str) -> str:
        """Publish a key."""
        return f"published {key}"

    lookup.metadata = {"side_effects": "read"}
    publish.metadata = {"side_effects": "irreversible"}
    turns: list[Any] = [
        ("lookup", {"key": "a"}),
        ("lookup", {"key": "b"}),
        ("publish", {"key": "a"}),
    ]

    def after_the_first(states: list[Any]) -> Any:
        return [s for s in states if said(s)[-1:] == ["value of a"]][-1]

    def conversation(model: ScriptedChatModel) -> list[tuple[str, str]]:
        return [(m.type, str(m.content)) for m in model.seen[-1] if m.type != "system"]

    # LangGraph's own: a thread on InMemorySaver, continued from the checkpoint
    theirs = ScriptedChatModel(turns=[*turns, "published", "only a"])
    plain = create_agent(theirs, tools=[lookup, publish], checkpointer=InMemorySaver())
    config: Any = {"configurable": {"thread_id": "t"}}
    await plain.ainvoke({"messages": [HumanMessage("look a and b up")]}, config)
    earlier = after_the_first([s async for s in plain.aget_state_history(config)])
    await plain.ainvoke(None, earlier.config)
    # the harness's: the same graph with the harness's middleware, its paused run rewound
    ours = ScriptedChatModel(turns=[*turns, "only a"])
    graph = create_agent(
        ours, tools=[lookup, publish], middleware=middleware(), checkpointer=RunCheckpointer()
    )
    agent = harness.wrap(graph, id="lookups")
    paused = await agent.run("look a and b up", user="u")
    assert paused.status is RunStatus.PAUSED
    checkpoint = after_the_first(await history(agent, paused.run_id))
    result = await agent.rewind(paused.run_id, id_of(checkpoint), reviewer="u")
    assert result.status is RunStatus.SUCCESS and result.answer == "only a"
    assert conversation(ours) == conversation(theirs)
    assert conversation(ours)[-1] == ("tool", "value of a")  # nothing past the checkpoint


async def test_a_write_made_past_the_checkpoint_asks_again_and_runs_once_when_approved(
    harness: Harness,
) -> None:
    agent, _, run_id = await paused_after_a_refund(harness, [LOOKUPS[0], REFUND, "refunded o1"])
    checkpoint = await before_the_refund(agent, run_id)
    asked = await agent.rewind(run_id, checkpoint, reviewer="u")
    # the lookup past the checkpoint ran again unasked; the refund, allowed for the
    # conversation before, is a second real-world effect: it asks
    assert asked.status is RunStatus.PAUSED and asked.interrupt is not None
    assert asked.run_id != run_id and done == ["refund o1"]
    assert asked.interrupt.tool_call is not None and asked.interrupt.tool_call.tool == "refund"
    assert "before the rewind" in asked.interrupt.question
    ended = await agent.resume(asked.interrupt.interrupt_id, "approve", reviewer="u")
    assert ended.status is RunStatus.SUCCESS and ended.answer == "refunded o1"
    assert done == ["refund o1", "refund o1"]  # once more, once


async def test_a_rejected_repeat_never_runs_and_the_next_one_is_decided_as_ever(
    harness: Harness,
) -> None:
    agent, model, run_id = await paused_after_a_refund(harness, [REFUND, REFUND, "left as it is"])
    asked = await agent.rewind(run_id, await before_the_refund(agent, run_id), reviewer="u")
    assert asked.interrupt is not None
    again = await agent.resume(
        asked.interrupt.interrupt_id, "reject", answer="once is enough", reviewer="u"
    )
    # asked about once: the model's next try of it is this conversation's allowed refund
    assert again.status is RunStatus.SUCCESS and again.answer == "left as it is"
    assert done == ["refund o1", "refund o1"]
    told = [str(m["content"]) for m in model.requests[-1]["messages"] if m["role"] == "tool"]
    assert "once is enough" in told[-2] and told[-1] == "refunded o1"


async def test_a_write_made_before_the_checkpoint_is_decided_as_ever(harness: Harness) -> None:
    agent, _, run_id = await paused_after_a_refund(harness, [REFUND, "refunded twice"])
    after = [s for s in await history(agent, run_id) if said(s)[-1:] == ["refunded o1"]][-1]
    result = await agent.rewind(run_id, id_of(after), reviewer="u")
    # the refund is behind the checkpoint: one more of it is this conversation's allowed call
    assert result.status is RunStatus.SUCCESS and result.answer == "refunded twice"
    assert done == ["refund o1", "refund o1"]


async def test_values_are_applied_as_langgraphs_update_state_applies_them(harness: Harness) -> None:
    agent, model, run_id = await paused_after_a_refund(harness, [])
    values = {"messages": [AIMessage("The customer withdrew the request.")]}
    asked_before = len(model.requests)
    result = await agent.rewind(
        run_id, await before_the_refund(agent, run_id), values=values, reviewer="u"
    )
    assert result.answer == "The customer withdrew the request." and done == ["refund o1"]
    assert len(model.requests) == asked_before  # the state's answer is the run's: no model call


async def test_the_rewind_is_a_new_run_on_record_and_the_run_rewound_is_as_it_was(
    harness: Harness,
) -> None:
    agent, _, run_id = await paused_after_a_refund(harness, ["nothing to do"])
    before = [id_of(s) for s in await history(agent, run_id)]
    checkpoint = await before_the_refund(agent, run_id)
    result = await agent.rewind(run_id, checkpoint, reviewer="u")
    assert result.status is RunStatus.SUCCESS and result.run_id != run_id
    record = await harness.runs.get(result.run_id)
    assert record is not None and (record.thread_id, record.user_id) == ("t", "u")
    rewound = record.metadata["rewound"]
    assert (rewound["run_id"], rewound["checkpoint_id"], rewound["reviewer"]) == (
        run_id,
        checkpoint,
        "u",
    )
    assert rewound["at"].startswith("20")
    source = await harness.runs.get(run_id)
    assert source is not None and source.status is RunStatus.PAUSED
    assert source.awaiting is not None and "rewound" not in source.metadata
    assert [id_of(s) for s in await history(agent, run_id)] == before
    # and it is answered as ever: the close it waits on runs
    closed = await agent.resume(run_id, "reject", reviewer="u")
    assert closed.status is RunStatus.ERROR  # its model's turns went to the rewound run


async def test_the_spend_before_the_rewind_stays_counted(harness: Harness) -> None:
    agent, _, run_id = await paused_after_a_refund(harness, [CLOSE])
    source = await harness.runs.get(run_id)
    assert source is not None
    spent = (await journal_of(harness.runs, source)).spent["model_calls"]
    paused = await agent.rewind(run_id, await before_the_refund(agent, run_id), reviewer="u")
    record = await harness.runs.get(paused.run_id)
    assert record is not None and record.status is RunStatus.PAUSED
    assert (await journal_of(harness.runs, record)).spent["model_calls"] == spent + 1


async def test_another_harness_on_the_same_run_store_rewinds_the_run(harness: Harness) -> None:
    _, _, run_id = await paused_after_a_refund(harness, [])
    async with Harness(config=Settings(), runs=harness.runs) as other:
        agent, model = refunds(other, ["seen from elsewhere"])
        result = await agent.rewind(run_id, await before_the_refund(agent, run_id), reviewer="u")
    assert result.status is RunStatus.SUCCESS and result.answer == "seen from elsewhere"
    assert len(model.requests) == 1 and done == ["refund o1"]


# --------------------------------------------------------------------------- refused
async def test_a_run_at_work_is_not_rewound(harness: Harness) -> None:
    reached, release = asyncio.Event(), asyncio.Event()

    @tool(side_effects="read")
    async def slow() -> str:
        """Takes a while."""
        reached.set()
        await release.wait()
        return "done"

    model = ScriptedChat([("slow", {}), "done"])
    agent = arm(harness.wrap(ReAct("s", model), id="slow"), [slow])
    running = asyncio.create_task(agent.run("go", user="u", thread="t"))
    await reached.wait()
    [summary] = [r async for r in harness.runs.iterate(agent_id="slow")]
    with pytest.raises(ConfigurationError, match="is running: a run is rewound while it is paused"):
        await agent.rewind(summary.run_id, "any", reviewer="u")
    release.set()
    assert (await running).answer == "done"


async def test_only_the_runs_user_rewinds_it_and_only_from_a_checkpoint_of_its_own(
    harness: Harness,
) -> None:
    agent, _, run_id = await paused_after_a_refund(harness, [CLOSE])
    checkpoint = await before_the_refund(agent, run_id)
    with pytest.raises(ConfigurationError, match="only its user rewinds it, not someone-else"):
        await agent.rewind(run_id, checkpoint, reviewer="someone-else")
    with pytest.raises(ConfigurationError, match=f"no checkpoint nowhere of run {run_id}"):
        await agent.rewind(run_id, "nowhere", reviewer="u")
    other = await agent.run("close o1", user="u", thread="t2")
    assert other.status is RunStatus.PAUSED  # another run, paused on its own close
    with pytest.raises(ConfigurationError, match=f"no checkpoint {checkpoint} of run"):
        await agent.rewind(other.run_id, checkpoint, reviewer="u")  # another run's checkpoint
    assert done == ["refund o1"]
    assert len([r async for r in harness.runs.iterate(agent_id="refunds")]) == 2  # none started


async def test_a_run_of_another_agent_or_tenant_is_refused_as_any_read_of_it_is(
    harness: Harness,
) -> None:
    agent, _, run_id = await paused_after_a_refund(harness, [])
    model = ScriptedChat([])
    stranger = harness.wrap(ReAct("You handle refunds.", model), id="strangers")
    refusal = f"no run {run_id} of agent strangers"
    with pytest.raises(ConfigurationError, match=refusal):
        await stranger.resume(run_id, "approve", reviewer="u")  # what any use of a run answers
    with pytest.raises(ConfigurationError, match=refusal):
        await history(stranger, run_id)
    with pytest.raises(ConfigurationError, match=refusal):
        await stranger.rewind(run_id, "any", reviewer="u")
    with pytest.raises(ConfigurationError, match="speaks for"):
        [s async for s in agent.checkpoints(run_id, reader="u", tenant="another-tenant")]
    with pytest.raises(ConfigurationError, match="only its user reads its checkpoints"):
        [s async for s in agent.checkpoints(run_id, reader="mallory")]  # its states: u's chat


async def answers(question: str, agent: Any) -> str:
    return "an answer"


@pytest.mark.parametrize(
    "target",
    [
        answers,
        OpenAIAgent(name="assistant"),
        create_agent(ScriptedChatModel(turns=["hi"]), checkpointer=InMemorySaver()),
    ],
    ids=["function", "openai-agents", "a-graph-with-its-own-checkpointer"],
)
async def test_an_agent_with_no_checkpoint_history_says_so(harness: Harness, target: Any) -> None:
    agent = harness.wrap(target, id="plain")
    unsupported = "checkpoints and rewind are not supported for this agent"
    with pytest.raises(ConfigurationError, match=unsupported):
        await history(agent, "run_1")
    with pytest.raises(ConfigurationError, match=unsupported):
        await agent.rewind("run_1", "any", reviewer="u")


async def test_a_rewound_run_records_its_question_once_in_memory(
    memory_harness: Harness, memory_service: Any
) -> None:
    """The rewound run's question is the run's it continues: recorded under that run's question
    id, so memory holds it once — the conversation does not gain a turn asked twice."""
    agent, _, run_id = await paused_after_a_refund(memory_harness, [REFUND, "refunded twice"])
    after = [s for s in await history(agent, run_id) if said(s)[-1:] == ["refunded o1"]][-1]
    result = await agent.rewind(run_id, id_of(after), reviewer="u")
    assert result.status is RunStatus.SUCCESS, result.error
    await memory_harness.writes.drain()
    asked = [
        m
        for kept in memory_service.threads.values()
        for m in kept
        if m["role"] == "USER" and "refund and close o1" in str(m.get("content"))
    ]
    assert len(asked) == 1, asked
