"""The guardrail recipe (``docs/frameworks/react.md``): LangChain's ``@before_agent`` and
``@after_agent`` with ``jump_to`` given to ``ReAct(middleware=)`` — an input the guard blocks
never reaches the model or a tool, an answer it blocks is replaced, and the blocked turn is the
run's as any other: its record, its stream, the conversation's memory."""

from __future__ import annotations

from typing import Any

from langchain.agents.middleware import AgentState, after_agent, before_agent
from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime as GraphRuntime

from tests.support.arm import arm
from tests.support.memory import FakeMemoryService
from tests.support.models import ScriptedChat
from trellis import Harness, ReAct, tool
from trellis.contracts import RunEventType, RunStatus

BLOCKED = "I cannot help with card numbers."
WITHHELD = "The answer was withheld: it held a card number."
ran: list[str] = []


@tool(side_effects="read")
def account(name: str) -> str:
    """An account's card on file."""
    ran.append(name)
    return "4111 1111 1111 1111"


@before_agent(can_jump_to=["end"])
def no_card_numbers(state: AgentState[Any], runtime: GraphRuntime[Any]) -> dict[str, Any] | None:
    """The input guard: a question that carries a card number ends the turn before any model."""
    if "4111" in str(state["messages"][-1].content):
        return {"messages": [AIMessage(BLOCKED)], "jump_to": "end"}
    return None


@after_agent
def no_cards_out(state: AgentState[Any], runtime: GraphRuntime[Any]) -> dict[str, Any] | None:
    """The output guard: an answer that carries a card number is replaced."""
    answer = state["messages"][-1]
    if "4111" in str(answer.content):
        return {"messages": [AIMessage(WITHHELD, id=answer.id)]}
    return None


def guarded(harness: Harness, turns: list[Any]) -> tuple[Any, ScriptedChat]:
    model = ScriptedChat(turns)
    graph = ReAct("You help with accounts.", model, middleware=[no_card_numbers, no_cards_out])
    return arm(harness.wrap(graph, id="accounts"), [account]), model


async def test_a_blocked_input_reaches_no_model_and_is_the_runs_turn(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    agent, model = guarded(memory_harness, [])
    events = [e async for e in agent.stream("charge 4111 1111 1111 1111", user="u", thread="t")]
    done = next(e for e in events if e.type is RunEventType.RUN_FINISHED)
    assert done.data["result"] == BLOCKED
    assert model.requests == [] and ran == []
    said = "".join(e.data["delta"] for e in events if e.type is RunEventType.TEXT_MESSAGE_CONTENT)
    assert said == BLOCKED  # on the stream as any answer
    record = await memory_harness.runs.get(done.run_id)
    assert record is not None and record.status is RunStatus.SUCCESS and record.output == BLOCKED
    kept = [
        m["content"] for batch in memory_service.named("messages") for m in batch.body["messages"]
    ]
    assert kept == ["charge 4111 1111 1111 1111", BLOCKED]  # the conversation remembers it


async def test_a_blocked_answer_is_replaced_and_the_tool_call_stays_recorded(
    harness: Harness,
) -> None:
    agent, _ = guarded(harness, [("account", {"name": "ada"}), "It is 4111 1111 1111 1111."])
    events = [e async for e in agent.stream("which card has ada?", user="u")]
    done = next(e for e in events if e.type is RunEventType.RUN_FINISHED)
    assert done.data["result"] == WITHHELD
    results = [e.data["tool"] for e in events if e.type is RunEventType.TOOL_CALL_RESULT]
    assert results == ["account"] and ran == ["ada"]


async def test_an_input_the_guard_lets_through_is_answered_as_ever(harness: Harness) -> None:
    agent, model = guarded(harness, ["Ada has one card."])
    result = await agent.run("how many cards has ada?", user="u")
    assert result.answer == "Ada has one card." and len(model.requests) == 1
