"""The conversation the memory service replays is sized for the request it goes into: a ReAct
agent with a known window asks the window with ``conversation_token_budget`` — what the window
leaves once the instructions, the offered tools, the question, the memory context and a reply's
room are in — and the turn under way never clears the replayed turn's results
(``middleware.ThisTurnsResults``)."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately

from tests.support.arm import arm
from tests.support.memory import FakeMemoryService
from tests.support.models import ScriptedChat
from trellis import Harness, ReAct, tool
from trellis.harness.agent import CONVERSATION_TOKENS, REPLY_TOKENS, conversation_budget
from trellis.harness.clients.memory import context_budget
from trellis.harness.middleware import ThisTurnsResults
from trellis.harness.react import CLEARED

SYSTEM = "You answer stock price questions."


@tool(side_effects="read")
def stock_quote(symbol: str) -> str:
    """The latest closing quote of a stock symbol.

    Args:
        symbol: the ticker, e.g. ACME
    """
    return "123.45"


@pytest.mark.asyncio
async def test_a_known_window_sizes_the_replayed_conversation(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    model = ScriptedChat(["Hello."])
    agent = arm(memory_harness.wrap(ReAct(SYSTEM, model, context_window=16_384), id="q"), [])
    await agent.run("What did ACME close at?", user="u1", thread="t1")
    [asked] = memory_service.named("context")
    budget = asked.body["conversation_token_budget"]
    room = int(16_384 * 0.95) - context_budget(16_384) - REPLY_TOKENS
    assert asked.body["window_format"] == "messages"
    assert room // 2 < budget < room  # less the instructions, the tools offered and the question


@pytest.mark.asyncio
async def test_an_unknown_window_leaves_the_budget_to_the_service(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    agent = arm(memory_harness.wrap(ReAct(SYSTEM, ScriptedChat(["Hi."])), id="q"), [])
    await agent.run("Hello?", user="u1", thread="t1")
    [asked] = memory_service.named("context")
    assert "conversation_token_budget" not in asked.body


def test_the_budget_is_what_the_window_leaves_within_the_services_bounds() -> None:
    assert conversation_budget(None, holds=10, context=2000) is None
    assert (
        conversation_budget(16_384, holds=500, context=2000)
        == int(16_384 * 0.95) - 500 - 2000 - REPLY_TOKENS
    )
    assert conversation_budget(2_000, holds=500, context=2000) == CONVERSATION_TOKENS[0]
    assert conversation_budget(10**8, holds=0, context=0) == CONVERSATION_TOKENS[1]


def _turns() -> list:
    earlier = [
        HumanMessage("What did ACME close at?"),
        AIMessage("", tool_calls=[{"id": "a1", "name": "stock_quote", "args": {"s": "A"}}]),
        ToolMessage("x" * 4000, tool_call_id="a1"),
        AIMessage("123.45"),
    ]
    now = [HumanMessage("And the tape?")]
    for n in range(4):
        now += [
            AIMessage("", tool_calls=[{"id": f"b{n}", "name": "tape", "args": {"p": n}}]),
            ToolMessage("y" * 400, tool_call_id=f"b{n}"),
        ]
    return earlier + now


def test_the_turn_under_way_clears_only_its_own_older_results() -> None:
    messages = _turns()
    ThisTurnsResults(trigger=500, keep=3, placeholder=CLEARED).apply(
        messages, count_tokens=count_tokens_approximately
    )
    results = [m.content for m in messages if isinstance(m, ToolMessage)]
    assert results[0] == "x" * 4000  # the replayed turn's result, kept
    assert results[1] == CLEARED and results[2:] == ["y" * 400] * 3


def test_a_request_under_the_trigger_is_left_as_it_is() -> None:
    messages = _turns()
    ThisTurnsResults(trigger=100_000, keep=1).apply(
        messages, count_tokens=count_tokens_approximately
    )
    assert messages == _turns()
