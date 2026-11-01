"""A large result a conversation's turn saved stays readable in its later turns (ADV-6): the
history names its path (``/large_tool_results/...``), and a follow-up's ``query_result`` or
``read_file`` on it finds it — on ``ReAct``, a ``create_agent`` graph with ``middleware()`` and
OpenAI Agents, as Deep Agents keeps its files in the thread's state. Kept in this process per
conversation (``Agent.results``), never another user's or another thread's; a path no longer
kept says so and how to fetch it again (``results.GONE``)."""

from __future__ import annotations

from typing import Any

import pytest
from agents import Agent
from langchain.agents import create_agent

from tests.support.arm import arm
from tests.support.chat_model import ScriptedChatModel
from tests.support.models import ScriptedChat
from tests.support.openai_model import ScriptedModel
from trellis import Harness, ReAct, middleware, tool
from trellis.contracts import RunStatus
from trellis.harness.journal import content_key
from trellis.harness.middleware import RunCheckpointer
from trellis.harness.tools import results

ROWS = 6000
TEXT = "sku,stock\n" + "\n".join(f"S{n:05d},{n % 13}" for n in range(ROWS))
PATH = f"{results.PREFIX}/{content_key('result', TEXT)}"
COUNT = sum(1 for n in range(ROWS) if n % 13 < 2)
SQL = "SELECT count(*) AS low FROM result WHERE stock < 2"
FOLLOW_UP: list[Any] = [("query_result", {"file_path": PATH, "sql": SQL}), "counted"]


@tool(side_effects="read")
def stock_list() -> str:
    """Every SKU's stock."""
    return TEXT


def targets() -> dict[str, tuple[Any, Any]]:
    """Each adapter's target, scripted over two turns: the large fetch, then a query on it."""
    turns: list[Any] = [("stock_list", {}), "saved", *FOLLOW_UP]
    react = ScriptedChat(list(turns))
    graph = ScriptedChatModel(turns=list(turns))
    openai = ScriptedModel(list(turns))
    return {
        "react": (ReAct(system_prompt="s", model=react), react),
        "middleware": (
            create_agent(graph, tools=[], middleware=middleware(), checkpointer=RunCheckpointer()),
            graph,
        ),
        "openai_agents": (Agent(name="stock", instructions="s", model=openai), openai),
    }


def said(model: Any) -> str:
    kept = getattr(model, "seen", None) or getattr(model, "inputs", None) or model.requests
    return str(kept[-1])


@pytest.mark.parametrize("framework", ["react", "middleware", "openai_agents"])
async def test_a_result_saved_in_one_turn_is_queried_in_the_next(
    harness: Harness, framework: str
) -> None:
    target, model = targets()[framework]
    agent = arm(harness.wrap(target, id=f"kept-{framework}", without={"memory"}), [stock_list])
    first = await agent.run("list the stock", user="ada", thread="t1")
    assert first.status is RunStatus.SUCCESS, first.error
    second = await agent.run("how many are low?", user="ada", thread="t1")
    assert second.status is RunStatus.SUCCESS, second.error
    told = said(model)
    assert f"low\\n{COUNT}" in told, told[-600:]


async def test_another_thread_or_user_does_not_read_it(harness: Harness) -> None:
    model = ScriptedChat([("stock_list", {}), "saved", *FOLLOW_UP, *FOLLOW_UP])
    agent = arm(harness.wrap(ReAct(system_prompt="s", model=model), id="scoped"), [stock_list])
    await agent.run("list the stock", user="ada", thread="t1")
    for user, thread in (("ada", "t2"), ("bob", "t1")):
        await agent.run("how many are low?", user=user, thread=thread)
        told = said(model)
        assert "not a result this conversation keeps" in told, told[-400:]


def test_the_least_recently_used_conversation_goes_past_the_bound() -> None:
    kept = results.Conversations(most=10)
    kept.add("a", {"/x": "12345"}, {"/x": "tool_a"})
    kept.add("b", {"/y": "123456"}, {})
    assert kept.of("a") == ({}, {}) and kept.of("b") == ({"/y": "123456"}, {})
    kept.add("b", {"/y": "123456", "/z": "1234"}, {"/z": "tool_z", "/w": "none"})
    assert kept.of("b") == ({"/y": "123456", "/z": "1234"}, {"/z": "tool_z"})
    kept.add("c", {"/q": "12345678901"}, {})  # alone past the bound: kept by nobody
    assert kept.chars == 0 and kept.of("c") == ({}, {})
