"""Opt-in: real models through a running Bifrost gateway (``BIFROST_URL``, e.g.
``http://localhost:8091/v1``). Skipped when it is unset. ``make test-live`` runs them."""

from __future__ import annotations

import os

import pytest
from agents import Agent, OpenAIChatCompletionsModel
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from openai import AsyncOpenAI
from pydantic import BaseModel

from trellis import Harness, ReAct, Settings, tool
from trellis.contracts import RunStatus

URL = os.environ.get("BIFROST_URL")
KEY = os.environ.get("BIFROST_VIRTUAL_KEY") or "unused"
#: Cheap, and reliable at tool calling through the gateway.
MODEL = "openrouter/openai/gpt-4.1-nano"

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not URL, reason="needs a Bifrost gateway (BIFROST_URL)"),
]


@tool(side_effects="read")
def stock(sku: str) -> int:
    """Units of a SKU in stock. Known SKUs: A-1, B-2."""
    return {"A-1": 42, "B-2": 0}.get(sku.upper(), 0)


class Stock(BaseModel):
    sku: str
    units: int


def harness() -> Harness:
    return Harness(config=Settings(bifrost_url=URL, bifrost_virtual_key=KEY, eval_sample=0.0))


async def test_react_calls_a_tool_and_answers_in_the_schema() -> None:
    async with harness() as h:
        agent = h.wrap(
            ReAct(system="Answer stock questions with the stock tool.", model=MODEL, output=Stock),
            id="live-react",
            tools=[stock],
        )
        result = await agent.run("How many units of A-1 are in stock?", user="live")
    assert result.status is RunStatus.SUCCESS, result.error
    assert result.answer == Stock(sku="A-1", units=42)


async def test_langgraph_with_a_model_pointed_at_bifrost() -> None:
    async with harness() as h:
        model = ChatOpenAI(base_url=URL, api_key=KEY, model=MODEL, max_tokens=2048)  # type: ignore[arg-type]
        graph = create_agent(model, tools=await h.tools(stock, framework="langgraph"))
        result = await h.wrap(graph, id="live-graph").run(
            "Units of A-1 in stock? Use the tool.", user="live"
        )
    assert result.status is RunStatus.SUCCESS, result.error
    assert "42" in str(result.answer)


async def test_openai_agents_with_a_model_pointed_at_bifrost() -> None:
    async with harness() as h:
        model = OpenAIChatCompletionsModel(
            model=MODEL, openai_client=AsyncOpenAI(base_url=URL, api_key=KEY)
        )
        target = Agent(name="stock", instructions="Use the stock tool.", model=model)
        result = await h.wrap(target, id="live-openai", tools=[stock]).run(
            "Units of A-1?", user="live"
        )
    assert result.status is RunStatus.SUCCESS, result.error
    assert "42" in str(result.answer)


async def test_the_gateway_lists_its_mcp_tools() -> None:
    async with harness() as h:
        assert h.gateway is not None
        tools = await h.gateway.tools(["*"], None)
    assert all("-" in t.name for t in tools)
