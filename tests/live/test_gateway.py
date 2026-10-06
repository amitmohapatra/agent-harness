"""Opt-in: real models through a running Bifrost gateway (``BIFROST_URL``, e.g.
``http://localhost:8091/v1``). Skipped when it is unset. ``make test-live`` runs them."""

from __future__ import annotations

import json
import os

import pytest
from agents import Agent, OpenAIChatCompletionsModel
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from openai import AsyncOpenAI
from pydantic import BaseModel

from tests.live.conftest import MODEL
from tests.live.proof import RUN_SECONDS, TEST_SECONDS, Proof, ended, governed, recorded, streamed
from tests.support.planned import PlannedChat
from trellis import Harness, ReAct, Settings, tool
from trellis.contracts import RunStatus

URL = os.environ.get("BIFROST_URL")
KEY = os.environ.get("BIFROST_VIRTUAL_KEY") or "unused"

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


#: How a ``ReAct`` run ends on its model's account: an answer not in the schema, no answer
#: within its steps, a stall.
REACT_MODEL_ERRORS = frozenset({"ValidationError", "ModelError"})


def harness() -> Harness:
    """The gateway alone: no memory, runs in process."""
    return Harness(config=Settings(bifrost_url=URL, bifrost_virtual_key=KEY))


@pytest.mark.timeout(TEST_SECONDS)
async def test_react_calls_a_tool_and_answers_in_the_schema() -> None:
    """``ReAct`` through the gateway: the run ends, with its events, as the harness ends it; a
    stock call the model makes runs through the harness (on the stream, journaled, governed);
    an answer is an instance of the schema — the harness parsed it — and one that does not
    fit is the run's error. The model's numbers are not judged; when it makes no call, its
    scripted twin makes one, and the twin's answer is the tool's result in the schema."""
    question = "How many units of A-1 are in stock?"
    system = "Answer stock questions with the stock tool."
    async with harness() as h:
        decisions = governed(h, await h.tenant())
        proof = Proof()
        target = ReAct(system=system, model=MODEL, output=Stock, max_steps=4)
        agent = h.wrap(target, id="live-react", tools=[stock], hooks=[proof])
        run = await streamed("react", agent.stream(question, user="live", timeout=RUN_SECONDS))
        ended(run)
        recorded(run, proof, decisions)
        assert proof.result is not None
        if proof.result.status is RunStatus.SUCCESS:
            assert isinstance(proof.result.answer, Stock)
        elif proof.result.status is not RunStatus.TIMEOUT:
            assert proof.result.error is not None, run.summary()
            assert proof.result.error.code in REACT_MODEL_ERRORS, run.summary()
        if "stock" in run.started():
            return
        twin = PlannedChat([("stock", {"sku": "A-1"})], final='{"sku": "A-1", "units": {last}}')
        target = ReAct(system=system, model=twin, output=Stock)
        proof = Proof()
        agent = h.wrap(target, id="live-react-twin", tools=[stock], hooks=[proof])
        run = await streamed("react-twin", agent.stream(question, user="live"))
        ended(run)
        recorded(run, proof, decisions)
        assert run.started() == ["stock"]
        assert proof.result is not None and proof.result.answer == Stock(sku="A-1", units=42)
        assert '"stock"' in json.dumps(twin.requests[0]["tools"])  # offered


async def test_langgraph_with_a_model_pointed_at_bifrost() -> None:
    async with harness() as h:
        model = ChatOpenAI(
            base_url=URL,
            api_key=KEY,  # type: ignore[arg-type]
            model=MODEL,
            max_tokens=2048,  # type: ignore[call-arg]
            default_headers=await h.model_headers(),
        )
        graph = create_agent(model, tools=await h.tools(stock, framework="langgraph"))
        result = await h.wrap(graph, id="live-graph").run(
            "Units of A-1 in stock? Use the tool.", user="live"
        )
    assert result.status is RunStatus.SUCCESS, result.error
    assert "42" in str(result.answer)


async def test_openai_agents_with_a_model_pointed_at_bifrost() -> None:
    async with harness() as h:
        model = OpenAIChatCompletionsModel(
            model=MODEL,
            openai_client=AsyncOpenAI(
                base_url=URL, api_key=KEY, default_headers=await h.model_headers()
            ),
        )
        target = Agent(name="stock", instructions="Use the stock tool.", model=model)
        result = await h.wrap(target, id="live-openai", tools=[stock]).run(
            "Units of A-1?", user="live"
        )
    assert result.status is RunStatus.SUCCESS, result.error
    assert "42" in str(result.answer)


async def test_the_gateway_lists_the_mcp_tools_the_key_allows() -> None:
    async with harness() as h:
        assert h.gateway is not None
        tools = await h.gateway.tools()
    assert all("-" in t.name for t in tools)
