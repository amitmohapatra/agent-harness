"""Opt-in: real models, through a running Bifrost gateway (``LLM_GATEWAY=bifrost``,
``LLM_BASE_URL`` e.g. ``http://localhost:8091/v1``) or — with no gateway — directly at the model's
own OpenAI-compatible endpoint (``LLM_BASE_URL``, ``LLM_API_KEY``). Skipped when neither is set;
the listing of the key's MCP tools is Bifrost's own and needs the gateway. ``make test-live``
runs them."""

from __future__ import annotations

import json
import os

import pytest
from agents import Agent, OpenAIChatCompletionsModel
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from openai import AsyncOpenAI
from pydantic import BaseModel

from tests.live.conftest import GATEWAY_URL, MODEL, needs_gateway, needs_model
from tests.live.proof import RUN_SECONDS, TEST_SECONDS, Proof, ended, governed, recorded, streamed
from tests.support.arm import arm, native_tools
from tests.support.planned import PlannedChat
from trellis import Harness, ReAct, Settings, tool
from trellis.contracts import ModelError, RunStatus
from trellis.harness.prompts import GatewayHeaders

#: The endpoint every model call goes to: the gateway's, or the model's own.
URL = os.environ.get("LLM_BASE_URL")
KEY = os.environ.get("LLM_API_KEY") or "unused"
#: The route: ``bifrost`` through the gateway, else the endpoint's (``openai-compatible``).
GATEWAY = "bifrost" if GATEWAY_URL else None
#: What a framework's own model client sends: the gateway's per-request headers through
#: Bifrost; nothing of the gateway's to an endpoint that is not one.
HEADERS = GatewayHeaders() if GATEWAY else None

pytestmark = [pytest.mark.live]


@tool(side_effects="read")
def stock(sku: str) -> int:
    """Units of a SKU in stock. Known SKUs: A-1, B-2."""
    return {"A-1": 42, "B-2": 0}.get(sku.upper(), 0)


class Stock(BaseModel):
    sku: str
    units: int


#: How a ``ReAct`` run ends on its model's account: an answer not in the schema (pydantic's
#: ``ValidationError``), no answer within its steps or a stall (``ModelError``, by its code).
REACT_MODEL_ERRORS = frozenset({"ValidationError", ModelError.code})


def harness() -> Harness:
    """The model's route alone — the gateway, or the endpoint directly: no memory, runs in
    process."""
    return Harness(config=Settings(llm_gateway=GATEWAY, llm_base_url=URL, llm_api_key=KEY))


@needs_model
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
        decisions = governed(h, await h._tenant())
        proof = Proof()
        target = ReAct(system_prompt=system, model=MODEL, response_format=Stock, max_steps=4)
        agent = arm(h.wrap(target, id="live-react", hooks=[proof]), [stock])
        run = await streamed("react", agent.stream(question, user="live", run_timeout=RUN_SECONDS))
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
        target = ReAct(system_prompt=system, model=twin, response_format=Stock)
        proof = Proof()
        agent = arm(h.wrap(target, id="live-react-twin", hooks=[proof]), [stock])
        run = await streamed("react-twin", agent.stream(question, user="live"))
        ended(run)
        recorded(run, proof, decisions)
        assert run.started() == ["stock"]
        assert proof.result is not None and proof.result.answer == Stock(sku="A-1", units=42)
        assert '"stock"' in json.dumps(twin.requests[0]["tools"])  # offered


@needs_model
async def test_langgraph_with_a_model_pointed_at_the_endpoint() -> None:
    async with harness() as h:
        model = ChatOpenAI(
            base_url=URL,
            api_key=KEY,  # type: ignore[arg-type]
            model=MODEL,
            max_tokens=2048,  # type: ignore[call-arg]
            default_headers=HEADERS,
        )
        graph = create_agent(model, tools=await native_tools(h, stock, framework="langgraph"))
        result = await h.wrap(graph, id="live-graph").run(
            "Units of A-1 in stock? Use the tool.", user="live"
        )
    assert result.status is RunStatus.SUCCESS, result.error
    assert "42" in str(result.answer)


@needs_model
async def test_openai_agents_with_a_model_pointed_at_the_endpoint() -> None:
    async with harness() as h:
        model = OpenAIChatCompletionsModel(
            model=MODEL,
            openai_client=AsyncOpenAI(base_url=URL, api_key=KEY, default_headers=HEADERS),
        )
        target = Agent(name="stock", instructions="Use the stock tool.", model=model)
        result = await arm(h.wrap(target, id="live-openai"), [stock]).run(
            "Units of A-1?", user="live"
        )
    assert result.status is RunStatus.SUCCESS, result.error
    assert "42" in str(result.answer)


@needs_gateway
async def test_the_gateway_lists_the_mcp_tools_the_key_allows() -> None:
    async with harness() as h:
        assert h.gateway is not None
        tools = await h.gateway.tools()
    assert all("-" in t.name for t in tools)
