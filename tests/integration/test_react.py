"""``ReAct``: the harness's own loop over chat completions — native tool messages,
structured output, and a bound on model calls."""

from __future__ import annotations

import httpx
import respx
from pydantic import BaseModel

from tests.support.models import ScriptedChat
from trellis import Harness, ReAct, Settings, tool
from trellis.contracts import RunStatus


@tool(side_effects="read")
def stock(sku: str) -> int:
    """Units of a SKU in stock."""
    return {"a": 7}.get(sku, 0)


class Answer(BaseModel):
    sku: str
    units: int


async def test_tool_calls_use_native_tool_messages(harness: Harness) -> None:
    model = ScriptedChat([("stock", {"sku": "a"}), "7 units"])
    agent = harness.wrap(
        ReAct(system="You answer stock questions.", model=model), id="stock", tools=[stock]
    )
    result = await agent.run("how many a?", user="u1")
    assert result.answer == "7 units"
    second = model.requests[1]["messages"]
    assert second[-2]["tool_calls"][0]["function"]["name"] == "stock"
    assert second[-1] == {"role": "tool", "tool_call_id": "call_1", "content": "7"}
    assert model.requests[0]["tools"][0]["function"]["name"] == "stock"


async def test_structured_output_is_parsed_into_the_model(harness: Harness) -> None:
    model = ScriptedChat(['```json\n{"sku": "a", "units": 7}\n```'])
    agent = harness.wrap(ReAct(system="s", model=model, output=Answer), id="structured")
    result = await agent.run("a?", user="u1")
    assert result.answer == Answer(sku="a", units=7)
    assert model.requests[0]["response_format"]["json_schema"]["name"] == "Answer"
    record = await harness.runs.get(result.run_id)
    assert record is not None and record.output == {"sku": "a", "units": 7}


async def test_an_unknown_tool_is_told_to_the_model(harness: Harness) -> None:
    model = ScriptedChat([("nope", {}), "sorry"])
    agent = harness.wrap(ReAct(system="s", model=model), id="unknown")
    assert (await agent.run("x", user="u1")).answer == "sorry"
    assert "no tool 'nope'" in model.requests[1]["messages"][-1]["content"]


async def test_a_loop_that_never_answers_is_stopped(harness: Harness) -> None:
    model = ScriptedChat([("stock", {"sku": "a"})] * 3)
    agent = harness.wrap(ReAct(system="s", model=model, max_steps=3), id="loop", tools=[stock])
    result = await agent.run("x", user="u1")
    assert result.status is RunStatus.ERROR and result.error is not None
    assert "3 model calls" in result.error.message


@respx.mock
async def test_a_model_name_goes_to_bifrost() -> None:
    respx.post("http://gw.test/mcp").mock(
        return_value=httpx.Response(200, json={"jsonrpc": "2.0", "id": 1, "result": {"tools": []}})
    )
    route = respx.post("http://gw.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200, json={"choices": [{"message": {"role": "assistant", "content": "hi"}}]}
        )
    )
    async with Harness(
        config=Settings(bifrost_url="http://gw.test/v1", bifrost_virtual_key="vk")
    ) as h:
        result = await h.wrap(ReAct(system="s", model="gemini/gemini-3.8-flash"), id="named").run(
            "x", user="u"
        )
    assert result.answer == "hi"
    request = route.calls[0].request
    assert request.headers["authorization"] == "Bearer vk"
    assert b'"model":"gemini/gemini-3.8-flash"' in request.content.replace(b" ", b"")


async def test_a_model_name_without_bifrost_is_an_error(harness: Harness) -> None:
    result = await harness.wrap(ReAct(system="s", model="some/model"), id="nobifrost").run(
        "x", user="u"
    )
    assert (
        result.status is RunStatus.ERROR
        and result.error is not None
        and "BIFROST_URL" in result.error.message
    )
