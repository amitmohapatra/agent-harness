"""``ReAct``: the harness's own loop over chat completions — native tool messages,
structured output, and a bound on model calls."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
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
    model = ScriptedChat([("stock", {"sku": sku}) for sku in "abc"])
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


# --------------------------------------------------------------------------- robustness
async def test_arguments_that_are_not_json_are_an_error_the_model_reads(harness: Harness) -> None:
    broken = {
        "role": "assistant",
        "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "stock", "arguments": "{sku:"}}
        ],
    }
    model = ScriptedChat([broken, ("stock", {"sku": "a"}), "7"])
    agent = harness.wrap(ReAct(system="s", model=model), id="bad-json", tools=[stock])
    result = await agent.run("a?", user="u")
    assert result.status is RunStatus.SUCCESS and result.answer == "7"
    told = model.requests[1]["messages"][-1]["content"]
    assert told.startswith("stock was not run: its arguments are not valid JSON")


@pytest.mark.parametrize(
    ("args", "problem"),
    [
        ("[1, 2]", "its arguments must be a JSON object"),
        ("{}", "missing required argument(s): sku"),
        ('{"sku": "a", "colour": "red"}', "unknown argument(s): colour"),
        ('{"sku": 7}', "sku must be of type string"),
    ],
)
async def test_arguments_that_do_not_fit_the_schema_do_not_run_the_tool(
    harness: Harness, args: str, problem: str
) -> None:
    ran: list[Any] = []

    @tool(side_effects="irreversible")
    def order(sku: str, qty: int = 1, rush: bool = False) -> str:
        """Order a SKU."""
        ran.append(sku)
        return "ordered"

    call = {
        "role": "assistant",
        "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "order", "arguments": args}}
        ],
    }
    model = ScriptedChat([call, "fixed"])
    agent = harness.wrap(ReAct(system="s", model=model), id="schema", tools=[order])
    result = await agent.run("x", user="u")
    assert result.status is RunStatus.SUCCESS and ran == []  # nobody was asked to approve it
    assert problem in model.requests[1]["messages"][-1]["content"]


def test_the_schema_check_knows_the_basic_json_types() -> None:
    from trellis.harness.tools.base import arguments_problem

    schema = {
        "properties": {
            "n": {"type": "integer"},
            "x": {"type": "number"},
            "flag": {"type": "boolean"},
            "any": {},
            "either": {"type": ["string", "null"]},
            "odd": {"type": "decimal"},
        }
    }
    assert arguments_problem(schema, {"n": 1, "x": 1.5, "flag": True, "any": [1]}) is None
    assert arguments_problem(schema, {"either": None, "odd": "1.0", "extra": 1}) is None
    assert arguments_problem(schema, {"n": True}) == "n must be of type integer"
    assert arguments_problem(schema, {"x": "1"}) == "x must be of type number"
    assert arguments_problem(schema, {"flag": 1}) == "flag must be of type boolean"
    assert arguments_problem({}, {"a": 1}) is None


async def test_a_huge_result_is_cut_and_kept_whole_as_a_run_artifact(harness: Harness) -> None:
    @tool(side_effects="read")
    def dump() -> str:
        """Everything."""
        return "x" * 500

    model = ScriptedChat([("dump", {}), "done"])
    target = ReAct(system="s", model=model, max_result_chars=100)
    result = await harness.wrap(target, id="dumper", tools=[dump]).run("x", user="u")
    assert result.status is RunStatus.SUCCESS
    told = model.requests[1]["messages"][-1]["content"]
    assert told.startswith("x" * 100 + "\n…[cut: dump returned 500 characters, 100 are shown")
    artifact_id = told.rsplit("run artifact ", 1)[1].rstrip("]")
    kept = await harness.runs.artifacts.download(artifact_id, tenant="default")
    assert kept is not None and json.loads(kept) == {"tool": "dump", "output": "x" * 500}


async def test_a_huge_result_is_still_cut_when_it_cannot_be_kept(
    harness: Harness, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    async def refused(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("no artifacts here")

    monkeypatch.setattr(harness.runs.artifacts, "upload", refused)

    @tool(side_effects="read")
    def dump() -> str:
        """Everything."""
        return "y" * 50

    model = ScriptedChat([("dump", {}), "done"])
    target = ReAct(system="s", model=model, max_result_chars=10)
    await harness.wrap(target, id="dumper", tools=[dump]).run("x", user="u")
    told = model.requests[1]["messages"][-1]["content"]
    assert told == "y" * 10 + "\n…[cut: dump returned 50 characters, 10 are shown]"
    assert "no artifacts here" in caplog.text


async def test_the_same_call_over_and_over_is_a_stall(harness: Harness) -> None:
    calls: list[str] = []

    @tool(side_effects="read")
    def poll(job: str) -> str:
        """A job's status."""
        calls.append(job)
        return "pending"

    model = ScriptedChat([("poll", {"job": "j"})] * 5)
    agent = harness.wrap(ReAct(system="s", model=model), id="stall", tools=[poll])
    result = await agent.run("x", user="u")
    assert result.status is RunStatus.ERROR and result.error is not None
    assert "called 'poll' with the same arguments 3 times in a row" in result.error.message
    assert calls == ["j", "j"]  # the third was not run


async def test_a_call_that_changes_resets_the_stall_count(harness: Harness) -> None:
    turns: list[Any] = [("stock", {"sku": s}) for s in "aabaab"]
    model = ScriptedChat([*turns, "done"])
    target = ReAct(system="s", model=model, max_repeats=3)
    result = await harness.wrap(target, id="no-stall", tools=[stock]).run("x", user="u")
    assert result.answer == "done"


async def test_a_resume_replays_the_model_steps_before_the_pause(harness: Harness) -> None:
    @tool(side_effects="irreversible")
    def refund(order: str) -> str:
        """Refund an order."""
        return f"refunded {order}"

    model = ScriptedChat([("stock", {"sku": "a"}), ("refund", {"order": "o1"}), "refunded"])
    agent = harness.wrap(ReAct(system="s", model=model), id="refunds", tools=[stock, refund])
    paused = await agent.run("refund o1", user="u")
    assert paused.interrupt is not None and len(model.requests) == 2
    done = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="r")
    assert done.answer == "refunded"
    assert len(model.requests) == 3  # only the step after the approval asked the model


async def test_arguments_a_gateway_already_parsed_are_taken_as_they_are(harness: Harness) -> None:
    parsed = {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "stock", "arguments": {"sku": "a"}},
            }
        ],
    }
    model = ScriptedChat([parsed, "7"])
    agent = harness.wrap(ReAct(system="s", model=model), id="parsed", tools=[stock])
    assert (await agent.run("a?", user="u")).answer == "7"
    assert model.requests[1]["messages"][-1]["content"] == "7"
