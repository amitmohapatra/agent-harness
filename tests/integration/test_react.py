"""``ReAct(...)``: a LangChain ``create_agent`` graph — native tool messages, structured output,
the native middleware on by default and the harness's on top (in their order), a model name
on the gateway, and the user's middleware added or replacing by name."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
import respx
from deepagents.middleware import FilesystemMiddleware
from deepagents.middleware.summarization import _DeepAgentsSummarizationMiddleware
from langchain.agents.middleware import (
    AgentMiddleware,
    ContextEditingMiddleware,
    SummarizationMiddleware,
)
from langgraph.pregel import Pregel
from pydantic import BaseModel

from tests.support.models import ScriptedChat
from trellis import Harness, ReAct, Settings, tool
from trellis.contracts import ConfigurationError, RunStatus
from trellis.harness import middleware as ours
from trellis.harness.adapters.base import context_window
from trellis.harness.adapters.langgraph import LangGraphAdapter
from trellis.harness.middleware import RunCheckpointer
from trellis.harness.react import stacked


@tool(side_effects="read")
def stock(sku: str) -> int:
    """Units of a SKU in stock."""
    return {"a": 7}.get(sku, 0)


class Answer(BaseModel):
    sku: str
    units: int


def names(request: dict[str, Any]) -> list[str]:
    return [t["function"]["name"] for t in request.get("tools", [])]


async def test_react_is_a_create_agent_graph_the_langgraph_adapter_runs(harness: Harness) -> None:
    model = ScriptedChat([("stock", {"sku": "a"}), "7 units"])
    graph = ReAct(system="You answer stock questions.", model=model)
    assert isinstance(graph, Pregel) and isinstance(graph.checkpointer, RunCheckpointer)
    agent = harness.wrap(graph, id="stock", tools=[stock])
    assert isinstance(agent.adapter, LangGraphAdapter)
    result = await agent.run("how many a?", user="u1")
    assert result.answer == "7 units"
    first, second = model.requests
    assert first["messages"][0]["role"] == "system"
    assert first["messages"][0]["content"].startswith("You answer stock questions.")
    assert names(first) == ["read_file", "read_result", "stock"]  # sorted: the cache holds
    assert second["messages"][-2]["tool_calls"][0]["function"]["name"] == "stock"
    assert second["messages"][-1] == {"role": "tool", "tool_call_id": "call_1", "content": "7"}


async def test_structured_output_is_the_native_response_format(harness: Harness) -> None:
    model = ScriptedChat([("Answer", {"sku": "a", "units": 7})])
    agent = harness.wrap(ReAct(system="s", model=model, output=Answer), id="structured")
    result = await agent.run("a?", user="u1")
    assert result.answer == Answer(sku="a", units=7)
    assert "Answer" in names(model.requests[0])
    record = await harness.runs.get(result.run_id)
    assert record is not None and record.output == {"sku": "a", "units": 7}


async def test_at_the_step_limit_a_structured_answer_is_the_only_tool(harness: Harness) -> None:
    model = ScriptedChat([("stock", {"sku": "a"}), ("Answer", {"sku": "a", "units": 7})])
    target = ReAct(system="s", model=model, output=Answer, max_steps=1)
    result = await harness.wrap(target, id="last", tools=[stock]).run("a?", user="u1")
    assert result.answer == Answer(sku="a", units=7)
    assert names(model.requests[-1]) == ["Answer"]  # its own tool, no other


async def test_an_unknown_tool_is_told_to_the_model(harness: Harness) -> None:
    model = ScriptedChat([("nope", {}), "sorry"])
    agent = harness.wrap(ReAct(system="s", model=model), id="unknown")
    assert (await agent.run("x", user="u1")).answer == "sorry"
    assert "nope is not a valid tool" in model.requests[1]["messages"][-1]["content"]


async def test_a_loop_that_never_answers_is_stopped(harness: Harness) -> None:
    # asked once more without tools at the step limit, the model still calls a tool
    model = ScriptedChat([*[("stock", {"sku": sku}) for sku in "abc"], ("stock", {"sku": "d"})])
    agent = harness.wrap(ReAct(system="s", model=model, max_steps=3), id="loop", tools=[stock])
    result = await agent.run("x", user="u1")
    assert result.status is RunStatus.ERROR and result.error is not None
    assert result.error.message == "stopped after 3 model calls without an answer"
    assert model.requests[-1]["tool_choice"] == "none"


@respx.mock
async def test_a_model_name_is_a_gateway_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BIFROST_URL", "http://gw.test/v1")
    monkeypatch.setenv("BIFROST_VIRTUAL_KEY", "vk")
    respx.post("http://gw.test/mcp").mock(
        return_value=httpx.Response(200, json={"jsonrpc": "2.0", "id": 1, "result": {"tools": []}})
    )
    reply = {
        "id": "c",
        "object": "chat.completion",
        "created": 0,
        "model": "provider/model",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}
        ],
    }
    route = respx.post("http://gw.test/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=reply)
    )
    graph = ReAct(system="s", model="provider/model", context_window=50_000)
    assert context_window(graph) == 50_000
    async with Harness(
        config=Settings(bifrost_url="http://gw.test/v1", bifrost_virtual_key="vk")
    ) as h:
        agent = h.wrap(graph, id="named")
        assert agent.evals.fallback_model == "provider/model"
        result = await agent.run("x", user="u")
    assert result.answer == "hi"
    request = route.calls[0].request
    assert request.headers["authorization"] == "Bearer vk"
    assert request.headers["x-bf-mcp-include-clients"] == ""  # the gateway adds no MCP tools
    assert request.headers["x-bf-mcp-include-tools"] == ""
    assert json.loads(request.content)["model"] == "provider/model"


def test_a_model_name_without_the_gateway_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BIFROST_URL", raising=False)
    with pytest.raises(ConfigurationError, match="BIFROST_URL"):
        ReAct(system="s", model="some/model")


def test_what_react_refuses() -> None:
    model = ScriptedChat([])
    with pytest.raises(ConfigurationError, match="model_timeout"):
        ReAct(system="s", model=model, model_timeout=0)
    with pytest.raises(ConfigurationError, match="context_window"):
        ReAct(system="s", model=model, context_window=0)
    with pytest.raises(ConfigurationError, match="prompt_vars"):
        ReAct(system="s", model=model, prompt_vars={"a": 1})
    with pytest.raises(ConfigurationError):
        ReAct(system="s", model=model, prompt="bad@")


def test_the_window_is_the_models_profile_else_the_default() -> None:
    profiled = ScriptedChat([], profile={"max_input_tokens": 32_000})
    assert context_window(ReAct(system="s", model=profiled)) == 32_000
    unknown = ReAct(system="s", model=ScriptedChat([]))
    assert context_window(unknown) is None  # assumed CONTEXT_WINDOW: not said to the harness


# --------------------------------------------------------------------------- the stack
class Named(AgentMiddleware):
    def __init__(self, name: str) -> None:
        super().__init__()
        self._name = name

    @property
    def name(self) -> str:
        return self._name


def test_given_middleware_replaces_a_default_of_its_name_and_goes_before_model_hooks() -> None:
    defaults = [Named("A"), Named("B"), Named("ModelHooks")]
    mine_b, extra = Named("B"), Named("C")
    stack = stacked(defaults, [extra, mine_b])
    assert [m.name for m in stack] == ["A", "B", "C", "ModelHooks"]
    assert stack[1] is mine_b


async def test_a_summarization_middleware_given_replaces_the_default(harness: Harness) -> None:
    model = ScriptedChat(["hi"])
    mine = SummarizationMiddleware(model, trigger=("messages", 100))
    graph = ReAct(system="s", model=model, middleware=[mine])
    assert (await harness.wrap(graph, id="mine").run("x", user="u")).answer == "hi"


class Mine(AgentMiddleware):
    async def awrap_model_call(self, request: Any, handler: Any) -> Any:
        return await handler(request)


async def test_the_middleware_run_in_their_order(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[str] = []

    def recorded(cls: type, label: str) -> None:
        model_call = cls.awrap_model_call
        tool_call = cls.awrap_tool_call

        async def on_model(self: Any, request: Any, handler: Any) -> Any:
            seen.append(f"model {label}")
            return await model_call(self, request, handler)

        async def on_tool(self: Any, request: Any, handler: Any) -> Any:
            seen.append(f"tool {label}")
            return await tool_call(self, request, handler)

        monkeypatch.setattr(cls, "awrap_model_call", on_model)
        if tool_call is not AgentMiddleware.awrap_tool_call:
            monkeypatch.setattr(cls, "awrap_tool_call", on_tool)

    for cls, label in [
        (FilesystemMiddleware, "files"),
        (ours.HarnessTools, "tools"),
        (ours.StallGuard, "stall"),
        (ours.StepLimit, "steps"),
        (ContextEditingMiddleware, "clear"),
        (_DeepAgentsSummarizationMiddleware, "summary"),
        (Mine, "mine"),
        (ours.ModelHooks, "hooks"),
    ]:
        recorded(cls, label)
    model = ScriptedChat([("stock", {"sku": "a"}), "7"])
    graph = ReAct(system="s", model=model, middleware=[Mine()])
    await harness.wrap(graph, id="order", tools=[stock]).run("x", user="u")
    model_order = ["files", "tools", "stall", "steps", "clear", "summary", "mine", "hooks"]
    first_call = [s.split()[1] for s in seen[: len(model_order)]]
    assert first_call == model_order  # outermost first: ModelHooks sees what is sent
    tool_calls = [s for s in seen if s.startswith("tool ")]
    assert tool_calls == ["tool files", "tool tools"]  # eviction wraps the harness's bridge


async def test_a_resume_continues_in_place(harness: Harness) -> None:
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


async def test_framework_options_are_the_graphs_run_config(harness: Harness) -> None:
    model = ScriptedChat([("stock", {"sku": "a"}), ("stock", {"sku": "b"}), "done"])
    graph = ReAct(system="s", model=model)
    agent = harness.wrap(graph, id="opts", tools=[stock], framework_options={"recursion_limit": 3})
    result = await agent.run("x", user="u")
    assert result.status is RunStatus.ERROR and result.error is not None
    assert "recursion" in result.error.message.lower()
    with pytest.raises(ConfigurationError, match="RunnableConfig"):
        harness.wrap(ReAct(system="s", model=model), id="bad", framework_options={"temp": 1})


# --------------------------------------------------------------------------- robustness
async def test_arguments_that_are_not_json_are_told_to_the_model(harness: Harness) -> None:
    broken = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "stock", "arguments": "{sku:"}}
        ],
    }
    model = ScriptedChat([broken, ("stock", {"sku": "a"}), "7"])
    agent = harness.wrap(ReAct(system="s", model=model), id="bad-json", tools=[stock])
    result = await agent.run("a?", user="u")
    assert result.status is RunStatus.SUCCESS and result.answer == "7"
    told = model.requests[1]["messages"][-1]["content"]
    assert told == (
        "Tool call stock with id c1 could not be executed - arguments were malformed or truncated."
    )


@pytest.mark.parametrize(
    ("args", "problem"),
    [
        ({}, "missing required argument(s): sku"),
        ({"sku": "a", "colour": "red"}, "unknown argument(s): colour"),
        ({"sku": 7}, "sku must be of type string"),
    ],
)
async def test_arguments_that_do_not_fit_the_schema_do_not_run_the_tool(
    harness: Harness, args: dict[str, Any], problem: str
) -> None:
    ran: list[Any] = []

    @tool(side_effects="irreversible")
    def order(sku: str, qty: int = 1, rush: bool = False) -> str:
        """Order a SKU."""
        ran.append(sku)
        return "ordered"

    model = ScriptedChat([("order", args), "fixed"])
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


async def test_a_huge_result_is_saved_as_a_file_the_model_reads_in_pages(
    harness: Harness,
) -> None:
    @tool(side_effects="read")
    def dump() -> str:
        """Everything."""
        return "".join(str(n % 10) for n in range(100_000))

    model = ScriptedChat(
        [("dump", {}), ("read_file", {"file_path": "/large_tool_results/call_1"}), "done"]
    )
    result = await harness.wrap(ReAct(system="s", model=model), id="dumper", tools=[dump]).run(
        "x", user="u"
    )
    assert result.status is RunStatus.SUCCESS, result.error
    told = model.requests[1]["messages"][-1]["content"]
    assert "saved in the filesystem at this path: /large_tool_results/call_1" in told
    assert len(told) < 10_000  # a preview, not the result
    assert "0123456789" in model.requests[2]["messages"][-1]["content"]


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
