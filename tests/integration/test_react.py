"""``ReAct(...)``: a Deep Agents ``create_deep_agent`` graph — native tool messages, structured
output, Deep Agents' middleware and the harness's on top (in their order), a model name on the
gateway, and the user's middleware added (refused when named as the harness's, replacing Deep
Agents' by name)."""

from __future__ import annotations

import asyncio
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
    TodoListMiddleware,
)
from langgraph.pregel import Pregel
from pydantic import BaseModel

from tests.support.arm import arm
from tests.support.models import ScriptedChat
from trellis import Harness, ReAct, Settings, middleware, tool
from trellis.contracts import ConfigurationError, RunStatus
from trellis.harness import middleware as ours
from trellis.harness.adapters.base import context_window
from trellis.harness.adapters.langgraph import LangGraphAdapter
from trellis.harness.automatic import JournalBackend
from trellis.harness.middleware import RunCheckpointer
from trellis.harness.react import defaults_off, stacked


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
    graph = ReAct(system_prompt="You answer stock questions.", model=model)
    assert isinstance(graph, Pregel) and isinstance(graph.checkpointer, RunCheckpointer)
    agent = arm(harness.wrap(graph, id="stock"), [stock])
    assert isinstance(agent.adapter, LangGraphAdapter)
    result = await agent.run("how many a?", user="u1")
    assert result.answer == "7 units"
    first, second = model.requests
    assert first["messages"][0]["role"] == "system"
    assert first["messages"][0]["content"].startswith("You answer stock questions.")
    assert names(first) == ["stock"]  # the read tools: once there is something to read
    assert second["messages"][-2]["tool_calls"][0]["function"]["name"] == "stock"
    assert second["messages"][-1] == {"role": "tool", "tool_call_id": "call_1", "content": "7"}


async def test_structured_output_is_the_native_response_format(harness: Harness) -> None:
    model = ScriptedChat([("Answer", {"sku": "a", "units": 7})])
    agent = harness.wrap(
        ReAct(system_prompt="s", model=model, response_format=Answer), id="structured"
    )
    result = await agent.run("a?", user="u1")
    assert result.answer == Answer(sku="a", units=7)
    assert "Answer" in names(model.requests[0])
    record = await harness.runs.get(result.run_id)
    assert record is not None and record.output == {"sku": "a", "units": 7}


async def test_at_the_step_limit_a_structured_answer_is_the_only_tool(harness: Harness) -> None:
    model = ScriptedChat([("stock", {"sku": "a"}), ("Answer", {"sku": "a", "units": 7})])
    target = ReAct(system_prompt="s", model=model, response_format=Answer, max_steps=1)
    result = await arm(harness.wrap(target, id="last"), [stock]).run("a?", user="u1")
    assert result.answer == Answer(sku="a", units=7)
    assert names(model.requests[-1]) == ["Answer"]  # its own tool, no other


async def test_an_unknown_tool_is_told_to_the_model(harness: Harness) -> None:
    model = ScriptedChat([("nope", {}), "sorry"])
    agent = harness.wrap(ReAct(system_prompt="s", model=model), id="unknown")
    assert (await agent.run("x", user="u1")).answer == "sorry"
    assert "nope is not a valid tool" in model.requests[1]["messages"][-1]["content"]


async def test_a_loop_that_never_answers_is_stopped(harness: Harness) -> None:
    # asked once more without tools at the step limit, the model still calls a tool
    model = ScriptedChat([*[("stock", {"sku": sku}) for sku in "abc"], ("stock", {"sku": "d"})])
    agent = arm(
        harness.wrap(ReAct(system_prompt="s", model=model, max_steps=3), id="loop"), [stock]
    )
    result = await agent.run("x", user="u1")
    assert result.status is RunStatus.ERROR and result.error is not None
    assert result.error.message == "stopped after 3 model calls without an answer"
    assert model.requests[-1]["tool_choice"] == "none"


@respx.mock
async def test_a_model_name_is_a_gateway_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_BASE_URL", "http://gw.test/v1")
    monkeypatch.setenv("LLM_API_KEY", "vk")
    monkeypatch.setenv("LLM_GATEWAY", "bifrost")
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
    graph = ReAct(system_prompt="s", model="provider/model", context_window=50_000)
    assert context_window(graph) == 50_000
    async with Harness(
        config=Settings(llm_gateway="bifrost", llm_base_url="http://gw.test/v1", llm_api_key="vk")
    ) as h:
        agent = h.wrap(graph, id="named")
        result = await agent.run("x", user="u")
    assert result.answer == "hi"
    request = route.calls[0].request
    assert request.headers["authorization"] == "Bearer vk"
    assert request.headers["x-bf-mcp-include-clients"] == ""  # the gateway adds no MCP tools
    assert request.headers["x-bf-mcp-include-tools"] == ""
    assert json.loads(request.content)["model"] == "provider/model"


def test_each_event_loop_has_a_client_of_its_own(monkeypatch: pytest.MonkeyPatch) -> None:
    """langchain-openai's default client is one per gateway URL for the whole process: its
    pooled connections would break a graph run in a later event loop. A routed name's chat
    model is made once per route and event loop."""
    import openai

    from trellis.harness.gateways import routed_chat_model

    monkeypatch.setenv("LLM_BASE_URL", "http://gw.test/v1")
    monkeypatch.setenv("LLM_GATEWAY", "bifrost")
    made: list[httpx.AsyncClient] = []

    def client(**params: Any) -> httpx.AsyncClient:
        made.append(httpx.AsyncClient(**params))
        return made[-1]

    monkeypatch.setattr(openai, "DefaultAsyncHttpxClient", client)
    model = routed_chat_model("provider/model")

    async def twice() -> tuple[Any, Any]:
        return model.resolved(), model.resolved()

    first, again = asyncio.run(twice())
    other, _ = asyncio.run(twice())
    assert first is again and first is not other and len(made) == 2


async def test_a_name_whose_provider_cannot_be_told_fails_the_run_saying_why(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    graph = ReAct(system_prompt="s", model="granite")  # built: the route is the run's to choose
    async with Harness(config=Settings()) as h:
        result = await h.wrap(graph, id="unnamed").run("x", user="u")
    assert (
        result.error is not None and "name its provider ('openai:granite'" in result.error.message
    )


def test_what_react_refuses() -> None:
    model = ScriptedChat([])
    with pytest.raises(ConfigurationError, match="call_timeout"):
        ReAct(system_prompt="s", model=model, call_timeout=0)
    with pytest.raises(ConfigurationError, match="context_window"):
        ReAct(system_prompt="s", model=model, context_window=0)


@pytest.mark.parametrize(
    ("name", "said"),
    [
        ("checkpointer", r"keeps its checkpoint in the run; .*create_agent\(.*checkpointer="),
        ("store", r"store is the memory service; .*create_agent\(.*store="),
        ("capabilities", r"can and cannot do in system_prompt"),
        ("tool_guidance", r"tool guidance was removed.*in system_prompt and in each tool"),
        ("answer_check", r"answer check was removed.*h\.evaluate \(tool_coverage\)"),
    ],
)
def test_reacts_removed_arguments_are_refused_naming_what_to_use(name: str, said: str) -> None:
    given: dict[str, Any] = {name: object()}
    with pytest.raises(TypeError, match=rf"ReAct\({name}=\) was removed: .*{said}"):
        ReAct(system_prompt="s", model=ScriptedChat([]), **given)


def test_a_react_graph_checkpoints_into_the_run_and_has_no_store_of_its_own(
    harness: Harness,
) -> None:
    graph = ReAct(system_prompt="s", model=ScriptedChat([]))
    assert isinstance(graph.checkpointer, ours.RunCheckpointer) and graph.store is None
    harness.wrap(graph, id="run")


def test_subagents_are_deep_agents_own_wrapped_agents_and_a2a(harness: Harness) -> None:
    with pytest.raises(ConfigurationError, match="not a ScriptedChat"):
        ReAct(system_prompt="s", model=ScriptedChat([]), subagents=[ScriptedChat([])])
    graph = ReAct(system_prompt="s", model=ScriptedChat([]))
    with pytest.raises(ConfigurationError, match="CompiledSubAgent"):
        ReAct(system_prompt="s", model=ScriptedChat([]), subagents=[graph])


def test_the_window_is_the_models_profile_else_the_default() -> None:
    profiled = ScriptedChat([], profile={"max_input_tokens": 32_000})
    assert context_window(ReAct(system_prompt="s", model=profiled)) == 32_000
    unknown = ReAct(system_prompt="s", model=ScriptedChat([]))
    assert context_window(unknown) is None  # assumed CONTEXT_WINDOW: not said to the harness


# --------------------------------------------------------------------------- the stack
class Named(AgentMiddleware):
    def __init__(self, name: str) -> None:
        super().__init__()
        self._name = name

    @property
    def name(self) -> str:
        return self._name


def test_given_middleware_is_added_before_model_hooks_and_replaces_nothing() -> None:
    defaults = [Named("A"), Named("StepLimit"), Named("ModelHooks")]
    extra, more = Named("C"), Named("D")
    stack = stacked(defaults, [extra, more])
    assert [m.name for m in stack] == ["A", "StepLimit", "C", "D", "ModelHooks"]
    assert stacked(defaults, []) == defaults
    with pytest.raises(ConfigurationError, match=r"StepLimit is the harness's \(max_steps=\)"):
        stacked(defaults, [Named("StepLimit")])
    with pytest.raises(ConfigurationError, match=r"A is the harness's \(always on\)"):
        stacked(defaults, [Named("A")])


def test_reacts_middleware_refuses_the_harness_part_saying_react_has_it() -> None:
    model = ScriptedChat(["hi"])
    with pytest.raises(ConfigurationError, match=r"got middleware\(\): .*ReAct already has it"):
        ReAct(system_prompt="s", model=model, middleware=middleware())
    # planning is LangChain's TodoListMiddleware, by its name: a team's own is refused, not
    # put in its place (ADR 0003)
    with pytest.raises(ConfigurationError, match=r"TodoListMiddleware is the harness's"):
        ReAct(system_prompt="s", model=model, middleware=[TodoListMiddleware()])


async def test_a_deep_agents_middleware_given_replaces_its_own_by_name() -> None:
    model = ScriptedChat(["hi"])
    mine = SummarizationMiddleware(model, trigger=("messages", 100))
    graph = ReAct(system_prompt="s", model=model, middleware=[mine])
    assert "SummarizationMiddleware.before_model" in graph.nodes  # LangChain's, in its place
    with pytest.raises(ConfigurationError, match=r"StepLimit is the harness's \(max_steps=\)"):
        ReAct(system_prompt="s", model=model, middleware=[ours.StepLimit(3)])


@pytest.mark.parametrize("value", [0, -1, True, 2.5, "3", None])
@pytest.mark.parametrize(("name", "least"), [("max_steps", 1), ("max_repeats", 2)])
def test_steps_and_repeats_are_whole_numbers(name: str, least: int, value: Any) -> None:
    with pytest.raises(ConfigurationError, match=f"{name} is a number of steps, {least} or more"):
        ReAct(system_prompt="s", model=ScriptedChat(["hi"]), **{name: value})


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
    graph = ReAct(
        system_prompt="s",
        model=model,
        middleware=[Mine()],
        context_window=128_000,
        backend=JournalBackend(),  # on Deep Agents
    )
    await arm(harness.wrap(graph, id="order"), [stock]).run("x", user="u")
    # Deep Agents' files first, the harness's next; Deep Agents' summarization after what
    # narrows the tools, so it counts what is sent
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
    agent = arm(harness.wrap(ReAct(system_prompt="s", model=model), id="refunds"), [stock, refund])
    paused = await agent.run("refund o1", user="u")
    assert paused.interrupt is not None and len(model.requests) == 2
    done = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="r")
    assert done.answer == "refunded"
    assert len(model.requests) == 3  # only the step after the approval asked the model


async def test_framework_options_are_the_graphs_run_config(harness: Harness) -> None:
    model = ScriptedChat([("stock", {"sku": "a"}), ("stock", {"sku": "b"}), "done"])
    graph = ReAct(system_prompt="s", model=model)
    agent = arm(harness.wrap(graph, id="opts", framework_options={"recursion_limit": 3}), [stock])
    result = await agent.run("x", user="u")
    assert result.status is RunStatus.ERROR and result.error is not None
    assert "recursion" in result.error.message.lower()
    with pytest.raises(ConfigurationError, match="RunnableConfig"):
        harness.wrap(ReAct(system_prompt="s", model=model), id="bad", framework_options={"temp": 1})


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
    agent = arm(harness.wrap(ReAct(system_prompt="s", model=model), id="bad-json"), [stock])
    result = await agent.run("a?", user="u")
    assert result.status is RunStatus.SUCCESS and result.answer == "7"
    sent = model.requests[1]["messages"]
    assert sent[-1]["content"].startswith("stock was not run: its arguments are not valid JSON")
    assert sent[-1]["content"].endswith("Call it again with arguments that fit its schema.")
    # the broken arguments are not sent back: a strict server refuses such a history
    assert sent[-2]["tool_calls"][0]["function"]["arguments"] == "{}"


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
    agent = arm(harness.wrap(ReAct(system_prompt="s", model=model), id="schema"), [order])
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
    result = await arm(
        harness.wrap(ReAct(system_prompt="s", model=model, backend=JournalBackend()), id="dumper"),
        [dump],
    ).run("x", user="u")
    assert result.status is RunStatus.SUCCESS, result.error
    told = model.requests[1]["messages"][-1]["content"]
    assert "saved in the filesystem at this path: /large_tool_results/call_1" in told
    assert len(told) < 10_000  # a preview, not the result
    assert "0123456789" in model.requests[2]["messages"][-1]["content"]
    offered = [[t["function"]["name"] for t in r.get("tools", [])] for r in model.requests]
    once_saved = ["dump", "query_result", "read_file"]  # query_result: SQL over its rows
    assert offered == [["dump"], once_saved, once_saved]


async def test_the_same_call_over_and_over_is_a_stall(harness: Harness) -> None:
    calls: list[str] = []

    @tool(side_effects="read")
    def poll(job: str) -> str:
        """A job's status."""
        calls.append(job)
        return "pending"

    model = ScriptedChat([("poll", {"job": "j"})] * 5)
    agent = arm(harness.wrap(ReAct(system_prompt="s", model=model), id="stall"), [poll])
    result = await agent.run("x", user="u")
    assert result.status is RunStatus.ERROR and result.error is not None
    assert "called 'poll' with the same arguments 3 times in a row" in result.error.message
    assert calls == ["j", "j"]  # the third was not run


async def test_a_call_that_changes_resets_the_stall_count(harness: Harness) -> None:
    turns: list[Any] = [("stock", {"sku": s}) for s in "aabaab"]
    model = ScriptedChat([*turns, "done"])
    target = ReAct(system_prompt="s", model=model, max_repeats=3)
    result = await arm(harness.wrap(target, id="no-stall"), [stock]).run("x", user="u")
    assert result.answer == "done"


def test_a_measurement_leaves_default_middleware_out_only_inside_defaults_off() -> None:
    defaults = [Named("A"), Named("SummarizationMiddleware"), Named("ModelHooks")]
    with defaults_off("SummarizationMiddleware", "ModelHooks"):  # the last is always kept
        assert [m.name for m in stacked(defaults, [])] == ["A", "ModelHooks"]
    assert [m.name for m in stacked(defaults, [])] == ["A", "SummarizationMiddleware", "ModelHooks"]
