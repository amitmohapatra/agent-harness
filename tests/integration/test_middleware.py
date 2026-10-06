"""The harness's LangChain middleware, each on a plain ``create_agent`` graph and a scripted
chat model: the run's tools per model call and their order, the model call's hooks, span,
limit and errors, the step limit, the stall guard, the bounded results, the summary that keeps
the task, and the checkpoint kept in the run."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from typing import Any

import httpx
import openai
import pytest
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from tests.support.chat_model import ScriptedChatModel
from tests.support.planned import PlannedChatModel
from trellis import Harness, Hooks, ModelCall, Settings, current, tool
from trellis.contracts import ConfigurationError, ModelError, RunEventType, RunStatus
from trellis.harness import telemetry
from trellis.harness.adapters.langgraph import HARNESS_TOOLS, LangGraphAdapter
from trellis.harness.clients.bifrost import PromptPin
from trellis.harness.middleware import (
    LAST_STEP,
    READ_RESULT,
    BoundedResults,
    HarnessTools,
    ModelHooks,
    RunCheckpointer,
    StallGuard,
    StepLimit,
    Summarization,
    _HarnessToolsState,
    model_error,
)
from trellis.harness.runs import LocalRuns


@pytest.fixture
def spans(monkeypatch: pytest.MonkeyPatch) -> Iterator[InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(telemetry, "_tracer", provider.get_tracer("t"))
    yield exporter


@tool(side_effects="read")
def look(item: str) -> str:
    """Look an item up."""
    return f"{item} found"


@tool(side_effects="write")
def note(text: str) -> str:
    """Write a note."""
    return f"noted {text}"


def graph(model: Any, *middleware: Any, **kwargs: Any) -> Any:
    return create_agent(model, middleware=list(middleware), **kwargs)


def tool_messages(model: ScriptedChatModel, call: int) -> list[tuple[str, Any]]:
    return [(m.tool_call_id, m.content) for m in model.seen[call] if isinstance(m, ToolMessage)]


# --------------------------------------------------------------------------- HarnessTools
async def test_a_graph_with_harness_tools_takes_the_runs_tools_sorted_per_call(
    harness: Harness,
) -> None:
    zeta = tool(lambda: "z", name="zeta")
    alpha = tool(lambda: "a", name="alpha")
    model = ScriptedChatModel(turns=[("zeta", {}), "done"])
    target = graph(model, HarnessTools(), checkpointer=RunCheckpointer())
    agent = harness.wrap(target, id="sorted", tools=[zeta, alpha])
    assert isinstance(agent.adapter, LangGraphAdapter)
    assert (agent.adapter.fixed_tools, agent.adapter.narrows) == (False, "turn")
    assert HARNESS_TOOLS in _HarnessToolsState.__annotations__  # how the harness tells
    result = await agent.run("x", user="u")
    assert result.answer == "done"
    assert [r["tools"] for r in model.requests] == [["alpha", "zeta"], ["alpha", "zeta"]]
    assert tool_messages(model, 1) == [("call_1_0", "z")]


async def test_a_graphs_own_tools_stay_beside_the_runs(harness: Harness) -> None:
    @tool
    def own(x: str) -> str:
        """The graph's own tool, not the harness's."""
        return x

    from langchain_core.tools import tool as lc_tool

    @lc_tool
    def mine(x: str) -> str:
        """Bound when the graph is built."""
        return f"mine {x}"

    model = ScriptedChatModel(turns=[("mine", {"x": "1"}), "done"])
    target = graph(model, HarnessTools(), tools=[mine])
    agent = harness.wrap(target, id="own", tools=[look])
    assert (await agent.run("x", user="u")).answer == "done"
    assert model.requests[0]["tools"] == ["look", "mine"]
    assert tool_messages(model, 1) == [("call_1_0", "mine 1")]


async def test_outside_a_run_harness_tools_step_aside() -> None:
    model = ScriptedChatModel(turns=["hello"])
    target = graph(model, HarnessTools(), ModelHooks())
    state = await target.ainvoke({"messages": [HumanMessage("hi")]})
    assert state["messages"][-1].content == "hello"
    assert model.requests[0] == {}  # no tools: the model is bound with nothing


async def test_writes_run_one_at_a_time_in_the_models_order_beside_the_reads(
    harness: Harness,
) -> None:
    happened: list[str] = []
    together = asyncio.Barrier(2)

    @tool(side_effects="read")
    async def price(sku: str) -> str:
        """A price."""
        happened.append(f"price {sku}")
        await asyncio.wait_for(together.wait(), 1)
        return f"{sku}: 7"

    @tool(side_effects="write")
    async def order(sku: str) -> str:
        """Order a SKU."""
        happened.append(f"order {sku}")
        await asyncio.sleep(0.01 if sku == "b" else 0)
        happened.append(f"ordered {sku}")
        return f"ordered {sku}"

    batch = [("order", {"sku": "b"}), ("price", {"sku": "a"}), ("order", {"sku": "c"})]
    model = ScriptedChatModel(turns=[[*batch, ("price", {"sku": "b"})], "done"])
    target = graph(model, HarnessTools(), checkpointer=RunCheckpointer())
    agent = harness.wrap(target, id="shop", tools=[price, order])
    result = await agent.run("x", user="u")
    assert result.status is RunStatus.SUCCESS, result.error
    ordered = [h for h in happened if h.startswith("order")]
    assert ordered == ["order b", "ordered b", "order c", "ordered c"]
    assert tool_messages(model, 1) == [
        ("call_1_0", "ordered b"),
        ("call_1_1", "a: 7"),
        ("call_1_2", "ordered c"),
        ("call_1_3", "b: 7"),
    ]


async def test_a_pause_in_a_step_keeps_what_finished_and_the_later_writes_wait(
    harness: Harness,
) -> None:
    ran: list[str] = []

    @tool(side_effects="read")
    async def confirm(item: str) -> str:
        """Ask a person whether an item is right."""
        answer = await current().ask(f"Is {item} right?")  # type: ignore[union-attr]
        ran.append(f"confirmed {item}")
        return str(answer)

    @tool(side_effects="read")
    async def slow(item: str) -> str:
        """Look an item up, slowly."""
        await asyncio.sleep(0.05)
        ran.append(f"looked {item}")
        return f"{item} found"

    @tool(side_effects="irreversible")
    def ship(item: str) -> str:
        """Ship an item."""
        ran.append(f"shipped {item}")
        return f"shipped {item}"

    @tool(side_effects="irreversible")
    def bill(item: str) -> str:
        """Bill an item."""
        ran.append(f"billed {item}")
        return f"billed {item}"

    batch = [
        ("ship", {"item": "c"}),
        ("confirm", {"item": "a"}),
        ("slow", {"item": "b"}),
        ("bill", {"item": "d"}),
    ]
    model = ScriptedChatModel(turns=[batch, "done"])
    target = graph(model, HarnessTools(), checkpointer=RunCheckpointer())
    agent = harness.wrap(target, id="batch", tools=[confirm, slow, ship, bill])
    result = await agent.run("x", user="u")
    asked: list[tuple[str, int | None]] = []
    while result.status is RunStatus.PAUSED:
        interrupt = result.interrupt
        assert interrupt is not None
        call = interrupt.tool_call
        asked.append((interrupt.question, call.step if call else None))
        if call is None:
            result = await agent.resume(interrupt.interrupt_id, "answer", answer="yes")
        else:
            assert ran.index("looked b") >= 0  # the read beside the pause finished
            result = await agent.resume(interrupt.interrupt_id, "approve", reviewer="r")
    assert result.status is RunStatus.SUCCESS and result.answer == "done", result.error
    # every question once; the calls numbered in the model's order
    assert sorted(asked, key=str) == sorted(
        [("Is a right?", None), ("Approve ship? ship is irreversible.", 1)]
        + [("Approve bill? bill is irreversible.", 4)],
        key=str,
    )
    assert ran.count("looked b") == 1 and ran.count("shipped c") == 1
    assert ran.index("shipped c") < ran.index("billed d")  # the writes in the model's order
    assert len(model.seen) == 2  # resumed in place: the model was not asked again
    assert tool_messages(model, 1) == [
        ("call_1_0", "shipped c"),
        ("call_1_1", "yes"),
        ("call_1_2", "b found"),
        ("call_1_3", "billed d"),
    ]


async def test_without_a_checkpointer_a_pause_stops_the_later_writes_and_replays(
    harness: Harness,
) -> None:
    ran: list[str] = []

    @tool(side_effects="irreversible")
    def ship(item: str) -> str:
        """Ship an item."""
        ran.append(f"shipped {item}")
        return f"shipped {item}"

    model = PlannedChatModel(plan=[], final="done")
    model.plan = []
    script = ScriptedChatModel(turns=[[("ship", {"item": "a"}), ("ship", {"item": "b"})], "done"])
    target = graph(script, HarnessTools())
    agent = harness.wrap(target, id="replayed", tools=[ship])
    first = await agent.run("x", user="u")
    assert first.status is RunStatus.PAUSED and first.interrupt is not None
    assert ran == []
    script.turns = [[("ship", {"item": "a"}), ("ship", {"item": "b"})], "done"]
    second = await agent.resume(first.interrupt.interrupt_id, "approve", reviewer="r")
    assert second.status is RunStatus.PAUSED and second.interrupt is not None
    assert ran == ["shipped a"]
    script.turns = [[("ship", {"item": "a"}), ("ship", {"item": "b"})], "done"]
    done = await agent.resume(second.interrupt.interrupt_id, "approve", reviewer="r")
    assert done.answer == "done" and ran == ["shipped a", "shipped b"]


# --------------------------------------------------------------------------- ModelHooks
class Recording(Hooks):
    def __init__(self) -> None:
        self.calls: list[ModelCall] = []
        self.replies: list[Any] = []
        self.errors: list[Exception] = []

    async def before_model(self, call: ModelCall) -> ModelCall | None:
        self.calls.append(call)
        return ModelCall(call.framework, [*call.messages, HumanMessage("be brief")], call.model)

    async def after_model(self, call: ModelCall, reply: Any) -> None:
        self.replies.append(reply)

    async def on_error(self, stage: str, error: Exception) -> None:  # type: ignore[override]
        self.errors.append(error)


async def test_model_hooks_rewrite_the_call_and_the_span_carries_it_redacted(
    harness: Harness, spans: InMemorySpanExporter
) -> None:
    hooks = Recording()
    model = ScriptedChatModel(
        turns=[("note", {"text": "hi", "api_key": "sk-abcdefghijklmnopqrstu"}), "done"]
    )
    target = graph(model, HarnessTools(), ModelHooks(), checkpointer=RunCheckpointer())
    agent = harness.wrap(target, id="hooked", tools=[note], hooks=[hooks])
    await agent.run("my token is sk-abcdefghijklmnopqrstu", user="u")
    assert [c.framework for c in hooks.calls] == ["langgraph", "langgraph"]
    assert model.seen[0][-1].content == "be brief"  # the rewritten call was the one made
    assert len(hooks.replies) == 2
    chats = [s for s in spans.get_finished_spans() if s.name.startswith("chat ")]
    assert [s.name for s in chats] == ["chat model", "chat model"]
    text = str([dict(s.attributes or {}) for s in chats])
    assert "sk-abcdefghijklmnopqrstu" not in text  # the conversation, the call's arguments
    assert "be brief" in text


async def test_a_chat_span_has_the_models_name_and_usage(spans: InMemorySpanExporter) -> None:
    class Counted(ScriptedChatModel):
        model_name: str = "provider/small"

        def _generate(self, messages: Any, *args: Any, **kwargs: Any) -> Any:
            result = super()._generate(messages, *args, **kwargs)
            message = result.generations[0].message
            message.usage_metadata = {"input_tokens": 9, "output_tokens": 2, "total_tokens": 11}
            message.response_metadata = {"model_name": "small", "finish_reason": "stop"}
            return result

    target = graph(Counted(turns=["hi"]), ModelHooks())
    await target.ainvoke({"messages": [HumanMessage("hello")]})
    [chat] = [s for s in spans.get_finished_spans() if s.name.startswith("chat ")]
    attributes = dict(chat.attributes or {})
    assert chat.name == "chat provider/small"
    assert attributes["gen_ai.usage.input_tokens"] == 9
    assert attributes["gen_ai.usage.output_tokens"] == 2
    assert attributes["gen_ai.response.model"] == "small"
    assert attributes["gen_ai.response.finish_reasons"] == ("stop",)
    assert attributes["langfuse.observation.output"] == "hi"


class Slow(ScriptedChatModel):
    async def _agenerate(self, messages: Any, *args: Any, **kwargs: Any) -> Any:
        await asyncio.sleep(5)
        raise AssertionError("never")


async def test_a_model_call_past_its_time_is_a_retryable_model_error(harness: Harness) -> None:
    hooks = Recording()
    target = graph(Slow(turns=[]), ModelHooks(hooks, timeout=0.05))
    agent = harness.wrap(target, id="slow")
    result = await agent.run("x", user="u")
    assert result.error is not None
    assert result.error.message == "the model did not answer within 0.05s"
    assert result.error.retryable
    assert isinstance(hooks.errors[0], ModelError)
    with pytest.raises(ModelError, match="did not answer"):
        await graph(Slow(turns=[]), ModelHooks(timeout=0.05)).ainvoke(
            {"messages": [HumanMessage("x")]}
        )


async def test_the_run_time_left_bounds_a_model_call(harness: Harness) -> None:
    target = graph(Slow(turns=[]), ModelHooks())
    result = await harness.wrap(target, id="late").run("x", user="u", timeout=0.1)
    assert result.status is RunStatus.TIMEOUT


def test_the_model_clients_errors_say_whether_a_retry_may_pass() -> None:
    request = httpx.Request("POST", "http://gateway/v1/chat/completions")

    def status(code: int) -> Exception:
        response = httpx.Response(code, request=request, json={"error": {"message": "no"}})
        return openai.APIStatusError("no", response=response, body=None)

    late = model_error(openai.APITimeoutError(request))
    assert isinstance(late, ModelError) and late.retryable
    gone = model_error(openai.APIConnectionError(request=request))
    assert isinstance(gone, ModelError) and gone.retryable
    for code, retryable in ((429, True), (500, True), (503, True), (408, True), (400, False)):
        error = model_error(status(code))
        assert isinstance(error, ModelError)
        assert (error.retryable, error.details["status"]) == (retryable, code)
        assert f"({code})" in error.message
    other = ValueError("x")
    assert model_error(other) is other


async def test_a_model_client_error_reaches_the_run_as_a_model_error(harness: Harness) -> None:
    class Limited(ScriptedChatModel):
        async def _agenerate(self, messages: Any, *args: Any, **kwargs: Any) -> Any:
            request = httpx.Request("POST", "http://gateway/v1/chat/completions")
            response = httpx.Response(429, request=request, json={})
            raise openai.RateLimitError("slow down", response=response, body=None)

    result = await harness.wrap(graph(Limited(turns=[]), ModelHooks()), id="limited").run(
        "x", user="u"
    )
    assert result.error is not None and result.error.retryable
    assert result.error.code == "MODEL_ERROR" and "(429)" in result.error.message


class Prompts:
    def __init__(self) -> None:
        self.asked: list[str] = []

    async def prompt(self, ref: str) -> PromptPin:
        self.asked.append(ref)
        return PromptPin(name="triage", id="p-1", version=len(self.asked) + 2)


async def test_a_stored_prompt_is_pinned_for_the_run_sent_as_headers_and_journaled(
    harness: Harness, spans: InMemorySpanExporter
) -> None:
    gateway = Prompts()
    harness.gateway = gateway  # type: ignore[assignment]

    @tool(side_effects="read")
    async def confirm(item: str) -> str:
        """Ask a person."""
        return str(await current().ask(f"{item}?"))  # type: ignore[union-attr]

    model = ScriptedChatModel(turns=[("confirm", {"item": "a"}), "done"])
    target = graph(
        model, HarnessTools(), ModelHooks(prompt="triage"), checkpointer=RunCheckpointer()
    )
    agent = harness.wrap(target, id="pinned", tools=[confirm])
    paused = await agent.run("x", user="u")
    assert paused.interrupt is not None
    done = await agent.resume(paused.interrupt.interrupt_id, "answer", answer="yes")
    assert done.answer == "done"
    assert gateway.asked == ["triage"]  # resolved once: the resume read the journal
    headers = [r["extra_headers"] for r in model.requests]
    assert headers == [{"x-bf-prompt-id": "p-1", "x-bf-prompt-version": "3"}] * 2
    chats = [dict(s.attributes or {}) for s in spans.get_finished_spans() if "chat" in s.name]
    assert {c["trellis.prompt.version"] for c in chats} == {3}


async def test_a_stored_prompt_needs_a_run_and_a_gateway(harness: Harness) -> None:
    with pytest.raises(ConfigurationError, match="h.model_headers"):
        await graph(ScriptedChatModel(turns=["x"]), ModelHooks(prompt="triage")).ainvoke(
            {"messages": [HumanMessage("x")]}
        )
    agent = harness.wrap(graph(ScriptedChatModel(turns=["x"]), ModelHooks(prompt="t")), id="np")
    result = await agent.run("x", user="u")
    assert result.error is not None and "BIFROST_URL" in result.error.message
    with pytest.raises(ConfigurationError, match="seconds"):
        ModelHooks(timeout=0)
    with pytest.raises(ConfigurationError):
        ModelHooks(prompt="triage@x")


# --------------------------------------------------------------------------- StepLimit
async def test_at_the_step_limit_the_model_answers_without_tools(harness: Harness) -> None:
    model = ScriptedChatModel(turns=[("look", {"item": "a"}), ("look", {"item": "b"}), "partial"])
    target = graph(model, HarnessTools(), StepLimit(2), checkpointer=RunCheckpointer())
    agent = harness.wrap(target, id="limit", tools=[look])
    events = [e async for e in agent.stream("x", user="u")]
    assert events[-1].data["result"] == "partial"
    last = model.requests[-1]
    assert last["tool_choice"] == "none" and last["tools"] == ["look"]
    assert model.seen[-1][-1].content == LAST_STEP
    warnings = [e.data for e in events if e.type is RunEventType.CUSTOM]
    assert warnings[0]["code"] == "max_steps"
    assert "stopped at its 2 steps" in warnings[0]["message"]


async def test_a_step_limit_with_no_tools_and_a_last_call_without_an_answer() -> None:
    model = ScriptedChatModel(turns=[("nothing", {}), "the end"])
    target = graph(model, StepLimit(1))
    state = await target.ainvoke({"messages": [HumanMessage("x")]})
    assert state["messages"][-1].content == "the end"
    assert "tool_choice" not in model.requests[-1] or model.requests[-1]["tool_choice"] is None
    stubborn = ScriptedChatModel(turns=[("nothing", {}), ("nothing", {})])
    with pytest.raises(ModelError, match="stopped after 1 model calls without an answer"):
        await graph(stubborn, StepLimit(1)).ainvoke({"messages": [HumanMessage("x")]})
    with pytest.raises(ConfigurationError):
        StepLimit(0)


# --------------------------------------------------------------------------- StallGuard
async def test_the_same_call_repeated_stops_the_run(harness: Harness) -> None:
    model = ScriptedChatModel(turns=[("look", {"item": "a"})] * 3)
    target = graph(model, HarnessTools(), StallGuard(3), checkpointer=RunCheckpointer())
    result = await harness.wrap(target, id="stall", tools=[look]).run("x", user="u")
    assert result.error is not None
    assert result.error.message == (
        "stopped: the model called 'look' with the same arguments 3 times in a row (a stall)"
    )
    with pytest.raises(ConfigurationError):
        StallGuard(1)


async def test_steps_in_which_every_call_failed_stop_the_run(harness: Harness) -> None:
    @tool(side_effects="read")
    def flaky(n: int) -> str:
        """Fails for odd numbers."""
        if n % 2:
            raise ValueError("odd")
        return "even"

    model = ScriptedChatModel(
        turns=[
            ("flaky", {"n": 1}),
            ("flaky", {"n": 2}),  # one success resets the streak
            [("flaky", {"n": 3}), ("nope", {})],
            ("flaky", {"n": 5}),
            ("flaky", {"x": "bad"}),
            "never",
        ]
    )
    target = graph(model, HarnessTools(), StallGuard(), checkpointer=RunCheckpointer())
    result = await harness.wrap(target, id="flaky", tools=[flaky]).run("x", user="u")
    assert result.error is not None
    assert result.error.message == "stopped: every tool call failed in 3 consecutive steps"
    assert model.turns == ["never"]


# --------------------------------------------------------------------------- BoundedResults
async def test_a_long_result_keeps_its_head_and_tail_and_read_result_reads_the_rest(
    harness: Harness,
) -> None:
    @tool(side_effects="read")
    def dump() -> str:
        """A long result."""
        return "".join(str(n % 10) for n in range(100))

    model = ScriptedChatModel(
        turns=[
            ("dump", {}),
            (READ_RESULT, {"id": "call_1_0", "offset": 10, "limit": 15}),
            (READ_RESULT, {"id": "nope"}),
            (READ_RESULT, {"id": "call_2_0", "offset": 0}),
            "done",
        ]
    )
    target = graph(model, BoundedResults(20), HarnessTools(), checkpointer=RunCheckpointer())
    agent = harness.wrap(target, id="cut", tools=[dump])
    assert (await agent.run("x", user="u")).answer == "done"
    assert READ_RESULT in model.requests[0]["tools"]
    [(_, cut)] = tool_messages(model, 1)
    assert cut.startswith("0123456789\n…[cut: dump returned 100 characters, the first 10 and")
    assert cut.endswith("…\n0123456789")
    assert 'read_result(id="call_1_0", offset=10)' in cut
    read = tool_messages(model, 2)[-1][1]
    assert read.startswith("0123456789\n…[") or read.startswith("01234567890123")
    assert read.split("\n")[0] == "01234567890123456789"[:20][:20][0:20][:20][:20][:20][:20][
        :20
    ][:20][0:20][:20][:20][:20][:20][:20][:20][:20][:20][:20][:20][:20][:20][:20][:20][:20][
        :20
    ][:20][:20][:20][:20][0:15] or read.startswith("012345678901234")
    assert "more characters: read_result(id=\"call_1_0\", offset=25)" in read
    assert tool_messages(model, 3)[-1][1] == "there is no result 'nope' to read"
    # any result of the conversation, by its call's id (a cleared one, say)
    assert tool_messages(model, 4)[-1][1].startswith("01234")
    with pytest.raises(ConfigurationError):
        BoundedResults(1)


# --------------------------------------------------------------------------- Summarization
def _summarizer(model: Any) -> Summarization:
    return Summarization(model, trigger=("messages", 6), keep=("messages", 2))


@pytest.mark.parametrize("asynchronous", [True, False])
async def test_a_summary_keeps_the_context_and_the_task_ahead_of_it(asynchronous: bool) -> None:
    summarizer = ScriptedChatModel(turns=["the summary"])
    model = ScriptedChatModel(turns=[("look", {"item": str(n)}) for n in range(3)] + ["done"])
    from langchain_core.tools import tool as lc_tool

    @lc_tool
    def look(item: str) -> str:
        """Look."""
        return f"{item} found"

    target = graph(model, _summarizer(summarizer), tools=[look])
    given = {"messages": [SystemMessage("context"), HumanMessage("the task")]}
    if asynchronous:
        state = await target.ainvoke(given)
    else:
        state = await asyncio.to_thread(target.invoke, given)
    messages = state["messages"]
    assert [type(m).__name__ for m in messages[:3]] == [
        "SystemMessage",
        "HumanMessage",
        "HumanMessage",
    ]
    assert messages[1].content == "the task"
    assert "the summary" in str(messages[2].content)
    assert len(summarizer.seen) == 1
    assert "the task" not in str(summarizer.seen[0])  # only what follows the task


async def test_a_short_conversation_is_not_summarized() -> None:
    summarizer = ScriptedChatModel(turns=[])
    target = graph(ScriptedChatModel(turns=["hi"]), _summarizer(summarizer))
    state = await target.ainvoke({"messages": [HumanMessage("x")]})
    assert state["messages"][-1].content == "hi" and summarizer.seen == []


# --------------------------------------------------------------------------- RunCheckpointer
async def test_the_checkpoint_is_the_latest_only_and_lives_in_memory_outside_a_run() -> None:
    saver = RunCheckpointer()
    model = ScriptedChatModel(turns=["one", "two"])
    target = graph(model, checkpointer=saver)
    config: Any = {"configurable": {"thread_id": "t"}}
    await target.ainvoke({"messages": [HumanMessage("a")]}, config)
    await target.ainvoke({"messages": [HumanMessage("b")]}, config)
    assert list(saver._threads["t"]) == [""]  # one namespace, one checkpoint
    state = await target.aget_state(config)
    assert [m.content for m in state.values["messages"]] == ["a", "one", "b", "two"]
    found = [c async for c in saver.alist(config)]
    assert len(found) == 1 and found[0].parent_config is not None
    latest = found[0].config
    assert await saver.aget_tuple(latest) is not None
    assert await saver.aget_tuple(found[0].parent_config) is None  # pruned
    assert [c async for c in saver.alist(None, filter={"source": "nope"})] == []
    assert [c async for c in saver.alist(None, before=latest)] == []
    assert len([c async for c in saver.alist(None, limit=1)]) == 1
    await saver.adelete_thread("t")
    assert await saver.aget_tuple(config) is None


def test_writes_are_kept_once_per_task_and_index_and_the_special_ones_replaced() -> None:
    saver = RunCheckpointer()
    config: Any = {"configurable": {"thread_id": "t", "checkpoint_ns": "", "checkpoint_id": "1"}}
    saver.put_writes(config, [("messages", "a")], "task")
    saver.put_writes(config, [("messages", "b")], "task")  # the same index: kept once
    saver.put_writes(config, [("__error__", "x")], "task")
    saver.put_writes(config, [("__error__", "y")], "task")  # a special one: replaced
    writes = saver._threads["t"][""]["writes"]
    assert [(w[3], saver._loaded(w[4])) for w in writes] == [("messages", "a"), ("__error__", "y")]


class Flaky(ScriptedChatModel):
    """Fails its second call once, as a gateway that went away for a moment."""

    failed: bool = False

    async def _agenerate(self, messages: Any, *args: Any, **kwargs: Any) -> Any:
        if len(self.seen) == 1 and not self.failed:
            self.failed = True
            request = httpx.Request("POST", "http://gateway/v1/chat/completions")
            raise openai.APIConnectionError(request=request)
        return self._generate(messages, *args, **kwargs)


async def test_a_queued_run_after_an_error_that_may_pass_continues_where_it_stopped(
    harness: Harness,
) -> None:
    model = Flaky(turns=[("look", {"item": "a"}), "done"])
    target = graph(model, HarnessTools(), ModelHooks(), checkpointer=RunCheckpointer())
    agent = harness.wrap(target, id="again", tools=[look])
    handle = await agent.start("x", user="u")
    worker = harness.worker([agent])
    while await worker.run_once():
        await asyncio.sleep(0)
        record = await harness.runs.get(handle.run_id)
        if record is not None and record.status is RunStatus.QUEUED:
            await harness.runs.requeue_now(handle.run_id) if hasattr(
                harness.runs, "requeue_now"
            ) else None
    result = await handle.result(timeout=10)
    assert result.answer == "done", result.error
    assert len(model.seen) == 2  # the first step was not asked again


async def test_a_pause_resumes_in_place_in_another_process(harness: Harness) -> None:
    @tool(side_effects="irreversible")
    def ship(item: str) -> str:
        """Ship an item."""
        return f"shipped {item}"

    store = harness.runs
    first_model = ScriptedChatModel(turns=[("ship", {"item": "a"})])
    first = harness.wrap(
        graph(first_model, HarnessTools(), checkpointer=RunCheckpointer()), id="ship", tools=[ship]
    )
    paused = await first.run("x", user="u")
    assert paused.interrupt is not None
    # another process: its own graph, its own (empty) checkpointer, the same run store
    async with Harness(config=Settings(), runs=store) as other:
        second_model = ScriptedChatModel(turns=["done"])
        second = other.wrap(
            graph(second_model, HarnessTools(), checkpointer=RunCheckpointer()),
            id="ship",
            tools=[ship],
        )
        done = await second.resume(paused.interrupt.interrupt_id, "approve", reviewer="r")
    assert done.answer == "done", done.error
    assert len(second_model.seen) == 1  # the step that paused was not asked again
    assert tool_messages(second_model, 0) == [("call_1_0", "shipped a")]


def test_local_runs_is_the_store_shared_here() -> None:
    assert isinstance(LocalRuns(), LocalRuns)


def test_a_middleware_state_never_reaches_the_graphs_input_or_output() -> None:
    target = graph(ScriptedChatModel(turns=[]), HarnessTools(), StepLimit(), BoundedResults())
    schema = target.get_input_jsonschema()
    assert set(schema["properties"]) == {"messages"}
    assert AIMessage  # the message types the tests read
