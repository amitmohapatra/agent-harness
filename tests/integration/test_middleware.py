"""The harness's LangChain middleware, each on a plain ``create_agent`` graph and a scripted
chat model: the run's tools per model call and their order, the model call's hooks, span,
limit and errors, the step limit, the stall guard, the bounded results, the summary that keeps
the task, and the checkpoint kept in the run."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from typing import Any, cast

import httpx2
import openai
import pytest
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from tests.support.chat_model import ScriptedChatModel
from trellis import Harness, Hooks, ModelCall, Settings, current, tool
from trellis.contracts import ConfigurationError, ModelError, RunEventType, RunStatus
from trellis.harness import telemetry
from trellis.harness.adapters.langgraph import HARNESS_TOOLS, LangGraphAdapter
from trellis.harness.middleware import (
    LAST_STEP,
    READ_RESULT,
    HarnessTools,
    ModelHooks,
    RunCheckpointer,
    StallGuard,
    StepLimit,
    _HarnessToolsState,
    model_error,
    read_result,
)
from trellis.harness.prompts import PromptSources, ResolvedPrompt


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
            result = await agent.resume(
                interrupt.interrupt_id, "answer", answer="yes", reviewer="r"
            )
        else:
            assert ran.index("looked b") >= 0  # the read beside the pause finished
            result = await agent.resume(interrupt.interrupt_id, "approve", reviewer="r")
    assert result.status is RunStatus.SUCCESS and result.answer == "done", result.error
    # every question once; the calls numbered in the model's order
    assert sorted(asked, key=str) == sorted(
        [
            ("Is a right?", None),
            ("Approve ship? ship is irreversible.", 1),
            ("Approve bill? bill is irreversible.", 4),
        ],
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
    @tool(side_effects="read")
    def echo(text: str, api_key: str = "") -> str:
        """Say it back."""
        return text

    hooks = Recording()
    secret = "sk-abcdefghijklmnopqrstu"
    model = ScriptedChatModel(
        turns=[("echo", {"text": "hi", "api_key": "k1"}), ("echo", {"text": secret}), "done"]
    )
    target = graph(model, HarnessTools(), ModelHooks(), checkpointer=RunCheckpointer())
    agent = harness.wrap(target, id="hooked", tools=[echo], hooks=[hooks])
    await agent.run({"messages": [{"role": "user", "content": "go"}], "password": "pw"}, user="u")
    assert [c.framework for c in hooks.calls] == ["langgraph"] * 3
    assert model.seen[0][-1].content == "be brief"  # the rewritten call was the one made
    assert len(hooks.replies) == 3
    chats = [s for s in spans.get_finished_spans() if s.name.startswith("chat ")]
    assert [s.name for s in chats] == ["chat model"] * 3
    text = str([dict(s.attributes or {}) for s in chats])
    # a call's arguments are kept structured, so the redactor sees each: a secret's name, a
    # credential's value, as the model sent them and as the next call carries them
    assert secret not in text and "k1" not in text
    assert "be brief" in text


async def test_a_chat_span_has_the_models_name_and_usage(spans: InMemorySpanExporter) -> None:
    class Counted(ScriptedChatModel):
        model_name: str = "provider/small"

        def _generate(self, messages: Any, *args: Any, **kwargs: Any) -> Any:
            result = super()._generate(messages, *args, **kwargs)
            message = cast(AIMessage, result.generations[0].message)
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
    request = httpx2.Request("POST", "http://gateway/v1/chat/completions")

    def status(code: int) -> Exception:
        response = httpx2.Response(code, request=request, json={"error": {"message": "no"}})
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
            request = httpx2.Request("POST", "http://gateway/v1/chat/completions")
            response = httpx2.Response(429, request=request, json={})
            raise openai.RateLimitError("slow down", response=response, body=None)

    result = await harness.wrap(graph(Limited(turns=[]), ModelHooks()), id="limited").run(
        "x", user="u"
    )
    assert result.error is not None and result.error.retryable
    assert result.error.code == "MODEL_ERROR" and "(429)" in result.error.message


class Stored:
    """A prompt source holding a stored prompt of the gateway and a text prompt."""

    label = "test"

    def __init__(self) -> None:
        self.asked: list[str] = []

    async def resolve(self, name: str, version: str | None) -> ResolvedPrompt:
        self.asked.append(name)
        if name == "triage":
            return ResolvedPrompt(
                name="triage", version=str(len(self.asked) + 2), source="bifrost", selection="p-1"
            )
        return ResolvedPrompt(
            name=name,
            version="1",
            source="test",
            chat=(
                {"role": "system", "content": "Answer as {{who}}."},
                {"role": "user", "content": "An example."},
            ),
        )


async def test_a_stored_prompt_is_pinned_for_the_run_sent_as_headers_and_journaled(
    harness: Harness, spans: InMemorySpanExporter
) -> None:
    source = Stored()
    harness.prompts = PromptSources([source])

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
    done = await agent.resume(paused.interrupt.interrupt_id, "answer", answer="yes", reviewer="r")
    assert done.answer == "done"
    assert source.asked == ["triage"]  # resolved once: the resume read the journal
    headers = [r["extra_headers"] for r in model.requests]
    assert headers == [{"x-bf-prompt-id": "p-1", "x-bf-prompt-version": "3"}] * 2
    chats = [dict(s.attributes or {}) for s in spans.get_finished_spans() if "chat" in s.name]
    assert {c["trellis.prompt.version"] for c in chats} == {3}


async def test_any_other_prompt_is_rendered_into_the_instructions(harness: Harness) -> None:
    harness.prompts = PromptSources([Stored()])
    model = ScriptedChatModel(turns=["done"])
    hooks = ModelHooks(prompt="persona", prompt_vars={"who": "a pirate"})
    agent = harness.wrap(graph(model, hooks, system_prompt="You sell."), id="persona")
    assert (await agent.run("x", user="u")).answer == "done"
    sent = model.seen[0]
    assert sent[0].content == "Answer as a pirate.\n\nYou sell."
    assert [m.content for m in sent[1:]] == ["An example.", "x"]
    stored = harness.wrap(graph(ScriptedChatModel(turns=["x"]), hooks_of("triage")), id="vars")
    failed = await stored.run("x", user="u")
    assert failed.error is not None and "no prompt_vars" in failed.error.message


def hooks_of(prompt: str) -> ModelHooks:
    return ModelHooks(prompt=prompt, prompt_vars={"who": "x"})


async def test_a_prompt_needs_a_run_and_a_source(harness: Harness) -> None:
    with pytest.raises(ConfigurationError, match=r"h\.model_headers"):
        await graph(ScriptedChatModel(turns=["x"]), ModelHooks(prompt="triage")).ainvoke(
            {"messages": [HumanMessage("x")]}
        )
    agent = harness.wrap(graph(ScriptedChatModel(turns=["x"]), ModelHooks(prompt="t")), id="np")
    result = await agent.run("x", user="u")
    assert result.error is not None and "no prompt source" in result.error.message
    with pytest.raises(ConfigurationError, match="seconds"):
        ModelHooks(timeout=0)
    with pytest.raises(ConfigurationError):
        ModelHooks(prompt="triage@")
    with pytest.raises(ConfigurationError, match="prompt_vars"):
        ModelHooks(prompt_vars={"a": 1})


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
    target = graph(model, HarnessTools(), StepLimit(1))
    state = await target.ainvoke({"messages": [HumanMessage("x")]})
    assert state["messages"][-1].content == "the end"
    assert "tool_choice" not in model.requests[-1] or model.requests[-1]["tool_choice"] is None
    stubborn = ScriptedChatModel(turns=[("nothing", {}), ("nothing", {})])
    with pytest.raises(ModelError, match="stopped after 1 model calls without an answer"):
        await graph(stubborn, HarnessTools(), StepLimit(1)).ainvoke(
            {"messages": [HumanMessage("x")]}
        )
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


# --------------------------------------------------------------------------- read_result
async def test_read_result_reads_a_result_of_the_conversation_in_parts(harness: Harness) -> None:
    @tool(side_effects="read")
    def dump() -> str:
        """A long result."""
        return "".join(str(n % 10) for n in range(100))

    model = ScriptedChatModel(
        turns=[
            ("dump", {}),
            (READ_RESULT, {"id": "call_1_0", "offset": 10, "limit": 15}),
            (READ_RESULT, {"id": "call_1_0", "offset": 90, "limit": 500}),
            (READ_RESULT, {"id": "nope"}),
            "done",
        ]
    )
    target = graph(model, HarnessTools(), tools=[read_result(30)], checkpointer=RunCheckpointer())
    agent = harness.wrap(target, id="reader", tools=[dump])
    assert (await agent.run("x", user="u")).answer == "done"
    assert READ_RESULT in model.requests[0]["tools"]
    assert tool_messages(model, 2)[-1][1] == (
        '012345678901234\n…[75 more characters: read_result(id="call_1_0", offset=25)]'
    )
    assert tool_messages(model, 3)[-1][1] == "0123456789"  # at most what is left
    assert tool_messages(model, 4)[-1][1] == "there is no result 'nope' to read"


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


def test_a_middleware_state_never_reaches_the_graphs_input_or_output() -> None:
    target = graph(ScriptedChatModel(turns=[]), HarnessTools(), StepLimit())
    schema = target.get_input_jsonschema()
    assert set(schema["properties"]) == {"messages"}


async def test_a_tool_that_asks_twice_is_answered_once_each(harness: Harness) -> None:
    @tool(side_effects="read")
    async def interview(name: str) -> str:
        """Ask two questions."""
        runtime = current()
        assert runtime is not None
        first = await runtime.ask("First?")
        second = await runtime.ask("Second?")
        return f"{first} then {second}"

    model = ScriptedChatModel(turns=[("interview", {"name": "a"}), "done"])
    target = graph(model, HarnessTools(), checkpointer=RunCheckpointer())
    agent = harness.wrap(target, id="twice", tools=[interview])
    paused = await agent.run("x", user="u")
    assert paused.interrupt is not None and paused.interrupt.question == "First?"
    again = await agent.resume(paused.interrupt.interrupt_id, "answer", answer="one", reviewer="r")
    # LangGraph hands the first answer back to the second question: it is passed over
    assert again.interrupt is not None and again.interrupt.question == "Second?"
    done = await agent.resume(again.interrupt.interrupt_id, "answer", answer="two", reviewer="r")
    assert done.answer == "done"
    assert tool_messages(model, 1) == [("call_1_0", "one then two")]


def test_the_helpers_read_what_they_are_given() -> None:
    from langchain.agents.middleware import ModelRequest, ModelResponse

    from trellis.harness.middleware import _named, _replied, _with_task

    assert _named({"type": "function", "function": {"name": "x"}}) == "x"
    assert _named({"name": "y"}) == "y"
    summary = HumanMessage("the summary", additional_kwargs={"lc_source": "summarization"})

    def request(messages: list[Any], state: list[Any]) -> Any:
        return ModelRequest(
            model=ScriptedChatModel(turns=[]), messages=messages, state={"messages": state}
        )

    # no user message to keep: nothing is put back; a summary in the state ends the head
    bare = request([summary], [SystemMessage("context", id="c")])
    assert _with_task(bare) is bare
    earlier = request([summary], [summary, HumanMessage("task", id="t")])
    assert _with_task(earlier) is earlier
    kept = request([summary], [SystemMessage("context", id="c"), HumanMessage("task", id="t")])
    assert [m.content for m in _with_task(kept).messages] == ["context", "task", "the summary"]
    _replied(None, ModelResponse(result=[]))  # nothing the model said: nothing on the span
