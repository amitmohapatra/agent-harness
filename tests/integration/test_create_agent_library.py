"""LangChain's ``create_agent`` and the middleware it ships, each run through the harness as the
library gives it: structured output, the hooks around the agent and the model, a jump, a
dynamic prompt, a context, state a middleware adds — and ``ToolCallLimitMiddleware``,
``ModelRetryMiddleware``, ``ModelFallbackMiddleware``, ``PIIMiddleware``,
``SummarizationMiddleware``, ``LLMToolSelectorMiddleware``, ``LLMToolEmulator``,
``FilesystemFileSearchMiddleware``, ``ShellToolMiddleware``, and LangGraph's older
``create_react_agent``. Each test says what the library does and what the harness still does
around it. (``ToolRetryMiddleware`` and ``ToolErrorMiddleware``: ``test_tool_errors_library``;
``HumanInTheLoopMiddleware``: ``test_hitl_middleware``; ``TodoListMiddleware``,
``ContextEditingMiddleware``, ``ModelCallLimitMiddleware``: ``ReAct``'s own tests.)"""

from __future__ import annotations

from pathlib import Path
from typing import Any, NotRequired

import pytest
from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    ExtendedModelResponse,
    FilesystemFileSearchMiddleware,
    HostExecutionPolicy,
    LLMToolEmulator,
    LLMToolSelectorMiddleware,
    ModelCallLimitMiddleware,
    ModelFallbackMiddleware,
    ModelRetryMiddleware,
    PIIMiddleware,
    ShellToolMiddleware,
    SummarizationMiddleware,
    ToolCallLimitMiddleware,
    after_agent,
    after_model,
    before_agent,
    before_model,
    dynamic_prompt,
    hook_config,
)
from langchain.agents.structured_output import ProviderStrategy, ToolStrategy
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool as lc_tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command
from pydantic import BaseModel, Field

from tests.support.chat_model import ScriptedChatModel, StreamingChatModel
from tests.support.memory import FakeMemoryService, pushed
from trellis import Harness
from trellis.contracts import RunEventType, RunStatus

ran: list[str] = []


@pytest.fixture(autouse=True)
def _reset() -> None:
    ran.clear()


def results(events: list[Any]) -> list[tuple[str, str]]:
    """The tool calls a stream reported, in order: each tool's name and how it ended."""
    return [
        (e.data["tool"], e.data["status"])
        for e in events
        if e.type is RunEventType.TOOL_CALL_RESULT
    ]


def text_of(message: Any) -> str:
    """A message's text, its parts joined."""
    content = message.content
    return content if isinstance(content, str) else "\n".join(p["text"] for p in content)


@lc_tool
def stock(sku: str) -> str:
    """Units of a SKU in stock."""
    ran.append(f"stock {sku}")
    return f"{sku}: 7"


stock.metadata = {"side_effects": "read"}


@lc_tool
def order(sku: str) -> str:
    """Order more of a SKU."""
    ran.append(f"order {sku}")
    return f"ordered {sku}"


order.metadata = {"side_effects": "irreversible"}


class Flaky(ScriptedChatModel):
    """A scripted model whose calls fail first with ``errors``, one each."""

    errors: list[Any] = Field(default_factory=list)

    def _generate(self, messages: Any, stop: Any = None, run_manager: Any = None, **kw: Any) -> Any:
        if self.errors:
            raise self.errors.pop(0)
        return super()._generate(messages, stop, run_manager, **kw)


# ------------------------------------------------------------------------ structured output
class Count(BaseModel):
    sku: str
    units: int


async def test_a_tool_strategy_answer_is_the_runs_answer_and_a_bad_one_is_asked_again(
    harness: Harness,
) -> None:
    """``response_format=ToolStrategy(Count)``: the model answers by calling the schema's tool;
    arguments that do not fit are told to it and it is asked again (``handle_errors``); the
    run's answer is the ``structured_response``, and the schema's tool is no call of the run."""
    model = ScriptedChatModel(
        turns=[
            ("stock", {"sku": "a"}),
            ("Count", {"sku": "a", "units": "many"}),
            ("Count", {"sku": "a", "units": 7}),
        ]
    )
    graph = create_agent(model, tools=[stock], response_format=ToolStrategy(Count))
    events = [e async for e in harness.wrap(graph, id="count").stream("how many a?", user="u")]
    assert results(events) == [("stock", "ok")]
    assert events[-1].data["result"] == {"sku": "a", "units": 7}
    assert len(model.seen) == 3 and model.seen[2][-1].type == "tool"


async def test_a_provider_strategy_answer_is_read_from_the_models_own_structured_reply(
    harness: Harness,
) -> None:
    """``response_format=ProviderStrategy(Count)``: the model is asked for the schema in its
    own way (its ``response_format``), and its reply is the run's structured answer."""
    model = ScriptedChatModel(turns=['{"sku": "a", "units": 7}'])
    graph = create_agent(model, tools=[stock], response_format=ProviderStrategy(Count))
    result = await harness.wrap(graph, id="count").run("how many a?", user="u")
    assert result.answer == Count(sku="a", units=7)
    assert "response_format" in model.requests[0]


# -------------------------------------------------------------- hooks, a jump, state, context
async def test_the_hooks_around_the_agent_and_the_model_run_in_the_librarys_order(
    harness: Harness,
) -> None:
    order_: list[str] = []

    @before_agent
    def start(state: Any, runtime: Any) -> None:
        order_.append("before_agent")

    @before_model
    def ahead(state: Any, runtime: Any) -> None:
        order_.append("before_model")

    @after_model
    def behind(state: Any, runtime: Any) -> None:
        order_.append("after_model")

    @after_agent
    def end(state: Any, runtime: Any) -> None:
        order_.append("after_agent")

    @dynamic_prompt
    def prompt(request: Any) -> str:
        return f"You count stock. Messages so far: {len(request.messages)}."

    model = ScriptedChatModel(turns=[("stock", {"sku": "a"}), "7"])
    graph = create_agent(model, tools=[stock], middleware=[start, ahead, behind, end, prompt])
    result = await harness.wrap(graph, id="hooks").run("a?", user="u")
    assert result.answer == "7"
    assert order_ == [
        "before_agent",
        "before_model",
        "after_model",
        "before_model",
        "after_model",
        "after_agent",
    ]
    assert text_of(model.seen[0][0]).startswith("You count stock. Messages so far: 1.")
    assert text_of(model.seen[1][0]).startswith("You count stock. Messages so far: 3.")


async def test_a_hook_that_jumps_to_the_end_ends_the_run_with_its_message(harness: Harness) -> None:
    class Closed(AgentMiddleware):
        @hook_config(can_jump_to=["end"])
        def before_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
            return {"messages": [AIMessage("We are closed.")], "jump_to": "end"}

    model = ScriptedChatModel(turns=[])
    graph = create_agent(model, tools=[stock], middleware=[Closed()])
    result = await harness.wrap(graph, id="closed").run("a?", user="u")
    assert result.status is RunStatus.SUCCESS and result.answer == "We are closed."
    assert model.seen == []


class Shop(BaseModel):
    region: str


class Desk(AgentState[Any]):
    desk: NotRequired[str]


async def test_a_middlewares_state_and_the_runs_context_reach_its_hooks_and_tools(
    harness: Harness,
) -> None:
    """A middleware's ``state_schema`` adds a state key the input may set, and the graph's
    ``context_schema`` is given per run (``framework_options``): both reach a hook and a tool
    (``ToolRuntime``)."""
    read: dict[str, Any] = {}

    class Reads(AgentMiddleware[Desk, Shop]):
        state_schema = Desk

        def before_model(self, state: Any, runtime: Any) -> None:
            read["hook"] = (state.get("desk"), runtime.context.region)

    @lc_tool
    def where(sku: str, runtime: ToolRuntime[Shop, Any]) -> str:
        """Where a SKU is."""
        read["tool"] = (runtime.state.get("desk"), runtime.context.region)
        return "north"

    where.metadata = {"side_effects": "read"}
    model = ScriptedChatModel(turns=[("where", {"sku": "a"}), "north"])
    graph = create_agent(model, tools=[where], middleware=[Reads()], context_schema=Shop)
    result = await harness.wrap(graph, id="desk").run(
        {"messages": [{"role": "user", "content": "where?"}], "desk": "d1"},
        user="u",
        framework_options={"context": Shop(region="eu")},
    )
    assert result.answer == "north"
    assert read == {"hook": ("d1", "eu"), "tool": ("d1", "eu")}


class Counted(AgentState[Any]):
    calls: NotRequired[int]


async def test_a_model_wrapper_updates_the_state_beside_the_models_reply(harness: Harness) -> None:
    """A ``wrap_model_call`` middleware may return an ``ExtendedModelResponse``: its
    ``Command`` is applied to the graph's state with the model's reply."""

    class Counts(AgentMiddleware[Counted, Any]):
        state_schema = Counted

        async def awrap_model_call(self, request: Any, handler: Any) -> Any:
            calls = (request.state.get("calls") or 0) + 1
            return ExtendedModelResponse(
                model_response=await handler(request), command=Command(update={"calls": calls})
            )

    model = ScriptedChatModel(turns=[("stock", {"sku": "a"}), "7"])
    graph = create_agent(model, tools=[stock], middleware=[Counts()], checkpointer=InMemorySaver())
    result = await harness.wrap(graph, id="counts").run("a?", user="u", thread="t")
    assert result.answer == "7"
    key = await harness.conversation(user="u", thread="t")
    state = await graph.aget_state({"configurable": {"thread_id": key}})
    assert state.values["calls"] == 2


async def test_a_tool_a_middleware_adds_per_call_is_offered_run_and_governed(
    harness: Harness,
) -> None:
    """A middleware may offer the model a tool the graph was not built with
    (``request.override(tools=...)``) and hand it to the tool node (``override(tool=...)``):
    the model is offered it, and its call is the run's, governed as the tool declares."""

    @lc_tool
    def shelf(sku: str) -> str:
        """Where a SKU is shelved."""
        ran.append(f"shelf {sku}")
        return f"{sku}: aisle 4"

    shelf.metadata = {"side_effects": "irreversible"}

    class Adds(AgentMiddleware):
        async def awrap_model_call(self, request: Any, handler: Any) -> Any:
            return await handler(request.override(tools=[*request.tools, shelf]))

        async def awrap_tool_call(self, request: Any, handler: Any) -> Any:
            named = request.tool_call["name"] == "shelf"
            return await handler(request.override(tool=shelf) if named else request)

    model = ScriptedChatModel(turns=[("shelf", {"sku": "a"}), "aisle 4"])
    graph = create_agent(model, tools=[stock], middleware=[Adds()], checkpointer=InMemorySaver())
    agent = harness.wrap(graph, id="adds")
    paused = await agent.run("where is a?", user="u", thread="t")
    assert model.requests[0]["tools"] == ["stock", "shelf"]
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None and ran == []
    done = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="u")
    assert done.answer == "aisle 4" and ran == ["shelf a"]


# ------------------------------------------------------------------------- tool results
@lc_tool(response_format="content_and_artifact")
def report(sku: str) -> tuple[str, dict[str, Any]]:
    """A stock report: the text the model reads, and the rows it does not."""
    ran.append(f"report {sku}")
    return f"report of {sku}", {"rows": [1, 2, 3]}


report.metadata = {"side_effects": "read"}


@lc_tool
def picture(sku: str) -> list[dict[str, Any]]:
    """A photo of a SKU, as content blocks."""
    ran.append(f"picture {sku}")
    return [
        {"type": "text", "text": f"photo of {sku}"},
        {"type": "image", "base64": "iVBORw0KGgo=", "mime_type": "image/png"},
    ]


picture.metadata = {"side_effects": "read"}


async def test_a_tools_artifact_and_content_blocks_reach_the_graph_as_the_tool_made_them(
    harness: Harness,
) -> None:
    """A ``content_and_artifact`` tool's message keeps its artifact (data the model does not
    read, kept in the graph's state), and a tool's content blocks reach the model as blocks —
    an image it sees, not its JSON as text — as the library gives them; each call is still
    governed and recorded with the text of its result."""
    model = ScriptedChatModel(turns=[[("report", {"sku": "a"}), ("picture", {"sku": "a"})], "ok"])
    graph = create_agent(model, tools=[report, picture])
    events = [e async for e in harness.wrap(graph, id="results").stream("report a", user="u")]
    assert sorted(results(events)) == [("picture", "ok"), ("report", "ok")]
    read = {m.name: m for m in model.seen[-1] if isinstance(m, ToolMessage)}
    assert read["report"].content == "report of a"
    assert read["report"].artifact == {"rows": [1, 2, 3]}
    assert read["picture"].content == picture.invoke({"sku": "a"})
    [recorded] = [
        e.data
        for e in events
        if e.type is RunEventType.TOOL_CALL_RESULT and e.data["tool"] == "picture"
    ]
    assert "photo of a" in str(recorded)


async def test_a_tools_artifact_is_kept_when_its_call_is_replayed_after_a_pause(
    harness: Harness,
) -> None:
    """Without a checkpointer, the run resumed is the graph run again, and a call made before
    the pause is answered from the journal: its message still carries the artifact the tool
    made, though the tool does not run again."""
    model = ScriptedChatModel(
        turns=[[("report", {"sku": "a"}), ("order", {"sku": "a"})]] * 2 + ["ordered"]
    )
    graph = create_agent(model, tools=[report, order])
    agent = harness.wrap(graph, id="replayed")
    paused = await agent.run("report and order a", user="u")
    assert paused.interrupt is not None
    done = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="u")
    assert done.status is RunStatus.SUCCESS and ran == ["report a", "order a"]
    [kept] = [m for m in model.seen[-1] if isinstance(m, ToolMessage) and m.name == "report"]
    assert kept.artifact == {"rows": [1, 2, 3]} and kept.content == "report of a"


async def test_a_call_made_twice_in_one_step_runs_twice_as_the_library_runs_it(
    harness: Harness,
) -> None:
    """``create_agent`` runs each call its model makes, the same call twice included: so does
    the run, each governed and recorded, and the model reads each one's own result."""
    model = ScriptedChatModel(turns=[[("stock", {"sku": "a"}), ("stock", {"sku": "a"})], "7"])
    graph = create_agent(model, tools=[stock])
    events = [e async for e in harness.wrap(graph, id="twice").stream("a twice", user="u")]
    assert ran == ["stock a", "stock a"]
    assert results(events) == [("stock", "ok"), ("stock", "ok")]
    read = [m.content for m in model.seen[-1] if isinstance(m, ToolMessage)]
    assert read == ["a: 7", "a: 7"]


async def test_a_large_result_reaches_a_graph_of_its_own_loop_whole(harness: Harness) -> None:
    """Without the harness's middleware the graph runs its own loop, and its tool's result is
    the library's: a result past the size the harness keeps, and a table, reach the model as the
    tool returned them, with no tool of the harness's added to read them back."""

    @lc_tool
    def big(n: int) -> str:
        """A long text."""
        return "word " * n

    big.metadata = {"side_effects": "read"}
    model = ScriptedChatModel(turns=[("big", {"n": 20_000}), "read"])
    graph = create_agent(model, tools=[big])
    assert (await harness.wrap(graph, id="big").run("x", user="u")).answer == "read"
    [read] = [m for m in model.seen[-1] if isinstance(m, ToolMessage)]
    assert read.content == "word " * 20_000
    assert model.requests[-1]["tools"] == ["big"]


# ---------------------------------------------------------------------- the calls' limits
@pytest.mark.parametrize("how", ["end", "error"])
async def test_model_call_limit_middleware_ends_the_run_as_it_is_told_to(
    harness: Harness, how: Any
) -> None:
    model = ScriptedChatModel(turns=[("stock", {"sku": "a"})])
    limit = ModelCallLimitMiddleware(run_limit=1, exit_behavior=how)
    graph = create_agent(model, tools=[stock], middleware=[limit])
    result = await harness.wrap(graph, id=f"limit-{how}").run("a?", user="u")
    said = "Model call limits exceeded: run limit (1/1)"
    if how == "end":
        assert result.status is RunStatus.SUCCESS and result.answer == said
    else:
        assert result.status is RunStatus.ERROR and result.error is not None
        assert result.error.message == said
    assert ran == ["stock a"] and len(model.seen) == 1


async def test_tool_call_limit_middleware_blocks_the_calls_past_its_limit(harness: Harness) -> None:
    """Past its limit ``ToolCallLimitMiddleware`` answers a call itself: the tool does not
    run, and the blocked call is no call of the run."""
    model = ScriptedChatModel(
        turns=[[("stock", {"sku": "a"}), ("stock", {"sku": "b"})], "a: 7, b not counted"]
    )
    limit = ToolCallLimitMiddleware(tool_name="stock", run_limit=1)
    graph = create_agent(model, tools=[stock], middleware=[limit])
    events = [e async for e in harness.wrap(graph, id="limit").stream("a and b?", user="u")]
    assert ran == ["stock a"] and results(events) == [("stock", "ok")]
    read = [m for m in model.seen[-1] if isinstance(m, ToolMessage)]
    told = {m.tool_call_id: (m.status, m.content) for m in read}
    assert told == {
        "call_1_0": ("success", "a: 7"),
        "call_1_1": ("error", "Tool call limit exceeded. Do not call 'stock' again."),
    }


# ------------------------------------------------------------------- the model: retry, fallback
async def test_model_retry_middleware_asks_the_model_again(harness: Harness) -> None:
    model = Flaky(turns=["7"], errors=[TimeoutError("busy")])
    retry = ModelRetryMiddleware(max_retries=2, initial_delay=0.0, jitter=False)
    graph = create_agent(model, tools=[stock], middleware=[retry])
    result = await harness.wrap(graph, id="again").run("a?", user="u")
    assert result.status is RunStatus.SUCCESS and result.answer == "7"


async def test_model_fallback_middleware_asks_the_next_model_with_the_same_tools(
    harness: Harness,
) -> None:
    first = Flaky(turns=[], errors=[RuntimeError("provider down")] * 2)
    second = ScriptedChatModel(turns=[("stock", {"sku": "a"}), "7 from the fallback"])
    graph = create_agent(first, tools=[stock], middleware=[ModelFallbackMiddleware(second)])
    events = [e async for e in harness.wrap(graph, id="fallback").stream("a?", user="u")]
    assert events[-1].data["result"] == "7 from the fallback"
    assert results(events) == [("stock", "ok")]
    assert second.requests[0]["tools"] == ["stock"]


# ------------------------------------------------------------------------------ PII
async def test_pii_middleware_redacts_what_the_model_is_sent_and_blocks_as_it_says(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    """``PIIMiddleware("email")`` redacts the address before the model call — the memory
    context still comes with the question — and ``strategy="block"`` ends the run in error."""
    model = ScriptedChatModel(turns=["noted"])
    graph = create_agent(model, tools=[], middleware=[PIIMiddleware("email", strategy="redact")])
    result = await memory_harness.wrap(graph, id="pii").run("write to ada@example.com", user="u")
    assert result.answer == "noted"
    sent = text_of(model.seen[0][-1])
    assert "ada@example.com" not in sent and "write to [REDACTED_EMAIL]" in sent
    assert pushed(memory_service.context_text) in sent

    blocking = create_agent(
        ScriptedChatModel(turns=[]),
        tools=[],
        middleware=[PIIMiddleware("email", strategy="block")],
    )
    refused = await memory_harness.wrap(blocking, id="pii-block").run("ada@example.com", user="u")
    assert refused.status is RunStatus.ERROR and refused.error is not None
    assert "email" in refused.error.message.lower()


async def test_pii_middleware_redacts_the_streamed_answer_as_it_streams(harness: Harness) -> None:
    """``PIIMiddleware(apply_to_output=True)`` redacts the model's reply in the state and, by
    its stream transformer, on the wire: no streamed piece of the answer carries the address,
    however the model split it, and the pieces make the redacted answer."""
    model = StreamingChatModel(turns=["write to ada@example.com now"], blocks=False, piece=10)
    pii = PIIMiddleware("email", strategy="redact", apply_to_output=True)
    graph = create_agent(model, tools=[], middleware=[pii])
    events = [e async for e in harness.wrap(graph, id="pii-stream").stream("hi", user="u")]
    deltas = [e.data["delta"] for e in events if e.type is RunEventType.TEXT_MESSAGE_CONTENT]
    assert events[-1].data["result"] == "write to [REDACTED_EMAIL] now"
    assert "".join(deltas) == "write to [REDACTED_EMAIL] now"
    assert not any("@" in d or "example" in d for d in deltas)


# ------------------------------------------------------------------------ summarization
async def test_summarization_middleware_replaces_older_messages_with_its_summary(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    """Past its trigger, ``SummarizationMiddleware`` replaces the older messages with a summary
    its own model writes: the model reads the summary, every tool still runs once, and the
    run's question and answer are what memory is told of the turn."""
    summarizer = ScriptedChatModel(turns=["They asked about a and b."] * 2)
    model = ScriptedChatModel(
        turns=[("stock", {"sku": "a"}), ("stock", {"sku": "b"}), "a: 7, b: 7"]
    )
    summary = SummarizationMiddleware(
        model=summarizer, trigger=("messages", 4), keep=("messages", 2)
    )
    graph = create_agent(model, tools=[stock], middleware=[summary])
    result = await memory_harness.wrap(graph, id="summary").run("a and b?", user="u")
    assert result.status is RunStatus.SUCCESS and result.answer == "a: 7, b: 7"
    assert summarizer.turns == [] and ran == ["stock a", "stock b"]
    read = [text_of(m) for m in model.seen[-1] if m.type == "human"]
    assert read == ["Here is a summary of the conversation to date:\n\nThey asked about a and b."]
    await memory_harness.writes.drain()
    told = [(m["role"], m["content"]) for m in memory_service.threads[result.run_id]]
    assert told[0] == ("USER", "a and b?") and told[-1] == ("ASSISTANT", "a: 7, b: 7")


# -------------------------------------------------------------- tool selection, tool emulation
async def test_llm_tool_selector_offers_the_model_the_tools_its_selector_picked(
    harness: Harness,
) -> None:
    selector = ScriptedChatModel(turns=[("ToolSelectionResponse", {"tools": ["stock"]})] * 2)
    model = ScriptedChatModel(turns=[("stock", {"sku": "a"}), "7"])
    graph = create_agent(
        model, tools=[stock, order], middleware=[LLMToolSelectorMiddleware(model=selector)]
    )
    result = await harness.wrap(graph, id="select").run("how many a?", user="u")
    assert result.answer == "7"
    assert [r["tools"] for r in model.requests] == [["stock"], ["stock"]]


async def test_llm_tool_emulator_answers_for_the_tool_which_does_not_run(harness: Harness) -> None:
    """``LLMToolEmulator`` answers a call with its own model in the tool's place: the tool does
    not run, so nothing is asked of a person about it."""
    emulator = ScriptedChatModel(turns=["ordered a (emulated)"])
    model = ScriptedChatModel(turns=[("order", {"sku": "a"}), "done"])
    graph = create_agent(
        model, tools=[order], middleware=[LLMToolEmulator(tools=["order"], model=emulator)]
    )
    result = await harness.wrap(graph, id="emulated").run("order a", user="u")
    assert result.status is RunStatus.SUCCESS and result.answer == "done"
    assert ran == [] and model.seen[-1][-1].content == "ordered a (emulated)"


# -------------------------------------------------------------------- file search, the shell
async def test_file_search_middlewares_tool_asks_then_searches_the_folder(
    harness: Harness, tmp_path: Path
) -> None:
    """``FilesystemFileSearchMiddleware`` gives ``glob_search`` and ``grep_search`` over a
    folder. The library's tools say nothing of what they do, so a call asks a person first (as
    any tool that declares no side effects), then runs as the library wrote it."""
    (tmp_path / "notes.md").write_text("tides at noon\n")
    search = FilesystemFileSearchMiddleware(root_path=str(tmp_path), use_ripgrep=False)
    model = ScriptedChatModel(turns=[("grep_search", {"pattern": "tides"}), "found"])
    graph = create_agent(model, tools=[], middleware=[search], checkpointer=InMemorySaver())
    agent = harness.wrap(graph, id="search")
    paused = await agent.run("find tides", user="u", thread="t")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
    assert paused.interrupt.tool_call is not None
    assert paused.interrupt.tool_call.tool == "grep_search"
    done = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="u")
    assert done.answer == "found"
    assert "notes.md" in str(model.seen[-1][-1].content)


async def test_shell_tool_middlewares_shell_asks_then_runs_in_the_librarys_session(
    harness: Harness, tmp_path: Path
) -> None:
    """``ShellToolMiddleware`` starts a shell before the agent and gives it as the ``shell``
    tool: a call asks a person first (the tool declares no side effects), then runs in the
    library's own session — started again by the library for the run that resumes."""
    shell = ShellToolMiddleware(
        workspace_root=str(tmp_path), execution_policy=HostExecutionPolicy()
    )
    command = "echo tides > t.txt && cat t.txt"
    model = ScriptedChatModel(turns=[("shell", {"command": command}), "ok"])
    graph = create_agent(model, tools=[], middleware=[shell], checkpointer=InMemorySaver())
    agent = harness.wrap(graph, id="shell")
    paused = await agent.run("write tides", user="u", thread="t")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
    assert paused.interrupt.tool_call is not None and paused.interrupt.tool_call.tool == "shell"
    assert paused.interrupt.tool_call.args == {"command": command}
    assert not (tmp_path / "t.txt").exists()
    done = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="u")
    assert done.status is RunStatus.SUCCESS and done.answer == "ok"
    assert "tides" in str(model.seen[-1][-1].content)
    assert (tmp_path / "t.txt").read_text().strip() == "tides"


# ------------------------------------------------------------------- create_react_agent
async def test_create_react_agents_hooks_run_and_its_tools_are_governed(harness: Harness) -> None:
    """LangGraph's older prebuilt agent: its ``pre_model_hook`` and ``post_model_hook`` run
    around each model call, and its tool node's calls are the run's."""
    from langgraph.prebuilt import create_react_agent

    hooks: list[str] = []

    def pre(state: Any) -> dict[str, Any]:
        hooks.append("pre")
        return {"llm_input_messages": state["messages"]}

    def post(state: Any) -> dict[str, Any]:
        hooks.append("post")
        return {}

    model = ScriptedChatModel(turns=[("stock", {"sku": "a"}), "7"])
    with pytest.warns(DeprecationWarning):
        graph = create_react_agent(model, [stock], pre_model_hook=pre, post_model_hook=post)
    events = [e async for e in harness.wrap(graph, id="prebuilt").stream("a?", user="u")]
    assert results(events) == [("stock", "ok")]
    assert events[-1].data["result"] == "7"
    assert hooks == ["pre", "post", "pre", "post"]
