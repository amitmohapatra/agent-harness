"""A tool of the graph's own that raises: the harness records the failed call, and what the
graph does with the error is the library's — LangChain's ``ToolRetryMiddleware`` tries the
call again, ``ToolErrorMiddleware`` words what the model reads, a wrapper of the builder's
answers in the tool's place, a ``ToolNode``'s ``handle_tool_errors`` gives its text, exactly
as the graph words it. An error nothing of the graph handles fails the run, as it fails the
graph without the harness — unless the graph runs the harness's own middleware (``ReAct``,
``middleware()``: ``GovernTools``), whose rule is the harness's: the model reads the failure."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import httpx
import pytest
from deepagents import create_deep_agent
from langchain.agents import create_agent
from langchain.agents.middleware import (
    ToolErrorMiddleware,
    ToolRetryMiddleware,
    wrap_tool_call,
)
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool as lc_tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import RetryPolicy

from tests.support.chat_model import ScriptedChatModel
from tests.support.journals import journal_of
from tests.support.memory import FakeMemoryService
from trellis import Budget, Harness, ReAct
from trellis.contracts import RunEventType, RunStatus, ToolOutcome, ToolStatus
from trellis.harness.hooks import Ask, Hooks
from trellis.harness.tools import base
from trellis.harness.tools.bridge import MAX_REPEATS

ran: list[str] = []
fails: list[Exception] = []


@pytest.fixture(autouse=True)
def _reset() -> None:
    ran.clear()
    fails.clear()


@lc_tool
def stock(sku: str) -> str:
    """Units of a SKU in stock."""
    ran.append(f"stock {sku}")
    if fails:
        raise fails.pop(0)
    return f"{sku}: 7"


stock.metadata = {"side_effects": "read"}


def results(events: list[Any]) -> list[tuple[str, str]]:
    """The tool calls a stream reported, in order: each tool's name and how it ended."""
    return [
        (e.data["tool"], e.data["status"])
        for e in events
        if e.type is RunEventType.TOOL_CALL_RESULT
    ]


def told(model: ScriptedChatModel) -> ToolMessage:
    """The tool result the model read last."""
    last = model.seen[-1][-1]
    assert isinstance(last, ToolMessage)
    return last


def retrying(max_retries: int) -> ToolRetryMiddleware:
    return ToolRetryMiddleware(
        max_retries=max_retries, initial_delay=0.0, jitter=False, retry_on=(ValueError,)
    )


async def test_tool_retry_middleware_tries_a_failing_tool_again(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    """The library's retry gets the tool's error and makes the call again: the second try's
    result is what the model reads. Each try is a call the run made — on its events, recorded —
    the failed one as failed."""
    fails.append(ValueError("no shelf"))
    model = ScriptedChatModel(turns=[("stock", {"sku": "a"}), "7"])
    graph = create_agent(model, tools=[stock], middleware=[retrying(2)])
    events = [e async for e in memory_harness.wrap(graph, id="retry").stream("a?", user="u")]
    assert ran == ["stock a", "stock a"]
    assert results(events) == [("stock", "error"), ("stock", "ok")]
    assert model.seen[-1][-1].content == "a: 7"
    await memory_harness.writes.drain()
    recorded = [c.body["status"] for c in memory_service.named("record_tool")]
    assert recorded == ["error", "ok"]


async def test_tool_retry_middleware_out_of_tries_tells_the_model_in_its_own_words(
    harness: Harness,
) -> None:
    fails.extend([ValueError("no shelf")] * 2)
    model = ScriptedChatModel(turns=[("stock", {"sku": "a"}), "it failed"])
    graph = create_agent(model, tools=[stock], middleware=[retrying(1)])
    result = await harness.wrap(graph, id="retry").run("a?", user="u")
    assert result.answer == "it failed" and ran == ["stock a", "stock a"]
    assert told(model).status == "error"
    assert told(model).content == (
        "Tool 'stock' failed after 2 attempts with ValueError: no shelf. Please try again."
    )


async def test_the_tries_of_one_call_are_not_a_model_repeating_itself(harness: Harness) -> None:
    """A call the run is given over and over with the same arguments is stopped as a stall
    (``MAX_REPEATS``): a model looping. The library's retries of one call are that call tried
    again, and each is made."""
    fails.extend([ValueError("no shelf")] * MAX_REPEATS)
    model = ScriptedChatModel(turns=[("stock", {"sku": "a"}), "7"])
    graph = create_agent(model, tools=[stock], middleware=[retrying(MAX_REPEATS)])
    result = await harness.wrap(graph, id="retry").run("a?", user="u")
    assert result.answer == "7" and ran == ["stock a"] * (MAX_REPEATS + 1)
    assert model.seen[-1][-1].content == "a: 7"


async def test_the_librarys_retry_works_under_the_harness_middleware_too(harness: Harness) -> None:
    """``ReAct`` governs its graph's own tools in a middleware of its own (``GovernTools``):
    a retry middleware given to it gets the tool's error all the same."""
    fails.append(ValueError("no shelf"))
    model = ScriptedChatModel(turns=[("stock", {"sku": "a"}), "7"])
    agent = harness.wrap(
        ReAct(system_prompt="Count.", model=model, tools=[stock], middleware=[retrying(1)]),
        id="react",
    )
    result = await agent.run("a?", user="u")
    assert result.answer == "7" and ran == ["stock a", "stock a"]


async def test_tool_error_middleware_words_the_error_the_model_reads(harness: Harness) -> None:
    fails.append(ValueError("no shelf"))

    def worded(exc: Exception, request: Any) -> str:
        return f"{request.tool_call['name']} is unavailable ({exc})"

    model = ScriptedChatModel(turns=[("stock", {"sku": "a"}), "unavailable"])
    graph = create_agent(model, tools=[stock], middleware=[ToolErrorMiddleware(worded)])
    events = [e async for e in harness.wrap(graph, id="worded").stream("a?", user="u")]
    assert results(events) == [("stock", "error")]
    assert (told(model).status, told(model).content) == (
        "error",
        "stock is unavailable (no shelf)",
    )


async def test_a_wrapper_of_yours_gets_the_tools_error_and_answers_in_its_place(
    harness: Harness,
) -> None:
    fails.append(ConnectionError("down"))

    @wrap_tool_call
    async def fallback(request: Any, handler: Any) -> Any:
        try:
            return await handler(request)
        except ConnectionError:
            return ToolMessage("a: 5 (yesterday's count)", tool_call_id=request.tool_call["id"])

    model = ScriptedChatModel(turns=[("stock", {"sku": "a"}), "5"])
    graph = create_agent(model, tools=[stock], middleware=[fallback])
    result = await harness.wrap(graph, id="fallback").run("a?", user="u")
    assert result.answer == "5"
    assert model.seen[-1][-1].content == "a: 5 (yesterday's count)"


async def test_an_error_a_wrapper_raises_of_its_own_is_the_graphs(harness: Harness) -> None:
    """A wrapper that turns the tool's error into another raises its own: no call's failure,
    so the graph fails with it, as the library has it."""
    fails.append(ValueError("no shelf"))

    @wrap_tool_call
    async def strict(request: Any, handler: Any) -> Any:
        try:
            return await handler(request)
        except ValueError as exc:
            raise RuntimeError("the stock desk is closed") from exc

    model = ScriptedChatModel(turns=[("stock", {"sku": "a"}), "unreached"])
    graph = create_agent(model, tools=[stock], middleware=[strict])
    result = await harness.wrap(graph, id="strict").run("a?", user="u")
    assert result.status is RunStatus.ERROR and result.error is not None
    assert "the stock desk is closed" in result.error.message


def by_hand(model: Any, node: ToolNode) -> Any:
    bound = model.bind_tools(list(node.tools_by_name.values()))

    async def think(state: Any) -> dict[str, Any]:
        return {"messages": [await bound.ainvoke(state["messages"])]}

    graph = StateGraph(MessagesState)
    graph.add_node("model", think)
    graph.add_node("tools", node)
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", tools_condition)
    graph.add_edge("tools", "model")
    return graph.compile()


async def test_a_tool_nodes_own_error_text_is_what_the_model_reads(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    """``ToolNode(handle_tool_errors="...")``: the node's text for a tool that failed is what
    the model reads, as the node wrote it; the call is the run's failed call all the same."""
    fails.append(ValueError("no shelf"))
    model = ScriptedChatModel(turns=[("stock", {"sku": "a"}), "it is down"])
    node = ToolNode([stock], handle_tool_errors="Shelf lookups are down: say so.")
    agent = memory_harness.wrap(by_hand(model, node), id="shelf")
    events = [e async for e in agent.stream("a?", user="u")]
    assert results(events) == [("stock", "error")]
    assert (told(model).status, told(model).content) == (
        "error",
        "Shelf lookups are down: say so.",
    )
    await memory_harness.writes.drain()
    assert [c.body["status"] for c in memory_service.named("record_tool")] == ["error"]


async def test_an_after_tool_hook_still_has_the_last_word_on_a_handled_error(
    harness: Harness,
) -> None:
    """The run's ``after_tool`` hooks change what the model reads of a call: a hook that
    rewrites a failed call's result is read, not the node's text."""

    class Kinder(Hooks):
        async def after_tool(self, call: Any, outcome: ToolOutcome) -> ToolOutcome:
            return outcome.model_copy(update={"output": "the shelf will be back at noon"})

    fails.append(ValueError("no shelf"))
    model = ScriptedChatModel(turns=[("stock", {"sku": "a"}), "noon"])
    node = ToolNode([stock], handle_tool_errors="Shelf lookups are down: say so.")
    agent = harness.wrap(by_hand(model, node), id="shelf", hooks=[Kinder()])
    assert (await agent.run("a?", user="u")).answer == "noon"
    assert told(model).content == "the shelf will be back at noon"


def wrapper_reraising() -> Any:
    @wrap_tool_call
    async def watching(request: Any, handler: Any) -> Any:
        try:
            return await handler(request)
        except ValueError:
            ran.append("seen by the wrapper")
            raise

    return watching


@pytest.mark.parametrize("how", ["node", "agent", "retry", "wrapper"])
async def test_an_error_nothing_of_the_graph_handles_fails_the_run_as_langgraph_does(
    memory_harness: Harness, memory_service: FakeMemoryService, how: str
) -> None:
    """LangGraph raises a tool's error that nothing handles, and the graph fails with it: a
    ``ToolNode(handle_tool_errors=False)``, ``create_agent``'s tool node (it handles a call's
    invalid arguments only), ``ToolRetryMiddleware(on_failure="error")`` out of tries, a wrapper
    that raises it again. The run fails with that error, as any graph's error ends a run; the
    call is recorded as the run's failed call."""
    fails.extend([ValueError("no shelf")] * 2)
    model = ScriptedChatModel(turns=[("stock", {"sku": "a"}), "unreached"])
    if how == "node":
        graph = by_hand(model, ToolNode([stock], handle_tool_errors=False))
    elif how == "agent":
        graph = create_agent(model, tools=[stock])
    elif how == "retry":
        gives_up = ToolRetryMiddleware(
            max_retries=1, initial_delay=0.0, jitter=False, on_failure="error"
        )
        graph = create_agent(model, tools=[stock], middleware=[gives_up])
    else:
        graph = create_agent(model, tools=[stock], middleware=[wrapper_reraising()])
    agent = memory_harness.wrap(graph, id=f"unhandled-{how}")
    events = [e async for e in agent.stream("a?", user="u")]
    finished = events[-1]
    assert finished.type is RunEventType.RUN_FINISHED and finished.error is not None
    assert (finished.error.code, finished.error.message) == ("ValueError", "no shelf")
    assert len(model.seen) == 1
    record = await memory_harness.runs.get(finished.run_id)
    assert record is not None and record.status is RunStatus.ERROR
    assert ("stock", "error") in results(events)
    await memory_harness.writes.drain()
    assert "error" in [c.body["status"] for c in memory_service.named("record_tool")]


async def test_under_the_harness_middleware_an_error_nothing_handles_is_read_by_the_model(
    harness: Harness,
) -> None:
    """``ReAct`` (and a graph built with ``middleware()``: ``GovernTools``) runs the graph's
    own tools under the harness's rule, as its docs say: a tool that raises becomes an error
    result the model reads, and the run goes on."""
    fails.append(ValueError("no shelf"))
    model = ScriptedChatModel(turns=[("stock", {"sku": "a"}), "it failed"])
    agent = harness.wrap(ReAct(system_prompt="Count.", model=model, tools=[stock]), id="react")
    result = await agent.run("a?", user="u")
    assert result.status is RunStatus.SUCCESS and result.answer == "it failed"
    assert (told(model).status, told(model).content) == ("error", "stock failed: no shelf")


# --------------------------------------------------------------------------- the hooks' word
class Kinder(Hooks):
    """The run's ``after_tool`` hook rewording a failed call (``ok``: making it a result)."""

    def __init__(self, *, ok: bool = False) -> None:
        self.ok = ok

    async def after_tool(self, call: Any, outcome: ToolOutcome) -> ToolOutcome:
        if outcome.ok:
            return outcome
        changed: dict[str, Any] = {"output": "the shelf will be back at noon"}
        ended = {**changed, "status": ToolStatus.OK} if self.ok else changed
        return outcome.model_copy(update=ended)


def answering() -> Any:
    @wrap_tool_call
    async def answer(request: Any, handler: Any) -> Any:
        try:
            return await handler(request)
        except ValueError as exc:
            ran.append("answered by the wrapper")
            return ToolMessage(
                f"wrapper: {exc}", tool_call_id=request.tool_call["id"], status="error"
            )

    return answer


@pytest.mark.parametrize("how", ["wrapper", "retry"])
async def test_an_after_tool_hook_has_the_last_word_on_an_error_the_graph_answers(
    harness: Harness, how: str
) -> None:
    """The tool raises and the run's ``after_tool`` hook rewords the failure: the error is
    raised into the graph all the same — a wrapper of yours gets it and answers in its place,
    ``ToolRetryMiddleware`` gives up and words its message — and the failed message the graph
    makes of it says the hook's words, as the hooks decide what the model reads."""
    fails.append(ValueError("no shelf"))
    middleware = (
        [answering()]
        if how == "wrapper"
        else [ToolRetryMiddleware(max_retries=0, on_failure="continue")]
    )
    model = ScriptedChatModel(turns=[("stock", {"sku": "a"}), "noon"])
    graph = create_agent(model, [stock], middleware=middleware)
    result = await harness.wrap(graph, id="shelf", hooks=[Kinder()]).run("a?", user="u")
    assert result.status is RunStatus.SUCCESS and result.answer == "noon"
    assert (told(model).status, told(model).content) == ("error", "the shelf will be back at noon")
    assert ran == ["stock a", *(["answered by the wrapper"] if how == "wrapper" else [])]


async def test_an_after_tool_hook_that_makes_a_failure_a_result_gives_the_graph_that_result(
    harness: Harness,
) -> None:
    """A hook that turns the failed call into a result: the graph gets the result — no error
    for its retry to try again, its wrapper to answer — and the model reads it."""
    fails.append(ValueError("no shelf"))
    model = ScriptedChatModel(turns=[("stock", {"sku": "a"}), "noon"])
    graph = create_agent(model, [stock], middleware=[retrying(2), answering()])
    result = await harness.wrap(graph, id="shelf", hooks=[Kinder(ok=True)]).run("a?", user="u")
    assert result.answer == "noon" and ran == ["stock a"]
    assert (told(model).status, told(model).content) == (
        "success",
        "the shelf will be back at noon",
    )


async def test_under_the_harness_middleware_the_hooks_word_on_an_error_is_read_too(
    harness: Harness,
) -> None:
    """``ReAct``'s rule for an error nothing of the graph handles: the model reads the failed
    result — the hook's words when it reworded it."""
    fails.append(ValueError("no shelf"))
    model = ScriptedChatModel(turns=[("stock", {"sku": "a"}), "noon"])
    react = ReAct(system_prompt="Count.", model=model, tools=[stock])
    result = await harness.wrap(react, id="react", hooks=[Kinder()]).run("a?", user="u")
    assert result.answer == "noon"
    assert (told(model).status, told(model).content) == ("error", "the shelf will be back at noon")


# --------------------------------------------------------------------------- one retry policy
@pytest.mark.parametrize("graph_of", ["create_agent", "react"])
async def test_the_graphs_retry_policy_is_the_one_that_tries_the_call(
    harness: Harness, graph_of: str
) -> None:
    """A read whose error may pass (the shelf service refusing the connection) in a graph that
    retries it itself (``ToolRetryMiddleware(max_retries=2)``): the tool is tried as often as
    the graph says — 3 times — not each of the graph's tries tried again by the harness."""
    fails.extend([httpx.ConnectError("shelf service refused the connection")] * 3)
    model = ScriptedChatModel(turns=[("stock", {"sku": "a"}), "down"])
    retry = ToolRetryMiddleware(max_retries=2, initial_delay=0, jitter=False)
    graph = (
        create_agent(model, [stock], middleware=[retry])
        if graph_of == "create_agent"
        else ReAct(system_prompt="Count.", model=model, tools=[stock], middleware=[retry])
    )
    result = await harness.wrap(graph, id="shelf").run("a?", user="u")
    assert result.answer == "down" and ran == ["stock a"] * 3


@pytest.fixture
def _quick_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(base, "RETRY_BACKOFF_SECONDS", 0.001)


def refused(times: int) -> None:
    """The shelf service refuses the next ``times`` connections: an error that may pass."""
    fails.extend([httpx.ConnectError("shelf service refused the connection")] * times)


@pytest.mark.usefixtures("_quick_retries")
async def test_a_nodes_retry_policy_is_the_one_that_tries_the_call(harness: Harness) -> None:
    """A tool node with no wrapper whose node has a ``RetryPolicy(max_attempts=3)``: the
    graph runs the node again on the error, so the tool is called 3 times in all — the
    harness tries none of the node's runs again itself."""
    refused(10)
    model = ScriptedChatModel(turns=[("stock", {"sku": "a"}), "never"])
    bound = model.bind_tools([stock])

    async def think(state: Any) -> dict[str, Any]:
        return {"messages": [await bound.ainvoke(state["messages"])]}

    graph = StateGraph(MessagesState)
    graph.add_node("model", think)
    policy = RetryPolicy(max_attempts=3, initial_interval=0.0, jitter=False)
    graph.add_node("tools", ToolNode([stock]), retry_policy=policy)
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", tools_condition)
    graph.add_edge("tools", "model")
    result = await harness.wrap(graph.compile(), id="policy").run("a?", user="u")
    assert result.status is RunStatus.ERROR and ran == ["stock a"] * 3


def plain(model: Any) -> Any:
    return create_agent(model, [stock])


def worded_errors(model: Any) -> Any:
    def worded(exc: Exception, request: Any) -> str:
        return f"stock is unavailable ({exc})"

    return create_agent(model, [stock], middleware=[ToolErrorMiddleware(worded)])


def deep(model: Any) -> Any:
    return create_deep_agent(model=model, tools=[stock])


def react(model: Any) -> Any:
    return ReAct(system_prompt="Count.", model=model, tools=[stock])


@pytest.mark.usefixtures("_quick_retries")
@pytest.mark.parametrize("graph_of", [plain, worded_errors, deep, react], ids=lambda f: f.__name__)
async def test_where_the_graph_does_not_retry_its_tool_the_harnesss_retries_stand(
    harness: Harness, graph_of: Callable[[Any], Any]
) -> None:
    """A read whose error may pass, in a graph that does not try it again itself — no
    wrapper (``create_agent`` with no tool-call middleware), or wrappers that only handle the
    error: ``ToolErrorMiddleware``, Deep Agents' own (its file middleware wraps every call),
    ``ReAct``'s ``GovernTools`` over its own LangChain tool: the harness tries it again as it
    tries any read, 3 times in all, and the model reads the result."""
    refused(2)
    model = ScriptedChatModel(turns=[("stock", {"sku": "a"}), "7"])
    result = await harness.wrap(graph_of(model), id="shelf").run("a?", user="u")
    assert result.answer == "7" and ran == ["stock a"] * 3
    assert told(model).content == "a: 7"


# ------------------------------------------------------- the harness's tries: one call
def builders(saver: Any = None) -> list[Any]:
    """``create_agent``, Deep Agents and ``ReAct`` over the graph's own ``stock`` (and
    ``tools``), with ``saver`` as their checkpointer where one is given (ReAct keeps its own)."""

    def agent(model: Any, tools: list[Any]) -> Any:
        return create_agent(model, tools, checkpointer=saver)

    def deep_agent(model: Any, tools: list[Any]) -> Any:
        return create_deep_agent(model=model, tools=tools, checkpointer=saver)

    def react_agent(model: Any, tools: list[Any]) -> Any:
        return ReAct(system_prompt="Count.", model=model, tools=tools)

    return [agent, deep_agent, react_agent]


BUILT = ["create_agent", "deep agent", "react"]


@pytest.mark.usefixtures("_quick_retries")
@pytest.mark.parametrize("built", BUILT)
async def test_the_harnesss_tries_of_a_graphs_tool_are_one_call_of_the_run(
    memory_harness: Harness, memory_service: FakeMemoryService, built: str
) -> None:
    """The harness tries a read whose error may pass again around the graph's handling of the
    error — and the run sees one call, as when it tried it within the call: one start and one
    result on the stream (AG-UI's ``TOOL_CALL_*``), one record, one call of the run's
    ``max_tool_calls`` budget (a budget of two has room for it, and is not reached), and no
    stall."""
    refused(2)
    build = dict(zip(BUILT, builders(), strict=True))[built]
    model = ScriptedChatModel(turns=[("stock", {"sku": "a"}), "7"])
    agent = memory_harness.wrap(build(model, [stock]), id="one", budget=Budget(max_tool_calls=2))
    events = [e async for e in agent.stream("a?", user="u")]
    read = [m for m in model.seen[-1] if isinstance(m, ToolMessage)]
    assert ran == ["stock a"] * 3 and read[-1].content == "a: 7"
    starts = [e for e in events if e.type is RunEventType.TOOL_CALL_START]
    assert [e.data["tool"] for e in starts] == ["stock"]
    assert results(events) == [("stock", "ok")]
    assert not [e for e in events if e.type is RunEventType.RUN_ERROR]
    finished = events[-1]
    assert finished.outcome is not None and finished.outcome.value == "success"
    assert finished.data["spent"]["tool_calls"] == 1
    await memory_harness.writes.drain()
    [record] = memory_service.named("record_tool")
    assert record.body["status"] == "ok"


@pytest.mark.usefixtures("_quick_retries")
@pytest.mark.parametrize("built", BUILT)
async def test_the_harnesss_tries_never_stop_a_call_as_a_stall(
    harness: Harness, built: str
) -> None:
    """``MAX_REPEATS`` calls with the same arguments in a row are a model looping; the
    harness's own tries of one call are not: the same call in two model steps, each tried 3
    times, runs every time, and nothing is said of a stall."""
    refused(2)
    build = dict(zip(BUILT, builders(), strict=True))[built]
    model = ScriptedChatModel(turns=[("stock", {"sku": "a"}), ("stock", {"sku": "a"}), "7"])
    events = [
        e async for e in harness.wrap(build(model, [stock]), id="loop").stream("a?", user="u")
    ]
    assert ran == ["stock a"] * 4 and told(model).content == "a: 7"
    assert not [e for e in events if e.type is RunEventType.CUSTOM and "stall" in str(e.data)]
    assert results(events) == [("stock", "ok"), ("stock", "ok")]


class Asking(Hooks):
    """A rule of the run's own that asks a person before each call of ``stock``."""

    async def before_tool(self, call: Any) -> Ask | None:
        return Ask("Look the stock up?") if call.tool == "stock" else None


@pytest.mark.usefixtures("_quick_retries")
@pytest.mark.parametrize("built", BUILT)
async def test_a_call_that_asks_is_asked_about_once_across_the_harnesss_tries(
    harness: Harness, built: str
) -> None:
    """A read a hook asks a person about, approved, fails twice and is tried again: the
    approval covers its tries — one question for one call."""
    refused(2)
    build = dict(zip(BUILT, builders(InMemorySaver()), strict=True))[built]
    model = ScriptedChatModel(turns=[("stock", {"sku": "a"}), "7"])
    agent = harness.wrap(build(model, [stock]), id="asks", hooks=[Asking()])
    paused = await agent.run("a?", user="u", thread="t")
    assert paused.interrupt is not None and ran == []
    done = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="u")
    assert done.status is RunStatus.SUCCESS and done.interrupt is None, done.error
    assert ran == ["stock a"] * 3 and done.answer == "7"


class Watching(Hooks):
    """The run's ``on_error`` hooks: each failed call they are told of."""

    def __init__(self) -> None:
        self.failed: list[str] = []

    async def on_error(self, stage: Any, error: Exception) -> None:
        self.failed.append(f"{stage}: {error}")


@pytest.mark.usefixtures("_quick_retries")
@pytest.mark.parametrize("built", BUILT)
@pytest.mark.parametrize("refusals", [2, 3])
async def test_the_on_error_hooks_are_told_of_a_failed_call_not_of_its_failed_tries(
    harness: Harness, built: str, refusals: int
) -> None:
    """A call the harness tries again: the run's ``on_error`` hooks hear of its failure once —
    when its last try fails — and not at all when a try works, as when it tried it within the
    call."""
    refused(refusals)
    build = dict(zip(BUILT, builders(), strict=True))[built]
    model = ScriptedChatModel(turns=[("stock", {"sku": "a"}), "done"])
    watching = Watching()
    graph = (  # create_agent's run goes on past the error: its wrapper answers it
        create_agent(model, [stock], middleware=[ToolErrorMiddleware(lambda e, r: str(e))])
        if built == "create_agent"
        else build(model, [stock])
    )
    await harness.wrap(graph, id="hooked", hooks=[watching]).run("a?", user="u")
    assert ran == ["stock a"] * 3
    said = [f for f in watching.failed if f.startswith("tool")]
    assert said == ([] if refusals == 2 else ["tool: shelf service refused the connection"])


paid: list[int] = []


@lc_tool
def pay_now(amount: int) -> str:
    """Pay."""
    paid.append(amount)
    return f"paid {amount}"


pay_now.metadata = {"side_effects": "irreversible"}


@pytest.mark.usefixtures("_quick_retries")
@pytest.mark.parametrize("built", BUILT)
async def test_a_call_tried_again_beside_a_pause_is_not_run_again_on_the_resume(
    harness: Harness, built: str
) -> None:
    """A read that fails twice runs beside a payment that asks: its tries end — the third
    working, journaled — before the run pauses; the resume replays it and pays once. The
    run's budget counts each call once: the read, and the payment once it ran."""
    refused(2)
    paid.clear()
    build = dict(zip(BUILT, builders(InMemorySaver()), strict=True))[built]
    model = ScriptedChatModel(turns=[[("stock", {"sku": "a"}), ("pay_now", {"amount": 5})], "done"])
    agent = harness.wrap(build(model, [stock, pay_now]), id="beside")
    paused = await agent.run("a? and pay 5", user="u", thread="t")
    assert paused.interrupt is not None and ran == ["stock a"] * 3 and paid == []
    journal = await journal_of(harness.runs, await harness.runs.get(paused.run_id))
    assert journal.spent["tool_calls"] == 1  # the read, once; the payment not yet
    done = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="u")
    assert done.status is RunStatus.SUCCESS and done.answer == "done", done.error
    assert ran == ["stock a"] * 3 and paid == [5]


# --------------------------------------------------------------------------- one question
charged: list[str] = []


@lc_tool
def charge(amount: int) -> str:
    """Charge the card."""
    if not charged:
        charged.append(f"charge {amount} failed")
        raise ValueError("the card network blinked")
    charged.append(f"charge {amount}")
    return f"charged {amount}"


charge.metadata = {"side_effects": "irreversible"}


async def test_a_retry_of_an_approved_call_is_not_asked_about_again(harness: Harness) -> None:
    """A person approves an irreversible call; it fails and the graph's retry tries it again
    (the same call of the same model step): the approval covers the retry — one call, one
    question. The same call made by the model's next step is a call of its own: asked."""
    charged.clear()
    model = ScriptedChatModel(
        turns=[("charge", {"amount": 5}), ("charge", {"amount": 5}), "charged twice"]
    )
    graph = create_agent(model, [charge], middleware=[retrying(1)], checkpointer=InMemorySaver())
    agent = harness.wrap(graph, id="card")
    paused = await agent.run("charge 5, twice", user="u", thread="t")
    assert paused.interrupt is not None and charged == []
    again = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="u")
    assert charged == ["charge 5 failed", "charge 5"]  # retried, unasked
    assert again.status is RunStatus.PAUSED and again.interrupt is not None  # the next step's
    done = await agent.resume(again.interrupt.interrupt_id, "approve", reviewer="u")
    assert done.status is RunStatus.SUCCESS and done.answer == "charged twice"
    assert charged == ["charge 5 failed", "charge 5", "charge 5"]
