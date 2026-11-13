"""A tool of the graph's own that raises: the harness records the failed call, and what the
graph does with the error is the library's — LangChain's ``ToolRetryMiddleware`` tries the
call again, ``ToolErrorMiddleware`` words what the model reads, a wrapper of the builder's
answers in the tool's place, a ``ToolNode``'s ``handle_tool_errors`` gives its text, exactly
as the graph words it. An error nothing of the graph handles fails the run, as it fails the
graph without the harness — unless the graph runs the harness's own middleware (``ReAct``,
``middleware()``: ``GovernTools``), whose rule is the harness's: the model reads the failure."""

from __future__ import annotations

from typing import Any

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import (
    ToolErrorMiddleware,
    ToolRetryMiddleware,
    wrap_tool_call,
)
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool as lc_tool
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from tests.support.chat_model import ScriptedChatModel
from tests.support.memory import FakeMemoryService
from trellis import Harness, ReAct
from trellis.contracts import RunEventType, RunStatus, ToolOutcome, ToolStatus
from trellis.harness.hooks import Hooks
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
