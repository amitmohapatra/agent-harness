"""LangGraph's own features, each run through the harness as the library gives it: a node's
``Command`` (``goto``, ``update``) and a ``Command`` as the run's input, ``Send`` branches,
sub-graphs, two interrupts in one step, a static breakpoint, retry, cache and timeout policies,
the recursion limit, the functional API, time travel, a store of the graph's own — and the tool
node's: what a tool is handed (``InjectedState``, ``InjectedStore``, ``ToolRuntime``), a tool
that returns a ``Command``, arguments that do not fit. Each test says what the library does and
what the harness still does around it: the calls recorded and governed, the run's events and
end, a pause, a cancel. (A tool's ``Command`` across a pause: ``test_tool_updates``; a tool
that raises: ``test_tool_errors_library``.)"""

from __future__ import annotations

import asyncio
import operator
from typing import Annotated, Any, NotRequired, TypedDict

import pytest
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool as lc_tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import InjectedState, InjectedStore, ToolNode, ToolRuntime, tools_condition
from langgraph.store.memory import InMemoryStore
from langgraph.types import CachePolicy, Command, RetryPolicy, Send, TimeoutPolicy, interrupt

from tests.support.chat_model import ScriptedChatModel
from tests.support.memory import FakeMemoryService
from trellis import Harness, current, tool
from trellis.contracts import RunEventType, RunStatus
from trellis.harness.tools.bridge import MAX_REPEATS, STALLED

ran: list[str] = []


@pytest.fixture(autouse=True)
def _reset() -> None:
    ran.clear()


@tool(side_effects="read")
async def count(sku: str) -> int:
    """Count a SKU."""
    ran.append(f"count {sku}")
    await asyncio.sleep(0)
    return len(sku)


def results(events: list[Any]) -> list[tuple[str, str]]:
    """The tool calls a stream reported, in order: each tool's name and how it ended."""
    return [
        (e.data["tool"], e.data["status"])
        for e in events
        if e.type is RunEventType.TOOL_CALL_RESULT
    ]


def by_hand(model: Any, tools: list[Any], *, state: Any = MessagesState, **compile: Any) -> Any:
    """A graph built by hand: a model node, ``ToolNode(tools)``, ``tools_condition``."""
    bound = model.bind_tools(tools)

    async def think(state: Any) -> dict[str, Any]:
        return {"messages": [await bound.ainvoke(state["messages"])]}

    graph = StateGraph(state)
    graph.add_node("model", think)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", tools_condition)
    graph.add_edge("tools", "model")
    return graph.compile(**compile)


async def state_of(harness: Harness, graph: Any, thread: str) -> dict[str, Any]:
    key = await harness.conversation(user="u", thread=thread)
    return (await graph.aget_state({"configurable": {"thread_id": key}})).values


# ------------------------------------------------------------------ a node's Command, Send
class Route(TypedDict):
    topic: str
    path: NotRequired[Annotated[list[str], operator.add]]
    done: NotRequired[str]


def routed() -> Any:
    def triage(state: Route) -> Command[Any]:
        where = "urgent" if "fire" in state["topic"] else "routine"
        return Command(update={"path": ["triage"]}, goto=where)

    def urgent(state: Route) -> dict[str, Any]:
        return {"path": ["urgent"], "done": "paged"}

    def routine(state: Route) -> dict[str, Any]:
        return {"path": ["routine"], "done": "filed"}

    graph = StateGraph(Route)
    graph.add_node("triage", triage, destinations=("urgent", "routine"))
    graph.add_node("urgent", urgent)
    graph.add_node("routine", routine)
    graph.add_edge(START, "triage")
    return graph.compile(checkpointer=InMemorySaver())


async def test_a_nodes_command_updates_the_state_and_goes_where_it_says(harness: Harness) -> None:
    graph = routed()
    agent = harness.wrap(graph, id="triage")
    fire = await agent.run({"topic": "fire in bay 2"}, user="u", thread="fire")
    calm = await agent.run({"topic": "a new label"}, user="u", thread="calm")
    assert fire.status is RunStatus.SUCCESS and calm.status is RunStatus.SUCCESS
    assert (await state_of(harness, graph, "fire"))["path"] == ["triage", "urgent"]
    assert (await state_of(harness, graph, "calm"))["path"] == ["triage", "routine"]


async def test_a_command_as_the_runs_input_is_the_graphs_as_ainvoke_takes_it(
    harness: Harness,
) -> None:
    """A run's input may be LangGraph's ``Command`` (an ``update``, a ``goto``): the graph is
    given it as it is, and the thread ends as ``graph.ainvoke(Command(...))`` leaves one."""

    def given() -> Command[Any]:
        return Command(update={"topic": "a drill", "path": ["by hand"]}, goto="urgent")

    alone = routed()
    expected = await alone.ainvoke(given(), {"configurable": {"thread_id": "alone"}})
    assert "urgent" in expected["path"] and "by hand" in expected["path"]
    graph = routed()
    result = await harness.wrap(graph, id="triage").run(given(), user="u", thread="t")
    assert result.status is RunStatus.SUCCESS
    assert await state_of(harness, graph, "t") == expected


class Fan(TypedDict):
    skus: list[str]
    counted: Annotated[list[str], operator.add]
    answer: NotRequired[str]


def fan(state: Fan) -> list[Send]:
    return [Send("one", {"sku": sku}) for sku in state["skus"]]


async def test_send_branches_run_side_by_side_and_their_calls_are_the_runs(
    harness: Harness,
) -> None:
    """``Send`` runs one branch per item, at once; each branch's harness call is the run's own:
    governed, on its events, counted once."""

    async def one(state: Any) -> dict[str, Any]:
        runtime = current()
        assert runtime is not None
        await runtime.tools.use(count)
        found = await runtime.tools.call("count", sku=state["sku"])
        return {"counted": [f"{state['sku']}={found}"]}

    def gather(state: Fan) -> dict[str, Any]:
        return {"answer": ", ".join(sorted(state["counted"]))}

    graph = StateGraph(Fan)
    graph.add_node("one", one)
    graph.add_node("gather", gather)
    graph.add_conditional_edges(START, fan, ["one"])
    graph.add_edge("one", "gather")
    graph.add_edge("gather", END)
    compiled = graph.compile(checkpointer=InMemorySaver())
    agent = harness.wrap(compiled, id="fan")
    given = {"skus": ["a", "bb", "ccc"], "counted": []}
    events = [e async for e in agent.stream(given, user="u", thread="t")]
    assert sorted(ran) == ["count a", "count bb", "count ccc"]
    assert results(events) == [("count", "ok")] * 3
    assert events[-1].type is RunEventType.RUN_FINISHED and events[-1].error is None
    assert events[-1].data["spent"]["tool_calls"] == 3
    assert (await state_of(harness, compiled, "t"))["answer"] == "a=1, bb=2, ccc=3"


async def test_a_run_cancelled_inside_send_branches_stops_every_branch(harness: Harness) -> None:
    started = asyncio.Event()
    stopped: list[str] = []

    async def one(state: Any) -> None:
        started.set()
        try:
            await asyncio.sleep(30)
        finally:
            stopped.append(state["sku"])

    graph = StateGraph(Fan)
    graph.add_node("one", one)
    graph.add_conditional_edges(START, fan, ["one"])
    graph.add_edge("one", END)
    agent = harness.wrap(graph.compile(), id="fan")
    running = asyncio.ensure_future(agent.run({"skus": ["a", "b"], "counted": []}, user="u"))
    await asyncio.wait_for(started.wait(), 5)
    [record] = [r async for r in harness.runs.iterate(agent_id="fan")]
    await agent.cancel(record.run_id, reason="enough")
    result = await running
    assert result.status is RunStatus.CANCELLED
    assert sorted(stopped) == ["a", "b"]


# ------------------------------------------------------------------------------ sub-graphs
class Inner(TypedDict):
    topic: str
    title: NotRequired[str]


async def test_a_sub_graphs_interrupt_pauses_the_run_and_its_answer_reaches_the_sub_graph(
    harness: Harness,
) -> None:
    def ask(state: Inner) -> dict[str, Any]:
        return {"title": interrupt({"question": f"Title for {state['topic']}?"})}

    inner = StateGraph(Inner)
    inner.add_node("ask", ask)
    inner.add_edge(START, "ask")
    inner.add_edge("ask", END)

    def close(state: Inner) -> dict[str, Any]:
        return {"title": f"{state.get('title')}!"}

    outer = StateGraph(Inner)
    outer.add_node("inner", inner.compile())
    outer.add_node("close", close)
    outer.add_edge(START, "inner")
    outer.add_edge("inner", "close")
    outer.add_edge("close", END)
    graph = outer.compile(checkpointer=InMemorySaver())
    agent = harness.wrap(graph, id="titled")
    paused = await agent.run({"topic": "tides"}, user="u", thread="t")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
    assert paused.interrupt.question == "Title for tides?"
    done = await agent.resume(paused.interrupt.interrupt_id, "answer", answer="Moon", reviewer="u")
    assert done.status is RunStatus.SUCCESS
    assert (await state_of(harness, graph, "t"))["title"] == "Moon!"


async def test_the_text_of_a_model_inside_a_sub_graph_is_streamed(harness: Harness) -> None:
    model = ScriptedChatModel(turns=["hello from inside"])

    async def think(state: MessagesState) -> dict[str, Any]:
        return {"messages": [await model.ainvoke(state["messages"])]}

    inner = StateGraph(MessagesState)
    inner.add_node("think", think)
    inner.add_edge(START, "think")
    inner.add_edge("think", END)
    outer = StateGraph(MessagesState)
    outer.add_node("inner", inner.compile())
    outer.add_edge(START, "inner")
    outer.add_edge("inner", END)
    events = [e async for e in harness.wrap(outer.compile(), id="nested").stream("hi", user="u")]
    text = [e.data["delta"] for e in events if e.type is RunEventType.TEXT_MESSAGE_CONTENT]
    assert text == ["hello from inside"] and events[-1].data["result"] == "hello from inside"


class Two(TypedDict):
    a: NotRequired[str]
    b: NotRequired[str]


async def test_two_interrupts_raised_in_one_step_are_asked_one_after_the_other(
    harness: Harness,
) -> None:
    """Two nodes of one step each call ``interrupt``: LangGraph holds both, and resumes each by
    its id. The run asks one, then the other, and each node gets its own answer."""

    def left(state: Two) -> dict[str, Any]:
        return {"a": interrupt({"question": "left?"})}

    def right(state: Two) -> dict[str, Any]:
        return {"b": interrupt({"question": "right?"})}

    graph = StateGraph(Two)
    graph.add_node("left", left)
    graph.add_node("right", right)
    graph.add_edge(START, "left")
    graph.add_edge(START, "right")
    compiled = graph.compile(checkpointer=InMemorySaver())
    agent = harness.wrap(compiled, id="two")
    answers = {"left?": "L", "right?": "R"}
    asked: list[str] = []
    result = await agent.run({}, user="u", thread="t")
    for _ in answers:
        assert result.status is RunStatus.PAUSED and result.interrupt is not None
        asked.append(result.interrupt.question)
        result = await agent.resume(
            result.interrupt.interrupt_id, "answer", answer=answers[asked[-1]], reviewer="u"
        )
    assert result.status is RunStatus.SUCCESS and sorted(asked) == sorted(answers)
    assert await state_of(harness, compiled, "t") == {"a": "L", "b": "R"}


class Tried(TypedDict):
    n: NotRequired[int]
    out: NotRequired[str]


async def test_a_static_breakpoint_stops_the_graph_and_a_run_given_nothing_goes_on(
    harness: Harness,
) -> None:
    """``interrupt_before`` stops the graph before a node with nothing asked: the run ends
    where the graph stopped, and a run of the thread given no input (``None``, as
    ``graph.ainvoke(None, config)``) goes on from there."""

    def write(state: Tried) -> dict[str, Any]:
        return {"n": 1}

    def publish(state: Tried) -> dict[str, Any]:
        return {"out": "published"}

    graph = StateGraph(Tried)
    graph.add_node("write", write)
    graph.add_node("publish", publish)
    graph.add_edge(START, "write")
    graph.add_edge("write", "publish")
    graph.add_edge("publish", END)
    compiled = graph.compile(checkpointer=InMemorySaver(), interrupt_before=["publish"])
    agent = harness.wrap(compiled, id="staged")
    stopped = await agent.run({}, user="u", thread="t")
    assert stopped.status is RunStatus.SUCCESS and stopped.interrupt is None
    assert await state_of(harness, compiled, "t") == {"n": 1}
    again = await agent.run(None, user="u", thread="t")
    assert again.status is RunStatus.SUCCESS
    assert await state_of(harness, compiled, "t") == {"n": 1, "out": "published"}


# -------------------------------------------------------------------- retries, cache, limit
def retried(attempts: int, tries: list[Any]) -> Any:
    """A graph of one node that fails until its last attempt, after a harness call."""

    async def flaky(state: Tried) -> dict[str, Any]:
        runtime = current()
        assert runtime is not None
        await runtime.tools.use(count)
        tries.append(await runtime.tools.call("count", sku="a"))
        if len(tries) < attempts:
            raise ConnectionError("try again")
        return {"out": f"{tries[-1]} after {len(tries)} tries"}

    policy = RetryPolicy(max_attempts=attempts, initial_interval=0.0, jitter=False)
    graph = StateGraph(Tried)
    graph.add_node("flaky", flaky, retry_policy=policy)
    graph.add_edge(START, "flaky")
    graph.add_edge("flaky", END)
    return graph.compile(checkpointer=InMemorySaver())


async def test_a_nodes_retry_policy_runs_it_again_with_the_calls_it_makes(harness: Harness) -> None:
    """A node that fails is run again as its ``RetryPolicy`` says, from its start: the harness
    call it makes is made again, as any of the node's code."""
    tries: list[Any] = []
    graph = retried(2, tries)
    result = await harness.wrap(graph, id="flaky").run({}, user="u", thread="t")
    assert result.status is RunStatus.SUCCESS and tries == [1, 1]
    assert ran == ["count a", "count a"]
    assert (await state_of(harness, graph, "t"))["out"] == "1 after 2 tries"


async def test_today_a_node_run_a_third_time_has_its_same_call_refused_as_a_stall(
    harness: Harness,
) -> None:
    """The harness stops a call made ``MAX_REPEATS`` times in a row with the same arguments (a
    model looping). A node its ``RetryPolicy`` runs that many times makes its call that many
    times too: the last is refused, and the node is handed the refusal as the call's result."""
    tries: list[Any] = []
    graph = retried(MAX_REPEATS, tries)
    result = await harness.wrap(graph, id="flaky").run({}, user="u", thread="t")
    assert result.status is RunStatus.SUCCESS
    assert ran == ["count a"] * (MAX_REPEATS - 1)
    assert tries[-1] == STALLED.format(tool="count", times=MAX_REPEATS)


async def test_a_nodes_cache_policy_answers_the_same_input_from_the_cache(harness: Harness) -> None:
    from langgraph.cache.memory import InMemoryCache

    def slow(state: Tried) -> dict[str, Any]:
        ran.append(f"slow {state.get('n')}")
        return {"out": f"n={state.get('n')}"}

    graph = StateGraph(Tried)
    graph.add_node("slow", slow, cache_policy=CachePolicy(ttl=60))
    graph.add_edge(START, "slow")
    graph.add_edge("slow", END)
    agent = harness.wrap(graph.compile(cache=InMemoryCache()), id="cached")
    for n in (1, 1, 2):
        assert (await agent.run({"n": n}, user="u")).status is RunStatus.SUCCESS
    assert ran == ["slow 1", "slow 2"]


async def test_a_node_past_its_timeout_policy_ends_the_run_saying_so(harness: Harness) -> None:
    async def slow(state: Tried) -> None:
        await asyncio.sleep(30)

    graph = StateGraph(Tried)
    graph.add_node("slow", slow, timeout=TimeoutPolicy(run_timeout=0.05))
    graph.add_edge(START, "slow")
    graph.add_edge("slow", END)
    result = await harness.wrap(graph.compile(), id="slow").run({}, user="u")
    assert result.status is RunStatus.ERROR and result.error is not None
    assert "Node 'slow' exceeded its run timeout" in result.error.message


async def test_a_graph_past_its_recursion_limit_ends_the_run_saying_so(harness: Harness) -> None:
    def again(state: Tried) -> dict[str, Any]:
        return {"n": (state.get("n") or 0) + 1}

    graph = StateGraph(Tried)
    graph.add_node("again", again)
    graph.add_edge(START, "again")
    graph.add_edge("again", "again")
    agent = harness.wrap(graph.compile(), id="loop", framework_options={"recursion_limit": 4})
    result = await agent.run({}, user="u")
    assert result.status is RunStatus.ERROR and result.error is not None
    assert "Recursion limit of 4" in result.error.message


# ------------------------------------------------------------------------- the functional API
async def test_an_entrypoint_and_its_tasks_run_and_pause_as_a_graph_does(harness: Harness) -> None:
    from langgraph.func import entrypoint, task

    @task
    def draft(topic: str) -> str:
        ran.append(f"draft {topic}")
        return f"about {topic}"

    @entrypoint(checkpointer=InMemorySaver())
    def workflow(given: dict[str, Any]) -> dict[str, Any]:
        text = draft(given["topic"]).result()
        verdict = interrupt({"question": f"Publish '{text}'?"})
        return {"text": text, "verdict": verdict}

    agent = harness.wrap(workflow, id="workflow")
    paused = await agent.run({"topic": "tides"}, user="u", thread="t")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
    assert paused.interrupt.question == "Publish 'about tides'?"
    done = await agent.resume(paused.interrupt.interrupt_id, "answer", answer="yes", reviewer="u")
    assert done.status is RunStatus.SUCCESS
    assert ran == ["draft tides"]  # the task's result came from its checkpoint, not a second run
    assert await state_of(harness, workflow, "t") == {"text": "about tides", "verdict": "yes"}


# ------------------------------------------------------------------------------ time travel
async def test_a_checkpoint_of_the_thread_is_forked_and_run_on_from(harness: Harness) -> None:
    """LangGraph's time travel on the run's thread: its history is read with the conversation's
    key, an earlier checkpoint is forked with ``update_state``, and a run given that
    ``checkpoint_id`` (the graph's own config key) and no input goes on from the fork."""

    def write(state: Tried) -> dict[str, Any]:
        ran.append(f"write {state.get('n')}")
        return {"out": f"wrote {state.get('n')}"}

    def publish(state: Tried) -> dict[str, Any]:
        ran.append(f"publish {state.get('out')}")
        return {}

    graph = StateGraph(Tried)
    graph.add_node("write", write)
    graph.add_node("publish", publish)
    graph.add_edge(START, "write")
    graph.add_edge("write", "publish")
    graph.add_edge("publish", END)
    compiled = graph.compile(checkpointer=InMemorySaver())
    agent = harness.wrap(compiled, id="travel")
    assert (await agent.run({"n": 1}, user="u", thread="t")).status is RunStatus.SUCCESS
    key = await harness.conversation(user="u", thread="t")
    config: RunnableConfig = {"configurable": {"thread_id": key}}
    history = [s async for s in compiled.aget_state_history(config)]
    before_publish = next(s for s in history if s.next == ("publish",))
    forked = await compiled.aupdate_state(before_publish.config, {"out": "edited"})
    again = await agent.run(
        None,
        user="u",
        thread="t",
        framework_options={
            "configurable": {"checkpoint_id": forked.get("configurable", {})["checkpoint_id"]}
        },
    )
    assert again.status is RunStatus.SUCCESS
    assert ran == ["write 1", "publish wrote 1", "publish edited"]


# ----------------------------------------------------------------- what a node writes to see
async def test_today_what_a_node_writes_to_the_custom_stream_is_not_on_the_runs_events(
    harness: Harness,
) -> None:
    """LangGraph's ``get_stream_writer()`` writes to the ``custom`` stream mode of
    ``graph.astream``. The run's stream is the harness's events, read from the graph's
    ``messages`` and ``values`` modes: what a node writes there is on none of them (the
    harness's own way is ``trellis.current().events``)."""

    def step(state: Tried) -> dict[str, Any]:
        get_stream_writer()({"progress": "half way"})
        return {"n": 1}

    graph = StateGraph(Tried)
    graph.add_node("step", step)
    graph.add_edge(START, "step")
    graph.add_edge("step", END)
    events = [e async for e in harness.wrap(graph.compile(), id="writes").stream({}, user="u")]
    assert [e.type for e in events] == [RunEventType.RUN_STARTED, RunEventType.RUN_FINISHED]


# ---------------------------------------------------------------------- what a tool is handed
async def test_a_graphs_own_store_is_what_its_tools_are_handed(harness: Harness) -> None:
    @lc_tool
    def remember(key: str, value: str, store: Annotated[Any, InjectedStore()]) -> str:
        """Keep a value."""
        store.put(("notes",), key, {"value": value})
        return "kept"

    remember.metadata = {"side_effects": "read"}
    own = InMemoryStore()
    model = ScriptedChatModel(turns=[("remember", {"key": "k", "value": "v"}), "kept it"])
    agent = harness.wrap(by_hand(model, [remember], store=own), id="keeper")
    result = await agent.run("keep k=v", user="u")
    assert result.answer == "kept it"
    item = own.get(("notes",), "k")
    assert item is not None and item.value == {"value": "v"}


class Desk(MessagesState):
    desk: NotRequired[str]


async def test_a_tool_is_handed_the_state_and_the_runtime_and_the_run_records_neither(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    """``InjectedState`` and ``ToolRuntime`` reach the tool as LangGraph gives them; what the
    harness governs and records are the arguments the model gave, nothing injected."""
    handed: dict[str, Any] = {}

    @lc_tool
    def where(sku: str, state: Annotated[dict[str, Any], InjectedState]) -> str:
        """Where a SKU is."""
        handed["state"] = state.get("desk")
        return f"{sku} at {state.get('desk')}"

    @lc_tool
    def who(sku: str, desk: Annotated[str, InjectedState("desk")], runtime: ToolRuntime) -> str:
        """Who has a SKU."""
        handed["field"] = desk
        handed["call"] = runtime.tool_call_id
        handed["runtime"] = runtime.state.get("desk")
        return f"{sku}: nobody"

    where.metadata = who.metadata = {"side_effects": "read"}
    model = ScriptedChatModel(
        turns=[[("where", {"sku": "a"}), ("who", {"sku": "a"})], "a is at north"]
    )
    agent = memory_harness.wrap(by_hand(model, [where, who], state=Desk), id="desk")
    given = {"messages": [{"role": "user", "content": "where?"}], "desk": "north"}
    events = [e async for e in agent.stream(given, user="u")]
    assert sorted(results(events)) == [("where", "ok"), ("who", "ok")]
    assert handed == {"state": "north", "field": "north", "call": "call_1_1", "runtime": "north"}
    await memory_harness.writes.drain()
    recorded = {c.body["tool"]: c.body["args"] for c in memory_service.named("record_tool")}
    assert recorded == {"where": {"sku": "a"}, "who": {"sku": "a"}}


# ------------------------------------------------------------------ a tool returning a Command
class Cart(MessagesState):
    cart: NotRequired[Annotated[list[str], operator.add]]


@pytest.mark.usefixtures("writes_unasked")  # not about the write's approval
async def test_a_tools_command_updates_the_graphs_state_and_is_a_call_of_the_run(
    harness: Harness,
) -> None:
    @lc_tool
    def add(sku: str, runtime: ToolRuntime) -> Command[Any]:
        """Add a SKU to the cart."""
        said = ToolMessage(f"added {sku}", tool_call_id=runtime.tool_call_id or "")
        return Command(update={"cart": [sku], "messages": [said]})

    add.metadata = {"side_effects": "write"}
    model = ScriptedChatModel(turns=[("add", {"sku": "a"}), "added"])
    compiled = by_hand(model, [add], state=Cart, checkpointer=InMemorySaver())
    events = [e async for e in harness.wrap(compiled, id="cart").stream("a", user="u", thread="t")]
    assert results(events) == [("add", "ok")]
    assert (await state_of(harness, compiled, "t"))["cart"] == ["a"]
    assert model.seen[-1][-1].content == "added a"


async def test_arguments_that_do_not_fit_a_tools_schema_are_told_to_the_model(
    harness: Harness,
) -> None:
    @lc_tool
    def shelve(n: int) -> str:
        """Shelve n units."""
        ran.append(f"shelve {n}")
        return str(n)

    shelve.metadata = {"side_effects": "read"}
    model = ScriptedChatModel(turns=[("shelve", {"n": "many"}), ("shelve", {"n": 3}), "3"])
    result = await harness.wrap(by_hand(model, [shelve]), id="shelver").run("shelve", user="u")
    assert result.answer == "3" and ran == ["shelve 3"]
    refused = model.seen[1][-1]
    assert isinstance(refused, ToolMessage)
    assert refused.status == "error" and "n" in str(refused.content)
