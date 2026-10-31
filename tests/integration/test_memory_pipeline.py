"""Memory in the pipeline, against the in-process memory service: who the key says the
deployment is, push (the context as a system message, the tool hints narrowing the tools),
pull (the service's agent tools), the background records, the run's outcome, the sampled
grounding check and people's feedback."""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from datetime import UTC, datetime
from typing import Any, cast

import httpx
import pytest
import respx
from agents import Agent as OpenAIAgent
from langchain.agents import create_agent

from tests.support.arm import arm, native_tools
from tests.support.chat_model import ScriptedChatModel
from tests.support.memory import MEMORY_TOOLS, FakeMemoryService
from tests.support.models import ScriptedChat
from tests.support.openai_model import ScriptedModel
from tests.unit.test_toolbox import Def
from tests.unit.test_toolbox import Gateway as StubGateway
from trellis import Harness, ReAct, Runtime, Settings, tool
from trellis.contracts import ConfigurationError, RunEventType, RunStatus
from trellis.harness import snapshots, telemetry
from trellis.harness.clients.bifrost import CODE_MODE_TOOLS, Gateway
from trellis.harness.clients.memory import RECORDED_OUTPUT_MAX_BYTES
from trellis.harness.governance import catalog
from trellis.harness.identity import Identity, conversation_key
from trellis.memory.models import ToolHints


@tool(side_effects="read")
def stock(sku: str) -> int:
    """Units in stock."""
    return 7


def many(n: int) -> list[Any]:
    """``n`` read tools: a toolbox large enough for tool hints."""

    def make(i: int) -> Any:
        def lookup(key: str) -> str:
            return f"t{i}:{key}"

        return tool(lookup, name=f"t{i}", side_effects="read", description=f"Tool {i}.")

    return [make(i) for i in range(n)]


def graph_tools(n: int) -> list[Any]:
    """``n`` LangChain tools of a graph's own (the framework's: the harness narrows them)."""
    from langchain_core.tools import StructuredTool

    def make(i: int) -> Any:
        def lookup(key: str) -> str:
            return f"t{i}:{key}"

        return StructuredTool.from_function(lookup, name=f"t{i}", description=f"Tool {i}.")

    return [make(i) for i in range(n)]


def harness_with(service: FakeMemoryService, **settings: Any) -> Harness:
    return Harness(config=Settings(**settings), memory=service.client())


# --------------------------------------------------------------------------- who we are
async def test_the_tenant_is_the_keys_own(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    async def fn(input: str, agent: Runtime) -> str:
        return agent.tenant

    result = await memory_harness.wrap(fn, id="t").run("x", user="u")
    assert result.answer == "acme"
    assert len(memory_service.named("key")) == 1  # asked once
    await memory_harness.wrap(fn, id="t2").run("y", user="u")
    assert len(memory_service.named("key")) == 1
    with pytest.raises(ConfigurationError, match="speaks for 'acme'"):
        await memory_harness.wrap(fn, id="t3").run("x", user="u", tenant="globex")


async def test_a_platform_key_names_the_tenant_per_call(memory_service: FakeMemoryService) -> None:
    memory_service.tenant = None

    async def fn(input: str, agent: Runtime) -> str:
        return agent.tenant

    async with harness_with(memory_service) as h:
        agent = h.wrap(fn, id="p")
        with pytest.raises(ConfigurationError, match="platform key"):
            await agent.run("x", user="u")
        assert (await agent.run("x", user="u", tenant="globex")).answer == "globex"


def test_agent_runs_needs_the_memory_services_keys() -> None:
    with pytest.raises(ConfigurationError, match="MEMORY_URL"):
        Harness(config=Settings(runs_url="http://runs"))


# --------------------------------------------------------------------------- push
async def test_push_injects_the_context_as_a_system_message(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    model = ScriptedChat(["7 units"])
    agent = memory_harness.wrap(
        ReAct(system_prompt="You answer stock questions.", model=model), id="s"
    )
    result = await agent.run("how many a?", user="u1", thread="t1")
    assert result.answer == "7 units"
    instructions, context = model.requests[0]["messages"][:2]
    assert instructions == {"role": "system", "content": "You answer stock questions."}
    assert context == {"role": "system", "content": memory_service.context_text}
    [call] = memory_service.named("context")
    assert call.scope["user_id"] == "u1"
    assert call.scope["thread_id"] == conversation_key("acme", "u1", "t1")  # u1's own "t1"
    assert call.body["query"] == "how many a?" and call.body["token_budget"] == 2000
    assert call.body["window"] is True  # ReAct keeps no conversation of its own
    assert "tools" not in call.body  # no tools of its own: nothing learned to offer
    assert memory_service.named("tool_hints") == []


async def test_a_small_toolbox_gets_its_learned_skills_and_every_tool(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    """Below five tools the context still carries what the agent learned for the task; no
    tool hints, so the model is offered every tool."""
    memory_service.learned_skills = "- refund-an-order: t0 -> t1 (worked 100% of 3 runs)"
    model = ScriptedChat(["done"])
    agent = arm(memory_harness.wrap(ReAct(system_prompt="s", model=model), id="refunds"), many(2))
    result = await agent.run("refund order O-1", user="u")
    assert result.status is RunStatus.SUCCESS, result.error
    [context] = memory_service.named("context")
    assert context.body["tools"] == {"available": ["t0", "t1"], "k": 8}
    pushed = model.requests[0]["messages"][1]["content"]
    assert "## Learned skills for this task\n- refund-an-order: t0 -> t1" in pushed
    assert "## Tools" not in pushed
    offered = [t["function"]["name"] for t in model.requests[0]["tools"]]
    assert {"t0", "t1"} <= set(offered), "no hints: every tool offered"


async def test_a_memory_outage_is_a_warning_not_a_failure(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    memory_service.fail = {"context", "messages"}

    async def fn(input: str, agent: Runtime) -> str:
        return "fine"

    events = [e async for e in memory_harness.wrap(fn, id="o").stream("x", user="u")]
    warnings = [
        e.data for e in events if e.type is RunEventType.CUSTOM and e.data["name"] == "warning"
    ]
    assert warnings[0]["code"] == "memory_unavailable"
    assert events[-1].type is RunEventType.RUN_FINISHED
    await memory_harness.writes.drain()
    assert memory_harness.writes.failed == 1  # the transcript write, reported and counted


# --------------------------------------------------------------------------- tool hints
async def test_from_five_tools_the_hints_narrow_what_react_is_offered_per_call(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    memory_service.candidates = ["t3", "t1"]
    memory_service.candidates_for = {"archive it": ["t5"]}
    model = ScriptedChat([("t3", {"key": "a"}), ("tool_search", {"task": "archive it"}), "done"])
    agent = arm(memory_harness.wrap(ReAct(system_prompt="s", model=model), id="n"), many(6))
    result = await agent.run("look up a", user="u")
    assert result.status is RunStatus.SUCCESS, result.error

    [context] = memory_service.named("context")
    assert context.body["tools"] == {"available": [f"t{i}" for i in range(6)], "k": 8}
    assert "## Tools" in model.requests[0]["messages"][1]["content"]  # confidence/args/missing
    offered = [[t["function"]["name"] for t in r["tools"]] for r in model.requests]
    memory_tools = list(MEMORY_TOOLS)
    # the candidates and the memory tools, never all six, sorted by name
    assert offered[0] == sorted(["t1", "t3", *memory_tools])
    assert offered[1] == sorted(["t1", "t3", *memory_tools])
    # tool_search found t5 among the run's own tools: offered from the next call on, after
    # the others (the order they were first offered: a prompt cache keeps the prefix)
    assert offered[2] == [*sorted(["t1", "t3", *memory_tools]), "t5"]


async def test_hints_below_the_floor_narrow_nothing(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    """B13: hints that match nothing (an index not built yet: confidence about 0) narrowed the
    offer to the toolbox's first names. Below ``NARROW_FLOOR`` every tool is offered, with
    ``tool_search`` among the memory tools."""
    memory_service.candidates = ["t3"]
    memory_service.top_confidence = 0.1
    model = ScriptedChat(["done"])
    agent = arm(memory_harness.wrap(ReAct(system_prompt="s", model=model), id="n"), many(6))
    result = await agent.run("look up a", user="u")
    assert result.status is RunStatus.SUCCESS, result.error
    offered = [t["function"]["name"] for t in model.requests[0]["tools"]]
    assert offered == sorted([f"t{i}" for i in range(6)] + list(MEMORY_TOOLS))
    assert "tool_search" in offered


@pytest.mark.parametrize(("top", "narrows"), [(0.5, True), (0.49, False)])
async def test_a_hint_exactly_at_the_floor_narrows_and_one_just_below_does_not(
    memory_harness: Harness, memory_service: FakeMemoryService, top: float, narrows: bool
) -> None:
    """B13 at ``NARROW_FLOOR``: a best hint of exactly 0.5 narrows the offer (and drops the
    abstain note: a tool answers); 0.49 narrows nothing and the note stays."""
    memory_service.candidates = ["t3"]
    memory_service.top_confidence = top
    memory_service.evidence_status = "INSUFFICIENT"
    model = ScriptedChat(["done"])
    agent = arm(memory_harness.wrap(ReAct(system_prompt="s", model=model), id="n"), many(6))
    result = await agent.run("look up a", user="u")
    assert result.status is RunStatus.SUCCESS, result.error
    offered = [t["function"]["name"] for t in model.requests[0]["tools"]]
    everything = sorted([f"t{i}" for i in range(6)] + list(MEMORY_TOOLS))
    assert offered == (sorted(["t3", *MEMORY_TOOLS]) if narrows else everything)
    noted = "## Memory" in model.requests[0]["messages"][1]["content"]
    assert noted is not narrows


async def test_a_hint_behind_code_mode_keeps_the_meta_tools_that_reach_it(
    memory_service: FakeMemoryService,
) -> None:
    """B16/B17: the hints are asked about the tools behind Code Mode too; when one is hinted,
    the narrowing keeps the meta-tools, the only way to reach it (they were dropped)."""
    stub = StubGateway([Def(f"wiki-t{i}", "wiki") for i in range(20)])
    memory_service.candidates = ["wiki-t4", "t3"]
    model = ScriptedChat(["done"])
    async with Harness(
        config=Settings(), memory=memory_service.client(), gateway=cast("Gateway", stub)
    ) as h:
        agent = arm(h.wrap(ReAct(system_prompt="s", model=model), id="cm"), many(6))
        result = await agent.run("what do the docs say", user="u")
    assert result.status is RunStatus.SUCCESS, result.error
    [context] = memory_service.named("context")
    assert "wiki-t4" in context.body["tools"]["available"]
    offered = {t["function"]["name"] for t in model.requests[0]["tools"]}
    assert {s.name for s in CODE_MODE_TOOLS} | {"t3"} <= offered
    assert "t1" not in offered, "still narrowed"


async def test_the_tools_a_learned_skill_names_are_offered_with_the_hints(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    """The offered set is the hints, the tools the learned plan names, the used ones and the
    memory tools — never all of the run's tools while hints exist."""
    memory_service.candidates = ["t3"]
    memory_service.learned_skills = "- archive: t3 then t4 (worked 90% of 10 runs)"
    model = ScriptedChat(["done"])
    agent = arm(memory_harness.wrap(ReAct(system_prompt="s", model=model), id="n"), many(6))
    result = await agent.run("look up a", user="u")
    assert result.status is RunStatus.SUCCESS, result.error
    offered = [t["function"]["name"] for t in model.requests[0]["tools"]]
    assert offered == sorted(["t3", "t4", *MEMORY_TOOLS])  # t4: the learned plan's step


async def test_the_hints_narrow_what_a_create_agent_graph_is_offered_per_call(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    """A graph's tools are bound when it is built: the harness runs a copy of a
    ``create_agent`` graph whose model calls offer only the tools the run offers — its own
    tools of the run's set too (``own`` is no candidate) — and the graph itself, run outside
    the harness, still offers every tool it was built with."""
    from langchain_core.tools import tool as langchain_tool

    @langchain_tool
    def own(key: str) -> str:
        """The graph's own tool, not the harness's."""
        return key

    memory_service.candidates = ["t3", "t1"]
    memory_service.candidates_for = {"archive it": ["t5"]}
    model = ScriptedChatModel(
        turns=[("t3", {"key": "a"}), ("tool_search", {"task": "archive it"}), "done", "plain"]
    )
    memory_tools = await native_tools(memory_harness, framework="langgraph")
    tools = [*graph_tools(6), *memory_tools, own]
    graph = create_agent(model, tools=tools)
    agent = memory_harness.wrap(graph, id="graph-hinted")
    assert (agent.adapter.fixed_tools, agent.adapter.narrows) == (True, "turn")
    result = await agent.run("look up a", user="u")
    assert result.status is RunStatus.SUCCESS, result.error

    offered = [r["tools"] for r in model.requests]
    kept = MEMORY_TOOLS  # the memory tools are not narrowed
    assert sorted(offered[0]) == sorted(["t1", "t3", *kept])
    assert sorted(offered[1]) == sorted(["t1", "t3", *kept])
    assert sorted(offered[2]) == sorted(["t1", "t3", "t5", *kept])  # tool_search found t5
    [asked] = memory_service.named("context")[:1]
    assert "own" in asked.body["tools"]["available"]  # the graph's own is of the run's set
    await graph.ainvoke({"messages": [{"role": "user", "content": "x"}]})
    assert len(offered := model.requests[-1]["tools"]) == 7 + len(kept), offered  # as built


async def test_the_hints_narrow_what_a_deep_agent_is_offered(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    """The harness tools are those the run offers; Deep Agents' state tools (its files) are no
    tools of the run's set, and its ``task`` hands work on: they stay offered."""
    from deepagents import create_deep_agent

    memory_service.candidates = ["t2"]
    model = ScriptedChatModel(turns=[("t2", {"key": "b"}), "done"])
    tools = [*graph_tools(6), *await native_tools(memory_harness, framework="deepagents")]
    graph = create_deep_agent(model=model, tools=tools)
    result = await memory_harness.wrap(graph, id="deep-hinted").run("look up b", user="u")
    assert result.status is RunStatus.SUCCESS, result.error
    for names in (r["tools"] for r in model.requests):
        assert "t2" in names and "write_file" in names and "task" in names, names
        assert not {"t0", "t1", "t3", "t4", "t5"} & set(names), names


def test_only_a_create_agent_model_node_is_narrowed() -> None:
    """A function named as ``create_agent``'s model node but not holding its handler (another
    LangChain release) leaves the graph as it is; a hand-built graph is run as it is."""
    from langgraph.graph import START, MessagesState, StateGraph

    from trellis.harness.adapters.langgraph import MODEL_NODE, LangGraphAdapter
    from trellis.harness.middleware import narrowing

    async def model(state: MessagesState) -> MessagesState:
        return state

    model.__qualname__ = MODEL_NODE
    builder = StateGraph(MessagesState)
    builder.add_node("model", model)
    builder.add_edge(START, "model")
    graph = builder.compile()
    assert narrowing(graph, model) is None
    adapter = LangGraphAdapter(graph)
    assert (adapter.narrowed, adapter.narrows) == (None, "none")


@pytest.mark.parametrize("omit", [True, False], ids=["no-candidates-field", "nothing-fits"])
async def test_without_candidates_every_tool_is_offered(
    memory_harness: Harness, memory_service: FakeMemoryService, omit: bool
) -> None:
    memory_service.omit_candidates = omit
    memory_service.candidates = []
    model = ScriptedChat(["done"])
    await arm(memory_harness.wrap(ReAct(system_prompt="s", model=model), id="n"), many(6)).run(
        "x", user="u"
    )
    # all the agent's, memory's (and ReAct's own)
    assert len(model.requests[0]["tools"]) == 6 + len(MEMORY_TOOLS)


async def test_tool_search_answers_among_the_runs_tools_and_offers_them(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    memory_service.candidates = ["t4"]

    async def fn(input: str, agent: Runtime) -> Any:
        return await agent.tools.call("tool_search", task="reorder")

    agent = arm(memory_harness.wrap(fn, id="h"), many(6))
    result = await agent.run("x", user="u")
    # what the model reads: the choice, its confidence, how it has done, the arguments found
    # and missing, the plan — and nothing empty
    assert result.answer == {
        "tools": [
            {
                "name": "t4",
                "confidence": 0.9,
                "next": True,
                "success_rate": 0.8,
                "args": {"sku": "SKU-1"},
                "missing": [{"arg": "qty", "question": "How many?"}],
            }
        ],
        "plan": {
            "id": "proc_1",
            "name": "reorder",
            "steps": ["t4"],
            "success_rate": 0.75,
            "runs": 4,
        },
    }
    assert memory_service.named("tool_hints")[-1].body["available"] == [f"t{i}" for i in range(6)]
    assert memory_service.named("call_agent_tool") == []  # answered by the harness


async def test_tools_hints_inside_a_run(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    async def fn(input: str, agent: Runtime) -> Any:
        return await agent.tools.hints("reorder")

    result = await arm(memory_harness.wrap(fn, id="h"), [stock]).run("reorder a", user="u")
    assert isinstance(result.answer, ToolHints) and result.answer.tools[0].name == "stock"
    # the candidates are the run's own tools, never the memory service's pull tools
    assert memory_service.named("tool_hints")[0].body["available"] == ["stock"]


# --------------------------------------------------------------------------- pull
async def test_pull_adds_the_memory_tools_and_they_call_the_service(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    model = ScriptedChat([("memory_search", {"query": "preferences"}), "email"])
    agent = memory_harness.wrap(ReAct(system_prompt="s", model=model), id="prefs")
    result = await agent.run("how do I like to be contacted?", user="u1")
    assert result.answer == "email"
    offered = [t["function"]["name"] for t in model.requests[0]["tools"]]
    assert offered == sorted(MEMORY_TOOLS)
    [call] = memory_service.named("call_agent_tool")
    assert call.path["name"] == "memory_search" and call.body["args"] == {"query": "preferences"}
    await memory_harness.writes.drain()
    assert memory_service.named("record_tool") == []  # the service logs its own tools


async def test_an_agent_searches_what_was_said_as_the_service_offers_it(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    """``kinds: ["message"]`` reads this conversation and the user's earlier ones (the memory
    service's behaviour, described in its own schema); the harness passes the call through as
    the model made it, in the run's scope, and adds nothing."""
    args = {"query": "order number", "kinds": ["message"]}
    model = ScriptedChat([("memory_search", args), "4471"])
    agent = memory_harness.wrap(ReAct(system_prompt="s", model=model), id="recall")
    result = await agent.run("what was my order number?", user="u1", thread="t-new")
    assert result.answer == "4471"
    [search] = [t for t in model.requests[0]["tools"] if t["function"]["name"] == "memory_search"]
    kinds = search["function"]["parameters"]["properties"]["kinds"]["description"]
    assert "earlier conversations" in kinds
    [call] = memory_service.named("call_agent_tool")
    assert call.body["args"] == args
    assert call.body["scope"]["user_id"] == "u1"
    assert call.body["scope"]["thread_id"] == conversation_key("acme", "u1", "t-new")


# --------------------------------------------------------------------------- records
async def test_the_transcript_tools_and_the_outcome_are_recorded_once(
    memory_harness: Harness, memory_service: FakeMemoryService, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(snapshots, "now", lambda: datetime(2026, 10, 30, 9, 0, tzinfo=UTC))
    model = ScriptedChat([("stock", {"sku": "a"}), "7 units"])
    agent = arm(memory_harness.wrap(ReAct(system_prompt="s", model=model), id="stock"), [stock])
    result = await agent.run("how many a?", user="u1")
    await memory_harness.writes.drain()
    [batch] = memory_service.named("messages")
    # the question, the call with its result, the answer: as a checkpoint would replay them
    assert [(m["role"], m["content"], m["source_message_id"]) for m in batch.body["messages"]] == [
        ("USER", "how many a?", f"{result.run_id}:user:0"),
        ("ASSISTANT", "", f"{result.run_id}:1:msg:1"),
        # which tool returned it, and when: a later turn replays it so (``snapshots``)
        ("TOOL", "[snapshot of stock from 2026-10-30T09:00:00Z]\n7", f"{result.run_id}:1:msg:2"),
        ("ASSISTANT", "7 units", f"{result.run_id}:1:msg:3"),
    ]
    call, answered = batch.body["messages"][1:3]
    [made] = call["tool_calls"]
    assert made["name"] == "stock" and made["arguments"] == {"sku": "a"}
    assert answered["tool_call_id"] == made["id"] and answered["kind"] == "INTERNAL"
    [recorded] = memory_service.named("record_tool")
    assert recorded.body["tool"] == "stock" and recorded.body["task"] == "how many a?"
    [outcome] = memory_service.named("feedback")
    assert outcome.body["target_kind"] == "run" and outcome.body["target_id"] == result.run_id
    assert outcome.body["verdict"] == "confirm" and outcome.body["source"] == "system"
    assert outcome.idempotency_key == f"{result.run_id}:outcome"


async def test_a_large_result_is_recorded_within_the_services_limit(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    """The service refuses a recorded output over 256 KiB (its ``TOOL_OUTPUT_MAX_BYTES``) with
    a 422, and the call's record was lost: a large result's call is recorded with its output
    cut to the limit, so the service still learns from the call."""

    @tool(side_effects="read")
    def sales_rows(region: str) -> list[dict[str, Any]]:
        """Every sales line of a region."""
        return [{"region": region, "line": n, "note": "x" * 80} for n in range(5000)]

    model = ScriptedChat([("sales_rows", {"region": "north"}), "5000 lines"])
    agent = arm(memory_harness.wrap(ReAct(system_prompt="s", model=model), id="rows"), [sales_rows])
    await agent.run("every north line?", user="u1")
    await memory_harness.writes.drain()
    [recorded] = memory_service.named("record_tool")
    output = recorded.body["output"]
    assert len(json.dumps(output).encode()) <= RECORDED_OUTPUT_MAX_BYTES
    assert isinstance(output, str) and output.startswith('[{"region": "north", "line": 0')
    assert output.endswith("bytes; cut to the service's limit)")


async def test_a_failed_run_is_rejected_and_a_cancelled_one_says_nothing(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    async def fn(input: str, agent: Runtime) -> str:
        if input == "fail":
            raise RuntimeError("boom")
        return str(await agent.ask("Sure?"))

    agent = memory_harness.wrap(fn, id="t")
    failed = await agent.run("fail", user="u")
    paused = await agent.run("ask", user="u")
    assert paused.interrupt is not None
    await agent.resume(paused.interrupt.interrupt_id, "cancel", reviewer="u")
    await memory_harness.writes.drain()
    [outcome] = memory_service.named("feedback")
    assert outcome.body["target_id"] == failed.run_id and outcome.body["verdict"] == "reject"
    assert outcome.body["comment"] == "boom"


async def test_the_transcript_is_recorded_on_a_pause_and_on_a_failure(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    async def fn(input: str, agent: Runtime) -> str:
        if input == "fail":
            raise RuntimeError("boom")
        return str(await agent.ask("Sure?"))

    agent = memory_harness.wrap(fn, id="t")
    paused = await agent.run("ask", user="u")
    assert paused.interrupt is not None
    await agent.resume(paused.interrupt.interrupt_id, "answer", answer="yes", reviewer="u")
    failed = await agent.run("fail", user="u")
    assert failed.status is RunStatus.ERROR
    await memory_harness.writes.drain()
    batches = memory_service.named("messages")
    sent = [
        (m["role"], m["content"], m["source_message_id"])
        for b in batches
        for m in b.body["messages"]
    ]
    run = paused.run_id
    assert sent[:3] == [
        ("USER", "ask", f"{run}:user:0"),  # on the pause
        ("USER", "ask", f"{run}:user:0"),  # again on the resumed attempt: the service stores
        ("ASSISTANT", "yes", f"{run}:2:msg:1"),  # it once (its source id is the same)
    ]
    assert sent[3][:2] == ("USER", "fail")  # a failed run's question is kept too
    # a run with no thread is its own thread
    assert {b.scope["thread_id"] for b in batches} == {paused.run_id, failed.run_id}


async def test_an_approval_decision_is_feedback_on_the_tool_call(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    @tool(side_effects="irreversible")
    def wipe(disk: str) -> str:
        """Wipe a disk."""
        return "wiped"

    async def fn(input: str, agent: Runtime) -> Any:
        return await agent.tools.call("wipe", disk="d1")

    agent = arm(memory_harness.wrap(fn, id="ops"), [wipe])
    paused = await agent.run("wipe d1", user="u")
    assert paused.interrupt is not None
    await agent.resume(paused.interrupt.interrupt_id, "reject", reviewer="boss")
    await memory_harness.writes.drain()
    decided = [f for f in memory_service.named("feedback") if f.body["target_kind"] == "tool_call"]
    [feedback] = decided
    assert feedback.body["reviewer"] == "boss" and feedback.body["verdict"] == "reject"
    # what approval patterns are learned from: the tool and the arguments it was asked about
    assert feedback.body["metadata"]["tool"] == "wipe"
    assert feedback.body["metadata"]["args"] == {"disk": "d1"}


ENDPOINT = "http://llm.test/v1"


async def ok(input: str, agent: Runtime) -> str:
    return "ok"


async def test_each_agent_tells_the_memory_service_its_model_endpoint_once(
    memory_service: FakeMemoryService,
) -> None:
    """The route's whole endpoint, its key in plain, once per process and agent; the agent's
    model and its profile when its builder names one (``ReAct``)."""
    settings = {"llm_base_url": ENDPOINT, "llm_api_key": "sk-agent"}
    async with harness_with(memory_service, **settings) as h:
        agent = h.wrap(ok, id="keyed")
        await agent.run("a", user="u")
        await agent.run("b", user="u")
        named = h.wrap(ReAct(system_prompt="s", model="openai/o3"), id="named")
        assert named.model == "openai/o3" and agent.model is None
        assert h.memory is not None
        whose = Identity(tenant="acme", user="u", agent_id="named", run_id="r1")
        await h._registered(h.memory, whose, named.model)
        await h.writes.drain()
    first, second = memory_service.named("model_key")
    assert first.body["endpoint"] == {
        "base_url": ENDPOINT,
        "gateway": "openai-compatible",
        "auth": "bearer",
        "model": None,
        "api_key": "sk-agent",
        "profile": {
            "reasoning_output": False,
            "temperature": True,
            "max_output_tokens": None,
            "tool_calling": True,
        },
    }
    assert first.idempotency_key is not None and first.idempotency_key.startswith(
        "model-key:keyed:"
    )
    assert first.scope["agent_id"] == "keyed" and "user_id" not in first.scope
    assert second.body["endpoint"]["model"] == "openai/o3"  # as the route names it
    profile = second.body["endpoint"]["profile"]
    assert profile["reasoning_output"] is True and profile["temperature"] is False


@pytest.mark.parametrize(
    ("settings", "told"),
    [
        ({"llm_base_url": ENDPOINT, "llm_auth": "google"}, {"auth": "google", "api_key": None}),
        (
            {"llm_base_url": ENDPOINT, "llm_gateway": "bifrost", "llm_api_key": "vk"},
            {"gateway": "bifrost", "api_key": "vk"},
        ),
        ({}, None),  # the providers' own SDKs: their keys are theirs, nothing is told
    ],
    ids=["login", "bifrost", "provider sdk"],
)
async def test_a_login_sends_no_secret_and_the_provider_sdks_nothing(
    memory_service: FakeMemoryService, settings: dict[str, Any], told: dict[str, Any] | None
) -> None:
    async with harness_with(memory_service, **settings) as h:
        await h.wrap(ok, id="a").run("x", user="u")
        await h.writes.drain()
    sent = [call.body["endpoint"] for call in memory_service.named("model_key")]
    assert sent == ([] if told is None else [sent[0] | told])


async def test_a_service_that_takes_no_endpoints_is_told_once(
    memory_service: FakeMemoryService, caplog: pytest.LogCaptureFixture
) -> None:
    """Credential encryption off in the memory service: the registration is logged once per
    process and never asked again, and no failed write is reported for it."""
    memory_service.model_keys = False
    async with harness_with(memory_service, llm_base_url=ENDPOINT, llm_api_key="sk") as h:
        with caplog.at_level("INFO", logger="trellis.harness"):
            await h.wrap(ok, id="a").run("x", user="u")
            await h.wrap(ok, id="b").run("x", user="u")  # both asked before either answer
            await h.writes.drain()
            await h.wrap(ok, id="c").run("x", user="u")  # after the answer: not asked
            await h.writes.drain()
        assert h.writes.failed == 0
    assert caplog.text.count("takes no model endpoints") == 1
    assert memory_service.named("model_key") == []  # refused each time it was asked: once


async def test_an_endpoint_the_service_refuses_is_said_once(
    memory_service: FakeMemoryService, caplog: pytest.LogCaptureFixture
) -> None:
    """One it cannot reach (a ``localhost`` only this machine reaches): a warning once, not a
    failed write, and not asked again; the service's own settings serve."""
    memory_service.failures["model_key"] = 422
    async with harness_with(memory_service, llm_base_url="http://localhost:11434/v1") as h:
        with caplog.at_level("WARNING", logger="trellis.harness"):
            await h.wrap(ok, id="a").run("x", user="u")
            await h.writes.drain()
            await h.wrap(ok, id="b").run("x", user="u")
            await h.writes.drain()
        assert h.writes.failed == 0 and h._endpoints_off
    assert caplog.text.count("refused http://localhost:11434/v1") == 1
    assert "(the fake memory service)" in caplog.text  # the service's reason, as it said it


async def test_a_registration_that_fails_otherwise_is_a_failed_write(
    memory_service: FakeMemoryService, monkeypatch: pytest.MonkeyPatch
) -> None:
    from trellis.harness import writes as writes_module

    monkeypatch.setattr(writes_module, "WRITE_BACKOFF_SECONDS", 0.0)
    memory_service.fail.add("model_key")
    async with harness_with(memory_service, llm_base_url=ENDPOINT) as h:
        await h.wrap(ok, id="a").run("x", user="u")
        await h.writes.drain()
        assert h.writes.failed == 1 and not h._endpoints_off


def test_a_memory_service_needs_the_api_key() -> None:
    with pytest.raises(ConfigurationError, match="need TRELLIS_API_KEY"):
        Harness(config=Settings(memory_url="http://m"))
    with pytest.raises(ConfigurationError, match="need TRELLIS_API_KEY"):
        Harness(config=Settings(memory_url="http://m", runs_url="http://r"))


async def test_the_context_budget_follows_the_models_window(
    memory_service: FakeMemoryService,
) -> None:
    class Model:
        context_window = 128_000

    async def fn(input: str, agent: Runtime) -> str:
        return "ok"

    fn.model = Model()  # type: ignore[attr-defined]
    async with harness_with(memory_service) as h:
        await h.wrap(fn, id="windowed").run("q", user="u")
    [asked] = memory_service.named("context")
    assert asked.body["token_budget"] == 6400  # 5 % of the window


# --------------------------------------------------------------------------- the catalog
async def test_the_catalog_decides_tiers_and_approvals(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    @tool(side_effects="write")
    def pay(amount: int) -> str:
        """Pay an invoice."""
        return f"paid {amount}"

    memory_service.catalog = {"pay": {"side_effects": "write", "approve_when": "amount > 100"}}

    async def fn(input: int, agent: Runtime) -> Any:
        return await agent.tools.call("pay", amount=input)

    agent = arm(memory_harness.wrap(fn, id="payer"), [pay])
    assert (await agent.run(50, user="u")).answer == "paid 50"
    paused = await agent.run(500, user="u")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
    assert paused.interrupt.question == "Approve pay? amount > 100."
    await memory_harness.writes.drain()
    [put] = memory_service.named("put_catalog")
    assert put.scope["tenant_id"] == "acme" and "user_id" not in put.scope
    assert put.body["tools"] == [
        {
            "name": "pay",
            "description": "Pay an invoice.",
            "input_schema": put.body["tools"][0]["input_schema"],
            "source": "local",
            "server": None,
            "side_effects": "write",
        }
    ]


# --------------------------------------------------------------------------- grounding
@pytest.mark.usefixtures("langfuse_env")
@respx.mock
async def test_a_sampled_run_is_verified_and_scored_on_its_trace(
    memory_service: FakeMemoryService,
    sampled: Any,
) -> None:
    scores = respx.post("https://lf.test/api/public/scores").mock(
        return_value=httpx.Response(200, json={"id": "x"})
    )
    sampled(1.0)
    async with harness_with(memory_service) as h:

        async def fn(input: str, agent: Runtime) -> str:
            return "you prefer email"

        result = await h.wrap(fn, id="judged").run("contact?", user="u")
        await h.writes.drain()
    [verified] = memory_service.named("verify")
    [context] = memory_service.named("context")
    assert verified.body["answer"] == "you prefer email"
    assert verified.body["bundle_id"].startswith("bnd_")
    [posted] = [c for c in scores.calls if b"grounding" in c.request.content]
    body = json.loads(posted.request.content)
    assert body == {
        "id": f"{result.run_id}:grounding",
        "traceId": telemetry.trace_hex(result.run_id),
        "name": "grounding",
        "value": 0.8,
        "dataType": "NUMERIC",
    }
    assert context.body["query"] == "contact?"


async def test_a_structured_answer_is_verified_as_its_json(
    memory_service: FakeMemoryService,
    sampled: Any,
) -> None:
    sampled(1.0)
    async with harness_with(memory_service) as h:

        async def fn(input: str, agent: Runtime) -> dict[str, str]:
            return {"contact": "email"}

        await h.wrap(fn, id="judged").run("contact?", user="u")
        await h.writes.drain()
    [verified] = memory_service.named("verify")
    assert verified.body["answer"] == '{"contact": "email"}'


async def test_an_unsampled_run_is_not_verified(
    memory_service: FakeMemoryService, sampled: Any
) -> None:
    async def fn(input: str, agent: Runtime) -> str:
        return "answer"

    sampled(0.0)
    async with harness_with(memory_service) as h:
        await h.wrap(fn, id="n").run("q", user="u")
        await h.writes.drain()
    assert memory_service.named("verify") == []


# --------------------------------------------------------------------------- feedback
@pytest.mark.usefixtures("langfuse_env")
@respx.mock
async def test_feedback_fans_out_to_langfuse_and_the_memory_service(
    memory_service: FakeMemoryService,
) -> None:
    scores = respx.post("https://lf.test/api/public/scores").mock(
        return_value=httpx.Response(200, json={"id": "x"})
    )
    async with harness_with(memory_service) as h:

        async def fn(input: str, agent: Runtime) -> str:
            return "12"

        result = await h.wrap(fn, id="f").run("stock?", user="u")
        await h.writes.drain()
        await h.feedback(result.run_id, "correct", "13")
    bodies = [json.loads(c.request.content) for c in scores.calls]
    [body] = [b for b in bodies if b["name"] == "feedback"]
    assert body["value"] == 0.0 and body["comment"] == "13"
    assert body["traceId"] == telemetry.trace_hex(result.run_id)
    human = [f for f in memory_service.named("feedback") if f.body["source"] == "human"]
    [sent] = human
    assert sent.body["target_kind"] == "run" and sent.body["target_id"] == result.run_id
    assert sent.body["verdict"] == "correct" and sent.body["correction"] == "13"
    assert sent.body["reviewer"] == "u"


async def test_feedback_without_langfuse_is_a_score_span_and_memory_feedback(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    async def fn(input: str, agent: Runtime) -> str:
        return "12"

    result = await memory_harness.wrap(fn, id="f").run("stock?", user="u")
    assert memory_harness.evals.langfuse is None
    await memory_harness.writes.drain()  # the run's own outcome is a queued write
    stored = await memory_harness.feedback(result.run_id, "confirm")
    assert [f.body["verdict"] for f in memory_service.named("feedback")][-1] == "confirm"
    # a person's verdict waits for the tenant administrator; the run's own status did not
    assert stored is not None and stored.review is not None
    assert stored.review.state == "pending"
    outcome = next(
        f for f in memory_service.stored_feedback.values() if f.get("source") == "system"
    )
    assert "review" not in outcome
    with pytest.raises(ConfigurationError, match="no run"):
        await memory_harness.feedback("run_missing", "confirm")


async def test_runtime_memory_is_the_sdk_in_the_runs_scope(memory_harness: Harness) -> None:
    async def fn(input: str, agent: Runtime) -> str:
        assert agent.context is not None
        remembered = await agent.memory.call_agent_tool("memory_search", {"query": input})
        return str(remembered)

    result = await memory_harness.wrap(fn, id="direct").run("x", user="u")
    assert result.status is RunStatus.SUCCESS and "the user prefers email" in result.answer


@pytest.mark.parametrize(
    ("status", "note"),
    [
        ("INSUFFICIENT", "Memory holds nothing about this question"),
        ("INCOMPLETE", "Memory covers only part of this question"),
        ("COMPLETE", None),
    ],
)
async def test_the_model_is_told_when_memory_has_nothing_to_go_on(
    memory_harness: Harness, memory_service: FakeMemoryService, status: str, note: str | None
) -> None:
    """Abstention: memory with no evidence for the question must turn into an honest "I don't
    know", not a confident guess; the harness says so to the model with the context — and
    what an honest answer does: it names the detail it does not have and offers to look it up
    or to be told it (a bare "I don't know" leaves the user nowhere)."""
    memory_service.evidence_status = status
    model = ScriptedChat(["I don't know your sister's name."])
    agent = memory_harness.wrap(ReAct(system_prompt="You help.", model=model), id="s")
    events = [e async for e in agent.stream("What is my sister's name?", user="u1")]
    system = model.requests[0]["messages"][1]["content"]
    assert system.startswith(memory_service.context_text)
    if note is None:
        assert "## Memory" not in system
    else:
        assert note in system
        assert "say which detail you do not have" in system
        assert "offer to look it up" in system and "ask the user for it" in system
    [loaded] = [e for e in events if e.type is RunEventType.CONTEXT_LOADED]
    assert loaded.data["evidence_status"] == status


async def test_no_abstain_note_when_a_hinted_tool_answers_the_question(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    """S4: "Memory holds nothing ... do not guess" on a question a hinted tool answers drove a
    weak model into memory_search loops; a confident hint means a tool answers it."""
    memory_service.evidence_status = "INSUFFICIENT"
    memory_service.candidates = ["t3"]
    model = ScriptedChat(["done"])
    agent = arm(memory_harness.wrap(ReAct(system_prompt="s", model=model), id="n"), many(6))
    await agent.run("what is the daily meal allowance", user="u")
    assert "## Memory" not in model.requests[0]["messages"][1]["content"]


async def test_a_document_added_for_a_user_is_uploaded_in_their_scope_and_indexed(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    info = await memory_harness.add_document(
        ("terms.txt", b"Returns are accepted for 30 days.", "text/plain"),
        user="u",
        thread="thr_1",
        title="Return policy",
    )
    assert info.status == "READY"
    [upload] = memory_service.named("add_document")
    assert upload.scope["tenant_id"] == "acme" and upload.scope["user_id"] == "u"
    kept = memory_service.documents[info.document_id]["thread_id"]
    assert kept == conversation_key("acme", "u", "thr_1")  # u's conversation thr_1


async def test_a_document_is_staged_then_ready_or_failed_with_its_error(
    memory_harness: Harness, memory_service: FakeMemoryService, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``add_document`` waits through STAGED (the parse job queued or running) and returns the
    document as it ended: READY, or FAILED with the parse error for the caller to act on."""
    import asyncio as aio

    real_sleep = aio.sleep

    async def no_wait(seconds: float) -> None:
        await real_sleep(0)

    monkeypatch.setattr("trellis.memory.advanced.asyncio.sleep", no_wait)
    memory_service.document_reads_staged = 3
    ready = await memory_harness.add_document(b"returns: 30 days", user="u")
    assert ready.status == "READY" and len(memory_service.named("document")) == 4
    memory_service.document_outcome = "FAILED"
    failed = await memory_harness.add_document(b"\x00garbage", user="u")
    assert failed.status == "FAILED"
    assert getattr(failed, "last_error", None) == "the file could not be parsed"  # an extra
    staged = await memory_harness.add_document(b"later", user="u", wait=None)
    assert staged.status == "STAGED"


def rules_expire(monkeypatch: pytest.MonkeyPatch) -> None:
    """The catalog's rules as read so far are older than their TTL: the next check reads it."""
    monkeypatch.setattr(
        catalog, "_now", lambda: time.monotonic() + catalog.GOVERNANCE_TTL_SECONDS + 1
    )


async def test_an_approval_rule_set_after_a_graph_was_built_governs_its_calls(
    memory_harness: Harness, memory_service: FakeMemoryService, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A compiled graph holds the tools ``h.tools`` built it with; the rule an administrator
    sets afterwards still decides its calls (governance looks each call up by name, as the
    catalog says within its TTL, not as it said when the graph was built)."""
    paid: list[int] = []

    @tool(side_effects="write")
    def pay(amount: int) -> str:
        """Pay an invoice."""
        paid.append(amount)
        return f"paid {amount}"

    model = ScriptedChatModel(turns=[("pay", {"amount": 500}), "paid"])
    graph = create_agent(
        model, tools=await native_tools(memory_harness, pay, framework="langgraph")
    )
    memory_service.catalog = {"pay": {"side_effects": "write", "approve_when": "amount > 100"}}
    rules_expire(monkeypatch)
    agent = memory_harness.wrap(graph, id="graph-payer")
    paused = await agent.run("pay the invoice", user="u")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None, paused
    assert paused.interrupt.question == "Approve pay? amount > 100."
    assert paid == []


async def test_a_handoffs_tools_built_by_h_tools_follow_the_catalog_too(
    memory_harness: Harness, memory_service: FakeMemoryService, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An OpenAI Agents specialist reached by a handoff carries tools ``h.tools`` built: the
    run's toolbox does not hold them, and governance looks their calls up by name all the
    same."""
    paid: list[int] = []

    @tool(side_effects="write")
    def pay(amount: int) -> str:
        """Pay an invoice."""
        paid.append(amount)
        return f"paid {amount}"

    class Model(ScriptedModel):
        """Call ids of its own, as two real models' would be."""

        def __init__(self, prefix: str, turns: list[Any]) -> None:
            super().__init__(turns)
            self.prefix = prefix

        def _next(self, input: Any, system: str | None) -> list[Any]:
            items = super()._next(input, system)
            for item in items:
                if hasattr(item, "call_id"):
                    item.call_id = self.prefix + item.call_id
            return items

    billing = OpenAIAgent(
        name="billing",
        model=Model("b", [("pay", {"amount": 500}), ("pay", {"amount": 500}), "paid"]),
        tools=await native_tools(memory_harness, pay, framework="openai_agents"),
    )
    triage = OpenAIAgent(
        name="triage",
        model=Model("t", [("transfer_to_billing", {}), ("transfer_to_billing", {})]),
        handoffs=[billing],
    )
    memory_service.catalog = {"pay": {"side_effects": "write", "approve_when": "amount > 100"}}
    rules_expire(monkeypatch)
    agent = memory_harness.wrap(triage, id="triage")
    paused = await agent.run("pay the invoice", user="u")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None, paused
    assert paused.interrupt.question == "Approve pay? amount > 100."
    assert paid == []
    # the re-run hands off again; the approved call runs once
    done = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="lead")
    assert done.status is RunStatus.SUCCESS and done.answer == "paid" and paid == [500]


# --------------------------------------------------------------------------- publish at start
async def test_a_worker_publishes_the_agents_tools_before_its_first_claim(
    memory_service: FakeMemoryService, caplog: pytest.LogCaptureFixture
) -> None:
    """B13: the catalog was written on the first run's ``tools()`` call, and the index landed
    seconds later: the first runs' hints were zeros. A worker publishes each agent's tools to
    the key's tenant before it claims anything; a failure is a warning."""

    @tool(side_effects="read")
    def stock(sku: str) -> str:
        """Units in stock."""
        return sku

    async def fn(input: str, agent: Runtime) -> str:
        return input

    async with harness_with(memory_service) as h:
        agent = arm(h.wrap(fn, id="pub"), [stock])
        task = asyncio.create_task(h.worker([agent]).run())
        for _ in range(200):
            if memory_service.named("put_catalog"):
                break
            await asyncio.sleep(0.01)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        published = json.dumps([c.body for c in memory_service.named("put_catalog")])
        assert "stock" in published

        broken = h.wrap(fn, id="broken")

        class Failing:
            async def tools(self) -> list[Any]:
                raise ConnectionError("the gateway is down")

        broken._toolbox = lambda tenant: Failing()  # type: ignore[method-assign]
        with caplog.at_level("WARNING", logger="trellis.harness"):
            await h._published([broken])
        assert "broken: its tools were not published at start" in caplog.text

    platform = FakeMemoryService(tenant=None)  # a platform key knows no tenant
    async with harness_with(platform) as h:
        await h._published([arm(h.wrap(fn, id="p"), [stock])])
        assert platform.named("put_catalog") == []

    down = FakeMemoryService(fail={"key"})
    async with harness_with(down) as h:
        with caplog.at_level("WARNING", logger="trellis.harness"):
            await h._published([h.wrap(fn, id="d")])
        assert "the tools were not published at start" in caplog.text
