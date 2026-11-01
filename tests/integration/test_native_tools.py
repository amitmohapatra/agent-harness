"""The run's tool set: the harness's tools and the framework's own — a graph's tool nodes', a
Deep Agent's sub-agents', an OpenAI agent's function, hosted, MCP server and handoff tools,
Claude's built-ins and MCP tools — one set, sorted, each once, that the tool hints and
``tool_search`` choose among, the catalog is told about and the narrowing narrows; and the
framework's own tools governed, journaled and recorded as the harness's are, with no
middleware of the harness's in the agent."""

from __future__ import annotations

import json
import types
from pathlib import Path
from typing import Any

import pytest
from agents import Agent, function_tool
from deepagents import SubAgent
from langchain.agents import create_agent
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool as langchain_tool
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from tests.support.arm import with_governance
from tests.support.chat_model import ScriptedChatModel
from tests.support.memory import FakeMemoryService
from tests.support.openai_model import ScriptedModel
from tests.support.planned import PlannedChatModel, PlannedModel
from trellis import Harness
from trellis.contracts import RunStatus, ToolSpec
from trellis.harness.adapters.base import HELPER, helper, unique
from trellis.harness.adapters.langgraph import graph_tools, subagent_graphs

refunded: list[str] = []


@pytest.fixture(autouse=True)
def _reset() -> None:
    refunded.clear()


@langchain_tool
def refund(order: str) -> str:
    """Refund an order."""
    refunded.append(order)
    return f"refunded {order}"


@langchain_tool
def lookup(order: str) -> str:
    """Look an order up."""
    return f"{order}: shipped"


def hand_built(model: ScriptedChatModel, tools: list[Any]) -> Any:
    """A graph built by hand: a model node binding its tools, a ``ToolNode``, the loop."""
    bound = model.bind_tools(tools)

    async def think(state: MessagesState) -> dict[str, Any]:
        return {"messages": [await bound.ainvoke([SystemMessage("s"), *state["messages"]])]}

    graph = StateGraph(MessagesState)
    graph.add_node("model", think)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", tools_condition)
    graph.add_edge("tools", "model")
    return graph.compile()


def published(memory_service: FakeMemoryService) -> dict[str, dict[str, Any]]:
    """What the catalog was told, by tool name."""
    calls = memory_service.named("put_catalog")
    return {e["name"]: e for c in calls for e in c.body["tools"]}


def recorded(memory_service: FakeMemoryService) -> dict[str, dict[str, Any]]:
    return {c.body["tool"]: c.body for c in memory_service.named("record_tool")}


# --------------------------------------------------------------------------- LangGraph


@pytest.mark.parametrize("build", ["hand_built", "create_agent"])
async def test_a_graphs_own_tool_is_governed_journaled_recorded_and_published(
    memory_harness: Harness, memory_service: FakeMemoryService, build: str
) -> None:
    """No middleware of the harness's: the graph's own ``refund`` is of the run's tool set
    (the context is asked with it, the catalog told about it) and its call is the run's — the
    catalog's rule asks a person, the approved call runs once and is recorded."""
    memory_service.catalog = {"refund": {"side_effects": "irreversible"}}
    model = PlannedChatModel(plan=[("refund", {"order": "o1"})], final="done")
    graph = (
        hand_built(model, [refund, lookup])
        if build == "hand_built"
        else create_agent(model, tools=[refund, lookup])
    )
    agent = memory_harness.wrap(graph, id=f"own-{build}")
    paused = await agent.run("refund o1", user="u1")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
    assert paused.interrupt.tool_call is not None and paused.interrupt.tool_call.tool == "refund"
    assert refunded == []
    done = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="u1")
    assert done.status is RunStatus.SUCCESS and refunded == ["o1"]  # once
    await memory_harness.writes.drain()
    assert recorded(memory_service)["refund"]["status"] == "ok"
    told = published(memory_service)
    assert {"refund", "lookup"} <= set(told) and told["refund"]["source"] == "local"
    [asked, *_] = memory_service.named("context")
    assert {"refund", "lookup"} <= set(asked.body["tools"]["available"])


async def test_a_graphs_own_tools_are_narrowed_with_the_harnesss(
    memory_harness: Harness, memory_service: FakeMemoryService, narrowing: None
) -> None:
    from tests.integration.test_memory_pipeline import graph_tools

    memory_service.candidates = ["t1", "lookup"]
    model = ScriptedChatModel(turns=["done"])
    tools = [*graph_tools(6), refund, lookup]
    graph = create_agent(model, tools=tools)
    result = await memory_harness.wrap(graph, id="narrowed").run("look o1 up", user="u")
    assert result.status is RunStatus.SUCCESS
    offered = set(model.requests[0]["tools"])
    assert {"t1", "lookup"} <= offered and not {"refund", "t0", "t2"} & offered


async def test_a_deep_agents_sub_agents_tools_are_of_the_run_and_governed(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    """A Deep Agent's ``task`` sub-agent runs its own tools in a graph of its own: they are of
    the run's tool set, and a call of one is the run's (the catalog's rule asks); ``task`` only
    hands the work on (a helper: read, never narrowed away)."""
    from deepagents import create_deep_agent

    memory_service.catalog = {"refund": {"side_effects": "irreversible"}}
    model = ScriptedChatModel(
        turns=[
            ("task", {"description": "refund o1", "subagent_type": "clerk"}),
            ("refund", {"order": "o1"}),
        ]
    )
    clerk = SubAgent(name="clerk", description="Refunds.", system_prompt="s", tools=[refund])
    graph = create_deep_agent(model=model, subagents=[clerk])
    names = {t.name for t in graph_tools(graph)}
    assert {"task", "refund"} <= names
    agent = memory_harness.wrap(graph, id="deep-own")
    paused = await agent.run("refund o1", user="u1")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
    assert paused.interrupt.tool_call is not None and paused.interrupt.tool_call.tool == "refund"
    assert refunded == []
    told = published(memory_service)
    assert told["task"]["side_effects"] == "read" and "refund" in told


def test_a_task_tool_whose_code_holds_no_graphs_has_no_sub_agents() -> None:
    @langchain_tool
    def task(description: str) -> str:
        """Not Deep Agents' own."""
        return description

    assert subagent_graphs(task) == []
    assert subagent_graphs(object()) == []


# --------------------------------------------------------------------------- OpenAI Agents


class Server:
    """An MCP server of the agent's own (the SDK's ``MCPServer`` shape), its tools listed and
    called in process."""

    def __init__(self, name: str = "crm", *, connected: bool = True) -> None:
        self._name = name
        self.connected = connected
        self.called: list[tuple[str, dict[str, Any] | None]] = []

    async def connect(self) -> None:
        self.connected = True

    async def cleanup(self) -> None:
        self.connected = False

    async def list_tools(self, run_context: Any = None, agent: Any = None) -> list[Any]:
        from mcp.types import Tool, ToolAnnotations

        if not self.connected:
            raise RuntimeError("not connected")
        return [
            Tool(
                name="find",
                description="Find an order.",
                input_schema={"type": "object", "properties": {"order": {"type": "string"}}},
                annotations=ToolAnnotations(read_only_hint=True),
            ),
            Tool(
                name="erase",
                description="Erase an order.",
                input_schema={"type": "object", "properties": {"order": {"type": "string"}}},
                annotations=ToolAnnotations(destructive_hint=True),
            ),
        ]

    async def call_tool(self, tool_name: str, arguments: Any, meta: Any = None) -> Any:
        from mcp.types import CallToolResult, TextContent

        self.called.append((tool_name, arguments))
        if tool_name == "boom":
            return CallToolResult(
                content=[TextContent(type="text", text="it broke")], is_error=True
            )
        return CallToolResult(content=[TextContent(type="text", text=f"{tool_name} done")])

    async def list_prompts(self) -> Any:
        return []

    async def get_prompt(self, name: str, arguments: Any = None) -> Any:
        return None


def mcp_server(name: str = "crm", *, connected: bool = True) -> Any:
    from agents.mcp import MCPServer

    class Own(Server, MCPServer):  # type: ignore[misc]
        @property
        def name(self) -> str:
            return self._name

    found = Own.__new__(Own)
    MCPServer.__init__(found)  # the SDK's own settings, its defaults
    Server.__init__(found, name, connected=connected)
    return found


async def test_an_openai_agents_own_tools_handoffs_and_mcp_tools_are_its_tool_set(
    harness: Harness,
) -> None:
    from agents import FileSearchTool, HostedMCPTool, handoff

    from trellis.harness.adapters.openai_agents import OpenAIAgentsAdapter

    @function_tool
    def eta(order: str) -> str:
        return "tomorrow"

    clerk = Agent(name="Clerk", tools=[eta])
    checker = Agent(name="checker", tools=[eta])
    hosted = HostedMCPTool(
        tool_config={
            "type": "mcp",
            "server_label": "gh",
            "server_url": "http://x",
            "allowed_tools": ["issues"],
        }
    )
    target = Agent(
        name="desk",
        tools=[FileSearchTool(vector_store_ids=["v"]), hosted, checker.as_tool("check", "C.")],
        handoffs=[clerk, handoff(Agent(name="hidden"))],
        mcp_servers=[mcp_server(), mcp_server("down", connected=False)],
    )
    adapter = OpenAIAgentsAdapter()
    specs = await adapter.native(target)
    assert await adapter.native(target) == specs  # the down server said once, still left out
    by_name = {s.name: s for s in specs}
    assert [s.name for s in specs] == sorted(by_name)
    assert set(by_name) == {
        "check",
        "erase",
        "eta",
        "file_search",
        "find",
        "issues",
        "transfer_to_clerk",
        "transfer_to_hidden",
    }
    assert HELPER in by_name["check"].tags and by_name["check"].side_effects == "read"
    assert HELPER in by_name["transfer_to_clerk"].tags
    assert (by_name["find"].source, by_name["find"].server) == ("mcp", "crm")
    assert (
        by_name["find"].side_effects == "read" and by_name["erase"].side_effects == "irreversible"
    )
    assert by_name["issues"].server == "gh"


async def test_an_openai_agents_mcp_server_calls_are_governed(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    """The agent's own MCP server's tools: a read runs, recorded; a destructive one (its
    annotations) waits for a person — the server is called once, after the approval."""
    server = mcp_server()
    model = PlannedModel([("find", {"order": "o1"}), ("erase", {"order": "o1"})], final="done")
    target = Agent(name="desk", model=model, mcp_servers=[server])
    agent = memory_harness.wrap(target, id="desk-mcp")
    paused = await agent.run("erase o1", user="u1")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
    assert paused.interrupt.tool_call is not None and paused.interrupt.tool_call.tool == "erase"
    assert server.called == [("find", {"order": "o1"})]
    done = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="u1")
    assert done.status is RunStatus.SUCCESS
    assert [c[0] for c in server.called] == ["find", "erase"]  # each once
    await memory_harness.writes.drain()
    assert {"find", "erase"} <= set(recorded(memory_service))
    assert published(memory_service)["erase"]["server"] == "crm"


async def test_a_governed_mcp_server_passes_through_outside_a_run_and_reports_errors() -> None:
    from mcp.types import CallToolResult

    from trellis.harness.adapters.openai_agents import GovernedServer

    inner = mcp_server()
    governed = GovernedServer.of(inner)
    assert GovernedServer.of(governed) is governed  # governed once
    assert governed.name == "crm"
    await governed.connect()
    tools = await governed.list_tools()
    assert [t.name for t in tools] == ["find", "erase"]
    assert await governed.list_prompts() == [] and await governed.get_prompt("p") is None
    result = await governed.call_tool("find", {"order": "o1"})  # outside a run: as it is
    assert isinstance(result, CallToolResult) and inner.called == [("find", {"order": "o1"})]
    await governed.cleanup()
    assert not inner.connected


# --------------------------------------------------------------------------- Claude


async def test_claudes_own_tools_stay_offered_whatever_the_hints(
    memory_harness: Harness, memory_service: FakeMemoryService, tmp_path: Path, narrowing: None
) -> None:
    """Claude's tools sit in its cached prompt prefix: the hints narrow none of them (a set
    changed per turn rewrote the prefix every turn); the team's own ``disallowed_tools`` stay."""
    from claude_agent_sdk import ClaudeAgentOptions

    from tests.integration.test_claude import CLI

    memory_service.candidates = ["Read", "Grep", "Bash", "Write", "Edit"]
    record = tmp_path / "cli.json"
    options = ClaudeAgentOptions(
        cli_path=CLI,
        disallowed_tools=["WebFetch"],
        env={"FAKE_CLAUDE_SCRIPT": json.dumps([{"text": "ok"}]), "FAKE_CLAUDE_RECORD": str(record)},
    )
    await memory_harness.wrap(options, id="claude-narrowed").run("read it", user="u1")
    assert json.loads(record.read_text())["disallowed_tools"] == "WebFetch"


# --------------------------------------------------------------------------- the set


def test_a_tool_set_is_sorted_and_each_name_once_the_first_wins() -> None:
    first, second = ToolSpec(name="b", description="first"), ToolSpec(name="b", description="2")
    assert [s.description for s in unique([first, ToolSpec(name="a"), second])] == ["", "first"]
    assert unique([]) == []
    assert [s.name for s in unique([ToolSpec(name="é"), ToolSpec(name="e")])] == ["e", "é"]
    tagged = helper(ToolSpec(name="t", side_effects="write", tags=["x"]))
    assert tagged.tags == ["x", HELPER] and tagged.side_effects == "read"


async def test_the_framework_tools_are_published_once_per_set_and_tenant(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    model = ScriptedChatModel(turns=["one", "two"])
    agent = memory_harness.wrap(create_agent(model, tools=[lookup]), id="once")
    for _ in range(2):
        await agent.run("x", user="u")
    await memory_harness.writes.drain()
    sent = [c for c in memory_service.named("put_catalog") if "lookup" in json.dumps(c.body)]
    assert len(sent) == 1


def test_a_graphs_tools_are_read_to_a_depth_and_past_an_empty_cell() -> None:
    from deepagents import create_deep_agent

    clerk = SubAgent(name="clerk", description="Refunds.", system_prompt="s", tools=[refund])
    graph = create_deep_agent(model=ScriptedChatModel(turns=[]), subagents=[clerk])
    assert "refund" not in {t.name for t in graph_tools(graph, depth=0)}  # the top only
    assert "refund" in {t.name for t in graph_tools(graph)}

    def holder() -> Any:
        held = None

        def task_code() -> Any:
            return held

        # the same code over an empty cell (a variable its scope has not assigned yet)
        return types.FunctionType(
            task_code.__code__, task_code.__globals__, "task_code", None, (types.CellType(),)
        )

    @langchain_tool
    def task(description: str) -> str:
        """A task tool whose code holds an empty cell."""
        return description

    task.coroutine = holder()
    assert subagent_graphs(task) == []


async def test_an_mcp_servers_error_and_a_rejected_call_reach_the_model_as_errors(
    harness: Harness,
) -> None:
    from mcp.types import CallToolResult

    from tests.support.catalog import FakeCatalog
    from trellis.contracts import ToolStatus
    from trellis.harness.adapters.openai_agents import GovernedServer
    from trellis.harness.governance import Governance
    from trellis.harness.governance.catalog import Rule
    from trellis.harness.runtime import current

    inner = mcp_server()
    governed = GovernedServer.of(inner)
    results: list[CallToolResult] = []

    async def call(input: str, agent: Any) -> str:
        assert current() is not None
        results.append(await governed.call_tool("boom", {}))
        results.append(await governed.call_tool("find", {"order": "o1"}))
        return "ok"

    async with with_governance(Harness(), Governance(FakeCatalog({"boom": Rule("read")}))) as h:
        assert (await h.wrap(call, id="mcp-errors").run("x", user="u")).status is RunStatus.SUCCESS
    broke, found = results
    assert broke.is_error and broke.content[0].text == "boom failed: it broke"  # type: ignore[union-attr]
    assert not found.is_error and found.content[0].text == "find done"  # type: ignore[union-attr]
    assert ToolStatus.ERROR.value == "error"


async def test_an_agent_tool_of_an_unknown_shape_runs_its_agent_as_it_is(
    harness: Harness, caplog: pytest.LogCaptureFixture
) -> None:
    """An agent tool whose invoke this SDK release does not build: its agent's own tools
    cannot be reached, said once; the tool itself is governed as any."""
    import dataclasses
    import logging

    from agents import FunctionTool

    from trellis.harness.adapters.openai_agents import _running, governed_agents
    from trellis.harness.tools.convert.openai_agents import AGENT_OF_TOOL, AGENT_TOOL

    @function_tool
    def peek(order: str) -> str:
        return "fine"

    inner = Agent(name="inner", tools=[peek])

    async def invoke(context: Any, arguments: str) -> str:
        return "checked"

    odd = FunctionTool(
        name="check",
        description="Check.",
        params_json_schema={"type": "object", "properties": {}},
        on_invoke_tool=invoke,
    )
    setattr(odd, AGENT_TOOL, True)
    setattr(odd, AGENT_OF_TOOL, inner)
    assert _running(odd, inner) is None
    assert _running(dataclasses.replace(odd), inner) is None
    empty = dataclasses.replace(odd, name="ask")
    setattr(empty, AGENT_OF_TOOL, Agent(name="bare"))  # an agent with no tools: nothing to say
    governed_agents(Agent(name="desk", tools=[empty]), added=[], agent_id="d")
    with caplog.at_level(logging.WARNING, logger="trellis.harness"):
        copied = governed_agents(Agent(name="desk", tools=[odd]), added=[], agent_id="d")
    assert [t.name for t in copied.tools] == ["check"]
    assert any("agent called as a tool" in r.getMessage() for r in caplog.records)


async def test_a_hosted_mcp_tool_without_a_list_and_a_tool_set_that_cannot_be_read(
    memory_harness: Harness, memory_service: FakeMemoryService, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agents import HostedMCPTool

    from trellis.contracts import RunEventType
    from trellis.harness.adapters import openai_agents

    hosted = HostedMCPTool(tool_config={"type": "mcp", "server_label": "gh", "server_url": "u"})
    specs = await openai_agents.OpenAIAgentsAdapter().native(Agent(name="a", tools=[hosted]))
    assert specs == []  # its tools are listed by the provider only

    async def broken(self: Any, target: Any) -> list[ToolSpec]:
        raise RuntimeError("cannot read the tools")

    monkeypatch.setattr(openai_agents.OpenAIAgentsAdapter, "native", broken)
    model = ScriptedModel(["done"])
    agent = memory_harness.wrap(Agent(name="b", model=model), id="unread")
    events = [e async for e in agent.stream("x", user="u")]
    said = [e.data for e in events if e.type is RunEventType.CUSTOM]
    assert any(d.get("code") == "native_tools" for d in said)
    assert events[-1].outcome is not None and events[-1].outcome.value == "success"


async def test_an_agents_own_tool_keeps_its_own_enabled_rule(harness: Harness) -> None:
    @function_tool(is_enabled=lambda context, agent: False)
    def hidden(order: str) -> str:
        return "never offered"

    @function_tool(is_enabled=lambda context, agent: True)
    def shown(order: str) -> str:
        return "offered"

    model = ScriptedModel(["done"])
    await harness.wrap(Agent(name="c", model=model, tools=[hidden, shown]), id="own-rule").run(
        "x", user="u"
    )
    assert model.tools[0] == ["shown"]
