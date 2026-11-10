"""Every tool says what it does, or a person is asked.

A tool defined in code declares its side effects where it is defined (``@tool``, ``tool(fn)``,
a bare function in ``tools=[...]``, ``governed``): one that does not is a configuration error
there, naming it and what to add. A tool nobody could declare in code — an MCP server's with no
annotations, a framework's own tool object that says nothing — is never run on a guess: each
call waits for a person, on every adapter, and the approval covers that one call. The three
cases a live test found running unasked are here as planned runs: a purge, a bulk price change
and another account's data."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
from agents import Agent, function_tool
from deepagents import create_deep_agent
from langchain.agents import create_agent
from langchain_core.tools import tool as lc_tool
from langgraph.checkpoint.memory import InMemorySaver

from tests.support.adapters import BUILDERS
from tests.support.arm import arm
from tests.support.chat_model import ScriptedChatModel
from tests.support.memory import FakeMemoryService
from tests.support.openai_model import ScriptedModel
from tests.support.planned import Call
from trellis import Harness, ReAct, Runtime, mcp, stdio, tool
from trellis.contracts import ConfigurationError, InterruptReason, RunEventType, RunStatus, ToolSpec
from trellis.harness.adapters.langgraph import graph_tools
from trellis.harness.governance import Governance, governed
from trellis.harness.governance.catalog import entry
from trellis.harness.governance.decision import Action, decide
from trellis.harness.middleware import GovernTools, declare_files, spec_of
from trellis.harness.tools.base import Tool
from trellis.harness.tools.mcp import MCPSource
from trellis.harness.tools.sources import as_source

SERVER = Path(__file__).parents[1] / "support" / "mcp_undeclared_server.py"
#: The three calls the live test saw run unasked, each with the arguments of a second call.
RISKY: list[tuple[str, dict[str, Any], dict[str, Any]]] = [
    ("purge_records", {"table": "orders"}, {"table": "customers"}),
    ("set_prices", {"percent": -90}, {"percent": -50}),
    ("read_account", {"account": "other-co"}, {"account": "third-co"}),
]


def back_office(ran: Path) -> MCPSource:
    """The server whose tools declare nothing; ``ran`` gets a line for each body that ran."""
    return mcp(stdio(sys.executable, str(SERVER), env={"RAN": str(ran)}), name="office")


def lines(ran: Path) -> list[str]:
    return ran.read_text().splitlines() if ran.exists() else []


# --------------------------------------------------------------------------- defined in code


def purge(table: str) -> str:
    """Purge every record of a table."""
    raise AssertionError("a tool that declares nothing never runs")


def test_a_function_that_declares_nothing_is_refused_where_it_is_made_a_tool() -> None:
    ways: dict[str, Any] = {
        "@tool": lambda: tool(purge),  # pyright: ignore[reportCallIssue]
        "tool(fn, name=)": lambda: tool(purge, name="wipe"),  # pyright: ignore[reportCallIssue]
        "tools=[fn]": lambda: as_source(purge),
        "governed(fn)": lambda: governed(purge, Governance(), on_ask=lambda decision: True),
        "another word": lambda: tool(purge, side_effects="delete"),  # pyright: ignore
    }
    for way, make in ways.items():
        with pytest.raises(ConfigurationError) as refused:
            make()
        said = str(refused.value)
        named = "wipe" if way == "tool(fn, name=)" else "purge"
        assert said.startswith(f"{named} must declare its side effects: add side_effects="), way
        assert '"read"' in said and '"write"' in said and '"irreversible"' in said, way


def test_a_bare_function_among_an_agents_tools_is_refused_when_it_is_wrapped(
    harness: Harness,
) -> None:
    built = ReAct("Tidy up.", model=ScriptedChatModel(turns=["x"]), tools=[purge])
    with pytest.raises(ConfigurationError, match="purge must declare its side effects"):
        harness.wrap(built, id="wiper")


async def test_a_function_targets_own_tool_that_declares_nothing_is_refused(
    harness: Harness,
) -> None:
    async def target(input: str, agent: Runtime) -> Any:
        return await agent.tools.call(purge, table="orders")

    result = await harness.wrap(target, id="own").run("go", user="u")
    assert result.status is RunStatus.ERROR and result.error is not None
    assert "purge must declare its side effects" in result.error.message


async def test_a_declared_tool_runs_as_its_risk_says(harness: Harness) -> None:
    done: list[str] = []

    @tool(side_effects="read")
    def count(table: str) -> int:
        """Count a table's records."""
        done.append("count")
        return 3

    @tool(side_effects="write")
    def archive(table: str) -> str:
        """Archive a table."""
        done.append("archive")
        return f"archived {table}"

    async def target(input: str, agent: Runtime) -> Any:
        count = await agent.tools.call("count", table="t")
        return [count, await agent.tools.call("archive", table="t")]

    agent = arm(harness.wrap(target, id="declared"), [count, archive])
    events = [e async for e in agent.stream("go", user="u")]
    notices = [
        e.data["tool"]
        for e in events
        if e.type is RunEventType.CUSTOM and e.data.get("name") == "tool_notice"
    ]
    assert done == ["count", "archive"] and notices == ["archive"]  # nobody asked


# --------------------------------------------------------------------------- nobody's to declare


def test_a_risk_nobody_declared_asks() -> None:
    decision = decide("purge", ToolSpec(name="purge").side_effects, None, {})
    assert decision.action is Action.ASK
    assert decision.question == "Approve purge? purge has not declared its side effects."


async def test_a_check_that_names_no_side_effects_asks() -> None:
    governance = Governance()
    assert (await governance.check("purge", {})).asks
    assert await governance.may_ask([ToolSpec(name="purge")]) == {"purge"}


def test_the_catalog_is_told_the_default_of_a_tool_that_says_nothing() -> None:
    """The service derives a tool's risk from its annotations: one the harness holds
    irreversible for what it does not say is published saying so, never with a
    ``side_effects`` of its own — an administrator's word in the catalog stays."""
    silent = ToolSpec(name="office-purge_records", source="mcp", server="office")
    assert entry(silent, None)["annotations"] == {"destructiveHint": True}
    assert entry(ToolSpec(name="native"), None)["annotations"] == {"destructiveHint": True}
    for sent in (entry(silent, None), entry(ToolSpec(name="native"), None)):
        assert "side_effects" not in sent
    assert entry(silent, {"readOnlyHint": True})["annotations"] == {"readOnlyHint": True}
    assert entry(silent, {"destructiveHint": False})["annotations"] == {"destructiveHint": False}
    said = entry(ToolSpec(name="count", side_effects="read"), None)
    assert said["side_effects"] == "read" and "annotations" not in said


@pytest.mark.parametrize("framework", list(BUILDERS))
@pytest.mark.parametrize("risky", RISKY, ids=[name for name, _, _ in RISKY])
async def test_a_servers_tool_that_declares_nothing_never_runs_unapproved_on_any_adapter(
    harness: Harness,
    framework: str,
    risky: tuple[str, dict[str, Any], dict[str, Any]],
    tmp_path: Path,
) -> None:
    """The purge, the bulk price change and another account's data: each call pauses for a
    person before its body runs; an approval runs that one call, the next asks again, and a
    rejected call never runs."""
    name, args, again = risky
    ran = tmp_path / "ran"
    plan: list[Call] = [(f"office-{name}", args), (f"office-{name}", again)]
    target, tools = await BUILDERS[framework](harness, [back_office(ran)], tmp_path, plan)
    agent = arm(harness.wrap(target, id=f"office-{framework}"), tools)
    first = await agent.run("do it", user="ada")
    assert first.status is RunStatus.PAUSED and first.interrupt is not None, first
    assert first.interrupt.reason is InterruptReason.APPROVAL
    assert first.interrupt.question == f"Approve office-{name}? office-{name} is irreversible."
    assert first.interrupt.tool_call is not None and first.interrupt.tool_call.args == args
    assert lines(ran) == []  # nothing ran
    second = await agent.resume(first.interrupt.interrupt_id, "approve", reviewer="lee")
    assert second.status is RunStatus.PAUSED and second.interrupt is not None, second
    assert second.interrupt.tool_call is not None and second.interrupt.tool_call.args == again
    assert lines(ran) == [name]  # the approved call, once: the approval covers no other
    await agent.resume(second.interrupt.interrupt_id, "reject", reviewer="lee")
    assert lines(ran) == [name]


async def test_the_catalogs_word_on_a_tool_that_says_nothing(
    memory_harness: Harness, memory_service: FakeMemoryService, tmp_path: Path
) -> None:
    """With the tool catalog, what the harness published keeps the tool asking, and an
    administrator's risk — the one place a tool nobody could declare is declared — lets it run."""
    ran = tmp_path / "ran"

    async def reader(input: str, agent: Runtime) -> Any:
        return await agent.tools.call("office-read_account", account="ours")

    agent = arm(memory_harness.wrap(reader, id="reader"), [back_office(ran)])
    paused = await agent.run("read", user="u")  # the catalog says nothing yet
    assert paused.status is RunStatus.PAUSED and lines(ran) == []
    await memory_harness.writes.drain()
    published = [
        t
        for call in memory_service.named("put_catalog")
        for t in call.body["tools"]
        if t["name"] == "office-read_account"
    ]
    assert published and "side_effects" not in published[-1], published
    assert published[-1]["annotations"] == {"destructiveHint": True}
    memory_service.catalog["office-read_account"] = dict(published[-1])  # as the service keeps it
    still = await arm(memory_harness.wrap(reader, id="reader-2"), [back_office(ran)]).run(
        "read", user="u"
    )
    assert still.status is RunStatus.PAUSED and lines(ran) == []  # the catalog derives the same
    memory_service.catalog["office-read_account"]["side_effects"] = "read"  # an administrator's
    done = await arm(memory_harness.wrap(reader, id="reader-3"), [back_office(ran)]).run(
        "read", user="u"
    )
    assert done.status is RunStatus.SUCCESS and done.answer == "the data of ours"
    assert lines(ran) == ["read_account"]


# --------------------------------------------------------------------------- a framework's own


async def test_a_graphs_own_tool_that_declares_nothing_asks(harness: Harness) -> None:
    done: list[str] = []

    @lc_tool
    def purge_records(table: str) -> str:
        """Purge every record of a table."""
        done.append(table)
        return f"purged {table}"

    model = ScriptedChatModel(turns=[[("purge_records", {"table": "orders"})], "purged"])
    graph = create_agent(
        model, tools=[purge_records], middleware=[GovernTools()], checkpointer=InMemorySaver()
    )
    agent = harness.wrap(graph, id="graph")
    paused = await agent.run("purge", user="u", thread="t")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None and done == []
    assert paused.interrupt.question == (
        "Approve purge_records? purge_records has not declared its side effects."
    )
    finished = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="r")
    assert finished.answer == "purged" and done == ["orders"]


async def test_a_graphs_own_tool_says_what_it_does_in_its_metadata(harness: Harness) -> None:
    done: list[str] = []

    @lc_tool
    def read_account(account: str) -> str:
        """Read an account's data."""
        done.append(account)
        return f"the data of {account}"

    read_account.metadata = {"side_effects": "read"}
    model = ScriptedChatModel(turns=[[("read_account", {"account": "ours"})], "read"])
    graph = create_agent(model, tools=[read_account], middleware=[GovernTools()])
    result = await harness.wrap(graph, id="graph").run("read", user="u")
    assert result.answer == "read" and done == ["ours"]


async def test_an_sdk_agents_own_tool_asks(harness: Harness) -> None:
    done: list[int] = []

    @function_tool
    def set_prices(percent: int) -> str:
        """Change every price by a percentage."""
        done.append(percent)
        return f"prices changed by {percent}%"

    target = Agent(name="s", model=ScriptedModel([("set_prices", {"percent": -90}), "changed"]))
    target.tools.append(set_prices)
    agent = harness.wrap(target, id="sdk")
    paused = await agent.run("cut prices", user="u")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None and done == []
    assert paused.interrupt.question == (
        "Approve set_prices? set_prices has not declared its side effects."
    )
    finished = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="r")
    assert finished.answer == "changed" and done == [-90]


async def test_a_graphs_own_tool_named_as_a_built_in_is_not_the_built_in(harness: Harness) -> None:
    """The harness knows the framework's built-ins by name (Deep Agents' file tools, the
    graph's state tools); a tool of the agent's own that shares a name is not one of them."""
    done: list[str] = []

    @lc_tool
    def delete(path: str) -> str:
        """Delete a customer's folder."""
        done.append(path)
        return f"deleted {path}"

    model = ScriptedChatModel(turns=[[("delete", {"path": "/acme"})], "deleted"])
    graph = create_agent(model, tools=[delete], checkpointer=InMemorySaver())
    paused = await harness.wrap(graph, id="named").run("delete it", user="u", thread="t")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None and done == []
    assert paused.interrupt.question == "Approve delete? delete has not declared its side effects."


def test_deep_agents_execute_is_declared_a_write_and_a_tool_of_that_name_is_not() -> None:
    @lc_tool
    def execute(command: str) -> str:
        """Run a command on the production host."""
        return command

    deep = create_deep_agent(model=ScriptedChatModel(turns=["x"]))
    own = create_agent(ScriptedChatModel(turns=["x"]), tools=[execute])
    for graph in (deep, own):
        declare_files(graph)
    theirs = [t for t in graph_tools(deep) if t.name == "execute"]
    assert theirs and {spec_of(t).side_effects for t in theirs} == {"write"}
    [mine] = [t for t in graph_tools(own) if t.name == "execute"]
    assert spec_of(mine).side_effects == "unknown"  # it asks


class Silent:
    """A source of the team's own whose tool's ``ToolSpec`` says nothing of what it does."""

    async def resolve(self) -> list[Tool]:
        async def run(args: dict[str, Any]) -> Any:
            raise AssertionError("a tool that declares nothing never runs")

        return [Tool(ToolSpec(name="wipe", description="Wipe the store."), run)]


@pytest.mark.parametrize("given", ["declared on the agent", "added in the run"])
async def test_a_source_of_yours_whose_tool_says_nothing_is_refused(
    harness: Harness, given: str
) -> None:
    async def target(input: str, agent: Runtime) -> Any:
        if given == "added in the run":
            await agent.tools.use(Silent())
        return await agent.tools.call("wipe")

    agent = harness.wrap(target, id="silent")
    if given == "declared on the agent":
        arm(agent, [Silent()])
    result = await agent.run("go", user="u")
    assert result.status is RunStatus.ERROR and result.error is not None
    assert result.error.message.startswith(
        "wipe must declare its side effects: add side_effects= to its ToolSpec"
    )
