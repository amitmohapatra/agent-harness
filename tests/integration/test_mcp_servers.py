"""MCP servers as ``tools=[...]`` entries, against a real server on the official SDK
(``tests/support/mcp_server.py``) over stdio and over HTTP: its tools named ``<server>-<tool>``,
their risk from their annotations (a read runs, a change is announced, a destructive one asks),
every call journaled (a resume runs nothing twice) and on the run's events; a server that dies
is reached again; a Virtual MCP named in the list is the gateway's; one path per server; and
``mcp=`` and ``as_tool()`` still work, said deprecated."""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import subprocess
import sys
import time
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import pytest

from tests.support.gateway import URL, FakeGateway
from trellis import Harness, Runtime, Settings, mcp, stdio
from trellis.contracts import ConfigurationError, RunEventType, RunStatus
from trellis.harness.tools.mcp import (
    MCPSource,
    retired_as_tool,
    retired_mcp,
    server_name,
    side_effects_of,
)

SERVER = Path(__file__).parents[1] / "support" / "mcp_server.py"


def local(**env: str) -> MCPSource:
    return mcp(stdio(sys.executable, str(SERVER), env=env or None))


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture(scope="module")
def http_server() -> Iterator[str]:
    """The server over HTTP, its every request needing the token ``t0k``."""
    port = free_port()
    process = subprocess.Popen(
        [sys.executable, str(SERVER), "http", str(port)],
        env={**os.environ, "TOKEN": "t0k"},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                break
        time.sleep(0.1)
    yield f"http://127.0.0.1:{port}/mcp"
    process.terminate()
    process.wait(timeout=10)


async def names_and(agent_fn: Any, harness: Harness, *tools: Any, **wrap: Any) -> Any:
    return await harness.wrap(agent_fn, id="notes", tools=list(tools), **wrap).run("go", user="u")


async def toolbox(input: str, agent: Runtime) -> list[str]:
    return sorted(agent.toolbox)


# --------------------------------------------------------------------------- stdio
async def test_a_local_servers_tools_are_named_by_it_and_judged_by_their_annotations(
    harness: Harness,
) -> None:
    seen: list[Any] = []

    async def notes(input: str, agent: Runtime) -> Any:
        read = await agent.tools.call("notes_box-read_note", path="a.md")
        added = await agent.tools.call("notes_box-add_note", text="hi")
        seen.extend([sorted(agent.toolbox), read, added])
        return "done"

    agent = harness.wrap(notes, id="notes", tools=[local()])
    events = [e async for e in agent.stream("go", user="u")]
    names, read, added = seen
    assert names == [
        "notes_box-add_note",
        "notes_box-crash",
        "notes_box-delete_note",
        "notes_box-fail",
        "notes_box-read_note",
        "notes_box-served_calls",
    ]
    assert (read, added) == ("note at a.md", "added hi")
    notices = [
        e for e in events if e.type is RunEventType.CUSTOM and e.data["name"] == "tool_notice"
    ]
    assert [n.data["tool"] for n in notices] == ["notes_box-add_note"]  # a change, announced


async def test_a_destructive_tool_asks_and_a_resume_runs_nothing_twice(harness: Harness) -> None:
    async def cleaner(input: str, agent: Runtime) -> Any:
        await agent.tools.call("notes-add_note", text="x")
        deleted = await agent.tools.call("notes-delete_note", path="a.md")
        return [deleted, await agent.tools.call("notes-served_calls")]

    agent = harness.wrap(
        cleaner, id="cleaner", tools=[mcp(stdio(sys.executable, str(SERVER)), name="Notes")]
    )
    paused = await agent.run("clean", user="u")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
    assert paused.interrupt.question.startswith("Approve notes-delete_note?")
    done = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="r")
    deleted, served = done.answer
    assert deleted == "deleted a.md"
    assert served == ["add_note", "delete_note"]  # the change before the pause ran once


async def test_an_error_the_server_reports_is_an_error_the_agent_reads(harness: Harness) -> None:
    async def failing(input: str, agent: Runtime) -> Any:
        return await agent.tools.call("notes_box-fail", why="the disk is full")

    said = (await names_and(failing, harness, local())).answer
    assert said.startswith("notes_box-fail failed") and "Error executing tool fail" in said


async def test_a_server_that_died_is_started_again(harness: Harness, tmp_path: Path) -> None:
    async def twice(input: str, agent: Runtime) -> Any:
        return await agent.tools.call("notes_box-crash")

    result = await names_and(twice, harness, local(MARK=str(tmp_path / "mark")))
    assert result.answer == "back"  # a read: tried again, on a new process
    assert (tmp_path / "mark").read_text() == "crashed"


async def test_without_mcp_the_server_is_not_started(harness: Harness) -> None:
    source = local()
    result = await names_and(toolbox, harness, source, without={"mcp"})
    assert result.answer == [] and source._connection._task is None


# --------------------------------------------------------------------------- HTTP
async def test_a_server_over_http_with_its_token(harness: Harness, http_server: str) -> None:
    async def reader(input: str, agent: Runtime) -> Any:
        return await agent.tools.call("box-read_note", path="b.md")

    token = mcp(http_server, name="box", headers={"Authorization": "Bearer t0k"})
    assert (await names_and(reader, harness, token)).answer == "note at b.md"
    refused = await harness.wrap(reader, id="refused", tools=[mcp(http_server)]).run("go", user="u")
    assert refused.status is RunStatus.ERROR  # no token: the server refuses the listing


# --------------------------------------------------------------------------- the gateway
@pytest.fixture
async def gw(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[tuple[Harness, FakeGateway]]:
    fake = FakeGateway()
    fake.serve_models(monkeypatch)
    settings = Settings(bifrost_url=URL, bifrost_virtual_key="vk")
    async with Harness(config=settings, gateway=fake.gateway()) as made:
        yield made, fake


async def test_mcp_entries_name_exactly_the_agents_mcp_tools(
    gw: tuple[Harness, FakeGateway],
) -> None:
    h, fake = gw
    fake.bundles = {"": ["erp-pay", "crm-get"], "finance": ["erp-pay"]}
    assert (await names_and(toolbox, h)).answer == ["crm-get", "erp-pay"]  # none: the key's
    named = await h.wrap(toolbox, id="fin", tools=[mcp("finance")]).run("go", user="u")
    assert named.answer == ["erp-pay"]
    own = await h.wrap(toolbox, id="own", tools=[local()]).run("go", user="u")
    assert "erp-pay" not in own.answer and "notes_box-read_note" in own.answer


async def test_one_path_per_server(gw: tuple[Harness, FakeGateway]) -> None:
    h, fake = gw
    fake.bundles = {"": ["erp-pay"], "finance": ["erp-pay"]}
    both = mcp(stdio(sys.executable, str(SERVER)), name="erp")
    found = await h.wrap(toolbox, id="both", tools=[mcp("finance"), both]).run("go", user="u")
    assert found.status is RunStatus.ERROR and "both through the gateway" in found.error.message  # type: ignore[union-attr]
    twice = await h.wrap(toolbox, id="twice", tools=[local(), local()]).run("go", user="u")
    assert "two mcp(...) entries are the server notes_box" in twice.error.message  # type: ignore[union-attr]


def test_a_virtual_mcp_needs_the_gateway(harness: Harness) -> None:
    with pytest.raises(ConfigurationError, match="a Virtual MCP, needs a gateway with mcp"):
        harness.wrap(toolbox, id="v", tools=[mcp("finance-tools")])
    harness.wrap(toolbox, id="off", tools=[mcp("finance-tools")], without={"mcp"})


def test_what_mcp_takes() -> None:
    assert mcp("https://h/mcp").kind == "http" and mcp("finance").virtual == "finance"
    for wrong in ("not a name", ""):
        with pytest.raises(ConfigurationError, match="give a server's URL"):
            mcp(wrong)
    with pytest.raises(ConfigurationError, match="takes no name"):
        mcp("finance", name="f")
    with pytest.raises(ConfigurationError, match="no letter or digit"):
        mcp("https://h/mcp", name="--")
    with pytest.raises(ConfigurationError, match="needs the command"):
        stdio("")
    assert server_name("Notes Box!") == "notes_box"
    assert side_effects_of(None) == "write"
    assert repr(mcp("finance")) == "mcp('finance')"


# --------------------------------------------------------------------------- deprecated
async def test_mcp_and_as_tool_still_work_said_deprecated(
    gw: tuple[Harness, FakeGateway], caplog: pytest.LogCaptureFixture
) -> None:
    h, fake = gw
    fake.bundles = {"": ["erp-pay"], "finance": ["erp-pay"]}
    caplog.set_level(logging.WARNING)
    retired_mcp.cache_clear()
    retired_as_tool.cache_clear()
    old = await h.wrap(toolbox, id="old", mcp=["finance"]).run("go", user="u")
    assert old.answer == ["erp-pay"]
    await h.tools(framework="langgraph", mcp=["finance"])
    child = h.wrap(toolbox, id="child")
    child.as_tool()
    said = caplog.text
    assert "h.wrap(mcp=[...]) is deprecated" in said and "h.tools(mcp=[...]) is deprecated" in said
    assert "agent.as_tool() is deprecated" in said


async def test_an_agent_in_the_list_is_a_sub_agent(harness: Harness) -> None:
    async def child(input: Any, agent: Runtime) -> str:
        return f"child saw {input}"

    async def parent(input: str, agent: Runtime) -> Any:
        return await agent.tools.call("helper", message="x")

    helper = harness.wrap(child, id="helper")
    result = await harness.wrap(parent, id="parent", tools=[helper]).run("go", user="u")
    assert result.answer == "child saw x", result.error


# --------------------------------------------------------------------------- connection
def test_a_connection_is_opened_again_in_another_event_loop() -> None:
    source = local()

    async def listed() -> list[str]:
        return [t.name for t in await source.resolve()]

    first = asyncio.run(listed())
    second = asyncio.run(listed())  # the first loop's connection is gone
    assert first == second and "notes_box-read_note" in first
    asyncio.run(source.aclose())


# --------------------------------------------------------------------------- h.tools
async def test_h_tools_gives_every_framework_the_servers_tools(harness: Harness) -> None:
    """``h.tools(mcp(...), framework=...)``: the server's tools in each framework's own form,
    each call still the harness's (governed, journaled)."""
    from agents import Agent as OpenAIAgent
    from langchain.agents import create_agent

    from tests.support.chat_model import ScriptedChatModel
    from tests.support.openai_model import ScriptedModel

    names = {f"notes_box-{t}" for t in ("read_note", "add_note", "delete_note")}
    for framework in ("langgraph", "deepagents", "openai_agents"):
        built = await harness.tools(local(), framework=framework)  # type: ignore[arg-type]
        assert names <= {t.name for t in built}, framework
    server = await harness.tools(local(), framework="claude_agent_sdk")
    assert server["name"] == "trellis" and server["type"] == "sdk"

    chat = ScriptedChatModel(turns=[("notes_box-read_note", {"path": "g.md"}), "read it"])
    graph = create_agent(chat, tools=await harness.tools(local(), framework="langgraph"))
    assert (await harness.wrap(graph, id="g").run("x", user="u")).answer == "read it"
    assert "note at g.md" in str(chat.seen[-1][-1].content)

    sdk = ScriptedModel([("notes_box-read_note", {"path": "o.md"}), "read too"])
    tools = await harness.tools(local(), framework="openai_agents")
    sdk_agent = harness.wrap(OpenAIAgent(name="o", model=sdk, tools=tools), id="o")
    assert (await sdk_agent.run("x", user="u")).answer == "read too"
