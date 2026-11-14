"""An OpenAI agent's own MCP servers (the SDK's ``MCPServerStdio`` and
``MCPServerStreamableHttp`` against a real server, ``tests.support.mcp_server``), run through
the harness: the SDK's own settings of a server — its tool filter, its approval policy, its
structured content, its resources — work as the SDK says, and each call of its tools is the
run's: governed by the tool's annotations, journaled (a resume runs nothing twice) and
recorded."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from agents import Agent
from agents.mcp import MCPServerStdio, MCPServerStreamableHttp, create_static_tool_filter

from tests.support.memory import MEMORY_TOOLS, FakeMemoryService
from tests.support.openai_model import ScriptedModel
from tests.support.planned import PlannedModel
from trellis import Harness
from trellis.contracts import RunStatus, ToolOutcome, ToolStatus
from trellis.harness.adapters.openai_agents import GovernedServer
from trellis.harness.hooks import Hooks

SERVER = Path(__file__).resolve().parents[1] / "support" / "mcp_server.py"


def stdio(**settings: Any) -> MCPServerStdio:
    """The notes server over stdio, as the SDK starts it."""
    return MCPServerStdio(
        params={"command": sys.executable, "args": [str(SERVER)]},
        name="notes",
        client_session_timeout_seconds=30,
        **settings,
    )


@pytest.fixture(scope="module")
def http_url() -> Iterator[str]:
    """The notes server over streamable HTTP, every request needing the token ``t0k``."""
    with socket.socket() as found:
        found.bind(("127.0.0.1", 0))
        port = int(found.getsockname()[1])
    process = subprocess.Popen(
        [sys.executable, str(SERVER), "http", str(port)],
        env={**os.environ, "TOKEN": "t0k"},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                break
        time.sleep(0.1)
    yield f"http://127.0.0.1:{port}/mcp"
    process.terminate()
    process.wait(timeout=10)


def recorded(memory_service: FakeMemoryService) -> list[tuple[str, str]]:
    return [(c.body["tool"], c.body["status"]) for c in memory_service.named("record_tool")]


def said(output: Any) -> str:
    """A tool result the model was given, as text."""
    if isinstance(output, str):
        return output
    return "".join(part.get("text", "") for part in output)


def results(model: ScriptedModel) -> list[str]:
    return [
        said(item["output"])
        for item in model.inputs[-1]
        if isinstance(item, dict) and item.get("type") == "function_call_output"
    ]


def own(offered: list[str]) -> list[str]:
    """The tools a call was offered, the memory tools left out."""
    return [name for name in offered if name not in MEMORY_TOOLS]


async def test_a_stdio_servers_tools_are_filtered_by_the_sdk_and_governed_by_the_harness(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    """``tool_filter`` (the SDK's static filter) decides what the model is offered; a read
    runs, a destructive tool (its annotations) waits for a person, and the resume — which runs
    the agent again against the journal — calls the server once for each."""
    allowed = create_static_tool_filter(
        allowed_tool_names=["read_note", "delete_note", "served_calls"]
    )
    async with stdio(tool_filter=allowed) as server:
        plan = [("read_note", {"path": "a"}), ("delete_note", {"path": "a"}), ("served_calls", {})]
        model = PlannedModel(plan, final="done")
        agent = memory_harness.wrap(
            Agent(name="desk", model=model, mcp_servers=[server]),
            id="filtered",
            without={"schedules"},
        )
        paused = await agent.run("tidy the notes", user="u")
        assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
        assert paused.interrupt.tool_call is not None
        assert paused.interrupt.tool_call.tool == "delete_note"
        assert paused.interrupt.question.startswith("Approve delete_note?")
        assert own(model.tools[0]) == ["read_note", "delete_note", "served_calls"]
        done = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="r")
        assert done.status is RunStatus.SUCCESS and done.answer == "done"
        served = results(model)[-1]
        assert served.count("read_note") == 1 and served.count("delete_note") == 1
        await memory_harness.writes.drain()
    assert recorded(memory_service) == [
        ("read_note", "ok"),
        ("delete_note", "ok"),
        ("served_calls", "ok"),
    ]


async def test_a_servers_dynamic_filter_and_tool_cache_are_the_sdks(harness: Harness) -> None:
    def reads_only(context: Any, tool: Any) -> bool:
        return tool.name.startswith("read")

    async with stdio(tool_filter=reads_only, cache_tools_list=True) as server:
        model = PlannedModel([("read_note", {"path": "b"})], final="done")
        agent = harness.wrap(
            Agent(name="desk", model=model, mcp_servers=[server]),
            id="dynamic",
            without={"schedules"},
        )
        result = await agent.run("read b", user="u")
        assert result.status is RunStatus.SUCCESS and results(model) == ["note at b"]
        assert own(model.tools[0]) == ["read_note"]


@pytest.mark.parametrize("decision", ["approve", "reject"])
async def test_a_servers_own_approval_policy_pauses_the_run_as_the_sdk_does(
    memory_harness: Harness, memory_service: FakeMemoryService, decision: str
) -> None:
    """``require_approval`` (the SDK's own, per tool): the run pauses on the SDK's approval
    with its saved state; approved, the call is made once — still the run's call, recorded —
    and the model is not asked again; rejected, the server is never called and the model reads
    why. A tool the policy does not name runs unasked."""
    policy = {"always": {"tool_names": ["read_note"]}, "never": {"tool_names": ["served_calls"]}}
    async with stdio(require_approval=policy) as server:
        model = ScriptedModel([("read_note", {"path": "a"}), ("served_calls", {}), "done"])
        agent = memory_harness.wrap(
            Agent(name="desk", model=model, mcp_servers=[server]),
            id=f"policy-{decision}",
            without={"schedules"},
        )
        paused = await agent.run("read a", user="u")
        assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
        assert paused.interrupt.question == "Approve read_note?"  # the SDK's own approval
        assert paused.interrupt.tool_call is not None
        assert paused.interrupt.tool_call.args == {"path": "a"}
        done = await agent.resume(
            paused.interrupt.interrupt_id, decision, answer="not this note", reviewer="r"
        )
        assert done.status is RunStatus.SUCCESS and done.answer == "done"
        assert len(model.inputs) == 3  # continued from the SDK's state
        first, served = results(model)
        await memory_harness.writes.drain()
    if decision == "approve":
        assert first == "note at a" and served.count("read_note") == 1
        assert recorded(memory_service) == [("read_note", "ok"), ("served_calls", "ok")]
    else:
        assert first == "not this note" and "read_note" not in served
        assert recorded(memory_service) == [("served_calls", "ok")]


async def test_a_write_the_servers_policy_asked_about_is_not_asked_about_twice(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    """A tool both the server's policy and governance ask about (a write): the person is asked
    once — the SDK's approval is the answer governance takes for that call."""
    async with stdio(require_approval="always") as server:
        model = ScriptedModel([("add_note", {"text": "milk"}), "added"])
        agent = memory_harness.wrap(
            Agent(name="desk", model=model, mcp_servers=[server]), id="once", without={"schedules"}
        )
        paused = await agent.run("note milk", user="u")
        assert paused.interrupt is not None
        done = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="r")
        assert done.status is RunStatus.SUCCESS and done.answer == "added"
        assert results(model) == ["added milk"]
        await memory_harness.writes.drain()
    assert recorded(memory_service) == [("add_note", "ok")]


async def test_a_servers_error_is_what_the_model_reads_and_structured_content_is_used(
    memory_harness: Harness, memory_service: FakeMemoryService
) -> None:
    """A tool that fails on the server is an error the model reads, recorded as one, and the
    run goes on; with ``use_structured_content`` the model reads the result's structured
    content, as the SDK gives it."""
    memory_service.catalog = {"fail": {"side_effects": "read"}}
    async with stdio(use_structured_content=True) as server:
        model = PlannedModel([("fail", {"why": "no"}), ("served_calls", {})], final="done")
        agent = memory_harness.wrap(
            Agent(name="desk", model=model, mcp_servers=[server]),
            id="errors",
            without={"schedules"},
        )
        result = await agent.run("try", user="u")
        assert result.status is RunStatus.SUCCESS and result.answer == "done"
        failed, served = results(model)
        assert "fail failed" in failed and served == '{"result": []}'
        await memory_harness.writes.drain()
    assert recorded(memory_service) == [("fail", "error"), ("served_calls", "ok")]


async def test_a_server_over_http_with_its_headers_is_governed_the_same(
    memory_harness: Harness, memory_service: FakeMemoryService, http_url: str
) -> None:
    params: Any = {"url": http_url, "headers": {"Authorization": "Bearer t0k"}}
    async with MCPServerStreamableHttp(
        params=params, name="notes-http", client_session_timeout_seconds=30
    ) as server:
        plan = [("whoami", {}), ("delete_note", {"path": "a"})]
        model = PlannedModel(plan, final="done")
        agent = memory_harness.wrap(
            Agent(name="desk", model=model, mcp_servers=[server]), id="http", without={"schedules"}
        )
        paused = await agent.run("who am I, then delete a", user="u")
        assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
        assert paused.interrupt.tool_call is not None
        assert paused.interrupt.tool_call.tool == "delete_note"
        done = await agent.resume(paused.interrupt.interrupt_id, "reject", reviewer="r")
        assert done.status is RunStatus.SUCCESS
        assert results(model)[0] == "t0k"  # the server's own headers reached it
        await memory_harness.writes.drain()
    assert ("whoami", "ok") in recorded(memory_service)
    assert ("delete_note", "ok") not in recorded(memory_service)


async def test_the_governed_copy_of_a_server_reads_its_resources_and_cached_tools() -> None:
    """What the agent's code reads of its server inside a run (the copy the harness governs
    its calls on) is the server's own: its cached tools, its resources, its templates."""
    async with stdio(cache_tools_list=True) as server:
        governed = GovernedServer.of(server)
        assert governed.cached_tools is None  # nothing listed yet
        listed = await governed.list_tools()
        assert [t.name for t in governed.cached_tools or ()] == [t.name for t in listed]
        assert (await governed.list_resources()).resources == []
        assert (await governed.list_resource_templates()).resource_templates == []
        with pytest.raises(Exception, match=r"(?i)unknown resource|not found"):
            await governed.read_resource("note://missing")


class Unreachable:
    """An MCP server of the agent's own whose calls cannot be made (its transport is gone)."""

    def __init__(self) -> None:
        self.called: list[str] = []

    async def connect(self) -> None:
        return None

    async def cleanup(self) -> None:
        return None

    async def list_tools(self, run_context: Any = None, agent: Any = None) -> list[Any]:
        from mcp.types import Tool, ToolAnnotations

        return [
            Tool(
                name="find",
                description="Find an order.",
                input_schema={"type": "object", "properties": {"order": {"type": "string"}}},
                annotations=ToolAnnotations(read_only_hint=True, idempotent_hint=False),
            )
        ]

    async def call_tool(self, tool_name: str, arguments: Any, meta: Any = None) -> Any:
        self.called.append(tool_name)
        raise ConnectionResetError("the server hung up")

    async def list_prompts(self) -> Any:
        return []

    async def get_prompt(self, name: str, arguments: Any = None) -> Any:
        return None


def unreachable(**settings: Any) -> Any:
    from agents.mcp import MCPServer

    class Own(Unreachable, MCPServer):  # type: ignore[misc]
        @property
        def name(self) -> str:
            return "crm"

    found = Own.__new__(Own)
    MCPServer.__init__(found, **settings)
    Unreachable.__init__(found)
    return found


@pytest.mark.parametrize("policy", ["default", "custom", "none"])
async def test_a_call_the_server_could_not_take_is_told_as_its_failure_function_says(
    memory_harness: Harness, memory_service: FakeMemoryService, policy: str
) -> None:
    """A call that raises (the transport, not the tool): recorded as the call's failure, then
    the SDK's to tell — its default text (from openai-agents 0.23 without the error's detail),
    the server's own ``failure_error_function``'s, or with ``failure_error_function=None`` the
    run ends with the error."""
    settings: dict[str, Any] = {
        "default": {},
        "custom": {"failure_error_function": lambda _context, error: "the CRM is down"},
        "none": {"failure_error_function": None},
    }[policy]
    server = unreachable(**settings)
    model = ScriptedModel([("find", {"order": "o1"}), "it failed"])
    agent = memory_harness.wrap(
        Agent(name="desk", model=model, mcp_servers=[server]),
        id=f"unreachable-{policy}",
        without={"schedules"},
    )
    result = await agent.run("find o1", user="u")
    await memory_harness.writes.drain()
    assert server.called and recorded(memory_service)[-1] == ("find", "error")
    if policy == "none":
        assert result.status is RunStatus.ERROR and result.error is not None
        assert "the server hung up" in result.error.message and len(model.inputs) == 1
        return
    assert result.status is RunStatus.SUCCESS and result.answer == "it failed"
    [told] = results(model)
    assert told == (
        "the CRM is down"
        if policy == "custom"
        else "An error occurred while running the tool. Please try again."
    )


class Withheld(Hooks):
    """The run's ``after_tool`` hook: a failed call's words withheld (``ok``: made a result)."""

    def __init__(self, *, ok: bool = False) -> None:
        self.ok = ok

    async def after_tool(self, call: Any, outcome: ToolOutcome) -> ToolOutcome:
        if outcome.ok:
            return outcome
        if self.ok:
            return outcome.model_copy(update={"output": "o1: open", "status": ToolStatus.OK})
        return outcome.model_copy(update={"output": "find failed (details withheld)"})


@pytest.mark.parametrize("policy", ["default", "custom", "none"])
async def test_a_failed_call_a_hook_rewords_is_told_in_its_words_as_the_sdk_tells_it(
    harness: Harness, policy: str
) -> None:
    """A call the server could not take, its failure reworded by the run's ``after_tool``
    hook: the SDK's policy still decides — the run goes on, or with
    ``failure_error_function=None`` ends with the error — and the model reads the hook's
    words, not the SDK's text or the server's own function's."""
    settings: dict[str, Any] = {
        "default": {},
        "custom": {"failure_error_function": lambda _context, error: "the CRM is down"},
        "none": {"failure_error_function": None},
    }[policy]
    model = ScriptedModel([("find", {"order": "o1"}), "it failed"])
    agent = harness.wrap(
        Agent(name="desk", model=model, mcp_servers=[unreachable(**settings)]),
        id=f"withheld-{policy}",
        without={"schedules"},
        hooks=[Withheld()],
    )
    result = await agent.run("find o1", user="u")
    if policy == "none":
        assert result.status is RunStatus.ERROR and len(model.inputs) == 1
        return
    assert result.answer == "it failed" and results(model) == ["find failed (details withheld)"]


async def test_a_failed_call_a_hook_makes_a_result_is_read_as_one(harness: Harness) -> None:
    """A hook that turns the call the server could not take into a result: the model reads
    that result, even where the failure would end the run (``failure_error_function=None``)."""
    model = ScriptedModel([("find", {"order": "o1"}), "it is open"])
    agent = harness.wrap(
        Agent(
            name="desk",
            model=model,
            mcp_servers=[unreachable(failure_error_function=None)],
        ),
        id="withheld-ok",
        without={"schedules"},
        hooks=[Withheld(ok=True)],
    )
    result = await agent.run("find o1", user="u")
    assert result.status is RunStatus.SUCCESS and result.answer == "it is open"
    assert results(model) == ["o1: open"]
