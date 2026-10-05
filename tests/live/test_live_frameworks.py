"""Every target against the real services: memory pushed into the run and pulled by the
agent, the MCP tool the agent's virtual key allows (nothing else) and a local function, the
transcript, the tool records and the outcome written back — with a real model through the
gateway (the Claude target drives the scripted Claude Code CLI; everything else about it is
real)."""

from __future__ import annotations

import json
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
from agents import Agent as OpenAIAgent
from agents import OpenAIChatCompletionsModel
from claude_agent_sdk import ClaudeAgentOptions
from deepagents import create_deep_agent
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from openai import AsyncOpenAI

from tests.live.conftest import (
    BIFROST_URL,
    MODEL,
    WIKI_TOOL,
    live_harness,
    needs_gateway,
    needs_memory,
)
from tests.live.support import eventually, memory_scope
from trellis import Harness, ReAct, Runtime, tool
from trellis.contracts import RunEvent, RunEventType, RunOutcome

pytestmark = [pytest.mark.live, needs_gateway, needs_memory]

CLI = str(Path(__file__).resolve().parents[1] / "support" / "fake_claude_cli.py")
QUESTION = (
    "First call memory_search with the query 'warehouse'. Then use the stock tool for SKU A-1, "
    f"and the {WIKI_TOOL} tool for the repository facebook/react. Answer with the number of "
    "units in stock and the first documentation topic."
)
SYSTEM = "You are a stock assistant. Always use the tools you are asked to use."


@tool(side_effects="read")
def stock(sku: str) -> int:
    """Units of a SKU in stock. Known SKUs: A-1, B-2."""
    return {"A-1": 42, "B-2": 0}.get(sku.upper(), 0)


async def chat_model(h: Harness) -> ChatOpenAI:
    """A model through Bifrost with the harness's virtual key and ``h.model_headers()`` (the
    gateway adds none of the key's MCP tools), with its own HTTP client: langchain-openai
    otherwise shares one per process, bound to the event loop of the test that made it."""
    return ChatOpenAI(
        base_url=BIFROST_URL,
        api_key=h.settings.bifrost_virtual_key,  # type: ignore[arg-type]
        model=MODEL,
        max_tokens=2048,  # type: ignore[call-arg]
        default_headers=await h.model_headers(),
        http_async_client=httpx.AsyncClient(timeout=120),
    )


Build = Callable[[Harness, str, Path], Awaitable[tuple[Any, list[Any]]]]


async def langgraph(h: Harness, wiki: str, tmp: Path) -> tuple[Any, list[Any]]:
    tools = await h.tools(stock, framework="langgraph")  # + the key's MCP tool + memory tools
    return create_agent(await chat_model(h), tools=tools, system_prompt=SYSTEM), []


async def deep_agent(h: Harness, wiki: str, tmp: Path) -> tuple[Any, list[Any]]:
    tools = await h.tools(stock, framework="deepagents")
    return create_deep_agent(model=await chat_model(h), tools=tools, system_prompt=SYSTEM), []


async def openai_agents(h: Harness, wiki: str, tmp: Path) -> tuple[Any, list[Any]]:
    model = OpenAIChatCompletionsModel(
        model=MODEL,
        openai_client=AsyncOpenAI(
            base_url=BIFROST_URL,
            api_key=h.settings.bifrost_virtual_key,
            default_headers=await h.model_headers(),
        ),
    )
    return OpenAIAgent(name="stock", instructions=SYSTEM, model=model), [stock]


async def react(h: Harness, wiki: str, tmp: Path) -> tuple[Any, list[Any]]:
    return ReAct(system=SYSTEM, model=MODEL), [stock]


async def function(h: Harness, wiki: str, tmp: Path) -> tuple[Any, list[Any]]:
    async def answer(input: str, agent: Runtime) -> str:
        assert agent.context is not None and "Berlin" in agent.context  # pushed
        await agent.tools.call("memory_search", query="warehouse")  # pulled
        units = await agent.tools.call("stock", sku="A-1")
        wiki_out = await agent.tools.call(f"{wiki}-{WIKI_TOOL}", repoName="facebook/react")
        return f"{units} units; {str(wiki_out)[:60]}"

    return answer, [stock]


async def claude(h: Harness, wiki: str, tmp: Path) -> tuple[Any, list[Any]]:
    script = [
        {"tool": "memory_search", "args": {"query": "warehouse"}},
        {"tool": "stock", "args": {"sku": "A-1"}},
        {"tool": f"{wiki}-{WIKI_TOOL}", "args": {"repoName": "facebook/react"}},
        {"text": "42 units in stock; the first topic is the overview."},
    ]
    options = ClaudeAgentOptions(
        cli_path=CLI,
        system_prompt=SYSTEM,
        env={"FAKE_CLAUDE_SCRIPT": json.dumps(script), "FAKE_CLAUDE_RECORD": str(tmp / "cli.json")},
    )
    return options, [stock]


TARGETS: dict[str, Build] = {
    "langgraph": langgraph,
    "deepagents": deep_agent,
    "openai_agents": openai_agents,
    "react": react,
    "function": function,
    "claude_agent_sdk": claude,
}


@pytest.mark.parametrize("framework", list(TARGETS))
async def test_a_target_runs_with_memory_and_tools(
    framework: str, deepwiki: str, wiki_key: str, tmp_path: Path
) -> None:
    suffix = uuid.uuid4().hex[:8]
    user, thread = f"live-user-{suffix}", f"live-thread-{suffix}"
    async with live_harness(wiki_key) as h:
        target, tools = await TARGETS[framework](h, deepwiki, tmp_path)
        agent = h.wrap(target, id=f"live-{framework}", tools=tools)
        scope = await memory_scope(h, user=user, agent_id=agent.id, thread=thread)
        await scope.remember(f"The warehouse of {user} is in Berlin.", visibility="USER")

        events: list[RunEvent] = [e async for e in agent.stream(QUESTION, user=user, thread=thread)]
        await h.writes.drain()

        finished = events[-1]
        assert finished.type is RunEventType.RUN_FINISHED, finished
        assert finished.outcome is RunOutcome.SUCCESS, finished.error
        assert "42" in json.dumps(finished.data.get("result"))
        loaded = [e for e in events if e.type is RunEventType.CONTEXT_LOADED]
        assert loaded and loaded[0].data["chars"] > 0  # push
        called = {e.data.get("tool") for e in events if e.type is RunEventType.TOOL_CALL_START}
        assert {"memory_search", "stock"} <= called, called  # pull, and the local tool
        if framework in ("function", "claude_agent_sdk", "react"):
            assert f"{deepwiki}-{WIKI_TOOL}" in called  # the MCP tool, through Bifrost
        # the key allows one tool of one wiki: the toolbox holds exactly that MCP tool
        toolbox = await h.resolve(agent.sources, tenant=await h.tenant())
        assert [t.name for t in toolbox if t.spec.source == "mcp"] == [f"{deepwiki}-{WIKI_TOOL}"]
        if framework == "claude_agent_sdk":
            started = json.loads((tmp_path / "cli.json").read_text())
            assert "Berlin" in started["system_prompt"]
        assert h.writes.failed == 0  # every background write landed

        # the transcript, once
        history = await scope.history()
        assert [m.role for m in history] == ["USER", "ASSISTANT"]
        assert history[0].content == QUESTION

        # the tool records reach the catalog's statistics
        async def stock_counted() -> bool:
            [entry] = await scope.advanced.tools.catalog(names=["stock"])
            return entry.stats.calls >= 1

        assert await eventually(stock_counted)
