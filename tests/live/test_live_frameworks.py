"""Every target against the real services, checked for what the harness does with it: the
memory it pushed into the run, the tools it offered the model (the memory tools, the MCP tool
the agent's virtual key allows and nothing else, a local function), the calls it ran — each
on the stream, journaled, decided by governance, recorded in the memory service's catalog —
the transcript, and a run that ended in a final status with its events. The model is real
(through the gateway) and its answer is not judged (``tests.live.proof``); where it made
none of the calls, its scripted twin makes them against the same services. The Claude target
drives the scripted Claude Code CLI and the function target calls the tools itself: everything
else about them is real."""

from __future__ import annotations

import json
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from agents import Agent as OpenAIAgent
from agents import ModelSettings, OpenAIChatCompletionsModel
from claude_agent_sdk import ClaudeAgentOptions
from deepagents import create_deep_agent
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from openai import AsyncOpenAI, DefaultAsyncHttpxClient

from tests.live.conftest import (
    BIFROST_URL,
    MODEL,
    WIKI_TOOL,
    live_harness,
    needs_gateway,
    needs_memory,
)
from tests.live.proof import (
    MAX_TOKENS,
    RUN_SECONDS,
    SETTLE_SECONDS,
    TEST_SECONDS,
    Proof,
    Run,
    Wire,
    ended,
    governed,
    harness_ok,
    recorded,
    streamed,
)
from tests.live.support import eventually, memory_scope
from tests.support.adapters import BUILDERS
from trellis import Agent, Harness, ReAct, Runtime, tool
from trellis.contracts import RunEventType, RunOutcome

pytestmark = [pytest.mark.live, needs_gateway, needs_memory]

CLI = str(Path(__file__).resolve().parents[1] / "support" / "fake_claude_cli.py")
QUESTION = (
    "First call memory_search with the query 'warehouse'. Then use the stock tool for SKU A-1, "
    f"and the {WIKI_TOOL} tool for the repository facebook/react. Answer with the number of "
    "units in stock and the first documentation topic."
)
SYSTEM = "You are a stock assistant. Always use the tools you are asked to use."
#: The most model turns a framework takes before it stops a model that keeps calling tools.
MAX_TURNS = 6


@tool(side_effects="read")
def stock(sku: str) -> int:
    """Units of a SKU in stock. Known SKUs: A-1, B-2."""
    return {"A-1": 42, "B-2": 0}.get(sku.upper(), 0)


def plan(wiki: str) -> list[tuple[str, dict[str, Any]]]:
    """The calls the question asks for: memory pulled, the local tool, the MCP tool."""
    return [
        ("memory_search", {"query": "warehouse"}),
        ("stock", {"sku": "A-1"}),
        (f"{wiki}-{WIKI_TOOL}", {"repoName": "facebook/react"}),
    ]


@dataclass
class Built:
    """A target as its team builds it, the tools it is wrapped with, its framework's options,
    what its model was sent (``wire``: a model this test built), and whether it follows a
    script (it is its own twin)."""

    target: Any
    tools: list[Any] = field(default_factory=list)
    options: dict[str, Any] = field(default_factory=dict)
    wire: Wire | None = None
    scripted: bool = False


async def chat_model(h: Harness, wire: Wire) -> ChatOpenAI:
    """A model through Bifrost with the harness's virtual key and ``h.model_headers()`` (the
    gateway adds none of the key's MCP tools), short replies, no retries (a slow reply is the
    run's time limit's), and its own HTTP client, recording what it sends: langchain-openai
    otherwise shares one per process, bound to the event loop of the test that made it."""
    return ChatOpenAI(
        base_url=BIFROST_URL,
        api_key=h.settings.bifrost_virtual_key,  # type: ignore[arg-type]
        model=MODEL,
        max_tokens=MAX_TOKENS,  # type: ignore[call-arg]
        max_retries=0,
        default_headers=await h.model_headers(),
        http_async_client=wire.client(),
    )


Build = Callable[[Harness, str, Path], Awaitable[Built]]


async def langgraph(h: Harness, wiki: str, tmp: Path) -> Built:
    wire = Wire()
    tools = await h.tools(stock, framework="langgraph")  # + the key's MCP tool + memory tools
    graph = create_agent(await chat_model(h, wire), tools=tools, system_prompt=SYSTEM)
    return Built(graph, options={"recursion_limit": 2 * MAX_TURNS + 1}, wire=wire)


async def deep_agent(h: Harness, wiki: str, tmp: Path) -> Built:
    wire = Wire()
    tools = await h.tools(stock, framework="deepagents")
    graph = create_deep_agent(model=await chat_model(h, wire), tools=tools, system_prompt=SYSTEM)
    return Built(graph, options={"recursion_limit": 2 * MAX_TURNS + 1}, wire=wire)


async def openai_agents(h: Harness, wiki: str, tmp: Path) -> Built:
    wire = Wire()
    model = OpenAIChatCompletionsModel(
        model=MODEL,
        openai_client=AsyncOpenAI(
            base_url=BIFROST_URL,
            api_key=h.settings.bifrost_virtual_key,
            default_headers=await h.model_headers(),
            http_client=wire.client(DefaultAsyncHttpxClient),
            max_retries=0,
        ),
    )
    settings = ModelSettings(max_tokens=MAX_TOKENS)
    agent = OpenAIAgent(name="stock", instructions=SYSTEM, model=model, model_settings=settings)
    return Built(agent, [stock], options={"max_turns": MAX_TURNS}, wire=wire)


async def react(h: Harness, wiki: str, tmp: Path) -> Built:
    return Built(ReAct(system=SYSTEM, model=MODEL, max_steps=MAX_TURNS), [stock])


async def function(h: Harness, wiki: str, tmp: Path) -> Built:
    async def answer(input: str, agent: Runtime) -> str:
        results = [await agent.tools.call(name, **args) for name, args in plan(wiki)]
        return f"{results[1]} units; {str(results[2])[:60]}"

    return Built(answer, [stock], scripted=True)


async def claude(h: Harness, wiki: str, tmp: Path) -> Built:
    script: list[dict[str, Any]] = [{"tool": name, "args": args} for name, args in plan(wiki)]
    script.append({"text": "42 units in stock; the first topic is the overview."})
    options = ClaudeAgentOptions(
        cli_path=CLI,
        system_prompt=SYSTEM,
        env={"FAKE_CLAUDE_SCRIPT": json.dumps(script), "FAKE_CLAUDE_RECORD": str(tmp / "cli.json")},
    )
    return Built(options, [stock], scripted=True)


TARGETS: dict[str, Build] = {
    "langgraph": langgraph,
    "deepagents": deep_agent,
    "openai_agents": openai_agents,
    "react": react,
    "function": function,
    "claude_agent_sdk": claude,
}


async def twin(h: Harness, framework: str, wiki: str, tmp: Path) -> Built:
    """The same framework with a scripted model that makes the planned calls (each against
    the live services), then answers."""
    target, tools = await BUILDERS[framework](h, [stock], tmp, plan(wiki), system=SYSTEM)
    return Built(target, tools, scripted=True)


@dataclass
class Checked:
    run: Run
    proof: Proof
    agent: Agent
    user: str


async def checked(h: Harness, name: str, built: Built, wiki: str, tenant: str) -> Checked:
    """``built`` run once on the question, with memory to push, and what the harness did
    checked: the run ended, as the harness may end it; the context pushed (and, where it is
    seen, sent to the model); the tools offered; every call it ran recorded; the transcript."""
    suffix = uuid.uuid4().hex[:8]
    user, thread = f"live-user-{suffix}", f"live-thread-{suffix}"
    proof = Proof()
    agent = h.wrap(built.target, id=f"live-{name}-{suffix}", tools=built.tools, hooks=[proof])
    scope = await memory_scope(h, user=user, agent_id=agent.id, thread=thread)
    await scope.remember(f"The warehouse of {user} is in Berlin.", visibility="USER")
    decisions = governed(h, tenant)
    stream = agent.stream(
        QUESTION, user=user, thread=thread, timeout=RUN_SECONDS, framework_options=built.options
    )
    run = await streamed(name, stream)
    if built.wire is not None:
        print(f"[live] {run.summary(built.wire)}")
    await h.writes.drain()

    ended(run)
    harness_ok(run)
    loaded = [e for e in run.events if e.type is RunEventType.CONTEXT_LOADED]
    assert loaded and loaded[0].data["chars"] > 0  # push
    assert proof.context is not None and "Berlin" in proof.context
    if built.wire is not None:  # what the model was sent: the context, and the run's tools
        assert built.wire.requests, run.summary(built.wire)
        assert "Berlin" in built.wire.said()
        offered = built.wire.offered()
        assert {tool for tool, _ in plan(wiki)} <= offered, offered
    recorded(run, proof, decisions)
    assert h.writes.failed == 0  # every background write landed

    history = await scope.history()
    assert history and history[0].role == "USER" and history[0].content == QUESTION
    if run.finished.outcome is RunOutcome.SUCCESS and run.finished.data.get("result"):
        assert [m.role for m in history] == ["USER", "ASSISTANT"]  # the transcript, once
    return Checked(run, proof, agent, user)


@pytest.mark.timeout(TEST_SECONDS)
@pytest.mark.parametrize("framework", list(TARGETS))
async def test_a_target_runs_with_memory_and_tools(
    framework: str, deepwiki: str, wiki_key: str, tmp_path: Path
) -> None:
    planned = [name for name, _ in plan(deepwiki)]
    async with live_harness(wiki_key) as h:
        tenant = await h.tenant()
        built = await TARGETS[framework](h, deepwiki, tmp_path)
        proven = await checked(h, framework, built, deepwiki, tenant)
        live = proven.run
        if live.finished.outcome is not RunOutcome.SUCCESS or not set(planned) <= set(
            live.started()
        ):
            # the model did not make every call, or did not get to an answer: its scripted
            # twin makes them, through the same harness and services, and answers
            assert not built.scripted, live.summary()
            built = await twin(h, framework, deepwiki, tmp_path)
            proven = await checked(h, f"{framework}-twin", built, deepwiki, tenant)
        run = proven.run
        assert run.finished.outcome is RunOutcome.SUCCESS, run.summary()
        assert set(planned) <= set(run.started()), run.summary()
        # each ran against its service at least once (a call the model got wrong is an
        # error it read: on the stream and recorded, as any other)
        ok = {r["tool"] for r in run.results() if r["status"] == "ok"}
        assert set(planned) <= ok, run.results()
        # the key allows one tool of one wiki: the toolbox holds exactly that MCP tool
        toolbox = await h.resolve(proven.agent.sources, tenant=tenant)
        assert [t.name for t in toolbox if t.spec.source == "mcp"] == [f"{deepwiki}-{WIKI_TOOL}"]
        if framework == "claude_agent_sdk":
            started = json.loads((tmp_path / "cli.json").read_text())
            assert "Berlin" in started["system_prompt"]
        scope = await memory_scope(h, user=proven.user, agent_id=proven.agent.id)

        # the tool records reach the catalog's statistics
        async def stock_counted() -> bool:
            [entry] = await scope.advanced.tools.catalog(names=["stock"])
            return entry.stats.calls >= 1

        assert await eventually(stock_counted, within=SETTLE_SECONDS)
