"""Every adapter, built to follow one plan of tool calls: the table the behaviour tests run over
(``BUILDERS``) — the function target, ``ReAct``, LangChain's ``create_agent``, Deep Agents, the
OpenAI Agents SDK and the Claude Agent SDK (the real SDK driving the scripted CLI). Each model
makes the plan's calls in order and answers ``FINAL`` with the last result it read
(``tests.support.planned``), so a test sees what reached the model."""

from __future__ import annotations

import json
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Final

from agents import Agent as OpenAIAgent
from claude_agent_sdk import ClaudeAgentOptions
from deepagents import create_deep_agent
from langchain.agents import create_agent

from tests.support.planned import FINAL, Call, PlannedChat, PlannedChatModel, PlannedModel
from trellis import Harness, ReAct, Runtime
from trellis.harness.tools.convert import text_of

CLI: Final = str(Path(__file__).resolve().parent / "fake_claude_cli.py")

#: The target and the tools to wrap it with (none for a graph: they are built into it).
Built = tuple[Any, list[Any]]
Builder = Callable[[Harness, list[Any], Path, list[Call]], Awaitable[Built]]


async def _function(h: Harness, tools: list[Any], tmp: Path, plan: list[Call]) -> Built:
    async def follow(input: Any, agent: Runtime) -> str:
        results = [text_of(await agent.tools.call(name, **args)) for name, args in plan]
        return FINAL.replace("{last}", results[-1] if results else "")

    return follow, tools


async def _react(h: Harness, tools: list[Any], tmp: Path, plan: list[Call]) -> Built:
    return ReAct(system="You work.", model=PlannedChat(plan)), tools


async def _langgraph(h: Harness, tools: list[Any], tmp: Path, plan: list[Call]) -> Built:
    graph = create_agent(
        PlannedChatModel(plan=plan), tools=await h.tools(*tools, framework="langgraph")
    )
    return graph, []


async def _deepagents(h: Harness, tools: list[Any], tmp: Path, plan: list[Call]) -> Built:
    graph = create_deep_agent(
        model=PlannedChatModel(plan=plan), tools=await h.tools(*tools, framework="deepagents")
    )
    return graph, []


async def _openai_agents(h: Harness, tools: list[Any], tmp: Path, plan: list[Call]) -> Built:
    return OpenAIAgent(name="worker", instructions="You work.", model=PlannedModel(plan)), tools


async def _claude(h: Harness, tools: list[Any], tmp: Path, plan: list[Call]) -> Built:
    script = [{"tool": name, "args": args} for name, args in plan] + [{"text": FINAL}]
    options = ClaudeAgentOptions(
        cli_path=CLI,
        env={
            "FAKE_CLAUDE_SCRIPT": json.dumps(script),
            "FAKE_CLAUDE_RECORD": str(tmp / f"cli-{uuid.uuid4().hex[:6]}.json"),
        },
    )
    return options, tools


BUILDERS: Final[dict[str, Builder]] = {
    "function": _function,
    "react": _react,
    "langgraph": _langgraph,
    "deepagents": _deepagents,
    "openai_agents": _openai_agents,
    "claude_agent_sdk": _claude,
}
