"""Every adapter, built to follow one plan of tool calls: the table the behaviour tests run over
(``BUILDERS``) — the function target, ``ReAct``, LangChain's ``create_agent``, Deep Agents, the
OpenAI Agents SDK and the Claude Agent SDK (the real SDK driving the scripted CLI). Each model
makes the plan's calls in order and answers ``FINAL`` with the last result it read
(``tests.support.planned``), so a test sees what reached the model. ``system`` is the
instructions the framework's agent is built with (a function has none), and ``seen`` gets the
model built — for the Claude Agent SDK, the file its CLI records what it was started with in."""

from __future__ import annotations

import json
import uuid
from collections.abc import Awaitable
from pathlib import Path
from typing import Any, Final, Protocol

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
SYSTEM: Final = "You work."


class Builder(Protocol):
    def __call__(
        self,
        h: Harness,
        tools: list[Any],
        tmp: Path,
        plan: list[Call],
        *,
        system: str = ...,
        seen: list[Any] | None = ...,
    ) -> Awaitable[Built]: ...


async def _function(
    h: Harness,
    tools: list[Any],
    tmp: Path,
    plan: list[Call],
    *,
    system: str = SYSTEM,
    seen: list[Any] | None = None,
) -> Built:
    async def follow(input: Any, agent: Runtime) -> str:
        results = [text_of(await agent.tools.call(name, **args)) for name, args in plan]
        return FINAL.replace("{last}", results[-1] if results else "")

    return follow, tools


async def _react(
    h: Harness,
    tools: list[Any],
    tmp: Path,
    plan: list[Call],
    *,
    system: str = SYSTEM,
    seen: list[Any] | None = None,
) -> Built:
    model = PlannedChat(plan)
    _saw(seen, model)
    return ReAct(system=system, model=model), tools


async def _langgraph(
    h: Harness,
    tools: list[Any],
    tmp: Path,
    plan: list[Call],
    *,
    system: str = SYSTEM,
    seen: list[Any] | None = None,
) -> Built:
    model = PlannedChatModel(plan=plan)
    _saw(seen, model)
    built = await h.tools(*tools, framework="langgraph")
    return create_agent(model, tools=built, system_prompt=system), []


async def _deepagents(
    h: Harness,
    tools: list[Any],
    tmp: Path,
    plan: list[Call],
    *,
    system: str = SYSTEM,
    seen: list[Any] | None = None,
) -> Built:
    model = PlannedChatModel(plan=plan)
    _saw(seen, model)
    built = await h.tools(*tools, framework="deepagents")
    return create_deep_agent(model=model, tools=built, system_prompt=system), []


async def _openai_agents(
    h: Harness,
    tools: list[Any],
    tmp: Path,
    plan: list[Call],
    *,
    system: str = SYSTEM,
    seen: list[Any] | None = None,
) -> Built:
    model = PlannedModel(plan)
    _saw(seen, model)
    return OpenAIAgent(name="worker", instructions=system, model=model), tools


async def _claude(
    h: Harness,
    tools: list[Any],
    tmp: Path,
    plan: list[Call],
    *,
    system: str = SYSTEM,
    seen: list[Any] | None = None,
) -> Built:
    script = [{"tool": name, "args": args} for name, args in plan] + [{"text": FINAL}]
    record = tmp / f"cli-{uuid.uuid4().hex[:6]}.json"
    _saw(seen, record)
    options = ClaudeAgentOptions(
        cli_path=CLI,
        system_prompt=system,
        env={"FAKE_CLAUDE_SCRIPT": json.dumps(script), "FAKE_CLAUDE_RECORD": str(record)},
    )
    return options, tools


def _saw(seen: list[Any] | None, model: Any) -> None:
    if seen is not None:
        seen.append(model)


BUILDERS: Final[dict[str, Builder]] = {
    "function": _function,
    "react": _react,
    "langgraph": _langgraph,
    "deepagents": _deepagents,
    "openai_agents": _openai_agents,
    "claude_agent_sdk": _claude,
}
