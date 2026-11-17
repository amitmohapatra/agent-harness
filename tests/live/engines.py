"""Every adapter on the live model, built the way its team builds it: the table the
model-driven workflow tests run over (:data:`ENGINES`). The model is the suite's
(``TRELLIS_LIVE_MODEL``), reached as a deployment reaches it — through the gateway, or directly
at ``LLM_BASE_URL`` — by each framework's own means:

* a **function** runs a loop of its own: the routed chat model, LangChain tools made of the
  harness's (``native_tools``), each call through the bridge;
* ``ReAct`` takes the model's name, and its tools, skills, sandbox and plan mode in its own
  arguments;
* LangChain's ``create_agent`` — plain (**langgraph**: its tools bound when it is built) and
  with ``trellis.middleware()`` (**middleware**: the run's tools per model call) — and **Deep
  Agents** take the routed chat model (``gateways.routed_chat_model``), each compiled with the
  run's checkpointer (``RunCheckpointer``), so a pause resumes in place: without one a resume
  runs the graph again from its input and asks the model again (docs/frameworks/langgraph.md,
  "Approvals"), and a model asked again may ask its question again instead of acting on the
  answer it was given (seen live: Haiku, ``create_agent``, ``ask_user``);
* the **OpenAI Agents SDK** takes a chat completions model pointed at the route's endpoint;
* the **Claude Agent SDK** runs the real Claude Code CLI (``claude_cli_env``) and **Kimi Code**
  the real ``kimi acp``, each in a folder of the test's own.

Skills go where each engine reads them (``docs/skills.md``): its own slot, else the harness's
two tools. What a test asserts is what a model cannot fake: its tools' own ledger, the calls
the harness ran (:class:`Seen`), and figures computed from seeded data (:func:`says`)."""

from __future__ import annotations

import re
import uuid
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import pytest
from agents import Agent as OpenAIAgent
from agents import OpenAIChatCompletionsModel
from claude_agent_sdk import ClaudeAgentOptions
from deepagents import create_deep_agent
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from openai import AsyncOpenAI

from tests.live.conftest import (
    DIRECT_URL,
    GATEWAY_URL,
    MODEL,
    claude_cli_env,
    needs_claude_cli,
    needs_claude_model,
    needs_model,
    settings,
)
from tests.live.proof import RUN_SECONDS
from tests.support.arm import arm, native_tools
from tests.support.kimi import needs_kimi
from trellis import Agent, Harness, Hooks, ReAct, Result, Runtime, deep_agent, kimi, middleware
from trellis.contracts import ToolCall, ToolOutcome
from trellis.harness.adapters.claude import PLAN_TOOL, SKILLS_SLOT
from trellis.harness.gateways import routed_chat_model
from trellis.harness.history import Speaker
from trellis.harness.middleware import RunCheckpointer
from trellis.harness.plan_mode import PlanMode
from trellis.harness.prompts import GatewayHeaders
from trellis.harness.skills import LOAD_SKILL, SkillSet

#: The most model turns an engine takes before it stops a model that keeps calling tools.
MAX_TURNS: Final = 14
#: What every engine's agent is told (Kimi Code keeps its own instructions: a scenario's
#: question says all it needs).
SYSTEM: Final = (
    "You are an operations assistant. Use your tools for every fact and every change; never "
    "invent a value. Be brief."
)
#: The calls that read a skill's instructions: the harness's tool, Deep Agents' ``read_file``
#: of its skills route (``ReAct`` too), Claude Code's and Kimi Code's ``Skill``.
SKILL_LOADS: Final = frozenset({LOAD_SKILL, "read_file", SKILLS_SLOT})


@dataclass
class Kit:
    """What a scenario's agent is made of: the tools the harness runs for it (functions,
    wrapped agents, ``ask_user()``), its skills, a sandbox, and whether it can plan first
    (``ReAct``'s ``PlanMode``)."""

    tools: list[Any]
    skills: SkillSet | None = None
    box: Any = None
    plan: bool = False
    system: str = SYSTEM

    def sources(self, *, slot: bool, shell: bool = False) -> list[Any]:
        """The tool sources of an engine: the kit's, the two skill tools where the engine has
        no skills ``slot``, and the sandbox's where it has no ``shell`` of its own."""
        found = list(self.tools)
        if self.skills is not None and not slot:
            found.append(self.skills.tools())
        if self.box is not None and not shell:
            found.append(self.box.tools())
        return found


@dataclass
class Built:
    """A target, the sources it is armed with, and its framework's options."""

    target: Any
    sources: list[Any]
    options: dict[str, Any] = field(default_factory=dict)


Build = Callable[[Harness, Kit, Path], Awaitable[Built]]


async def _function(h: Harness, kit: Kit, tmp: Path) -> Built:
    sources = kit.sources(slot=False)
    native = {t.name: t for t in await native_tools(h, *sources, framework="langgraph")}

    async def loop(input: str, agent: Runtime) -> str:
        """The team's own loop: the model asked, its calls run, until it answers."""
        model = routed_chat_model(MODEL).bind_tools(list(native.values()))
        earlier = [
            (HumanMessage if said.role is Speaker.USER else AIMessage)(said.content)
            for said in agent.history or ()
        ]
        messages: list[Any] = [
            SystemMessage(f"{kit.system}\n\n{agent.context or ''}"),
            *earlier,
            HumanMessage(input),
        ]
        for _ in range(MAX_TURNS):
            reply = await model.ainvoke(messages)
            messages.append(reply)
            if not reply.tool_calls:
                return str(reply.text)
            for call in reply.tool_calls:
                tool = native.get(call["name"])
                messages.append(
                    await tool.ainvoke(call)
                    if tool is not None
                    else ToolMessage(f"no tool {call['name']}", tool_call_id=call["id"])
                )
        return ""

    return Built(loop, sources)


async def _react(h: Harness, kit: Kit, tmp: Path) -> Built:
    target = ReAct(
        system_prompt=kit.system,
        model=MODEL,
        skills=kit.skills,
        backend=kit.box,
        max_steps=MAX_TURNS,
        middleware=[PlanMode()] if kit.plan else [],
    )
    plan = {"configurable": {"plan_mode": True}} if kit.plan else {}
    return Built(target, kit.sources(slot=True, shell=True), plan)


async def _langgraph(h: Harness, kit: Kit, tmp: Path) -> Built:
    sources = kit.sources(slot=False)
    tools = await native_tools(h, *sources, framework="langgraph")
    graph = create_agent(
        routed_chat_model(MODEL),
        tools=tools,
        system_prompt=kit.system,
        checkpointer=RunCheckpointer(),
    )
    return Built(graph, sources, {"recursion_limit": 2 * MAX_TURNS + 1})


async def _middleware(h: Harness, kit: Kit, tmp: Path) -> Built:
    graph = create_agent(
        routed_chat_model(MODEL),
        system_prompt=kit.system,
        middleware=middleware(max_steps=MAX_TURNS),
        checkpointer=RunCheckpointer(),
    )
    return Built(graph, kit.sources(slot=False))


async def _deepagents(h: Harness, kit: Kit, tmp: Path) -> Built:
    sources = kit.sources(slot=True, shell=True)
    graph = create_deep_agent(
        model=routed_chat_model(MODEL),
        tools=await native_tools(h, *sources, framework="deepagents"),
        system_prompt=kit.system,
        checkpointer=RunCheckpointer(),
        **(deep_agent(kit.box, kit.skills) if kit.box or kit.skills else {}),
    )
    return Built(graph, sources, {"recursion_limit": 2 * MAX_TURNS + 1})


async def _openai_agents(h: Harness, kit: Kit, tmp: Path) -> Built:
    client = AsyncOpenAI(
        base_url=h.settings.llm_base_url,
        api_key=h.settings.llm_api_key or "none",
        default_headers=GatewayHeaders() if GATEWAY_URL is not None else None,
        timeout=RUN_SECONDS,
    )
    model = OpenAIChatCompletionsModel(model=MODEL, openai_client=client)
    agent = OpenAIAgent(name="operations", instructions=kit.system, model=model)
    return Built(agent, kit.sources(slot=False), {"max_turns": MAX_TURNS})


async def _claude(h: Harness, kit: Kit, tmp: Path) -> Built:
    """Claude Code itself, given none of its built-in tools but — with skills — ``Skill``,
    its skills slot."""
    home = tmp / f"claude-{uuid.uuid4().hex[:6]}"
    home.mkdir()
    slot = [SKILLS_SLOT] if kit.skills is not None else []
    # plan mode: Claude Code's own, held by the harness (its plan written, then presented)
    planning = {"permission_mode": "plan"} if kit.plan else {}
    options = ClaudeAgentOptions(
        model=MODEL,
        system_prompt=kit.system,
        tools=[*slot, *(("Write", PLAN_TOOL) if kit.plan else ())],
        max_turns=MAX_TURNS,
        cwd=home,
        env=claude_cli_env(),
        **planning,
        **(kit.skills.claude() if kit.skills is not None else {}),
    )
    return Built(options, kit.sources(slot=True))


async def _kimi(h: Harness, kit: Kit, tmp: Path) -> Built:
    """Kimi Code itself (``kimi acp``), its model the gateway's or — with no gateway — the
    endpoint its own configuration names (``test_live_kimi``)."""
    home, work = tmp / f"kimi-{uuid.uuid4().hex[:6]}", tmp / f"work-{uuid.uuid4().hex[:6]}"
    home.mkdir()
    work.mkdir()
    env = {"KIMI_CODE_HOME": str(home), "KIMI_DISABLE_TELEMETRY": "1"}
    if DIRECT_URL is not None:
        env |= {
            "KIMI_MODEL_PROVIDER_TYPE": "openai",
            "KIMI_MODEL_BASE_URL": DIRECT_URL,
            "KIMI_MODEL_API_KEY": settings().llm_api_key or "none",
            "KIMI_MODEL_NAME": MODEL,
        }
    skills = kit.skills.kimi() if kit.skills is not None else {}
    # plan mode: Kimi Code's own (its plan presented for review, a question with its options)
    planning = {"mode": "plan"} if kit.plan else {}
    return Built(
        kimi(model=MODEL, cwd=str(work), env=env, **planning, **skills), kit.sources(slot=True)
    )


BUILDS: Final[dict[str, Build]] = {
    "function": _function,
    "react": _react,
    "langgraph": _langgraph,
    "middleware": _middleware,
    "deepagents": _deepagents,
    "openai_agents": _openai_agents,
    "claude_agent_sdk": _claude,
    "kimi": _kimi,
}
#: What each engine's tests need beside the harness: a model, and for the two CLIs the CLI.
NEEDS: Final[dict[str, list[pytest.MarkDecorator]]] = {
    **{name: [needs_model] for name in BUILDS},
    "claude_agent_sdk": [needs_claude_model, needs_claude_cli],
    "kimi": [needs_model, needs_kimi],
}
#: The engines whose framework has a skills slot of its own (``docs/skills.md``).
SLOTS: Final = ("react", "deepagents", "claude_agent_sdk", "kimi")


def engines(*, only: Sequence[str] | None = None, without: Sequence[str] = ()) -> list[Any]:
    """The engines a test runs on, as ``parametrize`` takes them: all, ``only`` some, or all
    ``without`` some — each with what it needs (:data:`NEEDS`)."""
    names = [name for name in (only or BUILDS) if name not in without]
    return [pytest.param(name, marks=NEEDS[name], id=name) for name in names]


ENGINES: Final = engines()


class Seen(Hooks):
    """The calls of an agent's runs as the harness ran them: each as it was asked for
    (``asked``: a framework's own tools too, a skill's load among them) and how it ended."""

    def __init__(self) -> None:
        self.asked: list[ToolCall] = []
        self.ended: list[tuple[ToolCall, ToolOutcome]] = []

    async def before_tool(self, call: ToolCall) -> None:
        self.asked.append(call)

    async def after_tool(self, call: ToolCall, outcome: ToolOutcome) -> ToolOutcome:
        self.ended.append((call, outcome))
        return outcome

    def called(self, tool: str) -> list[dict[str, Any]]:
        """The arguments of each call of ``tool`` asked for, in order."""
        return [dict(call.args) for call in self.asked if call.tool == tool]

    def loaded(self, skill: str) -> bool:
        """Whether the skill's instructions were read, through whichever slot the engine has."""
        return any(
            call.tool in SKILL_LOADS and skill in str(dict(call.args)) for call in self.asked
        )


@dataclass
class Made:
    """A scenario's agent on one engine, and the calls of its runs."""

    agent: Agent
    seen: Seen
    options: dict[str, Any]

    async def run(
        self, question: str, *, user: str, thread: str | None = None, mode: str = "queue"
    ) -> Result:
        """One run to its end or its first question, a skill's load approved on the way
        (:meth:`settled`); ``mode``: what it does to a run working in its conversation."""
        asked = await self.agent.run(
            question, user=user, thread=thread, mode=mode, framework_options=self.options or None
        )
        return await self.settled(asked, user)

    async def resume(self, paused: Result, decision: str, *, user: str, **given: Any) -> Result:
        """The paused run answered (``agent.resume``), on to its end or its next question."""
        assert paused.interrupt is not None, paused
        answered = await self.agent.resume(
            paused.interrupt.interrupt_id, decision, reviewer=user, **given
        )
        return await self.settled(answered, user)

    async def settled(self, result: Result, user: str) -> Result:
        """Claude Code asks its user before it loads a skill (``adapters.claude.CLAUDE_ASKS``):
        the run's user allows it, as a person would, until the run asks something else or
        ends."""
        while (
            result.interrupt is not None
            and result.interrupt.tool_call is not None
            and result.interrupt.tool_call.tool == SKILLS_SLOT
        ):
            result = await self.agent.resume(
                result.interrupt.interrupt_id, "approve", reviewer=user
            )
        return result


async def made(
    engine: str, h: Harness, kit: Kit, tmp: Path, name: str, *, without: Sequence[str] = ()
) -> Made:
    """``kit``'s agent on ``engine``, wrapped (its id the scenario's ``name`` and a suffix of
    its own, so the services' state is this test's; ``without`` what the harness does not do
    for its runs), armed, and watched."""
    built = await BUILDS[engine](h, kit, tmp)
    seen = Seen()
    wrapped = h.wrap(
        built.target,
        id=f"live-{name}-{uuid.uuid4().hex[:8]}",
        hooks=[seen],
        run_timeout=RUN_SECONDS,
        without=without,
    )
    return Made(arm(wrapped, built.sources), seen, built.options)


def says(answer: Any, number: int) -> bool:
    """Whether ``answer`` holds the whole number ``number``, however its thousands are
    written (``4150``, ``4,150``, ``4 150``, ``4150.00``)."""
    plain = re.sub(r"(?<=\d)[, ](?=\d{3}(?!\d))", "", str(answer))
    return re.search(rf"(?<![\d.]){number}(?:\.0+)?(?!\d)", plain) is not None
