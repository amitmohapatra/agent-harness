"""``Harness``: the one object an application constructs. It reads the deployment from the
environment, owns the clients (Bifrost, memory, runs), the background writes and the judge,
and attaches all of it to agents with :meth:`Harness.wrap`."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Final, Literal

from trellis.contracts import (
    ConfigurationError,
    Feedback,
    FeedbackSource,
    FeedbackTargetKind,
    FeedbackVerdict,
    ToolSpec,
)
from trellis.eval.budget import JudgeBudget
from trellis.eval.judge import GroundedJudge
from trellis.harness import telemetry
from trellis.harness.adapters import convert
from trellis.harness.agent import Agent, MemoryMode
from trellis.harness.artifacts import Artifacts
from trellis.harness.clients.bifrost import Gateway
from trellis.harness.clients.memory import Memory, RunMemory
from trellis.harness.clients.runs import HttpRuns, LocalRuns, Runs
from trellis.harness.identity import Identity
from trellis.harness.runtime import current
from trellis.harness.settings import Settings
from trellis.harness.tools.base import Source, Tool
from trellis.harness.tools.policy import Rule
from trellis.harness.tools.sources import as_source
from trellis.harness.worker import WORKER_CONCURRENCY, Worker
from trellis.harness.writes import Writes

Framework = Literal["langgraph", "openai-agents", "claude-agent-sdk"]
#: The native tool format each framework's agents are built with.
FORMATS: Final = {
    "langgraph": "langchain",
    "openai-agents": "openai_agents",
    "claude-agent-sdk": "claude",
}
#: Sources whose side effects the harness knows, and so publishes to the tool catalog.
KNOWN_EFFECTS: Final = frozenset({"local", "openapi", "a2a"})


class Harness:
    """``Harness()`` reads the environment (``.env.example`` lists every variable);
    ``Harness(config=Settings(...))`` is the same without it."""

    def __init__(self, config: Settings | None = None) -> None:
        self.settings = config or Settings.from_env()
        s = self.settings
        self.gateway = Gateway(s.bifrost_url, s.bifrost_virtual_key) if s.bifrost_url else None
        self.memory = Memory(s.memory_url, s.memory_api_key) if s.memory_url else None
        self.runs: Runs = HttpRuns(s.runs_url, s.runs_api_key) if s.runs_url else LocalRuns()
        self.writes = Writes()
        self.artifacts = Artifacts()
        self.judge = GroundedJudge(budget=JudgeBudget(s.eval_sample), model=self.gateway)
        #: every agent wrapped here, by id (what ``python -m trellis.worker`` serves)
        self.agents: dict[str, Agent] = {}
        self._registered: set[tuple[str, str]] = set()
        telemetry.configure(s)

    # ------------------------------------------------------------------ attaching
    def wrap(
        self,
        target: Any,
        *,
        id: str,
        tools: Sequence[Source | Callable[..., Any]] = (),
        memory: MemoryMode = "off",
        approve: Mapping[str, Rule] | None = None,
        tool_hints: bool = False,
    ) -> Agent:
        """Attach the harness to ``target`` (a compiled LangGraph graph, an OpenAI Agents
        ``Agent``, ``ClaudeAgentOptions``, a ``ReAct``, or ``async (input, agent) -> answer``)."""
        agent = Agent(
            self, target, id=id, tools=tools, memory=memory, approve=approve, tool_hints=tool_hints
        )
        if agent.id in self.agents:
            raise ConfigurationError(f"an agent {agent.id!r} is already wrapped by this harness")
        self.agents[agent.id] = agent
        return agent

    async def tools(
        self, *sources: Source | Callable[..., Any], framework: Framework, memory: bool = False
    ) -> Any:
        """The sources as ``framework``'s own tools, for building an agent with them before
        wrapping it: LangChain tools (LangGraph, Deep Agents), ``FunctionTool``\\ s (OpenAI
        Agents), or one in-process MCP server (Claude). ``memory=True`` adds the memory
        service's agent tools. Every call is still the harness's: policy, approval, record."""
        tools = await self.resolve([as_source(s) for s in sources], tenant=self.settings.tenant)
        if memory:
            if self.memory is None:
                raise ConfigurationError("memory tools need MEMORY_URL")
            scope = RunMemory(self.memory, self.memory.client.bind(tenant_id=self.settings.tenant))
            tools.extend(await self.memory_tools(scope, read_only=False))
        return convert(FORMATS[framework], tools)  # type: ignore[arg-type]

    def worker(self, agents: Sequence[Agent], *, concurrency: int = WORKER_CONCURRENCY) -> Worker:
        """A worker that claims these agents' queued runs and executes them."""
        return Worker(self, agents, concurrency=concurrency)

    async def feedback(
        self,
        run_id: str,
        verdict: FeedbackVerdict | str,
        correction: Any = None,
        *,
        reviewer: str | None = None,
    ) -> Feedback:
        """What a person said about a run, stored in the memory service."""
        if self.memory is None:
            raise ConfigurationError("feedback is stored in the memory service: set MEMORY_URL")
        record = await self.runs.get(run_id)
        if record is None:
            raise ConfigurationError(f"no run {run_id}")
        feedback = Feedback(
            tenant_id=record.tenant_id,
            workspace_id=record.workspace_id,
            user_id=record.user_id,
            agent_id=record.agent_id,
            agent_run_id=run_id,
            target_kind=FeedbackTargetKind.RUN,
            target_id=run_id,
            verdict=FeedbackVerdict(verdict),
            correction=correction,
            reviewer=reviewer or record.user_id,
            source=FeedbackSource.HUMAN,
        )
        scope = Identity(
            tenant=record.tenant_id,
            user=record.user_id or "system",
            agent_id=record.agent_id,
            run_id=run_id,
            thread=record.thread_id,
        )
        await self.memory.bind(scope).feedback(feedback)
        return feedback

    async def aclose(self) -> None:
        """Finish the queued writes and close the clients."""
        await self.writes.aclose()
        closers = [c.aclose() for c in (self.gateway, self.memory, self.runs) if c is not None]
        await asyncio.gather(*closers)

    async def __aenter__(self) -> Harness:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    # ------------------------------------------------------------------ used by agents
    async def resolve(self, sources: Sequence[Source], *, tenant: str) -> list[Tool]:
        """Every source's tools. Names must be unique across an agent's sources."""
        services = _Services(self, tenant)
        tools: dict[str, Tool] = {}
        for source in sources:
            for found in await source.resolve(services):
                if found.name in tools:
                    raise ConfigurationError(f"two tools are named {found.name!r}")
                tools[found.name] = found
        known = [t.spec for t in tools.values() if t.spec.source in KNOWN_EFFECTS]
        if known and self.memory is not None:
            catalog = services.catalog()
            self.writes.submit("memory.tool_catalog", lambda: catalog.publish_catalog(known))
        return list(tools.values())

    async def memory_tools(self, run_memory: RunMemory, read_only: bool) -> list[Tool]:
        """The memory service's agent tools, each calling the service in the current run."""
        return [
            Tool(spec, _memory_call(spec))
            for spec in await run_memory.agent_tools(read_only=read_only)
        ]

    def registered(self, memory: Memory, identity: Identity) -> None:
        """Register ``TRELLIS_MEMORY_MODEL_KEY`` for an agent once per process (idempotent)."""
        key = self.settings.memory_model_key
        scope = (identity.tenant, identity.agent_id)
        if key is None or scope in self._registered:
            return
        self._registered.add(scope)
        run_memory = memory.bind(identity)
        self.writes.submit(
            "memory.model_key", lambda: run_memory.register_model_key(key, identity.agent_id)
        )


class _Services:
    """What sources may use while resolving: the gateway, and the catalog's side effects."""

    def __init__(self, harness: Harness, tenant: str) -> None:
        self.harness = harness
        self.tenant = tenant

    @property
    def gateway(self) -> Gateway:
        if self.harness.gateway is None:
            raise ConfigurationError("mcp() tools are served by Bifrost: set BIFROST_URL")
        return self.harness.gateway

    def catalog(self) -> RunMemory:
        memory = self.harness.memory
        assert memory is not None
        return RunMemory(memory, memory.client.bind(tenant_id=self.tenant))

    async def side_effects(self, names: list[str]) -> dict[str, str]:
        if self.harness.memory is None or not names:
            return {}
        return await self.catalog().side_effects(names)


def _memory_call(spec: ToolSpec) -> Callable[[dict[str, Any]], Any]:
    async def run(args: dict[str, Any]) -> Any:
        runtime = current()
        if runtime is None or runtime.run_memory is None:
            raise ConfigurationError(f"{spec.name} needs a run with memory on")
        return await runtime.run_memory.call_agent_tool(spec.name, args)

    return run
