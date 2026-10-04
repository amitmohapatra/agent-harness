"""``Harness``: the one object an application constructs. It reads the deployment from the
environment, owns the clients (Bifrost, memory, runs), the background writes and the scores,
and attaches all of it to agents with :meth:`Harness.wrap`.

Nothing about an agent is configured beyond ``h.wrap(target, id=...)``: memory is on when the
deployment has a memory service, the MCP tools are the ones the Bifrost virtual key allows,
risk tiers and approval rules come from the tools and the catalog, and who the deployment is
(its tenant) comes from ``TRELLIS_API_KEY`` itself — asked of the memory service, remembered
for the process (asked again every :data:`KEY_TTL_SECONDS`, the last answer kept while the
service is down).
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, Final, Literal

from trellis.contracts import ConfigurationError, FeedbackVerdict, ToolSpec
from trellis.harness import telemetry
from trellis.harness.adapters import convert
from trellis.harness.agent import Agent
from trellis.harness.clients.bifrost import Gateway
from trellis.harness.clients.memory import TOOL_SEARCH, Memory, RunMemory
from trellis.harness.clients.runs import HttpRuns, LocalRuns, Runs, RunSummary
from trellis.harness.fresh import Fresh
from trellis.harness.identity import Identity
from trellis.harness.runtime import current
from trellis.harness.settings import Settings
from trellis.harness.tools.base import Source, Tool
from trellis.harness.tools.sources import as_source
from trellis.harness.tools.toolbox import Published, Toolbox
from trellis.harness.worker import Worker
from trellis.harness.writes import Writes
from trellis.memory.errors import AuthenticationError, AuthorizationError
from trellis.memory.models import DocumentInfo, Feedback, KeyInfo

Framework = Literal["langgraph", "openai-agents", "claude-agent-sdk"]
#: The native tool format each framework's agents are built with.
FORMATS: Final = {
    "langgraph": "langchain",
    "openai-agents": "openai_agents",
    "claude-agent-sdk": "claude",
}
#: The metadata key on a LangChain tool built by :meth:`Harness.tools`: which call built it.
TOOLBOX: Final = "trellis_toolbox"
#: The tenant of a deployment with no memory service (development: nothing to ask).
LOCAL_TENANT: Final = "default"
#: How long what the memory service said about ``TRELLIS_API_KEY`` is kept before it is asked
#: again; while the service cannot be reached the last answer is kept (asked again after the
#: retry interval).
KEY_TTL_SECONDS: Final = 600.0
KEY_RETRY_SECONDS: Final = 30.0
#: How a person's verdict reads as a Langfuse score.
VERDICT_SCORES: Final = {"confirm": 1.0, "approve": 1.0, "edit": 0.5, "correct": 0.0, "reject": 0.0}


class Harness:
    """``Harness()`` reads the environment (``.env.example`` lists every variable);
    ``Harness(config=Settings(...))`` is the same without it."""

    def __init__(self, config: Settings | None = None) -> None:
        self.settings = config or Settings.from_env()
        s = self.settings
        if s.runs_url and not s.memory_url:
            raise ConfigurationError(
                "RUNS_URL needs MEMORY_URL: agent-runs accepts the keys the memory service "
                "issues, and the harness learns its tenant from there"
            )
        self.gateway = Gateway(s.bifrost_url, s.bifrost_virtual_key) if s.bifrost_url else None
        self.memory = Memory(s.memory_url, s.api_key) if s.memory_url else None
        self.runs: Runs = HttpRuns(s.runs_url, s.api_key) if s.runs_url else LocalRuns()
        self.writes = Writes(spool=s.spool_dir, replay=self._replay)
        self.scores = telemetry.Scores.of(s)
        #: every agent wrapped here, by id (what ``python -m trellis.worker`` serves)
        self.agents: dict[str, Agent] = {}
        #: the sources of each :meth:`tools` call, by the toolbox number its tools carry: a
        #: LangGraph agent's toolbox is the sources of the calls its graph's tools came from
        self._built: dict[int, list[Source]] = {}
        self._key = Fresh(
            self._whoami,
            what="who TRELLIS_API_KEY is",
            ttl=KEY_TTL_SECONDS,
            retry=KEY_RETRY_SECONDS,
            fatal=(ConfigurationError,),
        )
        self._registered: set[tuple[str, str]] = set()
        self._published: dict[str, Published] = {}
        telemetry.configure(s)

    # ------------------------------------------------------------------ attaching
    def wrap(
        self, target: Any, *, id: str, tools: Sequence[Source | Callable[..., Any]] = ()
    ) -> Agent:
        """Attach the harness to ``target`` (a compiled LangGraph graph, an OpenAI Agents
        ``Agent``, ``ClaudeAgentOptions``, a ``ReAct``, or ``async (input, agent) -> answer``).
        ``tools`` are the agent's own, run in this process (functions, ``a2a``, ``openapi``);
        its MCP tools are the ones the Bifrost virtual key allows."""
        agent = Agent(self, target, id=id, tools=tools)
        if agent.id in self.agents:
            raise ConfigurationError(f"an agent {agent.id!r} is already wrapped by this harness")
        self.agents[agent.id] = agent
        return agent

    async def tools(self, *sources: Source | Callable[..., Any], framework: Framework) -> Any:
        """The toolbox as ``framework``'s own tools, for building an agent with them before
        wrapping it: LangChain tools (LangGraph, Deep Agents), ``FunctionTool``\\ s (OpenAI
        Agents), or one in-process MCP server (Claude). It holds ``sources``, the MCP tools
        the virtual key allows and — memory on — the memory service's agent tools. Every call
        is still the harness's: policy, approval, record."""
        mine = [as_source(s) for s in sources]
        tenant = await self.tenant()
        tools = await self.resolve(mine, tenant=tenant)
        if self.memory is not None:
            scope = self.memory.scoped(tenant)
            tools.extend(await self.memory_tools(scope))
        native = convert(FORMATS[framework], tools)  # type: ignore[arg-type]
        if framework == "langgraph" and native:
            number = len(self._built)
            self._built[number] = mine
            for tool in native:
                tool.metadata = {**(tool.metadata or {}), TOOLBOX: number}
        return native

    def worker(self, agents: Sequence[Agent], *, concurrency: int | None = None) -> Worker:
        """A worker that claims these agents' queued runs and executes them, ``concurrency``
        at a time (else ``TRELLIS_WORKER_CONCURRENCY``, else the CPU count from 1 to 8)."""
        return Worker(self, agents, concurrency=concurrency)

    async def inbox(self, assignee: str | None = None) -> list[RunSummary]:
        """The paused runs waiting on a person — ``assignee`` (``user:…``, ``role:…``), or
        everyone in the tenant — newest first. Answer one with ``agent.resume``."""
        return list(await self.runs.inbox(await self.tenant(), assignee))

    async def feedback(
        self, run_id: str, verdict: FeedbackVerdict | str, correction: Any = None
    ) -> Feedback | None:
        """What a person said about a run: a score on its trace (Langfuse, when the OTLP
        settings reach it; a ``score`` span otherwise) and — memory on — the run's ``human``
        feedback. The memory service stores it pending (``review.state``) until the tenant
        administrator approves it, and only then does it outrank the judge's and the run's
        own; the stored record is returned (None with memory off)."""
        chosen = FeedbackVerdict(verdict)
        record = await self.runs.get(run_id)
        if record is None:
            raise ConfigurationError(f"no run {run_id}")
        stored: Feedback | None = None
        if self.memory is not None:
            scope = Identity(
                tenant=record.tenant_id,
                user=record.user_id or record.on_behalf_of or "system",
                agent_id=record.agent_id,
                run_id=run_id,
                thread=record.thread_id,
            )
            stored = await self.memory.bind(scope).run_feedback(
                chosen.value,
                source="human",
                correction=correction,
                reviewer=scope.user,
                key=f"{run_id}:human:{chosen.value}",
            )
        comment = None if correction is None else str(correction)
        await self.score(
            run_id,
            "feedback",
            VERDICT_SCORES[chosen.value],
            key=f"{run_id}:feedback",
            comment=comment,
        )
        return stored

    async def add_document(
        self,
        file: Any,
        *,
        user: str,
        tenant: str | None = None,
        thread: str | None = None,
        title: str | None = None,
        visibility: str | None = None,
        wait: float | None = 60.0,
    ) -> DocumentInfo:
        """Add a file to ``user``'s document memory (or one thread's, with ``thread``) so the
        agents' context cites it: bytes, a path, or a (filename, bytes, media_type) tuple.
        Waits until it is indexed unless ``wait`` is None. ``visibility`` widens who may
        retrieve it (``WORKSPACE``, ``TENANT``)."""
        if self.memory is None:
            raise ConfigurationError("memory is off in this deployment: set MEMORY_URL")
        scope = self.memory.for_user(await self.tenant(tenant), user, thread)
        return await scope.add_document(file, title=title, visibility=visibility, wait=wait)

    async def aclose(self) -> None:
        """Finish the queued writes and close the clients."""
        await self.writes.aclose()
        closers = [
            c.aclose() for c in (self.gateway, self.memory, self.runs, self.scores) if c is not None
        ]
        await asyncio.gather(*closers)

    async def __aenter__(self) -> Harness:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    # ------------------------------------------------------------------ who we are
    async def key(self) -> KeyInfo:
        """What the memory service says about ``TRELLIS_API_KEY``: asked at first use, then
        every :data:`KEY_TTL_SECONDS`; while the service cannot be reached the last answer
        stands. ``ConfigurationError`` when the service refuses the key, or could never be
        reached to say who it is."""
        return await self._key.get()

    async def _whoami(self) -> KeyInfo:
        if self.memory is None:
            return KeyInfo(
                key_id="local", tenant_id=LOCAL_TENANT, principal="local", role="service"
            )
        try:
            return await self.memory.key()
        except (AuthenticationError, AuthorizationError) as exc:
            raise ConfigurationError(
                f"the memory service at MEMORY_URL refused TRELLIS_API_KEY: {exc}"
            ) from exc
        except Exception as exc:
            if self._key.value is not None:
                raise  # a known key stands while the service is away
            raise ConfigurationError(
                "the memory service at MEMORY_URL could not be reached to say who "
                f"TRELLIS_API_KEY is (its tenant): {type(exc).__name__}: {exc}"
            ) from exc

    async def tenant(self, requested: str | None = None) -> str:
        """The tenant a call runs in: the key's own. Only a platform key (no tenant of its
        own) names one per call."""
        own = (await self.key()).tenant_id
        if own is None:
            if requested is None:
                raise ConfigurationError("a platform key names the tenant: pass tenant=")
            return requested
        if requested is not None and requested != own:
            raise ConfigurationError(f"TRELLIS_API_KEY speaks for {own!r}, not {requested!r}")
        return own

    def built_for(self, tools: Sequence[Any]) -> list[Source]:
        """The sources of the :meth:`tools` calls that ``tools`` (a graph's bound tools) came
        from — each graph has its own toolbox, so two graphs may each have a ``search``."""
        numbers = {
            n for t in tools if (n := (getattr(t, "metadata", None) or {}).get(TOOLBOX)) is not None
        }
        return [s for n in sorted(numbers) for s in self._built[n]]

    @property
    def known_tenant(self) -> str | None:
        """The key's tenant once :meth:`key` has been asked (``None`` before, or for a
        platform key)."""
        return self._key.value.tenant_id if self._key.value is not None else None

    async def writes_memory(self) -> bool:
        """Whether runs record their transcript, tool calls and outcome: memory is on."""
        return self.memory is not None

    # ------------------------------------------------------------------ used by agents
    def toolbox(self, sources: Sequence[Source], *, tenant: str) -> Toolbox:
        """A toolbox of ``sources`` and the MCP tools in ``tenant``, tiered by the catalog and
        kept fresh (``tools/toolbox.py``)."""
        catalog = self.memory.scoped(tenant) if self.memory is not None else None
        return Toolbox(
            sources,
            gateway=self.gateway,
            catalog=catalog,
            writes=self.writes,
            published=self._published.setdefault(tenant, Published()),
        )

    async def resolve(self, sources: Sequence[Source], *, tenant: str) -> list[Tool]:
        """The toolbox once: ``sources`` and the MCP tools, tiered by the catalog."""
        return await self.toolbox(sources, tenant=tenant).tools()

    def _replay(self, record: dict[str, Any]) -> Callable[[], Awaitable[object]] | None:
        """A memory write an earlier process spooled, as a write again (memory on)."""
        return self.memory.replay(record) if self.memory is not None else None

    async def memory_tools(self, run_memory: RunMemory) -> list[Tool]:
        """The memory service's agent tools, each calling the service in the current run."""
        return [Tool(spec, _memory_call(spec)) for spec in await run_memory.agent_tools()]

    async def registered(self, memory: Memory, identity: Identity) -> None:
        """Register ``BIFROST_VIRTUAL_KEY`` as the agent's memory model key, once per process
        and agent (idempotent in the service)."""
        key = self.settings.bifrost_virtual_key
        scope = (identity.tenant, identity.agent_id)
        if key is None or scope in self._registered or not await self.writes_memory():
            return
        self._registered.add(scope)
        agent_memory = memory.scoped(identity.tenant, identity.agent_id)
        await self.writes.submit("memory.model_key", lambda: agent_memory.register_model_key(key))

    async def score(
        self, run_id: str, name: str, value: float, *, key: str, comment: str | None = None
    ) -> None:
        """A score on the run's trace: a ``score`` span always, and Langfuse's scores API
        when the OTLP settings reach it."""
        telemetry.score_span(run_id, name, value, comment)
        if self.scores is not None:
            await self.scores.post(
                run_id, name, value, data_type="NUMERIC", comment=comment, key=key
            )


def _memory_call(spec: ToolSpec) -> Callable[[dict[str, Any]], Any]:
    async def run(args: dict[str, Any]) -> Any:
        runtime = current()
        if runtime is None or runtime.run_memory is None:
            raise ConfigurationError(f"{spec.name} needs a run with memory on (MEMORY_URL)")
        if spec.name == TOOL_SEARCH:  # among the tools this run can actually call
            hints = await runtime.tools.hints(str(args.get("task", "")))
            # what the model reads: each tool's confidence, arguments and gaps, and the plan
            return hints.model_dump(mode="json", exclude_defaults=True)
        return await runtime.run_memory.call_agent_tool(spec.name, args)

    return run
