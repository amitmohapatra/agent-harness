"""``Harness``: the one object an application constructs — its connections to the services
(the run store, the memory service, the Bifrost gateway), each built from the environment when
it is not given (or left off: ``memory=False``), with the background writes and the scores —
and attaches all of it to agents with :meth:`Harness.wrap`.

Nothing about an agent is configured beyond ``h.wrap(target, id=...)``: memory is on when the
deployment has a memory service, the MCP tools are the ones the Bifrost virtual key allows,
whether a call runs, is announced or asks is governance's (the tools' risks and the memory
service's catalog rules), and who the deployment is (its tenant) comes from
``TRELLIS_API_KEY`` itself — asked of the memory service, remembered for the process (asked
again every :data:`KEY_TTL_SECONDS`, the last answer kept while the service is down).
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Awaitable, Callable, Collection, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Final, Literal, TypeVar

from trellis.contracts import (
    ConfigurationError,
    FeedbackSource,
    FeedbackVerdict,
    RunStatus,
    ToolSpec,
)
from trellis.harness import telemetry
from trellis.harness.adapters import convert
from trellis.harness.agent import Agent
from trellis.harness.bounded import LRU
from trellis.harness.budget import Budget
from trellis.harness.clients.bifrost import Gateway
from trellis.harness.clients.memory import TOOL_SEARCH, Memory, RunMemory
from trellis.harness.evals import EvalReport, EvalServices, Evaluator
from trellis.harness.evals import evaluate as run_evaluation
from trellis.harness.features import Feature
from trellis.harness.fresh import Fresh
from trellis.harness.gateways import GatewayKind, bifrost_wins, model_gateway
from trellis.harness.governance import Governance
from trellis.harness.governance.catalog import MemoryCatalog
from trellis.harness.hooks import Hooks
from trellis.harness.identity import Identity
from trellis.harness.prompts import Prompts, Rendered, Stored
from trellis.harness.removed import removed
from trellis.harness.runs import LocalRuns, RunStore
from trellis.harness.runtime import current
from trellis.harness.settings import Settings
from trellis.harness.subagents import asked_by
from trellis.harness.tools.base import Source, Tool
from trellis.harness.tools.mcp import MCPSource
from trellis.harness.tools.toolbox import Toolbox
from trellis.harness.worker import Worker
from trellis.harness.writes import Writes
from trellis.memory import MemoryClient
from trellis.memory.errors import (
    AuthenticationError,
    AuthorizationError,
    DependencyUnavailableError,
)
from trellis.memory.models import DocumentInfo, Feedback, KeyInfo
from trellis.runs import RunsClient, RunSummary

log = logging.getLogger("trellis.harness")

_Made = TypeVar("_Made", Gateway, MemoryClient, RunsClient)

#: The tenant of a deployment with no memory service (development: nothing to ask).
LOCAL_TENANT: Final = "default"
#: How long what the memory service said about ``TRELLIS_API_KEY`` is kept before it is asked
#: again; while the service cannot be reached the last answer is kept (asked again after the
#: retry interval).
KEY_TTL_SECONDS: Final = 600.0
KEY_RETRY_SECONDS: Final = 30.0
#: What the memory service says when it takes no model keys (``ProviderNotConfigured``: its
#: credential encryption is not configured), answered as DEPENDENCY_UNAVAILABLE.
NOT_CONFIGURED: Final = "not configured"
#: A person's verdict on a run (``h.feedback``), and how it reads as a Langfuse score. ``approve``
#: and ``edit`` are not among them: they are the system's record of an approval decision
#: (``agent.resume``), written by governance.
VERDICT_SCORES: Final = {"confirm": 1.0, "correct": 0.0, "reject": 0.0}
#: Methods ``Harness`` no longer has, and what to use instead (said when one is called).
REMOVED_METHODS: Final = {
    "governance_middleware": "a wrapped create_agent or Deep Agents graph's own tools are "
    "governed already, middleware() includes the same governance, and outside a harness run "
    "there is nothing to govern",
}
#: Paused runs one inbox page asks for (agent-runs' page limit), and the pages one inbox read
#: follows at most: past that the newest are returned and a warning is logged.
INBOX_LIMIT: Final = 500
INBOX_MAX_PAGES: Final = 10


class Harness:
    """The connections to the services, the rest from the environment: ``Harness()`` reads it
    (``.env.example`` lists every variable); ``Harness(config=Settings(...))`` is the same
    without it.

    ``memory`` (a ``trellis.memory.MemoryClient``), ``runs`` (a
    :class:`~trellis.harness.runs.RunStore`: ``trellis.runs.RunsClient``, ``LocalRuns``) and
    ``gateway`` (a ``trellis.harness.clients.bifrost.Gateway``) replace that service (tests, in
    process): given, each is used as it is (and closed by its owner, not here); not given, each
    is built from the environment (``MEMORY_URL``, ``RUNS_URL``, ``BIFROST_URL``); ``False``
    leaves memory, the gateway or agent-runs off (runs kept in this process) even where the
    environment names them."""

    def __init__(
        self,
        config: Settings | None = None,
        *,
        memory: MemoryClient | Literal[False] | None = None,
        runs: RunStore | Literal[False] | None = None,
        gateway: Gateway | Literal[False] | None = None,
    ) -> None:
        self.settings = config or Settings.from_env()
        s = self.settings
        builds_runs = runs is None and s.runs_url is not None
        builds_memory = memory is None and s.memory_url is not None
        if builds_runs and not (builds_memory or memory):
            raise ConfigurationError(
                "RUNS_URL needs MEMORY_URL: agent-runs accepts the keys the memory service "
                "issues, and the harness learns its tenant from there"
            )
        if (builds_runs or builds_memory) and not s.api_key:
            raise ConfigurationError(
                "MEMORY_URL (and RUNS_URL) need TRELLIS_API_KEY: both services refuse a call "
                "without a key (a development memory service takes one of its trusted_dev keys)"
            )
        #: the clients built here from the environment, closed by :meth:`aclose`
        self._made: list[Any] = []
        if gateway is None and s.bifrost_url:
            gateway = self._making(Gateway(s.bifrost_url, s.bifrost_virtual_key))
        self.gateway: Gateway | None = gateway or None
        #: what evaluation reaches: Langfuse (scores, datasets, dataset runs) and the judge —
        #: the gateway, with the judge's own virtual key when it has one (its budget apart from
        #: the agents'); each agent's are ``agent.evals``
        self.evals = EvalServices.of(s, gateway=self.gateway)
        if builds_memory:
            memory = self._making(MemoryClient(s.memory_url, api_key=s.api_key))
        self.memory = Memory(memory) if memory else None
        if builds_runs:
            runs = self._making(RunsClient(s.runs_url, api_key=s.api_key))
        self.runs: RunStore = runs or LocalRuns()
        self.writes = Writes(spool=s.spool_dir, replay=self._replay)
        #: every agent wrapped here, by id (what ``python -m trellis.harness.worker`` serves)
        self.agents: dict[str, Agent] = {}
        self._key = Fresh(
            self._whoami,
            what="who TRELLIS_API_KEY is",
            ttl=KEY_TTL_SECONDS,
            retry=KEY_RETRY_SECONDS,
            fatal=(ConfigurationError,),
        )
        self._keys_registered: set[tuple[str, str]] = set()
        #: the memory service said it takes no model keys: none is registered again
        self._model_keys_off = False
        #: governance per tenant (:meth:`governance`), the least recently used forgotten past
        #: the bound; ``_governance_given`` replaces it for every tenant (tests)
        self._governances: LRU[str, Governance] = LRU()
        self._governance_given: Governance | None = None
        #: the gateway's stored prompts (``stored``, :meth:`prompt`)
        self.prompts = Prompts(self.gateway)
        telemetry.configure()
        #: where a model name goes (``trellis.harness.gateways``): Bifrost, else the
        #: provider's own SDK with its own key; these settings', not the environment's
        self.model_gateway = model_gateway(s)
        if self.model_gateway.kind is GatewayKind.BIFROST:
            bifrost_wins(os.environ)
        log.info(
            "trellis harness: models %s, memory %s, runs %s, spool %s, traces %s "
            "(trellis doctor prints each setting and its source)",
            f"{self.model_gateway.title} {self.model_gateway.url}",
            s.memory_url or "off",
            s.runs_url or "in process",
            s.spool_dir or "none",
            "on" if telemetry.exporting() else "off",
        )

    # ------------------------------------------------------------------ attaching
    def wrap(
        self,
        agent: Any,
        *,
        id: str,
        version: str | None = None,
        description: str | None = None,
        hooks: Sequence[Hooks] = (),
        judges: Sequence[Evaluator] = (),
        budget: Budget | None = None,
        run_timeout: float | None = None,
        without: Collection[str] = (),
        framework_options: Mapping[str, Any] | None = None,
    ) -> Agent:
        """Attach the harness to ``agent`` — a compiled LangGraph graph (Deep Agents and
        ``ReAct`` included), an OpenAI Agents ``Agent``, ``ClaudeAgentOptions``, or
        ``async (input, agent) -> answer`` — and say who it is and how it is run. What the agent
        *is* (its model, instructions and tools) is its builder's: ``ReAct(tools=,
        subagents=)``, or the framework's own (its tools are governed as the harness's); every
        run also has the MCP tools the virtual key allows and the memory tools.

        ``id`` names it (its runs, memory, catalog, traces); ``version`` is the version of its
        code (else ``service.version`` in ``OTEL_RESOURCE_ATTRIBUTES``), recorded with each run
        it starts: a run resumed on another version goes on, with a warning naming both;
        ``description`` says what it does (its A2A card, an agent calling it; else its
        function's docstring). ``hooks`` run around its runs, model calls and tool calls
        (``trellis.harness.hooks``); ``judges`` score a share of its successful runs online
        (``llm_judge(sample=)``). For every run unless the run says otherwise: ``budget`` is
        the most one may spend — tokens, dollars, tool calls (``trellis.harness.budget``) —,
        ``run_timeout`` the most working time it may take, in seconds — on every entry:
        ``serve_chat``, ``serve_a2a``, ``h.evaluate``, scheduled runs too —, ``without`` what
        the harness does not do (``memory``, ``tool_search``, ``mcp``, ``code_mode``,
        ``grounding``, ``judges``: ``trellis.harness.features``), and ``framework_options``
        the framework's own options for each run's call, passed through unchanged (a LangGraph
        config's keys, ``Runner.run``'s arguments, ``ClaudeAgentOptions`` fields; each
        framework's page says which the harness keeps for itself): a run's own go over them."""
        wrapped = Agent(
            self,
            agent,
            id=id,
            version=version,
            description=description,
            hooks=hooks,
            judges=judges,
            budget=budget,
            run_timeout=run_timeout,
            without=without,
            framework_options=framework_options,
        )
        if wrapped.id in self.agents:
            raise ConfigurationError(f"an agent {wrapped.id!r} is already wrapped by this harness")
        self.agents[wrapped.id] = wrapped
        return wrapped

    def worker(self, agents: Sequence[Agent], *, concurrency: int | None = None) -> Worker:
        """A worker that claims these agents' queued runs and executes them, ``concurrency``
        at a time (else the CPU count, from 1 to 8)."""
        return Worker(self, agents, concurrency=concurrency)

    async def inbox(
        self, assignee: str | None = None, *, tenant: str | None = None
    ) -> list[RunSummary]:
        """The paused runs waiting on a person — ``assignee`` (``user:…``, ``role:…``), or
        everyone in the tenant — newest first, at most :data:`INBOX_MAX_PAGES` pages of
        :data:`INBOX_LIMIT` (a warning says when there may be more). Answer one with
        ``agent.resume``. A sub-agent's question is listed once, on its parent's run (which
        is answered); its own paused run is left out. ``tenant`` is named by a platform key
        only."""
        waiting = [
            summary
            async for summary in self.runs.iterate(
                status=RunStatus.PAUSED,
                assignee=assignee,
                limit=INBOX_LIMIT,
                tenant=await self._tenant(tenant),
                max_pages=INBOX_MAX_PAGES,
            )
        ]
        if len(waiting) >= INBOX_LIMIT * INBOX_MAX_PAGES:
            log.warning(
                "the inbox of %s holds at least %d paused runs: the newest are returned",
                assignee or "the tenant",
                len(waiting),
            )
        asked = [asked_by(s.awaiting) for s in waiting if s.awaiting is not None]
        children = {named["run_id"] for named in asked if named is not None}
        return [s for s in waiting if s.run_id not in children]

    async def feedback(
        self,
        run_id: str,
        verdict: FeedbackVerdict | str,
        correction: Any = None,
        *,
        tenant: str | None = None,
    ) -> Feedback | None:
        """What a person said about a run: a score on its trace (Langfuse, when the OTLP
        settings reach it; a ``score`` span otherwise) and — memory on — the run's ``human``
        feedback, citing the memories the run's answer rested on (``RunMemory.cited``), whose
        confidence it moves once it is applied. ``verdict`` is ``confirm``, ``reject`` or
        ``correct`` (with the ``correction``); ``approve`` and ``edit`` are refused: they record
        a decision on a paused action, which ``agent.resume`` makes. The memory service stores it
        pending (``review.state``) until the tenant administrator reviews it (accepts or
        dismisses it), and only once accepted does it outrank the judge's and the run's own; the
        stored record is returned (None with memory off). ``tenant`` is the run's, named by a
        platform key only."""
        chosen = FeedbackVerdict(verdict)
        if chosen.value not in VERDICT_SCORES:
            raise ConfigurationError(
                f"h.feedback takes a person's verdict on a run: confirm, reject or correct, not "
                f"{chosen.value!r} — approve and edit record a decision on a paused action "
                "(agent.resume(..., decision=...))"
            )
        record = await self.runs.get(run_id, tenant=await self._tenant(tenant))
        if record is None:
            raise ConfigurationError(f"no run {run_id}")
        stored: Feedback | None = None
        if self.memory is not None:
            scope = Identity.of(record)
            run_memory = self.memory.bind(scope)
            stored = await run_memory.run_feedback(
                chosen.value,
                source=FeedbackSource.HUMAN.value,
                correction=correction,
                reviewer=scope.user,
                key=f"{run_id}:human:{chosen.value}",
                evidence_refs=await run_memory.cited(),
            )
        comment = None if correction is None else str(correction)
        await self._score(
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
        scope = self.memory.for_user(await self._tenant(tenant), user, thread)
        return await scope.add_document(file, title=title, visibility=visibility, wait=wait)

    @removed(evaluators="it is judges= now (h.evaluate(agent, dataset, judges))")
    async def evaluate(
        self,
        agent: Agent,
        dataset: str | Sequence[Any],
        judges: Sequence[Evaluator],
        *,
        run_name: str | None = None,
        description: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        concurrency: int = 4,
        limit: int | None = None,
        user: str | None = None,
    ) -> EvalReport:
        """Run ``agent`` on every item of ``dataset`` and score its answers with ``judges``
        (from ``trellis.harness.evals``: ``grounding()``, ``exact_match()``,
        ``contains()``, ``llm_judge(criteria)``, or any ``async (EvalCase) -> EvalScore |
        None``) — ``evaluate(agent, ...)``, with the agent's own services.

        ``dataset`` is a Langfuse dataset's name (Langfuse configured through the OTLP
        settings) or the items themselves (``{"input", "expected"?, "metadata"?}`` or
        ``EvalItem``\\ s). Each item runs through the normal pipeline — memory, tools,
        approvals — ``concurrency`` at a time (the first ``limit`` items only, with ``limit``),
        acting for ``user``; its scores go on its run's trace, and an item of a Langfuse dataset
        is linked to the dataset run ``run_name`` (with ``description`` and ``metadata``), and
        every run's spans carry Langfuse's experiment attributes (Langfuse v4 builds the
        experiment from them). An item that pauses for a person is
        ``interrupted`` (its run is cancelled) and one that fails is ``error``: neither stops the
        evaluation. The report lists the items in dataset order."""
        return await run_evaluation(
            agent,
            dataset,
            judges,
            run_name=run_name,
            description=description,
            metadata=metadata,
            concurrency=concurrency,
            limit=limit,
            user=user,
        )

    async def prompt(self, ref: str | Stored, /, **values: Any) -> Rendered:
        """The gateway's stored prompt ``ref`` (``"name"``, ``"name@version"``, or
        ``stored(...)``) with ``values`` filling its ``{{variables}}``: its ``.text`` and its
        ``.messages`` — for a framework's own instructions (LangGraph, OpenAI Agents, Claude,
        Deep Agents, a function). Inside a run it is pinned: journaled, so a resumed run reads
        the same version."""
        return await self.prompts.rendered(ref, **values)

    if not TYPE_CHECKING:  # pragma: no branch - a type checker refuses a removed method itself

        def __getattr__(self, name: str) -> Any:
            if name in REMOVED_METHODS:
                raise AttributeError(f"h.{name}() was removed: {REMOVED_METHODS[name]}")
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    async def aclose(self) -> None:
        """Finish the queued writes, export the queued spans and close the clients built here
        (the blocks given are their owner's to close)."""
        await self.writes.aclose()
        await telemetry.flush()
        await asyncio.gather(*(s.aclose() for s in self._mcp_sources()))
        judge = self.evals.judge_gateway
        judge = judge if judge is not self.gateway else None
        clients = (*self._made, judge, self.evals.langfuse)
        closers = [c.aclose() for c in clients if c is not None]
        await asyncio.gather(*closers)

    def _mcp_sources(self) -> list[MCPSource]:
        """The MCP servers this harness's agents reach themselves."""
        found = [s for a in self.agents.values() for s in a.sources]
        return list({id(s): s for s in found if isinstance(s, MCPSource)}.values())

    async def __aenter__(self) -> Harness:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    # ------------------------------------------------------------------ who we are
    async def _key_info(self) -> KeyInfo:
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

    async def _tenant(self, requested: str | None = None) -> str:
        """The tenant a call runs in: the key's own. Only a platform key (no tenant of its
        own) names one per call."""
        own = (await self._key_info()).tenant_id
        if own is None:
            if requested is None:
                raise ConfigurationError("a platform key names the tenant: pass tenant=")
            return requested
        if requested is not None and requested != own:
            raise ConfigurationError(f"TRELLIS_API_KEY speaks for {own!r}, not {requested!r}")
        return own

    async def _published(self, agents: Sequence[Agent]) -> None:
        """Each agent's tools published to the key's tenant now - at a worker's start - so
        tool search has them indexed before the first run asks for hints. A platform key
        knows no tenant (each run's first one publishes); a failure is a warning, and the
        first run publishes them anyway."""
        if self.memory is None:
            return
        try:
            tenant = (await self._key_info()).tenant_id
        except Exception as exc:
            log.warning("the tools were not published at start: %s", exc)
            return
        if tenant is None:
            return
        for agent in agents:
            try:
                await agent._toolbox(tenant).tools()
            except Exception as exc:
                log.warning("%s: its tools were not published at start: %s", agent.id, exc)

    @property
    def _known_tenant(self) -> str | None:
        """The key's tenant once it has been asked (``None`` before, or for a platform key)."""
        return self._key.value.tenant_id if self._key.value is not None else None

    def _making(self, client: _Made) -> _Made:
        """``client``, built here from the environment: :meth:`aclose` closes it."""
        self._made.append(client)
        return client

    # ------------------------------------------------------------------ used by agents
    def _governance(self, tenant: str) -> Governance:
        """Governance in ``tenant``: one per tenant — the memory service's tool catalog in its
        scope (memory on; publishes go through the background writes), or the tools' own risks
        only."""
        if self._governance_given is not None:
            return self._governance_given
        found = self._governances.get(tenant)
        if found is None:
            found = self._governances[tenant] = self._governed(tenant)
        return found

    def _governed(self, tenant: str) -> Governance:
        if self.memory is None:
            return Governance(tenant=tenant)
        scope = self.memory.scoped(tenant)
        writes = self.writes

        async def submit(entries: list[dict[str, object]], send: Callable[[], Any]) -> None:
            record = scope.record("publish_catalog", entries=entries)
            await writes.submit("memory.tool_catalog", send, record=record)

        return Governance(MemoryCatalog(scope.ctx), submit=submit, tenant=tenant)

    def _toolbox(
        self,
        sources: Sequence[Source],
        *,
        tenant: str,
        without: Collection[Feature] = (),
        agent_id: str | None = None,
    ) -> Toolbox:
        """A toolbox of ``sources`` and the MCP tools the virtual key allows in ``tenant``
        (none ``without`` ``mcp``: no MCP server is asked), published to its catalog and kept
        fresh (``tools/toolbox.py``) — ``agent_id``'s, whose learned approval rules governance
        reads."""
        gateway = None if "mcp" in without else self.gateway
        if "mcp" in without:  # its own MCP servers not reached either
            sources = [s for s in sources if not isinstance(s, MCPSource)]
        governance = self._governance(tenant)
        return Toolbox(sources, gateway=gateway, governance=governance, agent_id=agent_id)

    def _replay(self, record: dict[str, Any]) -> Callable[[], Awaitable[object]] | None:
        """A memory write an earlier process spooled, as a write again (memory on)."""
        return self.memory.replay(record) if self.memory is not None else None

    async def memory_tools(self, *, tenant: str | None = None) -> list[Any]:
        """The memory service's agent tools — its pull tools (``memory_search``...) and
        ``tool_search`` — as LangChain tools, for a graph you build yourself: put them in its
        ``ToolNode`` and bind them to its model. Each call goes to the memory service in the
        scope of the run making it (its user, thread and run; a call outside a run is refused)
        and is governed and recorded like any tool of the run. Memory off: none. ``tenant``
        is the runs', named by a platform key only."""
        if self.memory is None:
            return []
        scope = self.memory.scoped(await self._tenant(tenant))
        return convert("langchain", await self._pull_tools(scope))

    async def _pull_tools(self, run_memory: RunMemory) -> list[Tool]:
        """The memory service's agent tools, each calling the service in the current run."""
        return [
            Tool(
                spec,
                _memory_call(spec),
                feature="tool_search" if spec.name == TOOL_SEARCH else "memory_pull",
            )
            for spec in await run_memory.agent_tools()
        ]

    async def _registered(self, memory: Memory, identity: Identity) -> None:
        """Register ``BIFROST_VIRTUAL_KEY`` as the agent's memory model key, once per process
        and agent (idempotent in the service). A memory service that takes no model keys (its
        credential encryption is not configured) is told so once per process, in the log, and
        not asked again."""
        key = self.settings.bifrost_virtual_key
        scope = (identity.tenant, identity.agent_id)
        if key is None or self._model_keys_off or scope in self._keys_registered:
            return
        self._keys_registered.add(scope)
        agent_memory = memory.scoped(identity.tenant, identity.agent_id)

        async def register() -> None:
            try:
                await agent_memory.register_model_key(key)
            except DependencyUnavailableError as exc:
                if NOT_CONFIGURED not in str(exc):
                    raise
                if not self._model_keys_off:
                    self._model_keys_off = True
                    log.info(
                        "the memory service takes no model keys (%s): its LLM work for these "
                        "agents runs on the tenant's or the operator's key",
                        exc,
                    )

        await self.writes.submit("memory.model_key", register)

    async def _score(
        self,
        run_id: str,
        name: str,
        value: float | bool | str,
        *,
        key: str,
        comment: str | None = None,
        trace: str | None = None,
    ) -> None:
        """A score on the run's trace (``EvalServices.score``) — ``trace``'s, the run whose
        trace it is in, for a sub-agent's run —: a ``score`` span always, and Langfuse's scores
        API when the OTLP settings reach it — a number (``NUMERIC``), a bool (``BOOLEAN``, 1 or
        0) or a category (``CATEGORICAL``)."""
        trace_id = telemetry.trace_hex(trace or run_id)
        await self.evals.score(trace_id, name, value, key=key, comment=comment, run_id=run_id)


def _memory_call(spec: ToolSpec) -> Callable[[dict[str, Any]], Any]:
    async def run(args: dict[str, Any]) -> Any:
        # the bridge runs it only inside a run whose memory is on (``memory_pull`` among its
        # features): outside one, or without memory, it says so before this is reached
        runtime = current()
        assert runtime is not None and runtime.run_memory is not None
        if spec.name == TOOL_SEARCH:  # among the tools this run can actually call
            hints = await runtime.tools.hints(str(args.get("task", "")))
            # what the model reads: each tool's confidence, arguments and gaps, and the plan
            return hints.model_dump(mode="json", exclude_defaults=True)
        return await runtime.run_memory.call_agent_tool(spec.name, args)

    return run
