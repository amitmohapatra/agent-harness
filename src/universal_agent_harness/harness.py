"""``AgentHarness`` — the one object an application constructs (§1).

    harness = AgentHarness(memory=MemoryClient(...))
    wrapped = harness.wrap(existing_agent, agent_id="inventory-agent")
    result  = await wrapped(payload, context=context)

Three integration modes, all backed by the same pipeline:

* **Level 1** ``harness.wrap(callable, ...)`` — an existing agent, unchanged;
* **Level 2** ``@harness.agent(...)`` — the agent receives an :class:`AgentRuntime`;
* **Level 3** ``async with harness.execution(context)`` — a block of code you call yourself.

The harness composes providers behind protocols and owns no framework knowledge. Framework
support arrives as an adapter (``harness.langgraph``), never as core code.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import AsyncIterator, Callable, Iterable, Mapping
from contextlib import asynccontextmanager
from typing import Any

from universal_agent_contracts.context import AgentExecutionContext
from universal_agent_contracts.descriptors import AgentDescriptor, SkillDescriptor
from universal_agent_contracts.errors import AgentError
from universal_agent_contracts.events import LifecycleEvent
from universal_agent_contracts.messages import AgentRequest, AgentResponse

from universal_agent_harness.artifacts.stores import (
    FileArtifactStore,
    InMemoryArtifactStore,
    NoArtifactStore,
)
from universal_agent_harness.config.settings import HarnessConfig
from universal_agent_harness.evaluation.events import (
    CompositeEvaluationSink,
    LifecycleDispatcher,
    LoggingEvaluationSink,
    NoOpEvaluationProvider,
)
from universal_agent_harness.execution.context_factory import ContextFactory
from universal_agent_harness.execution.coordinator import ExecutionCoordinator, RuntimeBuilder
from universal_agent_harness.execution.retry import RetryPolicy
from universal_agent_harness.execution.sync import run_sync
from universal_agent_harness.interceptors.base import BaseInterceptor, InterceptorChain
from universal_agent_harness.interceptors.evaluation import EvaluationEventInterceptor
from universal_agent_harness.interceptors.identity import IdentityInterceptor
from universal_agent_harness.interceptors.memory import (
    MemoryContextInterceptor,
    MemoryObservationInterceptor,
)
from universal_agent_harness.interceptors.policy import PolicyInterceptor
from universal_agent_harness.interceptors.result import ResultValidationInterceptor
from universal_agent_harness.interceptors.telemetry import TelemetryInterceptor
from universal_agent_harness.interceptors.timeout import TimeoutInterceptor
from universal_agent_harness.memory.client import MemoryFactory
from universal_agent_harness.memory.policy import MemoryPolicy
from universal_agent_harness.memory.writeback import WritebackQueue
from universal_agent_harness.models.providers import DirectModelClient, UnconfiguredModelClient
from universal_agent_harness.policy.providers import NoOpPolicyProvider
from universal_agent_harness.registry.client import NoOpAgentRegistry
from universal_agent_harness.runs import NoRunStore, RunRecorder
from universal_agent_harness.runtime.agent_runtime import AgentRuntime
from universal_agent_harness.runtime.logging import configure_logging
from universal_agent_harness.runtime.propagation import bind, current_context
from universal_agent_harness.telemetry.metrics import MetricsRecorder
from universal_agent_harness.telemetry.noop import NoOpTelemetryProvider
from universal_agent_harness.telemetry.otel import OpenTelemetryTelemetryProvider
from universal_agent_harness.telemetry.redaction import DefaultRedactor
from universal_agent_harness.telemetry.sampling import Sampler
from universal_agent_harness.telemetry.tracer import HarnessTracer
from universal_agent_harness.tools.local import LocalToolClient, NoToolsClient
from universal_agent_harness.tools.wrappers import wrap_tool as _wrap_tool

__version__ = "0.1.0"

#: How many memory/evaluation writes may be in flight before the harness writes inline.
MAX_PENDING_WRITEBACKS = 256


class AgentHarness:
    """The cross-cutting runtime layer around agents you already have."""

    def __init__(
        self,
        *,
        memory: Any = None,
        model: Any = None,
        tools: Any = None,
        artifacts: Any = None,
        config: HarnessConfig | str | dict[str, Any] | None = None,
        defaults: Mapping[str, Any] | None = None,
        telemetry: Any = None,
        policy: Any = None,
        registry: Any = None,
        runs: Any = None,
        evaluation_sink: Any = None,
        evaluation_provider: Any = None,
        prompts: Any = None,
        redactor: Any = None,
        interceptors: Iterable[BaseInterceptor] = (),
        listeners: Iterable[Any] = (),
        error_mode: str = "raise",
    ) -> None:
        self.config = config if isinstance(config, HarnessConfig) else HarnessConfig.load(config)
        self.defaults = dict(defaults or {})
        self.error_mode = error_mode
        self.version = __version__

        obs = self.config.observability
        configure_logging(obs.log_level, json_output=obs.structured_logging)

        # -- telemetry: OpenTelemetry first, Langfuse layered on the same spans (§21/§22)
        self.telemetry = telemetry or self._build_telemetry()
        # Capture flags decide *whether* a payload is attached; the redactor decides what an
        # allowed payload may contain. Gating twice would make ``capture.inputs`` do nothing.
        self.redactor = redactor or DefaultRedactor()
        self.tracer = HarnessTracer(
            self.telemetry,
            capture=self.config.telemetry.capture,
            redactor=self.redactor,
            metrics=MetricsRecorder(self.telemetry, enabled=self.config.telemetry.metrics_enabled),
            enabled=self.config.telemetry.enabled,
        )
        self.sampler = Sampler(self.config.telemetry.sampling)

        # -- providers
        self.memory_factory = MemoryFactory(
            memory, self.config.memory, timeout_seconds=self.config.timeouts.memory_seconds
        )
        self.default_memory_policy = self.memory_factory.default_policy
        self.model_client = self._build_model(model)
        self.tool_client = self._build_tools(tools)
        self.artifact_store = self._build_artifacts(artifacts)
        # A provider is enabled by being passed: no flag that has to agree with it.
        self.policy = policy or NoOpPolicyProvider()
        self.policy_enabled = policy is not None
        self.registry = registry or self._build_registry()
        self.registry_enabled = self.registry.name != "noop"
        self.runs = runs or self._build_runs()
        self.prompts = prompts
        self.evaluation_provider = evaluation_provider or NoOpEvaluationProvider()

        self.events = LifecycleDispatcher(list(listeners))
        if self.runs.name != "noop":
            # A listener, not an interceptor: recording a run is an observation of the turn,
            # not a step in it. An interceptor that failed would change the outcome, which
            # is the coupling the store's best-effort default exists to avoid.
            self._recorder = RunRecorder(self.runs)
            self.events.add(self._recorder)
        #: Bounded so a backlog can never grow without limit; writing inline is the
        #: fallback when it saturates.
        self.writeback = WritebackQueue(MAX_PENDING_WRITEBACKS)
        self.evaluation_sink = evaluation_sink or self._build_evaluation_sink()
        self.context_factory = ContextFactory(self.defaults)
        self.descriptors: dict[str, AgentDescriptor] = {}

        self.chain = InterceptorChain([*self._core_interceptors(), *interceptors])
        self.runtime_builder = RuntimeBuilder(
            memory_factory=self.memory_factory,
            model_client=self.model_client,
            tool_client=self.tool_client,
            artifact_store=self.artifact_store,
            policy=self.policy if self.policy_enabled else None,
            events=self.events,
            timeouts=self.config.timeouts,
            tools_config=self.config.tools,
            artifacts_config=self.config.artifacts,
        )
        self.coordinator = ExecutionCoordinator(
            chain=self.chain,
            runtime_builder=self.runtime_builder,
            events=self.events,
            sampler=self.sampler,
            tracer=self.tracer,
            error_mode=error_mode,
        )
        self._langgraph: Any = None

    # ------------------------------------------------------------------ construction
    def _build_telemetry(self) -> Any:
        if not self.config.telemetry.enabled:
            return NoOpTelemetryProvider()
        providers: list[Any] = [OpenTelemetryTelemetryProvider(self.config.telemetry)]
        langfuse_config = self.config.observability.langfuse
        if langfuse_config.enabled:
            # Optional extra: imported only when Langfuse is switched on.
            from universal_agent_harness.langfuse.provider import (  # noqa: PLC0415
                LangfuseTelemetryProvider,
            )

            self.langfuse = LangfuseTelemetryProvider(
                langfuse_config,
                sample_rate=self.config.telemetry.sampling.sample_rate,
                strict=self.strict_observability,
            )
            providers.append(self.langfuse)
        else:
            self.langfuse = None
        if len(providers) == 1:
            return providers[0]
        # Imported here so a single-provider harness never builds the composite machinery.
        from universal_agent_harness.telemetry.composite import (  # noqa: PLC0415
            CompositeTelemetryProvider,
        )

        return CompositeTelemetryProvider(
            providers, strict=self.config.observability.failure_mode == "fail_closed"
        )

    @property
    def strict_observability(self) -> bool:
        return self.config.observability.failure_mode == "fail_closed"

    def _build_model(self, model: Any) -> Any:
        if model is None:
            return UnconfiguredModelClient()
        if hasattr(model, "invoke") and hasattr(model, "structured"):
            return model
        return DirectModelClient(model, model=self.config.models.default_model)

    def _build_tools(self, tools: Any) -> Any:
        if tools is None:
            return NoToolsClient()
        if isinstance(tools, Mapping):
            return LocalToolClient(tools)
        if isinstance(tools, list | tuple):
            client = LocalToolClient()
            for fn in tools:
                client.register(fn)
            return client
        return tools

    def _build_artifacts(self, artifacts: Any) -> Any:
        if not self.config.artifacts.enabled:
            return NoArtifactStore()
        if artifacts is None:
            return InMemoryArtifactStore()
        if isinstance(artifacts, str):
            return FileArtifactStore(artifacts)
        return artifacts

    def _build_evaluation_sink(self) -> Any:
        sinks: list[Any] = [LoggingEvaluationSink()]
        langfuse = getattr(self, "langfuse", None)
        if langfuse is not None and langfuse.client is not None:
            from universal_agent_harness.langfuse.evaluation import (  # noqa: PLC0415
                LangfuseEvaluationProvider,
                LangfuseEvaluationSink,
            )

            provider = LangfuseEvaluationProvider(langfuse.client)
            if isinstance(self.evaluation_provider, NoOpEvaluationProvider):
                self.evaluation_provider = provider
            sinks.append(LangfuseEvaluationSink(provider))
        return CompositeEvaluationSink(sinks) if len(sinks) > 1 else sinks[0]

    def _core_interceptors(self) -> list[BaseInterceptor]:
        chain: list[BaseInterceptor] = [
            IdentityInterceptor(self.version),
            TelemetryInterceptor(),
            TimeoutInterceptor(self.config.timeouts.default_seconds),
            ResultValidationInterceptor(output_schema=self._output_schema),
        ]
        if self.policy_enabled:
            chain.append(PolicyInterceptor(self.policy))
        if self.memory_factory.enabled:
            chain.append(MemoryContextInterceptor(self.events))
            if self.default_memory_policy.writes_anything:
                chain.append(
                    MemoryObservationInterceptor(
                        self.writeback,
                        fail_closed=self.config.memory.failure_mode == "fail_closed",
                    )
                )
        langfuse = getattr(self, "langfuse", None)
        if langfuse is not None:
            from universal_agent_harness.langfuse.interceptor import (  # noqa: PLC0415
                LangfuseInterceptor,
            )

            chain.append(
                LangfuseInterceptor(
                    langfuse,
                    self.config.telemetry.capture,
                    fail_closed=self.strict_observability,
                )
            )
        if self.config.evaluation_events.enabled:
            chain.append(
                EvaluationEventInterceptor(
                    self.evaluation_sink,
                    synchronous=self.config.evaluation_events.synchronous,
                    sample_rate=self.config.evaluation_events.sample_rate,
                    queue=self.writeback,
                )
            )
        return chain

    # ------------------------------------------------------------------ level 1: wrap
    def wrap(
        self,
        target: Callable[..., Any],
        *,
        agent_id: str | None = None,
        skills: list[str | SkillDescriptor] | None = None,
        version: str = "0.1.0",
        state_mapper: Callable[[AgentResponse], Any] | None = None,
        memory_policy: MemoryPolicy | dict[str, Any] | None = None,
        timeout_seconds: float | None = None,
        idempotent: bool = False,
        retry: RetryPolicy | None = None,
        error_mode: str | None = None,
        interceptors: Iterable[BaseInterceptor] = (),
        framework: str | None = None,
        framework_version: str | None = None,
        agent_group: str | None = None,
        **descriptor_fields: Any,
    ) -> Callable[..., Any]:
        """Wrap an existing agent. The wrapper keeps the target's sync/async nature.

        ``state_mapper`` maps the :class:`AgentResponse` onto whatever the caller's framework
        expects (a graph state update, a dict, the raw data) — the application keeps owning
        its own state shape (§8).
        """
        descriptor = self.describe(
            agent_id or getattr(target, "__name__", "agent"),
            skills=skills,
            version=version,
            framework=framework,
            framework_version=framework_version,
            agent_group_id=agent_group,
            **descriptor_fields,
        )
        policy = retry or RetryPolicy(self.config.retries, idempotent=idempotent)
        extra = tuple(interceptors)

        async def run(
            payload: Any = None,
            *,
            context: AgentExecutionContext | None = None,
            objective: str | None = None,
            **fields: Any,
        ) -> Any:
            if agent_group:
                fields.setdefault("agent_group_id", agent_group)
            request = self.request(
                payload,
                descriptor=descriptor,
                context=context,
                objective=objective,
                timeout_seconds=timeout_seconds,
                **fields,
            )
            result = await self.coordinator.execute(
                target,
                request,
                descriptor,
                retry=policy,
                error_mode=error_mode,
                extra_interceptors=extra,
                memory_policy=memory_policy,
            )
            return state_mapper(result) if state_mapper else result

        if _is_async_callable(target):
            wrapper: Callable[..., Any] = _copy_metadata(run, target)
        else:

            async def run_and_settle(payload: Any, **kwargs: Any) -> Any:
                """A synchronous call is synchronous to the end.

                ``run_sync`` executes on a loop that closes when it returns, so a write
                scheduled fire-and-forget during the call would race the loop's shutdown —
                sometimes landing, sometimes cancelled. Draining here makes a sync agent's
                memory writes as durable as an async agent's, at the cost of the sync caller
                waiting for them (which is what "synchronous" means).
                """
                result = await run(payload, **kwargs)
                await self.drain()
                return result

            def sync_wrapper(payload: Any = None, **kwargs: Any) -> Any:
                return run_sync(run_and_settle(payload, **kwargs))

            wrapper = _copy_metadata(sync_wrapper, target)
            wrapper.arun = run  # type: ignore[attr-defined]

        wrapper.descriptor = descriptor  # type: ignore[attr-defined]
        wrapper.harness = self  # type: ignore[attr-defined]
        wrapper.__wrapped_agent__ = target  # type: ignore[attr-defined]
        return wrapper

    # ------------------------------------------------------------------ level 2: decorator
    def agent(
        self,
        agent_id: str | None = None,
        *,
        skills: list[str | SkillDescriptor] | None = None,
        **options: Any,
    ) -> Callable[[Callable[..., Any]], Any]:
        """Decorator for a runtime-aware agent: ``async def fn(state, agent) -> ...``."""

        def decorate(fn: Callable[..., Any]) -> Any:
            return self.wrap(fn, agent_id=agent_id or fn.__name__, skills=skills, **options)

        return decorate

    # ------------------------------------------------------------------ level 3: block
    @asynccontextmanager
    async def execution(
        self,
        context: AgentExecutionContext | None = None,
        *,
        agent_id: str = "agent",
        input: Any = None,
        objective: str | None = None,
        skills: list[str | SkillDescriptor] | None = None,
        **fields: Any,
    ) -> AsyncIterator[AgentRuntime]:
        """Instrument a block of code you call yourself.

        Everything the pipeline does around a wrapped agent happens here too: the span, the
        memory context, the deadline, the observations. What it cannot do is inspect the
        code inside the block — a plain call to an un-wrapped library stays un-instrumented,
        and the harness says so rather than implying otherwise (§13).
        """
        descriptor = self.describe(agent_id, skills=skills)
        request = self.request(
            input, descriptor=descriptor, context=context, objective=objective, **fields
        )
        decision = self.sampler.decide(
            agent_id=request.context.agent_id, run_id=request.context.agent_run_id
        )
        tracer = self.tracer.for_decision(decision)
        runtime = self.runtime_builder.build(request.context, descriptor, tracer=tracer)
        with tracer.agent_span(request.context, skills=descriptor.skill_ids or None) as span:
            runtime.state.update({"span": span, "request": request, "sampling": decision})
            with bind(request.context, runtime):
                self.events.emit(
                    LifecycleEvent.AGENT_START,
                    {"context": request.context, "descriptor": descriptor, "request": request},
                )
                prepared = await self.chain.before(request, runtime)
                runtime.state["request"] = prepared
                try:
                    yield runtime
                except BaseException as exc:
                    error = AgentError.of(exc, trace_id=request.context.trace_id)
                    await self.chain.on_error(error, runtime)
                    self.events.emit(
                        LifecycleEvent.AGENT_ERROR, {"context": request.context, "error": error}
                    )
                    self.events.emit(
                        LifecycleEvent.AGENT_FINISH,
                        {"context": request.context, "status": "ERROR", "error": error},
                    )
                    raise
                # The block reports what it produced through ``runtime.state["result"]``;
                # with nothing set, the execution is recorded as a bare success.
                result = AgentResponse.coerce(runtime.state.get("result"))
                result = await self.chain.after(result, runtime)
                if result.succeeded:
                    self.events.emit(
                        LifecycleEvent.AGENT_SUCCESS,
                        {"context": request.context, "result": result},
                    )
                self.events.emit(
                    LifecycleEvent.AGENT_FINISH,
                    {"context": request.context, "status": str(result.status), "result": result},
                )

    # ------------------------------------------------------------------ direct run
    async def run(
        self,
        target: Callable[..., Any],
        payload: Any = None,
        *,
        agent_id: str | None = None,
        context: AgentExecutionContext | None = None,
        **options: Any,
    ) -> AgentResponse:
        """One-shot execution without keeping a wrapper around."""
        wrapper_options = {k: v for k, v in options.items() if k in _WRAP_OPTIONS}
        call_fields = {k: v for k, v in options.items() if k not in _WRAP_OPTIONS}
        wrapped = self.wrap(target, agent_id=agent_id, **wrapper_options)
        runner = getattr(wrapped, "arun", wrapped)
        return await runner(payload, context=context, **call_fields)

    def run_sync(self, target: Callable[..., Any], payload: Any = None, **options: Any) -> Any:
        """Synchronous convenience. Never nests ``asyncio.run`` inside a running loop (§72)."""
        return run_sync(self.run(target, payload, **options))

    # ------------------------------------------------------------------ helpers
    async def _output_schema(self, agent_id: str) -> dict[str, Any] | None:
        """The output schema the registry holds for ``agent_id``, if any.

        A bound method rather than a lambda so the interceptor keeps working when the
        registry is swapped after construction, and so an unbound deployment answers
        ``None`` without the interceptor needing to know a registry exists.
        """
        resolver = getattr(self.registry, "output_schema", None)
        if resolver is None:
            return None
        return await resolver(agent_id)

    def _build_runs(self) -> Any:
        """The durable-runs client when the deployment records to one, else a no-op.

        Partial configuration is treated as unconfigured, for the same reason as the
        registry: a URL with no key fails on the first turn, at startup, reading as an
        outage rather than as the missing setting it is.
        """
        settings = self.config.runs
        if not settings.configured:
            return NoRunStore()
        from universal_agent_harness.runs import RunStoreClient  # noqa: PLC0415 - optional

        return RunStoreClient(
            str(settings.url), api_key=str(settings.api_key), required=settings.required
        )

    def _build_registry(self) -> Any:
        """The AI Registry client when the deployment is bound to one, else a no-op.

        Built from configuration rather than asked for in code: a deployment that has
        already told the harness its registry URL, product and key should not have to
        construct a client as well. Partial configuration is treated as *not configured* —
        a URL without a key would otherwise fail on the first call, at startup, in a way
        that reads as an outage rather than a missing setting.
        """
        settings = self.config.registry
        if not settings.configured:
            return NoOpAgentRegistry()
        from universal_agent_harness.registry.ai_registry import (  # noqa: PLC0415 - optional
            AIRegistryClient,
        )

        return AIRegistryClient(
            str(settings.url),
            product_key=str(settings.product_key),
            api_key=str(settings.api_key),
        )

    def qualify(self, agent_id: str) -> str:
        """``agent_id`` as the rest of the world must see it.

        Agents are declared bare — ``@harness.agent(agent_id="refund-agent")`` — because the
        product a deployment serves is configuration, not source. When ``product_key`` is
        set, the id that leaves this process becomes ``{product_key}:{agent_id}``, which is
        what keeps two teams' identically-named agents from sharing a memory scope inside
        one tenant.

        ``:`` rather than ``/``: ``safe_id`` keeps ``:`` and rewrites ``/`` to a hyphen, so
        ``billing/refund-agent`` would arrive downstream as ``billing-refund-agent`` —
        indistinguishable from product ``billing-refund``'s agent ``agent``.

        An id that already carries a prefix is left alone, so a process serving several
        products can name them explicitly.
        """
        product = self.config.registry.product_key
        if not product or ":" in agent_id:
            return agent_id
        return f"{product}:{agent_id}"

    def describe(
        self,
        agent_id: str,
        *,
        skills: list[str | SkillDescriptor] | None = None,
        version: str = "0.1.0",
        **fields: Any,
    ) -> AgentDescriptor:
        """Build (and remember) an agent descriptor, for the registry hook and telemetry."""
        descriptor = AgentDescriptor.build(
            self.qualify(agent_id),
            skills=skills,
            version=version,
            harness_version=self.version,
            **{k: v for k, v in fields.items() if v is not None},
        )
        self.descriptors[descriptor.agent_id] = descriptor
        return descriptor

    def request(
        self,
        payload: Any = None,
        *,
        descriptor: AgentDescriptor,
        context: AgentExecutionContext | None = None,
        objective: str | None = None,
        timeout_seconds: float | None = None,
        **fields: Any,
    ) -> AgentRequest:
        """Build the :class:`AgentRequest` for one call, resolving the execution context."""
        request_fields = {
            k: fields.pop(k)
            for k in (
                "skills_requested",
                "constraints",
                "artifact_refs",
                "evidence_refs",
                "metadata",
            )
            if k in fields
        }
        ctx = self.context_factory.build(
            agent_id=descriptor.agent_id,
            context=context,
            overrides=fields,
            timeout_seconds=timeout_seconds,
        )
        if timeout_seconds is not None:
            request_fields.setdefault("constraints", {})["timeout_seconds"] = timeout_seconds
        return AgentRequest.create(
            ctx,
            payload,
            objective=objective,
            **request_fields,
        )

    def wrap_tool(self, fn: Callable[..., Any] | None = None, /, **options: Any) -> Any:
        """Instrument a tool the developer calls directly (§12)."""
        return _wrap_tool(fn, **options) if fn is not None else _wrap_tool(**options)

    def wrap_model(self, client: Any, **options: Any) -> DirectModelClient:
        """Adapt an existing model client/callable to the instrumented port (§17)."""
        return DirectModelClient(client, **options)

    def register_tool(self, fn: Callable[..., Any], **spec_fields: Any) -> Any:
        """Add a tool to the harness's local tool runtime."""
        if not isinstance(self.tool_client, LocalToolClient):
            self.tool_client = LocalToolClient()
            self.runtime_builder.tool_client = self.tool_client
        return self.tool_client.register(fn, **spec_fields)

    def on(self, listener: Any) -> Any:
        """Register a lifecycle listener: ``listener(event_name, payload)``."""
        self.events.add(listener)
        return listener

    def add_interceptor(self, interceptor: BaseInterceptor) -> None:
        """Add an interceptor after construction. Ordering stays deterministic (§19)."""
        self.chain = self.chain.with_extra([interceptor])
        self.coordinator.chain = self.chain

    @property
    def current_context(self) -> AgentExecutionContext | None:
        return current_context()

    @property
    def langgraph(self) -> Any:
        """The LangGraph adapter. Importing it requires the ``langgraph`` extra (§2)."""
        if self._langgraph is None:
            try:
                from universal_agent_harness_langgraph import LangGraphHarness  # noqa: PLC0415
            except ImportError as exc:  # pragma: no cover - documented degradation
                raise ImportError(
                    "LangGraph support needs the adapter: "
                    'pip install "universal-agent-harness[langgraph]"'
                ) from exc
            self._langgraph = LangGraphHarness(self)
        return self._langgraph

    async def register_agents(self) -> None:
        """Push known descriptors to the registry hook (a no-op by default, §47)."""
        if not self.registry_enabled:
            return
        for descriptor in self.descriptors.values():
            await self.registry.register(descriptor)

    async def drain(self, timeout: float | None = 30.0) -> int:  # noqa: ASYNC109
        """Await outstanding memory/evaluation writebacks and run records.

        Run records are drained too, and before the count is returned: a process that exits
        without this may have opened a run it never closed, leaving a turn that finished
        looking like one still in flight.
        """
        recorder = getattr(self, "_recorder", None)
        if recorder is not None:
            await recorder.drain()
        return await self.writeback.drain(timeout)

    def flush(self, timeout_seconds: float = 5.0) -> None:
        """Flush telemetry exporters."""
        self.tracer.flush(timeout_seconds)

    async def aclose(self) -> None:
        await self.drain()
        self.flush()
        langfuse = getattr(self, "langfuse", None)
        if langfuse is not None:
            langfuse.shutdown()

    async def __aenter__(self) -> AgentHarness:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()


def _is_async_callable(target: Any) -> bool:
    """True for ``async def`` functions *and* objects whose ``__call__`` is async.

    Getting this wrong turns an async agent into a blocking one, so it is checked on the
    object actually invoked, not just on the outer reference.
    """
    if inspect.iscoroutinefunction(target):
        return True
    if not callable(target):
        return True
    call = getattr(type(target), "__call__", None)  # noqa: B004 - fetching, not testing
    return call is not None and inspect.iscoroutinefunction(call)


def _copy_metadata(wrapper: Callable[..., Any], target: Any) -> Callable[..., Any]:
    """``functools.wraps`` where possible; callables without ``__name__`` are common
    (partials, callable objects) and must not break wrapping."""
    try:
        return functools.wraps(target)(wrapper)
    except (AttributeError, TypeError):
        wrapper.__name__ = getattr(target, "__name__", type(target).__name__)
        return wrapper


#: Keyword arguments of :meth:`AgentHarness.wrap` (so ``run`` can split them from call fields).
_WRAP_OPTIONS = frozenset(
    {
        "skills",
        "version",
        "state_mapper",
        "memory_policy",
        "timeout_seconds",
        "idempotent",
        "retry",
        "error_mode",
        "interceptors",
        "framework",
        "framework_version",
    }
)
