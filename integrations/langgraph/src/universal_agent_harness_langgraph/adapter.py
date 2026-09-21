"""``LangGraphHarness``: the harness around LangGraph nodes (§51-§55).

    harness = AgentHarness(memory=memory_client)
    graph.add_node("inventory", harness.langgraph.wrap_node(existing_node,
                                                            agent_id="inventory-agent"))

What the adapter does, and only this:

* derive identity from the ``RunnableConfig`` (thread, subgraph lineage, step) — public keys;
* build a wrapper whose signature declares the injectables the node asked for;
* run the node through the harness pipeline;
* map the :class:`AgentResponse` back to a **state update the graph already understands**.

What it deliberately does not do: own the graph topology, the routing, the reducers, the
checkpoint backend or the state schema. A wrapped node returns exactly what the original
node returned unless the caller asks for a different mapping (§52, §54).
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from typing import Any

from universal_agent_contracts.context import AgentExecutionContext
from universal_agent_contracts.messages import AgentResponse
from universal_agent_contracts.tool import ToolSpec

from universal_agent_harness_langgraph.lineage import Lineage, context_fields, lineage_from_config
from universal_agent_harness_langgraph.signature import (
    build_node,
    declared_injectables,
    positional_arity,
)

__all__ = ["LangGraphHarness", "langgraph_version"]

#: Query extractors: a state key, or a callable taking the state.
Query = str | Callable[[Any], str | None]
#: Result mappers: ``AgentResponse -> state update``.
StateMapper = Callable[[AgentResponse], Any]


def langgraph_version() -> str | None:
    """The installed LangGraph version, for the compatibility matrix and span attributes."""
    try:
        from importlib.metadata import version  # noqa: PLC0415

        return version("langgraph")
    except Exception:  # pragma: no cover - langgraph not installed
        return None


class LangGraphHarness:
    """Framework adapter. The only module in the project that imports LangGraph."""

    name = "langgraph"

    def __init__(self, harness: Any, *, thread_prefix: str = "") -> None:
        self.harness = harness
        self.thread_prefix = thread_prefix
        self.version = langgraph_version()

    # ------------------------------------------------------------------ capability check
    def supports(self, target: object) -> bool:
        """Whether this adapter can wrap ``target``: any callable is a candidate node."""
        return callable(target)

    @staticmethod
    def available() -> bool:
        try:
            import langgraph  # noqa: F401, PLC0415 - capability probe
        except ImportError:
            return False
        return True

    # ------------------------------------------------------------------ wrapping
    def wrap_node(
        self,
        node: Callable[..., Any],
        *,
        agent_id: str | None = None,
        skills: list[str] | None = None,
        query: Query | None = None,
        state_mapper: StateMapper | None = None,
        objective: str | None = None,
        **options: Any,
    ) -> Callable[..., Any]:
        """Wrap an existing node. Its semantics are unchanged: same arguments in, same state
        update out, same exceptions (§52).

        ``query`` names what the memory retrieval should search for — a state key or a
        callable. Without it, memory retrieval is skipped for this node rather than guessing.
        """
        name = agent_id or getattr(node, "__name__", "node")
        runtime_aware = positional_arity(node) >= 2
        wrapped_agent = self.harness.wrap(
            _as_agent(node, runtime_aware=runtime_aware),
            agent_id=name,
            skills=skills,
            framework="langgraph",
            framework_version=self.version,
            **options,
        )
        runner = getattr(wrapped_agent, "arun", wrapped_agent)
        mapper = state_mapper or _default_mapper

        async def impl(state: Any, injected: dict[str, Any]) -> Any:
            config = injected.get("config")
            context = self.extract_context(config, agent_id=name)
            payload = _NodeCall(state=state, injected=injected, node=node)
            result = await runner(
                payload,
                context=context,
                objective=objective or _query_of(query, state),
            )
            return mapper(result)

        return build_node(
            impl,
            name=name,
            injectables=declared_injectables(node) or ("config",),
            is_async=True,
        )

    def agent(
        self,
        agent_id: str | None = None,
        *,
        skills: list[str] | None = None,
        **options: Any,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator for a runtime-aware node: ``async def node(state, agent)``.

        The second argument is the :class:`AgentRuntime`; ``agent.model``, ``agent.tools``
        and ``agent.memory`` are instrumented.
        """

        def decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
            return self.wrap_node(fn, agent_id=agent_id or fn.__name__, skills=skills, **options)

        return decorate

    def wrap_tool(self, fn: Callable[..., Any] | None = None, /, **options: Any) -> Any:
        """Instrument a tool used inside the graph (including inside a ``ToolNode``)."""
        return self.harness.wrap_tool(fn, **options)

    # ------------------------------------------------------------------ adapter contract
    def extract_context(
        self, config: Mapping[str, Any] | None, *, agent_id: str
    ) -> AgentExecutionContext | None:
        """Build the execution context for this graph position, or ``None`` to let the
        harness fall back to its defaults / the ambient context."""
        lineage = lineage_from_config(config)
        fields = context_fields(lineage, thread_prefix=self.thread_prefix)
        overrides = dict(lineage.overrides)
        tenant_id = overrides.get("tenant_id") or self.harness.defaults.get("tenant_id")
        if not tenant_id:
            return None
        fields.pop("agent_id", None)
        explicit_run = overrides.get("agent_run_id")
        return AgentExecutionContext.create(
            tenant_id=tenant_id,
            agent_id=agent_id,
            agent_run_id=explicit_run or lineage.run_id_for(agent_id),
            **{
                **{k: v for k, v in self.harness.defaults.items() if k != "tenant_id"},
                **{k: v for k, v in fields.items() if k not in ("tenant_id", "agent_run_id")},
            },
        ).with_fields(parent_agent_run_id=lineage.parent_run_id())

    def map_result(self, result: AgentResponse, mapper: StateMapper | None = None) -> Any:
        return (mapper or _default_mapper)(result)

    def lineage(self, config: Mapping[str, Any] | None) -> Lineage:
        """Exposed for applications that want the parsed position themselves."""
        return lineage_from_config(config)

    def tool_specs(self, tools: list[Any]) -> list[ToolSpec]:
        """Describe LangChain-style tools (``name``/``description``/``args_schema``) for the
        harness's tool contracts, without importing langchain."""
        specs: list[ToolSpec] = []
        for tool in tools:
            name = getattr(tool, "name", None) or getattr(tool, "__name__", None)
            if not name:
                continue
            specs.append(
                ToolSpec(
                    name=name,
                    description=getattr(tool, "description", "") or "",
                    source="langgraph",
                )
            )
        return specs


class _NodeCall:
    """The payload handed to the wrapped node: the graph state plus its injectables."""

    __slots__ = ("injected", "node", "state")

    def __init__(self, state: Any, injected: dict[str, Any], node: Callable[..., Any]) -> None:
        self.state = state
        self.injected = injected
        self.node = node

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<NodeCall state_keys={_state_keys(self.state)}>"


def _as_agent(node: Callable[..., Any], *, runtime_aware: bool) -> Callable[..., Any]:
    """Adapt a LangGraph node into the ``(payload, runtime)`` shape the harness calls."""

    async def agent(payload: _NodeCall, runtime: Any) -> Any:
        kwargs = {k: v for k, v in payload.injected.items() if k in declared_injectables(node)}
        args: tuple[Any, ...] = (payload.state, runtime) if runtime_aware else (payload.state,)
        value = node(*args, **kwargs)
        if inspect.isawaitable(value):
            return await value
        return value

    return agent


def _default_mapper(result: AgentResponse) -> Any:
    """Return exactly what the node returned (§52). The harness's own metadata stays on the
    result object, which callers can opt into with an explicit ``state_mapper``."""
    return result.data


def _query_of(query: Query | None, state: Any) -> str | None:
    if query is None:
        return None
    if callable(query):
        return query(state)
    value = state.get(query) if isinstance(state, Mapping) else getattr(state, query, None)
    if isinstance(value, str):
        return value
    if isinstance(value, list | tuple) and value:
        last = value[-1]
        content = getattr(last, "content", None)
        if isinstance(content, str):
            return content
        if isinstance(last, Mapping):
            text = last.get("content")
            return text if isinstance(text, str) else None
    return None


def _state_keys(state: Any) -> Any:
    return sorted(state) if isinstance(state, Mapping) else type(state).__name__
