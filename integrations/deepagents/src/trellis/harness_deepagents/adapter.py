"""``DeepAgentsHarness``: the harness around a Deep Agents agent (design §8).

Two ways in, both giving the same six bindings:

    # 1. an agent you already have: add the middleware, the backend and the model
    agent = create_deep_agent(
        model=harness.deepagents.model(),
        tools=[search],
        backend=harness.deepagents.backend(),
        middleware=[harness.deepagents.middleware()],
    )
    run = harness.deepagents.wrap(agent, agent_id="researcher", query="question")

    # 2. let the adapter assemble it
    run = harness.deepagents.agent(
        agent_id="researcher", tools=[search], system_prompt="...", query="question"
    )

    answer = await run({"messages": [{"role": "user", "content": "how much stock?"}]},
                      context=context)

What it does not do: own the graph, the subagents, the skills, the todo list, the
checkpointer or the state schema. ``interrupt_on=`` keeps working — Deep Agents' own
human-in-the-loop raises a ``GraphInterrupt``, which the harness already reads structurally
and turns into the same :class:`Interrupt` a policy ``require_approval`` produces.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from trellis.contracts.messages import AgentResponse

from trellis.harness.reasoning.assembler import DEFAULT_BUDGET_TOKENS
from trellis.harness_deepagents.backend import MemoryServiceBackend
from trellis.harness_deepagents.middleware import FRAMEWORK, TrellisMiddleware
from trellis.harness_deepagents.models import BifrostChatModel

__all__ = ["DeepAgentsHarness", "deepagents_version"]

#: Query extractors: a state key, or a callable taking the input state.
Query = str | Callable[[Any], str | None]


def deepagents_version() -> str | None:
    """The installed Deep Agents version, for the compatibility matrix and span attributes."""
    try:
        from importlib.metadata import version  # noqa: PLC0415

        return version("deepagents")
    except Exception:  # pragma: no cover - deepagents not installed
        return None


class DeepAgentsHarness:
    """Framework adapter. With :mod:`trellis.harness_deepagents.backend`, the only place
    this project imports Deep Agents."""

    name = FRAMEWORK

    def __init__(self, harness: Any) -> None:
        self.harness = harness
        self.version = deepagents_version()

    # ------------------------------------------------------------------ capability check
    @staticmethod
    def available() -> bool:
        try:
            import deepagents  # noqa: F401, PLC0415 - capability probe
        except ImportError:
            return False
        return True

    def supports(self, target: object) -> bool:
        """Whether this adapter can wrap ``target``: anything invocable as a graph."""
        return callable(getattr(target, "ainvoke", None)) or callable(target)

    # ------------------------------------------------------------------ the bindings
    def model(
        self,
        *,
        client: Any = None,
        model: str | None = None,
        params: Mapping[str, Any] | None = None,
    ) -> BifrostChatModel:
        """The chat model Deep Agents should use: the harness's, over the gateway.

        With no ``client`` it resolves the running execution's instrumented ``runtime.model``
        per call, so one compiled agent serves every run and every call is still traced,
        metered, policy-checked and deadline-bounded.
        """
        return BifrostChatModel(
            client=client,
            model_name=model or self.harness.config.models.default_model,
            params=dict(params or {}),
        )

    def backend(self, *, memory: Any = None, visibility: str = "USER") -> MemoryServiceBackend:
        """``/memories/*`` served by the Memory Service instead of a local store."""
        return MemoryServiceBackend(memory, visibility=visibility)

    def middleware(
        self,
        *,
        runtime: Any = None,
        model: Any = None,
        skills: Sequence[Any] = (),
        budget_tokens: int = DEFAULT_BUDGET_TOKENS,
        memory_tokens: int | None = None,
        inject_context: bool = True,
        record_to_memory: bool = True,
    ) -> TrellisMiddleware:
        """The middleware that binds context, model, tools, pause, compaction and run end."""
        return TrellisMiddleware(
            runtime,
            policy=self.harness.policy if self.harness.policy_enabled else None,
            model=model,
            skills=skills,
            budget_tokens=budget_tokens,
            memory_tokens=memory_tokens,
            inject_context=inject_context,
            record_to_memory=record_to_memory,
        )

    # ------------------------------------------------------------------ wrapping
    def wrap(
        self,
        agent: Any,
        *,
        agent_id: str | None = None,
        query: Query | None = None,
        skills: list[Any] | None = None,
        config: Mapping[str, Any] | None = None,
        state_mapper: Callable[[AgentResponse], Any] | None = None,
        **options: Any,
    ) -> Callable[..., Any]:
        """Run a compiled Deep Agents agent through the harness pipeline.

        ``query`` names what memory retrieval should search for and what the turn is recorded
        as having been asked — a key of the input state, or a callable over it. Without one
        the last human message in ``messages`` is used, because for a chat-shaped agent that
        *is* the question; ``query=lambda _: None`` opts out.
        """
        name = agent_id or getattr(agent, "name", None) or "deepagents-agent"
        invoke = getattr(agent, "ainvoke", None) or agent
        run_config = dict(config or {})

        async def target(payload: Any, runtime: Any) -> Any:
            merged = _merge_config(run_config, payload)
            state = payload.get("state", payload) if isinstance(payload, dict) else payload
            result = await invoke(state, merged) if merged else await invoke(state)
            runtime.logger.debug("deepagents.finished", agent_id=name)
            return result

        wrapped = self.harness.wrap(
            target,
            agent_id=name,
            skills=skills,
            framework=FRAMEWORK,
            framework_version=self.version,
            state_mapper=state_mapper,
            **options,
        )
        runner = getattr(wrapped, "arun", wrapped)

        async def run(payload: Any = None, **fields: Any) -> Any:
            fields.setdefault("objective", _query_of(query, payload))
            return await runner(payload, **fields)

        run.harness = self.harness  # type: ignore[attr-defined]
        run.descriptor = wrapped.descriptor  # type: ignore[attr-defined]
        run.arun = run  # type: ignore[attr-defined]
        run.agent = agent  # type: ignore[attr-defined]
        return run

    def agent(
        self,
        *,
        agent_id: str,
        tools: Sequence[Any] = (),
        system_prompt: str | None = None,
        model: str | None = None,
        middleware: Sequence[Any] = (),
        backend: Any = None,
        interrupt_on: Mapping[str, Any] | None = None,
        skills: list[Any] | None = None,
        query: Query | None = None,
        config: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> Callable[..., Any]:
        """Build a Deep Agents agent with the harness already bound, and wrap it.

        The harness middleware goes **first** so it is the outermost layer: it sees the model
        request after every other middleware has shaped it (which is what makes the
        compaction summary visible to it) and it authorizes a tool call before any other
        middleware can run one.
        """
        from deepagents import create_deep_agent  # noqa: PLC0415 - the adapter's own import

        chat_model = self.model(model=model)
        compiled = create_deep_agent(
            model=chat_model,
            tools=list(tools),
            system_prompt=system_prompt,
            middleware=[self.middleware(), *middleware],
            backend=backend if backend is not None else self.backend(),
            **({"interrupt_on": dict(interrupt_on)} if interrupt_on else {}),
            **kwargs,
        )
        return self.wrap(compiled, agent_id=agent_id, query=query, skills=skills, config=config)


def _merge_config(base: Mapping[str, Any], payload: Any) -> dict[str, Any]:
    """The graph config for this turn: the adapter's, plus a ``config`` in the payload.

    A caller resuming a Deep Agents human-in-the-loop passes its own ``configurable``
    (a ``thread_id`` for the checkpointer); it must not be dropped.
    """
    merged = dict(base)
    extra = payload.get("config") if isinstance(payload, dict) else None
    if isinstance(extra, Mapping):
        configurable = {
            **(merged.get("configurable") or {}),
            **(extra.get("configurable") or {}),
        }
        merged = {**merged, **extra}
        if configurable:
            merged["configurable"] = configurable
    return merged


def _query_of(query: Query | None, payload: Any) -> str | None:
    """What this turn is about, for retrieval and for the recorded question."""
    if callable(query):
        return query(payload)
    state = payload.get("state", payload) if isinstance(payload, dict) else payload
    if isinstance(query, str):
        value = state.get(query) if isinstance(state, Mapping) else getattr(state, query, None)
        return value if isinstance(value, str) else None
    return _last_human(state)


def _last_human(state: Any) -> str | None:
    messages = (
        state.get("messages") if isinstance(state, Mapping) else getattr(state, "messages", None)
    )
    for message in reversed(list(messages or ())):
        role = message.get("role") if isinstance(message, Mapping) else getattr(message, "type", "")
        if role in ("user", "human"):
            content = (
                message.get("content")
                if isinstance(message, Mapping)
                else getattr(message, "content", None)
            )
            if isinstance(content, str) and content.strip():
                return content.strip()
    return None
