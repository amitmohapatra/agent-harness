"""Where a model *name* is served — model routing, one strategy (:class:`ModelGateway`) with
one adapter per route, each declaring what it can do (:class:`Capability`). Exactly one serves a
deployment, never a chain of fallbacks:

1. :class:`Bifrost` — ``BIFROST_URL`` (and ``BIFROST_VIRTUAL_KEY``): every model call of every
   framework goes through it (it wins over a provider key: said once, and by ``doctor``), and
   everything else the harness uses the gateway for;
2. :class:`ProviderKey` — otherwise: the provider's own SDK, reading its own environment
   (``OPENAI_API_KEY``, ``OPENAI_BASE_URL``, ``ANTHROPIC_API_KEY``...; never read here): models
   only.

A model name is mapped to the route's form (:func:`split`): ``gpt-4o`` and ``openai:gpt-4o``
are ``openai/gpt-4o`` on Bifrost, and ``openai/gpt-4o`` is ``openai:gpt-4o`` for LangChain's
``init_chat_model``. A feature that needs a capability the deployment's route lacks is off —
``trellis doctor`` lists them — and one asked for explicitly says why it cannot be
(:func:`missing`)."""

from __future__ import annotations

import functools
import logging
import weakref
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Mapping
from enum import StrEnum
from typing import TYPE_CHECKING, Any, ClassVar, Final, Self

from trellis.contracts import ConfigurationError
from trellis.harness.bounded import LRU

if TYPE_CHECKING:
    from trellis.harness.settings import Settings

log = logging.getLogger("trellis.harness")


class Capability(StrEnum):
    MODELS = "models"
    MCP = "mcp"
    CODE_MODE = "code_mode"
    COST = "cost"
    FALLBACKS = "fallbacks"
    PROMPTS = "prompts"
    ATTRIBUTION = "attribution"


class GatewayKind(StrEnum):
    BIFROST = "bifrost"
    PROVIDER_KEY = "provider_key"


#: What each capability is, as ``doctor`` says it when it is off.
CAPABILITIES: Final[dict[Capability, str]] = {
    Capability.MODELS: "model names (ReAct(model='name'), llm_judge(model='name'))",
    Capability.MCP: "the gateway's MCP tools (the ones the virtual key allows)",
    Capability.CODE_MODE: "Code Mode",
    Capability.COST: "dollar budgets (Budget(max_cost=)) from the gateway's cost of each call",
    Capability.FALLBACKS: "model fallbacks and retries in the gateway",
    Capability.PROMPTS: "stored prompts (stored('name'))",
    Capability.ATTRIBUTION: "each call's run, agent, tenant and user as the gateway's dimensions",
}
#: The provider a model name is of, by how the name starts, for a name that does not say
#: (``gpt-4o``): the route's provider id (Bifrost's; LangChain's is :data:`LANGCHAIN_PROVIDERS`').
PROVIDER_PREFIXES: Final = (
    ("gpt-", "openai"),
    ("chatgpt-", "openai"),
    ("o1", "openai"),
    ("o3", "openai"),
    ("o4", "openai"),
    ("claude-", "anthropic"),
    ("gemini-", "gemini"),
    ("mistral-", "mistral"),
    ("codestral-", "mistral"),
    ("command", "cohere"),
)
#: A provider id as LangChain's ``init_chat_model`` names it, where it differs from Bifrost's.
LANGCHAIN_PROVIDERS: Final = {"gemini": "google_genai", "mistral": "mistralai"}
#: Where the provider keys live that a provider SDK reads itself (``doctor`` says which are
#: set; the harness reads none of them).
PROVIDER_KEYS: Final = (
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "MISTRAL_API_KEY",
    "COHERE_API_KEY",
    "GROQ_API_KEY",
)
#: The ``Authorization`` a gateway without a key is sent (the client needs one).
NO_KEY: Final = "no-key"
#: Retries of a model call Bifrost's client makes itself: none — the gateway retries (a
#: provider's ``network_config.max_retries``) and falls back, the one place that does.
BIFROST_CLIENT_RETRIES: Final = 0
#: How many chat models made of routed names are kept (per route, name, effort, event loop).
ROUTED_KEPT: Final = 64


class ModelGateway(ABC):
    """The port: a gateway that serves model names, and what it can do."""

    kind: ClassVar[GatewayKind]
    #: how ``doctor`` and the start-up line name it
    title: ClassVar[str]
    capabilities: ClassVar[frozenset[Capability]]

    def __init__(self, url: str, source: str) -> None:
        self.url = url
        #: the variables that named it
        self.source = source

    def has(self, capability: Capability) -> bool:
        return capability in self.capabilities

    @property
    @abstractmethod
    def probe_key(self) -> str:
        """The key ``trellis doctor`` asks the gateway's models with."""

    @classmethod
    @abstractmethod
    def detect(cls, settings: Settings) -> Self | None:
        """This gateway, when the deployment's configuration names it."""

    @abstractmethod
    def chat_model(self, model: str, *, effort: str | None) -> Any:
        """A LangChain chat model of ``model`` on this gateway. It has no timeout of its own:
        a model call's bound is the run's (``ModelHooks``, the stall watchdog), applied once."""

    @abstractmethod
    def routed(self, model: str) -> str:
        """``model`` as this route names it (:func:`split`)."""


def split(model: str) -> tuple[str | None, str]:
    """A model name's provider and its model: ``openai/gpt-4o``, ``openai:gpt-4o`` and
    ``gpt-4o`` (by :data:`PROVIDER_PREFIXES`) are ``("openai", "gpt-4o")``; a name whose
    provider cannot be told is ``(None, name)``."""
    for mark in ("/", ":"):
        provider, found, rest = model.partition(mark)
        if found and provider and rest:
            return provider, rest
    for prefix, provider in PROVIDER_PREFIXES:
        if model.startswith(prefix):
            return provider, model
    return None, model


def _chat(model: str, effort: str | None, **more: Any) -> Any:
    import openai  # noqa: PLC0415 - the react extra
    from langchain_openai import ChatOpenAI  # noqa: PLC0415

    return ChatOpenAI(
        model=model,
        stream_usage=True,
        reasoning_effort=effort,
        # a client of its own: langchain-openai shares one per gateway URL across the process,
        # whose pooled connections break the models of a later event loop
        http_async_client=openai.DefaultAsyncHttpxClient(),
        **more,
    )


class Bifrost(ModelGateway):
    """Bifrost: model names on its chat completions with the deny-all MCP scope (the harness
    lists and runs the MCP tools itself), its retries and fallbacks, every capability."""

    kind = GatewayKind.BIFROST
    title = "Bifrost"
    capabilities = frozenset(Capability)

    def __init__(self, url: str, key: str | None) -> None:
        super().__init__(url, "BIFROST_URL")
        self.key = key

    @property
    def probe_key(self) -> str:
        return self.key or NO_KEY

    @classmethod
    def detect(cls, settings: Settings) -> Self | None:
        if not settings.bifrost_url:
            return None
        return cls(settings.bifrost_url, settings.bifrost_virtual_key)

    @property
    def origin(self) -> str:
        """The gateway's root (``BIFROST_URL`` less its ``/v1``): its other routes hang there
        (``/anthropic``)."""
        return self.url.rstrip("/").removesuffix("/v1")

    def routed(self, model: str) -> str:
        provider, name = split(model)
        return name if provider is None else f"{provider}/{name}"

    def chat_model(self, model: str, *, effort: str | None) -> Any:
        from bifrost_sdk import NO_GATEWAY_TOOLS  # noqa: PLC0415

        return _chat(
            self.routed(model),
            effort,
            base_url=self.url,
            api_key=self.key or NO_KEY,
            default_headers=dict(NO_GATEWAY_TOOLS),
            max_retries=BIFROST_CLIENT_RETRIES,
        )


class ProviderKey(ModelGateway):
    """No gateway: a model name goes to its provider, through the provider's own SDK, which
    reads its own key and endpoint from the environment (``OPENAI_API_KEY``,
    ``OPENAI_BASE_URL`` for any OpenAI-compatible endpoint, ``ANTHROPIC_API_KEY``...): models
    only, the client's own retries (nothing else retries there)."""

    kind = GatewayKind.PROVIDER_KEY
    title = "provider key"
    capabilities = frozenset({Capability.MODELS})

    def __init__(self) -> None:
        super().__init__("the provider's own endpoint", "no BIFROST_URL")

    @property
    def probe_key(self) -> str:
        return NO_KEY

    @classmethod
    def detect(cls, settings: Settings) -> Self | None:
        return cls()

    def routed(self, model: str) -> str:
        provider, name = split(model)
        if provider is None:
            raise ConfigurationError(
                f"model {model!r}: name its provider ('openai:{model}', 'anthropic:{model}'), "
                "whose key its SDK reads (OPENAI_API_KEY, ANTHROPIC_API_KEY...), or set "
                "BIFROST_URL to route every model through Bifrost"
            )
        return f"{LANGCHAIN_PROVIDERS.get(provider, provider)}:{name}"

    def chat_model(self, model: str, *, effort: str | None) -> Any:
        provider, _, name = self.routed(model).partition(":")
        if provider == "openai":  # the client the harness tunes (usage on streams, effort)
            return _chat(name, effort)
        from langchain.chat_models import init_chat_model  # noqa: PLC0415

        return init_chat_model(name, model_provider=provider)

    def judge(self, model: str) -> Any:
        """``model`` as a judge asks it: an OpenAI model on the OpenAI SDK's own endpoint
        (Bifrost's judges go through the harness's own client, with the judge's key:
        ``EvalServices.judge_gateway``)."""
        from trellis.harness.evals import OpenAICompatibleJudge  # noqa: PLC0415

        provider, _, name = self.routed(model).partition(":")
        if provider != "openai":
            raise ConfigurationError(
                f"a judge by name without BIFROST_URL is an OpenAI model on the OpenAI SDK's "
                f"endpoint (OPENAI_API_KEY, OPENAI_BASE_URL), not {model!r}: set BIFROST_URL, or "
                "give llm_judge a model object"
            )
        return OpenAICompatibleJudge(name)


#: The routes, in the order a deployment's configuration is read: the first it names serves.
GATEWAYS: Final[tuple[type[ModelGateway], ...]] = (Bifrost, ProviderKey)


def model_gateway(settings: Settings) -> ModelGateway:
    """The route a model name goes by: Bifrost when the settings name it, else the provider's
    own SDK (:class:`ProviderKey`)."""
    for gateway in GATEWAYS:
        found = gateway.detect(settings)
        if found is not None:
            return found
    raise AssertionError("ProviderKey always serves")  # pragma: no cover


@functools.cache  # said once per process
def _bifrost_wins(keys: tuple[str, ...]) -> None:
    log.info(
        "model calls go through Bifrost (BIFROST_URL), not to the providers with %s: Bifrost "
        "wins over a provider key (unset BIFROST_URL to call the providers directly)",
        ", ".join(keys),
    )


def provider_keys(environ: Mapping[str, str]) -> tuple[str, ...]:
    """The provider keys set in ``environ`` (:data:`PROVIDER_KEYS`), by name."""
    return tuple(name for name in PROVIDER_KEYS if environ.get(name, "").strip())


def bifrost_wins(environ: Mapping[str, str]) -> None:
    """Bifrost serves this deployment: a provider key also set is not used for model calls,
    which is said once."""
    keys = provider_keys(environ)
    if keys:
        _bifrost_wins(keys)


def missing(gateway: ModelGateway, capability: Capability, what: str) -> str | None:
    """Why ``what`` cannot be (``None``: it can): a route without ``capability``."""
    if not gateway.has(capability):
        return f"{what} needs a gateway with {capability}: Bifrost (BIFROST_URL)"
    return None


# ---------------------------------------------------------------------- LangChain, routed per run
#: The chat models made of routed names, per event loop (an HTTP client's pooled connections
#: belong to the loop that opened them: :func:`per_loop`), by route, name and effort.
_MADE: weakref.WeakKeyDictionary[Any, LRU[tuple[Any, ...], Any]] = weakref.WeakKeyDictionary()
#: The ones made outside an event loop.
_MADE_NO_LOOP: LRU[tuple[Any, ...], Any] = LRU(ROUTED_KEPT)


def per_loop(kept: weakref.WeakKeyDictionary[Any, LRU[Any, Any]], size: int) -> LRU[Any, Any]:
    """The running event loop's cache in ``kept`` (made on first use, gone with the loop)."""
    import asyncio  # noqa: PLC0415

    loop = asyncio.get_running_loop()
    found = kept.get(loop)
    if found is None:
        found = kept[loop] = LRU(size)
    return found


def routed_chat_model(model: str, *, effort: str | None = None) -> Any:
    """A LangChain chat model of the name ``model`` that each call routes as the harness of the
    run making it says (:attr:`Harness.model_gateway`: the settings given in code over the
    environment's), outside a run as the environment does — Bifrost, else the provider's own
    SDK — the name mapped to the route's form. Its ``profile`` is the provider's model's (its
    context window), when LangChain knows it."""
    return _routed_class()(model_name=model, effort=effort, profile=_profile_of(model))


def routed(model: Any) -> bool:
    """Whether ``model`` is a model name the harness routes (:func:`routed_chat_model`)."""
    return isinstance(model, _routed_class())


def route_now() -> ModelGateway:
    """The route a model call made now takes: the harness's of the run in this task, else the
    environment's."""
    from trellis.harness.runtime import current  # noqa: PLC0415 - runtime imports this module
    from trellis.harness.settings import Settings  # noqa: PLC0415

    runtime = current()
    if runtime is not None:
        return runtime.agent.harness.model_gateway
    return model_gateway(Settings.from_env())


def _resolved(model: str, effort: str | None) -> Any:
    gateway = route_now()
    try:
        made = per_loop(_MADE, ROUTED_KEPT)
    except RuntimeError:  # no event loop: a sync call
        made = _MADE_NO_LOOP
    key = (gateway.kind, gateway.url, getattr(gateway, "key", None), model, effort)
    found = made.get(key)
    if found is None:
        found = made[key] = gateway.chat_model(model, effort=effort)
    return found


def _profile_of(model: str) -> dict[str, Any] | None:
    """What LangChain knows of the model (its ``profile``: the context window among it), by
    its provider's chat model made offline; ``None`` when it knows nothing."""
    provider, name = split(model)
    try:
        if provider in (None, "openai"):
            from langchain_openai import ChatOpenAI  # noqa: PLC0415

            profile = ChatOpenAI(model=name, api_key=NO_KEY).profile  # type: ignore[arg-type]
        elif provider == "anthropic":
            from langchain_anthropic import ChatAnthropic  # noqa: PLC0415

            profile = ChatAnthropic(model=name, api_key=NO_KEY).profile  # type: ignore[call-arg,arg-type]
        else:
            return None
    except Exception:
        return None
    return dict(profile) if profile else None


@functools.cache
def _routed_class() -> type:
    from langchain_core.language_models import BaseChatModel  # noqa: PLC0415 - react extra

    class RoutedChatModel(BaseChatModel):
        """A model name, the route chosen per call (:func:`routed_chat_model`): each call,
        tool binding and structured output is the routed chat model's."""

        model_name: str
        effort: str | None = None

        @property
        def _llm_type(self) -> str:
            return "trellis-routed"

        @property
        def _identifying_params(self) -> dict[str, Any]:
            return {"model_name": self.model_name, "effort": self.effort}

        def resolved(self) -> Any:
            """The chat model the route makes of the name now."""
            return _resolved(self.model_name, self.effort)

        def bind_tools(self, tools: Any, **kwargs: Any) -> Any:  # type: ignore[override]
            return self.resolved().bind_tools(tools, **kwargs)

        def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:  # type: ignore[override]
            return self.resolved().with_structured_output(schema, **kwargs)

        def _generate(
            self, messages: Any, stop: Any = None, run_manager: Any = None, **kw: Any
        ) -> Any:
            return self.resolved()._generate(messages, stop=stop, run_manager=run_manager, **kw)

        async def _agenerate(
            self, messages: Any, stop: Any = None, run_manager: Any = None, **kw: Any
        ) -> Any:
            chat = self.resolved()
            return await chat._agenerate(messages, stop=stop, run_manager=run_manager, **kw)

        def _stream(
            self, messages: Any, stop: Any = None, run_manager: Any = None, **kw: Any
        ) -> Any:
            yield from self.resolved()._stream(messages, stop=stop, run_manager=run_manager, **kw)

        async def _astream(
            self, messages: Any, stop: Any = None, run_manager: Any = None, **kw: Any
        ) -> AsyncIterator[Any]:
            chat = self.resolved()
            async for chunk in chat._astream(messages, stop=stop, run_manager=run_manager, **kw):
                yield chunk

    return RoutedChatModel
