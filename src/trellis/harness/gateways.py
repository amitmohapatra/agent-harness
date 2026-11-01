"""Where a model *name* is served — model routing: one strategy (:class:`ModelGateway`), one
adapter per route, each declaring what it can do (:class:`Capability`). Exactly one serves a
deployment, never a chain of fallbacks, chosen by name from one registry (:func:`registry`:
the routes below, and those Python packages add under the entry-point group
``trellis.gateways``):

1. :class:`OpenAICompatible` (``openai-compatible``) — ``LLM_BASE_URL``, any endpoint that
   speaks OpenAI's chat completions (OpenAI, OpenRouter, a LiteLLM proxy, Ollama, vLLM,
   llama.cpp, Anthropic's and Gemini's OpenAI-compatible endpoints): every model call of every
   framework goes there, the model name as given; models only;
2. :class:`Bifrost` (``bifrost``) — ``LLM_GATEWAY=bifrost``: the same, the name as Bifrost's
   ``provider/model``, and everything else the harness uses a gateway for (MCP tools, Code
   Mode, stored prompts, skills, the Claude Agent SDK on its Anthropic route...);
3. :class:`ProviderSDK` (``provider-sdk``) — no ``LLM_BASE_URL``: the provider's own SDK
   (LangChain's ``init_chat_model``), reading its own environment (``OPENAI_API_KEY``,
   ``ANTHROPIC_API_KEY``, Google's and AWS's credentials...; never read here): models only.

A URL route wins over a provider key (said once, and by ``doctor``). Its requests are
authenticated by ``LLM_AUTH``'s strategy (bifrost-sdk's registry: ``bearer`` with
``LLM_API_KEY``, ``google``, ``aws``). A model name is mapped to the route's form
(:meth:`ModelGateway.routed`). A feature that needs a capability the deployment's route lacks
is off — ``trellis doctor`` lists them — and one asked for explicitly says why it cannot be
(:func:`missing`). What the memory service is told to call for an agent's own model work is the
route's :meth:`~ModelGateway.endpoint` (``trellis.contracts.Endpoint``)."""

from __future__ import annotations

import functools
import importlib
import importlib.metadata
import logging
import weakref
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Mapping
from enum import StrEnum
from typing import TYPE_CHECKING, Any, ClassVar, Final, Self

from trellis.contracts import ConfigurationError, Endpoint, ModelProfile
from trellis.harness.bounded import LRU

if TYPE_CHECKING:
    import httpx

    from trellis.harness.clients.bifrost import Gateway
    from trellis.harness.evals import OpenAICompatibleJudge
    from trellis.harness.settings import Settings

log = logging.getLogger("trellis.harness")


class Capability(StrEnum):
    MODELS = "models"
    MCP = "mcp"
    CODE_MODE = "code_mode"
    COST = "cost"
    FALLBACKS = "fallbacks"
    PROMPTS = "prompts"
    SKILLS = "skills"
    ATTRIBUTION = "attribution"
    ANTHROPIC_ROUTE = "anthropic_route"


#: What each capability is, as ``doctor`` says it when it is off.
CAPABILITIES: Final[dict[Capability, str]] = {
    Capability.MODELS: "model names (ReAct(model='name'), llm_judge(model='name'))",
    Capability.MCP: "the gateway's MCP tools (the ones the virtual key allows)",
    Capability.CODE_MODE: "Code Mode",
    Capability.COST: "dollar budgets (Budget(max_cost=)) from the gateway's cost of each call",
    Capability.FALLBACKS: "model fallbacks and retries in the gateway",
    Capability.PROMPTS: "stored prompts (stored('name'))",
    Capability.SKILLS: "the gateway's skills repository (skills.bifrost())",
    Capability.ATTRIBUTION: (
        "each call's run, agent, tenant and user as the gateway's dimensions (the OpenAI "
        "Agents SDK's and Kimi Code's model calls routed through the gateway to carry them)"
    ),
    Capability.ANTHROPIC_ROUTE: (
        "the Claude Agent SDK's model calls on the gateway's Anthropic route"
    ),
}
#: The entry-point group a package registers a route of its own under: the entry point's name
#: is the route's ``LLM_GATEWAY`` name, its object a :class:`ModelGateway` subclass.
ENTRY_POINTS: Final = "trellis.gateways"
#: The route a set ``LLM_BASE_URL`` takes when ``LLM_GATEWAY`` does not name one.
DEFAULT_ROUTE: Final = "openai-compatible"
#: The auth strategy ``LLM_AUTH`` names when unset (bifrost-sdk's ``AUTH_STRATEGIES``).
BEARER: Final = "bearer"
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
#: A provider id as LangChain's ``init_chat_model`` names it, where it differs from Bifrost's
#: (the ``[gemini]``, ``[vertex]`` and ``[bedrock]`` extras install the first three).
LANGCHAIN_PROVIDERS: Final = {
    "gemini": "google_genai",
    "vertex": "google_vertexai",
    "bedrock": "bedrock_converse",
    "mistral": "mistralai",
}
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
#: The ``Authorization`` a bearer endpoint without a key is sent (an OpenAI client needs one).
NO_KEY: Final = "no-key"
#: The key each auth strategy is built with when ``LLM_API_KEY`` is unset, by its name: a
#: bearer key's placeholder; a login (``google``, ``aws``) takes none.
KEYLESS: Final[dict[str, str]] = {BEARER: NO_KEY}
#: Retries of a model call Bifrost's client makes itself: none — the gateway retries (a
#: provider's ``network_config.max_retries``) and falls back, the one place that does.
BIFROST_CLIENT_RETRIES: Final = 0
#: How many chat models made of routed names are kept (per route, name, effort, event loop).
ROUTED_KEPT: Final = 64
#: Anthropic's own API. As an OpenAI-compatible endpoint (``LLM_BASE_URL`` its ``/v1``) it is
#: its OpenAI SDK compatibility layer, which drops prompt caching ("Prompt caching is not
#: supported, but it is supported in the Anthropic SDKs":
#: https://platform.claude.com/docs/en/api/openai-sdk): its models are made with Anthropic's own
#: client, on the same endpoint and key (:meth:`OpenAICompatible.chat_model`).
ANTHROPIC_API: Final = "https://api.anthropic.com"


class ModelGateway(ABC):
    """The port: a route that serves model names, and what it can do. A subclass names itself
    (``name``: its ``LLM_GATEWAY`` value) and is found through :func:`registry`."""

    #: the name ``LLM_GATEWAY`` gives it (the registry's key)
    name: ClassVar[str]
    #: how ``doctor`` and the start-up line name it
    title: ClassVar[str]
    capabilities: ClassVar[frozenset[Capability]]

    def __init__(self, url: str | None, key: str | None = None, auth: str = BEARER) -> None:
        #: where model calls go (``None``: each provider's own endpoint)
        self.url = url
        #: the bearer key (``LLM_API_KEY``); ``None`` for a login, or a server that takes none
        self.key = key
        #: the auth strategy's name (``LLM_AUTH``)
        self.auth = auth

    def has(self, capability: Capability) -> bool:
        return capability in self.capabilities

    @property
    def where(self) -> str:
        """Where its model calls go, as the start-up line and ``doctor`` say it."""
        return self.url or "each provider's own endpoint"

    @classmethod
    @abstractmethod
    def of(cls, settings: Settings) -> Self:
        """This route, as ``settings`` configure it (a :class:`ConfigurationError` when they
        do not fit it)."""

    @abstractmethod
    def chat_model(self, model: str, *, effort: str | None) -> Any:
        """A LangChain chat model of ``model`` on this route. It has no timeout of its own:
        a model call's bound is the run's (``ModelHooks``, the stall watchdog), applied once."""

    @abstractmethod
    def routed(self, model: str) -> str:
        """``model`` as this route names it."""

    @abstractmethod
    def judge(self, model: str) -> OpenAICompatibleJudge:
        """``model`` as a judge asks it, on this route (a gateway's own judge client, with the
        judge's key, is ``EvalServices.judge_gateway``)."""

    @abstractmethod
    def endpoint(self, model: str | None) -> Endpoint | None:
        """What the memory service is told to call for an agent's own model work, whose model
        is ``model`` (``None``: not known; the service picks): this route's endpoint, its key
        only when it is a bearer key, and the model's profile as LangChain knows it.
        ``None``: there is none to tell (the service uses its own settings)."""

    def gateway_client(self, key: str | None = None) -> Gateway | None:
        """The client of the gateway's own API (MCP tools, stored prompts, skills, the judge
        alias), with ``key`` over the route's; ``None``: the route has none."""
        return None

    def announce(self, environ: Mapping[str, str]) -> None:
        """Say once what this route means for the environment's provider keys (nothing by
        default)."""
        return None

    def notes(self) -> list[str]:
        """What ``doctor`` says of how this route calls its models, beyond its name and auth
        (nothing by default)."""
        return []


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


def profile(model: str | None) -> ModelProfile:
    """What ``model`` accepts as LangChain's profile says it (:func:`_profile_of`); a plain
    chat model when it says nothing."""
    found = _profile_of(model) if model else None
    return ModelProfile.model_validate(found or {})


def openai_auth(auth: httpx.Auth | None) -> Any:
    """``auth`` (an ``httpx.Auth``: bifrost-sdk's strategies) as the OpenAI SDK's clients take
    it: they build on their own HTTP library (a fork of httpx, ``httpx2``, whose requests have
    httpx's API) and take only its ``Auth``, so the strategy is delegated to from one."""
    if auth is None:
        return None
    import openai  # noqa: PLC0415 - the react extra

    return _delegating(openai.DefaultAsyncHttpxClient)(auth)


@functools.cache
def _delegating(client: type) -> type:
    """An ``Auth`` of ``client``'s HTTP library (the module its base class is of) that runs
    another library's ``httpx.Auth`` flow on its requests."""
    library = importlib.import_module(client.__mro__[1].__module__.partition(".")[0])

    class Delegating(library.Auth):  # type: ignore[name-defined,misc]
        def __init__(self, inner: httpx.Auth) -> None:
            self.inner = inner
            self.requires_request_body = inner.requires_request_body
            self.requires_response_body = inner.requires_response_body

        def sync_auth_flow(self, request: Any) -> Any:
            return (yield from self.inner.sync_auth_flow(request))

        async def async_auth_flow(self, request: Any) -> Any:
            flow = self.inner.async_auth_flow(request)
            sent = await anext(flow)
            while True:
                response = yield sent
                try:
                    sent = await flow.asend(response)
                except StopAsyncIteration:
                    return

    return Delegating


def _chat(model: str, effort: str | None, auth: httpx.Auth | None = None, **more: Any) -> Any:
    import openai  # noqa: PLC0415 - the react extra
    from langchain_openai import ChatOpenAI  # noqa: PLC0415

    return ChatOpenAI(
        model=model,
        stream_usage=True,
        reasoning_effort=effort,
        # clients of its own: langchain-openai shares one per URL across the process, whose
        # pooled connections break the models of a later event loop; each authenticates its
        # requests with the route's strategy
        http_client=openai.DefaultHttpxClient(auth=openai_auth(auth)),
        http_async_client=openai.DefaultAsyncHttpxClient(auth=openai_auth(auth)),
        **more,
    )


def _anthropic(model: str, effort: str | None, url: str, key: str | None) -> Any:
    """A ``ChatAnthropic`` of ``model`` on Anthropic's API at ``url`` with ``key``."""
    from langchain_anthropic import ChatAnthropic  # noqa: PLC0415 - the react extra's

    effort_given = {"reasoning_effort": effort} if effort is not None else {}
    return ChatAnthropic(model_name=model, base_url=url, api_key=key, **effort_given)  # type: ignore[call-arg,arg-type]


class OpenAICompatible(ModelGateway):
    """An OpenAI-compatible endpoint (``LLM_BASE_URL``): model names as given, on its chat
    completions, authenticated by ``LLM_AUTH``'s strategy; models only, the client's own
    retries. Anthropic's own API (:data:`ANTHROPIC_API`) is called with Anthropic's client
    instead, which keeps its prompt cache, unless ``LLM_GATEWAY`` names this route
    (``forced``: the OpenAI-compatible client asked for)."""

    name = "openai-compatible"
    title = "OpenAI-compatible endpoint"
    capabilities = frozenset({Capability.MODELS})
    url: str

    def __init__(
        self, url: str, key: str | None = None, auth: str = BEARER, *, forced: bool = False
    ) -> None:
        super().__init__(url, key, auth)
        #: the OpenAI-compatible client asked for by name (``LLM_GATEWAY``), whatever the endpoint
        self.forced = forced

    @classmethod
    def of(cls, settings: Settings) -> Self:
        url = settings.llm_base_url
        if not url:
            raise ConfigurationError(
                f"LLM_GATEWAY={cls.name} needs LLM_BASE_URL: the endpoint's OpenAI-compatible "
                "/v1 root"
            )
        if not url.startswith(("http://", "https://")):
            raise ConfigurationError(f"LLM_BASE_URL is an http(s) URL, not {url!r}")
        from bifrost_sdk import AUTH_STRATEGIES  # noqa: PLC0415

        auth = settings.llm_auth or BEARER
        if auth not in AUTH_STRATEGIES:
            known = ", ".join(sorted(AUTH_STRATEGIES))
            raise ConfigurationError(f"LLM_AUTH={auth!r} names no auth strategy; known: {known}")
        if auth not in KEYLESS and settings.llm_api_key:
            raise ConfigurationError(
                f"LLM_AUTH={auth} signs with its own credentials: unset LLM_API_KEY"
            )
        return cls(url, settings.llm_api_key, auth, forced=settings.llm_gateway == cls.name)

    @property
    def origin(self) -> str:
        """The endpoint's root (``LLM_BASE_URL`` less its ``/v1``): a gateway's other routes
        hang there (Bifrost's ``/anthropic``, ``/mcp``)."""
        return self.url.rstrip("/").removesuffix("/v1")

    @functools.cached_property
    def http_auth(self) -> httpx.Auth:
        """How each request is authenticated: ``LLM_AUTH``'s strategy, made on first use (a
        login reads its credentials then)."""
        from bifrost_sdk import auth_strategy  # noqa: PLC0415

        return auth_strategy(self.auth, self.key or KEYLESS.get(self.auth))

    def routed(self, model: str) -> str:
        return model

    def client_options(self) -> dict[str, Any]:
        """What this route adds to each chat model's client."""
        return {}

    @property
    def signer(self) -> httpx.Auth | None:
        """What signs each model request beyond the OpenAI client's own key: nothing for a
        bearer key (the client sends it), the login's strategy otherwise."""
        return None if self.auth in KEYLESS else self.http_auth

    @property
    def anthropic(self) -> bool:
        """Whether its models are made with Anthropic's own client: the endpoint is Anthropic's
        API, reached with a bearer key, and the OpenAI-compatible client was not asked for."""
        return (
            not self.forced
            and self.origin == ANTHROPIC_API
            and self.auth == BEARER
            and self.key is not None
        )

    def notes(self) -> list[str]:
        if not self.anthropic:
            return []
        return [
            "client: Anthropic's own, on its API (its OpenAI compatibility layer drops prompt "
            "caching); LLM_GATEWAY=openai-compatible keeps the OpenAI-compatible client"
        ]

    def chat_model(self, model: str, *, effort: str | None) -> Any:
        if self.anthropic:
            return _anthropic(self.routed(model), effort, self.origin, self.key)
        return _chat(
            self.routed(model),
            effort,
            self.signer,
            base_url=self.url,
            api_key=self.key or NO_KEY,
            **self.client_options(),
        )

    def judge(self, model: str) -> OpenAICompatibleJudge:
        from trellis.harness.evals import OpenAICompatibleJudge  # noqa: PLC0415

        return OpenAICompatibleJudge(
            self.routed(model),
            self.key or NO_KEY,
            base_url=self.url,
            auth=self.signer,
            options=self.client_options(),
        )

    def endpoint(self, model: str | None) -> Endpoint:
        from pydantic import SecretStr  # noqa: PLC0415

        return Endpoint(
            base_url=self.url,
            gateway=self.name,
            auth=self.auth,
            model=self.routed(model) if model else None,
            api_key=SecretStr(self.key) if self.key and self.auth in KEYLESS else None,
            profile=profile(model),
        )

    def announce(self, environ: Mapping[str, str]) -> None:
        keys = provider_keys(environ)
        if keys:
            _endpoint_wins(self.title, keys)


class Bifrost(OpenAICompatible):
    """Bifrost (``LLM_GATEWAY=bifrost``): model names as its ``provider/model``, on its chat
    completions with the deny-all MCP scope (the harness lists and runs the MCP tools itself),
    its retries and fallbacks, the virtual key as the bearer key; every capability."""

    name = "bifrost"
    title = "Bifrost"
    capabilities = frozenset(Capability)

    @classmethod
    def of(cls, settings: Settings) -> Self:
        found = super().of(settings)
        if found.auth != BEARER:
            raise ConfigurationError(
                f"LLM_GATEWAY=bifrost authenticates with its virtual key (LLM_AUTH={BEARER}, "
                f"LLM_API_KEY), not LLM_AUTH={found.auth}"
            )
        return found

    def routed(self, model: str) -> str:
        provider, name = split(model)
        return name if provider is None else f"{provider}/{name}"

    def client_options(self) -> dict[str, Any]:
        from bifrost_sdk import NO_GATEWAY_TOOLS  # noqa: PLC0415

        return {"default_headers": dict(NO_GATEWAY_TOOLS), "max_retries": BIFROST_CLIENT_RETRIES}

    def gateway_client(self, key: str | None = None) -> Gateway:
        from trellis.harness.clients.bifrost import Gateway  # noqa: PLC0415

        return Gateway(self.url, key or self.key)


class ProviderSDK(ModelGateway):
    """No endpoint: a model name goes to its provider, through the provider's own SDK
    (LangChain's ``init_chat_model``), which reads its own key and endpoint from the
    environment (``OPENAI_API_KEY``, ``OPENAI_BASE_URL``, ``ANTHROPIC_API_KEY``, Google's and
    AWS's credentials...): models only, the client's own retries (nothing else retries there)."""

    name = "provider-sdk"
    title = "provider SDK"
    capabilities = frozenset({Capability.MODELS})

    def __init__(self) -> None:
        super().__init__(None)

    @classmethod
    def of(cls, settings: Settings) -> Self:
        given = [
            name
            for name, value in (
                ("LLM_BASE_URL", settings.llm_base_url),
                ("LLM_API_KEY", settings.llm_api_key),
                ("LLM_AUTH", settings.llm_auth),
            )
            if value
        ]
        if given:
            raise ConfigurationError(
                f"{', '.join(given)} set with LLM_GATEWAY={cls.name}: the providers' own SDKs "
                "read their own endpoint and credentials (unset them, or set LLM_BASE_URL with "
                "an endpoint's route)"
            )
        return cls()

    def routed(self, model: str) -> str:
        provider, name = split(model)
        if provider is None:
            raise ConfigurationError(
                f"model {model!r}: name its provider ('openai:{model}', 'anthropic:{model}'), "
                "whose key its SDK reads (OPENAI_API_KEY, ANTHROPIC_API_KEY...), or set "
                "LLM_BASE_URL to send every model to one endpoint"
            )
        return f"{LANGCHAIN_PROVIDERS.get(provider, provider)}:{name}"

    def chat_model(self, model: str, *, effort: str | None) -> Any:
        provider, _, name = self.routed(model).partition(":")
        if provider == "openai":  # the client the harness tunes (usage on streams, effort)
            return _chat(name, effort)
        from langchain.chat_models import init_chat_model  # noqa: PLC0415

        return init_chat_model(name, model_provider=provider)

    def judge(self, model: str) -> OpenAICompatibleJudge:
        """``model`` as a judge asks it: an OpenAI model on the OpenAI SDK's own endpoint."""
        from trellis.harness.evals import OpenAICompatibleJudge  # noqa: PLC0415

        provider, _, name = self.routed(model).partition(":")
        if provider != "openai":
            raise ConfigurationError(
                f"a judge by name without LLM_BASE_URL is an OpenAI model on the OpenAI SDK's "
                f"endpoint (OPENAI_API_KEY, OPENAI_BASE_URL), not {model!r}: set LLM_BASE_URL, "
                "or give llm_judge a model object"
            )
        return OpenAICompatibleJudge(name)

    def endpoint(self, model: str | None) -> None:
        """None: the providers' keys are their SDKs' own, never read here, so the memory
        service is told no endpoint (it uses its own settings)."""
        return None


#: The routes built in, by name; :func:`registry` adds the entry points'.
GATEWAYS: Final[dict[str, type[ModelGateway]]] = {
    g.name: g for g in (OpenAICompatible, Bifrost, ProviderSDK)
}


@functools.cache
def registry() -> dict[str, type[ModelGateway]]:
    """Every route by its ``LLM_GATEWAY`` name: the built-in ones (:data:`GATEWAYS`) and each
    a package registers under the entry-point group :data:`ENTRY_POINTS`, read once."""
    found = dict(GATEWAYS)
    for entry in importlib.metadata.entry_points(group=ENTRY_POINTS):
        if entry.name in found:
            raise ConfigurationError(
                f"the {ENTRY_POINTS} entry point {entry.value!r} is named {entry.name!r}, "
                "which a route already has"
            )
        loaded = entry.load()
        if not (isinstance(loaded, type) and issubclass(loaded, ModelGateway)):
            raise ConfigurationError(
                f"the {ENTRY_POINTS} entry point {entry.name!r} ({entry.value}) is not a "
                "ModelGateway subclass"
            )
        found[entry.name] = loaded
    return found


def model_gateway(settings: Settings) -> ModelGateway:
    """The route a model name goes by: the one ``LLM_GATEWAY`` names, else
    ``openai-compatible`` with ``LLM_BASE_URL`` set, else the provider's own SDK."""
    name = settings.llm_gateway or (DEFAULT_ROUTE if settings.llm_base_url else ProviderSDK.name)
    routes = registry()
    if name not in routes:
        raise ConfigurationError(
            f"LLM_GATEWAY={name!r} names no route; known: {', '.join(sorted(routes))}"
        )
    return routes[name].of(settings)


@functools.cache  # said once per process
def _endpoint_wins(title: str, keys: tuple[str, ...]) -> None:
    log.info(
        "model calls go to the %s (LLM_BASE_URL), not to the providers with %s: the endpoint "
        "wins over a provider key (unset LLM_BASE_URL to call the providers directly)",
        title,
        ", ".join(keys),
    )


def provider_keys(environ: Mapping[str, str]) -> tuple[str, ...]:
    """The provider keys set in ``environ`` (:data:`PROVIDER_KEYS`), by name."""
    return tuple(name for name in PROVIDER_KEYS if environ.get(name, "").strip())


def missing(gateway: ModelGateway, capability: Capability, what: str) -> str | None:
    """Why ``what`` cannot be (``None``: it can): a route without ``capability``, naming the
    routes that have it."""
    if gateway.has(capability):
        return None
    having = [
        f"{g.title} (LLM_GATEWAY={n})"
        for n, g in registry().items()
        if capability in g.capabilities
    ]
    return f"{what} needs a gateway with {capability}: {', '.join(having)}"


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
    environment's), outside a run as the environment does (:func:`model_gateway`) — the name
    mapped to the route's form. Its ``profile`` is the provider's model's (its
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
    # a route that keeps the OpenAI-compatible client when asked (``forced``) makes another one
    forced = getattr(gateway, "forced", False)
    key = (gateway.name, gateway.url, gateway.key, gateway.auth, forced, model, effort)
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
