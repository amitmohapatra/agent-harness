"""Where a model *name* is served: a port (:class:`ModelGateway`) with one adapter per kind of
gateway, each declaring what it can do (:class:`Capability`). Exactly one serves a deployment —
the first adapter in :data:`GATEWAYS` its configuration names; never a chain of fallbacks:

1. :class:`Bifrost` — ``BIFROST_URL`` (and ``BIFROST_VIRTUAL_KEY``): models, and everything
   else the harness uses the gateway for;
2. :class:`OpenAICompatible` — the OpenAI SDK's own ``OPENAI_BASE_URL`` and
   ``OPENAI_API_KEY``, read by the SDK itself (LiteLLM's proxy, Portkey, TrueFoundry,
   OpenRouter, vLLM, Ollama, OpenAI): models only;
3. none — the agent's model is an object passed in code, and a model name is an error naming
   the three (:func:`unserved`).

A feature that needs a capability the deployment's gateway lacks is off — ``trellis doctor``
lists them — and one asked for explicitly says why it cannot be (:func:`missing`)."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Mapping
from enum import StrEnum
from typing import TYPE_CHECKING, Any, ClassVar, Final, Self

from trellis.contracts import ConfigurationError

if TYPE_CHECKING:
    from trellis.harness.settings import Settings


class Capability(StrEnum):
    MODELS = "models"
    MCP = "mcp"
    CODE_MODE = "code_mode"
    COST = "cost"
    FALLBACKS = "fallbacks"
    PROMPTS = "prompts"
    SKILLS = "skills"
    ATTRIBUTION = "attribution"


class GatewayKind(StrEnum):
    BIFROST = "bifrost"
    OPENAI_COMPATIBLE = "openai_compatible"


#: What each capability is, as ``doctor`` says it when it is off.
CAPABILITIES: Final[dict[Capability, str]] = {
    Capability.MODELS: "model names (ReAct(model='name'), llm_judge(model='name'))",
    Capability.MCP: "the gateway's MCP tools and Virtual MCPs (mcp('name'))",
    Capability.CODE_MODE: "Code Mode",
    Capability.COST: "dollar budgets (Budget(max_cost=)) from the gateway's cost of each call",
    Capability.FALLBACKS: "model fallbacks and retries in the gateway",
    Capability.PROMPTS: "stored prompts (prompt='name')",
    Capability.SKILLS: "the gateway's Skills Repository",
    Capability.ATTRIBUTION: "each call's run, agent, tenant and user as the gateway's dimensions",
}
#: The OpenAI SDK's own variables: either one names an OpenAI-compatible endpoint.
OPENAI_ENV: Final = ("OPENAI_BASE_URL", "OPENAI_API_KEY")
#: Where the OpenAI SDK sends a request when ``OPENAI_BASE_URL`` is not set.
OPENAI_DEFAULT_URL: Final = "https://api.openai.com/v1"
#: The ``Authorization`` a gateway without a key is sent (the client needs one).
NO_KEY: Final = "no-key"
#: Retries of a model call Bifrost's client makes itself: none — the gateway retries (a
#: provider's ``network_config.max_retries``) and falls back, the one place that does.
BIFROST_CLIENT_RETRIES: Final = 0


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
    def detect(cls, settings: Settings, environ: Mapping[str, str]) -> Self | None:
        """This gateway, when the deployment's configuration names it."""

    @abstractmethod
    def chat_model(self, model: str, *, timeout: float | None, effort: str | None) -> Any:
        """A LangChain chat model of ``model`` on this gateway."""


def _chat(model: str, timeout: float | None, effort: str | None, **more: Any) -> Any:
    import openai  # noqa: PLC0415 - the react extra
    from langchain_openai import ChatOpenAI  # noqa: PLC0415

    return ChatOpenAI(
        model=model,
        timeout=timeout,
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
    def detect(cls, settings: Settings, environ: Mapping[str, str]) -> Self | None:
        if not settings.bifrost_url:
            return None
        return cls(settings.bifrost_url, settings.bifrost_virtual_key)

    def chat_model(self, model: str, *, timeout: float | None, effort: str | None) -> Any:
        from bifrost_sdk import NO_GATEWAY_TOOLS  # noqa: PLC0415

        return _chat(
            model,
            timeout,
            effort,
            base_url=self.url,
            api_key=self.key or NO_KEY,
            default_headers=dict(NO_GATEWAY_TOOLS),
            max_retries=BIFROST_CLIENT_RETRIES,
        )


class OpenAICompatible(ModelGateway):
    """Any OpenAI-compatible endpoint the OpenAI SDK's own variables name: models only; the
    client's own retries (nothing else retries there)."""

    kind = GatewayKind.OPENAI_COMPATIBLE
    title = "OpenAI-compatible"
    capabilities = frozenset({Capability.MODELS})

    def __init__(self, url: str, source: str, *, key: str | None) -> None:
        super().__init__(url, source)
        #: ``OPENAI_API_KEY`` (the SDK reads it itself; ``None``: a server that takes none)
        self.key = key

    @property
    def probe_key(self) -> str:
        return self.key or NO_KEY

    @classmethod
    def detect(cls, settings: Settings, environ: Mapping[str, str]) -> Self | None:
        named = [name for name in OPENAI_ENV if environ.get(name, "").strip()]
        if not named:
            return None
        url = environ.get("OPENAI_BASE_URL", "").strip() or OPENAI_DEFAULT_URL
        key = environ.get("OPENAI_API_KEY", "").strip() or None
        return cls(url, " and ".join(named), key=key)

    def chat_model(self, model: str, *, timeout: float | None, effort: str | None) -> Any:
        return _chat(model, timeout, effort, **({} if self.key else {"api_key": NO_KEY}))

    def judge(self, model: str) -> Any:
        """``model`` as a judge asks it (Bifrost's judges go through the harness's own client,
        with the judge's key: ``EvalServices.judge_gateway``)."""
        from trellis.harness.evals import OpenAICompatibleJudge  # noqa: PLC0415

        return OpenAICompatibleJudge(model, key=None if self.key else NO_KEY)


#: The gateways, in the order a deployment's configuration is read: the first it names serves.
GATEWAYS: Final[tuple[type[ModelGateway], ...]] = (Bifrost, OpenAICompatible)


def model_gateway(
    settings: Settings, environ: Mapping[str, str] | None = None
) -> ModelGateway | None:
    """The gateway a model name goes to (``None``: none is configured)."""
    env = os.environ if environ is None else environ
    for gateway in GATEWAYS:
        found = gateway.detect(settings, env)
        if found is not None:
            return found
    return None


def unserved(what: str) -> ConfigurationError:
    """The error for a model name with no gateway to serve it."""
    return ConfigurationError(
        f"{what} with a model name needs a gateway: BIFROST_URL (Bifrost), or "
        "OPENAI_BASE_URL / OPENAI_API_KEY (any OpenAI-compatible endpoint); or pass a model "
        "object instead of a name"
    )


def missing(gateway: ModelGateway | None, capability: Capability, what: str) -> str | None:
    """Why ``what`` cannot be (``None``: it can): no gateway, or one without ``capability``."""
    if gateway is None:
        return f"{what} needs a gateway with {capability}: Bifrost (BIFROST_URL)"
    if not gateway.has(capability):
        return (
            f"{what} needs a gateway with {capability}: the {gateway.title} gateway at "
            f"{gateway.url} has none (Bifrost has it)"
        )
    return None
