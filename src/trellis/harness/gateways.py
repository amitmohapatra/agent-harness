"""Where a model *name* is served — exactly one place per deployment, never a chain of
fallbacks:

1. **Bifrost** — ``BIFROST_URL`` (and ``BIFROST_VIRTUAL_KEY``): models, and everything else the
   harness uses the gateway for (:data:`BIFROST`);
2. else **any OpenAI-compatible endpoint** — the OpenAI SDK's own ``OPENAI_BASE_URL`` and
   ``OPENAI_API_KEY``, read by the SDK itself (LiteLLM's proxy, Portkey, TrueFoundry, OpenRouter,
   vLLM, Ollama, OpenAI): models only (:data:`OPENAI_COMPATIBLE`);
3. else **none** — the agent's model is an object passed in code (``ReAct(model=ChatX(...))``,
   a framework's own client), and a model name is an error naming these three.

Each gateway declares what it can do (:data:`Capability`). A feature that needs one the
deployment's gateway does not have is off — ``trellis doctor`` lists them — and a feature asked
for explicitly (a Virtual MCP, a judge alias, a stored prompt) says why it cannot be."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Literal

from trellis.contracts import ConfigurationError

Capability = Literal["models", "mcp", "code_mode", "cost", "fallbacks", "prompts", "skills"]
#: What each capability is, as ``doctor`` says it when it is off.
CAPABILITIES: Final[dict[str, str]] = {
    "models": "model names (ReAct(model='name'), llm_judge(model='name'))",
    "mcp": "the gateway's MCP tools and Virtual MCPs (mcp('name'))",
    "code_mode": "Code Mode",
    "cost": "dollar budgets (Budget(max_cost=)) from the gateway's cost of each call",
    "fallbacks": "model fallbacks and retries in the gateway",
    "prompts": "stored prompts (prompt='name')",
    "skills": "the gateway's Skills Repository",
}
BIFROST: Final[frozenset[str]] = frozenset(CAPABILITIES)
OPENAI_COMPATIBLE: Final[frozenset[str]] = frozenset({"models"})
#: The OpenAI SDK's own variables: either one names an OpenAI-compatible endpoint.
OPENAI_ENV: Final = ("OPENAI_BASE_URL", "OPENAI_API_KEY")
#: Where the OpenAI SDK sends a request when ``OPENAI_BASE_URL`` is not set.
OPENAI_DEFAULT_URL: Final = "https://api.openai.com/v1"


@dataclass(frozen=True)
class ModelGateway:
    """The deployment's gateway: which kind, where, what it can do, and what named it."""

    kind: Literal["bifrost", "openai_compatible"]
    url: str
    capabilities: frozenset[str]
    source: str

    def has(self, capability: Capability) -> bool:
        return capability in self.capabilities


def model_gateway(
    bifrost_url: str | None, environ: Mapping[str, str] | None = None
) -> ModelGateway | None:
    """The gateway a model name goes to: Bifrost when ``bifrost_url`` is set, else the
    OpenAI-compatible endpoint the OpenAI SDK's variables name, else none."""
    if bifrost_url:
        return ModelGateway("bifrost", bifrost_url, BIFROST, "BIFROST_URL")
    env = os.environ if environ is None else environ
    named = [name for name in OPENAI_ENV if env.get(name, "").strip()]
    if named:
        url = env.get("OPENAI_BASE_URL", "").strip() or OPENAI_DEFAULT_URL
        return ModelGateway("openai_compatible", url, OPENAI_COMPATIBLE, " and ".join(named))
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
            f"{what} needs a gateway with {capability}: the {gateway.kind.replace('_', '-')} "
            f"gateway at {gateway.url} has none (Bifrost has it)"
        )
    return None
