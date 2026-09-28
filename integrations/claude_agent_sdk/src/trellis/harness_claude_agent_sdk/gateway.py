"""Pointing the Claude Agent SDK's CLI at the gateway, without ever handling the key.

The Claude Agent SDK is the only one of the three with **no model seam**: it does not call a
model, it spawns the ``claude`` CLI, which does. There is no client object to replace and no
``Model`` interface to implement, so the whole model moment is two environment variables the
CLI reads — which ``ClaudeAgentOptions.env`` can set per run.

``ANTHROPIC_BASE_URL`` is the binding. Verified against the running gateway on 2026-09-28:
Bifrost serves an Anthropic-compatible API under ``/anthropic`` (``POST
/anthropic/v1/messages`` answers with Anthropic's own error envelope, ``{"type": "error",
"error": {"type": "api_error", …}}``, where ``/v1/chat/completions`` answers with Bifrost's
OpenAI-shaped one). The Anthropic clients append ``/v1/messages`` to the base URL, so the
base URL is the gateway plus ``/anthropic``.

The credential is deliberately **not** something this module touches. The SDK's transport
builds the CLI's environment as ``{**os.environ, **options.env}``, so a token exported in the
process that starts the agent is already inherited; copying it into ``options.env`` would put
a secret into a dataclass that gets logged, repr'd and attached to traces. So this module
*checks* that the variable is set and says so when it is not — a clear failure at build time
instead of a 401 from a subprocess later — and never reads, returns, logs or stores its value.
"""

from __future__ import annotations

import os
from typing import Final

from trellis.contracts.errors import ConfigurationError

__all__ = ["ANTHROPIC_PREFIX", "BASE_URL_VAR", "TOKEN_VAR", "gateway_env"]

#: The path Bifrost serves its Anthropic-compatible API under.
ANTHROPIC_PREFIX: Final = "/anthropic"
#: The variable the CLI reads its endpoint from.
BASE_URL_VAR: Final = "ANTHROPIC_BASE_URL"
#: The variable the CLI reads its credential from. Its value never enters this process's
#: data structures: it is inherited by the subprocess from the environment.
TOKEN_VAR: Final = "ANTHROPIC_AUTH_TOKEN"


def gateway_env(base_url: str | None, *, require_token: bool = True) -> dict[str, str]:
    """The environment that points the CLI at the gateway. Contains no secret.

    Raises when the deployment has not been configured, rather than letting the CLI reach
    Anthropic directly: an agent that silently bypassed the gateway would spend against a
    budget nobody set and leave no record on the team's virtual key.
    """
    if not base_url:
        raise ConfigurationError(
            "the Claude Agent SDK adapter needs the gateway's URL: pass gateway_url= or set "
            "models.gateway_url, so the CLI talks to Bifrost rather than to a provider"
        )
    if require_token and not os.environ.get(TOKEN_VAR):
        raise ConfigurationError(
            f"{TOKEN_VAR} is not set in this process: the CLI inherits its credential from "
            "the environment, and the harness never copies a key into options.env. Export "
            "the gateway's virtual key as that variable before starting the agent"
        )
    return {BASE_URL_VAR: anthropic_base_url(base_url)}


def anthropic_base_url(base_url: str) -> str:
    """``base_url`` as the CLI must see it: the gateway's Anthropic-compatible prefix.

    Idempotent, so a deployment that already configured the full prefix is not given it twice.
    """
    trimmed = base_url.rstrip("/")
    if trimmed.endswith(ANTHROPIC_PREFIX):
        return trimmed
    return f"{trimmed}{ANTHROPIC_PREFIX}"
