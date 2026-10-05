"""Who is calling, on an A2A call: identity is the deployment's, never the message's.

The trusted form is a header the platform's authenticating edge sets (and overwrites) before
the call reaches this process: ``X-Trellis-Identity: {"tenant_id": ..., "user_id": ...}``,
announced on the card as the trusted-identity extension. A deployment that authenticates some
other way passes its own resolver (``serve_a2a(app, url, identity=fn)``, ``fn(call_context) ->
user``).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Final

from a2a.server.context import ServerCallContext

from trellis.harness.identity import IDENTITY_HEADER

if TYPE_CHECKING:
    from trellis.harness.harness import Harness

log = logging.getLogger("trellis.a2a")

EXTENSION_URI: Final = "https://trellis.dev/a2a/extensions/trusted-identity/v1"
EXTENSION_DESCRIPTION: Final = (
    "The caller's tenant and user as trusted platform context, set by the deployment's "
    "authenticated edge and never read from the message."
)
#: A header longer than this is refused rather than parsed.
HEADER_MAX_CHARS: Final = 4096
#: The user a call without any identity runs as (with a warning, once per server).
ANONYMOUS: Final = "anonymous"

UserResolver = Callable[[ServerCallContext], str]


class IdentityRefused(PermissionError):
    """The call carries an identity this deployment will not act on."""


class HeaderIdentity:
    """The default resolver: the trusted header when present (its tenant must be the one
    ``TRELLIS_API_KEY`` speaks for), otherwise :data:`ANONYMOUS`."""

    def __init__(self, harness: Harness) -> None:
        self.harness = harness
        self._warned = False

    def __call__(self, context: ServerCallContext) -> str:
        raw = header(context, IDENTITY_HEADER)
        if not raw:
            if not self._warned:
                self._warned = True
                log.warning(
                    "A2A calls without %s run as user %r: pass serve_a2a(identity=...) or put "
                    "an authenticating edge in front",
                    IDENTITY_HEADER,
                    ANONYMOUS,
                )
            return ANONYMOUS
        if len(raw) > HEADER_MAX_CHARS:
            raise IdentityRefused("the platform identity header is too long")
        try:
            claimed = json.loads(raw)
        except ValueError as exc:
            raise IdentityRefused("the platform identity header is not JSON") from exc
        if not isinstance(claimed, Mapping):
            raise IdentityRefused("the platform identity header must be a JSON object")
        tenant = claimed.get("tenant_id")
        served = self.harness.known_tenant
        if tenant and tenant != served:
            raise IdentityRefused(f"tenant {tenant!r} is not served here")
        user = claimed.get("user_id")
        if not user:
            raise IdentityRefused("the platform identity header names no user")
        return str(user)


def header(context: ServerCallContext, name: str) -> str:
    headers: Any = context.state.get("headers")
    if not isinstance(headers, Mapping):
        return ""
    value = headers.get(name)
    return str(value) if value is not None else ""
