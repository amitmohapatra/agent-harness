"""Who is calling, on an A2A call: identity is the deployment's, never the message's.

The trusted form is a header the platform's authenticating edge sets (and overwrites) before
the call reaches this process: ``X-Trellis-Identity: {"tenant_id": ..., "user_id": ...}``,
announced on the card as the trusted-identity extension. A deployment that authenticates some
other way passes its own resolver (``serve_a2a(app, url, identity=fn)``, ``fn(call_context) ->
user``).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Final

from a2a.server.context import ServerCallContext

from trellis.harness.identity import IDENTITY_HEADER, trusted_user

if TYPE_CHECKING:
    from trellis.harness.harness import Harness

log = logging.getLogger("trellis.a2a")

EXTENSION_URI: Final = "https://trellis.dev/a2a/extensions/trusted-identity/v1"
EXTENSION_DESCRIPTION: Final = (
    "The caller's tenant and user as trusted platform context, set by the deployment's "
    "authenticated edge and never read from the message."
)
#: The user a call without any identity runs as (with a warning, once per server).
ANONYMOUS: Final = "anonymous"

UserResolver = Callable[[ServerCallContext], str]


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
        return trusted_user(raw, self.harness._known_tenant)


def header(context: ServerCallContext, name: str) -> str:
    headers: Any = context.state.get("headers")
    if not isinstance(headers, Mapping):
        return ""
    value = headers.get(name)
    return str(value) if value is not None else ""
