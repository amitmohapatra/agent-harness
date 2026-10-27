"""Who is calling, on an A2A call: identity is the deployment's, never the message's.

The trusted form is a header the platform's authenticating edge sets (and overwrites) before
the call reaches this process: ``X-Trellis-Identity: {"tenant_id": ..., "user_id": ...}``,
announced on the card as the trusted-identity extension; a call without it is refused (``401``
at the JSON-RPC route), never run as anonymous — the rule ``serve_chat`` keeps. A deployment
that authenticates some other way passes its own resolver (``serve_a2a(app, url,
identity=fn)``, ``fn(call_context) -> user``).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Final

from a2a.server.context import ServerCallContext

from trellis.harness.identity import IDENTITY_HEADER, trusted_user

if TYPE_CHECKING:
    from trellis.harness.harness import Harness

EXTENSION_URI: Final = "https://trellis.dev/a2a/extensions/trusted-identity/v1"
EXTENSION_DESCRIPTION: Final = (
    "The caller's tenant and user as trusted platform context, set by the deployment's "
    "authenticated edge and never read from the message."
)
UserResolver = Callable[[ServerCallContext], str]


class HeaderIdentity:
    """The default resolver: the trusted header (its tenant, when it names one, the one
    ``TRELLIS_API_KEY`` speaks for); without it the call is refused (:func:`trusted_user`)."""

    def __init__(self, harness: Harness) -> None:
        self.harness = harness

    def __call__(self, context: ServerCallContext) -> str:
        return trusted_user(header(context, IDENTITY_HEADER), self.harness._known_tenant)


def header(context: ServerCallContext, name: str) -> str:
    headers: Any = context.state.get("headers")
    if not isinstance(headers, Mapping):
        return ""
    value = headers.get(name)
    return str(value) if value is not None else ""
