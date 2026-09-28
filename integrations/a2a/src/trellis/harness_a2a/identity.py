"""Who is asking, on an A2A call (design §9: "the caller's tenant/workspace travel as trusted
context on the A2A extension header, never inferred by the callee").

The rule this module enforces, and the reason it is small: **identity is the deployment's, never
the caller's**. An A2A request carries a message and a `tenant` field, both written by whoever
sent it; neither is identity. What the harness trusts is a header the platform's own edge sets
*after* it authenticated the caller, exactly as the AG-UI route trusts its `context_factory`.

```mermaid
sequenceDiagram
  participant C as Calling agent
  participant E as Platform edge (authn)
  participant S as A2A server
  participant H as Harness
  C->>E: JSON-RPC + credential (card security scheme)
  E->>E: authenticate, strip any client-sent identity header
  E->>S: same call + X-Trellis-Identity {tenant, user, workspace}
  S->>S: IdentityResolver.resolve(call context)
  alt refused
    S-->>C: error (no run starts, no memory is touched)
  else trusted
    S->>H: AgentExecutionContext(tenant, user, workspace)
  end
```

A deployment that cannot put an authenticating edge in front of the server uses
:class:`FixedIdentity` instead and serves exactly one tenant. There is deliberately no third
option: a server that believed a header nobody checked would be a tenant-impersonation hole with
the shape of a working system.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any, Final, Protocol, runtime_checkable

#: The A2A extension this surface declares on its card and activates on its calls. The identity
#: fields themselves travel in the header below, which is what an edge can strip and re-set.
EXTENSION_URI: Final = "https://trellis.dev/a2a/extensions/trusted-identity/v1"
EXTENSION_DESCRIPTION: Final = (
    "The caller's tenant, user and workspace as trusted platform context, set by the "
    "deployment's authenticated edge and never read from the message."
)
#: Lower-case: the SDK hands an executor ``call_context.state['headers']`` with lower-cased keys.
IDENTITY_HEADER: Final = "x-trellis-identity"
#: The only fields read out of the header. Anything else in it is dropped.
IDENTITY_FIELDS: Final = ("tenant_id", "user_id", "workspace_id")
#: A header longer than this is refused rather than parsed: it is a header, not a payload.
HEADER_MAX_CHARS: Final = 4096


class IdentityRefused(PermissionError):
    """The call carries no identity this deployment will act on. The server turns it into an A2A
    error before a run starts, so nothing downstream ever sees a caller it could not place."""


@runtime_checkable
class IdentityResolver(Protocol):
    """How a deployment answers "who is asking?" for an A2A call.

    ``resolve`` receives the SDK's ``ServerCallContext`` (headers in ``state['headers']``, the
    authenticated principal in ``user``) and returns the identity fields the harness builds its
    :class:`~trellis.contracts.context.AgentExecutionContext` from. It raises
    :class:`IdentityRefused` rather than returning a default.
    """

    def resolve(self, call_context: Any) -> dict[str, Any]: ...


class FixedIdentity:
    """One tenant, from configuration. For a server that is not multi-tenant.

    The identity header is ignored outright — not merged, not used as a fallback — so a
    single-tenant deployment cannot be talked into another tenant by a caller who guessed the
    header name.
    """

    name = "fixed"

    def __init__(
        self, tenant_id: str, *, user_id: str | None = None, workspace_id: str | None = None
    ) -> None:
        if not tenant_id:
            raise ValueError("FixedIdentity needs a tenant_id")
        self._identity = {"tenant_id": tenant_id, "user_id": user_id, "workspace_id": workspace_id}

    def resolve(self, call_context: Any) -> dict[str, Any]:
        return {k: v for k, v in self._identity.items() if v}


class TrustedHeaderIdentity:
    """The platform edge's identity header, checked before it is believed.

    The deployment contract, which this class cannot verify on its own and therefore states:
    the server must not be reachable except through an edge that authenticates every caller and
    **overwrites** ``X-Trellis-Identity`` on the way in. Given that, this resolver adds the
    checks that are local: the header must be present, small, valid JSON and name a tenant; the
    tenant must be one this deployment serves when ``allowed_tenants`` says which; and a
    deployment that needs a per-user identity says so with ``require_user``.

    ``require_authenticated`` turns that contract into something this process can check: with it on,
    a call is refused unless the deployment's own middleware authenticated it (the SDK's
    ``ServerCallContext.user``, populated by Starlette's ``AuthenticationMiddleware``, mTLS or a JWT
    middleware). Turn it on whenever the process authenticates as well as the edge — then a
    deployment that accidentally becomes reachable directly fails closed instead of believing a
    header a client set. It is off by default because the documented topology authenticates at the
    edge, where this process sees no principal at all.
    """

    name = "trusted-header"

    def __init__(
        self,
        *,
        allowed_tenants: Iterable[str] | None = None,
        header: str = IDENTITY_HEADER,
        require_user: bool = False,
        require_authenticated: bool = False,
    ) -> None:
        self.header = header.lower()
        self.allowed_tenants = frozenset(allowed_tenants) if allowed_tenants is not None else None
        self.require_user = require_user
        self.require_authenticated = require_authenticated
        #: Refusals are counted so a misconfigured edge is visible as a number, not only in logs.
        self.refused = 0

    def resolve(self, call_context: Any) -> dict[str, Any]:
        if self.require_authenticated and not _authenticated(call_context):
            self.refused += 1
            raise IdentityRefused("this deployment only trusts the header on an authenticated call")
        raw = _header(call_context, self.header)
        if not raw:
            self.refused += 1
            raise IdentityRefused("the call carries no platform identity header")
        if len(raw) > HEADER_MAX_CHARS:
            self.refused += 1
            raise IdentityRefused("the platform identity header is too long")
        try:
            claimed = json.loads(raw)
        except ValueError as exc:
            self.refused += 1
            raise IdentityRefused("the platform identity header is not JSON") from exc
        if not isinstance(claimed, Mapping):
            self.refused += 1
            raise IdentityRefused("the platform identity header must be a JSON object")
        identity = {
            field: str(claimed[field])
            for field in IDENTITY_FIELDS
            if claimed.get(field) not in (None, "")
        }
        tenant = identity.get("tenant_id")
        if not tenant:
            self.refused += 1
            raise IdentityRefused("the platform identity header names no tenant")
        if self.allowed_tenants is not None and tenant not in self.allowed_tenants:
            self.refused += 1
            raise IdentityRefused(f"tenant {tenant!r} is not served here")
        if self.require_user and not identity.get("user_id"):
            self.refused += 1
            raise IdentityRefused("this deployment requires the calling user's identity")
        return identity


def identity_headers(identity: Mapping[str, Any]) -> dict[str, str]:
    """The outbound form of the trusted context, for a caller the platform trusts.

    Used by :class:`~trellis.harness_a2a.client.A2AAgentClient`: the local run's tenant, user and
    workspace travel to the remote agent on the same header its own resolver reads.
    """
    fields = {f: str(identity[f]) for f in IDENTITY_FIELDS if identity.get(f) not in (None, "")}
    if not fields.get("tenant_id"):
        raise ValueError("an A2A call needs the caller's tenant")
    return {IDENTITY_HEADER: json.dumps(fields, separators=(",", ":"), sort_keys=True)}


def check_claimed_tenant(trusted: str, claimed: str | None) -> None:
    """Refuse a call that names a different tenant than the authenticated caller.

    A2A requests carry a tenant of their own, and the SDK lifts it onto the call context: from the
    request body on JSON-RPC (``jsonrpc_dispatcher``), from the path segment on the REST routes.
    Either way it is *not* identity here — the header is — but a call that contradicts the
    authenticated caller is a bug or an attempt, and answering it as if the two agreed is how
    tenant confusion starts. A call that names nothing is fine: it simply says less.
    """
    if claimed and claimed != trusted:
        raise IdentityRefused("the request names a different tenant than the authenticated caller")


def _authenticated(call_context: Any) -> bool:
    """Whether the deployment's own middleware authenticated this call.

    Read from the SDK's call context rather than from a header, because a header is the thing being
    checked. A context with no user at all is not authenticated.
    """
    user = getattr(call_context, "user", None)
    return bool(getattr(user, "is_authenticated", False))


def _header(call_context: Any, name: str) -> str:
    """One header from the SDK's call context, whatever shape it carries."""
    state = getattr(call_context, "state", None)
    headers = state.get("headers") if isinstance(state, Mapping) else None
    if not isinstance(headers, Mapping):
        return ""
    value = headers.get(name)
    if value is None:  # a caller-supplied mapping may not be lower-cased
        value = next((v for k, v in headers.items() if str(k).lower() == name), None)
    return str(value) if value is not None else ""


__all__ = [
    "EXTENSION_DESCRIPTION",
    "EXTENSION_URI",
    "HEADER_MAX_CHARS",
    "IDENTITY_FIELDS",
    "IDENTITY_HEADER",
    "FixedIdentity",
    "IdentityRefused",
    "IdentityResolver",
    "TrustedHeaderIdentity",
    "check_claimed_tenant",
    "identity_headers",
]
