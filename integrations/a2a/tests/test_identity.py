"""The trust boundary: identity comes from the platform's header, never from the message, and a
call this deployment cannot place is refused rather than defaulted."""

from __future__ import annotations

import json

import pytest
from a2a.server.context import ServerCallContext

from trellis.harness_a2a import (
    EXTENSION_URI,
    IDENTITY_HEADER,
    FixedIdentity,
    IdentityRefused,
    TrustedHeaderIdentity,
    identity_headers,
)
from trellis.harness_a2a.identity import HEADER_MAX_CHARS, check_claimed_tenant

ACME = {"tenant_id": "acme", "user_id": "u1", "workspace_id": "ws1"}


def call_context(**headers: str) -> ServerCallContext:
    """A call context shaped as the SDK builds one (lower-cased headers in ``state``)."""
    return ServerCallContext(state={"headers": {k.lower(): v for k, v in headers.items()}})


def test_the_trusted_header_places_the_caller() -> None:
    resolver = TrustedHeaderIdentity(allowed_tenants={"acme"})
    context = call_context(**identity_headers(ACME))
    assert resolver.resolve(context) == ACME


def test_a_foreign_tenant_on_the_header_is_refused() -> None:
    resolver = TrustedHeaderIdentity(allowed_tenants={"acme"})
    context = call_context(**identity_headers({"tenant_id": "globex"}))
    with pytest.raises(IdentityRefused, match="not served here"):
        resolver.resolve(context)
    assert resolver.refused == 1


@pytest.mark.parametrize(
    ("header", "reason"),
    [
        ("", "no platform identity header"),
        ("not json", "not JSON"),
        ('["acme"]', "must be a JSON object"),
        ("{}", "names no tenant"),
        (json.dumps({"tenant_id": ""}), "names no tenant"),
        ("x" * (HEADER_MAX_CHARS + 1), "too long"),
    ],
)
def test_a_header_that_cannot_be_trusted_is_refused(header: str, reason: str) -> None:
    resolver = TrustedHeaderIdentity()
    context = call_context(**{IDENTITY_HEADER: header}) if header else call_context()
    with pytest.raises(IdentityRefused, match=reason):
        resolver.resolve(context)


def test_unknown_fields_in_the_header_are_dropped() -> None:
    resolver = TrustedHeaderIdentity()
    context = call_context(
        **{IDENTITY_HEADER: json.dumps({**ACME, "role": "admin", "tenant": "globex"})}
    )
    assert resolver.resolve(context) == ACME


def test_a_deployment_may_require_the_calling_user() -> None:
    resolver = TrustedHeaderIdentity(require_user=True)
    with pytest.raises(IdentityRefused, match="calling user"):
        resolver.resolve(call_context(**identity_headers({"tenant_id": "acme"})))
    assert resolver.resolve(call_context(**identity_headers(ACME)))["user_id"] == "u1"


def test_a_header_with_any_casing_is_read() -> None:
    """Deployments and proxies differ; the SDK lower-cases, a test double might not."""
    resolver = TrustedHeaderIdentity()
    context = ServerCallContext(state={"headers": {"X-Trellis-Identity": json.dumps(ACME)}})
    assert resolver.resolve(context)["tenant_id"] == "acme"


def test_a_fixed_identity_ignores_the_header_entirely() -> None:
    """A single-tenant server cannot be talked into another tenant by guessing a header name."""
    resolver = FixedIdentity("acme", user_id="service")
    seen = resolver.resolve(call_context(**identity_headers({"tenant_id": "globex"})))
    assert seen == {"tenant_id": "acme", "user_id": "service"}
    with pytest.raises(ValueError, match="needs a tenant_id"):
        FixedIdentity("")


def test_a_context_without_headers_is_not_an_identity() -> None:
    resolver = TrustedHeaderIdentity()
    with pytest.raises(IdentityRefused):
        resolver.resolve(ServerCallContext())


def test_the_outbound_header_is_compact_and_needs_a_tenant() -> None:
    headers = identity_headers(ACME)
    assert list(headers) == [IDENTITY_HEADER]
    assert json.loads(headers[IDENTITY_HEADER]) == ACME
    assert " " not in headers[IDENTITY_HEADER]
    with pytest.raises(ValueError, match="needs the caller's tenant"):
        identity_headers({"user_id": "u1"})


def test_a_transport_tenant_that_contradicts_the_caller_is_refused() -> None:
    check_claimed_tenant("acme", None)
    check_claimed_tenant("acme", "acme")
    with pytest.raises(IdentityRefused, match="different tenant"):
        check_claimed_tenant("acme", "globex")


def test_the_extension_uri_is_versioned() -> None:
    assert EXTENSION_URI.startswith("https://") and EXTENSION_URI.endswith("/v1")


def test_a_deployment_may_require_the_call_itself_to_be_authenticated() -> None:
    """The header's integrity rests on an edge stripping it. With this on, a process that becomes
    reachable directly fails closed instead of believing a header a client set."""

    from a2a.auth.user import User

    class Principal(User):
        @property
        def is_authenticated(self) -> bool:
            return True

        @property
        def user_name(self) -> str:
            return "edge"

    resolver = TrustedHeaderIdentity(require_authenticated=True)
    unauthenticated = call_context(**identity_headers(ACME))
    with pytest.raises(IdentityRefused, match="authenticated call"):
        resolver.resolve(unauthenticated)
    assert resolver.refused == 1
    authenticated = ServerCallContext(state={"headers": identity_headers(ACME)}, user=Principal())
    assert resolver.resolve(authenticated) == ACME
    # off by default: the documented topology authenticates at the edge, out of this process
    assert TrustedHeaderIdentity().resolve(unauthenticated) == ACME
