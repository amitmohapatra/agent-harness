"""The AI Registry client, against a scripted registry.

The behaviours worth pinning are all about what happens when the registry is *not* perfectly
available — that is the whole point of the control-plane/data-plane split it is built on.
"""

from __future__ import annotations

import httpx
import pytest

from universal_agent_harness.registry.ai_registry import AIRegistryClient

MANIFEST = {
    "contract": "v1",
    "product_key": "billing",
    "seq": 7,
    "audiences": ["internal", "external"],
    "default_audience": "external",
    "entities": [
        {
            "id": "e1",
            "type": "agent",
            "name": "refund-agent",
            "version": 3,
            "views": {
                "internal": {"enabled": True, "spec": {"description": "issues refunds"}},
                "external": {"enabled": False},
            },
        },
        {
            "id": "e2",
            "type": "agent",
            "name": "public-agent",
            "version": 1,
            "views": {
                "internal": {"enabled": True, "spec": {"description": "public"}},
                "external": {"enabled": True, "spec": {"description": "public"}},
            },
        },
        {
            "id": "e3",
            "type": "tool",
            "name": "lookup",
            "version": 2,
            "views": {"external": {"enabled": True, "spec": {"description": "a tool"}}},
        },
    ],
}


def registry(handler) -> AIRegistryClient:
    transport = httpx.MockTransport(handler)
    return AIRegistryClient(
        "http://registry.test",
        product_key="billing",
        api_key="k",
        client=httpx.AsyncClient(transport=transport, base_url="http://registry.test"),
    )


def serving(payload=MANIFEST, *, etag='W/"7"'):
    calls: list[httpx.Request] = []

    def handler(request):
        calls.append(request)
        if request.headers.get("If-None-Match") == etag:
            return httpx.Response(304, headers={"ETag": etag})
        return httpx.Response(200, json=payload, headers={"ETag": etag})

    return calls, handler


async def test_the_api_key_goes_on_the_data_plane_header() -> None:
    calls, handler = serving()
    r = registry(handler)
    await r.manifest()
    assert calls[0].headers["x-api-key"] == "k"
    assert calls[0].url.path == "/v1/products/billing/manifest"
    await r.aclose()


async def test_an_unchanged_manifest_costs_a_304_not_a_refetch() -> None:
    """The ETag is what makes polling affordable when no Redis channel is configured."""
    calls, handler = serving()
    r = registry(handler)
    first = await r.manifest()
    second = await r.manifest()
    assert first == second
    assert len(calls) == 2
    assert calls[1].headers["If-None-Match"] == 'W/"7"'
    await r.aclose()


async def test_an_unreachable_registry_serves_last_known_good() -> None:
    """The registry is a control plane: losing it must degrade freshness, not availability.
    An agent that refused to run because the registry blinked would invert that."""
    state = {"up": True}

    def handler(request):
        if not state["up"]:
            raise httpx.ConnectError("refused")
        return httpx.Response(200, json=MANIFEST, headers={"ETag": 'W/"7"'})

    r = registry(handler)
    await r.manifest()
    state["up"] = False
    assert (await r.manifest())["seq"] == 7
    await r.aclose()


async def test_with_no_cache_an_unreachable_registry_is_an_error() -> None:
    """Last-known-good is only honest when there *is* a last known good. Returning an empty
    manifest on a cold start would look like 'this product has no agents'."""

    def handler(request):
        raise httpx.ConnectError("refused")

    r = registry(handler)
    with pytest.raises(httpx.ConnectError):
        await r.manifest()
    await r.aclose()


async def test_an_audience_only_sees_what_it_is_entitled_to() -> None:
    _, handler = serving()
    r = registry(handler)
    external = [a["name"] for a in await r.agents(audience="external")]
    internal = [a["name"] for a in await r.agents(audience="internal")]
    assert external == ["public-agent"]
    assert sorted(internal) == ["public-agent", "refund-agent"]
    await r.aclose()


async def test_an_unknown_audience_falls_back_to_the_default_not_to_everything() -> None:
    """A misspelled audience must not be the way an internal agent reaches a public surface."""
    _, handler = serving()
    r = registry(handler)
    assert [a["name"] for a in await r.agents(audience="typo")] == []
    await r.aclose()


async def test_the_resolved_view_is_used_verbatim() -> None:
    """Views are pre-resolved per audience by the registry. A client that merged overlays
    itself would be a second implementation of that logic, and the two would drift."""
    _, handler = serving()
    r = registry(handler)
    agent = (await r.agents(audience="internal"))[0]
    assert agent["version"] in (1, 3)
    assert "description" in agent
    await r.aclose()


async def test_tools_and_agents_are_separate_listings() -> None:
    _, handler = serving()
    r = registry(handler)
    assert [t["name"] for t in await r.tools(audience="external")] == ["lookup"]
    assert "lookup" not in [a["name"] for a in await r.agents(audience="external")]
    await r.aclose()


async def test_registration_is_refused_to_the_data_plane() -> None:
    """A process that could register itself would let a deployment add an agent nobody
    approved — the property the registry exists to prevent."""
    from universal_agent_contracts.descriptors import AgentDescriptor

    calls, handler = serving()
    r = registry(handler)
    await r.register(AgentDescriptor(agent_id="sneaky"))
    await r.heartbeat(AgentDescriptor(agent_id="sneaky"))
    assert calls == []
    await r.aclose()


async def test_it_refuses_to_be_built_without_credentials() -> None:
    with pytest.raises(ValueError, match="base_url, product_key and api_key"):
        AIRegistryClient("http://registry.test", product_key="billing", api_key="")


# ----------------------------------------------------------------- code <-> registry sync


async def test_an_agent_listed_and_implemented_is_bound() -> None:
    """The binding key is the NAME: harness agent_id == registry entity name, exactly as the
    MCP SDK binds a handler with @server.tool("get_invoice")."""
    _, handler = serving()
    r = registry(handler)
    state = await r.reconcile(["public-agent", "refund-agent"])
    assert state.bound == ["billing:public-agent", "billing:refund-agent"]
    assert state.in_sync
    await r.aclose()


async def test_a_registry_agent_with_no_handler_is_reported_not_served() -> None:
    """Fail-safe, never fail-crash. A deploy that dropped a handler must be loud without
    taking the rest of the process down — the same rule the MCP SDK applies to tools."""
    _, handler = serving()
    r = registry(handler)
    state = await r.reconcile(["public-agent"])
    assert state.unbound == ["billing:refund-agent"]
    assert state.bound == ["billing:public-agent"]
    assert not state.in_sync
    await r.aclose()


async def test_an_agent_implemented_but_not_listed_is_not_exposed() -> None:
    """The registry is the source of truth for exposure. Without this a deployment could
    ship an agent nobody approved and nobody would notice until it answered a customer."""
    _, handler = serving()
    r = registry(handler)
    state = await r.reconcile(["public-agent", "shadow-agent"])
    assert state.unregistered == ["billing:shadow-agent"]
    assert not state.in_sync
    await r.aclose()


async def test_reconciliation_ignores_audience_entirely() -> None:
    """An audience answers "who may call this?" — a property of a request, not a deployment.
    refund-agent is internal-only, yet the process still implements and is accountable for
    it. Filtering by the manifest's default audience would report it as unregistered on
    every single start, which is a false alarm that trains people to ignore real ones."""
    _, handler = serving()
    r = registry(handler)
    state = await r.reconcile(["refund-agent", "public-agent"])
    # internal-only, yet bound: the default audience would have called it unregistered
    assert "billing:refund-agent" in state.bound
    assert state.unregistered == []
    assert state.in_sync
    await r.aclose()


async def test_discovery_still_narrows_by_audience() -> None:
    """The per-request question keeps its answer: an external caller must not discover an
    internal agent just because the deployment is accountable for it."""
    _, handler = serving()
    r = registry(handler)
    assert [a["name"] for a in await r.agents(audience="external")] == ["public-agent"]
    assert {a["name"] for a in await r.all_agents()} == {"public-agent", "refund-agent"}
    await r.aclose()


# ----------------------------------------------------------------- uniqueness


def other_product(handler) -> AIRegistryClient:
    return AIRegistryClient(
        "http://registry.test",
        product_key="support",
        api_key="k",
        client=httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="http://registry.test"
        ),
    )


async def test_two_products_may_each_own_an_agent_of_the_same_name() -> None:
    """The registry's constraint is UNIQUE(product_id, type, name), so this is legal and
    teams will do it. Nothing downstream knows about products, so the identity that leaves
    the process has to carry one."""
    _, handler = serving()
    billing, support = registry(handler), other_product(handler)
    assert billing.qualified("refund-agent") == "billing:refund-agent"
    assert support.qualified("refund-agent") == "support:refund-agent"
    assert billing.qualified("refund-agent") != support.qualified("refund-agent")
    await billing.aclose()
    await support.aclose()


async def test_the_separator_survives_id_sanitisation() -> None:
    """``safe_id`` rewrites ``/`` to ``-``, which would make billing/refund-agent
    indistinguishable from product "billing-refund"'s agent "agent". ``:`` survives."""
    from universal_agent_contracts.ids import safe_id

    _, handler = serving()
    r = registry(handler)
    qualified = r.qualified("refund-agent")
    assert safe_id(qualified) == qualified
    assert safe_id("billing/refund-agent") != "billing/refund-agent"
    await r.aclose()


async def test_an_already_qualified_name_is_left_alone() -> None:
    """A process serving several products declares fully-qualified ids itself; qualifying
    them twice would produce billing:support:refund-agent and match nothing."""
    _, handler = serving()
    r = registry(handler)
    assert r.qualified("support:refund-agent") == "support:refund-agent"
    await r.aclose()


async def test_discovery_hands_back_the_id_the_harness_must_use() -> None:
    """A developer should never have to build the qualified id by hand — getting it wrong
    is silent, and the symptom appears later as two agents sharing a memory scope."""
    _, handler = serving()
    r = registry(handler)
    found = await r.agents(audience="internal")
    assert {a["agent_id"] for a in found} == {"billing:public-agent", "billing:refund-agent"}
    assert {a["name"] for a in found} == {"public-agent", "refund-agent"}
    await r.aclose()
