"""One deployment serves one product, and the product comes from configuration.

The failure this prevents is silent and slow: two teams each own an agent named
"refund-agent" — which the registry permits, since its constraint is
UNIQUE(product_id, type, name) — and the Memory Service keys a private memory as
``principal:{tenant}/agent:{agent_id}``, scoped by tenant alone. Bare ids inside one tenant
therefore share a memory scope, and the symptom is one agent slowly learning things the
other was told.
"""

from __future__ import annotations

import pytest
from universal_agent_contracts.ids import safe_id

from universal_agent_harness import AgentHarness
from universal_agent_harness.config.settings import env_overrides

TENANT = {"tenant_id": "acme"}


def harness(product: str | None = None) -> AgentHarness:
    config: dict = {"memory": {"enabled": False}}
    if product:
        config["registry"] = {"product_key": product}
    return AgentHarness(config=config, defaults=TENANT)


def test_the_product_comes_from_configuration_not_from_the_call_site() -> None:
    """Agents are declared bare so the same source can be deployed for two products, and a
    product rename is not a code change."""
    assert harness("billing").qualify("refund-agent") == "billing:refund-agent"
    assert harness("support").qualify("refund-agent") == "support:refund-agent"


def test_the_same_name_in_two_products_yields_two_identities() -> None:
    billing, support = harness("billing"), harness("support")
    assert billing.qualify("refund-agent") != support.qualify("refund-agent")


def test_no_product_key_means_no_qualification() -> None:
    """A single-product deployment with no registry should not be forced to invent a prefix."""
    assert harness().qualify("refund-agent") == "refund-agent"


def test_an_explicitly_qualified_id_is_left_alone() -> None:
    """A process serving several products names them itself; qualifying twice would produce
    billing:support:refund-agent and match nothing in the registry."""
    assert harness("billing").qualify("support:refund-agent") == "support:refund-agent"


def test_the_qualified_id_survives_id_sanitisation() -> None:
    """``safe_id`` rewrites ``/`` to ``-``, which would make billing/refund-agent
    indistinguishable from product "billing-refund"'s agent "agent". ``:`` survives, and a
    separator that does not survive cannot carry a guarantee."""
    qualified = harness("billing").qualify("refund-agent")
    assert safe_id(qualified) == qualified
    assert safe_id("billing/refund-agent") == "billing-refund-agent"


def test_the_descriptor_carries_the_qualified_id() -> None:
    """The descriptor is what reaches the registry hook and telemetry, so qualification has
    to happen before it is built, not at the edges afterwards."""
    assert harness("billing").describe("refund-agent").agent_id == "billing:refund-agent"


def test_it_is_configurable_from_the_environment() -> None:
    """A deployment property belongs in the deployment, so it has to be reachable without
    touching code."""
    env = env_overrides(
        {
            "UAH_REGISTRY_URL": "http://registry.test",
            "UAH_REGISTRY_PRODUCT_KEY": "billing",
            "UAH_REGISTRY_API_KEY": "k",
        }
    )
    assert env["registry"] == {
        "url": "http://registry.test",
        "product_key": "billing",
        "api_key": "k",
    }


async def test_the_execution_context_carries_the_qualified_id() -> None:
    """What memory and telemetry actually see. If qualification stopped at the descriptor,
    the context would still leak a bare id into the principal key."""
    h = harness("billing")
    async with h.execution(agent_id="refund-agent") as runtime:
        assert runtime.context.agent_id == "billing:refund-agent"
        assert runtime.descriptor.agent_id == "billing:refund-agent"


@pytest.mark.parametrize("product", ["billing", "support"])
async def test_two_deployments_of_one_agent_do_not_share_a_principal(product: str) -> None:
    h = harness(product)
    async with h.execution(agent_id="refund-agent") as runtime:
        assert runtime.context.agent_id == f"{product}:refund-agent"


# ----------------------------------------------------------------- wiring from config


def test_a_bound_deployment_gets_a_registry_client_without_asking() -> None:
    """Configure-once: a deployment that already stated its registry should not also have to
    construct a client in code."""
    h = AgentHarness(
        defaults=TENANT,
        config={
            "memory": {"enabled": False},
            "registry": {"url": "http://registry.test", "product_key": "billing", "api_key": "k"},
        },
    )
    assert h.registry.name == "ai-registry"
    assert h.registry_enabled


def test_an_unbound_deployment_stays_on_the_no_op() -> None:
    h = AgentHarness(defaults=TENANT, config={"memory": {"enabled": False}})
    assert h.registry.name == "noop"
    assert not h.registry_enabled


def test_partial_configuration_is_not_configuration() -> None:
    """A URL with no key would fail on the first call, at startup, reading as an outage
    rather than as the missing setting it is."""
    h = AgentHarness(
        defaults=TENANT,
        config={"memory": {"enabled": False}, "registry": {"url": "http://registry.test"}},
    )
    assert h.registry.name == "noop"


def test_an_explicit_registry_still_wins() -> None:
    """Passing one in code is how a test, or an unusual deployment, overrides config."""
    from universal_agent_harness.registry.client import InMemoryAgentRegistry

    h = AgentHarness(
        defaults=TENANT,
        registry=InMemoryAgentRegistry(),
        config={
            "memory": {"enabled": False},
            "registry": {"url": "http://registry.test", "product_key": "billing", "api_key": "k"},
        },
    )
    assert h.registry.name == "memory"
