"""An agent is held to the output contract it published in the registry.

The schema describes ``AgentResponse.data`` — the only part that varies. The envelope
(status, claims, evidence, error) is fixed by the contract and identical for every agent, so
an entity restating it would be the contract copied once per agent, drifting from the day it
next changed.

Before this, ``ResultValidationError`` existed in the contracts and was never raised by
anything: an agent could declare a shape in the registry and return something else forever.
"""

from __future__ import annotations

from typing import Any

import pytest
from universal_agent_contracts.errors import ResultValidationError
from universal_agent_contracts.messages import AgentResponse, AgentStatus

from universal_agent_harness import AgentHarness

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"outcome": {"enum": ["refunded", "declined"]}},
    "required": ["outcome"],
}


class FakeRegistry:
    """Stands in for a bound registry. Only the one method the harness asks for."""

    name = "fake-registry"

    def __init__(self, schema: dict[str, Any] | None = SCHEMA) -> None:
        self.schema = schema
        self.asked: list[str] = []

    async def register(self, descriptor) -> None: ...

    async def heartbeat(self, descriptor, *, status: str = "healthy") -> None: ...

    async def output_schema(self, agent_id: str) -> dict[str, Any] | None:
        self.asked.append(agent_id)
        return self.schema


def build(client: Any = None, **config: Any) -> AgentHarness:
    return AgentHarness(
        registry=client,
        defaults={"tenant_id": "acme"},
        config={"memory": {"enabled": False}, **config},
    )


async def run(harness: AgentHarness, data: Any, status: AgentStatus = AgentStatus.SUCCESS):
    @harness.agent(agent_id="refund-agent")
    async def agent(state, runtime):
        return AgentResponse(status=status, data=data)

    return await agent({})


async def test_a_result_matching_the_declared_shape_passes() -> None:
    harness = build(FakeRegistry())
    result = await run(harness, {"outcome": "refunded"})
    assert result.status == AgentStatus.SUCCESS
    assert result.data == {"outcome": "refunded"}


async def test_a_result_violating_the_declared_shape_is_refused() -> None:
    """The schema exists so callers can rely on the shape without defending against it. A
    payload that quietly violates it is worse downstream than a loud failure here."""
    harness = build(FakeRegistry())
    with pytest.raises(ResultValidationError, match="output_schema"):
        await run(harness, {"outcome": "maybe"})


async def test_a_missing_required_field_is_refused() -> None:
    harness = build(FakeRegistry())
    with pytest.raises(ResultValidationError):
        await run(harness, {"refund_id": "r-1"})


async def test_the_schema_is_looked_up_by_the_qualified_agent_id() -> None:
    """Two products may each own a "refund-agent"; the lookup must not be ambiguous."""
    fake = FakeRegistry()
    harness = build(fake, registry={"product_key": "billing"})
    await run(harness, {"outcome": "refunded"})
    assert fake.asked == ["billing:refund-agent"]


async def test_an_agent_that_declared_nothing_is_not_constrained() -> None:
    """A conversational agent with a free-text contract is legitimate. Requiring a schema
    would push people into writing a meaningless one."""
    harness = build(FakeRegistry(schema=None))
    result = await run(harness, "anything at all")
    assert result.data == "anything at all"


async def test_an_unbound_deployment_pays_nothing() -> None:
    """No registry means no lookup, no jsonschema import, and no behaviour change."""
    harness = build()
    result = await run(harness, {"whatever": True})
    assert result.data == {"whatever": True}


async def test_a_failed_run_is_not_also_reported_as_a_schema_violation() -> None:
    """A run that already failed has no data to speak of, and a schema complaint on top
    would bury the real error."""
    harness = build(FakeRegistry())
    result = await run(harness, None, status=AgentStatus.ERROR)
    assert result.status == AgentStatus.ERROR


async def test_the_check_runs_before_an_oversized_payload_is_offloaded() -> None:
    """Once data is replaced by an artifact reference there is nothing left to validate, so
    an agent could escape its own contract simply by returning too much."""
    harness = build(FakeRegistry(), artifacts={"inline_max_bytes": 8})
    with pytest.raises(ResultValidationError):
        await run(harness, {"outcome": "definitely not one of the allowed values"})
