"""Errors, timeouts, cancellation, retries (§37-§40, §77)."""

from __future__ import annotations

import asyncio

import pytest
from tests.support import span_by_name

from universal_agent_harness import (
    AgentHarness,
    AgentStatus,
    AgentTimeoutError,
    ErrorCategory,
    PolicyDeniedError,
)
from universal_agent_harness.policy.providers import AllowListPolicyProvider


class DomainError(Exception):
    """An application's own exception type."""


async def test_original_exception_type_is_preserved(harness, context):
    async def agent(payload):
        raise DomainError("inventory feed is stale")

    with pytest.raises(DomainError) as exc:
        await harness.wrap(agent, agent_id="inv")(None, context=context)
    # ...and the normalized error rides along for callers who want it
    assert exc.value.agent_error.category is ErrorCategory.UNKNOWN
    assert exc.value.agent_error.trace_id == context.trace_id


async def test_error_mode_result_returns_a_normalized_result(harness, context):
    async def agent(payload):
        raise DomainError("boom")

    result = await harness.wrap(agent, agent_id="inv", error_mode="result")(None, context=context)
    assert result.status is AgentStatus.ERROR
    assert result.error.code == "DomainError"
    assert result.error.message == "boom"


async def test_errors_are_recorded_on_the_span(harness, context, spans):
    async def agent(payload):
        raise DomainError("boom")

    with pytest.raises(DomainError):
        await harness.wrap(agent, agent_id="inv")(None, context=context)
    span = span_by_name(spans, "agent.run")
    assert span.attributes["status"] == "error"
    assert span.attributes["error.code"] == "DomainError"
    assert span.status.status_code.name == "ERROR"


async def test_timeout_produces_a_timeout_error(memory, context):
    harness = AgentHarness(
        memory=memory,
        defaults={"tenant_id": "acme"},
        config={"timeouts": {"default_seconds": 0.05}},
    )

    async def agent(payload):
        await asyncio.sleep(5)
        return "never"

    with pytest.raises(AgentTimeoutError):
        await harness.wrap(agent, agent_id="slow")(None, context=context)


async def test_timeout_result_mode_reports_timeout_status(memory, context):
    harness = AgentHarness(
        memory=memory,
        defaults={"tenant_id": "acme"},
        config={"timeouts": {"default_seconds": 0.05}},
    )

    async def agent(payload):
        await asyncio.sleep(5)

    result = await harness.wrap(agent, agent_id="slow", error_mode="result")(None, context=context)
    assert result.status is AgentStatus.TIMEOUT
    assert result.error.category is ErrorCategory.TIMEOUT
    assert result.error.retryable is True


async def test_per_call_timeout_overrides_the_default(harness, context):
    async def agent(payload):
        await asyncio.sleep(1)
        return "slow but fine"

    with pytest.raises(AgentTimeoutError):
        await harness.wrap(agent, agent_id="inv", timeout_seconds=0.05)(None, context=context)


async def test_timeout_cancels_the_agent_task(memory, context):
    cancelled = asyncio.Event()
    harness = AgentHarness(
        memory=memory,
        defaults={"tenant_id": "acme"},
        config={"timeouts": {"default_seconds": 0.05}},
    )

    async def agent(payload):
        try:
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    with pytest.raises(AgentTimeoutError):
        await harness.wrap(agent, agent_id="slow")(None, context=context)
    assert cancelled.is_set()


async def test_cancellation_propagates_and_is_never_swallowed(harness, context):
    started = asyncio.Event()

    async def agent(payload):
        started.set()
        await asyncio.sleep(5)

    task = asyncio.ensure_future(harness.wrap(agent, agent_id="inv")(None, context=context))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_cooperative_cancellation_token(harness, context):
    async def agent(payload, runtime):
        runtime.cancellation.cancel("operator")
        runtime.check_cancelled()
        return "unreachable"

    with pytest.raises(asyncio.CancelledError):
        await harness.wrap(agent, agent_id="inv")(None, context=context)


async def test_retries_are_off_by_default(harness, context):
    attempts = 0

    async def agent(payload):
        nonlocal attempts
        attempts += 1
        raise TimeoutError("transient")

    with pytest.raises(TimeoutError):
        await harness.wrap(agent, agent_id="inv")(None, context=context)
    assert attempts == 1


async def test_retries_need_both_configuration_and_an_idempotent_agent(memory, context):
    harness = AgentHarness(
        memory=memory,
        defaults={"tenant_id": "acme"},
        config={"retries": {"enabled": True, "max_attempts": 3, "backoff_seconds": 0.001}},
    )
    attempts = 0

    async def agent(payload):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise TimeoutError("transient")
        return "third time lucky"

    non_idempotent = harness.wrap(agent, agent_id="inv")
    with pytest.raises(TimeoutError):
        await non_idempotent(None, context=context)
    assert attempts == 1

    attempts = 0
    idempotent = harness.wrap(agent, agent_id="inv", idempotent=True)
    assert (await idempotent(None, context=context)).data == "third time lucky"
    assert attempts == 3


async def test_validation_errors_are_never_retried(memory, context):
    harness = AgentHarness(
        memory=memory,
        defaults={"tenant_id": "acme"},
        config={"retries": {"enabled": True, "max_attempts": 3}},
    )
    attempts = 0

    async def agent(payload):
        nonlocal attempts
        attempts += 1
        raise ValueError("bad input")

    with pytest.raises(ValueError):
        await harness.wrap(agent, agent_id="inv", idempotent=True)(None, context=context)
    assert attempts == 1


async def test_policy_denial_rejects_before_the_agent_runs(memory, context):
    ran = False
    harness = AgentHarness(
        memory=memory,
        defaults={"tenant_id": "acme"},
        policy=AllowListPolicyProvider(agents={"allowed-agent"}),
    )

    async def agent(payload):
        nonlocal ran
        ran = True
        return "ok"

    with pytest.raises(PolicyDeniedError, match="not in the allow list"):
        await harness.wrap(agent, agent_id="denied-agent")(None, context=context)
    assert ran is False


async def test_policy_denial_status_in_result_mode(memory, context):
    harness = AgentHarness(
        memory=memory,
        defaults={"tenant_id": "acme"},
        policy=AllowListPolicyProvider(agents=set()),
    )

    async def agent(payload):
        return "ok"

    result = await harness.wrap(agent, agent_id="x", error_mode="result")(None, context=context)
    assert result.status is AgentStatus.REJECTED
    assert result.error.category is ErrorCategory.POLICY


async def test_tool_policy_blocks_a_specific_tool(memory, context):
    async def dangerous(x: int) -> int:
        return x

    harness = AgentHarness(
        memory=memory,
        tools=[dangerous],
        defaults={"tenant_id": "acme"},
        policy=AllowListPolicyProvider(tools=set()),
    )

    async def agent(payload, runtime):
        with pytest.raises(PolicyDeniedError, match="dangerous"):
            await runtime.tools.call("dangerous", x=1)
        return "blocked"

    assert (await harness.wrap(agent, agent_id="inv")(None, context=context)).data == "blocked"


async def test_interceptor_failures_do_not_hide_the_original_error(harness, context):
    from universal_agent_harness import BaseInterceptor

    class Broken(BaseInterceptor):
        name, order = "broken", 95

        async def on_error(self, error, runtime):
            raise RuntimeError("interceptor is broken")

    harness.add_interceptor(Broken())

    async def agent(payload):
        raise DomainError("the real problem")

    with pytest.raises(DomainError, match="the real problem"):
        await harness.wrap(agent, agent_id="inv")(None, context=context)
