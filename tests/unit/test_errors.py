"""Error normalization (§37) and the never-retry rules (§40)."""

from __future__ import annotations

import asyncio

import pytest
from universal_agent_contracts.errors import (
    AgentTimeoutError,
    PolicyDeniedError,
    classify,
)

from universal_agent_harness import AgentError, ErrorCategory
from universal_agent_harness.config.settings import RetryConfig
from universal_agent_harness.execution.retry import NEVER_RETRY, RetryPolicy


@pytest.mark.parametrize(
    ("exc", "category"),
    [
        (asyncio.CancelledError(), ErrorCategory.CANCELLED),
        (TimeoutError(), ErrorCategory.TIMEOUT),
        (ValueError("bad"), ErrorCategory.VALIDATION),
        (PermissionError("no"), ErrorCategory.AUTHORIZATION),
        (RuntimeError("?"), ErrorCategory.UNKNOWN),
    ],
)
def test_classification(exc, category):
    assert classify(exc) is category


def test_sdk_errors_are_classified_by_module_and_name():
    from universal_memory.errors import RateLimitedError

    assert classify(RateLimitedError("slow down")) is ErrorCategory.RATE_LIMIT


def test_unknown_errors_are_not_retryable():
    error = AgentError.of(RuntimeError("boom"))
    assert error.category is ErrorCategory.UNKNOWN
    assert error.retryable is False


def test_harness_errors_carry_their_own_classification():
    error = AgentError.of(AgentTimeoutError("too slow"), trace_id="abc")
    assert error.code == "AGENT_TIMEOUT"
    assert error.category is ErrorCategory.TIMEOUT
    assert error.retryable is True
    assert error.trace_id == "abc"


def test_policy_denial_is_never_retried():
    policy = RetryPolicy(RetryConfig(enabled=True, max_attempts=5), idempotent=True)
    error = AgentError.of(PolicyDeniedError("nope"))
    assert error.category in NEVER_RETRY
    assert policy.should_retry(error, attempt=1) is False


def test_retry_requires_idempotency_and_enabled_config():
    error = AgentError(code="X", category=ErrorCategory.TIMEOUT, retryable=True)
    assert RetryPolicy(RetryConfig(enabled=True), idempotent=False).should_retry(error, 1) is False
    assert RetryPolicy(RetryConfig(enabled=False), idempotent=True).should_retry(error, 1) is False
    assert RetryPolicy(RetryConfig(enabled=True), idempotent=True).should_retry(error, 1) is True


def test_backoff_is_bounded():
    policy = RetryPolicy(
        RetryConfig(enabled=True, max_attempts=8, backoff_seconds=1), idempotent=True
    )
    assert all(policy.backoff(n) <= 30.0 for n in range(1, 9))
    assert policy.backoff(1) <= policy.backoff(4)  # exponential, with jitter
