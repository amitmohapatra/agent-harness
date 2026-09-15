"""Retry policy (§40).

Retries are *opt-in*, *category-gated* and *idempotency-gated*. Nothing is retried because
it merely failed: only categories the configuration names (timeout, rate limit, dependency)
and only when the wrapped agent has been declared idempotent. Authorization failures,
validation failures, policy denials and cancellation are never retried.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from typing import TypeVar

from universal_agent_harness.config.settings import RetryConfig
from universal_agent_harness.contracts.errors import AgentError, ErrorCategory

T = TypeVar("T")

#: Categories that are never retried, whatever the configuration says.
NEVER_RETRY = frozenset(
    {
        ErrorCategory.AUTHORIZATION,
        ErrorCategory.VALIDATION,
        ErrorCategory.POLICY,
        ErrorCategory.CANCELLED,
    }
)


class RetryPolicy:
    """Decides whether to try again, and how long to wait first."""

    def __init__(self, config: RetryConfig | None = None, *, idempotent: bool = False) -> None:
        self.config = config or RetryConfig()
        self.idempotent = idempotent
        self._categories = {ErrorCategory(c) for c in self.config.retry_categories}

    @property
    def max_attempts(self) -> int:
        return self.config.max_attempts if self.enabled else 1

    @property
    def enabled(self) -> bool:
        return self.config.enabled and self.idempotent

    def should_retry(self, error: AgentError, attempt: int) -> bool:
        if not self.enabled or attempt >= self.max_attempts:
            return False
        if error.category in NEVER_RETRY:
            return False
        return error.retryable and error.category in self._categories

    def backoff(self, attempt: int) -> float:
        cfg = self.config
        delay = min(
            cfg.initial_backoff_seconds * (cfg.backoff_multiplier ** (attempt - 1)),
            cfg.max_backoff_seconds,
        )
        return delay * (0.5 + random.random() / 2) if cfg.jitter else delay

    async def sleep(self, attempt: int) -> None:
        await asyncio.sleep(self.backoff(attempt))


async def with_retry[T](
    operation: Callable[[int], Awaitable[T]],
    policy: RetryPolicy,
    *,
    classify: Callable[[BaseException], AgentError],
    on_retry: Callable[[AgentError, int], None] | None = None,
) -> T:
    """Run ``operation(attempt)`` under ``policy``. Cancellation always propagates."""
    attempt = 1
    while True:
        try:
            return await operation(attempt)
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            error = classify(exc)
            if not policy.should_retry(error, attempt):
                raise
            if on_retry is not None:
                on_retry(error, attempt)
            await policy.sleep(attempt)
            attempt += 1
