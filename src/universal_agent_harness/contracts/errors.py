"""The standardized error model (§37) and the exceptions the harness raises.

Every failure that crosses the harness boundary is normalized into an :class:`AgentError`
so callers can branch on ``category``/``retryable`` instead of on a framework's exception
zoo. Cancellation is *never* normalized away: :class:`asyncio.CancelledError` propagates.
"""

from __future__ import annotations

import asyncio
import builtins
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ErrorCategory(StrEnum):
    VALIDATION = "VALIDATION"
    AUTHORIZATION = "AUTHORIZATION"
    MODEL = "MODEL"
    TOOL = "TOOL"
    MEMORY = "MEMORY"
    TIMEOUT = "TIMEOUT"
    CANCELLED = "CANCELLED"
    RATE_LIMIT = "RATE_LIMIT"
    DEPENDENCY = "DEPENDENCY"
    POLICY = "POLICY"
    UNKNOWN = "UNKNOWN"


#: Categories a retry policy may consider (§40). Everything else is never auto-retried.
RETRYABLE_CATEGORIES = frozenset(
    {ErrorCategory.TIMEOUT, ErrorCategory.RATE_LIMIT, ErrorCategory.DEPENDENCY}
)


class AgentError(BaseModel):
    """A normalized, serializable failure."""

    model_config = ConfigDict(frozen=True)

    code: str
    category: ErrorCategory = ErrorCategory.UNKNOWN
    message: str = ""
    retryable: bool = False
    source: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)
    trace_id: str | None = None

    @classmethod
    def of(
        cls,
        exc: BaseException,
        *,
        category: ErrorCategory | None = None,
        source: str | None = None,
        trace_id: str | None = None,
        retryable: bool | None = None,
    ) -> AgentError:
        """Normalize an exception. A :class:`HarnessError` carries its own classification."""
        if isinstance(exc, HarnessError):
            return exc.to_error(trace_id=trace_id, source=source or exc.source)
        cat = category or classify(exc)
        return cls(
            code=type(exc).__name__,
            category=cat,
            message=str(exc)[:2000],
            retryable=cat in RETRYABLE_CATEGORIES if retryable is None else retryable,
            source=source,
            trace_id=trace_id,
        )


def classify(exc: BaseException) -> ErrorCategory:
    """Best-effort category for an arbitrary exception.

    Known Memory Service SDK errors and stdlib timeouts are mapped explicitly; everything
    else is ``UNKNOWN`` (and therefore *not* retryable) rather than optimistically retried.
    """
    if isinstance(exc, asyncio.CancelledError):
        return ErrorCategory.CANCELLED
    if isinstance(exc, asyncio.TimeoutError | builtins.TimeoutError):
        return ErrorCategory.TIMEOUT
    if isinstance(exc, ValueError | TypeError | KeyError):
        return ErrorCategory.VALIDATION
    if isinstance(exc, PermissionError):
        return ErrorCategory.AUTHORIZATION
    return _sdk_category(exc)


def _sdk_category(exc: BaseException) -> ErrorCategory:
    """Map ``universal_memory`` SDK errors without importing the SDK eagerly."""
    name = type(exc).__name__
    mapping = {
        "AuthenticationError": ErrorCategory.AUTHORIZATION,
        "AuthorizationError": ErrorCategory.AUTHORIZATION,
        "ValidationError": ErrorCategory.VALIDATION,
        "ConflictError": ErrorCategory.VALIDATION,
        "NotFoundError": ErrorCategory.DEPENDENCY,
        "RateLimitedError": ErrorCategory.RATE_LIMIT,
        "DependencyUnavailableError": ErrorCategory.DEPENDENCY,
        "InsufficientEvidence": ErrorCategory.MEMORY,
        "MemoryError": ErrorCategory.MEMORY,
        "ConnectError": ErrorCategory.DEPENDENCY,
        "ConnectTimeout": ErrorCategory.TIMEOUT,
        "ReadTimeout": ErrorCategory.TIMEOUT,
        "PoolTimeout": ErrorCategory.TIMEOUT,
    }
    module = type(exc).__module__.split(".")[0]
    if module in ("universal_memory", "httpx") and name in mapping:
        return mapping[name]
    return ErrorCategory.UNKNOWN


# --------------------------------------------------------------------------- exceptions


class HarnessError(Exception):
    """Base class for failures the harness itself raises."""

    code = "HARNESS_ERROR"
    category = ErrorCategory.UNKNOWN
    retryable = False
    source: str | None = None

    def __init__(
        self,
        message: str = "",
        *,
        details: dict[str, Any] | None = None,
        source: str | None = None,
        retryable: bool | None = None,
    ) -> None:
        super().__init__(message or self.code)
        self.message = message or self.code
        self.details = details or {}
        if source is not None:
            self.source = source
        if retryable is not None:
            self.retryable = retryable

    def to_error(self, *, trace_id: str | None = None, source: str | None = None) -> AgentError:
        return AgentError(
            code=self.code,
            category=self.category,
            message=self.message,
            retryable=self.retryable,
            source=source or self.source,
            details=self.details,
            trace_id=trace_id,
        )


class ConfigurationError(HarnessError):
    code = "CONFIGURATION_ERROR"
    category = ErrorCategory.VALIDATION


class AgentTimeoutError(HarnessError):
    code = "AGENT_TIMEOUT"
    category = ErrorCategory.TIMEOUT
    retryable = True


class AgentCancelledError(HarnessError):
    code = "AGENT_CANCELLED"
    category = ErrorCategory.CANCELLED


class PolicyDeniedError(HarnessError):
    code = "POLICY_DENIED"
    category = ErrorCategory.POLICY


class MemoryUnavailableError(HarnessError):
    code = "MEMORY_UNAVAILABLE"
    category = ErrorCategory.MEMORY
    retryable = True


class ModelError(HarnessError):
    code = "MODEL_ERROR"
    category = ErrorCategory.MODEL


class ToolError(HarnessError):
    code = "TOOL_ERROR"
    category = ErrorCategory.TOOL


class ToolNotFoundError(ToolError):
    code = "TOOL_NOT_FOUND"
    category = ErrorCategory.VALIDATION


class ResultValidationError(HarnessError):
    code = "RESULT_VALIDATION_ERROR"
    category = ErrorCategory.VALIDATION
