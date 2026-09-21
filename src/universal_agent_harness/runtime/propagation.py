"""Context propagation (§33).

*Within* a process, ``contextvars`` carry the current context/runtime as a convenience so
tools and helper functions can find them without threading arguments through. *Across*
services, W3C Trace Context (OpenTelemetry's propagator) is the contract — and nothing
sensitive ever goes into baggage.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import TYPE_CHECKING, Any

from universal_agent_contracts.context import AgentExecutionContext

if TYPE_CHECKING:  # pragma: no cover
    from universal_agent_harness.runtime.agent_runtime import AgentRuntime

_context: ContextVar[AgentExecutionContext | None] = ContextVar("uah_context", default=None)
_runtime: ContextVar[Any] = ContextVar("uah_runtime", default=None)

#: Identity that may cross a service boundary in baggage. Ids only — never content,
#: never a principal's name, never credentials (§33).
BAGGAGE_FIELDS = ("tenant_id", "agent_id", "agent_run_id", "correlation_id")


def current_context() -> AgentExecutionContext | None:
    """The execution context bound in this async task, if any."""
    return _context.get()


def current_runtime() -> AgentRuntime | None:
    """The runtime bound in this async task, if any."""
    return _runtime.get()


def require_runtime() -> AgentRuntime:
    runtime = _runtime.get()
    if runtime is None:
        raise RuntimeError(
            "no AgentRuntime is bound here; call this inside a harness-wrapped agent "
            "or pass the runtime explicitly"
        )
    return runtime


@contextmanager
def bind(context: AgentExecutionContext | None = None, runtime: Any = None) -> Iterator[None]:
    """Bind context/runtime for the duration of the block, restoring previous values after."""
    tokens: list[tuple[ContextVar[Any], Token[Any]]] = []
    if context is not None:
        tokens.append((_context, _context.set(context)))
    if runtime is not None:
        tokens.append((_runtime, _runtime.set(runtime)))
    try:
        yield
    finally:
        for var, token in reversed(tokens):
            var.reset(token)


def trace_headers(context: AgentExecutionContext | None = None) -> dict[str, str]:
    """W3C Trace Context headers for an outbound call, plus harness correlation ids.

    Uses the configured OpenTelemetry propagator so whatever the application installed
    (tracecontext, b3, ...) is honoured.
    """
    headers: dict[str, str] = {}
    try:
        # Propagation is best effort: a process without a propagator still runs.
        from opentelemetry.propagate import inject  # noqa: PLC0415

        inject(headers)
    except Exception:  # pragma: no cover - propagation is best-effort
        pass
    ctx = context or current_context()
    if ctx is not None:
        headers.setdefault("x-request-id", ctx.request_id)
        headers.setdefault("x-correlation-id", ctx.correlation_id)
    return headers


def baggage_fields(context: AgentExecutionContext) -> dict[str, str]:
    """The safe subset of identity for cross-service baggage."""
    return {f: str(getattr(context, f)) for f in BAGGAGE_FIELDS if getattr(context, f, None)}


def extract_trace_id(carrier: Mapping[str, str] | None = None) -> str | None:
    """The current (or carrier's) OpenTelemetry trace id as a hex string."""
    from opentelemetry import trace  # noqa: PLC0415

    if carrier:
        try:
            from opentelemetry.propagate import extract  # noqa: PLC0415

            ctx = extract(dict(carrier))
            span = trace.get_current_span(ctx)
        except Exception:  # pragma: no cover
            span = trace.get_current_span()
    else:
        span = trace.get_current_span()
    span_context = span.get_span_context()
    if span_context and span_context.trace_id:
        return format(span_context.trace_id, "032x")
    return None
