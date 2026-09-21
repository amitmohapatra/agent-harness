"""``harness.wrap_tool`` — instrumentation for a tool the developer calls directly (§13).

This is priority 2 of the instrumentation ladder: the harness cannot intercept an arbitrary
unwrapped Python function, but a wrapped one is instrumented wherever it is called from,
including inside a framework's own tool node, as long as an execution is in scope.

Outside an execution the wrapper is a pass-through: the tool still works, it is simply not
instrumented, and that is stated rather than silently implied.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from typing import Any, overload

from universal_agent_contracts.tool import ToolCall, ToolOutcome, ToolSpec

from universal_agent_harness.runtime.propagation import current_runtime
from universal_agent_harness.tools.local import LocalToolClient


@overload
def wrap_tool[F: Callable[..., Any]](fn: F, /) -> F: ...
@overload
def wrap_tool[F: Callable[..., Any]](
    *, name: str | None = ..., spec: ToolSpec | None = ..., **spec_fields: Any
) -> Callable[[F], F]: ...


def wrap_tool(
    fn: Callable[..., Any] | None = None,
    /,
    *,
    name: str | None = None,
    spec: ToolSpec | None = None,
    unwrap: bool = True,
    **spec_fields: Any,
) -> Any:
    """Wrap a tool callable so its calls are traced, metered and recorded.

    ``unwrap=True`` (the default) keeps the wrapped function's return type: callers get the
    tool's own value, not a :class:`ToolOutcome`. Pass ``unwrap=False`` to receive the
    outcome (status, latency, artifacts) instead.
    """

    def decorate(target: Callable[..., Any]) -> Callable[..., Any]:
        tool_name = name or spec.name if spec else name or getattr(target, "__name__", "tool")
        registry = LocalToolClient()
        tool_spec = registry.register(target, name=tool_name, spec=spec, **spec_fields)

        async def _run(args: dict[str, Any]) -> Any:
            runtime = current_runtime()
            call = ToolCall(tool=tool_spec.name, args=args)
            if runtime is None:
                result = target(**args)
                return await result if inspect.isawaitable(result) else result
            # Deferred: the instrumented client imports the runtime, which imports this.
            from universal_agent_harness.tools.client import InstrumentedToolClient  # noqa: PLC0415

            client = InstrumentedToolClient(
                registry,
                runtime_ref=runtime,
                timeout=getattr(runtime.tools, "timeout", None),
                policy=getattr(runtime.tools, "_policy", None),
                events=getattr(runtime.tools, "_events", None),
                record_to_memory=getattr(runtime.tools, "record_to_memory", True),
            )
            outcome = await client.call(call)
            return outcome.output if unwrap else outcome

        if inspect.iscoroutinefunction(target):

            @functools.wraps(target)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                return await _run(_bind(target, args, kwargs))

            wrapper: Callable[..., Any] = async_wrapper
        else:

            @functools.wraps(target)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                from universal_agent_harness.execution.sync import run_sync  # noqa: PLC0415

                if current_runtime() is None:
                    return target(*args, **kwargs)
                return run_sync(_run(_bind(target, args, kwargs)))

            wrapper = sync_wrapper

        wrapper.tool_spec = tool_spec  # type: ignore[attr-defined]
        wrapper.__wrapped_tool__ = target  # type: ignore[attr-defined]
        return wrapper

    return decorate(fn) if fn is not None else decorate


def _bind(
    target: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]
) -> dict[str, Any]:
    """Positional arguments become named ones so a tool call records argument *names*."""
    if not args:
        return dict(kwargs)
    try:
        bound = inspect.signature(target).bind(*args, **kwargs)
        bound.apply_defaults()
        return dict(bound.arguments)
    except (TypeError, ValueError):  # pragma: no cover - *args-style tools
        return {"args": list(args), **kwargs}


def unwrap_outcome(value: Any) -> Any:
    return value.output if isinstance(value, ToolOutcome) else value
