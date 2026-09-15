"""Concrete :class:`ModelClient` implementations (§16).

``DirectModelClient`` adapts whatever the application already has — an async callable, a
sync callable, or an object exposing ``invoke``/``ainvoke``/``acomplete`` — into the port.
That is the "priority 2" path of §59: a provider wrapper, chosen explicitly, rather than
global monkeypatching (§17).
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterator, Callable
from typing import Any

from universal_agent_harness.contracts.errors import ConfigurationError, ModelError
from universal_agent_harness.contracts.model import ModelRequest, ModelResponse

#: Method names, in preference order, that a "model-like" object may expose.
ASYNC_METHODS = ("ainvoke", "acomplete", "agenerate", "acreate", "invoke", "complete", "generate")
STREAM_METHODS = ("astream", "stream")


class DirectModelClient:
    """Wraps a callable or a model object. No provider SDK is imported here."""

    name = "direct"

    def __init__(
        self,
        target: Callable[..., Any] | Any,
        *,
        model: str | None = None,
        provider: str | None = None,
        structured_target: Callable[..., Any] | None = None,
        stream_target: Callable[..., Any] | None = None,
    ) -> None:
        self._target = target
        self._structured = structured_target
        self._stream = stream_target or _find(target, STREAM_METHODS)
        self.default_model = model
        self.provider = provider or _infer_provider(target)
        resolved = target if callable(target) else _find(target, ASYNC_METHODS)
        if resolved is None:
            raise ConfigurationError(
                f"{target!r} is not usable as a model client: pass a callable or an object "
                f"exposing one of {ASYNC_METHODS}"
            )
        self._call: Callable[..., Any] = resolved

    async def invoke(self, request: ModelRequest | str, /, **kwargs: Any) -> ModelResponse:
        req = _as_request(request, self.default_model, self.provider)
        raw = await _maybe_await(self._call(*_call_args(req), **kwargs))
        return ModelResponse.coerce(raw, request=req)

    async def structured(
        self, request: ModelRequest | str, /, schema: Any, **kwargs: Any
    ) -> ModelResponse:
        req = _as_request(request, self.default_model, self.provider)
        target = self._structured or _find(self._target, ("astructured", "structured"))
        if target is None:
            raw = await _maybe_await(self._call(*_call_args(req), schema=schema, **kwargs))
        else:
            raw = await _maybe_await(target(*_call_args(req), schema=schema, **kwargs))
        response = ModelResponse.coerce(raw, request=req)
        return response if response.data is not None else response.model_copy(update={"data": raw})

    def stream(self, request: ModelRequest | str, /, **kwargs: Any) -> AsyncIterator[Any]:
        if self._stream is None:
            raise ModelError(f"model client {self.provider or self._target!r} does not stream")
        req = _as_request(request, self.default_model, self.provider)
        return _as_async_iterator(self._stream(*_call_args(req), **kwargs))


class UnconfiguredModelClient:
    """The default. Fails loudly and usefully instead of pretending to be a model."""

    name = "unconfigured"
    default_model = None
    provider = None

    async def invoke(self, request: ModelRequest | str, /, **kwargs: Any) -> ModelResponse:
        raise ConfigurationError(
            "no model client is configured; pass model=... to AgentHarness(...) or call your "
            "provider directly (model calls you make yourself are not instrumented)"
        )

    async def structured(
        self, request: ModelRequest | str, /, schema: Any, **kwargs: Any
    ) -> ModelResponse:
        return await self.invoke(request)

    def stream(self, request: ModelRequest | str, /, **kwargs: Any) -> AsyncIterator[Any]:
        raise ConfigurationError("no model client is configured")


def _as_request(value: ModelRequest | str, model: str | None, provider: str | None) -> ModelRequest:
    req = ModelRequest(prompt=value) if isinstance(value, str) else value
    updates: dict[str, Any] = {}
    if req.model is None and model:
        updates["model"] = model
    if req.provider is None and provider:
        updates["provider"] = provider
    return req.model_copy(update=updates) if updates else req


def _call_args(request: ModelRequest) -> tuple[Any, ...]:
    """Call the target with the most natural argument: messages if present, else the prompt."""
    if request.messages is not None:
        return (request.messages,)
    if request.prompt is not None:
        return (request.prompt,)
    return (request,)


def _find(target: Any, names: tuple[str, ...]) -> Callable[..., Any] | None:
    for name in names:
        candidate = getattr(target, name, None)
        if callable(candidate):
            return candidate
    return None


def _infer_provider(target: Any) -> str | None:
    module = type(target).__module__.split(".")[0]
    return module if module not in ("builtins", "functools", "types", "__main__") else None


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _as_async_iterator(value: Any) -> AsyncIterator[Any]:
    if hasattr(value, "__aiter__"):
        return value.__aiter__()
    if inspect.isawaitable(value):

        async def _awaited() -> AsyncIterator[Any]:
            resolved = await value
            async for chunk in _as_async_iterator(resolved):
                yield chunk

        return _awaited()

    async def _sync() -> AsyncIterator[Any]:
        for chunk in value:
            yield chunk
            await asyncio.sleep(0)

    return _sync()
