"""The Bifrost gateway as a :class:`ModelClient` (§16).

Bifrost (maximhq/bifrost) is an OpenAI-compatible LLM gateway: it holds the provider keys and
does routing, fallbacks and budgets behind ``POST {base_url}/chat/completions``. The harness
therefore knows a URL, a virtual key and a model *name* — no provider SDK is imported here,
which is what lets the same agent run against OpenAI, Anthropic or a local model by changing
a string.

Bounded by construction, because a chat node that hangs holds a graph open:

* one deadline per call, and the harness narrows it further to whatever the execution has
  left (``InstrumentedModelClient`` passes its own timeout);
* ``max_retries`` retries with exponential backoff, and **only** for failures that a retry
  can fix — a 400 is a bug in the request, so retrying it just spends the budget;
* a circuit breaker: after ``circuit_failure_threshold`` consecutive failures the next calls
  fail immediately for ``circuit_open_seconds`` instead of each paying the full timeout. A
  gateway outage costs one timeout, not one per request.

The first two of those are the ``bifrost-sdk`` client's, shared with the Memory Service; the
breaker and everything below it are the harness's. They were one lump of code here until the
same three bugs had to be fixed twice in two repositories on the same day.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator, Iterable, Sequence
from typing import Any

from bifrost_sdk import Bifrost, BifrostError, RateLimited, Unreachable
from universal_agent_contracts.errors import ConfigurationError, ModelError
from universal_agent_contracts.model import ModelRequest, ModelResponse, ModelUsage
from universal_agent_contracts.tool import ToolSpec


def tool_schemas(tools: Iterable[ToolSpec]) -> list[dict[str, Any]]:
    """Harness tool specs in the OpenAI ``tools`` shape the gateway expects.

    The harness owns one description of a tool; this projects it for the wire rather than
    asking applications to maintain a second copy that can drift from the executable one.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.input_schema
                or {"type": "object", "properties": {}, "additionalProperties": False},
            },
        }
        for spec in tools
    ]


class BifrostModelClient:
    """An OpenAI-compatible gateway, spoken over plain HTTP."""

    name = "bifrost"

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 2,
        backoff_seconds: float = 0.25,
        circuit_failure_threshold: int = 5,
        circuit_open_seconds: float = 15.0,
        default_params: dict[str, Any] | None = None,
        http_client: Any = None,
    ) -> None:
        if not base_url:
            raise ConfigurationError("BifrostModelClient: base_url is required")
        self.default_model = model
        self.provider = "bifrost"
        self.circuit_failure_threshold = circuit_failure_threshold
        self.circuit_open_seconds = circuit_open_seconds
        self.default_params = dict(default_params or {})
        #: Transport, retries and rate-limit handling come from the shared gateway client.
        #: What stays here is what is the *harness's*: the ModelRequest/ModelResponse
        #: contract, tool projection, and the circuit breaker. Keeping a second copy of the
        #: transport is what let the two drift — both read ``Retry-After`` from the header
        #: alone, so a provider that puts the delay in the body was retried on a backoff
        #: measured in milliseconds against a window measured in a minute.
        try:
            self._gateway = Bifrost(
                base_url,
                api_key=api_key,
                timeout=timeout,
                max_retries=max_retries,
                backoff_seconds=backoff_seconds,
                client=http_client,
            )
        except ImportError as exc:  # pragma: no cover - documented degradation
            raise ConfigurationError("BifrostModelClient needs httpx: pip install httpx") from exc
        self._consecutive_failures = 0
        self._circuit_open_until = 0.0

    # ------------------------------------------------------------------ port
    async def invoke(self, request: ModelRequest | str, /, **kwargs: Any) -> ModelResponse:
        req = _as_request(request, self.default_model)
        body = self._body(req, **kwargs)
        started = time.perf_counter()
        data = await self._chat(body, deadline=kwargs.pop("timeout", None))
        return self._response(data, req, started)

    async def structured(
        self, request: ModelRequest | str, /, schema: Any, **kwargs: Any
    ) -> ModelResponse:
        """A JSON answer that satisfies ``schema``.

        ``response_format`` is the gateway's job, but not every upstream provider honours it,
        so the parse is still checked here and one repair round is allowed: the invalid output
        and the reason go back as another turn. A model that cannot produce the shape twice is
        an error, not something to retry forever on the caller's deadline.
        """
        req = _as_request(request, self.default_model)
        body = self._body(req, **kwargs)
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "result", "schema": schema, "strict": True},
        }
        started = time.perf_counter()
        last: Exception | None = None
        for attempt in range(2):
            data = await self._chat(body, deadline=kwargs.get("timeout"))
            response = self._response(data, req, started)
            try:
                parsed = json.loads(_json_slice(response.text or ""))
            except ValueError as exc:
                last = ModelError(
                    f"gateway returned non-JSON structured output: {exc}", source="bifrost"
                )
                if attempt == 0:
                    body = _repair_turn(body, response.text or "", str(exc))
                continue
            return response.model_copy(update={"data": parsed})
        raise last or ModelError("structured output invalid")

    async def stream(self, request: ModelRequest | str, /, **kwargs: Any) -> AsyncIterator[str]:
        """Server-sent text deltas, yielded as they arrive.

        Nothing is buffered: a chat UI's time-to-first-token is the point of streaming, and
        collecting the whole answer before yielding would throw that away.
        """
        req = _as_request(request, self.default_model)
        body = self._body(req, **kwargs)
        params = {k: v for k, v in body.items() if k not in ("model", "messages", "tools")}
        try:
            async for delta in self._gateway.stream(
                body["messages"], model=body["model"], tools=body.get("tools"), **params
            ):
                yield delta
        except BifrostError as exc:
            # A stream is not retried — a partly consumed one cannot be replayed without
            # showing the caller duplicate text — so this is the only failure it can have.
            raise _as_model_error(exc) from exc

    async def ping(self) -> bool:
        """Whether the gateway answers at all. For readiness probes, not for the hot path."""
        return await self._gateway.ping()

    async def aclose(self) -> None:
        await self._gateway.aclose()

    # ------------------------------------------------------------------ internals
    def _body(self, req: ModelRequest, **kwargs: Any) -> dict[str, Any]:
        model = req.model or self.default_model
        if not model:
            raise ConfigurationError(
                "no model: pass model= to BifrostModelClient or set it on the request"
            )
        params = {**self.default_params, **req.params, **kwargs}
        params.pop("timeout", None)
        body: dict[str, Any] = {"model": model, "messages": _messages(req), **params}
        if req.tools:
            body["tools"] = req.tools
        return body

    def _response(self, data: dict[str, Any], req: ModelRequest, started: float) -> ModelResponse:
        choice = _first_choice(data)
        message = choice.get("message") or {}
        # A reasoning model spends the output budget on thinking before it emits any text, so
        # a `max_tokens` that looks generous returns 200 OK with no content and
        # finish_reason="length". Handing that back as a response with `text=None` makes a
        # graph node see "the model said nothing" and carry on. Measured against
        # gemini-3.6-flash: answering "Reply with exactly: OK" consumed 57 reasoning tokens,
        # so max_tokens=16 produced no text at all. A tool call with no text is a different
        # thing and is left alone.
        empty = not message.get("content") and not message.get("tool_calls")
        if empty and choice.get("finish_reason") == "length":
            usage = ModelUsage.extract(data)
            raise ModelError(
                "bifrost returned no content: the output budget was exhausted before any "
                "text was produced (raise max_tokens)",
                source="bifrost",
                details={
                    "finish_reason": "length",
                    "model": data.get("model") or req.model,
                    "usage": usage.model_dump() if hasattr(usage, "model_dump") else None,
                },
            )
        return ModelResponse(
            text=message.get("content"),
            raw=data,
            model=data.get("model") or req.model,
            provider=self.provider,
            finish_reason=choice.get("finish_reason"),
            usage=ModelUsage.extract(data),
            tool_calls=list(message.get("tool_calls") or []),
            latency_ms=round((time.perf_counter() - started) * 1000, 3),
        )

    async def _chat(self, body: dict[str, Any], *, deadline: float | None) -> dict[str, Any]:
        now = time.monotonic()
        if now < self._circuit_open_until:
            raise ModelError(
                "bifrost circuit open",
                details={"retry_after_seconds": round(self._circuit_open_until - now, 1)},
            )
        params = {k: v for k, v in body.items() if k not in ("model", "messages", "tools")}
        try:
            data = await self._gateway.complete(
                body["messages"],
                model=body["model"],
                tools=body.get("tools"),
                # the execution's remaining time, narrowed by InstrumentedModelClient
                timeout=deadline,
                **params,
            )
        except BifrostError as exc:
            # A 429 is the gateway working and asking for less, not the gateway being broken.
            # Counting it toward the breaker turns backpressure into an outage: measured on a
            # real run, 17 rate limits opened the circuit and the next 62 calls failed
            # instantly without a request ever being sent.
            self._failed(trips_circuit=not isinstance(exc, RateLimited))
            raise _as_model_error(exc) from exc
        self._consecutive_failures = 0
        return data

    def _failed(self, *, trips_circuit: bool = True) -> None:
        if not trips_circuit:
            return
        self._consecutive_failures += 1
        if self._consecutive_failures >= self.circuit_failure_threshold:
            self._circuit_open_until = time.monotonic() + self.circuit_open_seconds


# ---------------------------------------------------------------------- helpers
def _as_model_error(exc: BifrostError) -> ModelError:
    """The gateway client's failure, in the harness's own error vocabulary (§16).

    Only the status survives as text, because that is what callers match on and what the
    tests assert; the rest of what the gateway said stays in ``details`` for the log.
    """
    if isinstance(exc, Unreachable):
        return ModelError(f"bifrost unreachable ({exc})", source="bifrost", retryable=True)
    status = exc.details.get("status")
    if status is None:
        return ModelError("bifrost returned a non-JSON body", source="bifrost")
    return ModelError(f"bifrost returned {status}", source="bifrost", details=dict(exc.details))


def _as_request(request: ModelRequest | str, default_model: str | None) -> ModelRequest:
    if isinstance(request, ModelRequest):
        return request if request.model else request.model_copy(update={"model": default_model})
    return ModelRequest(prompt=request, model=default_model)


def _messages(req: ModelRequest) -> list[dict[str, Any]]:
    if req.messages:
        return list(req.messages)
    if req.prompt:
        return [{"role": "user", "content": req.prompt}]
    raise ConfigurationError("ModelRequest carries neither messages nor prompt")


def _first_choice(payload: dict[str, Any]) -> dict[str, Any]:
    choices: Sequence[Any] = payload.get("choices") or []
    return dict(choices[0]) if choices else {}


def _json_slice(text: str) -> str:
    """The JSON object inside a reply that may be fenced or prefaced with prose."""
    start, end = text.find("{"), text.rfind("}")
    return text[start : end + 1] if 0 <= start < end else text


def _repair_turn(body: dict[str, Any], text: str, reason: str) -> dict[str, Any]:
    return {
        **body,
        "messages": [
            *body["messages"],
            {"role": "assistant", "content": text[:4000]},
            {
                "role": "user",
                "content": f"That output was invalid: {reason}. Return only JSON matching "
                "the schema.",
            },
        ],
    }
