"""Model-call contracts (§15/§16). Provider-neutral: no OpenAI/Anthropic/Bifrost types here."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from universal_agent_harness.contracts.artifacts import ArtifactRef


class ModelUsage(BaseModel):
    """Token and cost accounting, as far as the provider reports it."""

    model_config = ConfigDict(frozen=True, extra="allow")

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cached_input_tokens: int | None = None
    reasoning_tokens: int | None = None
    cost_usd: float | None = None

    @property
    def tokens(self) -> int | None:
        if self.total_tokens is not None:
            return self.total_tokens
        if self.input_tokens is None and self.output_tokens is None:
            return None
        return (self.input_tokens or 0) + (self.output_tokens or 0)

    @classmethod
    def extract(cls, payload: Any) -> ModelUsage | None:
        """Best-effort usage from a provider response (dict or object) (§15 "where available").

        Recognises the common spellings (``prompt_tokens``/``input_tokens``,
        ``completion_tokens``/``output_tokens``). Returns ``None`` when nothing is reported —
        the harness never invents token counts.
        """
        usage = _get(payload, "usage") if not _is_usage_like(payload) else payload
        if usage is None:
            return None
        fields = {
            "input_tokens": _first(usage, "input_tokens", "prompt_tokens"),
            "output_tokens": _first(usage, "output_tokens", "completion_tokens"),
            "total_tokens": _first(usage, "total_tokens"),
            "cached_input_tokens": _first(usage, "cached_input_tokens", "cache_read_input_tokens"),
            "reasoning_tokens": _first(usage, "reasoning_tokens"),
            "cost_usd": _first(usage, "cost_usd", "cost", "total_cost"),
        }
        known: dict[str, Any] = {}
        for name, value in fields.items():
            if not isinstance(value, int | float) or isinstance(value, bool):
                continue
            known[name] = float(value) if name == "cost_usd" else int(value)
        return cls(**known) if known else None


class ModelRequest(BaseModel):
    """One model invocation, as the harness sees it."""

    model_config = ConfigDict(extra="allow")

    model: str | None = None
    provider: str | None = None
    profile: str | None = None
    prompt_id: str | None = None
    prompt_version: str | None = None
    messages: list[dict[str, Any]] | None = None
    prompt: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    tools: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelResponse(BaseModel):
    """The normalized result of a model call. ``raw`` keeps the provider object for callers."""

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    text: str | None = None
    data: Any = None
    raw: Any = None
    model: str | None = None
    provider: str | None = None
    finish_reason: str | None = None
    usage: ModelUsage | None = None
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    fallback_used: bool = False
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    latency_ms: float | None = None

    @classmethod
    def coerce(cls, value: Any, *, request: ModelRequest | None = None) -> ModelResponse:
        """Wrap a provider response without losing it."""
        if isinstance(value, cls):
            return value
        usage = ModelUsage.extract(value)
        text = value if isinstance(value, str) else _text_of(value)
        return cls(
            text=text,
            data=None if isinstance(value, str) else value,
            raw=value,
            model=_get(value, "model") or (request.model if request else None),
            provider=request.provider if request else None,
            finish_reason=_first(value, "finish_reason", "stop_reason"),
            usage=usage,
        )


def _is_usage_like(payload: Any) -> bool:
    return any(
        _get(payload, k) is not None for k in ("input_tokens", "prompt_tokens", "total_tokens")
    )


def _get(obj: Any, key: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _first(obj: Any, *keys: str) -> Any:
    for key in keys:
        value = _get(obj, key)
        if value is not None:
            return value
    return None


def _text_of(value: Any) -> str | None:
    for key in ("text", "content", "output_text"):
        found = _get(value, key)
        if isinstance(found, str):
            return found
    return None
