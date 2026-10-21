"""What the harness knows of a model named by a string (a gateway model name): its vendor and
its context window — what ``ReAct`` builds its client with and scales its limits by.

**The window** is the ``context_window`` given (code wins), else the model's LangChain profile
— the model object's, or the one LangChain bundles for a vendor's model name — never fetched;
when none says, :data:`DEFAULT_WINDOW`, said once in the log. Where the number came from is
logged once per model. :func:`serves` asks the gateway, once per process, waiting at most
:data:`PROBE_SECONDS`.

**The vendor**, of a name the gateway routes to a provider: ``anthropic/...`` or ``openai/...``
(a bare ``claude-...`` or ``gpt-...`` name too). Another provider in front of a vendor's model
(``openrouter/openai/...``) is not that vendor: the gateway does not serve it on the vendor's
own API. :func:`serves` says whether the gateway serves a vendor's API at all.
"""

from __future__ import annotations

import functools
import importlib
import logging
from dataclasses import dataclass
from typing import Any, Final, Literal, TypeGuard

import httpx

log = logging.getLogger("trellis.models")

#: The window assumed when nothing says (the most common window of current hosted models).
DEFAULT_WINDOW: Final = 128_000
#: How long a probe of the gateway may take, in seconds.
PROBE_SECONDS: Final = 2.0
#: Every effort level, least first.
EFFORTS: Final = ("none", "minimal", "low", "medium", "high", "xhigh", "max")

Vendor = Literal["anthropic", "openai"]
Source = Literal["given", "profile", "default"]
#: A vendor's own gateway provider name, and the bare model names that are its.
VENDOR_NAMES: Final[dict[Vendor, tuple[str, ...]]] = {
    "anthropic": ("claude",),
    "openai": ("gpt", "o1", "o3", "o4", "chatgpt"),
}


@dataclass(frozen=True)
class Window:
    """A model's context window, in tokens, and what said so."""

    tokens: int
    source: Source


def bare(name: str) -> str:
    """A gateway model name's last part: the model's own name."""
    return name.rsplit("/", 1)[-1]


def vendor_of(name: str) -> Vendor | None:
    """The vendor whose own API serves ``name`` through the gateway, if any."""
    provider = name.split("/", 1)[0] if "/" in name else None
    if provider is not None:
        return provider if provider in VENDOR_NAMES else None  # type: ignore[return-value]
    return next(
        (v for v, prefixes in VENDOR_NAMES.items() if name.startswith(prefixes)),
        None,
    )


def window_of(name: str | None, *, given: int | None = None, profile: int | None = None) -> Window:
    """The window of the model ``name`` (or of a model object whose ``profile`` says): the
    ``given`` one, else its LangChain profile's (bundled with LangChain, never fetched), else
    :data:`DEFAULT_WINDOW`, said once. Where it came from is logged once per model."""
    if given is not None:
        found = Window(given, "given")
    elif profile is not None:
        found = Window(profile, "profile")
    else:
        listed = bundled_profile(name).get("max_input_tokens") if name else None
        found = Window(listed, "profile") if _tokens(listed) else None  # type: ignore[arg-type]
    if found is None:
        _assumed(name or "the model")
        return Window(DEFAULT_WINDOW, "default")
    _said(name or "the model", found.tokens, found.source)
    return found


def bundled_profile(name: str) -> dict[str, Any]:
    """The LangChain profile bundled for a vendor's model name (``anthropic/...``,
    ``openai/...``, a bare ``claude-...`` or ``gpt-...``): its window, its effort levels...
    Empty for any other model."""
    vendor = vendor_of(name)
    modules = {
        "anthropic": "langchain_anthropic.chat_models",
        "openai": "langchain_openai.chat_models.base",
    }
    if vendor is None:
        return {}
    profiles = importlib.import_module(modules[vendor])  # the react extra installs both
    return dict(profiles._get_default_model_profile(bare(name)) or {})


def effort_levels(model: Any) -> tuple[str, ...]:
    """The reasoning effort levels a chat model takes, as its LangChain profile lists them
    (``reasoning_effort_levels``); none when it lists none."""
    profile = getattr(model, "profile", None)
    listed = profile.get("reasoning_effort_levels") if isinstance(profile, dict) else None
    return tuple(str(v) for v in listed if v in EFFORTS) if isinstance(listed, list | tuple) else ()


def nearest(level: str, levels: tuple[str, ...]) -> str:
    """``level`` if the model takes it, else the one it takes closest to it (the lower on a
    tie)."""
    if level in levels:
        return level
    at = EFFORTS.index(level)
    return min(levels, key=lambda lv: (abs(EFFORTS.index(lv) - at), EFFORTS.index(lv)))


def _tokens(value: Any) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


@functools.cache
def _said(name: str, tokens: int, source: str) -> None:
    log.info("the context window of %s: %d tokens (%s)", name, tokens, source)


@functools.cache
def _assumed(name: str) -> None:
    log.warning(
        "the context window of %s is not known: %d tokens assumed (ReAct(context_window=) says it)",
        name,
        DEFAULT_WINDOW,
    )


@functools.cache
def serves(base_url: str, key: str | None, vendor: Vendor) -> bool:
    """Whether the gateway serves ``vendor``'s own API: an empty request to it is refused as
    invalid (it is there), not as not found. Nothing is generated, nothing is charged."""
    root = base_url.rstrip("/").removesuffix("/v1")
    url = f"{root}/anthropic/v1/messages" if vendor == "anthropic" else f"{root}/v1/responses"
    try:
        response = httpx.post(
            url,
            json={},
            headers={"Authorization": f"Bearer {key or 'none'}", "x-api-key": key or "none"},
            timeout=PROBE_SECONDS,
        )
    except Exception:
        return False
    return response.status_code not in (404, 405, 501)


__all__ = [
    "DEFAULT_WINDOW",
    "EFFORTS",
    "Window",
    "bare",
    "effort_levels",
    "nearest",
    "serves",
    "vendor_of",
    "window_of",
]
