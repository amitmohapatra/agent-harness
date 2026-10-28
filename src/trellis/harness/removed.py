"""Arguments the public API no longer takes, refused by name.

A removed or renamed keyword argument is not kept as an alias: passing it raises Python's own
``TypeError`` for an argument a call does not take, with what to use instead in the message
(``h.evaluate(evaluators=)`` names ``judges``), rather than the bare "unexpected keyword
argument". The signature itself is unchanged, so a type checker refuses the old name too.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import ParamSpec, TypeVar

_P = ParamSpec("_P")
_R = TypeVar("_R")


def removed(**instead: str) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Refuse each keyword argument named in ``instead`` (its value: what to do instead)."""

    def decorate(func: Callable[_P, _R]) -> Callable[_P, _R]:
        @functools.wraps(func)
        def refusing(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            for name in kwargs:
                if name in instead:
                    raise TypeError(f"{_shown(func)}({name}=) was removed: {instead[name]}")
            return func(*args, **kwargs)

        return refusing

    return decorate


def _shown(func: Callable[..., object]) -> str:
    """How the call is written: ``h.evaluate``, ``ReAct``, ``middleware``, ``FitWindow``."""
    name = func.__qualname__.removesuffix(".__init__")
    return "h." + name.split(".", 1)[1] if name.startswith("Harness.") else name
