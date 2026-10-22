"""What the harness's own code guarantees where a type cannot say it.

A broken guarantee is a :class:`~trellis.contracts.HarnessError` naming it, never an
``assert``: Python drops asserts under ``-O``, and the failure would then surface far away as
an ``AttributeError`` on ``None``.
"""

from __future__ import annotations

from trellis.contracts import HarnessError


def present[T](value: T | None, guarantee: str) -> T:
    """``value``, which ``guarantee`` says is never ``None`` here."""
    if value is None:  # pragma: no cover - a guarantee of the harness's own code
        raise HarnessError(f"internal error: {guarantee}")
    return value


__all__ = ["present"]
