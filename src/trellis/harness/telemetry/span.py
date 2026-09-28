"""The span object the harness hands to interceptors and runtime clients.

It is intentionally tiny: set attributes, record an event, mark an error. Backends
implement it over their own span type; :class:`NoOpSpan` costs nothing when telemetry is
off, which is what keeps the "harness overhead" budget (§78) reachable.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class HarnessSpan(Protocol):
    def set_attribute(self, key: str, value: Any) -> None: ...

    def set_attributes(self, attributes: Mapping[str, Any]) -> None: ...

    def add_event(self, name: str, attributes: Mapping[str, Any] | None = None) -> None: ...

    def record_error(self, error: BaseException | str, **attributes: Any) -> None: ...

    def set_status_ok(self) -> None: ...


class NoOpSpan:
    """Does nothing, cheaply. Shared singleton: it holds no state."""

    __slots__ = ()

    def set_attribute(self, key: str, value: Any) -> None:
        return None

    def set_attributes(self, attributes: Mapping[str, Any]) -> None:
        return None

    def add_event(self, name: str, attributes: Mapping[str, Any] | None = None) -> None:
        return None

    def record_error(self, error: BaseException | str, **attributes: Any) -> None:
        return None

    def set_status_ok(self) -> None:
        return None


NOOP_SPAN = NoOpSpan()
