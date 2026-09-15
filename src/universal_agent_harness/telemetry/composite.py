"""Fan telemetry out to several providers with failure isolation (§32).

One backend being down — Langfuse, typically — must not break the others and must never
break the business execution. In ``non_blocking`` mode every provider error is logged once
and swallowed; ``fail_closed`` re-raises, for the rare regulated deployment that needs it.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from typing import Any

from universal_agent_harness.telemetry.span import NOOP_SPAN

log = logging.getLogger("universal_agent_harness.telemetry")


class CompositeSpan:
    """One logical span backed by several provider spans."""

    __slots__ = ("_spans", "_strict")

    def __init__(self, spans: Sequence[Any], *, strict: bool) -> None:
        self._spans = spans
        self._strict = strict

    def _each(self, method: str, *args: Any, **kwargs: Any) -> None:
        for span in self._spans:
            try:
                getattr(span, method)(*args, **kwargs)
            except Exception:
                if self._strict:
                    raise
                log.debug("telemetry span.%s failed", method, exc_info=True)

    def set_attribute(self, key: str, value: Any) -> None:
        self._each("set_attribute", key, value)

    def set_attributes(self, attributes: Mapping[str, Any]) -> None:
        self._each("set_attributes", attributes)

    def add_event(self, name: str, attributes: Mapping[str, Any] | None = None) -> None:
        self._each("add_event", name, attributes)

    def record_error(self, error: BaseException | str, **attributes: Any) -> None:
        self._each("record_error", error, **attributes)

    def set_status_ok(self) -> None:
        self._each("set_status_ok")

    @property
    def spans(self) -> Sequence[Any]:
        return self._spans


class CompositeTelemetryProvider:
    """Composition over inheritance: any number of providers behind one port."""

    name = "composite"

    def __init__(self, providers: Sequence[Any], *, strict: bool = False) -> None:
        self.providers = [p for p in providers if p is not None]
        self.strict = strict

    @contextmanager
    def start_span(
        self, name: str, *, kind: str = "internal", attributes: Mapping[str, Any] | None = None
    ) -> Iterator[Any]:
        if not self.providers:
            yield NOOP_SPAN
            return
        spans: list[Any] = []
        with ExitStack() as stack:
            for provider in self.providers:
                try:
                    spans.append(
                        stack.enter_context(
                            provider.start_span(name, kind=kind, attributes=attributes)
                        )
                    )
                except Exception:
                    if self.strict:
                        raise
                    log.warning(
                        "telemetry provider %s failed to start a span", provider, exc_info=True
                    )
            yield CompositeSpan(spans, strict=self.strict) if len(spans) != 1 else spans[0]

    def record_event(self, name: str, attributes: Mapping[str, Any] | None = None) -> None:
        self._each("record_event", name, attributes)

    def record_metric(
        self,
        name: str,
        value: float,
        *,
        unit: str = "",
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        self._each("record_metric", name, value, unit=unit, attributes=attributes)

    def flush(self, timeout_seconds: float = 5.0) -> None:
        self._each("flush", timeout_seconds)

    def _each(self, method: str, *args: Any, **kwargs: Any) -> None:
        for provider in self.providers:
            try:
                getattr(provider, method)(*args, **kwargs)
            except Exception:
                if self.strict:
                    raise
                log.debug("telemetry %s.%s failed", provider, method, exc_info=True)
