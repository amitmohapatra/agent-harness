"""The telemetry provider used when observability is disabled."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

from universal_agent_harness.telemetry.span import NOOP_SPAN, NoOpSpan


class NoOpTelemetryProvider:
    name = "noop"

    @contextmanager
    def start_span(
        self, name: str, *, kind: str = "internal", attributes: Mapping[str, Any] | None = None
    ) -> Iterator[NoOpSpan]:
        yield NOOP_SPAN

    def record_event(self, name: str, attributes: Mapping[str, Any] | None = None) -> None:
        return None

    def record_metric(
        self,
        name: str,
        value: float,
        *,
        unit: str = "",
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        return None

    def flush(self, timeout_seconds: float = 5.0) -> None:
        return None
