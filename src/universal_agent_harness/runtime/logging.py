"""Structured JSON logging (§36) with the execution's identifiers on every line.

structlog is used when installed; otherwise the stdlib logger is wrapped and the same
fields are emitted in ``extra``. Payloads are never logged by default — the log line
carries ids, an event name, a duration and a status, and nothing else.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

BASE_LOGGER = "universal_agent_harness"

try:  # pragma: no cover - exercised by whichever branch the environment provides
    import structlog
    _HAS_STRUCTLOG = True
except ImportError:  # pragma: no cover - stdlib logging is the documented fallback
    structlog = None  # type: ignore[assignment]
    _HAS_STRUCTLOG = False


def configure_logging(level: str = "INFO", *, json_output: bool = True) -> None:
    """Opt-in JSON logging setup. Applications that configure logging keep their own setup:
    this only touches the harness logger's level unless structlog is present."""
    logging.getLogger(BASE_LOGGER).setLevel(level.upper())
    if structlog is None or not json_output:
        return
    if structlog.is_configured():  # pragma: no cover - respect the application's setup
        return
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelNamesMapping().get(level.upper(), logging.INFO)
        ),
        cache_logger_on_first_use=True,
    )


class HarnessLogger:
    """A logger bound to one execution's identifiers."""

    __slots__ = ("_fields", "_log")

    def __init__(self, name: str = BASE_LOGGER, fields: Mapping[str, Any] | None = None) -> None:
        self._fields = dict(fields or {})
        self._log = structlog.get_logger(name) if structlog is not None else logging.getLogger(name)

    def bind(self, **fields: Any) -> HarnessLogger:
        return HarnessLogger(
            getattr(self._log, "name", BASE_LOGGER), {**self._fields, **_clean(fields)}
        )

    def _emit(self, level: str, event: str, **fields: Any) -> None:
        payload = {**self._fields, **_clean(fields)}
        if structlog is not None:
            getattr(self._log, level)(event, **payload)
        else:
            getattr(self._log, level)(event, extra={"harness": payload})

    def debug(self, event: str, /, **fields: Any) -> None:
        self._emit("debug", event, **fields)

    def info(self, event: str, /, **fields: Any) -> None:
        self._emit("info", event, **fields)

    def warning(self, event: str, /, **fields: Any) -> None:
        self._emit("warning", event, **fields)

    def error(self, event: str, /, **fields: Any) -> None:
        self._emit("error", event, **fields)

    def exception(self, event: str, /, **fields: Any) -> None:
        if structlog is not None:
            self._log.exception(event, **{**self._fields, **_clean(fields)})
        else:
            self._log.exception(event, extra={"harness": {**self._fields, **_clean(fields)}})


def get_logger(name: str = BASE_LOGGER, **fields: Any) -> HarnessLogger:
    return HarnessLogger(name, fields)


#: Keys the rendering layer owns. A caller's field of the same name is kept, under a
#: prefix, rather than colliding with the log record (or raising).
_RESERVED = ("event", "level", "timestamp", "logger")


def _clean(fields: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in fields.items():
        if value is None:
            continue
        out[f"field_{key}" if key in _RESERVED else key] = value
    return out
