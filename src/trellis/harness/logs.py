"""Log lines as JSON, one object per line: what a log pipeline (Datadog, Loki, Cloud Logging)
reads without parsing text.

The harness logs through the standard ``logging`` module (``trellis.*`` loggers, with the
run's id and the like as ``extra`` fields) and leaves the configuration to the application,
except in the worker it starts itself (``python -m trellis.harness.worker``), which logs JSON
when its output is not a terminal (a container, a service manager) and text when it is —
nothing to set. An application of its own installs :class:`JSONFormatter` the usual way::

    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())
    logging.basicConfig(level=logging.INFO, handlers=[handler])

Each line has ``time`` (ISO 8601, UTC), ``level``, ``logger``, ``message``, every ``extra``
field (``run_id``...), and ``exception`` when there is one.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import IO, Any, Final

#: The attributes every ``LogRecord`` has: anything else on one is an ``extra`` field.
_STANDARD: Final = frozenset(
    {*vars(logging.LogRecord("", logging.INFO, "", 0, "", None, None)), "message", "asctime"}
)


class JSONFormatter(logging.Formatter):
    """One JSON object per record."""

    def format(self, record: logging.LogRecord) -> str:
        line: dict[str, Any] = {
            "time": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        line.update({k: v for k, v in vars(record).items() if k not in _STANDARD})
        if record.exc_info:
            line["exception"] = self.formatException(record.exc_info)
        return json.dumps(line, default=str)


def configure(stream: IO[str], *, level: int = logging.INFO) -> None:
    """Log to ``stream`` at ``level``: JSON lines unless ``stream`` is a terminal (nothing,
    when the root logger already has a handler: the application configured logging)."""
    handler = logging.StreamHandler(stream)
    if not stream.isatty():
        handler.setFormatter(JSONFormatter())
    logging.basicConfig(level=level, handlers=[handler])


__all__ = ["JSONFormatter", "configure"]
