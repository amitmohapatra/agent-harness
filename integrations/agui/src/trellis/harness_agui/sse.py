"""Server-sent events: one ``data:`` line per AG-UI event, as the protocol's SSE transport
expects (``text/event-stream``, the event's JSON as the data, a blank line between events)."""

from __future__ import annotations

import json
from typing import Any

from trellis.harness_agui.events import AGUIEvent

MEDIA_TYPE = "text/event-stream"


def encode(event: AGUIEvent) -> str:
    return f"data: {json.dumps(event.wire(), default=str)}\n\n"


def decode(body: str) -> list[dict[str, Any]]:
    """The events of an SSE body (tests and clients)."""
    out: list[dict[str, Any]] = []
    for block in body.split("\n\n"):
        for line in block.splitlines():
            if line.startswith("data:"):
                out.append(json.loads(line[5:].strip()))
    return out
