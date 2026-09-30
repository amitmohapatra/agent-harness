"""Server-sent events: each AG-UI event as one ``data:`` line, numbered with ``id:`` so a
client that reconnects sends ``Last-Event-ID`` and gets what came after."""

from __future__ import annotations

import json
from typing import Any, Final

from trellis.harness.surfaces.agui.events import AGUIEvent

MEDIA_TYPE: Final = "text/event-stream"


def encode(event: AGUIEvent, number: int) -> str:
    return f"id: {number}\ndata: {json.dumps(event.wire(), default=str)}\n\n"


def decode(body: str) -> list[tuple[int, dict[str, Any]]]:
    """``(id, event)`` pairs of an SSE body (tests and Python clients)."""
    out: list[tuple[int, dict[str, Any]]] = []
    for block in body.split("\n\n"):
        number, data = -1, None
        for line in block.splitlines():
            if line.startswith("id:"):
                number = int(line[3:].strip())
            elif line.startswith("data:"):
                data = json.loads(line[5:].strip())
        if data is not None:
            out.append((number, data))
    return out
