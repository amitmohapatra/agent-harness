"""Harness tools as chat-completions function definitions (the ``ReAct`` target)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from trellis.harness.tools.base import Tool


def convert(tools: Sequence[Tool]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.spec.description or t.name,
                "parameters": t.spec.input_schema or {"type": "object", "properties": {}},
            },
        }
        for t in tools
    ]
