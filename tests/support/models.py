"""A scripted chat-completions endpoint (the shape ``ReAct`` and the judge call)."""

from __future__ import annotations

import json
from typing import Any

Call = tuple[str, dict[str, Any]]
Turn = str | Call | list[Call] | dict[str, Any]


class ScriptedChat:
    """Each call answers with the next turn: text, ``(tool, args)``, several calls in one
    message (``[(tool, args), ...]``), or a raw message."""

    def __init__(self, turns: list[Turn]) -> None:
        self.turns = list(turns)
        self.requests: list[dict[str, Any]] = []

    async def complete(self, messages: list[dict[str, Any]], **body: Any) -> dict[str, Any]:
        self.requests.append({"messages": [dict(m) for m in messages], **body})
        turn = self.turns.pop(0)
        if isinstance(turn, str):
            message: dict[str, Any] = {"role": "assistant", "content": turn}
        elif isinstance(turn, tuple | list):
            calls = turn if isinstance(turn, list) else [turn]
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"call_{len(self.requests)}" + (f"_{n}" if n else ""),
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(args)},
                    }
                    for n, (name, args) in enumerate(calls)
                ],
            }
        else:
            message = turn
        return {"model": "scripted", "choices": [{"index": 0, "message": message}]}
