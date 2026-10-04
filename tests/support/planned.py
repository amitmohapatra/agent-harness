"""Planned models: each call answers from what the conversation already holds, not from a
script consumed call by call.

A plan is the tool calls a model makes in order; the model counts the tool results in the
messages it is given and makes the next planned call, or — every call made — answers
``final`` with ``{last}`` replaced by the last tool result it read (a rejected call's reason,
an approved call's output). So a framework that re-runs a resumed run from its input, and one
that resumes in place from its checkpoint, drive the same model the same way, and an answer
shows what the model read. Each model keeps what it was sent (the pushed memory context is in
there)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from langchain_core.messages import BaseMessage, ToolMessage
from pydantic import Field

from tests.support.chat_model import ScriptedChatModel
from tests.support.models import ScriptedChat
from tests.support.openai_model import ScriptedModel

Call = tuple[str, dict[str, Any]]
FINAL = "Done. {last}"


def planned(plan: Sequence[Call], results: Sequence[str], final: str) -> str | Call:
    """The next turn of ``plan`` after ``results`` (the tool results so far)."""
    if len(results) < len(plan):
        return plan[len(results)]
    return final.replace("{last}", results[-1] if results else "")


class PlannedChatModel(ScriptedChatModel):
    """A LangChain chat model (``create_agent``, Deep Agents, a hand-built graph)."""

    plan: list[Call]
    final: str = FINAL
    turns: list[Any] = Field(default_factory=list)

    def _generate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> Any:
        results = [_text(m.content) for m in messages if isinstance(m, ToolMessage)]
        self.turns = [planned(self.plan, results, self.final)]
        return super()._generate(messages, *args, **kwargs)

    def said(self) -> str:
        """Everything the model was sent, as one text."""
        return "\n".join(_text(m.content) for call in self.seen for m in call)


class PlannedChat(ScriptedChat):
    """A chat-completions endpoint (``ReAct``)."""

    def __init__(self, plan: Sequence[Call], final: str = FINAL) -> None:
        super().__init__([])
        self.plan, self.final = list(plan), final

    async def complete(self, messages: list[dict[str, Any]], **body: Any) -> dict[str, Any]:
        results = [_text(m.get("content")) for m in messages if m.get("role") == "tool"]
        self.turns = [planned(self.plan, results, self.final)]
        return await super().complete(messages, **body)

    def said(self) -> str:
        return json.dumps([r["messages"] for r in self.requests], default=str)


class PlannedModel(ScriptedModel):
    """An OpenAI Agents SDK ``Model``."""

    def __init__(self, plan: Sequence[Call], final: str = FINAL) -> None:
        super().__init__([])
        self.plan, self.final = list(plan), final

    def _next(self, input: Any, system: str | None) -> list[Any]:
        items = input if isinstance(input, list) else []
        results = [
            _text(_field(i, "output")) for i in items if _field(i, "type") == "function_call_output"
        ]
        self.turns = [planned(self.plan, results, self.final)]
        return super()._next(input, system)

    def said(self) -> str:
        return json.dumps([self.system, self.inputs], default=str)


def _field(item: Any, name: str) -> Any:
    return item.get(name) if isinstance(item, dict) else getattr(item, name, None)


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in content
        )
    return "" if content is None else str(content)
