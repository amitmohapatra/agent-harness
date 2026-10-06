"""Planned models whose plan may hold steps of several calls (one message, several tool calls),
and what every planned model of a cell was sent and offered (:class:`ModelLog`).

A step of a grouped plan is a call, or a list of calls the model makes in one message; the
model counts the tool results it was given and makes the next step's calls, or answers.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import pytest
from langchain_core.messages import BaseMessage, ToolMessage
from pydantic import Field

from tests.support.chat_model import ScriptedChatModel
from tests.support.models import Script, ScriptedChat
from tests.support.openai_model import ScriptedModel
from tests.support.planned import FINAL, Call, PlannedChatModel, PlannedModel, _text

Step = Call | list[Call]


def grouped(plan: Sequence[Step], results: Sequence[str], final: str = FINAL) -> Any:
    """The next turn of a grouped ``plan`` after ``results``."""
    done = 0
    for step in plan:
        if len(results) == done:
            return step
        done += len(step) if isinstance(step, list) else 1
    return final.replace("{last}", results[-1] if results else "")


class _Grouped(Script):
    def __init__(self, plan: Sequence[Step]) -> None:
        super().__init__([])
        self.steps = list(plan)

    def next(self, body: dict[str, Any]) -> Any:
        messages = body.get("messages") or []
        results = [_text(m.get("content")) for m in messages if m.get("role") == "tool"]
        return grouped(self.steps, results)


class GroupedChat(ScriptedChat):
    """``ReAct``'s gateway model, following a grouped plan."""

    def __init__(self, plan: Sequence[Step]) -> None:
        super().__init__(script=_Grouped(plan))


class GroupedChatModel(PlannedChatModel):
    """A LangChain chat model, following a grouped plan (``steps``)."""

    steps: list[Any] = Field(default_factory=list)

    def _generate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> Any:
        results = [_text(m.content) for m in messages if isinstance(m, ToolMessage)]
        self.turns = [grouped(self.steps, results, self.final)]
        return ScriptedChatModel._generate(self, messages, *args, **kwargs)


class GroupedModel(PlannedModel):
    """An OpenAI Agents SDK model, following a grouped plan."""

    def __init__(self, plan: Sequence[Step]) -> None:
        super().__init__([])
        self.steps = list(plan)

    def _next(self, input: Any, system: str | None) -> list[Any]:
        items = input if isinstance(input, list) else []
        results = [
            _text(_field(i, "output")) for i in items if _field(i, "type") == "function_call_output"
        ]
        self.turns = [grouped(self.steps, results, self.final)]
        return ScriptedModel._next(self, input, system)


def _field(item: Any, name: str) -> Any:
    return item.get(name) if isinstance(item, dict) else getattr(item, name, None)


class ModelLog:
    """What every planned model in the cell was sent (``said``) and offered (``offered``), per
    call, whatever its framework: the scripted models are patched to tell it."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.calls: list[tuple[str, list[str]]] = []
        self._bound: dict[int, list[str]] = {}
        log = self

        reply = Script.reply

        def replied(script: Script, body: dict[str, Any]) -> Any:
            offered = [t["function"]["name"] for t in body.get("tools") or []]
            log.calls.append((json.dumps(body.get("messages"), default=str), offered))
            return reply(script, body)

        monkeypatch.setattr(Script, "reply", replied)

        def bind_tools(model: ScriptedChatModel, tools: Sequence[Any], **kw: Any) -> Any:
            log._bound[id(model)] = [getattr(t, "name", str(t)) for t in tools]
            return model

        generate = ScriptedChatModel._generate

        def generating(
            model: ScriptedChatModel, messages: list[BaseMessage], *a: Any, **k: Any
        ) -> Any:
            said = "\n".join(_text(m.content) for m in messages)
            log.calls.append((said, list(log._bound.get(id(model), []))))
            return generate(model, messages, *a, **k)

        monkeypatch.setattr(ScriptedChatModel, "bind_tools", bind_tools)
        monkeypatch.setattr(ScriptedChatModel, "_generate", generating)

        respond = ScriptedModel.get_response
        stream = ScriptedModel.stream_response

        def heard(a: tuple[Any, ...], k: dict[str, Any]) -> None:
            """The SDK calls with keywords (or positionally): system, input, settings, tools."""
            system = k.get("system_instructions", a[0] if a else None)
            given = k.get("input", a[1] if len(a) > 1 else None)
            tools = k.get("tools", a[3] if len(a) > 3 else ())
            said = json.dumps([system, given], default=str)
            log.calls.append((said, [t.name for t in tools or ()]))

        async def get_response(model: ScriptedModel, *a: Any, **k: Any) -> Any:
            heard(a, k)
            return await respond(model, *a, **k)

        async def stream_response(model: ScriptedModel, *a: Any, **k: Any) -> Any:
            heard(a, k)
            async for item in stream(model, *a, **k):
                yield item

        monkeypatch.setattr(ScriptedModel, "get_response", get_response)
        monkeypatch.setattr(ScriptedModel, "stream_response", stream_response)

    def said(self) -> str:
        return "\n".join(said for said, _ in self.calls)

    def offered(self) -> list[list[str]]:
        return [offered for _, offered in self.calls]
