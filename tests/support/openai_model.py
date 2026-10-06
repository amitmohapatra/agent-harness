"""A scripted OpenAI Agents SDK ``Model``: the real ``Runner`` drives it, turn by turn."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from agents import Model, ModelResponse, Usage
from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseTextDeltaEvent,
)

#: a call's arguments: an object, or the text the model wrote (which may not be JSON)
Args = dict[str, Any] | str
Turn = str | tuple[str, Args] | list[tuple[str, Args]]


class ScriptedModel(Model):
    """Each call answers with the next turn: text, ``(tool, args)`` for a tool call (``args``
    as text: the arguments exactly as written), or a list of them for several calls in one
    response."""

    def __init__(self, turns: list[Turn]) -> None:
        self.turns = list(turns)
        self.inputs: list[Any] = []
        self.system: list[str | None] = []
        #: the tool names each call was offered
        self.tools: list[list[str]] = []

    def _next(self, input: Any, system: str | None) -> list[Any]:
        self.inputs.append(input)
        self.system.append(system)
        turn = self.turns.pop(0)
        if isinstance(turn, str):
            return [
                ResponseOutputMessage(
                    id=f"msg_{len(self.inputs)}",
                    role="assistant",
                    status="completed",
                    type="message",
                    content=[ResponseOutputText(type="output_text", text=turn, annotations=[])],
                )
            ]
        calls = turn if isinstance(turn, list) else [turn]
        return [
            ResponseFunctionToolCall(
                id=f"fc_{len(self.inputs)}" + (f"_{n}" if n else ""),
                call_id=f"call_{len(self.inputs)}" + (f"_{n}" if n else ""),
                type="function_call",
                name=name,
                arguments=args if isinstance(args, str) else json.dumps(args),
                status="completed",
            )
            for n, (name, args) in enumerate(calls)
        ]

    async def get_response(
        self,
        system_instructions: str | None,
        input: Any,
        model_settings: Any = None,
        tools: Any = (),
        *args: Any,
        **kwargs: Any,
    ) -> ModelResponse:
        self.tools.append([t.name for t in tools])
        return ModelResponse(
            output=self._next(input, system_instructions), usage=Usage(), response_id=None
        )

    async def stream_response(  # type: ignore[override]
        self, system_instructions: str | None, input: Any, *args: Any, **kwargs: Any
    ) -> AsyncIterator[Any]:
        output = self._next(input, system_instructions)
        for item in output:
            if isinstance(item, ResponseOutputMessage):
                for index, word in enumerate(item.content[0].text.split(" ")):  # type: ignore[union-attr]
                    yield ResponseTextDeltaEvent(
                        type="response.output_text.delta",
                        item_id=item.id,
                        output_index=0,
                        content_index=0,
                        delta=word if index == 0 else f" {word}",
                        logprobs=[],
                        sequence_number=index,
                    )
        yield ResponseCompletedEvent(
            type="response.completed",
            sequence_number=99,
            response=Response(
                id="resp_1",
                created_at=0,
                model="scripted",
                object="response",
                output=output,
                tool_choice="auto",
                tools=[],
                parallel_tool_calls=False,
            ),
        )
