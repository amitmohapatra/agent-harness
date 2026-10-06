"""A scripted LangChain chat model: ``create_agent`` and Deep Agents drive it like a real one."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field

Turn = str | tuple[str, dict[str, Any]] | list[tuple[str, dict[str, Any]]]


class ScriptedChatModel(BaseChatModel):
    """Each call answers with the next turn: text, ``(tool, args)`` for a tool call, or a list
    of them for several calls in one message."""

    turns: list[Turn]
    seen: list[list[BaseMessage]] = Field(default_factory=list)
    #: what each call was bound with: the tools' names, ``tool_choice`` and the settings
    requests: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> Runnable[Any, Any]:  # type: ignore[override]
        names = [convert_to_openai_tool(t)["function"]["name"] for t in tools]
        return self.bind(tools=names, **kwargs)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.seen.append(list(messages))
        self.requests.append(kwargs)
        turn = self.turns.pop(0)
        if isinstance(turn, str):
            message = AIMessage(content=turn)
        else:
            calls = turn if isinstance(turn, list) else [turn]
            message = AIMessage(
                content="",
                tool_calls=[
                    {"name": name, "args": args, "id": f"call_{len(self.seen)}_{n}"}
                    for n, (name, args) in enumerate(calls)
                ],
            )
        return ChatResult(generations=[ChatGeneration(message=message)])
