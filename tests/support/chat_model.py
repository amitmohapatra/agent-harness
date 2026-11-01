"""A scripted LangChain chat model: ``create_agent`` and Deep Agents drive it like a real one."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Sequence
from typing import Any, Final

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field

Turn = str | tuple[str, dict[str, Any]] | list[tuple[str, dict[str, Any]]]


#: The bound keyword holding the tools a call was offered, as given (``ScriptedChatModel.spends``).
OFFERED: Final = "offered_tools"


class ScriptedChatModel(BaseChatModel):
    """Each call answers with the next turn: text, ``(tool, args)`` for a tool call, or a list
    of them for several calls in one message."""

    turns: list[Turn]
    seen: list[list[BaseMessage]] = Field(default_factory=list)
    #: what each call was bound with: the tools' names, ``tool_choice`` and the settings
    requests: list[dict[str, Any]] = Field(default_factory=list)
    #: the input and output tokens every reply reports (none by default); an input of ``None``:
    #: the size of the request it was sent, counted as ``middleware.FitWindow`` counts one (its
    #: messages and its tools), so the window fitting learns no false ratio from it
    spends: tuple[int | None, int] | None = None
    #: how long every call takes before it answers (a stalled provider: longer than its bound)
    stall: float = 0.0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> Runnable[Any, Any]:  # type: ignore[override]
        names = [convert_to_openai_tool(t)["function"]["name"] for t in tools]
        return self.bind(tools=names, **{OFFERED: list(tools)}, **kwargs)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.stall:
            await asyncio.sleep(self.stall)
        return self._generate(messages, stop, **kwargs)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        offered = kwargs.pop(OFFERED, None)  # counted, not recorded: requests stay as bound
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
        if self.spends is not None:
            given, said = self.spends
            if given is None:
                given = count_tokens_approximately(messages, tools=offered or None)
            message.usage_metadata = {
                "input_tokens": given,
                "output_tokens": said,
                "total_tokens": given + said,
            }
        return ChatResult(generations=[ChatGeneration(message=message)])


#: A string argument holding this is given the path of the last saved large result the model
#: was told of (the tool message names it), so a scripted call can read it back.
SAVED: Final = "{saved}"
_SAVED_PATH: Final = re.compile(r"/large_tool_results/[^\s'\"`),;]+")


def _said(content: Any) -> str:
    if isinstance(content, list):
        return "\n".join(str(p.get("text", "")) if isinstance(p, dict) else str(p) for p in content)
    return str(content)


class KeyedChatModel(ScriptedChatModel):
    """Answers by the conversation it is given, not by the order of its calls: ``script`` holds
    each user message (a run's question, a steered message, a helper's task) with its rounds,
    the answer last. A call is answered with the round of the latest user message the script
    holds (its k-th time said: the script's k-th entry of it), counted by the assistant
    messages since — so runs that interleave, are steered or are cut short each still get
    their own rounds; a call offered no tool gets the answer. A call no scripted message is
    in is answered from ``turns``, in order. A string argument holding :data:`SAVED` gets the
    last saved result's path."""

    script: list[tuple[str, list[Turn]]] = Field(default_factory=list)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        humans = [n for n, m in enumerate(messages) if isinstance(m, HumanMessage)]
        for n in reversed(humans):
            text = _said(messages[n].content)
            entries = [r for key, r in self.script if key == text or (key.strip() and key in text)]
            if not entries:
                continue
            said = sum(_said(messages[k].content) == text for k in humans if k <= n)
            rounds = entries[min(max(said, 1), len(entries)) - 1]
            done = sum(isinstance(m, AIMessage) for m in messages[n + 1 :])
            chosen = rounds[min(done, len(rounds) - 1)]
            if not isinstance(chosen, str) and _no_tools(kwargs):
                chosen = rounds[-1]  # offered no tool: it answers
            self.turns.insert(0, _with_path(chosen, messages))
            break
        if not self.turns:
            self.turns.append("I have nothing more to add.")
        return super()._generate(messages, stop, run_manager, **kwargs)


def _no_tools(kwargs: dict[str, Any]) -> bool:
    """Whether the call was offered no tool (none bound, or ``tool_choice="none"``)."""
    return not kwargs.get("tools") or kwargs.get("tool_choice") == "none"


def _with_path(turn: Turn, messages: list[BaseMessage]) -> Turn:
    if isinstance(turn, str):
        return turn
    found = [
        p.rstrip(".:")
        for m in messages
        if isinstance(m, ToolMessage)
        for p in _SAVED_PATH.findall(_said(m.content))
    ]
    path = found[-1] if found else SAVED

    def filled(call: tuple[str, dict[str, Any]]) -> tuple[str, dict[str, Any]]:
        name, args = call
        return name, {
            k: v.replace(SAVED, path) if isinstance(v, str) else v for k, v in args.items()
        }

    return [filled(c) for c in turn] if isinstance(turn, list) else filled(turn)
