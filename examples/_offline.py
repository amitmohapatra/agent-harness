"""Models for running the examples with no services.

With ``BIFROST_URL`` set, the examples use a real model through Bifrost's OpenAI-compatible
endpoint (:data:`MODEL`). Without it they use the scripted models here, which answer from a
fixed script: the harness, the frameworks and the tools are all real; only the model is not.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any

#: The model the examples use through Bifrost.
MODEL = "openrouter/openai/gpt-4.1-nano"
MAX_TOKENS = 2048
#: A stand-in for the Claude Code CLI that speaks its stream-json protocol.
FAKE_CLAUDE_CLI = str(
    Path(__file__).resolve().parents[1] / "tests" / "support" / "fake_claude_cli.py"
)

Turn = str | tuple[str, dict[str, Any]]


def online() -> bool:
    return bool(os.environ.get("BIFROST_URL"))


def gateway() -> tuple[str, str]:
    """Bifrost's base URL and virtual key, for a framework's own OpenAI client."""
    return os.environ["BIFROST_URL"], os.environ.get("BIFROST_VIRTUAL_KEY") or "unused"


# --------------------------------------------------------------------------- LangChain
def langchain_model(turns: Sequence[Turn]) -> Any:
    """A chat model for LangGraph / Deep Agents: Bifrost when online, else the script."""
    if online():
        from langchain_openai import ChatOpenAI

        url, key = gateway()
        return ChatOpenAI(model=MODEL, base_url=url, api_key=key, max_completion_tokens=MAX_TOKENS)  # type: ignore[arg-type]
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    script = list(turns)

    class Scripted(BaseChatModel):
        @property
        def _llm_type(self) -> str:
            return "scripted"

        def bind_tools(self, tools: Any, **kwargs: Any) -> Any:  # type: ignore[override]
            return self

        def _generate(
            self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any
        ) -> ChatResult:
            turn = script.pop(0)
            if isinstance(turn, str):
                message = AIMessage(content=turn)
            else:
                message = AIMessage(
                    content="",
                    tool_calls=[{"name": turn[0], "args": turn[1], "id": f"call_{len(script)}"}],
                )
            return ChatResult(generations=[ChatGeneration(message=message)])

    return Scripted()


# --------------------------------------------------------------------------- OpenAI Agents
def openai_agents_model(turns: Sequence[Turn]) -> Any:
    """A model for the OpenAI Agents SDK: Bifrost when online, else the script."""
    from agents import Model, ModelResponse, OpenAIChatCompletionsModel, Usage
    from openai.types.responses import (
        ResponseFunctionToolCall,
        ResponseOutputMessage,
        ResponseOutputText,
    )

    if online():
        from openai import AsyncOpenAI

        url, key = gateway()
        return OpenAIChatCompletionsModel(
            model=MODEL, openai_client=AsyncOpenAI(base_url=url, api_key=key)
        )
    script = list(turns)

    class Scripted(Model):
        async def get_response(self, *args: Any, **kwargs: Any) -> ModelResponse:
            turn = script.pop(0)
            if isinstance(turn, str):
                item: Any = ResponseOutputMessage(
                    id="msg",
                    role="assistant",
                    status="completed",
                    type="message",
                    content=[ResponseOutputText(type="output_text", text=turn, annotations=[])],
                )
            else:
                item = ResponseFunctionToolCall(
                    id="fc",
                    call_id=f"call_{len(script)}",
                    type="function_call",
                    name=turn[0],
                    arguments=json.dumps(turn[1]),
                    status="completed",
                )
            return ModelResponse(output=[item], usage=Usage(), response_id=None)

        def stream_response(self, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
            raise NotImplementedError("the offline examples do not stream this model")

    return Scripted()


# --------------------------------------------------------------------------- ReAct
class ScriptedChat:
    """A chat-completions endpoint answering from a script (the ``ReAct`` model shape)."""

    def __init__(self, turns: Sequence[Turn]) -> None:
        self.turns = list(turns)

    async def complete(self, messages: list[dict[str, Any]], **body: Any) -> dict[str, Any]:
        turn = self.turns.pop(0)
        if isinstance(turn, str):
            message: dict[str, Any] = {"role": "assistant", "content": turn}
        else:
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"call_{len(self.turns)}",
                        "type": "function",
                        "function": {"name": turn[0], "arguments": json.dumps(turn[1])},
                    }
                ],
            }
        return {"choices": [{"message": message}]}


def react_model(turns: Sequence[Turn]) -> Any:
    """A model name for Bifrost when online, else the script."""
    return MODEL if online() else ScriptedChat(turns)


# --------------------------------------------------------------------------- evaluation
class Answering:
    """A chat-completions endpoint that answers each question from a table, and — asked by the
    evaluation judge (``llm_judge``) — grades an answer 1 when it contains the expected one
    (with nothing expected: when it is an answer at all), else 0, as the strict JSON the judge
    asks for."""

    def __init__(self, answers: dict[str, str]) -> None:
        self.answers = answers

    async def complete(self, messages: list[dict[str, Any]], **body: Any) -> dict[str, Any]:
        if "strict evaluator" in str(messages[0].get("content")):
            content = self._grade(str(messages[-1]["content"]))
        else:
            question = next(m["content"] for m in reversed(messages) if m["role"] == "user")
            content = self.answers.get(question, "I don't know.")
        return {"choices": [{"message": {"role": "assistant", "content": content}}]}

    @staticmethod
    def _grade(prompt: str) -> str:
        sections = dict(part.split("\n", 1) for part in prompt.split("## ")[1:])
        expected = sections.get("Expected answer", "").strip().casefold()
        graded = sections.get("Answer to grade", "").strip().casefold()
        good = expected in graded if expected else graded not in ("", "i don't know.")
        reasoning = "names the expected answer" if good else "does not name the expected answer"
        return json.dumps({"score": 1.0 if good else 0.0, "reasoning": reasoning})


def answering_model(answers: dict[str, str]) -> Any:
    """A model name for Bifrost when online, else the answer table (which also judges)."""
    return MODEL if online() else Answering(answers)
