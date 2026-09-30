"""``ReAct``: a tool-calling loop over chat completions, for teams with no framework.

    agent = h.wrap(ReAct(system="You answer stock questions.", model="gemini/gemini-3.8-flash"),
                   id="stock", tools=[mcp("erp")])

Native tool messages (``tool_calls`` in, ``role: tool`` out), structured output when
``output=`` names a pydantic model (``response_format`` JSON schema), and at most
``max_steps`` model calls. A model name goes to Bifrost (``BIFROST_URL``); any object with the
:class:`ChatModel` shape can stand in for it (a scripted model in a test).
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, ClassVar, Final, Protocol

from pydantic import BaseModel

from trellis.contracts import ConfigurationError, InterruptResolution, ModelError
from trellis.harness.adapters.base import Extracted, Invocation, Output, ToolFormat
from trellis.harness.journal import Pending
from trellis.harness.tools import bridge
from trellis.harness.tools.convert import text_of

#: Model calls one run may make before it is stopped.
MAX_STEPS: Final = 12


class ChatModel(Protocol):
    """A chat-completions endpoint: the request body in, the response object out."""

    async def complete(self, messages: list[dict[str, Any]], **body: Any) -> dict[str, Any]: ...


@dataclass(frozen=True)
class ReAct:
    """The target: a system prompt, a model, and optionally the answer's schema."""

    system: str
    model: str | ChatModel
    output: type[BaseModel] | None = None
    max_steps: int = MAX_STEPS


@dataclass(slots=True)
class ReActResult:
    messages: list[dict[str, Any]] = field(default_factory=list)
    answer: Any = None


class ReActAdapter:
    name: ClassVar[str] = "react"
    tool_format: ClassVar[ToolFormat] = "openai_chat"
    fixed_tools: ClassVar[bool] = False

    def keeps_conversation(self, target: Any) -> bool:
        return False

    def prepare_input(self, target: ReAct, input: Any, context: str | None) -> Any:
        system = f"{target.system}\n\n{context}" if context else target.system
        if isinstance(input, list):
            return [{"role": "system", "content": system}, *input]
        text = input if isinstance(input, str) else json.dumps(input, default=str)
        return [{"role": "system", "content": system}, {"role": "user", "content": text}]

    async def invoke(self, target: ReAct, native_input: Any, run: Invocation) -> ReActResult:
        result: ReActResult | None = None
        async for item in self.stream(target, native_input, run):
            if isinstance(item, Output):
                result = item.value
        assert result is not None
        return result

    async def stream(self, target: ReAct, native_input: Any, run: Invocation) -> AsyncIterator[Any]:
        model = _model(target, run)
        tools = {t.name: t for t in run.tools}
        body: dict[str, Any] = {}
        if run.native_tools:
            body["tools"] = run.native_tools
        if target.output is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": target.output.__name__,
                    "schema": target.output.model_json_schema(),
                },
            }
        messages = list(native_input)
        result = ReActResult(messages=messages)
        for _ in range(target.max_steps):
            reply = await model.complete(messages, **body)
            message = _message(reply)
            messages.append(message)
            content = message.get("content")
            if isinstance(content, str) and content:
                yield content
            calls = message.get("tool_calls") or []
            if not calls:
                result.answer = _answer(target, content)
                yield Output(result)
                return
            for call in calls:
                function = call.get("function") or {}
                found = tools.get(function.get("name", ""))
                if found is None:
                    output = f"there is no tool {function.get('name')!r}"
                else:
                    args = json.loads(function.get("arguments") or "{}")
                    output = (await bridge.call(found, args, call_id=call.get("id"))).output
                messages.append(
                    {"role": "tool", "tool_call_id": call.get("id"), "content": text_of(output)}
                )
        raise ModelError(f"ReAct stopped after {target.max_steps} model calls without an answer")

    def extract(self, target: ReAct, output: ReActResult) -> Extracted:
        transcript = [
            ("assistant", m["content"])
            for m in output.messages
            if m.get("role") == "assistant" and isinstance(m.get("content"), str) and m["content"]
        ]
        return Extracted(answer=output.answer, transcript=transcript)  # type: ignore[arg-type]

    def resume_input(
        self,
        target: ReAct,
        native_input: Any,
        pending: Pending,
        resolution: InterruptResolution,
    ) -> Any:
        return native_input


def _model(target: ReAct, run: Invocation) -> Any:
    if not isinstance(target.model, str):
        return target.model
    gateway = run.runtime.agent.harness.gateway
    if gateway is None:
        raise ConfigurationError("ReAct with a model name needs BIFROST_URL")
    return _Named(gateway, target.model)


@dataclass(frozen=True, slots=True)
class _Named:
    gateway: Any
    model: str

    async def complete(self, messages: list[dict[str, Any]], **body: Any) -> dict[str, Any]:
        return await self.gateway.complete(messages, model=self.model, **body)


def _message(reply: dict[str, Any]) -> dict[str, Any]:
    try:
        message = dict(reply["choices"][0]["message"])
    except (KeyError, IndexError, TypeError) as exc:
        raise ModelError(f"the model returned no message: {str(reply)[:300]}") from exc
    message.setdefault("role", "assistant")
    return {k: v for k, v in message.items() if v is not None}


def _answer(target: ReAct, content: Any) -> Any:
    if target.output is None:
        return content
    text = content if isinstance(content, str) else json.dumps(content)
    return target.output.model_validate_json(_unfenced(text))


def _unfenced(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1] if "\n" in stripped else ""
        stripped = stripped.rsplit("```", 1)[0]
    return stripped.strip()
