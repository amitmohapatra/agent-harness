"""``ReAct``: a tool-calling loop over chat completions, for teams with no framework.

    agent = h.wrap(ReAct(system="You answer stock questions.", model="gemini/gemini-3.8-flash"),
                   id="stock")

Native tool messages (``tool_calls`` in, ``role: tool`` out), structured output when
``output=`` names a pydantic model (``response_format`` JSON schema), and at most
``max_steps`` model calls. Each call is sent the tools the run offers at that moment (the
tool hints' candidates, the memory tools, the tools already used) and is a ``chat`` span. A
model name goes to Bifrost (``BIFROST_URL``); any object with the :class:`ChatModel` shape can
stand in for it (a scripted model in a test).

What keeps a run going when the model slips:

* arguments that are not a JSON object, or that miss what the tool's schema requires (a
  required field, a basic type, an unknown field where none are allowed), are an error result
  the model reads and can correct — the tool does not run;
* a tool result longer than ``max_result_chars`` is cut there with a marker saying so (the
  full result is kept as a run artifact, named in the marker), so one large result cannot
  fill the context;
* the same call with the same arguments in ``max_repeats`` consecutive model steps stops the
  run (a stall), as ``max_steps`` does;
* every model step is journaled (keyed by the step and the conversation so far): a resumed
  run — after a pause, or a worker crash — replays the steps it already took instead of
  calling the model again, and the journal replays their tool calls.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, ClassVar, Final, Protocol

from pydantic import BaseModel

from trellis.contracts import ConfigurationError, InterruptResolution, ModelError
from trellis.harness.adapters.base import Extracted, Invocation, Narrowing, Output, ToolFormat
from trellis.harness.journal import Pending, content_key
from trellis.harness.runtime import Runtime
from trellis.harness.telemetry import model_span, usage
from trellis.harness.telemetry import output as span_output
from trellis.harness.tools import bridge
from trellis.harness.tools.base import Tool
from trellis.harness.tools.convert import openai_chat, text_of

log = logging.getLogger("trellis.run")

#: Model calls one run may make before it is stopped.
MAX_STEPS: Final = 12
#: The longest tool result the model reads, in characters (about 5k tokens); the rest is cut.
MAX_RESULT_CHARS: Final = 20_000
#: Consecutive model steps making the same call with the same arguments that stop the run.
MAX_REPEATS: Final = 3
#: A JSON schema ``type`` and the Python values that are one (a bool is not a number).
JSON_TYPES: Final[dict[str, tuple[type, ...]]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
    "null": (type(None),),
}


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
    #: the longest tool result the model reads (characters); longer ones are cut, with a marker
    max_result_chars: int = field(default=MAX_RESULT_CHARS, kw_only=True)
    #: consecutive steps repeating one call (same tool, same arguments) that stop the run
    max_repeats: int = field(default=MAX_REPEATS, kw_only=True)


@dataclass(slots=True)
class ReActResult:
    messages: list[dict[str, Any]] = field(default_factory=list)
    answer: Any = None


class ReActAdapter:
    name: ClassVar[str] = "react"
    tool_format: ClassVar[ToolFormat] = "openai_chat"
    fixed_tools: ClassVar[bool] = False
    narrows: ClassVar[Narrowing] = "turn"

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
        runtime = run.runtime
        tools = {t.name: t for t in run.tools}
        body: dict[str, Any] = {}
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
        name = target.model if isinstance(target.model, str) else type(target.model).__name__
        streaks: dict[str, int] = {}
        for step in range(target.max_steps):
            offered = openai_chat.convert([t for t in run.tools if runtime.offers(t.name)])
            request = {**body, "tools": offered} if offered else body
            message = await _step(runtime, model, messages, name=name, step=step, request=request)
            messages.append(message)
            content = message.get("content")
            if isinstance(content, str) and content:
                yield content
            calls = message.get("tool_calls") or []
            if not calls:
                result.answer = _answer(target, content)
                yield Output(result)
                return
            streaks = _stalled(target, calls, streaks)
            for call in calls:
                output = await _run_call(target, runtime, tools, call)
                messages.append({"role": "tool", "tool_call_id": call.get("id"), "content": output})
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


async def _step(
    runtime: Runtime,
    model: Any,
    messages: list[dict[str, Any]],
    *,
    name: str,
    step: int,
    request: dict[str, Any],
) -> dict[str, Any]:
    """One model step: the journal's, when a resumed run already took it with this
    conversation, else the model's (journaled, and saved as progress in a worker)."""
    key = content_key("react", step, [m for m in messages if m.get("role") != "system"])
    replayed, recorded = runtime.replay.call(key)
    if replayed:
        return dict(recorded)
    with model_span(name, messages) as span:
        reply = await model.complete(messages, **request)
        usage(span, reply)
        message = _message(reply)
        span_output(span, message.get("content") or message.get("tool_calls"))
    runtime.replay.record_call(key, message)
    await runtime.progress(now=False)
    return message


def _stalled(target: ReAct, calls: list[dict[str, Any]], streaks: dict[str, int]) -> dict[str, int]:
    """The streak of each call (tool and arguments) over consecutive steps; a call repeated
    ``max_repeats`` steps in a row stops the run before it runs again."""
    now: dict[str, int] = {}
    for call in calls:
        function = call.get("function") or {}
        key = content_key("stall", function.get("name"), function.get("arguments"))
        now[key] = streaks.get(key, 0) + 1
        if now[key] >= target.max_repeats:
            raise ModelError(
                f"ReAct stopped: the model called {function.get('name')!r} with the same "
                f"arguments {now[key]} times in a row (a stall)"
            )
    return now


async def _run_call(
    target: ReAct, runtime: Runtime, tools: dict[str, Tool], call: dict[str, Any]
) -> str:
    """What the model reads for one of its calls: the result (bounded), or why it did not
    run."""
    function = call.get("function") or {}
    called = function.get("name", "")
    found = tools.get(called)
    if found is None:
        return f"there is no tool {called!r}"
    args, problem = _arguments(found, function.get("arguments"))
    if args is None:
        return f"{called} was not run: {problem}. Call it again with arguments that fit its schema."
    output = (await bridge.call(found, args, call_id=call.get("id"))).output
    return await _bounded(runtime, called, text_of(output), target.max_result_chars)


def _arguments(tool: Tool, raw: Any) -> tuple[dict[str, Any] | None, str]:
    """The call's arguments as an object, or ``None`` and what is wrong with them."""
    if isinstance(raw, dict):
        args: Any = raw
    else:
        try:
            args = json.loads(raw or "{}")
        except (TypeError, ValueError) as exc:
            return None, f"its arguments are not valid JSON ({exc})"
    if not isinstance(args, dict):
        return None, "its arguments must be a JSON object"
    problem = _schema_problem(tool.spec.input_schema or {}, args)
    return (None, problem) if problem else (args, "")


def _schema_problem(schema: dict[str, Any], args: dict[str, Any]) -> str | None:
    """A light check of the tool's JSON schema — required fields, the basic types of the
    declared ones, unknown fields where none are allowed; the tool itself validates the rest
    (a local tool through pydantic)."""
    missing = [name for name in schema.get("required") or [] if name not in args]
    if missing:
        return f"missing required argument(s): {', '.join(missing)}"
    properties: dict[str, Any] = schema.get("properties") or {}
    if schema.get("additionalProperties") is False:
        unknown = [name for name in args if name not in properties]
        if unknown:
            return f"unknown argument(s): {', '.join(unknown)}"
    for name, value in args.items():
        expected = (properties.get(name) or {}).get("type")
        allowed = JSON_TYPES.get(expected) if isinstance(expected, str) else None
        if allowed is None:
            continue
        wrong_bool = isinstance(value, bool) and expected != "boolean"
        if wrong_bool or not isinstance(value, allowed):
            return f"{name} must be of type {expected}"
    return None


async def _bounded(runtime: Runtime, tool: str, text: str, limit: int) -> str:
    """``text`` cut at ``limit`` characters, with a marker saying so; the full text is kept as a
    run artifact when the run store takes one."""
    if len(text) <= limit:
        return text
    kept = ""
    try:
        data = json.dumps({"tool": tool, "output": text}).encode()
        ref = await runtime.agent.harness.runs.artifacts.upload(
            runtime.run_id, data, worker_id=runtime.worker_id, tenant=runtime.tenant
        )
        kept = f"; the full result is run artifact {ref.artifact_id}"
    except Exception as exc:  # the cut result still goes to the model
        log.warning("the full result of %s in run %s was not kept: %s", tool, runtime.run_id, exc)
    return (
        f"{text[:limit]}\n…[cut: {tool} returned {len(text)} characters, {limit} are shown{kept}]"
    )


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
