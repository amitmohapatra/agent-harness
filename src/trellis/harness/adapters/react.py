"""``ReAct``: a tool-calling loop over chat completions, for teams with no framework.

    agent = h.wrap(ReAct(system="You answer stock questions.", model="gemini/gemini-3.8-flash"),
                   id="stock")

Native tool messages (``tool_calls`` in, ``role: tool`` out), structured output when
``output=`` names a pydantic model (``response_format`` JSON schema), and at most
``max_steps`` model calls with tools. Each call is sent the tools the run offers at that moment
(the tool hints' candidates, the memory tools, the tools already used), sorted by name, and is
a ``chat`` span. A model name goes to Bifrost (``BIFROST_URL``); any object with the
:class:`ChatModel` shape can stand in for it (a scripted model in a test).

Several calls in one step: the ones whose tool only reads (``side_effects="read"``, or
idempotent) run at once, then the others one at a time, in the order the model gave; the tool
messages follow the model's order, one per call. A call that pauses lets the calls running
beside it finish (they are journaled) before the run pauses.

What keeps a run going when the model slips:

* arguments that are not a JSON object, or that miss what the tool's schema requires (a
  required field, a basic type, an unknown field where none are allowed), are an error result
  the model reads and can correct — the tool does not run;
* a tool result longer than ``max_result_chars`` keeps its head and its tail, with a marker in
  the middle (the full result is kept as a run artifact, named in the marker), so one large
  result cannot fill the context; ``read_result`` reads the rest;
* the same call with the same arguments in ``max_repeats`` consecutive model steps stops the
  run (a stall), and so do :data:`ERROR_STREAK` consecutive steps in which every call failed;
* at ``max_steps`` the model is asked once more, without tools, for its best answer with what
  it has (a ``warning`` event, ``max_steps``, says the run stopped there);
* every model step is journaled (keyed by the step and the conversation so far): a resumed
  run — after a pause, or a worker crash — replays the steps it already took instead of
  calling the model again, and the journal replays their tool calls;
* a model call takes at most ``model_timeout`` seconds (and what is left of the run's time),
  the gateway's own retries included: past it the run fails with a ``ModelError`` that may be
  retried.

``prompt=`` names a stored prompt of the gateway's Prompt Repository (``"triage"``, or
``"triage@3"`` for that version): its id is resolved once (``Gateway.prompt``, kept fresh),
the version is pinned for the run (journaled: a resumed run sends the same) and every model
call selects it — the gateway prepends its messages — and says so on its ``chat`` span.

What keeps the conversation within the model's window (``context_window``: the target's, else
the model object's, else :data:`CONTEXT_WINDOW`), estimated as :data:`CHARS_PER_TOKEN`
characters a token:

* past :data:`CLEAR_AT` of it, the older tool results (all but the last :data:`KEEP_RESULTS`)
  are replaced by a placeholder naming how to read them again (``read_result``), all at once;
* past :data:`COMPACT_AT`, the older turns are compacted into one summary (the task, the
  state, the decisions, what failed, the next steps) by one model call; the system message,
  the first user message, the summary and the recent turns that fit in :data:`KEEP_RECENT` of
  the window stay, cut only between turns (a call and its results are never split);
* either only when it frees :data:`MIN_FREED` of the window, so the cached prompt breaks
  rarely.

What was decided before each model call, and the summary, are journaled like the model's
steps, so a resumed run reads exactly what the model read.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, ClassVar, Final, Protocol

from pydantic import BaseModel

from trellis.contracts import ConfigurationError, InterruptResolution, ModelError, ToolStatus
from trellis.harness.adapters.base import (
    Extracted,
    Invocation,
    Narrowing,
    Output,
    ToolFormat,
    context_window,
)
from trellis.harness.clients.bifrost import PromptPin, prompt_ref
from trellis.harness.hooks import ModelCall
from trellis.harness.journal import Pending, content_key
from trellis.harness.runtime import Runtime
from trellis.harness.telemetry import model_span, usage
from trellis.harness.telemetry import output as span_output
from trellis.harness.tools import bridge
from trellis.harness.tools.base import Tool, arguments_problem
from trellis.harness.tools.convert import openai_chat, text_of

log = logging.getLogger("trellis.run")

#: Model calls with tools one run may make; then one more, without tools, for the answer.
MAX_STEPS: Final = 12
#: The longest tool result the model reads, in characters (about 5k tokens); the middle of a
#: longer one is cut.
MAX_RESULT_CHARS: Final = 20_000
#: Consecutive model steps making the same call with the same arguments that stop the run.
MAX_REPEATS: Final = 3
#: Consecutive model steps in which every call failed (no such tool, arguments that do not
#: fit, an error or a timeout) that stop the run.
ERROR_STREAK: Final = 3
#: The window assumed when neither the target nor its model says, in tokens.
CONTEXT_WINDOW: Final = 128_000
#: Characters per token, in the estimate of a request's size.
CHARS_PER_TOKEN: Final = 4
#: The share of the window past which older tool results are cleared, and the most recent
#: ones kept.
CLEAR_AT: Final = 0.5
KEEP_RESULTS: Final = 3
#: The share of the window past which older turns are compacted, and the share the recent
#: turns kept may take.
COMPACT_AT: Final = 0.75
KEEP_RECENT: Final = 0.25
#: The share of the window a clearing or a compaction must free, so the cached prompt breaks
#: rarely (and a prompt that is mostly tools and instructions is not compacted every step).
MIN_FREED: Final = 0.1
#: The tool that reads a result back, by its call id: offered once a result was cut or cleared.
READ_RESULT: Final = "read_result"
READ_RESULT_SCHEMA: Final = {
    "type": "object",
    "properties": {
        "id": {"type": "string", "description": "the id of the tool call that returned it"},
        "offset": {"type": "integer", "description": "the first character to read (0)"},
        "limit": {"type": "integer", "description": "how many characters to read"},
    },
    "required": ["id"],
    "additionalProperties": False,
}
#: What the model is asked when its steps run out, and to compact older turns.
LAST_STEP: Final = (
    "You have used every step this run allows. Do not call tools: answer now with what you "
    "have, and say what is left undone."
)
COMPACT: Final = (
    "Summarize the conversation so far for yourself, to continue the task from it: the task, "
    "the current state, the decisions made and why, what was tried and failed, the next "
    "steps, and the facts (names, ids, numbers) still needed. Do not call tools."
)
SUMMARY: Final = "Summary of the earlier steps:"
#: The outcomes that count as a failed call in an error streak.
FAILED: Final = frozenset({ToolStatus.ERROR, ToolStatus.TIMEOUT})


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
    #: the most one model call may take, in seconds (``None``: the gateway's own limit)
    model_timeout: float | None = field(default=None, kw_only=True)
    #: the model's context window in tokens, when the model object does not say
    #: (``None``: its ``context_window``, else :data:`CONTEXT_WINDOW`)
    context_window: int | None = field(default=None, kw_only=True)
    #: a stored prompt of the gateway (``"name"``, ``"name@version"``) every model call selects
    prompt: str | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        if self.model_timeout is not None and self.model_timeout <= 0:
            raise ConfigurationError("model_timeout is a number of seconds over 0")
        if self.context_window is not None and self.context_window <= 0:
            raise ConfigurationError("context_window is a number of tokens over 0")
        if self.prompt is not None:
            prompt_ref(self.prompt)
            if not isinstance(self.model, str):
                raise ConfigurationError(
                    "prompt= needs a Bifrost model name: the gateway prepends the stored prompt"
                )


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
        body: dict[str, Any] = {}
        if target.output is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": target.output.__name__,
                    "schema": target.output.model_json_schema(),
                },
            }
        loop = _Loop(target, run, await _model(target, run), body, list(native_input))
        messages = loop.messages
        result = ReActResult(messages=messages)
        streaks: dict[str, int] = {}
        failing = 0
        for step in range(target.max_steps):
            message = await loop.ask(step)
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
            ran = await loop.run(calls)
            for call, (shown, _) in zip(calls, ran, strict=True):
                messages.append({"role": "tool", "tool_call_id": call.get("id"), "content": shown})
            failing = failing + 1 if all(failed for _, failed in ran) else 0
            if failing >= ERROR_STREAK:
                raise ModelError(
                    f"ReAct stopped: every tool call failed in {failing} consecutive steps"
                )
        messages.append({"role": "user", "content": LAST_STEP})
        message = await loop.ask(target.max_steps, last=True)
        messages.append(message)
        content = message.get("content")
        if not isinstance(content, str) or not content:
            raise ModelError(
                f"ReAct stopped after {target.max_steps} model calls without an answer"
            )
        stopped = f"run {loop.runtime.run_id} stopped at its {target.max_steps} steps"
        log.warning("%s", stopped)
        loop.runtime.events.warning("max_steps", stopped)
        yield content
        result.answer = _answer(target, content)
        yield Output(result)

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


@dataclass(slots=True)
class _Loop:
    """One attempt of the loop: what it sends the model, and every result it read."""

    target: ReAct
    invocation: Invocation
    model: Any
    body: dict[str, Any]
    messages: list[dict[str, Any]]
    #: every tool result in full, by its call id (what ``read_result`` reads)
    results: dict[str, str] = field(default_factory=dict)
    #: the results replaced by a placeholder, by call id
    cleared: set[Any] = field(default_factory=set)
    #: ``read_result`` is offered (a result was cut or cleared)
    reading: bool = False

    @property
    def runtime(self) -> Runtime:
        return self.invocation.runtime

    @property
    def tools(self) -> dict[str, Tool]:
        return {t.name: t for t in self.invocation.tools}

    @property
    def window(self) -> int:
        return context_window(self.target) or CONTEXT_WINDOW

    def offered(self) -> list[dict[str, Any]]:
        """The tools offered now, sorted by name (the set only grows within a run)."""
        offered = [t for t in self.invocation.tools if self.runtime.offers(t.name)]
        tools = openai_chat.convert(offered)
        if self.reading and READ_RESULT not in self.tools:
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": READ_RESULT,
                        "description": "Read part of an earlier tool result that was cut or "
                        "cleared, by the id of the call that returned it.",
                        "parameters": READ_RESULT_SCHEMA,
                    },
                }
            )
        return sorted(tools, key=lambda t: t["function"]["name"])

    async def ask(self, step: int, *, last: bool = False) -> dict[str, Any]:
        """The model's next step, the conversation fitted to the window first; the ``last``
        one without tools."""
        await self._fitted(step)
        offered = self.offered()
        request = {**self.body, "tools": offered} if offered else dict(self.body)
        if last and offered:
            request["tool_choice"] = "none"
        return await self._step(step, self.messages, request)

    async def _step(
        self,
        step: int,
        messages: list[dict[str, Any]],
        request: dict[str, Any],
        *,
        kind: str = "react",
    ) -> dict[str, Any]:
        """One model call: the journal's, when a resumed run already made it with this
        conversation, else the model's within ``model_timeout`` — the run's ``before_model``
        hooks may rewrite its messages, and its ``after_model`` hooks see the reply —
        (journaled, and saved as progress in a worker)."""
        runtime, limit = self.runtime, self.target.model_timeout
        key = content_key(kind, step, _history(messages))
        replayed, recorded = runtime.replay.call(key)
        if replayed:
            return dict(recorded)
        target = self.target.model
        name = target if isinstance(target, str) else type(target).__name__
        prompt = self.model.prompt if isinstance(self.model, _Named) else None
        extra = prompt.attributes() if prompt is not None else None
        hooks = runtime.agent.hooks
        call = await hooks.model(ModelCall("react", messages, model=name))
        with model_span(name, call.messages, extra=extra) as span:
            try:
                async with runtime.limited(limit):
                    reply = await self.model.complete(call.messages, **request)
            except TimeoutError as exc:
                within = "" if limit is None else f" within {limit:g}s"
                late = ModelError(
                    f"the model did not answer{within}", source="react", retryable=True
                )
                await hooks.failed("model", late)
                raise late from exc
            except Exception as exc:
                await hooks.failed("model", exc)
                raise
            await hooks.answered(call, reply)
            usage(span, reply)
            message = _message(reply)
            span_output(span, message.get("content") or message.get("tool_calls"))
        runtime.replay.record_call(key, message)
        await runtime.progress(now=False)
        return message

    # ------------------------------------------------------------------ the calls of a step
    async def run(self, calls: list[dict[str, Any]]) -> list[tuple[str, bool]]:
        """What the model reads for each of its calls, in its order, and whether the call
        failed: the reads at once, then the others one at a time. Each call's step is numbered
        in the model's order before any runs."""
        ran: list[tuple[str, bool] | None] = [None] * len(calls)
        together: list[tuple[int, Tool, dict[str, Any], int]] = []
        after: list[tuple[int, Tool, dict[str, Any], int]] = []
        for n, call in enumerate(calls):
            function = call.get("function") or {}
            called = function.get("name", "")
            found = self.tools.get(called)
            if found is None:
                ran[n] = (
                    self._read(function.get("arguments"))
                    if called == READ_RESULT
                    else (f"there is no tool {called!r}", True)
                )
                continue
            args, problem = _arguments(found.spec.input_schema, function.get("arguments"))
            if args is None:
                fix = "Call it again with arguments that fit its schema."
                ran[n] = (f"{called} was not run: {problem}. {fix}", True)
                continue
            reads = found.spec.side_effects == "read" or found.spec.idempotent
            (together if reads else after).append((n, found, args, self.runtime.next_step()))
        done = await asyncio.gather(
            *(self._call(tool, args, calls[n], step) for n, tool, args, step in together),
            return_exceptions=True,
        )
        for (n, *_), outcome in zip(together, done, strict=True):
            if isinstance(outcome, BaseException):
                raise outcome
            ran[n] = outcome
        for n, tool, args, step in after:
            ran[n] = await self._call(tool, args, calls[n], step)
        shown: list[tuple[str, bool]] = []
        for call, outcome in zip(calls, ran, strict=True):
            assert outcome is not None
            self.results.setdefault(str(call.get("id")), outcome[0])
            shown.append(outcome)
        return shown

    async def _call(
        self, tool: Tool, args: dict[str, Any], call: dict[str, Any], step: int
    ) -> tuple[str, bool]:
        outcome = await bridge.call(tool, args, call_id=call.get("id"), step=step)
        text = text_of(outcome.output)
        call_id = str(call.get("id"))
        self.results[call_id] = text
        return await self._bounded(tool.name, call_id, text, step), outcome.status in FAILED

    def _read(self, raw: Any) -> tuple[str, bool]:
        """``read_result``: part of a result the run already has, from where it was cut."""
        args, problem = _arguments(READ_RESULT_SCHEMA, raw)
        if args is None:
            return f"{READ_RESULT} was not run: {problem}", True
        ref, limit = str(args["id"]), self.target.max_result_chars
        text = self.results.get(ref)
        if text is None:
            return f"there is no result {ref!r} to read", True
        offset = max(0, args.get("offset", 0))
        end = offset + min(max(1, args.get("limit", limit)), limit)
        part = text[offset:end]
        if end < len(text):
            more = f'{READ_RESULT}(id="{ref}", offset={end})'
            part += f"\n…[{len(text) - end} more characters: {more}]"
        return part, False

    async def _bounded(self, tool: str, call_id: str, text: str, step: int) -> str:
        """``text`` with its middle cut when it is longer than ``max_result_chars``: its head and
        tail, and a marker saying how to read the rest. The full text is kept as a run
        artifact when the run store takes one, once: a re-run reads which from the journal."""
        limit = self.target.max_result_chars
        if len(text) <= limit:
            return text
        self.reading = True
        runtime = self.runtime
        key = content_key("react-kept", step, call_id)
        replayed, kept = runtime.replay.call(key)
        if not replayed:
            try:
                data = json.dumps({"tool": tool, "output": text}).encode()
                ref = await runtime.agent.harness.runs.artifacts.upload(
                    runtime.run_id, data, worker_id=runtime.worker_id, tenant=runtime.tenant
                )
                kept = ref.artifact_id
            except Exception as exc:  # the cut result still goes to the model
                log.warning(
                    "the full result of %s in run %s was not kept: %s", tool, runtime.run_id, exc
                )
            runtime.replay.record_call(key, kept)
        head = limit // 2
        tail = limit - head
        named = f"; the full result is run artifact {kept}" if kept else ""
        return (
            f"{text[:head]}\n…[cut: {tool} returned {len(text)} characters, the first {head} "
            f'and the last {tail} are shown; {READ_RESULT}(id="{call_id}", offset={head}) '
            f"reads the rest{named}]…\n{text[-tail:]}"
        )

    # ------------------------------------------------------------------ the context
    async def _fitted(self, step: int) -> None:
        """Before a model call: the older tool results cleared, or the older turns compacted,
        when the conversation grew past their share of the window — as the journal says a
        resumed run decided at this step, else as decided now (and journaled)."""
        runtime = self.runtime
        key = content_key("react-context", step, _history(self.messages))
        replayed, decided = runtime.replay.call(key)
        if not replayed:
            decided = self._decided()
            runtime.replay.record_call(key, decided)
        if decided.get("cleared"):
            self._clear(set(decided["cleared"]))
        if decided.get("compacted") is not None:
            await self._compact(step, decided["compacted"])

    def _decided(self) -> dict[str, Any]:
        window = self.window
        used = _tokens(self.messages, self.offered())
        decided: dict[str, Any] = {}
        results = [m for m in self.messages if m.get("role") == "tool"]
        older = [m for m in results[:-KEEP_RESULTS] if m.get("tool_call_id") not in self.cleared]
        freed = sum(
            _tokens(m.get("content")) - _tokens(_placeholder(m.get("tool_call_id"))) for m in older
        )
        if used > window * CLEAR_AT and freed >= window * MIN_FREED:
            decided["cleared"] = [m.get("tool_call_id") for m in older]
            used -= freed
        if used > window * COMPACT_AT and (cut := self._cut()) is not None:
            decided["compacted"] = cut
        return decided

    def _clear(self, ids: set[Any]) -> None:
        for n, message in enumerate(self.messages):
            ref = message.get("tool_call_id")
            if message.get("role") == "tool" and ref in ids and ref not in self.cleared:
                self.messages[n] = {**message, "content": _placeholder(ref)}
        self.cleared |= ids
        self.reading = True

    def _cut(self) -> int | None:
        """Where the recent turns kept begin: the most recent turns that fit in
        :data:`KEEP_RECENT` of the window, the last one at least, each turn whole (a message
        and its tool results); ``None`` when the older turns are too few to compact."""
        messages, head = self.messages, _head(self.messages)
        turns = [n for n in range(head, len(messages)) if messages[n].get("role") != "tool"]
        kept = turns[-1] if turns else head
        for start in reversed(turns[:-1]):
            if _tokens(messages[start:]) > self.window * KEEP_RECENT:
                break
            kept = start
        worth = kept > head and _tokens(messages[head:kept]) >= self.window * MIN_FREED
        return kept if worth else None

    async def _compact(self, step: int, cut: int) -> None:
        """The turns between the first user message and ``cut`` replaced by one summary the
        model writes (journaled as a model step of its own)."""
        offered = self.offered()
        request: dict[str, Any] = {"tools": offered, "tool_choice": "none"} if offered else {}
        asked = [*self.messages[:cut], {"role": "user", "content": COMPACT}]
        reply = await self._step(step, asked, request, kind="react-compact")
        summary = reply.get("content")
        if not isinstance(summary, str) or not summary:
            log.warning("run %s: no summary was written; nothing compacted", self.runtime.run_id)
            return
        head = _head(self.messages)
        self.messages[head:cut] = [{"role": "user", "content": f"{SUMMARY}\n\n{summary}"}]


def _history(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The conversation a step is keyed by: the system message (the memory context may differ
    between attempts) left out."""
    return [m for m in messages if m.get("role") != "system"]


def _head(messages: list[dict[str, Any]]) -> int:
    """Where the turns begin: after the first user message."""
    for n, message in enumerate(messages):
        if message.get("role") == "user":
            return n + 1
    return 1


def _tokens(*parts: Any) -> int:
    return len(json.dumps(parts, default=str)) // CHARS_PER_TOKEN


def _placeholder(call_id: Any) -> str:
    return (
        f'[result cleared to keep the context small: {READ_RESULT}(id="{call_id}") reads it again]'
    )


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


def _arguments(schema: dict[str, Any] | None, raw: Any) -> tuple[dict[str, Any] | None, str]:
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
    problem = arguments_problem(schema or {}, args)
    return (None, problem) if problem else (args, "")


async def _model(target: ReAct, run: Invocation) -> Any:
    if not isinstance(target.model, str):
        return target.model
    runtime = run.runtime
    gateway = runtime.agent.harness.gateway
    if gateway is None:
        raise ConfigurationError("ReAct with a model name needs BIFROST_URL")
    if target.prompt is None:
        return _Named(gateway, target.model)
    # the version a resumed run pinned, else the one the gateway resolves now (journaled)
    key = content_key("react-prompt", target.prompt)
    replayed, recorded = runtime.replay.call(key)
    if replayed:
        return _Named(gateway, target.model, PromptPin(**recorded))
    prompt = await gateway.prompt(target.prompt)
    runtime.replay.record_call(key, dataclasses.asdict(prompt))
    return _Named(gateway, target.model, prompt)


@dataclass(frozen=True, slots=True)
class _Named:
    gateway: Any
    model: str
    #: the stored prompt every call selects
    prompt: PromptPin | None = None

    async def complete(self, messages: list[dict[str, Any]], **body: Any) -> dict[str, Any]:
        return await self.gateway.complete(messages, model=self.model, prompt=self.prompt, **body)


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
