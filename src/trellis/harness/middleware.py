"""The harness as LangChain middleware: what a ``create_agent`` graph (Deep Agents' too) gets
from the harness, added where LangChain and Deep Agents have no native piece. ``ReAct(...)``
is a ``create_agent`` graph built with all of them, beside the native middleware it turns on;
any ``create_agent`` or Deep Agents graph may take any of them:

    graph = create_agent(model, middleware=[HarnessTools(), ModelHooks()],
                         checkpointer=RunCheckpointer())
    agent = h.wrap(graph, id="support", tools=[refund])

* :class:`HarnessTools` — the run's harness tools offered per model call (the tool hints'
  narrowing, the memory tools, what ``tool_search`` found), sorted by name so the cached prompt
  holds; every call goes through the bridge (``tools.bridge``: governance, journal, record),
  numbered in the model's order, and a call that does more than read waits for the earlier
  ones of the same model step (the model's order); a graph with it takes ``h.wrap(tools=)``;
* :class:`ModelHooks` — around every model call: the run's ``before_model`` / ``after_model`` /
  ``on_error`` hooks, a ``chat`` span with the usage (redacted, as every span), the most the
  call may take (``timeout``, and what is left of the run's time), the gateway's stored prompt
  pinned for the run (``prompt``: journaled, sent as headers, on the span), and the model
  client's errors as a :class:`~trellis.contracts.ModelError` saying whether it may pass;
* :class:`StepLimit` — at ``max_steps`` model calls one more, without tools, for the best
  answer (and a ``max_steps`` warning event), instead of LangChain's canned limit message;
* :class:`StallGuard` — the same call with the same arguments in ``max_repeats`` consecutive
  steps, or :data:`ERROR_STREAK` steps in which every call failed, stop the run;
* :func:`read_result` — the tool that reads an earlier result back by its call's id: what a
  ``ContextEditingMiddleware`` placeholder points to (the state keeps the result whole);
* :class:`RunCheckpointer` — a LangGraph checkpointer that keeps only the latest checkpoint,
  in the run's journal (``thread_id`` is the run): a resume — in any worker, after a pause or
  a crash — continues where the graph stopped, without asking the model again.

The harness's middleware run inside a harness run (``trellis.current()``); outside one they
step aside (:class:`ModelHooks` runs the hooks it was given, :class:`RunCheckpointer` keeps the
checkpoint in memory). They are asynchronous: a graph is run with ``ainvoke``/``astream``.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import logging
import weakref
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Annotated, Any, Final, NotRequired, cast

from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
)
from langchain.agents.middleware.types import (
    ExtendedModelResponse,
    PrivateStateAttr,
    hook_config,
)
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    convert_to_messages,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
    get_checkpoint_metadata,
)
from langgraph.prebuilt import InjectedState
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

from trellis.contracts import ConfigurationError, ModelError, ToolStatus
from trellis.harness.hooks import Hooks, ModelCall, running
from trellis.harness.journal import content_key
from trellis.harness.prompts import Prompt, ResolvedPrompt
from trellis.harness.repository import pinned
from trellis.harness.runtime import DEFERRED, Runtime, current
from trellis.harness.telemetry import model_span, usage
from trellis.harness.telemetry import output as span_output
from trellis.harness.tools import bridge
from trellis.harness.tools.base import Tool
from trellis.harness.tools.convert import langchain as converted
from trellis.harness.tools.convert import text_of

log = logging.getLogger("trellis.run")

#: Model calls with tools one run may make (:class:`StepLimit`); then one more, without tools.
MAX_STEPS: Final = 12
#: Consecutive steps making the same call with the same arguments that stop the run.
MAX_REPEATS: Final = 3
#: Consecutive steps in which every call failed (no such tool, arguments that do not fit, an
#: error or a timeout) that stop the run.
ERROR_STREAK: Final = 3
#: The tool that reads a cleared result back by its call's id (:func:`read_result`), and the
#: most it reads at once, in characters.
READ_RESULT: Final = "read_result"
READ_LIMIT: Final = 20_000
#: What the model is asked when its steps run out.
LAST_STEP: Final = (
    "You have used every step this run allows. Do not call tools: answer now with what you "
    "have, and say what is left undone."
)
#: The tool outcomes that count as a failed call (in a streak; a ``ToolMessage``'s status).
FAILED: Final = frozenset({ToolStatus.ERROR, ToolStatus.TIMEOUT})
#: The state key of :class:`StepLimit`: the model calls of this invocation.
STEPS: Final = "trellis_steps"
#: The openai client's statuses a call may pass after (its own retry rule).
RETRYABLE_STATUS: Final = frozenset({408, 409, 429})


# --------------------------------------------------------------------------- tools
class _HarnessToolsState(AgentState[Any]):
    # the key is ``adapters.langgraph.HARNESS_TOOLS``: the harness reads it on the graph
    trellis_harness_tools: NotRequired[Annotated[bool, PrivateStateAttr]]


@dataclass(eq=False)
class _Step:
    """One model step's calls as they run: the number of its first call, its harness calls
    that do more than read in the model's order, the ones that finished (in this attempt, or
    in an earlier one: a resume runs only the calls that had not), and the exception one of
    them stopped on (a pause: the later ones wait for the resume)."""

    first: int
    writes: list[str]
    finished: set[str]
    stopped: BaseException | None = None
    turn: asyncio.Condition = field(default_factory=asyncio.Condition)


class HarnessTools(AgentMiddleware):
    """The run's harness tools, offered per model call and run through the bridge."""

    state_schema = _HarnessToolsState

    def __init__(self) -> None:
        super().__init__()
        self._tools: weakref.WeakKeyDictionary[Runtime, dict[str, BaseTool]] = (
            weakref.WeakKeyDictionary()
        )
        self._steps: weakref.WeakKeyDictionary[Runtime, dict[str, _Step]] = (
            weakref.WeakKeyDictionary()
        )

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        runtime = current()
        if runtime is None:
            return await handler(request)
        harness = self._tools.get(runtime)
        if harness is None:
            made = converted.convert(list(runtime.toolbox.values()))
            harness = self._tools[runtime] = {t.name: t for t in made}
        own = [t for t in request.tools if _named(t) not in runtime.toolbox]
        offered = [t for name, t in harness.items() if runtime.offers(name)]
        return await handler(request.override(tools=sorted([*own, *offered], key=_named)))

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        runtime = current()
        call = request.tool_call
        tool = runtime.toolbox.get(call["name"]) if runtime is not None else None
        if runtime is None or tool is None:
            return await handler(request)
        step, n = self._step(runtime, request)
        call_id = str(call["id"])
        if call_id in step.writes:
            async with _turn(runtime, step, call_id):
                outcome = await bridge.call(
                    tool, call["args"], call_id=call_id, step=step.first + n
                )
        else:
            outcome = await bridge.call(tool, call["args"], call_id=call_id, step=step.first + n)
        return ToolMessage(
            content=text_of(outcome.output),
            tool_call_id=call_id,
            name=tool.name,
            status="error" if outcome.status in FAILED else "success",
        )

    def _step(self, runtime: Runtime, request: ToolCallRequest) -> tuple[_Step, int]:
        """The model step the call is one of (numbered on its first call's arrival), and the
        call's place in it."""
        call_id = request.tool_call["id"]
        messages = request.state.get("messages", []) if isinstance(request.state, dict) else []
        message = next(
            (
                m
                for m in reversed(messages)
                if isinstance(m, AIMessage) and any(c["id"] == call_id for c in m.tool_calls)
            ),
            AIMessage(content="", tool_calls=[request.tool_call]),
        )
        calls = message.tool_calls
        key = message.id or ",".join(str(c["id"]) for c in calls)
        steps = self._steps.setdefault(runtime, {})
        step = steps.get(key)
        if step is None:
            writes = [str(c["id"]) for c in calls if _writes(runtime.toolbox.get(c["name"]))]
            ran = {w for w in writes if runtime.replay.journal.calls.get(_ran(w))}
            step = steps[key] = _Step(runtime.next_step(len(calls)), writes, ran)
        return step, next(n for n, c in enumerate(calls) if c["id"] == call_id)


def _writes(tool: Tool | None) -> bool:
    """A harness tool that does more than read, and is not idempotent: it runs in its turn."""
    return tool is not None and tool.spec.side_effects != "read" and not tool.spec.idempotent


@contextlib.asynccontextmanager
async def _turn(runtime: Runtime, step: _Step, call_id: str) -> AsyncIterator[None]:
    """The call runs once the step's earlier writes finished (journaled, so a resume knows);
    when one of them stopped (it paused), the call waits for the resume instead: a LangGraph
    interrupt that asks nothing (:data:`~trellis.harness.runtime.DEFERRED`), or the run's own
    pause without a checkpointer."""
    earlier = step.writes[: step.writes.index(call_id)]
    async with step.turn:
        await step.turn.wait_for(
            lambda: step.stopped is not None or all(w in step.finished for w in earlier)
        )
    if step.stopped is not None:
        if runtime.suspend is not None and runtime.pending is not None:
            runtime.suspend({DEFERRED: True})
        raise step.stopped
    try:
        yield
    except BaseException as exc:
        step.stopped = exc
        raise
    else:
        runtime.replay.record_call(_ran(call_id), True)
        step.finished.add(call_id)
    finally:
        async with step.turn:
            step.turn.notify_all()


def _ran(call_id: str) -> str:
    """The journal key saying a call of a model step finished."""
    return content_key("ran", call_id)


def _named(tool: BaseTool | dict[str, Any]) -> str:
    if isinstance(tool, dict):
        return str((tool.get("function") or tool).get("name", ""))
    return tool.name


# --------------------------------------------------------------------------- the model call
class ModelHooks(AgentMiddleware):
    """Around every model call: the task kept ahead of a summary (the conversation's leading
    system messages, the pushed memory context among them, and its first user message), the
    run's model hooks (then ``hooks``), a ``chat`` span with the usage, at most ``timeout``
    seconds (and what is left of the run's time), and the ``prompt`` of the harness's prompt
    sources (``"name"``, ``"name@version"``, a ``Prompt``), pinned for the run: a stored prompt
    of the gateway is selected by every call (headers the gateway reads: the model is a gateway
    model, ``ChatOpenAI``); any other is rendered (``prompt_vars`` fill its ``{{variables}}``)
    into the instructions, before the system prompt, its other messages before the
    conversation."""

    def __init__(
        self,
        *hooks: Hooks,
        timeout: float | None = None,
        prompt: str | Prompt | None = None,
        prompt_vars: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if timeout is not None and timeout <= 0:
            raise ConfigurationError("a model timeout is a number of seconds over 0")
        if isinstance(prompt, str):
            pinned(prompt)
        if prompt_vars is not None and prompt is None:
            raise ConfigurationError("prompt_vars= fills the variables of a prompt=")
        self.given = list(hooks)
        self.timeout = timeout
        self.prompt = prompt
        self.prompt_vars = dict(prompt_vars or {})
        self._prompts: weakref.WeakKeyDictionary[Runtime, ResolvedPrompt] = (
            weakref.WeakKeyDictionary()
        )

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        runtime = current()
        hooks = running(*self.given)
        request = _with_task(request)
        prompt = await self._pinned(runtime)
        if prompt is not None:
            request = self._prompted(request, prompt)
        name = _model_name(request.model)
        asked = ModelCall(
            "langgraph", list(request.messages), model=name, system=request.system_message
        )
        call = await hooks.model(asked) if hooks else asked
        if call is not asked:
            request = request.override(messages=call.messages, system_message=call.system)
        extra = prompt.attributes() if prompt is not None else None
        with model_span(name or "model", _said(request), extra=extra) as span:
            try:
                async with _limited(runtime, self.timeout):
                    reply = await handler(request)
            except Exception as exc:
                error = model_error(exc, self.timeout)
                await hooks.failed("model", error)
                if error is exc:
                    raise
                raise error from exc
            await hooks.answered(call, reply)
            _replied(span, reply)
        return reply

    async def _pinned(self, runtime: Runtime | None) -> ResolvedPrompt | None:
        """The prompt's version this run uses: the journal's (a resumed run uses the same),
        else the one its source gives now (journaled)."""
        if self.prompt is None:
            return None
        if runtime is None:
            raise ConfigurationError(
                "ModelHooks(prompt=) pins the prompt for a harness run: outside one, render it "
                "yourself (h.prompt_messages), or give the model client h.model_headers(prompt=)"
            )
        found = self._prompts.get(runtime)
        if found is None:
            found = await runtime.agent.harness.prompts.get(self.prompt, runtime=runtime)
            if found.pin() is not None and self.prompt_vars:
                raise ConfigurationError(
                    f"prompt= names {found.name}, a stored prompt of the gateway: the gateway "
                    "prepends it as it is stored (no prompt_vars=)"
                )
            self._prompts[runtime] = found
        return found

    def _prompted(self, request: ModelRequest[Any], prompt: ResolvedPrompt) -> ModelRequest[Any]:
        pin = prompt.pin()
        if pin is not None:
            sent = request.model_settings.get("extra_headers", {})
            headers = {**sent, **pin.options().headers()}
            return request.override(
                model_settings={**request.model_settings, "extra_headers": headers}
            )
        rendered = prompt.messages(**self.prompt_vars)
        own = [str(m["content"]) for m in rendered if m.get("role") == "system"]
        rest = [m for m in rendered if m.get("role") != "system"]
        system = request.system_message
        text = "\n\n".join(part for part in [*own, system.text if system else ""] if part)
        return request.override(
            system_message=SystemMessage(content=text),
            messages=[*cast(list[AnyMessage], convert_to_messages(rest)), *request.messages],
        )


def _with_task(request: ModelRequest[Any]) -> ModelRequest[Any]:
    """The request with the task kept ahead of a summary: a summarization middleware (Deep
    Agents', LangChain's) replaces the older turns by a summary, the run's task and its pushed
    memory context among them — the conversation's leading system messages and first user
    message are put back before it."""
    messages = request.messages
    if not messages or not _summary(messages[0]):
        return request
    state = list(request.state.get("messages", []))
    head: list[AnyMessage] = []
    for message in state:
        if _summary(message):
            break
        head.append(message)
        if isinstance(message, HumanMessage):
            break
    else:
        return request
    sent = {m.id for m in messages if m.id is not None}
    kept = [m for m in head if m.id is None or m.id not in sent]
    return request.override(messages=[*kept, *messages]) if kept else request


def _summary(message: BaseMessage) -> bool:
    return isinstance(message, HumanMessage) and (
        message.additional_kwargs.get("lc_source") == "summarization"
    )


@contextlib.asynccontextmanager
async def _limited(runtime: Runtime | None, seconds: float | None) -> AsyncIterator[None]:
    """At most ``seconds`` and what is left of the run's time."""
    left = runtime.remaining() if runtime is not None else None
    bounds = [s for s in (seconds, left) if s is not None]
    limit = min(bounds) if bounds else None
    if runtime is None:
        async with asyncio.timeout(limit):
            yield
        return
    async with runtime.limited(limit):
        yield


def model_error(exc: Exception, timeout: float | None = None) -> Exception:
    """A failed model call as the harness reports it: out of time, or the openai client's
    error, as a :class:`ModelError` that says whether a retry may pass (the client's own rule:
    no answer, a timeout, 408, 409, 429, 5xx); anything else as it is."""
    if isinstance(exc, TimeoutError):
        within = "" if timeout is None else f" within {timeout:g}s"
        return ModelError(f"the model did not answer{within}", source="langgraph", retryable=True)
    try:
        import openai  # noqa: PLC0415 - the model client of the react extra
    except ImportError:  # pragma: no cover - the react extra installs it
        return exc
    if isinstance(exc, openai.APIConnectionError):  # a timeout is one
        return ModelError(f"the model call failed: {exc}", source="langgraph", retryable=True)
    if isinstance(exc, openai.APIStatusError):
        status = exc.status_code
        retryable = status in RETRYABLE_STATUS or status >= 500
        return ModelError(
            f"the model call failed ({status}): {exc.message}",
            source="langgraph",
            retryable=retryable,
            details={"status": status},
        )
    return exc


def _model_name(model: Any) -> str | None:
    for attribute in ("model_name", "model"):
        name = getattr(model, attribute, None)
        if isinstance(name, str) and name:
            return name
    return None


def _said(request: ModelRequest[Any]) -> list[dict[str, Any]]:
    """The call's messages as a span keeps them: structured (the redactor walks them)."""
    system = [request.system_message] if request.system_message is not None else []
    return [_spoken(m) for m in [*system, *request.messages]]


def _spoken(message: BaseMessage) -> dict[str, Any]:
    said: dict[str, Any] = {"role": message.type, "content": message.content}
    if isinstance(message, AIMessage) and message.tool_calls:
        said["tool_calls"] = [{"name": c["name"], "args": c["args"]} for c in message.tool_calls]
    return said


def _replied(span: Any, reply: ModelResponse[Any]) -> None:
    message = _ai(reply)
    if message is None:
        return
    metadata = message.response_metadata or {}
    counts = message.usage_metadata or {}
    reason = metadata.get("finish_reason")
    usage(
        span,
        model=metadata.get("model_name"),
        input_tokens=counts.get("input_tokens"),
        output_tokens=counts.get("output_tokens"),
        finish_reasons=[reason] if isinstance(reason, str) else [],
    )
    span_output(span, message.text or _spoken(message).get("tool_calls"))


def _ai(reply: ModelResponse[Any]) -> AIMessage | None:
    return next((m for m in reversed(reply.result) if isinstance(m, AIMessage)), None)


# --------------------------------------------------------------------------- limits
class _StepsState(AgentState[Any]):
    trellis_steps: NotRequired[Annotated[int, PrivateStateAttr]]


class StepLimit(AgentMiddleware):
    """At most ``max_steps`` model calls with tools in one invocation; then one more, told to
    answer with what it has (:data:`LAST_STEP`) and to call no tool (``tool_choice="none"``;
    with structured output, offered none but the answer's own), and a ``max_steps`` warning on
    the run's stream. No answer then fails the run (``ModelError``)."""

    state_schema = _StepsState

    def __init__(self, max_steps: int = MAX_STEPS) -> None:
        super().__init__()
        if max_steps < 1:
            raise ConfigurationError("max_steps is a number of model calls from 1")
        self.max_steps = max_steps

    async def abefore_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        return {STEPS: 0}

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any] | ExtendedModelResponse[Any]:
        taken = request.state.get(STEPS, 0)
        if taken < self.max_steps:
            reply = await handler(request)
            return ExtendedModelResponse(
                model_response=reply, command=Command(update={STEPS: taken + 1})
            )
        told = [*request.messages, HumanMessage(content=LAST_STEP)]
        if request.response_format is not None:
            # the answer may be the structured output's own tool call: no other tool
            last = request.override(messages=told, tools=[])
        else:
            # the same tools (the cached prompt holds), none of them called
            last = request.override(
                messages=told, tool_choice="none" if request.tools else request.tool_choice
            )
        reply = await handler(last)
        message = _ai(reply)
        answered = reply.structured_response is not None or (
            message is not None and not message.tool_calls and bool(message.text)
        )
        if not answered:
            raise ModelError(
                f"stopped after {self.max_steps} model calls without an answer",
                source="langgraph",
            )
        runtime = current()
        stopped = f"run {runtime.run_id if runtime else '-'} stopped at its {self.max_steps} steps"
        log.warning("%s", stopped)
        if runtime is not None:
            runtime.events.warning("max_steps", stopped)
        return reply


class StallGuard(AgentMiddleware):
    """A run that goes nowhere stops (``ModelError``): the same call with the same arguments
    in ``max_repeats`` consecutive steps (before it runs again), or :data:`ERROR_STREAK`
    consecutive steps in which every call failed. A step whose calls were all malformed (their
    arguments not JSON) is not the end of the run: the model is asked again, and LangChain
    tells it what was wrong."""

    @hook_config(can_jump_to=["model"])
    async def aafter_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        last = state["messages"][-1] if state.get("messages") else None
        if isinstance(last, AIMessage) and last.invalid_tool_calls and not last.tool_calls:
            return {"jump_to": "model"}
        return None

    def __init__(self, max_repeats: int = MAX_REPEATS) -> None:
        super().__init__()
        if max_repeats < 2:
            raise ConfigurationError("max_repeats is a number of steps from 2")
        self.max_repeats = max_repeats

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        history = list(request.state.get("messages", []))
        failing = _failing(history)
        if failing >= ERROR_STREAK:
            raise ModelError(
                f"stopped: every tool call failed in {failing} consecutive steps",
                source="langgraph",
            )
        reply = await handler(request)
        message = _ai(reply)
        if message is not None and message.tool_calls:
            steps = _steps(history)
            for call in message.tool_calls:
                key = _call_key(call)
                streak = 1
                for earlier in steps:
                    if key not in earlier:
                        break
                    streak += 1
                if streak >= self.max_repeats:
                    raise ModelError(
                        f"stopped: the model called {call['name']!r} with the same arguments "
                        f"{streak} times in a row (a stall)",
                        source="langgraph",
                    )
        return reply


def _call_key(call: Any) -> str:
    return content_key("stall", call["name"], call["args"])


def _turn_messages(history: list[AnyMessage]) -> list[AnyMessage]:
    """The messages since the last user message (this invocation's steps)."""
    for n in range(len(history) - 1, -1, -1):
        if isinstance(history[n], HumanMessage):
            return history[n + 1 :]
    return history


def _steps(history: list[AnyMessage]) -> list[set[str]]:
    """The calls of each step of this invocation, the latest first."""
    return [
        {_call_key(c) for c in m.tool_calls}
        for m in reversed(_turn_messages(history))
        if isinstance(m, AIMessage)
    ]


def _failing(history: list[AnyMessage]) -> int:
    """How many of the latest steps had every call fail."""
    results = {m.tool_call_id: m.status for m in history if isinstance(m, ToolMessage)}
    streak = 0
    for message in reversed(_turn_messages(history)):
        if not isinstance(message, AIMessage):
            continue
        statuses = [results.get(c["id"] or "") for c in message.tool_calls]
        if not statuses or any(s != "error" for s in statuses):
            break
        streak += 1
    return streak


# --------------------------------------------------------------------------- results
def read_result(most: int = READ_LIMIT) -> BaseTool:
    """The ``read_result`` tool: part of an earlier tool result, by the id of the call that
    returned it — what a result ``ContextEditingMiddleware`` cleared from the request (the graph's
    state keeps it whole) is read again with, at most ``most`` characters at a time."""

    async def read(
        id: str,
        state: Annotated[dict[str, Any], InjectedState],
        offset: int = 0,
        limit: int = most,
    ) -> str:
        """Read an earlier tool result that was cleared to keep the context small, by the id
        of the call that returned it (from offset, at most limit characters)."""
        text = next(
            (
                m.content
                for m in state.get("messages", [])
                if isinstance(m, ToolMessage) and m.tool_call_id == id
            ),
            None,
        )
        if not isinstance(text, str):
            return f"there is no result {id!r} to read"
        start = max(0, offset)
        end = start + min(max(1, limit), most)
        part = text[start:end]
        if end < len(text):
            part += (
                f'\n…[{len(text) - end} more characters: {READ_RESULT}(id="{id}", offset={end})]'
            )
        return part

    return StructuredTool.from_function(coroutine=read, name=READ_RESULT)


# --------------------------------------------------------------------------- checkpoints
class RunCheckpointer(BaseCheckpointSaver[int]):
    """A checkpointer that keeps the latest checkpoint only — and, in a harness run whose
    ``thread_id`` is the run (``LangGraphAdapter`` sees to it), keeps it in the run's journal
    (``Journal.graph``), saved with the run's progress and with its pause. A resume in any
    worker continues where the graph stopped: the model is not asked again, a summary is not
    written again, a tool that finished beside a pause is not run again. Outside a run it is
    an in-memory checkpointer. A graph whose state uses LangGraph's ``DeltaChannel`` needs the
    whole history and is not for it."""

    def __init__(self) -> None:
        super().__init__()
        self._threads: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------ storage
    def _slot(self, thread_id: str) -> dict[str, Any]:
        """Where the thread's checkpoints are: the run's journal, for the run's own thread."""
        runtime = current()
        if runtime is not None and runtime.run_id == thread_id:
            journal = runtime.replay.journal
            if journal.graph is None:
                journal.graph = {}
            return journal.graph
        return self._threads.setdefault(thread_id, {})

    def _dumped(self, value: Any) -> list[str]:
        kind, data = self.serde.dumps_typed(value)
        return [kind, base64.b64encode(data).decode()]

    def _loaded(self, dumped: list[str]) -> Any:
        return self.serde.loads_typed((dumped[0], base64.b64decode(dumped[1])))

    # ------------------------------------------------------------------ the saver
    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        thread_id, ns = _where(config)
        record = self._slot(thread_id).get(ns) or {}
        wanted = get_checkpoint_id(config)
        latest = record.get("id")
        if latest is None or (wanted and wanted != latest):
            return None
        parent = record.get("parent")
        return CheckpointTuple(
            config=_config(thread_id, ns, latest),
            checkpoint=self._loaded(record["checkpoint"]),
            metadata=self._loaded(record["metadata"]),
            pending_writes=[
                (task, channel, self._loaded(value))
                for checkpoint_id, task, _, channel, value, _ in record.get("writes", [])
                if checkpoint_id == latest
            ],
            parent_config=_config(thread_id, ns, parent) if parent else None,
        )

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        threads = [_configurable(config)["thread_id"]] if config else list(self._threads)
        found = [
            tuple_
            for thread_id in threads
            for ns in list(self._slot(thread_id))
            if (tuple_ := self.get_tuple(_config(thread_id, ns, None))) is not None
            and all(tuple_.metadata.get(k) == v for k, v in (filter or {}).items())
            and (before is None or _id(tuple_) < str(get_checkpoint_id(before)))
        ]
        yield from found[:limit]

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        thread_id, ns = _where(config)
        slot = self._slot(thread_id)
        writes = [w for w in (slot.get(ns) or {}).get("writes", []) if w[0] >= checkpoint["id"]]
        slot[ns] = {
            "id": checkpoint["id"],
            "checkpoint": self._dumped(checkpoint),
            "metadata": self._dumped(get_checkpoint_metadata(config, metadata)),
            "parent": _configurable(config).get("checkpoint_id"),
            "writes": writes,
        }
        return _config(thread_id, ns, checkpoint["id"])

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        thread_id, ns = _where(config)
        checkpoint_id = _configurable(config)["checkpoint_id"]
        record = self._slot(thread_id).setdefault(ns, {})
        kept: list[list[Any]] = record.setdefault("writes", [])
        for n, (channel, value) in enumerate(writes):
            index = WRITES_IDX_MAP.get(channel, n)
            same = [w for w in kept if w[0] == checkpoint_id and w[1] == task_id and w[2] == index]
            if same and index >= 0:
                continue
            for old in same:
                kept.remove(old)
            kept.append([checkpoint_id, task_id, index, channel, self._dumped(value), task_path])

    def delete_thread(self, thread_id: str) -> None:
        """Forget a thread kept outside a run (a run's own goes with its journal)."""
        self._threads.pop(thread_id, None)

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        return self.get_tuple(config)

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        for found in self.list(config, filter=filter, before=before, limit=limit):
            yield found

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        saved = self.put(config, checkpoint, metadata, new_versions)
        runtime = current()
        if runtime is not None and runtime.run_id == _where(config)[0]:
            await runtime.progress(now=False)  # a worker's run: saved with its progress
        return saved

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        self.put_writes(config, writes, task_id, task_path)

    async def adelete_thread(self, thread_id: str) -> None:
        self.delete_thread(thread_id)


def _where(config: RunnableConfig) -> tuple[str, str]:
    configurable = _configurable(config)
    return str(configurable["thread_id"]), str(configurable.get("checkpoint_ns", ""))


def _configurable(config: RunnableConfig) -> dict[str, Any]:
    return dict(config.get("configurable") or {})


def _id(found: CheckpointTuple) -> str:
    return str(_configurable(found.config)["checkpoint_id"])


def _config(thread_id: str, ns: str, checkpoint_id: str | None) -> RunnableConfig:
    configurable: dict[str, Any] = {"thread_id": thread_id, "checkpoint_ns": ns}
    if checkpoint_id is not None:
        configurable["checkpoint_id"] = checkpoint_id
    return {"configurable": configurable}


__all__ = [
    "HarnessTools",
    "ModelHooks",
    "RunCheckpointer",
    "StallGuard",
    "StepLimit",
    "model_error",
    "read_result",
]
