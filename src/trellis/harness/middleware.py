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
* :class:`ToolGuidance` — how to choose among tools added to the instructions (every part of
  the request, the specific tool over a general one, current figures from tools, a change made
  by calling its tool, conclusions from the results), the run's tools that change data named;
* :class:`AnswerCheck` — the model's draft answer checked against the request once before it
  is final (not streamed): it calls the tools still needed, or answers in its place; a run
  whose tool hints narrowed its tools is offered every one from the check on;
* :class:`StepLimit` — at ``max_steps`` model calls one more, without tools, for the best
  answer (and a ``max_steps`` warning event), instead of LangChain's canned limit message;
* :class:`StallGuard` — the same call with the same arguments in ``max_repeats`` consecutive
  steps, or :data:`ERROR_STREAK` steps in which every call failed, stop the run;
* :func:`read_result` — the tool that reads an earlier result back by its call's id: what a
  ``ContextEditingMiddleware`` placeholder points to (the state keeps the result whole);
* :class:`RunCheckpointer` — a LangGraph checkpointer that keeps only the latest checkpoint,
  in the run's journal (``thread_id`` is the run): a resume — in any worker, after a pause or
  a crash — continues where the graph stopped, without asking the model again;
* :func:`narrowing` — not given by anyone: the harness runs a ``create_agent`` graph built
  without :class:`HarnessTools` as a copy whose model calls are offered only the tools the run
  offers (the tool hints, the parts it is without), as :class:`ModelHooks` does.

The harness's middleware run inside a harness run (``trellis.current()``); outside one they
step aside (:class:`ModelHooks` runs the hooks it was given, :class:`RunCheckpointer` keeps the
checkpoint in memory). They are asynchronous: a graph is run with ``ainvoke``/``astream``.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import copy
import functools
import itertools
import json
import logging
import re
import types
import weakref
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Annotated, Any, Final, NotRequired, cast, get_args

from bifrost_sdk.headers import PROMPT_ID
from langchain.agents.factory import _chain_async_model_call_handlers
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
from langchain_core.runnables.config import var_child_runnable_config
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

from trellis.contracts import ConfigurationError, ModelError, ToolError, ToolSpec, ToolStatus
from trellis.harness import (
    accounting,  # noqa: F401 - every model call accounted
)
from trellis.harness.bounded import LRU, SAID_ONCE
from trellis.harness.hooks import Hooks, ModelCall, running
from trellis.harness.journal import content_key
from trellis.harness.prompts import Prompt, ResolvedPrompt, selected
from trellis.harness.repository import pinned
from trellis.harness.runtime import DEFERRED, Runtime, current
from trellis.harness.telemetry import LoopEvent, gateway_headers, metrics, model_span, usage
from trellis.harness.telemetry import output as span_output
from trellis.harness.tools import bridge
from trellis.harness.tools.base import SideEffects, Tool, arguments_of, not_run
from trellis.harness.tools.convert import langchain as converted
from trellis.harness.tools.convert import text_of
from trellis.harness.tools.lint import MOST_TOOLS
from trellis.harness.tools.mcp import side_effects_of as annotated

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
#: Deep Agents' tool that reads a large result saved as a file.
READ_FILE: Final = "read_file"
#: The think tool (``trellis.harness.adaptive``): a thought noted, nothing read or changed.
THINK: Final = "think"
#: Deep Agents' ``task`` tool (``adaptive.Helpers``): a helper's run, offered on a task of
#: several parts.
TASK: Final = "task"
READ_LIMIT: Final = 20_000
#: What the model is asked when its steps run out.
LAST_STEP: Final = (
    "You have used every step this run allows. Do not call tools: answer now with what you "
    "have, and say what is left undone."
)
#: The tool outcomes that count as a failed call (in a streak; a ``ToolMessage``'s status).
FAILED: Final = frozenset({ToolStatus.ERROR, ToolStatus.TIMEOUT})
#: The free variable of ``create_agent``'s model node holding its composed ``awrap_model_call``
#: handler (:func:`narrowing`).
HANDLER: Final = "awrap_model_call_handler"
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
    """One model step's calls as they run: which step (its place in the conversation), the
    number of its first call, its harness calls that do more than read in the model's order,
    the ones that finished (in this attempt, or in an earlier one: a resume runs only the calls
    that had not), and the exception one of
    them stopped on (a pause: the later ones wait for the resume)."""

    key: str
    first: int
    writes: list[str]
    finished: set[str]
    stopped: BaseException | None = None
    turn: asyncio.Condition = field(default_factory=asyncio.Condition)


#: The metadata key of a framework tool whose call, cut by a crash, continues where it was
#: when it runs again (a helper's run: ``adaptive.Helpers``), so ``GovernTools`` runs it again
#: rather than reporting its effect unknown.
RESUMABLE: Final = "trellis_resumable"
#: The tool sets whose LangChain tools a :class:`HarnessTools` keeps converted, at most.
TOOLSETS: Final = 64
#: The tools that only change the graph's own state (no effect outside the run, kept by its
#: checkpoint): a ``GovernTools`` lets them through. Deep Agents' file tools are among them
#: on its default ``StateBackend``; on a backend that writes to a real disk, give the graph
#: governed tools instead (``h.tools``).
STATE_TOOLS: Final = frozenset(
    {
        "write_todos",
        "read_result",
        "ls",
        "read_file",
        "write_file",
        "edit_file",
        "glob",
        "grep",
        THINK,
    }
)


class GovernTools(AgentMiddleware):
    """The framework's own tools of a ``create_agent`` or Deep Agents graph — those not built
    with ``h.tools`` — run as the harness's are (``h.governance_middleware()``; ``ReAct`` has
    it): governance decides each call (the catalog's risk, else the tool's declared one in its
    ``metadata["side_effects"]``, else its MCP-style annotations, else ``write``: announced),
    a person approves when it asks, and the call is journaled (a resume does not run it again),
    timed and traced. The harness's tools and the graph's state tools (:data:`STATE_TOOLS`)
    are let through as they are."""

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        runtime = current()
        call = request.tool_call
        own = request.tool
        if (
            runtime is None
            or own is None
            or call["name"] in runtime.toolbox
            or call["name"] in STATE_TOOLS
        ):
            return await handler(request)

        async def run(args: dict[str, Any]) -> Any:
            done = _answer_of(await handler(request.override(tool_call={**call, "args": args})))
            if isinstance(done, ToolMessage) and done.status == "error":
                raise ToolError(text_of(done.content), source="tools")
            return text_of(getattr(done, "content", done))

        spec = ToolSpec(
            name=call["name"],
            description=own.description or call["name"],
            input_schema=_schema_of(own),
            side_effects=side_effects_of(own.metadata),
        )
        resumable = bool((own.metadata or {}).get(RESUMABLE))
        outcome = await bridge.call(
            Tool(spec, run, resumable=resumable), dict(call["args"]), call_id=str(call["id"])
        )
        return ToolMessage(
            content=text_of(outcome.output),
            tool_call_id=str(call["id"]),
            name=call["name"],
            status="error" if outcome.status in FAILED else "success",
        )


def _answer_of(done: ToolMessage | Command[Any]) -> Any:
    """A tool's answer: its message, or the one its command adds (Deep Agents' ``task``)."""
    if not isinstance(done, Command) or not isinstance(done.update, Mapping):
        return done
    said = [m for m in done.update.get("messages") or () if isinstance(m, ToolMessage)]
    return said[-1] if said else done


def side_effects_of(metadata: Mapping[str, Any] | None) -> str:
    """A framework tool's risk as it declares it: ``side_effects`` (``read``, ``write``,
    ``irreversible``), else MCP-style annotations (``readOnlyHint``: read, ``destructiveHint``:
    irreversible), else ``write``."""
    found = dict(metadata or {})
    declared = found.get("side_effects")
    if declared in get_args(SideEffects):
        return str(declared)
    return annotated(found.get("annotations"))


def _schema_of(tool: Any) -> dict[str, Any]:
    from langchain_core.utils.function_calling import convert_to_openai_tool  # noqa: PLC0415

    return dict(convert_to_openai_tool(tool)["function"].get("parameters") or {})


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
        #: each tool set's LangChain tools, by its tools (kept: the toolbox gives a tenant's
        #: runs the same ``Tool``s, so a run converts none), the most recent :data:`TOOLSETS`
        self._toolsets: LRU[tuple[int, ...], tuple[tuple[Tool, ...], dict[str, BaseTool]]] = LRU(
            TOOLSETS
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
            harness = self._tools[runtime] = self._converted(runtime)
        own = [t for t in request.tools if _named(t) not in runtime.toolbox]
        offered = [t for name, t in harness.items() if runtime.offers(name)]
        return await handler(request.override(tools=sorted([*own, *offered], key=_named)))

    def _converted(self, runtime: Runtime) -> dict[str, BaseTool]:
        """The run's tools as LangChain's, by name: its tool set's, converted once."""
        tools = tuple(runtime.toolbox.values())
        # by the tools' identity (a ``Tool`` holds its schema, a dict: not hashable); the
        # entry keeps them, so no id is reused while it is kept
        _, made = self._toolsets.setdefault(
            tuple(map(id, tools)),
            lambda: (tools, {t.name: t for t in converted.convert(list(tools))}),
        )
        return made

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
        at, message = _calling(messages, call_id) or (
            len(messages),
            AIMessage(content="", tool_calls=[request.tool_call]),
        )
        calls = message.tool_calls
        # where the step is in the conversation, and what it is: a model may reuse call ids
        key = content_key("step", at, message.id, [c["id"] for c in calls])
        steps = self._steps.setdefault(runtime, {})
        step = steps.get(key)
        if step is None:
            writes = [str(c["id"]) for c in calls if _writes(runtime.toolbox.get(c["name"]))]
            ran = {w for w in writes if runtime.replay.journal.calls.get(_ran(key, w))}
            step = steps[key] = _Step(key, runtime.next_step(len(calls)), writes, ran)
        return step, next(n for n, c in enumerate(calls) if c["id"] == call_id)


def _calling(messages: Sequence[AnyMessage], call_id: str) -> tuple[int, AIMessage] | None:
    """The latest model message making the call ``call_id``, and where it is: read from the
    end (it is the latest step's, but for a resume's)."""
    for n in range(len(messages) - 1, -1, -1):
        message = messages[n]
        if isinstance(message, AIMessage) and any(c["id"] == call_id for c in message.tool_calls):
            return n, message
    return None


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
        runtime.replay.record_call(_ran(step.key, call_id), True)
        step.finished.add(call_id)
    finally:
        async with step.turn:
            step.turn.notify_all()


def _ran(step: str, call_id: str) -> str:
    """The journal key saying a call of a model step finished."""
    return content_key("ran", step, call_id)


def _malformed(call: Any) -> str:
    """What is wrong with the arguments of a call the chat model could not parse."""
    problem = arguments_of(call.get("args"))
    return problem if isinstance(problem, str) else str(call.get("error") or "they are invalid")


def _unbroken(request: ModelRequest[Any]) -> ModelRequest[Any]:
    """The request with the arguments of the conversation's malformed calls sent as ``{}``: a
    strict server refuses a history that holds them (the model reads what was wrong in the
    call's answer)."""
    if not any(isinstance(m, AIMessage) and m.invalid_tool_calls for m in request.messages):
        return request
    messages = [
        m.model_copy(
            update={"invalid_tool_calls": [{**c, "args": "{}"} for c in m.invalid_tool_calls]}
        )
        if isinstance(m, AIMessage) and m.invalid_tool_calls
        else m
        for m in request.messages
    ]
    return request.override(messages=messages)


def _offered(request: ModelRequest[Any], runtime: Runtime | None) -> ModelRequest[Any]:
    """The request with only the harness tools the run offers now: a graph's are bound when it
    is built (``h.tools``), so a part turned off afterwards (``h.wrap(without=)``, a run's
    ``without=``) would still be offered (a call of it is refused: it is off), and so would the
    tools the tool hints leave out (``Runtime.offers``). The graph's own tools stay."""
    if runtime is None:
        return request
    kept = [t for t in request.tools if _offers(runtime, t)]
    return request if len(kept) == len(request.tools) else request.override(tools=kept)


def _offers(runtime: Runtime, tool: BaseTool | dict[str, Any]) -> bool:
    feature = (getattr(tool, "metadata", None) or {}).get(converted.FEATURE)
    if feature is not None and not runtime.uses(feature):
        return False
    name = _named(tool)
    return name not in runtime.toolbox or runtime.offers(name)


async def _narrow(
    request: ModelRequest[Any],
    handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
) -> ModelResponse[Any]:
    """A model call offered only the tools the run offers (:func:`_offered`)."""
    return await handler(_offered(request, current()))


def narrowing(graph: Any, node: Callable[..., Any]) -> Any | None:
    """A copy of a ``create_agent`` graph (Deep Agents' too) — ``node`` is its model node's
    async function — whose every model call is offered only the tools the run offers
    (:func:`_offered`: the tool hints' narrowing, the parts the run is without), or ``None``
    when ``node`` is not of the shape this LangChain release builds.

    ``create_agent`` composes its middleware's ``awrap_model_call`` into one handler when the
    graph is built; the copy's model node composes the same handlers with :func:`_narrow`
    innermost — what the model is finally offered — as a middleware given to ``create_agent``
    would be. The graph itself is left as it is (run outside the harness, it offers every
    tool); the harness runs graphs asynchronously, so only the async model node is changed."""
    code = node.__code__
    if HANDLER not in code.co_freevars:
        _unnarrowed(node.__qualname__)
        return None
    cells = dict(zip(code.co_freevars, node.__closure__ or (), strict=True))
    given = cells[HANDLER].cell_contents
    handler = _chain_async_model_call_handlers([*([given] if given else []), _narrow])
    closure = tuple(
        types.CellType(handler) if name == HANDLER else cell for name, cell in cells.items()
    )
    narrowed = types.FunctionType(code, node.__globals__, node.__name__, node.__defaults__, closure)
    narrowed.__kwdefaults__ = node.__kwdefaults__
    narrowed.__qualname__ = node.__qualname__
    model = graph.nodes["model"]
    runnable = copy.copy(model.bound)
    runnable.afunc = narrowed
    return graph.copy({"nodes": {**graph.nodes, "model": model.copy({"bound": runnable})}})


@functools.lru_cache(maxsize=SAID_ONCE)  # said once per node, of the most recent
def _unnarrowed(node: str) -> None:
    """Said when a graph's model node, named as ``create_agent``'s, is not of the shape this
    harness reads (a LangChain release it was not tested with): its tools are not narrowed."""
    log.warning(
        "the graph's model node %s is not of the shape the harness reads (no %s): the run's "
        "tool narrowing is off for it; use the LangChain release trellis-harness pins",
        node,
        HANDLER,
    )


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
    conversation. Without ``prompt``, a chat model whose own headers select a stored prompt
    (``ChatOpenAI(default_headers=await h.model_headers(prompt=...))``) selects, on every call,
    the version the run pinned."""

    def __init__(
        self,
        *hooks: Hooks,
        timeout: float | None = None,
        prompt: str | Prompt | None = None,
        prompt_vars: dict[str, Any] | None = None,
        gateway: bool = False,
    ) -> None:
        super().__init__()
        #: the model is the gateway's: every call carries the run's Bifrost dimensions and
        #: its ``traceparent`` (``telemetry.gateway_headers``)
        self.gateway = gateway
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

    @hook_config(can_jump_to=["model"])
    async def aafter_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        """A call whose arguments the chat model could not parse (``invalid_tool_calls``: not
        JSON, not an object) is answered with what was wrong, as a call of the harness's is;
        a step of such calls only asks the model again (LangChain would end the run)."""
        last = state["messages"][-1] if state.get("messages") else None
        if not isinstance(last, AIMessage) or not last.invalid_tool_calls:
            return None
        for _ in last.invalid_tool_calls:
            metrics.loop(LoopEvent.MALFORMED_ARGS)
        told = [
            ToolMessage(
                not_run(c.get("name") or "unknown", _malformed(c)),
                name=c.get("name") or "unknown",
                tool_call_id=c["id"],
                status="error",
            )
            for c in last.invalid_tool_calls
            if c.get("id")
        ]
        return {"messages": told} | ({} if last.tool_calls else {"jump_to": "model"})

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        runtime = current()
        hooks = running(*self.given)
        request = _with_task(_unbroken(_offered(request, runtime)))
        prompt = await self._pinned(runtime) or _selected(request, runtime)
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
        with model_span(name or "model", lambda: _said(request), extra=extra) as span:
            if self.gateway:  # in the model's span: the gateway's span of the call joins it
                sent = request.model_settings.get("extra_headers") or {}
                headers = {**sent, **gateway_headers()}
                request = request.override(
                    model_settings={**request.model_settings, "extra_headers": headers}
                )
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
        found = await runtime.agent.harness.prompts.pin(self.prompt, runtime)
        if found.pin() is not None and self.prompt_vars:
            raise ConfigurationError(
                f"prompt= names {found.name}, a stored prompt of the gateway: the gateway "
                "prepends it as it is stored (no prompt_vars=)"
            )
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


def _selected(request: ModelRequest[Any], runtime: Runtime | None) -> ResolvedPrompt | None:
    """The stored prompt the run pinned that the chat model's own headers select (a
    ``ChatOpenAI(default_headers=await h.model_headers(prompt=...))``, which copied them when it
    was built): each call then selects the version the run pinned."""
    headers = getattr(request.model, "default_headers", None)
    if runtime is None or not isinstance(headers, Mapping):
        return None
    return selected(runtime, headers.get(PROMPT_ID))


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
    counts: dict[str, Any] = dict(message.usage_metadata or {})
    details = counts.get("input_token_details") or {}
    reason = metadata.get("finish_reason")
    usage(
        span,
        model=metadata.get("model_name"),
        input_tokens=counts.get("input_tokens"),
        output_tokens=counts.get("output_tokens"),
        finish_reasons=[reason] if isinstance(reason, str) else [],
        cached_tokens=details.get("cache_read"),
        written_tokens=details.get("cache_creation"),
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
    consecutive steps in which every call failed (a malformed one among them)."""

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
        history = request.state.get("messages", [])
        failing = _failing(history)
        if failing >= ERROR_STREAK:
            raise ModelError(
                f"stopped: every tool call failed in {failing} consecutive steps",
                source="langgraph",
            )
        reply = await handler(request)
        message = _ai(reply)
        if message is not None and message.tool_calls:
            # a stall needs only the latest steps: no more than max_repeats - 1 are looked at
            steps = list(itertools.islice(_steps(history), self.max_repeats - 1))
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


def _turn_back(history: Sequence[AnyMessage]) -> Iterator[AnyMessage]:
    """This invocation's messages (since the last user message), the latest first — read from
    the end, as far as the caller reads."""
    for n in range(len(history) - 1, -1, -1):
        message = history[n]
        if isinstance(message, HumanMessage):
            return
        yield message


def _steps(history: Sequence[AnyMessage]) -> Iterator[set[str]]:
    """The calls of each step of this invocation, the latest first."""
    for message in _turn_back(history):
        if isinstance(message, AIMessage):
            yield {_call_key(c) for c in message.tool_calls}


def _failing(history: Sequence[AnyMessage]) -> int:
    """How many of the latest steps had every call fail: read from the end, a step's results
    (after it) before the step, as far as the first step with a call that did not fail."""
    statuses: dict[str, str] = {}
    streak = 0
    for message in _turn_back(history):
        if isinstance(message, ToolMessage):
            statuses[message.tool_call_id] = message.status
            continue
        if not isinstance(message, AIMessage):
            continue
        made = [statuses.get(c["id"] or "") for c in message.tool_calls]
        if not made or any(s != "error" for s in made):
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
                for m in reversed(state.get("messages", []))
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


class ReadTools(AgentMiddleware):
    """``read_file`` offered once the graph's state has a file to read (a large result saved, a
    summarized history: Deep Agents' ``StateBackend``), ``read_result`` once a result reads as
    ``placeholder`` (cleared): a small model offered them from the start reads nothing, again
    and again. Both stay callable."""

    def __init__(self, placeholder: str) -> None:
        super().__init__()
        self.placeholder = placeholder

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        unused = set()
        if not request.state.get("files"):
            unused.add("read_file")
        if not any(
            isinstance(m, ToolMessage) and m.content == self.placeholder for m in request.messages
        ):
            unused.add(READ_RESULT)
        kept = [t for t in request.tools if _named(t) not in unused]
        return await handler(request.override(tools=kept))


# --------------------------------------------------------------------------- tool choice
#: What :class:`ToolGuidance` adds to the instructions: how to choose among tools, the same for
#: any tool set (the tools' own descriptions say what each is for).
TOOL_GUIDANCE: Final = (
    "How to use your tools:\n"
    "- A request may have several parts (questions, actions). Address every part, each with "
    "the tool made for it.\n"
    "- When several tools could answer a part, use the most specific one whose description "
    "matches it; use a general or overview tool only when no specific tool fits.\n"
    "- Current figures come from a tool call, even when the context remembers older values; "
    "the context gives background, preferences and rules, and its rules still apply.\n"
    "- To create, change or delete something, call the tool that does it: the system asks for "
    "any approval itself. Never say an action was done unless its tool returned success.\n"
    "- Before calling a tool that changes data, check the call against the rules in your "
    "instructions and context: if one forbids it, do not call it; say which rule and why.\n"
    "- When the tools and the context have nothing about what was asked (an empty result, "
    "not found), say plainly that there is no record of it: no hedging, no estimates, no "
    "figures that are not in the results.\n"
    "- Base each conclusion on the tool results. If no tool can answer a part, say so.\n"
    "- Asked whether you can do something, answer plainly yes or no, from your tools and the "
    "capabilities you are told of; do not hedge."
)
#: What the model is told about the tools that change data (:class:`ToolGuidance`,
#: :class:`AnswerCheck`): their names.
WRITE_TOOLS: Final = "These tools change data: {tools}."
#: What the model is told of what the assistant can and cannot do (``ToolGuidance``'s
#: ``capabilities``), before one line per capability.
CAPABILITIES: Final = (
    "What you can and cannot do here (asked about one of these, answer plainly yes or no):"
)
#: What :class:`AnswerCheck` asks the model, after its draft answer.
ANSWER_CHECK: Final = (
    "Before this answer goes to the user, check it against the user's request. Do not mention "
    "this check.\n"
    "1. List the parts of the request: each question asked and each action requested.\n"
    "2. For each part, find what answers it: a tool result in this conversation, the context "
    "you were given, or a plain statement that it cannot be done. A part with none of these "
    "needs a tool call: make it now, with the most specific tool for it.\n"
    "3. A part that asks to create, change or delete something needs the call of the tool "
    "that does it{writes}; the system asks for approval itself. Do not say it was done unless "
    "that tool returned success. If a rule in your instructions or context forbids the "
    "change, do not call the tool: say which rule forbids it.\n"
    "4. Make sure each conclusion follows from the tool results.\n"
    "5. If there is no record of what was asked, say so plainly: no hedging, no estimates.\n"
    "If a part is still open, call the tools it needs now. Otherwise reply with the final "
    "answer for the user, and only that."
)
#: What :class:`AnswerCheck` adds when nothing found is about what was asked: every tool result
#: of the request empty or not found, or no tool called and memory holding nothing.
NO_RECORD: Final = (
    "\nNothing {source} is about what was asked. Unless a tool you have not called yet "
    "would find it, say plainly that there is no record of it, in those words: no hedging, no "
    "estimates, no figures."
)
#: What :class:`AnswerCheck` asks before the first change of an invocation is made (and before
#: anyone is asked to approve it).
WRITE_CHECK: Final = (
    "You were about to call {calls}, which changes data. Before it is made, and before anyone "
    "is asked to approve it, check it against every rule in your instructions and in the "
    "context you were given (a standing rule, a limit, a preference). If a rule forbids it, do "
    "not call it: tell the user plainly which rule forbids it and why. Otherwise make the same "
    "call again. Do not mention this check."
)
#: A tool result that holds nothing: empty, or saying it found nothing.
_NOTHING: Final = frozenset({"", "[]", "{}", "null", "none", '""', "''"})
_NOT_FOUND: Final = re.compile(
    r"\bnot found\b|\bno (?:such|records?|results?|data|matches?)\b|\bdoes not exist\b|\b404\b",
    re.IGNORECASE,
)
#: The state keys of :class:`AnswerCheck`: the checks made in this invocation, and whether its
#: first change was checked.
CHECKED: Final = "trellis_checked"
WRITE_CHECKED: Final = "trellis_write_checked"
#: The tag that keeps a model call's tokens off the graph's ``messages`` stream (LangGraph's
#: ``TAG_NOSTREAM``): a draft answer the user is not shown.
NO_STREAM: Final = "nostream"


class ToolGuidance(AgentMiddleware):
    """How to choose among tools, added to the instructions of every model call that offers
    tools (:data:`TOOL_GUIDANCE`): every part of a request addressed, the most specific tool
    over a general one, current figures from tools rather than the remembered context (whose
    rules still apply), an action done by calling its tool (never claimed without it, never
    against a rule), "no record" said plainly, conclusions from the results, a plain yes or no
    to what it can do. Inside a run it also names the offered harness tools that change data
    (their declared side effects). ``text`` replaces the guidance (``None``: none, the
    capabilities only). ``capabilities`` says what
    the assistant can and cannot do beyond its tools (``{"accept an image upload": False}``),
    one line each, on every call (:data:`CAPABILITIES`). No extra model call."""

    def __init__(
        self,
        text: str | None = TOOL_GUIDANCE,
        *,
        capabilities: Mapping[str, bool] | None = None,
    ) -> None:
        super().__init__()
        if text is not None and not text.strip():
            raise ConfigurationError("ToolGuidance needs a text (None: the capabilities only)")
        self.text = text
        self.capabilities = dict(capabilities or {})

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        parts = []
        if request.tools and self.text is not None:
            writes = _writes_offered(request)
            parts.append(self.text)
            if writes:
                parts.append(WRITE_TOOLS.format(tools=", ".join(writes)))
        if self.capabilities:
            lines = [
                f"- {name}: {'yes' if can else 'no'}" for name, can in self.capabilities.items()
            ]
            parts.append("\n".join([CAPABILITIES, *lines]))
        if not parts:
            return await handler(request)
        return await handler(request.override(system_message=_appended(request, "\n".join(parts))))


class _CheckedState(AgentState[Any]):
    trellis_checked: NotRequired[Annotated[int, PrivateStateAttr]]
    trellis_write_checked: NotRequired[Annotated[bool, PrivateStateAttr]]


class AnswerCheck(AgentMiddleware):
    """An answer is checked before it is final, and a change before it is made.

    **The answer**: when the model answers without calling a tool (its draft, not streamed), it
    is asked once more, with the draft, to check it against the request (:data:`ANSWER_CHECK`)
    — every part answered by a tool result, the context or a plain "cannot"; a requested change
    made by calling its tool (inside a run, the tools that change data are named), never
    claimed without it, never against a rule; the conclusions following from the results;
    "no record" said plainly — and either calls the tools still needed (the loop goes on) or
    gives the final answer, which replaces the draft. When nothing found is about what was
    asked — every tool result of the request empty or not found, or no tool called and memory
    holding nothing (``INSUFFICIENT``) — it is told so (:data:`NO_RECORD`). At most ``checks``
    checks per invocation of the graph: one more model call each.

    **A change** (``writes``, on by default): the first step of an invocation that calls a
    harness tool changing data is drafted (not streamed) and the model asked to check the
    call against the rules in its instructions and context (:data:`WRITE_CHECK`) before it
    runs — before governance asks anyone to approve it: it makes the call again, or tells the
    user which rule forbids it. One more model call, once per invocation.

    Neither on a call offered no tools, or told to call none (behind :class:`StepLimit`, as in
    ``ReAct``, its last call is checked with no tool offered: only the text can be corrected).
    Inside a run whose tool hints narrowed what the model is offered, the answer check is
    offered every tool (only those that change data past :data:`~.lint.MOST_TOOLS`): a part
    the hints did not foresee can still be served. For that it goes before
    :class:`HarnessTools` (as in ``ReAct``), which then offers them."""

    state_schema = _CheckedState

    def __init__(self, checks: int = 1, *, writes: bool = True, text: str = ANSWER_CHECK) -> None:
        super().__init__()
        if checks < 1:
            raise ConfigurationError("AnswerCheck makes at least 1 check (or leave it out)")
        if "{writes}" not in text:
            raise ConfigurationError("an AnswerCheck text has a {writes} placeholder")
        self.checks = checks
        self.writes = writes
        self.text = text

    async def abefore_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        return {CHECKED: 0, WRITE_CHECKED: False}

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any] | ExtendedModelResponse[Any]:
        runtime = current()
        done = request.state.get(CHECKED, 0)
        write_checked = not self.writes or request.state.get(WRITE_CHECKED, False)
        if (
            (done >= self.checks and write_checked)
            or not _tools(request, runtime)
            or request.tool_choice == "none"
        ):
            return await handler(request)
        with _quiet():
            draft = await handler(request)
        said = _drafted(draft)
        if said is None:  # a step that calls tools: no answer yet
            calls = [] if write_checked else _write_calls(draft, runtime)
            if not calls:
                return draft
            told = [
                *request.messages,
                AIMessage(content=f"(about to call {'; '.join(calls)})"),
                HumanMessage(content=WRITE_CHECK.format(calls="; ".join(calls))),
            ]
            checked = await handler(request.override(messages=told))
            return ExtendedModelResponse(
                model_response=draft if _drafted(checked) == "" else checked,
                command=Command(update={WRITE_CHECKED: True}),
            )
        if done >= self.checks:
            return draft
        writes = _changing(runtime)
        if runtime is not None:
            _widened(runtime, writes)
        text = self.text.format(writes=f" ({', '.join(writes)})" if writes else "")
        told = [
            *request.messages,
            AIMessage(content=said),
            HumanMessage(content=text + _no_record(request.messages, runtime)),
        ]
        checked = await handler(request.override(messages=told))
        if _drafted(checked) == "":  # nothing said, nothing called: the draft stands
            checked = draft
        return ExtendedModelResponse(
            model_response=checked, command=Command(update={CHECKED: done + 1})
        )


def _write_calls(reply: ModelResponse[Any], runtime: Runtime | None) -> list[str]:
    """The calls of a reply whose tool changes data, as the model is shown them."""
    message = _ai(reply)
    changing = set(_changing(runtime))
    return [
        f"{c['name']}({json.dumps(c['args'], default=str)})"
        for c in (message.tool_calls if message is not None else [])
        if c["name"] in changing
    ]


def _no_record(messages: Sequence[BaseMessage], runtime: Runtime | None) -> str:
    """:data:`NO_RECORD` when nothing found is about what was asked: every tool result since
    the request empty or not found, or none and memory holding nothing; else nothing."""
    turn: list[BaseMessage] = []
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            break
        turn.append(message)
    results = [m for m in turn if isinstance(m, ToolMessage)]
    insufficient = runtime is not None and runtime.evidence == "INSUFFICIENT"
    if results and all(_empty(m) for m in results):
        source = "in memory or the tool results" if insufficient else "the tools returned"
        return NO_RECORD.format(source=source)
    if not results and insufficient:
        return NO_RECORD.format(source="in memory")
    return ""


def _empty(message: ToolMessage) -> bool:
    """A tool result that holds nothing: empty, or saying nothing was found."""
    text = message.text.strip()
    return text.casefold() in _NOTHING or bool(_NOT_FOUND.search(text))


def _tools(request: ModelRequest[Any], runtime: Runtime | None) -> bool:
    """Whether the model has tools to call beyond the readers of earlier results, the think
    tool and the helpers' ``task`` (each offered on its own signal): the run's own, or the
    graph's."""
    own = [t for t in request.tools if _named(t) not in (READ_RESULT, READ_FILE, THINK, TASK)]
    return bool(own) or bool(runtime is not None and runtime.tool_names())


def _changing(runtime: Runtime | None) -> list[str]:
    """The run's own tools that change data (their side effects, as declared), sorted; none
    outside a run (a graph's own tools say nothing of theirs)."""
    if runtime is None:
        return []
    return sorted(
        name for name in runtime.tool_names() if runtime.toolbox[name].spec.side_effects != "read"
    )


def _widened(runtime: Runtime, writes: list[str]) -> None:
    """A run whose tool hints narrowed its tools offers every one of them from now on (only
    those that change data, past :data:`~.lint.MOST_TOOLS`)."""
    if runtime.offered is None:
        return
    names = runtime.tool_names()
    runtime.offer(names if len(names) <= MOST_TOOLS else writes)


@contextlib.contextmanager
def _quiet() -> Iterator[None]:
    """The model calls made inside, tagged :data:`NO_STREAM`: their tokens are not streamed
    (a reply the graph keeps still reaches the stream whole, as its node's output)."""
    config = var_child_runnable_config.get() or {}
    tags = [*(config.get("tags") or []), NO_STREAM]
    token = var_child_runnable_config.set(cast(RunnableConfig, {**config, "tags": tags}))
    try:
        yield
    finally:
        var_child_runnable_config.reset(token)


def _drafted(reply: ModelResponse[Any]) -> str | None:
    """What a model reply answers — its structured answer as JSON, else its text — or ``None``
    when it calls tools (not an answer yet)."""
    if reply.structured_response is not None:
        return _json(reply.structured_response)
    message = _ai(reply)
    if message is None or message.tool_calls or message.invalid_tool_calls:
        return None
    return message.text


def _json(value: Any) -> str:
    dumped = value.model_dump(mode="json") if hasattr(value, "model_dump") else value
    return json.dumps(dumped, default=str)


def _writes_offered(request: ModelRequest[Any]) -> list[str]:
    """Of the run's tools that change data (:func:`_changing`), those the call offers."""
    offered = {_named(t) for t in request.tools}
    return [name for name in _changing(current()) if name in offered]


def _appended(request: ModelRequest[Any], text: str) -> SystemMessage:
    """The request's instructions with ``text`` after them (a system message of content
    blocks gets one more block)."""
    system = request.system_message
    if system is None:
        return SystemMessage(content=text)
    if isinstance(system.content, str):
        joined = f"{system.content}\n\n{text}" if system.content else text
        return SystemMessage(content=joined)
    return SystemMessage(content=[*system.content, {"type": "text", "text": text}])


# --------------------------------------------------------------------------- checkpoints
#: Outside a run, how many threads :class:`RunCheckpointer` keeps (the most recently used).
THREADS_KEPT: Final = 10_000


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
        #: outside a run: the threads' latest checkpoints, the least recently used forgotten
        #: past the bound (an in-memory checkpointer; inside a run, the run's journal)
        self._threads: LRU[str, dict[str, Any]] = LRU(THREADS_KEPT)

    # ------------------------------------------------------------------ storage
    def _slot(self, thread_id: str) -> dict[str, Any]:
        """Where the thread's checkpoints are: the run's journal, for the run's own thread."""
        runtime = current()
        if runtime is not None and runtime.run_id == thread_id:
            journal = runtime.replay.journal
            if journal.graph is None:
                journal.graph = {}
            return journal.graph
        return self._threads.setdefault(thread_id, dict)

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
        # by (checkpoint, task, index), in the order kept: a write of a key it holds is the
        # same write (kept), unless a special one (a negative index), which replaces it
        found = {(w[0], w[1], w[2]): w for w in kept}
        for n, (channel, value) in enumerate(writes):
            index = WRITES_IDX_MAP.get(channel, n)
            key = (checkpoint_id, task_id, index)
            if key in found and index >= 0:
                continue
            found.pop(key, None)
            found[key] = [checkpoint_id, task_id, index, channel, self._dumped(value), task_path]
        kept[:] = found.values()

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
    "AnswerCheck",
    "HarnessTools",
    "ModelHooks",
    "ReadTools",
    "RunCheckpointer",
    "StallGuard",
    "StepLimit",
    "ToolGuidance",
    "model_error",
    "read_result",
]
