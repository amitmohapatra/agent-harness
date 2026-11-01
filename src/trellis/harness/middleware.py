"""The harness as LangChain middleware: what a ``create_agent`` graph (Deep Agents' too) gets
from the harness, added where LangChain and Deep Agents have no native piece. ``ReAct(...)``
is a ``create_agent`` graph built with all of them, beside the native middleware it turns on;
any ``create_agent`` or Deep Agents graph may take any of them:

    graph = create_agent(model, tools=[refund], middleware=[ModelHooks()],
                         checkpointer=RunCheckpointer())
    agent = h.wrap(graph, id="support")

(A graph built with ``middleware()`` — :class:`HarnessTools` among it — and no checkpointer is
run with a :class:`RunCheckpointer` without being given one: ``adapters.langgraph.checkpointing``.)

* :class:`HarnessTools` — the run's harness tools offered per model call (the tool hints'
  narrowing, the memory tools, what ``tool_search`` found), sorted by name so the cached prompt
  holds; every call goes through the bridge (``tools.bridge``: governance, journal, record),
  numbered in the model's order, and a call that does more than read waits for the earlier
  ones of the same model step (the model's order); a call repeating an earlier one of its step
  (the same tool, the same arguments) runs once (``bridge.same``) — ``ReAct``'s ``tools=``;
* :class:`ModelHooks` — around every model call: the run's ``on_error`` hooks for a call that
  failed, a ``chat`` span with the usage (redacted, as every span), the most the
  call may take (``timeout``, and what is left of the run's time), the gateway's stored prompt
  pinned for the run (``prompt``: journaled, sent as headers, on the span), and the model
  client's errors as a :class:`~trellis.contracts.ModelError` saying whether it may pass;
* :func:`step_limit` — ``max_steps`` model calls: LangChain's own ``ModelCallLimitMiddleware``
  (:class:`AnswerAtLimit`), then one more that answers, offered no tool;
* :class:`StallGuard` — the same call with the same arguments in ``max_repeats`` consecutive
  steps, or :data:`ERROR_STREAK` steps in which every call failed and a failure repeated, stop
  the run (a repeat within one step runs once: :class:`HarnessTools`, :func:`governed_call`);
* :class:`ReadTools` — ``read_file`` offered once there is something to read back: a large
  result kept, a file in the graph's state, or a result LangChain's ``ClearToolUsesEdit``
  cleared (paged from the graph's state by its call's id, which keeps it whole);
* :class:`RunCheckpointer` — a LangGraph checkpointer that keeps only the latest checkpoint,
  in the run's journal (``thread_id`` is the run): a resume — in any worker, after a pause or
  a crash — continues where the graph stopped, without asking the model again;
* :func:`narrowing` — not given by anyone: the harness runs a ``create_agent`` graph built
  without :class:`HarnessTools` as a copy whose model calls are offered only the tools the run
  offers (the tool hints, the parts it is without), as :class:`ModelHooks` does.

The harness's middleware run inside a harness run (``trellis.current()``); outside one they
step aside (:class:`RunCheckpointer` keeps the checkpoint in memory). They are asynchronous: a
graph is run with ``ainvoke``/``astream``.
"""

from __future__ import annotations

import asyncio
import base64
import collections
import contextlib
import copy
import functools
import itertools
import logging
import re
import types
import weakref
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Collection,
    Iterator,
    Mapping,
    Sequence,
)
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from typing import Annotated, Any, Final, NotRequired, cast, get_args

from bifrost_sdk import RETRYABLE
from langchain.agents.factory import _chain_async_model_call_handlers
from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    ClearToolUsesEdit,
    ModelCallLimitMiddleware,
    ModelRequest,
    ModelResponse,
)
from langchain.agents.middleware.context_editing import TokenCounter
from langchain.agents.middleware.types import (
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
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    DeltaChannelHistory,
    get_checkpoint_id,
    get_checkpoint_metadata,
)
from langgraph.checkpoint.serde.types import _DeltaSnapshot
from langgraph.prebuilt.tool_node import ToolCallRequest, ToolNode
from langgraph.types import Command

from trellis.contracts import (
    ConfigurationError,
    HarnessError,
    ModelError,
    ToolError,
    ToolOutcome,
    ToolSpec,
    ToolStatus,
)
from trellis.contracts.errors import is_pause_signal
from trellis.harness import (
    accounting,  # noqa: F401 - every model call accounted
    budget,
)
from trellis.harness.adapters.base import FIT_SHARE
from trellis.harness.adapters.langgraph import CONTEXT_MESSAGE_ID, TASK_TOOL
from trellis.harness.bounded import LRU, SAID_ONCE, LRUSet
from trellis.harness.bridges.helpers import Kind
from trellis.harness.conversation import question_id
from trellis.harness.gateways import Capability, route_now, routed
from trellis.harness.hooks import running
from trellis.harness.journal import content_key
from trellis.harness.prompts import ResolvedPrompt, Stored, message_text
from trellis.harness.removed import removed
from trellis.harness.runtime import DEFERRED, ModelTimeout, Runtime, current
from trellis.harness.telemetry import gateway_headers, model_span, usage
from trellis.harness.telemetry import output as span_output
from trellis.harness.tools import bridge, query, results
from trellis.harness.tools.base import (
    SideEffects,
    Sourced,
    Tool,
    arguments_of,
    citations,
    data_of,
    not_run,
)
from trellis.harness.tools.convert import langchain as converted
from trellis.harness.tools.convert import text_of
from trellis.harness.tools.mcp import side_effects_of as annotated

log = logging.getLogger("trellis.run")

#: Model calls one run may make (:func:`step_limit`: LangChain's ``ModelCallLimitMiddleware``).
MAX_STEPS: Final = 12
#: Consecutive steps making the same call with the same arguments that stop the run.
MAX_REPEATS: Final = 3
#: Consecutive steps in which every call failed (no such tool, arguments that do not fit, an
#: error or a timeout), one failure among them met again, that stop the run.
ERROR_STREAK: Final = 3
#: Deep Agents' tool that reads a large result saved as a file — and, by the path
#: ``/large_tool_results/<call id>`` (Deep Agents' own), a result cleared from the request.
READ_FILE: Final = "read_file"
#: What a model whose every call failed in a row is told, for its one answer without tools.
FAILING: Final = (
    "Every tool call failed in the last {steps} steps. Do not call tools: answer now with what "
    "you have, and say what failed."
)
#: The tool outcomes that count as a failed call (in a streak; a ``ToolMessage``'s status).
FAILED: Final = frozenset({ToolStatus.ERROR, ToolStatus.TIMEOUT})
#: The free variable of ``create_agent``'s model node holding its composed ``awrap_model_call``
#: handler (:func:`narrowing`).
HANDLER: Final = "awrap_model_call_handler"


# --------------------------------------------------------------------------- tools
class _HarnessToolsState(AgentState[Any]):
    # the key is ``adapters.langgraph.HARNESS_TOOLS``: the harness reads it on the graph
    trellis_harness_tools: NotRequired[Annotated[bool, PrivateStateAttr]]


@dataclass(eq=False)
class _Step:
    """One model step's calls as they run: which step (its place in the conversation), the
    number of its first call, its harness calls that do more than read in the model's order
    (a repeat left out: it does not run), the ones that finished (in this attempt, or in an
    earlier one: a resume runs only the calls that had not), the exception one of them stopped
    on (a pause: the later ones wait for the resume), and the calls repeating an earlier one of
    the step (``bridge.firsts``)."""

    key: str
    first: int
    writes: list[str]
    finished: set[str]
    stopped: BaseException | None = None
    turn: asyncio.Condition = field(default_factory=asyncio.Condition)
    repeats: dict[str, str] = field(default_factory=dict)


#: The tool sets whose LangChain tools a :class:`HarnessTools` keeps converted, at most.
TOOLSETS: Final = 64
#: The tools that only change the graph's own state (no effect outside the run, kept by its
#: checkpoint): a ``GovernTools`` lets them through. Deep Agents' file tools are among them
#: on its default ``StateBackend``; on a backend that writes to a real disk, its file tools are
#: governed as any other (give them their risk in the catalog).
STATE_TOOLS: Final = frozenset(
    {
        "write_todos",
        "ls",
        "read_file",
        "write_file",
        "edit_file",
        "delete",
        "glob",
        "grep",
    }
)
#: The state tools that write a file, and what their result then says of it: the run's own
#: scratch copy, which the user cannot reach (asked for a chart, a model saved an SVG in the
#: graph's state and told the user to open it).
SCRATCH_WRITES: Final = frozenset({"write_file", "edit_file"})
#: The runs whose model was told :data:`SCRATCH_NOTE` (once a run, however many it writes).
_SCRATCH_SAID: LRUSet[str] = LRUSet(maxsize=SAID_ONCE)
SCRATCH_NOTE: Final = (
    "This file is in this run's own scratch space: the user cannot open or download it, so "
    "never offer it to them as a file; give them its content instead."
)
#: The metadata key marking a file tool whose backend is outside the graph's state (a disk, a
#: store, a sandbox: ``ReAct(backend=)``): governed as any other tool, its risk as declared.
OUTSIDE_STATE: Final = "trellis_outside_state"
#: The metadata key marking a tool of the graph's own machinery, not of the agent's tool set
#: (``ReAct``'s: Deep Agents' ``execute`` on a backend that runs nothing, its general-purpose
#: ``task`` helper, offered on a long task only): it is left out of the run's tool set.
MACHINERY: Final = "trellis_machinery"


class GovernTools(AgentMiddleware):
    """The framework's own tools of a ``create_agent`` or Deep Agents graph run as the
    harness's are (:func:`~trellis.harness.react.middleware` and ``ReAct`` have it):
    :func:`governed_call`.
    Inside a harness run every tool node is governed so already (:func:`govern_tool_nodes`);
    a call is governed once, whichever comes first."""

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        return await governed_call(request, handler)


#: The call a governed run of a framework's own tool is executing (:func:`governed_call`): the
#: layers under it (a tool node inside a ``GovernTools``) let it through, governed once.
#: By the tool's name and the call's id: a call id is unique in one conversation only, and a
#: sub-agent's calls are another's.
_GOVERNING: ContextVar[tuple[str, str] | None] = ContextVar("trellis_governing", default=None)


def _scratch_said(
    runtime: Runtime, tool: str, done: ToolMessage | Command[Any]
) -> ToolMessage | Command[Any]:
    """A state tool's result: the run's first scratch write that worked says
    :data:`SCRATCH_NOTE` (once a run: its history keeps saying it)."""
    if tool not in SCRATCH_WRITES or runtime.run_id in _SCRATCH_SAID:
        return done
    noted = _scratch_noted(done)
    if noted is not done:
        _SCRATCH_SAID.add(runtime.run_id)
    return noted


def _scratch_noted(done: ToolMessage | Command[Any]) -> ToolMessage | Command[Any]:
    """A scratch write's result with :data:`SCRATCH_NOTE` after it: on its message, or on the
    message of the ``Command`` that also updates the graph's files (Deep Agents' state
    backend). A failed write is left as it is."""

    def noted(m: Any) -> Any:
        if not isinstance(m, ToolMessage) or m.status == "error" or not isinstance(m.content, str):
            return m
        return m.model_copy(update={"content": f"{m.content}\n{SCRATCH_NOTE}"})

    if isinstance(done, ToolMessage):
        return noted(done)
    update = done.update
    if not isinstance(update, dict) or not isinstance(update.get("messages"), list):
        return done
    said = [noted(m) for m in update["messages"]]
    if all(a is b for a, b in zip(said, update["messages"], strict=True)):
        return done
    return Command(
        graph=done.graph, update={**update, "messages": said}, resume=done.resume, goto=done.goto
    )


async def governed_call(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
) -> ToolMessage | Command[Any]:
    """A framework's own tool call, inside a harness run, run as the harness's are: governance
    decides it (the catalog's risk, else the tool's declared one in its
    ``metadata["side_effects"]``, else its MCP-style annotations, else ``write``: announced), a
    person approves when it asks, and it is journaled (a resume does not run it again), timed,
    traced and recorded (``tools.bridge``). What the tool returned is returned as it is — a
    ``Command`` (a Deep Agent's ``task`` updating the graph's state) included — and, replayed
    from the journal, as the message the model read. The harness's tools, the graph's state
    tools (:data:`STATE_TOOLS`), a call already governed and any call outside a run are let
    through as they are — a file tool of a backend outside the graph's state
    (:data:`OUTSIDE_STATE`) is governed. A call repeating an earlier one of its model step
    (the same tool, the same arguments) does not run — the harness's tools' included — and
    points the model to the earlier one's result (``bridge.same``)."""
    runtime = current()
    call = request.tool_call
    own = request.tool
    if runtime is None or own is None or _GOVERNING.get() == (call["name"], str(call["id"])):
        return await handler(request)
    if call["name"] in STATE_TOOLS and not (own.metadata or {}).get(OUTSIDE_STATE):
        return _scratch_said(runtime, call["name"], await handler(request))
    _, step = _step_of(request)
    first = _repeats(step).get(str(call["id"]))
    if first is not None:
        repeat = bridge.same(
            runtime, call["name"], call["args"], call_id=str(call["id"]), first=first
        )
        return _message(repeat, str(call["id"]), call["name"], kept=_paged(runtime, request))
    if converted.made(own) or call["name"] in runtime.toolbox:
        return await handler(request)  # the harness's own: its call goes through the bridge
    produced: list[ToolMessage | Command[Any]] = []

    async def run(args: dict[str, Any]) -> Any:
        token = _GOVERNING.set((call["name"], str(call["id"])))
        try:
            done = await _helped(runtime, call["name"], str(call["id"]), args, ran)
        finally:
            _GOVERNING.reset(token)
        if isinstance(done, ToolMessage) and done.status == "error":
            raise ToolError(text_of(done.content), source="tools")
        produced.append(done)
        # a content_and_artifact tool's artifact: the sources of its result, when it names any
        cited = citations(getattr(done, "artifact", None))
        return Sourced(text_of(_told(done)), cited) if cited else text_of(_told(done))

    async def ran(args: dict[str, Any]) -> ToolMessage | Command[Any]:
        return await handler(request.override(tool_call={**call, "args": args}))

    spec = runtime.native.get(call["name"]) or spec_of(own)  # as the run's tool set has it
    outcome = await bridge.call(Tool(spec, run), dict(call["args"]), call_id=str(call["id"]))
    if produced and outcome.ok and isinstance(produced[-1], Command):
        return produced[-1]  # the graph's state update, as the tool made it
    return _message(outcome, str(call["id"]), call["name"], kept=_paged(runtime, request))


def _message(outcome: ToolOutcome, call_id: str, name: str, *, kept: bool = False) -> ToolMessage:
    """A harness call's outcome as the tool message the model reads — ``kept``: a large one
    kept in the run and previewed, ``read_file`` paging it (``tools.results``); a table of
    many rows read whole is kept too, for ``query_result`` (``results.tabled``)."""
    return ToolMessage(
        content=(
            results.readable(outcome.output, name)
            if kept
            else results.tabled(text_of(outcome.output), name)
        ),
        tool_call_id=call_id,
        name=name,
        status="error" if outcome.status in FAILED else "success",
    )


def _paged(runtime: Runtime, request: ToolCallRequest) -> bool:
    """Whether a large result of this call is kept in the run and previewed, ``read_file``
    paging it (``tools.results``, the convention every adapter's model reads): in a graph that
    is offered the harness's tools per model call (:class:`HarnessTools`, which offers and
    serves that ``read_file``) and has no ``read_file`` of its own — Deep Agents' saves a large
    result on its backend itself (a sandbox's among them, where code reads it)."""
    if runtime not in _PAGED:
        return False
    tools = getattr(request.runtime, "tools", None) or ()
    return not any(getattr(t, "name", None) == READ_FILE for t in tools)


async def _kept_read(runtime: Runtime, request: ToolCallRequest) -> ToolMessage | None:
    """A call reading a result the run keeps, answered: ``read_file`` with its page (the graph
    has no ``read_file`` of its own, or the path is one the run keeps: a result cut to fit the
    window, in a Deep Agents graph); ``query_result`` with what it computes (``tools.query``)
    — over a result the run keeps, or one Deep Agents saved in the graph's state. Any other
    call: ``None``."""
    call = request.tool_call
    args = dict(call["args"])
    path = "/" + str(args.get("file_path", "")).lstrip("/")
    kept = path in runtime.replay.journal.results
    said: str | None = None
    cleared = None if kept else _cleared_result(request.state, path)
    if call["name"] == READ_FILE and cleared is not None:
        said = results.read(args, text=cleared)
    elif call["name"] == READ_FILE and (request.tool is None or kept):
        said = results.read(args)
    elif call["name"] == query.QUERY_RESULT and (request.tool is None or kept):
        said = await query.query(args, None if kept else _state_file(request.state, path))
    if said is None:
        return None  # another tool, or the graph's own read_file of its own files
    return ToolMessage(content=said, tool_call_id=str(call["id"]), name=call["name"])


def _cleared_result(state: Any, path: str) -> str | None:
    """The whole text of a tool result the request no longer shows (LangChain's
    ``ClearToolUsesEdit`` cleared it: the graph's state keeps it), named by Deep Agents' path
    for its call (``/large_tool_results/<call id>``) — ``None`` for any other path, and for a
    result Deep Agents evicted to its backend (its own ``read_file`` reads that file)."""
    prefix = results.PREFIX + "/"
    if not path.startswith(prefix) or not isinstance(state, Mapping):
        return None
    ident = path[len(prefix) :]
    for message in reversed(state.get("messages") or ()):
        if isinstance(message, ToolMessage) and message.tool_call_id == ident:
            text = message.text
            return None if text.startswith(results.EVICTED) else text
    return None


def _state_file(state: Any, path: str) -> str | None:
    """The text of a file Deep Agents keeps in the graph's state (a large result it saved), or
    ``None``."""
    found = (state.get("files") if isinstance(state, Mapping) else None) or {}
    content = (found.get(path) or {}).get("content")
    return "\n".join(content) if isinstance(content, list) else content


def _saved(state: Any) -> bool:
    """Whether the run keeps a result, or the graph's state holds one Deep Agents saved."""
    files = (state.get("files") if isinstance(state, Mapping) else None) or {}
    return results.kept() or any(str(p).startswith(results.PREFIX) for p in files)


def _reader(name: str, description: str, schema: dict[str, Any], answer: Any) -> BaseTool:
    """A tool reading a kept result (``tools.results.READERS``) as a LangChain tool, offered
    once there is one, its calls answered by :class:`HarnessTools` (no tool node runs it)."""

    async def run(**args: Any) -> str:
        return str(await answer(args))

    return StructuredTool.from_function(
        coroutine=run, name=name, description=description, args_schema=schema
    )


#: The harness's ``read_file`` and ``query_result`` as LangChain tools.
READERS: Final = {r[0]: _reader(*r) for r in results.READERS}
READ_FILE_TOOL: Final = READERS[READ_FILE]
#: The runs a :class:`HarnessTools` offers the harness's tools to (:func:`_paged`).
_PAGED: Final[weakref.WeakSet[Runtime]] = weakref.WeakSet()


async def _helped(
    runtime: Runtime,
    name: str,
    key: str,
    args: dict[str, Any],
    ran: Callable[[dict[str, Any]], Awaitable[ToolMessage | Command[Any]]],
) -> ToolMessage | Command[Any]:
    """A Deep Agents ``task`` call (:data:`TASK_TOOL`, with its arguments) as a helper of the
    run (``bridges.helpers``): its child run started, the sub-agent given the memory context
    for its task before the task, its tool calls recorded under its run while it works, and
    ended with its answer or its error (a pause leaves it to the next attempt). Any other call
    runs as it is."""
    helpers = runtime.helpers
    if helpers is None or name != TASK_TOOL or not set(args) >= TASK_ARGUMENTS:
        return await ran(args)
    task = str(args["description"])
    helper = await helpers.start(key, kind=Kind.TASK, name=str(args["subagent_type"]), task=task)
    try:
        with helpers.working(helper):
            done = await ran({**args, "description": helpers.briefed(helper, task)})
    except BaseException as exc:
        if not is_pause_signal(exc):  # a pause: the next attempt ends it
            await helpers.end(key, error=f"{type(exc).__name__}: {exc}")
        raise
    if isinstance(done, ToolMessage) and done.status == "error":
        await helpers.end(key, error=text_of(done.content))
    else:
        await helpers.end(key, output=text_of(_told(done)))
    return done


#: The arguments Deep Agents' sub-agent tool (:data:`TASK_TOOL`) takes (``SubAgentMiddleware``).
TASK_ARGUMENTS: Final = frozenset({"description", "subagent_type"})


def _told(done: Any) -> Any:
    """What a tool's result says to the model: a message's content, or a ``Command``'s last
    message's (its state update's)."""
    if isinstance(done, Command):
        update = done.update if isinstance(done.update, Mapping) else {}
        messages = update.get("messages") or []
        return getattr(messages[-1], "content", messages[-1]) if messages else ""
    return getattr(done, "content", done)


def govern_tool_nodes() -> None:
    """Every LangGraph tool node, inside a harness run, governed (:func:`governed_call`): a
    ``create_agent`` or Deep Agents graph's (its ``task`` sub-agents' graphs too), a hand-built
    graph's ``ToolNode`` — whoever built them, with no middleware of the harness's. Installed
    once per process, on the tool node's one execution of a call (after its own wrappers);
    outside a harness run it changes nothing."""
    execute = ToolNode._execute_tool_async
    if getattr(execute, _INSTALLED, False):
        return

    async def governed(
        self: ToolNode, request: ToolCallRequest, input_type: Any, config: RunnableConfig
    ) -> Any:
        if current() is None:
            return await execute(self, request, input_type, config)

        async def handler(given: ToolCallRequest) -> Any:
            return await execute(self, given, input_type, config)

        return await governed_call(request, handler)

    setattr(governed, _INSTALLED, True)
    ToolNode._execute_tool_async = governed  # type: ignore[method-assign]


#: The mark of the tool node's execution :func:`govern_tool_nodes` installed.
_INSTALLED: Final = "_trellis_governed"


def side_effects_of(metadata: Mapping[str, Any] | None) -> str:
    """A framework tool's risk as it declares it: ``side_effects`` (``read``, ``write``,
    ``irreversible``), else MCP-style annotations (``readOnlyHint``: read, ``destructiveHint``:
    irreversible), else ``write``."""
    found = dict(metadata or {})
    declared = found.get("side_effects")
    if declared in get_args(SideEffects):
        return str(declared)
    return annotated(found.get("annotations"))


def spec_of(tool: BaseTool) -> ToolSpec:
    """A framework's own LangChain tool as the harness governs, records and publishes it: its
    name, description and schema, its risk as it declares it (:func:`side_effects_of`)."""
    return ToolSpec(
        name=tool.name,
        description=tool.description or tool.name,
        input_schema=_schema_of(tool),
        side_effects=side_effects_of(tool.metadata),
        **data_of(tool.metadata, name=tool.name),
    )


def _schema_of(tool: Any) -> dict[str, Any]:
    from langchain_core.utils.function_calling import convert_to_openai_tool  # noqa: PLC0415

    return dict(convert_to_openai_tool(tool)["function"].get("parameters") or {})


#: The ``response_metadata`` key of a user message steered into a run (``mode="steer"``):
#: the run it was sent as (its transcript keeps it as said, not as the run's question).
STEERED: Final = "trellis_steered"


class HarnessTools(AgentMiddleware):
    """The run's harness tools, offered per model call and run through the bridge — and,
    before each model call, the messages sent to the run's conversation with
    ``mode="steer"`` since the last one (``Runtime.steered``), added to the graph's messages
    as the user's, each with its id (``conversation.question_id``)."""

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

    def before_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        run = current()
        steered = run.steered() if run is not None else []
        if not steered:
            return None
        assert run is not None
        said = [
            HumanMessage(
                content=s.text,
                id=question_id(run.conversation, s.run_id),
                response_metadata={STEERED: s.run_id},
            )
            for s in steered
        ]
        return {"messages": said}

    async def abefore_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        return self.before_model(state, runtime)

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
        _PAGED.add(runtime)
        own = [t for t in request.tools if _named(t) not in runtime.toolbox]
        offered = [t for name, t in harness.items() if runtime.offers(name)]
        if _saved(request.state):  # a large result kept: read in pages, or queried
            names = {_named(t) for t in own}
            offered.extend(t for name, t in READERS.items() if name not in names)
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
        page = await _kept_read(runtime, request) if runtime is not None else None
        if page is not None:
            return page
        tool = runtime.toolbox.get(call["name"]) if runtime is not None else None
        if runtime is None or tool is None:
            return await handler(request)
        step, n = self._step(runtime, request)
        call_id = str(call["id"])
        first = step.repeats.get(call_id)
        if first is not None:
            outcome = bridge.same(runtime, tool.name, call["args"], call_id=call_id, first=first)
        elif call_id in step.writes:
            async with _turn(runtime, step, call_id):
                outcome = await bridge.call(
                    tool, call["args"], call_id=call_id, step=step.first + n
                )
        else:
            outcome = await bridge.call(tool, call["args"], call_id=call_id, step=step.first + n)
        return _message(outcome, call_id, tool.name, kept=_paged(runtime, request))

    def _step(self, runtime: Runtime, request: ToolCallRequest) -> tuple[_Step, int]:
        """The model step the call is one of (numbered on its first call's arrival), and the
        call's place in it."""
        call_id = request.tool_call["id"]
        at, message = _step_of(request)
        calls = message.tool_calls
        # where the step is in the conversation, and what it is: a model may reuse call ids
        key = content_key("step", at, message.id, [c["id"] for c in calls])
        steps = self._steps.setdefault(runtime, {})
        step = steps.get(key)
        if step is None:
            repeats = _repeats(message)
            writes = [
                str(c["id"])
                for c in calls
                if _writes(runtime.toolbox.get(c["name"])) and str(c["id"]) not in repeats
            ]
            ran = {w for w in writes if runtime.replay.journal.calls.get(_ran(key, w))}
            step = steps[key] = _Step(
                key, runtime.next_step(len(calls)), writes, ran, repeats=repeats
            )
        return step, next(n for n, c in enumerate(calls) if c["id"] == call_id)


def _step_of(request: ToolCallRequest) -> tuple[int, AIMessage]:
    """The model message making a tool call's request, and where it is in the conversation the
    request carries (its graph's state); one of the call alone, at the end, when the
    conversation does not hold it."""
    messages = request.state.get("messages", []) if isinstance(request.state, dict) else []
    return _calling(messages, request.tool_call["id"]) or (
        len(messages),
        AIMessage(content="", tool_calls=[request.tool_call]),
    )


def _repeats(message: AIMessage) -> dict[str, str]:
    """The calls of a model step repeating an earlier one of it, by id: the earlier one's
    (``bridge.firsts``)."""
    return bridge.firsts((str(c["id"]), c["name"], c["args"]) for c in message.tool_calls)


def _calling(messages: Sequence[AnyMessage], call_id: str | None) -> tuple[int, AIMessage] | None:
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
        runtime.replay.record_call(_ran(step.key, call_id), True, stepped=True)
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
    """The request with only the tools the run offers now: a graph's are bound when it is
    built, so a part turned off afterwards (``h.wrap(without=)``, a run's
    ``without=``) would still be offered (a call of it is refused: it is off), and so would the
    tools the tool hints leave out (``Runtime.offers``) — the harness's and the graph's own
    tools of the run's set alike; a tool outside the set (the graph's state tools) stays."""
    if runtime is None:
        return request
    kept = [t for t in request.tools if _offers(runtime, t)]
    return request if len(kept) == len(request.tools) else request.override(tools=kept)


def _offers(runtime: Runtime, tool: BaseTool | dict[str, Any]) -> bool:
    feature = (getattr(tool, "metadata", None) or {}).get(converted.FEATURE)
    if feature is not None and not runtime.uses(feature):
        return False
    return runtime.offers(_named(tool))  # the graph's own tools of the run's set narrowed too


async def _narrow(
    request: ModelRequest[Any],
    handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
) -> ModelResponse[Any]:
    """A model call offered only the tools the run offers (:func:`_offered`), the memory
    context sent with the question (:func:`with_question`)."""
    return await handler(with_question(_offered(request, current())))


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
    """Around every model call: the task kept ahead of a summary (the conversation's leading system
    messages, the pushed memory context among them, and its first user message), the run's
    ``on_error`` hooks for a call that failed, a ``chat`` span with the usage, at most ``timeout``
    seconds (and what is left of the run's time), and the stored ``prompt`` of the gateway
    (``stored("name", **vars)``), pinned for the run: with no variables to fill, on a routed
    model through Bifrost, selected by every call (headers the gateway reads: it prepends it);
    otherwise rendered into the instructions, its other messages before the conversation."""

    def __init__(self, *, timeout: float | None = None, prompt: Stored | None = None) -> None:
        super().__init__()
        if timeout is not None and timeout <= 0:
            raise ConfigurationError("a model timeout is a number of seconds over 0")
        self.timeout = timeout
        self.prompt = prompt
        #: each run's tools in the order they were first offered (:meth:`_in_order`)
        self._order: weakref.WeakKeyDictionary[Runtime, dict[str, int]] = (
            weakref.WeakKeyDictionary()
        )

    @hook_config(can_jump_to=["model"])
    async def aafter_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        """A call whose arguments the chat model could not parse (``invalid_tool_calls``: not
        JSON, not an object) is answered with what was wrong, as a call of the harness's is;
        a step of such calls only asks the model again (LangChain would end the run)."""
        last = state["messages"][-1] if state.get("messages") else None
        if not isinstance(last, AIMessage) or not last.invalid_tool_calls:
            return None
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
        request = _with_task(_unbroken(_offered(request, runtime)))
        request = self._in_order(with_question(request), runtime)
        prompt = await self._pinned(runtime)
        if prompt is not None:
            request = self._prompted(request, prompt)
        name = _model_name(request.model)
        extra = prompt.attributes() if prompt is not None else None
        with model_span(name or "model", lambda: _said(request), extra=extra) as span:
            # in the model's span: the gateway's span of the call joins it. A model name the
            # harness routes (``gateways.routed_chat_model``), on a route that attributes
            # (Bifrost): every call carries the run's dimensions and its ``traceparent``
            if routed(request.model) and route_now().has(Capability.ATTRIBUTION):
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
                await running().failed("model", error)
                if error is exc:
                    raise
                raise error from exc
            _replied(span, reply)
        return reply

    def _in_order(self, request: ModelRequest[Any], runtime: Runtime | None) -> ModelRequest[Any]:
        """The tools in the order the run first offered them — the first call's by name, each
        tool offered later (a ``tool_search`` find, a long task's tools) after them — so a
        provider's prompt cache, whose prefix the tools open, keeps it while the set only
        grows: a tool offered again returns to its place, never moving the others."""
        if runtime is None or len(request.tools) < 2:
            return request
        order = self._order.setdefault(runtime, {})
        for name in sorted({_named(t) for t in request.tools} - order.keys()):
            order[name] = len(order)
        tools = sorted(request.tools, key=lambda t: order[_named(t)])
        same = all(a is b for a, b in zip(tools, request.tools, strict=True))
        return request if same else request.override(tools=tools)

    async def _pinned(self, runtime: Runtime | None) -> ResolvedPrompt | None:
        """The stored prompt's version this run uses: the journal's (a resumed run uses the
        same), else the one the gateway serves now (journaled)."""
        if self.prompt is None:
            return None
        if runtime is None:
            raise ConfigurationError(
                "a stored system_prompt is pinned for a harness run: outside one, render it "
                "yourself (await h.prompt(...))"
            )
        return await runtime.agent.harness.prompts.pin(self.prompt.ref, runtime)

    def _prompted(self, request: ModelRequest[Any], prompt: ResolvedPrompt) -> ModelRequest[Any]:
        assert self.prompt is not None
        pin = prompt.pin()
        if (
            pin is not None
            and not prompt.variables
            and routed(request.model)
            and route_now().has(Capability.PROMPTS)
        ):
            sent = request.model_settings.get("extra_headers", {})
            headers = {**sent, **pin.options().headers()}
            return request.override(
                model_settings={**request.model_settings, "extra_headers": headers}
            )
        rendered = prompt.messages(**self.prompt.values)
        own = [message_text(m.get("content")) for m in rendered if m.get("role") == "system"]
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


#: The memory context as it is sent with the user's message (:func:`with_question`): marked
#: off, so the message after it reads as the question, not one more note.
ASKED_WITH: Final = (
    "<memory_context>\nWhat memory holds for this message (background; not the user's "
    "message, which follows):\n\n{context}\n</memory_context>"
)


def with_question(request: ModelRequest[Any]) -> ModelRequest[Any]:
    """The request with the pushed memory context (the system message
    :data:`~trellis.harness.adapters.langgraph.CONTEXT_MESSAGE_ID`) sent as the first part of
    this turn's question rather than ahead of the history. It is new for every question, and a
    provider's prompt cache keeps only what a request shares with an earlier one from its start:
    so the next turn's request repeats this one's history as it was sent (audit K1). It is a
    part of the user's message, never a system message after the first: some providers refuse
    one there. Only the request changes; the conversation the graph keeps does not. The
    question is the latest user message the graph's state holds (a spent budget's last word
    is not). The context is marked off (:data:`ASKED_WITH`): after a long one, a model read
    the question as one more saved note.

    A cache that keeps only what a request marks (Anthropic's: a prompt-caching middleware
    sets the request's ``cache_control``) keeps the history too: the answer just before the
    question is marked, the end of what the next turn repeats (:func:`_history_marked`)."""
    messages = list(request.messages)
    at = next(
        (
            n
            for n, m in enumerate(messages)
            if isinstance(m, SystemMessage) and m.id == CONTEXT_MESSAGE_ID
        ),
        None,
    )
    if at is None:
        return request
    kept = {m.id for m in request.state.get("messages", []) if m.id is not None}
    asked = next(
        (
            n
            for n in range(len(messages) - 1, at, -1)
            if isinstance(messages[n], HumanMessage) and messages[n].id in kept
        ),
        None,
    )
    if asked is None:
        return request
    question = messages[asked]
    said = (
        question.content
        if isinstance(question.content, list)
        else [{"type": "text", "text": question.content}]
    )
    told = [{"type": "text", "text": ASKED_WITH.format(context=messages[at].text)}, *said]
    messages[asked] = question.model_copy(update={"content": told})
    marker = request.model_settings.get("cache_control")
    if marker and asked > at + 1:
        messages[asked - 1] = _history_marked(messages[asked - 1], marker)
    del messages[at]
    return request.override(messages=messages)


def _history_marked(answer: AnyMessage, marker: Any) -> AnyMessage:
    """The answer that ends the history with a cache breakpoint (``marker``) on its last text
    part; any other message (a call, a result) as it is, and so is an answer that has one."""
    if not isinstance(answer, AIMessage) or answer.tool_calls:
        return answer
    content = answer.content
    parts: list[Any] = (
        list(content) if isinstance(content, list) else [{"type": "text", "text": content}]
    )
    texts = [
        n
        for n, p in enumerate(parts)
        if isinstance(p, dict) and p.get("type") == "text" and p.get("text")
    ]
    if not texts or any(isinstance(p, dict) and "cache_control" in p for p in parts):
        return answer
    parts[texts[-1]] = {**parts[texts[-1]], "cache_control": marker}
    return answer.model_copy(update={"content": parts})


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
    error, as a :class:`ModelError` that says whether a retry may pass (no answer, a timeout,
    or a status in bifrost-sdk's ``RETRYABLE``, the set the gateway client and the memory
    service retry on); anything else as it is."""
    if isinstance(exc, TimeoutError):
        within = "" if timeout is None else f" within {timeout:g}s"
        return ModelTimeout(
            f"the model did not answer{within}: the step stalled",
            source="langgraph",
            details={} if timeout is None else {"seconds": timeout},
        )
    try:
        import openai  # noqa: PLC0415 - the model client of the react extra
    except ImportError:  # pragma: no cover - the react extra installs it
        return exc
    if isinstance(exc, openai.APIConnectionError):  # a timeout is one
        return ModelError(f"the model call failed: {exc}", source="langgraph", retryable=True)
    if isinstance(exc, openai.APIStatusError):
        status = exc.status_code
        retryable = status in RETRYABLE  # bifrost-sdk's: the one set every model call retries on
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
def step_limit(
    max_steps: int = MAX_STEPS, *, scoped: bool = False
) -> AgentMiddleware[Any, Any, Any]:
    """At most ``max_steps`` model calls that may call tools, then one that answers
    (:class:`AnswerAtLimit`). ``scoped``: counted per thread (a ``ReAct`` graph's thread is its run,
    so a resume keeps the count), else per invocation (``middleware()`` in a graph whose thread
    outlives a run)."""
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
        raise ConfigurationError("max_steps is a number of model calls from 1")
    return AnswerAtLimit(max_steps, scoped=scoped)


class AnswerAtLimit(ModelCallLimitMiddleware):
    """LangChain's own ``ModelCallLimitMiddleware`` counting the model calls, which after
    ``max_steps`` of them makes one more — the model told the limit is reached
    (``budget.LIMIT``) and offered no tool, as :class:`StallGuard` does on a stall
    (:func:`_answer_now`) — so the run ends with an answer saying what was found and what is
    left, not LangChain's "Model call limits exceeded" message. LangChain's limit, one call
    past ``max_steps``, stays the backstop (its ``exit_behavior="end"``). Named as LangChain's
    (a ``ReAct(middleware=)`` of that name is refused, naming ``max_steps=``)."""

    def __init__(self, max_steps: int, *, scoped: bool) -> None:
        self.max_steps = max_steps
        self.counted = "thread_model_call_count" if scoped else "run_model_call_count"
        limit = {"thread_limit" if scoped else "run_limit": max_steps + 1}
        super().__init__(exit_behavior="end", **limit)

    @property
    def name(self) -> str:
        return ModelCallLimitMiddleware.__name__

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        if request.state.get(self.counted, 0) < self.max_steps:
            return await handler(request)
        return await _answer_now(
            request,
            handler,
            budget.LIMIT.format(limit=self.max_steps),
            f"stopped: the step limit ({self.max_steps}) was reached",
            warning="step_limit",
        )


class StallGuard(AgentMiddleware):
    """A run that goes nowhere ends with one answer: the same call with the same arguments in
    ``max_repeats`` consecutive steps (before it runs again), or :data:`ERROR_STREAK`
    consecutive steps in which every call failed (a malformed one among them) and a failure
    was met again (:func:`_failing`). The model is told so (``bridge.STALLED``,
    :data:`FAILING`) and called once more, offered no tool (``tool_choice="none"``), so the
    user is told what went wrong; no answer then stops the run (``ModelError``)."""

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
            return await _answer_now(
                request,
                handler,
                FAILING.format(steps=failing),
                f"stopped: every tool call failed in {failing} consecutive steps",
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
                    return await _answer_now(
                        request,
                        handler,
                        bridge.STALLED.format(tool=call["name"], times=streak),
                        f"stopped: the model called {call['name']!r} with the same arguments "
                        f"{streak} times in a row (a stall)",
                    )
        return reply


async def _answer_now(
    request: ModelRequest[Any],
    handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    told: str,
    stopped: str,
    *,
    warning: str = "stalled",
) -> ModelResponse[Any]:
    """The model called once more, told ``told`` and offered no tool; its answer (``warning``
    on the run's stream), or the run stopped (``ModelError(stopped)``) when it gives none."""
    last = request.override(
        messages=[*request.messages, HumanMessage(content=told)],
        tool_choice="none" if request.tools else request.tool_choice,
    )
    reply = await handler(last)
    message = _ai(reply)
    if message is None or message.tool_calls or not message.text:
        raise ModelError(stopped, source="langgraph")
    runtime = current()
    log.warning("run %s %s", runtime.run_id if runtime else "-", stopped)
    if runtime is not None:
        runtime.events.warning(warning, stopped)
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
    """How many of the latest steps had every call fail — when a failure among them repeats
    (the same tool failing with the same error), else 0: a model whose every failed call
    failed differently read each error and is correcting itself (a wrong type, then a missing
    argument, then a made-up name), which only the step limit bounds; one that meets an error
    it met before is making no progress (OpenHands' stuck detector counts the same action
    meeting the same error). Read from the end, a step's results (after it) before the step,
    as far as the first step with a call that did not fail."""
    results: dict[str, ToolMessage] = {}
    streak = 0
    met: set[str] = set()
    repeated = False
    for message in _turn_back(history):
        if isinstance(message, ToolMessage):
            results[message.tool_call_id] = message
            continue
        if not isinstance(message, AIMessage):
            continue
        repeats = _repeats(message)  # a repeat did not run: its first call's status counts
        made = [
            (c["name"], results.get(c["id"] or ""))
            for c in message.tool_calls
            if c["id"] not in repeats
        ]
        failed = [(name, r) for name, r in made if r is not None and r.status == "error"]
        if not made or len(failed) < len(made):
            break
        streak += 1
        failures = {content_key("failure", name, r.text) for name, r in failed}
        repeated = repeated or bool(failures & met)
        met |= failures
    return streak if repeated else 0


# --------------------------------------------------------------------------- results
class ReadTools(AgentMiddleware):
    """``read_file``, the one tool that reads back what was put aside, offered once there is
    something to read: once a result reads as ``placeholder`` (cleared by LangChain's
    ``ClearToolUsesEdit``: ``read_file`` pages it from the graph's state by its call's id, the
    harness's own when the graph has none), once the graph's state holds a file (a large result
    saved, a summarized history) or the run keeps a result — a small model offered it from the
    start reads nothing, again and again. ``files=True``: ``read_file`` always offered (its
    files are outside the graph's state: a disk, a store, a sandbox, the skills)."""

    def __init__(self, placeholder: str | None, *, files: bool = False) -> None:
        super().__init__()
        self.placeholder = placeholder
        self.files = files

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        tools = list(request.tools)
        has = any(_named(t) == READ_FILE for t in tools)
        if self.placeholder is not None and any(
            isinstance(m, ToolMessage) and m.content == self.placeholder for m in request.messages
        ):
            if not has:  # a result was cleared: read_file reads it back
                tools.append(READ_FILE_TOOL)
        elif has and not (self.files or request.state.get("files") or results.kept()):
            tools = [t for t in tools if _named(t) != READ_FILE]
        return await handler(
            request if tools == list(request.tools) else request.override(tools=tools)
        )


# --------------------------------------------------------------------------- the window
#: Tools a narrowing keeps at least, beside the ones the run already used (then the
#: descriptions are compacted before any of them is dropped).
FIT_KEEP: Final = 8
#: A compacted tool description's most characters (its first sentence, at most this).
FIT_DESCRIPTION: Final = 120
#: The characters a cut result's marker takes at most (its size, and where the rest is read).
CUT_MARKER: Final = 192
#: The steps :class:`FitWindow` takes, in order, each only while the request does not fit;
#: how often each fired in this process (a bounded counter: these keys only).
FIT_STEPS: Final = ("tools", "descriptions", "results", "turns", "memory")
#: ``refit``: a request the model's server refused as over its window, fitted again with the
#: server's own count (:data:`_RATIOS`) and sent once more.
FITTED: Final[collections.Counter[str]] = collections.Counter()
#: A model's tokens per counted token, by model name: learned from what its replies say their
#: requests were, and from a request its server refused as too large (the count is an
#: estimate: a tokenizer makes more or fewer tokens of the same text); within ``FIT_RATIOS``.
_RATIOS: Final[LRU[str, float]] = LRU(64)
FIT_RATIOS: Final = (0.8, 2.0)
#: A model server's refusal of a request over its window (OpenAI's, Anthropic's, llama.cpp's,
#: vLLM's words), and the size it says the request was.
_OVERFLOW: Final = re.compile(
    r"context_length_exceeded|exceed\w* (?:the )?(?:available |maximum )?context|"
    r"maximum context length|prompt is too long|input tokens exceed",
    re.IGNORECASE,
)
_OVERFLOW_SIZE: Final = re.compile(r"(?:request \(|resulted in |too long: )(\d+) tokens")
_WORD: Final = re.compile(r"[a-z0-9]+")


#: What says a line of the memory context states a rule (``FitWindow`` never trims it): a rule,
#: a policy, a "never", an "always", a "must", something forbidden or not allowed.
_RULE: Final = re.compile(
    r"\b(?:rules?|polic(?:y|ies)|never|always|must|forbid\w*|prohibit\w*|not allowed)\b",
    re.IGNORECASE,
)


class ContextTooLarge(ModelError):
    """The user's own message, with the instructions, does not fit the model's window."""

    code = "CONTEXT_TOO_LARGE"


@dataclass(slots=True)
class ThisTurnsResults(ClearToolUsesEdit):
    """LangChain's ``ClearToolUsesEdit`` for the turn under way: past its ``trigger`` (counted
    over the whole request, as LangChain counts it), it clears the older results of the calls
    made since the user's latest message, never those of the earlier turns before it. The
    conversation the memory service replays was sized for the request it goes into (its
    budget is what the window leaves: ``Agent.conversation_budget``), so clearing its results
    at the turn's start would only undo that; a request still over the window is fitted by
    :class:`FitWindow`, which leaves older turns out first."""

    def apply(self, messages: list[AnyMessage], *, count_tokens: TokenCounter) -> None:
        start = next(
            (i for i in range(len(messages) - 1, -1, -1) if isinstance(messages[i], HumanMessage)),
            0,
        )
        turn = messages[start:]
        earlier = count_tokens(messages[:start]) if start else 0
        ClearToolUsesEdit.apply(
            replace(self, trigger=self.trigger - earlier), turn, count_tokens=count_tokens
        )
        messages[start:] = turn


class FitWindow(AgentMiddleware):
    """A model call never fails because its request does not fit the window. Before each call,
    when the request (its messages, instructions and tools, counted as Deep Agents counts them)
    is over :data:`FIT_SHARE` of the window less the reply's tokens, it is made smaller, step
    by step, only as far as it must, each step said on the run (a ``context_fit`` warning and a
    log line) and counted (:data:`FITTED`):

    1. ``tools`` — the offered tools narrowed to the most relevant: those the run already used
       always kept, then those the tool hints chose, then by the words they share with the
       request, at least :data:`FIT_KEEP`;
    2. ``descriptions`` — the kept tools' descriptions compacted (the first sentence, no
       parameter descriptions);
    3. ``results`` — the tools narrowed past :data:`FIT_KEEP`, then the longest results cut
       to their head and tail, each kept whole and its cut naming where ``read_file`` reads the
       rest (clearing older results is LangChain's ``ClearToolUsesEdit``'s, before it: one
       clearing pass). ``turns=False`` (a Deep Agents summarization after it) leaves the
       results to it too: it offloads a result whole to its backend and reads it back with its
       own ``read_file``, so a result is stored once, where its sandbox's code reads it;
    4. ``turns`` — the older turns left out, the latest kept whole (``turns=False`` leaves them
       to a summarization after it: ``ReAct``'s summarizes them);
    5. ``memory`` — last, only when the latest turn would still not fit: the memory context's
       lowest-ranked items left out (those its ``## Most relevant`` does not name first, the
       last first), never a line that states a rule, nor one of the standing instructions (the
       profile), the summary, the recent turns or the learned plan (its skills and tools).

    The count is an estimate (Deep Agents'): when the model's server still refuses the request
    as over its window (its tokenizer makes more tokens of it), it is fitted again by the
    server's own count and sent once more (``refit``), that model's later calls fitted by the
    learned ratio from the start (a refusal that states no count tightens that call only); the
    ratio is also learned from what each reply says its request was, so a request that fits by
    the model's own count is sent as it is.

    When the instructions and the user's own message alone do not fit, the call fails with a
    :class:`ContextTooLarge` naming the window and the size. A request that fits is sent as it
    is, the same object. ``window``: the model's input window in tokens, else its LangChain
    profile's; neither known, it steps aside."""

    @removed(optional="it dropped the tool guidance, which is gone: nothing else is optional")
    def __init__(self, window: int | None = None, *, turns: bool = True) -> None:
        super().__init__()
        if window is not None and window <= 0:
            raise ConfigurationError("a window is a number of tokens over 0")
        self.window = window
        self.turns = turns
        #: each run's tools its last narrowed call kept (:func:`_narrowed`'s ``before``)
        self._kept: weakref.WeakKeyDictionary[Runtime, frozenset[str]] = weakref.WeakKeyDictionary()

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        budget = self._budget(request)
        if budget is None:
            return await handler(request)
        name = _model_name(request.model) or ""
        ratio = _RATIOS.get(name) or 1.0
        sent, counted = self._within(request, int(budget / ratio))
        try:
            reply = await handler(sent)
            _calibrated(name, counted, reply)
            return reply
        except Exception as exc:
            real = _refused_size(exc)
            if real is None:
                raise
            # the server counts more than the estimate: fitted again by its count, once — a
            # count it stated is the model's from then on; a refusal that states none (Deep
            # Agents' own budget check after its summary) tightens this call only
            if real and counted:
                learned = _RATIOS[name] = max(ratio * 1.1, real / counted)
            else:
                learned = ratio * 1.25
            FITTED["refit"] += 1
            log.info("the model's server refused %s tokens: fitted again", real or "too many")
            return await handler(self._within(request, int(budget / learned))[0])

    def _within(self, request: ModelRequest[Any], budget: int) -> tuple[ModelRequest[Any], int]:
        """The request fitted to ``budget`` (counted tokens), and its count."""
        size = _size(request)
        if size <= budget:
            return request, size
        fitted = self._fitted(request, budget)
        return fitted, _size(fitted)

    def _budget(self, request: ModelRequest[Any]) -> int | None:
        profile = getattr(request.model, "profile", None)
        window = self.window or (
            profile.get("max_input_tokens") if isinstance(profile, dict) else None
        )
        if not isinstance(window, int) or isinstance(window, bool):
            return None
        reply = 0
        for key in ("max_tokens", "max_completion_tokens", "max_output_tokens"):
            value = request.model_settings.get(key, getattr(request.model, key, None))
            if isinstance(value, int) and not isinstance(value, bool):
                reply = max(reply, value)
        return max(0, int(window * FIT_SHARE) - reply)

    def _results(self, request: ModelRequest[Any], budget: int) -> ModelRequest[Any]:
        """The longest results cut (:func:`_cut`) — none with a summarization after it, whose
        own overflow handling offloads them whole (Deep Agents')."""
        return _cut(request, budget) if self.turns else request

    def _fitted(self, request: ModelRequest[Any], budget: int) -> ModelRequest[Any]:
        runtime = current()
        before = self._kept.get(runtime, frozenset()) if runtime is not None else frozenset()
        taken: list[str] = []
        steps: tuple[Callable[[ModelRequest[Any], int], ModelRequest[Any]], ...] = (
            lambda r, b: _narrowed(r, b, runtime, least=FIT_KEEP, before=before),
            lambda r, b: r.override(tools=[_compact(t) for t in r.tools]),
            lambda r, b: self._results(_narrowed(r, b, runtime, least=0, before=before), b),
            lambda r, b: _latest(r) if self.turns else r,
            lambda r, b: _remembered_less(r, b, after=None if self.turns else _latest),
        )
        for name, step in zip(FIT_STEPS, steps, strict=True):
            if name == "memory" and not self.turns and _size(_latest(request)) <= budget:
                break  # the summarization after it makes it fit: memory stays whole
            smaller = step(request, budget)
            if smaller is not request:
                taken.append(name)
                FITTED[name] += 1
                request = smaller
            if _size(request) <= budget:
                break
        size = _size(request if self.turns else _latest(request))
        if size > budget and _size(_bare(request)) > budget:
            raise ContextTooLarge(
                f"the instructions and the user's message alone are about "
                f"{_size(_bare(request))} tokens, more than the {budget} this model's window "
                "holds for a request: shorten the message or use a model with a larger window",
                source="langgraph",
                details={"budget": budget, "size": size},
            )
        if runtime is not None and ("tools" in taken or "results" in taken):
            self._kept[runtime] = frozenset(_named(t) for t in request.tools)
        said = (
            f"the request was made to fit the model's window ({budget} tokens): {', '.join(taken)}"
        )
        log.info("%s", said)
        if runtime is not None and taken:
            runtime.events.warning("context_fit", said)
        return request


def _calibrated(name: str, counted: int, reply: Any) -> None:
    """The model's tokens per counted token, from what its reply says the request was
    (``usage_metadata``): the count is an estimate, over a tokenizer's count or under it; a
    request that fits the window by the model's own count is then not narrowed. Averaged with
    what was learned, within :data:`FIT_RATIOS`."""
    said = reply.result[-1] if isinstance(reply, ModelResponse) and reply.result else reply
    usage = getattr(said, "usage_metadata", None) or {}
    real = usage.get("input_tokens")
    if not isinstance(real, int) or real <= 0 or counted <= 0:
        return
    low, high = FIT_RATIOS
    observed = min(high, max(low, real / counted))
    known = _RATIOS.get(name)
    _RATIOS[name] = observed if known is None else (known + observed) / 2


def _refused_size(exc: BaseException) -> int | None:
    """The size a model server said a request it refused as over its window was (0: not
    said), or ``None`` when ``exc`` is no such refusal (nor caused by one)."""
    from langchain_core.exceptions import ContextOverflowError  # noqa: PLC0415

    seen: BaseException | None = exc
    while seen is not None:
        text = str(seen)
        if isinstance(seen, ContextOverflowError) or _OVERFLOW.search(text):
            said = _OVERFLOW_SIZE.search(text)
            return int(said.group(1)) if said else 0
        seen = seen.__cause__
    return None


def _size(request: ModelRequest[Any]) -> int:
    """The request's tokens, as Deep Agents' summarization counts them."""
    system = [request.system_message] if request.system_message is not None else []
    return count_tokens_approximately([*system, *request.messages], tools=list(request.tools))


def _bare(request: ModelRequest[Any]) -> ModelRequest[Any]:
    """The request with only its instructions and the user's latest message, no tools."""
    asked = next((m for m in reversed(request.messages) if isinstance(m, HumanMessage)), None)
    return request.override(messages=[asked] if asked is not None else [], tools=[])


def _narrowed(
    request: ModelRequest[Any],
    budget: int,
    runtime: Runtime | None,
    *,
    least: int,
    before: Collection[str] = (),
) -> ModelRequest[Any]:
    """The request's tools ranked — those the run used (and the graph's own state tools),
    those an earlier call of the run kept (``before``: the same set call after call, so a
    prompt cache keeps its prefix), those the tool hints chose, then by the words they share
    with the user's latest message — and kept in that order while they fit what the messages
    leave, at least ``least`` (the used ones always)."""
    tools = list(request.tools)
    if not tools:
        return request
    used = set(runtime.used) if runtime is not None else set()
    hinted = (runtime.offered or set()) if runtime is not None else set()
    asked = next((m.text for m in reversed(request.messages) if isinstance(m, HumanMessage)), "")
    words = set(_WORD.findall(asked.lower()))

    def rank(item: tuple[int, BaseTool | dict[str, Any]]) -> tuple[int, int, int]:
        n, tool = item
        name = _named(tool)
        must = name in used or name in STATE_TOOLS
        shared = len(words & set(_WORD.findall(f"{name} {_described(tool)}".lower())))
        tier = 0 if must else 1 if name in before else 2 if name in hinted else 3
        return (tier, -shared, n)

    room = budget - _size(request.override(tools=[]))
    kept: list[BaseTool | dict[str, Any]] = []
    for _, tool in sorted(enumerate(tools), key=rank):
        must = _named(tool) in used or _named(tool) in STATE_TOOLS
        cost = count_tokens_approximately([], tools=[tool])
        if must or len(kept) < least or cost <= room:
            kept.append(tool)
            room -= cost
    if len(kept) == len(tools):
        return request
    order = {id(t): n for n, t in enumerate(tools)}  # the offered order kept (a cache's prefix)
    return request.override(tools=sorted(kept, key=lambda t: order[id(t)]))


def _described(tool: BaseTool | dict[str, Any]) -> str:
    if isinstance(tool, dict):
        return str((tool.get("function") or tool).get("description") or "")
    return tool.description or ""


def _compact(tool: BaseTool | dict[str, Any]) -> dict[str, Any]:
    """A tool as its definition with the description cut to its first sentence (at most
    :data:`FIT_DESCRIPTION` characters) and no parameter descriptions."""
    from langchain_core.utils.function_calling import convert_to_openai_tool  # noqa: PLC0415

    found = copy.deepcopy(convert_to_openai_tool(tool))
    function = found.get("function", found)
    first = re.split(r"(?<=[.!?])\s", str(function.get("description") or ""), maxsplit=1)[0]
    function["description"] = first[:FIT_DESCRIPTION]
    for spec in (function.get("parameters") or {}).get("properties", {}).values():
        spec.pop("description", None)
    return found


def _latest(request: ModelRequest[Any]) -> ModelRequest[Any]:
    """The request with its leading system messages and its latest turn only (from the user's
    latest message on)."""
    messages = list(request.messages)
    lead = list(itertools.takewhile(lambda m: isinstance(m, SystemMessage), messages))
    asked = max((n for n, m in enumerate(messages) if isinstance(m, HumanMessage)), default=None)
    if asked is None or asked <= len(lead):
        return request
    return request.override(messages=[*lead, *messages[asked:]])


#: The memory context's sections (``## ...``), its items' lines (``- ...``) and their handles
#: (``[m3]``), the sections kept whole — the standing instructions (the profile), the summary,
#: the recent turns, the learned plan (its skills, tools, procedures) — and the section that
#: names the best-ranked items (the memory service's rendering).
_SECTION: Final = re.compile(r"^##\s+(.*)$")
_ITEM_LINE: Final = re.compile(r"^\s*[-*] ")
_HANDLE: Final = re.compile(r"\[([^\]\s]+)\]")
_KEPT_SECTION: Final = re.compile(
    r"^(profile|conversation summary|recent conversation|learned skills|tools|procedures?|plan)\b",
    re.IGNORECASE,
)
_RANKED_SECTION: Final = "most relevant"


def _remembered_less(
    request: ModelRequest[Any],
    budget: int,
    *,
    after: Callable[[ModelRequest[Any]], ModelRequest[Any]] | None = None,
) -> ModelRequest[Any]:
    """The request with the memory context's lowest-ranked items left out until it (``after``
    it is reduced so: the latest turn) fits — those the best-ranked section does not name
    first, the last line first, then the best-ranked ones, their last first; never a line of a
    kept section (:data:`_KEPT_SECTION`), nor one that states a rule (:data:`_RULE`)."""
    messages = list(request.messages)
    at = next(
        (
            n
            for n, m in enumerate(messages)
            if isinstance(m, SystemMessage) and m.id == CONTEXT_MESSAGE_ID
        ),
        None,
    )
    if at is None:
        return request
    lines = messages[at].text.split("\n")
    items: list[tuple[int, str]] = []  # (line, its section)
    section = ""
    for n, line in enumerate(lines):
        if heading := _SECTION.match(line.strip()):
            section = heading.group(1).strip().casefold()
        elif _ITEM_LINE.match(line) and not _KEPT_SECTION.match(section) and not _RULE.search(line):
            items.append((n, section))
    best = {
        handle
        for n, where in items
        if where == _RANKED_SECTION
        for handle in _HANDLE.findall(lines[n])
    }
    ranked = {
        n
        for n, where in items
        if where == _RANKED_SECTION or not best.isdisjoint(_HANDLE.findall(lines[n]))
    }
    lowest = [n for n, _ in reversed(items) if n not in ranked]
    lowest += sorted(ranked, reverse=True)
    over = _size(after(request) if after is not None else request) - budget
    gone: set[int] = set()
    for n in lowest:
        if over <= 0:
            break
        gone.add(n)
        over -= len(lines[n]) // 4 + 1
    if not gone:
        return request
    kept = "\n".join(line for n, line in enumerate(lines) if n not in gone)
    messages[at] = messages[at].model_copy(update={"content": kept})
    return request.override(messages=messages)


def _cut(request: ModelRequest[Any], budget: int) -> ModelRequest[Any]:
    """The longest tool results cut to their head and tail until the request fits — inside a
    run, each kept whole first (``tools.results``) and its cut saying where the rest is: the
    ``read_file`` call that reads on from the line the cut starts on (offered with it), so no
    row of it is out of the model's reach."""
    tools = list(request.tools)
    inside = current() is not None and any(isinstance(m, ToolMessage) for m in request.messages)
    if inside and all(_named(t) != READ_FILE for t in tools):
        tools.append(READ_FILE_TOOL)
    messages = list(request.messages)
    over = _size(request.override(tools=tools)) - budget
    for n in sorted(
        (n for n, m in enumerate(messages) if isinstance(m, ToolMessage)),
        key=lambda n: -len(messages[n].text),
    ):
        if over <= 0:
            break
        text = messages[n].text
        # tokens to characters, as the count does, and room for the marker said in its place
        keep = max(200, len(text) - over * 4 - CUT_MARKER)
        if keep >= len(text):
            continue
        half = keep // 2
        said = f"{len(text) - keep} characters cut to fit the window"
        path = results.keep(text, tool=messages[n].name) if inside else None
        if path is not None:
            line = results.line_at(text, half)
            said += f'; the rest: {results.READ_FILE}(file_path="{path}", offset={line})'
        cut = f"{text[:half]}\n…[{said}]…\n{text[-half:]}"
        messages[n] = messages[n].model_copy(update={"content": cut})
        over -= (len(text) - len(cut)) // 4
    if messages == list(request.messages):
        return request
    return request.override(messages=messages, tools=tools)


# --------------------------------------------------------------------------- checkpoints
class CheckpointUnreadable(HarnessError):
    """A graph checkpoint :class:`RunCheckpointer` kept that cannot be read back (damaged, or
    written with types this release cannot load): the run fails with it, and is not run again
    — starting the graph over would make again the calls its checkpoint had moved past."""

    code = "CHECKPOINT_UNREADABLE"


#: Outside a run, how many threads :class:`RunCheckpointer` keeps (the most recently used).
THREADS_KEPT: Final = 10_000


class RunCheckpointer(BaseCheckpointSaver[int]):
    """A checkpointer that keeps the latest checkpoint only — and, in a harness run whose
    ``thread_id`` is the run (``LangGraphAdapter`` sees to it), keeps it in the run's journal
    (``Journal.graph``), saved with the run's progress and with its pause. A resume in any
    worker continues where the graph stopped: the model is not asked again, a summary is not
    written again, a tool that finished beside a pause is not run again. Outside a run it is
    an in-memory checkpointer.

    A channel kept as deltas (LangGraph's ``DeltaChannel``: Deep Agents' ``messages`` and
    ``files``) is stored in a checkpoint only every so many updates (its snapshot); in between
    the graph rebuilds it from the writes since (``get_delta_channel_history``). With the
    latest checkpoint only, those writes would be gone: each such channel keeps its last
    snapshot and the writes applied since, beside the checkpoint (``deltas``), and they are
    what the history returns — a new snapshot starts them again, so they never hold more than
    the channel's own updates since its last one.

    As the root graph's checkpoint moves past a step, the journal's records of that step's
    calls go (``Replay.passed``): the graph never makes them again, so a long run's journal is
    its latest checkpoint and the records of the step under way. A checkpoint that cannot be
    read back fails the run (:class:`CheckpointUnreadable`)."""

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
        try:
            return self.serde.loads_typed((dumped[0], base64.b64decode(dumped[1])))
        except Exception as exc:
            raise CheckpointUnreadable(
                f"the graph's checkpoint cannot be read back ({type(exc).__name__}: {exc}): "
                "the run stops rather than start the graph over, which would make its calls "
                "again; start a new run"
            ) from exc

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
        before = slot.get(ns) or {}
        parent = _configurable(config).get("checkpoint_id")
        kept = before.get("writes", [])
        # the writes the new checkpoint applied — those of the one it follows — in the order
        # they were applied (the tasks' paths, LangGraph's ``apply_writes``), not finished in
        applied = sorted(
            (w for w in kept if w[0] == parent and before.get("id") == parent),
            key=lambda w: (w[5], w[2]),
        )
        slot[ns] = {
            "id": checkpoint["id"],
            "checkpoint": self._dumped(checkpoint),
            "metadata": self._dumped(get_checkpoint_metadata(config, metadata)),
            "parent": parent,
            "writes": [w for w in kept if w[0] >= checkpoint["id"]],
            "deltas": self._deltas(before.get("deltas") or {}, checkpoint, metadata, applied),
        }
        runtime = current()
        if not ns and runtime is not None and runtime.run_id == thread_id:
            runtime.replay.passed(parent)  # the steps it follows are done: not made again
        return _config(thread_id, ns, checkpoint["id"])

    def _deltas(
        self,
        kept: Mapping[str, Any],
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        applied: Sequence[Sequence[Any]],
    ) -> dict[str, Any]:
        """Each delta channel's last stored value and the writes applied since (``kept``, then
        what this checkpoint holds or ``applied``): the channels the checkpoint's metadata
        counts updates of (``counters_since_delta_snapshot``), those it stores a snapshot of,
        and those kept before."""
        values = checkpoint["channel_values"]
        counted = cast(Mapping[str, Any], metadata.get("counters_since_delta_snapshot") or {})
        names = {*kept, *counted, *(k for k, v in values.items() if isinstance(v, _DeltaSnapshot))}
        deltas: dict[str, Any] = {}
        for name in sorted(names):
            if name in values:  # stored whole: the history starts again from it
                deltas[name] = {"seed": self._dumped(values[name]), "writes": []}
                continue
            entry = kept.get(name) or {"seed": None, "writes": []}
            added = [[w[1], w[4]] for w in applied if w[3] == name]
            deltas[name] = {**entry, "writes": [*entry["writes"], *added]} if added else entry
        return deltas

    def get_delta_channel_history(
        self, *, config: RunnableConfig, channels: Sequence[str]
    ) -> Mapping[str, DeltaChannelHistory]:
        """The delta channels' last stored values and the writes since, for the latest
        checkpoint (an earlier one is not kept: nothing)."""
        thread_id, ns = _where(config)
        record = self._slot(thread_id).get(ns) or {}
        wanted = get_checkpoint_id(config)
        latest = not wanted or wanted == record.get("id")
        deltas = (record.get("deltas") or {}) if latest else {}
        found: dict[str, DeltaChannelHistory] = {}
        for name in channels:
            entry = deltas.get(name) or {"seed": None, "writes": []}
            history: DeltaChannelHistory = {
                "writes": [(task, name, self._loaded(value)) for task, value in entry["writes"]]
            }
            if entry["seed"] is not None:
                history["seed"] = self._loaded(entry["seed"])
            found[name] = history
        return found

    async def aget_delta_channel_history(
        self, *, config: RunnableConfig, channels: Sequence[str]
    ) -> Mapping[str, DeltaChannelHistory]:
        return self.get_delta_channel_history(config=config, channels=channels)

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
    "CheckpointUnreadable",
    "ContextTooLarge",
    "FitWindow",
    "HarnessTools",
    "ModelHooks",
    "ReadTools",
    "RunCheckpointer",
    "StallGuard",
    "model_error",
    "step_limit",
]
