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
* :class:`PromptCache` — Anthropic's prompt cache (LangChain's
  ``AnthropicPromptCachingMiddleware``) for an Anthropic chat model however it is given, a
  model name the harness routes too, and the answer that ends a replayed history marked with it;
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
import json
import logging
import re
import sys
import time
import types
import weakref
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Collection,
    Hashable,
    Iterator,
    Mapping,
    Sequence,
)
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from typing import Annotated, Any, Final, NoReturn, NotRequired, TypeGuard, cast, get_args

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
    ModelCallResult,
    PrivateStateAttr,
    hook_config,
)
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
    convert_to_messages,
)
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, StructuredTool
from langchain_core.utils.function_calling import convert_to_openai_tool
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
from langgraph.checkpoint.serde.base import SerializerProtocol
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
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
    is_pause_signal,
)
from trellis.harness import (
    accounting,  # every model call accounted, and a reply cut at the output limit noted
    budget,
    compat,
    snapshots,
    task_tools,
)
from trellis.harness.adapters.base import FIT_SHARE, helper
from trellis.harness.adapters.langgraph import (
    ASKED,
    CONTEXT_MESSAGE_ID,
    TASK_TOOL,
    carried_plain,
    notified,
    screened,
)
from trellis.harness.bounded import LRU, SAID_ONCE, LRUSet
from trellis.harness.bridges.helpers import Kind
from trellis.harness.clients.memory import TOOL_SEARCH
from trellis.harness.conversation import question_id
from trellis.harness.gateways import Capability, route_now, routed
from trellis.harness.history import ASKED_WITH
from trellis.harness.hooks import running
from trellis.harness.journal import call_key, content_key
from trellis.harness.outbound import TargetRefused
from trellis.harness.prompts import ResolvedPrompt, Stored, message_text
from trellis.harness.removed import removed
from trellis.harness.runtime import DEFERRED, ModelTimeout, Runtime, current
from trellis.harness.schedules import schedule_tools
from trellis.harness.skills import serves_skills
from trellis.harness.telemetry import failed, gateway_headers, model_span, tool_span, usage
from trellis.harness.telemetry import output as span_output
from trellis.harness.tools import ask, bridge, export, query, results, stored
from trellis.harness.tools.base import (
    CUT_SHORT,
    FAILED,
    UNDECLARED,
    SideEffects,
    Sourced,
    Tool,
    arguments_of,
    backoff,
    citations,
    cut_short,
    data_of,
    failed_output,
    max_age,
    not_run,
)
from trellis.harness.tools.bridge import MAX_REPEATS
from trellis.harness.tools.convert import langchain as converted
from trellis.harness.tools.convert import text_of
from trellis.harness.tools.mcp import side_effects_of as annotated
from trellis.harness.tools.results import READ_FILE

log = logging.getLogger("trellis.run")

#: Model calls one run may make (:func:`step_limit`: LangChain's ``ModelCallLimitMiddleware``).
MAX_STEPS: Final = 12
#: Consecutive steps in which every call failed (no such tool, arguments that do not fit, an
#: error or a timeout), one failure among them met again, that stop the run.
ERROR_STREAK: Final = 3
#: What a model whose every call failed in a row is told, for its one answer without tools.
FAILING: Final = (
    "Every tool call failed in the last {steps} steps. Do not call tools: answer now with what "
    "you have, and say what failed."
)
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
    "This file is in this run's own scratch space: the user cannot open or download it."
)
#: What :data:`SCRATCH_NOTE` goes on to say, by what the run has to give the user a file
#: (:func:`_scratch_to_user`): ``query_result``'s ``save_as`` over a result it keeps; code of
#: its own; or nothing — said, so the model tells the user why (told only "give them its
#: content", a Deep Agents model answered "I can't produce a native Excel file" with no reason).
SCRATCH_TO_USER: Final = {
    "save": " query_result's save_as saves a file they can.",
    "code": " Never offer it to them as a file; give them its content instead.",
    "none": " No tool of this run makes a file they can: give them its content, and say so.",
}
#: The metadata key marking a file tool whose backend is outside the graph's state (a disk, a
#: store, a sandbox: ``ReAct(backend=)``): governed as any other tool, its risk as declared.
OUTSIDE_STATE: Final = "trellis_outside_state"
#: Deep Agents' file tools that read, and those that change files: on a backend outside the
#: graph's state they are governed as any tool, with these risks (:func:`outside_state`).
FILE_READS: Final = frozenset({"ls", "read_file", "glob", "grep"})
FILE_WRITES: Final = frozenset({"write_file", "edit_file", "delete"})
FILE_TOOLS: Final = FILE_READS | FILE_WRITES
#: The metadata key holding the backend of a Deep Agents file tool that keeps some paths in
#: the graph's state and others outside it (a ``CompositeBackend``: a state route beside a
#: disk, a store or a sandbox — the run's skills beside a sandbox): each call is the graph's
#: scratch or a governed call of the run by where its path lands (:func:`in_state`).
ROUTED: Final = "trellis_routed_backend"


#: Deep Agents' shell tool, a command in its backend's sandbox: a write, as the harness's own
#: sandbox tool is (``sandbox().tools()``).
EXECUTE: Final = "execute"
#: Deep Agents' tool that compacts the conversation on request (its
#: ``SummarizationToolMiddleware``): the graph's own state — a summary in the history's place,
#: the history kept in its backend as its automatic summarization keeps it —, so a read.
COMPACT: Final = "compact_conversation"
#: Where the code of a tool the framework made lives: Deep Agents', LangChain's agent middleware's.
FRAMEWORK_CODE: Final = ("deepagents.", "langchain.agents.middleware.")
#: Where the code of a tool the harness made for a graph lives: its background-task tools.
HARNESS_CODE: Final = ("trellis.harness.bridges.background_tasks",)


def framework_made(tool: Any) -> bool:
    """Whether the framework made ``tool`` (its code is the framework's): only then is it one
    of the built-ins the harness knows by name — the graph's state tools, Deep Agents' file
    tools and ``execute``. A tool of the agent's own with a built-in's name says what it does
    itself, as any other does."""
    return _code_of(tool).startswith(FRAMEWORK_CODE)


def confined(tool: Any) -> bool:
    """Whether a write of the graph's own ``tool`` keeps to what the run owns, so it is
    announced rather than asked (``Tool.confined``): a framework's own tool, under its own
    permission flow, or one the harness made for the graph (:data:`HARNESS_CODE`: a
    sub-agent's task, whose own calls are governed in its run) — never a tool of the agent's,
    and never a Deep Agents file tool that changes real files (:data:`FILE_WRITES` on a disk, a
    store or a sandbox: :data:`OUTSIDE_STATE`): a write is a write, and asks as any does. One
    that only changes the graph's state is its scratch, not governed at all (:func:`scratch`)."""
    if getattr(tool, "name", None) in FILE_WRITES and (tool.metadata or {}).get(OUTSIDE_STATE):
        return False
    return framework_made(tool) or _code_of(tool).startswith(HARNESS_CODE)


def _code_of(tool: Any) -> str:
    code = getattr(tool, "coroutine", None) or getattr(tool, "func", None)
    return getattr(code, "__module__", None) or ""


def outside_state(name: str) -> dict[str, Any]:
    """The marks of a Deep Agents file tool ``name`` on a backend outside the graph's state:
    governed (:data:`OUTSIDE_STATE`), a read or a write."""
    return {OUTSIDE_STATE: True, "side_effects": "read" if name in FILE_READS else "write"}


def files_backend(tool: Any) -> Any | None:
    """The backend of a Deep Agents file tool, as the ``FilesystemMiddleware`` that made it
    holds it (the tool's code closes over its middleware); ``None`` when it cannot be read: a
    tool of another middleware, a release laid out otherwise, or a backend made per call (a
    factory)."""
    from deepagents.backends.protocol import BackendProtocol  # noqa: PLC0415 - an extra
    from deepagents.middleware.filesystem import FilesystemMiddleware  # noqa: PLC0415

    for code in (getattr(tool, "coroutine", None), getattr(tool, "func", None)):
        for cell in getattr(code, "__closure__", None) or ():
            try:
                held = cell.cell_contents
            except ValueError:  # an empty cell
                continue
            if isinstance(held, FilesystemMiddleware):
                backend = held.backend
                return backend if isinstance(backend, BackendProtocol) else None
    return None


def _state(backend: Any) -> bool:
    """Whether ``backend`` keeps its files in the graph's state: a ``StateBackend`` — not the
    run's skills (``skills.serves_skills``), which it reads as one, from their store: reading
    a skill is a call of the run."""
    from deepagents.backends import StateBackend  # noqa: PLC0415 - an extra

    return isinstance(backend, StateBackend) and not serves_skills(backend)


def _all_state(backend: Any) -> bool:
    """Whether every file of ``backend`` is kept in the graph's state (:func:`_state`), or a
    composite of nothing else."""
    from deepagents.backends import CompositeBackend  # noqa: PLC0415 - an extra

    if isinstance(backend, CompositeBackend):
        return all(_all_state(b) for b in (backend.default, *backend.routes.values()))
    return _state(backend)


def in_state(backend: Any, args: Mapping[str, Any]) -> bool:
    """Whether a Deep Agents file call with ``args`` only touches the graph's state on
    ``backend``: its path (``file_path``, else ``path``, else the root, as the tools default)
    is routed to a state backend as Deep Agents routes it, and no route outside the state lies
    under it — a listing or a search from a folder reads the routes below it too. A path
    Deep Agents would refuse, or anything that cannot be decided, is not: the call is then
    governed (it fails closed)."""
    from deepagents.backends import CompositeBackend  # noqa: PLC0415 - an extra
    from deepagents.backends.composite import _route_for_path  # noqa: PLC0415
    from deepagents.backends.utils import validate_path  # noqa: PLC0415

    given = args.get("file_path") or args.get("path") or "/"
    try:
        path = validate_path(given)
    except (ValueError, TypeError, AttributeError):
        return False
    if not isinstance(backend, CompositeBackend):
        return _all_state(backend)
    under = path if path.endswith("/") else f"{path}/"
    below = [b for prefix, b in backend.sorted_routes if prefix.startswith(under)]
    if not all(_all_state(b) for b in below):
        return False
    routed, routed_path, _ = _route_for_path(
        default=backend.default, sorted_routes=backend.sorted_routes, path=path
    )
    return in_state(routed, {"path": routed_path})


def _file_marks(tool: Any, name: str) -> dict[str, Any]:
    """What a Deep Agents file tool ``name`` is marked with, by where its backend keeps files
    (:func:`files_backend`): nothing when all of them are the graph's state (its calls only
    change the state, which the checkpoint keeps); :func:`outside_state` when none is, or when
    the backend cannot be read (governed: it fails closed); and with the backend to decide
    each call by (:data:`ROUTED`) when some are."""
    backend = files_backend(tool)
    if backend is not None and _all_state(backend):
        return {}
    routed = backend is not None and _routes_some_to_state(backend)
    return {**outside_state(name), **({ROUTED: backend} if routed else {})}


def _routes_some_to_state(backend: Any) -> bool:
    from deepagents.backends import CompositeBackend  # noqa: PLC0415 - an extra

    if isinstance(backend, CompositeBackend):
        return any(_routes_some_to_state(b) for b in (backend.default, *backend.routes.values()))
    return _state(backend)


def scratch(tool: Any, call: Mapping[str, Any]) -> bool:
    """Whether ``call`` of a state tool (:data:`STATE_TOOLS`, the framework's) only changes
    the graph's own state: its tool is not marked :data:`OUTSIDE_STATE`, or — a backend that
    routes some paths to the state (:data:`ROUTED`) — this call's path is one of them."""
    marks = tool.metadata or {}
    if not marks.get(OUTSIDE_STATE):
        return True
    backend = marks.get(ROUTED)
    return backend is not None and in_state(backend, call.get("args") or {})


def declare_files(graph: Any) -> None:
    """Deep Agents' file tools of ``graph`` and of its ``task`` sub-agents' graphs, marked
    :func:`outside_state` wherever their backend keeps files outside the graph's state (a
    disk, a store, a sandbox — alone, or a route of a composite beside a state route:
    :func:`_file_marks`), so those calls are governed as any tool's — as ``ReAct(backend=)``
    marks its own. A tool whose files are all the graph's state is left as it is: its calls
    only change the state, which the checkpoint keeps. Deep Agents'
    :data:`EXECUTE` is declared a write, and its async sub-agents' task tools what each does
    and what its result is (``task_tools``), its :data:`COMPACT` a read — a tool that says so
    itself keeps its word. Only the framework's own (:func:`framework_made`)."""
    from trellis.harness.adapters.langgraph import (  # noqa: PLC0415 - the adapter's, circular
        TASK_TOOL,
        bound_tools,
        subagent_graphs,
    )

    pending = [graph]
    seen: set[int] = set()  # a sub-agent that forks its parent's conversation runs its graphs too
    while pending:
        found = pending.pop()
        if id(found) in seen:
            continue
        seen.add(id(found))
        for tool in bound_tools(found):
            name = getattr(tool, "name", None)
            said = tool.metadata or {}
            if name == TASK_TOOL:
                pending.extend(subagent_graphs(tool))
            elif not framework_made(tool):
                continue
            elif name in FILE_TOOLS and not said.get(OUTSIDE_STATE):
                tool.metadata = {**said, **_file_marks(tool, name)}
            elif name == EXECUTE and "side_effects" not in said:
                tool.metadata = {**said, "side_effects": "write"}
            elif (risk := task_tools.TOOLS.get(str(name))) and "side_effects" not in said:
                given = {} if data_of(said, strict=False) else task_tools.result_of(str(name))
                tool.metadata = {**said, **given, "side_effects": risk}
            elif name == COMPACT and "side_effects" not in said:
                tool.metadata = {**said, "side_effects": "read"}


#: The metadata key marking a tool of the graph's own machinery, not of the agent's tool set
#: (``ReAct``'s: Deep Agents' ``execute`` on a backend that runs nothing, its general-purpose
#: ``task`` helper, offered on a long task only): it is left out of the run's tool set.
MACHINERY: Final = "trellis_machinery"
#: Deep Agents' ``read_file`` in a graph no Deep Agents capability was asked for (``ReAct``
#: alone): a large result is kept by the run, as on any graph without file tools of its own
#: (:func:`_paged`), not saved by Deep Agents in the graph's state.
KEPT_BY_RUN: Final = "trellis_kept_by_run"


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
        # the harness's rule for the call's error: nothing of the graph's handles it, the model
        # reads it (:func:`govern_tool_nodes`)
        _under_way().owned = True
        return await governed_call(request, handler)


#: The call a governed run of a framework's own tool is executing (:func:`governed_call`): the
#: layers under it (a tool node inside a ``GovernTools``) let it through, governed once.
#: By the tool's name and the call's id: a call id is unique in one conversation only, and a
#: sub-agent's calls are another's.
_GOVERNING: ContextVar[tuple[str, str] | None] = ContextVar("trellis_governing", default=None)


def _scratch_said(
    runtime: Runtime, tool: str, done: ToolMessage | Command[Any], state: Any = None
) -> ToolMessage | Command[Any]:
    """A state tool's result: the run's first scratch write that worked says
    :data:`SCRATCH_NOTE` and what gives the user a file (:func:`_scratch_to_user`) — once a
    run: its history keeps saying it."""
    if tool not in SCRATCH_WRITES or runtime.run_id in _SCRATCH_SAID:
        return done
    noted = _scratch_noted(done, SCRATCH_NOTE + _scratch_to_user(runtime, state))
    if noted is not done:
        _SCRATCH_SAID.add(runtime.run_id)
    return noted


def _scratch_to_user(runtime: Runtime, state: Any) -> str:
    """What gives the run's user a file (:data:`SCRATCH_TO_USER`), from what the run has: code
    of its own (``tools.export.runs_code``), else ``query_result``'s ``save_as`` once a result
    is held for it to save (:func:`_saved`: it is offered then), else nothing."""
    if export.runs_code(runtime) is not None:
        return SCRATCH_TO_USER["code"]
    return SCRATCH_TO_USER["save" if _saved(state) else "none"]


def _scratch_noted(
    done: ToolMessage | Command[Any], note: str = SCRATCH_NOTE
) -> ToolMessage | Command[Any]:
    """A scratch write's result with ``note`` after it: on its message, or on the message of
    the ``Command`` that also updates the graph's files (Deep Agents' state backend). A failed
    write is left as it is."""

    def noted(m: Any) -> Any:
        if not isinstance(m, ToolMessage) or m.status == "error" or not isinstance(m.content, str):
            return m
        return m.model_copy(update={"content": f"{m.content}\n{note}"})

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


async def _remote_tasks(runtime: Runtime, call: Any, own: Any, args: Mapping[str, Any]) -> Any:
    """The adapter's record of Deep Agents' tasks on a server (``bridges.async_tasks``), when
    ``call`` is of one of Deep Agents' own task tools (``own``) — a start checked first: past
    the chain of outcomes, or a webhook that leads somewhere private, fails the call, and its
    server is sent nothing.
    ``None`` for any other call."""
    tasks = runtime.agent.adapter.tasks
    if tasks is None or call["name"] not in task_tools.TOOLS or not framework_made(own):
        return None
    try:
        await tasks.starting(runtime, own, args)
    except TargetRefused as exc:
        refused = f"{call['name']} was not run: its webhook: {exc}"
        raise ToolError(refused, source="tools") from exc
    return tasks


async def governed_call(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
) -> ToolMessage | Command[Any]:
    """A framework's own tool call, inside a harness run, run as the harness's are: governance
    decides it (the catalog's risk, else the tool's declared one in its
    ``metadata["side_effects"]``, else its MCP-style annotations, else none: it asks), a
    person approves when it asks, and it is journaled (a resume does not run it again), timed,
    traced and recorded (``tools.bridge``). What the tool returned is returned as it is — a
    ``Command`` (a Deep Agent's ``task`` updating the graph's state), or the list of them and
    messages a tool may return, included — and, replayed from the journal, the same: the
    message the model read, or the ``Command`` as the tool made it (:func:`_kept_update`), so
    the state update of a call that ran before a pause is not lost with the step run again.
    An error the tool raised is recorded as the call's failure and raised again as it was
    (:func:`_failed`), tried once: a read whose error may pass is tried again around the
    graph's own handling of the error, where the graph did not try it again itself
    (``Tool.deferring``, :func:`govern_tool_nodes`).
    The graph's own handling of the error — LangChain's ``ToolRetryMiddleware`` and
    ``ToolErrorMiddleware``, a wrapper of the builder's, the tool node's ``handle_tool_errors``
    — sees what the library gives it, and the failed message the graph made of it (the tool
    node's, a wrapper's) is the one the model reads, as the graph worded it, unless the run's
    ``after_tool`` hooks changed what the call says: then the model reads theirs (a result, or
    their words in place of the graph's).
    The harness's tools, the graph's state tools (:data:`STATE_TOOLS`, the framework's:
    :func:`framework_made`), a call already governed and any call outside a run are let
    through as they are — a file tool of a backend outside the graph's state
    (:data:`OUTSIDE_STATE`) is governed. A call of the graph's own tools runs however often
    its model step makes it, as the library runs it (a write is still made once: its
    idempotency key); a harness tool's call repeating an earlier one of its step (the same
    tool, the same arguments) does not run, and points the model to the earlier one's result
    (``bridge.same``). The message the call ends with is its tool's, as the library made it
    (:func:`_answered`)."""
    runtime = current()
    call = request.tool_call
    own = request.tool
    if runtime is None or own is None or _GOVERNING.get() == (call["name"], str(call["id"])):
        return await handler(request)
    if call["name"] in STATE_TOOLS and framework_made(own) and scratch(own, call):
        traced = await _scratch_traced(runtime, own, request, handler)
        return _scratch_said(runtime, call["name"], traced, request.state)
    if converted.made(own) or call["name"] in runtime.toolbox:
        # the harness's own: its call goes through the bridge, a repeat of its step's too
        first = _repeats(_step_of(request)[1]).get(str(call["id"]))
        if first is not None:
            repeat = bridge.same(
                runtime, call["name"], call["args"], call_id=str(call["id"]), first=first
            )
            return _message(repeat, request, kept=_paged(runtime, request))
        return _kept_own(runtime, request, await handler(request))
    produced: list[Any] = []
    raised: list[Exception] = []
    #: the failed message the graph made of the tool's error, and what the call raised for it
    reported: list[tuple[ToolMessage, ToolError]] = []

    async def run(args: dict[str, Any]) -> Any:
        tasks = await _remote_tasks(runtime, call, own, args)
        token = _GOVERNING.set((call["name"], str(call["id"])))
        try:
            done = await _helped(runtime, call["name"], str(call["id"]), args, ran)
        finally:
            _GOVERNING.reset(token)
        if isinstance(done, ToolMessage) and done.status == "error":
            failure = ToolError(text_of(done.content), source="tools")
            reported[:] = [(done, failure)]
            raise failure
        if tasks is not None:
            # a task started is a child run; one Deep Agents found ended, ends (``async_tasks``)
            try:
                await tasks.said(runtime, own, args=args, done=done, told=text_of(_told(done)))
            except Exception as exc:  # the task's run is a record: the call's result stands
                log.warning(
                    "run %s: the run of a task of %s not kept: %s", runtime.run_id, own.name, exc
                )
        produced.append(done)
        # a content_and_artifact tool's artifact: the sources of its result, when it names any
        cited = citations(getattr(done, "artifact", None))
        return Sourced(text_of(_told(done)), cited) if cited else text_of(_told(done))

    async def ran(args: dict[str, Any]) -> ToolMessage | Command[Any]:
        raised.clear()
        reported.clear()
        try:
            return await handler(request.override(tool_call={**call, "args": args}))
        except Exception as exc:
            raised.append(exc)
            raise

    spec = runtime.native.get(call["name"]) or spec_of(own)  # as the run's tool set has it
    # Deep Agents' files, a task handed on: the run's own. In a tool node, tried again around
    # the graph's handling
    made = Tool(spec, run, confined=confined(own), deferring=_CALLING.get())
    outcome = await bridge.call(made, dict(call["args"]), call_id=str(call["id"]))
    update = await _update(runtime, call, outcome, produced)
    if update is not None:
        return update  # the graph's state update, as the tool made it
    own_message = await _made(runtime, call, outcome, produced)
    return _answered(runtime, request, outcome, reported, raised, made=own_message)


def _answered(
    runtime: Runtime,
    request: ToolCallRequest,
    outcome: ToolOutcome,
    reported: Sequence[tuple[ToolMessage, ToolError]],
    raised: Sequence[Exception],
    *,
    made: ToolMessage | None = None,
) -> ToolMessage:
    """The message a governed call that gave the graph no update ends with: the failed message
    the graph made of the tool's error (``reported``: the tool node's, a wrapper's), as the
    graph worded it — unless the run's ``after_tool`` hooks changed what the call says — else
    the harness's of the call's outcome; an error the tool raised (``raised``) is raised again
    (:func:`_failed`), for the graph to handle as it does — unless the hooks made the call a
    result: then it is that result. A failure the hooks worded otherwise is raised all the same
    (the graph's retry, its wrappers and its tool node act on it as ever), and the failed
    message the graph makes of it is the hooks' (``_Calling.said``). A result the tool's own
    message carries more of than its text (``made``: :func:`_made`) is that message, as the
    library made it, unless the hooks changed it: its content blocks (an image, a file) as
    they are, and its artifact beside the text, which the harness gives as any result's."""
    calling = _under_way()
    calling.said = None  # this try's: what an earlier one's hooks said is not this one's
    if reported and outcome.output == failed_output(outcome.tool, reported[0][1]):
        return reported[0][0]
    if made is not None and not bridge.hooked(outcome):
        if runtime not in _PAGED or not isinstance(made.content, str):
            return made
        said = _message(outcome, request, kept=_paged(runtime, request)).content
        return made.model_copy(update={"content": said})
    message = _message(outcome, request, kept=_paged(runtime, request))
    if raised and not outcome.ok:
        if bridge.hooked(outcome):
            # a failure the run's after_tool hooks worded: the graph's own handling of the
            # error still runs, and the failed message it makes of it says the hooks' words
            calling.said = message
        _failed(raised[-1], message)
    return message


#: The journal's mark, in a call's record, of what a framework's own tool returned in a
#: message's place (:func:`_kept_update`): a ``Command``, or a list of them and messages.
UPDATE: Final = "trellis_update"
#: LangGraph's own serializer for what a run keeps of its graph — its checkpoint
#: (:class:`RunCheckpointer`) and what a tool returned in a message's place
#: (:func:`_kept_update`) — in LangGraph's strict mode: read back, stored data is made only
#: into the types LangGraph itself lists as safe (its messages, ``Command``, ``Send``,
#: ``Interrupt``...), and anything else is left as the data it holds, never imported or
#: called. A run's journal and checkpoint are written by whoever holds a key of the run
#: store: no trust to construct with. A graph's own state types are added by LangGraph's own
#: means, per graph (``adapters.langgraph.checkpointing``: ``with_allowlist``).
STRICT_SERDE: Final = JsonPlusSerializer(allowed_msgpack_modules=None)


async def _update(
    runtime: Runtime, call: Mapping[str, Any], outcome: ToolOutcome, produced: Sequence[Any]
) -> Any | None:
    """What a governed call that ended well gives the graph in a message's place: what its
    tool returned now (``produced``: its last try's result) when that is no message — kept for
    a replay (:func:`_kept_update`) — or, the call replayed from the journal, what was kept of
    it (:func:`_update_of`); ``None``: its message."""
    if not outcome.ok:
        return None
    if produced and not isinstance(produced[-1], ToolMessage):
        await _kept_update(runtime, call, outcome, produced[-1])
        return produced[-1]
    if outcome.cached and UPDATE in outcome.metadata:
        kept = _kept(runtime, outcome, call)
        return None if isinstance(kept, ToolMessage) else kept  # a message: :func:`_made`
    return None


async def _made(
    runtime: Runtime, call: Mapping[str, Any], outcome: ToolOutcome, produced: Sequence[Any]
) -> ToolMessage | None:
    """The tool's own message for a governed call that ended well, where it carries more than
    the text the run records of it: content blocks (an image, a file: a multimodal result the
    model reads as such) or an artifact (``response_format="content_and_artifact"``: data the
    model does not read, which the graph's state keeps) — the message its tool returned now
    (kept for a replay: :func:`_kept_update`), or, the call replayed from the journal, the one
    kept; else ``None``: the harness's message of the outcome says all of it."""
    if not outcome.ok:
        return None
    if produced and isinstance(done := produced[-1], ToolMessage):
        if done.artifact is not None or not isinstance(done.content, str):
            await _kept_update(runtime, call, outcome, done)  # its text alone is recorded
        return done
    if outcome.cached and UPDATE in outcome.metadata:  # a message (:func:`_update`: no other)
        return _kept(runtime, outcome, call)
    return None


def _kept(runtime: Runtime, outcome: ToolOutcome, call: Mapping[str, Any]) -> Any:
    """What was kept of a call replayed from the journal (:func:`_update_of`), read back as
    the run's graph may be: strictly, its own state types allowed."""
    serde = STRICT_SERDE.with_msgpack_allowlist(state_types(runtime.agent.target))
    return _update_of(outcome.metadata[UPDATE], str(call["id"]), serde)


async def _kept_update(
    runtime: Runtime, call: Mapping[str, Any], outcome: ToolOutcome, made: Any
) -> None:
    """What a framework's own tool returned in a message's place (``made``: a ``Command``
    updating the graph's state, a list of them and messages), kept in the journal with the
    record of its call (``Journal.calls``, the bridge's: the call as the model made it): a
    later attempt that replays the call gives the graph the same update, where the bridge's
    record alone gives the message the model read. Serialized as LangGraph serializes a step's
    writes; an update that cannot be (a value its serializer does not take) is left out, said
    in the log — the call's replay is then its message."""
    recorded = runtime.replay.journal.calls.get(call_key(call["name"], call["args"]))
    if not recorded or bridge.journal_outcome(call["name"], recorded[-1]).output != outcome.output:
        return  # not journaled as this call's (the paused call made again in other words)
    try:
        kind, data = STRICT_SERDE.dumps_typed(made)
    except Exception as exc:
        log.warning(
            "run %s: what %s returned in a message's place cannot be kept for a replay (%s: %s)",
            runtime.run_id,
            call["name"],
            type(exc).__name__,
            exc,
        )
        return
    kept = [kind, base64.b64encode(data).decode(), str(call["id"])]
    recorded[-1] = bridge.journal_record(outcome.model_copy(update={"metadata": {UPDATE: kept}}))
    await runtime.progress(now=True)  # saved with the call it is the result of


def state_types(graph: Any) -> set[tuple[str, ...]]:
    """The types ``graph``'s own state declares (its schemas' and channels': a Pydantic model,
    a dataclass, an enum a state key holds), as LangGraph itself lists them for its strict
    serializer (``build_serde_allowlist``, what a graph compiled under
    ``LANGGRAPH_STRICT_MSGPACK`` is given): what :data:`STRICT_SERDE` may also make of the
    run's stored data for this graph."""
    from langgraph._internal._serde import build_serde_allowlist  # noqa: PLC0415 - LangGraph's own

    schemas: list[type[Any]] = list(getattr(getattr(graph, "builder", None), "schemas", ()))
    return build_serde_allowlist(schemas=schemas, channels=getattr(graph, "channels", None))


def _update_of(kept: Sequence[str], call_id: str, serde: SerializerProtocol) -> Any:
    """What :func:`_kept_update` kept, read back with ``serde`` (:data:`STRICT_SERDE`, told
    the graph's own :func:`state_types`), for the call ``call_id`` now: the messages answering
    the call it was made for answer this one (a model asked again gives its call another
    id)."""
    kind, data, was = kept
    made = serde.loads_typed((kind, base64.b64decode(data)))

    def answering(found: Any) -> None:
        for message in found if isinstance(found, list) else ():
            if isinstance(message, ToolMessage) and message.tool_call_id == was:
                message.tool_call_id = call_id

    for item in made if isinstance(made, list) else [made]:
        answering([item])
        update = getattr(item, "update", None)
        answering(update.get("messages") if isinstance(update, dict) else update)
    return made


@dataclass(slots=True, eq=False)
class _Calling:
    """One call of a tool node, from its first wrapper to its tool: the latest governed try
    of it that failed — the error its tool raised, and the message the model reads for it
    where the harness's rule holds; whether the harness's middleware governs it (``owned``:
    ``GovernTools``, which ``ReAct`` and ``middleware()`` have); whether it is ``waiting`` (it
    paused, and waits for the node's other calls) or has ``ended``."""

    failed: tuple[BaseException, ToolMessage] | None = None
    #: the message the run's after_tool hooks made of its latest governed try, which failed:
    #: the failed message the graph makes of that error is this one (:func:`govern_tool_nodes`)
    said: ToolMessage | None = None
    #: the error of its latest try that the harness would try it again after, how many times,
    #: and what ends that try on the run's stream and records it, when no try follows
    #: (``tools.base.Deferring``); how often the graph ran the call in that try — a second time
    #: is the graph trying it again itself (a wrapper calling its handler again); and the
    #: harness's tries of it before this one
    owed: tuple[Exception, int, Callable[[], Awaitable[None]]] | None = None
    entries: int = 0
    tried: int = 0
    owned: bool = False
    waiting: bool = False
    ended: asyncio.Event = field(default_factory=asyncio.Event)

    async def owe(
        self, error: Exception, times: int, finish: Callable[[], Awaitable[None]]
    ) -> None:
        await self.ended_try()  # the graph's own try before this one: a call of its own
        self.owed = (error, times, finish)

    async def ended_try(self) -> None:
        """The latest failed try that was to be tried again, ended and recorded as it was:
        none follows it (the graph tried the call again itself, or the harness will not)."""
        owed, self.owed = self.owed, None
        if owed is not None:
            await owed[2]()

    def trying(self) -> None:
        """A try of the call begins: what an earlier one left is not this one's (a failed
        try the harness tries again is an attempt of the call, not ended on its own)."""
        self.failed, self.said, self.owed, self.entries = None, None, None, 0


#: The tool node's call under way (:func:`govern_tool_nodes`), where its tries meet.
_CALLING: ContextVar[_Calling | None] = ContextVar("trellis_calling", default=None)


def _step_key(state: Any, call_id: str) -> Hashable:
    """The model step a tool call is of, as the calls made beside it share it — the model
    message that made it (its id, else its calls' ids), in the run — whichever task LangGraph
    runs each call in and whatever state object each is handed; the state object itself where
    its messages do not hold the call."""
    runtime = current()
    messages = state.get("messages", []) if isinstance(state, dict) else []
    found = _calling(messages, call_id)
    if runtime is None or found is None:
        return ("state", id(state))
    made = found[1]
    return (runtime.run_id, made.id or tuple(str(c["id"]) for c in made.tool_calls))


async def _others_ended(calling: _Calling, beside: Collection[_Calling]) -> None:
    """A tool node's call paused (it asks a person): it leaves the node once the other calls
    under way of its model step have ended (``beside``: :func:`_step_key`), or wait as it does.
    A ``ToolNode`` runs its step's calls together and stops at the first that raises, and
    ``create_agent`` runs each call as a task of its own (a ``Send``), which LangGraph cancels
    when another task of the step raises: either way a write beside the call would end after
    the run's pause was recorded, or never, and the resumed run, finding it started and never
    ended, could only tell the model its effect is unknown. LangGraph itself waits so for the
    other tasks of a step in which one interrupts."""
    calling.waiting = True
    for other in [c for c in beside if c is not calling and not c.waiting]:
        await other.ended.wait()


def _under_way() -> _Calling:
    """The tool node's call under way — or, where a wrapper of the graph's ran the call
    outside the node's context (a thread, a context of its own), one nothing else reads."""
    return _CALLING.get() or _Calling()


def _failed(error: Exception, message: ToolMessage) -> NoReturn:
    """A governed call's tool raised ``error`` (recorded: the call failed): raised again as it
    was, so what the graph does with a tool's error is the library's — a wrapper that retries,
    rewords or answers in its place gets the error, the tool node's ``handle_tool_errors``
    decides, as without the harness. An error nothing of the graph handles fails the graph, as
    LangGraph has it — unless the harness's middleware governs the call: it ends as
    ``message``, which the model reads (:func:`govern_tool_nodes`)."""
    _under_way().failed = (error, message)
    raise error


def _message(outcome: ToolOutcome, request: ToolCallRequest, *, kept: bool = False) -> ToolMessage:
    """A harness call's outcome as the tool message the model reads — ``kept``: a large one
    kept in the run and previewed, ``read_file`` paging it (``tools.results``); where the
    harness runs the loop (:data:`_PAGED`), a table of many rows read whole is kept too, for
    ``query_result`` (``results.tabled``) — its table named by what the call asked
    (:func:`_table`). Elsewhere the result's text, as it is."""
    call = request.tool_call
    text = text_of(outcome.output)
    if kept:
        content = results.readable(outcome.output, _table(_step_of(request)[1], call))
    elif current() in _PAGED:
        content = results.tabled(text, _table(_step_of(request)[1], call))
    else:
        content = text
    return ToolMessage(
        content=content,
        tool_call_id=str(call["id"]),
        name=call["name"],
        status="error" if outcome.status in FAILED else "success",
    )


#: The most characters of a call's arguments a kept result's table is named with.
TABLE_SAID: Final = 40


def _table(reply: AIMessage, call: Mapping[str, Any]) -> str:
    """What a kept result of ``call`` is named as a table (``tools.query.tables``): its tool —
    and, where the model's ``reply`` calls that tool more than once, the values of the
    arguments that differ between those calls (``sales_summary_revenue``,
    ``sales_summary_units``). Calls made side by side answer in any order: numbered by which
    answered first, the model took ``sales_summary_2`` for its second call and joined revenue
    as units. Values that are not plain (a list, an object), or longer than :data:`TABLE_SAID`
    together, leave the tool's name: numbered as the results arrive."""
    name, args = str(call["name"]), call.get("args") or {}
    others = [
        c.get("args") or {} for c in reply.tool_calls if c["name"] == name and c["id"] != call["id"]
    ]
    apart = [
        str(args[key])
        for key in sorted(args)
        if isinstance(args[key], str | int | float)
        and any(other.get(key) != args[key] for other in others)
    ]
    said = "_".join(apart)
    return f"{name}_{said}" if said and len(said) <= TABLE_SAID else name


def _paged(runtime: Runtime, request: ToolCallRequest) -> bool:
    """Whether a large result of this call is kept in the run and previewed, ``read_file``
    paging it (``tools.results``, the convention every adapter's model reads): in a graph that
    is offered the harness's tools per model call (:class:`HarnessTools`, which offers and
    serves that ``read_file``) and has no ``read_file`` of its own keeping files in the
    graph's state — there Deep Agents saves a large result itself, where the harness reads it
    (:func:`_graph_results`). On a backend outside the state (a sandbox, a disk, a store:
    :data:`OUTSIDE_STATE`) the run keeps it, as without a backend: the harness cannot read a
    file there, so no table was named and ``query_result`` was never offered; a sandbox gets
    the kept result as a file for code all the same (``SandboxSource.with_results``)."""
    if runtime not in _PAGED:
        return False
    tools = getattr(request.runtime, "tools", None) or ()
    return not any(
        getattr(t, "name", None) == READ_FILE
        and not (marks := getattr(t, "metadata", None) or {}).get(KEPT_BY_RUN)
        and not marks.get(OUTSIDE_STATE)
        for t in tools
    )


def _asked_at(messages: Sequence[AnyMessage]) -> int:
    """Where the run's question is in ``messages``: its last user message not steered into it
    (a message steered in joins the run under way: what the run fetched before it is this
    turn's, not an earlier one's); ``-1``: none."""
    return max(
        (n for n, m in enumerate(messages) if typed(m) and STEERED not in m.response_metadata),
        default=-1,
    )


def _earlier(runtime: Runtime, messages: Sequence[AnyMessage]) -> tuple[list[AnyMessage], set[str]]:
    """``messages`` (a thread the framework keeps, as the model is sent it) with each result
    from before the run's question (:func:`_asked_at`) of a live tool of the run (``max_age``
    0: every earlier value is out of date) marked ``snapshots.EARLIER``, and those tools. A result
    already stamped (a memory window's replay), or of a tool whose value stays current a while
    (its age is not known here) or for good, is left as it is; and so is the result of a call
    that failed or was stopped before it came back: it returned no value to be out of date."""
    asked = _asked_at(messages)
    called = {
        c["id"]: c["name"]
        for m in messages[:asked]
        if isinstance(m, AIMessage)
        for c in m.tool_calls
    }
    out = list(messages)
    stale: set[str] = set()
    for n, m in enumerate(messages[:asked]):
        if not isinstance(m, ToolMessage) or not isinstance(m.content, str) or m.status == "error":
            continue
        tool = called.get(m.tool_call_id) or m.name or ""
        found = runtime.toolbox.get(tool)
        spec = found.spec if found is not None else runtime.native.get(tool)
        marked = m.content.startswith(snapshots.STALE.format(tool=tool))  # memory's replay
        if spec is None or marked or max_age(spec) != 0:
            continue
        stale.add(tool)
        # the request's copy is marked, never the graph's state: each call marks it anew
        mark = snapshots.EARLIER.format(tool=tool)
        out[n] = m.model_copy(update={"content": f"{mark}\n{m.content}"})
    return out, stale


def _with_readers(runtime: Runtime, request: ModelRequest[Any]) -> ModelRequest[Any]:
    """The request offering the harness's readers too, once a result is held (:func:`_readers`)
    — as schemas: ``create_agent`` refuses a tool it was not built with, and the governed tool
    node answers a reader's call before it looks for the tool (:func:`_kept_read`)."""
    present = {_named(t) for t in request.tools}
    held = [convert_to_openai_tool(t) for t in _readers(runtime, request.state, present)]
    return request.override(tools=[*request.tools, *held]) if held else request


def _stale_marked(
    runtime: Runtime, request: ModelRequest[Any]
) -> tuple[ModelRequest[Any], set[str]]:
    """The request with each earlier turn's live result marked out of date (:func:`_earlier`)
    and its latest question saying which tools (:func:`_stale_noted`), and those tools."""
    messages, stale = _earlier(runtime, request.messages)
    return (request.override(messages=_stale_noted(messages, stale)) if stale else request), stale


def _stale_noted(messages: list[AnyMessage], stale: Collection[str]) -> list[AnyMessage]:
    """``messages`` with the latest user message saying which tools' earlier results are out of
    date (``snapshots.note``, as a memory window's replay says it in the run's context), after
    its text: the request's copy, the earlier messages untouched (one cached prefix)."""
    asked = _asked_at(messages)
    note = snapshots.note(sorted(stale))
    if asked < 0 or note is None:
        return messages
    said = messages[asked]
    content = said.content
    more: str | list[Any] = (
        [*content, {"type": "text", "text": note}]
        if isinstance(content, list)
        else f"{content}\n\n{note}"
    )
    return [*messages[:asked], said.model_copy(update={"content": more}), *messages[asked + 1 :]]


async def _kept_read(runtime: Runtime, request: ToolCallRequest) -> ToolMessage | None:
    """A call reading a result the run keeps, answered: ``read_file`` with its page (the graph
    has no ``read_file`` of its own, or the path is one the run keeps: a result cut to fit the
    window, in a Deep Agents graph); ``query_result`` with what it computes (``tools.query``)
    — over a result the run keeps, or one Deep Agents saved in the graph's state. Any other
    call: ``None``."""
    call = request.tool_call
    args = dict(call["args"])
    path = "/" + str(args.get("file_path", "")).lstrip("/")
    named = _stub_names(request.state, path)
    if named is not None:  # the call's result is a stub: the file it names is the result
        path = args["file_path"] = named
    # a result the run can read (its own, its conversation's earlier turns'), or a file kept
    # for its user (``tools.stored``): the harness's
    kept = path in query.saved(runtime) or stored.is_stored(path)
    said: str | None = None
    cleared = None if kept else _cleared_result(request.state, path)
    if call["name"] == READ_FILE and cleared is not None:
        said = results.read(args, text=cleared)
    elif call["name"] == READ_FILE and (request.tool is None or kept):
        said = await results.aread(args)
    elif call["name"] == query.QUERY_RESULT and (request.tool is None or kept):
        with query.graph_results(*_graph_results(request.state, runtime)):
            # a file Deep Agents saved in the state, else a result still in its messages —
            # either as the table the run already keeps of it (:func:`_kept_copy`)
            given = None if kept else _state_file(request.state, path, runtime)
            said = await query.query(args, given)
    if said is None:
        return None  # another tool, or the graph's own read_file of its own files
    return ToolMessage(content=said, tool_call_id=str(call["id"]), name=call["name"])


#: Where a stub says its result was saved (``results.STUB``'s wording, Deep Agents' own).
_STUB_AT: Final = re.compile(
    re.escape(results.STUB.split("{file_path}")[0].rsplit("{tool_call_id}", 1)[1]) + r"(\S+)"
)


def _result_of(state: Any, path: str) -> str | None:
    """The text of the tool result a call's path names (``/large_tool_results/<call id>``: what
    a cleared result's mark gives, and Deep Agents' stubs), as the graph's state keeps it — the
    latest of that call's id; ``None`` for any other path."""
    prefix = results.PREFIX + "/"
    if not path.startswith(prefix) or not isinstance(state, Mapping):
        return None
    ident = path[len(prefix) :]
    for message in reversed(state.get("messages") or ()):
        if isinstance(message, ToolMessage) and message.tool_call_id == ident:
            return message.text
    return None


def _stub_names(state: Any, path: str) -> str | None:
    """The file a call's result was saved at, when its result in the graph's state is a stub
    naming another path than the call's own (the run keeps a large result under a path of its
    own): a model given the call's path — by a cleared result's mark — reads or queries the
    result itself. ``None`` otherwise."""
    text = _result_of(state, path)
    found = _STUB_AT.search(text) if text is not None and text.startswith(results.EVICTED) else None
    return found.group(1) if found is not None and found.group(1) != path else None


def _cleared_result(state: Any, path: str) -> str | None:
    """The whole text of a tool result the request no longer shows (LangChain's
    ``ClearToolUsesEdit`` cleared it: the graph's state keeps it), named by Deep Agents' path
    for its call (``/large_tool_results/<call id>``) — ``None`` for any other path, and for a
    result Deep Agents evicted to its backend (its own ``read_file`` reads that file)."""
    text = _result_of(state, path)
    gone = text is None or text.startswith(results.EVICTED) or _marked(text)
    return None if gone else text


def _state_file(state: Any, path: str, runtime: Runtime | None = None) -> str | None:
    """The text of a file Deep Agents keeps in the graph's state (a large result it saved),
    else of a result still in the graph's messages, by its call's path (:func:`_cleared_result`:
    what a cleared result's mark names), or ``None`` — a copy of a result the run keeps
    (:func:`_kept_copy`: the message of a table read whole carries its note after the rows) as
    that result."""
    found = (state.get("files") if isinstance(state, Mapping) else None) or {}
    content = (found.get(path) or {}).get("content")
    text = "\n".join(content) if isinstance(content, list) else content
    if text is None or _marked(text):  # a mark saved as a file is no result
        text = _cleared_result(state, path)
    if text is None or runtime is None:
        return text
    copied = _kept_copy(runtime, text)
    return text if copied is None else runtime.replay.journal.results[copied]


def _kept_copy(runtime: Runtime, text: str) -> str | None:
    """The path of the result the run keeps that ``text`` — a file Deep Agents saved — is a
    copy of: a table read whole is kept and sent with its note after it (``results.tabled``),
    and Deep Agents saves that whole message when it is too large for the window. The file
    then starts with the kept result, and what follows is the note: no table of its own (read
    as one, a JSON result was "CSV" of thousands of columns)."""
    kept: Mapping[str, str] = runtime.replay.journal.results
    return next((p for p, k in kept.items() if _copy_of(text, k)), None)


#: How the note a table read whole carries starts, after its rows (``results.tabled``).
_NOTED: Final = "\n\n" + results.TABLED.split("{", 1)[0]


def _copy_of(text: str, kept: str) -> bool:
    """Whether ``text`` is the ``kept`` result as its message carries it: its rows, then the
    note. A result that only begins with another's rows (a longer page, a later date range)
    is itself: taken for a copy, a count over its 20 rows answered the other's 10."""
    return len(kept) < len(text) and text.startswith(kept) and text.startswith(_NOTED, len(kept))


def _shown(text: str, path: str) -> tuple[str, str | None]:
    """The result a tool message's whole ``text`` shows (``path``: its call's), as the run
    holds it — where it is read, and its rows: the result the run keeps, at its own path,
    when the text is a copy of one (its rows, then the note: :func:`_copy_of`) or a stub
    naming one; the text itself at the call's path, when it is neither; no rows for a stub of
    a file the run does not keep (Deep Agents' own). A kept result's own path stays readable
    whatever the graph does with its messages: a summarization writes the cleared request
    back as the state, and the call's path then led to the mark."""
    runtime = current()
    kept: Mapping[str, str] = runtime.replay.journal.results if runtime is not None else {}
    if text.startswith(results.EVICTED):
        return next(((p, kept[p]) for p in _STUB_AT.findall(text) if p in kept), (path, None))
    return next(((p, k) for p, k in kept.items() if _copy_of(text, k)), (path, text))


def _marked(text: str) -> bool:
    """Whether ``text`` is a cleared result's mark: no result, wherever it was written (Deep
    Agents saves a request's large messages as files, and a summarization writes the cleared
    request back as the graph's state: a query of that path read the mark's one line)."""
    from trellis.harness.react import CLEARED  # noqa: PLC0415 - it builds this module's stack

    return text.startswith(CLEARED.split("{path}", 1)[0])


def _graph_results(
    state: Any, runtime: Runtime | None = None
) -> tuple[dict[str, str], dict[str, str]]:
    """The results Deep Agents saved in the graph's state (``/large_tool_results/<call id>``),
    by path, and the tool whose call returned each (named in the state's messages) — less the
    copies of results ``runtime``'s run keeps itself (:func:`_kept_copy`): tables already."""
    held: Mapping[str, Any] = state if isinstance(state, Mapping) else {}
    prefix = results.PREFIX + "/"
    files = {
        str(p): text
        for p in held.get("files") or {}
        if str(p).startswith(prefix)
        and (text := _state_file(held, str(p))) is not None
        and (runtime is None or _kept_copy(runtime, text) is None)
    }
    calls = {
        str(c["id"]): _table(m, c)
        for m in held.get("messages") or ()
        if isinstance(m, AIMessage)
        for c in m.tool_calls
    }
    tools = {p: calls[p[len(prefix) :]] for p in files if p[len(prefix) :] in calls}
    return files, tools


def _saved(state: Any) -> bool:
    """Whether the run keeps a result, or the graph's state holds one Deep Agents saved."""
    files = (state.get("files") if isinstance(state, Mapping) else None) or {}
    return results.kept() or any(str(p).startswith(results.PREFIX) for p in files)


def _kept_own(
    runtime: Runtime, request: ToolCallRequest, done: ToolMessage | Command[Any]
) -> ToolMessage | Command[Any]:
    """A harness tool's result in a graph that was built with it and is offered the harness's
    readers (:func:`_narrow`): a large one kept and previewed, a table read whole kept too, as
    ``HarnessTools`` keeps them (:func:`_message`); elsewhere as it is."""
    if runtime not in _PAGED or not isinstance(done, ToolMessage) or done.status == "error":
        return done
    call, said = request.tool_call, text_of(done.content)
    name = _table(_step_of(request)[1], call)
    kept = results.readable(said, name) if _paged(runtime, request) else results.tabled(said, name)
    return done.model_copy(update={"content": kept})


def _readers(
    runtime: Runtime | None,
    state: Any,
    present: Collection[str | None],
    *,
    held: bool | None = None,
) -> list[BaseTool]:
    """The harness's readers of a held result (``read_file``; ``query_result`` too, saving its
    result as a file where the run runs no code: :data:`SAVING`) the model is offered beside
    ``present``, once a result is held anywhere: the run keeps one, the graph's state holds
    one Deep Agents saved (:func:`_saved`), or the caller knows one is held — ``held``: the
    request shows one cleared from the window (:class:`ReadTools`, after the clearing), or one
    is being cut to fit (:func:`_cut`); ``True`` when it holds rows, ``False`` when it is
    text only, which ``read_file`` alone reads (a query has nothing to compute over). Their
    calls are answered by the harness (:func:`_kept_read`), from wherever the result is."""
    if held is None and not _saved(state):
        return []
    readers = _OFFERED[id(results.readers())]  # the run's: saving, with a sandbox, or plain
    only = {READ_FILE} if held is False and not _saved(state) else set(readers)
    return [t for name, t in readers.items() if name in only and name not in present]


def _reader(name: str, description: str, schema: dict[str, Any], answer: Any) -> BaseTool:
    """A tool reading a kept result (``tools.results.READERS``) as a LangChain tool, offered
    once there is one, its calls answered by :class:`HarnessTools` (no tool node runs it)."""

    async def run(**args: Any) -> str:
        return str(await answer(args))

    return StructuredTool.from_function(
        coroutine=run, name=name, description=description, args_schema=schema
    )


#: The harness's ``read_file`` and ``query_result`` as LangChain tools; and as a run that saves a
#: query's result as a file is offered them (``results.SAVING``), made once each.
READERS: Final = {r[0]: _reader(*r) for r in results.READERS}
READ_FILE_TOOL: Final = READERS[READ_FILE]
SAVING: Final = {**READERS, query.QUERY_RESULT: _reader(*results.SAVING[1])}
#: And as a run with a sandbox is offered them (``results.CODING``).
CODING: Final = {**READERS, query.QUERY_RESULT: _reader(*results.CODING[1])}
#: Each set of ``results.readers()`` as LangChain tools.
_OFFERED: Final = {
    id(results.READERS): READERS,
    id(results.SAVING): SAVING,
    id(results.CODING): CODING,
}
#: The runs a :class:`HarnessTools` offers the harness's tools to (:func:`_paged`).
_PAGED: Final[weakref.WeakSet[Runtime]] = weakref.WeakSet()


#: What a run whose results the harness keeps (:data:`_PAGED`) is told before any result is
#: there: ``read_file`` and ``query_result`` are offered only once one is kept
#: (``results.kept``), and a model told nothing of them declined a question over more rows
#: than it thought it could read, never calling the tool that returns them. A fact, as each of
#: Claude Code's tools states its own limits: worded as an order ("never decline"), it also
#: fetched every match of an ambiguous request instead of asking which one.
KEPT_SAID: Final = (
    "A tool result too large to read is kept whole: query_result ranks, sums or counts all "
    "its rows exactly."
)


def _kept_said(request: ModelRequest[Any]) -> ModelRequest[Any]:
    """The request with :data:`KEPT_SAID` after its instructions, said once, where it offers
    a tool: the same line for every call, so the instructions stay one cached prefix."""
    system = request.system_message
    if not request.tools or (system is not None and KEPT_SAID in system.text):
        return request
    given = system.content if system is not None else ""
    content: str | list[str | dict[Any, Any]] = (
        [*given, {"type": "text", "text": KEPT_SAID}]
        if isinstance(given, list)
        else "\n\n".join(part for part in (given, KEPT_SAID) if part)
    )
    return request.override(system_message=SystemMessage(content=content))


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


async def _scratch_traced(
    runtime: Runtime,
    own: BaseTool,
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
) -> ToolMessage | Command[Any]:
    """A state tool's call (the graph's own state: not governed), still a step of the run:
    traced as every call is, its governance action ``run``."""
    call = request.tool_call
    spec = runtime.native.get(call["name"]) or spec_of(own)
    with tool_span(
        call["name"], str(call["id"]), call["args"], source=spec.source, action="run"
    ) as span:
        done = await handler(request)
        told = text_of(_told(done))
        span_output(span, told, key="gen_ai.tool.call.result")
        if _errored(done):
            failed(span, ToolOutcome(tool=call["name"], status=ToolStatus.ERROR, output=told))
    return done


def _errored(done: Any) -> bool:
    """Whether a tool's result is a failed message: its own, or a ``Command``'s last one."""
    if isinstance(done, Command):
        update = done.update if isinstance(done.update, Mapping) else {}
        done = (update.get("messages") or [None])[-1]
    return isinstance(done, ToolMessage) and done.status == "error"


def _told(done: Any) -> Any:
    """What a tool's result says to the model: a message's content, a ``Command``'s last
    message's (its state update's), or — a list of them — the last that says anything."""
    if isinstance(done, list):
        said = [told for told in map(_told, done) if told != ""]
        return said[-1] if said else ""
    if isinstance(done, Command):
        update = done.update if isinstance(done.update, Mapping) else {}
        messages = update.get("messages") or []
        return getattr(messages[-1], "content", messages[-1]) if messages else ""
    return getattr(done, "content", done)


def govern_tool_nodes() -> None:
    """Every LangGraph tool node, inside a harness run, governed (:func:`governed_call`): a
    ``create_agent`` or Deep Agents graph's (its ``task`` sub-agents' graphs too), a hand-built
    graph's ``ToolNode`` — whoever built them, with no middleware of the harness's. A call
    reading a result the run keeps is answered first (:func:`_kept_read`), the harness's
    readers among them where the graph was not built with them: ``ToolNode`` defers checking
    a call's tool to this execution so that a wrapper may answer a tool it lacks. Installed
    once per process, on the tool node's one execution of a call (after its own wrappers);
    outside a harness run it changes nothing.

    The node's one call (before its wrappers) is installed on too: the error of a governed
    call's tool that nothing of the graph handled — the node's own ``handle_tool_errors``, a
    wrapper — fails the graph there as LangGraph has it, or, where the harness's middleware
    governs the call (``GovernTools``: ``ReAct``, ``middleware()``), ends as the failed result
    the model reads (:func:`_failed`); and a call that pauses (it asks a person) leaves the
    node once the node's other calls have ended (:func:`_others_ended`)."""
    execute = ToolNode._execute_tool_async
    if getattr(execute, _INSTALLED, False):
        return

    async def governed(
        self: ToolNode, request: ToolCallRequest, input_type: Any, config: RunnableConfig
    ) -> Any:
        runtime = current()
        if runtime is None:
            return await execute(self, request, input_type, config)
        # a reader of a kept result the graph was not built with (:func:`_narrow` offers it):
        # the tool node validates a call only here, so a tool it lacks can be answered
        page = await _kept_read(runtime, request) if runtime in _PAGED else None
        if page is not None:
            return page

        async def handler(given: ToolCallRequest) -> Any:
            return await execute(self, given, input_type, config)

        calling = _under_way()
        calling.entries += 1  # the graph running the call: again, its own retry
        if calling.entries > 1:
            await calling.ended_try()  # its earlier try a call of its own, ended first
        return await governed_call(request, handler)

    run_one = ToolNode._arun_one

    async def called(self: ToolNode, call: Any, input_type: Any, tool_runtime: Any) -> Any:
        runtime = current()
        if runtime is None:
            return await run_one(self, call, input_type, tool_runtime)
        calling = _Calling()
        token = _CALLING.set(calling)
        # the calls of one model step, gathered by one tool node or each a task of its own
        step = _step_key(tool_runtime.state, str(call["id"]))
        beside = runtime.beside.setdefault(step, set())
        beside.add(calling)
        try:
            while True:
                calling.trying()
                done: Any = None
                error: Exception | None = None
                try:
                    done = await _tried(
                        calling, beside, run_one(self, call, input_type, tool_runtime)
                    )
                except Exception as exc:
                    error = exc
                if not await _again(calling, error, tool_runtime.config):
                    await calling.ended_try()
                    if error is not None:
                        raise error
                    await _refused_by_graph(runtime, self, call, calling, done)
                    return done
                calling.tried += 1
        finally:
            _CALLING.reset(token)
            calling.ended.set()
            beside.discard(calling)
            if not beside:
                runtime.beside.pop(step, None)

    setattr(governed, _INSTALLED, True)
    ToolNode._execute_tool_async = governed  # type: ignore[method-assign]
    ToolNode._arun_one = called  # type: ignore[method-assign]


async def _refused_by_graph(
    runtime: Runtime, node: ToolNode, call: Mapping[str, Any], calling: _Calling, done: Any
) -> None:
    """A call of the graph's own tool that a wrapper of the graph refused in the tool's place
    — LangChain's ``AutoModeMiddleware`` blocking a call its decision model finds risky, a
    per-call approval rejected: the tool never ran and the call never reached the harness
    (``_Calling.entries``), so it would be on no record of the run. It is recorded as the
    failed call the model reads: on the run's stream, traced and in the memory service
    (``bridge.observed``; nothing is run, journaled or counted against the budget). A call
    the wrapper answered well (a cache, an emulator) is the wrapper's answer, not a refusal;
    the harness's own tools are the bridge's. A call a person rejected at the library's own
    approval gate is their decision, recorded as such, never as a failed call
    (``Journal.rejected_by_person``: by what the call is)."""
    name = str(call["name"])
    tool = node.tools_by_name.get(name)
    refused = isinstance(done, ToolMessage) and done.status == "error" and not calling.entries
    if not refused or tool is None or converted.made(tool) or name in runtime.toolbox:
        return
    if runtime.replay.journal.rejected_by_person(name, dict(call["args"])):
        return
    outcome = ToolOutcome(tool=name, status=ToolStatus.ERROR, output=text_of(done.content))
    spec = runtime.native.get(name) or spec_of(tool)
    await bridge.observed(
        runtime, spec, dict(call["args"]), outcome, ref=str(call["id"]), started=time.time()
    )


async def _tried(calling: _Calling, beside: Collection[_Calling], running: Awaitable[Any]) -> Any:
    """One try of a tool node's call (``running``: the node's own, its wrappers and all), as
    the graph ends it — or a pause, once the step's other calls have ended
    (:func:`_others_ended`); an error nothing of the graph handled, as the harness's
    middleware's rule has it where it governs the call (:func:`_failed`); a failed message the
    graph made of an error the run's hooks worded, theirs."""
    try:
        done = await running
    except Exception as exc:
        if is_pause_signal(exc):
            await _others_ended(calling, beside)
        if not calling.owned or calling.failed is None or calling.failed[0] is not exc:
            raise
        return calling.failed[1]
    said = calling.said
    failed = isinstance(done, ToolMessage) and done.status == "error"
    return said if said is not None and failed else done


async def _again(calling: _Calling, error: Exception | None, config: RunnableConfig) -> bool:
    """Whether a tool node's call is tried again (after a wait, when so), as the harness tries
    any read whose error may pass (``tools.base.retried``): its try failed with such an error
    (``_Calling.owed``), it has tries again left, and the graph did not — and will not — try it
    again itself: a wrapper of the graph's ran it again within the try (``ToolRetryMiddleware``
    calling its handler again), or the error leaves the node for the node's own
    ``RetryPolicy`` to run it again (:func:`graph_retries`). A wrapper that only handles the
    error (``ToolErrorMiddleware``, Deep Agents' file middleware, ``GovernTools``) leaves the
    harness's tries standing: each try runs through it, as the graph runs a call."""
    owed = calling.owed
    if owed is None or calling.tried >= owed[1] or calling.entries > 1:
        return False
    if error is not None and graph_retries(config, error):
        return False
    await asyncio.sleep(backoff(calling.tried + 1, owed[0]))
    return True


#: The prefix of the keys LangGraph gives the config of a node it runs.
PREGEL_KEYS: Final = "__pregel"


def graph_retries(config: RunnableConfig, error: Exception) -> bool:
    """Whether the graph runs the node again on ``error``: a retry policy of its task's (else
    the graph's) that retries it, as LangGraph reads them (its task, from the call it hands the
    node's config: ``CONFIG_KEY_CALL``). These are LangGraph's internals: where they are not
    as the harness reads them (:func:`_internals_moved`) it says so — taken for "no policy",
    the harness would try the call again on top of the graph's own tries. A tool node no
    graph runs (its config has none of LangGraph's keys) has no node policy."""
    try:
        from langgraph._internal._constants import CONFIG_KEY_CALL  # noqa: PLC0415
        from langgraph.pregel._retry import _should_retry_on  # noqa: PLC0415
    except ImportError as exc:
        raise _internals_moved(str(exc)) from exc
    configurable: Mapping[str, Any] = config.get("configurable") or {}
    call = configurable.get(CONFIG_KEY_CALL)
    if call is None and not any(key.startswith(PREGEL_KEYS) for key in configurable):
        return False
    referred = next(iter(getattr(call, "args", ())), None)  # a weak reference to the task
    task: Any = referred() if isinstance(referred, weakref.ref) else None
    graphs = getattr(call, "keywords", None)
    if task is None or not hasattr(task, "retry_policy") or not isinstance(graphs, Mapping):
        raise _internals_moved(f"the call a node's config holds is {type(call).__name__}")
    policies = task.retry_policy or graphs.get("retry_policy") or ()
    return any(_should_retry_on(policy, error) for policy in policies)


def _internals_moved(found: str) -> ConfigurationError:
    """The error of a LangGraph whose internals are not where the harness reads whether the
    graph retries a node itself: named, with the releases the harness is tested with."""
    low, high = compat.TESTED["langgraph"]
    return ConfigurationError(
        "this langgraph does not hand a tool node its task as the harness reads it "
        f"({found}), so whether the graph tries a failed call again itself cannot be told and "
        "the harness will not try it again on top: use a langgraph release trellis-harness is "
        f"tested with (>={low},<{high}: docs/versioning.md)"
    )


#: The mark of the tool node's execution :func:`govern_tool_nodes` installed.
_INSTALLED: Final = "_trellis_governed"


def side_effects_of(metadata: Mapping[str, Any] | None) -> str:
    """A framework tool's risk as it declares it: ``side_effects`` (``read``, ``write``,
    ``irreversible``), else MCP-style annotations — ``annotations``, or as ``langchain.mcp``
    keeps a server's (``mcp.tool.annotations``) — read as an MCP tool's
    (``tools.mcp.side_effects_of``), else none (``UNDECLARED``: a person approves its calls —
    and may allow it for the conversation or always — until the tool, or the catalog, says what
    it does). A ``langchain.mcp`` tool whose server gave no annotations says nothing too."""
    found = dict(metadata or {})
    declared = found.get("side_effects")
    if declared in get_args(SideEffects):
        return str(declared)
    hints = found.get("annotations")
    if hints is None:  # ``langchain.mcp``'s place for them: the server's tool, under ``mcp``
        listed = (found.get("mcp") or {}).get("tool") or {}
        hints = listed.get("annotations")
    return UNDECLARED if hints is None else annotated(hints)


def spec_of(tool: BaseTool) -> ToolSpec:
    """A framework's own LangChain tool as the harness governs, records and publishes it: its
    name, description and schema, its risk as it declares it (:func:`side_effects_of`) — a
    hand-over to another agent of the graph a helper's (:func:`_hands_off`)."""
    spec = ToolSpec(
        name=tool.name,
        description=tool.description or tool.name,
        input_schema=_schema_of(tool),
        side_effects=side_effects_of(tool.metadata),
        **data_of(tool.metadata, name=tool.name),
    )
    return helper(spec) if _hands_off(tool) else spec


def _hands_off(tool: BaseTool) -> bool:
    """Whether ``tool`` is a langgraph-swarm handoff (``create_handoff_tool``: its own mark in
    the tool's metadata): the call only passes the conversation to another agent of the graph,
    whose own calls are governed — a hand-over, as Deep Agents' ``task`` is
    (``adapters.base.helper``). Only a process that built one has the library loaded."""
    swarm = sys.modules.get("langgraph_swarm.handoff")
    return swarm is not None and swarm.METADATA_KEY_HANDOFF_DESTINATION in (tool.metadata or {})


def _schema_of(tool: Any) -> dict[str, Any]:
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
        # a thread the framework keeps: an earlier turn's live value is out of date, and its
        # tool stays offered (the memory window's replay is stamped where it is read)
        request, stale = _stale_marked(runtime, request)
        if stale and runtime.offered is not None:
            runtime.offered |= stale
        request = _stubs_tabled(runtime, request)
        own = [t for t in request.tools if _named(t) not in runtime.toolbox]
        offered = [t for name, t in harness.items() if runtime.offers(name)]
        offered.extend(_readers(runtime, request.state, {_named(t) for t in own}))
        tools = sorted([*own, *offered], key=_named)
        return await handler(_kept_said(request.override(tools=tools)))

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
            return _cut_said(request, page)
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
        return _cut_said(request, _message(outcome, request, kept=_paged(runtime, request)))

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


def _cut_said(request: ToolCallRequest, done: ToolMessage) -> ToolMessage:
    """``done``, saying its call was cut where it was refused for its arguments and the reply
    that made the call reached the model's output limit (``accounting.at_output_limit``): the
    arguments stop where the reply did (``tools.base.CUT_SHORT``). Any other answer as it is.
    The reply was a model call either way, and counts as a step: a limit that did not count
    it would never end a run whose every reply is cut."""
    name = str(request.tool_call["name"])
    cut = cut_short(name, done.text) if accounting.at_output_limit(_step_of(request)[1]) else None
    return done if cut is None else done.model_copy(update={"content": cut})


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
    now = _said_now(runtime, request.tools)
    kept = [now.get(_named(t), t) for t in request.tools if _offers(runtime, t)]
    unchanged = len(kept) == len(request.tools) and not now
    return request if unchanged else request.override(tools=kept)


#: The features of the harness's own tools whose words are the run's (:func:`_said_now`).
BUILT_IN: Final = frozenset({"schedules", "ask"})


def _said_now(
    runtime: Runtime, tools: Sequence[BaseTool | dict[str, Any]]
) -> dict[str, BaseTool | dict[str, Any]]:
    """The harness's own tools a graph was built with (``h.harness_tools()``) as the run offers
    them now, by name, where they say more than when they were bound: ``schedule_task`` when it
    is in the user's zone, ``ask_user`` the app's own controls — each as ``Agent.tools_for``
    gives it to every other target (``schedules.schedule_tools``, ``ask.for_run``)."""
    bound = {
        t.name: t
        for t in tools
        if isinstance(t, BaseTool) and (t.metadata or {}).get(converted.FEATURE) in BUILT_IN
    }
    if not bound:
        return {}
    run = {t.name: t.spec for t in (*schedule_tools(runtime), ask.for_run(runtime))}
    return {
        name: tool.model_copy(
            update={"description": spec.description, "args_schema": spec.input_schema}
        )
        for name, tool in bound.items()
        if (spec := run.get(name)) is not None and spec.description != tool.description
    }


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
    context sent with the question (:func:`with_question`), the end of the history marked when
    a caching middleware marks the request (:func:`history_marked`), and the day it is after
    the instructions (:func:`_dated`). The run's results are kept as ``HarnessTools`` keeps
    them (:data:`_PAGED`: a table read whole kept for ``query_result``, its calls answered by
    the governed tool node), and the harness's readers offered once one is held
    (:func:`_readers`: a run that holds none is sent nothing more); an earlier turn's live
    result is marked out of date, and the question says which (:func:`_stale_noted`)."""
    runtime = current()
    if runtime is not None:
        _PAGED.add(runtime)
        request = _with_readers(runtime, _stale_marked(runtime, request)[0])
    asked = history_marked(with_question(_offered(request, runtime)))
    return await handler(_dated(asked, runtime))


#: What the stub of each result Deep Agents saved says of its table (``results.table_said``),
#: by run, path and the table's name: a saved file does not change, so it is parsed once.
_STUB_TABLES: LRU[tuple[str, str, str], str] = LRU(SAID_ONCE)


def _stubs_tabled(runtime: Runtime, request: ModelRequest[Any]) -> ModelRequest[Any]:
    """The request with Deep Agents' stub of each result it saved in the graph's state saying
    which table its rows are in a query, and its columns, as the harness's own stubs do
    (``results.readable``): ``query_result`` reads it as that table (:func:`_kept_read`), and a
    model not told its name took ``result`` for it in another file's query (the priced pilot's
    ``long-data-02`` on ``ReAct`` with a backend: a failed query, one step more over 80,000
    tokens of results). The request's copy says it, once, never the graph's state."""
    files, tools = _graph_results(request.state, runtime)
    # a copy of a result the run keeps (:func:`_kept_copy`) is that result's table
    whole = _graph_results(request.state)[0]
    copies = {p: k for p in whole.keys() - files.keys() if (k := _kept_copy(runtime, whole[p]))}
    if not files and not copies:
        return request
    with query.graph_results(files, tools):
        named = query.tables(runtime)
    kept: Mapping[str, str] = runtime.replay.journal.results
    messages = list(request.messages)
    for n, m in enumerate(messages):
        if not isinstance(m, ToolMessage) or not isinstance(m.content, str):
            continue
        path = f"{results.PREFIX}/{m.tool_call_id}"
        text = files.get(path) or kept.get(copies.get(path, ""))
        if text is None or not m.content.startswith(results.EVICTED):
            continue
        table = named[copies.get(path, path)]
        said = _STUB_TABLES.setdefault(
            (runtime.run_id, path, table),
            lambda path=path, text=text, table=table: results.table_said(path, text, table),
        )
        if said not in m.content:
            messages[n] = m.model_copy(update={"content": results.stub_tabled(m.content, said)})
    return request.override(messages=messages)


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
    handler = _chain_async_model_call_handlers(
        [*([given] if given else []), answered_in_full, _narrow]
    )
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
    if isinstance(tool, dict):  # a schema, or a provider's own tool: by its name, else its type
        found = tool.get("function") or tool
        return str(found.get("name") or ("" if "function" in tool else tool.get("type", "")))
    return tool.name


# --------------------------------------------------------------------------- the model call
class ModelHooks(AgentMiddleware):
    """Around every model call: the task kept ahead of a summary (the conversation's leading system
    messages, the pushed memory context among them, and the run's question), the run's
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
        cut = accounting.at_output_limit(last)  # arguments that stop where the reply did
        told = [
            ToolMessage(
                (CUT_SHORT.format(tool=c.get("name") or "unknown", problem=_malformed(c)))
                if cut
                else not_run(c.get("name") or "unknown", _malformed(c)),
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
        request = _with_task(_unbroken(_offered(request, runtime)), runtime)
        request = self._in_order(with_question(request), runtime)
        prompt = await self._pinned(runtime)
        if prompt is not None:
            request = self._prompted(request, prompt)
        request = _held_named(runtime, _dated(request, runtime))
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
                if runtime is not None and _retries_itself(request.model):
                    runtime.model_failures += 1  # its client tried again before it failed
                await running().failed("model", error)
                if error is exc:
                    raise
                raise error from exc
            _replied(span, reply)
        # what the model counted of the request it was sent, as sent: a summarization between
        # ``FitWindow`` and here made it smaller than what ``FitWindow`` counted
        _calibrated(name or "", _size(request), reply)
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


def _dated(request: ModelRequest[Any], runtime: Runtime | None) -> ModelRequest[Any]:
    """The request with the day it is said after its instructions (``snapshots.today``: the
    run's start, the same line for every call of the run), so the instructions before it are
    the same from day to day, as a prompt cache keeps them — instructions given as content
    blocks keep their blocks, the day one more after them. Said once (a graph given
    ``ModelHooks`` alone is run through :func:`_narrow` too). Outside a run, as it is."""
    if runtime is None:
        return request
    system, day = request.system_message, snapshots.today(runtime)
    if system is not None and system.text.endswith(day):
        return request
    given = system.content if system is not None else ""
    content: str | list[str | dict[Any, Any]] = (
        [*given, {"type": "text", "text": day}]
        if isinstance(given, list)
        else "\n\n".join(part for part in (given, day) if part)
    )
    return request.override(system_message=SystemMessage(content=content))


def _with_task(request: ModelRequest[Any], runtime: Runtime | None) -> ModelRequest[Any]:
    """The request with the task kept ahead of a summary: a summarization middleware (Deep
    Agents', LangChain's) replaces the older turns by a summary, the run's task and its pushed
    memory context among them — the conversation's leading system messages and the run's
    question (its last user message not steered into it: earlier turns replayed before it are
    the summary's) are put back before it; where the summary took them out of the graph's
    state too (LangChain's removes what it summarized), the run's own (:func:`_asked_again`)."""
    messages = request.messages
    if not messages or not _summary(messages[0]):
        return request
    # a summary kept in the state (LangChain's) ends what is looked at
    state = list(itertools.takewhile(lambda m: not _summary(m), request.state.get("messages", [])))
    asked = next(
        (m for m in reversed(state) if typed(m) and STEERED not in m.response_metadata),
        None,
    )
    if asked is None:
        again = _asked_again(messages, runtime)
        return request if again is None else request.override(messages=[again, *messages])
    lead = itertools.takewhile(lambda m: isinstance(m, SystemMessage), state)
    sent = {m.id for m in messages if m.id is not None}
    kept = [m for m in [*lead, asked] if m.id is None or m.id not in sent]
    return request.override(messages=[*kept, *messages]) if kept else request


def _asked_again(messages: Sequence[BaseMessage], runtime: Runtime | None) -> HumanMessage | None:
    """The run's question as its turn asked it — the memory context pushed with it marked off
    ahead of it (``history.ASKED_WITH``), screened as the graph screens the user's message —
    for a request whose summary left neither in the graph's state; ``None`` with no run, no
    question, or one the request still sends."""
    if runtime is None or not runtime.task:
        return None
    if any(isinstance(m, HumanMessage) and runtime.task in m.text for m in messages):
        return None
    said: list[str | dict[str, Any]] = [{"type": "text", "text": runtime.task}]
    context = runtime.pushed_context
    if context:
        told = ASKED_WITH.format(context=screened(runtime.agent.target, context))
        said.insert(0, {"type": "text", "text": told})
    return HumanMessage(content=said)


def with_question(request: ModelRequest[Any]) -> ModelRequest[Any]:
    """The request with the pushed memory context (the system message
    :data:`~trellis.harness.adapters.langgraph.CONTEXT_MESSAGE_ID`) sent as the first part of
    this turn's question rather than ahead of the history. It is new for every question, and a
    provider's prompt cache keeps only what a request shares with an earlier one from its start:
    so the next turn's request repeats this one's history as it was sent (audit K1). It is a
    part of the user's message, never a system message after the first: some providers refuse
    one there. Only the request changes; the conversation the graph keeps does not. The
    question is the latest user message the graph's state holds (a spent budget's last word
    is not). The context is marked off (``history.ASKED_WITH``): after a long one, a model read
    the question as one more saved note. A cache that keeps only what a request marks keeps
    the history too (:func:`history_marked`)."""
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
            if typed(messages[n]) and messages[n].id in kept
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
    del messages[at]
    return request.override(messages=messages)


def history_marked(request: ModelRequest[Any]) -> ModelRequest[Any]:
    """The request with the answer that ends the history — just before the latest question the
    graph's state holds (a spent budget's last word is not one), the end of what the next
    turn repeats — given the request's cache breakpoint, for a cache that keeps only what a
    request marks (Anthropic's: a prompt-caching middleware sets the request's
    ``cache_control``, :class:`PromptCache`); the request as it is without one."""
    marker = request.model_settings.get("cache_control")
    if not marker:
        return request
    messages = list(request.messages)
    kept = {m.id for m in request.state.get("messages", []) if m.id is not None}
    asked = next(
        (
            n
            for n in range(len(messages) - 1, 0, -1)
            if typed(messages[n]) and messages[n].id in kept
        ),
        None,
    )
    if asked is None:
        return request
    answer = _answer_marked(messages[asked - 1], marker)
    if answer is messages[asked - 1]:
        return request
    messages[asked - 1] = answer
    return request.override(messages=messages)


class PromptCache(AgentMiddleware):
    """Anthropic's prompt cache for every Anthropic chat model, however it is given: a
    ``ChatAnthropic``, or a model name the harness routes (``gateways.routed_chat_model``) whose
    route makes one of it now (the provider's own SDK). LangChain's
    ``AnthropicPromptCachingMiddleware`` (``native``: the one given to ``ReAct``, else its
    defaults) decides on the chat model the call is made with and marks what it marks — the
    tools, the instructions, the request's own breakpoint — and the answer that ends a history
    the harness replays gets the same mark (:func:`history_marked`; the model's own answer in
    a graph's thread is sent as it gave it): 4 breakpoints, Anthropic's most. Any
    other chat model's request is sent as it was. Named as LangChain's, it stands in Deep
    Agents' place for it (Deep Agents replaces a middleware of the same name), last of what
    changes the request."""

    def __init__(self, native: AgentMiddleware | None = None) -> None:
        super().__init__()
        if native is None:
            from langchain_anthropic.middleware import (  # noqa: PLC0415 - Deep Agents' own
                AnthropicPromptCachingMiddleware,
            )

            native = AnthropicPromptCachingMiddleware(unsupported_model_behavior="ignore")
        self.native = native

    @property
    def name(self) -> str:
        return CACHING

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelCallResult[Any]:
        given = request.model
        # what comes after it — the model call, and Deep Agents' memory, which marks its own
        # block only for a chat model it knows as Anthropic's — is handed the routed name's
        # model of this call (the route chosen once: the gateway's headers are on already)
        made = cast(Any, given).resolved() if routed(given) else given

        async def sent(marked: ModelRequest[Any]) -> ModelResponse[Any]:
            return await handler(history_marked(marked).override(model=made))

        return await self.native.awrap_model_call(request.override(model=made), sent)


#: The name of LangChain's Anthropic prompt-caching middleware, which :class:`PromptCache` has.
CACHING: Final = "AnthropicPromptCachingMiddleware"


def prompt_cache(given: Sequence[Any]) -> tuple[PromptCache, list[Any]]:
    """The :class:`PromptCache` a graph is built with — around LangChain's caching middleware
    among ``given``, its settings kept, else around its defaults — and the rest of ``given``."""
    own = [m for m in given if m.name == CACHING]
    rest = [m for m in given if m.name != CACHING]
    return PromptCache(own[0] if own else None), rest


def _answer_marked(answer: AnyMessage, marker: Any) -> AnyMessage:
    """The answer that ends the history with a cache breakpoint (``marker``) on its last text
    part; any other message (a call, a result) as it is, and so is an answer that has one.
    So is an answer the model itself gave in this thread (a graph's own checkpointer keeps it,
    with its usage): Anthropic holds a model's own earlier turns to what it produced (its
    thinking stays valid only in that conversation), and marking one measurably changed the
    answers after it — the request's own breakpoint caches that thread anyway. What is marked
    is a history the harness replays (memory's window)."""
    if not isinstance(answer, AIMessage) or answer.tool_calls or answer.usage_metadata:
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


def typed(message: Any) -> bool:
    """Whether ``message`` is one the user sent: a human message the harness or a framework did
    not add (``adapters.langgraph.notified``: a background task's update, a summary of earlier
    turns) — or a run's input marked as its question (``adapters.langgraph.ASKED``: a
    notification run's). The turn's question is the last such message, never a notification
    after it."""
    return isinstance(message, HumanMessage) and (
        ASKED in message.response_metadata or notified(message) is None
    )


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


def _retries_itself(model: Any) -> bool:
    """Whether a chat model's own client tries a failed request again: LangChain's chat
    models' ``max_retries`` (unset: the provider SDK's default, which retries). A model the
    harness routes sets none (the gateway retries), and its failure is the run's to retry."""
    return hasattr(model, "max_retries") and model.max_retries != 0


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
        cost=budget.gateway_cost(metadata),
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
    left, not LangChain's "Model call limits exceeded" message. A model turn asked for after
    that answer (a background task's outcome, after a final answer: Deep Agents' async
    sub-agents, the harness's own tasks) counts as any call, and past the limit it is not made:
    the run ends with the answer it has, where LangChain's limit, one call past ``max_steps``,
    would put its message in the answer's place. Named as LangChain's (a ``ReAct(middleware=)``
    of that name is refused, naming ``max_steps=``)."""

    def __init__(self, max_steps: int, *, scoped: bool) -> None:
        self.max_steps = max_steps
        self.counted = "thread_model_call_count" if scoped else "run_model_call_count"
        limit = {"thread_limit" if scoped else "run_limit": max_steps + 1}
        super().__init__(exit_behavior="end", **limit)

    @property
    def name(self) -> str:
        return ModelCallLimitMiddleware.__name__

    @hook_config(can_jump_to=["end"])
    async def abefore_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        if state.get(self.counted, 0) <= self.max_steps:
            return None
        return {"jump_to": "end", **_unread(state)}

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


def _unread(state: Mapping[str, Any]) -> dict[str, Any]:
    """The background tasks' outcomes read after the final answer (Deep Agents' async
    sub-agents' notifications) for a model turn the step limit does not make: not read after
    all — their notifications leave the turn, and their tasks' records say they still work, so
    the conversation's next turn reads and tells them again."""
    messages = state.get("messages") or []
    answered = max((n for n, m in enumerate(messages) if isinstance(m, AIMessage)), default=-1)
    told = [
        m
        for m in messages[answered + 1 :]
        if (notified(m) or {}).get("source") == task_tools.SOURCE
    ]
    if not told:
        return {}
    tasks = state.get("async_tasks") or {}
    ids = {str(((notified(m) or {}).get("data") or {}).get("task_id")) for m in told}
    return {
        "messages": [RemoveMessage(id=str(m.id)) for m in told],
        "async_tasks": {t: {**tasks[t], "status": "running"} for t in ids if t in tasks},
    }


class StallGuard(AgentMiddleware):
    """A run that goes nowhere ends with one answer: the same call with the same arguments in
    ``max_repeats`` consecutive steps (before it runs again), or :data:`ERROR_STREAK`
    consecutive steps in which every call failed (a malformed one among them) and a failure
    was met again (:func:`_failing`), or a reply cut at the model's output limit with no answer
    (:func:`answered_in_full`). The model is told so (``bridge.STALLED``, :data:`FAILING`,
    ``budget.CUT_OFF``) and called once more, offered no tool (:func:`no_tool`), so the
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
        reply = await answered_in_full(request, handler)
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


#: What a run whose answer's reply reached the model's output limit says of it (a warning: the
#: answer is still the run's).
ANSWER_CUT: Final = "the model's answer reached its output limit: it may stop before its end"
#: Why a run whose model's reply reached its output limit with no answer, twice, stopped.
CUT_STOPPED: Final = "stopped: the model's reply reached its output limit before it gave an answer"


async def answered_in_full(
    request: ModelRequest[Any],
    handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
) -> ModelResponse[Any]:
    """The model's reply — or, when it reached the model's output limit before it gave an
    answer (``accounting.cut_off``), the model called once more, told so (``budget.CUT_OFF``)
    and offered no tool (:func:`_answer_now`, an ``output_limit`` warning): LangChain ends its
    loop on a reply with no tool call, so the run ended with no answer (the break-and-fix
    gate's LangGraph turns, 4,096 tokens of rows summed by hand). An answer the limit cut
    short is the run's as it is, said to be cut (:data:`ANSWER_CUT`, the same warning): it
    went out as a plain success, a table stopping mid-row. ``StallGuard`` has it, and a
    ``create_agent`` graph without the harness's middleware runs it (:func:`narrowing`)."""
    reply = await handler(request)
    message = _ai(reply)
    if message is None or not accounting.at_output_limit(message):
        return reply
    if accounting.cut_off(message):
        return await _answer_now(
            request, handler, budget.CUT_OFF, CUT_STOPPED, warning="output_limit"
        )
    if not message.tool_calls:  # an answer, cut where the limit fell: the run's, said so
        runtime = current()
        log.warning("run %s %s", runtime.run_id if runtime else "-", ANSWER_CUT)
        if runtime is not None:
            runtime.events.warning("output_limit", ANSWER_CUT)
    return reply


def no_tool(model: Any) -> Any:
    """The ``tool_choice`` that offers ``model`` no tool, in its own terms: LangChain's
    ``"none"`` (OpenAI's word), and Anthropic's ``{"type": "none"}`` for a ``ChatAnthropic`` — a
    model name routed to one too — whose ``bind_tools`` reads any other string as the name of
    a tool to force (Anthropic's API then refuses the request). The tools stay defined: a
    history with tool calls needs them."""
    from langchain_anthropic import ChatAnthropic  # noqa: PLC0415 - Deep Agents' own

    made = cast(Any, model).resolved() if routed(model) else model
    return {"type": "none"} if isinstance(made, ChatAnthropic) else "none"


async def _answer_now(
    request: ModelRequest[Any],
    handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    told: str,
    stopped: str,
    *,
    warning: str = "stalled",
) -> ModelResponse[Any]:
    """The model called once more, told ``told`` and offered no tool; its answer (``warning``
    on the run's stream), or the run stopped (``ModelError(stopped)``) when it gives none.
    A run whose answer is a schema (``response_format``) is offered none of its tools instead:
    LangChain forces a tool call for a structured answer (``tool_choice`` ``any``, beside the
    schema's own tool), so ``no_tool`` would not reach the model, and its answer is the
    schema's (``structured_response``), a call of the schema's tool."""
    told_now = [*request.messages, HumanMessage(content=told)]
    if request.response_format is not None:
        last = request.override(messages=told_now, tools=[])
    else:
        choice = no_tool(request.model) if request.tools else request.tool_choice
        last = request.override(messages=told_now, tool_choice=choice)
    reply = await handler(last)
    message = _ai(reply)
    in_schema = reply.structured_response is not None
    if message is None or (not in_schema and (message.tool_calls or not message.text)):
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
        if typed(message):
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
    something to read: once a result was cleared (``cleared``: the graph clears older results,
    LangChain's ``ClearToolUsesEdit``; ``read_file`` pages one from the graph's state by the
    path its placeholder names, the harness's own when the graph has none), once the graph's
    state holds a file (a large result saved, a summarized history) or the run keeps a result
    — a small model offered it from the start reads nothing, again and again. ``files=True``:
    ``read_file`` always offered (its files are outside the graph's state: a disk, a store, a
    sandbox, the skills)."""

    def __init__(self, *, cleared: bool, files: bool = False) -> None:
        super().__init__()
        self.cleared = cleared
        self.files = files

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        tools = list(request.tools)
        has = any(_named(t) == READ_FILE for t in tools)
        if self.cleared and any(_cleared(m) for m in request.messages):
            # a result was cleared: read_file reads it back; query_result computes over one
            # that holds rows (its placeholder names the table: ``_rows_said``)
            rows = any(_cleared(m) and query.QUERY_RESULT in m.text for m in request.messages)
            tools += _readers(current(), request.state, {_named(t) for t in tools}, held=rows)
        elif has and not (self.files or request.state.get("files") or results.kept()):
            tools = [t for t in tools if _named(t) != READ_FILE]
        return await handler(
            request if tools == list(request.tools) else request.override(tools=tools)
        )


# --------------------------------------------------------------------------- the window
#: Where a model's profile keeps the window the model said, when ``ReAct`` was given another
#: (``react._profiled``): the reply is reserved out of the model's own (``FitWindow._budget``).
OWN_WINDOW: Final = "trellis_own_window"


def _sending(model: Any) -> Any:
    """The chat model that sends a request made of ``model``: a model name's, the one its
    route resolves to now (its ``max_tokens`` is the reply the call asks for); any other
    model, itself. The name itself where the route cannot make one: the call says why."""
    if not routed(model):
        return model
    try:
        return cast(Any, model).resolved()
    except Exception:
        return model


#: Tools a narrowing keeps at least, beside the ones it must keep (the run's used ones, the
#: graph's own state tools): the descriptions are compacted, the longest results cut and the
#: older turns left out before any of them is dropped.
FIT_KEEP: Final = 8
#: A compacted tool description's most characters (its first sentence, at most this).
FIT_DESCRIPTION: Final = 120
#: What a tool offered by its signature alone says where its description was (:func:`_signature`).
FIT_SIGNATURE: Final = "(description left out to fit the window)"
#: What the instructions say of the tools a request no longer offers at all, and what finds
#: one (``tool_search``, where the run has it) or that nothing does at this size.
FIT_HELD: Final = "Tools held back to fit the window ({how}): {names}"
FIT_FOUND: Final = f"{TOOL_SEARCH} finds one and offers it again"
FIT_NOT_FOUND: Final = "not offered at this window size"
_HELD: Final = FIT_HELD.split("(", maxsplit=1)[0]
#: What the latest question of a request opens with when no ``write_todos`` call in it shows
#: the todo list the graph's state keeps (:func:`todos_shown`).
TODOS_SHOWN: Final = "Your todo list (write_todos), as you last wrote it:\n{todos}"
_TODOS_SHOWN: Final = TODOS_SHOWN.split(":", maxsplit=1)[0]
_HELD_SIZE: Final = len(FIT_HELD.format(how=FIT_FOUND, names="")) / 4 + 1
#: The characters a cut result's marker takes at most (its size, and where the rest is read).
CUT_MARKER: Final = 192
#: The steps :class:`FitWindow` takes, in order, each only while the request does not fit;
#: how often each fired in this process (a bounded counter: these keys only).
FIT_STEPS: Final = ("tools", "descriptions", "results", "turns", "steps", "memory")
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
#: And the window it says it has, in the same refusal (OpenAI's and vLLM's "maximum context
#: length is N", Anthropic's "> N maximum", llama.cpp's "context size (N").
_OVERFLOW_LIMIT: Final = re.compile(
    r"maximum context length is (\d+)|> (\d+) maximum|context size \((\d+)", re.IGNORECASE
)
#: The window a model's server named when it refused a request, by model name: what is known
#: of a model whose window no one else said (:meth:`FitWindow._learning`).
_LEARNED: Final[LRU[str, int]] = LRU(64)
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
    :class:`FitWindow`, which leaves older turns out first. Each cleared result's
    ``placeholder`` names where it is read again (``{path}``: Deep Agents' path for its call,
    ``/large_tool_results/<call id>``, which :func:`_cleared_result` reads) — a model is not
    shown the ids of its calls, so a placeholder that left the id to it sent it to fetch the
    result again."""

    def apply(self, messages: list[AnyMessage], *, count_tokens: TokenCounter) -> None:
        start = next(
            (i for i in range(len(messages) - 1, -1, -1) if typed(messages[i])),
            0,
        )
        turn = messages[start:]
        shown = [m.text if isinstance(m, ToolMessage) else "" for m in turn]  # before clearing
        earlier = count_tokens(messages[:start]) if start else 0
        ClearToolUsesEdit.apply(
            replace(self, trigger=self.trigger - earlier), turn, count_tokens=count_tokens
        )
        for n, message in enumerate(turn):
            if _cleared(message) and message.content == self.placeholder:
                # the result's own rows, never the message's words: a kept table's message
                # ends with its note, and read with it a JSON result's "columns" were its data
                path, result = _shown(shown[n], f"{results.PREFIX}/{message.tool_call_id}")
                rows = "" if result is None else _rows_said(path, result, query.TABLE)
                said = self.placeholder.replace("{path}", path) + rows
                turn[n] = message.model_copy(update={"content": said})
        messages[start:] = turn


@functools.lru_cache(maxsize=256)
def _rows_said(path: str, text: str, table: str) -> str:
    """What the mark of a result the window no longer shows whole (cleared, or cut to fit)
    adds when the result holds rows: the ``table`` ``query_result`` computes over them at
    ``path``, as every held result's note says (``results.table_said``) — so it is counted
    exactly, not read back and added up. ``""`` for one with no rows (prose, a document).
    Each result is parsed once: the request is cleared and fitted anew before every model
    call."""
    columns, rows = query.rows_of(text)
    if not rows or columns == ["line"]:
        return ""
    return " " + results.table_said(path, text, table)


#: What a summarized request says, after its summary, of the results the run keeps that no
#: message left in it names; and the most it names (the rest are counted).
HELD_NAMED: Final = (
    "Results of this run that are still kept, each a table query_result computes over and "
    "read_file pages: {named}{more}."
)
HELD_MOST: Final = 100


def _held_named(runtime: Runtime | None, request: ModelRequest[Any]) -> ModelRequest[Any]:
    """The request with its summary — a summarization's, LangChain's or Deep Agents' — saying
    which results the run still keeps that nothing left in the request names: each by its
    table and its path (:data:`HELD_NAMED`). A summarization replaces the messages that named
    them with a summary in the summarizer's own words, which may name none (0 of 56, and the
    model said it had lost track of the results); the run keeps them all the same. The
    request's copy says it, never the graph's state. As it is with no summary, outside a run,
    or when every kept result is still named."""
    at = next(
        (
            n
            for n in range(len(request.messages) - 1, -1, -1)
            if request.messages[n].additional_kwargs.get("lc_source") == "summarization"
        ),
        None,
    )
    if runtime is None or at is None:
        return request
    read = " ".join(m.text for m in request.messages)
    names = query.tables(runtime)
    lost = [p for p in runtime.replay.journal.results if p not in read]
    if not lost:
        return request
    named = "; ".join(f"{names[p]} at {p}" for p in lost[:HELD_MOST])
    more = f"; and {len(lost) - HELD_MOST} more" if len(lost) > HELD_MOST else ""
    summary = request.messages[at]
    said = f"{summary.text}\n\n{HELD_NAMED.format(named=named, more=more)}"
    messages = list(request.messages)
    messages[at] = summary.model_copy(update={"content": said})
    return request.override(messages=messages)


def _start(text: str, room: int) -> str:
    """The first ``room`` characters of ``text``, saying how much was left out; ``text`` when
    it fits."""
    left = results.SHORTENED.format(left=len(text) - room)
    return text if room + len(left) >= len(text) else f"{text[:room]}{left}"


def _page_cut(messages: Sequence[AnyMessage], at: int, room: int) -> str | None:
    """The page ``read_file`` returned in ``messages[at]``, in about ``room`` characters: its
    first whole lines, then where to read on — the file the call read, from the line after
    the last one shown. A page is a view of a result (its lines, numbered), not a result:
    kept as one, its rows were the numbered lines, named as a table, and a join against it
    matched nothing. ``None``: the page fits, shows no whole line in that room, or its call
    is not in the request (it is then cut at its start as it stands: :func:`_start`)."""
    page = messages[at].text
    head = page[: page.rfind("\n", 0, room) + 1]
    call = next(
        (
            c
            for m in messages[:at]
            if isinstance(m, AIMessage)
            for c in m.tool_calls
            if c["id"] == cast(ToolMessage, messages[at]).tool_call_id
        ),
        None,
    )
    if room >= len(page) or not head.strip() or call is None:
        return None
    asked = call["args"]
    offset = asked.get("offset")
    start = max(0, offset) if isinstance(offset, int) else 0
    on = start + sum(1 for line in head.splitlines() if line.strip())
    cut = f"{len(page) - len(head)} characters cut to fit the window"
    path = "/" + str(asked.get("file_path", "")).lstrip("/")
    return f'{head}…[{cut}; read on: {READ_FILE}(file_path="{path}", offset={on})]'


def _mark(message: AnyMessage) -> bool:
    """Whether a tool message is a mark — what stands in the request for a result it does not
    show: a cleared result's line, a stub — and so never a result itself."""
    return _cleared(message) or (
        isinstance(message, ToolMessage) and message.text.startswith(results.EVICTED)
    )


def _cleared(message: AnyMessage) -> TypeGuard[ToolMessage]:
    """Whether LangChain's ``ClearToolUsesEdit`` cleared this result from the request (its
    own mark on the message)."""
    return isinstance(message, ToolMessage) and bool(
        message.response_metadata.get("context_editing", {}).get("cleared")
    )


class FitWindow(AgentMiddleware):
    """A model call never fails because its request does not fit the window. Before each call,
    when the request (its messages, instructions and tools, counted as Deep Agents counts them)
    is over :data:`FIT_SHARE` of the window less the reply's tokens, it is made smaller, step
    by step, only as far as it must, each step said on the run (a ``context_fit`` warning and a
    log line) and counted (:data:`FITTED`):

    1. ``tools`` — the offered tools narrowed to the most relevant: those the run already used
       (and the graph's own state tools) always kept, then those the tool hints chose, then by
       the words they share with the request, :data:`FIT_KEEP` of them at least besides; a
       tool not kept is still offered by its signature while that fits (always, when the
       hints chose it), and one held back altogether is named after the instructions;
    2. ``descriptions`` — the kept tools' descriptions compacted (the first sentence, no
       parameter descriptions);
    3. ``results`` — the longest results cut to their head and tail, each kept whole and its
       cut naming where ``read_file`` reads the rest (clearing older results is LangChain's
       ``ClearToolUsesEdit``'s, before it: one clearing pass). ``turns=False`` (a Deep Agents
       summarization after it) leaves the results to it: it offloads a result whole to its
       backend and reads it back with its own ``read_file``, so a result is stored once, where
       its sandbox's code reads it;
    4. ``turns`` — the older turns left out, the latest kept whole: first their tool calls and
       results, the oldest turn's first, each such turn then its question and answer (what
       memory's window gives of an older turn), and whole only when those do not fit either
       (``turns=False`` leaves the turns to a summarization after it: ``ReAct``'s summarizes
       them); only then the tools narrowed past :data:`FIT_KEEP`, as far as the latest turn
       leaves no room, those :data:`FIT_KEEP` staying by their signature: the tools the
       question needs are never dropped for older turns, nor for the latest turn's own bulk;
    5. ``steps`` — the latest turn's older steps (a call and its results), the oldest first,
       never the latest :data:`KEEP_STEPS`: said in one note after the question, a line each
       (the call and the start of its result) — a turn of many small steps is over a small
       window by itself (``turns=False`` leaves them to the summarization after it);
    6. ``memory`` — last, only when the latest turn would still not fit: the memory context's
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
        #: and every tool it offered, whole or by signature (``shown``): no other is added later
        self._shown: weakref.WeakKeyDictionary[Runtime, frozenset[str]] = (
            weakref.WeakKeyDictionary()
        )

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        request = todos_shown(request)
        budget = self._budget(request)
        if budget is None:
            return await self._learning(request, handler)
        name = _model_name(request.model) or ""
        ratio = _RATIOS.get(name) or 1.0
        sent, counted = self._within(request, int(budget / ratio))
        try:
            return await handler(sent)
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
            return await handler(self._within(request, int(budget / learned), refit=True)[0])

    async def _learning(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        """The call of a model whose window no one said — not the caller, not the library's
        profile, not the route: sent as it is, and when the server refuses it as over its
        window and names that window (:func:`_refused_limit`), fitted to it and sent again.
        The window is the model's from then on (:data:`_LEARNED`, by its name). Nothing was
        fitted for such a model, so its first long request ended the run with the refusal;
        a refusal that names no window still does."""
        try:
            return await handler(request)
        except Exception as exc:
            limit = _refused_limit(exc)
            if limit is None:
                raise
        name = _model_name(request.model)
        if name:
            _LEARNED[name] = limit
        budget = self._budget(request, limit)
        assert budget is not None  # a window is known now
        FITTED["refit"] += 1
        log.info("the model's server said its window is %s tokens: fitted to it", limit)
        return await handler(self._within(request, budget, refit=True)[0])

    def _within(
        self, request: ModelRequest[Any], budget: int, *, refit: bool = False
    ) -> tuple[ModelRequest[Any], int]:
        """The request fitted to ``budget`` (counted tokens), and its count."""
        size = _size(request)
        if size <= budget:
            return request, size
        fitted = self._fitted(request, budget, refit=refit)
        return fitted, _size(fitted)

    def _budget(self, request: ModelRequest[Any], said_now: int | None = None) -> int | None:
        """What the request may hold: the window's share (:data:`FIT_SHARE`), and no more than
        the model's own window leaves beside the reply the call will send — a provider takes
        input and reply together. The reply is the call's own setting, else the chat model's
        that makes the call (a model name's: the one its route resolves to; read from the
        name there was none, and 950,000 tokens were sent where the provider took 872,000).
        A window the caller gives below the model's own is not what the reply is taken out
        of: given 32,000 for a model replying up to 64,000, nothing was left for the input —
        the model's own is its profile's (``ReAct``'s copy keeps it as :data:`OWN_WINDOW`).
        The window is the one given, else the model's profile's, else the one its server
        named when it refused a request (``said_now``, :data:`_LEARNED`); ``None``: no one
        has said it."""
        profile = getattr(request.model, "profile", None)
        said = profile if isinstance(profile, dict) else {}
        learned = said_now or _LEARNED.get(_model_name(request.model) or "")
        window = self.window or said.get("max_input_tokens") or learned
        if not isinstance(window, int) or isinstance(window, bool):
            return None
        sends = _sending(request.model)
        reply = 0
        for key in ("max_tokens", "max_completion_tokens", "max_output_tokens"):
            value = request.model_settings.get(key, getattr(sends, key, None))
            if isinstance(value, int) and not isinstance(value, bool):
                reply = max(reply, value)
        # the model's own window: kept beside the given one by ReAct, the profile's where the
        # window was given here (``middleware(context_window=)``), else the one in use
        own = said.get(OWN_WINDOW) or said.get("max_input_tokens") or learned or window
        return max(0, min(int(window * FIT_SHARE), int(own * FIT_SHARE) - reply))

    def _results(
        self, request: ModelRequest[Any], budget: int, *, refit: bool
    ) -> ModelRequest[Any]:
        """The longest results cut (:func:`_cut`) — with a summarization after it, whose own
        overflow handling offloads them whole (Deep Agents'), only once the request was refused
        all the same (``refit``: what it keeps, its ``read_file``'s pages, did not fit beside its
        summary), and only the latest turn's: the messages it cuts the summary at stay where
        they are."""
        if self.turns:
            return _cut(request, budget)
        return _cut(request, budget, latest=True) if refit else request

    def _turns(
        self,
        request: ModelRequest[Any],
        budget: int,
        runtime: Runtime | None,
        before: tuple[Collection[str], Collection[str]],
        given: Sequence[BaseTool | dict[str, Any]],
        *,
        refit: bool,
    ) -> ModelRequest[Any]:
        """The older turns left out (none with a summarization after it: ``turns=False``) —
        their calls and results first (:func:`_answers`), whole when their questions and answers
        do not fit either — then the tools past :data:`FIT_KEEP` narrowed only as far as what is
        left does not fit, and those :data:`FIT_KEEP` kept by their signature: a latest turn
        that fills the window by itself is not made to fit by taking away the tools it goes on
        with. Only a request the model's server itself refused (``refit``) gives them up too."""
        if self.turns:
            request = todos_shown(_answers(request, budget))
            if _size(request) <= budget:
                return request
            request = todos_shown(_latest(request))
        return _narrowed(
            request,
            budget,
            runtime,
            least=0,
            signed=0 if refit else FIT_KEEP,
            before=before[0],
            shown=before[1],
            given=given,
        )

    def _fitted(
        self, request: ModelRequest[Any], budget: int, *, refit: bool = False
    ) -> ModelRequest[Any]:
        runtime = current()
        before = self._kept.get(runtime, frozenset()) if runtime is not None else frozenset()
        shown = self._shown.get(runtime, frozenset()) if runtime is not None else frozenset()
        given = request.tools
        taken: list[str] = []
        steps: tuple[Callable[[ModelRequest[Any], int], ModelRequest[Any]], ...] = (
            lambda r, b: _narrowed(
                r, b, runtime, least=FIT_KEEP, before=before, shown=shown, given=given
            ),
            lambda r, b: r.override(tools=[_compact(t) for t in r.tools]),
            lambda r, b: self._results(r, b, refit=refit),
            lambda r, b: self._turns(r, b, runtime, (before, shown), given, refit=refit),
            lambda r, b: _steps_said(r, b) if self.turns else r,
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
        if runtime is not None and ("tools" in taken or "turns" in taken):
            self._kept[runtime] = frozenset(_named(t) for t in request.tools if _whole(t))
            self._shown[runtime] = frozenset(_named(t) for t in request.tools)
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


def _refused_limit(exc: BaseException) -> int | None:
    """The window a model server named in refusing a request as over it, or ``None``: ``exc``
    is no such refusal (:func:`_refused_size`), or names none."""
    seen: BaseException | None = exc if _refused_size(exc) is not None else None
    while seen is not None:
        said = _OVERFLOW_LIMIT.search(str(seen))
        if said is not None:
            return int(next(n for n in said.groups() if n))
        seen = seen.__cause__
    return None


def _size(request: ModelRequest[Any]) -> int:
    """The request's tokens, as Deep Agents' summarization counts them."""
    system = [request.system_message] if request.system_message is not None else []
    return count_tokens_approximately([*system, *request.messages], tools=list(request.tools))


def _bare(request: ModelRequest[Any]) -> ModelRequest[Any]:
    """The request with only its instructions and the user's latest message, no tools."""
    asked = next((m for m in reversed(request.messages) if typed(m)), None)
    return request.override(messages=[asked] if asked is not None else [], tools=[])


def _narrowed(
    request: ModelRequest[Any],
    budget: int,
    runtime: Runtime | None,
    *,
    least: int,
    signed: int = FIT_KEEP,
    before: Collection[str] = (),
    shown: Collection[str] = (),
    given: Sequence[BaseTool | dict[str, Any]] | None = None,
) -> ModelRequest[Any]:
    """The request's tools ranked — those it must keep (the run's used ones, the graph's own
    state tools, the readers of a result the run keeps, :data:`SAVING`, which its preview says
    to call, and what finds a held tool, ``tool_search``), those an earlier call of the run
    kept whole (``before``: the same set call after call, so a prompt cache keeps its prefix;
    a tool none offered, ``shown``, is not offered later unless the hints choose it), those
    the tool hints chose, then by the words they share with the user's latest message — and
    kept in that order while they fit what the messages leave, ``least`` of them at least
    besides those it must keep — which are set aside first by their first sentence
    (:func:`_compact`), and kept whole only where that leaves the others room.

    A tool it does not keep is still offered while its signature fits (:func:`_signature`:
    its name and arguments, what a call takes), and whatever the room when the tool hints chose
    it or it is among the first ``signed`` besides those it must keep: the model reaches for a
    tool it has not called yet, and it is there at every window size. Only past those is a
    tool held back, and the instructions name it (:func:`_held_said`: of ``given``, the tools
    before any narrowing of the call) — a line whose room is set aside first."""
    tools, incoming = list(request.tools), request
    if not tools:
        return request
    request = _unheld(request)  # an earlier narrowing's line: said anew, of what is held now
    must = set(runtime.used) if runtime is not None else set()
    must |= {*STATE_TOOLS, TOOL_SEARCH, *(SAVING if _saved(request.state) else ())}
    # a provider's own tool (``provider_tools``): its definition is the provider's, not cut
    must |= {_named(t) for t in tools if isinstance(t, dict) and _provider_own(t)}
    hinted = (runtime.offered or set()) if runtime is not None else set()
    asked = next((m.text for m in reversed(request.messages) if typed(m)), "")
    words = set(_WORD.findall(asked.lower()))

    def rank(item: tuple[int, BaseTool | dict[str, Any]]) -> tuple[int, int, int]:
        n, tool = item
        name = _named(tool)
        shared = len(words & set(_WORD.findall(f"{name} {_described(tool)}".lower())))
        tier = 0 if name in must else 1 if name in before else 2 if name in hinted else 3
        return (tier, -shared, n)

    offered = [_named(t) for t in (tools if given is None else given)]
    # what naming a held tool takes is set aside for each that could be, and given back by
    # each that stays: the line is part of what must fit
    aside = {name: (len(name) + 2) / 4 for name in offered if name not in must}
    room = budget - _size(request.override(tools=[])) - (_HELD_SIZE + sum(aside.values()))
    kept: dict[int, BaseTool | dict[str, Any]] = {}
    ranked = sorted(enumerate(tools), key=rank)
    # those it must keep take their compacted room first, and are kept whole only where the
    # others still fit beside them (``write_todos`` whole held back the task's tools at 8,000)
    musts = {n: tool for n, tool in ranked if _named(tool) in must}
    least_of = {n: count_tokens_approximately([], tools=[_compact(t)]) for n, t in musts.items()}
    room -= sum(least_of.values())
    more = 0  # kept, or offered by signature, beside those it must keep
    for n, tool in ranked:
        name = _named(tool)
        if n in musts or (shown and name not in shown and name not in hinted):
            continue  # a tool an earlier call held back stays held: the same set call after call
        cost = count_tokens_approximately([], tools=[tool]) - aside.get(name, 0)
        if more < least or cost <= room:
            kept[n] = tool
        else:
            bare = _signature(tool)
            cost = count_tokens_approximately([], tools=[bare]) - aside.get(name, 0)
            if not (name in hinted or more < signed or cost <= room):
                continue  # held back: named in the instructions
            kept[n] = bare
        more += 1
        room -= cost
    for n, tool in musts.items():
        extra = count_tokens_approximately([], tools=[tool]) - least_of[n]
        whole = extra <= room
        kept[n] = tool if whole else _compact(tool)
        room -= extra if whole else 0
    if all(kept.get(n) is tool for n, tool in enumerate(tools)):
        return incoming
    out = request.override(tools=[kept[n] for n in sorted(kept)])  # the offered order kept
    return _held_said(out, offered)


def _provider_own(tool: dict[str, Any]) -> bool:
    """Whether a tool given as a ``dict`` is a provider's own (a ``type`` other than a
    function's), not a schema."""
    return "function" not in tool and tool.get("type") not in (None, "function")


def _signature(tool: BaseTool | dict[str, Any]) -> BaseTool | dict[str, Any]:
    """A tool as its name and the arguments a call takes (their names, types and which are
    required), no description: what a call needs, in a few tokens."""
    return _compact(tool, FIT_SIGNATURE)


def _whole(tool: BaseTool | dict[str, Any]) -> bool:
    """Whether a tool is offered with its description (not by its signature alone)."""
    return _described(tool) != FIT_SIGNATURE


def _held_said(request: ModelRequest[Any], offered: Sequence[str]) -> ModelRequest[Any]:
    """The request saying, after its instructions, which of the tools ``offered`` before any
    narrowing it no longer offers, and what finds one: a tool is never held back in silence (a
    model that was not told answered that no tool held the data)."""
    now = {_named(t) for t in request.tools}
    held = [name for name in offered if name not in now]
    if not held:
        return request
    how = FIT_FOUND if TOOL_SEARCH in now else FIT_NOT_FOUND
    line = FIT_HELD.format(how=how, names=", ".join(held))
    system = request.system_message
    content = system.content if system is not None else ""
    said: str | list[str | dict[Any, Any]] = (
        [*content, {"type": "text", "text": line}]
        if isinstance(content, list)
        else "\n\n".join(part for part in (content, line) if part)
    )
    return request.override(system_message=SystemMessage(content=said))


def _unheld(request: ModelRequest[Any]) -> ModelRequest[Any]:
    """The request without the line an earlier narrowing of the call said (:func:`_held_said`)."""
    system = request.system_message
    if system is None or _HELD not in system.text:
        return request
    content = system.content  # the line is the last thing said: a block of its own, or a part
    left: str | list[str | dict[Any, Any]] = (
        content[:-1] if isinstance(content, list) else content[: content.rindex(_HELD)].rstrip()
    )
    return request.override(system_message=SystemMessage(content=left) if left else None)


def _described(tool: BaseTool | dict[str, Any]) -> str:
    if isinstance(tool, dict):
        return str((tool.get("function") or tool).get("description") or "")
    return tool.description or ""


def _compact(tool: BaseTool | dict[str, Any], said: str | None = None) -> BaseTool | dict[str, Any]:
    """A tool as its definition with the description cut to its first sentence (at most
    :data:`FIT_DESCRIPTION` characters; ``said``: in its place) and no parameter
    descriptions. A provider's own tool (a definition that is no function: web search, code
    execution) is as it is: nothing of it is the harness's to cut. A tool carrying provider
    fields (LangChain's ``extras``: Anthropic's ``defer_loading``, a ``cache_control``) stays
    a tool carrying them — a provider reads them from the tool, not from a definition."""
    if isinstance(tool, dict) and tool.get("type", "function") != "function":
        return tool
    found = copy.deepcopy(convert_to_openai_tool(tool))
    function = found.get("function", found)
    first = re.split(r"(?<=[.!?])\s", str(function.get("description") or ""), maxsplit=1)[0]
    function["description"] = first[:FIT_DESCRIPTION] if said is None else said
    for spec in (function.get("parameters") or {}).get("properties", {}).values():
        spec.pop("description", None)
    if isinstance(tool, dict) or not tool.extras:
        return found
    cut = {"description": function["description"], "args_schema": function["parameters"]}
    return tool.model_copy(update=cut)


def _asked(request: ModelRequest[Any]) -> int | None:
    """Where the request's latest turn starts (:func:`_question`) — or ``None`` when no turn is
    before it."""
    asked = _question(request)
    return None if asked is None or asked <= _lead(request.messages) else asked


def _question(request: ModelRequest[Any]) -> int | None:
    """Where the user's latest message in the conversation is — a note appended to this call
    alone, as the step limit's and the budget's are, is not one: the turn starts at the
    question before it — or ``None`` when the request holds none."""
    state = request.state.get("messages") or []
    said = {id(m) for m in state}
    named = {m.id for m in state if m.id is not None}
    return max(
        (
            n
            for n, m in enumerate(request.messages)
            if typed(m) and (id(m) in said or m.id in named)
        ),
        default=None,
    )


def _lead(messages: Sequence[AnyMessage]) -> int:
    """How many messages open a request before its turns: its leading system messages, and
    the summary a summarization of the graph's own (LangChain's, Deep Agents') put in its
    older turns' place. The summary is what is left of those turns: left out with the turns
    that follow it, they would be lost twice."""
    opens = itertools.takewhile(lambda m: isinstance(m, SystemMessage) or _summary(m), messages)
    return sum(1 for _ in opens)


def _latest(request: ModelRequest[Any]) -> ModelRequest[Any]:
    """The request with what opens it (:func:`_lead`) and its latest turn only
    (:func:`_asked`)."""
    asked = _asked(request)
    if asked is None:
        return request
    messages = list(request.messages)
    return request.override(messages=[*messages[: _lead(messages)], *messages[asked:]])


def _answers(request: ModelRequest[Any], budget: int) -> ModelRequest[Any]:
    """The request with its older turns' tool calls and their results left out — the oldest
    turn's first, a turn's all together, only as far as it must — each such turn then its
    question and what the assistant answered: what the memory service's window gives of an
    older turn, and the run store's turns with memory off (``conversation.earlier``)."""
    asked = _asked(request)
    messages = list(request.messages)
    over = _size(request) - budget
    gone: set[int] = set()
    for n, message in enumerate(messages[: asked or 0]):
        if over <= 0 and typed(message):
            break  # the next turn: its calls and results stay
        if isinstance(message, ToolMessage) or (
            isinstance(message, AIMessage) and message.tool_calls
        ):
            gone.add(n)
            over -= count_tokens_approximately([message])
    if not gone:
        return request
    return request.override(messages=[m for n, m in enumerate(messages) if n not in gone])


#: The latest turn's steps (a call and its results) :func:`_steps_said` always keeps whole.
KEEP_STEPS: Final = 3
#: What the note in the place of the latest turn's older steps says (:func:`_steps_said`),
#: then a line of each step's calls: ``- name(arguments) -> the start of its result``, each part
#: at most :data:`STEP_LINE` characters.
STEPS_LEFT: Final = (
    "The {count} earliest steps of this turn, left out to fit the window: each call made and "
    "the start of its result (the latest steps follow whole){unlisted}:"
)
STEPS_UNLISTED: Final = "; the first {count} of them not listed"
#: How many steps a note of :func:`_steps_said` stands for (its first group).
STEPS_LEFT_COUNT: Final = re.compile(r"The (\d+) earliest steps of this turn, left out to fit")
STEP_LINE: Final = 100
#: The source of the note's notification metadata: no message of the user's (:func:`typed`).
STEPS_SOURCE: Final = "trellis_fit_window"


def _steps_said(request: ModelRequest[Any], budget: int) -> ModelRequest[Any]:
    """The request with its latest turn's older steps — each a model's call and its results —
    left out until it fits, the oldest first, never the latest :data:`KEEP_STEPS`, and said in
    one note in their place after the question (:data:`STEPS_LEFT`): a line for each call, its
    name and arguments and the start of its result, the earliest lines left out too when the
    lines alone do not fit. A turn of many small steps (200 lookups) is over a small window by
    itself, where no result is long enough to cut and no older turn is there to leave out: it
    was sent whole, over the window. The note is a notification (no question of the user's),
    so the turn still starts at the question."""
    from langchain_core.messages import add_notification_metadata  # noqa: PLC0415

    asked = _question(request)
    if asked is None:
        return request
    messages = list(request.messages)
    groups: list[list[int]] = []
    n = asked + 1
    while n < len(messages) and _calls_tools(messages[n]):
        group = [n]
        n += 1
        while n < len(messages) and isinstance(messages[n], ToolMessage):
            group.append(n)
            n += 1
        groups.append(group)
    # the note's own head, its role and what a message adds besides (Deep Agents' count)
    over = _size(request) - budget + _tokens_of([STEPS_LEFT, STEPS_UNLISTED, "user"]) + 3
    lines: list[str] = []
    gone: set[int] = set()
    left = 0
    for group in groups[:-KEEP_STEPS]:
        if over <= 0:
            break
        said = _step_lines(messages, group)
        lines += said
        gone.update(group)
        left += 1
        over -= count_tokens_approximately([messages[i] for i in group]) - _tokens_of(said)
    if not left:
        return request
    unlisted = 0
    while over > 0 and unlisted < len(lines):  # the lines alone do not fit: the earliest go
        over -= _tokens_of(lines[unlisted : unlisted + 1])
        unlisted += 1
    head = STEPS_LEFT.format(
        count=left,
        unlisted=STEPS_UNLISTED.format(count=unlisted) if unlisted else "",
    )
    note = add_notification_metadata(
        HumanMessage(content="\n".join([head, *lines[unlisted:]])), source=STEPS_SOURCE
    )
    kept = [m for i, m in enumerate(messages) if i not in gone]
    return request.override(messages=[*kept[: asked + 1], note, *kept[asked + 1 :]])


def _calls_tools(message: AnyMessage) -> bool:
    return isinstance(message, AIMessage) and bool(message.tool_calls)


def _step_lines(messages: Sequence[AnyMessage], group: Sequence[int]) -> list[str]:
    """A step's calls, a line each: its name and arguments, and the start of its result."""
    call_message = cast(AIMessage, messages[group[0]])
    results = {cast(ToolMessage, messages[i]).tool_call_id: messages[i].text for i in group[1:]}
    lines = []
    for call in call_message.tool_calls:
        args = _clipped(json.dumps(call["args"], separators=(",", ":"), default=str))
        lines.append(f"- {call['name']}({args}) -> {_clipped(results.get(call['id'] or '', ''))}")
    return lines


def _clipped(text: str) -> str:
    """``text`` on one line, at most :data:`STEP_LINE` characters."""
    flat = " ".join(text.split())
    return flat if len(flat) <= STEP_LINE else f"{flat[: STEP_LINE - 1]}…"


def _tokens_of(lines: Sequence[str]) -> int:
    """What ``lines`` add to a request, counted as :func:`_size` counts it."""
    return sum(len(line) + 1 for line in lines) // 4 + 1


def todos_shown(request: ModelRequest[Any]) -> ModelRequest[Any]:
    """The request with the todo list the graph's state keeps (``todos``: LangChain's
    ``TodoListMiddleware``) said at the start of its latest question (:data:`TODOS_SHOWN`)
    when none of its messages is a ``write_todos`` call — the call that wrote it left out to
    fit the window (:class:`FitWindow`), a summary in its place, or the earlier turns given as
    text (a graph with no thread of its own): the model reads the list only in that call. Said
    once; only the request changes, never the conversation the graph keeps."""
    todos = request.state.get("todos")
    messages = list(request.messages)
    at = max((n for n, m in enumerate(messages) if typed(m)), default=None)
    if not todos or at is None or _TODOS_SHOWN in messages[at].text:
        return request
    if any(isinstance(m, AIMessage) and _wrote_todos(m) for m in messages):
        return request
    listed = "\n".join(f"- [{t.get('status', 'pending')}] {t.get('content', '')}" for t in todos)
    question = messages[at]
    said = (
        question.content
        if isinstance(question.content, list)
        else [{"type": "text", "text": question.content}]
    )
    told = [{"type": "text", "text": TODOS_SHOWN.format(todos=listed)}, *said]
    messages[at] = question.model_copy(update={"content": told})
    return request.override(messages=messages)


def _wrote_todos(message: AIMessage) -> bool:
    return any(call["name"] == "write_todos" for call in message.tool_calls)


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


def _cut(request: ModelRequest[Any], budget: int, *, latest: bool = False) -> ModelRequest[Any]:
    """The longest tool results cut to their head and tail until the request (``latest``: its
    latest turn, :func:`_latest`, whose results alone are cut) fits — inside a run, each kept
    whole first (``tools.results``) and its cut saying where the rest is: the ``read_file`` call
    that reads on from the line the cut starts on (offered with it), so no row of it is out of
    the model's reach. A kept table's note (``results.noted``: its path and the table to query)
    stays whole after the cut: without it, the model guessed a table name and read another
    result's rows; a result with rows and no note is given one (:func:`_rows_said`). A mark
    (:func:`_mark`) is no result: it is shortened as one (``results.shortened``), never kept —
    cut as a result, a cleared result's mark was kept as one, and "the rest" of it was the
    mark's own words."""
    tools = list(request.tools)
    runtime = current()
    inside = runtime is not None and any(isinstance(m, ToolMessage) for m in request.messages)
    if inside:  # what is cut is held: read_file is offered with it
        tools += _readers(runtime, request.state, {_named(t) for t in tools}, held=False)
    messages = list(request.messages)
    measured = request.override(tools=tools)
    start = 0
    if latest:
        measured = _latest(measured)
        start = _asked(request) or 0
    over = _size(measured) - budget
    for n in sorted(
        (n for n, m in enumerate(messages) if n >= start and isinstance(m, ToolMessage)),
        key=lambda n: -len(messages[n].text),
    ):
        if over <= 0:
            break
        if _mark(messages[n]) or messages[n].name == READ_FILE:
            # no result: a mark is shortened as a mark, a page read of a result as a page
            # (where to read on in the file it read) — nothing of either is kept
            whole = messages[n].text
            room = max(200, len(whole) - over * 4 - CUT_MARKER)
            if _mark(messages[n]):
                less = results.shortened(whole, room)
            else:  # a page: its whole lines and where to read on, else its start as it stands
                less = _page_cut(messages, n, room) or _start(whole, room)
            messages[n] = messages[n].model_copy(update={"content": less})
            over -= (len(whole) - len(less)) // 4
            continue
        text, note = results.noted(messages[n].text)
        # tokens to characters, as the count does, and room for the marker said in its place
        keep = max(200, len(text) - over * 4 - CUT_MARKER)
        if keep >= len(text):
            continue
        half = keep // 2
        said = f"{len(text) - keep} characters cut to fit the window"
        # a kept table's rows are kept already, at the path its note names: the same copy
        path = results.keep(text, tool=messages[n].name) if inside else None
        if path is not None and runtime is not None:
            line = results.line_at(text, half)
            said += f'; the rest: {results.READ_FILE}(file_path="{path}", offset={line})'
            # no note of its own (fewer rows than a table read whole is kept at): its rows'
            note = note or _rows_said(path, text, query.tables(runtime)[path])
            if note:  # rows: query_result too, which the note names
                tools += _readers(runtime, request.state, {_named(t) for t in tools}, held=True)
        cut = f"{text[:half]}\n…[{said}]…\n{text[-half:]}{note}"
        messages[n] = messages[n].model_copy(update={"content": cut})
        over -= (len(text) + len(note) - len(cut)) // 4
    if messages == list(request.messages):
        return request
    return request.override(messages=messages, tools=tools)


# --------------------------------------------------------------------------- checkpoints
class CheckpointUnreadable(HarnessError):
    """A graph checkpoint :class:`RunCheckpointer` kept that cannot be read back (damaged, or
    written with types this release cannot load): the run fails with it, and is not run again
    — starting the graph over would make again the calls its checkpoint had moved past."""

    code = "CHECKPOINT_UNREADABLE"


class RewindRefused(HarnessError):
    """A harness run's graph was continued from an earlier checkpoint than its latest
    (LangGraph's ``update_state`` or a ``checkpoint_id`` on the run's own thread, from code
    inside the run): refused — the run's history is read (``get_state_history``), never forked."""

    code = "REWIND_REFUSED"


#: The thread whose checkpoints a task reads outside the thread's run, and where they are
#: (:meth:`RunCheckpointer.bound`).
_BOUND: ContextVar[tuple[str, dict[str, Any]] | None] = ContextVar("trellis_bound", default=None)
#: Outside a run, how many threads :class:`RunCheckpointer` keeps (the most recently used).
THREADS_KEPT: Final = 10_000


class RunCheckpointer(BaseCheckpointSaver[int]):
    """A checkpointer that keeps every checkpoint, as LangGraph's own savers do (its
    ``get_state_history``, ``update_state`` and a thread continued from a ``checkpoint_id``
    work on it as on ``InMemorySaver``) — and, in a harness run whose ``thread_id`` is the
    run (``LangGraphAdapter`` sees to it), keeps them in the run's journal (``Journal.graph``),
    saved with the run's progress and with its pause, until the run ends (the run store
    clears a run's checkpoint with its ending). A resume in any worker continues where the
    graph stopped: the model is not asked
    again, a summary is not written again, a tool that finished beside a pause is not run
    again. Outside a run it is an in-memory checkpointer.

    A namespace's record is its latest checkpoint — its pending writes, and each delta
    channel's last snapshot with the writes applied since (``deltas``: LangGraph's
    ``DeltaChannel``, Deep Agents' ``messages`` and ``files``, is stored in a checkpoint only
    every so many updates and rebuilt from the writes in between) — and its ``history``: every
    earlier checkpoint, oldest first, each with the writes applied after it. A delta channel
    at an earlier checkpoint is rebuilt from that history (:meth:`_delta_history`).

    As the root graph's checkpoint moves past a step, the journal's records of that step's
    calls go (``Replay.passed``): the graph does not make them again. Inside a run, a
    checkpoint that follows an earlier one than the latest — a fork from the run's own code —
    is refused (:class:`RewindRefused`): a run is rewound from outside it, into a new run
    that asks a person about the writes it would repeat (``Agent.rewind``: :meth:`rewound`,
    :meth:`bound`). A checkpoint that
    cannot be read back fails the run (:class:`CheckpointUnreadable`). The thread ends with
    the run; what its conversation carries to its next turn
    (``adapters.langgraph.CARRIED``) is read from each checkpoint as it is kept
    (``Runtime.carried``)."""

    def __init__(self) -> None:
        super().__init__(serde=STRICT_SERDE)
        #: outside a run: the threads' latest checkpoints, the least recently used forgotten
        #: past the bound (an in-memory checkpointer; inside a run, the run's journal)
        self._threads: LRU[str, dict[str, Any]] = LRU(THREADS_KEPT)

    # ------------------------------------------------------------------ storage
    @contextlib.contextmanager
    def bound(self, thread_id: str, graph: dict[str, Any]) -> Iterator[None]:
        """Inside, ``thread_id``'s checkpoints are ``graph`` — a run's ``Journal.graph``, read
        from the run store — for the task that entered: LangGraph's own API reads a run's
        history (``get_state_history``), or updates the state a rewound run starts from
        (``update_state``), outside any attempt of the run (``bridges.rewind``)."""
        token = _BOUND.set((thread_id, graph))
        try:
            yield
        finally:
            _BOUND.reset(token)

    def rewound(self, graph: Mapping[str, Any], checkpoint_id: str) -> dict[str, Any] | None:
        """``graph`` (a thread's checkpoints, by namespace) as a thread that stopped at its
        root graph's checkpoint ``checkpoint_id``: that one its latest — with none of the
        writes made after it, so the steps that followed it are made again —, the earlier
        ones its history, a sub-graph's as they are. ``None``: no checkpoint it kept."""
        root = graph.get("") or {}
        target = _kept_checkpoints(root).get(checkpoint_id)
        if target is None:
            return None
        names = list(root.get("deltas") or {})
        latest = {
            **{key: target[key] for key in ("id", "checkpoint", "metadata", "parent")},
            "writes": [],
            "deltas": self._delta_history(root, checkpoint_id, names),
            "history": [r for r in root.get("history", ()) if r["id"] < checkpoint_id],
        }
        return {**graph, "": latest}

    def _slot(self, thread_id: str) -> dict[str, Any]:
        """Where the thread's checkpoints are: what the task is :meth:`bound` to, else the
        run's journal, for the run's own thread."""
        bound = _BOUND.get()
        if bound is not None and bound[0] == thread_id:
            return bound[1]
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
        found = _kept_checkpoints(record).get(str(get_checkpoint_id(config) or record.get("id")))
        return None if found is None else self._tuple(thread_id, ns, found)

    def _tuple(self, thread_id: str, ns: str, record: Mapping[str, Any]) -> CheckpointTuple:
        """A kept checkpoint (the latest's record, or one of its ``history``) as LangGraph's."""
        parent = record.get("parent")
        return CheckpointTuple(
            config=_config(thread_id, ns, record["id"]),
            checkpoint=self._loaded(record["checkpoint"]),
            metadata=self._loaded(record["metadata"]),
            pending_writes=[
                (task, channel, self._loaded(value))
                for checkpoint_id, task, _, channel, value, _ in record.get("writes", [])
                if checkpoint_id == record["id"]
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
        """The kept checkpoints, the newest first (LangGraph's order), read back one at a
        time: ``limit`` of them cost ``limit`` reads."""
        threads = [_configurable(config)["thread_id"]] if config else list(self._threads)
        ceiling = str(get_checkpoint_id(before)) if before is not None else None
        found = (
            self._tuple(thread_id, ns, record)
            for thread_id in threads
            for ns, kept in list(self._slot(thread_id).items())
            for checkpoint_id, record in sorted(_kept_checkpoints(kept).items(), reverse=True)
            if ceiling is None or checkpoint_id < ceiling
        )
        matching = (
            t for t in found if all(t.metadata.get(k) == v for k, v in (filter or {}).items())
        )
        yield from itertools.islice(matching, limit)

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
        latest = before.get("id")
        history: list[dict[str, Any]] = before.get("history") or []
        if latest is not None and latest not in (parent, checkpoint["id"]):
            # it follows an earlier checkpoint than the latest: a fork of the thread
            runtime = current()
            if runtime is not None and runtime.run_id == thread_id:
                raise RewindRefused(
                    f"run {thread_id}: its graph continues from its latest checkpoint, not "
                    f"from an earlier checkpoint ({parent}): a run's history is read "
                    "(get_state_history), and a run is not rewound from code inside it"
                )
            followed = _kept_checkpoints(before).get(str(parent)) or {}
            deltas = self._delta_history(before, str(parent), list(before.get("deltas") or {}))
        else:
            followed, deltas = before, before.get("deltas") or {}
        # the writes the new checkpoint applied — those of the one it follows — in the order
        # they were applied (the tasks' paths, LangGraph's ``apply_writes``), not finished in
        applied = sorted(
            (w for w in followed.get("writes", []) if w[0] == parent), key=lambda w: (w[5], w[2])
        )
        if latest is not None and latest != checkpoint["id"]:
            history.append(_archived(before, first=not history))
        slot[ns] = {
            "id": checkpoint["id"],
            "checkpoint": self._dumped(checkpoint),
            "metadata": self._dumped(get_checkpoint_metadata(config, metadata)),
            "parent": parent,
            "writes": [w for w in before.get("writes", []) if w[0] >= checkpoint["id"]],
            "deltas": self._deltas(deltas, checkpoint, metadata, applied),
            "history": history,
        }
        runtime = current()
        if not ns and runtime is not None and runtime.run_id == thread_id:
            runtime.replay.passed(parent)  # the steps it follows are done: not made again
            # what the run's conversation carries to its next turn, as the graph now holds it
            values = checkpoint["channel_values"]
            runtime.carried = carried_plain(values, runtime.carried or {})
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

    def _delta_history(
        self, record: Mapping[str, Any], checkpoint_id: str, channels: Sequence[str]
    ) -> dict[str, Any]:
        """Each of ``channels`` at the kept checkpoint ``checkpoint_id``, as ``deltas`` keeps
        one (its last stored value and the writes applied since): the latest's and the first
        kept one's as kept; any other's from the writes applied after each checkpoint before
        it, back to the nearest that stores the channel whole — or to the first one kept
        (LangGraph's own walk of the parent chain, over what the record holds)."""
        kept = _kept_checkpoints(record)
        target = kept.get(checkpoint_id) or {}
        if "deltas" in target:  # the latest, or the first one kept: as it was rebuilt
            return {name: target["deltas"].get(name) or _EMPTY for name in channels}
        found: dict[str, Any] = {name: {"seed": None, "writes": []} for name in channels}
        left = set(channels)
        cursor = kept.get(str(target.get("parent")))
        while cursor is not None and left:
            applied = sorted(cursor.get("writes", []), key=lambda w: (w[5], w[2]))
            values = self._loaded(cursor["checkpoint"])["channel_values"]
            base = cursor.get("deltas") or {}
            for name in sorted(left):
                added = [[w[1], w[4]] for w in applied if w[0] == cursor["id"] and w[3] == name]
                entry = found[name]
                entry["writes"] = [*added, *entry["writes"]]
                if name in values:  # stored whole: the history starts from it
                    entry["seed"] = self._dumped(values[name])
                elif name in base:  # the first checkpoint kept: what it was rebuilt from
                    entry["seed"] = base[name]["seed"]
                    entry["writes"] = [*base[name]["writes"], *entry["writes"]]
                else:
                    continue
                left.discard(name)
            cursor = kept.get(str(cursor.get("parent")))
        return found

    def get_delta_channel_history(
        self, *, config: RunnableConfig, channels: Sequence[str]
    ) -> Mapping[str, DeltaChannelHistory]:
        """The delta channels' last stored values and the writes since, at a kept checkpoint
        (the latest when the config names none; one not kept: nothing)."""
        thread_id, ns = _where(config)
        record = self._slot(thread_id).get(ns) or {}
        wanted = str(get_checkpoint_id(config) or record.get("id"))
        deltas = (
            self._delta_history(record, wanted, channels)
            if wanted in _kept_checkpoints(record)
            else {}
        )
        found: dict[str, DeltaChannelHistory] = {}
        for name in channels:
            entry = deltas.get(name) or _EMPTY
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
        latest = self._slot(thread_id).setdefault(ns, {})
        # an earlier checkpoint's (the thread continued from it) are kept with it
        record = next((r for r in latest.get("history", ()) if r["id"] == checkpoint_id), latest)
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


#: A delta channel nothing is kept of.
_EMPTY: Final[Mapping[str, Any]] = {"seed": None, "writes": []}


def _kept_checkpoints(record: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    """A namespace's kept checkpoints by id: its ``history`` and its latest (none: a record
    that holds only writes, or nothing)."""
    if record.get("id") is None:
        return {}
    return {r["id"]: r for r in (*record.get("history", ()), record)}


def _archived(record: Mapping[str, Any], *, first: bool) -> dict[str, Any]:
    """The latest checkpoint's ``record`` as its ``history`` keeps it once another follows:
    with the writes applied after it; its ``deltas`` only when it is the ``first`` kept (what
    every later one is rebuilt from: :meth:`RunCheckpointer._delta_history`)."""
    kept = {k: record[k] for k in ("id", "checkpoint", "metadata", "parent")}
    kept["writes"] = [w for w in record.get("writes", []) if w[0] == record["id"]]
    if first:
        kept["deltas"] = record.get("deltas") or {}
    return kept


def _where(config: RunnableConfig) -> tuple[str, str]:
    configurable = _configurable(config)
    return str(configurable["thread_id"]), str(configurable.get("checkpoint_ns", ""))


def _configurable(config: RunnableConfig) -> dict[str, Any]:
    return dict(config.get("configurable") or {})


def _config(thread_id: str, ns: str, checkpoint_id: str) -> RunnableConfig:
    """A kept checkpoint's config: every caller names the checkpoint."""
    return {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": ns,
            "checkpoint_id": checkpoint_id,
        }
    }


__all__ = [
    "CheckpointUnreadable",
    "ContextTooLarge",
    "FitWindow",
    "HarnessTools",
    "ModelHooks",
    "ReadTools",
    "RewindRefused",
    "RunCheckpointer",
    "StallGuard",
    "model_error",
    "step_limit",
]
