"""What ``ReAct`` turns on by itself, from what a run shows — each a LangChain or Deep Agents
middleware used as it is, the harness's code only the trigger and the glue that keeps it
governed, journaled and resumable. A task that does not need one never sees it: a short task's
model calls are what they were before any of these existed (the tools, the instructions).

* **A long task** — :func:`long_task`: once the model has made :data:`LONG_AFTER` calls in this
  invocation, or the request asks for steps (a list of :data:`STEP_ITEMS` items or more, "step
  by step", "a plan") — on a model whose window has room for them (:data:`MIN_WINDOW`) gets:

  * ``planning`` — LangChain's ``TodoListMiddleware`` (:class:`Planning`): the ``write_todos``
    tool and its instructions;
  * ``files`` — Deep Agents' ``FilesystemMiddleware`` with ``ls``, ``write_file`` and
    ``edit_file`` besides ``read_file`` (:class:`ScratchFiles` offers them), on a
    :class:`JournalBackend`: the files are the graph's state, so the run's journal keeps them
    (a pause, a crash, another worker), each at most :data:`MAX_FILE_CHARS` and all of a run's
    at most :data:`MAX_RUN_CHARS`;
  * ``subagent_task`` — Deep Agents' ``SubAgentMiddleware`` (:class:`Helpers`): the ``task``
    tool, a general-purpose helper doing a part of the work in its own context. Each call is a
    child run of the same agent (as ``agent.as_tool()`` makes one: governed, journaled, its
    pause the parent's, cancelled with it), without ``subagent_task`` itself — a helper starts
    no helper — and at most :data:`MAX_HELPERS` at once per run.

``without={"planning"}`` (``"files"``, ``"subagent_task"``) turns one off, for an agent or a
run: the model is not offered it, and a call of it is refused as not available.
"""

from __future__ import annotations

import asyncio
import re
import weakref
from collections.abc import Awaitable, Callable, Collection
from typing import Annotated, Any, Final, NotRequired

from deepagents.backends import StateBackend
from deepagents.backends.protocol import EditResult, WriteResult
from deepagents.backends.utils import file_data_to_string
from deepagents.middleware import FilesystemMiddleware
from deepagents.middleware.filesystem import _file_data_reducer
from deepagents.middleware.subagents import SubAgentMiddleware
from langchain.agents.middleware import AgentMiddleware, AgentState, ModelRequest, ModelResponse
from langchain.agents.middleware.todo import TodoListMiddleware
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

from trellis.contracts import ToolStatus
from trellis.harness.features import Feature
from trellis.harness.middleware import _named
from trellis.harness.runtime import Runtime, current
from trellis.harness.tools import bridge
from trellis.harness.tools.convert import text_of

#: The smallest context window the long-task tools are offered in, in tokens: together they
#: cost about 2,600 tokens of instructions and definitions on every call, which a smaller
#: window cannot spare.
MIN_WINDOW: Final = 32_000
#: Model calls in one invocation after which a task is long (the next call gets the tools).
LONG_AFTER: Final = 3
#: List items in the request from which it asks for steps.
STEP_ITEMS: Final = 3
_ITEM: Final = re.compile(r"(?m)^\s*(?:\d{1,2}[.)]|[-*•])\s+\S")
_CUE: Final = re.compile(r"\bstep[- ]by[- ]step\b|\bmulti-?step\b|\b(?:a|the|your) plan\b", re.I)
#: ``write_todos``: the planning tool.
TODOS: Final = "write_todos"
#: The scratch file tools a long task is offered (``read_file`` reads paged results anyway).
SCRATCH: Final = frozenset({"ls", "write_file", "edit_file"})
#: The helper tool, and its one kind of helper.
TASK: Final = "task"
HELPER: Final = "general-purpose"
HELPER_DESCRIPTION: Final = (
    "A general-purpose helper with the same tools as you, working in a context of its own: give "
    "it one self-contained part of a long task (what to find or do, and what to answer with); "
    "it answers when it is done. Several parts that do not depend on each other can go to "
    "helpers at once."
)
#: Helpers one run has working at once (more wait their turn).
MAX_HELPERS: Final = 4
#: A file the model writes holds at most this many characters; all of a run's at most
#: :data:`MAX_RUN_CHARS`. What the harness saves itself (a large result, a summarized
#: history) is not counted.
MAX_FILE_CHARS: Final = 100_000
MAX_RUN_CHARS: Final = 1_000_000
RESERVED: Final = ("/large_tool_results/", "/conversation_history/")
#: What a call of a feature the run is without reads.
NOT_AVAILABLE: Final = "{tool} is not available in this run"


def on(feature: Feature) -> bool:
    """Whether ``feature`` is on: the run (and its agent) is not without it; outside a run,
    always."""
    runtime = current()
    return runtime is None or runtime.uses(feature)


def calls_made(request: ModelRequest[Any]) -> int:
    """The model calls of this invocation so far (the steps since the user's message)."""
    made = 0
    for message in reversed(_history(request)):
        if isinstance(message, HumanMessage):
            break
        made += isinstance(message, AIMessage)
    return made


def asks_for_steps(request: ModelRequest[Any]) -> bool:
    """Whether the user's latest message asks for steps: a list of :data:`STEP_ITEMS` items or
    more, or says so ("step by step", "multi-step", "a plan"). The graph's state is read, not
    the request: a middleware's own message to the model (a check) is not the user's."""
    asked = next((m.text for m in reversed(_history(request)) if isinstance(m, HumanMessage)), "")
    return len(_ITEM.findall(asked)) >= STEP_ITEMS or bool(_CUE.search(asked))


def _history(request: ModelRequest[Any]) -> list[Any]:
    return list((request.state or {}).get("messages", []))


def long_task(request: ModelRequest[Any]) -> bool:
    """Whether the task is long: :data:`LONG_AFTER` model calls made, or steps asked for."""
    return calls_made(request) >= LONG_AFTER or asks_for_steps(request)


class _LongTasks:
    """What the long-task middleware share: the model's context window (``None``: not known,
    taken as large enough), and whether this call gets their tools."""

    window: int | None = None

    def wanted(self, feature: Feature, request: ModelRequest[Any]) -> bool:
        roomy = self.window is None or self.window >= MIN_WINDOW
        return roomy and on(feature) and long_task(request)


def without_tools(request: ModelRequest[Any], names: Collection[str]) -> ModelRequest[Any]:
    """The request without the tools ``names`` (the same request when it offers none)."""
    kept = [t for t in request.tools if _named(t) not in names]
    return request if len(kept) == len(request.tools) else request.override(tools=kept)


def refused(call: dict[str, Any]) -> ToolMessage:
    """The answer to a call of a tool of a feature the run is without."""
    return ToolMessage(
        content=NOT_AVAILABLE.format(tool=call["name"]),
        tool_call_id=str(call["id"]),
        name=call["name"],
        status="error",
    )


# --------------------------------------------------------------------------- planning
class Planning(_LongTasks, TodoListMiddleware):
    """LangChain's ``TodoListMiddleware``, offered on a long task (``planning``): until then
    the model is sent neither ``write_todos`` nor its instructions. Named as LangChain's, so a
    ``TodoListMiddleware`` given to ``ReAct(middleware=)`` replaces it (always on)."""

    def __init__(self, *, window: int | None = None) -> None:
        super().__init__()
        self.window = window

    @property
    def name(self) -> str:
        return TodoListMiddleware.__name__

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> Any:
        if self.wanted("planning", request):
            return await super().awrap_model_call(request, handler)
        return await handler(without_tools(request, {TODOS}))

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        if request.tool_call["name"] == TODOS and not on("planning"):
            return refused(dict(request.tool_call))
        return await handler(request)


# --------------------------------------------------------------------------- files
class JournalBackend(StateBackend):
    """Deep Agents' ``StateBackend`` — the files are the graph's state, which a ``ReAct`` run
    keeps in its journal (``RunCheckpointer``): they survive a pause, a crash and a move to
    another worker — with the model's files capped: one at most ``max_file`` characters, all
    of the run's at most ``max_run``. What the harness saves itself (:data:`RESERVED`: a large
    result, a summarized history) is not capped here."""

    def __init__(self, *, max_file: int = MAX_FILE_CHARS, max_run: int = MAX_RUN_CHARS) -> None:
        super().__init__()
        self.max_file = max_file
        self.max_run = max_run

    def _too_large(self, file_path: str, size: int) -> str | None:
        if file_path.startswith(RESERVED):
            return None
        if size > self.max_file:
            return (
                f"Error: {file_path} would hold {size} characters; a file holds at most "
                f"{self.max_file}. Write it in parts, or keep only what is needed."
            )
        others = sum(
            len(file_data_to_string(data))
            for path, data in self._read_files().items()
            if path != file_path and data is not None and not path.startswith(RESERVED)
        )
        if others + size > self.max_run:
            return (
                f"Error: this run's files would hold {others + size} characters; a run's "
                f"files hold at most {self.max_run}. Edit or shorten the files already written."
            )
        return None

    def write(self, file_path: str, content: str) -> WriteResult:
        problem = self._too_large(file_path, len(content))
        return WriteResult(error=problem) if problem else super().write(file_path, content)

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        data = self._read_files().get(file_path)
        if data is not None and old_string:
            text = file_data_to_string(data)
            edited = text.replace(old_string, new_string, -1 if replace_all else 1)
            problem = self._too_large(file_path, len(edited))
            if problem:
                return EditResult(error=problem)
        return super().edit(file_path, old_string, new_string, replace_all)


class FilesState(AgentState[Any]):
    """The graph's files, merged by Deep Agents' own reducer as a plain channel: Deep Agents
    keeps them in a ``DeltaChannel``, which needs every earlier checkpoint — the run's
    checkpointer keeps the latest only (``RunCheckpointer``), so a resume would find no
    files."""

    files: Annotated[NotRequired[dict[str, Any]], _file_data_reducer]


class JournalFiles(FilesystemMiddleware):
    """Deep Agents' ``FilesystemMiddleware`` on a :class:`JournalBackend`, its files in a
    channel the run's journal keeps whole (:class:`FilesState`): a large result saved there, a
    summarized history and the scratch files are all there after a pause, a crash or a move to
    another worker. ``tools``: the file tools it has (``ReAct``: ``read_file`` and the scratch
    tools :class:`ScratchFiles` offers). Named as Deep Agents', so a ``FilesystemMiddleware``
    given to ``ReAct(middleware=)`` replaces it."""

    def __init__(self, backend: JournalBackend, *, tools: list[Any]) -> None:
        super().__init__(backend=backend, tools=tools)
        self.state_schema = FilesState  # type: ignore[assignment]

    @property
    def name(self) -> str:
        return FilesystemMiddleware.__name__


class ScratchFiles(_LongTasks, AgentMiddleware):
    """``ls``, ``write_file`` and ``edit_file`` (Deep Agents' ``FilesystemMiddleware``, which
    ``ReAct`` builds with them) offered on a long task (``files``); refused without it."""

    def __init__(self, *, window: int | None = None) -> None:
        super().__init__()
        self.window = window

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        if self.wanted("files", request):
            return await handler(request)
        return await handler(without_tools(request, SCRATCH))

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        if request.tool_call["name"] in SCRATCH and not on("files"):
            return refused(dict(request.tool_call))
        return await handler(request)


# --------------------------------------------------------------------------- helpers
def _unused(_: Any) -> Any:  # pragma: no cover - Helpers runs the task tool itself
    raise RuntimeError("the task tool runs as a child run (Helpers)")


class Helpers(_LongTasks, SubAgentMiddleware):
    """Deep Agents' ``SubAgentMiddleware``: the ``task`` tool, offered on a long task
    (``subagent_task``), with one general-purpose helper. A call runs as a child run of the
    run's own agent through the harness's tool bridge (``agent.as_tool()``'s machinery:
    governed, journaled, resumed from its journal after a crash, its pause the parent's,
    cancelled with the parent), without ``subagent_task`` (a helper calls no helper), at most
    ``concurrency`` at once per run. Named as Deep Agents', so a ``SubAgentMiddleware`` given
    to ``ReAct(middleware=)`` replaces it."""

    def __init__(
        self, backend: Any, *, concurrency: int = MAX_HELPERS, window: int | None = None
    ) -> None:
        super().__init__(
            backend=backend,
            subagents=[
                {
                    "name": HELPER,
                    "description": HELPER_DESCRIPTION,
                    "runnable": RunnableLambda(_unused),
                }
            ],
        )
        self.concurrency = concurrency
        self.window = window
        self._slots: weakref.WeakKeyDictionary[Runtime, asyncio.Semaphore] = (
            weakref.WeakKeyDictionary()
        )

    @property
    def name(self) -> str:
        return SubAgentMiddleware.__name__

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        if self.wanted("subagent_task", request):
            return await super().awrap_model_call(request, handler)
        return await handler(without_tools(request, {TASK}))

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        call = request.tool_call
        if call["name"] != TASK:
            return await handler(request)
        runtime = current()
        if runtime is None or not runtime.uses("subagent_task"):
            return refused(dict(call))
        from trellis.harness.subagents import SubAgent  # noqa: PLC0415 - imports adapters

        [helper] = await SubAgent(
            runtime.agent,
            name=TASK,
            description=HELPER_DESCRIPTION,
            side_effects=None,
            without={"subagent_task"},
        ).resolve()
        slots = self._slots.setdefault(runtime, asyncio.Semaphore(self.concurrency))
        async with slots:
            asked = {"message": str(call["args"].get("description", ""))}
            outcome = await bridge.call(helper, asked, call_id=call["id"])
        return ToolMessage(
            content=text_of(outcome.output),
            tool_call_id=str(call["id"]),
            name=TASK,
            status="error"
            if outcome.status in (ToolStatus.ERROR, ToolStatus.TIMEOUT)
            else "success",
        )
