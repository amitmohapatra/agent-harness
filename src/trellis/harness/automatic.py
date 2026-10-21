"""What ``ReAct`` turns on by itself, from what a run shows — each a LangChain or Deep Agents
middleware used as it is, the harness's code only the trigger and the glue that keeps it
governed, journaled and resumable. A task that does not need one never sees it: a short task's
model calls are what they were before any of these existed (the tools, the instructions).

* **A long task** — :func:`long_task`: once the model has made :data:`LONG_AFTER` calls in this
  invocation, or the request asks for steps (a list of :data:`STEP_ITEMS` items or more, "step
  by step", "a plan") — on a model whose window is known and has room for them
  (:data:`MIN_WINDOW`; an unknown window: off) gets:

  * ``planning`` — LangChain's ``TodoListMiddleware`` (:class:`Planning`): the ``write_todos``
    tool and its instructions;
  * ``files`` — Deep Agents' ``FilesystemMiddleware`` with ``ls``, ``write_file`` and
    ``edit_file`` besides ``read_file`` (:class:`ScratchFiles` offers them), on a
    :class:`JournalBackend`: the files are the graph's state, so the run's journal keeps them
    (a pause, a crash, another worker), each at most :data:`MAX_FILE_CHARS` and all of a run's
    at most :data:`MAX_RUN_CHARS`.

* **A budget** given (``h.wrap(budget=)``) — :class:`Budgeted`: once it is spent, one more call
  without tools for the best answer with what the run has.

``without={"planning"}`` (``"files"``) turns one off, for an agent or a run: the model is not
offered it, and a call of it is refused as not available.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable, Collection, Sequence
from typing import Annotated, Any, Final, NotRequired

from deepagents.backends import StateBackend
from deepagents.backends.protocol import EditResult, WriteResult
from deepagents.backends.utils import file_data_to_string
from deepagents.middleware import FilesystemMiddleware
from deepagents.middleware.filesystem import _file_data_reducer
from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
)
from langchain.agents.middleware.todo import TodoListMiddleware
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

from trellis.harness import budget
from trellis.harness.features import Feature
from trellis.harness.middleware import _named
from trellis.harness.runtime import current

log = logging.getLogger("trellis.react")

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


def _history(request: ModelRequest[Any]) -> Sequence[Any]:
    return (request.state or {}).get("messages", [])


def long_task(request: ModelRequest[Any]) -> bool:
    """Whether the task is long: :data:`LONG_AFTER` model calls made, or steps asked for."""
    return calls_made(request) >= LONG_AFTER or asks_for_steps(request)


class PromptCache:
    """Whether the model's prompt cache is seen to work: a reply of the graph's model reported
    cached prompt tokens. From then on the long-task tools are offered from the first call
    (``stable_prefix``): the tools are near the front of the prompt, and adding them at a
    later step makes the cache miss everything after them."""

    def __init__(self) -> None:
        self.seen = False

    def note(self, reply: ModelResponse[Any]) -> None:
        """Note what ``reply`` says of the cache."""
        if self.seen:
            return
        message = next((m for m in reversed(reply.result) if isinstance(m, AIMessage)), None)
        usage = (message.usage_metadata if message is not None else None) or {}
        if (usage.get("input_token_details") or {}).get("cache_read"):
            self.seen = True
            log.info("the model's prompt cache works: the long-task tools from the first call")

    def holds(self) -> bool:
        """Whether the long-task tools go from the first call: the cache works, and
        ``stable_prefix`` is on."""
        return self.seen and on("stable_prefix")


class _LongTasks:
    """What the long-task middleware share: the model's context window (``None``: not known —
    no room is assumed, so they are off), the model's prompt cache, and whether this call gets
    their tools."""

    window: int | None = None
    cache: PromptCache

    @property
    def roomy(self) -> bool:
        return self.window is not None and self.window >= MIN_WINDOW

    def wanted(self, feature: Feature, request: ModelRequest[Any]) -> bool:
        return self.roomy and on(feature) and (self.cache.holds() or long_task(request))

    def noted(self, reply: ModelResponse[Any]) -> ModelResponse[Any]:
        """``reply``, its prompt cache noted (on a window with room for the tools)."""
        if self.roomy:
            self.cache.note(reply)
        return reply


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


# --------------------------------------------------------------------------- the budget
class Budgeted(AgentMiddleware):
    """The run's budget (``budget``; ``trellis.harness.budget``; ``ModelHooks`` counts what
    every model reply spent): once it — the run's, or a run it works inside — is spent, one
    more call with no tool, told to answer with what it has (``SPENT``); the run then ends
    ``PARTIAL`` (``BUDGET_EXCEEDED``)."""

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        runtime = current()
        spent = budget.reached(runtime) if runtime is not None else None
        if runtime is not None and spent is not None:
            budget.stopped(runtime, spent)
            told = [*request.messages, HumanMessage(content=budget.SPENT)]
            request = request.override(
                messages=told, tools=[] if request.response_format is not None else request.tools
            )
            if request.tools:
                request = request.override(tool_choice="none")
        return await handler(request)


# --------------------------------------------------------------------------- planning
class Planning(_LongTasks, TodoListMiddleware):
    """LangChain's ``TodoListMiddleware``, offered on a long task (``planning``): until then
    the model is sent neither ``write_todos`` nor its instructions. Named as LangChain's, so a
    ``TodoListMiddleware`` given to ``ReAct(middleware=)`` replaces it (always on)."""

    def __init__(self, *, window: int | None = None, cache: PromptCache | None = None) -> None:
        super().__init__()
        self.window = window
        self.cache = cache or PromptCache()

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
        return self.noted(await handler(without_tools(request, {TODOS})))

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

    def __init__(self, *, window: int | None = None, cache: PromptCache | None = None) -> None:
        super().__init__()
        self.window = window
        self.cache = cache or PromptCache()

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        if self.wanted("files", request):
            return await handler(request)
        return self.noted(await handler(without_tools(request, SCRATCH)))

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        if request.tool_call["name"] in SCRATCH and not on("files"):
            return refused(dict(request.tool_call))
        return await handler(request)
