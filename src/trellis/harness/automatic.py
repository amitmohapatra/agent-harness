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
  * ``files`` — Deep Agents' scratch file tools (:data:`SCRATCH`: ``ls``, ``write_file``,
    ``edit_file``, ``delete``, ``glob``, ``grep``) beside ``read_file``, and its
    general-purpose ``task`` helper when the agent has no sub-agents of its own
    (:class:`LongTaskTools` offers them; Deep Agents' ``FilesystemMiddleware`` and
    ``SubAgentMiddleware`` have them all along), on a :class:`JournalBackend`: the files are
    the graph's state, so the run's journal keeps them (a pause, a crash, another worker), each
    at most :data:`MAX_FILE_CHARS` and all of a run's at most :data:`MAX_RUN_CHARS`.

* **Many tools and no tool hints** — :class:`ToolSelection` (off unless a measurement turns it
  on: ``react.defaults_on``): LangChain's ``LLMToolSelectorMiddleware`` picks the tools that
  fit, by one model call of its own, when the run has no hints from memory (off, unreachable
  or cold) and a call offers more than :data:`SELECT_ABOVE` tools.

* **A wide task** — :class:`Delegation`: on a large model (:func:`model_tier`) with a known
  window of :data:`MIN_WINDOW` or more, a request of :data:`WIDE_ITEMS` independent items of
  one kind or more (:func:`wide`: a list of them, or "each / every / compare" over them; not
  steps that build on each other, nor a request to change anything) is offered ``delegate``
  (``trellis.harness.delegate``): helpers that only read, at once, merged for the model to
  check. A helper is not offered it (``delegate.MAX_DEPTH``); a small model never is.

* **A budget** given (``h.wrap(budget=)``) — :class:`Budgeted`: once it is spent, one more call
  without tools for the best answer with what the run has.

"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable, Collection, Sequence
from typing import Any, Final, cast

from deepagents.backends import StateBackend
from deepagents.backends.protocol import EditResult, WriteResult
from deepagents.backends.utils import file_data_to_string
from langchain.agents.middleware import (
    AgentMiddleware,
    LLMToolSelectorMiddleware,
    ModelRequest,
    ModelResponse,
)
from langchain.agents.middleware.todo import TodoListMiddleware
from langchain_core.messages import AIMessage, HumanMessage

from trellis.harness import budget
from trellis.harness.middleware import _named
from trellis.harness.runtime import current

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
#: Deep Agents' scratch file tools a long task is offered (``read_file`` reads saved results
#: anyway: ``ReadTools``).
SCRATCH: Final = frozenset({"ls", "write_file", "edit_file", "delete", "glob", "grep"})
#: Deep Agents' sub-agent tool: its general-purpose helper is a long task's.
TASK: Final = "task"
#: A file the model writes holds at most this many characters; all of a run's at most
#: :data:`MAX_RUN_CHARS`. What the harness saves itself (a large result, a summarized
#: history) is not counted.
MAX_FILE_CHARS: Final = 100_000
MAX_RUN_CHARS: Final = 1_000_000
RESERVED: Final = ("/large_tool_results/", "/conversation_history/")


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
    asked = _latest(request)
    return len(_ITEM.findall(asked)) >= STEP_ITEMS or bool(_CUE.search(asked))


def _latest(request: ModelRequest[Any]) -> str:
    """The user's latest message, from the graph's state."""
    return next((m.text for m in reversed(_history(request)) if isinstance(m, HumanMessage)), "")


def _history(request: ModelRequest[Any]) -> Sequence[Any]:
    return (request.state or {}).get("messages", [])


def long_task(request: ModelRequest[Any]) -> bool:
    """Whether the task is long: :data:`LONG_AFTER` model calls made, or steps asked for."""
    return calls_made(request) >= LONG_AFTER or asks_for_steps(request)


class _LongTasks:
    """What the long-task middleware share: the model's context window (``None``: not known —
    no room is assumed, so they are off), and whether this call gets their tools."""

    window: int | None = None

    def wanted(self, request: ModelRequest[Any]) -> bool:
        roomy = self.window is not None and self.window >= MIN_WINDOW
        return roomy and long_task(request)


def without_tools(request: ModelRequest[Any], names: Collection[str]) -> ModelRequest[Any]:
    """The request without the tools ``names`` (the same request when it offers none)."""
    kept = [t for t in request.tools if _named(t) not in names]
    return request if len(kept) == len(request.tools) else request.override(tools=kept)


# --------------------------------------------------------------------------- tool selection
#: The tools a call offers past which :class:`ToolSelection` selects, and how many it keeps.
SELECT_ABOVE: Final = 20
SELECT_KEEP: Final = 10


class ToolSelection(AgentMiddleware):
    """LangChain's ``LLMToolSelectorMiddleware`` (its own model call picks at most ``keep``
    tools that fit the request; a reply it cannot read offers every tool), used only when the
    run has no tool hints — memory off, unreachable or cold — and a call offers more than
    ``above`` tools; the tools the run already used are kept."""

    def __init__(self, *, above: int = SELECT_ABOVE, keep: int = SELECT_KEEP) -> None:
        super().__init__()
        self.above = above
        self.keep = keep

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        runtime = current()
        hinted = runtime is not None and runtime.offered is not None
        if hinted or len(request.tools) <= self.above:
            return await handler(request)
        used = sorted(runtime.used) if runtime is not None else []
        selector = LLMToolSelectorMiddleware(
            max_tools=self.keep, always_include=used, on_parsing_failure="all"
        )
        # it calls the handler with the tools it picked, and returns its reply
        return cast("ModelResponse[Any]", await selector.awrap_model_call(request, handler))


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
    the model is not sent ``write_todos``. Its guidance travels in the tool's own description
    (LangChain's), never in the instructions: the system prompt stays the same from the first
    call to the last, so a provider's prompt cache keeps its prefix. Named as LangChain's."""

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
        if self.wanted(request):
            return await handler(request)  # the tool, its description its guidance
        return await handler(without_tools(request, {TODOS}))


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


class LongTaskTools(_LongTasks, AgentMiddleware):
    """Deep Agents' tools that only a long task is offered (``tools``: the scratch file tools,
    and ``task`` when its only sub-agent is its general-purpose one): until then the model is
    sent neither them nor what they cost. Deep Agents' ``FilesystemMiddleware`` and
    ``SubAgentMiddleware`` have them all along."""

    def __init__(self, *, window: int | None = None, tools: Collection[str] = SCRATCH) -> None:
        super().__init__()
        self.window = window
        self.held = frozenset(tools)

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        if self.wanted(request):
            return await handler(request)
        return await handler(without_tools(request, self.held))


# --------------------------------------------------------------------------- delegation
#: Independent items of one kind from which a request is wide (fewer are one agent's work).
WIDE_ITEMS: Final = 4
_WIDE_CUE: Final = re.compile(
    r"\b(?:each|every|all (?:of )?(?:the|these|those)|for (?:all|each|every)|compare|across|"
    r"respectively)\b",
    re.I,
)
_SEQUENTIAL: Final = re.compile(
    r"\b(?:then|after that|afterwards|next|finally|step[- ]by[- ]step|in order)\b", re.I
)
_CHANGES: Final = re.compile(
    r"\b(?:create|update|delete|remove|send|book|refund|cancel|pay|transfer|publish|write)\b",
    re.I,
)
_SENTENCE: Final = re.compile(r"[.?!]+(?=\s|$)|\n+")
_PART: Final = re.compile(r",|;|\band\b|\bor\b", re.I)
#: A model named with this many billion parameters or fewer is small (``granite-3.3-2b``,
#: ``llama-3.1-8b``); a mixture of experts is named by its total (``30b-a3b``).
SMALL_PARAMS: Final = 14.0
_PARAMS: Final = re.compile(r"(?<![a-z\d.])(\d+(?:\.\d+)?)b\b", re.I)
#: Model families small-model mode is for, whatever their size says (no helpers at once on
#: Granite until a measurement says they pay).
SMALL_FAMILIES: Final = ("granite",)


def wide(text: str) -> bool:
    """Whether ``text`` asks for :data:`WIDE_ITEMS` independent items of one kind or more: a
    list of them, or an enumeration with a wide cue ("each", "every", "compare", ...) — not
    when it asks for steps in order ("then", "finally"...) or to change something (a write's
    verb: those stay the agent's own, one at a time)."""
    if _SEQUENTIAL.search(text) or _CHANGES.search(text):
        return False
    if len(_ITEM.findall(text)) >= WIDE_ITEMS:
        return True
    return bool(_WIDE_CUE.search(text)) and _enumerated(text) >= WIDE_ITEMS


def _enumerated(text: str) -> int:
    """The most items one sentence of ``text`` lists ("A, B, C and D": 4)."""
    counts = [
        len([p for p in _PART.split(sentence) if p.strip()])
        for sentence in _SENTENCE.split(text)
        if sentence.count(",") + sentence.count(";") >= 2
    ]
    return max(counts, default=0)


def model_tier(name: str | None, window: int | None) -> str:
    """``"small"`` or ``"large"``, from what is known without a model call (tier 0): small
    with a window under :data:`MIN_WINDOW` or not known, a name of a small family
    (:data:`SMALL_FAMILIES`) or of :data:`SMALL_PARAMS` billion parameters or fewer."""
    if window is None or window < MIN_WINDOW:
        return "small"
    named = (name or "").lower()
    if any(family in named for family in SMALL_FAMILIES):
        return "small"
    sizes = [float(size) for size in _PARAMS.findall(named)]
    return "small" if sizes and max(sizes) <= SMALL_PARAMS else "large"


class Delegation(AgentMiddleware):
    """``delegate`` (``trellis.harness.delegate``), offered only to a wide task (:func:`wide`,
    the user's latest message) on a large model (``tier``) — never in a helper's own run: until
    then the model is sent neither the tool nor what it costs. Its jobs' ``subagent_type`` is
    one of ``subagents`` (the agent's wrapped sub-agents)."""

    def __init__(self, *, tier: str, subagents: Sequence[str] = ()) -> None:
        from trellis.harness import delegate  # noqa: PLC0415 - it runs child runs

        super().__init__()
        self.tier = tier
        self.tools = [delegate.tool(subagents)]
        self._depth = delegate.MAX_DEPTH

    def wanted(self, request: ModelRequest[Any]) -> bool:
        runtime = current()
        if self.tier != "large" or runtime is None or runtime.delegated >= self._depth:
            return False
        return wide(_latest(request))

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        if self.wanted(request):
            return await handler(request)
        from trellis.harness.delegate import DELEGATE  # noqa: PLC0415

        return await handler(without_tools(request, {DELEGATE}))
