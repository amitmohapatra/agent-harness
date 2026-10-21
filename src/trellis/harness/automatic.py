"""What ``ReAct`` turns on by itself, from what a run shows — each a LangChain or Deep Agents
middleware used as it is, the harness's code only the trigger and the glue that keeps it
governed, journaled and resumable. A task that does not need one never sees it: a short task's
model calls are what they were before any of these existed (the tools, the instructions).

Each is one trigger (a small function of the request or the run) and the LangChain or Deep
Agents part it turns on. For every task: the vendor's own client (``native_client``), each
step's effort (``effort``), the prompt cache (``prompt_cache``), the gateway's fallbacks
(``fallback``), the run's budget (``budget``) and one repair of a structured answer
(``structured_output``).

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
from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
)
from langchain.agents.middleware.todo import TodoListMiddleware
from langchain.agents.structured_output import StructuredOutputValidationError
from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

from trellis.contracts import ConfigurationError, ModelError, ToolStatus
from trellis.harness import budget, models
from trellis.harness.features import Feature
from trellis.harness.middleware import ANSWER_CHECK, WRITE_CHECK, _named
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
#: How the checks ``AnswerCheck`` asks begin (a check is a deciding step).
_CHECKS: Final = (ANSWER_CHECK.split(".")[0], WRITE_CHECK.split("{")[0])
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


# --------------------------------------------------------------------------- the model
class NativeClient(AgentMiddleware):
    """A model named by its vendor's name asked through the vendor's own API on the gateway
    (``native_client``: ``ChatAnthropic`` on the gateway's Anthropic endpoint, ``ChatOpenAI`` on
    its Responses API), not the generic chat completions the graph was built with: the
    vendor's caching, effort and tool search then work. Without it, the generic client."""

    def __init__(self, native: Any) -> None:
        super().__init__()
        self.native = native

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        if on("native_client"):
            return await handler(request.override(model=self.native))
        return await handler(request)


#: The effort of a routine step, and of a step that decides more (``effort``).
ROUTINE: Final = "low"
DECIDING: Final = "high"


class Effort(AgentMiddleware):
    """How hard a model that reasons thinks, per step (``effort``): :data:`DECIDING` for the
    steps that decide the run — its first call (the plan), the step after a tool failed, a
    check of an answer or of a change before it is made — :data:`ROUTINE` for the others.
    ``level`` fixes it for every step. A model that takes no effort is sent none (a local
    model: nothing changes); a level it does not take is the nearest it does."""

    def __init__(self, level: str | None = None) -> None:
        super().__init__()
        if level is not None and level not in models.EFFORTS:
            raise ConfigurationError(f"effort is one of {', '.join(models.EFFORTS)}")
        self.level = level

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        levels = models.effort_levels(request.model)
        if not levels or not on("effort"):
            return await handler(request)
        level = models.nearest(self.level or step_effort(request), levels)
        settings = {**request.model_settings, "reasoning_effort": level}
        return await handler(request.override(model_settings=settings))


def step_effort(request: ModelRequest[Any]) -> str:
    """:data:`DECIDING` for the run's first call, the step after a failed tool call, and a
    check (an ``AnswerCheck`` message last); :data:`ROUTINE` otherwise."""
    last = request.messages[-1] if request.messages else None
    checking = isinstance(last, HumanMessage) and last.text.startswith(_CHECKS)
    failed = isinstance(last, ToolMessage) and last.status == "error"
    return DECIDING if checking or failed or calls_made(request) == 0 else ROUTINE


# --------------------------------------------------------------------------- spending
class PromptCache(AnthropicPromptCachingMiddleware):
    """LangChain's ``AnthropicPromptCachingMiddleware`` on the calls that reach an Anthropic
    model through its own API (``native_client``): the conversation's prefix — the
    instructions, the tools (sorted), the memory context, the history — is marked for the
    prompt cache (``prompt_cache``). Every other model is left as it is: OpenAI's caching is
    automatic, and the order is the same for it. Named as LangChain's, so one given to
    ``ReAct(middleware=)`` replaces it."""

    def __init__(self) -> None:
        super().__init__(unsupported_model_behavior="ignore")

    @property
    def name(self) -> str:
        return AnthropicPromptCachingMiddleware.__name__

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> Any:
        if not on("prompt_cache"):
            return await handler(request)
        return await super().awrap_model_call(request, handler)


class GatewayFallbacks(AgentMiddleware):
    """``ReAct(fallback_models=)``: sent with every model call as Bifrost's own ``fallbacks``
    (``provider/model``), so the gateway — the one place that falls back, as it does for the
    fallbacks configured on the virtual key or the provider — tries each in order when the
    model fails (``fallback``)."""

    def __init__(self, names: Collection[str]) -> None:
        super().__init__()
        self.names = list(names)

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> Any:
        if not on("fallback"):
            return await handler(request)
        settings = dict(request.model_settings)
        settings["extra_body"] = {**(settings.get("extra_body") or {}), "fallbacks": self.names}
        return await handler(request.override(model_settings=settings))


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


class StructuredRepair(AgentMiddleware):
    """An answer that does not fit ``ReAct(output=)``'s schema (``structured_output``): the
    model is told what was wrong and asked once more — by LangChain's tool strategy (its error
    tool message), or here (:data:`REPAIR`) when the provider's strategy refused it — and a
    second answer that does not fit either fails the run, saying why. Without it, LangChain's
    own: the tool strategy asks again until the step limit, a provider's fails at once."""

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        if request.response_format is None or not on("structured_output"):
            return await handler(request)
        if _failed_answers(request) >= 2:
            raise ModelError(
                "the answer did not fit the output schema twice; it was told what was wrong "
                "once and answered wrong again",
                source="langgraph",
            )
        try:
            return await handler(request)
        except StructuredOutputValidationError as exc:
            told = [
                *request.messages,
                exc.ai_message,
                HumanMessage(content=f"{REPAIR}\n{exc.source}"),
            ]
            try:
                return await handler(request.override(messages=told))
            except StructuredOutputValidationError as again:
                raise ModelError(
                    f"the answer did not fit the output schema twice: {again.source}",
                    source="langgraph",
                ) from again


#: What a model is told when its answer did not fit the output schema.
REPAIR: Final = (
    "That answer does not fit the schema the answer must follow. Answer again so it does:"
)


def _failed_answers(request: ModelRequest[Any]) -> int:
    """The answers of this invocation that the output schema refused (LangChain answers each
    with an error tool message under the output tool's name)."""
    names = _answer_tools(request.response_format)
    count = 0
    for message in _history(request):
        if isinstance(message, HumanMessage):
            count = 0
        elif isinstance(message, ToolMessage) and message.name in names:
            count += message.text.startswith("Error")
    return count


def _answer_tools(response_format: Any) -> set[str]:
    """The name of the tool an answer to ``response_format`` (``ReAct(output=)``: a pydantic
    model or a JSON schema) is given with: its schema's."""
    schema = getattr(response_format, "schema", response_format)
    named = schema.get("title") if isinstance(schema, dict) else getattr(schema, "__name__", None)
    return {named} if isinstance(named, str) else set()


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
