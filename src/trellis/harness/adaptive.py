"""What ``ReAct`` adds to its loop when a run shows the need — each a library's piece used as
it is (LangChain, Deep Agents, ``json_repair``), the harness's code only the signal that
switches it on and the glue. Without its signal a piece leaves every model call as it was;
``without={<feature>}`` (``trellis.harness.features``) turns it off for an agent or a run.

* ``token_count`` — :class:`ClearOldResults`: the trigger of LangChain's
  ``ContextEditingMiddleware`` counts what the model is really sent — the instructions and
  the tool definitions besides the messages — with ``count_tokens_approximately`` scaled by
  the usage the model reported (:class:`CalibratedCounter`). On a known window.
* ``clear_inputs`` — the same clearing also clears the arguments of the calls whose results
  it clears (``ClearToolUsesEdit(clear_tool_inputs=True)``), when the scratch files are
  offered (their bodies are what is large).
* ``strict_tools`` — :class:`StrictTools`: after the conversation's first malformed call (its
  arguments not JSON, or not fitting the tool's schema), the tools are sent ``strict``.
* ``native_output`` — :class:`NativeOutput`: ``output=`` answered through the provider's
  structured output (LangChain's ``ProviderStrategy(strict=True)``) instead of an answer tool,
  on a model that has it (its profile says ``structured_output``, or a gateway model).
* ``tool_repair`` — :class:`RepairToolCalls`: a call whose arguments are not JSON is repaired
  with ``json_repair``, and a misspelled tool name (case, ``-``/``_``) matched to the one tool
  it can mean, kept only when the repaired arguments fit the tool's schema; else the model is
  asked again as before.
* ``think`` — :class:`ThinkTool`: Anthropic's ``think`` tool, offered to a model that does not
  reason (no ``effort=``) when the call offers a tool that changes data.
* ``unsupported_content`` — Deep Agents' ``UnsupportedContentMiddleware``
  (:class:`UnsupportedContent`): content the model's profile says it cannot take replaced by
  a notice.
* ``rubric`` — Deep Agents' ``RubricMiddleware`` (:class:`Rubric`): a run whose input carries a
  ``rubric`` has its answer graded against it by the run's model, and revised.

``stable_prefix`` is :mod:`trellis.harness.automatic`'s (the long-task tools offered from the
first call once the model's prompt cache is seen to work).
"""

from __future__ import annotations

import asyncio
import json
import logging
import operator
import weakref
from collections.abc import Awaitable, Callable, Sequence
from typing import Annotated, Any, Final, NotRequired

import json_repair
from deepagents.backends import StateBackend
from deepagents.middleware.rubric import RubricMiddleware
from deepagents.middleware.subagents import (
    GENERAL_PURPOSE_SUBAGENT,
    TASK_TOOL_DESCRIPTION,
    CompiledSubAgent,
    SubAgentMiddleware,
)
from deepagents.middleware.unsupported_content import UnsupportedContentMiddleware
from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    ClearToolUsesEdit,
    ContextEditingMiddleware,
    ModelRequest,
    ModelResponse,
)
from langchain.agents.middleware.types import ExtendedModelResponse, PrivateStateAttr
from langchain.agents.structured_output import ProviderStrategy
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import BaseTool, StructuredTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

from trellis.contracts import ConfigurationError
from trellis.harness.automatic import (
    MIN_WINDOW,
    SCRATCH,
    asks_for_steps,
    on,
    refused,
    without_tools,
)
from trellis.harness.bounded import LRU
from trellis.harness.middleware import (
    FAILED,
    RESUMABLE,
    TASK,
    THINK,
    _named,
    _writes_offered,
)
from trellis.harness.runtime import Runtime, current
from trellis.harness.subagents import SubAgent
from trellis.harness.telemetry import LoopEvent, metrics
from trellis.harness.tools import bridge
from trellis.harness.tools.base import FIX_ARGUMENTS, arguments_problem
from trellis.harness.tools.convert import text_of

log = logging.getLogger("trellis.react")

Tools = Sequence[BaseTool | dict[str, Any]]


# --------------------------------------------------------------------------- token count
#: The tool sets whose definitions :class:`CalibratedCounter` keeps as text, at most (the
#: least recently counted is dropped first).
TOOLSETS: Final = 32


class CalibratedCounter:
    """LangChain's ``count_tokens_approximately`` of what a model call is sent — the
    instructions, the tool definitions and the messages — scaled by the usage the model
    reported for the conversation so far (``use_usage_metadata_scaling``). The definitions of
    a tool set are made into text once (``convert_to_openai_tool``, as the model is sent them)
    and kept for the next calls: at most :data:`TOOLSETS` sets."""

    def __init__(self, toolsets: int = TOOLSETS) -> None:
        self._texts: LRU[tuple[str, ...], SystemMessage] = LRU(toolsets)

    def __call__(
        self,
        messages: Sequence[BaseMessage],
        *,
        system: SystemMessage | None = None,
        tools: Tools = (),
    ) -> int:
        head: list[BaseMessage] = [system] if system is not None else []
        if tools:
            head.append(self._definitions(tools))
        return count_tokens_approximately([*head, *messages], use_usage_metadata_scaling=True)

    def _definitions(self, tools: Tools) -> SystemMessage:
        """The tool set's definitions as one message (kept: a tool set is keyed by its tools'
        names, in the order sent)."""
        return self._texts.setdefault(
            tuple(_named(t) for t in tools),
            lambda: SystemMessage(content=json.dumps([convert_to_openai_tool(t) for t in tools])),
        )


class _Editing(ContextEditingMiddleware):
    """LangChain's ``ContextEditingMiddleware``, its trigger counting with ``counter`` while
    ``token_count`` is on (LangChain's own count, of the messages alone, when it is off)."""

    def __init__(self, edit: ClearToolUsesEdit, counter: CalibratedCounter) -> None:
        super().__init__(edits=[edit])
        self.counter = counter

    def _resolve_token_counter(self, request: ModelRequest[Any]) -> Any:
        if not on("token_count"):
            return super()._resolve_token_counter(request)
        system, tools = request.system_message, list(request.tools)
        return lambda messages: self.counter(messages, system=system, tools=tools)


class ClearOldResults(AgentMiddleware):
    """Older tool results cleared from the request past ``trigger`` tokens (LangChain's
    ``ContextEditingMiddleware`` with ``edit``), counted by a :class:`CalibratedCounter`
    (``token_count``); with the scratch files offered (``clear_inputs``), the arguments of the
    calls whose results are cleared are cleared too. Named as LangChain's, so a
    ``ContextEditingMiddleware`` given to ``ReAct(middleware=)`` replaces it."""

    def __init__(self, edit: ClearToolUsesEdit, counter: CalibratedCounter | None = None) -> None:
        super().__init__()
        counter = counter or CalibratedCounter()
        inputs = ClearToolUsesEdit(
            trigger=edit.trigger,
            clear_at_least=edit.clear_at_least,
            keep=edit.keep,
            clear_tool_inputs=True,
            exclude_tools=edit.exclude_tools,
            placeholder=edit.placeholder,
        )
        self.results = _Editing(edit, counter)
        self.inputs = _Editing(inputs, counter)

    @property
    def name(self) -> str:
        return ContextEditingMiddleware.__name__

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> Any:
        scratch = any(_named(t) in SCRATCH for t in request.tools)
        editing = self.inputs if scratch and on("clear_inputs") else self.results
        return await editing.awrap_model_call(request, handler)


# --------------------------------------------------------------------------- strict tools
#: The state key of :class:`StrictTools`: the conversation has had a malformed call.
STRICT: Final = "trellis_strict"


class _StrictState(AgentState[Any]):
    # or-ed: the calls of one step may all say so at once
    trellis_strict: NotRequired[Annotated[bool, operator.or_, PrivateStateAttr]]


def _malformed(message: BaseMessage) -> bool:
    """A tool's answer that says its call's arguments could not be used (not JSON, or not
    fitting its schema): what ``ModelHooks`` and the bridge answer such a call with."""
    return (
        isinstance(message, ToolMessage)
        and message.status == "error"
        and message.text.endswith(FIX_ARGUMENTS)
    )


class StrictTools(AgentMiddleware):
    """``strict`` tool schemas (``bind_tools(strict=True)``: the provider holds the
    arguments to the schema) from the conversation's first malformed call on — a call whose
    arguments were not JSON, or did not fit its tool's schema. Not before: a model that never
    errs keeps the request it always had (and its cached prompt)."""

    state_schema = _StrictState

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any] | ExtendedModelResponse[Any]:
        strict = bool(request.state.get(STRICT))
        if strict and request.tools and on("strict_tools"):
            request = request.override(model_settings={**request.model_settings, "strict": True})
        reply = await handler(request)
        message = next((m for m in reversed(reply.result) if isinstance(m, AIMessage)), None)
        if strict or message is None or not message.invalid_tool_calls:
            return reply
        return ExtendedModelResponse(model_response=reply, command=Command(update=_turned_on()))

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        done = await handler(request)
        state = request.state if isinstance(request.state, dict) else {}
        if isinstance(done, ToolMessage) and _malformed(done) and not state.get(STRICT):
            return Command(update={"messages": [done], **_turned_on()})
        return done


def _turned_on() -> dict[str, Any]:
    """The state update that turns strict schemas on (said once, when ``strict_tools`` is)."""
    if on("strict_tools"):
        metrics.loop(LoopEvent.STRICT_ON)
        log.info("strict tool schemas on: the model made a malformed call")
    return {STRICT: True}


# --------------------------------------------------------------------------- native output
class NativeOutput(AgentMiddleware):
    """``output=`` answered through the provider's structured output — ``strategy``, a
    LangChain ``ProviderStrategy(schema, strict=True)`` — instead of the answer tool LangChain
    otherwise forces (``ToolStrategy``: every step must call a tool, and the answer is one
    more). Built only for a model that has it; ``native_output`` turns it off."""

    def __init__(self, strategy: ProviderStrategy[Any]) -> None:
        super().__init__()
        self.strategy = strategy

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        if request.response_format is not None and on("native_output"):
            request = request.override(response_format=self.strategy)
        return await handler(request)


def native_output(output: Any, model: Any, *, gateway: bool) -> NativeOutput | None:
    """:class:`NativeOutput` for ``output``, when the model has structured output: its profile
    says ``structured_output``, or it is a gateway model (the gateway passes the JSON schema on
    to the provider); else ``None``."""
    profile = getattr(model, "profile", None) or {}
    if output is None or not (gateway or profile.get("structured_output")):
        return None
    return NativeOutput(ProviderStrategy(output, strict=True))


# --------------------------------------------------------------------------- tool repair
#: The values ``json_repair`` completes a cut-off value with: a repair holding one is no repair.
MADE_UP: Final = ("", None)


def _key(name: str) -> str:
    """A tool name as a misspelling of it still reads: case and ``-``/``_`` ignored."""
    return name.casefold().replace("-", "_")


class RepairToolCalls(AgentMiddleware):
    """A model reply's broken calls repaired before anything reads it: arguments that are not
    JSON (``invalid_tool_calls``) repaired with ``json_repair``, and a tool name with no exact
    match given the one offered tool it can mean (case, ``-``/``_``). A repair stands only
    when the arguments then fit the tool's schema (``arguments_problem``); else the call is
    left as it was, and answered with what is wrong, as before. Nothing is done to a reply
    with nothing broken."""

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        reply = await handler(request)
        if not on("tool_repair"):
            return reply
        message = next((m for m in reversed(reply.result) if isinstance(m, AIMessage)), None)
        if message is None:
            return reply
        fixed = repaired(message, request.tools)
        if fixed is message:
            return reply
        result = [fixed if m is message else m for m in reply.result]
        return ModelResponse(result=result, structured_response=reply.structured_response)


def repaired(message: AIMessage, tools: Tools) -> AIMessage:
    """``message`` with its broken calls repaired (see :class:`RepairToolCalls`), or the same
    message when none is, or none could be."""
    named = {_named(t): t for t in tools}
    if all(c["name"] in named for c in message.tool_calls) and not message.invalid_tool_calls:
        return message
    keys: dict[str, list[str]] = {}
    for name in named:
        keys.setdefault(_key(name), []).append(name)

    def resolved(name: str | None) -> str | None:
        if name in named:
            return name
        found = keys.get(_key(name or ""), [])
        return found[0] if len(found) == 1 else None

    calls, invalid, count = [], [], 0
    for call in message.tool_calls:
        name = resolved(call["name"])
        if name is not None and name != call["name"]:
            calls.append({**call, "name": name})
            count += 1
        else:
            calls.append(call)
    for call in message.invalid_tool_calls:
        name = resolved(call.get("name"))
        args = _repaired_args(call.get("args"), named[name]) if name is not None else None
        if name is None or args is None or not call.get("id"):
            invalid.append(call)
            continue
        calls.append({"name": name, "args": args, "id": call["id"], "type": "tool_call"})
        metrics.loop(LoopEvent.MALFORMED_ARGS)
        count += 1
    if not count:
        return message
    for _ in range(count):
        metrics.loop(LoopEvent.REPAIRED)
    log.info("repaired %d tool call(s)", count)
    return message.model_copy(update={"tool_calls": calls, "invalid_tool_calls": invalid})


def _repaired_args(raw: str | None, tool: BaseTool | dict[str, Any]) -> dict[str, Any] | None:
    """The arguments ``json_repair`` makes of ``raw``, when they are an object that fits the
    tool's schema and has no value it had to make up (``json_repair`` completes a cut-off
    value as empty); else ``None``."""
    args = json_repair.loads(raw or "")
    if not isinstance(args, dict) or any(v in MADE_UP for v in args.values()):
        return None
    schema = convert_to_openai_tool(tool)["function"].get("parameters") or {}
    return None if arguments_problem(schema, args) is not None else args


# --------------------------------------------------------------------------- think
#: The think tool, as Anthropic published it (anthropic.com/engineering/claude-think-tool).
THINK_DESCRIPTION: Final = (
    "Use the tool to think about something. It will not obtain new information or change the "
    "database, but just append the thought to the log. Use it when complex reasoning or some "
    "cache memory is needed."
)
#: What a call of the think tool answers.
THOUGHT: Final = "Thought noted."


def think_tool() -> BaseTool:
    """The ``think`` tool: a thought, noted; nothing read, nothing changed."""

    async def think(thought: str) -> str:
        """Think about something.

        Args:
            thought: A thought to think about.
        """
        return THOUGHT

    return StructuredTool.from_function(
        coroutine=think, name=THINK, description=THINK_DESCRIPTION, parse_docstring=True
    )


def _ruled(request: ModelRequest[Any]) -> bool:
    """Whether the conversation opens with context that may hold rules: a leading system
    message (the memory context the harness pushes, a remembered rule)."""
    return bool(request.messages) and isinstance(request.messages[0], SystemMessage)


class ThinkTool(AgentMiddleware):
    """The ``think`` tool offered (``think``) to a model that does not reason (``reasons``:
    ``ReAct(effort=)`` was given), in a conversation that opens with context that may hold
    rules (the pushed memory context), on a call that offers a tool that changes data — a
    change a rule may forbid; not offered otherwise (the graph keeps it: a call of it is
    answered)."""

    def __init__(self, *, reasons: bool) -> None:
        super().__init__()
        self.reasons = reasons

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        wanted = (
            not self.reasons and _ruled(request) and bool(_writes_offered(request)) and on("think")
        )
        return await handler(request if wanted else without_tools(request, {THINK}))


# --------------------------------------------------------------------------- Deep Agents
class UnsupportedContent(UnsupportedContentMiddleware):
    """Deep Agents' ``UnsupportedContentMiddleware`` (``unsupported_content``): an image, audio
    or a file the model's profile says it cannot take replaced in the request by a notice — the
    thread keeps it. Named as Deep Agents'."""

    @property
    def name(self) -> str:
        return UnsupportedContentMiddleware.__name__

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> Any:
        if not on("unsupported_content"):
            return await handler(request)
        return await super().awrap_model_call(request, handler)


#: The most grading rounds of a run's answer against its rubric.
RUBRIC_ROUNDS: Final = 2


class Rubric(RubricMiddleware):
    """Deep Agents' ``RubricMiddleware`` (``rubric``): a run whose input carries a ``rubric``
    (``{"messages": [...], "rubric": "..."}``) has its answer graded against it by a grader
    on ``model`` once it would end, revised on the grader's feedback, at most
    :data:`RUBRIC_ROUNDS` rounds. Without a rubric it does nothing. Named as Deep Agents'."""

    def __init__(self, model: Any) -> None:
        super().__init__(model=model, max_iterations=RUBRIC_ROUNDS)

    @property
    def name(self) -> str:
        return RubricMiddleware.__name__

    async def abefore_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        return await super().abefore_agent(state, runtime) if on("rubric") else None

    async def aafter_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        return await super().aafter_agent(state, runtime) if on("rubric") else None


# --------------------------------------------------------------------------- helpers
#: The helpers one run has working at once (the parent's step may call more: they wait).
HELPERS_AT_ONCE: Final = 4
#: The general-purpose helper's name, description and instructions (Deep Agents').
GENERAL: Final = str(GENERAL_PURPOSE_SUBAGENT["name"])
#: The parent agents whose general-purpose helper is kept, at most.
HELPER_AGENTS: Final = 64
#: The todos of a plan from which a task is several independent parts.
PARTS: Final = 3


class Helpers(SubAgentMiddleware):
    """Deep Agents' ``SubAgentMiddleware`` (``subagents``): its ``task`` tool offered on a
    task of several parts — a long task's window (:data:`~.automatic.MIN_WINDOW`), and the
    request asking for steps, a plan of :data:`PARTS` todos or more, or ``task`` already
    called in the conversation — for a general-purpose helper and every agent the run has as a
    tool (each then reached through ``task`` only). Each ``task`` call is a child run of the
    harness (``trellis.harness.subagents``): its own record naming its parent, its tools
    governed and journaled, its journal kept in its parent's (a crash continues it), its
    spending the parent's (a parent's budget caps its helpers), at most
    :data:`HELPERS_AT_ONCE` at once; only its answer comes back. Named as Deep Agents'."""

    def __init__(self, model: Any, *, window: int | None) -> None:
        general: CompiledSubAgent = {
            "name": GENERAL,
            "description": str(GENERAL_PURPOSE_SUBAGENT["description"]),
            "runnable": RunnableLambda(self._general),
        }
        super().__init__(backend=StateBackend(), subagents=[general])
        self.model = model
        self.window = window
        task = self.tools[0]
        task.metadata = {**(task.metadata or {}), "side_effects": "write", RESUMABLE: True}
        self._helpers: LRU[str, SubAgentTool] = LRU(HELPER_AGENTS)
        self._slots: weakref.WeakKeyDictionary[Runtime, asyncio.Semaphore] = (
            weakref.WeakKeyDictionary()
        )

    @property
    def name(self) -> str:
        return SubAgentMiddleware.__name__

    def wanted(self, request: ModelRequest[Any]) -> bool:
        """Whether this call is offered ``task``: see the class."""
        roomy = self.window is not None and self.window >= MIN_WINDOW
        return roomy and on("subagents") and _parts(request)

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        runtime = current()
        if runtime is None or not self.wanted(request):
            return await handler(without_tools(request, {TASK}))
        agents = _agents(runtime)
        listed = [(GENERAL, str(GENERAL_PURPOSE_SUBAGENT["description"]))]
        listed += [(name, runtime.toolbox[name].spec.description) for name in agents]
        lines = "\n".join(f"- {name}: {description}" for name, description in listed)
        tools = [
            t.model_copy(
                update={"description": TASK_TOOL_DESCRIPTION.format(available_agents=lines)}
            )
            if _named(t) == TASK and isinstance(t, BaseTool)
            else t
            for t in request.tools
            if _named(t) not in agents
        ]
        return await handler(request.override(tools=tools))

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        call = request.tool_call
        runtime = current()
        if call["name"] != TASK or runtime is None:
            return await handler(request)
        if not on("subagents"):
            return refused(dict(call))
        named = call["args"].get("subagent_type")
        if named not in _agents(runtime):
            return await handler(request)  # Deep Agents' task: the general-purpose helper
        async with self._slot(runtime):
            outcome = await bridge.call(
                runtime.toolbox[named],
                {"message": call["args"].get("description", "")},
                call_id=str(call["id"]),
            )
        return ToolMessage(
            content=text_of(outcome.output),
            tool_call_id=str(call["id"]),
            name=TASK,
            status="error" if outcome.status in FAILED else "success",
        )

    def _slot(self, runtime: Runtime) -> asyncio.Semaphore:
        slots = self._slots.get(runtime)
        if slots is None:
            slots = self._slots[runtime] = asyncio.Semaphore(HELPERS_AT_ONCE)
        return slots

    async def _general(self, state: dict[str, Any]) -> dict[str, Any]:
        """Deep Agents' general-purpose helper as a child run of the harness: an agent of the
        run's own agent — its instructions Deep Agents', its model and tools the parent's — run
        on the task, its answer the helper's last message."""
        runtime = current()
        if runtime is None:
            raise ConfigurationError("a ReAct's helpers run inside a harness run")
        helper = self._helpers.setdefault(runtime.agent.id, lambda: self._made(runtime))
        message = state["messages"][-1]
        async with self._slot(runtime):
            answer = await helper.call({"message": text_of(message.content)})
        return {"messages": [AIMessage(content=text_of(answer))]}

    def _made(self, runtime: Runtime) -> SubAgentTool:
        from trellis.harness.react import ReAct  # noqa: PLC0415 - react builds this

        parent = runtime.agent
        helper = parent.harness.agents.get(f"{parent.id}{HELPER}") or parent.harness.wrap(
            ReAct(
                str(GENERAL_PURPOSE_SUBAGENT["system_prompt"]),
                self.model,
                context_window=self.window,
            ),
            id=f"{parent.id}{HELPER}",
            tools=list(parent.sources),
            without={*parent.without, "subagents"},
            budget=None,
        )
        return SubAgentTool(helper)


#: The general-purpose helper's agent id: its parent's, and this.
HELPER: Final = "-helper"


class SubAgentTool:
    """A helper agent's child run for a task (``trellis.harness.subagents``)."""

    def __init__(self, agent: Any) -> None:
        self.source = SubAgent(agent, name=agent.id, description=None, side_effects=None)

    async def call(self, args: dict[str, Any]) -> Any:
        return await self.source._run(args)


def _agents(runtime: Runtime) -> list[str]:
    """The run's tools that are agents (sub-agents: their calls continue after a crash)."""
    return sorted(name for name, tool in runtime.toolbox.items() if tool.resumable)


def _parts(request: ModelRequest[Any]) -> bool:
    """Whether the task has several parts: the request asks for steps, the plan has
    :data:`PARTS` todos or more, or ``task`` was called in this conversation already."""
    state = request.state or {}
    if asks_for_steps(request) or len(state.get("todos") or ()) >= PARTS:
        return True
    return any(
        isinstance(m, AIMessage) and any(c["name"] == TASK for c in m.tool_calls)
        for m in reversed(state.get("messages", []))
    )


__all__ = [
    "CalibratedCounter",
    "ClearOldResults",
    "Helpers",
    "NativeOutput",
    "RepairToolCalls",
    "Rubric",
    "StrictTools",
    "ThinkTool",
    "UnsupportedContent",
    "native_output",
    "repaired",
    "think_tool",
]
