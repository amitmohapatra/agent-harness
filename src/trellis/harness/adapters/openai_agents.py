"""OpenAI Agents SDK ``Agent``\\ s.

* input: a message list, the memory context first as a ``system`` message; where memory keeps
  the conversation (no ``session`` given), its recent messages follow as the SDK's own input
  items — messages, ``function_call`` and ``function_call_output`` — as a session replays them;
* transcript: the run's new items — its answers, its tool calls and their results — in order;
* run: ``Runner.run``/``Runner.run_streamed`` on a clone of the agent carrying the harness
  tools next to its own (the caller's agent is never changed) — and every agent it can hand off
  to cloned too, recursively, each one's own tools governed (:func:`governed_agents`) — with
  the harness's ``RunHooks`` always (``hooks.openai_agents``): the run's hooks around its model
  calls, and every reply's usage counted toward the run's budget;
* history: a ``session``, ``conversation_id``, ``previous_response_id`` or
  ``auto_previous_response_id`` given as a run option is the framework keeping the thread
  (``history``): memory's window is left out, unless the session is kept in RAM only. Given
  none, while memory is on, the run's session is the memory service's thread
  (``bridges.session.MemorySession``), which owns the history the same way;
* helpers: an agent called as a tool (``Agent.as_tool``) is a child run of the run
  (``bridges.helpers``): its call under the catalog's rules, what it spent on the run's budget,
  its run record naming the run, its turns in memory. Its own tools are governed when built with
  ``h.tools`` (the SDK runs the agent itself, which the harness cannot copy): any other is said
  once;
* pause: ``trellis.current().ask`` stops the run and a resume re-runs it against the journal;
  the SDK's own ``needs_approval`` tools pause with the SDK's ``RunState``, and a resume
  approves or rejects on that state and continues it (``RunState.approve``/``reject``). The SDK
  takes no edited arguments and no answer in place of a tool's result, so: a reject carries the
  reviewer's reason (``rejection_message``, the text the model reads); an answer is a reject
  whose message is the answer; an edit is a reject telling the model to call the tool again
  with the edited arguments — and that call, when the model makes it with exactly those
  arguments, is approved in the same attempt instead of asking again;
* framework options: keyword arguments of ``Runner.run``/``run_streamed`` (``max_turns``,
  ``run_config``, ``context``...), except the ones the harness gives itself
  (:data:`OWN_ARGUMENTS`), which are refused.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import json
import logging
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Final

from trellis.contracts import (
    ConfigurationError,
    InterruptDecision,
    InterruptResolution,
    ToolOutcome,
    ToolSpec,
    ToolStatus,
)
from trellis.harness.adapters.base import (
    Extracted,
    Invocation,
    Narrowing,
    NativePause,
    Output,
    ToolFormat,
)
from trellis.harness.bounded import SAID_ONCE
from trellis.harness.bridges.helpers import Helpers
from trellis.harness.bridges.session import MemorySession
from trellis.harness.history import MEMORY_THREAD, Called, Said, Speaker
from trellis.harness.journal import Pending
from trellis.harness.runtime import Runtime, reason_of
from trellis.harness.tools import bridge
from trellis.harness.tools.base import DEFAULT_SIDE_EFFECTS

log = logging.getLogger("trellis.harness")

#: ``Runner.run``'s arguments the harness gives itself: the agent (a clone carrying the harness
#: tools), its input (with the memory context, or the resumed state) and the run's hooks.
OWN_ARGUMENTS: Final = ("starting_agent", "input", "hooks")
#: ``Runner.run``'s arguments that keep the conversation on OpenAI's servers: the framework's
#: thread, durable (``history``).
SERVER_THREAD: Final = ("conversation_id", "previous_response_id", "auto_previous_response_id")
#: The raw streaming event that carries a text delta.
TEXT_DELTA = "response.output_text.delta"
#: What the model reads for a call a reviewer edited (the SDK cannot run it with other
#: arguments): call it again with these, and that call runs approved.
EDITED = (
    "A reviewer changed the arguments of this {tool} call, so it was not run. Call {tool} "
    "again with exactly these arguments: {args}"
)


@dataclass(frozen=True, slots=True)
class _Continue:
    """A resume from the SDK's serialised run state."""

    state: dict[str, Any]
    call_id: str | None
    approve: bool
    #: what the model reads instead of the tool's result: a reject's reason, an answer, an edit
    message: str | None = None
    #: an edit: the tool and its edited arguments, approved when the model calls it with them
    tool: str | None = None
    edited: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class ServerThread:
    """A conversation OpenAI's servers keep (``key``: the run option that names it): durable."""

    key: str


@functools.cache
def sdk_tracing_said() -> bool:
    """Say once, when an agent is wrapped, that the SDK's own tracing — which exports to
    OpenAI unless turned off — is left as the application set it (``True`` when said)."""
    from agents.tracing import get_trace_provider

    provider = get_trace_provider()
    disabled = getattr(provider, "_disabled", False)
    if disabled:
        return False
    log.info(
        "the OpenAI Agents SDK's own tracing is on and exports to OpenAI: the harness leaves it "
        "as you set it (agents.set_tracing_disabled(True), or OPENAI_AGENTS_DISABLE_TRACING=1, "
        "turns it off); the run's trace is the harness's and the gateway's"
    )
    return True


class OpenAIAgentsAdapter:
    name: ClassVar[str] = "openai_agents"
    tool_format: ToolFormat = "openai_agents"
    fixed_tools: bool = False
    narrows: Narrowing = "turn"
    #: its history comes through its session: the run's own, or the memory service's thread
    #: (``bridges.session.MemorySession``), never as messages in the input
    replays_history = False

    def thread_store(self, target: Any, options: Mapping[str, Any]) -> Any | None:
        """The run's ``session``, or the conversation OpenAI's servers keep
        (:data:`SERVER_THREAD`); else the memory service's thread, the session a run with
        memory on is given (``bridges.session``)."""
        return _thread_store(options)

    def prepare_input(
        self, target: Any, input: Any, context: str | None, history: Sequence[Said] = ()
    ) -> Any:
        system = [{"role": "system", "content": context}] if context else []
        if isinstance(input, str):
            return [*system, {"role": "user", "content": input}] if system else input
        if isinstance(input, list):
            return [*system, *input]
        return [*system, {"role": "user", "content": json.dumps(input, default=str)}]

    async def invoke(self, target: Any, native_input: Any, run: Invocation) -> Any:
        from agents import Runner

        helpers = Helpers(run.runtime, self.name)
        agent, hooks = self._agent(target, run, helpers), _hooks()
        options = _options(run.runtime)
        error: BaseException | None = None
        try:
            given = await self._input(agent, native_input)
            result = await Runner.run(agent, given, hooks=hooks, **options)
            state = _edited_call(result, native_input)
            if state is not None:  # the model called the edited tool as told: it runs, approved
                result = await Runner.run(agent, state, hooks=hooks, **options)
        except BaseException as exc:
            error = exc
            raise
        finally:
            await helpers.closed(error)
        return result

    async def stream(self, target: Any, native_input: Any, run: Invocation) -> AsyncIterator[Any]:
        from agents import Runner

        helpers = Helpers(run.runtime, self.name)
        agent, hooks = self._agent(target, run, helpers), _hooks()
        options = _options(run.runtime)
        given = await self._input(agent, native_input)
        result = Runner.run_streamed(agent, given, hooks=hooks, **options)
        error: BaseException | None = None
        try:
            async for delta in _deltas(result):
                yield delta
            state = _edited_call(result, native_input)
            if state is not None:
                result = Runner.run_streamed(agent, state, hooks=hooks, **options)
                async for delta in _deltas(result):
                    yield delta
        except BaseException as exc:
            error = exc
            raise
        finally:
            # cut short (cancelled, out of time), the SDK stops its run without waiting for
            # it: its tool calls end (on the run's stream too) before the run does
            assert result.run_loop_task is not None  # set by run_streamed
            await asyncio.wait({result.run_loop_task})
            await helpers.closed(error)
        yield Output(result)

    def extract(self, target: Any, output: Any) -> Extracted:
        transcript = said_in(output.new_items)
        pause = None
        if output.interruptions:
            item = output.interruptions[0]
            # the SDK pauses on a call before reading its arguments: ones that are not a JSON
            # object reach the person as the model wrote them
            args = _arguments(item)
            pause = NativePause(
                value=None,
                native_id=item.call_id,
                state=output.to_state().to_json(),
                tool=item.name,
                args=args if isinstance(args, dict) else {"arguments": item.arguments},
            )
        return Extracted(answer=output.final_output, transcript=transcript, pause=pause)  # type: ignore[arg-type]

    def resume_input(
        self,
        target: Any,
        native_input: Any,
        pending: Pending,
        resolution: InterruptResolution,
    ) -> Any:
        if pending.native_state is None:
            return native_input
        state, call_id = pending.native_state, pending.native_id
        decision = resolution.decision
        if decision is InterruptDecision.APPROVE:
            return _Continue(state, call_id, approve=True)
        if decision is InterruptDecision.EDIT and pending.interrupt.tool_call is not None:
            tool = pending.interrupt.tool_call.tool
            edited = resolution.payload or {}
            message = EDITED.format(tool=tool, args=json.dumps(edited, default=str))
            return _Continue(
                state, call_id, approve=False, message=message, tool=tool, edited=edited
            )
        if decision is InterruptDecision.ANSWER and resolution.answer is not None:
            answer = resolution.answer
            message = answer if isinstance(answer, str) else json.dumps(answer, default=str)
            return _Continue(state, call_id, approve=False, message=message)
        return _Continue(state, call_id, approve=False, message=reason_of(resolution))

    def native_tools(self, target: Any) -> list[str]:
        """The agent's own ``tools`` (function and hosted tools alike), by name."""
        return [t.name for t in target.tools if isinstance(getattr(t, "name", None), str)]

    def check_options(self, options: Mapping[str, Any]) -> None:
        from agents import Runner

        own = [key for key in options if key in OWN_ARGUMENTS]
        if own:
            raise ConfigurationError(
                f"framework_options {', '.join(map(repr, own))}: the harness gives Runner.run "
                "its agent, its input and its hooks itself (hooks go on h.wrap(hooks=))"
            )
        taken = inspect.signature(Runner.run).parameters
        unknown = [key for key in options if key not in taken]
        if unknown:
            raise ConfigurationError(
                f"framework_options {', '.join(map(repr, unknown))}: Runner.run takes no such "
                f"argument (it takes {', '.join(k for k in taken if k not in OWN_ARGUMENTS)})"
            )

    # ------------------------------------------------------------------ internals
    @staticmethod
    def _agent(target: Any, run: Invocation, helpers: Helpers) -> Any:
        # a tool the agent carries already (built with h.tools: read_file, say) is not added twice
        carried = {getattr(t, "name", None) for t in target.tools}
        added = [t for t in run.native_tools or () if t.name not in carried]
        return governed_agents(target, added=added, agent_id=run.runtime.agent_id, helpers=helpers)

    @staticmethod
    async def _input(agent: Any, native_input: Any) -> Any:
        if not isinstance(native_input, _Continue):
            return native_input
        from agents import RunState

        state = await RunState.from_json(agent, native_input.state)
        for item in state.get_interruptions():
            if item.call_id != native_input.call_id:
                continue  # still waiting: the run pauses on it next
            if native_input.approve:
                state.approve(item)
            else:
                state.reject(item, rejection_message=native_input.message)
        return state


def governed_agents(
    target: Any, *, added: list[Any], agent_id: str, helpers: Helpers | None = None
) -> Any:
    """``target`` and every agent reachable through its handoffs, each a copy whose own tools
    are governed (``tools.convert.openai_agents.governed``: hooks, the catalog's rules, the
    journal) — recursively, each agent copied once, so handoffs that lead back are kept —
    ``added`` (the run's harness tools) on ``target``'s. A ``Handoff`` object hides its agent
    until it is invoked: its tools cannot be reached, which is said once; so are an agent tool's
    (``Agent.as_tool``) own tools not built with ``h.tools``. With ``helpers``, each agent tool's
    call is a helper of the run. The agents you wrapped are not changed."""
    from agents import Agent

    from trellis.harness.tools.convert.openai_agents import governed

    copies: dict[int, Any] = {}

    def copied(agent: Any, extra: list[Any]) -> Any:
        found = copies.get(id(agent))
        if found is not None:
            return found
        for tool in agent.tools:
            _ungoverned_inside(agent_id, tool)
        own = [governed(t, helpers) for t in agent.tools]
        copy = copies[id(agent)] = agent.clone(tools=[*own, *extra])
        handoffs = []
        for handoff in agent.handoffs:
            if isinstance(handoff, Agent):
                handoffs.append(copied(handoff, []))
            else:
                _hidden_handoff(agent_id, str(getattr(handoff, "agent_name", "")))
                handoffs.append(handoff)
        copy.handoffs = handoffs
        return copy

    return copied(target, added)


@functools.lru_cache(maxsize=SAID_ONCE)  # said once per agent and handoff, of the most recent
def _hidden_handoff(agent_id: str, to: str) -> None:
    log.warning(
        "a handoff is a Handoff object, which hides its agent until it is invoked: that "
        "agent's own tools are not governed (hand off to the Agent itself)",
        extra={"agent_id": agent_id, "handoff": to},
    )


def _ungoverned_inside(agent_id: str, tool: Any) -> None:
    """Say once when an agent tool's agent has tools of its own not built with ``h.tools``:
    the SDK runs that agent itself, so their calls are not governed."""
    from agents import FunctionTool

    from trellis.harness.tools.convert.openai_agents import AGENT_OF_TOOL, MADE

    inner = getattr(tool, AGENT_OF_TOOL, None)
    plain = [
        t.name
        for t in getattr(inner, "tools", ())
        if isinstance(t, FunctionTool) and not getattr(t.on_invoke_tool, MADE, False)
    ]
    if plain:
        _said_inside(agent_id, tool.name, ", ".join(plain))


@functools.lru_cache(maxsize=SAID_ONCE)  # said once per agent and agent tool, of the most recent
def _said_inside(agent_id: str, tool: str, tools: str) -> None:
    log.warning(
        "an agent called as a tool runs its own tools inside the SDK, where the harness cannot "
        "govern them: build them with h.tools so the catalog's rules apply",
        extra={"agent_id": agent_id, "agent_tool": tool, "tools": tools},
    )


def _options(runtime: Runtime) -> dict[str, Any]:
    """``Runner.run``'s options: the run's framework options, with the memory service's thread
    as the session (``bridges.session``) when the run keeps no thread of its own and memory is
    on."""
    options = dict(runtime.framework_options)
    if runtime.run_memory is not None and _thread_store(options) is MEMORY_THREAD:
        options["session"] = MemorySession(runtime)
    return options


def _thread_store(options: Mapping[str, Any]) -> Any:
    """The thread store of a run with these options (``OpenAIAgentsAdapter.thread_store``)."""
    if options.get("session") is not None:
        return options["session"]
    server = next((key for key in SERVER_THREAD if options.get(key)), None)
    return ServerThread(server) if server is not None else MEMORY_THREAD


def _hooks() -> Any:
    """The harness's ``RunHooks``, always: the run's model hooks around every model call, and
    every reply's usage counted toward the run's budget (``budget.py``)."""
    from trellis.harness.hooks.openai_agents import ModelHooks

    return ModelHooks()


async def _deltas(result: Any) -> AsyncIterator[str]:
    async for event in result.stream_events():
        data = getattr(event, "data", None)
        if event.type == "raw_response_event" and getattr(data, "type", None) == TEXT_DELTA:
            yield data.delta  # type: ignore[union-attr]


def _edited_call(result: Any, native_input: Any) -> Any:
    """The run state with the edited call approved, when the run paused on the call a reviewer
    edited, made again with exactly the edited arguments (else ``None``)."""
    if not isinstance(native_input, _Continue) or native_input.edited is None:
        return None
    for item in result.interruptions:
        if item.name == native_input.tool and _arguments(item) == native_input.edited:
            state = result.to_state()
            state.approve(item)
            return state
    return None


def said_in(items: Sequence[Any]) -> list[Said]:
    """A run's new items as messages, in order: the function calls one model response made,
    together, on one assistant message (with what it said, when it said something first), and
    their results. A hosted tool's call and result (web search, a computer) are the SDK's own
    and are not replayed."""
    from agents import ItemHelpers, MessageOutputItem, ToolCallItem, ToolCallOutputItem
    from openai.types.responses import ResponseFunctionToolCall

    out: list[Said] = []
    for item in items:
        if isinstance(item, MessageOutputItem):
            if text := ItemHelpers.text_message_output(item):
                out.append(Said(Speaker.ASSISTANT, text))
        elif isinstance(item, ToolCallItem) and isinstance(item.raw_item, ResponseFunctionToolCall):
            raw = item.raw_item
            args = _arguments(raw)
            call = Called(
                raw.call_id,
                raw.name,
                args if isinstance(args, dict) else {"arguments": raw.arguments},
            )
            last = out[-1] if out else None
            if last is not None and last.role is Speaker.ASSISTANT:
                # no result between them: the same model response
                out[-1] = Said(Speaker.ASSISTANT, last.content, (*last.tool_calls, call))
            else:
                out.append(Said(Speaker.ASSISTANT, "", (call,)))
        elif isinstance(item, ToolCallOutputItem) and (answers := _function_output(item.raw_item)):
            out.append(Said(Speaker.TOOL, _output(item.output), tool_call_id=answers))
    return out


def _function_output(raw: Any) -> str | None:
    """The call a function call's result answers; ``None`` for a hosted tool's result."""
    if isinstance(raw, dict) and raw.get("type") == "function_call_output":
        return str(raw["call_id"])
    return None


def _output(output: Any) -> str:
    return output if isinstance(output, str) else json.dumps(output, default=str)


def _arguments(item: Any) -> Any:
    try:
        return json.loads(item.arguments or "{}")
    except ValueError:
        return None


#: The tools a provider runs itself, inside the model's step, by the type of the item that
#: reports a call of one: the name it is recorded under (a hosted MCP server's tool: its own).
HOSTED: Final[dict[str, str]] = {
    "file_search_call": "file_search",
    "web_search_call": "web_search",
    "code_interpreter_call": "code_interpreter",
    "image_generation_call": "image_generation",
    "mcp_call": "hosted_mcp",
}
#: A hosted call's status that is no failure.
HOSTED_DONE: Final = frozenset({"completed", "searching", "in_progress", "interpreting"})


async def hosted_calls(runtime: Runtime, items: Sequence[Any], *, started: float) -> None:
    """The provider's hosted tool calls in one model reply (``items``: its output), recorded
    as the run's tool calls (``tools.bridge.observed``): each with what it was asked, what it
    found (its results, never an image's bytes), whether it failed, and the duration of the
    step it ran in (``started``: when the step began)."""
    for item in items:
        kind = getattr(item, "type", None)
        name = HOSTED.get(kind) if isinstance(kind, str) else None
        if name is None:
            continue
        server = None
        if kind == "mcp_call":  # a hosted MCP server's tool: by its own name, from its server
            name, server = str(item.name), str(item.server_label)
        args, output = _hosted(item)
        failed = getattr(item, "status", "completed") not in HOSTED_DONE or bool(
            getattr(item, "error", None)
        )
        outcome = ToolOutcome(
            tool=name,
            status=ToolStatus.ERROR if failed else ToolStatus.OK,
            output=output,
            error_class="HostedToolError" if failed else None,
        )
        source = "local" if server is None else "mcp"
        spec = ToolSpec(name=name, source=source, server=server, side_effects=DEFAULT_SIDE_EFFECTS)
        ref = str(getattr(item, "id", None) or f"{runtime.run_id}:{name}")
        await bridge.observed(runtime, spec, args, outcome, ref=ref, started=started)


def _hosted(item: Any) -> tuple[dict[str, Any], Any]:
    """What a hosted call was asked (its arguments) and what came of it, as plain data."""
    kind = item.type
    if kind == "file_search_call":
        results = [
            {"file": r.filename, "score": r.score} for r in (item.results or ()) if r is not None
        ]
        return {"queries": list(item.queries or ())}, {"results": results}
    if kind == "web_search_call":
        action = item.action.model_dump(exclude_none=True) if item.action is not None else {}
        return action, {"status": item.status}
    if kind == "code_interpreter_call":
        outputs = [o.model_dump(exclude_none=True) for o in item.outputs or ()]
        return {"code": item.code or ""}, {"outputs": outputs}
    if kind == "image_generation_call":
        return {}, {"revised_prompt": getattr(item, "revised_prompt", None)}
    args = _arguments(item)
    error = item.error
    output = item.output if error is None else error.model_dump(mode="json", exclude_none=True)
    return (args if isinstance(args, dict) else {"arguments": item.arguments}), output
