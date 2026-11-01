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
  its run record naming the run, its memory context, tool records and outcome. It runs the
  governed copy of its agent; where the SDK's agent tool cannot be pointed at it, its own tools'
  calls are not governed, which is said once;
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
  (:data:`OWN_ARGUMENTS`), which are refused; a call to a tool the model made up is an error
  the model reads (``RunConfig.tool_not_found_behavior``), as on every other adapter.
"""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import functools
import inspect
import json
import logging
import types
import weakref
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Final, cast

from trellis.contracts import (
    ConfigurationError,
    InterruptDecision,
    InterruptResolution,
    ToolError,
    ToolOutcome,
    ToolSpec,
    ToolStatus,
)
from trellis.harness.adapters.base import (
    Extracted,
    HistoryIn,
    Invocation,
    Narrowing,
    NativePause,
    Output,
    ToolFormat,
    helper,
    unique,
)
from trellis.harness.bounded import LRU, SAID_ONCE
from trellis.harness.bridges.helpers import Helpers
from trellis.harness.bridges.session import MemorySession, input_items
from trellis.harness.features import MEMORY_TOOLS
from trellis.harness.gateways import Capability, per_loop
from trellis.harness.history import MEMORY_THREAD, Called, Said, Speaker, answered
from trellis.harness.journal import Pending
from trellis.harness.runtime import Runtime, current, reason_of
from trellis.harness.tools import bridge
from trellis.harness.tools.base import DEFAULT_SIDE_EFFECTS, Tool, data_of
from trellis.harness.tools.mcp import data_of_listed, side_effects_of

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


if TYPE_CHECKING:
    from trellis.harness.gateways import Bifrost

#: Bifrost's model providers, per event loop, by gateway (:func:`bifrost_provider`).
_PROVIDERS: weakref.WeakKeyDictionary[Any, LRU[Any, Any]] = weakref.WeakKeyDictionary()
#: How many of them one event loop keeps.
PROVIDERS_KEPT: Final = 8


class OpenAIAgentsAdapter:
    name: ClassVar[str] = "openai_agents"
    tool_format: ToolFormat = "openai_agents"
    fixed_tools: bool = False
    narrows: Narrowing = "turn"
    #: its history comes through its session: the run's own, or the memory service's thread
    #: (``bridges.session.MemorySession``), never as messages in the input
    replays_history = False
    #: with memory off and no session, the conversation's earlier turns come as input items
    history_in: HistoryIn = "messages"
    sessions = False

    def __init__(self) -> None:
        #: the tools the agents declare (:meth:`native`), once read
        self._static: list[ToolSpec] | None = None

    def thread_store(self, target: Any, options: Mapping[str, Any]) -> Any | None:
        """The run's ``session``, or the conversation OpenAI's servers keep
        (:data:`SERVER_THREAD`); else the memory service's thread, the session a run with
        memory on is given (``bridges.session``)."""
        return _thread_store(options)

    def steers(self, target: Any) -> bool:
        """Every agent: a steered message joins the input of its next model call
        (``RunConfig.call_model_input_filter``, :func:`_steering`)."""
        return True

    def prepare_input(
        self, target: Any, input: Any, context: str | None, history: Sequence[Said] = ()
    ) -> Any:
        system: list[Any] = [{"role": "system", "content": context}] if context else []
        system += [item for said in answered(history) for item in input_items(said)]
        if isinstance(input, str):
            return [*system, {"role": "user", "content": input}] if system else input
        if isinstance(input, list):
            return [*system, *input]
        return [*system, {"role": "user", "content": json.dumps(input, default=str)}]

    async def invoke(self, target: Any, native_input: Any, run: Invocation) -> Any:
        from agents import Runner

        helpers = Helpers(run.runtime, self.name)
        agent, hooks = self._agent(target, run, helpers), _hooks()
        options = _options(run.runtime, target)
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
        options = _options(run.runtime, target)
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

    async def native(self, target: Any) -> list[ToolSpec]:
        """The tools of every agent the run can reach (:func:`reachable`: ``target``, the agents
        it hands off to and the agents its agent tools run): their function and hosted tools,
        the handoffs themselves (``transfer_to_...``), and their MCP servers' tools — listed
        from each server (a server that cannot list them now, not connected, is left out and
        said once). The harness's own tools are not among them."""
        if self._static is None:
            self._static = [spec for agent in reachable(target) for spec in _specs(agent)]
        listed: list[ToolSpec] = []
        for agent in reachable(target):
            for server in agent.mcp_servers:
                listed.extend(await _listed(server))
        return unique([*self._static, *listed])

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
        # the memory tools (pull, tool_search) go to every agent the run can reach: an agent
        # it hands off to answers from memory as the one it started with
        memory = {t.spec.name for t in run.tools if t.feature in MEMORY_TOOLS}
        shared = [t for t in run.native_tools or () if t.name in memory]
        return governed_agents(
            target,
            added=list(run.native_tools or ()),
            shared=shared,
            agent_id=run.runtime.agent_id,
            helpers=helpers,
        )

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
    target: Any,
    *,
    added: list[Any],
    agent_id: str,
    shared: Sequence[Any] = (),
    helpers: Helpers | None = None,
) -> Any:
    """``target`` and every agent reachable through its handoffs and its agent tools
    (``Agent.as_tool``), each a copy whose own tools are governed
    (``tools.convert.openai_agents.governed``: hooks, the catalog's rules, the journal, and
    offered per turn only while the run offers them: the tool hints' narrowing), whose MCP
    servers' calls are governed the same way (:class:`GovernedServer`) and whose agent tools
    run the governed copy of their agent — recursively, each agent copied once, so handoffs that
    lead back are kept — ``added`` (the run's harness tools) on ``target``'s and ``shared``
    (the run's memory tools) on every other's. A tool an agent carries already (built with
    the harness) is not added twice. A ``Handoff`` object hides its agent until it is invoked:
    its tools cannot be reached, which is said once. With ``helpers``, each agent tool's call is
    a helper of the run. The agents you wrapped are not changed."""
    from agents import Agent

    from trellis.harness.tools.convert.openai_agents import AGENT_OF_TOOL, governed

    copies: dict[int, Any] = {}

    def copied(agent: Any, extra: Sequence[Any]) -> Any:
        found = copies.get(id(agent))
        if found is not None:
            return found
        copy = copies[id(agent)] = agent.clone()
        own = []
        for tool in agent.tools:
            inner = getattr(tool, AGENT_OF_TOOL, None)
            ran = _running(tool, copied(inner, shared)) if isinstance(inner, Agent) else None
            if inner is not None and ran is None:
                _ungoverned_inside(agent_id, tool)
            own.append(governed(ran or tool, helpers))
        carried = {getattr(t, "name", None) for t in agent.tools}
        copy.tools = [*own, *(t for t in extra if t.name not in carried)]
        copy.mcp_servers = [GovernedServer.of(s) for s in agent.mcp_servers]
        handoffs = []
        for handoff in agent.handoffs:
            if isinstance(handoff, Agent):
                handoffs.append(copied(handoff, shared))
            else:
                _hidden_handoff(agent_id, str(getattr(handoff, "agent_name", "")))
                handoffs.append(handoff)
        copy.handoffs = handoffs
        return copy

    return copied(target, added)


class GovernedServer:
    """An OpenAI agent's MCP server as the harness runs it: every call of its tools inside a
    run goes through the bridge (``tools.bridge.call``: hooks, the catalog's rules — its tools'
    annotations as they were listed for their risk —, a person when governance asks, the
    journal, the record), the server itself as it was (connected, listed, cleaned up by its
    owner). A call that pauses the run raises out of the SDK's run (``_failure_error_function``
    ``None``: the SDK raises instead of telling the model), and a tool's own error is the
    result the model reads, as before. Made by :meth:`of`, a subclass of the SDK's
    ``MCPServer`` built when first needed."""

    @staticmethod
    def of(server: Any) -> Any:
        """``server`` governed (a server already governed is kept as it is)."""
        cls = _governed_server()
        return server if isinstance(server, cls) else cls(server)


@functools.cache
def _governed_server() -> type:
    from agents.mcp import MCPServer
    from mcp.types import CallToolResult, TextContent

    class Governed(MCPServer, GovernedServer):  # type: ignore[misc]
        #: a pause raises out of the SDK's run rather than becoming an error text
        _failure_error_function = None

        def __init__(self, inner: Any) -> None:  # the SDK's own state is the inner server's
            self._inner = inner
            self._annotations: dict[str, Any] = {}

        def __getattr__(self, name: str) -> Any:
            return getattr(self._inner, name)

        @property
        def name(self) -> str:
            return str(self._inner.name)

        async def connect(self) -> None:
            await self._inner.connect()

        async def cleanup(self) -> None:
            await self._inner.cleanup()

        async def list_tools(self, run_context: Any = None, agent: Any = None) -> Any:
            tools = await self._inner.list_tools(run_context, agent)
            self._annotations.update((t.name, t.annotations) for t in tools)
            return tools

        async def list_prompts(self) -> Any:
            return await self._inner.list_prompts()

        async def get_prompt(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
            return await self._inner.get_prompt(name, arguments)

        async def call_tool(
            self, tool_name: str, arguments: dict[str, Any] | None, meta: Any = None
        ) -> Any:
            inner = self._inner
            if current() is None:
                return await inner.call_tool(tool_name, arguments, meta)

            async def run(args: dict[str, Any]) -> Any:
                result = await inner.call_tool(tool_name, args, meta)
                if result.is_error:
                    said = "".join(getattr(c, "text", "") for c in result.content)
                    raise ToolError(said or f"{tool_name} failed", source="mcp")
                return result.model_dump(mode="json", by_alias=True, exclude_none=True)

            spec = ToolSpec(
                name=tool_name,
                source="mcp",
                server=self.name,
                side_effects=side_effects_of(self._annotations.get(tool_name)),
                **data_of(self._annotations.get(tool_name), strict=False),
            )
            outcome = await bridge.call(Tool(spec, run), dict(arguments or {}))
            if outcome.ok and isinstance(outcome.output, dict):
                return CallToolResult.model_validate(outcome.output)
            text = outcome.output if isinstance(outcome.output, str) else json.dumps(outcome.output)
            return CallToolResult(content=[TextContent(type="text", text=text)], is_error=True)

    return Governed


#: The name of the agent an agent tool's (``Agent.as_tool``) invoke runs, in its code.
RUNS_AGENT: Final = "self"


def _running(tool: Any, agent: Any) -> Any | None:
    """An agent tool (``Agent.as_tool``) as a copy that runs ``agent`` — the governed copy of
    its own — with everything else as it was given (its input, its run options, its output
    extractor): ``None`` when its invoke is not of the shape this SDK release builds (its own
    agent then runs as it is, said once)."""
    from agents import FunctionTool

    from trellis.harness.tools.convert.openai_agents import AGENT_OF_TOOL, AGENT_TOOL

    invoker = tool.on_invoke_tool
    impl = getattr(invoker, "_invoke_tool_impl", None)
    code = getattr(impl, "__code__", None)
    if not isinstance(tool, FunctionTool) or code is None or RUNS_AGENT not in code.co_freevars:
        return None
    assert impl is not None
    cells = dict(zip(code.co_freevars, impl.__closure__ or (), strict=True))
    closure = tuple(
        types.CellType(agent) if name == RUNS_AGENT else cell for name, cell in cells.items()
    )
    running = types.FunctionType(code, impl.__globals__, impl.__name__, impl.__defaults__, closure)
    running.__kwdefaults__ = impl.__kwdefaults__
    copied_invoker = copy.copy(invoker)
    copied_invoker._invoke_tool_impl = running
    made = dataclasses.replace(tool, on_invoke_tool=copied_invoker)
    copied_invoker._function_tool = made
    setattr(made, AGENT_TOOL, True)
    setattr(made, AGENT_OF_TOOL, agent)
    return made


def reachable(target: Any) -> list[Any]:
    """``target`` and every agent it can reach: those it hands off to (an ``Agent``; a
    ``Handoff`` object hides its own) and those its agent tools run, recursively, each once."""
    from agents import Agent

    from trellis.harness.tools.convert.openai_agents import AGENT_OF_TOOL

    found: dict[int, Any] = {}
    pending = [target]
    while pending:
        agent = pending.pop()
        if id(agent) in found:
            continue
        found[id(agent)] = agent
        pending.extend(h for h in agent.handoffs if isinstance(h, Agent))
        inner = (getattr(tool, AGENT_OF_TOOL, None) for tool in agent.tools)
        pending.extend(i for i in inner if isinstance(i, Agent))
    return list(found.values())


def _specs(agent: Any) -> list[ToolSpec]:
    """One agent's own tools, as specs: its function tools (the harness's left out), its
    hosted tools (a hosted MCP server's allowed tools by name, from it), its handoffs."""
    from agents import Agent, FunctionTool, Handoff, HostedMCPTool

    from trellis.harness.tools.convert.openai_agents import AGENT_TOOL, MADE

    found: list[ToolSpec] = []
    for tool in agent.tools:
        if isinstance(tool, FunctionTool):
            if getattr(tool.on_invoke_tool, MADE, False):
                continue  # the harness's own
            spec = ToolSpec(
                name=tool.name,
                description=tool.description or tool.name,
                input_schema=tool.params_json_schema or {"type": "object"},
            )
            found.append(helper(spec) if getattr(tool, AGENT_TOOL, False) else spec)
        elif isinstance(tool, HostedMCPTool):
            config = dict(tool.tool_config)
            allowed = config.get("allowed_tools")
            names = allowed if isinstance(allowed, list) else []
            server = str(config.get("server_label") or tool.name)
            found.extend(ToolSpec(name=str(n), source="mcp", server=server) for n in names)
        else:  # a hosted tool, a computer, a shell: by its name
            found.append(ToolSpec(name=str(tool.name), description=type(tool).__name__))
    for handoff in agent.handoffs:
        if isinstance(handoff, Agent):
            name, said = Handoff.default_tool_name(handoff), handoff.handoff_description
        else:
            name, said = handoff.tool_name, handoff.tool_description
        found.append(helper(ToolSpec(name=name, description=said or name, side_effects="read")))
    return found


#: The MCP servers whose tools could not be listed for the run's tool set (said once each).
_UNLISTED: set[str] = set()


async def _listed(server: Any) -> list[ToolSpec]:
    """An MCP server's tools, as the server lists them now (``[]``: it cannot, said once)."""
    try:
        tools = await server.list_tools()
    except Exception as exc:
        if server.name not in _UNLISTED:
            _UNLISTED.add(server.name)
            log.info(
                "an MCP server's tools could not be listed for the run's tool set: they are "
                "left out of the tool hints and the catalog until it can",
                extra={"server": server.name, "reason": type(exc).__name__},
            )
        return []
    return [
        ToolSpec(
            name=tool.name,
            description=tool.description or tool.name,
            input_schema=dict(tool.input_schema or {"type": "object"}),
            source="mcp",
            server=server.name,
            side_effects=side_effects_of(tool.annotations),
            **data_of_listed(tool),
        )
        for tool in tools
    ]


@functools.lru_cache(maxsize=SAID_ONCE)  # said once per agent and handoff, of the most recent
def _hidden_handoff(agent_id: str, to: str) -> None:
    log.warning(
        "a handoff is a Handoff object, which hides its agent until it is invoked: that "
        "agent's own tools are not governed (hand off to the Agent itself)",
        extra={"agent_id": agent_id, "handoff": to},
    )


def _ungoverned_inside(agent_id: str, tool: Any) -> None:
    """Say once when an agent tool's agent has tools of its own the harness did not make and
    the agent tool could not be pointed at the governed copy: the SDK runs that agent itself,
    so their calls are not governed."""
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
        "govern them: hand off to the agent instead, or give its tools to the agent calling it",
        extra={"agent_id": agent_id, "agent_tool": tool, "tools": tools},
    )


def _options(runtime: Runtime, target: Any) -> dict[str, Any]:
    """``Runner.run``'s options: the run's framework options, with the memory service's thread
    as the session (``bridges.session``) when the run keeps no thread of its own and memory is
    on, and the model names routed through Bifrost when it serves the harness
    (:func:`_routed`)."""
    options = dict(runtime.framework_options)
    if runtime.run_memory is not None and _thread_store(options) is MEMORY_THREAD:
        options["session"] = MemorySession(runtime)
    return _routed(runtime, target, _steering(runtime, _told_unknown_tools(options)))


def _told_unknown_tools(options: dict[str, Any]) -> dict[str, Any]:
    """The options with a call to a tool the model made up answered as an error the model
    reads, so the run goes on — as on every other adapter — and not a ``ModelBehaviorError``
    that ends it: the SDK's own ``RunConfig.tool_not_found_behavior="return_error_to_model"``
    (its message: ``Tool '<name>' not found.``)."""
    from agents import RunConfig

    config = options.get("run_config") or RunConfig()
    told = dataclasses.replace(config, tool_not_found_behavior="return_error_to_model")
    return {**options, "run_config": told}


def _steering(runtime: Runtime, options: dict[str, Any]) -> dict[str, Any]:
    """The options with the messages steered into the run (``Runtime.steered``) joining the
    input of each model call where they were taken, as the user's
    (``RunConfig.call_model_input_filter``, after the run's own filter): the SDK keeps a
    filter's input for that call only, so each call is given them again, in place."""
    from agents import RunConfig
    from agents.run_config import ModelInputData

    config = options.get("run_config") or RunConfig()
    theirs = config.call_model_input_filter
    kept: list[tuple[int, dict[str, Any]]] = []

    async def steered(data: Any) -> Any:
        given = data.model_data
        if theirs is not None:
            given = theirs(data)
            if inspect.isawaitable(given):
                given = await given
        items: list[Any] = list(given.input)
        kept.extend((len(items), {"role": "user", "content": s.text}) for s in runtime.steered())
        for offset, (at, item) in enumerate(kept):
            items.insert(at + offset, item)
        return ModelInputData(input=items, instructions=given.instructions)

    return {**options, "run_config": dataclasses.replace(config, call_model_input_filter=steered)}


def _routed(runtime: Runtime, target: Any, options: dict[str, Any]) -> dict[str, Any]:
    """With Bifrost serving the harness (a route with ``Capability.ATTRIBUTION``), every model
    name of the run's agents goes through it
    (``RunConfig.model_provider``: :func:`bifrost_provider`, the name mapped to Bifrost's
    ``provider/model``) — unless the run's own ``run_config`` names a model provider of its
    own (code over the environment). An agent whose model is an object of its own calls its
    provider directly, which is said once per agent."""
    from agents import Agent, RunConfig
    from agents.models.multi_provider import MultiProvider

    gateway = runtime.agent.harness.model_gateway
    if not gateway.has(Capability.ATTRIBUTION):
        return options
    config = options.get("run_config") or RunConfig()
    if not isinstance(config.model_provider, MultiProvider):
        return options  # the run's own provider
    for agent in reachable(target):
        model = agent.model if isinstance(agent, Agent) else None
        if model is not None and not isinstance(model, str):
            _unrouted(runtime.agent_id, agent.name, type(model).__name__)
    provider = bifrost_provider(cast("Bifrost", gateway))
    return {**options, "run_config": dataclasses.replace(config, model_provider=provider)}


@functools.lru_cache(maxsize=SAID_ONCE)  # said once per agent, of the most recent
def _unrouted(agent_id: str, name: str, model: str) -> None:
    log.warning(
        "%s: the OpenAI agent %r has a model object (%s), which calls its provider directly, "
        "not Bifrost: give it a model name to route it through LLM_BASE_URL",
        agent_id,
        name,
        model,
        extra={"agent_id": agent_id, "agent": name},
    )


def bifrost_provider(gateway: Bifrost) -> Any:
    """The OpenAI Agents SDK's model provider for Bifrost: each model name (the SDK's default
    for an agent naming none) mapped to Bifrost's form, asked on its chat completions with the
    virtual key, the deny-all MCP scope and the run's dimensions and ``traceparent`` (read per
    request), no retries of the client's own (the gateway retries). One per gateway and event
    loop (a client's pooled connections belong to the loop that opened them)."""
    providers = per_loop(_PROVIDERS, PROVIDERS_KEPT)
    key = (gateway.url, gateway.key)
    found = providers.get(key)
    if found is None:
        found = providers[key] = _bifrost_provider_class()(gateway)
    return found


@functools.cache
def _bifrost_provider_class() -> type:
    import openai
    from agents.models import get_default_model
    from agents.models.interface import ModelProvider
    from agents.models.openai_provider import OpenAIProvider

    from trellis.harness.gateways import BIFROST_CLIENT_RETRIES, NO_KEY
    from trellis.harness.prompts import GatewayHeaders

    class BifrostProvider(ModelProvider):
        def __init__(self, gateway: Bifrost) -> None:
            self.gateway = gateway
            client = openai.AsyncOpenAI(
                base_url=gateway.url,
                api_key=gateway.key or NO_KEY,
                default_headers=GatewayHeaders(),  # type: ignore[arg-type] # read per request
                max_retries=BIFROST_CLIENT_RETRIES,
            )
            self.inner = OpenAIProvider(openai_client=client, use_responses=False)

        def get_model(self, model_name: str | None) -> Any:
            return self.inner.get_model(self.gateway.routed(model_name or get_default_model()))

    return BifrostProvider


def _thread_store(options: Mapping[str, Any]) -> Any:
    """The thread store of a run with these options (``OpenAIAgentsAdapter.thread_store``)."""
    if options.get("session") is not None:
        return options["session"]
    server = next((key for key in SERVER_THREAD if options.get(key)), None)
    return ServerThread(server) if server is not None else MEMORY_THREAD


def _hooks() -> Any:
    """The harness's ``RunHooks``, always: every model call watched for a stall, its hosted
    tool calls recorded, and its usage counted toward the run's budget (``budget.py``)."""
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
