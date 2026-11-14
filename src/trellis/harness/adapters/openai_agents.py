"""OpenAI Agents SDK ``Agent``\\ s.

* input: a message list, the memory context first as a ``system`` message; where memory keeps
  the conversation (no ``session`` given), its recent messages follow as the SDK's own input
  items — messages, ``function_call`` and ``function_call_output`` — as a session replays them;
* transcript: the run's new items — its answers, its tool calls and their results — in order;
* run: ``Runner.run``/``Runner.run_streamed`` on a clone of the agent carrying the harness
  tools next to its own (the caller's agent is never changed) — and every agent it can hand off
  to cloned too, recursively, each one's own tools governed (:func:`governed_agents`) — with
  the harness's ``RunHooks`` always (``hooks.openai_agents``): every reply's usage counted
  toward the run's budget, and the run's own ``RunHooks`` (``hooks=``, a framework option)
  called as the SDK calls them;
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
* pause: ``trellis.current().ask`` stops the run and a resume re-runs it against the journal,
  its model told which call the person answered about (``adapters.claude.RESUMED_ASKED``, as a
  resumed Claude session is: a model run again from the same input may decide otherwise);
  that call, made again in other words, runs as it was (``Replay.made``);
  a call governance asks about is held at the SDK's own gate (:func:`_gated`: a
  ``needs_approval`` policy asking governance, ``tools.convert.openai_agents.held``), as are
  the SDK's own ``needs_approval`` tools: they pause with the SDK's ``RunState``, and a resume
  approves or rejects on that state and continues it (``RunState.approve``/``reject``) — the
  other calls governance's gate held in the same reply decided again then, one by one
  (:func:`_decided`: the SDK keeps them waiting and does not ask the gate again). The SDK
  takes no edited arguments and no answer in place of a tool's result, so: a reject carries the
  reviewer's reason (``rejection_message``, the text the model reads); an answer is a reject
  whose message is the answer; an edit is a reject telling the model to call the tool again
  with the edited arguments — and that call, when the model makes it with exactly those
  arguments, is approved in the same attempt instead of asking again;
* framework options: keyword arguments of ``Runner.run``/``run_streamed`` (``max_turns``,
  ``run_config``, ``context``, ``hooks``...), except the ones the harness gives itself
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
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
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
from trellis.contracts.errors import is_pause_signal
from trellis.harness import snapshots
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
from trellis.harness.adapters.claude import resumed_call
from trellis.harness.bounded import LRU, SAID_ONCE
from trellis.harness.bridges.helpers import Helpers
from trellis.harness.bridges.session import MemorySession, input_items
from trellis.harness.conversation import remembered
from trellis.harness.features import MEMORY_TOOLS
from trellis.harness.gateways import Capability, per_loop
from trellis.harness.history import ASKED_WITH, MEMORY_THREAD, Called, Said, Speaker, answered
from trellis.harness.journal import Pending
from trellis.harness.runtime import Runtime, current, reason_of
from trellis.harness.tools import bridge
from trellis.harness.tools.base import Tool, data_of
from trellis.harness.tools.mcp import data_of_listed, side_effects_of

log = logging.getLogger("trellis.harness")

#: ``Runner.run``'s arguments the harness gives itself: the agent (a clone carrying the harness
#: tools) and its input (with the memory context, or the resumed state).
OWN_ARGUMENTS: Final = ("starting_agent", "input")
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
    #: the person's decision covers the tool's later calls (``remember`` beyond ``"once"``):
    #: the SDK's own ``always_approve``/``always_reject`` for the rest of its run
    always: bool = False


@dataclass(frozen=True, slots=True)
class _Again:
    """A resume that runs the agent again from its input, its model told (``told``) which call
    the person answered about."""

    input: Any
    told: str


@dataclass(frozen=True, slots=True)
class ServerThread:
    """A conversation OpenAI's servers keep (``key``: the run option that names it): durable."""

    key: str


#: What the harness says once when the SDK's own tracing is on (:func:`sdk_tracing_said`).
SDK_TRACING: Final = (
    "the OpenAI Agents SDK's own tracing is on: it exports every run's spans to OpenAI, tool "
    "inputs and outputs and model inputs included (RunConfig.trace_include_sensitive_data "
    "defaults to True), outside the harness's redaction. The harness leaves the SDK's default "
    "as it is; to turn it off call agents.set_tracing_disabled(True) or set "
    "OPENAI_AGENTS_DISABLE_TRACING=1 (or RunConfig(tracing_disabled=True) per run; "
    "RunConfig(trace_include_sensitive_data=False), or "
    "OPENAI_AGENTS_TRACE_INCLUDE_SENSITIVE_DATA=false, keeps the spans without the data). The "
    "run's own trace is the harness's and the gateway's"
)


@functools.cache
def sdk_tracing_said() -> bool:
    """Warn once, when an agent is wrapped, that the SDK's own tracing — which exports to
    OpenAI, tool inputs and outputs included, unless turned off — is left as the application
    set it (:data:`SDK_TRACING`: the exact switches; ``True`` when said)."""
    from agents.tracing import get_trace_provider

    provider = get_trace_provider()
    disabled = getattr(provider, "_disabled", False)
    if disabled:
        return False
    log.warning(SDK_TRACING)
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

    def __init__(self, target: Any = None) -> None:
        #: the tools the agents declare (:meth:`native`), once read
        self._static: list[ToolSpec] | None = None
        #: the harness's model route, once the agent is wrapped (:meth:`wrapped`)
        self.gateway: Any = None
        if target is not None:  # every agent it reaches can be read, or ``h.wrap`` says why
            reachable(target)

    def wrapped(self, target: Any, options: Mapping[str, Any], gateway: Any) -> None:
        """Where the agent is wrapped (``h.wrap``), with the options every one of its runs is
        given and the harness's model route (``gateway``, kept: who keeps a server-side thread
        depends on it, :meth:`thread_store`): the SDK's own tracing said once
        (:func:`sdk_tracing_said`); a ``session`` its runs would share made each
        conversation's own, or refused when it cannot be (:func:`per_conversation`); a live
        sandbox session or a sandbox state to resume, one sandbox for every user, refused
        (:func:`_one_sandbox`); and what the agents use that their model route cannot carry
        refused (:func:`unserved`)."""
        self.gateway = gateway
        sdk_tracing_said()
        session = options.get("session")
        if session is not None:
            _scoped_id(session)  # refused here when it cannot be one per conversation
        config = options.get("run_config")
        if config is not None:
            _one_sandbox(_coerced(config))
        unserved(gateway, target, dict(options))
        _memory_said(target)

    def thread_store(self, target: Any, options: Mapping[str, Any]) -> Any | None:
        """The run's ``session``, or the conversation OpenAI's servers keep
        (:data:`SERVER_THREAD`) when the run's model route keeps it (:func:`keeps_thread`: a
        route that ignores it keeps none, and memory keeps the history as with no thread);
        else the memory service's thread, the session a run with memory on is given
        (``bridges.session``)."""
        return _thread_store(options, keeps_thread(self.gateway, target, options))

    def steers(self, target: Any) -> bool:
        """Every agent: a steered message joins the input of its next model call
        (``RunConfig.call_model_input_filter``, :func:`_steering`)."""
        return True

    def asks(self, target: Any) -> bool:
        return False  # the SDK pauses for an approval, never for a question of the model's

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
        from agents.exceptions import MaxTurnsExceeded

        helpers, agent, hooks, options = await self._prepared(target, native_input, run)
        try:
            given = await self._input(agent, native_input)
            result = await Runner.run(agent, given, hooks=hooks, **options)
            resumed = native_input
            while (state := await _continued(result, resumed, run.runtime)) is not None:
                resumed = None  # an edited call is approved once
                result = await Runner.run(agent, state, hooks=hooks, **options)
        except MaxTurnsExceeded as exc:
            result = _at_turn_limit(run.runtime, exc, options)
        except BaseException as exc:
            await helpers.closed(exc)
            raise
        await _closed(helpers, result)
        return result

    async def stream(self, target: Any, native_input: Any, run: Invocation) -> AsyncIterator[Any]:
        from agents import Runner
        from agents.exceptions import MaxTurnsExceeded

        helpers, agent, hooks, options = await self._prepared(target, native_input, run)
        given = await self._input(agent, native_input)
        result = Runner.run_streamed(agent, given, hooks=hooks, **options)
        streamed = result
        error: BaseException | None = None
        try:
            async for delta in _deltas(result):
                yield delta
            resumed = native_input
            while (state := await _continued(result, resumed, run.runtime)) is not None:
                resumed = None
                result = streamed = Runner.run_streamed(agent, state, hooks=hooks, **options)
                async for delta in _deltas(result):
                    yield delta
        except MaxTurnsExceeded as exc:
            result = _at_turn_limit(run.runtime, exc, options)
        except BaseException as exc:
            error = exc
            raise
        finally:
            # cut short (cancelled, out of time), the SDK stops its run without waiting for
            # it: its tool calls end (on the run's stream too) before the run does
            assert streamed.run_loop_task is not None  # set by run_streamed
            await asyncio.wait({streamed.run_loop_task})
            if error is not None:
                await helpers.closed(error)
        await _closed(helpers, result)
        yield Output(result)

    def extract(self, target: Any, output: Any) -> Extracted:
        transcript = said_in(output.new_items)
        pause = None
        if output.interruptions:
            item = output.interruptions[0]
            pause = NativePause(
                value=None,
                native_id=item.call_id,
                state=output.to_state().to_json(),
                tool=item.name,
                args=_asked_about(item),
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
            call = pending.call
            if call is None or pending.interrupt.tool_call is not None:
                return native_input
            # a question a tool asked while it ran: the model makes that call again
            return _Again(native_input, resumed_call(call, call.tool, denied=False, asked=True))
        state, call_id = pending.native_state, pending.native_id
        decision = resolution.decision
        always = remembered(pending.interrupt, resolution)
        if decision is InterruptDecision.APPROVE:
            return _Continue(state, call_id, approve=True, always=always)
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
        return _Continue(
            state, call_id, approve=False, message=reason_of(resolution), always=always
        )

    def workdir(self, target: Any, options: Mapping[str, Any]) -> Path | None:
        return None  # no folder of its own: a sandbox's outputs/ are its outputs

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
        from agents.lifecycle import RunHooksBase

        own = [key for key in options if key in OWN_ARGUMENTS]
        if own:
            raise ConfigurationError(
                f"framework_options {', '.join(map(repr, own))}: the harness gives Runner.run "
                "its agent and its input itself"
            )
        hooks = options.get("hooks")
        if hooks is not None and not isinstance(hooks, RunHooksBase):
            raise ConfigurationError(
                f"framework_options 'hooks': Runner.run's hooks are the SDK's RunHooks, not "
                f"{type(hooks).__name__} (the harness's own hooks go on h.wrap(hooks=))"
            )
        taken = inspect.signature(Runner.run).parameters
        unknown = [key for key in options if key not in taken]
        if unknown:
            raise ConfigurationError(
                f"framework_options {', '.join(map(repr, unknown))}: Runner.run takes no such "
                f"argument (it takes {', '.join(k for k in taken if k not in OWN_ARGUMENTS)})"
            )

    # ------------------------------------------------------------------ internals
    async def _prepared(
        self, target: Any, native_input: Any, run: Invocation
    ) -> tuple[Helpers, Any, Any, dict[str, Any]]:
        """What an attempt runs with, streamed or not: its helpers, the governed copy of its
        agent, the harness's ``RunHooks`` around the run's own, and ``Runner.run``'s options."""
        helpers = Helpers(run.runtime, self.name)
        options = _options(run.runtime, target, native_input)
        agent = self._agent(target, run, helpers, await _gated(run))
        return helpers, agent, _hooks(options.pop("hooks", None)), options

    @staticmethod
    def _agent(
        target: Any,
        run: Invocation,
        helpers: Helpers,
        gated: Mapping[str, ToolSpec],
    ) -> Any:
        # the memory tools (pull, tool_search) go to every agent the run can reach: an agent
        # it hands off to answers from memory as the one it started with
        memory = {t.spec.name for t in run.tools if t.feature in MEMORY_TOOLS}
        shared = [t for t in run.native_tools or () if t.name in memory]
        runtime = run.runtime
        return governed_agents(
            target,
            added=list(run.native_tools or ()),
            shared=shared,
            agent_id=runtime.agent_id,
            helpers=helpers,
            gated=gated,
            nested=lambda agent, own: _child_config(runtime, agent, own),
        )

    @staticmethod
    async def _input(agent: Any, native_input: Any) -> Any:
        if isinstance(native_input, _Again):
            return native_input.input
        if not isinstance(native_input, _Continue):
            return native_input
        from agents import RunState

        state = await RunState.from_json(agent, native_input.state)
        for item in state.get_interruptions():
            if item.call_id != native_input.call_id:
                continue  # still waiting: the run pauses on it next
            if native_input.approve:
                state.approve(item, always_approve=native_input.always)
            else:
                state.reject(
                    item, always_reject=native_input.always, rejection_message=native_input.message
                )
        return state


async def _closed(helpers: Helpers, result: Any) -> None:
    """The attempt's helpers ended with the SDK's run (``Helpers.closed``) — unless the run
    paused on an approval of the SDK's own (its ``interruptions``): an agent tool waiting on
    one inside it (``tools.convert.openai_agents.waits_inside``) stays open, and the resume,
    which makes its call again, ends it."""
    if not result.interruptions:
        await helpers.closed(None)


def governed_agents(
    target: Any,
    *,
    added: list[Any],
    agent_id: str,
    shared: Sequence[Any] = (),
    helpers: Helpers | None = None,
    gated: Mapping[str, ToolSpec] | None = None,
    nested: Callable[[Any, Any], Any],
) -> Any:
    """``target`` and every agent reachable through its handoffs and its agent tools
    (``Agent.as_tool``), each a copy whose own tools are governed
    (``tools.convert.openai_agents.governed``: hooks, the catalog's rules, the journal, and
    offered per turn only while the run offers them: the tool hints' narrowing), whose MCP
    servers' calls are governed the same way (:class:`GovernedServer`), whose sandbox
    capabilities' tools — made inside the SDK's run (``exec_command``, ``apply_patch``...) —
    are governed as the SDK makes them (:func:`governed_capability`), and whose agent tools
    run the governed copy of their agent — recursively, each agent copied once, so handoffs that
    lead back are kept — ``added`` (the run's harness tools) on ``target``'s and ``shared``
    (the run's memory tools) on every other's, after the agent's own, in the order the agent
    gives them. A tool an agent carries already (built with the harness) is not added twice. A
    ``Handoff`` object (``handoff(agent, ...)``) is kept with all it gives the SDK — its name,
    its input filter, its callback — and hands off to the governed copy of its agent
    (:func:`_handed`). With ``helpers``, each agent tool's call is a helper of the run; its
    child run's config is the one ``nested`` makes of its agent and its own
    (:func:`_child_config`), and its model calls are watched (:func:`_running`). The function
    tools ``gated`` names (those governance may ask about, :func:`_gated`) are held at the
    SDK's own approval gate when it asks (``tools.convert.openai_agents.held``). The agents you
    wrapped are not changed."""
    from agents import Agent

    from trellis.harness.tools.convert.openai_agents import AGENT_OF_TOOL, governed, held

    gates = gated or {}

    def gate(tool: Any) -> Any:
        spec = gates.get(getattr(tool, "name", ""))
        return tool if spec is None else held(tool, spec)

    copies: dict[int, Any] = {}

    def copied(agent: Any, extra: Sequence[Any]) -> Any:
        found = copies.get(id(agent))
        if found is not None:
            return found
        copy = copies[id(agent)] = agent.clone()
        own = []
        for tool in agent.tools:
            inner = getattr(tool, AGENT_OF_TOOL, None)
            ran = (
                _running(tool, copied(inner, shared), nested) if isinstance(inner, Agent) else None
            )
            if inner is not None and ran is None:
                _ungoverned_inside(agent_id, tool)
            own.append(gate(governed(ran or tool, helpers)))
        carried = {getattr(t, "name", None) for t in agent.tools}
        copy.tools = [*own, *(gate(t) for t in extra if t.name not in carried)]
        copy.mcp_servers = [GovernedServer.of(s) for s in agent.mcp_servers]
        capabilities = getattr(agent, "capabilities", None)
        if capabilities is not None:  # a SandboxAgent's
            copy.capabilities = [governed_capability(c) for c in capabilities]
        handoffs = []
        for handoff in agent.handoffs:
            if isinstance(handoff, Agent):
                handoffs.append(copied(handoff, shared))
            else:
                handoffs.append(_handed(handoff, lambda a: copied(a, shared), agent_id))
        copy.handoffs = handoffs
        return copy

    return copied(target, added)


def governed_capability(capability: Any) -> Any:
    """A ``SandboxAgent``'s capability on the run's copy (the SDK's seam: what its ``tools()``
    makes inside the SDK's run, bound to the run's sandbox — ``exec_command``, ``write_stdin``,
    ``view_image``, ``apply_patch``, ``load_skill``...), each tool governed as the agent's own
    are (``tools.convert.openai_agents.governed``: the catalog, a person when it asks, the
    journal, the record, the stream) without changing what it does; refused before the first
    model call when the model's route would drop it (:func:`_capability_served`). A sandbox
    ``Memory``'s own model calls — its extraction and consolidation agents, run by the SDK when
    the sandbox closes — are made through the run's model route with their usage counted
    (:func:`through_run`). Everything else is the capability's own; the agent's capability is
    not changed."""
    from agents.sandbox.capabilities.memory import Memory

    made = capability.model_copy()
    if isinstance(made, Memory) and made.generate is not None:
        generate = made.generate
        runtime = current()
        assert runtime is not None  # the run's copy of an agent is made inside the run
        provider = _provider_of(runtime)
        made.generate = dataclasses.replace(
            generate,
            phase_one_model=through_run(generate.phase_one_model, provider),
            phase_two_model=through_run(generate.phase_two_model, provider),
        )
    object.__setattr__(made, "__class__", _governing(type(capability)))
    return made


#: How many sandbox capability classes are kept in their governed form (:func:`_governing`):
#: the SDK has under ten; one left out is made again.
CAPABILITY_KINDS: Final = 64


@functools.lru_cache(maxsize=CAPABILITY_KINDS)
def _governing(kind: type) -> type:
    """``kind`` (a sandbox ``Capability`` class) whose tools are governed
    (:func:`governed_capability`) and, like the agent's own (:func:`_gated`), held at the SDK's
    approval gate when governance asks — the approved call continues the SDK's paused run, in
    the same sandbox — unless the run has tool hooks (their question comes first: the call
    pauses as a harness pause does)."""
    from agents import FunctionTool

    from trellis.harness.tools.convert.openai_agents import governed, held, own_spec

    class Governing(kind):  # type: ignore[misc,valid-type]
        def tools(self) -> list[Any]:
            made = [governed(tool) for tool in super().tools()]
            runtime = current()
            if runtime is None or runtime.agent.hooks:
                return made
            return [
                held(tool, own_spec(runtime, tool)) if isinstance(tool, FunctionTool) else tool
                for tool in made
            ]

        def sampling_params(self, sampling_params: dict[str, Any]) -> dict[str, Any]:
            asked = super().sampling_params(sampling_params)
            _capability_served(self, sampling_params.get("model"), asked)
            return asked

    Governing.__name__ = Governing.__qualname__ = kind.__name__
    return Governing


def _capability_served(capability: Any, model: Any, asked: Mapping[str, Any]) -> None:
    """Refuse, as the SDK prepares the run's sandbox agent (before its first model call), a
    capability whose tools or request parameters (``asked``) its model's route drops: a name
    the harness routes through Bifrost, which serves it on chat completions only
    (:func:`_whole`), which keeps function tools only and no Responses parameter (a
    ``Compaction``'s ``context_management``)."""
    from agents.exceptions import UserError
    from agents.models.chatcmpl_converter import Converter

    if not isinstance(model, str):
        return
    runtime = current()
    assert runtime is not None  # the SDK prepares the run's sandbox agent inside the run
    gateway = runtime.agent.harness.model_gateway
    route = _route_of(gateway, runtime.framework_options)
    if route is None or not routes(route, model) or _whole(gateway, model):
        return
    found = [f"its request parameters ({', '.join(asked)})"] if asked else []
    for tool in capability.tools():
        try:
            Converter.tool_to_openai(tool)
        except UserError:
            found.append(f"its tool {getattr(tool, 'name', type(tool).__name__)!r}")
    if found:
        raise ConfigurationError(
            f"the sandbox capability {type(capability).__name__} needs the Responses API, and "
            f"Bifrost serves {model!r} through chat completions, which would drop "
            f"{' and '.join(found)}: use a model Bifrost serves on the Responses API "
            "(openai/...), or leave the capability out"
        )


def through_run(model: Any, provider: Any) -> Any:
    """A sandbox ``Memory``'s model (its ``phase_one_model``/``phase_two_model``) asked through
    the model route of the run whose sandbox it summarises: a name, through ``provider`` (the
    run's, :func:`_provider_of` — the route it was made with, so a sandbox you keep and close
    after the run still asks through it), each reply's
    usage counted toward the budget of the run it is asked in and its call watched for a
    stall. Two of one name are the same model (the SDK compares a session's memory configs).
    A model object is your own: as it is."""
    return _through_run_class()(model, provider) if isinstance(model, str) else model


def _provider_of(runtime: Runtime) -> Any:
    """The model provider of ``runtime``'s run as the harness routes it (:func:`_routed`): its
    ``RunConfig.model_provider``, on Bifrost for the names a ``MultiProvider`` leaves to its
    default provider."""
    from agents import RunConfig

    gateway = runtime.agent.harness.model_gateway
    route = _route_of(gateway, runtime.framework_options)
    if route is not None:
        return routed_provider(cast("Bifrost", gateway), route)
    given = runtime.framework_options.get("run_config")
    return (RunConfig() if given is None else _coerced(given)).model_provider


@functools.cache
def _through_run_class() -> type:
    from agents.models.interface import Model

    from trellis.harness.hooks.openai_agents import counted

    class ThroughRun(Model):  # type: ignore[misc]
        def __init__(self, name: str, provider: Any) -> None:
            self.name = name
            self._provider = provider

        def __eq__(self, other: object) -> bool:
            return isinstance(other, ThroughRun) and other.name == self.name

        def __hash__(self) -> int:
            return hash(self.name)

        def _model(self) -> Any:
            return self._provider.get_model(self.name)

        async def get_response(self, *args: Any, **kwargs: Any) -> Any:
            runtime = current()
            if runtime is None:  # after its run: asked through the run's route, no run to count
                return await self._model().get_response(*args, **kwargs)
            step = f"openai_agents:memory:{self.name}"
            runtime.model_started(step)
            try:
                response = await self._model().get_response(*args, **kwargs)
            finally:
                runtime.model_ended(step)
            counted(runtime, self.name, response.usage)
            return response

        def stream_response(self, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
            # the SDK's ``Model`` interface; its memory agents are not streamed
            return cast("AsyncIterator[Any]", self._model().stream_response(*args, **kwargs))

    return ThroughRun


def _memory_said(target: Any) -> None:
    """Say once, where an agent is wrapped, of a sandbox ``Memory`` that generates memories:
    its consolidation agent writes the memory files in the sandbox with tools the SDK makes
    for it outside any seam the harness has, so those writes are the SDK's own, not governed or
    recorded (its model calls go through the run: :func:`through_run`)."""
    from agents.sandbox.capabilities.memory import Memory

    for agent in reachable(target):
        for capability in getattr(agent, "capabilities", None) or ():
            if isinstance(capability, Memory) and capability.generate is not None:
                _said_memory(agent.name)


@functools.lru_cache(maxsize=SAID_ONCE)  # said once per agent name, of the most recent
def _said_memory(name: str) -> None:
    log.warning(
        "the sandbox agent %r has a Memory that generates memories: its model calls go through "
        "the run's model route and budget, but the SDK's consolidation agent writes the memory "
        "files in the sandbox itself, with tools the harness cannot reach, so those writes are "
        "not governed or recorded (Memory(generate=None) reads memories only; the memory "
        "service keeps long-term memory governed)",
        name,
        extra={"agent": name},
    )


async def _gated(run: Invocation) -> dict[str, ToolSpec]:
    """The run's tools — the harness's and the agents' own — whose calls governance may ask a
    person about, as the catalog says now (``Governance.may_ask``), by name: these are held at
    the SDK's own approval gate, so an approved call continues the paused run. Any other tool
    the SDK runs as ever (a call it cannot read the arguments of is not held for a person); a
    call the catalog starts asking about later in the run, and every call of a run with tool
    hooks, pause as a harness pause does."""
    runtime = run.runtime
    if runtime.agent.hooks:  # a tool hook may rewrite the call, deny it or ask its own question
        return {}  # first: its calls pause as the harness's do
    own = [t.spec for t in run.tools if not t.confined]  # a confined write is announced
    specs = {s.name: s for s in [*runtime.native.values(), *own]}
    governance = runtime.agent.harness._governance(runtime.tenant)
    asks = await governance.may_ask(list(specs.values()), agent_id=runtime.agent_id)
    return {name: specs[name] for name in asks}


class GovernedServer:
    """An OpenAI agent's MCP server as the harness runs it: every call of its tools inside a
    run goes through the bridge (``tools.bridge.call``: hooks, the catalog's rules — its tools'
    annotations as they were listed for their risk —, a person when governance asks, the
    journal, the record), the server itself as it was (connected, listed, its prompts and
    resources read, cleaned up by its owner). A tool's own error (the server's result) is the
    result the model reads, as before. A call the server could not take (the call itself
    raised) is recorded as the call's failure and is then the SDK's to tell, as without the
    harness: the model reads what the server's — else the agent's — ``failure_error_function``
    makes of it, and with ``failure_error_function=None`` the run ends with the error — the
    model reading the words of the run's ``after_tool`` hooks instead when they reworded the
    failure, and the result when they made it one (``bridge.hooked``). A call that pauses
    the run raises out of the SDK's run whatever that function is. Made by
    :meth:`of`, a subclass of the SDK's ``MCPServer`` built when first needed."""

    @staticmethod
    def of(server: Any) -> Any:
        """``server`` governed (a server already governed is kept as it is)."""
        cls = _governed_server()
        return server if isinstance(server, cls) else cls(server)


#: The attribute of the error a failed MCP call raised holding what the run's ``after_tool``
#: hooks made the call say (``bridge.hooked``): the SDK tells the model of that error in these
#: words (:func:`_worded`).
WORDED: Final = "_trellis_worded"


def _worded(error: BaseException) -> str | None:
    """What the run's hooks made an MCP call's failure say, for ``error`` as the SDK hands it to
    a ``failure_error_function``: the error the call raised, or the SDK's own error raised from
    it (``AgentsException(...) from``); ``None`` when they said nothing of it."""
    for held in (error, error.__cause__):
        said = getattr(held, WORDED, None)
        if isinstance(said, str):
            return said
    return None


#: The SDK's ``MCPServer`` method that says how a failed call is told to the model, which the
#: governed server wraps (a private name: checked where the class is made).
FAILURE_FUNCTION: Final = "_get_failure_error_function"


def _wraps_failure_function(server: type) -> None:
    """Refuse an SDK whose ``MCPServer`` no longer has :data:`FAILURE_FUNCTION`: the governed
    server's override of it would never be called, and a pause inside an MCP call would become
    an error text the model reads instead of stopping the run."""
    if not hasattr(server, FAILURE_FUNCTION):
        raise ConfigurationError(
            f"this OpenAI Agents SDK's MCPServer no longer has {FAILURE_FUNCTION}, which the "
            "harness wraps so a pause inside an MCP call stops the run: use a tested SDK "
            "release (docs/versioning.md)"
        )


@functools.cache
def _governed_server() -> type:
    from agents.mcp import MCPServer
    from mcp.types import CallToolResult, TextContent

    _wraps_failure_function(MCPServer)

    class Governed(MCPServer, GovernedServer):  # type: ignore[misc]
        def __init__(self, inner: Any) -> None:  # the SDK's own state is the inner server's
            self._inner = inner
            self._annotations: dict[str, Any] = {}

        def _get_failure_error_function(self, agent_failure_error_function: Any) -> Any:
            """The server's own way of telling the model a call failed (else its agent's),
            except while the run is pausing: a pause raises out of the SDK's run rather than
            becoming an error text."""
            theirs = self._inner._get_failure_error_function(agent_failure_error_function)
            if theirs is None:
                return None

            def told(context: Any, error: Exception) -> Any:
                runtime = current()
                if runtime is not None and runtime.pending is not None:
                    raise error
                text = theirs(context, error)  # the server's own way, which still runs
                worded = _worded(error)
                return text if worded is None else worded

            return told

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

        # what the SDK's base class answers itself (no tools cached, no resources): the
        # inner server's own
        @property
        def cached_tools(self) -> Any:
            return self._inner.cached_tools

        async def list_resources(self, cursor: str | None = None) -> Any:
            return await self._inner.list_resources(cursor)

        async def list_resource_templates(self, cursor: str | None = None) -> Any:
            return await self._inner.list_resource_templates(cursor)

        async def read_resource(self, uri: str) -> Any:
            return await self._inner.read_resource(uri)

        async def call_tool(
            self, tool_name: str, arguments: dict[str, Any] | None, meta: Any = None
        ) -> Any:
            inner = self._inner
            if current() is None:
                return await inner.call_tool(tool_name, arguments, meta)

            raised: list[Exception] = []

            async def run(args: dict[str, Any]) -> Any:
                try:
                    result = await inner.call_tool(tool_name, args, meta)
                except Exception as error:
                    raised.append(error)
                    raise
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
            # the server's own tries, when it makes them (``max_retry_attempts``), are its only
            retried = bool(getattr(inner, "max_retry_attempts", 0))
            outcome = await bridge.call(Tool(spec, run, retried=retried), dict(arguments or {}))
            if outcome.ok and isinstance(outcome.output, dict):
                return CallToolResult.model_validate(outcome.output)
            text = outcome.output if isinstance(outcome.output, str) else json.dumps(outcome.output)
            hooked = bridge.hooked(outcome)
            if raised and not is_pause_signal(raised[-1]) and not (hooked and outcome.ok):
                # the call itself failed: the SDK's to tell, or to raise — in the words of the
                # run's after_tool hooks when they worded it (:func:`_worded`)
                if hooked:
                    setattr(raised[-1], WORDED, text)
                raise raised[-1]
            return CallToolResult(
                content=[TextContent(type="text", text=text)], is_error=not outcome.ok
            )

    return Governed


#: The name of the agent an agent tool's (``Agent.as_tool``) invoke runs, in its code.
RUNS_AGENT: Final = "self"
#: The names of the run config and the ``RunHooks`` its child run is given, in its code.
CHILD_CONFIG: Final = "run_config"
CHILD_HOOKS: Final = "hooks"


def _running(tool: Any, agent: Any, nested: Callable[[Any, Any], Any]) -> Any | None:
    """An agent tool (``Agent.as_tool``) as a copy that runs ``agent`` — the governed copy of
    its own — with everything else as it was given (its input, its run options, its output
    extractor): ``None`` when its invoke is not of the shape this SDK release builds (its own
    agent then runs as it is, said once). Its child run is given the config ``nested`` makes
    of ``agent`` and the tool's own ``run_config`` (:func:`_child_config`) in place of the
    calling run's, and the harness's ``RunHooks`` around its own ``hooks`` —
    every model call of the child watched for a stall (its usage the agent tool's helper
    counts: ``tools.convert.openai_agents._helped``)."""
    from agents import FunctionTool

    from trellis.harness.hooks.openai_agents import ChildModelHooks
    from trellis.harness.tools.convert.openai_agents import AGENT_OF_TOOL, AGENT_TOOL

    invoker = tool.on_invoke_tool
    impl = getattr(invoker, "_invoke_tool_impl", None)
    code = getattr(impl, "__code__", None)
    closes = {RUNS_AGENT, CHILD_CONFIG, CHILD_HOOKS}
    if not isinstance(tool, FunctionTool) or code is None or not closes <= set(code.co_freevars):
        return None
    assert impl is not None
    cells = dict(zip(code.co_freevars, impl.__closure__ or (), strict=True))
    given = {
        RUNS_AGENT: agent,
        CHILD_CONFIG: nested(agent, cells[CHILD_CONFIG].cell_contents),
        CHILD_HOOKS: ChildModelHooks(cells[CHILD_HOOKS].cell_contents),
    }
    closure = tuple(
        types.CellType(given[name]) if name in given else cell for name, cell in cells.items()
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


#: The field of the SDK's ``Handoff`` that holds the agent a ``handoff(agent, ...)`` hands off
#: to (a weak reference; the SDK's own run state finds the agent by it). The SDK has no public
#: way to read it, so it is checked where an agent is wrapped (:func:`handed_to`).
HANDOFF_AGENT: Final = "_agent_ref"


def handed_to(handoff: Any) -> Any | None:
    """The agent a ``Handoff`` object hands off to, as the SDK keeps it (:data:`HANDOFF_AGENT`);
    ``None`` for one built by hand, which names none before it runs. An SDK whose ``Handoff``
    no longer has that field is refused: its agents would run ungoverned."""
    from agents import Handoff

    if HANDOFF_AGENT not in {f.name for f in dataclasses.fields(Handoff)}:
        raise ConfigurationError(
            "this OpenAI Agents SDK's Handoff no longer says the agent it hands off to "
            f"({HANDOFF_AGENT}), so that agent's tools cannot be governed: hand off to the "
            "Agent itself, or use a tested SDK release (docs/versioning.md)"
        )
    ref = getattr(handoff, HANDOFF_AGENT, None)
    return ref() if callable(ref) else None


def _handed(handoff: Any, copied: Callable[[Any], Any], agent_id: str) -> Any:
    """A ``Handoff`` object as the run's copy: everything it gives the SDK kept, the agent it
    returns when invoked replaced by that agent's governed copy (``copied``) — and the agent it
    names (:func:`handed_to`) the copy too, which the SDK's run state finds it by on a resume.
    One built by hand, which names no agent before it runs, is governed from the moment it
    hands off; its agent's tools are not of the run's tool set before then (said once)."""
    from agents import Agent

    target = handed_to(handoff)
    governed_target = copied(target) if isinstance(target, Agent) else None
    if governed_target is None:
        _hidden_handoff(agent_id, str(getattr(handoff, "agent_name", "")))
    original = handoff.on_invoke_handoff

    async def invoke(context: Any, input_json: str | None = None) -> Any:
        returned = await original(context, input_json)
        # the copy is held here (the SDK's handle on it is weak), the same one ``copied`` gives
        return governed_target if returned is target else copied(returned)

    made = dataclasses.replace(handoff, on_invoke_handoff=invoke)
    if governed_target is not None:
        setattr(made, HANDOFF_AGENT, weakref.ref(governed_target))
    return made


def reachable(target: Any) -> list[Any]:
    """``target`` and every agent it can reach: those it hands off to (an ``Agent``, or the
    agent a ``Handoff`` object names: :func:`handed_to`) and those its agent tools run,
    recursively, each once."""
    from agents import Agent

    from trellis.harness.tools.convert.openai_agents import AGENT_OF_TOOL

    found: dict[int, Any] = {}
    pending = [target]
    while pending:
        agent = pending.pop()
        if id(agent) in found:
            continue
        found[id(agent)] = agent
        for handoff in agent.handoffs:
            to = handoff if isinstance(handoff, Agent) else handed_to(handoff)
            if isinstance(to, Agent):
                pending.append(to)
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
        "a Handoff object built by hand names no agent until it is invoked: that agent's own "
        "tools are governed once it hands off, but are not of the run's tool set (tool hints, "
        "catalog) before (use handoff(agent, ...), or hand off to the Agent itself)",
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


def _options(runtime: Runtime, target: Any, native_input: Any) -> dict[str, Any]:
    """``Runner.run``'s options: the run's framework options — a ``run_config`` given as a dict
    made the SDK's ``RunConfig`` as ``Runner.run`` makes it (:func:`_coerced`), a ``session``
    the agent was wrapped with made the conversation's own (:func:`per_conversation`) — with
    the memory service's thread as the session (``bridges.session``) when the run keeps no
    thread of its own and memory is on, the model names routed through Bifrost when it serves
    the harness (:func:`_routed`), and the model of a run resumed from its input told which
    call the person answered about (:class:`_Again`)."""
    options = dict(runtime.framework_options)
    given = options.get("run_config")
    if given is not None:
        options["run_config"] = _coerced(given)
    session = options.get("session")
    if session is not None and session is runtime.agent.framework_options.get("session"):
        options["session"] = per_conversation(session, runtime.conversation)
    gateway = runtime.agent.harness.model_gateway
    kept = _thread_store(options, keeps_thread(gateway, target, options))
    if runtime.run_memory is not None and kept is MEMORY_THREAD:
        options["session"] = MemorySession(runtime)
    told = _told_unknown_tools(options, given)
    question = _question(native_input, runtime.pushed_context)
    asked = _dated(runtime, _asked_with(runtime, told, question))
    again = native_input.told if isinstance(native_input, _Again) else None
    return _routed(runtime, target, _steering(runtime, asked, again))


def _coerced(config: Any) -> Any:
    """A ``run_config`` as ``Runner.run`` takes it: a ``RunConfig``, or a dict of its fields,
    made one the SDK's way (its own ``TypeError`` for a field it does not have)."""
    from agents.run_config import _coerce_run_config

    return _coerce_run_config(config)


@dataclass(frozen=True, slots=True)
class TurnLimit:
    """A run the SDK stopped at its turn limit (``MaxTurnsExceeded``), as its outcome: what it
    did up to there (``new_items``, the SDK's ``run_data``) and no final output of the SDK's —
    the run ends ``PARTIAL`` (:func:`_at_turn_limit`), its answer the last thing its model
    said, if it said anything."""

    final_output: Any
    new_items: list[Any]
    interruptions: tuple[Any, ...] = ()


def _at_turn_limit(runtime: Runtime, error: Any, options: Mapping[str, Any]) -> TurnLimit:
    """The SDK's outcome of a run that reached its ``max_turns`` (``error``: its
    ``MaxTurnsExceeded``; an ``error_handlers`` entry of the run's own answers it before this),
    as the harness ends it: ``PARTIAL`` — the turn limit is the run's budget of turns, said as
    the budget's warning — with no model call of the harness's, its answer the last message
    the model wrote (``None`` when it wrote none)."""
    from agents.run_config import DEFAULT_MAX_TURNS

    limit = options.get("max_turns", DEFAULT_MAX_TURNS)
    items = list(error.run_data.new_items)  # the SDK's, set wherever it raises the error
    said = [s.content for s in said_in(items) if s.role is Speaker.ASSISTANT and s.content]
    runtime.budget_spent = f"its turn limit (max_turns={limit}) was reached"
    log.warning("run %s stopped: the turn limit (%s) was reached", runtime.run_id, limit)
    return TurnLimit(final_output=said[-1] if said else None, new_items=items)


def _told_unknown_tools(options: dict[str, Any], given: Any) -> dict[str, Any]:
    """The options with a call to a tool the model made up answered as an error the model
    reads, so the run goes on — as on every other adapter — and not a ``ModelBehaviorError``
    that ends it: the SDK's own ``RunConfig.tool_not_found_behavior="return_error_to_model"``
    (its message: ``Tool '<name>' not found.``) — unless the run says its own (``given``: the
    ``run_config`` as it was given; a dict naming the field, or a ``RunConfig`` whose value is
    not the field's default — the one value a ``RunConfig`` object cannot tell from unset)."""
    from agents import RunConfig

    config = options.get("run_config") or RunConfig()
    default = next(f.default for f in dataclasses.fields(RunConfig) if f.name == UNKNOWN_TOOLS)
    if (isinstance(given, Mapping) and UNKNOWN_TOOLS in given) or getattr(
        config, UNKNOWN_TOOLS
    ) != default:
        return options
    told = dataclasses.replace(config, tool_not_found_behavior="return_error_to_model")
    return {**options, "run_config": told}


#: The ``RunConfig`` field that says what a call to a tool the model made up does.
UNKNOWN_TOOLS: Final = "tool_not_found_behavior"


def _child_config(runtime: Runtime, agent: Any, own: Any) -> Any:
    """The run config of an agent tool's child run (``Agent.as_tool``): the tool's own
    ``run_config`` — else the calling run's as it was given, as the SDK passes it on — with
    what the harness gives every model call of the run that is not the calling conversation's:
    the model route (:func:`_routed`), the day (:func:`_dated`) and a made-up tool's error
    (:func:`_told_unknown_tools`, unless it says its own). The messages steered into the
    conversation, the memory context the run was pushed and why a computer action did not run
    are the calling run's, and are not given to the child (its helper is given its own
    memory context, ``bridges.helpers``)."""
    from agents import RunConfig

    theirs = runtime.framework_options.get("run_config") if own is None else own
    options: dict[str, Any] = {"run_config": RunConfig() if theirs is None else _coerced(theirs)}
    told = _told_unknown_tools(options, theirs)
    return _routed(runtime, agent, _dated(runtime, told))["run_config"]


def _filtered(options: dict[str, Any], change: Callable[[Any], Any]) -> dict[str, Any]:
    """The options with ``change`` made to the input of each model call (its
    ``ModelInputData``: the items and the instructions), after the run's own filter
    (``RunConfig.call_model_input_filter``). Only the model's input changes; the input a
    session keeps does not."""
    from agents import RunConfig

    config = options.get("run_config") or RunConfig()
    theirs = config.call_model_input_filter

    async def changed(data: Any) -> Any:
        given = data.model_data
        if theirs is not None:
            given = theirs(data)
            if inspect.isawaitable(given):
                given = await given
        return change(given)

    return {**options, "run_config": dataclasses.replace(config, call_model_input_filter=changed)}


def _question(native_input: Any, context: str | None) -> Any:
    """What this run asked — the content of the last user message of the input the harness
    made (``prepare_input``: after the memory context's system message), as a resumed run's
    saved state holds it too — or ``None`` when it cannot be read."""
    if isinstance(native_input, _Again):
        native_input = native_input.input
    elif isinstance(native_input, _Continue):
        native_input = native_input.state.get("original_input")
    if not isinstance(native_input, list):
        return None
    items = [i for i in native_input if isinstance(i, dict)]
    start = next(
        (
            n
            for n, i in enumerate(items)
            if i.get("role") == "system" and i.get("content") == context
        ),
        0,
    )
    asked = [i for i in items[start:] if i.get("role") == "user"]
    return asked[-1].get("content") if asked else None


def _asked_with(runtime: Runtime, options: dict[str, Any], question: Any = None) -> dict[str, Any]:
    """The options with the pushed memory context sent as the first part of the user's turn,
    marked off (``history.ASKED_WITH``), as ``ReAct``'s and ``create_agent``'s
    (``middleware.with_question``), and not as a system message of its own (:func:`_filtered`).
    Haiku, a question 14 products match: given the context as its own system message it
    answered for one of them 9 times in 10, given it in the user's turn it asked which one was
    meant. The turn is this run's question (``question``: its content, :func:`_question`) —
    the first user message after the context that says it, wherever the run's
    ``session_input_callback`` put the earlier turns —, else the last user message after the
    context."""
    from agents.run_config import ModelInputData

    context = runtime.pushed_context
    if not context:
        return options

    def asked(given: Any) -> Any:
        items: list[Any] = list(given.input)
        at = next(
            (
                n
                for n, i in enumerate(items)
                if isinstance(i, dict) and i.get("role") == "system" and i.get("content") == context
            ),
            None,
        )
        after = range(at + 1, len(items)) if at is not None else range(0)
        users = [n for n in after if isinstance(items[n], dict) and items[n].get("role") == "user"]
        said_it = [n for n in users if question is not None and items[n].get("content") == question]
        found = said_it[0] if said_it else users[-1] if users else None
        if at is None or found is None:
            return given
        said = items[found].get("content")
        parts = said if isinstance(said, list) else [{"type": "input_text", "text": str(said)}]
        marked = {"type": "input_text", "text": ASKED_WITH.format(context=context)}
        items[found] = {**items[found], "content": [marked, *parts]}
        del items[at]
        return ModelInputData(input=items, instructions=given.instructions)

    return _filtered(options, asked)


def _dated(runtime: Runtime, options: dict[str, Any]) -> dict[str, Any]:
    """The options with the day it is said after each agent's instructions
    (``snapshots.today``: the run's start, the same line for every model call of the run, the
    instructions before it the same from day to day, as a prompt cache keeps them)."""
    from agents.run_config import ModelInputData

    day = snapshots.today(runtime)

    def dated(given: Any) -> Any:
        told = "\n\n".join(part for part in (given.instructions, day) if part)
        return ModelInputData(input=given.input, instructions=told)

    return _filtered(options, dated)


def _steering(runtime: Runtime, options: dict[str, Any], again: str | None) -> dict[str, Any]:
    """The options with the messages steered into the run (``Runtime.steered``) joining the
    input of each model call where they were taken, as the user's (:func:`_filtered`): the SDK
    keeps a filter's input for that call only, so each call is given them again, in place.
    ``again``: what a run resumed from its input is told first, the same way (no session keeps
    it); and so is why a computer action did not run (``openai_local.unsaid``)."""
    from agents.run_config import ModelInputData

    from trellis.harness.tools.convert.openai_local import unsaid

    kept: list[tuple[int, dict[str, Any]]] = []
    said = [again] if again else []

    def steered(given: Any) -> Any:
        items: list[Any] = list(given.input)
        said.extend(s.text for s in runtime.steered())
        said.extend(unsaid(runtime))  # computer actions not run since: why (the bridge's words)
        kept.extend((len(items), {"role": "user", "content": text}) for text in said)
        said.clear()
        for offset, (at, item) in enumerate(kept):
            items.insert(at + offset, item)
        return ModelInputData(input=items, instructions=given.instructions)

    return _filtered(options, steered)


def _routed(runtime: Runtime, target: Any, options: dict[str, Any]) -> dict[str, Any]:
    """With Bifrost serving the harness (a route with ``Capability.ATTRIBUTION``), the model
    names the run leaves to the SDK's default provider go through it
    (``RunConfig.model_provider``: :func:`routed_provider` over the run's ``MultiProvider``,
    the name mapped to Bifrost's ``provider/model``) — every other choice of the run's kept: a
    model provider of its own (code over the environment), the ``MultiProvider``'s own
    ``provider_map`` and its ``litellm/`` and ``any-llm/`` prefixes. A model object (an
    agent's, or ``RunConfig.model``) calls its provider directly, which is said once. What an
    agent uses that its route cannot carry is refused before the run's first model call
    (:func:`unserved`)."""
    from agents import Agent, RunConfig
    from agents.models.multi_provider import MultiProvider

    gateway = runtime.agent.harness.model_gateway
    unserved(gateway, target, options)
    if not gateway.has(Capability.ATTRIBUTION):
        return options
    config = options.get("run_config") or RunConfig()
    if config.model is not None and not isinstance(config.model, str):
        _unrouted(runtime.agent_id, "RunConfig.model", type(config.model).__name__)
    if not isinstance(config.model_provider, MultiProvider):
        return options  # the run's own provider
    for agent in reachable(target):
        model = agent.model if isinstance(agent, Agent) else None
        if model is not None and not isinstance(model, str):
            _unrouted(runtime.agent_id, agent.name, type(model).__name__)
    provider = routed_provider(cast("Bifrost", gateway), config.model_provider)
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


def routed_provider(gateway: Bifrost, theirs: Any) -> Any:
    """The run's model provider on Bifrost: ``theirs`` (the run's ``MultiProvider``) for every
    name it gives a provider of its own, Bifrost (:func:`bifrost_provider`) for the names it
    leaves to its OpenAI provider (:meth:`_Routed.routes`)."""
    return _routed_class()(theirs, bifrost_provider(gateway))


@functools.cache
def _routed_class() -> type:
    from agents.models.interface import ModelProvider

    class Routed(ModelProvider):  # type: ignore[misc]
        def __init__(self, theirs: Any, bifrost: Any) -> None:
            self.theirs = theirs
            self.bifrost = bifrost

        def get_model(self, model_name: str | None) -> Any:
            if routes(self.theirs, model_name):
                return self.bifrost.get_model(model_name)
            return self.theirs.get_model(model_name)

    return Routed


def routes(theirs: Any, name: str | None) -> bool:
    """Whether the run's ``MultiProvider`` (``theirs``) leaves the model ``name`` to its
    default provider, which Bifrost stands in for: a bare name, an ``openai/`` one, or a name
    whose prefix it gives no provider of its own (a name in Bifrost's ``provider/model``
    form) — and not one its ``provider_map``, ``litellm/`` or ``any-llm/`` gives a provider
    (the SDK's own resolution, ``MultiProvider._resolve_prefixed_model``)."""
    from agents.exceptions import UserError

    prefix, stripped = theirs._get_prefix_and_model_name(name)
    if prefix is None:
        return True
    try:
        provider, _ = theirs._resolve_prefixed_model(
            original_model_name=name, prefix=prefix, stripped_model_name=stripped
        )
    except UserError:  # a prefix it cannot place: the gateway's provider
        return True
    return bool(provider is theirs.openai_provider)


def bifrost_provider(gateway: Bifrost) -> Any:
    """The OpenAI Agents SDK's model provider for Bifrost: each model name (the SDK's default
    for an agent naming none) mapped to Bifrost's form, asked on the SDK's default API — the
    Responses API (``/v1/responses``), or chat completions for a model the gateway answers it
    does not serve there (``_learning_class``), unless the application set another
    (``set_default_openai_api``) — with the virtual key,
    the deny-all MCP scope and the run's dimensions and ``traceparent`` (read per request), no
    retries of the client's own (the gateway retries). One per gateway and event loop (a
    client's pooled connections belong to the loop that opened them)."""
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
    from agents.models._openai_shared import get_use_responses_by_default
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
            self.inner = OpenAIProvider(openai_client=client)  # the SDK's default API
            self.chat = OpenAIProvider(openai_client=client, use_responses=False)

        def get_model(self, model_name: str | None) -> Any:
            name = self.gateway.routed(model_name or get_default_model())
            model = self.inner.get_model(name)
            if not get_use_responses_by_default():  # the application's own choice of API
                return model
            return _learning_class()(self.gateway, name, model, self.chat.get_model(name))

    return BifrostProvider


#: What a gateway answers a request on a route it does not serve (no such route, no such
#: method on it, not implemented): the model is then asked on chat completions.
NOT_SERVED: Final = frozenset({404, 405, 501})


@functools.cache
def _learning_class() -> type:
    import openai
    from agents.models.interface import Model

    class Learning(Model):  # type: ignore[misc]
        """A model name on Bifrost, asked on the Responses API until the gateway answers that
        it does not serve the route for it (:data:`NOT_SERVED`, on the model's first call
        there: nothing was asked, no tool has run for it) — then, and from then on for this
        gateway and model (``Bifrost.served``), on chat completions, as before the Responses
        route. What a chat completions request cannot carry is refused first, each feature
        named (:func:`_refused_on_chat`). One fallback per call at most; a model that has
        answered on Responses never falls back (its refusal is the run's error)."""

        def __init__(self, gateway: Bifrost, name: str, responses: Any, chat: Any) -> None:
            self.gateway, self.name, self.responses, self.chat = gateway, name, responses, chat

        def _falls_back(self, error: Exception, args: Any, kwargs: Any) -> bool:
            unserved = isinstance(error, openai.APIStatusError) and error.status_code in NOT_SERVED
            if not unserved or not self.gateway.falls_back(self.name):
                return False
            asked = inspect.signature(Model.get_response).bind(self, *args, **kwargs)
            _refused_on_chat(self.name, asked.arguments)
            return True

        async def get_response(self, *args: Any, **kwargs: Any) -> Any:
            if not self.gateway.keeps_responses(self.name):
                return await self.chat.get_response(*args, **kwargs)
            try:
                response = await self.responses.get_response(*args, **kwargs)
            except Exception as error:
                if not self._falls_back(error, args, kwargs):
                    raise
            else:
                self.gateway.served(self.name, responses=True)
                return response
            response = await self.chat.get_response(*args, **kwargs)
            self.gateway.served(self.name, responses=False)  # it answered there: kept
            return response

        async def stream_response(self, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
            if self.gateway.keeps_responses(self.name):
                try:
                    async for event in self.responses.stream_response(*args, **kwargs):
                        self.gateway.served(self.name, responses=True)
                        yield event
                    return
                except Exception as error:  # refused before its first event: nothing was said
                    if not self._falls_back(error, args, kwargs):
                        raise
            async for event in self.chat.stream_response(*args, **kwargs):
                self.gateway.served(self.name, responses=False)
                yield event

    return Learning


def _whole(gateway: Any, name: str | None) -> bool:
    """Whether a model name the harness routes through Bifrost is asked with all a Responses
    request carries kept — hosted tools, a stored prompt, server-side threads, compaction: the
    SDK's default API is the Responses API (else the application chose chat completions, whose
    limits are the SDK's own) and the gateway has not answered that it serves the model on
    chat completions only (``Bifrost.keeps_responses``: learned from its first call)."""
    from agents.models import _openai_shared, get_default_model

    if not _openai_shared.get_use_responses_by_default():
        return True
    return cast("Bifrost", gateway).keeps_responses(name or get_default_model())


def _route_of(gateway: Any, options: Mapping[str, Any]) -> Any | None:
    """The run's ``MultiProvider`` when the harness routes its default names through Bifrost
    (:func:`_routed`); ``None`` when it routes none (another route, a provider of the run's)."""
    from agents import RunConfig
    from agents.models.multi_provider import MultiProvider

    if gateway is None or not gateway.has(Capability.ATTRIBUTION):
        return None
    given = options.get("run_config")
    config = RunConfig() if given is None else _coerced(given)
    provider = config.model_provider
    return provider if isinstance(provider, MultiProvider) else None


def _model_of(agent: Any, options: Mapping[str, Any]) -> Any:
    """The model ``agent`` is called with in a run with these options: ``RunConfig.model``,
    else its own."""
    given = options.get("run_config")
    model = _coerced(given).model if given is not None else None
    return model if model is not None else getattr(agent, "model", None)


def keeps_thread(gateway: Any, target: Any, options: Mapping[str, Any]) -> bool:
    """Whether the model the run starts with keeps a conversation OpenAI's servers hold
    (:data:`SERVER_THREAD`): not the SDK's chat completions model, which ignores the ids (and
    says so) — a model object of that class, or a name the SDK's OpenAI provider answers on
    chat completions (the application's default API, or the provider's own) — nor a name the
    harness routes through Bifrost to a provider it converts to chat completions
    (:func:`_whole`). Any other model or provider is taken at its word, as a thread store the
    harness cannot see into is (``history``)."""
    from agents import RunConfig
    from agents.models import _openai_shared
    from agents.models.multi_provider import MultiProvider
    from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel

    model = _model_of(target, options)
    if model is not None and not isinstance(model, str):
        return not isinstance(model, OpenAIChatCompletionsModel)
    given = options.get("run_config")
    provider = (RunConfig() if given is None else _coerced(given)).model_provider
    if not isinstance(provider, MultiProvider) or not routes(provider, model):
        return True
    if _route_of(gateway, options) is None:  # the SDK's OpenAI provider, on the API it was given
        return bool(provider.openai_provider._use_responses)
    return _openai_shared.get_use_responses_by_default() and _whole(gateway, model)


def _converted(gateway: Any, route: Any, agent: Any, options: Mapping[str, Any]) -> bool:
    """Whether ``agent``'s model is a name the harness routes through Bifrost (``route``: the
    run's ``MultiProvider``) which the gateway serves on chat completions only (:func:`_whole`)."""
    model = _model_of(agent, options)
    named = model is None or isinstance(model, str)
    return named and routes(route, model) and not _whole(gateway, model)


#: A ``ModelSettings`` field a Responses request carries and a chat completions one does not.
RESPONSES_SETTINGS: Final = ("truncation", "response_include", "context_management")


def unserved(gateway: Any, target: Any, options: Mapping[str, Any]) -> None:
    """Refuse, before any model call, what a run's agents use that their model route drops: on
    Bifrost, a provider it calls by converting the request to chat completions (:func:`_whole`)
    keeps function tools only, and no stored prompt, Responses settings, sandbox request
    parameters or server-side thread — each named, with what to do, rather than dropped
    without a word or failing mid-run."""
    route = _route_of(gateway, options)
    if route is None:
        return
    found: list[str] = []
    for agent in reachable(target):
        if _converted(gateway, route, agent, options):
            found.extend(f"{agent.name!r}: {said}" for said in _chat_only(agent, options))
    server = [key for key in SERVER_THREAD if options.get(key)]
    if server and _converted(gateway, route, target, options):
        found.append(f"{target.name!r}: {', '.join(server)} (a server thread)")
    if found:
        raise ConfigurationError(DROPPED.format(found="; ".join(found)))


#: What a refusal says of what a chat completions request would drop (:func:`unserved`).
DROPPED: Final = (
    "the gateway serves these models on chat completions only (it answered that it does not "
    "serve them on /v1/responses), which would drop: {found}. Use a model the gateway serves "
    "on the Responses API, a model object of your own, or leave these out"
)


def _chat_only(agent: Any, options: Mapping[str, Any]) -> list[str]:
    """What ``agent`` uses that a chat completions request cannot carry (:func:`_dropped`):
    its tools, its stored prompt and its model settings as the run resolves them."""
    from agents import RunConfig

    given = options.get("run_config")
    config = RunConfig() if given is None else _coerced(given)
    settings = agent.model_settings.resolve(config.model_settings)
    return _dropped(agent.tools, agent.prompt, settings)


def _dropped(tools: Sequence[Any], prompt: Any, settings: Any) -> list[str]:
    """What a chat completions request cannot carry of a model call: each tool the SDK's own
    chat completions converter refuses (a hosted tool, a shell, a patch editor, a computer, a
    custom tool, a deferred or namespaced function tool: its words), a stored prompt and
    Responses settings (:data:`RESPONSES_SETTINGS`, ``reasoning.mode``/``context``)."""
    from agents.exceptions import UserError
    from agents.models.chatcmpl_converter import Converter

    found = []
    for tool in tools:
        try:
            Converter.tool_to_openai(tool)
        except UserError as exc:
            found.append(f"tool {getattr(tool, 'name', type(tool).__name__)!r} ({exc})")
    if prompt is not None:
        found.append("a stored prompt (Agent.prompt)")
    found += [f"ModelSettings.{name}" for name in RESPONSES_SETTINGS if getattr(settings, name)]
    reasoning = settings.reasoning
    found += [
        f"ModelSettings.reasoning.{name}"
        for name in ("mode", "context")
        if getattr(reasoning, name, None) is not None
    ]
    return found


def _refused_on_chat(model: str, asked: Mapping[str, Any]) -> None:
    """Refuse a model call about to fall back to chat completions (``_learning_class``) when
    that request cannot carry it: what the call's tools, prompt and settings need
    (:func:`_dropped`) and a server-side thread (``asked``: the call's arguments by name, as
    the SDK's ``Model.get_response`` takes them), each named."""
    found = _dropped(asked["tools"], asked["prompt"], asked["model_settings"])
    server = [key for key in SERVER_THREAD if asked.get(key)]
    if server:
        found.append(f"{', '.join(server)} (a server thread)")
    if found:
        raise ConfigurationError(DROPPED.format(found=f"{model!r}: " + "; ".join(found)))


def _thread_store(options: Mapping[str, Any], keeps: bool) -> Any:
    """The thread store of a run with these options (``OpenAIAgentsAdapter.thread_store``):
    ``keeps``, whether the run's model keeps a server-side thread (:func:`keeps_thread`) —
    when it does not, the ids are not a thread, and none is kept (``None``)."""
    if options.get("session") is not None:
        return options["session"]
    server = next((key for key in SERVER_THREAD if options.get(key)), None)
    if server is None:
        return MEMORY_THREAD
    return ServerThread(server) if keeps else None


def per_conversation(session: Any, conversation: str) -> Any:
    """A session the agent was wrapped with (``h.wrap(framework_options={"session": ...})``,
    one object for every run) as the conversation's own: a copy of it — its class, its storage
    — whose ``session_id`` is its own and the conversation's key (``Runtime.conversation``:
    the tenant's, the user's and the thread's; a run with no thread, its own), and a session
    it wraps (an encrypted session's) made the conversation's the same way, so two users, or
    two threads, never read each other's turns (:func:`_scoped_id` refuses a session that
    cannot be made so, where it is wrapped)."""
    from agents.memory.session import Session

    # its attributes on a new object of its class: ``copy.copy`` asks a blank object for its
    # ``__setstate__``, which a session that hands unknown attributes on to the one it wraps
    # (the SDK's ``EncryptedSession``) answers by recursing
    scoped = object.__new__(type(session))
    scoped.__dict__.update(vars(session))
    for name, value in vars(session).items():
        if isinstance(value, Session) and value is not session:
            setattr(scoped, name, per_conversation(value, conversation))
    scoped.session_id = _scoped_id(session) + f":{conversation}"
    return scoped


def _scoped_id(session: Any) -> str:
    """The ``session_id`` a session given on ``h.wrap`` keeps each conversation under, after
    its own (:func:`per_conversation`) — refused (``ConfigurationError``) when a copy with
    another id would still share its storage: its id is not a plain attribute of it (a
    property, a server's conversation), or it keeps a value made from its id when it was made
    (a key prefix), which a copy keeps too. A session it wraps is checked the same way."""
    from agents.memory.session import Session

    held = getattr(session, "__dict__", {})
    own = held.get("session_id")
    if not isinstance(own, str):
        raise ConfigurationError(_SHARED.format(kind=type(session).__name__, why=_NOT_ITS_OWN))
    for name, value in held.items():
        if isinstance(value, Session) and value is not session:
            _scoped_id(value)
        elif name != "session_id" and own and isinstance(value, str) and own in value:
            why = _MADE_FROM.format(name=name)
            raise ConfigurationError(_SHARED.format(kind=type(session).__name__, why=why))
    return own


#: Why a session given on ``h.wrap`` cannot be kept per conversation (:func:`_scoped_id`).
_SHARED: Final = (
    "framework_options 'session': a session given on h.wrap is every run's, so the harness "
    "keeps each conversation (tenant, user, thread) under its own session_id in it, and a "
    "{kind} cannot be: {why}. Give it per run instead (agent.run(..., framework_options="
    "{{'session': ...}}), one per conversation), or leave it out (memory keeps the thread)"
)
_NOT_ITS_OWN: Final = "its session_id is not a plain attribute a copy can change"
_MADE_FROM: Final = "its {name} was made from its session_id, and a copy would keep it"


def _one_sandbox(config: Any) -> None:
    """Refuse a sandbox that would be every user's: a ``SandboxRunConfig`` given on ``h.wrap``
    with a live ``session`` (one sandbox, its files and processes, for every run) or a
    ``session_state`` (every run resumed into the same one)."""
    sandbox = getattr(config, "sandbox", None)
    pinned = [
        name for name in ("session", "session_state") if getattr(sandbox, name, None) is not None
    ]
    if pinned:
        raise ConfigurationError(
            f"framework_options 'run_config': a SandboxRunConfig({pinned[0]}=...) on h.wrap "
            "is one sandbox — its files and its processes — for every user and thread. Give "
            "the sandbox client on h.wrap (SandboxRunConfig(client=...): the SDK makes a "
            "sandbox for each run), or the session you keep for a conversation per run "
            "(agent.run(..., framework_options={'run_config': RunConfig(sandbox="
            "SandboxRunConfig(session=...))}))"
        )


def _hooks(theirs: Any = None) -> Any:
    """The harness's ``RunHooks``, always: every model call watched for a stall, its hosted
    tool calls recorded, its usage counted toward the run's budget (``budget.py``) — and the
    run's own ``RunHooks`` (``theirs``: the ``hooks=`` framework option) called with everything
    the SDK reports."""
    from trellis.harness.hooks.openai_agents import ModelHooks

    return ModelHooks(theirs)


async def _deltas(result: Any) -> AsyncIterator[str]:
    async for event in result.stream_events():
        data = getattr(event, "data", None)
        if event.type == "raw_response_event" and getattr(data, "type", None) == TEXT_DELTA:
            yield data.delta  # type: ignore[union-attr]


async def _continued(result: Any, native_input: Any, runtime: Runtime) -> Any | None:
    """The SDK's run state to continue with when its run paused on a call a person already
    decided: the call a reviewer edited, made again with exactly the edited arguments
    (:func:`_edited_call`), or the next call governance's gate held in the same model reply as
    the one just answered (:func:`_decided`); else ``None``."""
    state = _edited_call(result, native_input)
    return state if state is not None else await _decided(result, runtime)


async def _decided(result: Any, runtime: Runtime) -> Any | None:
    """The run state with the first held call governance now lets run approved
    (``RunState.approve``), when the run paused again on calls its gate held in an earlier
    reply (``tools.convert.openai_agents.asked_again``) — the SDK keeps their pending
    approvals on a resume and does not ask the gate again: each is decided now, in order,
    after the answer to the one before (an approval for the conversation lets the rest of its
    tool's calls run). A call the SDK's own ``needs_approval`` holds runs unasked, its tool's
    later calls in the run too (``always_approve``), when the person allowed that tool beyond
    one call (``bridge.unasked``: for the conversation, its schedule or always, never an
    ``irreversible`` tool — the one rule governance's own questions follow). ``None`` when
    none waits, or the first asks: its pause is governance's, as the first call's was."""
    from trellis.harness.tools.convert.openai_agents import asked_again
    from trellis.harness.tools.convert.openai_local import spec_of

    if runtime.pending is not None:  # a call of this attempt's reply asked: the run's pause
        return None
    for item in result.interruptions:
        asks = await asked_again(item, result.context_wrapper)
        if asks is None:  # the SDK's own gate: unasked when its tool was allowed beyond a call
            spec = spec_of(runtime, item.name)
            if not await bridge.unasked(spec, _asked_about(item)):
                continue
            state = result.to_state()
            state.approve(item, always_approve=True)
            return state
        if asks:
            return None
        state = result.to_state()
        state.approve(item)
        return state
    return None


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


def _asked_about(item: Any) -> dict[str, Any]:
    """What a person is shown of the call the SDK waits on (its ``ToolApprovalItem``): its
    arguments — the SDK pauses on a call before reading them, so ones that are not a JSON
    object reach the person as the model wrote them — and, for a call that has none of that
    name (the SDK's tools that run here: a shell's commands, a patch's operation), what the
    call does as the model made it (``openai_local.call_args``: what governance decides on
    when it runs, so the person's answer is the one answer the call needs)."""
    if item.arguments is None:
        from trellis.harness.tools.convert.openai_local import call_args

        return call_args(item.raw_item)
    args = _arguments(item)
    return args if isinstance(args, dict) else {"arguments": item.arguments}


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
        spec = ToolSpec(name=name, source=source, server=server)
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
