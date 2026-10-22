"""OpenAI Agents SDK ``Agent``\\ s.

* input: a message list, the memory context first as a ``system`` message;
* run: ``Runner.run``/``Runner.run_streamed`` on a clone of the agent carrying the harness
  tools next to its own (the caller's agent is never changed), with the run's hooks around its
  model calls as the SDK's ``RunHooks`` (``hooks.openai_agents``);
* pause: ``trellis.current().ask`` stops the run and a resume re-runs it against the journal;
  the SDK's own ``needs_approval`` tools pause with the SDK's ``RunState``, and a resume
  approves or rejects on that state and continues it (``RunState.approve``/``reject``). The SDK
  takes no edited arguments and no answer in place of a tool's result, so: a reject carries the
  reviewer's reason (``rejection_message``, the text the model reads); an answer is a reject
  whose message is the answer; an edit is a reject telling the model to call the tool again
  with the edited arguments — and that call, when the model makes it with exactly those
  arguments, is approved in the same attempt instead of asking again;
* session: a run given no ``session`` of its own, with memory on, keeps the conversation in the
  memory service's thread (``bridges.session.MemorySession``): the SDK's own ``Session``
  interface, the memory service behind it;
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
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, Final

from trellis.contracts import ConfigurationError, InterruptDecision, InterruptResolution
from trellis.harness.adapters.base import (
    Extracted,
    Invocation,
    Narrowing,
    NativePause,
    Output,
    ToolFormat,
)
from trellis.harness.bridges import said
from trellis.harness.bridges.helpers import Helpers
from trellis.harness.bridges.history import Thread, owns, thread_for
from trellis.harness.bridges.session import MemorySession, bridgeable
from trellis.harness.journal import Pending
from trellis.harness.runtime import Runtime, reason_of

log = logging.getLogger("trellis.harness")

#: ``Runner.run``'s arguments the harness gives itself: the agent (a clone carrying the harness
#: tools) and its input (with the memory context, or the resumed state). Its ``hooks`` are the
#: harness's ``ModelHooks``, which tell the team's own ``RunHooks`` everything after them.
OWN_ARGUMENTS: Final = ("starting_agent", "input")
#: ``Runner.run``'s arguments that keep the conversation on OpenAI's servers: the framework's
#: thread, durable (``bridges.history``).
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

    def runs(self, target: Any, *, agent_id: str, memory: bool) -> Any:
        return target

    def keeps_conversation(self, target: Any, runtime: Runtime) -> bool:
        """Whether a session owns the conversation's history (``bridges.history``): the run's
        own when it is durable, or the memory service's thread (``bridges.session``)."""
        return owns(_thread(runtime))

    def prepare_input(self, target: Any, input: Any, context: str | None) -> Any:
        system = [{"role": "system", "content": context}] if context else []
        if isinstance(input, str):
            return [*system, {"role": "user", "content": input}] if system else input
        if isinstance(input, list):
            return [*system, *input]
        return [*system, {"role": "user", "content": json.dumps(input, default=str)}]

    async def invoke(self, target: Any, native_input: Any, run: Invocation) -> Any:
        from agents import Runner

        helpers = Helpers(run.runtime, self.name)
        agent, options = self._agent(target, run, helpers), _options(run, native_input)
        hooks = _hooks(options, helpers, agent)
        given = await self._input(agent, native_input)
        error: BaseException | None = None
        try:
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
        agent, options = self._agent(target, run, helpers), _options(run, native_input)
        hooks = _hooks(options, helpers, agent)
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
        from agents import ItemHelpers, MessageOutputItem

        transcript = [
            ("assistant", text)
            for item in output.new_items
            if isinstance(item, MessageOutputItem)
            and (text := ItemHelpers.text_message_output(item))
        ]
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

    def check_options(self, options: Mapping[str, Any]) -> None:
        from agents import Runner

        own = [key for key in options if key in OWN_ARGUMENTS]
        if own:
            raise ConfigurationError(
                f"framework_options {', '.join(map(repr, own))}: the harness gives Runner.run "
                "its agent and its input itself"
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
        """The agent run: a copy of ``target`` carrying the harness tools beside its own, and
        every agent it can hand off to a copy too (:func:`governed_agents`) — each one's own
        tools governed. The agents you wrapped are not changed."""
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


def _thread(runtime: Runtime) -> Thread:
    """The thread the run works on, by the history policy (``bridges.history``): its session,
    or the conversation OpenAI's servers keep (:data:`SERVER_THREAD`, durable)."""
    options = runtime.framework_options
    server = next((options[key] for key in SERVER_THREAD if options.get(key)), None)
    return thread_for(
        options.get("session") or server,
        bridged=bridgeable(runtime),
        agent_id=runtime.agent_id,
        memory=runtime.run_memory is not None,
    )


def governed_agents(target: Any, *, added: list[Any], agent_id: str, helpers: Helpers) -> Any:
    """``target`` and every agent reachable through its handoffs, each a copy whose own tools
    are governed (``tools.convert.openai_agents.governed``) — recursively, each agent copied
    once, so handoffs that lead back are kept — ``added`` (the run's harness tools) on
    ``target``'s. A ``Handoff`` object hides its agent until it is invoked: its tools cannot be
    reached, which is said once."""
    from agents import Agent

    from trellis.harness.tools.convert.openai_agents import governed

    copies: dict[int, Any] = {}

    def copied(agent: Any, extra: list[Any]) -> Any:
        found = copies.get(id(agent))
        if found is not None:
            return found
        own = [governed(t, helpers=helpers) for t in agent.tools]
        copy = copies[id(agent)] = agent.clone(tools=[*own, *extra])
        handoffs = []
        for handoff in agent.handoffs:
            if isinstance(handoff, Agent):
                handoffs.append(copied(handoff, []))
                continue
            said(
                f"handoff:{agent_id}:{getattr(handoff, 'agent_name', '?')}",
                "%s: the handoff to %s is a Handoff object, which hides its agent until it is "
                "invoked: that agent's own tools are not governed (hand off to the Agent itself)",
                agent_id,
                getattr(handoff, "agent_name", "?"),
                extra={"agent_id": agent_id},
            )
            handoffs.append(handoff)
        copy.handoffs = handoffs
        return copy

    return copied(target, added)


def _options(run: Invocation, native_input: Any) -> dict[str, Any]:
    """``Runner.run``'s options: the run's framework options, with the session the history
    policy gives the run — the memory service's thread (``Thread.BRIDGED``:
    ``bridges.session``), none (``Thread.RUN``: the run's own items, its saved state in the
    journal), or the run's own as given."""
    runtime = run.runtime
    options = dict(runtime.framework_options)
    thread = _thread(runtime)
    if thread is Thread.BRIDGED:
        fresh = not isinstance(native_input, _Continue)
        options["session"] = MemorySession(runtime, fresh=fresh)
        runtime.transcript_kept = True  # the session writes the thread
    elif thread is Thread.RUN:
        options.pop("session", None)
    return options


def _hooks(options: dict[str, Any], helpers: Helpers, agent: Any) -> Any:
    """The run's ``RunHooks``: the harness's ``ModelHooks`` — every reply's usage counted toward
    the run's budget, the run's model hooks, the agents handed off to as the run's helpers —
    then the team's own (``hooks`` of the run's options, taken out of them)."""
    from trellis.harness.hooks.openai_agents import ModelHooks

    return ModelHooks(chained=options.pop("hooks", None), helpers=helpers, starting=agent.name)


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


def _arguments(item: Any) -> Any:
    try:
        return json.loads(item.arguments or "{}")
    except ValueError:
        return None
