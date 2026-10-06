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
* framework options: keyword arguments of ``Runner.run``/``run_streamed`` (``max_turns``,
  ``run_config``, ``context``...), except the ones the harness gives itself
  (:data:`OWN_ARGUMENTS`), which are refused.
"""

from __future__ import annotations

import asyncio
import inspect
import json
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
from trellis.harness.journal import Pending
from trellis.harness.runtime import reason_of

#: ``Runner.run``'s arguments the harness gives itself: the agent (a clone carrying the harness
#: tools), its input (with the memory context, or the resumed state) and the run's hooks.
OWN_ARGUMENTS: Final = ("starting_agent", "input", "hooks")
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


class OpenAIAgentsAdapter:
    name: ClassVar[str] = "openai_agents"
    tool_format: ToolFormat = "openai_agents"
    fixed_tools: bool = False
    narrows: Narrowing = "turn"

    def keeps_conversation(self, target: Any) -> bool:
        return False

    def prepare_input(self, target: Any, input: Any, context: str | None) -> Any:
        system = [{"role": "system", "content": context}] if context else []
        if isinstance(input, str):
            return [*system, {"role": "user", "content": input}] if system else input
        if isinstance(input, list):
            return [*system, *input]
        return [*system, {"role": "user", "content": json.dumps(input, default=str)}]

    async def invoke(self, target: Any, native_input: Any, run: Invocation) -> Any:
        from agents import Runner

        agent, hooks = self._agent(target, run), _hooks(run)
        options = run.runtime.framework_options
        given = await self._input(agent, native_input)
        result = await Runner.run(agent, given, hooks=hooks, **options)
        state = _edited_call(result, native_input)
        if state is not None:  # the model called the edited tool as told: it runs, approved
            result = await Runner.run(agent, state, hooks=hooks, **options)
        return result

    async def stream(self, target: Any, native_input: Any, run: Invocation) -> AsyncIterator[Any]:
        from agents import Runner

        agent, hooks = self._agent(target, run), _hooks(run)
        options = run.runtime.framework_options
        given = await self._input(agent, native_input)
        result = Runner.run_streamed(agent, given, hooks=hooks, **options)
        try:
            async for delta in _deltas(result):
                yield delta
            state = _edited_call(result, native_input)
            if state is not None:
                result = Runner.run_streamed(agent, state, hooks=hooks, **options)
                async for delta in _deltas(result):
                    yield delta
        finally:
            # cut short (cancelled, out of time), the SDK stops its run without waiting for
            # it: its tool calls end (on the run's stream too) before the run does
            assert result.run_loop_task is not None  # set by run_streamed
            await asyncio.wait({result.run_loop_task})
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
    def _agent(target: Any, run: Invocation) -> Any:
        if not run.native_tools:
            return target
        return target.clone(tools=[*target.tools, *run.native_tools])

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


def _hooks(run: Invocation) -> Any:
    """The run's model hooks as the SDK's own ``RunHooks`` — none when it has no hooks."""
    if not run.runtime.agent.hooks:
        return None
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


def _arguments(item: Any) -> Any:
    try:
        return json.loads(item.arguments or "{}")
    except ValueError:
        return None
