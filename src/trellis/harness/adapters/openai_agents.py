"""OpenAI Agents SDK ``Agent``\\ s.

* input: a message list, the memory context first as a ``system`` message;
* run: ``Runner.run``/``Runner.run_streamed`` on a clone of the agent carrying the harness
  tools next to its own (the caller's agent is never changed);
* pause: ``trellis.current().ask`` stops the run and a resume re-runs it against the journal;
  the SDK's own ``needs_approval`` tools pause with the SDK's ``RunState``, and a resume
  approves or rejects on that state and continues it (``RunState.approve``/``reject``).
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, ClassVar

from trellis.contracts import InterruptDecision, InterruptResolution
from trellis.harness.adapters.base import Extracted, Invocation, NativePause, Output, ToolFormat
from trellis.harness.journal import Pending

#: The raw streaming event that carries a text delta.
TEXT_DELTA = "response.output_text.delta"


@dataclass(frozen=True, slots=True)
class _Continue:
    """A resume from the SDK's serialised run state."""

    state: dict[str, Any]
    call_id: str | None
    approve: bool


class OpenAIAgentsAdapter:
    name: ClassVar[str] = "openai_agents"
    tool_format: ClassVar[ToolFormat] = "openai_agents"
    fixed_tools: ClassVar[bool] = False

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

        agent = self._agent(target, run)
        return await Runner.run(agent, await self._input(agent, native_input))

    async def stream(self, target: Any, native_input: Any, run: Invocation) -> AsyncIterator[Any]:
        from agents import Runner

        agent = self._agent(target, run)
        result = Runner.run_streamed(agent, await self._input(agent, native_input))
        async for event in result.stream_events():
            data = getattr(event, "data", None)
            if event.type == "raw_response_event" and getattr(data, "type", None) == TEXT_DELTA:
                yield data.delta  # type: ignore[union-attr]
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
            args = json.loads(item.arguments) if item.arguments else {}
            pause = NativePause(
                value=None,
                native_id=item.call_id,
                state=output.to_state().to_json(),
                tool=item.name,
                args=args if isinstance(args, dict) else {"arguments": args},
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
        return _Continue(
            pending.native_state,
            pending.native_id,
            resolution.decision is InterruptDecision.APPROVE,
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
                state.reject(item)
        return state
