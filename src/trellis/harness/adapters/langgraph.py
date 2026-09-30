"""LangGraph compiled graphs — and Deep Agents, whose ``create_deep_agent`` returns one.

* input: ``{"messages": [...]}`` with the memory context as a leading system message (a
  string input becomes one user message; any other state dict passes through untouched).
  The context message has a fixed id, so a checkpointed thread holds at most one — each turn
  replaces it in place (``add_messages``) — and it leaves out the recent conversation, which
  the checkpointer already holds;
* run: ``ainvoke``/``astream`` with ``version="v2"``, on the LangGraph thread named by the
  run's thread (or the run id);
* pause: with a checkpointer, ``trellis.current().ask`` *is* LangGraph's ``interrupt``, and a
  resume is ``Command(resume=...)``; without one the run is re-executed from its input and the
  journal answers the questions already asked;
* tools: fixed when the graph is compiled, so harness tools come from ``h.tools(...)`` at
  build time.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, ClassVar, Final

from trellis.contracts import ConfigurationError, InterruptResolution
from trellis.harness.adapters.base import (
    Extracted,
    Invocation,
    Narrowing,
    NativePause,
    Output,
    ToolFormat,
)
from trellis.harness.journal import Pending
from trellis.harness.runtime import MARKER, answer_of

#: The journal key a graph's own ``interrupt(...)`` (not ``ask``) is filed under.
FOREIGN: Final = "langgraph"
#: The id of the memory context message: one per thread, replaced every turn.
CONTEXT_MESSAGE_ID: Final = "trellis-memory-context"


class LangGraphAdapter:
    name: ClassVar[str] = "langgraph"
    tool_format: ClassVar[ToolFormat] = "langchain"
    fixed_tools: ClassVar[bool] = True
    narrows: ClassVar[Narrowing] = "none"

    def keeps_conversation(self, target: Any) -> bool:
        return _checkpointed(target)

    def prepare_input(self, target: Any, input: Any, context: str | None) -> Any:
        from langchain_core.messages import SystemMessage

        system = [SystemMessage(content=context, id=CONTEXT_MESSAGE_ID)] if context else []
        if isinstance(input, str):
            return {"messages": [*system, {"role": "user", "content": input}]}
        if isinstance(input, list):
            return {"messages": [*system, *input]}
        if isinstance(input, dict) and isinstance(input.get("messages"), list):
            return {**input, "messages": [*system, *input["messages"]]}
        return input

    async def invoke(self, target: Any, native_input: Any, run: Invocation) -> Any:
        return await target.ainvoke(native_input, self._config(target, run), version="v2")

    async def stream(self, target: Any, native_input: Any, run: Invocation) -> AsyncIterator[Any]:
        from langchain_core.messages import AIMessage
        from langgraph.types import GraphOutput

        values: Any = None
        interrupts: tuple[Any, ...] = ()
        async for part in target.astream(
            native_input,
            self._config(target, run),
            stream_mode=["messages", "values"],
            version="v2",
        ):
            if part["type"] == "messages":
                chunk = part["data"][0]
                if (
                    isinstance(chunk, AIMessage)
                    and isinstance(chunk.content, str)
                    and chunk.content
                ):
                    yield chunk.content
            elif part["type"] == "values":
                values = part["data"]
                interrupts = part["interrupts"] or interrupts
        yield Output(GraphOutput(value=values, interrupts=interrupts))

    def extract(self, target: Any, output: Any) -> Extracted:
        state = output.value
        answer: Any = None
        if isinstance(state, dict):
            answer = state.get("structured_response")
            if answer is None:
                answer = _last_ai_text(state.get("messages") or [])
        transcript = [("assistant", answer)] if isinstance(answer, str) and answer else []
        pause = None
        if output.interrupts:
            ours = [
                i for i in output.interrupts if isinstance(i.value, dict) and i.value.get(MARKER)
            ]
            if not ours and not _checkpointed(target):
                raise ConfigurationError(
                    "the graph called interrupt() without a checkpointer, so it cannot be "
                    "resumed: compile it with one, or ask through trellis.current().ask"
                )
            first = (ours or list(output.interrupts))[0]
            pause = NativePause(value=first.value, native_id=first.id)
        return Extracted(answer=answer, transcript=transcript, pause=pause)  # type: ignore[arg-type]

    def resume_input(
        self,
        target: Any,
        native_input: Any,
        pending: Pending,
        resolution: InterruptResolution,
    ) -> Any:
        if pending.native_id is None or not _checkpointed(target):
            return native_input
        from langgraph.types import Command

        value = (
            answer_of(resolution) if pending.key == FOREIGN else resolution.model_dump(mode="json")
        )
        return Command(resume={pending.native_id: value})

    @staticmethod
    def _config(target: Any, run: Invocation) -> dict[str, Any]:
        runtime = run.runtime
        if _checkpointed(target):
            from langgraph.types import interrupt

            runtime.suspend = interrupt
        return {"configurable": {"thread_id": runtime.thread or runtime.run_id}}


def _checkpointed(target: Any) -> bool:
    from langgraph.checkpoint.base import BaseCheckpointSaver

    return isinstance(getattr(target, "checkpointer", None), BaseCheckpointSaver)


def _last_ai_text(messages: list[Any]) -> str | None:
    for message in reversed(messages):
        if getattr(message, "type", None) == "ai":
            text = (
                message.text if isinstance(getattr(message, "text", None), str) else message.content
            )
            if isinstance(text, str) and text:
                return text
    return None
