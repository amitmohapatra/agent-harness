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
  journal answers the questions already asked. A resume where the checkpointer no longer holds
  the pause (an ``InMemorySaver`` in another process: a worker, a replica) answers a harness
  pause from the journal the same way (:func:`holds`); a graph's own pause cannot be answered
  there, and the run fails saying so. LangChain's ``HumanInTheLoopMiddleware`` (Deep
  Agents' ``interrupt_on``) pauses with its own request: it is an approval of the calls it
  holds, and the harness's decision is turned into the ``{"decisions": [...]}`` it resumes
  with (:func:`hitl_response`);
* tools: fixed when the graph is compiled, so harness tools come from ``h.tools(...)`` at
  build time.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
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
from trellis.harness.asking import answer_of
from trellis.harness.journal import Pending
from trellis.harness.runtime import MARKER, reason_of

#: The journal key a graph's own ``interrupt(...)`` (not ``ask``) is filed under.
FOREIGN: Final = "langgraph"
#: The journal key of a ``HumanInTheLoopMiddleware`` pause (Deep Agents' ``interrupt_on``).
HITL: Final = "langchain_hitl"
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

        if pending.key == HITL:
            value: Any = hitl_response(pending.interrupt.payload or {}, resolution)
        elif pending.key == FOREIGN:
            value = answer_of(resolution)
        else:
            value = resolution.model_dump(mode="json")
        return Command(resume={pending.native_id: value})

    @staticmethod
    def _config(target: Any, run: Invocation) -> dict[str, Any]:
        runtime = run.runtime
        if _checkpointed(target):
            from langgraph.types import interrupt

            runtime.suspend = interrupt
        return {"configurable": {"thread_id": runtime.thread or runtime.run_id}}


async def holds(target: Any, thread: str, native_id: str) -> bool:
    """Whether the graph's checkpointer still holds the pause ``native_id`` on ``thread`` —
    what a resume in place needs. An ``InMemorySaver`` holds it in the process that paused,
    and nowhere else. (A pause with a native id is a checkpointed graph's.)"""
    state = await target.aget_state({"configurable": {"thread_id": thread}})
    return any(paused.id == native_id for paused in state.interrupts)


def is_hitl(value: Any) -> bool:
    """Whether an interrupt's value is a ``HumanInTheLoopMiddleware`` request (its
    ``action_requests`` and their ``review_configs``)."""
    return (
        isinstance(value, dict)
        and isinstance(value.get("action_requests"), list)
        and bool(value["action_requests"])
        and isinstance(value.get("review_configs"), list)
    )


def hitl_response(request: dict[str, Any], resolution: InterruptResolution) -> dict[str, Any]:
    """The ``{"decisions": [...]}`` a ``HumanInTheLoopMiddleware`` pause resumes with — one
    decision per call it holds, in order — for a harness decision:

    * ``approve`` approves every call;
    * ``edit`` runs the first call (the one the interrupt shows) with the edited arguments and
      approves the others;
    * ``reject`` rejects every call, with the reviewer's reason (``answer=``) as the message the
      model reads;
    * ``answer`` responds for every call with the answer (the tool does not run).

    An answer (or edit) that already is ``{"decisions": [...]}`` is sent as it is. A decision a
    call's ``allowed_decisions`` does not include raises ``ConfigurationError``."""
    raw = resolution.payload if resolution.decision is InterruptDecision.EDIT else resolution.answer
    if isinstance(raw, dict) and isinstance(raw.get("decisions"), list):
        return raw
    actions: list[dict[str, Any]] = request["action_requests"]
    configs: list[dict[str, Any]] = request["review_configs"]
    decisions = [_decision(resolution, action, first=n == 0) for n, action in enumerate(actions)]
    for action, config, decision in zip(actions, configs, decisions, strict=False):
        allowed = config.get("allowed_decisions") or []
        if decision["type"] not in allowed:
            raise ConfigurationError(
                f"{action.get('name')} does not allow the decision {decision['type']!r} "
                f"(allowed: {', '.join(allowed)})"
            )
    return {"decisions": decisions}


def _decision(resolution: InterruptResolution, action: dict[str, Any], *, first: bool) -> Any:
    decision = resolution.decision
    if decision is InterruptDecision.REJECT:
        reason = reason_of(resolution)
        return {"type": "reject", "message": reason} if reason else {"type": "reject"}
    if decision is InterruptDecision.EDIT and first:
        return {
            "type": "edit",
            "edited_action": {"name": action["name"], "args": resolution.payload},
        }
    if decision is InterruptDecision.ANSWER:
        answer = resolution.answer
        message = answer if isinstance(answer, str) else json.dumps(answer, default=str)
        return {"type": "respond", "message": message}
    return {"type": "approve"}


def bound_tools(target: Any) -> list[Any]:
    """The tools a compiled graph's tool nodes run (Deep Agents included)."""
    found: list[Any] = []
    for node in getattr(target, "nodes", {}).values():
        found.extend(getattr(getattr(node, "bound", None), "tools_by_name", {}).values())
    return found


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
