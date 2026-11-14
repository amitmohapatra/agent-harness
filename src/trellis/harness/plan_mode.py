"""Plan mode for ``ReAct``: read-only exploration, then a plan a person approves, then the work
— Claude Code's and Kimi's plan mode, built from LangChain's pieces:

    agent = h.wrap(ReAct("...", model, tools=[...], middleware=[PlanMode()]), id="refunds")
    await agent.run("refund o1", user="u", framework_options={"configurable": {"plan_mode": True}})

**Entered on request, never by default**, as theirs is (Claude's ``permission_mode`` is unset
unless a caller sets ``"plan"``; Kimi's is the user's to enter): a run is in plan mode when its
``framework_options`` say so (LangGraph's ``configurable``: per run, or for every run of an
agent at ``h.wrap``), where Claude's ``permission_mode`` and Kimi's ``mode`` go.

What it is made of:

* **the plan** is LangChain's ``write_todos`` (``TodoListMiddleware``), offered from the run's
  first call (``automatic.Planning``);
* **the approval** is LangChain's ``HumanInTheLoopMiddleware`` on ``write_todos``, with its own
  ``when`` predicate: the call pauses the run until the plan is approved — the harness's
  approval, one question (approve, edit the plan, or reject with what to change) —, never after;
* **the read-only rule**: until then a call runs only if governance rates it a read (the
  catalog's risk, else the tool's own). Any other — a write, a tool that says nothing, a
  hand-over to another agent, whose own calls this rule would not reach — is answered, not run,
  with what to do instead, and recorded as the run's rejected call.

Approved, the plan is in the graph's state (``plan_approved``, checkpointed with the run: a
resume in any worker keeps it) and every call is governed as ever."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable, Mapping
from typing import Annotated, Any, Final, NotRequired

from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.agents.middleware.types import (
    AgentState,
    ModelRequest,
    ModelResponse,
    PrivateStateAttr,
)
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.runtime import Runtime as GraphRuntime

from trellis.contracts import ToolCall, ToolSpec
from trellis.harness.adapters.base import HELPER
from trellis.harness.automatic import TODOS, plan_mode_asked
from trellis.harness.hooks import Deny, denied
from trellis.harness.middleware import FILE_READS, SAVING
from trellis.harness.runtime import Runtime, current
from trellis.harness.tools import bridge

#: The graph's state key that says the run's plan was approved.
APPROVED: Final = "plan_approved"
#: What the model is told while it plans, after its instructions.
PLANNING: Final = (
    "Plan mode is on. Explore with the tools that only read, then present your plan by calling "
    f"{TODOS}: a person approves it before you act. Until then no tool that changes anything "
    "runs."
)
#: What a call that does more than read is answered with while the run plans.
WAITS: Final = f"plan mode is on: present your plan with {TODOS} and wait for its approval"
#: The graph's own reads, of no spec in the run's tool set: its state's files and the results
#: the run keeps.
OWN_READS: Final = FILE_READS | SAVING.keys()


class _PlanState(AgentState[Any]):
    plan_approved: NotRequired[Annotated[bool, PrivateStateAttr]]


def planning(state: Mapping[str, Any]) -> bool:
    """Whether the run plans: it is in plan mode and its plan is not approved yet."""
    return plan_mode_asked() and not state.get(APPROVED)


def _awaits_approval(request: ToolCallRequest) -> bool:
    """LangChain's ``when`` predicate of ``write_todos``: the plan asks while the run plans."""
    return planning(request.state)


class PlanMode(HumanInTheLoopMiddleware[Any, Any, Any]):
    """Plan mode (the module's doc), for ``ReAct(middleware=[PlanMode()])``. Outside a harness
    run, and in a run that did not enter plan mode, it steps aside."""

    state_schema = _PlanState

    def __init__(self) -> None:
        super().__init__(
            {
                TODOS: {
                    "allowed_decisions": ["approve", "edit", "reject"],
                    "when": _awaits_approval,
                }
            },
            description_prefix="The plan needs approval",
        )

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        if current() is None or not planning(request.state):
            return await handler(request)
        system = request.system_message
        said = f"{system.text}\n\n{PLANNING}" if system is not None else PLANNING
        return await handler(request.override(system_message=SystemMessage(content=said)))

    async def aafter_model(
        self, state: AgentState[Any], runtime: GraphRuntime[Any]
    ) -> dict[str, Any] | None:
        run = current()
        reply = next((m for m in reversed(state["messages"]) if isinstance(m, AIMessage)), None)
        if run is None or reply is None or not planning(state):
            return await super().aafter_model(state, runtime)
        refused = await _refused(run, reply)
        update = await super().aafter_model(state, runtime)  # the plan's approval: its pause
        update = dict(update or {})
        answered = update.get("messages") or [reply]
        if any(c["name"] == TODOS for c in answered[0].tool_calls) and len(answered) == 1:
            update[APPROVED] = True  # approved (or edited): not rejected, which answers it
        if refused:
            update["messages"] = [*answered, *refused]
        return update or None


async def _refused(run: Runtime, reply: AIMessage) -> list[ToolMessage]:
    """The answers of ``reply``'s calls that may not run while the run plans (the module's
    doc), each recorded as the run's rejected call."""
    governance = run.agent.harness._governance(run.tenant)
    answers: list[ToolMessage] = []
    for call in reply.tool_calls:
        name, args = call["name"], dict(call["args"])
        tool = run.toolbox.get(name)
        spec = tool.spec if tool is not None else run.native.get(name)
        if name == TODOS or (spec is None and name in OWN_READS):
            continue
        spec = spec or ToolSpec(name=name)
        decision = await governance.check(
            name, args, side_effects=spec.side_effects, agent_id=run.agent_id
        )
        if decision.risk == "read" and HELPER not in spec.tags:
            continue
        outcome = denied(ToolCall(tool=name, args=args), Deny(WAITS))
        ref = str(call["id"])
        await bridge.observed(run, spec, args, outcome, ref=ref, started=time.time())
        answers.append(
            ToolMessage(content=str(outcome.output), name=name, tool_call_id=ref, status="error")
        )
    return answers


__all__ = ["PlanMode"]
