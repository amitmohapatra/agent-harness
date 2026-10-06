"""No harness at all: your own Deep Agent, with only the platform's SDKs stitched in by your code
— ``trellis.memory``, ``trellis.runs`` and ``trellis.contracts``. Nothing here imports
``trellis.harness``.

* memory — the context for the request is part of the agent's instructions for this run; the
  turn and the outcome are recorded;
* Deep Agents' own approvals — ``interrupt_on={"refund": True}`` pauses before every refund
  (LangChain's ``HumanInTheLoopMiddleware``), its checkpointer keeping the pause;
* runs — the pause waits in ``role:support-leads``'s inbox in agent-runs, and the reviewer's
  decision goes back as the middleware's own ``{"decisions": [...]}``.

Offline the memory service and the run store are in-process stand-ins that take the same calls
(``examples/_support``); with ``MEMORY_URL``/``RUNS_URL`` and ``TRELLIS_API_KEY`` they are real.

    python -m examples.04_no_harness.deepagents_sdks
"""

from __future__ import annotations

import asyncio
from typing import Any

from deepagents import create_deep_agent
from examples._support.memory import ScriptedMemory
from examples._support.offline import langchain_model, memory_client, runs_store
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from trellis.contracts import (
    Interrupt,
    InterruptDecision,
    InterruptReason,
    InterruptResolution,
    RunStart,
    RunStatus,
    ToolCall,
)

TENANT = "default"
AGENT = "refunds"


def refund(order: str, amount: float) -> str:
    """Refund an order."""
    return f"refunded {amount} on {order}"


async def main() -> None:
    runs = runs_store()
    memory = memory_client(ScriptedMemory(context="Ada is a gold customer: refund in full."))
    question, user = "Refund order o-7 (40 EUR), it arrived broken.", "ada"
    run = await runs.start(RunStart(tenant_id=TENANT, agent_id=AGENT, user_id=user, input=question))
    scope = memory.bind(tenant_id=TENANT, user_id=user).agent(AGENT, agent_run_id=run.run_id)
    pushed = await scope.context(question, window=False)

    agent = create_deep_agent(
        model=langchain_model([("refund", {"order": "o-7", "amount": 40.0}), "Refunded 40 EUR."]),
        tools=[refund],
        system_prompt=f"You process refunds.\n\n{pushed.rendered}",
        interrupt_on={"refund": True},
        checkpointer=InMemorySaver(),
    )
    config: Any = {"configurable": {"thread_id": run.run_id}}
    first: Any = {"messages": [("user", question)]}
    state = await agent.ainvoke(first, config)
    while "__interrupt__" in state:
        request = state["__interrupt__"][0].value  # the middleware's: action_requests, ...
        action = request["action_requests"][0]
        await runs.pause(
            Interrupt(
                tenant_id=TENANT,
                run_id=run.run_id,
                reason=InterruptReason.APPROVAL,
                question=f"Approve {action['name']}?",
                tool_call=ToolCall(tool=action["name"], args=action["args"]),
                assignee="role:support-leads",
                payload=request,
            )
        )
        record = await runs.get(run.run_id, tenant=TENANT)  # a reviewer reads it in the inbox
        assert record is not None and record.awaiting is not None
        call = record.awaiting.tool_call
        print("inbox:", record.awaiting.question, call.args if call else "")
        await runs.resume(
            InterruptResolution(
                interrupt_id=record.awaiting.interrupt_id,
                run_id=run.run_id,
                decision=InterruptDecision.APPROVE,
                reviewer="user:lead",
            ),
            tenant=TENANT,
        )
        state = await agent.ainvoke(Command(resume={"decisions": [{"type": "approve"}]}), config)

    answer = str(state["messages"][-1].content)
    await scope.history.add([("USER", question), ("ASSISTANT", answer)])
    await scope.feedback("run", run.run_id, "confirm", source="system")
    await runs.finish(run.run_id, RunStatus.SUCCESS, output=answer, tenant=TENANT)
    print(RunStatus.SUCCESS.value, answer)
    await memory.aclose()
    await runs.aclose()


if __name__ == "__main__":
    asyncio.run(main())
