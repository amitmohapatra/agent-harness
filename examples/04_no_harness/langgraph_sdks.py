"""No harness at all: your own LangGraph graph, with only the platform's SDKs stitched in by your
code — ``trellis.memory``, ``trellis.runs`` and ``trellis.contracts``. Nothing here imports
``trellis.harness``.

* memory — the context for the question goes in as a system message; each tool call, the turn
  and the outcome are recorded;
* your own approval rule — ``create_po`` over 100 units asks a person through LangGraph's own
  ``interrupt`` (the rule is the memory service's expression language,
  ``trellis.memory.approval.evaluate``, the same an administrator writes in the tool catalog);
* runs — the run is recorded in agent-runs, the pause waits in ``role:procurement``'s inbox,
  and the reviewer's ``InterruptResolution`` is what ``Command(resume=...)`` carries back.

Offline the memory service and the run store are in-process stand-ins that take the same calls
(``examples/_support``); with ``MEMORY_URL``/``RUNS_URL`` and ``TRELLIS_API_KEY`` they are real.

    python -m examples.04_no_harness.langgraph_sdks
"""

from __future__ import annotations

import asyncio
from typing import Any

from examples._support.memory import ScriptedMemory
from examples._support.offline import langchain_model, memory_client, runs_store
from langchain_core.messages import AnyMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt

from trellis.contracts import (
    Interrupt,
    InterruptDecision,
    InterruptReason,
    InterruptResolution,
    RunStart,
    RunStatus,
    ToolCall,
)
from trellis.memory import current_context
from trellis.memory.approval import evaluate as holds

TENANT = "default"  # the tenant your key speaks for (a development key's: default)
AGENT = "procurement"
RULE = "qty > 100"  # when an order needs a person


@tool
async def stock(sku: str) -> int:
    """Units of a SKU in stock."""
    count = {"SKU-1": 3}.get(sku, 0)
    if (scope := current_context()) is not None:
        await scope.record_tool("stock", {"sku": sku}, output=count)
    return count


@tool
async def create_po(sku: str, qty: int) -> str:
    """Order more units of a SKU from the supplier."""
    if holds(RULE, {"sku": sku, "qty": qty}):
        answer = InterruptResolution.model_validate(
            interrupt({"question": f"Order {qty} x {sku}?", "args": {"sku": sku, "qty": qty}})
        )
        if answer.decision is not InterruptDecision.APPROVE:
            return f"not ordered: {answer.comment or 'rejected'}"
    po = f"PO-{sku}-{qty}"
    if (scope := current_context()) is not None:
        await scope.record_tool("create_po", {"sku": sku, "qty": qty}, output=po)
    return po


def build(model: Any) -> Any:
    tools = [stock, create_po]
    bound = model.bind_tools(tools)

    async def think(state: MessagesState) -> dict[str, Any]:
        return {"messages": [await bound.ainvoke(state["messages"])]}

    def route(state: MessagesState) -> str:
        return "tools" if getattr(state["messages"][-1], "tool_calls", None) else END

    graph = StateGraph(MessagesState)
    graph.add_node("model", think)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", route)
    graph.add_edge("tools", "model")
    return graph.compile(checkpointer=InMemorySaver())  # production: a shared, durable saver


async def main() -> None:
    runs = runs_store()
    memory = memory_client(ScriptedMemory(context="Procurement orders in batches of 50."))
    model = langchain_model(
        [
            ("stock", {"sku": "SKU-1"}),
            ("create_po", {"sku": "SKU-1", "qty": 150}),
            "SKU-1 was at 3 units; ordered 150 (PO-SKU-1-150).",
        ]
    )
    graph = build(model)
    question, user = "Top up SKU-1 for the quarter.", "ada"
    run = await runs.start(RunStart(tenant_id=TENANT, agent_id=AGENT, user_id=user, input=question))
    scope = memory.bind(tenant_id=TENANT, user_id=user).agent(AGENT, agent_run_id=run.run_id)
    pushed = await scope.context(question, window=False)  # the graph keeps the conversation
    messages: list[AnyMessage | tuple[str, str]] = [
        SystemMessage(pushed.rendered),
        ("user", question),
    ]
    config: Any = {"configurable": {"thread_id": run.run_id}}

    async with scope:  # what current_context() returns inside the tools
        state = await graph.ainvoke({"messages": messages}, config)
        while "__interrupt__" in state:  # your rule asked: the run waits in agent-runs
            asked = state["__interrupt__"][0].value
            await runs.pause(
                Interrupt(
                    tenant_id=TENANT,
                    run_id=run.run_id,
                    reason=InterruptReason.APPROVAL,
                    question=asked["question"],
                    tool_call=ToolCall(tool="create_po", args=asked["args"]),
                    assignee="role:procurement",
                )
            )
            async for waiting in runs.iterate(
                status=RunStatus.PAUSED, assignee="role:procurement", tenant=TENANT
            ):  # a reviewer, later, from any process
                if waiting.run_id == run.run_id and waiting.awaiting is not None:
                    print("inbox:", waiting.awaiting.question)
                    await runs.resume(
                        InterruptResolution(
                            interrupt_id=waiting.awaiting.interrupt_id,
                            run_id=run.run_id,
                            decision=InterruptDecision.APPROVE,
                            reviewer="user:lead",
                        ),
                        tenant=TENANT,
                    )
            record = await runs.get(run.run_id, tenant=TENANT)
            assert record is not None and record.last_resolution is not None
            resume = record.last_resolution.model_dump(mode="json")
            state = await graph.ainvoke(Command(resume=resume), config)

        answer = str(state["messages"][-1].content)
        await scope.history.add([("USER", question), ("ASSISTANT", answer)])
        await scope.feedback("run", run.run_id, "confirm", source="system")
    await runs.finish(run.run_id, RunStatus.SUCCESS, output=answer, tenant=TENANT)
    print(RunStatus.SUCCESS.value, answer)
    await memory.aclose()
    await runs.aclose()


if __name__ == "__main__":
    asyncio.run(main())
