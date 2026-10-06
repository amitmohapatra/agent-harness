"""Way 2, pluggable blocks: a plain LangGraph graph — not wrapped, nothing in it is Trellis's —
with the blocks plugged in around it by the team's own code.

* Memory (``trellis.memory``): the context for the question goes in as a system message; the
  turn, each tool call and the outcome are recorded.
* Governance (``trellis.harness.governance``): ``governed`` checks every call of the graph's
  tools. ``create_po`` is irreversible, so its call asks — through LangGraph's own
  ``interrupt``, and the graph's checkpointer keeps the pause.
* Runs (``trellis.runs``): the run is recorded in agent-runs, the pause waits in
  ``role:procurement``'s inbox, and the reviewer's ``InterruptResolution`` is what
  ``Command(resume=...)`` carries back into the graph.
* Evaluation (``trellis.harness.evals``): the answer is judged on the run's trace.

Offline (no ``RUNS_URL``, ``MEMORY_URL`` or ``BIFROST_URL``) the run store is the in-process
one, the memory service a scripted one in this process, and the model and the judge are
scripted.

    python -m examples.03_way2_governance.langgraph_recipe
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import inspect
from collections.abc import Awaitable, Callable
from typing import Any, Literal

from examples._support.offline import (
    Runs,
    judge_services,
    langchain_model,
    memory_client,
    runs_store,
)
from langchain_core.messages import AnyMessage, SystemMessage
from langchain_core.tools import tool as langchain_tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_config
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
from trellis.harness.evals import EvalCase, EvalServices, grounding, judge, llm_judge
from trellis.harness.governance import Decision, Governance, governed
from trellis.memory import MemoryClient, MemoryContext, PromptContext, current_context

TENANT = "default"  # the tenant your key speaks for (a development key's: default)
AGENT = "procurement"
#: the run's memory scope and the context pushed into it (both None with memory off)
Recall = tuple[MemoryContext | None, PromptContext | None]
#: what governance records for a reviewer's decision (any other decision is a reject)
VERDICTS: dict[InterruptDecision, Literal["approve", "edit"]] = {
    InterruptDecision.APPROVE: "approve",
    InterruptDecision.EDIT: "edit",
}


# --------------------------------------------------------------------------- your tools
async def stock(sku: str) -> int:
    """Units of a SKU in stock."""
    return {"SKU-1": 3}.get(sku, 0)


async def create_po(sku: str, qty: int) -> str:
    """Order more units of a SKU from the supplier."""
    return f"PO-{sku}-{qty}"


def recorded(fn: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
    """Record each call in the run's memory scope (the one ``async with scope:`` entered)."""
    signature = inspect.signature(fn)

    @functools.wraps(fn)
    async def call(*args: Any, **kwargs: Any) -> Any:
        output = await fn(*args, **kwargs)
        if (scope := current_context()) is not None:
            named = dict(signature.bind(*args, **kwargs).arguments)
            await scope.record_tool(fn.__name__, named, output=output)
        return output

    return call


# --------------------------------------------------------------------------- the graph
def build(model: Any, gov: Governance) -> Any:
    """The graph as the team writes it; its tools are governed."""

    async def approval(decision: Decision) -> bool | dict[str, Any]:
        """Ask through LangGraph: ``interrupt`` pauses the graph (its checkpointer keeps the
        pause) and returns what ``Command(resume=...)`` sends — the reviewer's resolution."""
        asked = {"question": decision.question, "tool": decision.tool, "args": dict(decision.args)}
        resolution = InterruptResolution.model_validate(interrupt(asked))
        verdict: Literal["approve", "reject", "edit"] = VERDICTS.get(resolution.decision, "reject")
        await gov.decided(  # the memory service learns approval rules from it (memory on)
            decision,
            verdict,
            reviewer=resolution.reviewer or "unknown",
            run_id=resolution.run_id,
            user=get_config().get("configurable", {})["user"],
            edited=resolution.payload,
        )
        if verdict == "edit":
            return dict(resolution.payload or {})  # run with the reviewer's arguments
        return verdict == "approve"

    tools = [
        langchain_tool(governed(recorded(stock), gov, side_effects="read", on_ask=approval)),
        langchain_tool(
            governed(recorded(create_po), gov, side_effects="irreversible", on_ask=approval)
        ),
    ]
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


# --------------------------------------------------------------------------- one run
async def recall(memory: MemoryClient | None, run_id: str, user: str, question: str) -> Recall:
    """The run's memory scope and the context for the question (memory on)."""
    if memory is None:
        return None, None
    scope = memory.bind(tenant_id=TENANT, user_id=user).agent(AGENT, agent_run_id=run_id)
    # window=False: the graph's checkpointer keeps the conversation itself
    return scope, await scope.context(question, window=False)


async def review(runs: Runs, run_id: str) -> None:
    """A reviewer, any time later, from any process: the inbox, and an answer (here, to the
    run this example started)."""
    async for waiting in runs.iterate(
        status=RunStatus.PAUSED, assignee="role:procurement", tenant=TENANT
    ):
        assert waiting.awaiting is not None
        print("inbox:", waiting.run_id, waiting.awaiting.question)
        if waiting.run_id != run_id:
            continue
        answer = InterruptResolution(
            interrupt_id=waiting.awaiting.interrupt_id,
            run_id=waiting.run_id,
            decision=InterruptDecision.APPROVE,
            reviewer="user:lead",
        )
        await runs.resume(answer, tenant=TENANT)  # RUNNING again, attempt 2


async def finish(
    runs: Runs,
    services: EvalServices,
    memory: Recall,
    *,
    run_id: str,
    question: str,
    answer: str,
) -> None:
    """End the run, record the turn and its outcome (memory on), and judge the answer."""
    await runs.finish(run_id, RunStatus.SUCCESS, output=answer, tenant=TENANT)
    print(RunStatus.SUCCESS.value, answer)
    scope, pushed = memory
    if scope is not None:
        await scope.history.add([("USER", question), ("ASSISTANT", answer)])
        await scope.feedback("run", run_id, "confirm", source="system")
    case = EvalCase(
        input=question,
        output=answer,
        run_id=run_id,
        bundle_id=pushed.bundle_id if pushed is not None else None,
        memory=scope,
    )
    scores, failed = await judge(
        case, [grounding(), llm_judge("Says what was ordered.")], services=services
    )
    print("judged:", [(s.name, s.value) for s in scores], failed)


async def main() -> None:
    gov = Governance.from_env(agent_id=AGENT, tenant=TENANT)
    runs, memory, services = runs_store(), memory_client(), judge_services()
    model = langchain_model(
        [
            ("stock", {"sku": "SKU-1"}),
            ("create_po", {"sku": "SKU-1", "qty": 20}),
            "SKU-1 was at 3 units; ordered 20 (PO-SKU-1-20).",
        ]
    )
    graph = build(model, gov)
    question, user = "Top up SKU-1 if it is low.", "ada"
    run = await runs.start(RunStart(tenant_id=TENANT, agent_id=AGENT, user_id=user, input=question))
    scope, pushed = await recall(memory, run.run_id, user, question)
    messages: list[AnyMessage | tuple[str, str]] = [("user", question)]
    if pushed is not None:
        messages.insert(0, SystemMessage(pushed.rendered))
    config: Any = {"configurable": {"thread_id": run.run_id, "user": user}}

    async with scope or contextlib.nullcontext():  # what `recorded` records in
        state = await graph.ainvoke({"messages": messages}, config)
        while "__interrupt__" in state:  # governance asked: the run waits in agent-runs
            asked = state["__interrupt__"][0].value
            await runs.pause(
                Interrupt(
                    tenant_id=TENANT,
                    run_id=run.run_id,
                    reason=InterruptReason.APPROVAL,
                    question=asked["question"],
                    tool_call=ToolCall(tool=asked["tool"], args=asked["args"]),
                    assignee="role:procurement",
                )
            )  # the graph's own checkpointer keeps where it stopped

            await review(runs, run.run_id)

            # continue the graph where it stopped, with the reviewer's resolution
            record = await runs.get(run.run_id, tenant=TENANT)
            assert record is not None and record.last_resolution is not None
            resume = record.last_resolution.model_dump(mode="json")
            state = await graph.ainvoke(Command(resume=resume), config)

    answer = str(state["messages"][-1].content)
    await finish(
        runs, services, (scope, pushed), run_id=run.run_id, question=question, answer=answer
    )
    for client in (gov, runs, services, memory):
        if client is not None:
            await client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
