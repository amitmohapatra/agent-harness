"""Way 2, pluggable blocks: a plain OpenAI Agents SDK ``Agent`` run with ``Runner.run`` — not
wrapped — with the blocks plugged in by the team's own code.

* Memory (``trellis.memory``): the context for the question is a leading ``system`` input item;
  the turn, each tool call and the outcome are recorded.
* Governance (``trellis.harness.governance``): each tool's ``needs_approval`` asks
  ``Governance.check``; ``create_po`` is irreversible, so its call interrupts the run — the
  SDK's own approval pause.
* Runs (``trellis.runs``): the SDK's ``RunState`` (JSON) is the paused run's checkpoint in
  agent-runs, so any process can continue it; the pause waits in ``role:procurement``'s inbox
  and the reviewer's answer approves or rejects the call on the restored state.
* Evaluation (``trellis.harness.evals``): the answer is judged on the run's trace.

Offline (no ``RUNS_URL``, ``MEMORY_URL`` or ``BIFROST_URL``) the run store is the in-process
one, the memory service a scripted one in this process, and the model and the judge are
scripted.

    python -m examples.03_way2_governance.openai_agents_recipe
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import inspect
import json
from collections.abc import Awaitable, Callable
from typing import Any

from agents import (
    Agent,
    RunContextWrapper,
    Runner,
    RunState,
    function_tool,
    set_tracing_disabled,
)
from examples._support.offline import (
    Runs,
    judge_services,
    memory_client,
    openai_agents_model,
    runs_store,
)

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
from trellis.harness.governance import Governance
from trellis.memory import MemoryClient, MemoryContext, PromptContext, current_context

TENANT = "default"  # the tenant your key speaks for (a development key's: default)
AGENT = "procurement"
#: what each tool does: governance decides from it (and the catalog, memory on)
SIDE_EFFECTS = {"stock": "read", "create_po": "irreversible"}
#: the run's memory scope and the context pushed into it (both None with memory off)
Recall = tuple[MemoryContext | None, PromptContext | None]

set_tracing_disabled(True)  # the SDK's own tracing goes to OpenAI


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


def checked(gov: Governance, tool: str) -> Any:
    """A ``needs_approval`` that asks governance: the call needs approval when it asks."""

    async def needs_approval(ctx: RunContextWrapper[Any], args: dict[str, Any], _: str) -> bool:
        return (await gov.check(tool, args, side_effects=SIDE_EFFECTS[tool])).asks

    return needs_approval


# --------------------------------------------------------------------------- one run
async def recall(memory: MemoryClient | None, run_id: str, user: str, question: str) -> Recall:
    """The run's memory scope and the context for the question (memory on)."""
    if memory is None:
        return None, None
    scope = memory.bind(tenant_id=TENANT, user_id=user).agent(AGENT, agent_run_id=run_id)
    return scope, await scope.context(question)


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
    model = openai_agents_model(
        [
            ("stock", {"sku": "SKU-1"}),
            ("create_po", {"sku": "SKU-1", "qty": 20}),
            "SKU-1 was at 3 units; ordered 20 (PO-SKU-1-20).",
        ]
    )
    buyer = Agent(
        name="buyer",
        instructions="Keep stock topped up: check it, and order when it is low.",
        model=model,
        tools=[
            function_tool(recorded(stock), needs_approval=checked(gov, "stock")),
            function_tool(recorded(create_po), needs_approval=checked(gov, "create_po")),
        ],
    )
    question, user = "Top up SKU-1 if it is low.", "ada"
    run = await runs.start(RunStart(tenant_id=TENANT, agent_id=AGENT, user_id=user, input=question))
    scope, pushed = await recall(memory, run.run_id, user, question)
    items: list[Any] = [{"role": "user", "content": question}]
    if pushed is not None:
        items.insert(0, {"role": "system", "content": pushed.rendered})

    async with scope or contextlib.nullcontext():  # what `recorded` records in
        result = await Runner.run(buyer, items)
        while result.interruptions:  # governance asked: the run waits in agent-runs
            item = result.interruptions[0]
            tool, args = item.name or "", json.loads(item.arguments or "{}")
            decision = await gov.check(tool, args, side_effects=SIDE_EFFECTS[tool])
            asked = Interrupt(
                tenant_id=TENANT,
                run_id=run.run_id,
                reason=InterruptReason.APPROVAL,
                question=decision.question,
                tool_call=ToolCall(tool=tool, args=args),
                assignee="role:procurement",
            )
            # the SDK's state is the checkpoint: any process can continue the run
            state = {"state": result.to_state().to_json(), "call_id": item.call_id}
            await runs.pause(asked, checkpoint=state)

            await review(runs, run.run_id)

            record = await runs.get(run.run_id, tenant=TENANT)
            assert record is not None and record.checkpoint and record.last_resolution
            resolution = record.last_resolution
            approved = resolution.decision is InterruptDecision.APPROVE
            await gov.decided(  # the memory service learns approval rules from it (memory on)
                decision,
                "approve" if approved else "reject",
                reviewer=resolution.reviewer or "unknown",
                run_id=run.run_id,
                user=user,
            )
            restored = await RunState.from_json(buyer, record.checkpoint["state"])
            for pending in restored.get_interruptions():
                if pending.call_id != record.checkpoint["call_id"]:
                    continue
                if approved:
                    restored.approve(pending)
                else:
                    restored.reject(pending, rejection_message=str(resolution.answer or ""))
            result = await Runner.run(buyer, restored)

    answer = str(result.final_output)
    await finish(
        runs, services, (scope, pushed), run_id=run.run_id, question=question, answer=answer
    )
    for client in (gov, runs, services, memory):
        if client is not None:
            await client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
