"""Way 2, contracts: the records every block takes and returns (``trellis.contracts``), which is
why a run paused either way is one ``RunRecord`` in one inbox.

The same run is shown from both sides:

* a wrapped agent pauses: its ``Result.interrupt`` is a contracts ``Interrupt``, the run store
  holds a ``RunRecord``, ``agent.stream`` yields ``RunEvent``s;
* your own code (Way 2) answers it with an ``InterruptResolution`` — checked with ``resolves`` —
  and turns the decision into the ``Feedback`` record the memory service learns from;
* any failure becomes an ``AgentError`` (``classify``: may a retry help?).

    python -m examples.03_way2_contracts.records
"""

from __future__ import annotations

import asyncio

from trellis import Harness, Runtime
from trellis.contracts import (
    AgentError,
    AgentExecutionContext,
    InterruptDecision,
    InterruptResolution,
    RunEventType,
    RunStatus,
    classify,
)


async def buyer(sku: str, agent: Runtime) -> str:
    qty = await agent.ask(f"How many {sku}?", expects={"type": "integer"}, assignee="role:buyers")
    return f"ordered {qty} x {sku}"


async def main() -> None:
    async with Harness() as h:
        agent = h.wrap(buyer, id="buyer")
        events = [event async for event in agent.stream("SKU-1", user="ada")]
        print("events:", [e.type.value for e in events])
        finished = events[-1]
        assert finished.type is RunEventType.RUN_FINISHED

        [waiting] = await h.inbox("role:buyers")  # a trellis.runs.RunSummary
        record = await h.runs.get(waiting.run_id)  # a contracts RunRecord
        assert record is not None and record.awaiting is not None
        asked = record.awaiting  # a contracts Interrupt
        print(record.status.value, "->", asked.reason.value, asked.question, asked.expects)
        assert record.status.can_become(RunStatus.RUNNING)

        answer = InterruptResolution(
            interrupt_id=asked.interrupt_id,
            run_id=record.run_id,
            decision=InterruptDecision.ANSWER,
            answer=12,
            reviewer="user:lee",
        )
        assert answer.resolves(asked)
        context = AgentExecutionContext.create(
            tenant_id=record.tenant_id, user_id="ada", agent_id="buyer", agent_run_id=record.run_id
        )
        feedback = answer.to_feedback(asked, context)  # None: an answer is not a call's verdict
        print("feedback record:", feedback)

        done = await agent.resume(answer.interrupt_id, "answer", answer=12, reviewer="lee")
        print(done.status.value, done.answer)

    error = AgentError.of(TimeoutError("the supplier did not answer"), source="tools")
    print(error.code, error.category.value, "retryable:", error.retryable)
    print("a KeyError is", classify(KeyError("sku")).value)


if __name__ == "__main__":
    asyncio.run(main())
