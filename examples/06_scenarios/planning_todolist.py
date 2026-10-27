"""Scenario: planning on a long task. ``ReAct`` offers LangChain's ``write_todos`` plan (its
``TodoListMiddleware``) once a task is long — it asks for steps, or it took a few calls
already — on a model whose window has room: the agent keeps the plan, updates it as it works,
and answers when every step is done.

The plan is part of the graph's state, which the run's checkpoint keeps — a resume after a
pause or a crash continues the same plan. ``write_todos`` is the middleware's own tool: it
changes the graph's state, not the world, so it is not governed or recorded as a harness tool.

    python -m examples.06_scenarios.planning_todolist
"""

from __future__ import annotations

import asyncio
from typing import Any

from examples._support.offline import react_model
from langchain.agents.middleware import AgentMiddleware

from trellis import Harness, ReAct, tool
from trellis.contracts import RunEventType


class ShowPlan(AgentMiddleware):
    """LangChain's own slot for a model call: each plan the model writes, as it writes it."""

    async def awrap_model_call(self, request: Any, handler: Any) -> Any:
        reply = await handler(request)
        # the middleware's ModelResponse: its messages in .result
        for message in getattr(reply, "result", []):
            for planned in getattr(message, "tool_calls", None) or []:
                if planned["name"] == "write_todos":
                    todos = planned["args"]["todos"]
                    print("plan:", ", ".join(f"{t['content']} [{t['status']}]" for t in todos))
        return reply


@tool(side_effects="read")
def inventory(warehouse: str) -> str:
    """Stock levels in a warehouse."""
    return f"{warehouse}: 3 SKUs below their minimum"


@tool(side_effects="write")
def draft_order(warehouse: str) -> str:
    """Draft a replenishment order for a warehouse (a draft: nothing is sent)."""
    return f"draft order D-1 for {warehouse}"


def todos(*done: bool) -> dict[str, list[dict[str, str]]]:
    steps = ["check inventory", "draft the order", "summarise"]
    states = ["completed" if d else "pending" for d in done]
    states += ["pending"] * (len(steps) - len(states))
    return {"todos": [{"content": s, "status": t} for s, t in zip(steps, states, strict=True)]}


async def main() -> None:
    model = react_model(
        [
            ("write_todos", todos()),
            ("inventory", {"warehouse": "north"}),
            ("write_todos", todos(True)),
            ("draft_order", {"warehouse": "north"}),
            ("write_todos", todos(True, True, True)),
            "North is 3 SKUs short; draft order D-1 is ready for review.",
        ]
    )
    async with Harness() as h:
        target = ReAct(
            system_prompt="You plan replenishment. Keep a todo list; work through it.",
            model=model,
            tools=[inventory, draft_order],
            context_window=128_000,  # room for the long-task tools
            middleware=[ShowPlan()],
        )
        agent = h.wrap(target, id="planner")
        task = "Replenish the north warehouse, step by step."
        async for event in agent.stream(task, user="ada"):
            if event.type is RunEventType.TOOL_CALL_RESULT:
                print(f"{event.data['tool']}: {str(event.data['output'])[:80]}")
            if event.type is RunEventType.RUN_FINISHED:
                print(event.outcome, event.data.get("result"))


if __name__ == "__main__":
    asyncio.run(main())
