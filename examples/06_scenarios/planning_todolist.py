"""Scenario: opt-in planning. A ``ReAct`` agent given LangChain's ``TodoListMiddleware`` keeps a
``write_todos`` plan, updates it as it works, and answers when every step is done.

``ReAct(..., middleware=[...])`` adds any LangChain or Deep Agents middleware to the graph,
before the harness's ``ModelHooks`` (so the ``chat`` span shows what the model is sent); the
plan is part of the graph's state, which the run's checkpoint keeps — a resume after a pause or
a crash continues the same plan. ``write_todos`` is the middleware's own tool: it changes the
graph's state, not the world, so it is not governed or recorded as a harness tool.

    python -m examples.06_scenarios.planning_todolist
"""

from __future__ import annotations

import asyncio
from typing import Any

from examples._support.offline import react_model
from langchain.agents.middleware import TodoListMiddleware

from trellis import Harness, Hooks, ModelCall, ReAct, tool
from trellis.contracts import RunEventType


class ShowPlan(Hooks):
    """A model hook: each plan the model writes, as it writes it."""

    async def after_model(self, call: ModelCall, reply: Any) -> None:
        # a LangChain graph's reply: the middleware's ModelResponse, its messages in .result
        for message in getattr(reply, "result", []):
            for planned in getattr(message, "tool_calls", None) or []:
                if planned["name"] == "write_todos":
                    todos = planned["args"]["todos"]
                    print("plan:", ", ".join(f"{t['content']} [{t['status']}]" for t in todos))


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
            system="You plan replenishment. Keep a todo list; work through it.",
            model=model,
            middleware=[TodoListMiddleware()],
        )
        agent = h.wrap(target, id="planner", tools=[inventory, draft_order], hooks=[ShowPlan()])
        async for event in agent.stream("Replenish the north warehouse.", user="ada"):
            if event.type is RunEventType.TOOL_CALL_RESULT:
                print(f"{event.data['tool']}: {str(event.data['output'])[:80]}")
            if event.type is RunEventType.RUN_FINISHED:
                print(event.outcome, event.data.get("result"))


if __name__ == "__main__":
    asyncio.run(main())
