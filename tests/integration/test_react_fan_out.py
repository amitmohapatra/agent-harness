"""The fan-out recipe (``docs/frameworks/react.md``): a LangGraph ``Send`` map over the items
of a task, given to ``ReAct`` as a sub-agent of its own (Deep Agents' ``CompiledSubAgent``) —
each item a child run of a wrapped agent, worked at once under the harness's admission limit,
their answers gathered into the sub-agent's one."""

from __future__ import annotations

import asyncio
import operator
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, AnyMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Send

import trellis
from tests.support.models import ScriptedChat
from trellis import Harness, ReAct, Runtime
from trellis.contracts import RunStatus
from trellis.harness import admission

ITEMS = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]


class Fan(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    item: str
    found: Annotated[list[str], operator.add]


def fan_out(worker: Any) -> Any:
    """The recipe: one ``Send`` an item (a line of the task), each a run of ``worker``."""

    def items(state: Any) -> list[Send]:
        task = str(state["messages"][-1].content)
        return [Send("work", {"item": line}) for line in task.splitlines()]

    async def work(state: Any) -> dict[str, Any]:
        run = trellis.current()
        assert run is not None
        result = await worker.run(state["item"], user=run.user)  # the run's child
        return {"found": [str(result.answer)]}

    def gather(state: Any) -> dict[str, Any]:
        return {"messages": [AIMessage("\n".join(sorted(state["found"])))]}

    builder = StateGraph(Fan)
    builder.add_node("work", work)
    builder.add_node("gather", gather)
    builder.add_conditional_edges(START, items, ["work"])
    builder.add_edge("work", "gather")
    builder.add_edge("gather", END)
    return builder.compile()


async def test_each_item_is_a_child_run_worked_at_once_within_the_admission_limit(
    harness: Harness,
) -> None:
    at_once: list[int] = []
    working = 0

    async def research(item: str, agent: Runtime) -> str:
        nonlocal working
        working += 1
        at_once.append(working)
        await asyncio.sleep(0)  # the others start beside it
        working -= 1
        return f"{item}: found"

    worker = harness.wrap(research, id="researcher")
    swarm = {
        "name": "fan_out",
        "description": "Researches many items at once: one a line.",
        "runnable": fan_out(worker),
    }
    model = ScriptedChat(
        [("task", {"description": "\n".join(ITEMS), "subagent_type": "fan_out"}), "all found"]
    )
    agent = harness.wrap(ReAct("You research.", model, subagents=[swarm]), id="lead")
    result = await agent.run("research them", user="u", thread="t")
    assert result.status is RunStatus.SUCCESS and result.answer == "all found"
    told = next(m for m in model.requests[1]["messages"] if m["role"] == "tool")
    assert told["content"] == "\n".join(f"{item}: found" for item in sorted(ITEMS))
    found = [r async for r in harness.runs.iterate(agent_id="researcher")]
    kids = [k for k in [await harness.runs.get(r.run_id) for r in found] if k is not None]
    assert sorted(str(k.input) for k in kids) == sorted(ITEMS)
    assert all(k.parent_run_id is not None and k.thread_id == "t" for k in kids)
    assert all(k.status is RunStatus.SUCCESS for k in kids)
    assert 1 < max(at_once) <= admission.ADMISSION.limit  # at once, never past the limit
