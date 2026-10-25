"""Child runs, every framework: each framework's own way of handing work to another agent is a
child run of the run that hands it — its own run record naming the run, its memory context for
the task, its tool calls governed and recorded under its run, its tokens on the run's budget.

* ``ReAct``: ``subagents=[agent]``, a wrapped agent as a tool;
* a function: a wrapped agent's ``run`` called inside the run;
* LangGraph: the same, from a tool of the graph;
* Deep Agents: the ``task`` tool's sub-agent;
* OpenAI Agents SDK: an agent called as a tool (``Agent.as_tool``);
* Claude Agent SDK: a subagent of ``agents=`` (offline: the scripted CLI).

Every parent asks one specialist, ``stock``, how many units of SKU-1 are left; the run store
lists each parent's children (``h.runs.iterate(parent_run_id=...)``).

    python -m examples.06_scenarios.child_runs_every_framework
"""

from __future__ import annotations

import asyncio
from typing import Any

from agents import Agent, set_tracing_disabled
from claude_agent_sdk import AgentDefinition, ClaudeAgentOptions
from deepagents import create_deep_agent
from examples._support.offline import (
    claude_cli,
    langchain_model,
    offline_blocks,
    openai_agents_model,
    react_model,
)
from langchain.agents import create_agent
from langchain_core.tools import tool as langchain_tool

import trellis
from trellis import Agent as Wrapped
from trellis import Harness, ReAct, Runtime, tool

set_tracing_disabled(True)  # the SDK's own tracing goes to OpenAI; the harness traces via OTel
QUESTION = "How many units of SKU-1 are left?"


@tool(side_effects="read")
def units(sku: str) -> int:
    """Units of a SKU in stock.

    Args:
        sku: the SKU, like "SKU-1"
    """
    return {"SKU-1": 3}.get(sku, 0)


async def lineage(h: Harness, parent: Wrapped, input: Any) -> None:
    """Run ``parent`` and say which child runs its run made."""
    result = await parent.run(input, user="ada")
    children = [r async for r in h.runs.iterate(parent_run_id=result.run_id)]
    named = ", ".join(f"{c.agent_id} {c.status.value}" for c in children) or "none"
    print(f"{parent.id:>9}: {result.status.value} {result.answer!r} — children: {named}")
    assert children, f"{parent.id} made no child run"


async def main() -> None:
    async with Harness(**offline_blocks()) as h:
        # the specialist every parent asks: a wrapped ReAct with a tool of its own
        stock = h.wrap(
            ReAct(
                system_prompt="You answer stock questions.",
                model=react_model([("units", {"sku": "SKU-1"}), "3 units"] * 3),
                tools=[units],
            ),
            id="stock",
            description="Answers how many units of a SKU are in stock.",
        )

        # ReAct: a wrapped agent in subagents= is a tool whose call is a child run
        planner = ReAct(
            system_prompt="Ask stock about stock.",
            model=react_model([("stock", {"message": QUESTION}), "3 left"]),
            subagents=[stock],
        )
        await lineage(h, h.wrap(planner, id="react"), QUESTION)

        # a function: a wrapped agent's run() inside the run is a child run of it
        async def desk(question: str, agent: Runtime) -> str:
            known = agent.context  # what memory knows for the question (None: memory off)
            answer = await stock.run(question, user=agent.user)
            return f"stock says {answer.answer}" + (" (with memory's context)" if known else "")

        await lineage(h, h.wrap(desk, id="function"), QUESTION)

        # LangGraph: the same, from a tool of the graph (trellis.current() is the run)
        @langchain_tool
        async def ask_stock(question: str) -> str:
            """Ask the stock specialist a question about stock."""
            run = trellis.current()  # the run this tool is called in
            assert run is not None
            answer = await stock.run(question, user=run.user)
            return str(answer.answer)

        graph = create_agent(
            langchain_model([("ask_stock", {"question": QUESTION}), "3 left"]), tools=[ask_stock]
        )
        await lineage(h, h.wrap(graph, id="langgraph"), QUESTION)

        # Deep Agents: its task tool's sub-agent is a child run
        deep = create_deep_agent(
            model=langchain_model(
                [("task", {"description": QUESTION, "subagent_type": "counter"}), "3 left"]
            ),
            system_prompt="Hand stock questions to the counter.",
            subagents=[
                {
                    "name": "counter",
                    "description": "Counts the units of a SKU.",
                    "system_prompt": "Count the units.",
                    "model": langchain_model(["3 units"]),
                }
            ],
        )
        await lineage(h, h.wrap(deep, id="deepagent"), QUESTION)

        # OpenAI Agents SDK: an agent called as a tool is a child run
        counter = Agent(name="counter", model=openai_agents_model(["3 units"]))
        desk_agent = Agent(
            name="desk",
            instructions="Ask the counter about stock.",
            model=openai_agents_model([("count", {"input": QUESTION}), "3 left"]),
            tools=[counter.as_tool(tool_name="count", tool_description="Count units of a SKU.")],
        )
        await lineage(h, h.wrap(desk_agent, id="openai"), QUESTION)

        # Claude Agent SDK: a subagent of agents= is a child run
        script = [
            {"agent": "counter", "prompt": QUESTION, "steps": [{"text": "3 units"}]},
            {"text": "3 left"},
        ]
        options = ClaudeAgentOptions(
            system_prompt="Hand stock questions to the counter.",
            agents={"counter": AgentDefinition(description="Counts units.", prompt="Count.")},
            **claude_cli(script, "trellis"),
        )
        await lineage(h, h.wrap(options, id="claude"), QUESTION)


if __name__ == "__main__":
    asyncio.run(main())
