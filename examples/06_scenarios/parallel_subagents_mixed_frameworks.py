"""Scenario: a ``ReAct`` planner delegating to three sub-agents at once, each on another
framework — a LangGraph graph, an OpenAI Agents ``Agent`` and a plain function that asks a person.

* each ``agent.as_tool()`` is one tool of the planner; each call is a child run of it: its own
  run record, the planner's tenant, user, thread, time limit and trace;
* the three calls of one model step run at once (they only read);
* the function asks the traveller which budget: the question pauses the planner (the other two
  children finish first, journaled), it is answered with the planner's ``resume``, and it goes
  back to the child that asked — the finished children are not run again.

    python -m examples.06_scenarios.parallel_subagents_mixed_frameworks
"""

from __future__ import annotations

import asyncio

from agents import Agent as OpenAIAgent
from agents import set_tracing_disabled
from examples._support.offline import langchain_model, openai_agents_model, react_model
from langchain.agents import create_agent

from trellis import Harness, ReAct, Runtime, tool

set_tracing_disabled(True)


@tool(side_effects="read")
def flights(city: str) -> str:
    """The direct flights to a city."""
    return f"{city}: 2 direct flights a day, from 89 EUR"


@tool(side_effects="read")
def hotels(city: str, budget: str) -> str:
    """Hotels in a city, within a budget (low or high)."""
    return f"{city}: Hotel Nord, {'79' if budget == 'low' else '240'} EUR a night"


async def booker(city: str, agent: Runtime) -> str:
    """A function: the traveller chooses the budget, then a hotel is found."""
    budget = await agent.ask("Which budget?", options=["low", "high"])
    return str(await agent.tools.call("hotels", city=city, budget=budget))


async def main() -> None:
    async with Harness() as h:
        scout_graph = create_agent(
            langchain_model([("flights", {"city": "Oslo"}), "Oslo: 2 a day, from 89 EUR."]),
            tools=await h.tools(flights, framework="langgraph"),
            system_prompt="You find flights.",
        )
        scout = h.wrap(scout_graph, id="scout")  # LangGraph
        visas = h.wrap(
            OpenAIAgent(
                name="visas",
                instructions="You know entry rules.",
                model=openai_agents_model(["Norway: no visa for EU citizens."]),
            ),
            id="visas",
        )  # OpenAI Agents
        hotel = h.wrap(booker, id="booker", tools=[hotels])  # a function
        planner = h.wrap(
            ReAct(
                system="You plan trips: ask scout, visas and booker at once, then answer.",
                model=react_model(
                    [
                        [
                            ("scout", {"message": "Oslo"}),
                            ("visas", {"message": "Norway"}),
                            ("booker", {"message": "Oslo"}),
                        ],
                        "Flights from 89 EUR; no visa; Hotel Nord at 79 EUR a night.",
                    ]
                ),
            ),
            id="planner",
            tools=[s.as_tool(side_effects="read") for s in (scout, visas, hotel)],
        )

        result = await planner.run("Plan a weekend in Oslo", user="ada")
        while result.interrupt is not None:  # the booker asks, through the planner
            asker = (result.interrupt.payload or {})["subagent"]["agent_id"]
            print(f"{asker} asks: {result.interrupt.question}")
            result = await planner.resume(
                result.interrupt.interrupt_id, "answer", answer="low", reviewer="ada"
            )
        print(result.status.value, result.answer)
        children = [r async for r in h.runs.iterate(parent_run_id=result.run_id)]
        print("child runs:", sorted((c.agent_id, c.status.value) for c in children))


if __name__ == "__main__":
    asyncio.run(main())
