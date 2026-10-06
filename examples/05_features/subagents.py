"""Sub-agents: ``agent.as_tool()`` — a ``ReAct`` planner delegating to two agents at once, one
of which asks a person before it answers; the answer reaches it through the planner's resume.

    python -m examples.05_features.subagents
"""

from __future__ import annotations

import asyncio

from examples._support.offline import react_model

from trellis import Harness, ReAct, Runtime, tool


@tool(side_effects="read")
def flights(city: str) -> str:
    """The direct flights to a city."""
    return f"{city}: 2 direct flights a day, from 89 EUR"


@tool(side_effects="read")
def hotels(city: str, budget: str) -> str:
    """Hotels in a city, within a budget (low or high)."""
    return f"{city}: Hotel Nord, {'79' if budget == 'low' else '240'} EUR a night"


async def booker(input: str, agent: Runtime) -> str:
    """Find a hotel in a city, once the traveller has chosen a budget."""
    budget = await agent.ask("Which budget?", options=["low", "high"])
    return str(await agent.tools.call("hotels", city=input, budget=budget))


async def main() -> None:
    async with Harness() as h:
        scout = h.wrap(
            ReAct(
                system="You find flights. Use the tool, then answer in one line.",
                model=react_model([("flights", {"city": "Oslo"}), "Oslo: 2 a day, from 89 EUR"]),
            ),
            id="scout",
            tools=[flights],
        )
        hotel = h.wrap(booker, id="booker", tools=[hotels])
        planner = h.wrap(
            ReAct(
                system="You plan trips. Ask scout for flights and booker for a hotel, at once, "
                "then answer in two lines.",
                model=react_model(
                    [
                        [("scout", {"message": "Oslo"}), ("booker", {"message": "Oslo"})],
                        "Flights: 2 a day from 89 EUR.\nHotel: Hotel Nord, 79 EUR a night.",
                    ]
                ),
            ),
            id="planner",
            tools=[scout.as_tool(), hotel.as_tool()],  # both only read: they run at once
        )
        result = await planner.run("Plan a weekend in Oslo", user="ada")
        while result.interrupt is not None:  # the booker asks, through the planner
            asker = (result.interrupt.payload or {})["subagent"]["agent_id"]
            print(f"{asker} asks: {result.interrupt.question}")
            result = await planner.resume(
                result.interrupt.interrupt_id, "answer", answer="low", reviewer="ada"
            )
        print(result.status.value, result.answer)


if __name__ == "__main__":
    asyncio.run(main())
