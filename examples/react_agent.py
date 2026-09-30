"""``ReAct``: no framework — a tool-calling loop over chat completions, with the answer
parsed into a pydantic model.

    .venv/bin/python examples/react_agent.py
"""

from __future__ import annotations

import asyncio

from _offline import react_model
from pydantic import BaseModel

from trellis import Harness, ReAct, tool


class Forecast(BaseModel):
    city: str
    celsius: float
    advice: str


@tool(side_effects="read")
def temperature(city: str) -> float:
    """Today's temperature in a city, in Celsius."""
    return {"Oslo": 4.0}.get(city, 20.0)


async def main() -> None:
    async with Harness() as h:
        model = react_model(
            [
                ("temperature", {"city": "Oslo"}),
                '{"city": "Oslo", "celsius": 4.0, "advice": "Take a coat."}',
            ]
        )
        target = ReAct(
            system="You give weather advice. Use the tool, then answer as JSON.",
            model=model,
            output=Forecast,
        )
        agent = h.wrap(target, id="weather", tools=[temperature])
        result = await agent.run("Do I need a coat in Oslo?", user="ada")
        print(result.status.value, result.answer)


if __name__ == "__main__":
    asyncio.run(main())
