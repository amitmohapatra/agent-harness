"""``ReAct``: no framework of your own — LangChain's ``create_agent`` loop with the native
context middleware and the harness's on top, its answer a pydantic model (structured output:
offline the scripted model answers through the output's own tool).

    python -m examples.02_way1_react.structured_output
"""

from __future__ import annotations

import asyncio

from examples._support.offline import react_model
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
                ("Forecast", {"city": "Oslo", "celsius": 4.0, "advice": "Take a coat."}),
            ]
        )
        target = ReAct(
            system="You give weather advice. Use the tool, then answer.",
            model=model,
            output=Forecast,
        )
        agent = h.wrap(target, id="weather", tools=[temperature])
        result = await agent.run("Do I need a coat in Oslo?", user="ada")
        print(result.status.value, result.answer)


if __name__ == "__main__":
    asyncio.run(main())
