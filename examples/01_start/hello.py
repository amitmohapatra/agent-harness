"""Level 0: wrap an agent, run it. Three lines of Trellis around a plain async function.

    python -m examples.01_start.hello

With nothing configured everything runs in this process: no memory, runs kept in memory, the
tenant ``default``. Point it at the services and run it again — nothing else changes:

    export MEMORY_URL=http://localhost:8080 TRELLIS_API_KEY=dev-key   # agent-memory-service
    python -m examples.01_start.hello                                 # now it remembers
"""

from __future__ import annotations

import asyncio

from trellis import Harness, Runtime


async def answer(question: str, agent: Runtime) -> str:
    """Your agent: any async function of the input and the run (here it only echoes what
    memory said about the user)."""
    return f"{agent.context or 'Nothing remembered yet.'} (asked: {question})"


async def main() -> None:
    async with Harness() as h:  # the deployment is the environment
        agent = h.wrap(answer, id="hello")
        result = await agent.run("How do I like to be contacted?", user="ada")
        print(result.status.value, result.answer)


if __name__ == "__main__":
    asyncio.run(main())
