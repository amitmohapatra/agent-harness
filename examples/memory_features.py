"""Memory beyond the automatic push and pull: a document a user's runs cite, the agent's own
reads in a run (``agent.memory``), and a person's verdict on a run (``h.feedback``).

    MEMORY_URL=... TRELLIS_API_KEY=... python examples/memory_features.py   # the full tour
    .venv/bin/python examples/memory_features.py                            # memory off

With memory off there is no document memory and no memory context; feedback is then a score on
the run's trace only (Langfuse, when the OTLP settings reach it).
"""

from __future__ import annotations

import asyncio

from trellis import Harness, Runtime

POLICY = b"Returns policy: damaged pallets are refunded within 14 days of delivery."


async def support(question: str, agent: Runtime) -> str:
    if agent.context is None:  # memory off, or nothing remembered for this question
        return "I don't know the returns policy."
    found = await agent.memory.search("damaged pallets refund")  # the agent's own read
    return f"From memory ({len(found)} hits): {agent.context.splitlines()[-1]}"


async def main() -> None:
    async with Harness() as h:
        agent = h.wrap(support, id="support")
        if h.memory is not None:
            info = await h.add_document(
                ("returns.txt", POLICY, "text/plain"), user="ada", title="Returns"
            )
            print("document", info.title, info.status)  # indexed: the next context cites it
        result = await agent.run("How fast are damaged pallets refunded?", user="ada")
        print(result.status.value, result.answer)

        stored = await h.feedback(result.run_id, "confirm")
        if stored is None:
            print("feedback: a score on the run's trace (memory off)")
        else:  # a person's verdict waits for the tenant administrator before it counts
            print("feedback:", stored.verdict, stored.review.state if stored.review else "")


if __name__ == "__main__":
    asyncio.run(main())
