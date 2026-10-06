"""Memory beyond the automatic push and pull: a document a user's runs cite, the agent's own
reads in a run (``agent.memory``), and a person's verdict on a run (``h.feedback``).

    python -m examples.02_way1_function.memory_documents_feedback

Offline the memory service is a scripted one in this process (it cites the document it was
given and keeps a person's verdict pending, as the service does); with ``MEMORY_URL`` and
``TRELLIS_API_KEY`` set it is the real one. Feedback is also a score on the run's trace
(Langfuse, when the OTLP settings reach it).
"""

from __future__ import annotations

import asyncio

from examples._support.offline import offline_blocks

from trellis import Harness, Runtime

POLICY = b"Returns policy: damaged pallets are refunded within 14 days of delivery."


async def support(question: str, agent: Runtime) -> str:
    if agent.context is None:  # memory off, or nothing remembered for this question
        return "I don't know the returns policy."
    found = await agent.memory.search("damaged pallets refund")  # the agent's own read
    return f"From memory ({len(found)} hits): {agent.context.splitlines()[-1]}"


async def main() -> None:
    async with Harness(**offline_blocks()) as h:
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
