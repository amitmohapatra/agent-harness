"""Way 2, the memory block alone: ``trellis.memory`` around code of your own — no harness, any
framework (here a plain function stands in for your agent's model call).

* ``bind`` the scope (tenant, user), then ``agent(...)`` for one run of one agent;
* ``context(question)`` — what the service knows that bears on the question, rendered for a
  prompt, with the ``bundle_id`` a later check cites;
* ``remember`` a fact, ``search`` for it; ``record_tool`` each tool call; ``history.add`` the turn;
* ``feedback`` — how the run went (the run's own outcome, ``source="system"``).

Offline the service is a scripted one in this process; with ``MEMORY_URL`` and
``TRELLIS_API_KEY`` set it is the real one (``MemoryClient()`` reads both).

    python -m examples.03_way2_memory.memory_block
"""

from __future__ import annotations

import asyncio

from examples._support.memory import ScriptedMemory
from examples._support.offline import memory_client

from trellis.contracts import new_id

TENANT = "default"  # the tenant your key speaks for (a development key's: default)


async def my_agent(prompt: str) -> str:
    """Your agent, in any framework: it reads the context in its prompt."""
    return "Email Ada the shipping update." if "email" in prompt else "Call Ada."


async def main() -> None:
    memory = memory_client(ScriptedMemory(context="Ada prefers email."))
    run_id = new_id("run_")
    scope = memory.bind(tenant_id=TENANT, user_id="ada").agent("notifier", agent_run_id=run_id)
    async with scope:  # the scope the calls below are made in
        question = "How should I tell Ada her order shipped?"
        context = await scope.context(question)
        print("context:", context.rendered, f"(bundle {context.bundle_id})")

        await scope.remember("Ada's order o-7 shipped on Monday.")
        print("found:", [item.text for item in await scope.search("order o-7")])

        answer = await my_agent(f"{context.rendered}\n\n{question}")
        await scope.record_tool("send_email", {"to": "ada"}, output="sent")
        await scope.history.add([("USER", question), ("ASSISTANT", answer)])
        stored = await scope.feedback("run", run_id, "confirm", source="system")
        print(answer, "| feedback:", stored.verdict)
    await memory.aclose()


if __name__ == "__main__":
    asyncio.run(main())
