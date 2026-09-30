"""Co-work: a queued run that asks a person, waits in their inbox, and continues when they
answer — executed by a worker, not by the caller.

    .venv/bin/python examples/cowork.py                 # runs kept in process
    RUNS_URL=... RUNS_API_KEY=... python examples/cowork.py   # runs in agent-runs
"""

from __future__ import annotations

import asyncio
from typing import Any

from trellis import Harness, Runtime


async def draft_contract(input: dict[str, Any], agent: Runtime) -> str:
    draft = f"Contract with {input['customer']}: {input['terms']}"
    verdict = await agent.ask(
        f"Review the draft for {input['customer']}",
        ui="diff",
        expects={"type": "string"},
        assignee="role:legal",
    )
    return f"{draft} — legal: {verdict}"


async def main() -> None:
    async with Harness() as h:
        agent = h.wrap(draft_contract, id="contracts")
        worker = h.worker([agent])

        handle = await agent.start({"customer": "ACME", "terms": "net 30"}, user="ada")
        await worker.run_once()  # a worker claims the queued run; it pauses on the review
        paused = await handle.result(timeout=10)
        print("paused:", paused.status.value)

        [waiting] = await h.runs.list_paused(h.settings.tenant, assignee="role:legal")
        assert waiting.awaiting is not None
        print("legal inbox:", waiting.awaiting.question)
        resumed = await agent.resume(
            waiting.awaiting.interrupt_id, "answer", answer="approved", reviewer="lee"
        )
        print("after the answer:", resumed.status.value)  # QUEUED: a worker continues it

        await worker.run_once()
        done = await handle.result(timeout=10)
        print(done.status.value, done.answer)


if __name__ == "__main__":
    asyncio.run(main())
