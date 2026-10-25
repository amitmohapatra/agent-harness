"""Stored prompts: the gateway's Prompt Repository is where a team keeps its prompts — versioned,
changed without a redeploy — and ``stored(...)`` reads one.

* ``ReAct(system_prompt=stored("triage", team="EU"))``: the prompt is the agent's
  instructions, its ``{{variables}}`` filled; each run pins the version it starts with (a
  resumed run reads the same one, even after a newer was committed) and says which (a
  ``prompt`` event, the ``trellis.prompt.*`` span attributes);
* ``await h.prompt("triage@1", team="EU")`` — any framework: the text (``.text``) and the chat
  messages (``.messages``) to build an agent of your own with; inside a run it is pinned too.

Offline the gateway is scripted (two committed versions of ``triage``); with ``BIFROST_URL``
commit a prompt named ``triage`` in Bifrost first.

    python -m examples.05_features.stored_prompts
"""

from __future__ import annotations

import asyncio
from typing import Any

from examples._support.gateway import ScriptedGateway
from examples._support.offline import online, react_model

from trellis import Harness, ReAct, stored
from trellis.contracts import RunEventType

TRIAGE = [
    [{"role": "system", "content": "You triage tickets for the {{team}} team."}],
    [{"role": "system", "content": "You triage tickets for the {{team}} team. Be brief."}],
]


async def main() -> None:
    blocks: dict[str, Any] = {}
    if not online():
        blocks["gateway"] = ScriptedGateway({}, prompts={"triage": TRIAGE}).gateway()
    async with Harness(**blocks) as h:
        first = await h.prompt("triage@1", team="EU")
        print("version 1:", first.text)
        print("the latest, as messages:", (await h.prompt("triage", team="EU")).messages)

        model = react_model(["Ticket T-9: billing, high priority."])
        agent = h.wrap(ReAct(system_prompt=stored("triage", team="EU"), model=model), id="triage")
        async for event in agent.stream("Triage T-9: I was charged twice.", user="ada"):
            if event.type is RunEventType.CUSTOM and event.data.get("name") == "prompt":
                print("the run pinned:", event.data["prompt"], "version", event.data["version"])
            if event.type is RunEventType.RUN_FINISHED:
                print(event.outcome, event.data.get("result"))


if __name__ == "__main__":
    asyncio.run(main())
