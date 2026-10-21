"""Invariant I2: a short task — one tool, fewer than 3 steps, fewer than 20 tools — sends the
model exactly what ``ReAct`` sent before the automatic features (planning, files, the ``task``
helper, tool search, ...): the same tools, the same instructions, the same messages, the same
number of calls. The requests are compared with a snapshot taken from that ``ReAct``
(``golden/react_short.json``); a change that must change them updates the snapshot on purpose
(``TRELLIS_GOLDEN_UPDATE=1``)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from tests.support.models import ScriptedChat
from trellis import Harness, ReAct, tool

GOLDEN = Path(__file__).parent / "golden" / "react_short.json"


@tool(side_effects="read")
def stock(sku: str) -> int:
    """Units of a SKU in stock.

    Args:
        sku: The SKU, such as a-1.
    """
    return 7


def _other(n: int) -> Any:
    def report(store: str) -> str:
        """A report of one store.

        Args:
            store: The store's id.
        """
        return "nothing unusual"

    return tool(report, name=f"report_{n}", description=f"Report number {n} of one store.")


TOOLS = [stock, *(_other(n) for n in range(4))]


def wire(requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """What the model was sent, without what changes from run to run (none, today: the
    scripted model's call ids are numbered)."""
    return requests


async def test_a_short_task_sends_the_model_what_react_always_sent(harness: Harness) -> None:
    model = ScriptedChat([("stock", {"sku": "a-1"}), "7 units"])
    agent = harness.wrap(
        ReAct(system="You answer stock questions.", model=model), id="golden", tools=TOOLS
    )
    result = await agent.run("How many a-1 are in stock?", user="u1")
    assert result.answer == "7 units"
    sent = wire(model.requests)
    if os.environ.get("TRELLIS_GOLDEN_UPDATE"):
        GOLDEN.write_text(json.dumps(sent, indent=2, sort_keys=True) + "\n")
    assert sent == json.loads(GOLDEN.read_text())
    assert len(sent) == 2
