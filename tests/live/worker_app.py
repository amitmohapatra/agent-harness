"""The agents the live suite's worker processes serve, configured by the environment::

    python -m trellis.worker tests.live.worker_app:h

``TRELLIS_LIVE_SUFFIX`` keeps one session's agents apart from another's in a shared run store;
``TRELLIS_LIVE_LEDGER`` is a file each side effect of ``charge`` appends a line to, so a test
counts executions across processes.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from tests.live.conftest import live_harness
from trellis import Runtime, tool

SUFFIX = os.environ.get("TRELLIS_LIVE_SUFFIX", "local")
LEDGER = Path(os.environ.get("TRELLIS_LIVE_LEDGER", "/dev/null"))

h = live_harness()


#: named per session, so its statistics in the memory service count this session's calls only
CHARGE = f"charge_{SUFFIX}"


@tool(name=CHARGE, side_effects="write")
def charge(order: str, amount: int) -> str:
    """Charge an order (appends to the ledger: one line per real execution)."""
    with LEDGER.open("a") as ledger:
        ledger.write(f"{order}:{amount}:{os.getpid()}\n")
    return f"charged {order} {amount}"


async def billing(input: dict[str, Any], agent: Runtime) -> str:
    """Charge, then ask two people in turn: each answer may come to another worker."""
    receipt = await agent.tools.call(CHARGE, order=input["order"], amount=input["amount"])
    size = await agent.ask("Which size?", assignee="role:ops")
    ship = await agent.ask(f"Ship size {size}?", options=["yes", "no"], assignee="role:ops")
    return f"{receipt}; size {size}; ship {ship}"


async def briefing(input: str, agent: Runtime) -> str:
    return f"briefing for {agent.user}: {input}"


h.wrap(billing, id=f"live-billing-{SUFFIX}", tools=[charge], memory="read_write")
h.wrap(briefing, id=f"live-briefing-{SUFFIX}", memory="read_write")
