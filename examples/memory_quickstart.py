"""The README quick start, against a real Memory Service — including the onboarding it needs.

What it demonstrates, in the order it happens:

1. **onboarding**, the way ``tests/support.py::onboard`` does it: a tenant, a workspace and a
   member. A WORKSPACE-visible write needs all three, and skipping this is the most common
   reason a first script gets a 2xx and then finds nothing to read back;
2. **one wrapped agent** with the Memory Service attached, so the turn gets a context bundle
   before it runs and its input, output and claims are written back after it;
3. **explicit memory calls** from inside the agent — ``remember`` with a visibility the run's
   identity can actually express, then ``recall``;
4. **draining**, because writes are queued: a process that exits without ``aclose()`` loses
   them, and the service's 202 means "durably queued", not "already retrievable".

    MEMORY_SERVICE_URL=http://localhost:8080 MEMORY_API_KEY=dev-key \\
        python examples/memory_quickstart.py

Both variables have those values as defaults, so with the dev service running this is just
``python examples/memory_quickstart.py``. Without a reachable service it says so and exits 0 —
an example that cannot reach its dependency is not a failing test.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import uuid
from typing import Any

from trellis.harness import AgentExecutionContext, AgentHarness, AgentRuntime

URL = os.environ.get("MEMORY_SERVICE_URL", "http://localhost:8080")
KEY = os.environ.get("MEMORY_API_KEY", "dev-key")
TENANT = os.environ.get("MEMORY_TENANT", "acme")
WORKSPACE = "quickstart-ws"
USER = "u1"


async def onboard(client: Any) -> None:
    """Create the tenant, the workspace and the membership, tolerating that they exist.

    Exactly what the harness's own test support does. ``ConflictError`` means somebody got
    here first, which is the normal case on a second run.
    """
    from trellis.memory.errors import ConflictError  # noqa: PLC0415 - the live path only

    with contextlib.suppress(ConflictError):
        await client.admin.create_tenant(TENANT, tenant_id=TENANT)
    workspaces = client.administer(TENANT).workspaces
    with contextlib.suppress(ConflictError):
        await workspaces.create(WORKSPACE, workspace_id=WORKSPACE)
    await workspaces.set_member(WORKSPACE, f"user:{USER}")


async def inventory_agent(question: str, agent: AgentRuntime) -> dict[str, Any]:
    """The agent. Everything it knows about memory is that ``agent.memory`` exists."""
    bundle = agent.memory_context  # fetched before the run by the memory interceptor
    await agent.memory.remember(
        "SKU-1 is reordered in packs of 24.",
        memory_type="PROCEDURAL",
        visibility="USER",  # USER needs user_id on the context, and it is there
    )
    recalled = await agent.memory.recall(question)
    return {
        "question": question,
        "bundle_tokens": getattr(bundle, "token_estimate", None),
        "bundle_status": getattr(getattr(bundle, "evidence", None), "status", None),
        "recalled": len(recalled),
    }


async def main() -> None:
    try:
        from trellis.memory import MemoryClient  # noqa: PLC0415 - optional at import time
    except ImportError:  # pragma: no cover - the SDK is a core dependency
        print("the trellis-memory SDK is not installed")
        return

    client = MemoryClient(URL, api_key=KEY)
    try:
        health = await client.health()
    except Exception as exc:  # any transport failure means "not running"
        print(f"no Memory Service at {URL} ({type(exc).__name__}): nothing to demonstrate")
        await client.aclose()
        return

    await onboard(client)
    harness = AgentHarness(
        memory=client,
        defaults={"tenant_id": TENANT, "user_id": USER, "workspace_id": WORKSPACE},
    )
    wrapped = harness.wrap(inventory_agent, agent_id="inventory-agent", skills=["inventory.ask"])

    # Thread, session and turn are the application's own ids; the service creates them on
    # first use. A turn id belongs to one session, so a run derives its own.
    thread = await client.bind(tenant_id=TENANT, user_id=USER).chat.create(title="quick start")
    context = AgentExecutionContext.create(
        tenant_id=TENANT,
        agent_id="inventory-agent",
        workspace_id=WORKSPACE,
        user_id=USER,
        thread_id=thread.thread_id,
        turn_id=f"t_{uuid.uuid4().hex[:8]}",
    )

    result = await wrapped("how are packs of SKU-1 reordered?", context=context)

    print("\n=== memory quick start ===")
    print(f"service      : {URL} ({health.get('status')})")
    print(f"tenant       : {TENANT} / workspace {WORKSPACE} / user {USER}")
    print(f"thread       : {thread.thread_id}")
    print(f"status       : {result.status}")
    for key, value in (result.data or {}).items():
        print(f"{key:<13}: {value}")
    if result.warnings:
        print(f"warnings     : {[w.code for w in result.warnings]}")

    written = await harness.drain()  # the queued observations, awaited on purpose
    print(f"drained      : {written} queued write(s)")
    await harness.aclose()
    await client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
