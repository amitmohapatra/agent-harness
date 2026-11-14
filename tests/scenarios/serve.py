"""A harness process of the scenarios' own: ``python -m tests.scenarios.serve`` with ``SCEN_SPEC``
(JSON) saying what it serves —

* ``agents``: ``[{"adapter": ..., "id": ..., "claude": [...]}]``, each built by
  :mod:`tests.scenarios.brains` (the same agents the test process wraps, by id);
* ``mode``: ``worker`` (``h.worker(agents)``: claims queued runs, its lease ``lease`` seconds),
  ``agui`` (every agent's ``serve_chat`` and ``serve_a2a`` on uvicorn at ``port``), or both;
* ``hold``: the seconds a run this process executes is held for (``holding.HOLD_SECONDS``, 60 in
  a deployment): short here, so a killed server's runs end within seconds, not a minute.

It writes ``ready`` to ``SCEN_READY`` once serving. Its traffic with the services is checked
into ``SCEN_TRAFFIC`` for the test's invariants (``tests/support/invariants.py``).
The identity is the ``x-user`` header (a deployment's edge sets ``x-trellis-identity``).
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import tempfile
from pathlib import Path

from tests.scenarios import brains, workflows  # noqa: F401 - the workflows' builders
from tests.scenarios.conftest import harness
from tests.scenarios.kit import TOOLS
from tests.support.arm import writes_unasked_here
from tests.support.invariants import record_in_process
from trellis.harness import holding


async def main() -> None:
    spec = json.loads(os.environ["SCEN_SPEC"])
    writes_unasked_here(os.environ)
    record_in_process()
    holding.HOLD_SECONDS = spec.get("hold", holding.HOLD_SECONDS)  # type: ignore[misc]
    tmp = Path(spec.get("tmp") or tempfile.mkdtemp())
    async with harness() as h:
        agents = []
        for a in spec["agents"]:
            agent, _ = await brains.wrapped(
                h, a["adapter"], TOOLS, tmp, id=a["id"], claude=a.get("claude")
            )
            agents.append(agent)
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(signal.SIGTERM, stop.set)
        tasks = []
        mode = spec.get("mode", "worker")
        if mode in ("worker", "both"):
            worker = h.worker(agents, concurrency=spec.get("concurrency", 2))
            worker.loop.lease_seconds = spec.get("lease", worker.loop.lease_seconds)
            tasks.append(asyncio.create_task(worker.run()))
        if mode in ("agui", "both"):
            import uvicorn
            from fastapi import FastAPI, Request

            def user_of(request: Request) -> str:
                return request.headers.get("x-user", "")

            app = FastAPI()
            port = spec["port"]
            for agent in agents:
                agent.serve_chat(app, path=f"/agui/{agent.id}", identity=user_of)
                agent.serve_a2a(app, f"http://127.0.0.1:{port}/a2a/{agent.id}", identity=user_of)
            server = uvicorn.Server(
                uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
            )
            tasks.append(asyncio.create_task(server.serve()))
            while not server.started:  # noqa: ASYNC110 - uvicorn says so by a flag
                await asyncio.sleep(0.05)
        ready = os.environ.get("SCEN_READY")
        if ready:
            Path(ready).write_text("ready")  # noqa: ASYNC240 - one small file, once
        await stop.wait()
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


if __name__ == "__main__":
    asyncio.run(main())
