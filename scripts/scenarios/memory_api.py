"""The scenario stack's memory API (scripts/scenarios/stack.sh): the API and its job worker from the checkout on PYTHONPATH, in ONE process.

What is real: the service's own entry point (``run_api``), PostgreSQL, Redis, the filesystem
blob store, the frozen embedding / NLI models under ./models, and the job queue: Procrastinate
over PostgreSQL, run by the worker's own pools (``worker_pools_for`` / ``run_pools``, what
``memory_service.worker`` runs), so delayed jobs, retries and the periodic jobs behave as
deployed.

What is a stand-in, because this machine has no Qdrant and no OpenFGA server (and no Docker):
``Overrides(search_local_path=..., authorization="memory")``: qdrant-client local mode on disk
and the in-process authorization model, as /var/tmp/mx/stack/mem_api.py ran it. Both live in
one process only (the local Qdrant folder is locked by its client; the authorization model is
a dict), which is why the worker's pools run inside the API process here instead of in a
``memory-worker`` of their own. That earlier script also ran jobs inline
(``tasks="inline"``); this one does not: the inline queue runs a delayed job at once and
never runs a periodic one, and the harness live suite waits on both.
"""

import asyncio
import contextlib
import os
import signal

import memory_service.api.app as app
from memory_service.application.container import Overrides, worker_pools_for
from memory_service.config.constants import OVERLOAD, TASKS
from memory_service.observability.logging import get_logger

log = get_logger("scen")
TUNED = Overrides(search_local_path=os.environ["MH_QDRANT_PATH"], authorization="memory")
build = app.build_container


async def build_container(settings, version, *, overrides=None, **kw):
    # role="worker": the connection pools are sized for the worker's pools as well
    container = await build(
        settings, version, overrides=overrides or TUNED, **{**kw, "role": "worker"}
    )
    pools = worker_pools_for(settings)
    worker = asyncio.create_task(
        container.tasks.run_pools(
            pools,
            install_signal_handlers=False,
            shutdown_grace_seconds=OVERLOAD.worker_shutdown_grace_seconds,
            db_outage_seconds=TASKS.worker_db_outage_seconds,
        ),
        name="scen-worker",
    )
    log.info(
        "scen.worker_started",
        pools={",".join(q.value for q in queues): slots for queues, slots in pools},
    )
    closing = False

    def stopped(task: asyncio.Task) -> None:
        if closing or task.cancelled():
            return
        # the job worker ended on its own: stop the service rather than serve without jobs
        log.error("scen.worker_stopped", error=repr(task.exception()))
        os.kill(os.getpid(), signal.SIGTERM)

    worker.add_done_callback(stopped)
    close = container.close

    async def close_with_worker() -> None:
        nonlocal closing
        closing = True
        worker.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await worker
        await close()

    container.close = close_with_worker
    return container


app.build_container = build_container
from memory_service.__main__ import run_api  # noqa: E402

run_api()
