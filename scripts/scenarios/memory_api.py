"""The scenario stack's memory API (scripts/scenarios/stack.sh): the API and its job worker
from the checkout on PYTHONPATH, in ONE process.

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
``memory-worker`` of their own. OpenFGA keeps its relationship tuples in a database of its own,
so they outlive a restart; the in-process model's dict did not, and every thread, document and
key made before a restart was then refused ("Access denied": a test that stops and starts the
memory service). Its tuples are kept in ``authz.jsonl`` in the service's working folder
(:func:`durable`), and read back at start — as durable as the rows they guard. That earlier
script also ran jobs inline (``tasks="inline"``); this one does not: the inline queue runs a
delayed job at once and never runs a periodic one, and the harness live suite waits on both.
"""

import asyncio
import contextlib
import json
import os
import signal
from pathlib import Path

from memory_service.api import app
from memory_service.application.container import Overrides, worker_pools_for
from memory_service.config.constants import OVERLOAD, TASKS
from memory_service.observability.logging import get_logger
from memory_service.ports.authorization import RelationTuple

log = get_logger("scen")
TUNED = Overrides(search_local_path=os.environ["MH_QDRANT_PATH"], authorization="memory")
build = app.build_container
#: the authorization model's tuples, one write a line (in the working folder: a reset clears it)
TUPLES = Path("authz.jsonl")


async def durable(provider, path=TUPLES):
    """The in-process authorization model, its writes kept in ``path`` and read back first:
    what OpenFGA's own database does for its tuples."""
    write = provider.write
    if path.exists():
        for line in path.read_text().splitlines():
            done = json.loads(line)
            await write(
                [RelationTuple(**t) for t in done["add"]],
                [RelationTuple(**t) for t in done["delete"]],
            )
    log.info("scen.authorization_read_back", path=str(path.resolve()))

    async def kept(add, delete=()):
        await write(add, delete)  # refused (a tuple the model has no place for): not kept
        said = {"add": [t.model_dump() for t in add], "delete": [t.model_dump() for t in delete]}
        with path.open("a") as out:
            out.write(json.dumps(said) + "\n")

    provider.write = kept


async def build_container(settings, version, *, overrides=None, **kw):
    # role="combined": one process serves the API and runs the worker's pools, so its
    # connection pools are sized for both
    container = await build(
        settings, version, overrides=overrides or TUNED, **{**kw, "role": "combined"}
    )
    await durable(container.authorization)
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
