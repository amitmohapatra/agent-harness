"""The thread-stream load benchmark: ``python -m benchmark.load.threads`` — ``--streams`` open
chat-thread streams (the body of ``GET {path}/threads/{id}/events``, one poller per process)
while ``--schedules`` chat schedules fire into as many of those threads, on agent-runs
(``RUNS_URL``, ``TRELLIS_API_KEY``) with a fake model (a function target that answers at once).

It reports this process's CPU seconds and peak resident memory, agent-runs' database statements
a second (``pg_stat_statements`` on ``--dsn``) with the streams open and idle and while the
schedules fire, and the delivery latency from a fire to its reply on its thread's stream (p50,
p95, max). A private stack: agent-runs on its own database, its key registry a stub.

    RUNS_URL=http://127.0.0.1:55442 TRELLIS_API_KEY=k \\
      python -m benchmark.load.threads --dsn postgresql://memory@localhost:55441/runs_load
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import time
import uuid
from typing import Any

from benchmark import procstat
from trellis import Harness, Runtime, Settings
from trellis.contracts import ScheduleSpec
from trellis.harness.agui.threads import Threads
from trellis.harness.runs import THREAD
from trellis.harness.schedules import TASK


async def _answer(input: str, agent: Runtime) -> str:
    return f"done: {input[-12:]}"


def _statements(dsn: str) -> int:
    """Statements the database has run (``psql``: the harness has no PostgreSQL driver)."""
    query = "SELECT coalesce(sum(calls), 0) FROM pg_stat_statements"
    said = subprocess.run(["psql", dsn, "-tAc", query], capture_output=True, text=True, check=True)
    return int(said.stdout.strip())


async def _follow(
    stream: Any, thread: str, delivered: dict[str, float], wanted: int, all_in: asyncio.Event
) -> None:
    async for chunk in stream:
        if "RUN_FINISHED" in chunk and thread not in delivered:
            delivered[thread] = time.perf_counter()
            if len(delivered) >= wanted:
                all_in.set()


async def _work(h: Harness, agent: Any, until: asyncio.Event) -> None:
    worker = h.worker([agent])
    while not until.is_set():
        if not await worker.run_once():
            await asyncio.sleep(0.05)


async def main(streams: int, schedules: int, idle: float, workers: int, dsn: str) -> dict:
    run = uuid.uuid4().hex[:6]
    async with Harness(config=Settings.from_env()) as h:
        agent = h.wrap(_answer, id=f"load-{run}", without=["memory"])
        tenant = await h._tenant(None)
        threads = Threads(agent)
        ids = []
        for i in range(schedules):
            spec = ScheduleSpec(
                tenant_id=tenant, agent_id=agent.id, name=f"s{i}", cadence="manual",
                on_behalf_of=f"u{run}{i}", input=f"Scheduled task: report {i}",
                metadata={THREAD: f"t{run}{i}", TASK: {"task": f"report {i}", "once": False}},
            )  # fmt: skip
            ids.append((await h.runs.schedules.create(spec)).schedule_id)
        delivered: dict[str, float] = {}
        all_in = asyncio.Event()
        cpu0 = procstat.cpu_seconds()
        followers = []
        for i in range(streams):
            stream = threads.stream(tenant, f"u{run}{i}", f"t{run}{i}", None)
            follow = _follow(stream, f"t{run}{i}", delivered, schedules, all_in)
            followers.append(asyncio.create_task(follow))
        await asyncio.sleep(2)  # connected, replayed
        s0, t0 = _statements(dsn), time.perf_counter()
        await asyncio.sleep(idle)
        idle_rate = (_statements(dsn) - s0) / (time.perf_counter() - t0)
        cpu_idle = (procstat.cpu_seconds() or 0) - (cpu0 or 0)
        done = asyncio.Event()
        pool = [asyncio.create_task(_work(h, agent, done)) for _ in range(workers)]
        s1, t1 = _statements(dsn), time.perf_counter()
        fired = {}
        for i, sid in enumerate(ids):
            fired[f"t{run}{i}"] = time.perf_counter()
            await h.runs.schedules.fire(sid)
        async with asyncio.timeout(120):
            await all_in.wait()
        firing_rate = (_statements(dsn) - s1) / (time.perf_counter() - t1)
        done.set()
        await asyncio.gather(*pool)
        for task in followers:
            task.cancel()
        await asyncio.gather(*followers, return_exceptions=True)
        latencies = [delivered[t] - fired[t] for t in fired]
        return {
            "streams": streams,
            "schedules": schedules,
            "idle_seconds": idle,
            "cpu_seconds_idle": round(cpu_idle, 2),
            "cpu_seconds_total": round((procstat.cpu_seconds() or 0) - (cpu0 or 0), 2),
            "peak_rss_mb": procstat.peak_rss_mb(),
            "db_statements_per_s_idle": round(idle_rate, 1),
            "db_statements_per_s_firing": round(firing_rate, 1),
            "delivery_p50_s": round(procstat.percentile(latencies, 50) or 0, 3),
            "delivery_p95_s": round(procstat.percentile(latencies, 95) or 0, 3),
            "delivery_max_s": round(max(latencies), 3),
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="The thread-stream load benchmark.")
    parser.add_argument("--streams", type=int, default=1000)
    parser.add_argument("--schedules", type=int, default=100)
    parser.add_argument("--idle", type=float, default=10.0, help="seconds open and idle")
    parser.add_argument("--workers", type=int, default=10, help="worker loops in process")
    parser.add_argument("--dsn", required=True, help="agent-runs' database (pg_stat_statements)")
    args = parser.parse_args()
    report = asyncio.run(main(args.streams, args.schedules, args.idle, args.workers, args.dsn))
    sys.stdout.write(json.dumps(report, indent=2) + "\n")
