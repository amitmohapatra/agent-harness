"""The load benchmark on the production path (``--queued``): runs queued in agent-runs
(``RUNS_URL``) by this process — ``agent.start``, then the run's result — and executed by
worker processes (``h.worker([agent]).run()``) that claim them, as a deployment runs; with
the memory service (``MEMORY_URL``) as configured, or its memory parts off (``--no-memory``:
the service is still the key registry agent-runs asks).

It keeps ``concurrency`` runs in flight for ``seconds``, sampling every ``sample_seconds`` the
resident memory of each worker process and of each ``--watch`` process (the memory API, its
worker, agent-runs, its ticker), and reports each one's slope after the warm-up (the first
fifth), in MB per hour — a process that leaks shows a slope that does not flatten."""

from __future__ import annotations

import asyncio
import multiprocessing
import os
import statistics
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from benchmark import procstat
from benchmark.load.run import TASK, Plan, agent


@dataclass(frozen=True)
class Served:
    """What one worker process is asked to do."""

    plan: Plan
    seconds: float
    concurrency: int
    memory: bool


@dataclass
class Samples:
    #: (seconds since the start, resident MB)
    rss: list[tuple[float, float | None]] = field(default_factory=list)
    cpu_seconds: float | None = None


def harness() -> Any:
    """The environment's harness (``RUNS_URL``, ``MEMORY_URL``, ``TRELLIS_API_KEY``), tracing and
    the gateway's MCP tools off."""
    from trellis import Harness, Settings

    config = Settings.from_env().model_copy(update={"otlp_endpoint": None})
    return Harness(config=config, gateway=False)


def wrapped(h: Any, plan: Plan, memory: bool) -> Any:
    """The agent under load; without memory, the service is only the key registry."""
    return agent(h, plan, without=frozenset() if memory else frozenset({"memory"}))


async def _serve(served: Served) -> Samples:
    samples = Samples()
    began = time.monotonic()
    cpu = procstat.cpu_seconds()
    async with harness() as h:
        worker = h.worker([wrapped(h, served.plan, served.memory)], concurrency=served.concurrency)
        running = asyncio.create_task(worker.run())
        while time.monotonic() - began < served.seconds:
            samples.rss.append((time.monotonic() - began, procstat.rss_mb()))
            await asyncio.sleep(served.plan.sample_seconds)
        running.cancel()
        await asyncio.gather(running, return_exceptions=True)
    used = procstat.cpu_seconds()
    samples.cpu_seconds = None if used is None or cpu is None else used - cpu
    return samples


def serve(served: Served) -> Samples:
    """A worker process's entry: its own event loop."""
    return asyncio.run(_serve(served))


def slope(points: Sequence[tuple[float, float | None]]) -> float | None:
    """The least-squares slope of resident MB over time after the warm-up (the first fifth),
    in MB per hour."""
    known = [(t, m) for t, m in points if m is not None]
    if len(known) < 3:
        return None
    warm = [(t, m) for t, m in known if t >= known[-1][0] / 5]
    if len(warm) < 3:
        return None
    xs, ys = [t for t, _ in warm], [m for _, m in warm]
    return statistics.linear_regression(xs, ys).slope * 3600


async def drive(
    plan: Plan,
    *,
    seconds: float,
    concurrency: int,
    workers: int,
    memory: bool,
    watch: Mapping[str, int],
) -> dict[str, Any]:
    """``concurrency`` queued runs in flight for ``seconds``, executed by ``workers`` worker
    processes: the latencies, the errors, and each process's memory over time."""
    context = multiprocessing.get_context("spawn")
    lanes = max(1, concurrency // workers)
    pool = ProcessPoolExecutor(max_workers=workers, mp_context=context)
    loop = asyncio.get_running_loop()
    serving = [
        loop.run_in_executor(pool, serve, Served(plan, seconds + 30, lanes, memory))
        for _ in range(workers)
    ]
    latencies: list[float] = []
    errors: list[str] = []
    watched: dict[str, list[tuple[float, float | None]]] = {name: [] for name in watch}
    began = time.monotonic()
    async with harness() as h:
        queued = wrapped(h, plan, memory)

        async def lane(n: int) -> None:
            while time.monotonic() - began < seconds:
                t = time.monotonic()
                try:
                    handle = await queued.start(TASK, user=f"load-{n}", thread=f"load-{n}-{t}")
                    result = await handle.result(timeout=plan.run_seconds)
                except Exception as exc:  # a failed run is a measured error
                    errors.append(f"{type(exc).__name__}: {exc}"[:200])
                    continue
                if result.status.value.lower() != "success":
                    errors.append(str(result.error.message if result.error else result.status))
                    continue
                latencies.append(time.monotonic() - t)

        async def sampler() -> None:
            while True:
                for name, pid in watch.items():
                    watched[name].append((time.monotonic() - began, procstat.rss_mb(pid)))
                await asyncio.sleep(plan.sample_seconds)

        sampling = asyncio.create_task(sampler())
        await asyncio.gather(*(lane(n) for n in range(concurrency)))
        sampling.cancel()
    served = await asyncio.gather(*serving)
    pool.shutdown()
    wall = time.monotonic() - began
    processes = [
        {
            "name": f"worker {i + 1}",
            "slope_mb_per_hour": slope(s.rss),
            "rss": s.rss,
            "cpu_seconds": s.cpu_seconds,
        }
        for i, s in enumerate(served)
    ]
    processes += [
        {"name": name, "slope_mb_per_hour": slope(points), "rss": points, "cpu_seconds": None}
        for name, points in watched.items()
    ]
    return {
        "minutes": seconds / 60,
        "concurrency": concurrency,
        "workers": workers,
        "memory": memory,
        "runs": len(latencies) + len(errors),
        "throughput": round(len(latencies) / wall, 3) if wall else None,
        "p50": procstat.percentile(latencies, 50),
        "p95": procstat.percentile(latencies, 95),
        "p99": procstat.percentile(latencies, 99),
        "errors": len(errors),
        "error_examples": sorted(set(errors))[:3],
        "processes": processes,
        "pid": os.getpid(),
    }
