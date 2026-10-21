"""Kill-mid-run survival of a wrapped arm (B, C): a run queued in agent-runs, a worker process
that claims it killed (``SIGKILL``) once the run's first model call has answered, a second
worker started, and whether the run still ends — done, or paused at its approval — and how
long that took. Needs the run store (``RUNS_URL``) and the memory service.

    python -m benchmark.cost.frameworks.survive worker --arm langchain/B ...   # the worker

The caller is :func:`trial`; the worker is this module's ``worker`` command: it builds the arm's
agent exactly as the benchmark does and serves it (``trellis.harness.worker.serve``).
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import signal
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Final

from benchmark.cost import catalog
from benchmark.cost.frameworks import arms as built
from benchmark.cost.frameworks import meter, runner

#: The worker's lease lapses after 60 s (agent-runs' ``LEASE_SECONDS``); the ticker re-queues
#: it within seconds, claimable after a short backoff: this is ample for that and the rerun.
WAIT_SECONDS: Final = 600.0
#: The run's task: the first short task, on 10 tools.
CELL: Final = "short"
#: The environment the worker finds the proxy and the key in.
ROOT: Final = "FWBENCH_ROOT"
KEY: Final = "BIFROST_VIRTUAL_KEY"


def worker_command(arm: str, agent_id: str, setup: built.Setup, timeout: float) -> list[str]:
    """The worker's command line (the proxy and its key come from the environment:
    ``FWBENCH_ROOT``, ``BIFROST_VIRTUAL_KEY``)."""
    return [
        sys.executable,
        "-m",
        "benchmark.cost.frameworks.survive",
        "worker",
        "--arm",
        arm,
        "--agent-id",
        agent_id,
        "--model",
        setup.model,
        "--window",
        str(setup.window),
        "--timeout",
        str(timeout),
    ]


async def _first_call(meter_url: str, before: int, within: float) -> bool:
    """Whether a model call answered (the proxy counted one more) within ``within`` seconds."""
    deadline = time.monotonic() + within
    while time.monotonic() < deadline:
        if meter.snapshot(meter_url)["calls"] > before:
            return True
        await asyncio.sleep(0.5)
    return False


async def trial(options: runner.Options, index: int, *, root: Path, log: Path) -> dict[str, Any]:
    """One kill: the run queued, its first worker killed after the first model call, a second
    worker started; the run's end."""
    from trellis import Harness, Settings
    from trellis.contracts import RunStatus

    arm = options.arm
    planned = runner.plan(CELL, 1, options.seed)[0]
    agent_id = f"fwsurvive-{arm.name.replace('/', '-').replace('+', '-')}-{uuid.uuid4().hex[:6]}"
    command = worker_command(arm.name, agent_id, options.setup, options.timeout)
    config = Settings.from_env().model_copy(update={"otlp_endpoint": None})
    async with Harness(config=config, gateway=False) as h:

        async def queued_only(input: Any, agent: Any) -> None:  # the workers run it
            raise AssertionError("runs in a worker")

        agent = h.wrap(queued_only, id=agent_id)
        meter.snapshot(options.meter_url, seed=planned.seed)
        began = time.monotonic()
        handle = await agent.start(planned.task["input"], user="trellis-bench")
        with log.open("a") as out:
            first = await asyncio.create_subprocess_exec(*command, cwd=root, stdout=out, stderr=out)
            calls = meter.snapshot(options.meter_url)["calls"]
            answered = await _first_call(options.meter_url, calls, options.timeout)
            first.send_signal(signal.SIGKILL)
            await first.wait()
            killed_at = time.monotonic() - began
            second = await asyncio.create_subprocess_exec(
                *command, cwd=root, stdout=out, stderr=out
            )
            try:
                result = await handle.result(timeout=WAIT_SECONDS)
                status = result.status
                error = result.error.message if result.error is not None else None
            except Exception as exc:
                status, error = None, f"{type(exc).__name__}: {exc}"
            finally:
                second.send_signal(signal.SIGTERM)
                try:
                    await asyncio.wait_for(second.wait(), 40)
                except TimeoutError:
                    second.kill()
                    await second.wait()
        record = await handle.status()
    ended = status in (RunStatus.SUCCESS, RunStatus.PAUSED)
    found = {
        "arm": arm.name,
        "framework": arm.framework,
        "kind": arm.kind,
        "survive": True,
        "trial": index,
        "killed_after_a_model_call": answered,
        "killed_at_seconds": round(killed_at, 1),
        "status": None if status is None else str(status.value),
        "attempts": record.attempt,
        "survived": bool(ended and record.attempt >= 2),
        "seconds": round(time.monotonic() - began, 1),
        "error": (error or "")[:300] or None,
    }
    runner._write(options.out, found)
    return found


# --------------------------------------------------------------------------- the worker
async def serve(args: argparse.Namespace) -> None:
    from trellis import Harness, Settings
    from trellis.harness.worker.__main__ import serve as serving

    arm = built.ARMS[args.arm]
    setup = built.Setup.on(os.environ[ROOT], args.model, os.environ[KEY], window=args.window)
    planned = runner.plan(CELL, 1, 0)[0]
    config = Settings.from_env().model_copy(update={"otlp_endpoint": None})
    h = Harness(config=config, gateway=False)
    sources = catalog.catalogue(planned.taskset.tools, planned.size)
    target, wrap = await built.wrapped(arm, setup, h, planned.taskset.system, sources)
    h.wrap(target, id=args.agent_id, timeout=args.timeout, **wrap)
    await serving(h, concurrency=1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m benchmark.cost.frameworks.survive")
    sub = parser.add_subparsers(dest="command", required=True)
    worker = sub.add_parser("worker")
    worker.add_argument("--arm", required=True, choices=sorted(built.ARMS))
    worker.add_argument("--agent-id", required=True)
    worker.add_argument("--model", required=True)
    worker.add_argument("--window", type=int, default=8192)
    worker.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args(argv)
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(serve(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
