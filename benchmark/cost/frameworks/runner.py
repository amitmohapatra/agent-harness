"""One arm, in a process of its own (cold: nothing built before it starts): every run of its
cells, one at a time, each row appended to a JSON-lines file as it ends (a stopped arm keeps
what it measured; ``--resume`` skips those runs).

Per run it keeps what the cost benchmark keeps (:func:`benchmark.cost.run.row_of`: task
success, tool coverage, tokens, model calls, seconds) — the tokens and calls from the metering
proxy's counts before and after the run, so every framework is counted the same way — and the
seed, this process's CPU seconds (its children's too: the Claude Code CLI) and its peak
resident memory during the run (the peak reset before each run through
``/proc/self/clear_refs``).

**Cells**: a task type at a catalogue size (:data:`CELLS`) — ``short`` (the six short tasks on
10 tools), ``long`` (the three long ones on 10 tools), ``many`` (the short tasks on 100 tools),
``many50`` (on 50). A cell's ``n`` runs go through its tasks in turn, the k-th pass with seed
``seed + k``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import resource
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from benchmark import procstat
from benchmark.cost import catalog
from benchmark.cost.frameworks import arms as built
from benchmark.cost.frameworks import meter

#: The cells: task type -> (catalogue size, task length).
CELLS: Final = {
    "short": (10, "short"),
    "long": (10, "long"),
    "many": (100, "short"),
    "many50": (50, "short"),
}
#: The cells a run measures unless told.
DEFAULT_CELLS: Final = ("short", "long", "many")


@dataclass(frozen=True)
class Planned:
    """One run of a cell: its task, the set it is from, and its seed."""

    cell: str
    index: int
    taskset: catalog.TaskSet
    task: dict[str, Any]
    seed: int

    @property
    def size(self) -> int:
        return CELLS[self.cell][0]

    @property
    def key(self) -> tuple[str, int]:
        return (self.cell, self.index)


def plan(
    cell: str, n: int, seed: int, sets: Sequence[catalog.TaskSet] | None = None
) -> list[Planned]:
    """A cell's ``n`` runs: its tasks in turn, the k-th pass with seed ``seed + k``."""
    if cell not in CELLS:
        raise ValueError(f"no cell {cell!r}: the cells are {', '.join(CELLS)}")
    if n < 1:
        raise ValueError("a cell has at least one run")
    _, length = CELLS[cell]
    tasks = [
        (s, t)
        for s in (sets if sets is not None else catalog.task_sets())
        for t in s.tasks
        if t["metadata"]["length"] == length
    ]
    if not tasks:
        raise ValueError(f"no {length} task in the sets given")
    return [
        Planned(cell, i, tasks[i % len(tasks)][0], tasks[i % len(tasks)][1], seed + i // len(tasks))
        for i in range(n)
    ]


def needs_approval(taskset: catalog.TaskSet, task: dict[str, Any]) -> bool:
    """Whether the task asks for a change (its run may end at that change's approval)."""
    changes = {t.spec.name for t in taskset.tools if t.spec.side_effects != "read"}
    return bool(changes & set(task["metadata"]["needed_tools"]))


# --------------------------------------------------------------------------- per-run usage
def reset_peak() -> bool:
    """This process's peak resident memory reset to what it holds now (Linux 4.0+)."""
    try:
        Path("/proc/self/clear_refs").write_text("5")
    except OSError:
        return False
    return True


def cpu_now() -> float:
    """This process's CPU seconds and its waited-for children's (user + system)."""
    t = os.times()
    return t.user + t.system + t.children_user + t.children_system


def children_peak_mb() -> float:
    """The largest resident memory of any child this process waited for, in MB."""
    return resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024


def reset_peak_of(pid: int) -> bool:
    """Another process's peak resident memory reset (its owner, or root)."""
    try:
        Path(f"/proc/{pid}/clear_refs").write_text("5")
    except OSError:
        return False
    return True


@contextlib.contextmanager
def measured(helpers: Sequence[int] = ()) -> Iterator[dict[str, Any]]:
    """The run's wall seconds, CPU seconds and peak memory, filled in when the block ends:
    this process's and its children's, and those of ``helpers`` (processes working for the
    arm: their CPU added, their peak memory added to this process's)."""
    exact = reset_peak() and all(reset_peak_of(pid) for pid in helpers)
    before = {pid: procstat.cpu_seconds(pid) for pid in helpers}
    cpu, began = cpu_now(), time.monotonic()
    got: dict[str, Any] = {}
    try:
        yield got
    finally:
        got["seconds"] = time.monotonic() - began
        theirs = 0.0
        for pid, then in before.items():
            now = procstat.cpu_seconds(pid)
            theirs += (now - then) if now is not None and then is not None else 0.0
        got["cpu_seconds"] = round(cpu_now() - cpu + theirs, 3)
        peaks = [procstat.peak_rss_mb(pid) for pid in helpers]
        got["peak_rss_mb"] = (procstat.peak_rss_mb() or 0.0) + sum(p or 0.0 for p in peaks)
        got["peak_rss_per_run"] = exact
        got["children_peak_rss_mb"] = round(children_peak_mb(), 1)


def meter_of(used: dict[str, Any]) -> Any:
    """The proxy's counts for one run as the cost benchmark's ``Meter``."""
    from benchmark.cost.run import Meter

    m = Meter()
    m.calls = used["calls"]
    m.failed = used["failed"]
    m.input_tokens = used["input_tokens"]
    m.output_tokens = used["output_tokens"]
    m.cached_tokens = used["cached_tokens"]
    m.cache_reported = used["cache_reported"]
    m.refused_tokens = used["refused_tokens"]
    m.seconds = list(used["seconds"])
    return m


def row_for(
    arm: built.Arm,
    planned: Planned,
    *,
    result: Any,
    used: dict[str, Any],
    usage: dict[str, Any],
    called: list[str],
) -> dict[str, Any]:
    """One run as the report keeps it: the cost benchmark's row, and the arm's own fields."""
    from benchmark.cost.run import row_of

    row = row_of(
        result,
        planned.task,
        meter_of(used),
        called,
        usage["seconds"],
        approval=needs_approval(planned.taskset, planned.task),
        cpu={"arm": usage["cpu_seconds"]},
    )
    return row | {
        "arm": arm.name,
        "framework": arm.framework,
        "kind": arm.kind,
        "feature": arm.feature,
        "cell": planned.cell,
        "index": planned.index,
        "size": planned.size,
        "set": planned.taskset.name,
        "seed": planned.seed,
        "peak_rss_mb": usage["peak_rss_mb"],
        "peak_rss_per_run": usage["peak_rss_per_run"],
        "children_peak_rss_mb": usage["children_peak_rss_mb"],
        "date": datetime.now(UTC).strftime("%Y-%m-%d"),
    }


def done_keys(path: Path, arm: str) -> set[tuple[str, int]]:
    """The runs of ``arm`` a rows file already holds."""
    if not path.exists():
        return set()
    keys = set()
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("arm") == arm and "cell" in row:
                keys.add((row["cell"], row["index"]))
    return keys


# --------------------------------------------------------------------------- running an arm
@dataclass(frozen=True)
class Options:
    arm: built.Arm
    setup: built.Setup
    meter_url: str
    cells: tuple[str, ...]
    n: int
    seed: int
    timeout: float
    out: Path


async def resolved(sources: Sequence[Any]) -> list[Any]:
    from trellis.harness.tools.sources import as_source

    return [t for s in sources for t in await as_source(s).resolve()]


async def run_arm(options: Options, progress: Callable[[str], None] = print) -> list[dict]:
    """Every planned run of the arm not already in ``out``, in cell order."""
    arm = options.arm
    done = done_keys(options.out, arm.name)
    todo = [
        p
        for cell in options.cells
        for p in plan(cell, options.n, options.seed)
        if p.key not in done
    ]
    rows: list[dict] = []
    if not todo:
        return rows
    if arm.wrapped:
        rows = await _run_wrapped(options, todo, progress)
    else:
        rows = await _run_native(options, todo, progress)
    return rows


def _write(out: Path, row: dict[str, Any]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as f:
        f.write(json.dumps(row, default=str) + "\n")


def _said(row: dict[str, Any]) -> str:
    return (
        f"{row['arm']:<28} {row['cell']:<6} #{row['index']} {row['status']:<11} "
        f"ok={row['success']} cov={row['tool_coverage']} in={row['input_tokens']} "
        f"calls={row['model_calls']} {row['seconds']}s  {row['task']}"
    )


async def _run_native(
    options: Options, todo: Sequence[Planned], progress: Callable[[str], None]
) -> list[dict]:
    from benchmark.cost.run import scenario
    from trellis.harness.evals import EvalOutput, evaluate

    arm, rows = options.arm, []
    calls = built.Calls()
    runners: dict[tuple[str, int], built.Runner] = {}
    for planned in todo:
        where = (planned.taskset.name, planned.size)
        if where not in runners:
            tools = await resolved(catalog.catalogue(planned.taskset.tools, planned.size))
            runners[where] = built.native(arm, options.setup, planned.taskset.system, tools, calls)
        runner = runners[where]

        async def target(task: Any, _runner: built.Runner = runner) -> EvalOutput:
            answer = await built.bounded(_runner, str(task), options.timeout)
            return EvalOutput(answer, trajectory=list(calls.made))

        target.__name__ = arm.name.replace("/", "-")
        calls.reset()
        before = meter.snapshot(options.meter_url, seed=planned.seed)
        with measured(built.HELPERS) as usage:
            report = await evaluate(target, [planned.task], scenario.EVALUATORS, concurrency=1)
        used = meter.used(before, meter.snapshot(options.meter_url))
        row = row_for(
            arm, planned, result=report.items[0], used=used, usage=usage, called=calls.names
        )
        rows.append(row)
        _write(options.out, row)
        progress(_said(row))
    return rows


async def _run_wrapped(
    options: Options, todo: Sequence[Planned], progress: Callable[[str], None]
) -> list[dict]:
    from benchmark.cost.run import USER, Called, scenario
    from trellis import Harness, Settings

    arm, rows = options.arm, []
    config = Settings.from_env().model_copy(update={"otlp_endpoint": None})
    async with Harness(config=config, gateway=False) as h:
        agents: dict[tuple[str, int], Any] = {}
        called = Called()
        for planned in todo:
            where = (planned.taskset.name, planned.size)
            if where not in agents:
                sources = catalog.catalogue(planned.taskset.tools, planned.size)
                target, wrap = await built.wrapped(
                    arm, options.setup, h, planned.taskset.system, sources
                )
                agents[where] = h.wrap(
                    target,
                    id=agent_id(arm.name, planned.taskset.name, planned.size),
                    hooks=[called],
                    timeout=options.timeout,
                    **wrap,
                )
            called.tools = []
            before = meter.snapshot(options.meter_url, seed=planned.seed)
            with measured() as usage:
                report = await h.evaluate(
                    agents[where],
                    [planned.task],
                    scenario.EVALUATORS,
                    run_name=f"bench-frameworks-{arm.name.replace('/', '-')}",
                    concurrency=1,
                    user=USER,
                )
            used = meter.used(before, meter.snapshot(options.meter_url))
            row = row_for(
                arm,
                planned,
                result=report.items[0],
                used=used,
                usage=usage,
                called=list(called.tools),
            )
            rows.append(row)
            _write(options.out, row)
            progress(_said(row))
    return rows


def agent_id(arm: str, taskset: str, size: int) -> str:
    """The id an arm's agent is wrapped under, one per task set and catalogue size."""
    return f"fwbench-{arm.replace('/', '-').replace('+', '-')}-{taskset.replace(' ', '-')}-{size}"


# --------------------------------------------------------------------------- the load check
#: The task a load check runs (the first short task), on 10 tools.
LOAD_CELL: Final = "short"


async def load_check(
    options: Options, concurrency: int, runs: int, progress: Callable[[str], None] = print
) -> dict[str, Any]:
    """``runs`` runs of one short task, ``concurrency`` at a time: their latencies (p50, p95),
    successes, errors, the wall time and throughput, and the tokens all of them used."""
    arm = options.arm
    planned = plan(LOAD_CELL, 1, options.seed)[0]
    if arm.framework == "trueforge":  # its server works for the arm: measured with it
        from benchmark.cost.frameworks.trueforge import TrueForge

        built.HELPERS.append(TrueForge.current().pid)
    before = meter.snapshot(options.meter_url, seed=planned.seed)
    with measured(built.HELPERS) as usage:
        if arm.wrapped:
            results = await _load_wrapped(options, planned, concurrency, runs)
        else:
            results = await _load_native(options, planned, concurrency, runs)
    used = meter.used(before, meter.snapshot(options.meter_url))
    latencies = [r["seconds"] for r in results]
    found = {
        "arm": arm.name,
        "framework": arm.framework,
        "kind": arm.kind,
        "load": True,
        "concurrency": concurrency,
        "runs": runs,
        "task": planned.task["metadata"]["task"],
        "wall_seconds": round(usage["seconds"], 2),
        "throughput_per_min": round(60 * runs / usage["seconds"], 3) if usage["seconds"] else None,
        "p50": procstat.percentile(latencies, 50),
        "p95": procstat.percentile(latencies, 95),
        "successes": sum(1 for r in results if r["success"]),
        "errors": sum(1 for r in results if r["error"]),
        "model_calls": used["calls"],
        "failed_model_calls": used["failed"],
        "input_tokens": used["input_tokens"],
        "output_tokens": used["output_tokens"],
        "cpu_seconds": usage["cpu_seconds"],
        "peak_rss_mb": usage["peak_rss_mb"],
    }
    _write(options.out, found)
    progress(
        f"{arm.name:<28} load c={concurrency} runs={runs} ok={found['successes']} "
        f"errors={found['errors']} p50={found['p50']} p95={found['p95']} "
        f"{found['wall_seconds']}s"
    )
    return found


async def _load_native(
    options: Options, planned: Planned, concurrency: int, runs: int
) -> list[dict[str, Any]]:
    from benchmark.cost.run import scenario
    from trellis.harness.evals import EvalOutput, evaluate

    tools = await resolved(catalog.catalogue(planned.taskset.tools, planned.size))
    slots = asyncio.Semaphore(concurrency)
    approval = needs_approval(planned.taskset, planned.task)

    async def one() -> dict[str, Any]:
        calls = built.Calls()  # an agent of its own: the calls are this run's
        runner = built.native(options.arm, options.setup, planned.taskset.system, tools, calls)

        async def target(task: Any) -> EvalOutput:
            answer = await built.bounded(runner, str(task), options.timeout)
            return EvalOutput(answer, trajectory=list(calls.made))

        async with slots:
            began = time.monotonic()
            report = await evaluate(target, [planned.task], scenario.EVALUATORS, concurrency=1)
            seconds = time.monotonic() - began
        return outcome_of(report.items[0], seconds, approval=approval)

    return list(await asyncio.gather(*(one() for _ in range(runs))))


def outcome_of(item: Any, seconds: float, *, approval: bool) -> dict[str, Any]:
    """A load-check run: its seconds, whether it did its task (the cost benchmark's rule), and
    its error."""
    from benchmark.cost.run import succeeded

    scores = {s.name: s.value for s in item.scores}
    row = {
        "status": item.status,
        "tool_coverage": scores.get("tool_coverage"),
        "coverage_comment": next(
            (s.comment for s in item.scores if s.name == "tool_coverage"), None
        ),
        "checks": {k: v for k, v in scores.items() if k != "tool_coverage"},
    }
    return {
        "seconds": seconds,
        "success": succeeded(row, approval=approval),
        "error": item.error if item.status == "error" else None,
    }


async def _load_wrapped(
    options: Options, planned: Planned, concurrency: int, runs: int
) -> list[dict[str, Any]]:
    from benchmark.cost.run import USER, scenario
    from trellis import Harness, Settings

    config = Settings.from_env().model_copy(update={"otlp_endpoint": None})
    async with Harness(config=config, gateway=False) as h:
        sources = catalog.catalogue(planned.taskset.tools, planned.size)
        target, wrap = await built.wrapped(
            options.arm, options.setup, h, planned.taskset.system, sources
        )
        agent = h.wrap(
            target,
            id=agent_id(options.arm.name, "load", concurrency),
            timeout=options.timeout,
            **wrap,
        )
        slots = asyncio.Semaphore(concurrency)
        approval = needs_approval(planned.taskset, planned.task)

        async def one() -> dict[str, Any]:
            async with slots:
                began = time.monotonic()
                report = await h.evaluate(
                    agent, [planned.task], scenario.EVALUATORS, concurrency=1, user=USER
                )
                seconds = time.monotonic() - began
            return outcome_of(report.items[0], seconds, approval=approval)

        return list(await asyncio.gather(*(one() for _ in range(runs))))
