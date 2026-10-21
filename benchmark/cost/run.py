"""Running the benchmark: every task of every set, on every catalogue size, for every variant,
``repeats`` times — one run at a time, through the harness as a deployment runs an agent
(``h.wrap`` + ``h.evaluate``), memory and the run store off, no gateway MCP tools (the
catalogue is the whole tool set).

Per run it records what the model was sent and answered — input, output and cached tokens
and the model calls, counted from every chat model call the run made (a LangChain callback,
so a summarization call counts too, not only the loop's) — the wall time, ``tool_coverage``
(the share of the task's needed tools called: the evaluator's score, or, for a run that
ended in an error, from the calls it made before) and task success.
"""

from __future__ import annotations

import importlib
import os
import platform
import re
import time
from collections.abc import Callable, Collection, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any, Final

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.tracers.context import register_configure_hook

from benchmark.cost import catalog, prices, stats
from benchmark.cost.variants import VARIANTS, Setup
from trellis import Harness, Hooks, Settings
from trellis.contracts import ToolCall, ToolOutcome

scenario = importlib.import_module("examples.06_scenarios.tool_choice")

#: The user every benchmark run acts for.
USER: Final = "trellis-bench"
#: A call refused for its length, as llama.cpp's server says so (through the gateway).
REFUSED: Final = re.compile(r"request \((\d+) tokens\) exceeds the available context size")
#: What a run's answer is cut to in the report.
ANSWER_CHARS: Final = 300


class Meter(BaseCallbackHandler):
    """Every chat model call made while it is current (:func:`metered`): how many, how many
    failed, and the tokens their replies report."""

    run_inline = True  # counted in the run's own task, not a thread

    def __init__(self) -> None:
        self.calls = 0
        self.failed = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.cached_tokens = 0
        self.written_tokens = 0
        #: whether any reply said how much of its prompt the cache held
        self.cache_reported = False
        #: the prompt tokens of the calls the server refused as too long, as it counted them
        self.refused_tokens = 0

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        self.calls += 1
        for generations in response.generations:
            for generation in generations:
                usage = getattr(getattr(generation, "message", None), "usage_metadata", None)
                if not usage:
                    continue
                self.input_tokens += usage.get("input_tokens") or 0
                self.output_tokens += usage.get("output_tokens") or 0
                details = usage.get("input_token_details") or {}
                if "cache_read" in details:
                    self.cache_reported = True
                self.cached_tokens += details.get("cache_read") or 0
                self.written_tokens += details.get("cache_creation") or 0

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        self.calls += 1
        self.failed += 1
        refused = REFUSED.search(str(error))
        if refused:
            self.refused_tokens += int(refused.group(1))


_METER: ContextVar[Meter | None] = ContextVar("trellis_bench_meter", default=None)
register_configure_hook(_METER, inheritable=True)


@contextmanager
def metered() -> Iterator[Meter]:
    """A :class:`Meter` of every chat model call made in this context (and the tasks it
    starts) until the block ends."""
    meter = Meter()
    token = _METER.set(meter)
    try:
        yield meter
    finally:
        _METER.reset(token)


class Called(Hooks):
    """The tools the current run called (reset before each run)."""

    def __init__(self) -> None:
        self.tools: list[str] = []

    async def after_tool(self, call: ToolCall, outcome: ToolOutcome) -> ToolOutcome:
        self.tools.append(call.tool)
        return outcome


def harness() -> Harness:
    """The environment's harness with nothing but the model: memory, the run store, tracing
    and the gateway's MCP tools off, so no service's data is read or written."""
    config = Settings.from_env().model_copy(
        update={"memory_url": None, "runs_url": None, "otlp_endpoint": None}
    )
    return Harness(config=config, memory=False, gateway=False)


def coverage_of(needed: Sequence[str], called: Sequence[str]) -> float | None:
    if not needed:
        return None
    return sum(tool in called for tool in needed) / len(needed)


def succeeded(row: dict[str, Any], *, approval: bool) -> bool:
    """Whether a run did its task: it answered — or, for a task that asks for a change
    (``approval``), it paused at that change's approval — having called every needed tool and
    no broader one, and passed every other check of the task (``contains``, ``picks``). A run
    that paused at a change its task did not ask for did not."""
    ended = row["status"] == "success" or (row["status"] == "interrupted" and approval)
    return (
        ended
        and row["tool_coverage"] in (None, 1.0)
        and "called though not needed" not in (row["coverage_comment"] or "")
        and all(float(v) == 1.0 for v in row["checks"].values())
    )


def row_of(
    result: Any,
    task: dict[str, Any],
    meter: Meter,
    called: list[str],
    seconds: float,
    *,
    approval: bool = False,
) -> dict[str, Any]:
    """One run, as the report keeps it (``approval``: its task asks for a change)."""
    scores = {s.name: s for s in result.scores}
    needed = task["metadata"]["needed_tools"]
    scored = scores.get("tool_coverage")
    coverage = float(scored.value) if scored is not None else coverage_of(needed, called)
    comment = scored.comment if scored is not None else None
    row = {
        "task": task["metadata"]["task"],
        "length": task["metadata"].get("length", "short"),
        "status": result.status,
        "error": (result.error or "")[:300] or None,
        "tool_coverage": coverage,
        "coverage_comment": comment,
        # contains, picks...: what else the task expects
        "checks": {n: s.value for n, s in scores.items() if n != "tool_coverage"},
        "errored": result.status == "error",
        "called": called,
        "input_tokens": meter.input_tokens,
        "output_tokens": meter.output_tokens,
        "cached_tokens": meter.cached_tokens if meter.cache_reported else None,
        "cache_written_tokens": meter.written_tokens,
        "model_calls": meter.calls,
        "failed_model_calls": meter.failed,
        "refused_prompt_tokens": meter.refused_tokens,
        "seconds": round(seconds, 2),
        "answer": result.output[:ANSWER_CHARS] if isinstance(result.output, str) else None,
    }
    return row | {"success": succeeded(row, approval=approval)}


async def run_cell(
    h: Harness,
    variant: str,
    size: int,
    *,
    setup: dict[str, Any],
    repeats: int,
    sets: Sequence[catalog.TaskSet],
    run_seconds: float,
    progress: Any = print,
    done: Collection[tuple[Any, ...]] = (),
    on_row: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Every task of ``sets`` ``repeats`` times on catalogues of ``size`` tools, as
    ``variant``: one row per run, each given to ``on_row`` as it ends. A run whose
    :func:`key` is in ``done`` (measured before: ``--resume``) is not run again."""
    rows = []
    for taskset in sets:
        todo = [
            (repeat, task)
            for repeat in range(repeats)
            for task in taskset.tasks
            if (size, variant, taskset.name, repeat, task["metadata"]["task"]) not in done
        ]
        if not todo:
            continue
        built = Setup(system=taskset.system, **setup)
        target, options = VARIANTS[variant].build(built)
        called = Called()
        tools = catalog.catalogue(taskset.tools, size)
        changes = {t.spec.name for t in taskset.tools if t.spec.side_effects != "read"}
        agent = h.wrap(
            target,
            id=f"bench-{variant}-{taskset.name.replace(' ', '-')}-{size}",
            tools=tools,
            hooks=[called],
            timeout=run_seconds,
            **options,
        )
        for repeat, task in todo:
            called.tools = []
            began = time.monotonic()
            with metered() as meter:
                report = await h.evaluate(
                    agent,
                    [task],
                    scenario.EVALUATORS,
                    run_name=f"bench-cost-{variant}-{size}",
                    concurrency=1,
                    user=USER,
                )
            row = row_of(
                report.items[0],
                task,
                meter,
                list(called.tools),
                time.monotonic() - began,
                approval=bool(changes & set(task["metadata"]["needed_tools"])),
            )
            row |= {"size": size, "variant": variant, "set": taskset.name, "repeat": repeat}
            rows.append(row)
            if on_row is not None:
                on_row(row)
            progress(
                f"{size:>4} {variant:<9} r{repeat} {row['status']:<11} "
                f"cov={row['tool_coverage']} in={row['input_tokens']} "
                f"out={row['output_tokens']} calls={row['model_calls']} "
                f"{row['seconds']}s  {row['task']}"
            )
    return rows


def key(row: dict[str, Any]) -> tuple[Any, ...]:
    """What identifies a run: its catalogue size, variant, task set, repeat and task."""
    return (row["size"], row["variant"], row["set"], row["repeat"], row["task"])


#: The per-run figures summarized per cell.
FIGURES: Final = (
    "input_tokens",
    "cached_tokens",
    "output_tokens",
    "model_calls",
    "refused_prompt_tokens",
    "seconds",
    "tool_coverage",
    "success",
    "errored",
)


def summary(
    rows: list[dict[str, Any]], model: str, priced_as: Sequence[str]
) -> dict[str, dict[str, Any]]:
    """A cell's statistics, and its dollars per run at each price."""
    found: dict[str, Any] = {}
    for key in FIGURES:
        values = [float(r[key]) for r in rows if r.get(key) is not None]
        found[key] = stats.interval(values)
    own = model.rsplit("/", 1)[-1] if prices.price_of(model) else None
    dollars = {}
    for name in dict.fromkeys(n for n in (own, *priced_as) if n):
        price = prices.PRICES[name]
        # the cache the measured model reported is its own (a local server's prefix cache is
        # not a vendor's prompt cache): another model's price counts every input token in full
        mine = name == own
        per_run = [
            prices.dollars(
                price,
                input_tokens=r["input_tokens"],
                output_tokens=r["output_tokens"],
                cached=(r["cached_tokens"] or 0) if mine else 0,
                written=(r["cache_written_tokens"] or 0) if mine else 0,
            )
            for r in rows
        ]
        dollars[name] = stats.interval(per_run)
    return {"stats": found, "dollars": dollars}


def environment() -> dict[str, Any]:
    """What the run ran on, and which vendor keys were set (never their values)."""
    return {
        "machine": f"{platform.machine()}, {os.cpu_count()} CPUs, {platform.system()}",
        "vendor_keys": {
            name: bool(os.environ.get(name, "").strip())
            for name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY")
        },
        "bifrost": bool(os.environ.get("BIFROST_URL", "").strip()),
    }


def report_of(
    rows: list[dict[str, Any]],
    *,
    model: str,
    sizes: Sequence[int],
    variants: Sequence[str],
    repeats: int,
    sets: Sequence[catalog.TaskSet],
    setup: dict[str, Any],
    priced_as: Sequence[str],
    notes: Sequence[str] = (),
) -> dict[str, Any]:
    """The report: the settings, every cell's statistics and every run."""
    env = environment()
    listed = prices.price_of(model)
    cells = []
    for size in sizes:
        for variant in variants:
            mine = [r for r in rows if r["size"] == size and r["variant"] == variant]
            if mine:
                lengths = dict.fromkeys(r["length"] for r in mine)
                by_length = {
                    n: summary([r for r in mine if r["length"] == n], model, priced_as)
                    for n in lengths
                }
                cells.append(
                    {
                        "size": size,
                        "variant": variant,
                        "runs": len(mine),
                        **summary(mine, model, priced_as),
                        "by_length": by_length,
                    }
                )
    shown = {model.rsplit("/", 1)[-1]: listed} if listed else {}
    shown |= {name: prices.PRICES[name] for name in priced_as}
    return {
        "model": model,
        "model_price": model.rsplit("/", 1)[-1] if listed else None,
        "date": datetime.now(UTC).strftime("%Y-%m-%d"),
        "machine": env["machine"],
        "environment": env,
        "repeats": repeats,
        "tasks": sum(len(s.tasks) for s in sets),
        "sizes": list(sizes),
        "variants": {v: VARIANTS[v].description for v in variants},
        "setup": {k: v for k, v in setup.items() if k != "model"},
        "priced_as": list(priced_as),
        "prices": {name: asdict(price) for name, price in shown.items()},
        "notes": list(notes),
        "cells": cells,
        "runs": rows,
    }
