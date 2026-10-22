"""Who should own a conversation's history across runs — the framework's own thread, or the
memory service's window — measured (``python -m benchmark.thread``, under ``heavy.sh``):

* ``framework``: a ``create_agent`` graph with its own checkpointer (the thread is the
  conversation), one the harness takes as the thread's owner (:func:`own_thread`): it sends no
  memory window (``history.owner``);
* ``memory``: the same graph without a checkpointer (as ``ReAct``'s checkpoint is the run's
  own): the memory service's window gives the recent turns;
* ``framework-restart`` / ``memory-restart``: the same, with the memory service's worker
  killed (``-9``) and started again before turn 3 (the command ``MEM_WORKER_RESTART`` names)
  and a new harness and agent from then on, as after a process restart: what the history
  survives.

The variants differ in nothing else — the same model, instructions, no tools — so what is
measured is the history's owner.

All on the same model by name (the gateway the environment names), memory on (``MEMORY_URL``:
records, the context), the same conversations: facts said in the first turns, asked back in
the later ones. Per turn: the input and output tokens, the seconds, the model calls' seconds
and the rest (the harness's and the memory service's overhead), the memory context call's
milliseconds; per recall turn, whether the answer holds the fact. Without ``--drain`` each turn
follows the last at once, as a person's next message would: what keeps the turn just recorded
in the next window is the harness's own ordering. Writes ``<out>.json`` and ``<out>.md``."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import functools
import json
import os
import shlex
import statistics
import subprocess
import sys
import time
import uuid
from collections.abc import Callable, Sequence
from contextvars import ContextVar
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

from benchmark.cost.run import Meter, metered
from trellis import Harness, Settings

SYSTEM = "You are a support assistant. Answer in one short sentence."
#: Each conversation: its turns, and what a recall turn's answer must hold (by turn index).
CONVERSATIONS: list[tuple[list[str], dict[int, str]]] = [
    (
        [
            "Hi, my order number is QX-4471. Please keep it in mind.",
            "Also, I prefer deliveries on Tuesdays.",
            "Do you deliver on weekends in general?",
            "What is my order number?",
            "Which day do I prefer for deliveries?",
        ],
        {3: "QX-4471", 4: "Tuesday"},
    ),
    (
        [
            "I'm planning a trip to Lisbon in May with my sister Inês.",
            "We want a hotel near the river, budget 120 euros a night.",
            "What's a good way to get from the airport to the city?",
            "Which city am I travelling to?",
            "What is my hotel budget per night?",
        ],
        {3: "Lisbon", 4: "120"},
    ),
    (
        [
            "My account email is dana.k@example.org and my plan is Pro.",
            "Our team has 14 seats.",
            "How do I reset a password in general?",
            "How many seats does our team have?",
            "Which plan am I on?",
        ],
        {3: "14", 4: "Pro"},
    ),
]
#: The environment variable holding the command that restarts the memory service's worker.
RESTART_ENV: Final = "MEM_WORKER_RESTART"
#: The turn (0-based) a restart variant restarts before: between turns 2 and 3.
RESTART_BEFORE_TURN: Final = 2
#: What each prompt message is cut to in a ``--trace`` row.
TRACE_CHARS: Final = 2000
#: The percentile the report's tail columns give.
TAIL_PERCENTILE: Final = 95
MEDIAN: Final = 50


class Variant(StrEnum):
    FRAMEWORK = "framework"
    MEMORY = "memory"
    FRAMEWORK_RESTART = "framework-restart"
    MEMORY_RESTART = "memory-restart"

    @property
    def restarts(self) -> bool:
        return self in (Variant.FRAMEWORK_RESTART, Variant.MEMORY_RESTART)

    @property
    def framework_thread(self) -> bool:
        return self in (Variant.FRAMEWORK, Variant.FRAMEWORK_RESTART)


DEFAULT_VARIANTS: Final = (Variant.FRAMEWORK, Variant.MEMORY)


#: What a checkpointer does, all handed to the ``InMemorySaver`` inside :func:`own_thread`.
SAVER_METHODS: Final = (
    "get_tuple",
    "list",
    "put",
    "put_writes",
    "delete_thread",
    "aget_tuple",
    "alist",
    "aput",
    "aput_writes",
    "adelete_thread",
    "get_next_version",
)


def own_thread() -> Any:
    """The framework's own thread, owned by it as a durable checkpointer's would be: the
    harness leaves memory's window out (``history.owner``; an ``InMemorySaver`` itself would
    get memory's window as text, as a store lost on a restart does), so the variant measures
    the thread alone. It keeps the thread in this process (an ``InMemorySaver`` inside), so a
    restart loses it."""
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.checkpoint.memory import InMemorySaver

    inner = InMemorySaver()

    def handed(name: str) -> Callable[..., Any]:
        return lambda self, *args, **kwargs: getattr(inner, name)(*args, **kwargs)

    saver = type("OwnThread", (BaseCheckpointSaver,), {n: handed(n) for n in SAVER_METHODS})
    return saver(serde=inner.serde)


def framework_target(model: str) -> Any:
    from langchain.agents import create_agent

    from trellis.harness.react import _gateway_model

    chat = _gateway_model(model, timeout=None, effort=None, settings=None)
    return create_agent(chat, tools=[], system_prompt=SYSTEM, checkpointer=own_thread())


def memory_target(model: str) -> Any:
    from langchain.agents import create_agent

    from trellis.harness.react import _gateway_model

    chat = _gateway_model(model, timeout=None, effort=None, settings=None)
    return create_agent(chat, tools=[], system_prompt=SYSTEM)


# ---------------------------------------------------------------------- what a turn measures
class ContextCalls:
    """The memory context calls one turn made: how long each took, what each said."""

    def __init__(self) -> None:
        self.ms: list[float] = []
        self.evidence: list[str] = []


_CALLS: ContextVar[ContextCalls | None] = ContextVar("bench_thread_context", default=None)


def time_memory_context() -> None:
    """Time each memory context call the harness makes (``RunMemory.context``) for the turn
    current then; patched once per process."""
    from trellis.harness.clients.memory import RunMemory

    if getattr(RunMemory.context, "bench_timed", False):
        return
    plain = RunMemory.context

    @functools.wraps(plain)
    async def timed(self: Any, *args: Any, **kwargs: Any) -> Any:
        began = time.monotonic()
        pushed = await plain(self, *args, **kwargs)
        calls = _CALLS.get()
        if calls is not None:
            calls.ms.append((time.monotonic() - began) * 1000)
            calls.evidence.append(str(pushed.evidence_status))
        return pushed

    timed.bench_timed = True  # type: ignore[attr-defined]
    RunMemory.context = timed  # type: ignore[method-assign]


class Prompts(Meter):
    """A :class:`Meter` that also keeps what each model call was sent (``--trace``)."""

    def __init__(self) -> None:
        super().__init__()
        self.prompts: list[list[dict[str, str]]] = []

    def on_chat_model_start(self, serialized: Any, messages: Any, **kwargs: Any) -> None:
        super().on_chat_model_start(serialized, messages, **kwargs)
        for batch in messages:
            self.prompts.append(
                [{"role": m.type, "content": str(m.content)[:TRACE_CHARS]} for m in batch]
            )


def restart_worker() -> float:
    """Kill the memory service's worker and start it again (``MEM_WORKER_RESTART``); the
    seconds it took."""
    command = os.environ.get(RESTART_ENV)
    if not command:
        raise SystemExit(f"a restart variant needs the restart command in {RESTART_ENV}")
    began = time.monotonic()
    subprocess.run(shlex.split(command), check=True)
    return time.monotonic() - began


# ---------------------------------------------------------------------- the conversations
async def converse(
    h: Harness,
    fresh: Callable[[], Harness],
    target: Callable[[], Any],
    variant: Variant,
    repeat: int,
    *,
    drain: bool,
    trace: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for n, (turns, expected) in enumerate(CONVERSATIONS):
        agent_id = f"thread-{variant}-{n}-{repeat}-{uuid.uuid4().hex[:6]}"
        thread, user = f"conv-{uuid.uuid4().hex[:8]}", f"bench-{uuid.uuid4().hex[:6]}"
        async with contextlib.AsyncExitStack() as later:
            runner = h
            agent = runner.wrap(target(), id=agent_id)
            for i, text in enumerate(turns):
                restart: dict[str, Any] = {}
                if variant.restarts and i == RESTART_BEFORE_TURN:
                    await h.writes.drain()  # what a stopping process delivers before it goes
                    restart["restart_seconds"] = round(restart_worker(), 2)
                    # a new process: a new harness, and a new graph (an in-memory thread is gone)
                    runner = await later.enter_async_context(fresh())
                    agent = runner.wrap(target(), id=agent_id)
                row = await one_turn(runner, agent, text, user, thread, drain=drain, trace=trace)
                row |= {"variant": str(variant), "conversation": n, "repeat": repeat, "turn": i}
                row |= restart
                if i in expected:
                    row["recalled"] = expected[i].lower() in row["answer"].lower()
                rows.append(row)
                sys.stdout.write(json.dumps(row) + "\n")
    return rows


async def one_turn(
    h: Harness, agent: Any, text: str, user: str, thread: str, *, drain: bool, trace: bool
) -> dict[str, Any]:
    calls = ContextCalls()
    token = _CALLS.set(calls)
    try:
        began = time.monotonic()
        with metered(Prompts()) as meter:
            result = await agent.run(text, user=user, thread=thread)
        seconds = time.monotonic() - began
    finally:
        _CALLS.reset(token)
    drained = time.monotonic()
    if drain:
        await h.writes.drain()  # the turn recorded before the next asks for the window
    drained = time.monotonic() - drained
    answer = result.answer if isinstance(result.answer, str) else str(result.answer)
    model_seconds = sum(meter.seconds)
    row: dict[str, Any] = {
        "status": result.status.value,
        "input_tokens": meter.input_tokens,
        "output_tokens": meter.output_tokens,
        "model_calls": meter.calls,
        "seconds": round(seconds, 3),
        "model_seconds": round(model_seconds, 3),
        "overhead_ms": round((seconds - model_seconds) * 1000, 1),
        "context_ms": round(sum(calls.ms), 1),
        "evidence": calls.evidence,
        "drain_ms": round(drained * 1000, 1),
        "answer": (answer or "")[:200],
    }
    if trace and isinstance(meter, Prompts):
        row["prompt"] = meter.prompts[-1] if meter.prompts else []
    return row


# ---------------------------------------------------------------------- the report
def percentile(values: Sequence[float], p: float) -> float:
    """The ``p``-th percentile of ``values`` (linear between the closest ranks; 0 for none)."""
    ordered = sorted(values)
    if not ordered:
        return 0.0
    k = (len(ordered) - 1) * p / 100
    low = int(k)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (k - low)


def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    found: dict[str, Any] = {}
    for variant in dict.fromkeys(r["variant"] for r in rows):
        mine = [r for r in rows if r["variant"] == variant]
        recall = [r["recalled"] for r in mine if "recalled" in r]
        by_turn = {
            t: statistics.fmean(r["input_tokens"] for r in mine if r["turn"] == t)
            for t in sorted({r["turn"] for r in mine})
        }
        overhead = [r.get("overhead_ms", 0.0) for r in mine]
        found[variant] = {
            "recall": sum(recall) / len(recall) if recall else None,
            "recall_n": len(recall),
            "errors": sum(r["status"].lower() != "success" for r in mine),
            "input_tokens_by_turn": by_turn,
            "input_tokens_mean": statistics.fmean(r["input_tokens"] for r in mine),
            "output_tokens_mean": statistics.fmean(r.get("output_tokens", 0) for r in mine),
            "seconds_mean": statistics.fmean(r["seconds"] for r in mine),
            "model_seconds_mean": statistics.fmean(r.get("model_seconds", 0.0) for r in mine),
            "overhead_ms_p50": percentile(overhead, MEDIAN),
            "overhead_ms_p95": percentile(overhead, TAIL_PERCENTILE),
            "context_ms_p95": percentile([r.get("context_ms", 0.0) for r in mine], TAIL_PERCENTILE),
            "drain_ms_mean": statistics.fmean(r.get("drain_ms", 0.0) for r in mine),
        }
    return found


def markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Thread history: framework thread vs memory window ({report['model']})",
        "",
        f"{len(CONVERSATIONS)} conversations of 5 turns, {report['repeats']} repeat(s); recall: "
        "the later turns' answers holding the fact said in the first turns. Overhead: a turn's "
        "seconds less its model calls' (the harness's and the memory service's share).",
        "",
        "| variant | recall | errors | input tokens, mean per turn | by turn (1-5) "
        "| output tokens | seconds per turn | model seconds | overhead p50 / p95 ms "
        "| context p95 ms |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for variant, s in report["summary"].items():
        turns = " / ".join(f"{v:,.0f}" for v in s["input_tokens_by_turn"].values())
        recall = "—" if s["recall"] is None else f"{s['recall']:.2f} (n={s['recall_n']})"
        lines.append(
            f"| {variant} | {recall} | {s['errors']} | {s['input_tokens_mean']:,.0f} | {turns} | "
            f"{s['output_tokens_mean']:,.0f} | {s['seconds_mean']:.1f} | "
            f"{s['model_seconds_mean']:.1f} | "
            f"{s['overhead_ms_p50']:,.0f} / {s['overhead_ms_p95']:,.0f} | "
            f"{s['context_ms_p95']:,.0f} |"
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------- the run
def fresh_harness() -> Harness:
    config = Settings.from_env().model_copy(update={"runs_url": None, "otlp_endpoint": None})
    return Harness(config=config, gateway=False)


async def main_async(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    time_memory_context()
    async with fresh_harness() as h:
        if h.memory is None:
            raise SystemExit("the memory service is needed: MEMORY_URL")
        for repeat in range(args.repeats):
            for variant in map(Variant, args.variants):
                build = framework_target if variant.framework_thread else memory_target
                rows += await converse(
                    h,
                    fresh_harness,
                    functools.partial(build, args.model),
                    variant,
                    repeat,
                    drain=args.drain,
                    trace=args.trace,
                )
    return {
        "model": args.model,
        "repeats": args.repeats,
        "drain": args.drain,
        "summary": summary(rows),
        "runs": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m benchmark.thread", description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--out", default="build/bench-thread")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=[str(v) for v in DEFAULT_VARIANTS],
        choices=[str(v) for v in Variant],
    )
    parser.add_argument(
        "--drain",
        action="store_true",
        help="wait for the harness's queued writes after each turn",
    )
    parser.add_argument("--trace", action="store_true", help="keep each turn's prompt")
    args = parser.parse_args(argv)
    report = asyncio.run(main_async(args))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(report, indent=2, default=str))
    out.with_suffix(".md").write_text(markdown(report))
    sys.stdout.write(markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
