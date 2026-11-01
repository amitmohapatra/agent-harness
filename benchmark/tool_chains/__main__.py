"""The tool-chain benchmark: dependent tool calls — one call's output (an id, a name, a date) is
the next one's input, fanned out and fanned in, with a failing step, an id read from a large
kept result and a follow-up turn — measured per arm (``benchmark.tool_chains.cases``).

    python -m benchmark.tool_chains --scripted --out build/chains-fake            # no model
    LLM_BASE_URL=https://api.anthropic.com/v1/ LLM_GATEWAY=openai-compatible LLM_API_KEY=... \\
      python -m benchmark.tool_chains --model claude-haiku-5-5 --arms react,react-cold \\
      --max-usd 4 --out build/chains

**Arms** (:data:`ARMS`): an adapter (``react``: ``ReAct`` as it is built; ``langgraph``:
LangChain's ``create_agent`` with only the harness's tool running; ``deep``: ``ReAct`` on Deep
Agents) and its memory — off; ``cold``, the memory service of ``MEMORY_URL`` with nothing
learned (its context and tool hints, which narrow the tools offered from 30 tools on); or
``warm``, the same after the ``train`` split ran ``--train-passes`` times as the same agent,
so the memory service learned its plans. The tools are the world's fifteen and distractors
up to ``--size`` (``benchmark.cost.catalog``: realistic MCP-style tools of other systems).

Per case it records each turn's answer and score, the calls made (tool, arguments, status),
the tools each model call was offered (so a needed tool narrowed away shows), and the model
calls, tokens and seconds; ``--max-usd`` stops starting cases once the usage the responses
reported, at the model's listed price, reaches it. Writes ``<out>-<arm>.jsonl`` (a row per
case) and ``<out>.json`` / ``<out>.md`` (per arm and split: accuracy, calls, tokens, seconds,
dollars).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from collections.abc import Sequence
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from benchmark.cost import catalog, prices, stats
from benchmark.cost import run as runner
from benchmark.cost.variants import VARIANTS, Setup
from benchmark.tool_chains import cases as C
from benchmark.tool_chains import world
from trellis import Harness, Hooks
from trellis.contracts import ToolCall, ToolOutcome
from trellis.harness.react import window_of


@dataclass(frozen=True)
class Arm:
    name: str
    variant: str  # benchmark.cost.variants
    memory: str  # off, cold, warm


ARMS: Final = {
    a.name: a
    for a in (
        Arm("react", "main", "off"),
        Arm("react-cold", "main", "cold"),
        Arm("react-warm", "main", "warm"),
        Arm("langgraph", "baseline", "off"),
        Arm("deep", "deep", "off"),
    )
}
USER: Final = "chains-bench"
#: What the memory context's learned skills start with.
SKILLS: Final = "Learned skills for this task"
#: The calls of the case running in this task: (tool, arguments, status).
_CALLS: ContextVar[list[dict[str, Any]] | None] = ContextVar("chains_calls", default=None)


class Calls(Hooks):
    """Every tool call of the current case, with its arguments and how it ended."""

    async def after_tool(self, call: ToolCall, outcome: ToolOutcome) -> ToolOutcome:
        held = _CALLS.get()
        if held is not None:
            held.append({"tool": call.tool, "args": call.args, "status": outcome.status.value})
        return outcome


class Offered(runner.Meter):
    """A :class:`benchmark.cost.run.Meter` that also keeps the tools each model call was
    offered."""

    def __init__(self) -> None:
        super().__init__()
        self.offered: list[list[str]] = []
        #: whether a model call was sent a learned skill (the memory context's plan)
        self.skilled = False

    def on_chat_model_start(self, serialized: Any, messages: Any, **kwargs: Any) -> None:
        super().on_chat_model_start(serialized, messages, **kwargs)
        tools = (kwargs.get("invocation_params") or {}).get("tools") or []
        names = [
            t if isinstance(t, str) else (t.get("function") or t).get("name", "?") for t in tools
        ]
        self.offered.append(names)
        sent = " ".join(str(getattr(m, "content", "")) for batch in messages for m in batch)
        self.skilled = self.skilled or SKILLS in sent


def scripted(case: C.Case) -> Any:
    """A model that makes the case's gold calls, round by round, and gives its reference
    answers (``--scripted``: the pipeline at no cost)."""
    from tests.support.chat_model import ScriptedChatModel

    turns: list[Any] = []
    for turn in case.turns:
        turns += [list(r) for r in turn.rounds] + [turn.reference]
    return ScriptedChatModel(turns=turns, spends=(1000, 50))


@dataclass
class Budget:
    """Dollars spent so far, from the usage the responses reported."""

    cap: float | None
    price: prices.Price | None
    spent: float = 0.0

    def add(self, meter: runner.Meter) -> float:
        if self.price is None:
            return 0.0
        cost = prices.dollars(
            self.price,
            input_tokens=meter.input_tokens,
            output_tokens=meter.output_tokens,
            cached=meter.cached_tokens if meter.cache_reported else 0,
            written=meter.written_tokens,
        )
        self.spent += cost
        return cost

    @property
    def over(self) -> bool:
        return self.cap is not None and self.spent >= self.cap


def agent_of(h: Harness, arm: Arm, agent_id: str, args: argparse.Namespace, model: Any) -> Any:
    """``arm``'s agent, wrapped as ``agent_id``: the world's tools and the distractors."""
    window, _ = window_of(None, args.window)
    built = Setup(
        system=world.SYSTEM,
        model=model,
        window=window,
        max_steps=args.max_steps,
        tools=tuple(catalog.catalogue(list(world.TOOLS), args.size)),
    )
    target, options = VARIANTS[arm.variant].build(built)
    return h.wrap(target, id=agent_id, hooks=[Calls()], run_timeout=args.timeout, **options)


async def run_case(
    h: Harness, arm: Arm, case: C.Case, agent: Any, budget: Budget
) -> dict[str, Any] | None:
    """One case as ``arm``: every turn on one thread, scored."""
    if budget.over:
        return None
    world.reset()
    calls: list[dict[str, Any]] = []
    token = _CALLS.set(calls)
    thread = f"chains-{case.id}-{uuid.uuid4().hex[:8]}"
    turns, meter, began = [], Offered(), time.monotonic()
    try:
        with runner.metered(meter):
            for turn in case.turns:
                before = len(calls)
                result = await agent.run(turn.text, user=USER, thread=thread)
                answer = result.answer if isinstance(result.answer, str) else None
                score = C.scored(turn, answer)
                turns.append(
                    {
                        "run_id": result.run_id,
                        "text": turn.text,
                        "status": str(result.status),
                        "error": result.error.message[:300] if result.error else None,
                        "answer": (answer or "")[:600],
                        "calls": calls[before:],
                        **score,
                    }
                )
                if arm.memory != "off":
                    await h.writes.drain()  # the turn recorded before the next is asked
    finally:
        _CALLS.reset(token)
    called = {c["tool"] for c in calls}
    needed = case.needed
    unoffered = sorted(
        {n for n in needed if meter.offered and any(n not in names for names in meter.offered)}
    )
    return {
        "arm": arm.name,
        "case": case.id,
        "shape": case.shape,
        "split": case.split,
        "correct": all(t["correct"] for t in turns),
        "turns": turns,
        "needed": needed,
        "missed_tools": [n for n in needed if n not in called],
        "narrowed_away": unoffered,  # a needed tool some model call was not offered
        "offered_counts": [len(names) for names in meter.offered],
        "skill_offered": meter.skilled,
        "model_calls": meter.calls,
        "tool_calls": len(calls),
        "input_tokens": meter.input_tokens,
        "output_tokens": meter.output_tokens,
        "cached_tokens": meter.cached_tokens if meter.cache_reported else None,
        "seconds": round(time.monotonic() - began, 2),
        "usd": round(budget.add(meter), 6),
    }


def kept(out: Path, row: dict[str, Any], budget: Budget) -> None:
    """A case's row, appended to ``out`` as it ends (a run stopped part way keeps it)."""
    with out.open("a") as f:
        f.write(json.dumps(row, default=str) + "\n")
    sys.stdout.write(
        f"{row['arm']:<11} {row['phase']:<7} {row['case']:<18} "
        f"{'ok ' if row['correct'] else 'BAD'} calls={row['model_calls']} "
        f"in={row['input_tokens']} ${budget.spent:.3f}\n"
    )
    sys.stdout.flush()


async def learned(
    h: Harness, agent_id: str, verdicts: Sequence[str], within: float
) -> list[dict[str, Any]]:
    """The skills the memory service learned for ``agent_id`` from its confirmed runs: the
    people's ``verdicts`` approved by the tenant's administrator (``TRELLIS_API_KEY`` must be
    one, as the development key is) — only a verified outcome teaches, a run's own report of
    success does not — then its listing asked every few seconds until two agree and one skill
    is active, at most ``within`` seconds (the memory service's worker runs the learning job:
    every five minutes by default)."""
    from trellis.memory import MemoryClient

    tenant = await h._tenant()
    async with MemoryClient(os.environ["MEMORY_URL"], api_key=os.environ["TRELLIS_API_KEY"]) as m:
        admin = m.bind(tenant_id=tenant)
        for at in range(0, len(verdicts), APPROVE_BATCH):
            await admin.feedback.approve_many(list(verdicts[at : at + APPROVE_BATCH]))
        skills = admin.advanced.skills
        seen: list[Any] = []
        until = time.monotonic() + within
        while time.monotonic() < until:
            now = [
                (s.name, s.status, tuple(s.steps), s.runs)
                for s in await skills.list(agent=agent_id)
            ]
            if now == seen and any(status == "active" for _, status, _, _ in now):
                break
            seen = now
            await asyncio.sleep(LEARN_POLL)
    return [{"name": n, "status": st, "steps": list(sp), "runs": r} for n, st, sp, r in seen]


#: Verdicts approved in one request.
APPROVE_BATCH: Final = 50
#: Seconds between two listings of what the memory service learned.
LEARN_POLL: Final = 10.0
#: The skills each warm arm's teaching left (for the report).
taught: dict[str, list[dict[str, Any]]] = {}


async def run_arm(
    arm: Arm, chosen: Sequence[C.Case], args: argparse.Namespace, budget: Budget
) -> list[dict[str, Any]]:
    """``chosen`` as ``arm`` (``warm``: after the train split taught the same agent)."""
    stamp = uuid.uuid4().hex[:8]
    agent_id = f"chains-{arm.name}-{stamp}"
    rows: list[dict[str, Any]] = []
    gate = asyncio.Semaphore(args.concurrency)
    out = Path(f"{args.out}-{arm.name}.jsonl")
    verdicts: list[str] = []

    async with runner.harness(memory=arm.memory != "off") as h:
        # one agent for every case (what it learns is its own); a scripted model is a case's
        shared = None if args.scripted else agent_of(h, arm, agent_id, args, args.model)

        async def one(case: C.Case, *, keep: bool, phase: str) -> None:
            async with gate:
                agent = shared or agent_of(
                    h, arm, f"{agent_id}-{case.id}-{phase}", args, scripted(case)
                )
                row = await run_case(h, arm, case, agent, budget)
            if row is None:
                return
            row["phase"] = phase
            if phase.startswith("teach") and row["correct"]:
                # a person confirms the answer was right: the verified outcome that teaches
                for turn in row["turns"]:
                    try:
                        stored = await h.feedback(turn["run_id"], "confirm")
                    except Exception as exc:  # the memory service down: one run teaches less
                        sys.stderr.write(f"no verdict on {turn['run_id']}: {exc}\n")
                        continue
                    if stored is not None:
                        verdicts.append(stored.feedback_id)
            if keep:
                rows.append(row)
            kept(out, row, budget)

        if arm.memory == "warm":
            shapes = {c.shape for c in chosen}  # taught the shapes it is asked
            teach = [c for c in C.split(["train"]) if c.shape in shapes]
            for n in range(args.train_passes):
                await asyncio.gather(*(one(c, keep=False, phase=f"teach{n}") for c in teach))
                await h.writes.drain()
            taught[arm.name] = await learned(h, agent_id, verdicts, args.learn_wait)
        await asyncio.gather(*(one(c, keep=True, phase="eval") for c in chosen))
    return rows


def narrowing(every: Sequence[C.Case]) -> dict[str, Any]:
    """The check without a model: the hints narrowed to one needed tool of a case, which of
    its other needed tools the links keep offered (``tools.links.related``, as the toolbox
    asks it) — by shape, the (candidate, dropped tools) pairs, and how many of all pairs drop
    one."""
    from trellis.harness.tools.links import links, related

    specs = [t.spec for t in world.TOOLS]
    found = links(specs)
    required = {s.name: list((s.input_schema or {}).get("required") or []) for s in specs}
    drops: dict[str, set[tuple[str, tuple[str, ...]]]] = {}
    pairs = dropped = 0
    for case in every:
        needed = set(case.needed)
        for candidate in case.needed:
            pairs += 1
            lost = needed - {candidate} - related({candidate}, found, required)
            if lost:
                dropped += 1
                drops.setdefault(case.shape, set()).add((candidate, tuple(sorted(lost))))
    return {
        "pairs": pairs,
        "dropped": dropped,
        "by_shape": {k: sorted(v) for k, v in sorted(drops.items())},
    }


def _mean(rows: Sequence[dict[str, Any]], key: str) -> dict[str, Any]:
    return stats.interval([float(r[key]) for r in rows if r.get(key) is not None])


def summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_arm: dict[str, Any] = {}
    for arm in dict.fromkeys(r["arm"] for r in rows):
        mine = [r for r in rows if r["arm"] == arm]
        cell: dict[str, Any] = {}
        for split in ("all", *C.SPLITS):
            part = [r for r in mine if split == "all" or r["split"] == split]
            if not part:
                continue
            cell[split] = {
                "cases": len(part),
                "correct": sum(r["correct"] for r in part),
                "accuracy": _mean(part, "correct"),
                "model_calls": _mean(part, "model_calls"),
                "tool_calls": _mean(part, "tool_calls"),
                "input_tokens": _mean(part, "input_tokens"),
                "output_tokens": _mean(part, "output_tokens"),
                "seconds": _mean(part, "seconds"),
                "usd": sum(r["usd"] for r in part),
                "narrowed_away": sum(bool(r["narrowed_away"]) for r in part),
                "skill_offered": sum(bool(r.get("skill_offered")) for r in part),
            }
        shapes = dict.fromkeys(r["shape"] for r in mine)
        cell["by_shape"] = {
            s: f"{sum(r['correct'] for r in mine if r['shape'] == s)}/"
            f"{sum(r['shape'] == s for r in mine)}"
            for s in shapes
        }
        by_arm[arm] = cell
    return by_arm


def markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Tool chains — {report['model']}, {report['size']} tools",
        "",
        "| arm | split | correct | model calls | tool calls | input tokens | output tokens "
        "| seconds | US$ | narrowed away | skill offered |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for arm, cell in report["arms"].items():
        for split in ("all", *C.SPLITS):
            c = cell.get(split)
            if c:
                lines.append(
                    f"| {arm} | {split} | {c['correct']}/{c['cases']} "
                    f"| {stats.shown(c['model_calls'], 1)} | {stats.shown(c['tool_calls'], 1)} "
                    f"| {stats.shown(c['input_tokens'])} | {stats.shown(c['output_tokens'])} "
                    f"| {stats.shown(c['seconds'], 1)} | {c['usd']:.4f} | {c['narrowed_away']} "
                    f"| {c['skill_offered']} |"
                )
    lines += ["", "## By shape", ""]
    for arm, cell in report["arms"].items():
        shown = ", ".join(f"{s} {v}" for s, v in cell["by_shape"].items())
        lines.append(f"- **{arm}**: {shown}")
    n = report["narrowing"]
    lines += [
        "",
        "## Narrowed to one tool, without a model",
        "",
        f"The hints narrowed to one needed tool of a case: {n['dropped']} of {n['pairs']} "
        "(case, tool) pairs drop another needed tool from what is offered.",
        "",
        *(
            f"- **{shape}**: " + "; ".join(f"{c} drops {', '.join(lost)}" for c, lost in pairs)
            for shape, pairs in n["by_shape"].items()
        ),
        "",
        f"Spent: US${report['usd']:.4f}",
        "",
    ]
    return "\n".join(lines)


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    env = os.environ.get
    p = argparse.ArgumentParser(prog="python -m benchmark.tool_chains", description=__doc__)
    p.add_argument("--model", default=env("TRELLIS_BENCH_MODEL") or env("TRELLIS_LIVE_MODEL"))
    p.add_argument("--scripted", action="store_true", help="the gold calls, no model")
    p.add_argument("--arms", default="react", help=f"of {', '.join(ARMS)}")
    p.add_argument("--splits", default="train,heldout")
    p.add_argument("--cases", default="", help="only these case ids")
    p.add_argument("--size", type=int, default=50, help="tools offered, distractors included")
    p.add_argument("--window", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=14)
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--train-passes", type=int, default=2)
    p.add_argument(
        "--learn-wait", type=float, default=600.0, help="most seconds to wait for learning"
    )
    p.add_argument("--max-usd", type=float, default=None)
    p.add_argument("--out", default="build/tool-chains")
    args = p.parse_args(argv)
    if not args.model and not args.scripted:
        p.error("no model: --model, TRELLIS_BENCH_MODEL, or --scripted")
    unknown = [a for a in args.arms.split(",") if a not in ARMS]
    if unknown:
        p.error(f"unknown arms {unknown}: the arms are {', '.join(ARMS)}")
    return args


async def bench(args: argparse.Namespace) -> dict[str, Any]:
    chosen = C.split(args.splits.split(","))
    if args.cases:
        wanted = set(args.cases.split(","))
        chosen = [c for c in chosen if c.id in wanted]
    model = "scripted" if args.scripted else args.model
    budget = Budget(args.max_usd, prices.price_of(model))
    rows: list[dict[str, Any]] = []
    for name in args.arms.split(","):
        rows += await run_arm(ARMS[name], chosen, args, budget)
    report = {
        "model": model,
        "size": args.size,
        "usd": budget.spent,
        "stopped_at_cap": budget.over,
        "arms": summary(rows),
        "learned": taught,
        "narrowing": narrowing(chosen),
    }
    return report


def main(argv: list[str] | None = None) -> int:
    args = arguments(argv)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    report = asyncio.run(bench(args))
    Path(f"{args.out}.json").write_text(json.dumps(report, indent=2, default=str))
    Path(f"{args.out}.md").write_text(markdown(report))
    sys.stdout.write(markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
