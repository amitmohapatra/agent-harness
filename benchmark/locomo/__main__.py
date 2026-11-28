"""LoCoMo through a harness agent: ``python -m benchmark.locomo``.

    # what a run would ask and cost; nothing is sent
    python -m benchmark.locomo --conversations 2 --dry-run --price-in 0.10 --price-out 0.40

    # the run (the model and its endpoint as benchmark.cost takes them)
    LLM_BASE_URL=... LLM_API_KEY=... MEMORY_URL=... TRELLIS_API_KEY=... \\
      python -m benchmark.locomo --model provider/model --conversations 2 --yes

Every question the memory repository's own LoCoMo benchmark asks (its ``--conversations`` and
``--sample``: ``--limit`` here) is asked of a harness agent instead, in a new thread, memory
on (``--arm push``) or off (``--arm off``), and the answers are written in the shape that
benchmark's scorer reads. Before anything is sent the run's size and estimated cost are
printed; ``--dry-run`` stops there, and without ``--yes`` so does every run. A stopped run
goes on where it stopped: a question ``--out`` already answers is not asked again (the
status is 1 while a planned question has no answer, 2 when nothing was sent).
``README.md`` beside this file has the commands of both arms and what still differs.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from benchmark.locomo import direct as d
from benchmark.locomo import run


def _pairs(given: list[str], flag: str) -> dict[str, Any]:
    """``KEY=JSON`` pairs, as keyword arguments (a value that is not JSON is its text)."""
    found: dict[str, Any] = {}
    for pair in given:
        name, eq, value = pair.partition("=")
        if not name or not eq:
            raise argparse.ArgumentTypeError(f"{flag} takes KEY=JSON, not {pair!r}")
        try:
            found[name] = json.loads(value)
        except json.JSONDecodeError:
            found[name] = value
    return found


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    env = os.environ.get
    parser = argparse.ArgumentParser(
        prog="python -m benchmark.locomo",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    what = parser.add_argument_group("what is asked")
    what.add_argument(
        "--memory-repo",
        default=str(d.default_repo()),
        help=f"the memory repository's checkout ({d.REPO_ENV}; default: beside this one): its "
        "benchmark/locomo.py is the direct arm, its benchmark/data holds the dataset",
    )
    what.add_argument(
        "--dataset",
        default=None,
        help="locomo10.json, where it is not <memory-repo>/benchmark/data/locomo10.json",
    )
    what.add_argument(
        "--conversations", type=int, default=None, help="the first N conversations (all)"
    )
    what.add_argument(
        "--limit",
        type=int,
        default=None,
        help="N questions per conversation, drawn per category with the direct arm's seed "
        "(its --sample N: the same questions)",
    )
    how = parser.add_argument_group("how it is asked")
    how.add_argument("--arm", choices=run.ARMS, default="push", help="memory on, or off")
    how.add_argument("--adapter", choices=run.ADAPTERS, default="react")
    how.add_argument(
        "--model",
        default=env("TRELLIS_BENCH_MODEL") or env("TRELLIS_LIVE_MODEL"),
        help="the gateway model name (default: TRELLIS_BENCH_MODEL, TRELLIS_LIVE_MODEL)",
    )
    how.add_argument("--window", type=int, default=None, help="the model's context window")
    how.add_argument("--max-steps", type=int, default=8)
    how.add_argument("--timeout", type=float, default=300.0, help="seconds per question")
    how.add_argument("--concurrency", type=int, default=1, help="questions asked at once")
    how.add_argument(
        "--asker",
        choices=run.ASKERS,
        default="question",
        help="who asks: a user of its own per question (default: what one question's run "
        "records is not another's to read), or the direct arm's one user per conversation",
    )
    how.add_argument(
        "--reference",
        choices=run.REFERENCES,
        default="instructions",
        help="where the conversation's reference date is said: after the instructions "
        "(default: memory is then searched with the question alone, as the direct arm "
        "searches), or in the user's message, as the direct arm's own message has it",
    )
    how.add_argument(
        "--without",
        action="append",
        default=[],
        metavar="NAME",
        help="one more part the harness does not do (trellis.harness.features), repeatable",
    )
    how.add_argument(
        "--wrap",
        action="append",
        default=[],
        metavar="KEY=JSON",
        help="one more keyword argument of h.wrap, passed through unchanged, repeatable",
    )
    how.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="KEY=JSON",
        help="one more keyword argument of agent.run, passed through unchanged, repeatable",
    )
    how.add_argument(
        "--added",
        default=run.ADDED,
        help="what the agent is told after the direct arm's answer prompt, memory on",
    )
    how.add_argument(
        "--runs", action="store_true", help="keep the runs in agent-runs (RUNS_URL), not here"
    )
    corpus = parser.add_argument_group("the corpus")
    corpus.add_argument(
        "--ingest",
        choices=("none", "direct"),
        default="none",
        help="none: the direct arm's --reuse-corpus run already stored the conversations; "
        "direct: store them now with the direct arm's own entry point (no model call)",
    )
    corpus.add_argument(
        "--memory-python",
        default=None,
        help=f"the memory repository's interpreter, for --ingest direct ({d.PYTHON_ENV}; "
        "default: its .venv)",
    )
    spend = parser.add_argument_group("spend")
    spend.add_argument("--dry-run", action="store_true", help="print the estimate and stop")
    spend.add_argument("--yes", action="store_true", help="make the model calls")
    spend.add_argument("--price-in", type=float, default=None, help="$ per million input tokens")
    spend.add_argument("--price-out", type=float, default=None, help="$ per million output tokens")
    spend.add_argument(
        "--max-usd", type=float, default=None, help="stop asking once this much is spent"
    )
    assumed = run.Assumed()
    spend.add_argument(
        "--est-searches",
        type=float,
        default=assumed.searches,
        help="the estimate's memory_search calls per question",
    )
    spend.add_argument("--est-result-tokens", type=int, default=assumed.result_tokens)
    spend.add_argument("--est-output-tokens", type=int, default=assumed.output_tokens)
    parser.add_argument("--out", default="build/locomo-harness", help="<out>.json, .rows.jsonl")
    args = parser.parse_args(argv)
    try:
        args.wrap = _pairs(args.wrap, "--wrap")
        args.run = _pairs(args.run, "--run")
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    if (args.price_in is None) != (args.price_out is None):
        parser.error("--price-in and --price-out go together")
    if args.max_usd is not None and args.price_in is None:
        parser.error("--max-usd needs --price-in and --price-out")
    if args.concurrency < 1:
        parser.error("--concurrency is 1 or more")
    return args


def setup_of(args: argparse.Namespace, model: Any = None) -> run.Setup:
    return run.Setup(
        arm=args.arm,
        adapter=args.adapter,
        model=model if model is not None else args.model,
        model_name=args.model or "scripted",
        window=args.window,
        max_steps=args.max_steps,
        run_seconds=args.timeout,
        without=tuple(args.without),
        wrap=args.wrap,
        run=args.run,
        asker=args.asker,
        label=hashlib.sha256(args.out.encode()).hexdigest()[:8],
        added=args.added,
        reference=args.reference,
    )


def main(
    argv: list[str] | None = None,
    *,
    model: Any = None,
    harness: Any = None,
    progress: Any = None,
) -> int:
    """The command line. ``model`` (a chat model) and ``harness`` are a test's: the scripted
    model in the model name's place, a harness on a stand-in memory service in the
    environment's."""
    args = arguments(argv)
    say = progress or (lambda line: sys.stdout.write(line + "\n"))
    direct = d.read(Path(args.memory_repo))
    plan = run.planned(
        direct,
        count=args.conversations,
        limit=args.limit,
        dataset=Path(args.dataset) if args.dataset else None,
    )
    setup = setup_of(args, model)
    assumed = run.Assumed(
        searches=args.est_searches,
        result_tokens=args.est_result_tokens,
        output_tokens=args.est_output_tokens,
    )
    found = run.estimate(
        plan,
        arm=args.arm,
        system=run.instructions(
            direct, setup, plan.questions[0].reference if plan.questions else ""
        ),
        window=args.window,
        assumed=assumed,
        price_in=args.price_in,
        price_out=args.price_out,
    )
    say(run.said(found))
    if args.dry_run:
        say("dry run: nothing was sent")
        return 0
    if not args.yes:
        say("nothing was sent: --yes makes the model calls (--dry-run only prints the estimate)")
        return 2
    if model is None and not args.model:
        say("no model: --model, TRELLIS_BENCH_MODEL or TRELLIS_LIVE_MODEL")
        return 2
    if args.ingest == "direct":
        python = args.memory_python or d.default_python(direct.root)
        kept = d.ingest(
            direct,
            count=args.conversations,
            python=python,
            out=Path(f"{args.out}.ingest.json"),
            progress=say,
        )
        say(f"ingested: {json.dumps(kept, default=str)}")
    found = asyncio.run(_asked(args, plan=plan, direct=direct, setup=setup, given=harness, say=say))
    # a run that stopped, or left a question without an answer, is not done: the same
    # command goes on with what is left
    answered = found["dataset"]["questions"] - found["unanswered"]
    return 0 if answered == found["dataset"]["planned_questions"] else 1


async def _asked(
    args: argparse.Namespace,
    *,
    plan: run.Plan,
    direct: d.Direct,
    setup: run.Setup,
    given: Any,
    say: Any,
) -> dict[str, Any]:
    made = given or run.harnesses(plan, memory=args.arm != "off", runs=args.runs)
    try:
        found = await run.bench(
            made,
            plan=plan,
            direct=direct,
            setup=setup,
            out=args.out,
            concurrency=args.concurrency,
            prices=(args.price_in, args.price_out),
            max_usd=args.max_usd,
            progress=say,
        )
    finally:
        if given is None:
            for h in {id(h): h for h in made.values()}.values():
                await h.aclose()
    used = found["usage"]
    say(
        f"{found['dataset']['questions']} of {found['dataset']['planned_questions']} questions, "
        f"{found['unanswered']} without an answer; {used['model_calls']} model calls, "
        f"{used['input_tokens']} in / {used['output_tokens']} out"
        + (f", ${used['usd']:.4f}" if used["usd"] is not None else "")
    )
    say(f"wrote {args.out}.json (the scorer's shape) and {run.rows_path(args.out)}")
    return found


if __name__ == "__main__":
    raise SystemExit(main())
