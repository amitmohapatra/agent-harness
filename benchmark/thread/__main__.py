"""Who should own a conversation's history across runs — the framework's own thread, or the
memory service's window — measured (``python -m benchmark.thread``, under ``heavy.sh``):

* ``framework``: a ``create_agent`` graph with its own checkpointer (the thread is the
  conversation): the harness sends no memory window (``keeps_conversation``);
* ``memory``: the same graph without a checkpointer (as ``ReAct``'s checkpoint is the run's
  own): the memory service's window gives the recent turns.

The two differ in nothing else — the same model, instructions, no tools — so what is measured
is the history's owner.

Both on the same model by name (the gateway the environment names), memory on (``MEMORY_URL``:
records, the context), the same conversations: facts said in the first turns, asked back in
the later ones. Per turn: the input tokens the model was sent, the seconds; per recall turn:
whether the answer holds the fact. Writes ``<out>.json`` and ``<out>.md``."""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from benchmark.cost.run import metered
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


def framework_target(model: str) -> Any:
    from langchain.agents import create_agent
    from langgraph.checkpoint.memory import InMemorySaver

    from trellis.harness.react import _gateway_model

    chat = _gateway_model(model, timeout=None, effort=None, settings=None)
    return create_agent(chat, tools=[], system_prompt=SYSTEM, checkpointer=InMemorySaver())


def memory_target(model: str) -> Any:
    from langchain.agents import create_agent

    from trellis.harness.react import _gateway_model

    chat = _gateway_model(model, timeout=None, effort=None, settings=None)
    return create_agent(chat, tools=[], system_prompt=SYSTEM)


async def converse(h: Harness, target: Any, variant: str, repeat: int) -> list[dict[str, Any]]:
    rows = []
    for n, (turns, expected) in enumerate(CONVERSATIONS):
        agent = h.wrap(target, id=f"thread-{variant}-{n}-{repeat}-{uuid.uuid4().hex[:6]}")
        thread, user = f"conv-{uuid.uuid4().hex[:8]}", f"bench-{uuid.uuid4().hex[:6]}"
        for i, text in enumerate(turns):
            began = time.monotonic()
            with metered() as meter:
                result = await agent.run(text, user=user, thread=thread)
            seconds = time.monotonic() - began
            await h.writes.drain()  # the turn recorded before the next asks for the window
            answer = result.answer if isinstance(result.answer, str) else str(result.answer)
            row = {
                "variant": variant,
                "conversation": n,
                "repeat": repeat,
                "turn": i,
                "status": result.status.value,
                "input_tokens": meter.input_tokens,
                "model_calls": meter.calls,
                "seconds": round(seconds, 2),
                "answer": (answer or "")[:200],
            }
            if i in expected:
                row["recalled"] = expected[i].lower() in (answer or "").lower()
            rows.append(row)
            sys.stdout.write(json.dumps(row) + "\n")
    return rows


def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    found: dict[str, Any] = {}
    for variant in dict.fromkeys(r["variant"] for r in rows):
        mine = [r for r in rows if r["variant"] == variant]
        recall = [r["recalled"] for r in mine if "recalled" in r]
        by_turn = {
            t: statistics.fmean(r["input_tokens"] for r in mine if r["turn"] == t)
            for t in sorted({r["turn"] for r in mine})
        }
        found[variant] = {
            "recall": sum(recall) / len(recall) if recall else None,
            "recall_n": len(recall),
            "errors": sum(r["status"].lower() != "success" for r in mine),
            "input_tokens_by_turn": by_turn,
            "input_tokens_mean": statistics.fmean(r["input_tokens"] for r in mine),
            "seconds_mean": statistics.fmean(r["seconds"] for r in mine),
        }
    return found


def markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Thread history: framework thread vs memory window ({report['model']})",
        "",
        f"{len(CONVERSATIONS)} conversations of 5 turns, {report['repeats']} repeat(s); recall: "
        "the later turns' answers holding the fact said in the first turns.",
        "",
        "| variant | recall | errors | input tokens, mean per turn | by turn (1-5) "
        "| seconds per turn |",
        "|---|---|---|---|---|---|",
    ]
    for variant, s in report["summary"].items():
        turns = " / ".join(f"{v:,.0f}" for v in s["input_tokens_by_turn"].values())
        recall = "—" if s["recall"] is None else f"{s['recall']:.2f} (n={s['recall_n']})"
        lines.append(
            f"| {variant} | {recall} | {s['errors']} | {s['input_tokens_mean']:,.0f} | {turns} | "
            f"{s['seconds_mean']:.1f} |"
        )
    return "\n".join(lines) + "\n"


async def main_async(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    config = Settings.from_env().model_copy(update={"runs_url": None, "otlp_endpoint": None})
    async with Harness(config=config, gateway=False) as h:
        if h.memory is None:
            raise SystemExit("the memory service is needed: MEMORY_URL")
        for repeat in range(args.repeats):
            rows += await converse(h, framework_target(args.model), "framework", repeat)
            rows += await converse(h, memory_target(args.model), "memory", repeat)
    return {"model": args.model, "repeats": args.repeats, "summary": summary(rows), "runs": rows}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m benchmark.thread", description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--out", default="build/bench-thread")
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
