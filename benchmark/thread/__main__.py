"""Who should own a conversation's history across runs — the framework's own thread, or the
memory service's window — measured (``python -m benchmark.thread``, under ``heavy.sh``), with
the worker restarted mid-conversation:

* ``framework``: a ``create_agent`` graph on the harness's durable checkpointer
  (``RunCheckpointer``: the graph's checkpoint kept in the run's journal, in agent-runs);
* ``framework-ram``: the same graph on LangGraph's ``InMemorySaver`` (the framework's thread in
  the worker's memory: the harness sends no memory window, ``keeps_conversation``);
* ``memory``: the same graph without a checkpointer: the memory service's window gives the
  recent turns.

They differ in nothing else — the same model, instructions, no tools. After turn
:data:`RESTART_AFTER` the worker is restarted: its harness closed, a new one opened and the
graph built again, as a new worker process would (agent-runs, the memory service and the
model keep running). So what is measured is the history's owner, and whether it outlives a
worker.

All on the same model by name (the gateway the environment names), memory on (``MEMORY_URL``:
records, the context) and the run store agent-runs (``RUNS_URL``: a private instance), the
same conversations: facts said in the first turns, asked back in the later ones. Per turn: the
input tokens the model was sent, the seconds, and whether its input held any of the
conversation's facts (``history``); per recall turn: whether the answer holds the fact. The
decision rule (:func:`decision`): a variant whose model lost the history after the restart loses
outright; of the others, the memory window wins only with recall at least the framework
thread's and at least :data:`TOKEN_SAVING` fewer input tokens. Writes ``<out>.json`` and
``<out>.md``."""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Final

from langchain_core.tracers.context import register_configure_hook

from benchmark.cost.run import Meter
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
#: The worker is restarted after this many turns (between turns 2 and 3).
RESTART_AFTER: Final = 2
#: How much fewer input tokens the memory window needs, at least, to win (with recall at least
#: the framework thread's).
TOKEN_SAVING: Final = 0.20
#: The variant the memory window is compared with (the framework's thread, kept durably), and
#: the memory window's.
FRAMEWORK: Final = "framework"
MEMORY: Final = "memory"


class Heard(Meter):
    """A :class:`Meter` that also keeps the text of the last request the model was sent."""

    def __init__(self) -> None:
        super().__init__()
        self.sent = ""

    def on_chat_model_start(self, serialized: Any, messages: Any, **kwargs: Any) -> None:
        super().on_chat_model_start(serialized, messages, **kwargs)
        self.sent = "\n".join(str(m.content) for batch in messages for m in batch)


_HEARD: ContextVar[Heard | None] = ContextVar("trellis_bench_heard", default=None)
register_configure_hook(_HEARD, inheritable=True)


@contextmanager
def heard() -> Iterator[Heard]:
    """A :class:`Heard` of every chat model call made in this context until the block ends."""
    listener = Heard()
    token = _HEARD.set(listener)
    try:
        yield listener
    finally:
        _HEARD.reset(token)


def _graph(model: str, checkpointer: Any) -> Any:
    from langchain.agents import create_agent

    from trellis.harness.react import _gateway_model

    chat = _gateway_model(model, timeout=None, effort=None, settings=None)
    return create_agent(chat, tools=[], system_prompt=SYSTEM, checkpointer=checkpointer)


def framework_target(model: str) -> Any:
    from trellis.harness.middleware import RunCheckpointer

    return _graph(model, RunCheckpointer())


def framework_ram_target(model: str) -> Any:
    from langgraph.checkpoint.memory import InMemorySaver

    return _graph(model, InMemorySaver())


def memory_target(model: str) -> Any:
    return _graph(model, None)


#: Each variant's graph, built again by the restarted worker.
TARGETS: Final[dict[str, Callable[[str], Any]]] = {
    FRAMEWORK: framework_target,
    "framework-ram": framework_ram_target,
    MEMORY: memory_target,
}


def config() -> Settings:
    """The environment's settings, tracing off; the memory service and agent-runs required."""
    found = Settings.from_env().model_copy(update={"otlp_endpoint": None})
    if found.memory_url is None or found.runs_url is None:
        raise SystemExit("the memory service and agent-runs are needed: MEMORY_URL and RUNS_URL")
    return found


async def converse(variant: str, model: str, repeat: int) -> list[dict[str, Any]]:
    """Each conversation in ``variant``, its worker restarted after :data:`RESTART_AFTER`
    turns: a harness of its own (a worker's life) for the turns before, and one for those
    after, each with the graph built again."""
    rows = []
    for n, (turns, expected) in enumerate(CONVERSATIONS):
        agent_id = f"thread-{variant}-{n}-{repeat}-{uuid.uuid4().hex[:6]}"
        thread, user = f"conv-{uuid.uuid4().hex[:8]}", f"bench-{uuid.uuid4().hex[:6]}"
        facts = [fact.lower() for fact in expected.values()]
        for start, stop in ((0, RESTART_AFTER), (RESTART_AFTER, len(turns))):
            async with Harness(config=config(), gateway=False) as h:
                agent = h.wrap(TARGETS[variant](model), id=agent_id)
                for i in range(start, stop):
                    began = time.monotonic()
                    with heard() as listener:
                        result = await agent.run(turns[i], user=user, thread=thread)
                    seconds = time.monotonic() - began
                    await h.writes.drain()  # the turn recorded before the next asks for it
                    answer = result.answer if isinstance(result.answer, str) else str(result.answer)
                    row = {
                        "variant": variant,
                        "conversation": n,
                        "repeat": repeat,
                        "turn": i,
                        "after_restart": i >= RESTART_AFTER,
                        "status": result.status.value,
                        "input_tokens": listener.input_tokens,
                        "model_calls": listener.calls,
                        "seconds": round(seconds, 2),
                        "history": any(fact in listener.sent.lower() for fact in facts),
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
        later = [r["history"] for r in mine if r.get("after_restart")]
        by_turn = {
            t: statistics.fmean(r["input_tokens"] for r in mine if r["turn"] == t)
            for t in sorted({r["turn"] for r in mine})
        }
        found[variant] = {
            "recall": sum(recall) / len(recall) if recall else None,
            "recall_n": len(recall),
            "history_after_restart": sum(later) / len(later) if later else None,
            "errors": sum(r["status"].lower() != "success" for r in mine),
            "input_tokens_by_turn": by_turn,
            "input_tokens_mean": statistics.fmean(r["input_tokens"] for r in mine),
            "seconds_mean": statistics.fmean(r["seconds"] for r in mine),
        }
    return found


def decision(found: dict[str, Any]) -> str:
    """The owner by the rule: a variant whose model lost the history after the restart loses
    outright; between the framework's durable thread and the memory window, the window wins only
    with recall at least the thread's and :data:`TOKEN_SAVING` fewer input tokens."""
    lost = [v for v, s in found.items() if s["history_after_restart"] != 1.0]
    said = [f"{v} loses outright (its model lost the history after the restart)" for v in lost]
    standing = [v for v in (FRAMEWORK, MEMORY) if v in found and v not in lost]
    if standing != [FRAMEWORK, MEMORY]:
        said.append(f"the owner is {standing[0]}" if standing else "no variant stands")
        return "; ".join(said)
    thread, window = found[FRAMEWORK], found[MEMORY]
    saving = 1 - window["input_tokens_mean"] / thread["input_tokens_mean"]
    wins = window["recall"] >= thread["recall"] and saving >= TOKEN_SAVING
    said.append(
        f"the memory window saves {saving:.0%} of the input tokens at recall "
        f"{window['recall']:.2f} vs {thread['recall']:.2f}: the owner is "
        f"{MEMORY if wins else FRAMEWORK}"
    )
    return "; ".join(said)


def markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Thread history: framework thread vs memory window ({report['model']})",
        "",
        f"{len(CONVERSATIONS)} conversations of 5 turns, {report['repeats']} repeat(s), the "
        f"worker restarted after turn {RESTART_AFTER}; recall: the later turns' answers holding "
        "the fact said in the first turns; history: the model's input held the facts after the "
        "restart.",
        "",
        "| variant | recall | history after restart | errors | input tokens, mean per turn "
        "| by turn (1-5) | seconds per turn |",
        "|---|---|---|---|---|---|---|",
    ]
    for variant, s in report["summary"].items():
        turns = " / ".join(f"{v:,.0f}" for v in s["input_tokens_by_turn"].values())
        recall = "—" if s["recall"] is None else f"{s['recall']:.2f} (n={s['recall_n']})"
        kept = s["history_after_restart"]
        history = "—" if kept is None else f"{kept:.2f}"
        lines.append(
            f"| {variant} | {recall} | {history} | {s['errors']} | {s['input_tokens_mean']:,.0f} "
            f"| {turns} | {s['seconds_mean']:.1f} |"
        )
    return "\n".join([*lines, "", f"Decision: {report['decision']}."]) + "\n"


async def main_async(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for repeat in range(args.repeats):
        for variant in args.variants:
            rows += await converse(variant, args.model, repeat)
    found = summary(rows)
    return {
        "model": args.model,
        "repeats": args.repeats,
        "restart_after": RESTART_AFTER,
        "summary": found,
        "decision": decision(found),
        "runs": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m benchmark.thread", description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--variants", nargs="+", default=list(TARGETS), choices=list(TARGETS))
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
