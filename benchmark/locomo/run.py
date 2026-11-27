"""Asking: every planned LoCoMo question through a harness agent, one run each, in a thread
of its own — and what is kept of it.

Per question the agent is the chosen adapter's (``benchmark.cost.complex.target_of``, the
builders the other suites use), its instructions the direct arm's own answer prompt
(:func:`instructions`), its only tools the memory service's pull tools (:data:`WITHOUT`), and
its input the question as the direct arm words it (``direct.asked``). Two arms: ``push`` —
memory as the harness has it by default, the context pushed and the tools on demand — and
``off`` — no memory at all, the floor. Anything else the harness can be told about memory goes
through unchanged (``without``, ``wrap``, ``run``: :class:`Setup`), so an arm the harness
gains later needs no change here.

Two files are written, after every question (:func:`written`):

* ``<out>.rows.jsonl`` — one line per question asked (:func:`ask`): the answer whole, the
  tools called in order with their arguments, the tokens, model calls, seconds and run id.
  It is also what a rerun reads: a question answered there is not asked again.
* ``<out>.json`` — the same answers in the shape the memory repository's scorer reads
  (``benchmark/locomo_rescore.py``: :func:`result_of`), with no verdict in it: one judge
  grades both arms.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
import uuid
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from benchmark.cost import run as runner
from benchmark.locomo import direct as d
from trellis import Harness, Hooks
from trellis.contracts import ConfigurationError, RunStatus
from trellis.harness.clients.memory import context_budget

#: The arms: memory as the harness has it by default, and memory off.
ARMS: Final = ("push", "off")
#: The adapters a question can be asked through (``benchmark.cost.complex.ADAPTERS``; listed
#: here so the command line does not import seven frameworks to print its help).
ADAPTERS: Final = ("react", "langgraph", "graph", "deepagents", "openai", "claude", "kimi")
#: What the harness does for no question, so the memory tools are the only ones a run has:
#: no tool narrowing or ``tool_search`` (there is nothing to narrow), no gateway MCP tools or
#: Code Mode, no schedule tools, no ``ask_user``, no shell — and no sampled check or judge of
#: the answer, each a model call of its own nobody asked for.
WITHOUT: Final = ("tool_search", "mcp", "grounding", "judges", "schedules", "ask", "code")
#: The memory service's search tool.
SEARCH: Final = "memory_search"
#: What is said after the direct arm's prompt, memory on: that prompt speaks of a CONTEXT
#: block, which a harness agent is not given under that name.
ADDED: Final = (
    "The CONTEXT is what you are given about the earlier conversations with this request, "
    "and what your memory tools return. When what you are given does not hold the answer, "
    f"search memory ({SEARCH}) before you answer or decline."
)
#: Who asks: a user of its own per question (nothing an earlier question's run recorded is
#: this one's to read), or the direct arm's one user per conversation.
ASKERS: Final = ("question", "conversation")
#: Where the conversation's reference date is said. ``instructions`` (the default): after the
#: agent's instructions, in the direct arm's words, and the user's message is the question
#: alone — a harness run's message is also what memory is searched with, and the direct arm
#: searches with the question alone (``builder.build(ctx, question)``,
#: ``benchmark/locomo.py`` 1025). ``question``: in the user's message, as the direct arm's
#: own user message has it (``direct.asked``), which then is the search too.
REFERENCES: Final = ("instructions", "question")
#: The name the dataset is given in a result (the direct arm's: ``benchmark/locomo.py`` 1300).
DATASET_NAME: Final = "LoCoMo (snap-research/locomo)"
#: The length the direct arm cuts a stored answer to (``produced[:400]``, 1123).
DIRECT_ANSWER_CHARS: Final = 400


@dataclass(frozen=True)
class Asked:
    """One question of the plan."""

    conversation: int
    #: its place in the conversation's own question list
    n: int
    category: str
    question: str
    #: the gold answer, as the dataset has it (empty: none, the adversarial kind)
    gold: Any
    reference: str
    tenant: str

    @property
    def key(self) -> str:
        return f"c{self.conversation}-q{self.n}"


@dataclass(frozen=True)
class Plan:
    """What a run asks, and of what."""

    questions: tuple[Asked, ...]
    #: per conversation: its tenant, its stored turns and the questions asked of it
    conversations: tuple[dict[str, Any], ...]
    dataset_sha256: str

    def by_category(self) -> dict[str, int]:
        return dict(Counter(q.category for q in self.questions))


def planned(
    direct: d.Direct,
    *,
    count: int | None,
    limit: int | None,
    dataset: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> Plan:
    """The questions the direct arm asks with ``--conversations count --sample limit``, in
    its order, each in its conversation's tenant."""
    data, digest = d.conversations(direct, count, dataset)
    questions: list[Asked] = []
    listed = []
    for index, item in enumerate(data):
        conversation = item["conversation"]
        tenant = d.tenant_of(direct, index, environ)
        reference = d.reference_date(conversation)
        picked = d.questions_of(direct, item, limit)
        for n, question in picked:
            category = int(question.get("category", 0))
            questions.append(
                Asked(
                    conversation=index,
                    n=n,
                    category=direct.category_names.get(category, str(category)),
                    question=question["question"],
                    gold=question.get("answer") or "",  # benchmark/locomo.py 1045
                    reference=reference,
                    tenant=tenant,
                )
            )
        listed.append(
            {
                "conversation": index,
                "tenant": tenant,
                "turns": len(d.turns(conversation)),
                "questions": len(picked),
                "reference_date": reference,
            }
        )
    return Plan(tuple(questions), tuple(listed), digest)


# --------------------------------------------------------------------------- the estimate
@dataclass(frozen=True)
class Assumed:
    """What an estimate assumes of a question's run (the command line names each)."""

    #: ``memory_search`` calls per question, memory on (each one more model call)
    searches: float = 2.0
    #: the tokens one search's result adds to every later request of the run
    result_tokens: int = 1500
    #: the memory tools' descriptions, sent with every request, memory on
    tools_tokens: int = 1500
    #: what the harness adds around the context (its notes on answering and asking)
    notes_tokens: int = 400
    question_tokens: int = 60
    #: a model reply (an answer, or a tool call), reasoning not counted
    output_tokens: int = 150
    #: the scorer's judge call per answer (the memory repository's, not this run's)
    judge_input_tokens: int = 300
    judge_output_tokens: int = 100


def tokens_of(text: str) -> int:
    """About how many tokens a text is (four characters a token)."""
    return len(text) // 4


def estimate(
    plan: Plan,
    *,
    arm: str,
    system: str,
    window: int | None,
    assumed: Assumed = Assumed(),
    price_in: float | None = None,
    price_out: float | None = None,
) -> dict[str, Any]:
    """What asking the plan's questions is expected to cost: model calls and tokens, and
    dollars at ``price_in`` / ``price_out`` per million tokens. Memory on, a question is one
    request plus one per search, each carrying the pushed context at the harness's budget for
    it (``context_budget``: an upper bound, the service may push less) and every earlier
    search's result."""
    count = len(plan.questions)
    base = tokens_of(system) + assumed.question_tokens
    if arm == "off":
        calls, context = 1.0, 0
        per_question_in = float(base)
    else:
        context = context_budget(window)
        searches = assumed.searches
        calls = 1.0 + searches
        base += assumed.tools_tokens + assumed.notes_tokens + context
        carried = (assumed.result_tokens + assumed.output_tokens) * searches * (searches + 1) / 2
        per_question_in = calls * base + carried
    found: dict[str, Any] = {
        "arm": arm,
        "conversations": len(plan.conversations),
        "turns": sum(c["turns"] for c in plan.conversations),
        "questions": count,
        "by_category": plan.by_category(),
        "model_calls": round(count * calls),
        "input_tokens": round(count * per_question_in),
        "output_tokens": round(count * calls * assumed.output_tokens),
        "per_question": {
            "model_calls": calls,
            "input_tokens": round(per_question_in),
            "output_tokens": round(calls * assumed.output_tokens),
            "context_tokens": context,
        },
        "assumed": asdict(assumed),
        "judge": {
            "calls": count,
            "input_tokens": count * assumed.judge_input_tokens,
            "output_tokens": count * assumed.judge_output_tokens,
        },
        "usd": None,
    }
    if price_in is not None and price_out is not None:
        found["usd"] = round(
            dollars(found["input_tokens"], found["output_tokens"], price_in, price_out), 4
        )
        found["judge"]["usd"] = round(
            dollars(
                found["judge"]["input_tokens"],
                found["judge"]["output_tokens"],
                price_in,
                price_out,
            ),
            4,
        )
        found["prices"] = {"input_per_million": price_in, "output_per_million": price_out}
    return found


def dollars(input_tokens: float, output_tokens: float, price_in: float, price_out: float) -> float:
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000


def said(found: dict[str, Any]) -> str:
    """An estimate, as it is printed before anything is sent."""
    each = found["per_question"]
    cost = (
        f"${found['usd']:.4f} at ${found['prices']['input_per_million']:g} in / "
        f"${found['prices']['output_per_million']:g} out per million tokens"
        if found["usd"] is not None
        else "not priced (--price-in and --price-out, dollars per million tokens)"
    )
    categories = ", ".join(f"{name} {n}" for name, n in sorted(found["by_category"].items()))
    judge = found["judge"]
    judged = f", ${judge['usd']:.4f} at the same price" if "usd" in judge else ""
    return "\n".join(
        [
            f"arm                 {found['arm']}",
            f"conversations       {found['conversations']} ({found['turns']} stored turns)",
            f"questions           {found['questions']} ({categories})",
            f"model calls (est.)  {found['model_calls']} ({each['model_calls']:g} a question)",
            f"input tokens (est.) {found['input_tokens']} ({each['input_tokens']} a question"
            + (
                f", {each['context_tokens']} of them the pushed context at most, each request)"
                if each["context_tokens"]
                else ")"
            ),
            f"output tokens (est.) {found['output_tokens']}",
            f"cost (est.)         {cost}",
            f"judge, not this run {judge['calls']} calls, about {judge['input_tokens']} in / "
            f"{judge['output_tokens']} out{judged} (the memory repository's scorer)",
        ]
    )


# --------------------------------------------------------------------------- the agent
@dataclass(frozen=True)
class Setup:
    """How the questions are asked."""

    arm: str = "push"
    adapter: str = "react"
    #: a gateway model name, or (tests) a chat model
    model: Any = None
    model_name: str = ""
    window: int | None = None
    max_steps: int = 8
    run_seconds: float | None = 300.0
    #: more of what the harness does not do, by name (``trellis.harness.features``)
    without: tuple[str, ...] = ()
    #: more keyword arguments of ``h.wrap`` and of ``agent.run``, passed through unchanged
    wrap: Mapping[str, Any] = field(default_factory=dict)
    run: Mapping[str, Any] = field(default_factory=dict)
    asker: str = "question"
    added: str = ADDED
    reference: str = "instructions"

    def off(self) -> tuple[str, ...]:
        """Every feature the runs are without."""
        named = [*WITHOUT, *(("memory",) if self.arm == "off" else ()), *self.without]
        return tuple(dict.fromkeys(named))

    def described(self) -> dict[str, Any]:
        """The setup as a result keeps it (the model by its name)."""
        return {
            "arm": self.arm,
            "adapter": self.adapter,
            "model": self.model_name,
            "window": self.window,
            "max_steps": self.max_steps,
            "run_seconds": self.run_seconds,
            "without": list(self.off()),
            "wrap": dict(self.wrap),
            "run": dict(self.run),
            "asker": self.asker,
            "added": self.added if self.arm != "off" else "",
            "reference": self.reference,
        }


def instructions(direct: d.Direct, setup: Setup, reference: str = "") -> str:
    """The agent's instructions: the direct arm's answer prompt, word for word; memory on,
    what its CONTEXT is here (``setup.added``); and the conversation's ``reference`` date,
    unless the question carries it (:data:`REFERENCES`)."""
    parts = [direct.answer_system]
    if setup.arm != "off" and setup.added:
        parts.append(setup.added)
    if setup.reference == "instructions" and reference:
        parts.append(d.reference_line(reference))
    return "\n\n".join(parts)


def said_to(question: Asked, setup: Setup) -> str:
    """The user's message: the question, with its reference date where the setup says so."""
    if setup.reference == "question":
        return d.asked(question.question, question.reference)
    return question.question


class Seen(Hooks):
    """What each run did, by run id, read as it ends: its tool calls in order, with their
    arguments, and the memory context it was pushed."""

    def __init__(self) -> None:
        self.runs: dict[str, dict[str, Any]] = {}

    async def on_run_end(self, run: Any, result: Any) -> None:
        self.runs[result.run_id] = {
            "calls": [
                {
                    "tool": call.tool,
                    "args": dict(call.args),
                    "status": outcome.status.value,
                    "ms": outcome.latency_ms,
                    "result_chars": len(json.dumps(outcome.output, default=str)),
                }
                for call, outcome in run.replay.journal.trajectory
            ],
            "context_chars": len(run.pushed_context or ""),
            "evidence_status": run.evidence,
        }


async def built(h: Harness, direct: d.Direct, setup: Setup, seen: Seen, question: Asked) -> Any:
    """The wrapped agent the questions of ``question``'s conversation are asked of."""
    from benchmark.cost import complex as cx

    tenant = question.tenant if h.memory is not None else None
    memory = h.memory is not None and setup.arm != "off"
    # a graph built by hand binds its tools itself: the memory tools, and no other
    bound = await h.memory_tools(tenant=tenant) if memory and setup.adapter == "graph" else []
    target, options = cx.target_of(
        setup.model,
        "memory" if memory else "thread",
        setup.window,
        setup.max_steps,
        adapter=setup.adapter,
        tools=(),
        bound=bound,
        system=instructions(direct, setup, question.reference),
    )
    given = {"hooks": [seen], "run_timeout": setup.run_seconds, **options, **setup.wrap}
    given["without"] = {*given.get("without", ()), *setup.off()}
    return h.wrap(
        target, id=f"locomo-{setup.adapter}-{setup.arm}-c{question.conversation}", **given
    )


def user_of(question: Asked, asker: str) -> str:
    """Who asks: the direct arm's user for the conversation, or one of the question's own."""
    own = d.asker(question.conversation)
    return own if asker == "conversation" else f"{own}-q{question.n}"


async def ask(h: Harness, agent: Any, seen: Seen, question: Asked, setup: Setup) -> dict[str, Any]:
    """One question, asked in a new thread; its row."""
    from benchmark.cost import complex as cx

    user = user_of(question, setup.asker)
    thread = f"locomo-{question.key}-{uuid.uuid4().hex[:8]}"
    text = said_to(question, setup)
    # a platform key names the tenant per run; with memory off there is none to name
    tenant = question.tenant if h.memory is not None else None
    row: dict[str, Any] = {
        "key": question.key,
        "arm": setup.arm,
        "adapter": setup.adapter,
        "model": setup.model_name,
        "conversation": question.conversation,
        "n": question.n,
        "category": question.category,
        "question": question.question,
        "gold": question.gold,
        "asked": text,
        "tenant": tenant,
        "user": user,
        "thread": thread,
        "at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    began = time.monotonic()
    with runner.metered(adapter=setup.adapter) as meter:
        try:
            result = await agent.run(text, user=user, thread=thread, tenant=tenant, **setup.run)
        except Exception as exc:  # the row says it failed; the next question is still asked
            done: dict[str, Any] = {
                "run_id": None,
                "status": "RAISED",
                "error": f"{type(exc).__name__}: {exc}"[:400],
                "answer": None,
            }
        else:
            ended = result.status is RunStatus.SUCCESS
            done = {
                "run_id": result.run_id,
                "status": result.status.value,
                "error": result.error.message[:400] if result.error is not None else None,
                "answer": cx.said_by(result) if ended else None,
            }
    did = seen.runs.pop(done["run_id"] or "", {})
    calls = did.get("calls", [])
    searches = [c for c in calls if c["tool"] == SEARCH]
    return (
        row
        | done
        | {
            "calls": calls,
            "searches": len(searches),
            "search_orders": [c["args"].get("order") for c in searches],
            "search_kinds": [c["args"].get("kinds") for c in searches],
            "context_chars": did.get("context_chars"),
            "evidence_status": did.get("evidence_status"),
            "input_tokens": meter.input_tokens,
            "output_tokens": meter.output_tokens,
            "cached_tokens": meter.cached_tokens if meter.cache_reported else None,
            "model_calls": meter.calls,
            "failed_model_calls": meter.failed,
            "seconds": round(time.monotonic() - began, 3),
        }
    )


# --------------------------------------------------------------------------- the files
def rows_path(out: str) -> Path:
    return Path(f"{out}.rows.jsonl")


def saved(out: str) -> dict[str, dict[str, Any]]:
    """The rows a run already wrote, the latest of each question."""
    path = rows_path(out)
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {row["key"]: row for row in rows}


def answered(row: Mapping[str, Any]) -> bool:
    """Whether a row holds an answer (one that does is not asked again)."""
    return row.get("answer") is not None


def same_run(rows: Mapping[str, Mapping[str, Any]], setup: Setup) -> None:
    """Refuse to go on in a file another arm, adapter or model wrote: its rows would be
    counted as this one's."""
    for row in rows.values():
        theirs = (row["arm"], row["adapter"], row["model"])
        if theirs != (setup.arm, setup.adapter, setup.model_name):
            raise SystemExit(
                f"--out already holds rows of arm {theirs[0]}, adapter {theirs[1]}, model "
                f"{theirs[2]}: give this run another --out"
            )


def record_of(row: Mapping[str, Any]) -> dict[str, Any]:
    """A row as the scorer reads a record (``benchmark/locomo_rescore.py`` 56-82: the
    question, the gold ``answer``, the category's name and ``judged.produced``). A question
    with no answer carries the error and no ``produced``, as a direct-arm row whose
    generation failed does (``benchmark/locomo.py`` 1102-1112): the scorer counts it among
    its ``judge_failures`` and grades it for neither arm."""
    if answered(row):
        judged: dict[str, Any] = {"produced": row["answer"]}
    else:
        judged = {"error": f"{row['status']}: {row['error'] or 'no answer'}"[:300], "detail": ""}
    return {
        "conversation": row["conversation"],
        "category": row["category"],
        "question": row["question"],
        "answer": row["gold"],
        "judged": judged,
        "judge_ruler": None,
        "hit": None,
        "latency_ms": round(row["seconds"] * 1000, 1),
        "harness": {
            name: row.get(name)
            for name in (
                "key", "run_id", "status", "searches", "search_orders", "search_kinds",
                "input_tokens", "output_tokens", "model_calls", "context_chars",
                "evidence_status", "user", "thread", "tenant",
            )
        },
    }  # fmt: skip


def result_of(
    rows: Sequence[Mapping[str, Any]],
    *,
    plan: Plan,
    direct: d.Direct,
    setup: Setup,
    prices: tuple[float | None, float | None] = (None, None),
    stopped: str | None = None,
) -> dict[str, Any]:
    """The run as the memory repository's scorer reads a LoCoMo result: ``judged`` true (it
    holds produced answers, which is what the scorer asks), ``records`` in the plan's order,
    and no score — ``scored`` false until ``benchmark.locomo_rescore`` wrote its own file."""
    tokens_in = sum(r["input_tokens"] for r in rows)
    tokens_out = sum(r["output_tokens"] for r in rows)
    price_in, price_out = prices
    usd = None
    if price_in is not None and price_out is not None:
        usd = round(dollars(tokens_in, tokens_out, price_in, price_out), 4)
    return {
        "dataset": {
            "name": DATASET_NAME,
            "conversations": len(plan.conversations),
            "questions": len(rows),
            "planned_questions": len(plan.questions),
            "sha256": plan.dataset_sha256,
        },
        "judged": True,
        "scored": False,
        "judge_ruler": None,
        "scoring": {
            "note": "answers only, no verdict: grade this file and the direct arm's with the "
            "memory repository's benchmark.locomo_rescore, one ruler and one judge for both. "
            "Per-question detail is in the rows file beside it."
        },
        "arm": {"name": "harness"} | setup.described(),
        # without a conversation's reference date, which differs by conversation
        "instructions_sha256": hashlib.sha256(instructions(direct, setup).encode()).hexdigest(),
        "direct": {
            "source_sha256": direct.source_sha256,
            "commit": direct.commit(),
            "answer_system_sha256": hashlib.sha256(direct.answer_system.encode()).hexdigest(),
        },
        "conversations": list(plan.conversations),
        "asked_by_category": dict(Counter(r["category"] for r in rows)),
        "unanswered": sum(1 for r in rows if not answered(r)),
        # the direct arm keeps 400 characters of an answer; these are whole
        "answers_over_direct_cut": sum(
            1 for r in rows if answered(r) and len(str(r["answer"])) > DIRECT_ANSWER_CHARS
        ),
        "usage": {
            "input_tokens": tokens_in,
            "output_tokens": tokens_out,
            "model_calls": sum(r["model_calls"] for r in rows),
            "searches": sum(r["searches"] for r in rows),
            "seconds": round(sum(r["seconds"] for r in rows), 1),
            "usd": usd,
        },
        "stopped": stopped,
        "records": [record_of(r) for r in rows],
        "provenance": {
            "harness_commit": d.commit_of(d.HERE),
            "written": datetime.now(UTC).isoformat(timespec="seconds"),
        },
    }


def written(out: str, result: Mapping[str, Any]) -> Path:
    path = Path(f"{out}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    return path


def _append(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(row, default=str) + "\n")


# --------------------------------------------------------------------------- the run
#: The variable that holds conversation ``<index>``'s own Trellis key, where one key does
#: not speak for every conversation's tenant (``LOCOMO_API_KEY_0``, ``LOCOMO_API_KEY_1``...).
KEY_ENV: Final = "LOCOMO_API_KEY_{index}"


def harnesses(
    plan: Plan, *, memory: bool, runs: bool = False, environ: Mapping[str, str] | None = None
) -> dict[int, Harness]:
    """The harness each conversation's questions are asked in: the environment's
    (``benchmark.cost.run.harness``: its model endpoint, ``MEMORY_URL``, ``TRELLIS_API_KEY``),
    and for a conversation whose tenant has a key of its own (:data:`KEY_ENV`), one with that
    key. The memory service's keys each speak for one tenant, and a ``--reuse-corpus`` corpus
    is a tenant a conversation."""
    env = os.environ if environ is None else environ
    made: dict[str | None, Harness] = {}
    found: dict[int, Harness] = {}
    for conversation in plan.conversations:
        index = conversation["conversation"]
        key = (env.get(KEY_ENV.format(index=index)) or None) if memory else None
        if key not in made:
            made[key] = runner.harness(memory=memory, runs=runs, api_key=key)
        found[index] = made[key]
    return found


async def in_tenant(h: Harness, tenant: str) -> None:
    """Refuse before any question is asked when the harness's key speaks for another tenant
    than the conversation's: every run would fail the same way."""
    if h.memory is None:
        return
    try:
        await h._tenant(tenant)
    except ConfigurationError as exc:
        raise SystemExit(
            f"{exc}: the conversation is kept in tenant {tenant!r}. A key of the memory "
            f"service speaks for one tenant: give each conversation its own "
            f"({KEY_ENV.format(index='<index>')}), or a key that names no tenant"
        ) from exc


async def bench(
    h: Harness | Mapping[int, Harness],
    *,
    plan: Plan,
    direct: d.Direct,
    setup: Setup,
    out: str,
    concurrency: int = 1,
    prices: tuple[float | None, float | None] = (None, None),
    max_usd: float | None = None,
    progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Ask every question of ``plan`` no row of ``out`` answers yet, ``concurrency`` at a
    time, each written as it ends; the result. ``h``: the harness, or each conversation's own
    (:func:`harnesses`). Asking stops once what was spent reaches ``max_usd`` (at
    ``prices``): the rows stay, and a rerun goes on from them."""
    if concurrency > 1 and setup.adapter not in runner.LANGCHAIN_ADAPTERS:
        progress(f"{setup.adapter}: its model calls are metered one run at a time")
        concurrency = 1
    rows = saved(out)
    same_run(rows, setup)
    todo = [q for q in plan.questions if not answered(rows.get(q.key, {}))]
    progress(f"{len(plan.questions) - len(todo)} of {len(plan.questions)} answered already")
    own = h if isinstance(h, Mapping) else dict.fromkeys((q.conversation for q in todo), h)
    for index, tenant in {q.conversation: q.tenant for q in todo}.items():
        if setup.arm != "off" and own[index].memory is None:
            raise SystemExit(
                f"--arm {setup.arm} asks with memory on, and this harness has none: set "
                "MEMORY_URL and TRELLIS_API_KEY (--arm off is the run without memory)"
            )
        await in_tenant(own[index], tenant)
    agents: dict[int, tuple[Any, Seen]] = {}  # by conversation
    order = {q.key: i for i, q in enumerate(plan.questions)}
    stopped: str | None = None
    lock = asyncio.Lock()

    async def agent_of(question: Asked) -> tuple[Harness, Any, Seen]:
        mine = own[question.conversation]
        async with lock:
            if question.conversation not in agents:
                seen = Seen()
                made = await built(mine, direct, setup, seen, question)
                agents[question.conversation] = (made, seen)
        return (mine, *agents[question.conversation])

    def spent() -> float:
        if prices[0] is None or prices[1] is None:
            return 0.0
        used = list(rows.values())
        return dollars(
            sum(r["input_tokens"] for r in used),
            sum(r["output_tokens"] for r in used),
            prices[0],
            prices[1],
        )

    def result() -> dict[str, Any]:
        kept = sorted((r for k, r in rows.items() if k in order), key=lambda r: order[r["key"]])
        return result_of(
            kept, plan=plan, direct=direct, setup=setup, prices=prices, stopped=stopped
        )

    queue = list(reversed(todo))

    async def worker() -> None:
        nonlocal stopped
        while queue and stopped is None:
            if max_usd is not None and spent() >= max_usd:
                stopped = f"--max-usd {max_usd:g} reached (${spent():.4f} spent)"
                break
            question = queue.pop()
            row = await ask(*await agent_of(question), question, setup)
            async with lock:
                rows[row["key"]] = row
                _append(rows_path(out), row)
                written(out, result())
            progress(
                f"{row['key']:<9} {row['category']:<11} {row['status']:<8} "
                f"searches={row['searches']} calls={row['model_calls']} "
                f"in={row['input_tokens']} out={row['output_tokens']} {row['seconds']}s"
            )

    await asyncio.gather(*(worker() for _ in range(max(1, concurrency))))
    if stopped:
        progress(f"stopped: {stopped}")
    found = result()
    written(out, found)
    return found
