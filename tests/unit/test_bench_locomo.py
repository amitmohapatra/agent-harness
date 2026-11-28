"""LoCoMo through a harness agent (``benchmark.locomo``), offline: a made-up conversation of
three questions, a scripted model, the memory fake, and a stand-in for the memory repository's
checkout — its direct arm's constants read from source, its entry point run for ingestion, the
agent searching memory and answering, the result in the scorer's shape, a rerun asking only
what is not answered, and nothing sent on a dry run or without ``--yes``. Where the memory
repository itself is beside this one, its own sampling and its own scorer are run against
what this runner does and writes."""

from __future__ import annotations

import ast
import asyncio
import copy
import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pytest

from benchmark.cost import complex as cx
from benchmark.locomo import direct as d
from benchmark.locomo import run
from benchmark.locomo.__main__ import arguments, main
from tests.support.memory import AGENT_TOOLS, FakeMemoryService
from tests.support.models import Script, ScriptedChat
from trellis import Harness, Settings

ANSWER_SYSTEM = (
    "You answer questions about a person's life using only the CONTEXT. "
    "If the context never mentions it, reply with exactly: I don't know."
)
#: The stand-in for the memory repository's ``benchmark/locomo.py``: the constants the runner
#: reads, written as the real module writes them (a prompt in parts, a comment between), and
#: an entry point that records how it was called in place of ingesting.
DIRECT_ARM = '''"""A stand-in for the direct arm."""
import json
import os
import sys

CATEGORY_NAMES = {
    1: "multi_hop",
    2: "temporal",
    3: "open_domain",
    4: "single_hop",
    5: "adversarial",
}
TENANT = "bench_conv"
ADVERSARIAL = 5
SAMPLE_SEED = 20240919
_SESSION_TIME = "%I:%M %p on %d %B, %Y"
ANSWER_SYSTEM = (
    "You answer questions about a person's life using only the CONTEXT. "
    # a comment between the parts, as the real prompt has
    "If the context never mentions it, reply with exactly: I don't know."
)

if __name__ == "__main__":
    args = sys.argv[1:]
    out = args[args.index("--out") + 1]
    seen = {
        "argv": args,
        "cwd": os.getcwd(),
        "pythonpath": os.environ.get("PYTHONPATH", "").split(os.pathsep),
        "database": os.environ.get("MEMORY__DATABASE__URL"),
    }
    with open(out + ".seen", "w") as fh:
        json.dump(seen, fh)
    if os.environ.get("STAND_IN_FAILS"):
        sys.stderr.write("no benchmark database\\n")
        sys.exit(3)
    with open(out, "w") as fh:
        json.dump({"ingested_messages": 4, "ingest_seconds": 0.1, "reused_conversations": 0}, fh)
'''
DOG = "What is Mara's dog called?"
RACE = "When did Biscuit run his first race?"
CAT = "What is Jonas's cat called?"
CONVERSATION: dict[str, Any] = {
    "sample_id": "made-up",
    "conversation": {
        "speaker_a": "Mara",
        "speaker_b": "Jonas",
        "session_1_date_time": "1:56 pm on 8 May, 2023",
        "session_1": [
            {"speaker": "Mara", "dia_id": "D1:1", "text": "I adopted a greyhound, Biscuit."},
            {
                "speaker": "Jonas",
                "dia_id": "D1:2",
                "text": "Lovely!",
                "blip_caption": "a dog on a beach",
                "query": "greyhound",
            },
        ],
        "session_2_date_time": "2:00 pm on 9 June, 2023",
        "session_2": [
            {"speaker": "Mara", "dia_id": "D2:1", "text": "Biscuit ran his first race yesterday."},
            {"speaker": "Jonas", "dia_id": "D2:2", "text": ""},  # nothing said: not stored
        ],
        # the tenth session is the last, though "session_10" sorts before "session_2"
        "session_10_date_time": "9:30 am on 2 July, 2023",
        "session_10": [{"speaker": "Jonas", "dia_id": "D10:1", "text": "How is Biscuit?"}],
    },
    "qa": [
        {"question": DOG, "answer": "Biscuit", "evidence": ["D1:1"], "category": 4},
        {"question": RACE, "answer": "8 June 2023", "evidence": ["D2:1"], "category": 2},
        {"question": CAT, "adversarial_answer": "Biscuit", "evidence": [], "category": 5},
        {"question": "", "answer": "never asked", "category": 1},
    ],
}
ANSWERS = {DOG: "Biscuit, a greyhound.", RACE: "On 8 June 2023."}
REFERENCE = "9:30 am on 2 July, 2023"
SEARCH_ARGS = {"kinds": ["memory", "message"], "order": "newest"}


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A stand-in memory checkout: the direct arm's module and the made-up dataset."""
    root = tmp_path / "memory-repo"
    (root / "benchmark" / "data").mkdir(parents=True)
    (root / "benchmark" / "__init__.py").write_text("")
    (root / d.MODULE).write_text(DIRECT_ARM)
    (root / d.DATASET).write_text(json.dumps([CONVERSATION]))
    return root


def _text(message: dict[str, Any]) -> str:
    content = message["content"]
    return content if isinstance(content, str) else " ".join(p["text"] for p in content)


def _question(messages: list[dict[str, Any]]) -> str:
    """The question a request asks: the one its last user message holds."""
    said = next(_text(m) for m in reversed(messages) if m["role"] == "user")
    return next(q for q in (DOG, RACE, CAT) if q in said)


class Answerer(Script):
    """Searches memory for what it is asked, then answers from a table; declines what the
    table does not hold; ``fails``: the questions whose call fails, each once."""

    def __init__(self, fails: tuple[str, ...] = ()) -> None:
        super().__init__([])
        self.fails = list(fails)
        self.asked: list[str] = []

    def next(self, body: dict[str, Any]) -> Any:
        messages = body["messages"]
        question = _question(messages)
        if messages[-1]["role"] != "tool":
            self.asked.append(question)
        if question in self.fails:
            self.fails.remove(question)
            raise IndexError("the endpoint is down")
        if question not in ANSWERS:
            return "I don't know."
        if messages[-1]["role"] == "tool":
            return ANSWERS[question]
        return ("memory_search", {"query": question, **SEARCH_ARGS})


def _memory() -> FakeMemoryService:
    """The memory fake, its key a platform's (it names no tenant: each run does), its search
    tool taking ``order`` as the service's does."""
    tools = copy.deepcopy(AGENT_TOOLS)
    search = next(t for t in tools if t["name"] == run.SEARCH)
    search["input_schema"]["properties"]["order"] = {
        "enum": ["relevance", "newest", "oldest"],
        "type": "string",
    }
    return FakeMemoryService(
        tenant=None,
        context_text="## Memories\n- [2023-05-08] mara: I adopted a greyhound, Biscuit.",
        agent_tools=tools,
    )


def _setup(script: Script, **given: Any) -> run.Setup:
    return run.Setup(model=ScriptedChat(script=script), model_name="scripted", **given)


def _rows(out: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in run.rows_path(out).read_text().splitlines()]


def _result(out: str) -> dict[str, Any]:
    return json.loads(Path(f"{out}.json").read_text())


# --------------------------------------------------------------------------- the direct arm
def test_the_direct_arms_constants_are_read_from_its_source(repo: Path) -> None:
    direct = d.read(repo)
    assert direct.answer_system == ANSWER_SYSTEM
    assert direct.category_names[5] == "adversarial" and direct.category_names[4] == "single_hop"
    assert (direct.sample_seed, direct.tenant) == (20240919, "bench_conv")
    assert direct.dataset == repo / "benchmark/data/locomo10.json"
    # a module that no longer says one of them is refused by name, never guessed
    (repo / d.MODULE).write_text(DIRECT_ARM.replace("SAMPLE_SEED", "SEED"))
    with pytest.raises(SystemExit, match="SAMPLE_SEED"):
        d.read(repo)
    with pytest.raises(SystemExit, match="memory repository"):
        d.read(repo / "elsewhere")


def test_two_runs_never_share_a_user_or_an_agent(repo: Path) -> None:
    """Two arms over one corpus ask the same questions: what one run records (its turns, under
    its users and its agents) is not the other's to read; a rerun of the same ``--out`` is the
    same run, and the direct arm's one user stays that user."""
    from benchmark.locomo.__main__ import setup_of

    asked = run.planned(d.read(repo), count=None, limit=None, environ={}).questions[0]
    push = setup_of(arguments(["--memory-repo", str(repo), "--out", "build/push"]))
    recall = setup_of(arguments(["--memory-repo", str(repo), "--out", "build/on-demand"]))
    again = setup_of(arguments(["--memory-repo", str(repo), "--out", "build/push"]))
    assert push.label and push.label != recall.label and push.label == again.label
    users = {run.user_of(asked, "question", s.label) for s in (push, recall)}
    assert users == {f"locomo-0-q0-{push.label}", f"locomo-0-q0-{recall.label}"}
    assert run.user_of(asked, "conversation", push.label) == "locomo-0"
    assert run.user_of(asked, "question") == "locomo-0-q0"


def test_the_plan_is_the_direct_arms_questions_in_its_tenants(repo: Path) -> None:
    direct = d.read(repo)
    plan = run.planned(direct, count=None, limit=None, environ={})
    assert [q.question for q in plan.questions] == [DOG, RACE, CAT]  # the empty one is not asked
    assert [q.category for q in plan.questions] == ["single_hop", "temporal", "adversarial"]
    assert [q.gold for q in plan.questions] == ["Biscuit", "8 June 2023", ""]
    assert [q.key for q in plan.questions] == ["c0-q0", "c0-q1", "c0-q2"]
    # the last session by its number, and the tenant a --reuse-corpus run keeps it in
    assert {q.reference for q in plan.questions} == {REFERENCE}
    assert {q.tenant for q in plan.questions} == {"bench_conv_0"}
    # three turns said something, one with the caption of the image it shared
    assert plan.conversations[0]["turns"] == 4
    turns = d.turns(CONVERSATION["conversation"])
    assert turns["D1:2"] == "Lovely! [Shared an image of greyhound: a dog on a beach]"
    assert "D2:2" not in turns and list(turns) == ["D1:1", "D1:2", "D2:1", "D10:1"]
    renamed = run.planned(direct, count=1, limit=None, environ={d.TENANT_PREFIX_ENV: "arm_b"})
    assert renamed.questions[0].tenant == "arm_b_0"
    line = f"REFERENCE DATE (the conversation's last session): {REFERENCE}"
    assert d.asked(DOG, REFERENCE) == f"{line}\n\nQUESTION: {DOG}"
    assert d.asked(DOG, "") == f"QUESTION: {DOG}" and d.reference_line("") == ""
    # the date after the instructions (the message is the question alone), or in the message
    asked = plan.questions[0]
    setup = run.Setup()
    assert run.said_to(asked, setup) == DOG
    assert run.instructions(direct, setup, REFERENCE) == f"{ANSWER_SYSTEM}\n\n{run.ADDED}\n\n{line}"
    worded = run.Setup(reference="question", arm="off")
    assert run.said_to(asked, worded) == d.asked(DOG, REFERENCE)
    assert run.instructions(direct, worded, REFERENCE) == ANSWER_SYSTEM
    with pytest.raises(SystemExit, match="--dataset"):
        run.planned(direct, count=1, limit=None, dataset=repo / "none.json")


def test_a_sample_keeps_every_category_and_is_the_same_each_time(repo: Path) -> None:
    direct = d.read(repo)
    questions = [{"question": f"q{n}", "category": 1 + n % 5} for n in range(50)]
    listed = {"qa": questions}
    first = d.questions_of(direct, listed, 10)
    assert first == d.questions_of(direct, listed, 10)
    assert {q["category"] for _, q in first} == {1, 2, 3, 4, 5} and len(first) == 10
    assert all(questions[n] == q for n, q in first)  # each with its place in the list
    assert len(d.questions_of(direct, listed, 0)) == 50 == len(d.questions_of(direct, listed, 99))


def test_ingestion_is_the_direct_arms_own_entry_point(repo: Path, tmp_path: Path) -> None:
    direct = d.read(repo)
    out = tmp_path / "build" / "run.ingest.json"
    env = {**os.environ, "MEMORY__DATABASE__URL": "postgresql://bench/p7_locomo"}
    said: list[str] = []
    kept = d.ingest(
        direct, count=2, python=sys.executable, out=out, environ=env, progress=said.append
    )
    seen = json.loads(Path(f"{out}.seen").read_text())
    # python -m benchmark.locomo in the memory checkout, on its own modules, with the
    # environment that names its stores: one tenant a conversation, kept; no --judge
    assert seen["argv"] == [
        "--reuse-corpus", "--sample", "1", "--out", str(out), "--conversations", "2",
    ]  # fmt: skip
    assert seen["cwd"] == str(repo)
    assert seen["pythonpath"][:3] == [f"{repo}/src", str(repo), f"{repo}/sdk/python/src"]
    assert seen["database"] == "postgresql://bench/p7_locomo"
    assert kept["ingested_messages"] == 4 and "benchmark.locomo --reuse-corpus" in said[0]
    argv, _, cwd = d.ingest_command(direct, count=None, python="py", out=out, environ={})
    assert "--conversations" not in argv and "--judge" not in argv and cwd == repo
    # a direct arm that can store and ask nothing is asked for exactly that
    (repo / d.MODULE).write_text(DIRECT_ARM + 'SWITCHES = ("--ingest-only",)\n')
    argv, _, _ = d.ingest_command(d.read(repo), count=1, python="py", out=out, environ={})
    assert argv[3:5] == ["--reuse-corpus", "--ingest-only"] and "--sample" not in argv
    with pytest.raises(SystemExit, match="no benchmark database"):
        d.ingest(
            direct, count=1, python=sys.executable, out=out, environ=env | {"STAND_IN_FAILS": "1"},
            progress=said.append,
        )  # fmt: skip


# --------------------------------------------------------------------------- asking
@pytest.mark.asyncio
async def test_each_question_is_asked_of_an_agent_that_searches_memory(
    repo: Path, tmp_path: Path
) -> None:
    direct = d.read(repo)
    plan = run.planned(direct, count=1, limit=None, environ={})
    fake = _memory()
    script = Answerer()
    out = str(tmp_path / "build" / "locomo")
    async with Harness(config=Settings(), memory=fake.client()) as h:
        found = await run.bench(
            h, plan=plan, direct=direct, setup=_setup(script), out=out, prices=(0.1, 0.4),
            progress=lambda _: None,
        )  # fmt: skip
    rows = _rows(out)
    assert [r["key"] for r in rows] == ["c0-q0", "c0-q1", "c0-q2"]
    dog, race, cat = rows
    # the answer whole, the tools in order with their arguments, the cost and the run
    assert (dog["answer"], race["answer"], cat["answer"]) == (
        ANSWERS[DOG], ANSWERS[RACE], "I don't know."
    )  # fmt: skip
    assert dog["calls"][0]["tool"] == "memory_search"
    assert dog["calls"][0]["args"] == {"query": DOG, **SEARCH_ARGS}
    assert dog["calls"][0]["status"] == "ok" and dog["calls"][0]["result_chars"] > 0
    assert (dog["searches"], dog["search_orders"], dog["search_kinds"]) == (
        1, ["newest"], [["memory", "message"]]
    )  # fmt: skip
    assert (cat["searches"], cat["calls"], cat["model_calls"]) == (0, [], 1)
    assert dog["model_calls"] == 2 and dog["input_tokens"] == 20 and dog["output_tokens"] == 10
    assert dog["status"] == "SUCCESS" and dog["run_id"] and dog["seconds"] >= 0
    assert dog["context_chars"] > 0 and dog["evidence_status"] == "COMPLETE"
    # memory is asked what the direct arm asks it, the question alone, in the conversation's
    # tenant, each question a user and a thread of its own
    assert dog["asked"] == DOG
    pushed = fake.named("context")
    assert [c.body["query"] for c in pushed] == [DOG, RACE, CAT]
    assert [c.scope["tenant_id"] for c in pushed] == ["bench_conv_0"] * 3
    assert [c.scope["user_id"] for c in pushed] == ["locomo-0-q0", "locomo-0-q1", "locomo-0-q2"]
    assert len({c.scope["thread_id"] for c in pushed}) == 3
    searched = fake.named("call_agent_tool")
    assert [c.body["args"] for c in searched] == [
        {"query": DOG, **SEARCH_ARGS}, {"query": RACE, **SEARCH_ARGS}
    ]  # fmt: skip
    assert searched[0].scope["user_id"] == "locomo-0-q0"
    # the model: the direct arm's prompt, what its CONTEXT is here and the date the question
    # is asked on, in the direct arm's words; the pushed context; the memory tools and no other
    first = script.requests[0]
    system = " ".join(_text(m) for m in first["messages"] if m["role"] == "system")
    assert ANSWER_SYSTEM in system and run.ADDED in system
    assert f"REFERENCE DATE (the conversation's last session): {REFERENCE}" in system
    assert "I adopted a greyhound, Biscuit." in json.dumps(first["messages"])
    offered = {t["function"]["name"] for t in first["tools"]}
    assert offered == {t["name"] for t in AGENT_TOOLS} - {"tool_search"}
    # the result: the scorer's shape, no verdict in it
    result = _result(out)
    assert result == json.loads(json.dumps(found, default=str))
    assert result["judged"] is True and result["scored"] is False
    assert result["dataset"]["questions"] == 3 == result["dataset"]["planned_questions"]
    assert [(r["category"], r["question"], r["answer"]) for r in result["records"]] == [
        ("single_hop", DOG, "Biscuit"),
        ("temporal", RACE, "8 June 2023"),
        ("adversarial", CAT, ""),
    ]
    assert [r["judged"] for r in result["records"]] == [
        {"produced": ANSWERS[DOG]}, {"produced": ANSWERS[RACE]}, {"produced": "I don't know."}
    ]  # fmt: skip
    assert all(r["hit"] is None and r["judge_ruler"] is None for r in result["records"])
    assert result["records"][0]["harness"]["run_id"] == dog["run_id"]
    assert result["records"][0]["harness"]["search_orders"] == ["newest"]
    assert result["arm"]["arm"] == "push" and result["arm"]["adapter"] == "react"
    assert set(run.WITHOUT) <= set(result["arm"]["without"])
    assert result["usage"] == {
        "input_tokens": 50, "output_tokens": 25, "model_calls": 5, "searches": 2,
        "seconds": result["usage"]["seconds"], "usd": round((50 * 0.1 + 25 * 0.4) / 1e6, 4),
    }  # fmt: skip
    assert result["unanswered"] == 0 and result["answers_over_direct_cut"] == 0
    assert result["direct"]["source_sha256"] == direct.source_sha256
    assert result["asked_by_category"] == {"single_hop": 1, "temporal": 1, "adversarial": 1}


@pytest.mark.asyncio
async def test_a_rerun_asks_only_what_has_no_answer(repo: Path, tmp_path: Path) -> None:
    direct = d.read(repo)
    plan = run.planned(direct, count=1, limit=None, environ={})
    out = str(tmp_path / "locomo")
    first = Answerer(fails=(RACE,))
    async with Harness(config=Settings(), memory=_memory().client()) as h:
        found = await run.bench(
            h, plan=plan, direct=direct, setup=_setup(first), out=out, progress=lambda _: None
        )
    # the question whose model call failed has a row and no answer: the scorer is told so
    assert found["unanswered"] == 1 and first.asked == [DOG, RACE, CAT]
    failed = found["records"][1]["judged"]
    assert "produced" not in failed and failed["error"].startswith("ERROR")
    assert _rows(out)[1]["answer"] is None and _rows(out)[1]["status"] == "ERROR"
    # the rerun asks that one and no other, and the result holds all three, in the plan's order
    again = Answerer()
    said: list[str] = []
    async with Harness(config=Settings(), memory=_memory().client()) as h:
        found = await run.bench(
            h, plan=plan, direct=direct, setup=_setup(again), out=out, progress=said.append
        )
    assert again.asked == [RACE] and said[0] == "2 of 3 answered already"
    assert len(_rows(out)) == 4 and found["unanswered"] == 0
    assert [r["judged"].get("produced") for r in found["records"]] == [
        ANSWERS[DOG], ANSWERS[RACE], "I don't know."
    ]  # fmt: skip
    # nothing left: no model is called at all
    idle = Answerer()
    async with Harness(config=Settings(), memory=_memory().client()) as h:
        await run.bench(
            h, plan=plan, direct=direct, setup=_setup(idle), out=out, progress=lambda _: None
        )
        assert idle.requests == [] and len(_rows(out)) == 4
        # another arm's rows are never counted as this one's
        with pytest.raises(SystemExit, match="another --out"):
            await run.bench(
                h, plan=plan, direct=direct, setup=_setup(idle, arm="off"), out=out,
                progress=lambda _: None,
            )  # fmt: skip


@pytest.mark.asyncio
async def test_memory_off_is_the_floor_and_asking_stops_at_the_spend_cap(
    repo: Path, tmp_path: Path
) -> None:
    direct = d.read(repo)
    plan = run.planned(direct, count=1, limit=None, environ={})
    out = str(tmp_path / "off")
    script = Answerer()
    said: list[str] = []
    async with Harness(config=Settings(), memory=False) as h:
        # ten dollars a token: the first question's run is past a one-dollar cap
        setup = _setup(script, arm="off", asker="conversation", reference="question")
        found = await run.bench(
            h, plan=plan, direct=direct, setup=setup, out=out, prices=(1e7, 1e7), max_usd=1.0,
            progress=said.append,
        )  # fmt: skip
    first = script.requests[0]
    # no memory: the direct arm's prompt alone, no context, no tool, the direct arm's own
    # user, and the question worded as the direct arm's message words it
    system = " ".join(_text(m) for m in first["messages"] if m["role"] == "system")
    assert ANSWER_SYSTEM in system and run.ADDED not in system and "REFERENCE" not in system
    assert not first.get("tools") and "greyhound" not in json.dumps(first["messages"])
    (row,) = _rows(out)
    assert (row["user"], row["tenant"], row["searches"]) == ("locomo-0", None, 0)
    assert row["asked"] == d.asked(DOG, REFERENCE) in _text(first["messages"][-1])
    assert found["arm"]["reference"] == "question"
    assert "memory" in found["arm"]["without"] and found["arm"]["added"] == ""
    assert found["stopped"].startswith("--max-usd 1 reached") and said[-1].startswith("stopped")
    assert found["dataset"]["questions"] == 1 and found["dataset"]["planned_questions"] == 3


@pytest.mark.asyncio
async def test_a_key_that_speaks_for_another_tenant_is_refused_before_any_question(
    repo: Path, tmp_path: Path, monkeypatch: Any
) -> None:
    direct = d.read(repo)
    plan = run.planned(direct, count=1, limit=None, environ={})
    script = Answerer()
    out = str(tmp_path / "locomo")
    other = _memory()
    other.tenant = "acme"
    async with Harness(config=Settings(), memory=other.client()) as h:
        with pytest.raises(SystemExit, match=r"speaks for 'acme'.*LOCOMO_API_KEY_<index>"):
            await run.bench(
                h, plan=plan, direct=direct, setup=_setup(script), out=out,
                progress=lambda _: None,
            )  # fmt: skip
    # nor is a memory arm ever asked of a harness that has no memory service
    async with Harness(config=Settings(), memory=False) as h:
        with pytest.raises(SystemExit, match="MEMORY_URL"):
            await run.bench(
                h, plan=plan, direct=direct, setup=_setup(script), out=out,
                progress=lambda _: None,
            )  # fmt: skip
    assert script.requests == [] and not run.rows_path(out).exists()
    # the conversation's own tenant's key asks, each conversation in the harness of its key
    own = _memory()
    own.tenant = "bench_conv_0"
    async with Harness(config=Settings(), memory=own.client()) as h:
        found = await run.bench(
            {0: h}, plan=plan, direct=direct, setup=_setup(script), out=out,
            progress=lambda _: None,
        )  # fmt: skip
    assert found["unanswered"] == 0 and len(_rows(out)) == 3
    # the environment's harness, and one more for a conversation whose key is its own
    two = run.Plan(plan.questions, (*plan.conversations, {"conversation": 1}), "")
    monkeypatch.setenv("MEMORY_URL", "http://memory.test")
    monkeypatch.setenv("TRELLIS_API_KEY", "the-environments")
    env = {run.KEY_ENV.format(index=1): "conversation-ones"}
    made = run.harnesses(two, memory=True, environ=env)
    assert made[0] is not made[1]
    assert [made[i].settings.api_key for i in (0, 1)] == ["the-environments", "conversation-ones"]
    shared = run.harnesses(two, memory=True, environ={})
    assert shared[0] is shared[1] and shared[0].memory is not None
    off = run.harnesses(two, memory=False, environ=env)
    assert off[0] is off[1] and off[0].memory is None
    for h in (made[0], made[1], shared[0], off[0]):
        await h.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter", ["langgraph", "deepagents", "graph"])
async def test_the_other_langchain_adapters_ask_the_same_way(
    adapter: str, repo: Path, tmp_path: Path
) -> None:
    direct = d.read(repo)
    plan = run.planned(direct, count=1, limit=None, environ={})
    plan = run.Plan(plan.questions[:1], plan.conversations, plan.dataset_sha256)
    fake = _memory()
    script = Answerer()
    out = str(tmp_path / adapter)
    async with Harness(config=Settings(), memory=fake.client()) as h:
        await run.bench(
            h, plan=plan, direct=direct, setup=_setup(script, adapter=adapter), out=out,
            concurrency=4, progress=lambda _: None,
        )  # fmt: skip
    (row,) = _rows(out)
    assert (row["adapter"], row["answer"], row["searches"]) == (adapter, ANSWERS[DOG], 1)
    assert row["model_calls"] == 2 and row["calls"][0]["args"]["order"] == "newest"
    offered = {t["function"]["name"] for t in script.requests[0]["tools"]}
    assert run.SEARCH in offered and not offered & {"ask_user", "schedule_task", "sandbox_exec"}


def test_every_adapter_of_the_harness_can_be_chosen(tmp_path: Path, monkeypatch: Any) -> None:
    assert run.ADAPTERS == cx.ADAPTERS
    assert arguments([]).adapter == "react"
    monkeypatch.setenv("KIMI_RUN_ROOT", str(tmp_path / "kimi"))
    monkeypatch.setenv("CLAUDE_RUN_ROOT", str(tmp_path / "claude"))
    # each framework is given the benchmark's instructions in its own slot
    openai, _ = cx.target_of("m", "memory", None, 8, adapter="openai", tools=(), system="S")
    assert openai.instructions == "S"
    claude, _ = cx.target_of("m", "memory", None, 8, adapter="claude", tools=(), system="S")
    assert claude.system_prompt == "S"
    cx.target_of("m", "memory", None, 8, adapter="kimi", tools=(), system="S")
    assert (tmp_path / "kimi" / "cwd" / "AGENTS.md").read_text() == "S"
    with pytest.raises(SystemExit):
        arguments(["--adapter", "crewai"])


# --------------------------------------------------------------------------- spend control
def test_the_estimate_counts_calls_and_tokens_and_prices_them(repo: Path) -> None:
    direct = d.read(repo)
    plan = run.planned(direct, count=1, limit=None, environ={})
    assumed = run.Assumed(
        searches=2, result_tokens=1000, tools_tokens=500, notes_tokens=100, question_tokens=50,
        output_tokens=100,
    )  # fmt: skip
    system = "x" * 400  # a hundred tokens
    push = run.estimate(
        plan, arm="push", system=system, window=16_000, assumed=assumed, price_in=1.0,
        price_out=2.0,
    )  # fmt: skip
    # a 16,000-token window: 4,000 of pushed context; three requests a question, the second
    # and third carrying the searches before them
    each = 3 * (100 + 50 + 500 + 100 + 4000) + (1000 + 100) * 3
    assert push["per_question"] == {
        "model_calls": 3.0, "input_tokens": each, "output_tokens": 300, "context_tokens": 4000
    }  # fmt: skip
    assert (push["questions"], push["model_calls"]) == (3, 9)
    assert (push["input_tokens"], push["output_tokens"]) == (3 * each, 900)
    assert push["usd"] == round((3 * each * 1.0 + 900 * 2.0) / 1e6, 4)
    assert push["by_category"] == {"single_hop": 1, "temporal": 1, "adversarial": 1}
    assert push["judge"]["calls"] == 3 and push["turns"] == 4
    off = run.estimate(plan, arm="off", system=system, window=None, assumed=assumed)
    assert (off["model_calls"], off["input_tokens"], off["output_tokens"]) == (3, 450, 300)
    assert off["usd"] is None and "not priced" in run.said(off)
    text = run.said(push)
    assert "questions           3 (adversarial 1, single_hop 1, temporal 1)" in text
    assert f"${push['usd']:.4f} at $1 in / $2 out per million tokens" in text


def test_a_dry_run_prints_the_estimate_and_sends_nothing(repo: Path, tmp_path: Path) -> None:
    fake = _memory()
    script = Answerer()
    out = str(tmp_path / "dry")
    said: list[str] = []
    args = [
        "--memory-repo", str(repo), "--out", out, "--ingest", "direct",
        "--memory-python", sys.executable, "--price-in", "0.1", "--price-out", "0.4",
    ]  # fmt: skip
    h = Harness(config=Settings(), memory=fake.client())
    code = main(
        [*args, "--dry-run", "--yes"], model=ScriptedChat(script=script), harness=h,
        progress=said.append,
    )  # fmt: skip
    text = "\n".join(said)
    assert code == 0 and said[-1] == "dry run: nothing was sent"
    assert "conversations       1 (4 stored turns)" in text
    assert "questions           3 (adversarial 1, single_hop 1, temporal 1)" in text
    assert "model calls (est.)  9 (3 a question)" in text
    assert "input tokens (est.)" in text and "cost (est.)         $" in text
    # without --yes a run stops at the same place, and says what would start it
    code = main(args, model=ScriptedChat(script=script), harness=h, progress=said.append)
    assert code == 2 and said[-1].startswith("nothing was sent: --yes")
    assert script.requests == [] and fake.calls == []
    assert list(tmp_path.glob("dry*")) == []  # no result, no row, no ingestion


def test_a_run_with_yes_ingests_then_asks(repo: Path, tmp_path: Path) -> None:
    fake = _memory()
    script = Answerer()
    out = str(tmp_path / "paid")
    said: list[str] = []
    h = Harness(config=Settings(), memory=fake.client())
    code = main(
        [
            "--memory-repo", str(repo), "--out", out, "--ingest", "direct", "--yes",
            "--memory-python", sys.executable, "--limit", "2", "--concurrency", "2",
            "--without", "judges", "--run", 'framework_options={"recursion_limit": 40}',
        ],
        model=ScriptedChat(script=script),
        harness=h,
        progress=said.append,
    )  # fmt: skip
    assert code == 0
    seen = json.loads(Path(f"{out}.ingest.json.seen").read_text())
    assert seen["argv"][:3] == ["--reuse-corpus", "--sample", "1"]
    result = _result(out)
    # --limit 2 of three categories is one of each, as the direct arm's --sample 2 is
    assert sorted(r["category"] for r in result["records"]) == [
        "adversarial", "single_hop", "temporal"
    ]  # fmt: skip
    assert result["arm"]["run"] == {"framework_options": {"recursion_limit": 40}}
    assert result["unanswered"] == 0 and len(_rows(out)) == 3
    assert said[-1].startswith(f"wrote {out}.json")
    # a run that left a question without an answer says so in its status
    failing = Answerer(fails=(DOG,))
    code = main(
        ["--memory-repo", str(repo), "--out", str(tmp_path / "part"), "--yes"],
        model=ScriptedChat(script=failing),
        harness=Harness(config=Settings(), memory=_memory().client()),
        progress=said.append,
    )
    assert code == 1 and "1 without an answer" in said[-2]
    with pytest.raises(SystemExit):
        arguments(["--wrap", "no-equals-sign"])
    with pytest.raises(SystemExit):
        arguments(["--max-usd", "1"])  # a cap in dollars needs a price
    assert arguments(["--wrap", "version=v2", "--wrap", "rewindable=true"]).wrap == {
        "version": "v2", "rewindable": True
    }  # fmt: skip


# --------------------------------------------------------------------------- the real checkout
REAL = d.default_repo()
REAL_PYTHON = d.default_python(REAL)
needs_checkout = pytest.mark.skipif(
    not (REAL / d.MODULE).is_file(),
    reason=f"no memory repository at {REAL} ({d.REPO_ENV})",
)


def _own(name: str) -> Any:
    """A function of the real direct arm, from its source (its module imports the memory
    service, which this environment does not have)."""
    source = (REAL / d.MODULE).read_text()
    wanted = {name, "_SESSION", "SAMPLE_SEED"}
    body = [
        node
        for node in ast.parse(source).body
        if getattr(node, "name", None) in wanted
        or any(getattr(t, "id", None) in wanted for t in getattr(node, "targets", []))
    ]
    scope: dict[str, Any] = {"random": random, "defaultdict": defaultdict, "re": re}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(REAL / d.MODULE), "exec"), scope)
    return scope[name]


@needs_checkout
def test_the_sample_and_the_sessions_are_the_real_direct_arms() -> None:
    direct = d.read(REAL)
    assert "I don't know" in direct.answer_system and direct.category_names[5] == "adversarial"
    theirs, sessions = _own("_stratified"), _own("_sessions")
    made_up = [{"question": f"q{n}", "category": 1 + (n * n) % 5} for n in range(199)]
    data = [{"conversation": CONVERSATION["conversation"], "qa": made_up}]
    if direct.dataset.is_file():
        data += json.loads(direct.dataset.read_text())[:3]
    for item in data:
        assert d.sessions(item["conversation"]) == sessions(item["conversation"])
        for size in (1, 2, 7, 40, 10_000):
            ours = [q for _, q in d.questions_of(direct, item, size)]
            assert ours == [q for q in theirs(list(item["qa"]), size) if q.get("question")]


#: The memory repository's own scorer over a result, its judge and its container stood in
#: for: every line of ``rescore`` runs but the model call.
RESCORE = """
import asyncio, json, sys
import benchmark.locomo_rescore as rescorer

class Container:
    class llm:
        enabled = True
        @staticmethod
        async def resolve_model(use):
            return "stand-in"
    async def close(self):
        pass

async def build_container(*args, **kwargs):
    return Container()

async def judge(llm, question, gold, got, *, ruler="strict"):
    declined = "don't know" in got
    return {"correct": bool(gold) and str(gold) in got, "abstained": declined, "reason": ""}

rescorer.build_container = build_container
rescorer._judge = judge
result = json.loads(open(sys.argv[1]).read())
assert result.get("judged"), "the scorer refuses a result that was not judged"
out = asyncio.run(rescorer.rescore(result, "strict", 1e9))
json.dump(out, sys.stdout)
"""


@needs_checkout
@pytest.mark.skipif(not Path(REAL_PYTHON).is_file(), reason=f"no interpreter {REAL_PYTHON}")
@pytest.mark.asyncio
async def test_the_memory_repositorys_scorer_scores_the_result(repo: Path, tmp_path: Path) -> None:
    direct = d.read(repo)
    plan = run.planned(direct, count=1, limit=None, environ={})
    out = str(tmp_path / "locomo")
    async with Harness(config=Settings(), memory=_memory().client()) as h:
        await run.bench(
            h, plan=plan, direct=direct, setup=_setup(Answerer(fails=(RACE,))), out=out,
            progress=lambda _: None,
        )  # fmt: skip
    script = tmp_path / "rescore.py"
    script.write_text(RESCORE)
    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": os.pathsep.join([f"{REAL}/src", str(REAL), f"{REAL}/sdk/python/src"]),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    done = await asyncio.create_subprocess_exec(
        REAL_PYTHON, "-P", str(script), f"{out}.json", env=env, cwd=tmp_path,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )  # fmt: skip
    said, errors = await done.communicate()
    assert done.returncode == 0, errors.decode()[-2000:]
    scored = json.loads(said)
    # its rule: a match of the gold answer, and for the adversarial question declining; the
    # question with no answer is a judge failure, graded for no arm
    assert scored["by_category"] == {
        "single_hop": {"n": 1, "score": 1.0}, "adversarial": {"n": 1, "score": 1.0}
    }  # fmt: skip
    assert scored["judge_failures"] == 1 and scored["judge_ruler"] == "strict"
    assert [r["hit"] for r in scored["records"]] == [True, None, True]
    assert scored["records"][0]["harness"]["searches"] == 1  # what the run kept rides along


def test_the_full_arm_reads_the_whole_conversation_and_has_no_memory(repo: Path) -> None:
    """``--arm full``: the ceiling of reading, the whole conversation in its instructions as
    the CONTEXT (every session in order, with its date, each turn as it was stored) and no
    memory service, tool or context at all."""
    direct = d.read(repo)
    asked = run.planned(direct, count=None, limit=None, environ={}).questions[0]
    full = run.Setup(arm="full")
    assert "memory" in full.off()
    said = run.instructions(
        direct, full, asked.reference, d.transcript(CONVERSATION["conversation"])
    )
    assert said.startswith(ANSWER_SYSTEM) and run.ADDED not in said
    assert "CONTEXT (the whole conversation" in said
    for _, session in d.sessions(CONVERSATION["conversation"]):
        for turn in session:
            if d.turn_text(turn):
                assert d.turn_text(turn) in said
    assert run.instructions(direct, run.Setup(arm="off"), asked.reference) == run.instructions(
        direct, run.Setup(arm="off"), asked.reference, d.transcript(CONVERSATION["conversation"])
    ), "only the full arm reads the conversation"
