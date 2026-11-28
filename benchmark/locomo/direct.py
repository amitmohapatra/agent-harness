"""What the direct arm does — the memory service's own LoCoMo benchmark
(``benchmark/locomo.py`` in the memory repository) — as this runner needs it, so that both
arms read the same memory and ask the same questions.

**Read from the memory checkout, never copied**: the answer prompt (``ANSWER_SYSTEM``), the
category names, the sample's seed and the benchmark tenant
(:func:`read`: the module's source, parsed, not imported — its imports are the memory
service's, which this environment does not have).

**Run from the memory checkout, never mirrored**: ingestion (:func:`ingest_command`). The
direct arm writes each turn in process (``_ingest_conversation``, ``benchmark/locomo.py``
152-212, through ``submit_observation``, ``benchmark/common.py`` 270-321): an observation of
kind MESSAGE whose user is the speaker's own name lower-cased, in the benchmark tenant and
workspace ``ws``, with TENANT visibility, ``occurred_at`` the session's date (UTC) and no
thread. No HTTP route stores that (``submit_observation``'s own words: "which no public route
offers"): ``POST /v1/messages`` needs a thread and takes no audience, ``POST /v1/memories``
stores a memory, not an observation the pipeline reads. So the memory SDK cannot mirror it,
and the corpus is written by the direct arm's own entry point, in its own interpreter.

**Mirrored, a few pure lines each** (cited where they are; ``tests/unit/test_bench_locomo.py``
runs them against the checkout's own functions when one is there): the order of a
conversation's sessions, its reference date, a turn's text, the seeded per-category sample,
the tenant and the asking user of a conversation, and how a question and its reference date
are worded.
"""

from __future__ import annotations

import ast
import functools
import hashlib
import json
import os
import random
import re
import subprocess
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

HERE: Final = Path(__file__).resolve().parents[2]
#: Where the memory repository is: ``LOCOMO_MEMORY_REPO``, else the checkout beside this one
#: (as ``scripts/scenarios/stack.sh`` finds it).
REPO_ENV: Final = "LOCOMO_MEMORY_REPO"
#: Its interpreter (``LOCOMO_MEMORY_PYTHON``), else its own virtual environment's.
PYTHON_ENV: Final = "LOCOMO_MEMORY_PYTHON"
#: The direct arm's module and its dataset, in the memory repository.
MODULE: Final = "benchmark/locomo.py"
DATASET: Final = "benchmark/data/locomo10.json"
#: The module-level names :func:`read` takes from the direct arm's source.
NAMES: Final = ("CATEGORY_NAMES", "SAMPLE_SEED", "ANSWER_SYSTEM", "TENANT")
#: The workspace every direct-arm context is in (``benchmark/locomo.py`` 970-972). A harness
#: run names no workspace; the corpus is TENANT-visible, which no workspace narrows.
WORKSPACE: Final = "ws"
#: The variable that renames the per-conversation tenants (``benchmark/corpus.py`` 34-38).
TENANT_PREFIX_ENV: Final = "BENCH_CORPUS_TENANT_PREFIX"
_SESSION: Final = re.compile(r"^session_(\d+)$")  # benchmark/locomo.py 76


def default_repo() -> Path:
    return Path(os.environ.get(REPO_ENV) or HERE.parent / "agent-memory-service")


def default_python(repo: Path) -> str:
    return os.environ.get(PYTHON_ENV) or str(repo / ".venv" / "bin" / "python")


@dataclass(frozen=True)
class Direct:
    """The direct arm's own constants, as its source says them."""

    root: Path
    #: the system prompt its answers are generated with (``benchmark/locomo.py`` 574-607)
    answer_system: str
    #: LoCoMo's category numbers, named (63-69)
    category_names: dict[int, str]
    #: the seed of the per-category sample (85)
    sample_seed: int
    #: the benchmark's tenant (71); a conversation's own is ``<prefix>_<index>``
    tenant: str
    #: the digest of the source these were read from, kept with every result
    source_sha256: str
    #: whether its entry point can store the conversations and ask nothing (:data:`INGEST_ONLY`)
    ingest_only: bool = False

    @property
    def dataset(self) -> Path:
        return self.root / DATASET

    def commit(self) -> str | None:
        """The memory checkout's commit (``None`` where git cannot say)."""
        return commit_of(self.root)


@functools.cache
def commit_of(root: Path) -> str | None:
    """The commit a checkout is at, asked of git once (``None`` where git cannot say)."""
    try:
        said = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return said.stdout.strip() or None


def read(root: Path) -> Direct:
    """The direct arm's constants, from ``<root>/benchmark/locomo.py``."""
    path = root / MODULE
    if not path.is_file():
        raise SystemExit(
            f"{path} is missing: --memory-repo (or {REPO_ENV}) names the memory repository's "
            "checkout, whose benchmark/locomo.py is the direct arm"
        )
    source = path.read_text(encoding="utf-8")
    found: dict[str, Any] = {}
    for node in ast.parse(source).body:
        target = (
            node.targets[0]
            if isinstance(node, ast.Assign) and len(node.targets) == 1
            else getattr(node, "target", None)
        )
        value = getattr(node, "value", None)
        if isinstance(target, ast.Name) and target.id in NAMES and value is not None:
            found[target.id] = ast.literal_eval(value)
    missing = [name for name in NAMES if name not in found]
    if missing:
        raise SystemExit(f"{path} no longer defines {', '.join(missing)}: the direct arm changed")
    return Direct(
        root=root,
        answer_system=found["ANSWER_SYSTEM"],
        category_names={int(k): str(v) for k, v in found["CATEGORY_NAMES"].items()},
        sample_seed=int(found["SAMPLE_SEED"]),
        tenant=found["TENANT"],
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        ingest_only=f'"{INGEST_ONLY}"' in source,
    )


# --------------------------------------------------------------------------- the dataset
def conversations(
    direct: Direct, count: int | None, dataset: Path | None = None
) -> tuple[list[dict[str, Any]], str]:
    """The first ``count`` conversations (``None``: all) as the direct arm takes them
    (``benchmark/locomo.py`` 913-914), and the dataset's digest. ``dataset``: the file, where
    it is not in the memory checkout itself (it is git-ignored there: a worktree has none)."""
    path = dataset or direct.dataset
    if not path.is_file():
        raise SystemExit(
            f"{path} is missing: run `make bench-locomo-prepare` in the memory repository, or "
            "name the file with --dataset"
        )
    raw = path.read_bytes()
    data = json.loads(raw)
    return (data[:count] if count else data), hashlib.sha256(raw).hexdigest()


def sessions(conversation: Mapping[str, Any]) -> list[tuple[str, list[dict[str, Any]]]]:
    """A conversation's sessions in their own order (``_sessions``, 106-111)."""
    keys = sorted(
        (k for k in conversation if _SESSION.match(k)),
        key=lambda k: int(k.rsplit("_", 1)[1]),
    )
    return [(k, conversation[k]) for k in keys if isinstance(conversation[k], list)]


def reference_date(conversation: Mapping[str, Any]) -> str:
    """The date a question is asked on: the last session's, as the dataset writes it
    (998-1002). Empty when the conversation has no session."""
    found = sessions(conversation)
    return conversation.get(f"{found[-1][0]}_date_time", "") if found else ""


def turn_text(turn: Mapping[str, Any]) -> str:
    """What a turn is stored as (163-171; ``benchmark/corpus.py`` 46-56): its text, and the
    caption of an image it shared."""
    body = turn.get("text") or turn.get("clean_text") or ""
    if caption := (turn.get("blip_caption") or "").strip():
        query = (turn.get("query") or "").strip()
        body = f"{body} [Shared an image{f' of {query}' if query else ''}: {caption}]".strip()
    return body


def transcript(conversation: Mapping[str, Any]) -> str:
    """The whole conversation as a reader of it has it: every session in its order with its
    date, each turn as it was stored (``turn_text``), its speaker first."""
    lines: list[str] = []
    for key, session in sessions(conversation):
        lines.append(
            f"Session {key.rsplit('_', 1)[1]} - {conversation.get(f'{key}_date_time', '')}"
        )
        lines += [f"{t.get('speaker', '')}: {body}" for t in session if (body := turn_text(t))]
        lines.append("")
    return "\n".join(lines).strip()


def turns(conversation: Mapping[str, Any]) -> dict[str, str]:
    """``dia_id`` to the text each stored turn was submitted as (``turn_texts``)."""
    found: dict[str, str] = {}
    for _, session in sessions(conversation):
        for turn in session:
            body = turn_text(turn)
            if turn.get("dia_id") and body:
                found[turn["dia_id"]] = body
    return found


def stratified(questions: list[dict[str, Any]], size: int, seed: int) -> list[dict[str, Any]]:
    """``size`` questions drawn from each category in proportion, never none of one, with the
    direct arm's seed (``_stratified``, 88-103): its ``--sample``."""
    if size <= 0 or size >= len(questions):
        return questions
    buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for question in questions:
        buckets[int(question.get("category", 0))].append(question)
    rng = random.Random(seed)
    picked: list[dict[str, Any]] = []
    for category in sorted(buckets):
        group = buckets[category]
        share = max(1, round(size * len(group) / len(questions)))
        picked.extend(rng.sample(group, min(share, len(group))))
    return picked


def tenant_of(direct: Direct, index: int, environ: Mapping[str, str] | None = None) -> str:
    """The tenant conversation ``index`` is kept in by a ``--reuse-corpus`` run
    (``conversation_tenant``, ``benchmark/corpus.py`` 34-38): the only kind whose corpus
    outlives the run. (Without it every conversation goes through the one tenant, emptied
    before each: 968 and 992.)"""
    env = os.environ if environ is None else environ
    return f"{env.get(TENANT_PREFIX_ENV) or direct.tenant}_{index}"


def asker(index: int) -> str:
    """The user the direct arm asks conversation ``index``'s questions as (970-972): none of
    its speakers, and one with no memory of their own."""
    return f"locomo-{index}"


def reference_line(reference: str) -> str:
    """How the direct arm tells its model the date a question is asked on (800): empty when
    the conversation has none."""
    return f"REFERENCE DATE (the conversation's last session): {reference}" if reference else ""


def asked(question: str, reference: str) -> str:
    """A question as the direct arm's user message words it (798-803), less the ``CONTEXT``
    block: a harness agent's context is what the harness pushes and what its tools return."""
    line = reference_line(reference)
    return (f"{line}\n\n" if line else "") + f"QUESTION: {question}"


# --------------------------------------------------------------------------- ingestion
#: What the direct arm's own entry point is asked for when it only has to fill the corpus:
#: one tenant per conversation, kept and vouched for by its ledger (``--reuse-corpus``), and
#: the fewest questions it can be asked (``--sample 1``: one per category, retrieval only —
#: without ``--judge`` it calls no model). It has no switch that asks none; a checkout that
#: gained one (:data:`INGEST_ONLY`, the change proposed with this runner) is given that.
INGEST_ARGS: Final = ("--reuse-corpus", "--sample", "1")
INGEST_ONLY: Final = "--ingest-only"


def ingest_command(
    direct: Direct,
    *,
    count: int | None,
    python: str,
    out: Path,
    environ: Mapping[str, str] | None = None,
) -> tuple[list[str], dict[str, str], Path]:
    """The command, environment and folder that ingest the first ``count`` conversations as
    the direct arm does, because it is the direct arm: ``python -m benchmark.locomo`` in the
    memory checkout, on the stores its environment names (``MEMORY__DATABASE__URL``,
    ``MEMORY__SEARCH__QDRANT_URL``, ``BENCH_*``: passed through unchanged, so the corpus
    key its ledger keeps is the one a later judged run computes). ``out``: where its own
    (retrieval-only) result goes, kept out of the memory checkout."""
    env = dict(os.environ if environ is None else environ)
    root = str(direct.root)
    env["PYTHONPATH"] = os.pathsep.join(
        [f"{root}/src", root, f"{root}/sdk/python/src", *filter(None, [env.get("PYTHONPATH")])]
    )
    how = ("--reuse-corpus", INGEST_ONLY) if direct.ingest_only else INGEST_ARGS
    argv = [python, "-m", "benchmark.locomo", *how, "--out", str(out.resolve())]
    if count:
        argv += ["--conversations", str(count)]
    return argv, env, direct.root


def ingest(
    direct: Direct,
    *,
    count: int | None,
    python: str,
    out: Path,
    environ: Mapping[str, str] | None = None,
    progress: Any = print,
) -> dict[str, Any]:
    """Ingest through the direct arm's entry point (:func:`ingest_command`); what it said of
    its run (its result's summary), or ``SystemExit`` with its error."""
    argv, env, cwd = ingest_command(direct, count=count, python=python, out=out, environ=environ)
    out.parent.mkdir(parents=True, exist_ok=True)
    progress(f"ingesting as the direct arm does: {' '.join(argv[1:])} (in {cwd})")
    done = subprocess.run(argv, env=env, cwd=cwd, capture_output=True, text=True, check=False)
    if done.returncode != 0:
        raise SystemExit(
            f"the direct arm's ingestion failed ({done.returncode}):\n{done.stderr[-2000:]}"
        )
    said = json.loads(out.read_text()) if out.is_file() else {}
    kept = ("ingest_seconds", "ingested_messages", "reused_conversations", "aborted", "dataset")
    return {name: said.get(name) for name in kept}


def questions_of(
    direct: Direct, conversation: Mapping[str, Any], limit: int | None
) -> list[tuple[int, dict[str, Any]]]:
    """The questions the direct arm asks of a conversation, each with its place in the
    conversation's own list: all of them, or its ``--sample`` of ``limit`` (1004-1008), a
    question with no text left out (1023-1024)."""
    listed: Sequence[dict[str, Any]] = conversation.get("qa", [])
    numbered = [{**item, "_n": n} for n, item in enumerate(listed)]
    picked = stratified(numbered, limit, direct.sample_seed) if limit else numbered
    return [
        (item["_n"], {k: v for k, v in item.items() if k != "_n"})
        for item in picked
        if item.get("question")
    ]
