"""Re-check the targeted refactor plan (``../refactor-plan-2026-11-15.md``) on a newer main.

Standard library only; run from the repository root with any Python 3.12+:

    python docs/research/refactor-plan-2026-11-15/recheck.py numbers
    python docs/research/refactor-plan-2026-11-15/recheck.py copies
    python docs/research/refactor-plan-2026-11-15/recheck.py overlap origin/main origin/<branch>...
    python docs/research/refactor-plan-2026-11-15/recheck.py overlap <main sha> <checkout dir>

``numbers``: lines of src; cyclomatic complexity counted as the memory service's
``tests/unit/test_complexity_budget.py`` counts it (functions, mean, the ones above 15, the
worst, the longest); duplicate blocks as pylint's ``duplicate-code`` finds them (stripped lines,
comments, docstrings, imports and def signatures left out; more than ``MIN_LINES`` such lines
in a row seen twice), within a file too.

``copies``: where each copy the plan folds is now, by file, function and line, found by what
the copy does (so a moved or renamed one is still found), and every site that still stops an
attempt, lists a run's children or opens an attempt's helpers by hand.

``overlap``: for each ref, the functions of the base its diff touches that the refactor also
touches (``git diff -U0 base...ref``, mapped onto the base's functions).
"""

from __future__ import annotations

import ast
import io
import re
import subprocess
import sys
import tokenize
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

SRC = Path("src/trellis")
HARNESS = SRC / "harness"
MIN_LINES = 4
_BRANCHING = (
    ast.If,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.ExceptHandler,
    ast.With,
    ast.AsyncWith,
    ast.Assert,
    ast.IfExp,
    ast.comprehension,
    ast.match_case,
)
#: the functions the refactor changes or deletes (file under src/trellis/harness: qualname)
TARGETS = {
    "agent.py": {"Agent.cancel", "Agent.stop", "Agent._continue"},
    "harness.py": {"<module>", "Harness._lost"},
    "conversation.py": {"<module>"},
    "subagents.py": {"cancel_children"},
    "pipeline.py": {
        "_arrived",
        "_left_the_line",
        "_cancelled",
        "_children_ended",
        "_concluded",
        "_open_children_ended",
        "_settle_cancelled",
    },
    "background.py": {
        "INTERRUPTED",
        "Tasks.stop",
        "Tasks.aclose",
        "follow_up",
        "stop",
        "abandoned",
    },
    "bridges/live.py": {
        "Kept.cancel",
        "Kept.served",
        "Kept.abandoned",
        "Keeper._detach",
    },
    "bridges/async_tasks.py": {"AsyncTasks._closed", "AsyncTasks.cancelled", "AsyncTasks.stopped"},
    "bridges/helpers.py": {"<module>", "Helpers.__init__", "Helpers.closed", "Helpers.suspended"},
    "adapters/claude.py": {"ClaudeAdapter._helped", "ClaudeAdapter.__init__"},
    "adapters/acp.py": {"ACPAdapter._turns", "ACPAdapter.__init__"},
    "adapters/openai_agents.py": {
        "OpenAIAgentsAdapter.invoke",
        "OpenAIAgentsAdapter.stream",
        "OpenAIAgentsAdapter._prepared",
        "_closed",
    },
    "adapters/langgraph.py": {
        "LangGraphAdapter.invoke",
        "LangGraphAdapter.stream",
        "LangGraphAdapter.child_cancelled",
    },
}


def say(text: str = "") -> None:
    sys.stdout.write(text + "\n")


# --------------------------------------------------------------------------- functions
def functions(root: Path) -> Iterator[tuple[Path, str, ast.FunctionDef | ast.AsyncFunctionDef]]:
    """Every function under ``root``, with its qualified name (``Class.method``, nested too)."""
    for path in sorted(root.rglob("*.py")):

        def walk(node: ast.AST, prefix: str) -> Iterator[tuple[str, ast.AST]]:
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                    yield f"{prefix}{child.name}", child
                    yield from walk(child, f"{prefix}{child.name}.")
                else:
                    yield from walk(child, prefix)

        for name, node in walk(ast.parse(path.read_text(encoding="utf-8")), ""):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                yield path, name, node


def complexity(fn: ast.AST) -> int:
    score = 1
    for node in ast.walk(fn):
        if isinstance(node, _BRANCHING):
            score += 1
        elif isinstance(node, ast.BoolOp):
            score += len(node.values) - 1
    return score


# --------------------------------------------------------------------------- duplicates
def _kept_lines(path: Path) -> list[tuple[int, str]]:
    text = path.read_text(encoding="utf-8")
    dropped: set[int] = set()
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.Import | ast.ImportFrom):
            dropped.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            start = min([node.lineno, *(d.lineno for d in node.decorator_list)])
            dropped.update(range(start, node.body[0].lineno))
        body = getattr(node, "body", None)
        if (
            isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
            and body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            dropped.update(range(body[0].lineno, (body[0].end_lineno or body[0].lineno) + 1))
    comments = {
        tok.start[0]: tok.start[1]
        for tok in tokenize.generate_tokens(io.StringIO(text).readline)
        if tok.type == tokenize.COMMENT
    }
    kept: list[tuple[int, str]] = []
    for number, line in enumerate(text.splitlines(), 1):
        if number in dropped:
            continue
        stripped = (line[: comments[number]] if number in comments else line).strip()
        if stripped:
            kept.append((number, stripped))
    return kept


def duplicates(root: Path) -> list[tuple[int, str, str]]:
    """Each run of more than :data:`MIN_LINES` kept lines seen twice: (its length, where, where
    again), the longest first."""
    files = {str(p.relative_to(root)): _kept_lines(p) for p in sorted(root.rglob("*.py"))}
    index: dict[tuple[str, ...], list[tuple[str, int]]] = defaultdict(list)
    for name, kept in files.items():
        for i in range(len(kept) - MIN_LINES + 1):
            index[tuple(line for _, line in kept[i : i + MIN_LINES])].append((name, i))
    found: list[tuple[int, str, str]] = []
    for spots in index.values():
        for x, (fa, a) in enumerate(spots):
            for fb, b in spots[x + 1 :]:
                ka, kb = files[fa], files[fb]
                if a > 0 and b > 0 and ka[a - 1][1] == kb[b - 1][1]:
                    continue  # inside a longer run, counted where it starts
                if fa == fb and abs(a - b) < MIN_LINES:
                    continue
                n = 0
                while a + n < len(ka) and b + n < len(kb) and ka[a + n][1] == kb[b + n][1]:
                    n += 1
                if n > MIN_LINES:
                    found.append((n, f"{fa}:{ka[a][0]}", f"{fb}:{kb[b][0]}"))
    return sorted(found, key=lambda row: (-row[0], row[1], row[2]))


def numbers() -> None:
    files = sorted(SRC.rglob("*.py"))
    lines = [line for p in files for line in p.read_text(encoding="utf-8").splitlines()]
    filled = sum(1 for x in lines if x.strip())
    say(f"src: {len(files)} files, {len(lines)} lines, {filled} non-blank")
    rows = [
        (
            complexity(fn),
            (fn.end_lineno or fn.lineno) - fn.lineno,
            f"{p.relative_to(SRC)}:{fn.lineno} {n}",
        )
        for p, n, fn in functions(SRC)
    ]
    over = sorted((r for r in rows if r[0] > 15), reverse=True)
    mean = sum(r[0] for r in rows) / len(rows)
    say(f"complexity: {len(rows)} functions, mean {mean:.4f}, {len(over)} above 15")
    say(f"  worst cc={max(rows)[0]} {max(rows)[2]}")
    longest = max(rows, key=lambda r: r[1])
    say(f"  longest {longest[1]} lines {longest[2]}")
    for c, length, where in over:
        say(f"  cc={c:3d} len={length:4d} {where}")
    found = duplicates(SRC)
    say(f"duplicate blocks (> {MIN_LINES} kept lines): {len(found)}")
    for n, one, two in found:
        say(f"  {n:3d}  {one}  {two}")


# --------------------------------------------------------------------------- copies
def _calls(fn: ast.AST, attr: str) -> list[int]:
    return [
        n.lineno
        for n in ast.walk(fn)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == attr
    ]


def _called(fn: ast.AST, *ends: str) -> list[int]:
    """Calls whose callee reads as one of ``ends`` (``agent.harness.background.stop``...)."""
    return [
        n.lineno
        for n in ast.walk(fn)
        if isinstance(n, ast.Call) and ast.unparse(n.func).endswith(ends)
    ]


def _assigns(fn: ast.AST, attr: str) -> list[int]:
    return [
        n.lineno
        for n in ast.walk(fn)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Attribute) and t.attr == attr for t in n.targets)
    ]


def _names(fn: ast.AST, name: str) -> list[int]:
    return [n.lineno for n in ast.walk(fn) if isinstance(n, ast.Name) and n.id == name]


def _keyword(fn: ast.AST, name: str) -> list[int]:
    return [
        n.lineno
        for n in ast.walk(fn)
        if isinstance(n, ast.Call) and any(k.arg == name for k in n.keywords)
    ]


def _suppressed(fn: ast.AST) -> list[int]:
    """``with contextlib.suppress(ConflictError...)`` and ``except (..)ConflictError``."""
    found = []
    for n in ast.walk(fn):
        if isinstance(n, ast.With | ast.AsyncWith):
            for item in n.items:
                call = item.context_expr
                if (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Attribute)
                    and call.func.attr == "suppress"
                    and any(isinstance(a, ast.Name) and a.id == "ConflictError" for a in call.args)
                ):
                    found.append(n.lineno)
        is_handler = isinstance(n, ast.ExceptHandler) and n.type is not None
        if is_handler and "ConflictError" in ast.unparse(n.type):  # type: ignore[union-attr]
            found.append(n.lineno)
    return found


def _innermost(fn: ast.AST, lines: list[int]) -> list[int]:
    """``lines`` that are not inside a function nested in ``fn`` (reported there instead)."""
    nested = [
        (n.lineno, n.end_lineno or n.lineno)
        for n in ast.walk(fn)
        if n is not fn and isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)
    ]
    return [x for x in lines if not any(s <= x <= e for s, e in nested)]


def copies() -> None:
    probes = {
        "stops an attempt (sets .cancelled and cancels running_in)": lambda fn: sorted(
            set(_assigns(fn, "cancelled")) & {x - 1 for x in _calls(fn, "cancel")}
            | set(_assigns(fn, "cancelled")) & {x - 2 for x in _calls(fn, "cancel")}
        ),
        "decides 'it ended elsewhere' (names CANCELLED_ELSEWHERE / ENDED_ELSEWHERE)": lambda fn: (
            _names(fn, "CANCELLED_ELSEWHERE") + _names(fn, "ENDED_ELSEWHERE")
        ),
        "treats a refused ending write as 'ended meanwhile' (ConflictError)": _suppressed,
        "lists a run's children (iterate(parent_run_id=...))": lambda fn: [
            x for x in _keyword(fn, "parent_run_id") if x in _calls(fn, "iterate")
        ],
        "stops what this process keeps for a child (background.stop, child_cancelled, ...)": (
            lambda fn: (
                _called(fn, "background.stop", "tasks.cancelled", "tasks.stopped", "found.cancel")
                + [
                    n.lineno
                    for n in ast.walk(fn)
                    if isinstance(n, ast.Constant) and n.value == "child_cancelled"
                ]
            )
        ),
        "opens an attempt's helpers (Helpers(...)) or closes them (.closed(...))": lambda fn: (
            [
                n.lineno
                for n in ast.walk(fn)
                if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name)
                and n.func.id == "Helpers"
            ]
            + _called(fn, "helpers.closed")
        ),
    }
    hits: dict[str, list[str]] = {title: [] for title in probes}
    for path, name, fn in functions(HARNESS):
        for title, probe in probes.items():
            lines = sorted(set(_innermost(fn, probe(fn))))
            if lines:
                at = ", ".join(map(str, lines))
                hits[title].append(f"{path.relative_to(HARNESS)}:{name} (def {fn.lineno}; at {at})")
    for title, found in hits.items():
        say(f"{title}: {len(found)}")
        for where in found:
            say(f"  {where}")


# --------------------------------------------------------------------------- overlap
def _spans(text: str) -> list[tuple[int, int, str]]:
    out: list[tuple[int, int, str]] = []

    def walk(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                name = f"{prefix}{child.name}"
                start = min([child.lineno, *(d.lineno for d in child.decorator_list)])
                out.append((start, child.end_lineno or child.lineno, name))
                walk(child, name + ".")
            elif isinstance(child, ast.Assign | ast.AnnAssign) and prefix == "":
                targets = child.targets if isinstance(child, ast.Assign) else [child.target]
                for t in targets:
                    if isinstance(t, ast.Name):
                        out.append((child.lineno, child.end_lineno or child.lineno, t.id))
            else:
                walk(child, prefix)

    walk(ast.parse(text), "")
    return out


def touched(base: str, ref: str) -> dict[str, set[str]]:
    """The base's functions (and module-level names) that ``base...ref`` changes, by file. A
    ``ref`` that is a directory is a checkout of a branch not pushed yet: its working tree,
    uncommitted changes included, against ``base`` (a commit that checkout has)."""
    if Path(ref).is_dir():
        where, against, merge = ["-C", ref], [base], base
    else:
        where, against = [], [f"{base}...{ref}"]
        merge = subprocess.run(
            ["git", "merge-base", base, ref], capture_output=True, text=True, check=True
        ).stdout.strip()
    diff = subprocess.run(
        ["git", *where, "diff", "-U0", *against, "--", str(HARNESS)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    found: dict[str, set[str]] = defaultdict(set)
    spans: dict[str, list[tuple[int, int, str]]] = {}
    path = None
    for line in diff.splitlines():
        if line.startswith("--- "):
            path = line[6:] if line.startswith("--- a/") else None
            continue
        hunk = re.match(r"@@ -(\d+)(?:,(\d+))? \+", line)
        if not hunk or path is None or not path.endswith(".py"):
            continue
        start, count = int(hunk.group(1)), int(hunk.group(2) or 1)
        lo, hi = start, start + max(count, 1) - 1
        if path not in spans:
            shown = subprocess.run(
                ["git", *where, "show", f"{merge}:{path}"],
                capture_output=True,
                text=True,
                check=False,
            ).stdout
            spans[path] = _spans(shown) if shown else []
        hit = [n for s, e, n in spans[path] if s <= hi and lo <= e]
        inner = [n for n in hit if not any(o.startswith(n + ".") for o in hit)] or ["<module>"]
        found[str(Path(path).relative_to(HARNESS))].update(inner)
    return found


def overlap(base: str, refs: list[str]) -> None:
    for ref in refs:
        found = touched(base, ref)
        shared = sorted(
            f"{f}:{n}"
            for f, names in found.items()
            for n in names
            if any(n == t or n.startswith(t + ".") for t in TARGETS.get(f, ()))
        )
        say(f"{ref}: {sum(len(v) for v in found.values())} functions touched, {len(shared)} shared")
        for where in shared:
            say(f"  {where}")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "numbers"
    if command == "numbers":
        numbers()
    elif command == "copies":
        copies()
    elif command == "overlap":
        overlap(sys.argv[2], sys.argv[3:])
    else:
        raise SystemExit(__doc__)
