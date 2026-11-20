"""A ratchet on complexity, not a style opinion.

Measured on 2026-11-16, once stopping and the children's cascade had one owner each
(``stopping``, ``children``): 3,124 functions in 131 files, mean cyclomatic complexity 4.07 -
the code base is not generally complex. The tail is: 63 functions above 15,
the worst being ``Agent.remembered`` and ``ACPAdapter._attempt`` at 34; the longest is ``ReAct``
at 198 lines.

These budgets are set at *today's* worst, so nothing may get worse and every improvement
tightens them. Lowering a number here is the point; raising one needs a reason in the commit
message. Complexity is counted as the memory service's test of the same name counts it - one
per branch, loop, handler, ``with``, assert, comprehension, ``match`` case and extra boolean
operand, a nested function's toward its parent too - so the two repositories compare, with no
extra dependency to install.
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "trellis"

#: the current worst function, by cyclomatic complexity
MAX_COMPLEXITY = 34
#: the current longest function, in lines
MAX_FUNCTION_LINES = 198
#: how many functions may exceed a complexity of 15
MAX_OVER_15 = 63
#: the mean (4.1114 today: the spend saves of a worker's run, the one answer rule, a graph's
#: harness tools, and the hold, event-log and turn-order fixes for load, 2026-11-19; then the
#: schedule settings, end dates, limits said with their numbers, pause reasons, the model's
#: keep question and notifications, 2026-11-20; then a new schedule's one confirmation, its
#: unattended runs' skipped steps and an unanswered question's deadline, each branch a case a
#: user meets) must not drift upwards
MAX_MEAN_COMPLEXITY = 4.12

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


def _complexity(fn: ast.AST) -> int:
    score = 1
    for node in ast.walk(fn):
        if isinstance(node, _BRANCHING):
            score += 1
        elif isinstance(node, ast.BoolOp):
            score += len(node.values) - 1
    return score


def _functions() -> list[tuple[int, int, str]]:
    found: list[tuple[int, int, str]] = []
    for path in sorted(SRC.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                length = (node.end_lineno or node.lineno) - node.lineno
                found.append(
                    (
                        _complexity(node),
                        length,
                        f"{path.relative_to(SRC)}:{node.lineno} {node.name}",
                    )
                )
    return found


def test_no_function_is_more_complex_than_the_worst_one_today() -> None:
    worst = sorted(_functions(), reverse=True)[:3]
    assert worst[0][0] <= MAX_COMPLEXITY, (
        f"complexity budget exceeded (limit {MAX_COMPLEXITY}):\n  "
        + "\n  ".join(f"cc={c} {where}" for c, _, where in worst)
    )


def test_no_function_is_longer_than_the_longest_one_today() -> None:
    longest = sorted(_functions(), key=lambda row: -row[1])[:3]
    assert longest[0][1] <= MAX_FUNCTION_LINES, (
        f"length budget exceeded (limit {MAX_FUNCTION_LINES}):\n  "
        + "\n  ".join(f"len={n} {where}" for _, n, where in longest)
    )


def test_the_complex_tail_does_not_grow() -> None:
    over = [row for row in _functions() if row[0] > 15]
    assert len(over) <= MAX_OVER_15, (
        f"{len(over)} functions above complexity 15 (budget {MAX_OVER_15}); "
        "the least complex of them, where a new one most likely is:\n  "
        + "\n  ".join(f"cc={c} {where}" for c, _, where in sorted(over)[:8])
    )


def test_the_average_function_stays_simple() -> None:
    rows = _functions()
    mean = sum(row[0] for row in rows) / len(rows)
    assert mean <= MAX_MEAN_COMPLEXITY, (
        f"mean complexity {mean:.4f} over {len(rows)} functions exceeds {MAX_MEAN_COMPLEXITY}"
    )
