"""A ratchet on copied code, not a style opinion.

Measured at the time of writing (2026-11-16): 41 duplicate blocks in ``src/trellis``. Most are
keyword pass-throughs (``Agent.run``/``stream``/``start``) and the legacy ``skills`` re-export;
six are the adapters' copies of opening and closing an attempt's helpers, which
``bridges.helpers.helping`` replaces.

The budget is set at *today's* count, so no copy may be added and every fold tightens it.
Lowering the number here is the point; raising it needs a reason in the commit message.

A block is more than ``MIN_LINES`` lines in a row seen twice, within one file too, counted as
pylint's ``duplicate-code`` counts lines - stripped, with comments, docstrings, imports and
def signatures left out - with the standard library only, so there is nothing to install.
"""

from __future__ import annotations

import ast
import io
import tokenize
from collections import defaultdict
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "trellis"

#: a block is more than this many kept lines in a row
MIN_LINES = 4
#: how many duplicate blocks there are today
MAX_DUPLICATE_BLOCKS = 41


def _dropped(tree: ast.Module) -> set[int]:
    """The lines of imports, def signatures (decorators included) and docstrings."""
    dropped: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import | ast.ImportFrom):
            dropped.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            start = min([node.lineno, *(d.lineno for d in node.decorator_list)])
            dropped.update(range(start, node.body[0].lineno))
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            first = node.body[0] if node.body else None
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                dropped.update(range(first.lineno, (first.end_lineno or first.lineno) + 1))
    return dropped


def _kept_lines(path: Path) -> list[tuple[int, str]]:
    """The file's lines that count, each with its number: stripped, comments cut off."""
    text = path.read_text(encoding="utf-8")
    dropped = _dropped(ast.parse(text))
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


def _duplicates() -> list[tuple[int, str, str]]:
    """Each run of more than ``MIN_LINES`` kept lines seen twice: (its length, where, where
    again), the longest first."""
    files = {str(p.relative_to(SRC)): _kept_lines(p) for p in sorted(SRC.rglob("*.py"))}
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


def test_no_block_of_code_is_copied_beyond_the_ones_today() -> None:
    found = _duplicates()
    assert len(found) <= MAX_DUPLICATE_BLOCKS, (
        f"{len(found)} duplicate blocks of more than {MIN_LINES} lines "
        f"(budget {MAX_DUPLICATE_BLOCKS}); all of them, the new one among them:\n  "
        + "\n  ".join(f"{n} lines  {one}  {two}" for n, one, two in found)
    )
