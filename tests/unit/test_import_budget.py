"""A budget on how the harness's packages import each other: what the source has today, so
neither can grow unseen. Lower a number when a change brings it down; raise one only with the
reason in the commit.

* **Pairs that import each other** at module level (``agent`` and ``bridges``, ``tools`` and
  ``governance``): each one is two packages that cannot be loaded, read or tested apart.
* **Imports inside a function** of another of the harness's packages: what breaks such a
  cycle at import time, and hides it.

A package is a folder or module directly under ``trellis/harness``; an import under
``TYPE_CHECKING`` is none (it is no import when the code runs).
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "src" / "trellis" / "harness"
PREFIX = "trellis.harness"

#: The pairs of packages that import each other at module level.
MAX_MUTUAL_PAIRS = 8
#: The imports of another package made inside a function.
MAX_IMPORTS_IN_FUNCTIONS = 141


def _package(path: Path) -> str:
    return path.relative_to(ROOT).with_suffix("").parts[0]


def _imported(node: ast.ImportFrom) -> list[str]:
    """The harness packages ``node`` imports."""
    parts = (node.module or "").split(".")
    if parts[:2] != PREFIX.split("."):
        return []
    if len(parts) > 2:
        return [parts[2]]
    names = [alias.name for alias in node.names]
    return [n for n in names if (ROOT / n).is_dir() or (ROOT / f"{n}.py").is_file()]


def measured() -> tuple[set[tuple[str, str]], int]:
    """The pairs of packages that import each other at module level, and how many imports of
    another package sit inside a function."""
    at_load: dict[str, set[str]] = {}
    in_functions = 0
    for path in sorted(ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text())
        typing_only = {
            id(inner)
            for node in ast.walk(tree)
            if isinstance(node, ast.If) and "TYPE_CHECKING" in ast.dump(node.test)
            for inner in ast.walk(node)
        }
        top = {id(node) for node in tree.body}
        mine = _package(path)
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or id(node) in typing_only:
                continue
            for other in _imported(node):
                if other == mine:
                    continue
                if id(node) in top:
                    at_load.setdefault(mine, set()).add(other)
                else:
                    in_functions += 1
    pairs = {(a, b) for a, others in at_load.items() for b in others if a < b}
    return {(a, b) for a, b in pairs if a in at_load.get(b, ())}, in_functions


def test_no_new_pair_of_packages_imports_each_other() -> None:
    pairs, _ = measured()
    assert len(pairs) <= MAX_MUTUAL_PAIRS, sorted(pairs)


def test_no_more_imports_of_another_package_inside_functions() -> None:
    _, in_functions = measured()
    assert in_functions <= MAX_IMPORTS_IN_FUNCTIONS, in_functions
