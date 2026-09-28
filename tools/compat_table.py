"""Print the README's compatibility table from ``compatibility-matrix.json``.

The matrix file is written by ``pytest tests/compatibility`` from the packages actually
imported in that run. Documenting versions by hand is how a table starts claiming a release
nobody exercised, so the table in README.md is this script's output, pasted:

    make docs-compat        # or: .venv/bin/python tools/compat_table.py

Exit code 1 when the file is missing, so a stale paste cannot be mistaken for a fresh one.
"""

from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
MATRIX = ROOT / "compatibility-matrix.json"

#: package name in the matrix -> how the table names it. Order is the table's order; a package
#: in the matrix but not here is appended, so a new dependency shows up rather than vanishing.
LABELS: dict[str, str] = {
    "langgraph": "LangGraph",
    "langchain": "LangChain",
    "langchain-core": "LangChain core",
    "deepagents": "Deep Agents",
    "openai-agents": "OpenAI Agents SDK",
    "claude-agent-sdk": "Claude Agent SDK",
    "a2a-sdk": "a2a-sdk (A2A protocol v1.0)",
    "temporalio": "temporalio",
    "fastapi": "FastAPI (AG-UI, A2A surfaces)",
    "langfuse": "Langfuse",
    "opentelemetry-api": "OpenTelemetry API",
    "opentelemetry-sdk": "OpenTelemetry SDK",
    "pydantic": "pydantic",
    "trellis-memory": "trellis-memory (Memory Service SDK)",
    "trellis-harness-langgraph": "trellis-harness-langgraph",
    "trellis-harness-agui": "trellis-harness-agui",
    "trellis-harness-a2a": "trellis-harness-a2a",
    "trellis-harness-deepagents": "trellis-harness-deepagents",
    "trellis-harness-openai-agents": "trellis-harness-openai-agents",
    "trellis-harness-claude-agent-sdk": "trellis-harness-claude-agent-sdk",
    "trellis-harness-temporal": "trellis-harness-temporal",
}


def rows(matrix: dict) -> list[tuple[str, str]]:
    packages: dict[str, str] = matrix.get("packages", {})
    out = [
        ("Python", str(matrix.get("python", "?"))),
        ("trellis-harness", str(matrix.get("harness_version", "?"))),
    ]
    out += [(LABELS[name], packages[name]) for name in LABELS if name in packages]
    out += [(name, version) for name, version in sorted(packages.items()) if name not in LABELS]
    return out


def table(matrix: dict) -> str:
    lines = ["| Component | Version exercised |", "| --- | --- |"]
    lines += [f"| {name} | {version} |" for name, version in rows(matrix)]
    return "\n".join(lines)


def main() -> int:
    if not MATRIX.is_file():
        print(f"{MATRIX} is missing: run `pytest tests/compatibility` first", file=sys.stderr)
        return 1
    print(table(json.loads(MATRIX.read_text())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
