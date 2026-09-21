"""The rules that a reviewer cannot hold in their head, checked by reading the source.

The harness's model docstring has always claimed that no provider SDK is imported here —
that is the property that lets one agent run against OpenAI, Anthropic or a local model by
changing a string, and the reason a gateway exists at all. Until now nothing checked it, so
it was a comment rather than a guarantee: any `import openai` added in a hurry would have
kept every test green while quietly pinning the harness to one vendor.
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "universal_agent_harness"

#: Talking to a model provider directly is the one thing this package must never do. The
#: gateway client (``bifrost``) is not on this list and must not be: it speaks to *our*
#: gateway over plain HTTP and knows a URL and a model name, never a provider credential.
PROVIDER_SDKS = {
    "openai",
    "anthropic",
    "google",
    "vertexai",
    "litellm",
    "langchain_openai",
    "langchain_anthropic",
    "langchain_google_genai",
    "mistralai",
    "cohere",
    "ollama",
    "groq",
    "together",
    "boto3",
}


def _imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    return roots


def test_the_harness_imports_no_provider_sdk() -> None:
    """Every model call leaves through the gateway, so no vendor's SDK belongs in here."""
    offenders: list[str] = []
    for path in SRC.rglob("*.py"):
        leaked = _imported_roots(path) & PROVIDER_SDKS
        if leaked:
            offenders.append(f"{path.relative_to(SRC)}: {sorted(leaked)}")
    assert not offenders, (
        "a provider SDK reached the harness; model calls go through the gateway:\n"
        + "\n".join(offenders)
    )


def test_the_guard_would_actually_catch_one(tmp_path: Path) -> None:
    """A guard that cannot fail is decoration. This proves the AST walk sees a real import."""
    sample = tmp_path / "leaky.py"
    sample.write_text("from anthropic import Anthropic\nimport openai.types\n")
    assert _imported_roots(sample) & PROVIDER_SDKS == {"anthropic", "openai"}
