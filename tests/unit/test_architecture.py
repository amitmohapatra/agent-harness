"""The rules that a reviewer cannot hold in their head, checked by reading the source.

The harness's model docstring has always claimed that no provider SDK is imported here —
that is the property that lets one agent run against OpenAI, Anthropic or a local model by
changing a string, and the reason a gateway exists at all.

Until 2026-09-28 nothing actually checked it, in two senses. The guard existed but pointed at
``src/trellis.harness`` — a dot where a separator belongs — so ``rglob`` walked a directory
that does not exist and the test passed over **zero files**: any ``import openai`` added in a
hurry would have kept every test green while quietly pinning the harness to one vendor. And
it covered only the core, while the framework adapters are where the temptation actually
lives: ``deepagents`` brings ``langchain_anthropic`` and ``langchain_google_genai``, and
``openai-agents`` brings ``openai``, so in this virtualenv all three are one line away.

So: the path is fixed, every trellis package is covered, and the walk asserts it found files.
A guard that cannot fail is decoration, and this one could not fail for its whole life.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

#: Every trellis package in this repository, by the distribution that ships it. A new
#: adapter adds a row here; leaving one out is the failure mode this test exists to prevent.
PACKAGES: dict[str, Path] = {
    "trellis-harness": ROOT / "src" / "trellis" / "harness",
    "trellis-harness-langgraph": (
        ROOT / "integrations" / "langgraph" / "src" / "trellis" / "harness_langgraph"
    ),
    "trellis-harness-agui": (ROOT / "integrations" / "agui" / "src" / "trellis" / "harness_agui"),
    "trellis-harness-deepagents": (
        ROOT / "integrations" / "deepagents" / "src" / "trellis" / "harness_deepagents"
    ),
    "trellis-harness-openai-agents": (
        ROOT / "integrations" / "openai_agents" / "src" / "trellis" / "harness_openai_agents"
    ),
    "trellis-harness-claude-agent-sdk": (
        ROOT / "integrations" / "claude_agent_sdk" / "src" / "trellis" / "harness_claude_agent_sdk"
    ),
}

#: Talking to a model provider directly is the one thing these packages must never do. The
#: gateway client (``bifrost``) is not on this list and must not be: it speaks to *our*
#: gateway over plain HTTP and knows a URL and a model name, never a provider credential.
#:
#: A framework's own package is not on the list either — an adapter imports the framework it
#: adapts, and ``deepagents``/``agents``/``claude_agent_sdk`` each pull a provider SDK in
#: transitively. What matters is that *trellis* code never reaches for one: each adapter
#: binds its framework's own model seam to ``BifrostModelClient`` instead (design §8).
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

#: The provider *client* classes, banned by name as well as by module: an adapter that
#: reached one through a framework's re-export would pass the import check above while doing
#: exactly what it forbids.
PROVIDER_CLIENTS = {
    "OpenAI",
    "AsyncOpenAI",
    "AzureOpenAI",
    "AsyncAzureOpenAI",
    "Anthropic",
    "AsyncAnthropic",
    "AnthropicBedrock",
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


def _imported_names(path: Path) -> set[str]:
    """The symbols a module imports, whatever they were imported from."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names.update(alias.name for alias in node.names)
    return names


@pytest.mark.parametrize("distribution", sorted(PACKAGES))
def test_the_guard_actually_reads_this_package(distribution: str) -> None:
    """The bug this file was written after: a path that matched nothing, forever green."""
    root = PACKAGES[distribution]
    assert root.is_dir(), f"{distribution}: {root} is not a directory"
    assert list(root.rglob("*.py")), f"{distribution}: the guard walked {root} and found no files"


@pytest.mark.parametrize("distribution", sorted(PACKAGES))
def test_no_trellis_package_imports_a_provider_sdk(distribution: str) -> None:
    """Every model call leaves through the gateway, so no vendor's SDK belongs in here."""
    root = PACKAGES[distribution]
    offenders: list[str] = []
    for path in root.rglob("*.py"):
        leaked = _imported_roots(path) & PROVIDER_SDKS
        if leaked:
            offenders.append(f"{path.relative_to(ROOT)}: {sorted(leaked)}")
    assert not offenders, (
        f"a provider SDK reached {distribution}; model calls go through the gateway:\n"
        + "\n".join(offenders)
    )


@pytest.mark.parametrize("distribution", sorted(PACKAGES))
def test_no_trellis_package_imports_a_provider_client(distribution: str) -> None:
    """The same rule, by symbol: a framework re-exporting ``AsyncOpenAI`` is still a client."""
    root = PACKAGES[distribution]
    offenders: list[str] = []
    for path in root.rglob("*.py"):
        leaked = _imported_names(path) & PROVIDER_CLIENTS
        if leaked:
            offenders.append(f"{path.relative_to(ROOT)}: {sorted(leaked)}")
    assert not offenders, (
        f"a provider client reached {distribution}; the gateway holds the credentials:\n"
        + "\n".join(offenders)
    )


def test_the_guard_would_actually_catch_one(tmp_path: Path) -> None:
    """A guard that cannot fail is decoration. This proves the AST walk sees real imports."""
    sample = tmp_path / "leaky.py"
    sample.write_text("from anthropic import Anthropic\nimport openai.types\n")
    assert _imported_roots(sample) & PROVIDER_SDKS == {"anthropic", "openai"}
    assert _imported_names(sample) & PROVIDER_CLIENTS == {"Anthropic"}


def test_the_client_guard_catches_a_framework_re_export(tmp_path: Path) -> None:
    """The case the module check alone would miss: a client imported from somewhere else."""
    sample = tmp_path / "sneaky.py"
    sample.write_text("from agents.models.openai_provider import AsyncOpenAI\n")
    assert not _imported_roots(sample) & PROVIDER_SDKS
    assert _imported_names(sample) & PROVIDER_CLIENTS == {"AsyncOpenAI"}
