"""The documentation keeps up with the code: every public name, every setting and every example
is documented, every framework has its page (Way 1) and every block its page (Way 2), and the
docs index links each page."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import trellis
from trellis import Settings

ROOT = Path(__file__).resolve().parents[2]
README = (ROOT / "README.md").read_text()
DOCS = ROOT / "docs"
#: one page per kind of target, under docs/frameworks/
FRAMEWORK_PAGES = (
    "langgraph.md",
    "deepagents.md",
    "openai-agents.md",
    "claude-agent-sdk.md",
    "react.md",
    "functions.md",
)
#: Way 2: one page per block and per framework recipe, under docs/blocks/ (the sibling
#: repositories' READMEs link to each)
BLOCK_PAGES = (
    "memory.md",
    "runs.md",
    "governance.md",
    "evaluation.md",
    "a2a.md",
    "contracts.md",
    "langgraph.md",
    "openai-agents.md",
    "claude-agent-sdk.md",
    "mixing.md",
)


@pytest.mark.parametrize("name", trellis.__all__)
def test_every_public_name_is_in_the_readme(name: str) -> None:
    assert re.search(rf"`{re.escape(name)}[`(\[ ]", README), name


@pytest.mark.parametrize("field", list(Settings.model_fields))
def test_every_setting_is_in_the_configuration_page(field: str) -> None:
    assert f"`{field}`" in (DOCS / "configuration.md").read_text(), field


@pytest.mark.parametrize(
    "example",
    sorted(p.name for p in (ROOT / "examples").glob("*.py") if not p.name.startswith("_")),
)
def test_every_example_is_listed_and_linked_from_a_page(example: str) -> None:
    assert f"`{example}`" in README, example
    pages = [*DOCS.glob("*.md"), *DOCS.glob("frameworks/*.md"), *DOCS.glob("blocks/*.md")]
    assert any(f"examples/{example}" in page.read_text() for page in pages), example


@pytest.mark.parametrize("page", FRAMEWORK_PAGES)
def test_every_framework_page_exists_and_is_linked(page: str) -> None:
    assert (DOCS / "frameworks" / page).exists()
    assert f"frameworks/{page}" in (DOCS / "README.md").read_text()
    assert f"docs/frameworks/{page}" in README


@pytest.mark.parametrize("page", BLOCK_PAGES)
def test_every_block_page_exists_and_is_linked(page: str) -> None:
    assert (DOCS / "blocks" / page).exists()
    assert f"](blocks/{page})" in (DOCS / "README.md").read_text()


def test_the_docs_index_links_every_page() -> None:
    index = (DOCS / "README.md").read_text()
    for page in DOCS.glob("*.md"):
        if page.name != "README.md":
            assert f"]({page.name})" in index, page.name
    for page in DOCS.glob("blocks/*.md"):
        assert f"](blocks/{page.name})" in index, page.name


def test_the_readme_explains_the_two_ways() -> None:
    # the sibling repositories link to README.md#two-ways-to-use-trellis
    assert "\n## Two ways to use Trellis\n" in README
