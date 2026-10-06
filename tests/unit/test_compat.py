"""The tested framework ranges: one place (``trellis.harness.compat``), the same in
``pyproject.toml``, and one warning at wrap time for a framework outside them."""

from __future__ import annotations

import logging
import re
import tomllib
from pathlib import Path

import pytest

from trellis import Harness, Settings
from trellis.harness import compat

ROOT = Path(__file__).resolve().parents[2]
REQUIREMENT = re.compile(r"^(?P<name>[A-Za-z0-9_.-]+)(?:\[[^\]]*\])?(?P<spec>.*)$")


def pins() -> dict[str, list[str]]:
    """Every requirement of the extras and the dev group, by distribution: its specifiers."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    found: dict[str, list[str]] = {}
    groups = [
        *project["project"]["optional-dependencies"].values(),
        *project["dependency-groups"].values(),
    ]
    for requirement in (r for group in groups for r in group):
        match = REQUIREMENT.match(requirement.replace(" ", ""))
        assert match is not None, requirement
        if match["name"] != "trellis-harness":
            found.setdefault(match["name"], []).append(match["spec"])
    return found


def test_pyproject_pins_exactly_the_tested_ranges() -> None:
    pinned = pins()
    for distribution, (low, high) in compat.TESTED.items():
        assert distribution in pinned, f"{distribution} is not pinned in pyproject.toml"
        assert set(pinned[distribution]) == {f">={low},<{high}"}, distribution


def test_every_framework_an_adapter_runs_on_has_a_tested_range() -> None:
    named = {d for distributions in compat.FRAMEWORKS.values() for d in distributions}
    assert named <= set(compat.TESTED)
    assert set(compat.FRAMEWORKS) == {"langgraph", "openai_agents", "claude_agent_sdk", "function"}


def test_versions_compare_by_their_release_numbers() -> None:
    assert compat.release("0.2.160") == (0, 2, 160)
    assert compat.release("1.4.0rc1") == (1, 4, 0)
    assert compat.release("dev") == ()
    assert compat.tested("langgraph", "1.2.11")
    assert compat.tested("langgraph", "1.2")
    assert not compat.tested("langgraph", "1.3.0")
    assert not compat.tested("langgraph", "1.1.9")
    assert compat.tested("deepagents", "0.7.19")
    assert not compat.tested("deepagents", "0.7.18")
    assert compat.tested("claude-agent-sdk", "0.2.999")
    assert compat.tested("not-a-framework", "9")  # not the harness's to judge


def test_an_untested_framework_is_said_once_at_wrap_time(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    versions = {"openai-agents": "0.30.1"}
    monkeypatch.setattr(compat, "installed", versions.get)
    monkeypatch.setattr(compat, "_warned", set())
    from agents import Agent

    caplog.set_level(logging.WARNING, logger="trellis.harness")
    h = Harness(config=Settings())
    h.wrap(Agent(name="one"), id="one")
    h.wrap(Agent(name="two"), id="two")  # the same distribution: not said again
    warned = [r.getMessage() for r in caplog.records if "trellis-harness was tested" in r.message]
    assert warned == [
        "openai-agents 0.30.1 is installed, but this trellis-harness was tested with "
        "openai-agents>=0.22.3,<0.23: use a newer trellis-harness (docs/versioning.md)"
    ]


def test_a_tested_or_missing_framework_says_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(compat, "_warned", set())
    monkeypatch.setattr(compat, "installed", {"langgraph": "1.2.11"}.get)
    assert compat.check("langgraph") == []  # in range; the others are not installed
    assert compat.check("function") == []
    assert compat.check("unknown") == []


def test_installed_reads_the_distribution_metadata() -> None:
    assert compat.installed("pydantic") is not None
    assert compat.installed("no-such-distribution-anywhere") is None
