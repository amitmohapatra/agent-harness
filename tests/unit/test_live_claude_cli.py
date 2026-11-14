"""The live Claude tests run where the Claude Agent SDK can run Claude Code: the CLI bundled
with the SDK (what it runs when no ``cli_path`` is given), not only a ``claude`` on PATH."""

from __future__ import annotations

from pathlib import Path

import claude_agent_sdk
import pytest

from tests.live.conftest import claude_cli


def test_the_live_skip_finds_the_cli_the_sdk_bundles(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", "")
    bundled = Path(claude_agent_sdk.__file__).parent / "_bundled" / "claude"
    assert claude_cli() == str(bundled)
