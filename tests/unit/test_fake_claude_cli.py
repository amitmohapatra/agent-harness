"""The scripted stand-in for Claude Code (``tests/support/fake_claude_cli.py``) itself: what a
test reads of it holds when several of its processes write at once (a matrix cell's runs)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
#: What each writer process does: record its turn this many times.
WRITER = """
import sys
from tests.support import fake_claude_cli as cli
fake = cli.Cli()
for n in range(int(sys.argv[1])):
    fake.recorded("s", f"prompt {n}", "", [], first=n == 0)
"""


def test_processes_recording_at_once_leave_a_record_any_reader_can_read(tmp_path: Path) -> None:
    """Several processes record their turns into one ``FAKE_CLAUDE_RECORD`` at once: each
    record is whole when read — by a test, or by the process itself — and every turn is a
    line of ``<record>.turns``."""
    record, times, writers = tmp_path / "cli.json", 150, 6
    env = {**os.environ, "FAKE_CLAUDE_RECORD": str(record), "FAKE_CLAUDE_SESSIONS": str(tmp_path)}
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT), env.get("PYTHONPATH", "")])
    env.pop("CLAUDE_CONFIG_DIR", None)
    command = [sys.executable, "-c", WRITER, str(times)]
    started = [
        subprocess.Popen(command, env=env, cwd=ROOT, stderr=subprocess.PIPE, text=True)
        for _ in range(writers)
    ]
    torn = 0
    while any(p.poll() is None for p in started):
        if record.exists():
            try:
                json.loads(record.read_text())
            except json.JSONDecodeError:
                torn += 1
    errors = [p.communicate()[1] for p in started]
    assert [p.returncode for p in started] == [0] * writers, errors
    assert torn == 0
    lines = (tmp_path / "cli.json.turns").read_text().splitlines()
    assert len(lines) == times * writers
    assert all(json.loads(line)["session"] == "s" for line in lines)
