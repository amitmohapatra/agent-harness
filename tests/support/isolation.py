"""What the isolation tests need of this machine: a way to run a command as root (the service
itself when it is root, else ``sudo -n``, as on a CI runner), and a workspace root of a test's
own, removed afterwards. A machine with neither root nor a password-less sudo, or without the
tools the layer launches commands with, skips them."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import uuid
from collections.abc import Iterator

import pytest

from trellis.harness.sandbox.isolation import TOOLS


def root_helper() -> tuple[str, ...]:
    """The prefix that runs a command as root here: none when this process is root, else
    ``sudo -n``; skips the test when there is neither (or a tool is missing)."""
    if not sys.platform.startswith("linux") or any(shutil.which(t) is None for t in TOOLS):
        pytest.skip("needs Linux with util-linux's unshare, setpriv and prlimit")
    if os.geteuid() == 0:
        return ()
    sudo = shutil.which("sudo")
    if sudo is not None and subprocess.run([sudo, "-n", "true"], check=False).returncode == 0:
        return ("sudo", "-n")
    pytest.skip("needs root, or a password-less sudo, to switch uid")


def as_root(helper: tuple[str, ...], *argv: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([*helper, *argv], check=False, capture_output=True, text=True)


def fresh_root(helper: tuple[str, ...]) -> Iterator[str]:
    """A workspace root of the test's own (under ``/var/lib``, out of the paths a run's command
    gets of its own), removed with everything in it afterwards."""
    root = f"/var/lib/trellis-test/{uuid.uuid4().hex[:12]}"
    try:
        yield root
    finally:
        as_root(helper, "rm", "-rf", root)
        as_root(helper, "rmdir", "/var/lib/trellis-test")


def processes_of(uid: int) -> list[str]:
    """The processes of ``uid`` on this machine (their pid and command)."""
    found = subprocess.run(
        ["ps", "-o", "pid=,args=", "-u", str(uid)], check=False, capture_output=True, text=True
    )
    return [line.strip() for line in found.stdout.splitlines() if line.strip()]
