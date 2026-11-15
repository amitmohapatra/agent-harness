"""What the isolation tests need of this machine: a way to run a command as root (the service
itself when it is root, else ``sudo -n``, as on a CI runner), and a workspace root of a test's
own, removed afterwards. A machine with neither root nor a password-less sudo, or without the
tools the layer launches commands with, skips them."""

from __future__ import annotations

import os
import secrets
import shutil
import subprocess
import sys
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Final

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


#: The layer every test has unless it gives one of its own: it cannot isolate (its helper is
#: ``false``) — wherever the tests run, root or not, a framework's own shell is off the same
#: way (Claude Code's left out, Kimi Code's and an OpenAI shell's refused, saying why), and no
#: workspace is made on this machine. The test processes a test starts set it too.
NOT_ISOLATING: Final = ("/var/lib/trellis-test-none", ("false",))


def not_isolating() -> Any:
    """Make this machine's layer one that cannot isolate (:data:`NOT_ISOLATING`); the one it
    replaces is returned."""
    from trellis.harness.sandbox.isolation import HostSandbox, use_host

    root, helper = NOT_ISOLATING
    return use_host(HostSandbox(root, helper=helper))


@contextmanager
def isolating() -> Iterator[Any]:
    """This machine's isolation layer for a test (``use_host``): a root of the test's own and
    a uid range of its own (another test's layer may still be ending its runs), put back
    afterwards; skips where it cannot isolate (:func:`root_helper`)."""
    from trellis.harness.sandbox.isolation import HostSandbox, use_host

    helper = root_helper()
    for root in fresh_root(helper):
        base = 1_800_000_000 + secrets.randbelow(100_000) * 1_000
        made = HostSandbox(root, helper=helper, uid_base=base, tenants=1)
        replaced = use_host(made)
        try:
            yield made
        finally:
            use_host(replaced)
