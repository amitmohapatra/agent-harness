"""What a framework's shell runs in place of a command the harness runs for it
(``sandbox._relayed``), and the relay's own edges (``sandbox.relay``) that need no command:
the client writes each stream as it comes and exits with the command's code, or says the
harness went; a token the relay does not hold runs nothing; a path outside the workspace is
never adopted."""

from __future__ import annotations

import asyncio
import os
import socket
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from trellis.harness.sandbox import _relayed
from trellis.harness.sandbox._relayed import HEADER, LOST, main
from trellis.harness.sandbox.relay import NOT_STARTED, UNKNOWN, Relay


def serve(frames: bytes) -> tuple[str, threading.Thread, list[bytes]]:
    folder = tempfile.mkdtemp()
    path = f"{folder}/s"
    server = socket.socket(socket.AF_UNIX)
    server.bind(path)
    server.listen(1)
    got: list[bytes] = []

    def answer() -> None:
        conn, _ = server.accept()
        got.append(conn.recv(100))
        conn.sendall(frames)
        conn.close()
        server.close()

    thread = threading.Thread(target=answer)
    thread.start()
    return path, thread, got


def test_the_client_writes_each_stream_and_exits_with_the_commands_code(
    capfd: pytest.CaptureFixture[str],
) -> None:
    frames = HEADER.pack(1, 3) + b"out" + HEADER.pack(2, 3) + b"err" + HEADER.pack(0, 1) + b"7"
    path, thread, got = serve(frames)
    assert main(path, "token") == 7
    thread.join()
    assert got == [b"token\n"]
    out, err = capfd.readouterr()
    assert (out, err) == ("out", "err")


def test_the_client_says_when_the_harness_went_before_the_command_ended() -> None:
    path, thread, _ = serve(HEADER.pack(1, 2) + b"hi")
    assert main(path, "token") == LOST
    thread.join()


async def test_a_token_the_relay_does_not_hold_runs_nothing(tmp_path: Path) -> None:
    box: Any = SimpleNamespace(home=str(tmp_path), uid=os.getuid())
    relay = await Relay(box).start()
    try:
        reader, writer = await asyncio.open_unix_connection(relay.path)
        writer.write(b"never-given\n")
        await writer.drain()
        data = await reader.read()
    finally:
        await relay.aclose()
    assert data == HEADER.pack(2, len(UNKNOWN)) + UNKNOWN + HEADER.pack(0, 3) + b"126"
    assert NOT_STARTED == 126 and not await asyncio.to_thread(os.path.exists, relay.folder)


async def test_only_a_file_in_the_workspace_is_adopted(tmp_path: Path) -> None:
    home = tmp_path / "home"
    (home / "a").mkdir(parents=True)
    (home / "a" / "f").write_text("x")
    outside = tmp_path / "outside"
    outside.write_text("y")
    calls: list[str] = []
    box: Any = SimpleNamespace(home=str(home), uid=12345)
    relay = Relay(box)
    original = os.chown
    try:
        os.chown = lambda path, *a, **k: calls.append(str(path))  # type: ignore[assignment]
        relay.adopt(str(home / "a" / "f"))
        relay.adopt("a/f")  # relative to where the shell is
        relay.adopt(str(outside))
        relay.adopt(str(home))
    finally:
        os.chown = original
        await relay.aclose()
    assert calls == [str(home / "a" / "f"), str(home / "a")] * 2


def test_the_client_is_standard_library_only() -> None:
    names = [line for line in Path(_relayed.__file__).read_text().splitlines() if "import" in line]
    assert names == ["import os", "import socket", "import struct", "import sys"]


class FakeBox:
    """A workspace whose command answers as told (``ran``: what it was asked to run)."""

    def __init__(self, home: str, outcome: Any) -> None:
        self.home, self.uid, self.outcome = home, os.getuid(), outcome
        self.ran: list[list[str]] = []
        self.cancelled = asyncio.Event()

    async def exec(self, command: str, *, argv: list[str], sink: Any, **_: Any) -> Any:
        from trellis.harness.sandbox.base import ExecResult

        self.ran.append(argv)
        if self.outcome == "forever":
            try:
                await asyncio.Event().wait()
            finally:
                self.cancelled.set()
        if isinstance(self.outcome, Exception):
            raise self.outcome
        sink(1, b"out")
        return ExecResult(exit_code=self.outcome, stdout="out", stderr="", duration_ms=1)


async def asked(relay: Relay, line: str) -> bytes:
    token = line.rsplit(maxsplit=1)[-1]
    reader, writer = await asyncio.open_unix_connection(relay.path)
    writer.write(f"{token}\n".encode())
    await writer.drain()
    return await reader.read()


async def test_powershell_runs_in_pwsh_and_a_command_the_layer_refuses_says_why(
    tmp_path: Path,
) -> None:
    from trellis.contracts import ToolError

    box: Any = FakeBox(str(tmp_path), 0)
    relay = await Relay(box).start()
    try:
        said = await asked(relay, relay.command("PowerShell", "Get-ChildItem"))
        assert box.ran == [["pwsh", "-NoProfile", "-NonInteractive", "-Command", "Get-ChildItem"]]
        assert said.endswith(HEADER.pack(0, 1) + b"0")
        box.outcome = ToolError("cannot be isolated here", source="tools")
        refused = await asked(relay, relay.command("Bash", "id"))
        assert b"cannot be isolated here" in refused and refused.endswith(b"126")
    finally:
        await relay.aclose()


async def test_a_client_that_goes_has_its_command_killed(tmp_path: Path) -> None:
    box: Any = FakeBox(str(tmp_path), "forever")
    relay = await Relay(box).start()
    try:
        token = relay.command("Bash", "sleep 300").split()[-1]
        reader, writer = await asyncio.open_unix_connection(relay.path)
        writer.write(f"{token}\n".encode())
        await writer.drain()
        while not box.ran:  # noqa: ASYNC110 - the command started
            await asyncio.sleep(0.01)
        writer.close()  # the framework killed its client: its timeout, a task stopped
        async with asyncio.timeout(5):
            await box.cancelled.wait()
    finally:
        await relay.aclose()
