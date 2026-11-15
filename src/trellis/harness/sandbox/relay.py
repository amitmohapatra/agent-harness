"""A framework's own shell, run through the isolation layer: Claude Code's ``Bash`` (and its
``Monitor``, its ``PowerShell``), whose command the CLI runs itself, as the service's user.

The run's ``can_use_tool`` lets the call through with its command replaced (:meth:`Relay.
command`): the CLI runs ``python -I -S _relayed.py <socket> <token>``, which connects to the
relay's socket and hands it a one-time token — the harness looks up the command the model
made, runs it in the run's workspace (``isolation.Workspace``: its uid, its namespaces, its
limits, an environment of its own, its network, audited), and streams its stdout and stderr
back as they come, then its exit code. The CLI reads them as its own command's: what Claude
sees of the call — its output, its ``Exit code N``, a background task's output, a monitor's
lines — is what it would see of the command run itself. The CLI's timeout, or a task stopped,
kills the client: the socket closes, and the harness kills the command.

Bound to the run: the socket is in a folder of the service's own (0700) — no command (another
uid) can reach it — and is the query's alone, gone with it; each token names one command of
this run, once, and is never told the model. The shell's working directory goes on from one
command to the next as Claude Code's does (a ``cd`` holds for the next command when the
command succeeded, not for a background one), inside the workspace."""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import itertools
import logging
import os
import secrets
import shlex
import shutil
import sys
import tempfile
from collections.abc import Coroutine
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from trellis.harness.sandbox._relayed import HEADER

if TYPE_CHECKING:
    from trellis.harness.sandbox.isolation import Workspace

log = logging.getLogger("trellis.sandbox")

#: The client the framework's shell runs (standard library only).
CLIENT: Final = str(Path(__file__).with_name("_relayed.py"))
#: How each shell tool's command runs in the workspace: Claude Code's ``Bash`` (and
#: ``Monitor``'s command) in ``bash``, as Claude Code runs it (``eval`` of the command, in the
#: shell's working directory, which is written down when the command ends); ``PowerShell``'s
#: in ``pwsh``.
TRACKED: Final = (
    'cd -- "$1" 2>/dev/null; f=$2; c=$3; set --; '
    'trap \'{ pwd -P > "$f"; } 2>/dev/null\' EXIT; eval "$c"'
)
SHELLS: Final = {
    "Bash": ("bash", "-c", TRACKED, "bash"),
    "Monitor": ("bash", "-c", TRACKED, "bash"),
    "PowerShell": ("pwsh", "-NoProfile", "-NonInteractive", "-Command"),
}
#: What a token the relay does not hold (used, or never given) is answered with.
UNKNOWN: Final = b"this command was not given to the harness to run\n"
#: The exit code of a command the harness could not start (as a shell's).
NOT_STARTED: Final = 126


class Relay:
    """The relay of one framework process working in ``box`` (a workspace of this machine's
    isolation layer)."""

    def __init__(self, box: Workspace) -> None:
        self.box = box
        #: where the shell is: the workspace, then where its last command left it
        self.cwd = box.home
        self.folder = tempfile.mkdtemp(prefix="trellis-relay-")
        self.path = f"{self.folder}/s"
        #: the commands given, by their token: the tool, the command, whether in the
        #: background, and the context it runs in (the run's: its audit, its stream)
        self._given: dict[str, tuple[str, str, bool, contextvars.Context]] = {}
        self._server: asyncio.Server | None = None
        self._running: set[asyncio.Task[None]] = set()

    async def start(self) -> Relay:
        self._server = await asyncio.start_unix_server(self._served, path=self.path)
        return self

    def command(self, tool: str, command: str, *, background: bool = False) -> str:
        """What the framework's shell runs in place of ``command`` (a call of its ``tool``):
        the client, with a token that names this command, once."""
        token = secrets.token_hex(16)
        self._given[token] = (tool, command, background, contextvars.copy_context())
        return shlex.join([sys.executable, "-I", "-S", CLIENT, self.path, token])

    def adopt(self, path: str) -> None:
        """A file the framework wrote itself (as the service's user) in the workspace, and the
        folders it made for it: the run's, so its commands may change them."""
        home = Path(self.box.home)
        found = Path(path) if os.path.isabs(path) else Path(self.cwd) / path
        target = Path(os.path.normpath(found))
        if target == home or home not in target.parents:
            return
        for each in itertools.takewhile(lambda p: p != home, [target, *target.parents]):
            with contextlib.suppress(OSError):
                os.chown(each, self.box.uid, self.box.uid, follow_symlinks=False)

    async def aclose(self) -> None:
        """The framework's process ended: its commands still running killed, the socket gone."""
        if self._server is not None:
            self._server.close()
        for task in list(self._running):
            task.cancel()
        await asyncio.gather(*self._running, return_exceptions=True)
        shutil.rmtree(self.folder, ignore_errors=True)

    async def _served(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.current_task()
        assert task is not None
        self._running.add(task)
        try:
            token = (await reader.readline()).strip().decode(errors="replace")
            given = self._given.pop(token, None)
            if given is None:
                _frame(writer, 2, UNKNOWN)
                _frame(writer, 0, str(NOT_STARTED).encode())
            else:
                tool, command, background, context = given
                code = await _cut_off(reader, self._ran(writer, tool, command, background), context)
                _frame(writer, 0, str(code).encode())
            await writer.drain()
        except (ConnectionError, asyncio.CancelledError):
            pass
        finally:
            self._running.discard(task)
            writer.close()

    async def _ran(
        self, writer: asyncio.StreamWriter, tool: str, command: str, background: bool
    ) -> int:
        """``command`` run in the workspace, its output written to the client as it comes: its
        exit code. A ``Bash`` command that succeeded, not in the background, leaves the shell
        where it ended (inside the workspace)."""
        marker = f"{self.box.home}/.tmp/.trellis-cwd-{secrets.token_hex(8)}"
        argv = [*SHELLS[tool], command]
        if tool != "PowerShell":
            argv = [*SHELLS[tool], self.cwd, marker, command]
        try:
            ran = await self.box.exec(
                command, argv=argv, sink=lambda fd, data: _frame(writer, fd, data)
            )
        except Exception as exc:  # refused before it ran (the layer cannot isolate it, say)
            _frame(writer, 2, f"{exc}\n".encode())
            return NOT_STARTED
        finally:
            where = await asyncio.to_thread(_taken, marker)
        inside = where == self.box.home or where.startswith(f"{self.box.home}/")
        if ran.exit_code == 0 and not background and inside:
            self.cwd = where
        return ran.exit_code


async def _cut_off(
    reader: asyncio.StreamReader, work: Coroutine[Any, Any, int], context: contextvars.Context
) -> int:
    """``work`` run in ``context`` while the client is there: its client gone (killed, its
    connection closed), ``work`` is cancelled — the command killed — and ``ConnectionError``
    raised."""
    ran = asyncio.get_running_loop().create_task(work, context=context)
    gone = asyncio.ensure_future(reader.read())
    try:
        await asyncio.wait({ran, gone}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        gone.cancel()
        if not ran.done():
            ran.cancel()
            await asyncio.gather(ran, return_exceptions=True)
    if ran.cancelled():
        raise ConnectionError("the client went before its command ended")
    return ran.result()


def _taken(marker: str) -> str:
    """What the file ``marker`` says (where the shell was left), the file removed; ``""``: none."""
    try:
        with open(marker, encoding="utf-8", errors="replace") as said:
            return said.read().strip()
    except OSError:
        return ""
    finally:
        with contextlib.suppress(OSError):
            os.unlink(marker)


def _frame(writer: asyncio.StreamWriter, fd: int, data: bytes) -> None:
    writer.write(HEADER.pack(fd, len(data)) + data)


__all__ = ["CLIENT", "SHELLS", "Relay"]
