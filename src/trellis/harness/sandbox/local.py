"""A run's working folder on this machine, with no isolation: what the harness gives an agent
that has no shell of its own (``sandbox.code_for``), as Claude Code's ``Bash`` and Kimi Code's
shell run — a command with ``sh -c`` as the service's own user, with its environment, in a
folder made for the run. Nothing confines it: ``sandbox()`` (Docker, this machine's isolation
layer, a provider of yours) is the isolated one, and ``TRELLIS_HOST_EXECUTION=deny`` the
switch that turns this one off (``host``).

:class:`LocalFolders` is the provider (``name`` ``"local"``): each run's folder is made under
``root`` (else :func:`default_root`: the system's temporary folder) at its first command, and
removed with its run as any sandbox is (``sandbox.ended``, the reaper); its labels are kept in
a file beside it. The run's kept results, which the model names at ``/large_tool_results``,
are files of the folder (:data:`RESULTS`): a command or a path naming one there is read as
that file (:meth:`Folder.placed`), since nothing on a plain host puts a folder at that path.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
import signal
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Final

from trellis.contracts import ToolError
from trellis.harness.sandbox.base import ExecResult, SandboxLost, SandboxRef, SandboxSpec
from trellis.harness.sandbox.docker import OUTPUT_BYTES, _Tail
from trellis.harness.sandbox.isolation import READ_MAX_BYTES, RESULTS_PATH, _drain

#: The folder, under the system's temporary one, that holds the runs' folders by default.
ROOT: Final = "trellis-runs"
#: The folder of a run's folder that holds its kept results (``RESULTS_PATH`` names it).
RESULTS: Final = ".results"
#: Where the labels of the runs' folders are kept, under the root.
LABELS: Final = ".labels"


def default_root() -> Path:
    """Where the runs' folders are made unless a root is given: under the system's temporary
    folder (``TMPDIR``'s)."""
    return Path(tempfile.gettempdir()) / ROOT


class Folder:
    """One run's folder, as its tools use it (the ``Sandbox`` protocol)."""

    def __init__(self, ref: SandboxRef, home: Path) -> None:
        self._ref = ref
        self.home = home

    @property
    def ref(self) -> SandboxRef:
        return self._ref

    def placed(self, text: str) -> str:
        """``text`` (a command, a path) with the run's kept results named where they are: the
        folder's own (:data:`RESULTS`) in place of ``RESULTS_PATH``."""
        return text.replace(RESULTS_PATH, str(self.home / RESULTS))

    async def exec(
        self,
        command: str,
        *,
        timeout: float | None = None,  # noqa: ASYNC109 - the command's, killed past it
        env: Mapping[str, str] | None = None,
    ) -> ExecResult:
        started = time.monotonic()
        proc = await asyncio.create_subprocess_exec(
            "sh",
            "-c",
            self.placed(command),
            cwd=self._live(),
            env={**os.environ, **(env or {})},
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,  # its own process group: killed whole past its time
        )
        out, err = _Tail(OUTPUT_BYTES), _Tail(OUTPUT_BYTES)
        assert proc.stdout is not None and proc.stderr is not None

        async def ran() -> int:
            assert proc.stdout is not None and proc.stderr is not None
            await asyncio.gather(_drain(proc.stdout, out), _drain(proc.stderr, err))
            return await proc.wait()

        try:
            code = await asyncio.wait_for(ran(), timeout)
        except BaseException:  # past its time, or its call cancelled: every process of it
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(proc.pid, signal.SIGKILL)
            await proc.wait()
            raise
        return ExecResult(
            exit_code=128 - code if code < 0 else code,
            stdout=out.text(),
            stderr=err.text(),
            duration_ms=int((time.monotonic() - started) * 1000),
        )

    async def read(self, path: str) -> bytes:
        where = self._where(path)
        if not where.exists():
            raise ToolError(f"there is no file {path} in the working folder", source="tools")
        if not where.is_file():
            raise ToolError(f"{path} in the working folder is not a file", source="tools")
        if where.stat().st_size > READ_MAX_BYTES:
            raise ToolError(
                f"{path} is more than the {READ_MAX_BYTES} bytes read at once: read a part of "
                "it with a command",
                source="tools",
            )
        return await asyncio.to_thread(where.read_bytes)

    async def write(self, path: str, data: bytes) -> None:
        where = self._where(path)

        def work() -> None:
            where.parent.mkdir(parents=True, exist_ok=True)
            where.write_bytes(data)

        await asyncio.to_thread(work)

    def _where(self, path: str) -> Path:
        """``path`` as a file of this machine: a relative one in the folder, an absolute one as
        it is (a kept result's in the folder: :meth:`placed`)."""
        return Path(self._live()) / self.placed(path)

    def _live(self) -> str:
        if not self.home.is_dir():
            raise SandboxLost(f"the working folder {self._ref.id} is gone", source="tools")
        return str(self.home)


class LocalFolders:
    """The runs' folders on this machine, with no isolation (the module's words): ``root``
    holds them (else :func:`default_root`)."""

    name = "local"

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root is not None else default_root()

    async def create(self, ref: SandboxRef, spec: SandboxSpec | None = None) -> Folder:
        home = self._home(ref)
        made = not home.is_dir()
        home.mkdir(parents=True, exist_ok=True)
        (home / RESULTS).mkdir(exist_ok=True)
        self._labels(ref).parent.mkdir(parents=True, exist_ok=True)
        self._labels(ref).write_text(json.dumps(dict(ref.labels)))
        box = Folder(ref, home)
        files = spec.files if spec is not None and made else {}
        for path, content in files.items():
            await box.write(path, content.encode() if isinstance(content, str) else content)
        return box

    async def attach(self, ref: SandboxRef) -> Folder:
        box = Folder(ref, self._home(ref))
        box._live()
        return box

    async def delete(self, ref: SandboxRef) -> None:
        await asyncio.to_thread(shutil.rmtree, self._home(ref), ignore_errors=True)
        self._labels(ref).unlink(missing_ok=True)

    async def labelled(self, labels: Mapping[str, str]) -> list[SandboxRef]:
        found = []
        for kept in sorted((self.root / LABELS).glob("*.json")):
            held = json.loads(kept.read_text())
            if all(held.get(k) == v for k, v in labels.items()):
                found.append(SandboxRef(provider=self.name, id=kept.stem, labels=held))
        return found

    def _home(self, ref: SandboxRef) -> Path:
        return self.root / Path(ref.id).name

    def _labels(self, ref: SandboxRef) -> Path:
        return self.root / LABELS / f"{Path(ref.id).name}.json"


__all__ = ["Folder", "LocalFolders", "default_root"]
