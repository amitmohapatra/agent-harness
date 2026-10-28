"""A run store shared by worker *processes*: :class:`LocalRuns` whose state lives in a file, read
before and written back after every call under an exclusive lock — as every worker process
reads and writes agent-runs. A process killed mid-run leaves the store as its last call did."""

from __future__ import annotations

import contextlib
import fcntl
import os
import pickle
from collections.abc import Awaitable, Callable, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Concatenate

from trellis.contracts import ArtifactRef
from trellis.harness.runs import JSON_MIME, LocalArtifacts, LocalRuns

#: What a process keeps of its own (its lock, its views of the store, the file's name).
OWN = frozenset({"_lock", "artifacts", "schedules", "path", "_depth"})
#: The store's calls a worker and its caller make, each one read from and written to the file.
SHARED = ("start", "claim", "heartbeat", "release", "cancel", "pause", "resume", "finish", "get")


class FileRuns(LocalRuns):
    """The in-process store, its state in the file ``path`` (``path.lock`` its lock)."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        super().__init__()
        self.path = Path(path)
        self._depth = 0
        self.artifacts = FileArtifacts(self)

    @contextlib.contextmanager
    def synced(self) -> Iterator[None]:
        """The store as the file holds it, for one call, written back when it returns."""
        if self._depth:
            yield
            return
        with open(f"{self.path}.lock", "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            self._depth += 1
            try:
                if self.path.exists():
                    vars(self).update(pickle.loads(self.path.read_bytes()))
                yield
                kept = {k: v for k, v in vars(self).items() if k not in OWN}
                written = self.path.with_suffix(".writing")
                written.write_bytes(pickle.dumps(kept))
                os.replace(written, self.path)
            finally:
                self._depth -= 1
                fcntl.flock(lock, fcntl.LOCK_UN)

    def lapse(self, run_id: str) -> None:
        """The lease on ``run_id`` runs out (its worker is dead): the next claim queues it."""
        with self.synced():
            worker, _ = self._leases[run_id]
            self._leases[run_id] = (worker, datetime.now(UTC) - timedelta(seconds=1))


class FileArtifacts(LocalArtifacts):
    """The store's artifacts, in its file."""

    def __init__(self, runs: FileRuns) -> None:
        super().__init__(runs)
        self._file = runs

    async def upload(
        self,
        run_id: str,
        data: bytes,
        *,
        mime_type: str = JSON_MIME,
        worker_id: str | None = None,
        tenant: str | None = None,
    ) -> ArtifactRef:
        with self._file.synced():
            return await super().upload(
                run_id, data, mime_type=mime_type, worker_id=worker_id, tenant=tenant
            )

    async def download(self, artifact_id: str, *, tenant: str | None = None) -> bytes | None:
        with self._file.synced():
            return await super().download(artifact_id, tenant=tenant)


def _shared[**P, T](
    call: Callable[Concatenate[LocalRuns, P], Awaitable[T]],
) -> Callable[Concatenate[FileRuns, P], Awaitable[T]]:
    async def synced(self: FileRuns, *args: P.args, **kwargs: P.kwargs) -> T:
        with self.synced():
            return await call(self, *args, **kwargs)

    return synced


for _name in SHARED:
    setattr(FileRuns, _name, _shared(getattr(LocalRuns, _name)))
