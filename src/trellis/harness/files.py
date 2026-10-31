"""The files a run makes for its user (an export, a sandbox's output): kept in one store, on
fsspec, where ``TRELLIS_FILES_URL`` says — ``gs://bucket/prefix``, ``s3://...``, ``az://...``,
``file:///path``; unset, this machine's XDG data folder (``settings.default_files_url``). A cloud
store needs its fsspec package installed (``gcsfs``, ``s3fs``, ``adlfs``: fsspec's error names
it) and reads that cloud's own credentials, as its SDK does. Files are stored only: nothing is
parsed, indexed or remembered.

A file is kept under its run's scope — ``<tenant>/<user>/<conversation>/<run>/<name>`` (each
part made safe for a path: :func:`part`) — so the AG-UI route that serves it rebuilds the path
from the run's record alone, and one user never reaches another's. Beside it is its record
(``.<name>.json``: :meth:`FileStore.record`), how it was made — the run, what made it, the
results it was made from and when each was fetched, the run's tool calls — so a later run reads
it back (``tools.stored``) without the run's journal, which a finished run does not keep. A
store that signs URLs (GCS, S3, Azure) gives a link valid :data:`LINK_SECONDS`; any other is
streamed by the harness
(:meth:`FileStore.chunks`). How long a file is kept is the store's: a bucket's lifecycle rule;
on local disk, until it is deleted.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import mimetypes
import re
from collections.abc import AsyncIterator
from typing import Any, Final

from pydantic import BaseModel, ConfigDict

from trellis.contracts import ToolError

#: How long a signed download link is valid, in seconds.
LINK_SECONDS: Final = 300
#: The unit a file is read and streamed in.
CHUNK_BYTES: Final = 1024 * 1024
#: The longest part of a path kept as it is; a longer (or unsafe) one is its digest.
PART_CHARS: Final = 120
#: What a file's name and a path's part may hold as they are.
SAFE: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.@=-]*")
#: The event a stored file is announced with on the run's stream.
EVENT: Final = "file"


class StoredFile(BaseModel):
    """A file the run made: its ``name``, its ``size`` in bytes, its media type, and the
    ``source`` that made it (a tool's name)."""

    model_config = ConfigDict(frozen=True)

    name: str
    size: int
    content_type: str
    source: str

    @staticmethod
    def kind_of(name: str) -> str:
        """The media type a file's name says (``application/octet-stream`` when it says
        none)."""
        return mimetypes.guess_type(name)[0] or "application/octet-stream"


def part(text: str) -> str:
    """One part of a stored file's path: ``text`` as it is when it is short and safe, else its
    digest (so no part can climb out of its scope)."""
    if len(text) <= PART_CHARS and SAFE.fullmatch(text) and ".." not in text:
        return text
    return hashlib.sha256(text.encode()).hexdigest()[:32]


def file_name(name: str) -> str:
    """``name`` checked as a stored file's name: one safe path part, refused otherwise (the
    model reads why)."""
    if len(name) > PART_CHARS or not SAFE.fullmatch(name) or ".." in name:
        raise ToolError(
            f"{name!r} is not a file name: use letters, digits, '.', '_' and '-' (at most "
            f"{PART_CHARS}), e.g. sales-2026.xlsx",
            source="tools",
        )
    return name


def scope(tenant: str, user: str, conversation: str, run_id: str) -> str:
    """The folder a run's files are kept in, below the store's root."""
    return "/".join(part(p) for p in (tenant, user, conversation, run_id))


class FileStore:
    """The store at ``url`` (an fsspec URL). Every call runs fsspec's synchronous API in a
    thread, so the event loop never waits on storage."""

    def __init__(self, url: str, **options: Any) -> None:
        from fsspec.core import url_to_fs  # noqa: PLC0415 - loaded when files are used

        self.url = url
        self.fs, root = url_to_fs(url, **options)
        self.root = root.rstrip("/")

    def path(self, folder: str, name: str) -> str:
        return f"{self.root}/{folder}/{file_name(name)}"

    def _record_path(self, folder: str, name: str) -> str:
        return f"{self.root}/{folder}/.{file_name(name)}.json"

    async def put(
        self, folder: str, name: str, data: bytes, *, source: str, content_type: str | None = None
    ) -> StoredFile:
        """Keep ``data`` as ``name`` in ``folder`` (a run's :func:`scope`): the same name
        again replaces it."""
        path = self.path(folder, name)
        kind = content_type or StoredFile.kind_of(name)

        def write() -> None:
            self.fs.makedirs(path.rsplit("/", 1)[0], exist_ok=True)
            self.fs.pipe_file(path, data)

        await asyncio.to_thread(write)
        return StoredFile(name=name, size=len(data), content_type=kind, source=source)

    async def keep_record(self, folder: str, name: str, record: dict[str, Any]) -> None:
        """Keep how the file ``name`` in ``folder`` was made (``record``, JSON) beside it."""
        data = json.dumps(record, default=str).encode()
        await asyncio.to_thread(self.fs.pipe_file, self._record_path(folder, name), data)

    async def record(self, folder: str, name: str) -> dict[str, Any] | None:
        """How the file ``name`` in ``folder`` was made (:meth:`keep_record`), ``None`` when
        nothing says."""
        path = self._record_path(folder, name)

        def read() -> dict[str, Any] | None:
            if not self.fs.isfile(path):
                return None
            found = json.loads(self.fs.cat_file(path))
            return found if isinstance(found, dict) else None

        return await asyncio.to_thread(read)

    async def files(self, folder: str) -> list[tuple[str, str, int]]:
        """Every file kept below ``folder`` (a scope's beginning: a user's, a conversation's):
        its folder (below the store's root), its name and its size — never a record."""
        base = f"{self.root}/{folder}"

        def find() -> list[tuple[str, str, int]]:
            found = self.fs.find(base, detail=True) if self.fs.exists(base) else {}
            kept = []
            for path, info in found.items():
                where, _, name = str(path).rstrip("/").rpartition("/")
                if not name.startswith("."):
                    kept.append((where[len(self.root) + 1 :], name, int(info.get("size") or 0)))
            return kept

        return await asyncio.to_thread(find)

    async def folders(self, folder: str) -> list[str]:
        """The names of the folders directly in ``folder`` (a user's: their conversations), one
        listing — none when there is no such folder."""
        base = f"{self.root}/{folder}"

        def ls() -> list[str]:
            if not self.fs.isdir(base):
                return []
            found = self.fs.ls(base, detail=True)
            return [
                str(f["name"]).rstrip("/").rsplit("/", 1)[-1]
                for f in found
                if f["type"] == "directory"
            ]

        return await asyncio.to_thread(ls)

    async def read(self, folder: str, name: str) -> bytes | None:
        """The file's bytes, ``None`` when there is none."""
        path = self.path(folder, name)

        def cat() -> bytes | None:
            return bytes(self.fs.cat_file(path)) if self.fs.isfile(path) else None

        return await asyncio.to_thread(cat)

    async def size(self, folder: str, name: str) -> int | None:
        """The file's size, ``None`` when there is none."""
        path = self.path(folder, name)

        def info() -> int | None:
            return int(self.fs.info(path)["size"]) if self.fs.isfile(path) else None

        return await asyncio.to_thread(info)

    async def link(self, folder: str, name: str) -> str | None:
        """A signed URL valid :data:`LINK_SECONDS`, where the store signs them (GCS, S3, Azure);
        ``None`` for one that does not (local disk): the harness streams it."""
        path = self.path(folder, name)
        try:
            return str(await asyncio.to_thread(self.fs.sign, path, expiration=LINK_SECONDS))
        except NotImplementedError:
            return None

    async def chunks(self, folder: str, name: str) -> AsyncIterator[bytes]:
        """The file's bytes, :data:`CHUNK_BYTES` at a time."""
        handle = await asyncio.to_thread(self.fs.open, self.path(folder, name), "rb")
        try:
            while chunk := await asyncio.to_thread(handle.read, CHUNK_BYTES):
                yield chunk
        finally:
            await asyncio.to_thread(handle.close)


__all__ = [
    "LINK_SECONDS",
    "FileStore",
    "StoredFile",
    "file_name",
    "scope",
]
