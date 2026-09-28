"""``MemoryServiceBackend``: Deep Agents' ``/memories/*`` files, served by the Memory Service.

Deep Agents gives the model a filesystem and lets it keep long-lived notes under
``/memories`` (``MemoryMiddleware``), backed by a ``BackendProtocol`` — state, a real
filesystem, a LangGraph store, a sandbox. None of those knows the platform's boundary: who
may read a note, when a superseded one stops being true, or who read what.

This backend puts the Memory Service behind the same protocol, so the agent keeps using
``read_file``/``write_file``/``edit_file`` exactly as it always has and the notes land where
tenant, workspace, user and run visibility is actually enforced, with revisions and
forgetting for free (design §3).

Addressing: a file is ``/memories/<memory_id>.md``, and the **service** names it. A write to
a path the service does not know creates a memory and the result carries the path it was
given — ``write_file("/memories/refund-policy.md", ...)`` comes back as
``/memories/mem_01J…\u200b.md``, because the Memory Service assigns memory ids and does not keep
a caller-chosen filename (verified against the running service: a ``title``/``path`` passed
with the write does not survive extraction). A write over a path the service *does* know
supersedes that memory, keeping the revision.

That makes ``ls`` the way to find a note again, which is how Deep Agents' memory middleware
already works: it lists ``/memories`` and reads each file into the prompt, so a note is
found by its content, not by a name the model has to remember. A memory written but not yet
extracted is addressable by its observation id until it is, so a read that follows a write
never falls into a hole.

Paths outside ``/memories`` are refused through the protocol's own ``error`` field rather
than by raising, which is how every other backend reports a bad path.
"""

from __future__ import annotations

import fnmatch
import re
from typing import Any, Final

from deepagents.backends.protocol import (
    BackendProtocol,
    DeleteResult,
    EditResult,
    FileInfo,
    GlobResult,
    GrepMatch,
    GrepResult,
    LsResult,
    ReadResult,
    WriteResult,
)

from trellis.harness.execution.sync import run_sync
from trellis.harness_deepagents.binding import active_runtime

__all__ = ["MEMORIES_ROOT", "MemoryServiceBackend"]

#: The directory Deep Agents' memory middleware reads and writes.
MEMORIES_ROOT: Final = "/memories"
#: The extension the notes are presented under, so the model treats them as text.
SUFFIX: Final = ".md"
#: How many memories one listing asks the service for.
LISTING_LIMIT: Final = 200


class MemoryServiceBackend(BackendProtocol):
    """The Memory Service as a Deep Agents file backend.

        agent = harness.deepagents.create_agent(runtime=runtime, tools=[...])

    Constructed by the adapter with the execution's :class:`MemoryRuntime`; every operation
    is therefore already scoped to this run's tenant, workspace, user and thread, and every
    read is audited by the service.
    """

    def __init__(self, memory: Any = None, *, visibility: str = "USER") -> None:
        """``memory`` is the execution's :class:`MemoryRuntime`; left unset it resolves the
        running execution's per call, so one compiled graph serves every run.

        ``visibility`` is the audience a note written by the agent gets: ``USER`` keeps a
        remembered preference with the person it is about, which is what the Deep Agents
        memory middleware is for. Pass ``RUN`` for scratch notes that must not outlive the run.
        """
        self._memory = memory
        self.visibility = visibility

    @property
    def memory(self) -> Any:
        return self._memory if self._memory is not None else active_runtime().memory

    # ------------------------------------------------------------------ async (the real path)
    async def als(self, path: str) -> LsResult:
        if not _within(path):
            return LsResult(error=_outside(path))
        entries: list[FileInfo] = []
        for item in await self._memories():
            entries.append(FileInfo(path=_path_of(item), size=len(_content_of(item))))
        return LsResult(entries=entries)

    async def aread(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        if not _within(file_path):
            return ReadResult(error=_outside(file_path))
        if limit <= 0:
            return ReadResult(no_lines_requested=True)
        item = await self._item(file_path)
        if item is None:
            return ReadResult(error=f"{file_path}: no such memory")
        return _window(_content_of(item), offset, limit)

    async def awrite(self, file_path: str, content: str) -> WriteResult:
        if not _within(file_path):
            return WriteResult(error=_outside(file_path))
        existing = await self._resolve(_id_of(file_path))
        if existing is not None:
            # a write over a known memory supersedes it: the service keeps the revision
            await self.memory.forget(existing)
        ack = await self.memory.remember(
            content,
            memory_type="SEMANTIC",
            lifetime="LONG_TERM",
            visibility=self.visibility,
        )
        observation_id = _attr(ack, "observation_id")
        written = await self._memory_from(observation_id)
        return WriteResult(path=_path(written or observation_id or _id_of(file_path)))

    async def aedit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        if not _within(file_path):
            return EditResult(error=_outside(file_path))
        item = await self._item(file_path)
        if item is None:
            return EditResult(error=f"{file_path}: no such memory")
        content = _content_of(item)
        occurrences = content.count(old_string)
        if occurrences == 0:
            return EditResult(error=f"{file_path}: the text to replace is not in this memory")
        if occurrences > 1 and not replace_all:
            return EditResult(
                error=(
                    f"{file_path}: {occurrences} occurrences of that text; pass replace_all "
                    "or give more surrounding context"
                )
            )
        edited = content.replace(old_string, new_string, -1 if replace_all else 1)
        written = await self.awrite(_path(_attr(item, "memory_id")), edited)
        if written.error is not None:
            return EditResult(error=written.error)
        return EditResult(path=written.path, occurrences=occurrences if replace_all else 1)

    async def adelete(self, file_path: str) -> DeleteResult:
        if not _within(file_path):
            return DeleteResult(error=_outside(file_path))
        memory_id = await self._resolve(_id_of(file_path))
        if memory_id is None:
            return DeleteResult(error=f"{file_path}: no such memory")
        await self.memory.forget(memory_id)
        return DeleteResult(path=file_path)

    async def agrep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
        *,
        max_count: int | None = None,
    ) -> GrepResult:
        """Regex search over the notes, done here rather than by the service.

        The Memory Service has no regex endpoint, and ``recall`` is a different thing — a
        ranked semantic search, not a literal match — so substituting it would answer a
        different question than the model asked. The listing is bounded, so this is a scan
        over at most :data:`LISTING_LIMIT` notes.
        """
        if path is not None and not _within(path):
            return GrepResult(error=_outside(path))
        try:
            expression = re.compile(pattern)
        except re.error as exc:
            return GrepResult(error=f"invalid pattern: {exc}")
        matches: list[GrepMatch] = []
        for item in await self._memories():
            file_path = _path_of(item)
            if glob and not fnmatch.fnmatch(file_path, glob):
                continue
            for number, line in enumerate(_content_of(item).splitlines(), start=1):
                if expression.search(line):
                    matches.append(GrepMatch(path=file_path, line=number, text=line))
                    if max_count is not None and len(matches) >= max_count:
                        return GrepResult(matches=matches, truncated=True)
        return GrepResult(matches=matches)

    async def aglob(self, pattern: str, path: str | None = None) -> GlobResult:
        if path is not None and not _within(path):
            return GlobResult(error=_outside(path))
        matches = [
            FileInfo(path=_path_of(item), size=len(_content_of(item)))
            for item in await self._memories()
            if fnmatch.fnmatch(
                _path_of(item), pattern if pattern.startswith("/") else f"*{pattern}"
            )
        ]
        return GlobResult(matches=matches)

    # ------------------------------------------------------------------ sync (the protocol's)
    # Deep Agents calls the sync methods for a synchronous graph. The Memory Service client is
    # async-only, so these run on the harness's shared bridge loop rather than pretending to
    # be blocking implementations; an async graph never reaches them.
    def ls(self, path: str) -> LsResult:
        return run_sync(self.als(path))

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        return run_sync(self.aread(file_path, offset, limit))

    def write(self, file_path: str, content: str) -> WriteResult:
        return run_sync(self.awrite(file_path, content))

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        return run_sync(self.aedit(file_path, old_string, new_string, replace_all))

    def delete(self, file_path: str) -> DeleteResult:
        return run_sync(self.adelete(file_path))

    def grep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
        *,
        max_count: int | None = None,
    ) -> GrepResult:
        return run_sync(self.agrep(pattern, path, glob, max_count=max_count))

    def glob(self, pattern: str, path: str | None = None) -> GlobResult:
        return run_sync(self.aglob(pattern, path))

    # ------------------------------------------------------------------ internals
    async def _memories(self) -> list[Any]:
        return list(await self.memory.memories(limit=LISTING_LIMIT))

    async def _item(self, file_path: str) -> Any | None:
        """The memory a path names, whether it is addressed by memory or observation id."""
        identifier = _id_of(file_path)
        item = await self.memory.get(identifier)
        if item is not None:
            return item
        memory_id = await self._memory_from(identifier)
        return await self.memory.get(memory_id) if memory_id else None

    async def _resolve(self, identifier: str) -> str | None:
        """The memory id a path names, or ``None`` when the service holds no such memory."""
        if not identifier:
            return None
        item = await self.memory.get(identifier)
        if item is not None:
            return _attr(item, "memory_id") or identifier
        return await self._memory_from(identifier)

    async def _memory_from(self, observation_id: str | None) -> str | None:
        """The memory an observation produced.

        The service assigns the memory id during extraction and returns only the observation
        id to the writer, so this is the one link between what was written and what can be
        read back: every memory carries the evidence it came from.
        """
        if not observation_id:
            return None
        for item in await self._memories():
            if observation_id in _sources(item):
                return _attr(item, "memory_id")
        return None


def _within(path: str) -> bool:
    """Whether ``path`` is inside the directory this backend owns."""
    return path.rstrip("/") == MEMORIES_ROOT or path.startswith(f"{MEMORIES_ROOT}/")


def _outside(path: str) -> str:
    return f"{path}: this backend serves {MEMORIES_ROOT}/ only"


def _id_of(file_path: str) -> str:
    """The memory id (or the title of a memory about to be created) a path names."""
    name = file_path.rsplit("/", 1)[-1]
    return name[: -len(SUFFIX)] if name.endswith(SUFFIX) else name


def _path_of(item: Any) -> str:
    return _path(_attr(item, "memory_id") or _attr(item, "id") or "memory")


def _path(identifier: str | None) -> str:
    return f"{MEMORIES_ROOT}/{identifier or 'memory'}{SUFFIX}"


def _sources(item: Any) -> set[str]:
    """The observation ids a memory was extracted from."""
    return {
        str(source)
        for evidence in (_attr(item, "evidence") or ())
        if (source := _attr(evidence, "source_id"))
    }


def _content_of(item: Any) -> str:
    return str(_attr(item, "content") or _attr(item, "text") or "")


def _attr(item: Any, name: str) -> Any:
    """A field of a memory, whether the SDK handed back a model or a mapping."""
    if isinstance(item, dict):
        return item.get(name)
    return getattr(item, name, None)


def _window(content: str, offset: int, limit: int) -> ReadResult:
    """``content``'s lines ``offset..offset+limit``, with the pagination the protocol wants."""
    lines = content.splitlines()
    if offset >= len(lines):
        return ReadResult(file_data={"content": "", "encoding": "utf-8"})
    shown = lines[offset : offset + limit]
    end = offset + len(shown)
    return ReadResult(
        file_data={"content": "\n".join(shown), "encoding": "utf-8"},
        total_lines=len(lines),
        start_line=offset + 1,
        end_line=end,
        next_offset=end,
    )
