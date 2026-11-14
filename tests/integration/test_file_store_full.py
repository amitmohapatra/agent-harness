"""A file that could not be kept leaves nothing behind (SCN-14, found by the scenario suite on a
64 KiB file system: a 512 KiB report failed with "No space left on device" and left its first
64 KiB there, so the 8 KiB report made next failed too)."""

from __future__ import annotations

import errno
from pathlib import Path
from typing import Any

import pytest

from trellis.harness.files import FileStore


def _a_disk_that_fills(store: FileStore, room: int, monkeypatch: pytest.MonkeyPatch) -> None:
    """``store``'s file system writes at most ``room`` bytes of a file, then is full (undone at
    the test's end: fsspec shares one local file system between stores)."""
    pipe = store.fs.pipe_file

    def full(path: str, data: bytes, **kwargs: Any) -> None:
        if len(data) > room:
            pipe(path, data[:room], **kwargs)
            raise OSError(errno.ENOSPC, "No space left on device")
        pipe(path, data, **kwargs)

    monkeypatch.setattr(store.fs, "pipe_file", full)


async def test_a_file_that_did_not_fit_leaves_nothing_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = FileStore(tmp_path.as_uri())
    _a_disk_that_fills(store, 10, monkeypatch)
    with pytest.raises(OSError, match="No space left"):
        await store.put("run-1", "report.bin", b"x" * 100, source="big_report")
    assert not store.fs.exists(store.path("run-1", "report.bin"))
    # and a file that fits is kept as before
    kept = await store.put("run-1", "small.bin", b"x" * 8, source="big_report")
    assert kept.size == 8
    assert store.fs.cat_file(store.path("run-1", "small.bin")) == b"x" * 8


async def test_a_failed_write_keeps_the_file_of_that_name_gone_not_half(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same name written again and failing: the half-written replacement is removed (the
    earlier file was already being replaced: there is no old copy to give back)."""
    store = FileStore(tmp_path.as_uri())
    await store.put("run-1", "report.bin", b"old", source="t")
    _a_disk_that_fills(store, 2, monkeypatch)
    with pytest.raises(OSError, match="No space left"):
        await store.put("run-1", "report.bin", b"new and longer", source="t")
    assert not store.fs.exists(store.path("run-1", "report.bin"))
