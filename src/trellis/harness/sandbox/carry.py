"""A conversation's working files, carried from one turn to the next: the files a run wrote in
its isolated workspace (a sandbox's, this machine's — Claude Code's and Kimi Code's among
them), which goes with the run, kept in the file store under the conversation and given back
to the conversation's next run, in its fresh workspace, before its first command.

* **kept** (:func:`kept`) when a run ends in a way that keeps state — it succeeded, or it
  paused — as one archive (:data:`ARCHIVE`) in a folder of its own (:data:`KEPT_IN`) in the
  conversation's folder of what it carries, the graph state's
  (``.carried/<tenant>/<user>/<conversation>/<agent>/workspace``), with its record (the
  files in it, those left out): every file under the working directory but the user's
  :data:`~trellis.harness.sandbox.OUTPUTS` (delivered as ever) and the caches
  (:data:`SKIPPED`). A file past :data:`FILE_BYTES` is left out, and past
  :data:`TOTAL_BYTES` all told the oldest go first; the run is told which (a warning). A run
  that failed keeps nothing new: the next run gets what the last kept.
* **given back** (:func:`restored`) to the next run of the same tenant, user, conversation and
  agent — never another's: the folder is theirs alone — when its workspace is made; the model
  is told in one line what came back (:func:`told`), with its first command's result (Claude
  Code: with its context).

What is kept goes with the conversation's other files in the store (its lifecycle)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Final

from trellis.harness.files import part

if TYPE_CHECKING:
    from trellis.harness.runtime import Runtime
    from trellis.harness.sandbox.base import Sandbox

log = logging.getLogger("trellis.sandbox")

#: The archive a conversation's working files are kept in, and what made it (its source).
ARCHIVE: Final = "workspace.tar"
SOURCE: Final = "workspace"
#: The folder the archive is kept in, in the conversation's folder of what it carries: not
#: beside the graph state's files there, each of which a turn reads as one (``pipeline``).
KEPT_IN: Final = "workspace"
#: The largest file carried (the largest output kept), and the most carried all told: what one
#: read of a workspace's file takes (``isolation.READ_MAX_BYTES``), less the archive's own room.
FILE_BYTES: Final = 50 * 1024 * 1024
TOTAL_BYTES: Final = 60 * 1024 * 1024
#: What is never carried: the workspace's own temporary folder and kept results (the harness
#: writes them again), caches, and the archive itself while it is made.
SKIPPED: Final = (".tmp", ".results", "__pycache__", ".cache", ".trellis-carry*")
#: Where the archive and its list are made in the workspace, while they are.
STAGED: Final = ".trellis-carry.tar"
LISTED: Final = ".trellis-carry.list"
#: Each file under the working directory (not the user's outputs, nor :data:`SKIPPED`): its
#: modification time, its size and its path (``./``...), one a line.
FILES: Final = (
    "find . \\( "
    + " -o ".join(f"-name '{name}'" for name in SKIPPED)
    + " -o -path ./outputs \\) -prune -o -type f -exec stat -c '%Y %s %n' {} + 2>/dev/null; true"
)
#: The archive made of the files :data:`LISTED` names (one a line).
PACK: Final = f"tar -cf {STAGED} -T {LISTED} && rm -f {LISTED}"
#: The archive unpacked where it was written, then removed.
UNPACK: Final = f"tar -xf {STAGED}; s=$?; rm -f {STAGED}; exit $s"
#: How many files the line a run is told names; the rest are counted.
NAMED: Final = 20
#: What a run is told of the files it got back, and of those its conversation left out.
RESTORED: Final = (
    "Restored from this conversation's earlier turns into your working directory: {files}."
)
LEFT_OUT: Final = "Left out, past what a conversation carries: {files}."


def folder(runtime: Runtime) -> str:
    """Where the conversation's working files are kept: in its folder of what it carries
    (:data:`KEPT_IN`)."""
    from trellis.harness.pipeline import CARRIED_FOLDER  # noqa: PLC0415 - imports this

    scope = (runtime.tenant, runtime.user, runtime.conversation, runtime.agent_id)
    return "/".join([CARRIED_FOLDER, *map(part, scope), KEPT_IN])


def _carries(runtime: Runtime) -> bool:
    from trellis.harness.bridges.live import converses  # noqa: PLC0415 - imports this

    return runtime.user is not None and converses(runtime)


def _listed(files: list[str]) -> str:
    more = f" and {len(files) - NAMED} more" if len(files) > NAMED else ""
    return ", ".join(files[:NAMED]) + more


def chosen(listing: str) -> tuple[list[str], list[str]]:
    """The files a listing (:data:`FILES`) names that are carried — the newest first while
    they fit :data:`TOTAL_BYTES`, none past :data:`FILE_BYTES` — and those left out."""
    found = []
    for line in listing.splitlines():
        stamp, _, rest = line.partition(" ")
        size, _, path = rest.partition(" ")
        if path.startswith("./") and stamp.isdigit() and size.isdigit():
            found.append((int(stamp), int(size), path[2:]))
    kept: list[str] = []
    left: list[str] = []
    total = 0
    for _, size, path in sorted(found, key=lambda f: (-f[0], f[2])):
        if size > FILE_BYTES or total + size > TOTAL_BYTES:
            left.append(path)
        else:
            kept.append(path)
            total += size
    return sorted(kept), sorted(left)


async def kept(runtime: Runtime, box: Sandbox) -> None:
    """The run ends keeping state: the files of its workspace ``box`` kept for the
    conversation's next run (the module's rules). A failure is a warning: the run ends as it
    does."""
    if not _carries(runtime):
        return
    store = runtime.agent.harness.files
    try:
        listing = await box.exec(FILES)
        files, left = chosen(listing.stdout)
        if left:
            runtime.events.warning(
                "workspace_carried", f"not carried to the conversation's next turn: {_listed(left)}"
            )
        if not files:
            await store.remove(folder(runtime), ARCHIVE)
            return
        await box.write(LISTED, "".join(f"{f}\n" for f in files).encode())
        packed = await box.exec(PACK)
        if packed.exit_code != 0:
            raise RuntimeError(packed.stderr.strip() or f"tar ended {packed.exit_code}")
        data = await box.read(STAGED)
        await box.exec(f"rm -f {STAGED}")
        await store.put(folder(runtime), ARCHIVE, data, source=SOURCE)
        await store.keep_record(folder(runtime), ARCHIVE, {"files": files, "left": left})
    except Exception as exc:
        runtime.events.warning("workspace_carried", f"the working files were not kept: {exc}")


async def restored(runtime: Runtime, box: Sandbox) -> None:
    """The run's workspace ``box`` was made: the files its conversation kept (:func:`kept`)
    written into it, and what the model is told of them kept for it (:func:`told`). A failure
    is a warning: the run goes on in an empty workspace."""
    if not _carries(runtime):
        return
    store = runtime.agent.harness.files
    try:
        data = await store.read(folder(runtime), ARCHIVE)
        if data is None:
            return
        record: dict[str, Any] = await store.record(folder(runtime), ARCHIVE) or {}
        await box.write(STAGED, data)
        unpacked = await box.exec(UNPACK)
        if unpacked.exit_code != 0:
            raise RuntimeError(unpacked.stderr.strip() or f"tar ended {unpacked.exit_code}")
    except Exception as exc:
        runtime.events.warning("workspace_carried", f"the working files were not restored: {exc}")
        return
    said = RESTORED.format(files=_listed(list(record.get("files") or ())))
    if record.get("left"):
        said = f"{said} {LEFT_OUT.format(files=_listed(list(record['left'])))}"
    runtime.restored = said


def told(runtime: Runtime | None) -> str | None:
    """What the model is told of the files its run got back (:func:`restored`), once: the
    first result that can say it takes it."""
    if runtime is None:
        return None
    said, runtime.restored = runtime.restored, None
    return said


__all__ = ["ARCHIVE", "FILE_BYTES", "TOTAL_BYTES", "chosen", "kept", "restored", "told"]
