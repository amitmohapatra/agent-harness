"""What the suite approves when a run pauses to ask a person, as a user reading the request
would: a shell command that only reads, and one that writes only inside the run's own working
folder ("save this file" there). Both are read by the harness's own check
(``trellis.harness.governance.shell``: one implementation): with it a read-only command no
longer pauses at all, so what still pauses here is a save the framework asked about (Claude
Code's or Kimi Code's own permission check). A command that writes outside that folder,
deletes or moves anything, installs, reaches the network or cannot be read for certain is left
paused, and counts as a pause."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

from trellis.contracts import RunStatus
from trellis.harness.governance.shell import SHELLS, approvable, file_of, read_only
from trellis.harness.tools import results

__all__ = ["APPROVALS_MAX", "SHELLS", "approvable", "approved_reads", "read_only"]

#: The most commands one turn is resumed past (a loop is a pause).
APPROVALS_MAX: Final = 8


async def approved_reads(
    agent: Any, result: Any, reviewer: str, folder: str | Path | None = None
) -> tuple[Any, list[str]]:
    """``result`` — a run's — resumed approved, as often as it pauses on a call the suite
    approves (:func:`approvable` with the run's working ``folder``, at most
    :data:`APPROVALS_MAX` times), and those calls; a pause on anything else is left as it is."""
    commands: list[str] = []
    for _ in range(APPROVALS_MAX):
        pending = result.interrupt if result.status is RunStatus.PAUSED else None
        call = pending.tool_call if pending is not None else None
        kept = [results.PREFIX]  # where a kept result is read: the run may read it, as its folder
        if (
            pending is None
            or call is None
            or not approvable(call.tool, call.args, folder, reads=kept)
        ):
            break
        commands.append(str(call.args.get("command") or f"{call.tool} {file_of(call.args)}"))
        result = await agent.resume(pending.interrupt_id, "approve", reviewer=reviewer)
    return result, commands
