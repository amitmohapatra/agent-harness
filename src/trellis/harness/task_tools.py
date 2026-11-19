"""The five background-task tools — Deep Agents' async sub-agent tools, by name — in one table:
what each does, and what its result is. One interface, governed one way wherever the task runs:
on a server of yours (Deep Agents' own tools: ``bridges.async_tasks``) or in the harness
(``background``, ``bridges.background_tasks``). And the source a task's notification is marked
with."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

from trellis.harness.tools.base import FRESHNESS, STATIC, SideEffects

START: Final = "start_async_task"
CHECK: Final = "check_async_task"
UPDATE: Final = "update_async_task"
CANCEL: Final = "cancel_async_task"
LIST: Final = "list_async_tasks"
#: What each tool does: a task started, sent new instructions or cancelled is a write — the
#: sub-agent's own calls are governed in its run, so the write is announced on the run's
#: stream and nobody is asked, unless the tool catalog says more; its status and the list of
#: tasks are reads.
TOOLS: Final[Mapping[str, SideEffects]] = {
    START: "write",
    CHECK: "read",
    UPDATE: "write",
    CANCEL: "write",
    LIST: "read",
}
#: The notification source of a task's outcome (Deep Agents' ``lc_source``): its data is the
#: ``TaskEvent``.
SOURCE: Final = "async_subagent"
#: The statuses of a task that has not ended (Deep Agents' ``AsyncTask.status``), in the order
#: a task goes through them.
UNFINISHED: Final = ("pending", "running", "waiting")


def result_of(name: str) -> Mapping[str, Any]:
    """What the tool ``name`` says of its result: a write's is the record of what was done
    (the task's id, that it was sent or stopped), no value that changes — an earlier one is
    never marked out of date with the advice to call the tool again, which would start a
    second task. A status and the list are values that change: live, as any tool's by default
    (it says nothing)."""
    return {FRESHNESS: STATIC} if TOOLS[name] == "write" else {}
