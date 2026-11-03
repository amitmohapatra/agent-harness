"""What a run did, grouped as AG-UI's ``ACTIVITY_SNAPSHOT``: one line a chat UI shows
collapsed above the answer — "used 3 tools · ran 1 command" — replaced as each tool call ends.

It is derived from the tool events the stream already carries (each call once, by its id: a
call replayed on resume is not counted twice), so nothing is recorded for it; a reconnect served
from agent-runs' event log (another replica) derives it again. Each line shares the number of
the result it follows (``hub.RunBuffer.publish(derived=True)``).
"""

from __future__ import annotations

from typing import Final

from trellis.contracts import RunEvent, RunEventType
from trellis.harness.agui.events import AGUIEvent, AGUIEventType
from trellis.harness.sandbox import EXEC
from trellis.harness.tools.bridge import PAUSED
from trellis.harness.tools.results import EXECUTE

#: The ``activityType`` of the line.
ACTIVITY_TYPE: Final = "tools"
#: The tools that run a command: the harness's sandbox, Deep Agents' backend, Claude's shell.
COMMANDS: Final = frozenset({EXEC, EXECUTE, "Bash"})


class Activity:
    """One run's line, as its tool calls end."""

    def __init__(self) -> None:
        #: the calls that ended, by id, and how many of them ran a command
        self.calls: set[str] = set()
        self.commands = 0

    def seen(self, event: RunEvent) -> AGUIEvent | None:
        """The line again after ``event`` when it ends a call not counted yet; else ``None``."""
        call = event.tool_call_id
        if (
            event.type is not RunEventType.TOOL_CALL_RESULT
            or event.data.get("status") == PAUSED  # it ends when the run resumes
            or call is None
            or call in self.calls
        ):
            return None
        self.calls.add(call)
        self.commands += event.data.get("tool") in COMMANDS
        commands, tools = self.commands, len(self.calls) - self.commands
        return AGUIEvent(
            type=AGUIEventType.ACTIVITY_SNAPSHOT,
            timestamp=int(event.timestamp.timestamp() * 1000),
            thread_id=event.thread_id,
            run_id=event.run_id,
            message_id=f"{event.run_id}:activity",
            activity_type=ACTIVITY_TYPE,
            content={"summary": summary(tools, commands), "tools": tools, "commands": commands},
            replace=True,
        )


def summary(tools: int, commands: int) -> str:
    """``used 3 tools · ran 1 command`` (a part with nothing to say left out)."""
    parts = [
        f"{verb} {n} {noun}{'' if n == 1 else 's'}"
        for verb, n, noun in (("used", tools, "tool"), ("ran", commands, "command"))
        if n
    ]
    return " · ".join(parts)
