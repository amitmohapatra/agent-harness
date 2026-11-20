"""Long-running work: a tool whose call starts a job that takes a while (``tool(long_running=
"export_status")``, :data:`~trellis.harness.tools.base.LONG_RUNNING`) is followed up by the
harness, never polled by the model, and no run waits on it.

* **Started.** The job tool's call ends as any call; its result tells the model the harness
  checks it and continues here (:data:`STARTED`), and the first follow-up is made
  (:func:`started`): a one-time schedule of the run's user in its chat, agent-runs' (durable,
  fired once, a worker held only while it fires), not one of the user's schedules — no limit
  counts it, and ``list_schedules`` says what it waits for.
* **Checked.** Its run reads the job's status with the status tool (a read, governed as any
  call) before any model call (:func:`checked`): still running, the next follow-up is made
  :data:`GAPS` later and the run says nothing (``NO_REPORT``); done, failed, or running past
  :data:`GIVE_UP`, the run continues the task from what the status said — one model run, in the
  chat — and the chain ends. A status that cannot be read is checked again, not taken as a
  failure.
* **Status.** The status tool's result says where the job is in ``status``: :data:`DONE`,
  :data:`FAILED`, or anything else (still under way).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Final

from trellis.contracts import ScheduleSpec, ToolOutcome, ToolStatus
from trellis.harness import schedules
from trellis.harness.tools.base import LONG_RUNNING

if TYPE_CHECKING:
    from trellis.harness.runtime import Runtime
    from trellis.harness.tools.base import Tool

#: The schedule metadata key of a follow-up (``schedules.FOLLOW``): the job tool, its status
#: tool and the arguments it reads the job with, the checks before, when the job started, the
#: task.
FOLLOW: Final = schedules.FOLLOW
#: Minutes from one check of a job to the next, the last one repeated: soon at first, then
#: hourly.
GAPS: Final = (2, 5, 15, 30, 60)
#: How long a job is followed before the task continues without it.
GIVE_UP: Final = timedelta(hours=24)
#: The statuses that end a job; any other is "still under way".
DONE: Final = "done"
FAILED: Final = "failed"
#: What the model reads after a job tool's result.
STARTED: Final = (
    "{tool} started a job that takes a while. The harness checks it with {status} and "
    "continues this task in this chat when it is done or has failed: do not check it yourself "
    "now. Tell the user it is under way."
)
#: What it reads instead when the job cannot be followed.
UNFOLLOWED: Final = (
    "{tool} started a job, but it cannot be followed up: {status} is not a read here. Tell the "
    "user to ask about it later."
)
#: What a follow-up's run continues the task with.
ENDED: Final = {
    DONE: "The job {tool} started for the task '{task}' has finished: {said}. Continue the task "
    "from this and tell the user.",
    FAILED: "The job {tool} started for the task '{task}' failed: {said}. Tell the user what "
    "failed, and what can be done.",
    None: "The job {tool} started for the task '{task}' did not finish in {hours} hours: {said}. "
    "It is no longer followed up. Tell the user, and what they can do.",
}


async def started(runtime: Runtime, tool: Tool, outcome: ToolOutcome) -> ToolOutcome:
    """A call of a ``long_running`` tool that ran: its first follow-up made, and what the model
    reads of it said; any other call as it ran."""
    status = (tool.spec.model_extra or {}).get(LONG_RUNNING)
    if not status or outcome.status is not ToolStatus.OK:
        return outcome
    reader = runtime.toolbox.get(status)
    if reader is None or reader.spec.side_effects != "read":
        said = UNFOLLOWED.format(tool=tool.name, status=status)
        return outcome.model_copy(update={"output": f"{outcome.output}\n{said}"})
    follow = {
        "job": tool.name,
        "status": status,
        "args": _read_with(reader, outcome.output),
        "attempt": 0,
        "since": datetime.now(UTC).isoformat(),
        "task": runtime.task or "",
    }
    await _next(runtime, follow)
    said = STARTED.format(tool=tool.name, status=status)
    return outcome.model_copy(update={"output": f"{outcome.output}\n{said}"})


async def checked(runtime: Runtime) -> str | None:
    """Before a follow-up's run works: the job read with its status tool. ``None``: not a
    follow-up; :data:`~trellis.harness.schedules.NO_REPORT`: still under way, the next one made;
    else what the run continues the task with."""
    fired = runtime.scheduled
    follow = fired.follow if fired is not None else None
    if not follow:
        return None
    try:
        result: Any = await runtime.tools.call(follow["status"], **follow["args"])
        state = _state(result)
    except Exception as exc:  # unread is not failed: read again at the next check
        result, state = f"its status could not be read ({exc})", None
    since = datetime.fromisoformat(follow["since"])
    if state not in (DONE, FAILED) and datetime.now(UTC) - since < GIVE_UP:
        await _next(runtime, {**follow, "attempt": int(follow["attempt"]) + 1})
        return schedules.NO_REPORT
    said = result if isinstance(result, str) else json.dumps(result, default=str)
    return ENDED[state if state in (DONE, FAILED) else None].format(
        tool=follow["job"],
        task=follow["task"],
        said=said,
        hours=int(GIVE_UP.total_seconds() // 3600),
    )


async def _next(runtime: Runtime, follow: dict[str, Any]) -> None:
    """The follow-up after ``follow``'s checks: one time, :data:`GAPS` from now, the run's user's
    in its chat."""
    gap = GAPS[min(int(follow["attempt"]), len(GAPS) - 1)]
    when, _ = schedules._in_minutes(float(gap), "UTC")
    held = await schedules.limits(runtime)
    about = {"task": f"waiting for {follow['job']}", "condition": None, "once": True}
    metadata = schedules._made_metadata(runtime, held, about, "UTC", None)
    metadata[FOLLOW] = follow
    agent = runtime.agent
    spec = ScheduleSpec(
        tenant_id=runtime.tenant,
        agent_id=agent.id,
        name=f"Follow-up of {follow['job']}"[:200],
        cadence=when,
        timezone="UTC",
        on_behalf_of=runtime.user,
        input=f"Follow-up: {follow['job']}",
        timeout_seconds=min(t for t in (agent.timeout, held.run_timeout) if t is not None),
        agent_version=agent.version,
        metadata=metadata,
    )
    await agent.harness.runs.schedules.create(spec)


def _read_with(reader: Tool, output: Any) -> dict[str, Any]:
    """The status tool's arguments from the job tool's result: its fields the status tool
    takes, or, one argument it needs, the whole result."""
    taken = set(((reader.spec.input_schema or {}).get("properties") or {}).keys())
    found = output if isinstance(output, dict) else _json(output)
    if isinstance(found, dict) and (picked := {k: v for k, v in found.items() if k in taken}):
        return picked
    needed = (reader.spec.input_schema or {}).get("required") or []
    return {needed[0]: str(output)} if len(needed) == 1 else {}


def _state(result: Any) -> str | None:
    """Where a status tool's result says the job is: its ``status``."""
    found = result if isinstance(result, dict) else _json(result)
    status = found.get("status") if isinstance(found, dict) else None
    return str(status).lower() if status is not None else None


def _json(text: Any) -> Any:
    try:
        return json.loads(text) if isinstance(text, str) else None
    except ValueError:
        return None
