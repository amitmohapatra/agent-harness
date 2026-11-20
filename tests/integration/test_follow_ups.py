"""Long-running work (item 10): a tool declared ``long_running`` starts a job and the run goes on;
the harness checks the job itself, with the status tool the job tool names and no model call,
in one-time follow-ups of the chat (agent-runs' schedules: durable, fired once), the gaps
growing; when the job is done or failed, or after a day, one run of the chat continues the task
from its result, and the chain ends.
"""

from __future__ import annotations

from collections import deque
from datetime import UTC, datetime, timedelta
from typing import Any

from tests.support.arm import arm
from trellis import Harness, Runtime, Settings, tool
from trellis.contracts import RunStatus
from trellis.harness import follow_ups
from trellis.harness.follow_ups import FOLLOW, GAPS
from trellis.harness.runs import LocalRuns
from trellis.harness.schedules import NO_REPORT, TASK
from trellis.harness.tools.convert import text_of
from trellis.runs import ScheduleUpdate

THREAD = "chat-jobs"
#: What each status check answers next.
statuses: deque[Any] = deque()


@tool(side_effects="read", long_running="export_status")
def start_export(report: str) -> dict[str, str]:
    """Start exporting a report; it takes a while.

    Args:
        report: Which report.
    """
    return {"job_id": "J1", "status": "running"}


@tool(side_effects="read")
def export_status(job_id: str) -> dict[str, str]:
    """Where an export is.

    Args:
        job_id: The export's id.
    """
    answer = statuses.popleft()
    if isinstance(answer, Exception):
        raise answer
    return {"job_id": job_id, **answer}


@tool(side_effects="read", long_running="post_note")
def wrongly_followed(report: str) -> str:
    """A job whose status tool is not a read.

    Args:
        report: Which report.
    """
    return "J9"


@tool(side_effects="write")
def post_note(job: str) -> str:
    """Post a note: a write, so no status tool.

    Args:
        job: Which job.
    """
    return "posted"


class Desk:
    """In chat: starts the export; a follow-up's continuation is answered with what it read."""

    def __init__(self) -> None:
        self.inputs: list[str] = []
        self.said: list[str] = []

    async def __call__(self, input: str, agent: Runtime) -> str:
        self.inputs.append(input)
        if input == "list":
            return text_of(await agent.tools.call("list_schedules"))
        if input.startswith("cancel "):
            return text_of(await agent.tools.call("cancel_schedule", id=input.split()[1]))
        if not input.startswith("export"):
            return f"Reported: {input}"
        said = text_of(await agent.tools.call("start_export", report="sales"))
        self.said.append(said)
        return said


def local(h: Harness) -> LocalRuns:
    runs = h.runs
    assert isinstance(runs, LocalRuns)
    return runs


def following(h: Harness) -> list[Any]:
    return [s for s in local(h)._schedules.values() if FOLLOW in s.metadata]


async def started(h: Harness, desk: Desk) -> Any:
    agent = arm(h.wrap(desk, id="desk"), [start_export, export_status])
    result = await agent.run("export the sales report", user="ada", thread=THREAD)
    assert result.status is RunStatus.SUCCESS, result.error
    return agent


async def fire(h: Harness, agent: Any, schedule_id: str) -> Any:
    fired = await local(h).schedules.fire(schedule_id)
    assert await h.worker([agent]).run_once() is True
    return await h.runs.get(fired.run_id)


async def test_a_long_job_started_is_followed_up_by_the_harness_not_the_model() -> None:
    """The job tool's result tells the model the harness checks it; one follow-up is made, one
    time, in the chat, a first gap away; it is not the user's schedule (no limit counts it)."""
    statuses.clear()
    desk = Desk()
    async with Harness(config=Settings(schedule_max_per_user=0)) as h:
        await started(h, desk)
        assert "checks it" in desk.said[0] and "do not check it yourself" in desk.said[0]
        [follow] = following(h)
        assert follow.metadata[TASK]["once"] is True and follow.metadata[FOLLOW]["job"] == (
            "start_export"
        )
        assert follow.metadata[FOLLOW]["args"] == {"job_id": "J1"}
        due = follow.next_fire_at
        assert due is not None
        assert (
            timedelta(minutes=GAPS[0] - 1)
            < due - datetime.now(UTC)
            <= timedelta(minutes=GAPS[0] + 1)
        )


async def test_a_job_still_running_is_checked_again_later_with_no_model_call() -> None:
    statuses.clear()
    desk = Desk()
    async with Harness(config=Settings()) as h:
        agent = await started(h, desk)
        [first] = following(h)
        statuses.append({"status": "running"})
        run = await fire(h, agent, first.schedule_id)
        assert run.status is RunStatus.SUCCESS and run.output == NO_REPORT  # told nobody
        assert desk.inputs == ["export the sales report"]  # no model call for the check
        [second] = following(h)
        assert second.schedule_id != first.schedule_id  # the first ended, a new one is due
        assert second.metadata[FOLLOW]["attempt"] == 1
        gap = (second.next_fire_at or datetime.now(UTC)) - datetime.now(UTC)
        assert timedelta(minutes=GAPS[1] - 1) < gap <= timedelta(minutes=GAPS[1] + 1)


async def test_a_job_done_or_failed_continues_the_task_once_and_the_chain_ends() -> None:
    for answer, said in (
        ({"status": "done", "rows": "42"}, "finished"),
        (
            {"status": "failed", "reason": "disk full"},
            "failed",
        ),
    ):
        statuses.clear()
        desk = Desk()
        async with Harness(config=Settings()) as h:
            agent = await started(h, desk)
            [first] = following(h)
            statuses.append(answer)
            run = await fire(h, agent, first.schedule_id)
            assert run.status is RunStatus.SUCCESS
            continued = desk.inputs[-1]
            assert said in continued and "export the sales report" in continued
            assert str(next(iter(answer.values()))) in continued or "disk full" in continued
            assert run.thread_id == THREAD and not following(h)


async def test_a_job_never_done_is_given_up_after_a_day_and_the_user_told() -> None:
    statuses.clear()
    desk = Desk()
    async with Harness(config=Settings()) as h:
        agent = await started(h, desk)
        [first] = following(h)
        old = (datetime.now(UTC) - follow_ups.GIVE_UP - timedelta(minutes=1)).isoformat()
        kept = {**first.metadata[FOLLOW], "since": old}
        await local(h).schedules.update(first.schedule_id, ScheduleUpdate(metadata={FOLLOW: kept}))
        statuses.append({"status": "running"})
        await fire(h, agent, first.schedule_id)
        assert "did not finish" in desk.inputs[-1] and not following(h)


async def test_a_status_that_cannot_be_read_is_checked_again_not_taken_as_a_failure() -> None:
    statuses.clear()
    desk = Desk()
    async with Harness(config=Settings()) as h:
        agent = await started(h, desk)
        [first] = following(h)
        statuses.append(RuntimeError("status service down"))
        run = await fire(h, agent, first.schedule_id)
        assert run.output == NO_REPORT and len(desk.inputs) == 1
        [again] = following(h)
        assert again.metadata[FOLLOW]["attempt"] == 1


async def test_a_job_being_followed_is_listed_and_cancelled_like_a_schedule() -> None:
    statuses.clear()
    desk = Desk()
    async with Harness(config=Settings()) as h:
        agent = await started(h, desk)
        [follow] = following(h)
        listed = await agent.run("list", user="ada", thread=THREAD)
        assert "waiting" in str(listed.answer) and "start_export" in str(listed.answer)
        await agent.run(f"cancel {follow.schedule_id}", user="ada", thread=THREAD)
        assert not following(h)


async def test_a_status_tool_that_writes_is_refused_as_a_follow_up() -> None:
    """Checking a job runs unasked, so its status tool must only read: a job tool naming one
    that writes gets no follow-up, and the model is told why."""

    async def desk(input: str, agent: Runtime) -> str:
        return text_of(await agent.tools.call("wrongly_followed", report="x"))

    async with Harness(config=Settings()) as h:
        agent = arm(h.wrap(desk, id="desk"), [wrongly_followed, post_note])
        result = await agent.run("go", user="ada", thread=THREAD)
        assert not following(h) and "is not a read" in str(result.answer), result.answer
