"""A server killed (``SIGKILL``: nothing closed — a crash, a restart) while runs it executes in its
own process are under way: an ordinary turn, a sub-agent's child run, a helper a framework
spawned in the foreground, a Claude subagent working in the background. Nothing else can run
them, and none has a deadline. The server held each in the run store while it executed it
(``holding``): its holds stop, and once the last lapses the store ends every one ``ERROR``,
saying its holder stopped — agent-runs' ticker (``end_abandoned``), here the in-process store
as it sweeps lapsed leases (at a claim). A run whose holder is alive is not touched.

The server is a real process (``tests/support/killed_server.py``) on a run store in a file
(``shared_runs.FileRuns``), as every server and worker shares agent-runs."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import sys
from pathlib import Path

import pytest

from tests.support import killed_server
from tests.support.arm import arm
from tests.support.shared_runs import FileRuns
from trellis import Harness, Settings
from trellis.contracts import RunEvent, RunStart, RunStatus
from trellis.harness.bridges import claude_live

ROOT = Path(__file__).resolve().parents[2]


async def until(check: object, seconds: float) -> None:
    async with asyncio.timeout(seconds):
        while not check():  # type: ignore[operator]  # noqa: ASYNC110 - another process
            await asyncio.sleep(0.05)


def ready(folder: Path) -> list[str]:
    return (folder / "ready").read_text().splitlines()


def stand_ins(folder: Path) -> list[int]:
    """The stand-in CLI processes the server started (each records its pid)."""
    log = folder / "cli.json.turns"
    lines = log.read_text().splitlines() if log.exists() else []
    return [json.loads(line)["pid"] for line in lines]


async def killed(folder: Path, scenario: str, *, leave: bool = False) -> list[str]:
    """A server process of ``scenario``, ``SIGKILL``-ed once its work is under way, with what
    it started (``leave``: the stand-in CLI it started is left to itself): what it said was
    under way (its ``ready``)."""
    server = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "tests.support.killed_server",
        str(folder / "runs"),
        str(folder),
        scenario,
        cwd=ROOT,
    )
    try:
        await until((folder / "ready").exists, seconds=60)
    finally:
        server.send_signal(signal.SIGKILL)
        await server.wait()
    for pid in [] if leave else stand_ins(folder):  # what the killed server started goes too
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, signal.SIGKILL)
    return ready(folder)


@pytest.mark.parametrize("scenario", ["turn", "subagent", "helper", "background"])
async def test_the_runs_a_killed_server_executed_end_once_their_holds_lapse(
    tmp_path: Path, scenario: str
) -> None:
    left = await killed(tmp_path, scenario)
    store_path = tmp_path / "runs"
    assert len(left) == killed_server.RUNNING[scenario]

    store = FileRuns(store_path)
    for run_id in left:
        record = await store.get(run_id, tenant="default")
        assert record is not None and record.status is RunStatus.RUNNING
        assert record.deadline is None and record.timeout_seconds is None
    alive_one = await store.start(
        RunStart(tenant_id="default", agent_id=killed_server.AGENT_ID, input="still here")
    )
    await store.heartbeat(alive_one.run_id, "server-2", lease_seconds=600, tenant="default")
    await asyncio.sleep(killed_server.HOLD_SECONDS + 0.5)
    assert await store.claim("w1", [killed_server.AGENT_ID]) is None  # the sweep

    for run_id in left:
        ended = await store.get(run_id, tenant="default")
        assert ended is not None and ended.status is RunStatus.ERROR, (run_id, ended)
        assert ended.error is not None
        assert (ended.error.code, ended.error.retryable) == ("holder_lost", False)
        assert "its holder stopped: trellis@" in ended.error.message
    kept = await store.get(alive_one.run_id, tenant="default")
    assert kept is not None and kept.status is RunStatus.RUNNING


async def test_an_answer_to_a_background_subagent_a_killed_server_left_waiting_ends_its_run(
    tmp_path: Path,
) -> None:
    """A Claude subagent at work in the background asked a person (its run ``PAUSED``), the
    conversation's next turn asked it (that turn ``PAUSED`` on it), and the server was killed.
    The person answers the turn on another server: the subagent's task went with the killed
    server's Claude session, so nothing can continue it. Its run records the answer and ends
    ``ERROR`` (``holder_lost``), saying why; the turn says so too (``claude_session``) and goes
    on with its own prompt — never a subagent's run left waiting for an answer that came."""
    left = await killed(tmp_path, "asked")
    said = json.loads(left[0])
    store = FileRuns(tmp_path / "runs")
    waiting = await store.get(said["child"], tenant="default")
    assert waiting is not None and waiting.status is RunStatus.PAUSED

    async with Harness(config=Settings(), runs=store) as other:
        target = killed_server.claude(tmp_path, [{"text": "carried on"}])
        agent = arm(other.wrap(target, id=killed_server.AGENT_ID), [killed_server.refund])
        events: list[RunEvent] = []

        async def follow() -> None:
            async for event in agent.events(said["turn"]):
                events.append(event)

        following = asyncio.create_task(follow())
        await until(lambda: agent.watchers(said["turn"]), seconds=5)
        done = await agent.resume(said["interrupt"], "approve", reviewer="u1")
        await following
    assert done.status is RunStatus.SUCCESS and done.answer == "carried on"
    ended = await store.get(said["child"], tenant="default")
    assert ended is not None and ended.status is RunStatus.ERROR and ended.error is not None
    assert (ended.error.code, ended.error.retryable) == ("holder_lost", False)
    assert ended.error.message == claude_live.GONE
    [answered] = (await store.resolutions(said["child"], tenant="default")).items
    assert answered.resolution.decision.value == "APPROVE"
    warned = [e.data["message"] for e in events if e.data.get("code") == "claude_session"]
    assert warned == [claude_live.GONE_SAID]


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"


async def test_a_cli_its_killed_server_left_idle_ends_at_once(tmp_path: Path) -> None:
    """As the bundled CLI (2.1.283) does when its parent dies — its stdin closes — with
    nothing in the background: it ends. (Probed: the real CLI under the real SDK, the parent
    ``SIGKILL``-ed, against a scripted model on 127.0.0.1.)"""
    await killed(tmp_path, "idle", leave=True)
    [cli] = stand_ins(tmp_path)
    await until(lambda: not alive(cli), seconds=5)


async def test_a_cli_its_killed_server_left_working_ends_when_its_background_work_does(
    tmp_path: Path,
) -> None:
    """As the bundled CLI (2.1.283) does when its parent dies with a subagent at work in the
    background: the subagent goes on to its end, and then the process ends. A call it makes
    meanwhile has nobody to decide it (its hooks and permission check were the SDK's): it does
    not run. (Probed as above: the real subagent's ``Bash`` call did not run, its process
    ended once the subagent's work did.) The run of that work was ended by the run store
    already (``holder_lost``): the harness does not stop it."""
    await killed(tmp_path, "background", leave=True)
    [cli] = stand_ins(tmp_path)
    try:
        await asyncio.sleep(1)
        assert alive(cli)  # its work goes on
        tasks = tmp_path / "tasks"
        (tasks / "t1.call").write_text(json.dumps({"builtin": "Bash", "args": {"command": "ls"}}))
        await until((tasks / "t1.called").exists, seconds=5)
        called = json.loads((tasks / "t1.called").read_text())
        assert called["ok"] is False and not (tmp_path / "builtins.jsonl").exists()
        assert alive(cli)
        (tasks / "t1.done").write_text("the market is growing")
        await until(lambda: not alive(cli), seconds=5)
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.kill(cli, signal.SIGKILL)
