"""Kimi Code's own ``Bash`` on this machine runs through the harness's isolation layer: the
real ``kimi acp`` (``tests/support/kimi.py``) with no ``sandbox=`` is given ACP's terminal and
file capabilities, served by a workspace of this machine's layer at the folder it works in
(``adapters.acp.Sandboxed``) — each command the run's uid's, in the run's workspace. Each test
tries what a run's command must not reach (the service's environment, another run's
workspace) or checks the clean-up, a machine that cannot isolate, and the working files a
conversation carries. Run as root, or with a password-less sudo."""

from __future__ import annotations

import os
import secrets
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest

from tests.support.isolation import fresh_root, processes_of, root_helper
from tests.support.kimi import PlannedEndpoint, kimi_target, needs_kimi
from trellis import Harness, Settings
from trellis.contracts import RunEventType
from trellis.harness.sandbox.base import RUN
from trellis.harness.sandbox.isolation import HostSandbox, use_host

pytestmark = needs_kimi
SECRET = "sk-service-kimi-0123456789"


@pytest.fixture
def helper() -> tuple[str, ...]:
    return root_helper()


@pytest.fixture
def root(helper: tuple[str, ...]) -> Iterator[str]:
    yield from fresh_root(helper)


@pytest.fixture
def layer(helper: tuple[str, ...], root: str) -> Iterator[HostSandbox]:
    base = 1_600_000_000 + secrets.randbelow(100_000) * 1_000
    made = HostSandbox(root, helper=helper, uid_base=base, tenants=1)
    replaced = use_host(made)
    try:
        yield made
    finally:
        use_host(replaced)


@pytest.fixture
async def endpoint() -> AsyncIterator[PlannedEndpoint]:
    async with PlannedEndpoint().serving() as served:
        yield served


async def test_kimis_bash_runs_in_the_runs_workspace_as_its_uid_with_no_secret_of_the_service(
    harness: Harness,
    layer: HostSandbox,
    endpoint: PlannedEndpoint,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRELLIS_TEST_SECRET", SECRET)
    endpoint.plan = [
        ("Bash", {"command": "echo made > made.txt; id -u; pwd; env; ls /proc | grep -c '^[0-9]'"})
    ]
    agent = harness.wrap(kimi_target(endpoint, tmp_path), id="kimi-iso")
    events = [e async for e in agent.stream("go", user="u1")]
    finished = events[-1]
    assert finished.error is None
    answer = str(finished.data["result"])
    work = str(tmp_path / "kimi-work")
    lines = answer.removeprefix("Done. ").splitlines()
    uid = next(int(line) for line in lines if line.isdigit() and int(line) >= 1_000_000_000)
    assert uid >= layer.isolation.uid_base
    assert work in lines  # its commands start in the folder Kimi names its files by
    assert SECRET not in answer and "KIMI_" not in answer
    assert not (tmp_path / "kimi-work" / "made.txt").exists()  # written in the workspace
    audited = [
        e.data
        for e in events
        if e.type is RunEventType.CUSTOM and e.data.get("name") == "sandbox_command"
    ]
    assert [a["uid"] for a in audited] == [uid]
    # the run's end removed the workspace, and killed what was left of its uid
    assert await layer.labelled({RUN: finished.run_id}) == []
    assert processes_of(uid) == []


async def test_another_runs_workspace_is_out_of_reach_of_kimis_bash(
    harness: Harness, layer: HostSandbox, endpoint: PlannedEndpoint, tmp_path: Path
) -> None:
    other = await layer.workspace("default", "u2", "another-run")
    await other.write("notes.txt", b"the other run's")
    endpoint.plan = [("Bash", {"command": f"cat {other.home}/notes.txt; echo rc=$?"})]
    try:
        agent = harness.wrap(kimi_target(endpoint, tmp_path), id="kimi-other")
        result = await agent.run("go", user="u1")
    finally:
        await layer.delete(other.ref)
    assert "the other run's" not in str(result.answer) and "rc=1" in str(result.answer)


async def test_where_commands_cannot_be_isolated_kimis_bash_is_refused_and_says_why(
    harness: Harness, root: str, endpoint: PlannedEndpoint, tmp_path: Path
) -> None:
    unable = HostSandbox(root, helper=("false",))
    replaced = use_host(unable)
    endpoint.plan = [("Bash", {"command": "touch made.txt; id -u"})]
    try:
        agent = harness.wrap(kimi_target(endpoint, tmp_path), id="kimi-unable")
        result = await agent.run("go", user="u1")
    finally:
        use_host(replaced)
    assert "cannot be isolated" in str(result.answer)
    assert not (tmp_path / "kimi-work" / "made.txt").exists()
    assert str(os.getuid()) not in str(result.answer).split()


async def test_a_conversations_working_files_carry_to_its_next_kimi_turn(
    layer: HostSandbox, endpoint: PlannedEndpoint, tmp_path: Path
) -> None:
    endpoint.plan = [("Bash", {"command": "mkdir -p data && echo kept > data/a.txt"})]
    async with Harness(config=Settings()) as first:
        await first.wrap(kimi_target(endpoint, tmp_path), id="kimi-carry").run(
            "go", user="u1", thread="t1"
        )
    endpoint.plan = [("Bash", {"command": "cat data/a.txt"})]
    async with Harness(config=Settings()) as second:  # another worker, the same file store
        agent = second.wrap(kimi_target(endpoint, tmp_path), id="kimi-carry")
        again = await agent.run("go", user="u1", thread="t1")
        other = await agent.run("go", user="u2", thread="t1")
    assert "kept" in str(again.answer)
    assert "Restored from this conversation's earlier turns" in str(again.answer)
    assert "kept" not in str(other.answer)
