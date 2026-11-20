"""The isolation layer on this machine, with real processes and real uids (``HostSandbox``,
``isolated_run``): each run works in a workspace of its own (mode 0700) as an unprivileged uid
of its own, inside its tenant's base directory and uid range; it cannot read, list or write
another run's workspace, see or signal another run's processes, or read the service's files
or environment; the network is off; each limit stops a runaway command and says which, while
other runs keep working; a run's end kills its processes, deletes its workspace and frees its
uid; the reaper removes what ended runs left; every command is audited; and a service that
cannot switch uid refuses to run commands here at all. Run as root, or with a password-less
sudo (``tests/support/isolation.py``)."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from collections.abc import AsyncIterator, Iterator, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Final

import pytest

from tests.support.isolation import as_root, fresh_root, processes_of, root_helper
from trellis import Harness, Runtime, Settings
from trellis.contracts import ConfigurationError, RunEventType, RunStatus, ToolError
from trellis.harness.sandbox import ExecResult, SandboxLost, SandboxRef, SandboxSpec, reap
from trellis.harness.sandbox.base import RUN, TENANT
from trellis.harness.sandbox.isolation import (
    AUDIT,
    OUTPUT_BYTES,
    REGISTRY,
    HostSandbox,
    Limits,
    Ran,
    Runner,
    isolated_end,
    isolated_run,
    name_of,
)

SECRET: Final = "sk-service-0123456789"
#: A ``mkdir`` that succeeds on a folder that is there, as uutils' (``/bin/mkdir`` on Ubuntu
#: 26.04, GitHub's ubuntu-latest) does when two calls race to make one folder: it checks that the
#: folder is not there, then takes its EEXIST for success when the other call made it first.
NOT_EXCLUSIVE: Final = (
    'mkdir() { command mkdir "$@" 2>/dev/null; for d; do :; done; [ -d "$d" ]; }\n'
)


@pytest.fixture
def helper() -> tuple[str, ...]:
    return root_helper()


@pytest.fixture
def root(helper: tuple[str, ...]) -> Iterator[str]:
    yield from fresh_root(helper)


@pytest.fixture
async def layer(helper: tuple[str, ...], root: str) -> AsyncIterator[HostSandbox]:
    yield HostSandbox(root, helper=helper)


class Runs:
    """The run store, as the reaper reads it: whether each run it knows has ended."""

    def __init__(self, ended: dict[str, bool]) -> None:
        self.ended = ended

    async def get(self, run_id: str, *, tenant: str) -> Any:
        known = self.ended.get(run_id)
        return None if known is None else SimpleNamespace(final=known)


class RacingMkdir:
    """Root commands whose ``mkdir`` is not exclusive (:data:`NOT_EXCLUSIVE`): every call
    loses the race."""

    def __init__(self, runner: Runner) -> None:
        self.runner = runner

    async def run(
        self, argv: Sequence[str], *, stdin: bytes | None = None, keep: int | None = OUTPUT_BYTES
    ) -> Ran:
        shell, flag, script, *rest = argv
        return await self.runner.run(
            [shell, flag, NOT_EXCLUSIVE + script, *rest], stdin=stdin, keep=keep
        )


async def run(layer: HostSandbox, run_id: str, command: str, **kw: Any) -> ExecResult:
    return await isolated_run(kw.pop("tenant", "acme"), "ada", run_id, command, layer=layer, **kw)


async def test_each_run_works_as_a_uid_of_its_own_in_a_0700_workspace(layer: HostSandbox) -> None:
    one = await run(layer, "r1", "id -u; id -g; id -G; stat -c '%a %u' .; pwd")
    two = await run(layer, "r2", "id -u")
    uid, gid, groups, mode, home = one.stdout.splitlines()
    assert one.exit_code == 0 and int(uid) >= layer.isolation.uid_base
    assert gid == uid and groups == uid  # no group of the service's, none shared
    assert mode == f"700 {uid}"
    assert int(two.stdout) != int(uid) and int(uid) != os.getuid()
    assert home.startswith(f"{layer.isolation.root}/") and home.endswith(f"/{name_of('r1')}")
    again = await run(layer, "r1", "id -u")  # the same run: the same workspace and uid
    assert again.stdout.strip() == uid


async def test_one_run_cannot_read_list_or_write_another_runs_workspace(
    layer: HostSandbox,
) -> None:
    mine = await run(layer, "r1", "echo private > secret.txt; pwd")
    home = mine.stdout.strip()
    other = await run(
        layer,
        "r2",
        f"cat {home}/secret.txt; echo rc=$?; ls {home}; echo rc=$?; "
        f"echo x > {home}/planted; echo rc=$?; ls {os.path.dirname(home)}; echo rc=$?",
    )
    assert "private" not in other.stdout
    assert re.findall(r"rc=(\d+)", other.stdout) == ["1", "2", "2", "2"], other
    assert "Permission denied" in other.stderr
    with pytest.raises(ToolError, match="no file"):  # by its path, through the tool too
        await (await layer.workspace("acme", "ada", "r2")).read(f"{home}/secret.txt")


async def test_one_run_cannot_see_or_signal_another_runs_processes(layer: HostSandbox) -> None:
    sleeper = asyncio.create_task(run(layer, "r1", "exec sleep 30"))
    ws = await layer.workspace("acme", "ada", "r1")
    for _ in range(100):  # until its command runs
        if processes_of(ws.uid):
            break
        await asyncio.sleep(0.05)
    [line] = processes_of(ws.uid)
    pid = line.split()[0]
    seen = await run(layer, "r2", f"ps -e -o pid=,user=,args=; kill -9 {pid}; echo rc=$?")
    assert "sleep 30" not in seen.stdout  # only its own processes
    assert seen.stdout.strip().endswith("rc=1") and processes_of(ws.uid)  # still running
    assert len([p for p in seen.stdout.splitlines() if "rc=" not in p]) <= 3, seen.stdout
    await isolated_end("r1", layer=layer)  # its end kills it
    assert processes_of(ws.uid) == []
    assert (await sleeper).exit_code == 137


async def test_a_run_cannot_read_the_services_files_or_environment(
    layer: HostSandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TRELLIS_TEST_SECRET", SECRET)
    here = os.getcwd()
    registry = f"{layer.isolation.root}/{REGISTRY}"
    seen = await run(
        layer,
        "r1",
        f"env; ls -A {here}; echo rc=$?; ls {registry}; echo rc=$?; "
        "cat /proc/1/environ; echo rc=$?; ls /root; echo rc=$?",
        env={"GIVEN": "by-the-caller"},
    )
    assert SECRET not in seen.stdout + seen.stderr
    names = {
        line.split("=", 1)[0]
        for line in seen.stdout.splitlines()
        if "=" in line and not line.startswith("rc=")
    }
    assert {"PATH", "HOME", "TMPDIR", "LANG", "GIVEN"} <= names
    assert names <= {"PATH", "HOME", "TMPDIR", "LANG", "USER", "PWD", "SHLVL", "GIVEN", "_"}
    assert re.findall(r"rc=(\d+)", seen.stdout) == ["2", "2", "1", "2"], seen
    assert "pyproject.toml" not in seen.stdout  # the service's working folder is hidden


async def test_the_network_is_off_by_default_and_a_list_of_hosts_is_refused(
    layer: HostSandbox, helper: tuple[str, ...], root: str
) -> None:
    probe = (
        'python3 -c "import socket; s = socket.socket(); s.settimeout(3); '
        "s.connect(('1.1.1.1', 53))\""
    )
    off = await run(layer, "r1", probe)
    assert off.exit_code != 0 and "unreachable" in off.stderr, off
    listed = HostSandbox(root, helper=helper)
    ref = SandboxRef(provider="host", id=name_of("r2"), labels={TENANT: "acme", RUN: "r2"})
    with pytest.raises(ConfigurationError, match="cannot limit it to some hosts"):
        await listed.create(ref, SandboxSpec(network=("api.example.com",)))
    with pytest.raises(ConfigurationError, match="the administrator's network is none"):
        await listed.create(ref, SandboxSpec(network="open"))
    opened = HostSandbox(root, helper=helper, network="open")
    made = await opened.create(ref, SandboxSpec(network="open", files={"a.txt": "given"}))
    assert made.ref.id == name_of("r2") and await made.read("a.txt") == b"given"
    assert (await opened.attach(made.ref)).uid == made.uid  # a later attempt finds it


async def test_tenants_have_base_directories_and_uid_ranges_of_their_own(
    layer: HostSandbox,
) -> None:
    a = await layer.workspace("acme", "ada", "r1")
    b = await layer.workspace("globex", "bob", "r2")
    a2 = await layer.workspace("acme", "cy", "r3")
    per = layer.isolation.tenant_uids
    block = lambda uid: (uid - layer.isolation.uid_base) // per  # noqa: E731
    assert block(a.uid) == block(a2.uid) != block(b.uid)
    assert os.path.dirname(a.home) == os.path.dirname(a2.home) != os.path.dirname(b.home)
    seen = await run(layer, "r1", f"ls {os.path.dirname(b.home)}; echo rc=$?")
    assert seen.stdout.strip().endswith("rc=2")  # another tenant's base: not listed
    with pytest.raises(ConfigurationError, match="a tenant"):
        await layer.create(SandboxRef(provider="host", id="x"), SandboxSpec())


async def test_runs_and_tenants_never_share_a_uid_where_mkdir_is_not_exclusive(
    layer: HostSandbox,
) -> None:
    """A run's uid, its workspace and its tenant's uid range are claimed by a file made only if
    it is not there (the shell's own ``set -C``), never by ``mkdir`` alone: two runs started
    at once on a runner whose ``mkdir`` succeeds on a folder another call just made got one uid,
    and each one's start killed the other's ("the processes of uid ... do not end")."""
    layer.isolation.runner = RacingMkdir(layer.isolation.runner)
    one = await layer.workspace("acme", "ada", "r1")
    two, three = await asyncio.gather(
        layer.workspace("acme", "ada", "r2"), layer.workspace("acme", "ada", "r3")
    )
    other = await layer.workspace("globex", "bob", "r4")
    assert len({one.uid, two.uid, three.uid, other.uid}) == 4
    per = layer.isolation.tenant_uids
    assert one.uid // per == two.uid // per == three.uid // per != other.uid // per
    assert (await layer.workspace("acme", "ada", "r1")).uid == one.uid  # found again
    assert (await run(layer, "r2", "echo still here")).stdout == "still here\n"


@pytest.mark.parametrize(
    ("limits", "command", "limit"),
    [
        (Limits(cpu_seconds=1), "while :; do :; done", "cpu"),
        (
            Limits(memory_mib=64),
            'python3 -c "b = bytearray(512 * 1024 * 1024)"',
            "memory",
        ),
        (Limits(processes=8), "for i in $(seq 30); do sleep 2 & done; wait", "processes"),
        (Limits(file_mib=1), "head -c 3000000 /dev/zero > big", "file"),
        (Limits(open_files=16), "python3 -c \"[open('/dev/null') for _ in range(64)]\"", "files"),
    ],
)
async def test_each_limit_stops_a_runaway_command_and_says_which_while_others_work(
    layer: HostSandbox, limits: Limits, command: str, limit: str
) -> None:
    runaway = asyncio.create_task(run(layer, "r1", command, limits=limits))
    other = await run(layer, "r2", "echo still here")
    stopped = await runaway
    assert other.stdout.strip() == "still here"
    assert stopped.exit_code != 0 and stopped.limit == limit, stopped
    assert stopped.duration_ms is not None


async def test_a_command_past_its_time_is_killed_with_everything_it_started(
    layer: HostSandbox,
) -> None:
    with pytest.raises(TimeoutError):
        await run(layer, "r1", "sleep 30 & sleep 31 & sleep 32", timeout=0.5)
    ws = await layer.workspace("acme", "ada", "r1")
    for _ in range(40):
        if not processes_of(ws.uid):
            break
        await asyncio.sleep(0.05)
    assert processes_of(ws.uid) == []
    assert (await run(layer, "r1", "echo next")).stdout.strip() == "next"


async def test_a_background_process_does_not_outlive_its_command(layer: HostSandbox) -> None:
    started = await run(layer, "r1", "(sleep 30 &) ; echo started")
    ws = await layer.workspace("acme", "ada", "r1")
    assert started.stdout.strip() == "started" and processes_of(ws.uid) == []


async def test_a_runs_end_kills_its_processes_deletes_its_workspace_and_frees_its_uid(
    layer: HostSandbox, helper: tuple[str, ...]
) -> None:
    ws = await layer.workspace("acme", "ada", "r1")
    await ws.write("outputs/report.csv", b"a,b\n1,2\n")
    assert await ws.read("outputs/report.csv") == b"a,b\n1,2\n"
    registry = f"{layer.isolation.root}/{REGISTRY}"
    assert as_root(helper, "test", "-d", ws.home).returncode == 0
    await isolated_end("r1", layer=layer)
    assert as_root(helper, "test", "-e", ws.home).returncode == 1
    assert as_root(helper, "test", "-e", f"{registry}/uids/{ws.uid}").returncode == 1
    with pytest.raises(SandboxLost):
        await layer.attach(ws.ref)
    await isolated_end("r1", layer=layer)  # ending it again: nothing left to do
    fresh = await layer.workspace("acme", "ada", "r1")  # a later run: a fresh workspace
    assert (await fresh.exec("ls -A")).stdout.split() == [".results", ".tmp"]


async def test_reads_and_writes_are_the_runs_own_and_say_what_failed(layer: HostSandbox) -> None:
    ws = await layer.workspace("acme", "ada", "r1")
    await ws.write("a/b/c.txt", b"deep")
    assert await ws.read("a/b/c.txt") == b"deep"
    await ws.write("/large_tool_results/k1", b"kept")  # the run's results, at their path
    assert (await ws.exec("cat /large_tool_results/k1")).stdout == "kept"
    with pytest.raises(ToolError, match="no file /nowhere in the sandbox"):
        await ws.read("/nowhere")
    with pytest.raises(ToolError, match="is not a file"):
        await ws.read("a")
    with pytest.raises(ToolError, match="was not written"):
        await ws.write("/etc/planted", b"x")
    await ws.exec("chmod 000 a/b/c.txt")
    with pytest.raises(ToolError, match="Permission denied"):
        await ws.read("a/b/c.txt")
    with pytest.raises(ToolError, match="is not a variable name"):
        await ws.exec("true", env={"BAD NAME": "x"})


async def test_a_file_past_the_largest_read_is_refused(
    layer: HostSandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    from trellis.harness.sandbox import isolation

    monkeypatch.setattr(isolation, "READ_MAX_BYTES", 4)
    ws = await layer.workspace("acme", "ada", "r1")
    await ws.write("big.txt", b"123456789")
    with pytest.raises(ToolError, match="more than the 4 bytes read at once"):
        await ws.read("big.txt")


async def test_the_reaper_removes_what_ended_runs_left_and_nothing_else(
    layer: HostSandbox,
) -> None:
    runs = Runs({"r1": True, "r2": False})
    for run_id in ("r1", "r2", "r3"):  # r3: a run the store does not know is left alone
        await layer.workspace("acme", "ada", run_id)
    await layer.workspace("globex", "bob", "r4")  # another tenant's: not this round's
    reaped = await reap(layer, runs, tenant="acme")  # type: ignore[arg-type]
    assert [r.id for r in reaped] == [name_of("r1")]
    left = sorted(r.id for r in await layer.labelled({}))
    assert left == sorted(name_of(r) for r in ("r2", "r3", "r4"))
    assert [r.id for r in await layer.labelled({TENANT: "globex"})] == [name_of("r4")]


async def test_host_execution_refuses_to_start_when_it_cannot_switch_uid(
    monkeypatch: pytest.MonkeyPatch, root: str
) -> None:
    from trellis.harness.sandbox import isolation

    monkeypatch.setattr(isolation.os, "geteuid", lambda: 1000)
    plain = HostSandbox(root)
    why = plain.cannot_isolate()
    assert why is not None and "runs as uid 1000, not root" in why
    with pytest.raises(ConfigurationError, match="commands cannot be isolated on this machine"):
        await isolated_run("acme", "ada", "r1", "id", layer=plain)
    broken = HostSandbox(root, helper=("false",))
    assert "the helper false cannot run a command as root" in (broken.cannot_isolate() or "")
    monkeypatch.setattr(isolation.shutil, "which", lambda name: None)
    assert "unshare, setpriv, prlimit" in (HostSandbox(root).cannot_isolate() or "")
    monkeypatch.setattr(isolation.sys, "platform", "darwin")
    assert "Linux" in (HostSandbox(root).cannot_isolate() or "")


def test_a_layer_is_refused_a_root_or_path_it_cannot_keep(root: str) -> None:
    with pytest.raises(ConfigurationError, match="an absolute path"):
        HostSandbox("relative/root")
    with pytest.raises(ConfigurationError, match="hides the workspaces"):
        HostSandbox(root, hidden=("/var",))
    with pytest.raises(ConfigurationError, match="an absolute path"):
        HostSandbox(root, hidden=("etc",))
    with pytest.raises(ValueError, match="greater than 0"):
        Limits(cpu_seconds=0)


async def test_every_command_is_audited_in_the_log_and_on_its_runs_stream(
    layer: HostSandbox, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger=AUDIT):
        await run(layer, "r1", "exit 3")
    [record] = [json.loads(r.getMessage()) for r in caplog.records if r.name == AUDIT]
    assert record["tenant"] == "acme" and record["user"] == "ada" and record["run"] == "r1"
    assert record["command"] == "exit 3" and record["exit_code"] == 3
    assert record["limit"] is None and isinstance(record["duration_ms"], int)
    assert record["uid"] >= layer.isolation.uid_base

    async def work(input: str, agent: Runtime) -> Any:
        said = await isolated_run(agent.tenant, agent.user, agent.run_id, "echo hi", layer=layer)
        await isolated_run(agent.tenant, agent.user, agent.run_id, "true", layer=layer)
        return said.stdout

    async with Harness(config=Settings()) as harness:
        agent = harness.wrap(work, id="host-shell")
        events = [e async for e in agent.stream("go", user="ada")]
        again = await agent.run("go", user="ada")  # reaped once: not again within 10 minutes
        await harness.writes.drain()
    finished = events[-1]
    assert finished.data["result"] == "hi\n"
    audited, _ = [
        e.data
        for e in events
        if e.type is RunEventType.CUSTOM and e.data.get("name") == "sandbox_command"
    ]
    assert audited["user"] == "ada" and audited["run"] == finished.run_id
    assert audited["command"] == "echo hi" and audited["exit_code"] == 0
    # the run's end removed its workspace (its journal named it)
    assert await layer.labelled({RUN: finished.run_id}) == []
    assert await layer.labelled({RUN: again.run_id}) == []


async def test_the_layers_own_failures_are_errors_that_say_so(
    layer: HostSandbox, helper: tuple[str, ...], root: str
) -> None:
    tiny = HostSandbox(root, helper=helper, tenant_uids=1)
    await tiny.workspace("acme", "ada", "r1")
    with pytest.raises(ToolError, match="no uid is free for this tenant"):
        await tiny.workspace("acme", "ada", "r2")
    one = HostSandbox(f"{root}-one", helper=helper, tenants=1)
    try:
        await one.workspace("acme", "ada", "r1")
        with pytest.raises(ToolError, match="no uid range is left for a tenant"):
            await one.workspace("globex", "bob", "r2")
    finally:
        as_root(helper, "rm", "-rf", f"{root}-one")
    blocked = HostSandbox(f"{root}/file", helper=helper)
    as_root(helper, "mkdir", "-p", root)
    as_root(helper, "touch", f"{root}/file")
    with pytest.raises(ToolError, match="the workspace was not made"):
        await blocked.workspace("acme", "ada", "r1")
    assert as_root(helper, "test", "-f", f"{root}/file").returncode == 0


async def test_kimi_in_a_host_workspace_works_in_its_folder_as_the_runs_uid(
    helper: tuple[str, ...], root: str, tmp_path: Path
) -> None:
    """Kimi names its files by the folder it works in on this machine: in a workspace of the
    layer's that folder is the run's workspace, and where its commands start."""
    from tests.integration.test_acp import fake, recorded
    from trellis.harness.sandbox import sandbox

    layer = HostSandbox(root, helper=helper)
    script = [
        {"builtin": "Write", "args": {"path": "notes.txt", "content": "from kimi"}},
        {"builtin": "Bash", "args": {"command": "pwd; id -u; cat notes.txt"}},
        {"text": "{last}"},
    ]
    target = fake(tmp_path, script, sandbox=sandbox(layer), env={"FAKE_ACP_RAW": "1"})
    async with Harness(config=Settings()) as harness:
        done = await harness.wrap(target, id="kimi-host").run("work", user="ada")
    assert done.status is RunStatus.SUCCESS, done.error
    work = recorded(tmp_path)["started"][0]["cwd"]
    [ran] = [t for t in recorded(tmp_path)["terminal"] if "exit" in t]
    folder, uid, said = ran["output"].split("\n")[:3]
    assert folder == work and int(uid) >= layer.isolation.uid_base and said == "from kimi"
    assert await layer.labelled({}) == []  # its end removed the workspace


def test_the_host_layer_is_one_given_or_one_with_every_default(
    helper: tuple[str, ...], root: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from trellis.harness.sandbox import isolation

    given = HostSandbox(root, helper=helper)
    before = isolation.use_host(given)
    try:
        assert isolation.host_layer() is given
        unlabelled = SandboxRef(provider="host", id="trellis-x", labels={TENANT: "acme"})
        assert isolation.layer_of(unlabelled) is given
        assert isolation.use_host(None) is given
        default = isolation.host_layer()
        assert default.isolation.root == isolation.HOST_ROOT and default is not given
    finally:
        isolation.use_host(before)
    # a helper that runs a command as root: isolated through it (env: this process is root)
    through = HostSandbox(root, helper=helper or ("env",))
    assert through.cannot_isolate() is None and through.cannot_isolate() is None
    assert "cannot run a command as root" in (
        HostSandbox(root, helper=("/nonexistent/helper",)).cannot_isolate() or ""
    )
    monkeypatch.setattr(isolation.os, "geteuid", lambda: 0)
    assert HostSandbox(root).cannot_isolate() is None


async def test_names_and_paths_a_shell_would_split_are_refused(layer: HostSandbox) -> None:
    with pytest.raises(ConfigurationError, match="not /"):
        HostSandbox("/")
    with pytest.raises(ConfigurationError, match="characters a shell would split"):
        HostSandbox("/var/lib/my workspaces")
    with pytest.raises(ToolError, match="is not a workspace name"):
        await layer.isolation.release("bad name")
    ws = await layer.workspace("acme", "ada", "r1")
    with pytest.raises(ConfigurationError, match="characters a shell would split"):
        ws.alias("/a folder")
