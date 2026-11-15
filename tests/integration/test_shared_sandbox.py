"""The tenant's shared sandbox (``SharedDockerSandbox``, ``sandbox()``'s default): one container
per tenant, never another tenant's; inside it each run has its own workspace and uid through
the isolation layer; every way a run ends deletes its workspace, kills its processes and frees
its uid — its outputs delivered first — a crash's takeover kills what the dead attempt left
running, and the next turn finds the files its conversation carries in a fresh workspace.

The Engine API is in process and its containers are this machine (``tests/support/
local_engine.py``): every script runs for real, as root, with real uids.
``tests/live/test_live_shared_sandbox.py`` runs the boundaries against a real daemon."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from typing import Any, Final

import pytest

import trellis
from tests.integration.test_progress import Crash, crash_once
from tests.support.arm import arm
from tests.support.isolation import as_root, fresh_root, processes_of, root_helper
from tests.support.journals import journal_of
from tests.support.local_engine import LocalEngine
from trellis import Harness, Runtime, Settings
from trellis.contracts import ConfigurationError, RunEventType, RunStatus
from trellis.harness.runs import LocalRuns
from trellis.harness.sandbox import SandboxLost, SandboxRef, SandboxSpec, sandbox
from trellis.harness.sandbox.base import RUN, TENANT
from trellis.harness.sandbox.shared import POOL, POOL_CAPS, POOL_OF, SharedDockerSandbox
from trellis.harness.tools import results

UNKNOWN: Final = "it may or may not have taken effect: check before calling it again"


@pytest.fixture
def helper() -> tuple[str, ...]:
    return root_helper()


@pytest.fixture
def root(helper: tuple[str, ...]) -> Iterator[str]:
    yield from fresh_root(helper)


@pytest.fixture
def engine(helper: tuple[str, ...]) -> Iterator[LocalEngine]:
    with LocalEngine(helper).running() as running:
        yield running


@pytest.fixture
async def provider(engine: LocalEngine, root: str) -> AsyncIterator[SharedDockerSandbox]:
    yield SharedDockerSandbox(root=root)


def ref(run_id: str, tenant: str = "acme") -> SandboxRef:
    return SandboxRef(
        provider="docker-shared", id=f"trellis-{run_id}", labels={RUN: run_id, TENANT: tenant}
    )


async def test_two_tenants_never_share_a_container_and_a_tenants_runs_share_its_own(
    provider: SharedDockerSandbox, engine: LocalEngine
) -> None:
    spec = SandboxSpec()
    assert provider.pool_of("acme", spec).id != provider.pool_of("globex", spec).id
    assert provider.pool_of("acme", spec) == provider.pool_of("acme", SandboxSpec())
    other = SandboxSpec(image="registry.test/data:1")
    assert provider.pool_of("acme", other).id != provider.pool_of("acme", spec).id
    one = await provider.create(ref("r1"), SandboxSpec(files={"seed.txt": "seeded"}))
    two = await provider.create(ref("r2"), spec)
    three = await provider.create(ref("r3", "globex"), spec)
    assert one.ref.labels[POOL_OF] == two.ref.labels[POOL_OF] != three.ref.labels[POOL_OF]
    pools = {name: made.config for name, made in engine.containers.items()}
    assert sorted(c["Labels"][POOL] for c in pools.values()) == ["acme", "globex"]
    for config in pools.values():
        host = config["HostConfig"]
        assert host["CapAdd"] == list(POOL_CAPS) and host["CapDrop"] == ["ALL"]
        assert host["NetworkMode"] == "none" and "no-new-privileges" in host["SecurityOpt"]
    assert one.uid not in (two.uid, three.uid)
    assert await one.read("seed.txt") == b"seeded"
    seen = await two.exec(f"cat {one.home}/seed.txt; echo rc=$?")
    assert "seeded" not in seen.stdout and seen.stdout.strip().endswith("rc=1")
    acme = sorted(r.id for r in await provider.labelled({TENANT: "acme"}))
    assert acme == ["trellis-r1", "trellis-r2"]
    assert len(await provider.labelled({})) == 3
    await provider.delete(one.ref)
    with pytest.raises(SandboxLost):
        await provider.attach(one.ref)
    again = await provider.attach(two.ref)  # a later attempt finds it
    assert again.uid == two.uid
    del engine.containers[three.ref.labels[POOL_OF]]  # the tenant's container went
    await provider.delete(three.ref)  # and its workspace with it: nothing to do
    with pytest.raises(SandboxLost):
        await provider.attach(three.ref)
    with pytest.raises(ConfigurationError, match="cannot limit it to some hosts"):
        await provider.create(ref("r4"), SandboxSpec(network=("api.example.com",)))


async def test_a_sandbox_of_no_tenant_is_a_container_of_its_own(
    provider: SharedDockerSandbox, engine: LocalEngine
) -> None:
    probe = SandboxRef(provider="docker-shared", id="trellis-probe-1", labels={})
    made = await provider.create(probe, SandboxSpec(files={"a.txt": "x"}))
    assert made.ref.id == "trellis-probe-1"
    assert "CapAdd" not in engine.containers["trellis-probe-1"].config["HostConfig"]
    assert (await provider.attach(probe)).ref == probe
    await provider.delete(probe)
    assert engine.containers == {} and provider.image == "python:3.12-slim"


async def test_every_ending_deletes_the_runs_workspace_its_outputs_delivered_first(
    provider: SharedDockerSandbox,
) -> None:
    harness = Harness(config=Settings(files_url=f"memory://shared-{id(object())}"))
    started = asyncio.Event()
    seen: dict[str, Any] = {}

    async def work(input: str, agent: Runtime) -> Any:
        said = await agent.tools.call(
            "sandbox_exec", command="mkdir -p outputs && printf 'a,b\\n' > outputs/report.csv"
        )
        seen["runtime"] = trellis.current()
        assert said["exit_code"] == 0 and isinstance(said["duration_ms"], int), said
        if input == "fail":
            raise ValueError("broken")
        if input == "wait":
            started.set()
            await asyncio.sleep(30)
        return "done"

    agent = arm(harness.wrap(work, id="ender"), [sandbox(provider).tools()])
    events = [e async for e in agent.stream("ok", user="ada")]
    assert events[-1].outcome is not None and events[-1].outcome.value == "success"
    files = [e.data for e in events if e.data.get("name") == "file"]
    assert [f["file"] for f in files] == ["report.csv"]  # delivered, before the workspace went
    runtime = seen["runtime"]
    assert await harness.files.size(runtime.files_folder(), "report.csv") == 4
    audited = [e.data for e in events if e.data.get("name") == "sandbox_command"]
    assert audited and all(a["user"] == "ada" and a["tenant"] for a in audited)
    failed = await agent.run("fail", user="ada")
    late = await agent.run("wait", user="ada", run_timeout=1.5)
    assert (failed.status, late.status) == (RunStatus.ERROR, RunStatus.TIMEOUT)
    started.clear()
    running = asyncio.create_task(agent.run("wait", user="ada"))
    await started.wait()
    [run_id] = list(agent.running)
    await agent.cancel(run_id, reason="not needed")
    assert (await running).status is RunStatus.CANCELLED
    assert await provider.labelled({}) == []  # every run's workspace is gone
    await harness.aclose()


async def test_a_crash_takeover_kills_what_the_dead_attempt_left_and_cleans_at_the_end(
    provider: SharedDockerSandbox, helper: tuple[str, ...]
) -> None:
    crashes = [Crash()]

    async def work(input: str, agent: Runtime) -> list[str]:
        said = [(await agent.tools.call("sandbox_exec", command="echo one"))["stdout"]]
        if crashes:
            raise crashes.pop()
        said.append((await agent.tools.call("sandbox_exec", command="echo two"))["stdout"])
        return said

    async with Harness(config=Settings()) as harness:
        store = harness.runs
        assert isinstance(store, LocalRuns)
        agent = arm(harness.wrap(work, id="taken-over"), [sandbox(provider).tools()])
        handle = await agent.start("go", user="ada")
        await crash_once(store, agent, handle)
        recorded = (await journal_of(store, await handle.status())).sandbox
        assert recorded is not None
        box = SandboxRef.model_validate(recorded)
        [found] = await provider.labelled({RUN: handle.id})
        layer = provider.isolation(
            SandboxRef(provider="x", id=box.labels[POOL_OF], labels={POOL: "default"})
        )
        uid, _, _ = await layer.open("default", found.id, found.labels)
        # what the dead worker's command left running in the workspace, as the run's uid
        stray = await asyncio.create_subprocess_exec(
            *helper, "setpriv", f"--reuid={uid}", f"--regid={uid}", "--clear-groups", "sleep", "60"
        )
        for _ in range(100):
            if processes_of(uid):
                break
            await asyncio.sleep(0.05)
        assert processes_of(uid)
        assert await harness.worker([agent]).run_once()
        done = await handle.result(wait=10)
        assert done.answer == ["one\n", "two\n"]
        assert await stray.wait() != 0 and processes_of(uid) == []  # killed at the takeover
        assert await provider.labelled({RUN: handle.id}) == []  # cleaned at the end
        assert as_root(helper, "test", "-e", f"{provider.root}/.trellis/uids/{uid}").returncode == 1


async def test_a_command_out_of_time_is_killed_and_its_effect_is_unknown(
    provider: SharedDockerSandbox,
) -> None:
    async def work(input: str, agent: Runtime) -> Any:
        return await agent.tools.call("sandbox_exec", command="sleep 30")

    async with Harness(config=Settings()) as harness:
        source = sandbox(provider, call_timeout=1.0)
        agent = arm(harness.wrap(work, id="slow"), [source.tools()])
        result = await agent.run("go", user="ada")
    assert result.answer == f"sandbox_exec timed out after 1s; {UNKNOWN}"
    assert await provider.labelled({}) == []


async def test_a_limit_a_command_hits_is_said_with_its_result(
    root: str, engine: LocalEngine
) -> None:
    from trellis.harness.sandbox import Limits

    tight = SharedDockerSandbox(root=root, limits=Limits(cpu_seconds=1))

    async def work(input: str, agent: Runtime) -> Any:
        return await agent.tools.call("sandbox_exec", command="while :; do :; done")

    async with Harness(config=Settings()) as harness:
        result = await arm(harness.wrap(work, id="spin"), [sandbox(tight).tools()]).run(
            "x", user="u"
        )
    assert result.answer["limit"] == "cpu" and result.answer["exit_code"] == 152
    assert result.answer["hint"].startswith("Stopped: the command used all the CPU time")


async def test_the_next_turn_finds_the_files_its_conversation_carries_in_a_fresh_workspace(
    provider: SharedDockerSandbox,
) -> None:
    table = "sku,stock\n" + "\n".join(f"SKU-{n:05d},{n % 50}" for n in range(1, 2_001))
    seen: list[str] = []

    async def work(input: str, agent: Runtime) -> Any:
        if input == "first":
            path = results.keep(table, tool="inventory")
            assert path is not None
            seen.append(path)
        said = await agent.tools.call("sandbox_exec", command=f"wc -l < {seen[0]}; pwd")
        return said["stdout"].split()

    async with Harness(config=Settings()) as harness:
        agent = arm(harness.wrap(work, id="carrier"), [sandbox(provider).tools()])
        first = await agent.run("first", user="ada", thread="t1")
        second = await agent.run("second", user="ada", thread="t1")
    assert first.status is RunStatus.SUCCESS and second.status is RunStatus.SUCCESS, second.error
    assert first.answer[0] == second.answer[0] == "2000"  # every row, in each turn
    assert first.answer[1] != second.answer[1]  # a workspace of each run's own
    assert await provider.labelled({}) == []


def test_without_a_provider_a_sandbox_is_the_tenants_shared_container() -> None:
    chosen = sandbox(image="registry.test/img:1").provider_for()
    assert isinstance(chosen, SharedDockerSandbox) and chosen.image == "registry.test/img:1"
    assert RunEventType.CUSTOM  # the stream's audit events are CUSTOM ones
