"""Sandboxes against the real Docker daemon: a run's tools work in a container made for the run
(its files written, its CPU and memory limited, no network unless opened) and deleted at its
end; a command out of time is killed with what it started; a run paused for a person keeps its
container snapshotted and paused, and its resume works in the same one; a worker that dies
leaves the container to the next attempt; the reaper deletes what an ended run left."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any, Final

import httpx
import pytest

from tests.integration.test_progress import Crash, crash_once
from tests.live.conftest import needs_docker
from trellis import Harness, Runtime, Settings
from trellis.contracts import RunStatus
from trellis.harness.runs import LocalRuns
from trellis.harness.sandbox import DockerSandbox, SandboxRef, SandboxSpec, reap, sandbox
from trellis.harness.sandbox.base import RUN, TENANT
from trellis.harness.sandbox.docker import API_VERSION, DOCKER_SOCKET

pytestmark = [pytest.mark.live, needs_docker]

#: The image the sandboxes are made of: the deployment's (``SANDBOX_IMAGE``), else Docker's
#: default.
PROVIDER: Final = DockerSandbox(Settings.from_env().sandbox_image)
UNKNOWN: Final = "it may or may not have taken effect: check before calling it again"
#: Lists the processes of the sandbox (its image has no ps): their command lines.
PROCESSES: Final = (
    "for p in /proc/[0-9]*; do tr '\\0' ' ' < $p/cmdline 2>/dev/null; echo; done | grep -v '^$'"
)


async def engine(method: str, path: str, **kwargs: Any) -> httpx.Response:
    """A call of the daemon's Engine API, to look at what the provider did."""
    transport = httpx.AsyncHTTPTransport(uds=DOCKER_SOCKET)
    async with httpx.AsyncClient(transport=transport, base_url="http://docker") as client:
        return await client.request(method, f"/{API_VERSION}{path}", **kwargs)


async def state(name: str) -> str | None:
    """The container's state, ``None`` once it is gone."""
    found = await engine("GET", f"/containers/{name}/json")
    return None if found.status_code == 404 else found.json()["State"]["Status"]


async def images(run_id: str) -> list[str]:
    filters = json.dumps({"label": [f"{RUN}={run_id}"]})
    return [
        i["Id"] for i in (await engine("GET", "/images/json", params={"filters": filters})).json()
    ]


@pytest.fixture
async def harness() -> AsyncIterator[Harness]:
    async with Harness(config=Settings()) as h:
        yield h


async def test_a_run_works_in_its_own_limited_offline_container_deleted_at_its_end(
    harness: Harness,
) -> None:
    seen: dict[str, Any] = {}

    async def analyse(input: str, agent: Runtime) -> Any:
        script = "import csv; print(sum(int(r['n']) for r in csv.DictReader(open('data.csv'))))"
        await agent.tools.call("sandbox_write", path="sum.py", content=script)
        total = await agent.tools.call("sandbox_exec", command="python sum.py > out.txt")
        seen["config"] = (await engine("GET", f"/containers/trellis-{agent.run_id}/json")).json()
        seen["network"] = await agent.tools.call("sandbox_exec", command="ls /sys/class/net")
        seen["online"] = await agent.tools.call(
            "sandbox_exec",
            command="python -c 'import socket; socket.create_connection((\"1.1.1.1\", 53), 3)'",
        )
        return [total["exit_code"], await agent.tools.call("sandbox_read", path="out.txt")]

    spec = SandboxSpec(files={"data.csv": "n\n1\n2\n39\n"}, cpu=0.5, memory=128)
    agent = harness.wrap(analyse, id="analyst", tools=[sandbox(PROVIDER, spec)])
    result = await agent.run("add them up", user="u")
    assert result.status is RunStatus.SUCCESS, result.error
    assert result.answer == [0, "42\n"]
    config = seen["config"]
    assert config["State"]["Status"] == "running"
    assert config["Config"]["Labels"][RUN] == result.run_id
    limits = {k: config["HostConfig"][k] for k in ("Memory", "NanoCpus", "NetworkMode", "CapDrop")}
    assert limits == {
        "Memory": 128 * 1024 * 1024,
        "NanoCpus": 500_000_000,
        "NetworkMode": "none",
        "CapDrop": ["ALL"],
    }
    assert seen["network"]["stdout"].split() == ["lo"]  # no network but the loopback
    assert seen["online"]["exit_code"] != 0 and "Network is unreachable" in seen["online"]["stderr"]
    assert await state(f"trellis-{result.run_id}") is None  # deleted with the run


async def test_an_open_network_has_an_interface_and_a_command_out_of_time_is_killed(
    harness: Harness,
) -> None:
    async def work(input: str, agent: Runtime) -> Any:
        net = await agent.tools.call("sandbox_exec", command="ls /sys/class/net")
        late = await agent.tools.call("sandbox_exec", command="sleep 300 & sleep 301; echo never")
        left = await agent.tools.call("sandbox_exec", command=PROCESSES)
        return [net["stdout"].split(), late, left["stdout"]]

    source = sandbox(PROVIDER, SandboxSpec(network="open"), timeout=10)
    agent = harness.wrap(work, id="opener", tools=[source])
    result = await agent.run("go", user="u")
    assert result.status is RunStatus.SUCCESS, result.error
    interfaces, late, left = result.answer
    assert "eth0" in interfaces
    assert late == f"sandbox_exec timed out after 10s; {UNKNOWN}"
    assert "sleep 30" not in left  # killed, with the command it started in the background


async def test_a_paused_run_keeps_its_container_snapshotted_and_paused_and_resumes_in_it(
    harness: Harness,
) -> None:
    async def work(input: str, agent: Runtime) -> Any:
        await agent.tools.call("sandbox_exec", command="echo draft > report.md")
        await agent.ask("Publish the draft?")
        return await agent.tools.call("sandbox_exec", command="cat report.md")

    agent = harness.wrap(work, id="publisher", tools=[sandbox(PROVIDER)])
    waiting = await agent.run("write it", user="u")
    assert waiting.interrupt is not None
    name = f"trellis-{waiting.run_id}"
    assert await state(name) == "paused"
    [snapshot] = await images(waiting.run_id)
    record = await harness.runs.get(waiting.run_id, tenant="default")
    assert record is not None and record.checkpoint is not None
    assert record.checkpoint["sandbox"]["snapshot"] == snapshot
    done = await agent.resume(waiting.interrupt.interrupt_id, "approve", reviewer="u")
    assert done.answer == {"exit_code": 0, "stdout": "draft\n", "stderr": ""}
    assert await state(name) is None and await images(waiting.run_id) == []


async def test_a_worker_that_dies_leaves_the_container_to_the_next_attempt(
    harness: Harness,
) -> None:
    crashes = [Crash()]

    async def work(input: str, agent: Runtime) -> Any:
        await agent.tools.call("sandbox_exec", command="echo 1 >> attempts")
        if crashes:
            raise crashes.pop()
        return (await agent.tools.call("sandbox_exec", command="cat attempts"))["stdout"]

    store = harness.runs
    assert isinstance(store, LocalRuns)
    agent = harness.wrap(work, id="survivor", tools=[sandbox(PROVIDER)])
    handle = await agent.start("go", user="u")
    await crash_once(store, agent, handle)
    name = f"trellis-{handle.run_id}"
    made = (await engine("GET", f"/containers/{name}/json")).json()["Id"]
    assert await harness.worker([agent]).run_once()
    done = await handle.result(timeout=60)
    assert done.answer == "1\n"  # the command ran once, in the one container
    assert await state(name) is None and made


async def test_the_reaper_deletes_the_container_an_ended_run_left(harness: Harness) -> None:
    async def work(input: str, agent: Runtime) -> str:
        return "done"

    ended = await harness.wrap(work, id="ended").run("x", user="u")
    ref = SandboxRef(
        provider="docker",
        id=f"trellis-{ended.run_id}",
        labels={RUN: ended.run_id, TENANT: "default"},
    )
    await PROVIDER.create(ref, SandboxSpec())  # its process died before deleting it
    assert await state(ref.id) == "running"
    deleted = await reap(PROVIDER, harness.runs, tenant="default")
    assert ref in deleted and await state(ref.id) is None
