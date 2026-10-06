"""``DockerSandbox`` against the Engine API (a daemon in process, ``tests/support/docker.py``): a
container named and labelled after its run, hardened and limited, its files written before it
starts; adopted when it exists and its image pulled when missing; attached (resumed when paused,
started when stopped, lost when gone or someone else's); commands with their output, killed with
what they started when out of time or cancelled; files read and written; paused, committed and
made again from the snapshot; deleted with its snapshots; found by label; and every daemon error
said as one (that may pass when the daemon failed)."""

from __future__ import annotations

import asyncio
from typing import Final

import httpx
import pytest

from tests.support.docker import FakeDocker
from trellis.contracts import ConfigurationError, ToolError
from trellis.harness.sandbox import DockerSandbox, SandboxLost, SandboxRef, SandboxSpec
from trellis.harness.sandbox.base import AGENT, RUN, TENANT, ExecResult
from trellis.harness.sandbox.docker import (
    DEFAULT_IMAGE,
    IDLE,
    OUTPUT_BYTES,
    PIDS_LIMIT,
    SNAPSHOT,
    WORKDIR,
    WRAPPER,
    Container,
)

NAME: Final = "trellis-run_1"
REF: Final = SandboxRef(provider="docker", id=NAME, labels={RUN: "run_1", TENANT: "t", AGENT: "a"})


async def made(docker: FakeDocker, spec: SandboxSpec | None = None) -> Container:
    return await DockerSandbox().create(REF, spec or SandboxSpec())


async def test_a_sandbox_is_a_hardened_container_named_after_its_run_with_its_files() -> None:
    docker = FakeDocker()
    spec = SandboxSpec(cpu=0.5, memory=256, files={"a.txt": "A", "/etc/b": b"B"})
    with docker.running():
        box = await made(docker, spec)
    assert box.ref == REF and box.provider.name == "docker"
    container = docker.containers[NAME]
    assert container.state == "running" and container.labels == REF.labels
    assert container.config["Image"] == DEFAULT_IMAGE
    assert (container.config["Entrypoint"], container.config["WorkingDir"]) == (IDLE, WORKDIR)
    assert container.config["HostConfig"] == {
        "NetworkMode": "none",
        "Init": True,
        "CapDrop": ["ALL"],
        "SecurityOpt": ["no-new-privileges"],
        "PidsLimit": PIDS_LIMIT,
        "NanoCpus": 500_000_000,
        "Memory": 256 * 1024 * 1024,
    }
    assert container.files == {"/workspace/a.txt": b"A", "/etc/b": b"B"}
    assert docker.requests == [  # the files are in before it starts
        "POST /containers/create",
        f"GET /containers/{NAME}/json",
        f"PUT /containers/{NAME}/archive",
        f"POST /containers/{NAME}/start",
    ]


async def test_an_existing_sandbox_is_adopted_and_a_missing_image_pulled() -> None:
    docker = FakeDocker()
    with docker.running():
        first = await made(docker, SandboxSpec(files={"a.txt": "A"}))
        await first.write("a.txt", b"changed")
        again = await made(docker, SandboxSpec(files={"a.txt": "A"}))  # an attempt after a crash
        assert await again.read("a.txt") == b"changed"  # adopted, not seeded again
        assert list(docker.containers) == [NAME]
        other = REF.model_copy(update={"id": "trellis-run_2"})
        await DockerSandbox("registry.test/img:1").create(other, SandboxSpec(network="open"))
    assert "registry.test/img:1" in docker.images
    assert docker.containers["trellis-run_2"].config["HostConfig"]["NetworkMode"] == "bridge"


async def test_an_image_that_cannot_be_pulled_and_hosts_docker_cannot_limit_are_refused() -> None:
    docker = FakeDocker()
    docker.pulls["registry.test/missing"] = {"error": "manifest unknown"}
    docker.pulls["registry.test/denied"] = {"status": 500, "message": "denied"}
    with docker.running():
        with pytest.raises(ToolError, match=r"registry\.test/missing cannot be pulled: manifest"):
            await DockerSandbox("registry.test/missing").create(REF, SandboxSpec())
        with pytest.raises(ToolError, match="cannot be pulled: denied"):
            await DockerSandbox("registry.test/denied").create(REF, SandboxSpec())
        with pytest.raises(ConfigurationError, match="cannot limit it to some hosts"):
            await made(docker, SandboxSpec(network=("pypi.org",)))
    assert docker.containers == {}


async def test_attach_resumes_a_paused_sandbox_starts_a_stopped_one_and_finds_a_lost_one() -> None:
    docker = FakeDocker()
    provider = DockerSandbox()
    with docker.running():
        await made(docker)
        await provider.pause(REF)
        await provider.pause(REF)  # paused already: nothing to do
        assert docker.containers[NAME].state == "paused"
        assert (await provider.attach(REF)).ref == REF
        assert docker.containers[NAME].state == "running"
        await provider.attach(REF)  # running: nothing to do
        docker.containers[NAME].state = "exited"  # the daemon restarted
        await provider.attach(REF)
        assert docker.containers[NAME].state == "running"
        docker.containers[NAME].config["Labels"] = {RUN: "run_9"}
        with pytest.raises(SandboxLost, match=f"the container {NAME} is not this run's sandbox"):
            await provider.attach(REF)
        del docker.containers[NAME]
        with pytest.raises(SandboxLost, match=f"the sandbox {NAME} is gone"):
            await provider.attach(REF)
        with pytest.raises(SandboxLost, match="is gone"):
            await provider.pause(REF)
    assert docker.requests.count(f"POST /containers/{NAME}/unpause") == 1


async def test_a_command_says_its_exit_code_and_output_and_its_env_reaches_it() -> None:
    docker = FakeDocker()
    with docker.running():
        box = await made(docker)
        said = await box.exec("echo hi", timeout=5, env={"TOKEN": "t"})
        failed = await box.exec("fail")
        noisy = await box.exec("noisy")
    assert said == ExecResult(exit_code=0, stdout="hi", stderr="")
    assert failed == ExecResult(exit_code=2, stdout="", stderr="boom")
    assert noisy.stdout.startswith("[4 bytes cut]\n") and noisy.stdout.endswith("tail")
    assert len(noisy.stdout.split("\n", 1)[1]) == OUTPUT_BYTES
    _, cmd, config = docker.execs["exec0"]
    assert cmd[:3] == ["sh", "-c", WRAPPER] and cmd[3].startswith("/tmp/.trellis-exec-")
    assert cmd[4] == "echo hi" and config["Env"] == ["TOKEN=t"] and config["WorkingDir"] == WORKDIR


async def test_a_command_out_of_time_or_cancelled_is_killed_with_what_it_started() -> None:
    docker = FakeDocker()
    with docker.running():
        box = await made(docker)
        with pytest.raises(TimeoutError):
            await box.exec("sleep", timeout=0.05)
        running = asyncio.create_task(box.exec("sleep"))
        await asyncio.sleep(0.05)
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
        docker.broken["/exec/exec4/start"] = httpx.Response(500, json={"message": "no exec"})
        with pytest.raises(ToolError, match="Docker answered 500: no exec") as failed:
            await box.exec("echo never")
        assert failed.value.retryable  # the daemon failed: it may pass
    pid_files = [docker.execs[e][1][3] for e in ("exec0", "exec2", "exec4")]
    assert docker.killed == pid_files  # each stopped by the process group its pid file names


async def test_a_command_whose_sandbox_went_away_is_lost_and_not_killed_twice() -> None:
    docker = FakeDocker()
    with docker.running():
        box = await made(docker)
        running = asyncio.create_task(box.exec("sleep", timeout=0.2))
        await asyncio.sleep(0.05)
        del docker.containers[NAME]  # removed while the command ran
        with pytest.raises(TimeoutError):
            await running
        with pytest.raises(SandboxLost, match="is gone"):
            await box.exec("echo hi")
    assert docker.killed == []  # nothing left to kill


async def test_files_are_read_and_written_where_the_commands_run() -> None:
    docker = FakeDocker()
    with docker.running():
        box = await made(docker)
        await box.write("src/app.py", b"print(1)")
        await box.write("/tmp/x", b"x")
        assert await box.read("src/app.py") == await box.read("/workspace/src/app.py")
        with pytest.raises(ToolError, match="there is no file /workspace/nope in the sandbox"):
            await box.read("nope")
        with pytest.raises(ToolError, match="/workspace/src in the sandbox is not a file"):
            await box.read("src")
        del docker.containers[NAME]
        with pytest.raises(SandboxLost, match="is gone"):
            await box.read("src/app.py")
    assert docker.snapshots == {}


async def test_a_snapshot_makes_the_sandbox_again_and_delete_removes_both() -> None:
    docker = FakeDocker()
    provider = DockerSandbox()
    with docker.running():
        box = await made(docker, SandboxSpec(files={"a.txt": "A"}))
        await box.write("b.txt", b"B")
        snapshot = await provider.snapshot(REF)
        assert docker.images[snapshot] == {**REF.labels, SNAPSHOT: NAME}
        del docker.containers[NAME]  # lost
        kept = REF.model_copy(update={"snapshot": snapshot})
        again = await provider.create(kept, SandboxSpec(files={"a.txt": "not again"}))
        assert docker.containers[NAME].config["Image"] == snapshot
        assert await again.read("a.txt") == b"A" and await again.read("b.txt") == b"B"
        await provider.delete(kept)
        assert NAME not in docker.containers and snapshot not in docker.images
        docker.images["team/app:1"] = {RUN: "run_1"}  # an image of the same run, not a snapshot
        await provider.delete(SandboxRef(provider="docker", id="unlabelled"))
        await provider.delete(kept)
        assert sorted(docker.images) == [DEFAULT_IMAGE, "team/app:1"]  # only its own snapshots
        await provider.delete(kept)  # gone already: nothing to do
        with pytest.raises(SandboxLost, match="its snapshot are gone"):
            await provider.create(kept, SandboxSpec())


async def test_a_delete_the_daemon_refuses_is_an_error_and_a_vanished_image_is_not() -> None:
    docker = FakeDocker()
    provider = DockerSandbox()
    with docker.running():
        await made(docker)
        snapshot = await provider.snapshot(REF)
        docker.broken[f"/containers/{NAME}"] = httpx.Response(409, text="removal in progress")
        with pytest.raises(ToolError, match="Docker answered 409: removal in progress") as refused:
            await provider.delete(REF)
        assert not refused.value.retryable
        docker.broken[f"/images/{snapshot}"] = httpx.Response(500, json={"message": "busy"})
        with pytest.raises(ToolError, match="busy"):
            await provider.delete(REF)
        docker.broken[f"/images/{snapshot}"] = httpx.Response(404, json={"message": "gone"})
        await provider.delete(REF)


async def test_sandboxes_are_found_by_label_with_the_harnesss_labels_only() -> None:
    docker = FakeDocker()
    provider = DockerSandbox()
    with docker.running():
        await made(docker)
        other = SandboxRef(
            provider="docker", id="trellis-run_2", labels={RUN: "run_2", TENANT: "u"}
        )
        await provider.create(other, SandboxSpec())
        found = await provider.labelled({TENANT: "t"})
        docker.broken["/containers/json"] = httpx.Response(500, json={"message": "down"})
        with pytest.raises(ToolError, match="down"):
            await provider.labelled({TENANT: "t"})
    assert found == [REF]  # the image's own labels (maintainer) left out
