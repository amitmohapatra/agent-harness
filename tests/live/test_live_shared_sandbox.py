"""The tenant's shared sandbox against the real Docker daemon: two tenants get containers of
their own; inside one, a run cannot read, list or write another run's workspace, see its
processes or signal them, nor find a secret of the service's or the network; a runaway command
is stopped at its limit while another run works; a run's end leaves no workspace, process or
uid behind; the tenant's containers are removed afterwards."""

from __future__ import annotations

import asyncio
import os
import re
import uuid
from collections.abc import AsyncIterator
from typing import Final

import pytest

from tests.live.conftest import needs_docker
from tests.live.test_live_sandbox import engine
from trellis.harness.sandbox import Limits, SandboxRef, SandboxSpec
from trellis.harness.sandbox.base import RUN, TENANT
from trellis.harness.sandbox.shared import POOL, POOL_OF, SharedDockerSandbox

pytestmark = [pytest.mark.live, needs_docker]

SECRET: Final = "sk-live-service-0123456789"
#: The processes a command sees (the image has no ps): their uid and command line.
PROCESSES: Final = (
    "for p in /proc/[0-9]*; do echo \"$(stat -c %u $p) $(tr '\\0' ' ' < $p/cmdline)\"; done"
)


@pytest.fixture
async def provider() -> AsyncIterator[SharedDockerSandbox]:
    shared = SharedDockerSandbox(limits=Limits(cpu_seconds=2))
    yield shared
    found = await engine("GET", "/containers/json", params={"all": "true"})
    for container in found.json():
        if POOL in (container.get("Labels") or {}) and container["Labels"][POOL].startswith(
            "live-"
        ):
            await engine("DELETE", f"/containers/{container['Id']}", params={"force": "true"})


def ref(run_id: str, tenant: str) -> SandboxRef:
    return SandboxRef(
        provider="docker-shared", id=f"trellis-{run_id}", labels={RUN: run_id, TENANT: tenant}
    )


async def test_the_boundaries_hold_in_a_real_shared_container(
    provider: SharedDockerSandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TRELLIS_LIVE_SECRET", SECRET)
    acme, globex = f"live-acme-{uuid.uuid4().hex[:6]}", f"live-globex-{uuid.uuid4().hex[:6]}"
    one = await provider.create(ref("l1", acme), SandboxSpec(files={"secret.txt": "private"}))
    two = await provider.create(ref("l2", acme), SandboxSpec())
    other = await provider.create(ref("l3", globex), SandboxSpec())
    assert one.ref.labels[POOL_OF] == two.ref.labels[POOL_OF] != other.ref.labels[POOL_OF]
    mode = await one.exec("stat -c '%a %u' .; id -u")
    assert mode.stdout.split() == ["700", str(one.uid), str(one.uid)]
    assert one.uid != two.uid and int(one.uid) != os.getuid()

    sleeper = asyncio.create_task(one.exec("exec sleep 30"))
    await asyncio.sleep(1.0)
    seen = await two.exec(
        f"cat {one.home}/secret.txt; echo rc=$?; ls {one.home}; echo rc=$?; "
        f"echo x > {one.home}/planted; echo rc=$?; {PROCESSES}; kill -9 -1; echo rc=$?; "
        "env; cat /proc/1/environ; echo rc=$?; ls /workspaces/.trellis; echo rc=$?; "
        "echo x > /etc/planted; echo rc=$?; "
        "python3 -c \"import socket; socket.create_connection(('1.1.1.1', 53), 3)\"; echo rc=$?"
    )
    said = seen.stdout
    assert "private" not in said and "sleep 30" not in said and SECRET not in said
    assert re.findall(r"rc=(\d+)", said) == ["1", "2", "2", "1", "1", "2", "2", "1"], seen
    assert not sleeper.done()  # its kill reached its own processes only
    processes = [line for line in said.splitlines() if re.match(r"^\d+ ", line)]
    assert {line.split()[0] for line in processes} <= {"0", str(two.uid)}, processes

    runaway = asyncio.create_task(one.exec("while :; do :; done"))
    alive = await two.exec("echo still here")
    assert alive.stdout.strip() == "still here"
    stopped = await runaway
    assert stopped.limit == "cpu" and stopped.exit_code == 152

    await provider.delete(one.ref)
    assert (await sleeper).exit_code == 137  # its end killed it
    left = await two.exec(f"ls {os.path.dirname(one.home)}; {PROCESSES}")
    assert str(one.uid) not in left.stdout
    for box in (two, other):
        await provider.delete(box.ref)
    assert await provider.labelled({TENANT: acme}) == []
    assert await provider.labelled({TENANT: globex}) == []
