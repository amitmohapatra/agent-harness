"""Sandboxes as containers of a Docker daemon, through its Engine API (httpx on its socket).

Each sandbox is a container named and labelled after its run, from the spec's image (else the
provider's), kept alive doing nothing until it is deleted, and hardened: no network unless the
spec opens it, every capability dropped, no new privileges, at most :data:`PIDS_LIMIT`
processes, the spec's CPU and memory as its limits, and an init process that reaps what a
killed command leaves. A command runs with ``sh -c`` in :data:`WORKDIR` and writes its process
id first, so a command out of time is killed with everything it started (its process group).
A snapshot is the container committed to an image (``trellis-snapshot:<id>``); a pause is
Docker's (its processes frozen until the run continues).

The daemon is this machine's (or the one whose socket is given): a run paused here and resumed
by a worker that reaches another daemon finds its sandbox lost. Docker shares the host's
kernel: for code nobody reviewed, run the daemon with a stronger runtime (gVisor) or use a
provider of microVMs.
"""

from __future__ import annotations

import asyncio
import io
import json
import posixpath
import tarfile
import time
import uuid
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from typing import Any, Final

import httpx

from trellis.contracts import ConfigurationError, ToolError
from trellis.harness.sandbox.base import (
    PREFIX,
    ExecResult,
    SandboxLost,
    SandboxRef,
    SandboxSpec,
)

#: Where the daemon listens, and the Engine API version spoken (Docker 24 and later).
DOCKER_SOCKET: Final = "/var/run/docker.sock"
API_VERSION: Final = "v1.43"
#: The image of a sandbox whose spec and provider name none (``SANDBOX_IMAGE`` names another).
DEFAULT_IMAGE: Final = "python:3.12-slim"
#: The label of a snapshot's image: the sandbox it was taken of (what ``delete`` removes by).
SNAPSHOT: Final = "trellis.snapshot_of"
#: Where a command runs, and what a relative path is relative to.
WORKDIR: Final = "/workspace"
#: The most processes a sandbox may have at once.
PIDS_LIMIT: Final = 512
#: What is kept of a command's output, per stream: its end (the rest is cut, and says so).
OUTPUT_BYTES: Final = 256 * 1024
#: How long one call of the Engine API may take (a command's output and a pull: as long as
#: they take, within the command's time).
REQUEST_SECONDS: Final = 30.0
#: What a sandbox's container does when no command runs: nothing, until it is deleted.
IDLE: Final = ["sh", "-c", "while :; do sleep 3600; done"]
#: A command, run as ``sh -c WRAPPER <pid file> <command>``: its process id written first (the
#: leader of its process group), the file removed when it ends.
WRAPPER: Final = 'echo $$ >"$0"; sh -c "$1"; status=$?; rm -f "$0"; exit $status'
#: Kills the process group a pid file names (waiting a moment for a command that has not
#: written it yet).
KILL: Final = (
    'for i in 1 2 3 4 5 6 7 8 9 10; do [ -s "$0" ] && break; sleep 0.1; done; '
    'kill -KILL -"$(cat "$0")" 2>/dev/null; rm -f "$0"'
)


class DockerSandbox:
    """The Docker provider (``SANDBOX=docker``): ``image``, the image of a sandbox whose spec
    names none (else :data:`DEFAULT_IMAGE`); ``socket``, the daemon's Unix socket."""

    name = "docker"

    def __init__(self, image: str | None = None, *, socket: str = DOCKER_SOCKET) -> None:
        self.image = image or DEFAULT_IMAGE
        self.socket = socket

    async def create(self, ref: SandboxRef, spec: SandboxSpec) -> Container:
        if isinstance(spec.network, tuple):
            raise ConfigurationError(
                "a Docker sandbox's network is none or open: Docker cannot limit it to some "
                "hosts (use a provider that can, or an egress proxy)"
            )
        config = self._config(ref, spec)
        async with self._client() as client:
            response = await client.post("/containers/create", params={"name": ref.id}, json=config)
            if response.status_code == 404:  # the image is not here yet
                if ref.snapshot is not None:
                    raise SandboxLost(
                        f"the sandbox {ref.id} and its snapshot are gone", source="tools"
                    )
                await _pulled(client, config["Image"])
                response = await client.post(
                    "/containers/create", params={"name": ref.id}, json=config
                )
            if response.status_code != 409:  # 409: it exists, an earlier attempt made it
                _checked(response, ref)
            files = {} if ref.snapshot is not None else spec.files
            await _running(client, ref, files)
        return Container(self, ref)

    async def attach(self, ref: SandboxRef) -> Container:
        async with self._client() as client:
            await _running(client, ref, {})
        return Container(self, ref)

    async def delete(self, ref: SandboxRef) -> None:
        async with self._client() as client:
            gone = await client.delete(f"/containers/{ref.id}", params={"force": "true"})
            if gone.status_code != 404:
                _checked(gone, ref)
            snapshots = {"filters": _filter({SNAPSHOT: ref.id})}  # its own images, none other
            images = await client.get("/images/json", params=snapshots)
            for image in _checked(images, ref).json():
                removed = await client.delete(f"/images/{image['Id']}", params={"force": "true"})
                if removed.status_code != 404:
                    _checked(removed, ref)

    async def labelled(self, labels: Mapping[str, str]) -> list[SandboxRef]:
        params = {"all": "true", "filters": _filter(labels)}
        async with self._client() as client:
            found = _checked(await client.get("/containers/json", params=params), None).json()
        return [
            SandboxRef(
                provider=self.name,
                id=container["Names"][0].lstrip("/"),
                labels={k: v for k, v in container["Labels"].items() if k.startswith(PREFIX)},
            )
            for container in found
        ]

    async def pause(self, ref: SandboxRef) -> None:
        async with self._client() as client:
            paused = await client.post(f"/containers/{ref.id}/pause")
            if paused.status_code != 409:  # 409: paused already
                _checked(paused, ref)

    async def snapshot(self, ref: SandboxRef) -> str:
        params = {"container": ref.id, "repo": "trellis-snapshot", "tag": ref.id}
        async with self._client() as client:
            labels = {**ref.labels, SNAPSHOT: ref.id}
            committed = await client.post("/commit", params=params, json={"Labels": labels})
            return str(_checked(committed, ref).json()["Id"])

    def _config(self, ref: SandboxRef, spec: SandboxSpec) -> dict[str, Any]:
        limits: dict[str, Any] = {}
        if spec.cpu is not None:
            limits["NanoCpus"] = int(spec.cpu * 1e9)
        if spec.memory is not None:
            limits["Memory"] = spec.memory * 1024 * 1024
        return {
            "Image": ref.snapshot or spec.image or self.image,
            "Entrypoint": IDLE,
            "Cmd": None,
            "WorkingDir": WORKDIR,
            "Labels": ref.labels,
            "HostConfig": {
                "NetworkMode": "none" if spec.network == "none" else "bridge",
                "Init": True,
                "CapDrop": ["ALL"],
                "SecurityOpt": ["no-new-privileges"],
                "PidsLimit": PIDS_LIMIT,
                **limits,
            },
        }

    @asynccontextmanager
    async def _client(self) -> AsyncIterator[httpx.AsyncClient]:
        transport = httpx.AsyncHTTPTransport(uds=self.socket)
        base = f"http://docker/{API_VERSION}"
        async with httpx.AsyncClient(
            transport=transport, base_url=base, timeout=REQUEST_SECONDS
        ) as client:
            yield client


class Container:
    """One Docker sandbox, attached."""

    def __init__(self, provider: DockerSandbox, ref: SandboxRef) -> None:
        self.provider = provider
        self._ref = ref

    @property
    def ref(self) -> SandboxRef:
        return self._ref

    async def exec(
        self,
        command: str,
        *,
        timeout: float | None = None,  # noqa: ASYNC109 - the command's, killed past it
        env: Mapping[str, str] | None = None,
    ) -> ExecResult:
        pid_file = f"/tmp/.trellis-exec-{uuid.uuid4().hex}"
        config = {
            "Cmd": ["sh", "-c", WRAPPER, pid_file, command],
            "AttachStdout": True,
            "AttachStderr": True,
            "WorkingDir": WORKDIR,
            "Env": [f"{name}={value}" for name, value in (env or {}).items()],
        }
        async with self.provider._client() as client:
            made = await client.post(f"/containers/{self.ref.id}/exec", json=config)
            exec_id = _checked(made, self.ref).json()["Id"]
            try:
                async with asyncio.timeout(timeout):
                    stdout, stderr = await _output(client, exec_id, self.ref)
            except BaseException:  # out of time, cancelled, cut off: stopped, not left running
                await asyncio.shield(self._kill(pid_file))
                raise
            ended = _checked(await client.get(f"/exec/{exec_id}/json"), self.ref).json()
        return ExecResult(exit_code=ended["ExitCode"], stdout=stdout, stderr=stderr)

    async def read(self, path: str) -> bytes:
        where = _absolute(path)
        async with self.provider._client() as client:
            response = await client.get(
                f"/containers/{self.ref.id}/archive", params={"path": where}
            )
        if response.status_code == 404 and "Could not find the file" in response.text:
            raise ToolError(f"there is no file {where} in the sandbox", source="tools")
        archive = _checked(response, self.ref).content
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            member = tar.next()
            content = tar.extractfile(member) if member is not None and member.isfile() else None
            if content is None:
                raise ToolError(f"{where} in the sandbox is not a file", source="tools")
            return content.read()

    async def write(self, path: str, data: bytes) -> None:
        async with self.provider._client() as client:
            await _put(client, self.ref, {path: data})

    async def _kill(self, pid_file: str) -> None:
        """Kill the command whose process id ``pid_file`` holds, and what it started."""
        config = {"Cmd": ["sh", "-c", KILL, pid_file]}
        async with self.provider._client() as client:
            made = await client.post(f"/containers/{self.ref.id}/exec", json=config)
            if made.is_success:
                await client.post(f"/exec/{made.json()['Id']}/start", json={"Detach": False})


async def _running(
    client: httpx.AsyncClient, ref: SandboxRef, files: Mapping[str, str | bytes]
) -> None:
    """The container ``ref`` names, checked to be the run's and running: a new one gets its
    ``files`` before it starts, a paused one is resumed, a stopped one started again."""
    found = await client.get(f"/containers/{ref.id}/json")
    if found.status_code == 404:
        raise SandboxLost(f"the sandbox {ref.id} is gone", source="tools")
    info = _checked(found, ref).json()
    labels = info["Config"]["Labels"] or {}
    if any(labels.get(key) != value for key, value in ref.labels.items()):
        raise SandboxLost(f"the container {ref.id} is not this run's sandbox", source="tools")
    state = info["State"]["Status"]
    if state == "running":
        return
    if state == "paused":
        _checked(await client.post(f"/containers/{ref.id}/unpause"), ref)
        return
    if state == "created" and files:
        await _put(client, ref, files)
    _checked(await client.post(f"/containers/{ref.id}/start"), ref)


async def _put(
    client: httpx.AsyncClient, ref: SandboxRef, files: Mapping[str, str | bytes]
) -> None:
    """Write ``files`` into the container (their directories made as needed)."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for path, content in files.items():
            data = content.encode() if isinstance(content, str) else content
            member = tarfile.TarInfo(_absolute(path).lstrip("/"))
            member.size, member.mode, member.mtime = len(data), 0o644, int(time.time())
            tar.addfile(member, io.BytesIO(data))
    put = await client.put(
        f"/containers/{ref.id}/archive",
        params={"path": "/"},
        content=buffer.getvalue(),
        headers={"Content-Type": "application/x-tar"},
    )
    _checked(put, ref)


async def _output(client: httpx.AsyncClient, exec_id: str, ref: SandboxRef) -> tuple[str, str]:
    """Start the exec and read its output to the end: the multiplexed stream's frames (a
    header — the stream, 3 bytes, the length — then the bytes), each stream's end kept."""
    streams = {1: _Tail(), 2: _Tail()}
    pending = bytearray()
    start = {"Detach": False, "Tty": False}
    async with client.stream("POST", f"/exec/{exec_id}/start", json=start, timeout=None) as got:
        if not got.is_success:
            await got.aread()
            _checked(got, ref)
        async for chunk in got.aiter_bytes():
            pending += chunk
            while len(pending) >= 8:
                size = int.from_bytes(pending[4:8], "big")
                if len(pending) < 8 + size:
                    break
                if pending[0] in streams:
                    streams[pending[0]].add(bytes(pending[8 : 8 + size]))
                del pending[: 8 + size]
    return streams[1].text(), streams[2].text()


class _Tail:
    """The last :data:`OUTPUT_BYTES` of a stream, and how much was cut before them."""

    def __init__(self) -> None:
        self.kept = bytearray()
        self.cut = 0

    def add(self, data: bytes) -> None:
        self.kept += data
        over = len(self.kept) - OUTPUT_BYTES
        if over > 0:
            del self.kept[:over]
            self.cut += over

    def text(self) -> str:
        text = self.kept.decode("utf-8", errors="replace")
        return f"[{self.cut} bytes cut]\n{text}" if self.cut else text


async def _pulled(client: httpx.AsyncClient, image: str) -> None:
    """Pull ``image``: the daemon streams its progress, and a failure may come as its last
    line."""
    async with client.stream(
        "POST", "/images/create", params={"fromImage": image}, timeout=None
    ) as pulling:
        lines = [line async for line in pulling.aiter_lines() if line.strip()]
    last = json.loads(lines[-1]) if lines else {}
    failure = last.get("error") or (last.get("message") if not pulling.is_success else None)
    if failure:
        raise ToolError(f"the sandbox image {image} cannot be pulled: {failure}", source="tools")


def _checked(response: httpx.Response, ref: SandboxRef | None) -> httpx.Response:
    """``response`` when it succeeded; else the daemon's message as an error — the sandbox
    lost when its container is gone, one that may pass when the daemon failed (5xx)."""
    if response.is_success:
        return response
    try:
        message = str(response.json().get("message"))
    except ValueError:
        message = response.text
    if ref is not None and response.status_code == 404 and "No such container" in message:
        raise SandboxLost(f"the sandbox {ref.id} is gone", source="tools")
    raise ToolError(
        f"Docker answered {response.status_code}: {message}",
        source="tools",
        retryable=response.status_code >= 500,
    )


def _filter(labels: Mapping[str, str]) -> str:
    """The Engine API's filter on every one of ``labels``."""
    return json.dumps({"label": [f"{key}={value}" for key, value in labels.items()]})


def _absolute(path: str) -> str:
    return posixpath.normpath(posixpath.join(WORKDIR, path))
