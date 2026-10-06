"""The Docker Engine API in process, as ``DockerSandbox`` speaks it (respx, on the daemon's
base URL): containers with their labels, state and files; execs of scripted commands — ``echo
<text>`` prints it, ``fail`` writes to stderr and exits 2, ``sleep`` never ends (until it is
killed), ``noisy`` prints more than is kept, in frames cut across chunks; images, pulls and
commits. Every request is kept (``requests``), and a path in ``broken`` is answered once with its
response instead."""

from __future__ import annotations

import asyncio
import io
import json
import re
import tarfile
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import httpx
import respx

from trellis.harness.sandbox.docker import API_VERSION, DEFAULT_IMAGE, KILL, OUTPUT_BYTES


@dataclass
class FakeContainer:
    config: dict[str, Any]
    state: str = "created"
    files: dict[str, bytes] = field(default_factory=dict)

    @property
    def labels(self) -> dict[str, str]:
        return self.config.get("Labels") or {}


def frame(stream: int, data: bytes) -> bytes:
    return bytes([stream, 0, 0, 0]) + len(data).to_bytes(4, "big") + data


class FakeDocker:
    def __init__(self) -> None:
        self.containers: dict[str, FakeContainer] = {}
        #: images by name or id: their labels
        self.images: dict[str, dict[str, str]] = {DEFAULT_IMAGE: {}}
        #: what a pull of an image the daemon does not have says: nothing (it is pulled), or
        #: its last progress line
        self.pulls: dict[str, dict[str, Any]] = {}
        self.execs: dict[str, tuple[str, list[str], dict[str, Any]]] = {}
        self.killed: list[str] = []
        self.requests: list[str] = []
        self.broken: dict[str, httpx.Response] = {}
        #: the files of each committed image
        self.snapshots: dict[str, dict[str, bytes]] = {}

    @contextmanager
    def running(self) -> Iterator[FakeDocker]:
        with respx.mock(base_url=f"http://docker/{API_VERSION}", assert_all_called=False) as api:
            api.route(path__regex=r".*").mock(side_effect=self._handle)
            yield self

    # ------------------------------------------------------------------ the API
    async def _handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path.removeprefix(f"/{API_VERSION}")
        self.requests.append(f"{request.method} {path}")
        if path in self.broken:
            return self.broken.pop(path)
        for (method, pattern), handler in self._routes().items():
            found = re.fullmatch(pattern, path)
            if method == request.method and found:
                return await handler(request, *found.groups())
        raise AssertionError(f"unexpected {request.method} {path}")

    def _routes(self) -> dict[tuple[str, str], Any]:
        return {
            ("POST", "/containers/create"): self._create,
            ("GET", "/containers/json"): self._list,
            ("GET", r"/containers/([^/]+)/json"): self._inspect,
            ("POST", r"/containers/([^/]+)/(start|pause|unpause)"): self._state,
            ("DELETE", r"/containers/([^/]+)"): self._remove,
            ("PUT", r"/containers/([^/]+)/archive"): self._put,
            ("GET", r"/containers/([^/]+)/archive"): self._get,
            ("POST", r"/containers/([^/]+)/exec"): self._exec,
            ("POST", r"/exec/([^/]+)/start"): self._start_exec,
            ("GET", r"/exec/([^/]+)/json"): self._exec_info,
            ("POST", "/images/create"): self._pull,
            ("GET", "/images/json"): self._images,
            ("DELETE", r"/images/(.+)"): self._remove_image,
            ("POST", "/commit"): self._commit,
        }

    async def _create(self, request: httpx.Request) -> httpx.Response:
        name = request.url.params["name"]
        config = json.loads(request.content)
        if config["Image"] not in self.images:
            return _error(404, f"No such image: {config['Image']}")
        if name in self.containers:
            return _error(409, f'Conflict. The container name "/{name}" is already in use')
        self.containers[name] = FakeContainer(
            config, files=dict(self.snapshots.get(config["Image"], {}))
        )
        return httpx.Response(201, json={"Id": name, "Warnings": []})

    async def _list(self, request: httpx.Request) -> httpx.Response:
        wanted = json.loads(request.url.params["filters"])["label"]
        return httpx.Response(
            200,
            json=[
                {"Names": [f"/{name}"], "Labels": {**c.labels, "maintainer": "someone"}}
                for name, c in self.containers.items()
                if all(_has(c.labels, label) for label in wanted)
            ],
        )

    async def _inspect(self, request: httpx.Request, name: str) -> httpx.Response:
        container = self.containers.get(name)
        if container is None:
            return _error(404, f"No such container: {name}")
        config = {"Labels": container.config.get("Labels")}
        return httpx.Response(200, json={"Config": config, "State": {"Status": container.state}})

    async def _state(self, request: httpx.Request, name: str, verb: str) -> httpx.Response:
        container = self.containers.get(name)
        if container is None:
            return _error(404, f"No such container: {name}")
        if verb == "pause" and container.state == "paused":
            return _error(409, f"Container {name} is already paused")
        container.state = {"start": "running", "pause": "paused", "unpause": "running"}[verb]
        return httpx.Response(204)

    async def _remove(self, request: httpx.Request, name: str) -> httpx.Response:
        if self.containers.pop(name, None) is None:
            return _error(404, f"No such container: {name}")
        return httpx.Response(204)

    async def _put(self, request: httpx.Request, name: str) -> httpx.Response:
        container = self.containers[name]
        with tarfile.open(fileobj=io.BytesIO(request.content)) as tar:
            for member in tar.getmembers():
                content = tar.extractfile(member)
                assert content is not None
                container.files["/" + member.name] = content.read()
        return httpx.Response(200)

    async def _get(self, request: httpx.Request, name: str) -> httpx.Response:
        container = self.containers.get(name)
        if container is None:
            return _error(404, f"No such container: {name}")
        path = request.url.params["path"]
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w") as tar:
            if path in container.files:
                data = container.files[path]
                member = tarfile.TarInfo(path.rsplit("/", 1)[-1])
                member.size = len(data)
                tar.addfile(member, io.BytesIO(data))
            elif any(f.startswith(path + "/") for f in container.files):
                member = tarfile.TarInfo(path.rsplit("/", 1)[-1])
                member.type = tarfile.DIRTYPE
                tar.addfile(member)
            else:
                return _error(404, f"Could not find the file {path} in container {name}")
        return httpx.Response(200, content=buffer.getvalue())

    async def _exec(self, request: httpx.Request, name: str) -> httpx.Response:
        if name not in self.containers:
            return _error(404, f"No such container: {name}")
        exec_id = f"exec{len(self.execs)}"
        config = json.loads(request.content)
        self.execs[exec_id] = (name, config["Cmd"], config)
        return httpx.Response(201, json={"Id": exec_id})

    async def _start_exec(self, request: httpx.Request, exec_id: str) -> httpx.Response:
        _, cmd, _ = self.execs[exec_id]
        if cmd[2] == KILL:
            self.killed.append(cmd[3])
            return httpx.Response(200, content=b"")
        return httpx.Response(200, content=_output(cmd[4]))

    async def _exec_info(self, request: httpx.Request, exec_id: str) -> httpx.Response:
        _, cmd, _ = self.execs[exec_id]
        return httpx.Response(200, json={"ExitCode": 2 if cmd[-1] == "fail" else 0})

    async def _pull(self, request: httpx.Request) -> httpx.Response:
        image = request.url.params["fromImage"]
        said = self.pulls.get(image)
        if said is None:
            self.images[image] = {}
            return httpx.Response(200, content=b'{"status":"Pulling"}\n{"status":"Done"}\n')
        return httpx.Response(said.pop("status", 200), content=json.dumps(said).encode())

    async def _images(self, request: httpx.Request) -> httpx.Response:
        wanted = json.loads(request.url.params["filters"])["label"]
        return httpx.Response(
            200,
            json=[
                {"Id": image}
                for image, labels in self.images.items()
                if all(_has(labels, label) for label in wanted)
            ],
        )

    async def _remove_image(self, request: httpx.Request, image: str) -> httpx.Response:
        if self.images.pop(image, None) is None:
            return _error(404, f"No such image: {image}")
        return httpx.Response(200, json=[{"Deleted": image}])

    async def _commit(self, request: httpx.Request) -> httpx.Response:
        params = request.url.params
        name = params["container"]
        image = f"sha256:{name}-{len(self.images)}"
        self.images[image] = json.loads(request.content)["Labels"]
        self.snapshots[image] = dict(self.containers[name].files)
        assert params["repo"] == "trellis-snapshot" and params["tag"] == name
        return httpx.Response(201, json={"Id": image})


def _has(labels: dict[str, str], label: str) -> bool:
    key, _, value = label.partition("=")
    return labels.get(key) == value


def _output(command: str) -> AsyncIterator[bytes] | bytes:
    if command.startswith("echo "):
        return frame(1, command.removeprefix("echo ").encode())
    if command == "fail":
        return frame(2, b"boom")
    if command == "noisy":
        return _noisy()
    return _forever()


async def _noisy() -> AsyncIterator[bytes]:
    data = frame(0, b"ignored") + frame(1, b"x" * OUTPUT_BYTES) + frame(1, b"tail")
    for start in range(0, len(data), 7000):  # frames cut across chunks
        yield data[start : start + 7000]


async def _forever() -> AsyncIterator[bytes]:
    yield frame(1, b"started")
    await asyncio.sleep(60)
    yield b""  # never: it is killed first


def _error(status: int, message: str) -> httpx.Response:
    return httpx.Response(status, json={"message": message})
