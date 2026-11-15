"""The Docker Engine API in process, whose containers are this machine: what ``docker exec``
runs is run here for real, as root (``tests/support/isolation.py``), and an archive put is
unpacked here. So the shared sandbox's every Engine API call and every script of the isolation
layer run in ``make test`` with real processes and real uids, without a daemon;
``tests/live/test_live_shared_sandbox.py`` runs the same against a real one."""

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

from trellis.harness.sandbox.docker import API_VERSION
from trellis.harness.sandbox.isolation import ROOT_ENV


def frame(stream: int, data: bytes) -> bytes:
    return bytes([stream, 0, 0, 0]) + len(data).to_bytes(4, "big") + data


@dataclass
class Made:
    config: dict[str, Any]
    state: str = "created"
    files: dict[str, bytes] = field(default_factory=dict)


class LocalEngine:
    def __init__(self, helper: tuple[str, ...]) -> None:
        self.helper = helper
        self.containers: dict[str, Made] = {}
        self.execs: dict[str, dict[str, Any]] = {}
        self.codes: dict[str, int] = {}
        self.requests: list[str] = []

    @contextmanager
    def running(self) -> Iterator[LocalEngine]:
        with respx.mock(base_url=f"http://docker/{API_VERSION}", assert_all_called=False) as api:
            api.route(path__regex=r".*").mock(side_effect=self._handle)
            yield self

    async def _handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path.removeprefix(f"/{API_VERSION}")
        self.requests.append(f"{request.method} {path}")
        routes = {
            ("POST", "/containers/create"): self._create,
            ("GET", "/containers/json"): self._list,
            ("GET", r"/containers/([^/]+)/json"): self._inspect,
            ("POST", r"/containers/([^/]+)/start"): self._start,
            ("DELETE", r"/containers/([^/]+)"): self._remove,
            ("PUT", r"/containers/([^/]+)/archive"): self._put,
            ("POST", r"/containers/([^/]+)/exec"): self._exec,
            ("POST", r"/exec/([^/]+)/start"): self._start_exec,
            ("GET", r"/exec/([^/]+)/json"): self._exec_info,
            ("GET", "/images/json"): self._images,
        }
        for (method, pattern), handler in routes.items():
            found = re.fullmatch(pattern, path)
            if method == request.method and found:
                return await handler(request, *found.groups())
        raise AssertionError(f"unexpected {request.method} {path}")

    async def _create(self, request: httpx.Request) -> httpx.Response:
        name = request.url.params["name"]
        if name in self.containers:
            return _error(409, f'Conflict. The container name "/{name}" is already in use')
        self.containers[name] = Made(json.loads(request.content))
        return httpx.Response(201, json={"Id": name})

    async def _list(self, request: httpx.Request) -> httpx.Response:
        wanted = json.loads(request.url.params["filters"])["label"]
        return httpx.Response(
            200,
            json=[
                {"Names": [f"/{name}"], "Labels": made.config.get("Labels") or {}}
                for name, made in self.containers.items()
                if all(_has(made.config.get("Labels") or {}, label) for label in wanted)
            ],
        )

    async def _inspect(self, request: httpx.Request, name: str) -> httpx.Response:
        made = self.containers.get(name)
        if made is None:
            return _error(404, f"No such container: {name}")
        config = {"Labels": made.config.get("Labels")}
        return httpx.Response(200, json={"Config": config, "State": {"Status": made.state}})

    async def _start(self, request: httpx.Request, name: str) -> httpx.Response:
        self.containers[name].state = "running"
        return httpx.Response(204)

    async def _remove(self, request: httpx.Request, name: str) -> httpx.Response:
        if self.containers.pop(name, None) is None:
            return _error(404, f"No such container: {name}")
        return httpx.Response(204)

    async def _images(self, request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[])

    async def _put(self, request: httpx.Request, name: str) -> httpx.Response:
        made = self.containers[name]
        if made.config.get("HostConfig", {}).get("CapAdd"):  # a shared container: this machine
            proc = await asyncio.create_subprocess_exec(
                *self.helper, "tar", "-x", "-C", "/", stdin=asyncio.subprocess.PIPE, env=ROOT_ENV
            )
            await proc.communicate(request.content)
            assert proc.returncode == 0
        else:  # a container of its own (a probe's): its files kept as they are
            with tarfile.open(fileobj=io.BytesIO(request.content)) as tar:
                for member in tar.getmembers():
                    content = tar.extractfile(member)
                    assert content is not None
                    made.files["/" + member.name] = content.read()
        return httpx.Response(200)

    async def _exec(self, request: httpx.Request, name: str) -> httpx.Response:
        if name not in self.containers:
            return _error(404, f"No such container: {name}")
        exec_id = f"exec{len(self.execs)}"
        self.execs[exec_id] = json.loads(request.content)
        return httpx.Response(201, json={"Id": exec_id})

    async def _start_exec(self, request: httpx.Request, exec_id: str) -> httpx.Response:
        return httpx.Response(200, content=self._ran(exec_id))

    async def _ran(self, exec_id: str) -> AsyncIterator[bytes]:
        config = self.execs[exec_id]
        assert config.get("User") == "0:0"  # the layer's commands run as root
        proc = await asyncio.create_subprocess_exec(
            *self.helper,
            *config["Cmd"],
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=ROOT_ENV,
        )
        out, err = await proc.communicate()
        self.codes[exec_id] = proc.returncode or 0
        yield frame(1, out) + frame(2, err)

    async def _exec_info(self, request: httpx.Request, exec_id: str) -> httpx.Response:
        return httpx.Response(200, json={"ExitCode": self.codes[exec_id]})


def _has(labels: dict[str, str], label: str) -> bool:
    key, eq, value = label.partition("=")
    return key in labels if not eq else labels.get(key) == value


def _error(status: int, message: str) -> httpx.Response:
    return httpx.Response(status, json={"message": message})
