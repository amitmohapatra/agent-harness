"""The tenant's shared sandbox: one container per tenant (and image, network, CPU and memory),
in which every run of the tenant's gets a workspace and a uid of its own through the isolation
layer (:mod:`trellis.harness.sandbox.isolation`) — ``sandbox()``'s default provider.

A container is never shared across tenants: its name is a digest of the tenant and what the
container is made of, and it is labelled with the tenant (:data:`POOL`). Inside it the layer
keeps runs apart (workspace 0700, a uid per run, a process and mount namespace per command,
read-only filesystems but the workspace, limits per command, no secret in the environment), and
a run's end deletes its workspace, kills its processes and frees its uid — the container stays,
for the tenant's next run. The container holds the capabilities the layer's own root commands
need (switching uid, mounting, making and removing workspaces) and nothing else; a run's
command has none of them (``setpriv``: no capability, no new privileges).

A probe (``SandboxSource.card``) and any sandbox named for no tenant get a container of their
own, as :class:`~trellis.harness.sandbox.docker.DockerSandbox` makes."""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping, Sequence
from typing import Any, Final

from trellis.contracts import ConfigurationError, stable_id
from trellis.harness.sandbox.base import TENANT, SandboxLost, SandboxRef, SandboxSpec
from trellis.harness.sandbox.docker import (
    OUTPUT_BYTES,
    DockerSandbox,
    Sink,
    _checked,
    _put,
    _streams,
)
from trellis.harness.sandbox.isolation import (
    METADATA,
    TENANT_UIDS,
    TENANTS,
    UID_BASE,
    Isolation,
    Limits,
    Ran,
    Workspace,
)

#: The label of a tenant's shared container: the tenant's id.
POOL: Final = "trellis.pool"
#: The label a workspace's reference names its container by.
POOL_OF: Final = "trellis.pool_of"
#: Where the workspaces are in a shared container.
ROOT: Final = "/workspaces"
#: The most processes a tenant's container may have at once (each run's are fewer: ``Limits``).
POOL_PIDS: Final = 4_096
#: What the layer's root commands need in the container: switch uid (``setpriv``), make the
#: command's namespaces and mounts (``unshare``, ``mount``), own and remove a run's files, and
#: kill a run's processes. A run's command keeps none of them.
POOL_CAPS: Final = (
    "SYS_ADMIN",
    "SETUID",
    "SETGID",
    "SETPCAP",
    "CHOWN",
    "DAC_OVERRIDE",
    "FOWNER",
    "KILL",
)
#: What keeps the cloud metadata server (``isolation.METADATA``) from a container whose
#: network is open: a blackhole route for each of its networks in the container's own network
#: (its commands' alone), added over netlink by the image's Python as the layer's root, which
#: has ``NET_ADMIN`` there (a command has no capability); an address of a family the container
#: has no network of (errno 95, 97) is out of reach already.
ROUTES: Final = (
    "python3 -I -c '"
    "import socket, struct, sys\n"
    "for net in sys.argv[1:]:\n"
    '    addr, bits = net.split("/")\n'
    '    fam = socket.AF_INET6 if ":" in addr else socket.AF_INET\n'
    "    dst = socket.inet_pton(fam, addr)\n"
    '    body = struct.pack("BBBBBBBBI", fam, int(bits), 0, 0, 254, 4, 0, 6, 0)\n'
    '    body += struct.pack("HH", 4 + len(dst), 1) + dst\n'
    "    s = socket.socket(socket.AF_NETLINK, socket.SOCK_RAW, 0)\n"
    '    s.sendto(struct.pack("IHHII", 16 + len(body), 24, 0x405, 1, 0) + body, (0, 0))\n'
    '    err = -struct.unpack("i", s.recv(4096)[16:20])[0]\n'
    "    if err not in (0, 17, 95, 97):\n"
    '        sys.exit(f"the metadata server {net} could not be refused: errno {err}")\n'
    '\' "$@"'
)
#: Runs ``$@`` with the staged file ``$0`` as its input, then removes it.
STAGED: Final = '"$@" < "$0"; s=$?; rm -f "$0"; exit $s'


class _Pools(DockerSandbox):
    """The tenants' containers: a Docker sandbox with the layer's capabilities and more
    processes."""

    def _config(self, ref: SandboxRef, spec: SandboxSpec) -> dict[str, Any]:
        config = super()._config(ref, spec)
        config["WorkingDir"] = "/"
        caps = [*POOL_CAPS, *(["NET_ADMIN"] if spec.network == "open" else [])]
        config["HostConfig"] |= {"CapAdd": caps, "PidsLimit": POOL_PIDS}
        return config


class DockerRunner:
    """The layer's root commands in a container (``docker exec`` as root, through the Engine
    API): an input is staged as a file of root's first (``stage``) and fed to the command."""

    def __init__(self, provider: DockerSandbox, container: SandboxRef, stage: str) -> None:
        self.provider = provider
        self.container = container
        self.stage = stage

    async def run(
        self,
        argv: Sequence[str],
        *,
        stdin: bytes | None = None,
        keep: int | None = OUTPUT_BYTES,
        sink: Sink | None = None,
    ) -> Ran:
        name = self.container.id
        async with self.provider._client() as client:
            command = list(argv)
            if stdin is not None:
                staged = f"{self.stage}/{uuid.uuid4().hex}"
                await _put(client, self.container, {staged: stdin})
                command = ["sh", "-c", STAGED, staged, *command]
            config = {
                "Cmd": command,
                "AttachStdout": True,
                "AttachStderr": True,
                "User": "0:0",
                "WorkingDir": "/",
            }
            made = await client.post(f"/containers/{name}/exec", json=config)
            exec_id = _checked(made, self.container).json()["Id"]
            out, err = await _streams(client, exec_id, self.container, keep, sink)
            ended = _checked(await client.get(f"/exec/{exec_id}/json"), self.container).json()
        return Ran(int(ended["ExitCode"]), out, err)


class SharedDockerSandbox:
    """Sandboxes as workspaces in the tenant's shared container (``name`` ``"docker-shared"``):
    ``image`` (else the spec's, else ``python:3.12-slim``), the daemon at ``host`` or
    ``socket`` (else ``DOCKER_HOST``, the docker context, the local socket), ``limits`` each
    command's (:class:`~trellis.harness.sandbox.isolation.Limits`), ``root`` where the
    workspaces are in the container."""

    name = "docker-shared"

    def __init__(
        self,
        image: str | None = None,
        *,
        host: str | None = None,
        socket: str | None = None,
        limits: Limits | None = None,
        root: str = ROOT,
    ) -> None:
        self.docker = DockerSandbox(image, host=host, socket=socket)
        self.pools = _Pools(image, host=self.docker.host)
        self.limits = limits or Limits()
        self.root = root
        #: the open containers the metadata server was refused in (:data:`ROUTES`)
        self._routed: set[str] = set()

    @property
    def image(self) -> str:
        return self.docker.image

    def pool_of(self, tenant: str, spec: SandboxSpec) -> SandboxRef:
        """The container of ``tenant``'s that a sandbox of ``spec`` works in: one per tenant
        and what it is made of (image, network, CPU, memory), never another tenant's."""
        made = (spec.image or self.image, spec.network, spec.cpu, spec.memory)
        name = f"trellis-pool-{stable_id(tenant, *map(str, made))}"
        return SandboxRef(provider=self.name, id=name, labels={POOL: tenant})

    def isolation(self, pool: SandboxRef) -> Isolation:
        """The layer in ``pool``: the tenant's uids its own range (a digest of the tenant: no
        two tenants' containers share a uid, but by a one-in-a-million chance)."""
        block = int(stable_id("uids", pool.labels[POOL]), 16) % TENANTS
        return Isolation(
            DockerRunner(self.pools, pool, f"{self.root}/.trellis/stage"),
            self.root,
            network="open",  # the container's own: none unless the spec opens it
            uid_base=UID_BASE + block * TENANT_UIDS,
        )

    async def create(self, ref: SandboxRef, spec: SandboxSpec) -> Any:
        tenant = ref.labels.get(TENANT)
        if tenant is None:  # a probe's: a container of its own
            return await self.docker.create(ref, spec)
        if isinstance(spec.network, tuple):
            raise ConfigurationError(
                "a Docker sandbox's network is none or open: Docker cannot limit it to some "
                "hosts (use a provider that can, or an egress proxy)"
            )
        pool = self.pool_of(tenant, spec)
        await self.pools.create(pool, spec.model_copy(update={"files": {}}))
        layer = self.isolation(pool)
        if spec.network == "open" and pool.id not in self._routed:
            await layer._root(ROUTES, *METADATA)
            self._routed.add(pool.id)
        labels = {**ref.labels, POOL_OF: pool.id}
        uid, home, made = await layer.open(tenant, ref.id, labels)
        box = self._box(layer, ref.model_copy(update={"labels": labels}), uid, home)
        for path, content in spec.files.items() if made else ():
            await box.write(path, content.encode() if isinstance(content, str) else content)
        return box

    async def attach(self, ref: SandboxRef) -> Any:
        pool = self._pool(ref)
        if pool is None:
            return await self.docker.attach(ref)
        await self.pools.attach(pool)
        layer = self.isolation(pool)
        uid, home = await layer.attach(ref.id)
        return self._box(layer, ref, uid, home)

    async def delete(self, ref: SandboxRef) -> None:
        pool = self._pool(ref)
        if pool is None:
            await self.docker.delete(ref)
            return
        try:
            await self.isolation(pool).release(ref.id)
        except SandboxLost:  # its container is gone, and the workspace with it
            return

    async def labelled(self, labels: Mapping[str, str]) -> list[SandboxRef]:
        tenant = labels.get(TENANT)
        wanted = {POOL: tenant} if tenant is not None else {}
        params = {"all": "true", "filters": _present(POOL, wanted)}
        async with self.pools._client() as client:
            found = _checked(await client.get("/containers/json", params=params), None).json()
        refs = []
        for container in found:
            held = {POOL: container["Labels"][POOL]}
            pool = SandboxRef(provider=self.name, id=container["Names"][0].lstrip("/"), labels=held)
            for name, kept in await self.isolation(pool).listed():
                wanted = {**labels, POOL_OF: pool.id}  # its own: none another container lists
                if all(kept.get(k) == v for k, v in wanted.items()):
                    refs.append(SandboxRef(provider=self.name, id=name, labels=kept))
        return refs

    def _pool(self, ref: SandboxRef) -> SandboxRef | None:
        name, tenant = ref.labels.get(POOL_OF), ref.labels.get(TENANT)
        if name is None or tenant is None:
            return None
        return SandboxRef(provider=self.name, id=name, labels={POOL: tenant})

    def _box(self, layer: Isolation, ref: SandboxRef, uid: int, home: str) -> Workspace:
        return Workspace(layer, ref, uid, home, limits=self.limits, network="open")


def _present(key: str, labels: Mapping[str, str]) -> str:
    """The Engine API's filter: every container with the label ``key`` (of the value
    ``labels`` gives it, when it does)."""
    wanted = [f"{k}={v}" for k, v in labels.items()] or [key]
    return json.dumps({"label": wanted})


__all__ = ["POOL", "POOL_CAPS", "DockerRunner", "SharedDockerSandbox"]
