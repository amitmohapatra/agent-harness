"""A sandbox provider in process, as the harness drives one: each sandbox a dict of files, its
commands scripted — ``echo <text>`` prints it, ``cat <path>`` a file, ``sleep <seconds>`` waits (and is killed past its
time), ``false`` exits 1, ``crash`` kills the worker (:attr:`FakeSandboxes.crash`) — and
everything the provider was asked to do, in order. :class:`FakeSandboxes` has the core only; :class:`PausingSandboxes` can also
snapshot and pause."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass, field

from trellis.contracts import ToolError
from trellis.harness.sandbox import ExecResult, SandboxLost, SandboxRef, SandboxSpec


@dataclass
class Box:
    """One sandbox's state."""

    labels: dict[str, str]
    files: dict[str, bytes] = field(default_factory=dict)
    paused: bool = False


class FakeSandboxes:
    name = "fake"

    def __init__(self) -> None:
        self.boxes: dict[str, Box] = {}
        #: what was asked, in order: ("create" | "adopt" | "attach" | "delete" | ..., id)
        self.calls: list[tuple[str, str]] = []
        self.commands: list[str] = []
        self.killed: list[str] = []
        #: how the worker dies: raised by the command ``crash``, and once right after a create
        #: made its sandbox when ``crash_after_create``
        self.crash: BaseException = SystemExit("the worker died")
        self.crash_after_create = False

    async def create(self, ref: SandboxRef, spec: SandboxSpec) -> FakeBox:
        if ref.id in self.boxes:
            self.calls.append(("adopt", ref.id))
        else:
            self.calls.append(("create", ref.id))
            box = self.boxes[ref.id] = Box(dict(ref.labels))
            box.files.update(
                {p: c.encode() if isinstance(c, str) else c for p, c in spec.files.items()}
            )
        if self.crash_after_create:
            self.crash_after_create = False
            raise self.crash
        return FakeBox(self, ref)

    async def attach(self, ref: SandboxRef) -> FakeBox:
        self.calls.append(("attach", ref.id))
        self.live(ref).paused = False
        return FakeBox(self, ref)

    async def delete(self, ref: SandboxRef) -> None:
        self.calls.append(("delete", ref.id))
        self.boxes.pop(ref.id, None)

    async def labelled(self, labels: Mapping[str, str]) -> list[SandboxRef]:
        return [
            SandboxRef(provider=self.name, id=name, labels=box.labels)
            for name, box in self.boxes.items()
            if all(box.labels.get(k) == v for k, v in labels.items())
        ]

    def live(self, ref: SandboxRef) -> Box:
        box = self.boxes.get(ref.id)
        if box is None:
            raise SandboxLost(f"the sandbox {ref.id} is gone", source="tools")
        return box

    def made(self) -> list[str]:
        """The sandboxes actually created."""
        return [name for call, name in self.calls if call == "create"]


class PausingSandboxes(FakeSandboxes):
    """The same, with snapshots (a copy of the files) and pauses."""

    def __init__(self) -> None:
        super().__init__()
        self.snapshots: dict[str, dict[str, bytes]] = {}

    async def create(self, ref: SandboxRef, spec: SandboxSpec) -> FakeBox:
        if ref.snapshot is not None and ref.id not in self.boxes:
            self.calls.append(("restore", ref.id))
            box = self.boxes[ref.id] = Box(dict(ref.labels))
            box.files.update(self.snapshots[ref.snapshot])
            return FakeBox(self, ref)
        return await super().create(ref, spec)

    async def delete(self, ref: SandboxRef) -> None:
        await super().delete(ref)
        for name in [s for s in self.snapshots if s.startswith(f"{ref.id}@")]:
            del self.snapshots[name]

    async def snapshot(self, ref: SandboxRef) -> str:
        self.calls.append(("snapshot", ref.id))
        name = f"{ref.id}@{len(self.snapshots)}"
        self.snapshots[name] = dict(self.live(ref).files)
        return name

    async def pause(self, ref: SandboxRef) -> None:
        self.calls.append(("pause", ref.id))
        self.live(ref).paused = True


class FakeBox:
    def __init__(self, provider: FakeSandboxes, ref: SandboxRef) -> None:
        self.provider = provider
        self.ref = ref

    async def exec(
        self,
        command: str,
        *,
        timeout: float | None = None,  # noqa: ASYNC109 - the command's
        env: Mapping[str, str] | None = None,
    ) -> ExecResult:
        box = self.provider.live(self.ref)
        assert not box.paused, "a command in a paused sandbox"
        self.provider.commands.append(command)
        if command == "crash":
            raise self.provider.crash
        try:
            async with asyncio.timeout(timeout):
                if command.startswith("sleep "):
                    await asyncio.sleep(float(command.split()[1]))
        except BaseException:
            self.provider.killed.append(command)
            raise
        said = command.removeprefix("echo ") if command.startswith("echo ") else ""
        if command.startswith("cat "):
            said = (await self.read(command.removeprefix("cat "))).decode()
        extra = "".join(f" {k}={v}" for k, v in (env or {}).items())
        return ExecResult(exit_code=1 if command == "false" else 0, stdout=said + extra)

    async def read(self, path: str) -> bytes:
        found = self.provider.live(self.ref).files.get(path)
        if found is None:
            raise ToolError(f"there is no file {path} in the sandbox", source="tools")
        return found

    async def write(self, path: str, data: bytes) -> None:
        self.provider.live(self.ref).files[path] = data
