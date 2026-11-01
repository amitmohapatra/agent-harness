"""The provider-neutral sandbox: what a provider implements, and nothing else.

A :class:`SandboxProvider` creates, attaches to and deletes sandboxes, and finds them by
label; a :class:`Sandbox` runs a command and reads and writes a file. That is the whole core,
the part every provider has (Docker, E2B, Daytona, Modal, Vercel...). What only some providers
can do is an optional capability the harness asks for by ``isinstance``:
:class:`SupportsPause` (stop the sandbox's processes while its run waits for a person, resumed
by ``attach``) and :class:`SupportsSnapshot` (keep its filesystem, so a lost sandbox can be
made again). A network limited to some hosts is a spec a provider either enforces or refuses
(``ConfigurationError``), never weakens.

The contract a provider keeps:

* ``create(ref, spec)`` is idempotent by ``ref.id``: a sandbox of that name that exists is
  adopted, not duplicated (an attempt after a crash creates "again"); ``spec.files`` are
  written before the sandbox's first command, and only into a sandbox it made; a ref that
  names a ``snapshot`` is made from it (no files: the snapshot has them);
* ``attach(ref)`` gives the sandbox a ref names, resumed when paused; one that is gone raises
  :class:`SandboxLost` — never a blank replacement;
* ``delete(ref)`` removes the sandbox and its snapshots; deleting one that is gone succeeds;
* a command's non-zero exit is a result, never an error; past ``timeout`` (or cancelled) the
  command is killed, with what it started, and ``TimeoutError`` (or the cancellation) raised;
* an error that may pass (the provider unreachable) propagates as it is: a read is tried again.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Final, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from trellis.contracts import ToolError

#: What every label the harness gives a sandbox begins with (the ones a provider lists).
PREFIX: Final = "trellis."
#: The labels naming a sandbox's run, its tenant and its agent.
RUN: Final = "trellis.run_id"
TENANT: Final = "trellis.tenant"
AGENT: Final = "trellis.agent_id"


class SandboxSpec(BaseModel):
    """What a sandbox is made of: ``image`` (else the provider's), at most ``cpu`` cores and
    ``memory`` MiB (else the provider's limits), its ``network`` — ``"none"``, the default: no
    network at all; ``"open"``; or the hosts it may reach, where the provider can enforce that
    — and the ``files`` it starts with (path, relative to its working directory or absolute →
    content)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    image: str | None = None
    cpu: float | None = Field(default=None, gt=0)
    memory: int | None = Field(default=None, gt=0)
    network: Literal["none", "open"] | tuple[str, ...] = "none"
    files: dict[str, str | bytes] = Field(default_factory=dict)


#: A command killed by SIGKILL, as the shell reports it (128 + 9): what the kernel's
#: out-of-memory killer does to a process past its sandbox's memory limit.
KILLED: Final = 137
#: Python's own words for a module it cannot import, and for memory it could not get.
NO_MODULE: Final = re.compile(r"ModuleNotFoundError: No module named '([^'.]+)")
OUT_OF_MEMORY: Final = "MemoryError"
#: What a hint says to do about memory.
IN_PARTS: Final = (
    "Work on the data in parts (read it in chunks, keep only the columns needed) rather than "
    "running it again as it is."
)


def hint(code: int, stderr: str, spec: SandboxSpec) -> str | None:
    """What the sandbox knows about why a command failed that its own output does not say:
    the memory limit it went past (a SIGKILL, or Python's ``MemoryError``), and whether a
    module Python could not import can be installed there (the sandbox's network). ``None``
    for any other outcome: the command's output says all there is."""
    limit = f"{spec.memory} MiB" if spec.memory is not None else "the provider's"
    if code == KILLED:
        return (
            f"Killed (exit {KILLED}): the command went past the sandbox's memory limit "
            f"({limit}), or was killed. {IN_PARTS}"
        )
    if code == 0:
        return None
    if OUT_OF_MEMORY in stderr:
        return (
            f"Out of memory: the command went past the sandbox's memory limit ({limit}). {IN_PARTS}"
        )
    missing = NO_MODULE.search(stderr)
    if missing is None:
        return None
    if spec.network == "none":
        return (
            f"The module {missing[1]} is not installed in the sandbox, and the sandbox has no "
            "network to install it: use the standard library (csv, json, statistics, sqlite3) "
            "or a module that is installed."
        )
    return (
        f"The module {missing[1]} is not installed in the sandbox: install it (pip install), "
        "then run the command again."
    )


class SandboxRef(BaseModel):
    """Which sandbox, as a run's journal keeps it: the ``provider``'s name, the sandbox's ``id``
    (the name it is created under), its ``labels`` (its run, tenant and agent: what the reaper
    reads) and the ``snapshot`` it can be made again from, when there is one. Never a secret
    or a host path: it travels in the run's checkpoint."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: str
    id: str
    labels: dict[str, str] = Field(default_factory=dict)
    snapshot: str | None = None


class ExecResult(BaseModel):
    """What a command did: its ``exit_code`` and what it wrote to ``stdout`` and ``stderr``
    (a provider may keep only their end)."""

    model_config = ConfigDict(frozen=True)

    exit_code: int
    stdout: str = ""
    stderr: str = ""


class SandboxLost(ToolError):
    """The sandbox a run works in is gone — deleted, expired, or kept where this process cannot
    reach — and cannot be made again: what earlier calls did in it is lost with it."""

    code = "SANDBOX_LOST"


class Sandbox(Protocol):
    """One sandbox, attached: what its tools call."""

    @property
    def ref(self) -> SandboxRef: ...

    async def exec(
        self,
        command: str,
        *,
        timeout: float | None = None,  # noqa: ASYNC109 - the command's, enforced by the provider
        env: Mapping[str, str] | None = None,
    ) -> ExecResult:
        """Run ``command`` with ``sh -c`` in the working directory, ``env`` added to the
        environment; past ``timeout`` seconds it is killed (``TimeoutError``)."""
        ...

    async def read(self, path: str) -> bytes:
        """A file's content (``ToolError`` when there is none at ``path``)."""
        ...

    async def write(self, path: str, data: bytes) -> None:
        """Write a file, its directories made as needed."""
        ...


class SandboxProvider(Protocol):
    """Where sandboxes come from (``name``: what a :class:`SandboxRef` says)."""

    @property
    def name(self) -> str: ...

    async def create(self, ref: SandboxRef, spec: SandboxSpec) -> Sandbox: ...

    async def attach(self, ref: SandboxRef) -> Sandbox: ...

    async def delete(self, ref: SandboxRef) -> None: ...

    async def labelled(self, labels: Mapping[str, str]) -> list[SandboxRef]:
        """The sandboxes that have all ``labels``, each with its :data:`PREFIX` labels."""
        ...


@runtime_checkable
class SupportsPause(Protocol):
    """A provider that can stop a sandbox's processes while its run waits (``attach`` resumes
    it)."""

    async def pause(self, ref: SandboxRef) -> None: ...


@runtime_checkable
class SupportsSnapshot(Protocol):
    """A provider that can keep a sandbox's filesystem: the snapshot's id, which ``create``
    makes the sandbox again from (``SandboxRef.snapshot``) and ``delete`` removes."""

    async def snapshot(self, ref: SandboxRef) -> str: ...
