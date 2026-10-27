"""Sandboxes: ``sandbox()``, a place where the model runs commands and changes files away from
the host — one entry, given where each engine takes it:

    ReAct("You analyse data.", model, backend=sandbox())        # ReAct: its backend
    create_deep_agent(model, **sandbox().deep_agent())          # Deep Agents: backend=
    await agent.tools.use(sandbox().tools())                    # a function, a LangGraph graph

As a backend (``ReAct``, Deep Agents: ``trellis.harness.sandbox.backend``) Deep Agents' file
tools and its ``execute`` work in the run's sandbox. As tools (:meth:`SandboxSource.tools`) the
model gets three harness tools: :data:`EXEC` (a shell command: its exit code, stdout and
stderr), :data:`READ` (a file, as text) and :data:`WRITE` (a file). Each call goes through the
bridge like any tool's: governed (``sandbox_exec`` and ``sandbox_write`` write,
``sandbox_read`` reads; the catalog may say more), journaled (a resumed run reads the same
output and runs nothing again), recorded and redacted, and bounded by its ``call_timeout`` — a
command out of time is killed, and its effect is unknown to the model, as any write's.

The sandbox lives as long as its run, with nothing to call:

* **made** at the run's first sandbox call (once, however many calls come at once), named and
  labelled after the run (:func:`ref_of`): ``create`` adopts a sandbox of that name, so the
  attempt after a crash works in the one its predecessor made;
* **recorded** in the run's journal, and saved as its progress, as soon as it exists — before
  any call uses it. Every later attempt (after a pause, after a crash, on another worker)
  **attaches** to it and never replaces it blindly: one that is gone is made again from the
  snapshot taken when the run last paused, if no call has changed it since; otherwise every
  call is refused, saying the sandbox is lost (:class:`SandboxLost`);
* **snapshotted, then paused**, as far as its provider can, when the run pauses for a person
  (:func:`paused`);
* **deleted** when the run ends, whichever way (:func:`ended`) — except a queued run that failed
  with an error agent-runs runs it again after, whose next attempt goes on in it. A sandbox
  whose run ended without deleting it (its process died; agent-runs ended it) is deleted by
  :func:`reap`, which runs in the background when a process makes a sandbox for a tenant (at
  most every :data:`REAP_SECONDS`).

The provider is the one given (``sandbox(provider)``), else Docker (:class:`DockerSandbox`, of
``sandbox(image=)``): the daemon ``DOCKER_HOST`` names.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Final

from trellis.contracts import ToolError, ToolSpec, stable_id
from trellis.harness.bounded import LRU
from trellis.harness.runtime import Runtime, current
from trellis.harness.sandbox.base import (
    AGENT,
    RUN,
    TENANT,
    ExecResult,
    Sandbox,
    SandboxLost,
    SandboxProvider,
    SandboxRef,
    SandboxSpec,
    SupportsPause,
    SupportsSnapshot,
)
from trellis.harness.sandbox.docker import DockerSandbox
from trellis.harness.tools.base import REMOTE_TIMEOUT_SECONDS, Tool

if TYPE_CHECKING:
    from trellis.harness.agent import Agent
    from trellis.harness.runs import RunStore

log = logging.getLogger("trellis.sandbox")

#: The tools.
EXEC: Final = "sandbox_exec"
READ: Final = "sandbox_read"
WRITE: Final = "sandbox_write"
#: The ``CUSTOM`` event a run's sandbox is reported with (``created``, ``attached``,
#: ``restored``, ``paused``), and the code of its warnings.
EVENT: Final = "sandbox"
#: The most of a file ``sandbox_read`` returns: a larger one is read in parts with a command.
READ_BYTES: Final = 256 * 1024
#: The longest sandbox name (the strictest provider's); a longer one is the run id's digest.
NAME_CHARS: Final = 63
#: How often a process deletes, in the background, a tenant's sandboxes whose runs ended
#: without deleting them (the first time it makes one, then at most this often).
REAP_SECONDS: Final = 600.0


class SandboxSource:
    """A sandbox of the run's own (what :func:`sandbox` returns), given where each engine takes
    it: ``ReAct(backend=...)`` as it is, :meth:`deep_agent` for a ``create_deep_agent`` of your
    own, :meth:`tools` for an engine with no sandbox of its own (a function, a LangGraph
    graph). Every way works in the same per-run sandbox."""

    def __init__(self, provider: SandboxProvider | None, spec: SandboxSpec, timeout: float) -> None:
        self.provider = provider
        self.spec = spec
        #: the most one call (a command) may take, in seconds
        self.call_timeout = timeout
        path = {"path": "the file's path"}
        self._tools = [
            Tool(
                _spec(
                    EXEC,
                    "Run a shell command in your sandbox (sh -c, in its working directory): "
                    "its exit code, stdout and stderr.",
                    {"command": "the command"},
                    side_effects="write",
                ),
                self._exec,
                timeout=timeout,
            ),
            Tool(
                _spec(READ, "Read a text file of your sandbox.", path, side_effects="read"),
                self._read,
                timeout=timeout,
            ),
            Tool(
                _spec(
                    WRITE,
                    "Write a text file in your sandbox (its directories are made).",
                    {**path, "content": "what the file holds"},
                    side_effects="write",
                    idempotent=True,  # the same content at the same path has one effect
                ),
                self._write,
                timeout=timeout,
            ),
        ]
        self._as_tools = SandboxTools(self)
        #: when this process last reaped each tenant's sandboxes (``time.monotonic``)
        self._reaped: LRU[str, float] = LRU()

    def tools(self) -> SandboxTools:
        """The sandbox as three harness tools (:data:`EXEC`, :data:`READ`, :data:`WRITE`), a
        tool source: ``agent.tools.use(sandbox().tools())``, for an engine with no sandbox of
        its own. The same object every time."""
        return self._as_tools

    def deep_agent(self) -> dict[str, Any]:
        """``create_deep_agent``'s arguments for this sandbox: its ``backend`` (Deep Agents'
        ``BaseSandbox`` on the run's sandbox: its file tools and ``execute`` work in it) —
        ``create_deep_agent(model, **sandbox().deep_agent())``."""
        from trellis.harness.sandbox.backend import SandboxBackend  # noqa: PLC0415 - Deep Agents

        return {"backend": SandboxBackend(self, call_timeout=self.call_timeout)}

    def provider_for(self) -> SandboxProvider:
        """The provider given: a ``sandbox()`` without one makes Docker containers (the daemon
        ``DOCKER_HOST`` names, else the docker context, else the local socket), of the spec's
        image."""
        if self.provider is None:
            self.provider = DockerSandbox(self.spec.image)
        return self.provider

    async def opened(self, runtime: Runtime, *, changes: bool = False) -> Sandbox:
        """The run's sandbox: the one this attempt attached, else the one the journal names,
        else a new one. ``changes``: the call changes the sandbox, so the snapshot of the run's
        last pause no longer makes it again (forgotten, and saved so, before the call)."""
        journal = runtime.replay.journal
        async with runtime.replay.exclusive(EVENT):
            if runtime.sandbox is None:
                runtime.sandbox = await self._opened(runtime)
            if changes and journal.sandbox is not None and journal.sandbox.pop("snapshot", None):
                await runtime.progress(now=True)
            return runtime.sandbox

    async def _opened(self, runtime: Runtime) -> Sandbox:
        provider = self.provider_for()
        journal = runtime.replay.journal
        if journal.sandbox is None:
            await self._reaping(provider, runtime)
            box = await provider.create(ref_of(runtime, provider.name), self.spec)
            journal.sandbox = _recorded(box.ref)
            await runtime.progress(now=True)  # named before any call uses it
            runtime.events.custom(EVENT, action="created", sandbox=box.ref.id)
            return box
        ref = SandboxRef.model_validate(journal.sandbox)
        if ref.provider != provider.name:
            raise SandboxLost(
                f"the run's sandbox {ref.id} is a {ref.provider} sandbox, and its tools make "
                f"{provider.name} sandboxes now",
                source="tools",
            )
        try:
            box, action = await provider.attach(ref), "attached"
        except SandboxLost:
            if ref.snapshot is None:
                raise
            box, action = await provider.create(ref, self.spec), "restored"
        runtime.events.custom(EVENT, action=action, sandbox=ref.id)
        return box

    async def _reaping(self, provider: SandboxProvider, runtime: Runtime) -> None:
        """At most every :data:`REAP_SECONDS` per tenant, in the background: the tenant's
        sandboxes whose runs ended without deleting them, deleted."""
        tenant, now = runtime.tenant, time.monotonic()
        last = self._reaped.get(tenant)
        if last is not None and now - last < REAP_SECONDS:
            return
        self._reaped[tenant] = now
        harness = runtime.agent.harness

        async def work() -> None:
            await reap(provider, harness.runs, tenant=tenant)

        await harness.writes.submit("sandbox.reap", work, events=runtime.events)

    async def _exec(self, args: dict[str, Any]) -> dict[str, Any]:
        runtime = _running()
        box = await self.opened(runtime, changes=True)
        return (await box.exec(str(args["command"]), timeout=runtime.remaining())).model_dump()

    async def _read(self, args: dict[str, Any]) -> str:
        runtime = _running()
        path = str(args["path"])
        data = await (await self.opened(runtime)).read(path)
        if len(data) > READ_BYTES:
            raise ToolError(
                f"{path} is {len(data)} bytes, more than the {READ_BYTES} read at once: read "
                f"a part of it with {EXEC} (head -c, tail -c, sed -n)",
                source="tools",
            )
        return data.decode("utf-8", errors="replace")

    async def _write(self, args: dict[str, Any]) -> str:
        runtime = _running()
        path, data = str(args["path"]), str(args["content"]).encode()
        await (await self.opened(runtime, changes=True)).write(path, data)
        return f"wrote {len(data)} bytes to {path}"


class SandboxTools:
    """A sandbox's three tools, as a tool source (:meth:`SandboxSource.tools`)."""

    def __init__(self, source: SandboxSource) -> None:
        self.source = source

    async def resolve(self) -> list[Tool]:
        return list(self.source._tools)


def sandbox(
    provider: SandboxProvider | None = None,
    *,
    image: str | None = None,
    spec: SandboxSpec | None = None,
    call_timeout: float = REMOTE_TIMEOUT_SECONDS,
) -> SandboxSource:
    """A sandbox of the run's own for an agent — granting code execution is this explicit entry:
    ``ReAct(backend=sandbox())``, ``create_deep_agent(**sandbox().deep_agent())``,
    ``agent.tools.use(sandbox().tools())``. ``provider`` makes the sandboxes (else Docker: the
    daemon ``DOCKER_HOST`` names); ``image`` is what each is made of (else the provider's), and
    ``spec`` the rest — limits, network (none by default), files; ``call_timeout`` is the most one
    call may take, in seconds (a command past it is killed)."""
    given = spec or SandboxSpec()
    if image is not None:
        given = given.model_copy(update={"image": image})
    return SandboxSource(provider, given, call_timeout)


def ref_of(runtime: Runtime, provider: str) -> SandboxRef:
    """The sandbox of ``runtime``'s run: named after it — the same name in every attempt, which
    makes ``create`` idempotent — and labelled with its run, tenant and agent."""
    run_id = runtime.run_id
    name = f"trellis-{run_id}"
    if len(name) > NAME_CHARS or not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id):
        name = f"trellis-{stable_id(run_id)}"
    labels = {RUN: run_id, TENANT: runtime.tenant, AGENT: runtime.agent_id}
    return SandboxRef(provider=provider, id=name, labels=labels)


async def paused(runtime: Runtime) -> None:
    """The run pauses for a person: the sandbox this attempt worked in snapshotted (the
    snapshot named in its reference, which the pause saves) and paused, as far as its provider
    can. A failure is a warning: the run pauses all the same."""
    source = _source(runtime.agent, runtime.added)
    if runtime.sandbox is None or source is None:
        return
    journal = runtime.replay.journal
    ref = SandboxRef.model_validate(journal.sandbox)
    try:
        provider = source.provider_for()
        if isinstance(provider, SupportsSnapshot):
            snapshot = await provider.snapshot(ref)
            journal.sandbox = _recorded(ref.model_copy(update={"snapshot": snapshot}))
        if isinstance(provider, SupportsPause):
            await provider.pause(ref)
    except Exception as exc:
        runtime.events.warning(EVENT, f"the sandbox {ref.id} was not snapshotted and paused: {exc}")
        return
    runtime.events.custom(EVENT, action="paused", sandbox=ref.id)


async def ended(
    agent: Agent, recorded: dict[str, Any] | None, run_id: str, added: Sequence[Any] = ()
) -> None:
    """The run ``run_id`` of ``agent`` ended: the sandbox its journal names (``recorded``:
    ``Journal.sandbox``) deleted, by the agent's sandbox source or one its code added
    (``added``). A failure is logged — the run's end stands, and :func:`reap` deletes the
    sandbox later."""
    if recorded is None:
        return
    ref = SandboxRef.model_validate(recorded)
    source = _source(agent, added)
    if source is None:  # not among the agent's own sources (a handoff's specialist's)
        log.warning("the sandbox %s of run %s is left to the reaper", ref.id, run_id)
        return
    try:
        await source.provider_for().delete(ref)
    except Exception as exc:
        log.warning("the sandbox %s of run %s was not deleted: %s", ref.id, run_id, exc)
        return
    log.info("the sandbox %s of run %s was deleted", ref.id, run_id)


async def reap(provider: SandboxProvider, runs: RunStore, *, tenant: str) -> list[SandboxRef]:
    """Delete ``tenant``'s sandboxes whose runs have ended — what a run left when its process
    died before deleting its own — and return them. A sandbox whose run the store does not
    know (kept in a process that is gone, without ``RUNS_URL``) is left alone: no store says
    it ended."""
    deleted: list[SandboxRef] = []
    for ref in await provider.labelled({TENANT: tenant}):
        record = await runs.get(ref.labels[RUN], tenant=tenant)
        if record is not None and record.final:
            await provider.delete(ref)
            deleted.append(ref)
    return deleted


def _spec(
    name: str,
    description: str,
    fields: dict[str, str],
    *,
    side_effects: str,
    idempotent: bool = False,
) -> ToolSpec:
    """A sandbox tool's spec: its string arguments, every one required."""
    properties = {key: {"type": "string", "description": text} for key, text in fields.items()}
    schema = {"type": "object", "properties": properties, "required": list(fields)}
    return ToolSpec(
        name=name,
        description=description,
        input_schema=schema,
        side_effects=side_effects,
        idempotent=idempotent,
    )


def _source(agent: Agent, added: Sequence[Any] = ()) -> SandboxSource | None:
    """The agent's sandbox, else one the run's code (or its backend) added (one at most: two
    would give it two tools of one name)."""
    given = (s.source if isinstance(s, SandboxTools) else s for s in [*agent.sources, *added])
    return next((s for s in given if isinstance(s, SandboxSource)), None)


def _running() -> Runtime:
    runtime = current()
    assert runtime is not None  # the bridge runs a tool inside a run
    return runtime


def _recorded(ref: SandboxRef) -> dict[str, Any]:
    return ref.model_dump(mode="json", exclude_none=True)


__all__ = [
    "DockerSandbox",
    "ExecResult",
    "Sandbox",
    "SandboxLost",
    "SandboxProvider",
    "SandboxRef",
    "SandboxSource",
    "SandboxSpec",
    "SandboxTools",
    "SupportsPause",
    "SupportsSnapshot",
    "reap",
    "sandbox",
]
