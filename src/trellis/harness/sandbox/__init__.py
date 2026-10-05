"""Sandboxes for any framework: ``sandbox()``, a place where the model runs commands and changes
files away from the host.

    agent = h.wrap(target, id="analyst", tools=[sandbox()])  # SANDBOX=docker
    tools = await h.tools(sandbox(DockerSandbox(), SandboxSpec(cpu=1)), framework="langgraph")

The source gives the model three harness tools: :data:`EXEC` (a shell command: its exit code,
stdout and stderr), :data:`READ` (a file, as text) and :data:`WRITE` (a file). Each call goes
through the bridge like any tool's: governed (``sandbox_exec`` and ``sandbox_write`` write,
``sandbox_read`` reads; the catalog may say more), journaled (a resumed run reads the same
output and runs nothing again), recorded and redacted, and bounded by its ``timeout`` — a
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

The provider is the one given (``sandbox(provider)``), else the deployment's
(:func:`configured`: ``SANDBOX=docker``, its image ``SANDBOX_IMAGE``).
"""

from __future__ import annotations

import logging
import re
import time
from typing import TYPE_CHECKING, Any, Final

from trellis.contracts import ConfigurationError, ToolError, ToolSpec, stable_id
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
from trellis.harness.settings import Settings
from trellis.harness.tools.base import REMOTE_TIMEOUT_SECONDS, Tool

if TYPE_CHECKING:
    from trellis.harness.agent import Agent
    from trellis.harness.journal import Journal
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
    """A sandbox's tools, as a tool source (``tools=[...]``, ``h.tools(...)``): what
    :func:`sandbox` returns."""

    def __init__(self, provider: SandboxProvider | None, spec: SandboxSpec, timeout: float) -> None:
        self.provider = provider
        self.spec = spec
        path = {"path": "the file's path"}
        self.tools = [
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
        #: when this process last reaped each tenant's sandboxes (``time.monotonic``)
        self._reaped: dict[str, float] = {}

    async def resolve(self) -> list[Tool]:
        return list(self.tools)

    def provider_for(self, settings: Settings) -> SandboxProvider:
        """The provider given, else the deployment's (``settings``: the harness's)."""
        return self.provider or configured(settings)

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
        provider = self.provider_for(runtime.agent.harness.settings)
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


def sandbox(
    provider: SandboxProvider | None = None,
    spec: SandboxSpec | None = None,
    *,
    timeout: float = REMOTE_TIMEOUT_SECONDS,
) -> SandboxSource:
    """A sandbox's tools for an agent: ``sandbox_exec``, ``sandbox_read``, ``sandbox_write``,
    in the sandbox of the run that calls them. ``provider`` makes the sandboxes (else
    ``SANDBOX``: :func:`configured`); ``spec`` says what each is made of (else the provider's
    image, its limits, and no network); ``timeout`` is the most one call may take, in seconds
    (a command past it is killed)."""
    return SandboxSource(provider, spec or SandboxSpec(), timeout)


def configured(settings: Settings) -> SandboxProvider:
    """The deployment's provider: ``SANDBOX=docker`` is :class:`DockerSandbox`, of
    ``SANDBOX_IMAGE`` when it is set."""
    if settings.sandbox is None:
        raise ConfigurationError(
            "sandbox() was given no provider and the deployment has none: set SANDBOX=docker, "
            "or pass one (sandbox(DockerSandbox()))"
        )
    return DockerSandbox(settings.sandbox_image)


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
    source = _source(runtime.agent)
    if runtime.sandbox is None or source is None:
        return
    journal = runtime.replay.journal
    ref = SandboxRef.model_validate(journal.sandbox)
    try:
        provider = source.provider_for(runtime.agent.harness.settings)
        if isinstance(provider, SupportsSnapshot):
            snapshot = await provider.snapshot(ref)
            journal.sandbox = _recorded(ref.model_copy(update={"snapshot": snapshot}))
        if isinstance(provider, SupportsPause):
            await provider.pause(ref)
    except Exception as exc:
        runtime.events.warning(EVENT, f"the sandbox {ref.id} was not snapshotted and paused: {exc}")
        return
    runtime.events.custom(EVENT, action="paused", sandbox=ref.id)


async def ended(agent: Agent, journal: Journal, run_id: str) -> None:
    """The run ``run_id`` of ``agent`` ended (``journal``: its own): its sandbox deleted. A
    failure is logged — the run's end stands, and :func:`reap` deletes the sandbox later."""
    if journal.sandbox is None:
        return
    ref = SandboxRef.model_validate(journal.sandbox)
    source = _source(agent)
    if source is None:  # not among the agent's own sources (a handoff's specialist's)
        log.warning("the sandbox %s of run %s is left to the reaper", ref.id, run_id)
        return
    try:
        await source.provider_for(agent.harness.settings).delete(ref)
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


def _source(agent: Agent) -> SandboxSource | None:
    """The agent's sandbox source (one at most: two would give it two tools of one name)."""
    return next((s for s in agent.sources if isinstance(s, SandboxSource)), None)


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
    "SupportsPause",
    "SupportsSnapshot",
    "configured",
    "reap",
    "sandbox",
]
