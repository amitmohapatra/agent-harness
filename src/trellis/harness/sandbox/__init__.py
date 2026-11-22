"""Sandboxes: ``sandbox()``, a place where the model runs commands and changes files away from
the host — one entry, given where each engine takes it:

    ReAct("You analyse data.", model, backend=sandbox())        # ReAct: its backend
    create_deep_agent(model, **sandbox().deep_agent())          # Deep Agents: backend=
    await agent.tools.use(sandbox().tools())                    # a function, a LangGraph graph

As a backend (``ReAct``, Deep Agents: ``trellis.harness.sandbox.backend``) Deep Agents' file
tools and its ``execute`` work in the run's sandbox. As tools (:meth:`SandboxSource.tools`) the
model gets three harness tools: :data:`EXEC` (a shell command: its exit code, stdout and
stderr), :data:`READ` (a file, as text) and :data:`WRITE` (a file). Each call goes through the
bridge like any tool's — a failed command's result adds a ``hint`` when the sandbox knows why
(:func:`~trellis.harness.sandbox.base.hint`: the memory limit it went past, a module it cannot
install) — governed (``sandbox_exec`` and ``sandbox_write`` write,
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

**A shell by default**: an agent the harness wraps gets one with nothing to write
(:func:`code_for`, at ``h.wrap``), as Claude Code and Kimi Code have their own — ``ReAct`` as
Deep Agents' ``execute`` on its backend (``backend.CodeBackend``), every other agent as
``sandbox_exec`` (:data:`CODE`: one tool, a few words) — in a working folder of the run's own on
this machine, with no isolation (:func:`default_sandbox`: ``local.LocalFolders``; ``sandbox()``
given to the agent is the isolated one). The folder is made at the run's first command, as any
sandbox. None is given ``without=["code"]`` (the agent's own switch), where the deployment
says ``TRELLIS_HOST_EXECUTION=deny``, or to an
agent that brings its own (``Adapter.code``: Claude Code's and Kimi Code's shells, Deep Agents'
or LangChain's or OpenAI's own shell, a sandbox backend, ``sandbox()`` among its tools).
"""

from __future__ import annotations

import asyncio
import dataclasses
import functools
import itertools
import logging
import os
import posixpath
import socket
import time
import uuid
import weakref
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from trellis.contracts import ConfigurationError, ToolError, ToolSpec
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
    audited,
    hint,
)
from trellis.harness.sandbox.docker import DockerSandbox, docker_host
from trellis.harness.sandbox.isolation import (
    HostSandbox,
    Limits,
    host_layer,
    isolated_end,
    isolated_run,
    layer_of,
    name_of,
)
from trellis.harness.sandbox.shared import SharedDockerSandbox
from trellis.harness.tools import query
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
#: When this process began to delete each sandbox (the most recent; :data:`_STEPS`): a run's
#: end and the reaper, which may come to one sandbox at once, delete it once.
_DELETING: LRU[tuple[str, str], int] = LRU()
_STEPS = itertools.count()


def _claimed(ref: SandboxRef, *, since: int | None = None) -> bool:
    """Whether this caller is the one to delete ``ref`` (and it is marked so, with no
    ``await`` between the look and the mark): a run's end (``since`` ``None``), unless the
    reaper began to — a run's sandbox is made once and ended once; the reaper, unless
    someone began to since its round began (``since``: its listing may be older than that
    delete; a sandbox deleted before it and made again since is its to delete)."""
    key = (ref.provider, ref.id)
    begun = _DELETING.get(key)
    if begun is not None and (since is None or begun >= since):
        return False
    _DELETING[key] = next(_STEPS)
    return True


#: What the environment card is read with, in a sandbox made for it alone: the Python version,
#: then each installed distribution as ``name==version`` (the standard library's
#: ``importlib.metadata``; pip's own tools left out).
PROBE: Final = (
    'python3 -c "import sys, importlib.metadata as m; '
    "print(sys.version.split()[0]); "
    "print(' '.join(sorted({d.metadata['Name'].lower() + '==' + d.version "
    "for d in m.distributions() if d.metadata['Name'] "
    "and d.metadata['Name'].lower() not in ('pip', 'setuptools', 'wheel')})))\""
)
#: The most a probe may take, in seconds (its sandbox made, the script run).
PROBE_SECONDS: Final = 60.0
#: The most a ``sandbox()`` given no provider waits for the Docker daemon's answer.
PING_SECONDS: Final = 1.0
#: The most distributions the card names; the rest are counted.
CARD_PACKAGES: Final = 80
#: The label of a sandbox made only to read the environment card.
PROBE_LABEL: Final = "trellis.probe"
#: Where code in the sandbox puts the files the user gets: each file directly in it is kept in
#: the file store when the run succeeds (:func:`delivered`), before the sandbox is deleted.
OUTPUTS: Final = "outputs"
#: Each output file's size and name, one per line (POSIX sh: any image's shell).
LIST_OUTPUTS: Final = (
    f'for f in {OUTPUTS}/* {OUTPUTS}/.[!.]*; do [ -f "$f" ] && '
    'printf "%s %s\\n" "$(wc -c < "$f")" "${f##*/}"; done; true'
)
#: The most files of the working directory one look lists (a tree of dependencies is cut off
#: here: what is new past it is not named).
LISTED_MOST: Final = 5_000
#: The working directory, each output file's size and name, an empty line, then the path of
#: each file under the working directory, at any depth, that is not directly in the kept
#: folder (hidden folders left out): what the model's command or file is told with its
#: result is read from this (:meth:`SandboxSource.outputs_said`). Only the files directly in
#: the working directory were listed, so one a command wrote a folder down — ``reports/``,
#: or a folder inside the kept one — was neither said to be the user's nor not.
OUTPUTS_NOW: Final = (
    f'printf "%s\\n" "$PWD"; {LIST_OUTPUTS}; printf "\\n"; '
    'find . -type d -name ".?*" -prune -o -type f '
    f'\\( ! -path "./{OUTPUTS}/*" -o -path "./{OUTPUTS}/*/*" \\) -print 2>/dev/null '
    f'| sed "s|^\\./||" | head -n {LISTED_MOST}; true'
)
#: What a command or a file that names :data:`OUTPUTS` is told: the files the user gets so
#: far, or that there is none yet — with the folder's whole path, the one place a file for
#: the user is written.
OUTPUTS_HOLD: Final = "In {folder}/ for the user: {files}."
OUTPUTS_EMPTY: Final = (
    "No file is in {folder}/ yet: only a file written directly there is given to the user."
)
#: What a command that left a new file anywhere but directly in that folder is told, once per
#: file; and a file written by a path that is not directly in that folder.
OUTPUTS_BESIDE: Final = "Not in {folder}/, so not given to the user: {files}."
#: The most new files beside the folder named at once; the rest are counted.
BESIDE_NAMED: Final = 20
#: The largest output file kept; a larger one is left, said in a warning.
OUTPUT_MAX_BYTES: Final = 50 * 1024 * 1024
#: What the command tool's description says about outputs.
DELIVERS: Final = (
    f"Files you write in {OUTPUTS}/ (its working directory's) are given to the user when you "
    "finish: only those directly in it. A file a command leaves elsewhere under the working "
    "directory is named with its result; one written outside the working directory is not "
    "seen, and is not given to the user."
)
#: What the command tool's description says of the run's data: where code finds it, and what
#: is the query's — facts the model chooses by.
STARTS: Final = (
    "A result this run saved, and a query_result's result of more than one row, is a file "
    "here at the path its note names: code starts from that file, for what a query cannot "
    "make (a chart, a formatted workbook, a script, a model of its own). A count, a sum, a "
    "filter, a ranking or a join over saved rows is query_result's: exact over every row."
)
#: What an agent working in a folder of its own on this machine (Claude Code's, Kimi Code's:
#: ``Adapter.workdir``) is told with its context: the same ``outputs/``, kept the same way —
#: for a file meant for the user that they named no place for; one they placed goes where
#: they said (its own ``Write`` as the SDK does it without the harness).
DELIVERS_HERE: Final = (
    f"Files you write in {OUTPUTS}/ of your working directory are given to the user when you "
    "finish. Write a file the user names, or gives a path or folder for, exactly there (a bare "
    f"name is in your working directory); use {OUTPUTS}/ for a file you make to give the user "
    "that they named no place for."
)


#: Which of the files a command writes the user is given: the one thing a shell the harness
#: turns on must say that its own description cannot (a library's ``execute``, its words kept:
#: ``Adapter.give_shell``).
DELIVERED: Final = f"Only files you write directly in {OUTPUTS}/ are given to the user."
#: What the shell the harness gives an agent by itself says (:func:`code_for`:
#: ``sandbox_exec``, ``ReAct``'s ``execute``): what the model needs, in as few words as the
#: other tools take — a kept result's note names its file there (``tools.results.CODE``), and a
#: failed command's result says what went wrong (``base.hint``).
CODE: Final = (
    "Run a shell command in your working folder (sh -c; python3 too): its exit code and output. "
    f"{DELIVERED}"
)
#: The sandbox the harness gives an agent by itself, once made (:func:`default_sandbox`;
#: ``None``: none, as the test suite's agents have it).
_DEFAULT: dict[str, SandboxSource | None] = {}


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
                    f"its exit code, stdout and stderr. {DELIVERS} {STARTS}",
                    {"command": "the command"},
                    side_effects="write",
                ),
                self._exec,
                timeout=timeout,
                confined=True,
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
                confined=True,
            ),
        ]
        self._as_tools = SandboxTools(self)
        #: the shell the harness gives an agent by itself (:func:`code_for`): the command
        #: alone, in a few words, off in a run ``without=["code"]``
        self._code = Tool(
            _spec(EXEC, CODE, {"command": "the command"}, side_effects="write"),
            self._exec,
            feature="code",
            timeout=timeout,
            confined=True,
        )
        self._as_code = SandboxTools(self, default=True)
        #: when this process last reaped each tenant's sandboxes (``time.monotonic``)
        self._reaped: LRU[str, float] = LRU()
        #: the kept results each attempt has written into its sandbox (:meth:`with_results`)
        self._written: weakref.WeakKeyDictionary[Runtime, set[str]] = weakref.WeakKeyDictionary()
        #: each attempt's working directory and the files seen directly in it (:meth:`seen`)
        self._beside: weakref.WeakKeyDictionary[Runtime, tuple[str, set[str]]] = (
            weakref.WeakKeyDictionary()
        )
        #: the environment card, once read (:meth:`card`); ``""`` when none could be
        self._card: str | None = None
        self._probing = asyncio.Lock()

    async def card(self) -> str:
        """What the sandbox's image holds, for the model to write code that runs there the
        first time: its Python version and installed distributions, and — with no network —
        that nothing else can be installed. Read once per process, in a sandbox made for it
        alone and deleted at once; ``""`` when it cannot be read (no daemon: said once in the
        log; an image without Python: said in the card)."""
        if _host_denied(self.provider_for(), current()):
            return ""  # no command on this machine: the deployment switched them off
        async with self._probing:
            if self._card is None:
                self._card = await self._probed()
            return self._card

    async def _probed(self) -> str:
        provider = self.provider_for()
        ref = SandboxRef(
            provider=provider.name,
            id=f"trellis-probe-{uuid.uuid4().hex[:16]}",
            labels={PROBE_LABEL: "1"},
        )
        try:
            async with asyncio.timeout(PROBE_SECONDS):
                box = await provider.create(ref, self.spec.model_copy(update={"files": {}}))
                try:
                    found = await box.exec(PROBE, timeout=PROBE_SECONDS)
                finally:
                    await provider.delete(ref)
        except Exception as exc:
            log.warning("the sandbox's environment card was not read: %s", exc)
            return ""
        return card_of(found, offline=self.spec.network == "none")

    def tools(self) -> SandboxTools:
        """The sandbox as three harness tools (:data:`EXEC`, :data:`READ`, :data:`WRITE`), a
        tool source: ``agent.tools.use(sandbox().tools())``, for an engine with no sandbox of
        its own. The same object every time."""
        return self._as_tools

    def code(self) -> SandboxTools:
        """The code tool the harness gives an agent by itself, a tool source: ``sandbox_exec``
        (:data:`CODE`), with no environment card — nothing is made until a run calls it."""
        return self._as_code

    def deep_agent(self) -> dict[str, Any]:
        """``create_deep_agent``'s arguments for this sandbox: its ``backend`` (Deep Agents'
        ``BaseSandbox`` on the run's sandbox: its file tools and ``execute`` work in it) —
        ``create_deep_agent(model, **sandbox().deep_agent())``."""
        from trellis.harness.sandbox.backend import SandboxBackend  # noqa: PLC0415 - Deep Agents

        return {"backend": SandboxBackend(self, call_timeout=self.call_timeout)}

    def provider_for(self) -> SandboxProvider:
        """The provider given, else :func:`default_provider`'s: the tenant's shared container
        on Docker when its daemon answers, else this machine's isolation layer, else no shell
        (each command refused with the reason). Chosen once."""
        if self.provider is None:
            self.provider = default_provider(self.spec.image)
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
        if _host_denied(provider, runtime):
            raise ConfigurationError(HOST_DENIED)
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

    async def with_results(self, runtime: Runtime, box: Sandbox) -> None:
        """The results the run keeps (``tools.results``: a large one, or one cut to fit the
        window; its conversation's earlier turns' too, ``tools.query.saved``) written into its
        sandbox at the path the model was given, before a command runs or a file is read
        there — so code counts, sums, filters and ranks every row of one, instead of the model
        paging it into its context. Each once per attempt (a later attempt's sandbox may be one
        made again from a snapshot)."""
        written = self._written.setdefault(runtime, set())
        kept = query.saved(runtime)
        for path in [p for p in kept if p not in written]:
            await box.write(path, kept[path].encode())
            written.add(path)

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
        await self.with_results(runtime, box)
        command = str(args["command"])
        await self.seen(runtime, box)
        result = await audited(box, command, timeout=runtime.remaining())
        why = hint(result.exit_code, result.stderr, self.spec, result.limit)
        kept = await self.outputs_said(runtime, box, command)
        told = {"hint": why, OUTPUTS: kept}
        return result.model_dump() | {k: v for k, v in told.items() if v is not None}

    #: which of the files a command writes the user is given (:data:`DELIVERED`): what a
    #: framework's own shell the harness turns on is told, where it has no description of
    #: its own to say it in
    says = DELIVERED

    async def command(self, command: str) -> dict[str, Any]:
        """One command of the current run, as ``sandbox_exec`` answers it (its exit code,
        output, hint and what the user gets of what it wrote): what a framework's own shell
        the harness turns on runs (the OpenAI Agents SDK's ``ShellTool``:
        ``tools.convert.openai_local.given_shell``)."""
        return await self._exec({"command": command})

    async def seen(self, runtime: Runtime, box: Sandbox) -> None:
        """Before the model's first command or file of an attempt: the files its working
        directory holds already (the spec's, an earlier attempt's) are not ones it just left
        beside the kept folder."""
        if runtime not in self._beside:
            await self._now(runtime, box)

    async def outputs_said(self, runtime: Runtime, box: Sandbox, named: str) -> str | None:
        """What the model's command, or the file it wrote (``named``: the command, the path),
        is told with its result, so a file that is not the user's is known at once. One that
        names an :data:`OUTPUTS` folder: the files the kept folder holds (each with its size),
        or that it holds none, with the folder's whole path — another ``outputs`` (the root's,
        another folder's) is not it. One that left a new file under the working directory, at
        any depth, that is not directly in the kept folder: that file, by its path, named
        once. ``None`` for neither."""
        folder, kept, new = await self._now(runtime, box)
        said = []
        if OUTPUTS in named:
            said.append((OUTPUTS_HOLD if kept else OUTPUTS_EMPTY).format(folder=folder, files=kept))
        if new:
            more = f" and {len(new) - BESIDE_NAMED} more" if len(new) > BESIDE_NAMED else ""
            files = ", ".join(new[:BESIDE_NAMED]) + more
            said.append(OUTPUTS_BESIDE.format(folder=folder, files=files))
        return " ".join(said) or None

    async def written_said(self, runtime: Runtime, box: Sandbox, path: str) -> str | None:
        """What a file the model wrote by its path is told, whichever path: what the kept
        folder holds when the file is directly in it, else that the file is not the user's —
        by its path, so nothing is left to infer. (Told here, a file beside the kept folder
        is not one its next command left.)"""
        where, known = self._beside[runtime]
        written = posixpath.normpath(posixpath.join(where, path))
        folder = f"{where.rstrip('/')}/{OUTPUTS}"
        if posixpath.dirname(written) != folder:
            known.add(posixpath.relpath(written, where))
            return OUTPUTS_BESIDE.format(folder=folder, files=path)
        _, kept, _ = await self._now(runtime, box)
        return OUTPUTS_HOLD.format(folder=folder, files=kept)

    async def _now(self, runtime: Runtime, box: Sandbox) -> tuple[str, str, list[str]]:
        """The kept folder's whole path, the files it holds (each with its size), and the files
        under the working directory this attempt had not seen (none at its first look:
        :meth:`seen`) — one command (:data:`OUTPUTS_NOW`)."""
        listed = await box.exec(OUTPUTS_NOW, timeout=self.call_timeout)
        where, _, rest = listed.stdout.partition("\n")
        kept, _, beside = f"\n{rest}".partition("\n\n")
        found = [line.strip().partition(" ") for line in kept.splitlines()]
        files = ", ".join(f"{name} ({size} bytes)" for size, _, name in found if name)
        names = [name for name in beside.splitlines() if name]
        first = runtime not in self._beside
        known = self._beside.setdefault(runtime, (where, set()))[1]
        new = [] if first else [name for name in names if name not in known]
        known.update(names)
        return f"{where.rstrip('/')}/{OUTPUTS}", files, new

    async def _read(self, args: dict[str, Any]) -> str:
        runtime = _running()
        path = str(args["path"])
        box = await self.opened(runtime)
        await self.with_results(runtime, box)
        data = await box.read(path)
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
        box = await self.opened(runtime, changes=True)
        await self.seen(runtime, box)
        await box.write(path, data)
        wrote = f"wrote {len(data)} bytes to {path}"
        said = await self.written_said(runtime, box, path)
        return f"{wrote}\n{said}" if said else wrote


class SandboxTools:
    """A sandbox's three tools, as a tool source (:meth:`SandboxSource.tools`); ``default``:
    the code tool alone (:meth:`SandboxSource.code`)."""

    def __init__(self, source: SandboxSource, *, default: bool = False) -> None:
        self.source = source
        self.default = default

    async def resolve(self) -> list[Tool]:
        """The three tools, ``sandbox_exec`` described with the environment card
        (:meth:`SandboxSource.card`); the default's one tool as it is."""
        if self.default:
            return [self.source._code]
        run, *rest = self.source._tools
        card = await self.source.card()
        if not card:
            return [run, *rest]
        described = run.spec.model_copy(update={"description": f"{run.spec.description} {card}"})
        return [dataclasses.replace(run, spec=described), *rest]


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


def default_provider(image: str | None = None) -> SandboxProvider:
    """Where a ``sandbox()`` given no provider works: the tenant's shared container
    (:class:`~trellis.harness.sandbox.shared.SharedDockerSandbox`, of ``image``) when the
    Docker daemon ``DOCKER_HOST`` names answers (:func:`docker_answers`); else this machine,
    through the isolation layer (:class:`~trellis.harness.sandbox.isolation.HostSandbox`), when
    it can isolate here; else :class:`NoShell`, its reason said once. The model sees the same
    command tool either way."""
    if docker_answers():
        return SharedDockerSandbox(image)
    layer = host_layer()
    why = layer.cannot_isolate()
    return layer if why is None else NoShell(why)


def default_sandbox() -> SandboxSource | None:
    """The sandbox the harness gives an agent by itself (:func:`code_for`), one per process: a
    working folder of the run's own on this machine, with no isolation
    (``local.LocalFolders``) — as Claude Code's and Kimi Code's own shells run."""
    if "default" not in _DEFAULT:
        from trellis.harness.sandbox.local import LocalFolders  # noqa: PLC0415

        _DEFAULT["default"] = sandbox(LocalFolders())
    return _DEFAULT["default"]


def code_for(agent: Agent) -> tuple[SandboxSource | None, list[SandboxTools]]:
    """The sandbox the harness gives ``agent`` by itself (``h.wrap``: ``Agent.code``) and the
    tool source it adds to the agent's tools for it. The framework's own shell first: where
    the agent runs Deep Agents' ``execute`` there itself (``ReAct``'s ``CodeBackend``:
    ``Adapter.code`` says ``"default"``), or has one the harness can turn on
    (``Adapter.give_shell``: a Deep Agents graph of your own — ``agent.codes`` is then
    ``"default"`` too), no tool is added; any other agent gets ``sandbox_exec``
    (:meth:`SandboxSource.code`). Neither ``without=["code"]``, where the deployment says
    ``deny`` (:func:`_given`), for an agent that brings its own (``Adapter.code`` says
    ``"own"``; a ``sandbox()`` among its tools), nor for one the framework's own rule gives no
    shell (``give_shell`` says ``False``)."""
    if "code" in agent.without or agent.codes == "own" or _source(agent) is not None:
        return None, []
    source = _given(agent.harness.settings.host_execution)
    if source is None:
        return None, []
    if agent.codes == "none":
        turned_on = agent.adapter.give_shell(agent.target, default_files, DELIVERED)
        if turned_on is False:
            return None, []
        if turned_on:
            agent.codes = "default"
    return source, [source.code()] if agent.codes == "none" else []


def _given(host_execution: str) -> SandboxSource | None:
    """:func:`default_sandbox`, unless the deployment switched commands on this machine off
    (``host_execution`` ``deny``)."""
    return None if host_execution == "deny" else default_sandbox()


def offered(runtime: Runtime, specs: list[ToolSpec]) -> list[ToolSpec]:
    """``specs`` — the framework's own tools — less Deep Agents' ``execute`` on the harness's
    code (``Agent.codes`` ``"default"``: ``ReAct``'s) in a run without it (:func:`code_of`)."""
    if runtime.agent.codes != "default" or code_of(runtime) is not None:
        return specs
    return [s for s in specs if s.name != "execute"]


def given_code(host_execution: str) -> list[Tool]:
    """The shell for a graph built by hand (``Harness.harness_tools``): the one
    :func:`default_sandbox` gives, unless the deployment says ``deny``."""
    source = _given(host_execution)
    return [source._code] if source is not None else []


def code_of(runtime: Runtime | None) -> SandboxSource | None:
    """The sandbox the harness gave ``runtime``'s agent by itself (:func:`code_for`), unless the
    run is ``without=["code"]``."""
    if runtime is None or not runtime.uses("code"):
        return None
    return getattr(runtime.agent, "code", None)


def default_files() -> Any:
    """``ReAct``'s files backend when it is given none: ``backend.CodeBackend`` (its files the
    graph's state, its ``execute`` in the working folder the harness gives the run's agent:
    :func:`default_sandbox`), else ``automatic.JournalBackend`` where none is given."""
    from trellis.harness.sandbox.backend import CodeBackend, JournalBackend  # noqa: PLC0415

    return CodeBackend() if default_sandbox() is not None else JournalBackend()


#: What a command in this machine's isolation layer is refused with where the deployment
#: switched host commands off.
HOST_DENIED: Final = (
    "commands on this machine are switched off in this deployment (TRELLIS_HOST_EXECUTION="
    "deny), the isolation layer's too: give the sandbox a Docker daemon, or another provider"
)


def _host_denied(provider: SandboxProvider, runtime: Runtime | None) -> bool:
    """Whether ``provider`` runs commands on this machine (the isolation layer,
    ``HostSandbox``) in a run whose deployment says ``deny``."""
    if not isinstance(provider, HostSandbox) or runtime is None:
        return False
    return runtime.agent.harness.settings.host_execution == "deny"


def docker_answers() -> bool:
    """Whether the Docker daemon (``sandbox.docker.docker_host``: a ``unix://`` socket or a
    ``tcp://`` address) answers its ``/_ping`` within :data:`PING_SECONDS`."""
    where = docker_host()
    try:
        if where.startswith("unix://"):
            held = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            held.settimeout(PING_SECONDS)
            held.connect(where.removeprefix("unix://"))
        elif where.startswith("tcp://"):
            name, _, port = where.removeprefix("tcp://").rstrip("/").partition(":")
            held = socket.create_connection((name, int(port or 2375)), timeout=PING_SECONDS)
        else:
            return False
        with held:
            held.sendall(b"GET /_ping HTTP/1.0\r\nHost: docker\r\n\r\n")
            return b" 200 " in held.recv(64)
    except (OSError, ValueError):
        return False


class NoShell:
    """``sandbox()`` where no Docker daemon answers and this machine cannot isolate a command
    (``why``): each command refused with the reason (a ``ConfigurationError`` the model reads),
    said once in the log; never run unisolated."""

    name = "none"

    def __init__(self, why: str) -> None:
        self.why = why
        self.said = (
            "no shell for this run: no Docker daemon answers, and commands cannot be isolated "
            f"on this machine: {why}"
        )
        _said_once(self.said)

    async def create(self, ref: SandboxRef, spec: SandboxSpec | None = None) -> Sandbox:
        raise ConfigurationError(self.said)

    async def attach(self, ref: SandboxRef) -> Sandbox:
        raise ConfigurationError(self.said)

    async def delete(self, ref: SandboxRef) -> None:
        return None

    async def labelled(self, labels: Mapping[str, str]) -> list[SandboxRef]:
        return []


@functools.lru_cache(maxsize=16)
def _said_once(said: str) -> None:
    log.warning("%s", said)


async def delivered(runtime: Runtime) -> None:
    """The run succeeded: each file written directly in :data:`OUTPUTS` kept in the file store
    for its user (``Runtime.save_file``: named in its journal, announced, served over AG-UI) —
    its sandbox's, before the sandbox is deleted, and that of the folder its framework works in
    on this machine (``Adapter.workdir``: Claude Code's, Kimi Code's), moved out of it so a
    later run does not give it again. A file that cannot be kept (its name, its size past
    :data:`OUTPUT_MAX_BYTES`, a link there: :func:`_written`) is left, said in a warning."""
    box, source = runtime.sandbox, _source(runtime.agent, runtime.added)
    if box is not None and source is not None:
        tool = EXEC if EXEC in runtime.toolbox else "execute"
        listed = await box.exec(LIST_OUTPUTS, timeout=source.call_timeout)
        found = [line.strip().partition(" ") for line in listed.stdout.splitlines()]
        sized = [(int(size), name) for size, _, name in found]
        await _kept(runtime, sized, lambda name: box.read(f"{OUTPUTS}/{name}"), tool)
    agent = runtime.agent
    folder = agent.adapter.workdir(agent.target, runtime.framework_options)
    outputs = folder / OUTPUTS if folder is not None else None
    if outputs is not None and outputs.is_dir():
        here = outputs
        files = sorted(f for f in here.iterdir() if f.is_file())

        async def read(name: str) -> bytes:
            return await asyncio.to_thread(_written, here / name)

        sized = [(f.lstat().st_size, f.name) for f in files]
        await _kept(runtime, sized, read, agent.adapter.name, moved=lambda n: (here / n).unlink())


def _written(path: Path) -> bytes:
    """The bytes of the file at ``path`` itself, never of where a link there leads: any file
    this process may read, which the run's own account may not (``O_NOFOLLOW``, so a link
    put there after it was listed is refused too)."""
    try:
        handle = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as exc:  # ELOOP: a link
        raise ToolError("a link, not a file written there", source="tools") from exc
    with os.fdopen(handle, "rb") as opened:
        return opened.read()


async def _kept(
    runtime: Runtime,
    sized: list[tuple[int, str]],
    read: Callable[[str], Awaitable[bytes]],
    source: str,
    *,
    moved: Callable[[str], None] | None = None,
) -> None:
    """Each of ``sized`` (its size and name) read and kept for the run's user, ``source`` what
    made it, then ``moved`` (taken from where it was), when given. One past
    :data:`OUTPUT_MAX_BYTES`, or that cannot be kept, is left, said in a warning."""
    for size, name in sized:
        try:
            if size > OUTPUT_MAX_BYTES:
                raise ToolError(f"{size} bytes, past the {OUTPUT_MAX_BYTES} kept", source="tools")
            await runtime.save_file(name, await read(name), source=source)
        except ToolError as exc:
            runtime.events.warning(EVENT, f"the output {name} was not kept: {exc}")
            continue
        if moved is not None:
            moved(name)


def card_of(found: ExecResult, *, offline: bool) -> str:
    """The environment card a probe's output makes (:data:`PROBE`): ``Python <version>; installed:
    name version, ...`` — at most :data:`CARD_PACKAGES` named, the rest counted — and, with no
    network, that nothing else can be installed."""
    lines = found.stdout.strip().splitlines()
    if found.exit_code != 0 or not lines:
        return "The sandbox has no Python (python3): run shell commands only."
    version, packages = lines[0], (lines[1].split() if len(lines) > 1 else [])
    named = ", ".join(p.replace("==", " ") for p in packages[:CARD_PACKAGES])
    more = len(packages) - CARD_PACKAGES
    listed = (f"{named} and {more} more" if more > 0 else named) or "the standard library only"
    card = f"Its environment: Python {version}; installed: {listed}."
    if offline:
        card += " It has no network: nothing else can be installed."
    return card


def ref_of(runtime: Runtime, provider: str) -> SandboxRef:
    """The sandbox of ``runtime``'s run: named after it — the same name in every attempt, which
    makes ``create`` idempotent — and labelled with its run, tenant and agent."""
    run_id = runtime.run_id
    labels = {RUN: run_id, TENANT: runtime.tenant, AGENT: runtime.agent_id}
    return SandboxRef(provider=provider, id=name_of(run_id), labels=labels)


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
    provider = source.provider_for() if source is not None else None
    if ref.provider == HostSandbox.name and (provider is None or provider.name != ref.provider):
        provider = layer_of(ref)  # a command on the host (isolated_run): the layer's
    if provider is None:  # not among the agent's own sources (a handoff's specialist's)
        log.warning("the sandbox %s of run %s is left to the reaper", ref.id, run_id)
        return
    if not _claimed(ref):  # the reaper is deleting it
        return
    try:
        await provider.delete(ref)
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
    since = next(_STEPS)
    for ref in await provider.labelled({TENANT: tenant}):
        record = await runs.get(ref.labels[RUN], tenant=tenant)
        # not one its run's end (in this process) is deleting meanwhile
        if record is not None and record.final and _claimed(ref, since=since):
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
    would give it two tools of one name), else the one the harness gave it (:func:`code_for`)."""
    code = [getattr(agent, "code", None)]  # the one the harness gave the agent: last
    given = (
        s.source if isinstance(s, SandboxTools) else s for s in [*agent.sources, *added, *code]
    )
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
    "HostSandbox",
    "Limits",
    "NoShell",
    "Sandbox",
    "SandboxLost",
    "SandboxProvider",
    "SandboxRef",
    "SandboxSource",
    "SandboxSpec",
    "SandboxTools",
    "SharedDockerSandbox",
    "SupportsPause",
    "SupportsSnapshot",
    "card_of",
    "code_for",
    "code_of",
    "default_sandbox",
    "delivered",
    "given_code",
    "isolated_end",
    "isolated_run",
    "offered",
    "reap",
    "sandbox",
]
