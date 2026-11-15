"""One isolation layer for commands that share a machine: the tenant's shared sandbox (a
container of the tenant's, :mod:`trellis.harness.sandbox.shared`) and the host itself, when an
administrator allows commands there (:class:`HostSandbox`, :func:`isolated_run`). Both run every
command through :class:`Isolation`, the same scripts, so the same boundaries hold in both:

* **a tenant's own** — a base directory and a range of uids per tenant (``<root>/t-<digest>``,
  :data:`TENANT_UIDS` uids from :data:`UID_BASE`); a tenant's runs never get another's;
* **a run's own** — a workspace (``<root>/t-<digest>/<run's name>``, mode 0700) and an
  unprivileged uid of its own, made when the run first needs them; its commands cannot read,
  list or write another run's workspace (its owner and mode), see another run's processes (each
  command runs in a process namespace of its own, with a ``/proc`` of its own) or signal them
  (another uid), nor read the service's files: the layer's registry (``<root>/.trellis``) is
  root's alone, the paths in ``hidden`` (the service's working folder and home, by default on
  the host) are covered, and every filesystem but the workspace is read-only to it;
* **limits** per command (:class:`Limits`): CPU time, memory (address space), processes (per
  uid: per run), the largest file and the open files, and the call's wall-clock timeout —
  a runaway command is stopped, said in its result (``ExecResult.limit``), and the others go on;
* **no secret** — a command gets an environment of its own (:data:`PATH`, its ``HOME`` and
  ``TMPDIR`` in its workspace, ``LANG``, and what its caller passes), never the service's;
  the network is off (a network namespace of its own) unless the administrator opens it, and
  the cloud metadata server is out of reach either way (:data:`METADATA`); a Kubernetes pod's
  secret folders are hidden (:data:`SECRET_PATHS`);
* **clean-up** — a command's processes end with it (its process namespace goes); a run's end
  (:meth:`Isolation.release`) kills whatever of its uid is left, deletes its workspace and frees
  its uid; the reaper (``sandbox.reap``) does so for runs that ended without it;
* **audit** — every command is logged (:data:`AUDIT`) and, in a run, on its stream (the
  ``sandbox_command`` event): who, tenant, run, the command, its exit code, its duration and the
  limit it hit.

What it needs of the machine (or the container): Linux, util-linux's ``unshare``, ``setpriv``
and ``prlimit``, and root — the service itself, or a helper that runs a command as root
(``HostSandbox(helper=("sudo", "-n"))``). Without them it refuses (:meth:`HostSandbox.
cannot_isolate`), never running a command as the service's own user."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import posixpath
import re
import shutil
import subprocess
import sys
import time
import uuid
import weakref
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from trellis.contracts import ConfigurationError, ToolError, stable_id
from trellis.harness.runtime import current
from trellis.harness.sandbox.base import (
    AUDIT,
    AUDIT_CHARS,
    COMMAND_EVENT,
    RUN,
    TENANT,
    ExecResult,
    SandboxLost,
    SandboxRef,
    SandboxSpec,
    record_command,
)
from trellis.harness.sandbox.docker import OUTPUT_BYTES, Sink, _Tail
from trellis.harness.tools.base import REMOTE_TIMEOUT_SECONDS

log = logging.getLogger("trellis.sandbox")

#: What the layer launches a command with (util-linux).
TOOLS: Final = ("unshare", "setpriv", "prlimit")
#: The first uid a run gets, the uids each tenant has, and the most tenants one root holds.
UID_BASE: Final = 1_000_000_000
TENANT_UIDS: Final = 1_000
TENANTS: Final = 1_000_000
#: The layer's own folder under its root: root's alone (the runs' uids, labels, pid files).
REGISTRY: Final = ".trellis"
#: Where the host's workspaces are, unless another root is given.
HOST_ROOT: Final = "/var/lib/trellis/workspaces"
#: The labels a host workspace adds: whose run it is, and the layer that keeps it (a digest).
USER: Final = "trellis.user"
LAYER: Final = "trellis.layer"
#: A command's ``PATH``, and the environment the layer's own root commands get.
PATH: Final = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
ROOT_ENV: Final = {"PATH": PATH, "LANG": "C.UTF-8"}
#: The largest file a read returns (a larger one is read in parts with a command).
READ_MAX_BYTES: Final = 64 * 1024 * 1024
#: Kept results are read at this path (``tools.results``): the run's own, in its workspace.
RESULTS_PATH: Final = "/large_tool_results"
#: The cloud metadata server and the platform's other credential endpoints: the link-local
#: 169.254.0.0/16 (169.254.169.254: AWS's, Google's — ``metadata.google.internal`` — and
#: Azure's; 169.254.170.2: an ECS task's role credentials; 169.254.170.23: EKS Pod Identity's)
#: and ``fd00:ec2::254`` (AWS's over IPv6): no command reaches them, whatever the network —
#: with none, a command's network namespace has no route; open, the layer's uids are refused
#: them (:data:`GUARD`).
METADATA: Final = ("169.254.0.0/16", "fd00:ec2::254/128")
#: What refuses the layer's uids the metadata server on the host's network (iptables' owner
#: match: the service and every other user keep it).
FIREWALL: Final = ("iptables", "ip6tables")
#: Where a Kubernetes pod finds its secrets (its service account's token at
#: ``/var/run/secrets/kubernetes.io/serviceaccount``, Docker's and Compose's at
#: ``/run/secrets``): hidden from every command on the host.
SECRET_PATHS: Final = ("/var/run/secrets", "/run/secrets")
#: What the tools the layer needs are for (util-linux; and, with an open network, what keeps
#: the metadata server from commands).
MISSING: Final = (
    "unshare, setpriv and prlimit (util-linux) run every command, and with an open network "
    "iptables and ip6tables keep the cloud metadata server from it"
)
#: What a command on a machine the layer cannot isolate it on is refused with.
UNISOLATED_HERE: Final = (
    "the command was not run: commands cannot be isolated on this machine, and the harness "
    "never runs them as the service's own user"
)
#: A path the layer passes to its scripts: absolute, nothing a shell would split.
SAFE_PATH: Final = re.compile(r"/[A-Za-z0-9_./-]*")
NAME: Final = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*")
VARIABLE: Final = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
#: What each limit a command hit is called (``ExecResult.limit``), by its exit (128 + the
#: signal) or by what it printed.
EXIT_LIMITS: Final = {152: "cpu", 153: "file"}  # SIGXCPU, SIGXFSZ
PRINTED_LIMITS: Final = (
    ("File too large", "file"),
    ("Too many open files", "files"),
    ("Cannot fork", "processes"),
    ("can't fork", "processes"),
    ("Resource temporarily unavailable", "processes"),
    ("MemoryError", "memory"),
    ("Cannot allocate memory", "memory"),
    ("bad_alloc", "memory"),
    ("out of memory", "memory"),
)

# ----------------------------------------------------------------------------- the scripts
#: Kills every process of the uid it runs as but itself, until none is left (as that uid: a
#: signal reaches its own processes only).
_KILL_OWN = (
    "i=0; while :; do found=; for p in /proc/[0-9]*; do "
    '[ "${p#/proc/}" = "$$" ] && continue; [ -O "$p" ] && found=1 && break; done; '
    '[ -z "$found" ] && exit 0; kill -9 -1 2>/dev/null; i=$((i + 1)); '
    '[ "$i" -lt 100 ] || exit 1; sleep 0.05; done'
)
_KILLALL = (
    'killall() { setpriv --reuid="$1" --regid="$1" --clear-groups sh -c \''
    + _KILL_OWN
    + '\' || { echo "the processes of uid $1 do not end" >&2; return 1; }; }\n'
)
#: Makes (or finds) a run's workspace: ``R tenant name labels base per most`` → ``uid home
#: new|old``. The tenant's uid range is taken the first time (a block of its own), the run's
#: uid is the first free one in it, the workspace is made empty, the run's uid's.
OPEN: Final = (
    _KILLALL
    + r"""set -u
R=$1 t=$2 n=$3 labels=$4 base=$5 per=$6 most=$7
reg=$R/.trellis
(umask 022 && mkdir -p "$R")  # its parents made for any uid to go through (a run's to its own)
umask 077
mkdir -p "$reg/ws" "$reg/uids" "$reg/blocks" "$reg/tenants" "$reg/stage" "$reg/pids" \
  && chmod 711 "$R" && chmod 700 "$reg" \
  || { echo "the workspace was not made: no registry under $R" >&2; exit 70; }
if [ -s "$reg/ws/$n/uid" ]; then
  echo "$(cat "$reg/ws/$n/uid") $(cat "$reg/ws/$n/home") old"; exit 0
fi
if mkdir "$reg/tenants/$t" 2>/dev/null; then
  b=0
  while [ "$b" -lt "$most" ] && ! mkdir "$reg/blocks/$b" 2>/dev/null; do b=$((b + 1)); done
  if [ "$b" -ge "$most" ]; then
    rmdir "$reg/tenants/$t"; echo "no uid range is left for a tenant ($most tenants)" >&2; exit 75
  fi
  echo "$b" > "$reg/tenants/$t/.block" && mv "$reg/tenants/$t/.block" "$reg/tenants/$t/block"
fi
i=0
while [ ! -s "$reg/tenants/$t/block" ]; do
  i=$((i + 1)); [ "$i" -lt 100 ] || exit 70; sleep 0.05
done
if ! mkdir "$reg/ws/$n" 2>/dev/null; then  # another call is making it: its workspace
  i=0
  while [ ! -s "$reg/ws/$n/uid" ] && [ "$i" -lt 200 ]; do i=$((i + 1)); sleep 0.05; done
  if [ -s "$reg/ws/$n/uid" ]; then
    echo "$(cat "$reg/ws/$n/uid") $(cat "$reg/ws/$n/home") old"; exit 0
  fi  # else what a call that died left: made again
fi
b=$(cat "$reg/tenants/$t/block"); lo=$((base + b * per)); hi=$((lo + per)); u=$lo
while [ "$u" -lt "$hi" ] && ! mkdir "$reg/uids/$u" 2>/dev/null; do u=$((u + 1)); done
if [ "$u" -ge "$hi" ]; then
  rmdir "$reg/ws/$n"; echo "no uid is free for this tenant ($per runs at once)" >&2; exit 75
fi
printf '%s' "$n" > "$reg/uids/$u/ws"; printf '%s' "$labels" > "$reg/ws/$n/labels"
w=$R/$t/$n
killall "$u" \
  && mkdir -p "$R/$t" && chmod 711 "$R/$t" && rm -rf "$w" && mkdir -p "$w/.tmp" "$w/.results" \
  && chown -R "$u:$u" "$w" && chmod 700 "$w" "$w/.tmp" "$w/.results" \
  || { rm -rf "$reg/ws/$n" "$reg/uids/$u"; echo "the workspace was not made: $w" >&2; exit 70; }
printf '%s' "$w" > "$reg/ws/$n/home"; printf '%s' "$u" > "$reg/ws/$n/uid"
echo "$u $w new"
"""
)
#: A run's workspace, found again by a later attempt (``R name`` → ``uid home``; exit 3: none),
#: whatever an earlier attempt's commands left running killed (a crash's).
ATTACH: Final = (
    _KILLALL
    + r"""reg=$1/.trellis n=$2
[ -s "$reg/ws/$n/uid" ] || exit 3
u=$(cat "$reg/ws/$n/uid"); killall "$u" || exit 70
echo "$u $(cat "$reg/ws/$n/home")"
"""
)
#: A run's end (``R name``): its uid's processes killed, its workspace deleted, its uid freed.
RELEASE: Final = (
    _KILLALL
    + r"""reg=$1/.trellis n=$2
[ -s "$reg/ws/$n/uid" ] || { rm -rf "$reg/ws/$n"; exit 0; }
u=$(cat "$reg/ws/$n/uid")
killall "$u" || exit 70
rm -rf "$(cat "$reg/ws/$n/home")" || exit 70
rm -rf "$reg/uids/$u" "$reg/ws/$n"
"""
)
#: Every workspace under the root: its name and labels, a tab between, one a line.
LIST: Final = r"""reg=$1/.trellis
[ -d "$reg/ws" ] || exit 0
for d in "$reg"/ws/*/; do
  [ -s "${d}uid" ] || continue
  n=${d%/}; printf '%s\t%s\n' "${n##*/}" "$(cat "${d}labels")"
done
"""
#: Runs ``$@`` (the launcher) in the background so its process id is known, written to the pid
#: file ``$0`` (root's: a run cannot change it), which :data:`KILL_PID` kills by; its input the
#: wrapper's (by way of fd 3: a job in the background reads /dev/null otherwise).
PIDWRAP: Final = 'exec 3<&0; "$@" <&3 3<&- & echo $! > "$0"; wait $!; s=$?; rm -f "$0"; exit $s'
KILL_PID: Final = (
    'for i in 1 2 3 4 5 6 7 8 9 10; do [ -s "$0" ] && break; sleep 0.1; done; '
    '[ -s "$0" ] && kill -KILL "$(cat "$0")" 2>/dev/null; rm -f "$0"; true'
)
#: Inside the command's own namespaces, as root: the service's paths hidden, the workspace (and
#: its temporary folder, as /tmp, /var/tmp and /dev/shm) the only writable places — each other
#: filesystem read-only — the run's aliases bound, the limits set, then the command run as the
#: run's uid with no capability and an environment of its own (``env -i``).
STAGE: Final = r"""w=$1 u=$2 hide=$3 alias=$4 cpu=$5 cpuhard=$6 mem=$7 nproc=$8 fsize=$9; shift 9
nofile=$1 cwd=$2; shift 2
fail() { echo "the command was not started: $1" >&2; exit 126; }
for p in $hide; do
  [ -e "$p" ] || continue
  mount -t tmpfs -o size=4k,mode=000 trellis-hidden "$p" || fail "$p is not hidden"
done
mount --bind "$w" "$w" || fail "its workspace cannot be mounted"
keep=" $w "
for d in /tmp /var/tmp /dev/shm; do
  [ -d "$d" ] || continue
  case "$w/" in "$d"/*) continue ;; esac
  mount --bind "$w/.tmp" "$d" || fail "$d cannot be its own"
  keep="$keep$d "
done
for a in $alias; do
  src=${a%%=*} dst=${a#*=}
  { mkdir -p "$dst" && mount --bind "$src" "$dst"; } || fail "$dst cannot be its own"
  keep="$keep$dst "
done
mounts=$(while read -r _ m _; do printf '%s\n' "$m"; done < /proc/self/mounts)
for m in $mounts; do
  case "$keep" in *" $m "*) continue ;; esac
  case "$m" in /proc|/proc/*|/sys|/sys/*|/dev|/dev/*) continue ;; esac
  mount -o remount,bind,ro "$m" 2>/dev/null || [ "$m" != / ] || fail "the filesystem stays writable"
done
echo 1000 > /proc/self/oom_score_adj 2>/dev/null
cd "$cwd" || fail "no folder $cwd"
prlimit --cpu="$cpu:$cpuhard" --as="$mem" --nproc="$nproc" --fsize="$fsize" --nofile="$nofile" \
  --core=0 -- setpriv --reuid="$u" --regid="$u" --clear-groups --inh-caps=-all \
  --bounding-set=-all --no-new-privs -- env -i "$@"
"""
#: The metadata server refused to the uids ``lo``-``hi`` (``lo hi``): a rule each of
#: :data:`FIREWALL`, added once (it stays: it holds for any layer of those uids).
GUARD: Final = (
    "lo=$1 hi=$2; for rule in "
    + " ".join(f'"{wall} {address}"' for wall, address in zip(FIREWALL, METADATA, strict=True))
    + '; do set -- $rule; r="OUTPUT -m owner --uid-owner $lo-$hi -d $2 -j REJECT"; '
    '"$1" -w -C $r 2>/dev/null || "$1" -w -I $r '
    '|| { echo "the metadata server could not be refused to commands ($1)" >&2; exit 70; }; done'
)
#: A file of the run's, as the run's uid (``path max``): exit 3, no such file; 4, not a file;
#: 5, past ``max`` bytes.
READ: Final = (
    '[ -e "$1" ] || exit 3; [ -f "$1" ] || exit 4; '
    's=$(wc -c < "$1") || exit 6; [ "$s" -le "$2" ] || exit 5; exec cat -- "$1"'
)
#: A file written, as the run's uid, from the standard input (its folders made).
WRITE: Final = 'mkdir -p -- "$(dirname -- "$1")" && cat > "$1"'


class Limits(BaseModel):
    """What one command may use: ``cpu_seconds`` of CPU time (each process), ``memory_mib`` of
    address space (each process), ``processes`` at once (the run's uid's, every command of the
    run together), a file of at most ``file_mib``, ``open_files`` at once; its wall-clock time
    is the call's timeout."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    cpu_seconds: int = Field(default=120, gt=0)
    memory_mib: int = Field(default=2_048, gt=0)
    processes: int = Field(default=256, gt=0)
    file_mib: int = Field(default=1_024, gt=0)
    open_files: int = Field(default=1_024, gt=0)


@dataclass(frozen=True, slots=True)
class Ran:
    """What a root command did: its exit (128 + the signal that killed it) and its output."""

    code: int
    out: _Tail
    err: _Tail


class Runner(Protocol):
    """Runs a command as root where the workspaces are (this machine, a container)."""

    async def run(
        self,
        argv: Sequence[str],
        *,
        stdin: bytes | None = None,
        keep: int | None = OUTPUT_BYTES,
        sink: Sink | None = None,
    ) -> Ran:
        """Run ``argv`` as root, ``stdin`` its input (else none), each stream's last ``keep``
        bytes kept (``None``: all) and handed to ``sink`` as it comes."""
        ...


class HostRunner:
    """Root commands on this machine: the service's own when it is root, else through
    ``helper`` (a command that runs another as root: ``("sudo", "-n")``); never with the
    service's environment (:data:`ROOT_ENV`)."""

    def __init__(self, helper: Sequence[str] = ()) -> None:
        self.helper = tuple(helper)

    async def run(
        self,
        argv: Sequence[str],
        *,
        stdin: bytes | None = None,
        keep: int | None = OUTPUT_BYTES,
        sink: Sink | None = None,
    ) -> Ran:
        proc = await asyncio.create_subprocess_exec(
            *self.helper,
            *argv,
            stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=ROOT_ENV,
            start_new_session=True,
        )
        out, err = _Tail(keep, sink, 1), _Tail(keep, sink, 2)
        assert proc.stdout is not None and proc.stderr is not None
        try:
            await asyncio.gather(
                _drain(proc.stdout, out), _drain(proc.stderr, err), _feed(proc.stdin, stdin)
            )
            code = await proc.wait()
        except BaseException:  # its command is killed by its pid file (Isolation.launch)
            with contextlib.suppress(ProcessLookupError, PermissionError):
                proc.kill()
            raise
        return Ran(128 - code if code < 0 else code, out, err)


async def _drain(stream: asyncio.StreamReader, into: _Tail) -> None:
    while chunk := await stream.read(65_536):
        into.add(chunk)


async def _feed(stream: asyncio.StreamWriter | None, data: bytes | None) -> None:
    if stream is None or data is None:
        return
    stream.write(data)
    await stream.drain()
    stream.close()


class Isolation:
    """The layer: workspaces under ``root`` made, found, listed and removed, and commands run
    in them (:meth:`launch`), every one as root through ``runner`` (this machine's, a
    container's). ``network``: ``"none"`` (each command in a network namespace of its own) or
    ``"open"`` (the machine's or the container's own); ``hidden``: paths a command finds empty."""

    def __init__(
        self,
        runner: Runner,
        root: str,
        *,
        network: Literal["none", "open"] = "none",
        hidden: Sequence[str] = (),
        uid_base: int = UID_BASE,
        tenant_uids: int = TENANT_UIDS,
        tenants: int = TENANTS,
    ) -> None:
        self.runner = runner
        self.root = _safe(root, "the workspaces' root")
        if self.root == "/":
            raise ConfigurationError("the workspaces' root is a folder of their own, not /")
        for path in hidden:
            if _within(self.root, _safe(path, "a hidden path")):
                raise ConfigurationError(f"the hidden path {path} hides the workspaces ({root})")
        self.hidden = tuple(hidden)
        self.network: Literal["none", "open"] = network
        self.uid_base, self.tenant_uids, self.tenants = uid_base, tenant_uids, tenants

    @property
    def registry(self) -> str:
        return f"{self.root}/{REGISTRY}"

    async def open(
        self, tenant: str, name: str, labels: Mapping[str, str]
    ) -> tuple[int, str, bool]:
        """The workspace ``name`` of ``tenant``'s, made unless it is: its uid, its folder, and
        whether it was made now."""
        ran = await self._root(
            OPEN,
            self.root,
            tenant_key(tenant),
            _name(name),
            json.dumps(dict(labels), sort_keys=True),
            str(self.uid_base),
            str(self.tenant_uids),
            str(self.tenants),
        )
        uid, home, made = ran.out.text().split()
        return int(uid), home, made == "new"

    async def attach(self, name: str) -> tuple[int, str]:
        """The workspace ``name`` again (a later attempt's), with what an earlier attempt left
        running killed; :class:`SandboxLost` when it is gone."""
        ran = await self._root(ATTACH, self.root, _name(name), missing=3)
        if ran.code == 3:
            raise SandboxLost(f"the sandbox {name} is gone", source="tools")
        uid, home = ran.out.text().split()
        return int(uid), home

    async def release(self, name: str) -> None:
        """The run ended: its uid's processes killed, its workspace deleted, its uid freed."""
        await self._root(RELEASE, self.root, _name(name))

    async def listed(self) -> list[tuple[str, dict[str, str]]]:
        """Every workspace under the root, with its labels."""
        ran = await self._root(LIST, self.root)
        found = [line.partition("\t") for line in ran.out.text().splitlines() if line]
        return [(name, json.loads(labels)) for name, _, labels in found]

    async def launch(
        self,
        box: Workspace,
        argv: Sequence[str],
        *,
        limits: Limits,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,  # noqa: ASYNC109 - the command's, killed past it
        stdin: bytes | None = None,
        keep: int | None = OUTPUT_BYTES,
        sink: Sink | None = None,
    ) -> Ran:
        """Run ``argv`` in ``box`` as its uid, in namespaces of its own (processes, mounts, and
        the network unless open), within ``limits``, its output handed to ``sink`` as it comes;
        past ``timeout`` (or cancelled) it is killed, with everything it started, and the error
        raised."""
        given = dict(env or {})
        for name in given:
            if not VARIABLE.fullmatch(name):
                raise ToolError(f"{name!r} is not a variable name", source="tools")
        variables = {
            "PATH": PATH,
            "HOME": box.home,
            "TMPDIR": f"{box.home}/.tmp",
            "LANG": "C.UTF-8",
            "USER": "sandbox",
            **given,
        }
        aliases = " ".join(f"{src}={dst}" for dst, src in box.aliases.items())
        pid_file = f"{self.registry}/pids/{uuid.uuid4().hex}"
        net = ["--net"] if box.network == "none" else []
        command = [
            *("sh", "-c", PIDWRAP, pid_file),
            *("unshare", "--pid", "--fork", "--mount-proc", "--kill-child", *net, "--"),
            *("sh", "-c", STAGE, "trellis", box.home, str(box.uid), " ".join(self.hidden)),
            aliases,
            str(limits.cpu_seconds),
            str(limits.cpu_seconds + 1),
            str(limits.memory_mib * 1024 * 1024),
            str(limits.processes),
            str(limits.file_mib * 1024 * 1024),
            str(limits.open_files),
            box.cwd,
            *(f"{k}={v}" for k, v in variables.items()),
            *argv,
        ]
        try:
            async with asyncio.timeout(timeout):
                return await self.runner.run(command, stdin=stdin, keep=keep, sink=sink)
        except BaseException:  # out of time, cancelled, cut off: stopped, not left running
            await asyncio.shield(self.runner.run(["sh", "-c", KILL_PID, pid_file]))
            raise

    async def _root(self, script: str, *args: str, missing: int | None = None) -> Ran:
        ran = await self.runner.run(["sh", "-c", script, "trellis", *args])
        if ran.code not in (0, missing):
            said = ran.err.text().strip() or f"exit {ran.code}"
            raise ToolError(said, source="tools", retryable=ran.code == 75)
        return ran


class Workspace:
    """A run's workspace, as its tools use it (the ``Sandbox`` protocol): ``uid`` the run's,
    ``home`` its folder. Every command is audited, once, here (:data:`AUDIT`,
    :data:`COMMAND_EVENT`, the tool call's span: ``base.record_command``)."""

    #: it audits its own commands (``base.audited`` leaves them to it)
    audits = True

    def __init__(
        self,
        isolation: Isolation,
        ref: SandboxRef,
        uid: int,
        home: str,
        *,
        limits: Limits,
        network: Literal["none", "open"] = "none",
    ) -> None:
        self.isolation = isolation
        self._ref = ref
        self.uid = uid
        self.home = home
        self.limits = limits
        self.network: Literal["none", "open"] = network
        #: paths a command finds its own folders at (the path → its folder): the run's kept
        #: results at theirs, and what :meth:`alias` adds
        self.aliases: dict[str, str] = {RESULTS_PATH: f"{home}/.results"}
        #: where a command starts
        self.cwd = home

    @property
    def ref(self) -> SandboxRef:
        return self._ref

    def alias(self, path: str) -> None:
        """Make ``path`` the workspace too, and where commands start: for an agent that names
        its files by the folder it works in elsewhere (Kimi Code's, on this machine)."""
        self.aliases[_safe(path, "an alias")] = self.home
        self.cwd = path

    async def exec(
        self,
        command: str,
        *,
        timeout: float | None = None,  # noqa: ASYNC109 - the command's, killed past it
        env: Mapping[str, str] | None = None,
        limits: Limits | None = None,
        argv: Sequence[str] | None = None,
        sink: Sink | None = None,
    ) -> ExecResult:
        """Run ``command`` (``sh -c``; ``argv``: the program that runs it, a framework's own
        shell's) in the workspace — its output handed to ``sink`` as it comes — audited."""
        started = time.monotonic()
        try:
            ran = await self.isolation.launch(
                self,
                argv or ["sh", "-c", command],
                limits=limits or self.limits,
                env=env,
                timeout=timeout,
                sink=sink,
            )
        except BaseException as exc:
            self._audit(command, None, started, "time" if isinstance(exc, TimeoutError) else None)
            raise
        stderr = ran.err.text()
        limit = limit_of(ran.code, stderr)
        self._audit(command, ran.code, started, limit)
        return ExecResult(
            exit_code=ran.code,
            stdout=ran.out.text(),
            stderr=stderr,
            duration_ms=_since(started),
            limit=limit,
        )

    async def read(self, path: str) -> bytes:
        where = self._where(path)
        argv = ["sh", "-c", READ, "read", path, str(READ_MAX_BYTES)]
        ran = await self.isolation.launch(self, argv, limits=self.limits, keep=None)
        if ran.code == 0:
            return bytes(ran.out.kept)
        said = {
            3: f"there is no file {where} in the sandbox",
            4: f"{where} in the sandbox is not a file",
            5: f"{where} is more than the {READ_MAX_BYTES} bytes read at once: read a part of "
            "it with a command",
        }.get(ran.code) or f"{where} was not read: {ran.err.text().strip()}"
        raise ToolError(said, source="tools")

    async def write(self, path: str, data: bytes) -> None:
        argv = ["sh", "-c", WRITE, "write", path]
        ran = await self.isolation.launch(self, argv, limits=self.limits, stdin=data)
        if ran.code != 0:
            raise ToolError(
                f"{self._where(path)} was not written: {ran.err.text().strip()}", source="tools"
            )

    def _where(self, path: str) -> str:
        return posixpath.normpath(posixpath.join(self.cwd, path))

    def _audit(self, command: str, code: int | None, started: float, limit: str | None) -> None:
        """The command's one audit record (``base.record_command``: the log, the run's stream,
        the tool call's span): where it ran, this machine's layer (``host``) or a container's
        (``sandbox``)."""
        runtime = current()
        labels = self._ref.labels
        record_command(
            {
                "user": labels.get(USER) or (runtime.user if runtime is not None else None),
                "tenant": labels.get(TENANT),
                "run": labels.get(RUN),
                "sandbox": self._ref.id,
                "uid": self.uid,
                "command": command[:AUDIT_CHARS],
                "exit_code": code,
                "duration_ms": _since(started),
                "limit": limit,
                "where": "host" if isinstance(self.isolation.runner, HostRunner) else "sandbox",
            }
        )


class HostSandbox:
    """Workspaces on this machine, through :class:`Isolation` (``name`` ``"host"``): what an
    administrator who allows commands on the host gives them (:func:`isolated_run`), and a
    provider ``sandbox(HostSandbox())`` takes. ``root`` holds the workspaces (else
    :data:`HOST_ROOT`); ``helper`` runs a command as root when the service is not root;
    ``limits`` each command's; ``network`` the administrator's (``"none"``, the default, or
    ``"open"``; a list of hosts is refused: nothing here can enforce one); ``hidden`` the paths
    a command finds empty (else the service's working folder, its home, ``/root``, ``/home``
    and the secret folders :data:`SECRET_PATHS`, those that are not above ``root``). The cloud
    metadata server is out of every command's reach (:data:`METADATA`)."""

    name = "host"

    def __init__(
        self,
        root: str = HOST_ROOT,
        *,
        helper: Sequence[str] = (),
        limits: Limits | None = None,
        network: Literal["none", "open"] = "none",
        hidden: Sequence[str] | None = None,
        uid_base: int = UID_BASE,
        tenant_uids: int = TENANT_UIDS,
        tenants: int = TENANTS,
    ) -> None:
        self.helper = tuple(helper)
        self.limits = limits or Limits()
        if hidden is None:
            ours = [os.getcwd(), str(Path.home()), "/root", "/home", *SECRET_PATHS]
            given = _safe(root, "the workspaces' root")
            hidden = [
                p for p in dict.fromkeys(ours) if SAFE_PATH.fullmatch(p) and not _within(given, p)
            ]
        self.isolation = Isolation(
            HostRunner(self.helper),
            root,
            network=network,
            hidden=hidden,
            uid_base=uid_base,
            tenant_uids=tenant_uids,
            tenants=tenants,
        )
        self.key = stable_id("host", self.isolation.root)
        self._isolated = False
        #: the metadata server is refused to this layer's uids (:data:`GUARD`)
        self._guarded = False
        #: when this process last reaped each tenant's workspaces (``time.monotonic``)
        self._reaped: dict[str, float] = {}
        _LAYERS[self.key] = self

    def cannot_isolate(self) -> str | None:
        """Why commands cannot be isolated on this machine (no Linux, a tool missing, no way to
        switch uid), else ``None``."""
        if not sys.platform.startswith("linux"):
            return f"commands are isolated on Linux only, and this is {sys.platform}"
        missing = [t for t in TOOLS if shutil.which(t) is None]
        walls = [t for t in FIREWALL if shutil.which(t) is None]
        if self.isolation.network == "open" and walls:
            missing = [*missing, *walls]
        if missing:
            return f"{', '.join(missing)} are missing: {MISSING}"
        if self._isolated:
            return None
        if self.helper:
            try:
                said = subprocess.run(
                    [*self.helper, "id", "-u"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
            except (OSError, subprocess.SubprocessError):
                said = None
            if said is None or said.returncode != 0 or said.stdout.strip() != "0":
                return f"the helper {' '.join(self.helper)} cannot run a command as root"
        elif os.geteuid() != 0:
            return (
                f"the service runs as uid {os.geteuid()}, not root, and no helper is given "
                "(HostSandbox(helper=...)): it cannot switch a command to a run's own uid"
            )
        self._isolated = True
        return None

    async def workspace(self, tenant: str, user: str | None, run: str) -> Workspace:
        """The workspace of ``tenant``'s run ``run`` for ``user``, made the first time; inside
        that run, named in its journal (its end removes it) and the tenant's ended runs reaped
        now and then."""
        labels = {RUN: run, TENANT: tenant, USER: user or "", LAYER: self.key}
        box = await self.create(SandboxRef(provider=self.name, id=name_of(run), labels=labels))
        runtime = current()
        if runtime is not None and runtime.run_id == run:
            await self._recorded(runtime, box)
        return box

    async def create(self, ref: SandboxRef, spec: SandboxSpec | None = None) -> Workspace:
        why = self.cannot_isolate()
        if why is not None:  # never run as the service's own user
            raise ToolError(f"{UNISOLATED_HERE}: {why}", source="tools")
        given = spec or SandboxSpec(network=self.isolation.network)
        if isinstance(given.network, tuple):
            raise ConfigurationError(
                "a host sandbox's network is none or open: it cannot limit it to some hosts"
            )
        if given.network == "open" and self.isolation.network == "none":
            raise ConfigurationError(
                "this sandbox asks for the network, and the administrator's network is none "
                "(HostSandbox(network='open') opens it)"
            )
        tenant = ref.labels.get(TENANT)
        if tenant is None:
            raise ConfigurationError(f"a host workspace is a tenant's: {ref.id} names none")
        labels = {**ref.labels, LAYER: self.key}
        if self.isolation.network == "open" and not self._guarded:
            layer = self.isolation
            top = layer.uid_base + layer.tenants * layer.tenant_uids - 1
            await layer._root(GUARD, str(layer.uid_base), str(top))
            self._guarded = True
        uid, home, made = await self.isolation.open(tenant, ref.id, labels)
        box = self._box(ref.model_copy(update={"labels": labels}), uid, home, given.network)
        for path, content in given.files.items() if made else ():
            await box.write(path, content.encode() if isinstance(content, str) else content)
        return box

    async def attach(self, ref: SandboxRef) -> Workspace:
        uid, home = await self.isolation.attach(ref.id)
        return self._box(ref, uid, home, self.isolation.network)

    async def delete(self, ref: SandboxRef) -> None:
        await self.isolation.release(ref.id)

    async def labelled(self, labels: Mapping[str, str]) -> list[SandboxRef]:
        return [
            SandboxRef(provider=self.name, id=name, labels=held)
            for name, held in await self.isolation.listed()
            if all(held.get(k) == v for k, v in labels.items())
        ]

    def _box(self, ref: SandboxRef, uid: int, home: str, network: Any) -> Workspace:
        return Workspace(self.isolation, ref, uid, home, limits=self.limits, network=network)

    async def _recorded(self, runtime: Any, box: Workspace) -> None:
        """The run's journal names its workspace ``box`` (its end removes it:
        ``sandbox.ended``) unless it names a sandbox already — made now, the working files of
        the conversation's earlier turns given back into it (``sandbox.carry``); it is the
        run's sandbox for its outputs and its working files, unless it has one; the tenant's
        ended runs reaped, at most every ``REAP_SECONDS``."""
        from trellis.harness.sandbox import REAP_SECONDS, carry, reap  # noqa: PLC0415

        ref = box.ref
        journal = runtime.replay.journal
        if journal.sandbox is None:
            journal.sandbox = ref.model_dump(mode="json", exclude_none=True)
            await runtime.progress(now=True)
            await carry.restored(runtime, box)
        if runtime.sandbox is None:  # its outputs and working files are kept from it
            runtime.sandbox = box
        tenant, now = runtime.tenant, time.monotonic()
        last = self._reaped.get(tenant)
        if last is not None and now - last < REAP_SECONDS:
            return
        self._reaped[tenant] = now
        harness = runtime.agent.harness

        async def work() -> None:
            await reap(self, harness.runs, tenant=tenant)

        await harness.writes.submit("sandbox.reap", work, events=runtime.events)


#: The host layers made in this process, by their key (``LAYER``): a run's end finds its own.
_LAYERS: weakref.WeakValueDictionary[str, HostSandbox] = weakref.WeakValueDictionary()
#: The host's layer (:func:`use_host`, else one with every default, made when first needed).
_HOST: dict[str, HostSandbox] = {}


def use_host(layer: HostSandbox | None) -> HostSandbox | None:
    """Make ``layer`` the host's (:func:`isolated_run`'s when given none); the one it replaces is
    returned."""
    replaced = _HOST.pop("host", None)
    if layer is not None:
        _HOST["host"] = layer
    return replaced


def host_layer() -> HostSandbox:
    """The host's layer: the one :func:`use_host` gave, else one with every default."""
    if "host" not in _HOST:
        _HOST["host"] = HostSandbox()
    return _HOST["host"]


def layer_of(ref: SandboxRef) -> HostSandbox:
    """The host layer that keeps the workspace ``ref`` names (else the host's)."""
    return _LAYERS.get(ref.labels.get(LAYER, "")) or host_layer()


async def isolated_run(
    tenant: str,
    user: str | None,
    run: str,
    command: str,
    limits: Limits | None = None,
    *,
    timeout: float | None = REMOTE_TIMEOUT_SECONDS,  # noqa: ASYNC109 - the command's
    env: Mapping[str, str] | None = None,
    layer: HostSandbox | None = None,
) -> ExecResult:
    """Run ``command`` on this machine for ``tenant``'s run ``run`` of ``user``: in the run's
    workspace, as its uid, within ``limits`` (else the layer's) and ``timeout`` seconds, with an
    environment of ``env`` and the layer's own only — the entry the host path calls. Refused
    (``ToolError``, :meth:`HostSandbox.create`) when the layer cannot isolate it here: never run
    as the service's own user. Inside the run, its end removes the workspace; outside one,
    :func:`isolated_end`."""
    held = layer or host_layer()
    box = await held.workspace(tenant, user, run)
    return await box.exec(command, timeout=timeout, env=env, limits=limits)


async def isolated_end(run: str, *, layer: HostSandbox | None = None) -> None:
    """The run ``run`` ended: its processes killed, its workspace deleted, its uid freed."""
    await (layer or host_layer()).isolation.release(name_of(run))


def name_of(run: str) -> str:
    """A run's sandbox name: ``trellis-<run id>``, or its digest for a run id no provider takes
    as a name (at most 63 characters)."""
    name = f"trellis-{run}"
    if len(name) > 63 or not re.fullmatch(r"[A-Za-z0-9_.-]+", run):
        name = f"trellis-{stable_id(run)}"
    return name


def tenant_key(tenant: str) -> str:
    """A tenant's base folder's name: its digest (any tenant id is a safe name)."""
    return f"t-{stable_id('tenant', tenant)}"


def limit_of(code: int, stderr: str) -> str | None:
    """The limit a command that failed hit (``cpu``, ``memory``, ``processes``, ``file``,
    ``files``), by its exit or by what it printed; ``None`` for any other end."""
    if code == 0:
        return None
    if code in EXIT_LIMITS:
        return EXIT_LIMITS[code]
    return next((limit for said, limit in PRINTED_LIMITS if said in stderr), None)


def _since(started: float) -> int:
    return int((time.monotonic() - started) * 1000)


def _safe(path: str, what: str) -> str:
    if not path.startswith("/"):
        raise ConfigurationError(f"{what} is an absolute path, not {path!r}")
    norm = posixpath.normpath(path)
    if not SAFE_PATH.fullmatch(norm):
        raise ConfigurationError(f"{what} {path!r} has characters a shell would split")
    return norm


def _within(path: str, above: str) -> bool:
    """Whether ``above`` is ``path`` or a folder above it."""
    return above in ("/", path) or path.startswith(f"{above}/")


def _name(name: str) -> str:
    if not NAME.fullmatch(name):
        raise ToolError(f"{name!r} is not a workspace name", source="tools")
    return name


__all__ = [
    "AUDIT",
    "COMMAND_EVENT",
    "HostRunner",
    "HostSandbox",
    "Isolation",
    "Limits",
    "Runner",
    "Workspace",
    "host_layer",
    "isolated_end",
    "isolated_run",
    "layer_of",
    "limit_of",
    "name_of",
    "use_host",
]
