"""How many child runs a run has working at once — adaptive, for every engine — and the live
task tree its stream shows of them.

* **The limit** — :class:`Admission`, one per process (:data:`ADMISSION`): AIMD, as TCP's
  congestion window. Each child that ends while the process is healthy adds ``1 / limit``
  (about one more per round of children); a sign of pressure halves it, at most once per
  :data:`COOLDOWN_SECONDS`. Pressure is the container's CPU or memory past
  :data:`CPU_HIGH` / :data:`MEMORY_HIGH` of its cgroup's limits (:class:`Machine`: cgroup v2,
  else v1, else the host's), a provider slower than :data:`LATENCY_RISE` times its fastest
  recent pace (seconds per output token, from every LangChain model call:
  ``accounting.py``), or a gateway ``429``. It stays between :data:`FLOOR` and
  :data:`CEILING`, from :data:`START`.
* **Where it applies** — per parent run (:func:`branch`): a run admits at most the limit of
  its own children at once, and each child admits its own (nested children never wait on
  their parent's slots, so nothing deadlocks). Every child run the harness starts goes
  through it: a sub-agent's call in any framework, ``agent.run`` inside a run, a ``ReAct``'s
  ``delegate`` jobs (``trellis.harness.delegate``). A framework that spawns its own helpers
  keeps its own parallelism; Kimi Code is told the limit
  (``KIMI_CODE_AGENT_SWARM_MAX_CONCURRENCY``, ``adapters.acp``).
* **Hard ceilings** — a child waits for a slot at most what is left of its parent's time;
  the budget is the parent's (``budget.py``: every child's spending counts toward it).
* **The task tree** — each child on its parent's stream: ``helper_queued`` (waiting for a
  slot), ``helper_started``, ``helper_finished`` (its status, seconds, tokens and cost), with
  its run, its agent and what the caller says of it (``RunEvents.helper``).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import time
import weakref
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Protocol

from trellis.contracts.errors import is_pause_signal

if TYPE_CHECKING:
    from trellis.harness.runtime import Runtime

log = logging.getLogger("trellis.run")

#: The limit's bounds and where it starts: children of one parent at once.
FLOOR: Final = 1
START: Final = 4
CEILING: Final = 16
#: What one pressure sign multiplies the limit by, and the least time between two cuts (a
#: burst of 429s is one sign).
BACKOFF: Final = 0.5
COOLDOWN_SECONDS: Final = 2.0
#: The share of the container's CPU quota and memory limit past which it is under pressure.
CPU_HIGH: Final = 0.85
MEMORY_HIGH: Final = 0.85
#: The provider is slow past this many times its fastest recent pace (seconds per output
#: token, smoothed by :data:`LATENCY_WEIGHT`), once :data:`LATENCY_SAMPLES` calls were seen;
#: the fastest pace drifts up by :data:`LATENCY_DRIFT` a call, so an old one does not hold.
LATENCY_RISE: Final = 2.0
LATENCY_WEIGHT: Final = 0.2
LATENCY_SAMPLES: Final = 5
LATENCY_DRIFT: Final = 0.01
#: A pace this fast (seconds per output token: 50 tokens a second) is never slow, however it
#: compares: tiny paces are noise.
PACE_FLOOR: Final = 0.02
#: How long a reading of the machine stands before it is read again.
SAMPLE_SECONDS: Final = 1.0
#: A cgroup v1 memory limit this high is no limit (the kernel's "unlimited").
NO_LIMIT: Final = 1 << 62


class Gauges(Protocol):
    """What the admission reads of the machine: CPU and memory use, as shares of its limits."""

    def cpu(self) -> float | None: ...

    def memory(self) -> float | None: ...


class Machine:
    """This container's CPU and memory use, as shares of its limits: cgroup v2 (``cpu.max``,
    ``cpu.stat``, ``memory.max``, ``memory.current``), else v1 (``cpu``, ``cpuacct``,
    ``memory``), else the host's (``/proc/meminfo``, the load average). Memory is the working
    set (inactive page cache left out, as the kubelet counts it)."""

    def __init__(
        self,
        root: Path = Path("/sys/fs/cgroup"),
        meminfo: Path = Path("/proc/meminfo"),
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.root = root
        self.meminfo = meminfo
        self.clock = clock
        self._last: tuple[float, float] | None = None

    def memory(self) -> float | None:
        """The share of the memory limit in use (``None``: not known)."""
        v2 = self._read("memory.max"), self._read("memory.current")
        if v2[1] is not None:
            limit, used = v2[0], float(v2[1])
            inactive = self._stat("memory.stat", "inactive_file")
        else:
            limit = self._read("memory/memory.limit_in_bytes")
            usage = self._read("memory/memory.usage_in_bytes")
            if usage is None:
                return self._host_memory()
            used = float(usage)
            inactive = self._stat("memory/memory.stat", "total_inactive_file")
        total = float(limit) if isinstance(limit, int) else None
        if total is None or total >= NO_LIMIT:
            total = self._host_total()
        if not total:
            return None
        return max(0.0, used - (inactive or 0.0)) / total

    def cpu(self) -> float | None:
        """The share of the CPU quota used since the last reading (``None``: the first one,
        or not known)."""
        used = self._cpu_seconds()
        if used is None:
            return os.getloadavg()[0] / (os.cpu_count() or 1)
        now = self.clock()
        last, self._last = self._last, (now, used)
        if last is None or now <= last[0]:
            return None
        return (used - last[1]) / ((now - last[0]) * self._cores())

    def _cpu_seconds(self) -> float | None:
        usec = self._stat("cpu.stat", "usage_usec")
        if usec is not None:
            return usec / 1e6
        nsec = self._read("cpuacct/cpuacct.usage")
        return None if nsec is None else float(nsec) / 1e9

    def _cores(self) -> float:
        quota = self._read("cpu.max")
        if quota is not None:
            given, _, period = str(quota).partition(" ")
            if given != "max" and period:
                return float(given) / float(period)
        v1, period = self._read("cpu/cpu.cfs_quota_us"), self._read("cpu/cpu.cfs_period_us")
        if v1 is not None and period and float(v1) > 0:
            return float(v1) / float(period)
        return float(os.cpu_count() or 1)

    def _read(self, name: str) -> str | int | None:
        try:
            text = (self.root / name).read_text().strip()
        except OSError:
            return None
        return int(text) if text.lstrip("-").isdigit() else text

    def _stat(self, name: str, key: str) -> float | None:
        try:
            lines = (self.root / name).read_text().splitlines()
        except OSError:
            return None
        for line in lines:
            found, _, value = line.partition(" ")
            if found == key and value.strip().isdigit():
                return float(value)
        return None

    def _meminfo(self) -> dict[str, float]:
        try:
            lines = self.meminfo.read_text().splitlines()
        except OSError:
            return {}
        found: dict[str, float] = {}
        for line in lines:
            name, _, rest = line.partition(":")
            words = rest.split()
            if words and words[0].isdigit():
                found[name] = float(words[0]) * 1024
        return found

    def _host_total(self) -> float | None:
        return self._meminfo().get("MemTotal")

    def _host_memory(self) -> float | None:
        info = self._meminfo()
        total, available = info.get("MemTotal"), info.get("MemAvailable")
        if not total or available is None:
            return None
        return 1 - available / total


class Admission:
    """The AIMD limit of child runs a parent has at once (see the module). ``machine`` reads
    the container (``None``: its pressure is not watched), ``clock`` keeps time."""

    def __init__(
        self,
        *,
        start: int = START,
        floor: int = FLOOR,
        ceiling: int = CEILING,
        machine: Gauges | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not 1 <= floor <= start <= ceiling:
            raise ValueError("an admission limit needs 1 <= floor <= start <= ceiling")
        self.floor = floor
        self.ceiling = ceiling
        self.machine = machine
        self.clock = clock
        self._limit = float(start)
        self._cut_at: float | None = None
        self._pace: float | None = None
        self._fastest: float | None = None
        self._paced = 0
        self._read_at: float | None = None
        self._reading: str | None = None

    @property
    def limit(self) -> int:
        """The children a parent may have working at once now."""
        return max(self.floor, min(self.ceiling, int(self._limit)))

    def ended(self, *, ok: bool) -> None:
        """A child ended: under pressure the limit is cut; else one that answered raises it by
        ``1 / limit``."""
        pressure = self.pressure()
        if pressure is not None:
            self._cut(pressure)
        elif ok:
            self._limit = min(float(self.ceiling), self._limit + 1 / self._limit)

    def admitting(self) -> None:
        """A child is about to be admitted: under pressure the limit is cut first."""
        pressure = self.pressure()
        if pressure is not None:
            self._cut(pressure)

    def throttled(self) -> None:
        """The provider or its gateway answered ``429``: the limit is cut."""
        self._cut("the model gateway is rate limiting (429)")

    def paced(self, seconds: float, output_tokens: int) -> None:
        """A model call took ``seconds`` for ``output_tokens``: the provider's pace."""
        if output_tokens <= 0 or seconds < 0:
            return
        pace = seconds / output_tokens
        self._pace = (
            pace
            if self._pace is None
            else (LATENCY_WEIGHT * pace + (1 - LATENCY_WEIGHT) * self._pace)
        )
        drifted = None if self._fastest is None else self._fastest * (1 + LATENCY_DRIFT)
        self._fastest = self._pace if drifted is None else min(drifted, self._pace)
        self._paced += 1

    def pressure(self) -> str | None:
        """Why the process should not take more children now (``None``: it is healthy)."""
        slow = (
            self._paced >= LATENCY_SAMPLES
            and self._pace is not None
            and self._fastest is not None
            and self._pace > max(LATENCY_RISE * self._fastest, PACE_FLOOR)
        )
        if slow:
            return "the model provider is slowing down"
        return self._machine()

    def _machine(self) -> str | None:
        if self.machine is None:
            return None
        now = self.clock()
        if self._read_at is not None and now - self._read_at < SAMPLE_SECONDS:
            return self._reading
        self._read_at = now
        cpu, memory = self.machine.cpu(), self.machine.memory()
        self._reading = (
            f"CPU at {cpu:.0%} of the container's quota"
            if cpu is not None and cpu > CPU_HIGH
            else f"memory at {memory:.0%} of the container's limit"
            if memory is not None and memory > MEMORY_HIGH
            else None
        )
        return self._reading

    def _cut(self, why: str) -> None:
        now = self.clock()
        if self._cut_at is not None and now - self._cut_at < COOLDOWN_SECONDS:
            return
        self._cut_at = now
        before = self.limit
        self._limit = max(float(self.floor), self._limit * BACKOFF)
        log.info("child runs at once: %d, was %d (%s)", self.limit, before, why)


#: The process's admission: every parent's children are limited by it.
ADMISSION = Admission(machine=Machine())


class Gate:
    """One parent's children at once: at most the admission's limit working, each in a slot
    of its own (numbered from 0, the lowest free one taken)."""

    def __init__(self) -> None:
        self.taken: set[int] = set()
        #: the most that ever worked at once
        self.peak = 0
        self._changed = asyncio.Condition()

    async def enter(self, admission: Admission, within: float | None) -> int:
        """A slot, once fewer than the limit work (at most ``within`` seconds: past it
        ``TimeoutError``)."""
        admission.admitting()
        async with asyncio.timeout(within), self._changed:
            await self._changed.wait_for(lambda: len(self.taken) < admission.limit)
            slot = min(set(range(len(self.taken) + 1)) - self.taken)
            self.taken.add(slot)
            self.peak = max(self.peak, len(self.taken))
            return slot

    async def leave(self, slot: int) -> None:
        async with self._changed:
            self.taken.discard(slot)
            self._changed.notify_all()


#: Each parent's gate, kept while its runtime is.
_GATES: weakref.WeakKeyDictionary[Runtime, Gate] = weakref.WeakKeyDictionary()


def gate(parent: Runtime) -> Gate:
    """``parent``'s gate."""
    found = _GATES.get(parent)
    if found is None:
        found = _GATES[parent] = Gate()
    return found


class Branch:
    """One child of a run, as its parent's stream shows it: what the caller says of it
    (``about``), its time, and what it spent (its journal's, set by whoever ran it)."""

    def __init__(self, parent: Runtime, run_id: str, agent_id: str, about: dict[str, Any]) -> None:
        self.parent = parent
        self.run_id = run_id
        self.agent_id = agent_id
        self.about = about
        self.spent: dict[str, float] = {}
        self.began: float | None = None
        self.done = False
        #: its slot in the parent's gate, once admitted
        self.slot = 0

    def emit(self, phase: str, **data: Any) -> None:
        self.parent.events.helper(
            phase, run_id=self.run_id, agent_id=self.agent_id, **self.about, **data
        )

    def started(self) -> None:
        self.began = time.monotonic()
        self.emit("started")

    def ended(self, status: str, *, error: str | None = None) -> None:
        """The child ended ``status`` (once): its time and spending on the stream, and the
        admission told how it went."""
        if self.done:
            return
        self.done = True
        seconds = 0.0 if self.began is None else time.monotonic() - self.began
        spent = {k: self.spent[k] for k in ("tokens", "cost") if k in self.spent}
        found = {"error": error} if error is not None else {}
        self.emit("finished", status=status, seconds=round(seconds, 3), **spent, **found)
        ADMISSION.ended(ok=status in ("success", "partial"))


@contextlib.asynccontextmanager
async def branch(
    parent: Runtime,
    run_id: str,
    agent_id: str,
    *,
    within: float | None = None,
    **about: Any,
) -> AsyncIterator[Branch]:
    """A child of ``parent``, admitted: queued on the stream, a slot of the parent's gate taken
    (waiting at most ``within`` seconds, else what is left of the parent's time: past it
    ``TimeoutError``), started, and — however
    it ends — finished and its slot given back. The caller says how it ended
    (:meth:`Branch.ended`); one that raised ends ``error`` (a pause: ``paused``; a cancellation:
    ``cancelled``)."""
    found = Branch(parent, run_id, agent_id, about)
    found.emit("queued", limit=ADMISSION.limit)
    held = gate(parent)
    try:
        found.slot = await held.enter(ADMISSION, parent.remaining() if within is None else within)
    except TimeoutError:
        found.ended("timeout", error="no slot before the run's time ran out")
        raise
    try:
        found.started()
        yield found
    except asyncio.CancelledError:
        found.ended("cancelled")
        raise
    except BaseException as exc:
        found.ended("paused" if is_pause_signal(exc) else "error", error=str(exc) or None)
        raise
    finally:
        await held.leave(found.slot)
