"""The adaptive limit of child runs (``admission.py``): AIMD up while healthy and down on
pressure (CPU or memory against the container's cgroup limits, a slowing provider, a 429), at
most one cut per cooldown, within its floor and ceiling; each parent's gate holds at most the
limit, in slots of their own; a branch is on its parent's task tree however it ends; the model
calls of every run feed it (``accounting``)."""

from __future__ import annotations

import asyncio
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from trellis.contracts import AgentExecutionContext, AgentPaused, RunEvent
from trellis.harness import accounting, admission
from trellis.harness.admission import (
    BACKOFF,
    COOLDOWN_SECONDS,
    LATENCY_SAMPLES,
    NO_LIMIT,
    SAMPLE_SECONDS,
    Admission,
    Gate,
    Machine,
    branch,
)
from trellis.harness.events import RunEvents
from trellis.harness.gateways import ProviderSDK
from trellis.harness.identity import Identity
from trellis.harness.journal import Journal, Replay
from trellis.harness.runtime import Runtime, _current


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


class Fixed:
    """A machine whose readings the test sets."""

    def __init__(self, cpu: float | None = None, memory: float | None = None) -> None:
        self.readings = {"cpu": cpu, "memory": memory}
        self.read = 0

    def cpu(self) -> float | None:
        self.read += 1
        return self.readings["cpu"]

    def memory(self) -> float | None:
        return self.readings["memory"]


def files(root: Path, **found: str) -> Path:
    for name, text in found.items():
        path = root / name.replace("__", "/")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return root


# --------------------------------------------------------------------------- AIMD
def test_the_limit_rises_by_one_a_round_while_children_answer() -> None:
    limit = Admission(start=4, ceiling=6)
    assert limit.limit == 4
    for _ in range(4):
        limit.ended(ok=True)
    assert limit.limit == 4  # 4 + 1/4 + ... just under 5
    limit.ended(ok=True)
    assert limit.limit == 5
    for _ in range(50):
        limit.ended(ok=True)
    assert limit.limit == 6  # the ceiling


def test_a_child_that_failed_does_not_raise_the_limit() -> None:
    limit = Admission(start=2)
    for _ in range(10):
        limit.ended(ok=False)
    assert limit.limit == 2


def test_pressure_halves_the_limit_once_per_cooldown_down_to_the_floor() -> None:
    clock = Clock()
    machine = Fixed(cpu=0.95)
    limit = Admission(start=16, ceiling=16, machine=machine, clock=clock)
    limit.ended(ok=True)
    assert limit.limit == 16 * BACKOFF
    limit.admitting()
    assert limit.limit == 8  # within the cooldown: one cut
    for _ in range(6):
        clock.now += COOLDOWN_SECONDS + SAMPLE_SECONDS
        limit.admitting()
    assert limit.limit == 1  # the floor
    machine.readings["cpu"] = 0.1
    clock.now += SAMPLE_SECONDS
    limit.ended(ok=True)
    assert limit._limit > 1


def test_memory_past_its_share_is_pressure_and_a_reading_stands_a_while() -> None:
    clock = Clock()
    machine = Fixed(cpu=None, memory=0.9)
    limit = Admission(machine=machine, clock=clock)
    assert limit.pressure() == "memory at 90% of the container's limit"
    machine.readings["memory"] = 0.1
    assert limit.pressure() is not None and machine.read == 1  # the last reading stands
    clock.now += SAMPLE_SECONDS
    assert limit.pressure() is None and machine.read == 2
    machine.readings["cpu"] = 0.9
    clock.now += SAMPLE_SECONDS
    assert limit.pressure() == "CPU at 90% of the container's quota"


def test_no_machine_is_no_pressure_from_it() -> None:
    assert Admission(machine=None).pressure() is None


def test_a_rate_limit_cuts_the_limit() -> None:
    limit = Admission(start=8, ceiling=8)
    limit.throttled()
    assert limit.limit == 4
    limit.throttled()  # the same burst
    assert limit.limit == 4


def test_a_provider_slowing_down_is_pressure() -> None:
    limit = Admission(start=8, ceiling=8)
    for _ in range(LATENCY_SAMPLES):
        limit.paced(2.0, 100)  # 0.02 s a token
    assert limit.pressure() is None
    for _ in range(10):
        limit.paced(10.0, 100)  # five times slower
    assert limit.pressure() == "the model provider is slowing down"
    limit.ended(ok=True)
    assert limit.limit == 4


def test_a_fast_provider_is_never_slow_and_unpaced_calls_are_not_counted() -> None:
    limit = Admission()
    for _ in range(LATENCY_SAMPLES):
        limit.paced(0.001, 100)
    for _ in range(10):
        limit.paced(0.01, 100)  # ten times slower, still 10,000 tokens a second
    assert limit.pressure() is None
    limit.paced(5.0, 0)
    limit.paced(-1.0, 10)
    assert limit._paced == LATENCY_SAMPLES + 10


@pytest.mark.parametrize(("start", "floor", "ceiling"), [(0, 1, 4), (5, 1, 4), (1, 2, 4)])
def test_a_limit_out_of_its_bounds_is_refused(start: int, floor: int, ceiling: int) -> None:
    with pytest.raises(ValueError, match="floor <= start <= ceiling"):
        Admission(start=start, floor=floor, ceiling=ceiling)


# --------------------------------------------------------------------------- the machine
def test_cgroup_v2_memory_is_the_working_set_of_its_limit(tmp_path: Path) -> None:
    root = files(
        tmp_path,
        **{
            "memory.max": "1000\n",
            "memory.current": "800\n",
            "memory.stat": "anon 600\ninactive_file 200\nactive_file 0\n",
        },
    )
    assert Machine(root, tmp_path / "none").memory() == pytest.approx(0.6)


def test_unlimited_memory_is_a_share_of_the_hosts(tmp_path: Path) -> None:
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal:  4 kB\nMemAvailable: 1 kB\nHugePages_Total: 0\nOdd:\n")
    v2 = files(tmp_path / "v2", **{"memory.max": "max", "memory.current": "2048"})
    assert Machine(v2, meminfo).memory() == pytest.approx(0.5)
    v1 = files(
        tmp_path / "v1",
        **{
            "memory__memory.limit_in_bytes": str(NO_LIMIT),
            "memory__memory.usage_in_bytes": "3072",
            "memory__memory.stat": "total_inactive_file 1024\n",
        },
    )
    assert Machine(v1, meminfo).memory() == pytest.approx(0.5)
    host = Machine(tmp_path / "nothing", meminfo)
    assert host.memory() == pytest.approx(0.75)


def test_cgroup_v1_memory_and_nothing_known(tmp_path: Path) -> None:
    v1 = files(
        tmp_path / "v1",
        **{"memory__memory.limit_in_bytes": "1000", "memory__memory.usage_in_bytes": "900"},
    )
    assert Machine(v1, tmp_path / "none").memory() == pytest.approx(0.9)
    nothing = Machine(tmp_path / "nothing", tmp_path / "none")
    assert nothing.memory() is None
    no_total = files(tmp_path / "v2", **{"memory.max": "max", "memory.current": "1"})
    assert Machine(no_total, tmp_path / "none").memory() is None
    partial = tmp_path / "partial"
    partial.write_text("MemTotal: 4 kB\n")
    assert Machine(tmp_path / "nothing", partial).memory() is None


def test_cgroup_cpu_is_the_share_of_its_quota_since_the_last_reading(tmp_path: Path) -> None:
    clock = Clock()
    root = files(tmp_path, **{"cpu.max": "200000 100000", "cpu.stat": "usage_usec 0\n"})
    machine = Machine(root, clock=clock)
    assert machine.cpu() is None  # the first reading
    assert machine.cpu() is None  # no time passed
    clock.now += 1.0
    files(root, **{"cpu.stat": "usage_usec 1000000\nuser_usec 1\n"})
    assert machine.cpu() == pytest.approx(0.5)  # one second of two cores


def test_cgroup_v1_cpu_and_its_quota(tmp_path: Path) -> None:
    clock = Clock()
    root = files(
        tmp_path,
        **{
            "cpu.stat": "user_usec 1\n",  # v2's file without its usage: not v2
            "cpuacct__cpuacct.usage": "0",
            "cpu__cpu.cfs_quota_us": "100000",
            "cpu__cpu.cfs_period_us": "100000",
        },
    )
    machine = Machine(root, clock=clock)
    machine.cpu()
    clock.now += 1.0
    files(root, **{"cpuacct__cpuacct.usage": str(500_000_000)})
    assert machine.cpu() == pytest.approx(0.5)


def test_without_a_quota_cpu_is_a_share_of_the_cores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(admission.os, "cpu_count", lambda: 4)
    clock = Clock()
    root = files(
        tmp_path,
        **{"cpu.max": "max 100000", "cpu.stat": "usage_usec 0", "cpu__cpu.cfs_quota_us": "-1"},
    )
    machine = Machine(root, clock=clock)
    machine.cpu()
    clock.now += 1.0
    files(root, **{"cpu.stat": "usage_usec 2000000"})
    assert machine.cpu() == pytest.approx(0.5)


def test_with_no_cgroup_cpu_is_the_load_of_the_cores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(admission.os, "getloadavg", lambda: (3.0, 0.0, 0.0))
    monkeypatch.setattr(admission.os, "cpu_count", lambda: 4)
    assert Machine(tmp_path).cpu() == pytest.approx(0.75)


def test_the_default_machine_reads_this_one() -> None:
    machine = Machine()
    memory = machine.memory()
    assert memory is None or memory >= 0
    machine.cpu()


# --------------------------------------------------------------------------- the gate
async def test_a_gate_holds_at_most_the_limit_in_slots_of_their_own() -> None:
    gate, limit = Gate(), Admission(start=2, ceiling=2)
    first = await gate.enter(limit, None)
    second = await gate.enter(limit, None)
    assert (first, second) == (0, 1)
    waiting = asyncio.create_task(gate.enter(limit, None))
    await asyncio.sleep(0.01)
    assert not waiting.done()
    await gate.leave(first)
    assert await waiting == 0  # the lowest free slot
    assert gate.peak == 2
    with pytest.raises(TimeoutError):
        await gate.enter(limit, 0.01)


def _runtime() -> tuple[Runtime, list[RunEvent]]:
    seen: list[RunEvent] = []
    events = RunEvents(AgentExecutionContext.create(tenant_id="t", agent_id="a"))
    events.listen(seen.append)
    agent: Any = SimpleNamespace(harness=SimpleNamespace(model_gateway=ProviderSDK()))
    identity = Identity(tenant="t", agent_id="a", run_id="run_1", user="u")
    return Runtime(identity=identity, agent=agent, events=events, replay=Replay(Journal())), seen


def _tree(seen: list[RunEvent]) -> list[tuple[str, Any]]:
    return [(e.data["name"], e.data.get("status")) for e in seen]


async def test_a_branch_is_on_the_tree_however_it_ends(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(admission, "ADMISSION", Admission(start=1, ceiling=1))
    parent, seen = _runtime()
    async with branch(parent, "run_a", "kid") as found:
        found.spent = {"tokens": 30, "cost": 0.5, "model_calls": 2}
        found.ended("success")
        found.ended("error")  # once
    assert _tree(seen) == [
        ("helper_queued", None),
        ("helper_started", None),
        ("helper_finished", "success"),
    ]
    assert seen[0].data["limit"] == 1
    finished = seen[2].data
    assert (finished["tokens"], finished["cost"], finished["agent_id"]) == (30, 0.5, "kid")
    for raised, status in [
        (RuntimeError("boom"), "error"),
        (AgentPaused("Which year?"), "paused"),
        (asyncio.CancelledError(), "cancelled"),
    ]:
        seen.clear()
        with pytest.raises(type(raised)):
            async with branch(parent, "run_b", "kid"):
                raise raised
        assert _tree(seen)[-1] == ("helper_finished", status)
    assert admission.gate(parent).taken == set()  # every slot given back


async def test_a_branch_with_no_slot_in_time_ends_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(admission, "ADMISSION", Admission(start=1, ceiling=1))
    parent, seen = _runtime()
    async with branch(parent, "run_a", "kid") as holding:
        parent.ends_at = time.monotonic() + 0.01  # what is left of the run's time
        with pytest.raises(TimeoutError):
            async with branch(parent, "run_b", "kid"):
                pass  # pragma: no cover - never admitted
        holding.ended("success")
    late = [e.data for e in seen if e.data.get("run_id") == "run_b"]
    assert [d["name"] for d in late] == ["helper_queued", "helper_finished"]
    assert late[-1]["status"] == "timeout"


# --------------------------------------------------------------------------- what feeds it
class Refused(Exception):
    """A model client's error: its status its own, or its response's."""

    def __init__(self, **fields: Any) -> None:
        super().__init__("refused")
        self.__dict__.update(fields)


def _reply(tokens: int) -> LLMResult:
    message = AIMessage(
        "hi", usage_metadata={"input_tokens": 1, "output_tokens": tokens, "total_tokens": 1}
    )
    return LLMResult(generations=[[ChatGeneration(message=message)]])


async def test_every_model_call_in_a_run_feeds_its_pace_and_429s(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    limit = Admission(start=8, ceiling=8)
    monkeypatch.setattr(admission, "ADMISSION", limit)
    runtime, _ = _runtime()
    callback = accounting.UsageAccounting()
    token = _current.set(runtime)
    try:
        call = uuid.uuid4()
        callback.on_chat_model_start({}, [[]], run_id=call)
        callback.on_llm_end(_reply(10), run_id=call)
        assert limit._paced == 1
        callback.on_llm_end(_reply(10), run_id=uuid.uuid4())  # never started: no pace
        assert limit._paced == 1
        failed = uuid.uuid4()
        callback.on_chat_model_start({}, [[]], run_id=failed)
        callback.on_llm_error(Refused(status_code=429), run_id=failed)
        assert limit.limit == 4
        callback.on_llm_error(Refused(response=SimpleNamespace(status_code=500)), run_id=failed)
        callback.on_llm_error(RuntimeError("down"), run_id=uuid.uuid4())
        assert limit.limit == 4
    finally:
        _current.reset(token)
