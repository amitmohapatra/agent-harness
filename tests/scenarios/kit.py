"""The scenarios' tools and helpers: tools every adapter is given, a ledger of their side effects
(a file when ``SCEN_LEDGER`` names one, so a worker process counts too), and reading back what
the services keep."""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any, Final

import httpx

from tests.scenarios.brains import TOKEN, parse
from trellis import Harness, tool
from trellis.contracts import RunRecord
from trellis.memory import MemoryContext

LEDGER: list[str] = []


def _book(entry: str) -> None:
    LEDGER.append(entry)
    path = os.environ.get("SCEN_LEDGER")
    if path:
        with Path(path).open("a") as out:
            out.write(entry + "\n")


def ledger() -> list[str]:
    path = os.environ.get("SCEN_LEDGER")
    if path and Path(path).exists():
        return Path(path).read_text().splitlines()
    return list(LEDGER)


@tool(side_effects="read")
def lookup(key: str) -> str:
    """Look a key up."""
    return f"value-of-{key}"


#: every ``wait`` call made in this process: (thread, started, ended), monotonic seconds
WAITS: list[tuple[str | None, float, float]] = []


@tool(side_effects="read")
async def wait(seconds: float) -> str:
    """Wait a while (a slow read)."""
    from trellis import current

    began = time.monotonic()
    await asyncio.sleep(seconds)
    run = current()
    WAITS.append((run.thread if run is not None else None, began, time.monotonic()))
    return f"waited {seconds}"


def overlapped(threads: set[str]) -> bool:
    """Whether ``wait`` calls of two of ``threads`` ran at the same time."""
    calls = [w for w in WAITS if w[0] in threads]
    return any(a[0] != b[0] and a[1] < b[2] and b[1] < a[2] for a in calls for b in calls)


@tool(side_effects="write")
def note(text: str) -> str:
    """Write a note (a write: asks once a conversation)."""
    _book(f"note:{text}")
    return f"noted {text}"


@tool(side_effects="irreversible")
def transfer(amount: int, to: str, meta: dict[str, Any] | None = None) -> str:
    """Transfer money (irreversible: asks every time)."""
    _book(f"transfer:{amount}:{to}:{json.dumps(meta or {}, sort_keys=True)}")
    return f"transferred {amount} to {to}"


TOOLS: Final = [lookup, wait, note, transfer]


async def eventually(
    check: Callable[[], Awaitable[bool]], *, within: float = 60.0, every: float = 0.5
) -> bool:
    deadline = time.monotonic() + within
    while True:
        if await check():
            return True
        if time.monotonic() >= deadline:
            return False
        await asyncio.sleep(every)


async def thread_scope(h: Harness, *, user: str, agent_id: str, thread: str) -> MemoryContext:
    """The memory service in a conversation's scope, as the agent reads it."""
    assert h.memory is not None
    return h.memory.client.bind(
        tenant_id=await h._tenant(),
        user_id=user,
        agent_id=agent_id,
        thread_id=await h.conversation(user=user, thread=thread),
    )


async def user_scope(h: Harness, *, user: str, agent_id: str) -> MemoryContext:
    assert h.memory is not None
    return h.memory.client.bind(tenant_id=await h._tenant(), user_id=user, agent_id=agent_id)


async def history(h: Harness, *, user: str, agent_id: str, thread: str) -> list[tuple[str, str]]:
    """The conversation as the memory service keeps it: (role, content), oldest first."""
    await h.writes.drain()
    scope = await thread_scope(h, user=user, agent_id=agent_id, thread=thread)
    return [(str(m.role), m.content) for m in await scope.history(limit=500)]


def answered(messages: Sequence[tuple[str, str]]) -> list[str]:
    """The tokens of the assistant messages, in order (each one's ``mine``)."""
    return [parse(c).mine for r, c in messages if r.upper() == "ASSISTANT" and "ANS " in c]


def asked(messages: Sequence[tuple[str, str]], tokens: Sequence[str]) -> list[str]:
    """The tokens of ``tokens`` the user messages hold, in order (one per user message)."""
    wanted = set(tokens)
    return [
        t
        for role, content in messages
        if role.upper() == "USER"
        for t in TOKEN.findall(content)
        if t in wanted
    ]


async def record(h: Harness, run_id: str) -> RunRecord:
    """The run as agent-runs keeps it, read over its own API and parsed as the contract."""
    url = os.environ["RUNS_URL"]
    async with httpx.AsyncClient(
        base_url=url, headers={"X-Api-Key": h.settings.api_key or ""}
    ) as c:
        got = await c.get(f"/v1/runs/{run_id}")
        got.raise_for_status()
        return RunRecord.model_validate(got.json())


# --------------------------------------------------------------------------- processes
class Served:
    """A harness process (``tests.scenarios.serve``): its ``proc``, its ``port``, and its
    traffic's contract violations (``violations()``)."""

    def __init__(self, spec: dict[str, Any], tmp: Path, name: str) -> None:
        import subprocess
        import sys

        from tests.support import invariants

        self.ready = tmp / f"{name}.ready"
        current = invariants.current()
        self.out = current.traffic_file(name) if current is not None else tmp / f"{name}.traffic"
        self.log = tmp / f"{name}.log"
        self.port = spec.get("port")
        env = {
            **os.environ,
            "SCEN_SPEC": json.dumps({"tmp": str(tmp / name), **spec}),
            "SCEN_READY": str(self.ready),
            "SCEN_TRAFFIC": str(self.out),
        }
        (tmp / name).mkdir(parents=True, exist_ok=True)
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "tests.scenarios.serve"],
            env=env,
            stdout=self.log.open("w"),
            stderr=subprocess.STDOUT,
        )

    async def started(self, within: float = 120.0) -> Served:
        async def up() -> bool:
            assert self.proc.poll() is None, self.log.read_text()[-3000:]
            return self.ready.exists()

        assert await eventually(up, within=within, every=0.2), "the server never started"
        return self

    def kill(self) -> None:
        import signal

        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGKILL)
            self.proc.wait(10)

    def stop(self) -> None:
        import signal

        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(30)
            except Exception:
                self.kill()

    def violations(self) -> list[str]:
        """The contract violations of its traffic (the invariants read the rest)."""
        if not self.out.exists():
            return []
        return [
            json.loads(x)["violation"]
            for x in self.out.read_text().splitlines()
            if '"violation"' in x
        ]


def free_port() -> int:
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
