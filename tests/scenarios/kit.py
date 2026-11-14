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


#: the thread of every ``wait`` call begun in this process (a gate: :func:`waiting`)
BEGUN: list[str | None] = []
#: every ``wait`` call made in this process: (thread, started, ended), monotonic seconds
WAITS: list[tuple[str | None, float, float]] = []


@tool(side_effects="read")
async def wait(seconds: float) -> str:
    """Wait a while (a slow read)."""
    from trellis import current

    began = time.monotonic()
    run = current()
    BEGUN.append(run.thread if run is not None else None)
    await asyncio.sleep(seconds)
    WAITS.append((run.thread if run is not None else None, began, time.monotonic()))
    return f"waited {seconds}"


async def waiting(thread: str, within: float = 120.0) -> None:
    """Until a ``wait`` call of ``thread`` is under way in this process: its run is in its slow
    call (a gate, where a sleep would guess)."""

    async def begun() -> bool:
        return thread in BEGUN

    assert await eventually(begun, within=within, every=0.05), f"no slow call began in {thread}"


async def reached(
    h: Harness, run_id: str, kind: str, tool: str | None = None, within: float = 180.0
) -> None:
    """Until run ``run_id`` — in whatever process — has an event of ``kind`` (about ``tool``, when
    named) in agent-runs' event log: a gate on what the run did."""
    url, key = os.environ["RUNS_URL"], h.settings.api_key or ""

    async def there() -> bool:
        async with httpx.AsyncClient(base_url=url, headers={"X-Api-Key": key}) as c:
            got = await c.get(f"/v1/runs/{run_id}/events", params={"limit": 500})
        return got.status_code == 200 and any(
            p["event"]["type"] == kind
            and (tool is None or (p["event"].get("data") or {}).get("tool") == tool)
            for p in got.json()
        )

    assert await eventually(there, within=within, every=0.2), f"{run_id} never reached {kind}"


async def calling(h: Harness, run_id: str, tool: str = "wait", within: float = 180.0) -> None:
    """Until run ``run_id`` has begun a call of ``tool``."""
    await reached(h, run_id, "TOOL_CALL_START", tool, within)


async def runs_of(h: Harness, agent_id: str, count: int, within: float = 60.0) -> list[RunRecord]:
    """The agent's runs once there are ``count`` of them, as agent-runs keeps them."""
    found: list[Any] = []

    async def there() -> bool:
        found[:] = [r async for r in h.runs.iterate(agent_id=agent_id)]
        return len(found) >= count

    assert await eventually(there, within=within, every=0.1), (agent_id, len(found))
    return [await record(h, r.run_id) for r in found]


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


def asked_id(result: Any) -> str:
    """The id of the question a paused run (a ``Result``) waits on."""
    assert result.interrupt is not None, (result.status, result.error)
    return str(result.interrupt.interrupt_id)


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
        self.port = int(spec.get("port") or 0)
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


# --------------------------------------------------------------------------- Claude, in the background
#: a subagent started in the background by the scripted Claude CLI: task ``t1`` (or ``task``)
def background(task: str = "t1", prompt: str = "research") -> dict[str, Any]:
    return {"background": {"id": task, "agent": "researcher", "prompt": prompt}}


class ClaudeChat:
    """A wrapped Claude agent whose scripted CLI answers each turn of a session with the script
    of its number (``turns``), its background tasks driven from the test through files:
    ``call`` has a task make a tool call, ``finish`` ends it with its summary."""

    def __init__(
        self, h: Harness, tmp: Path, agent_id: str, turns: list[list[dict[str, Any]]], **wrap: Any
    ) -> None:
        from claude_agent_sdk import ClaudeAgentOptions

        from tests.scenarios.brains import CLI, SYSTEM
        from tests.support.arm import arm

        self.h, self.tasks = h, tmp / "tasks"
        self.tasks.mkdir(exist_ok=True)
        env = {
            "FAKE_CLAUDE_TURNS": json.dumps(turns),
            "FAKE_CLAUDE_RECORD": str(tmp / "cli.json"),
            "FAKE_CLAUDE_SESSIONS": str(tmp),
            "FAKE_CLAUDE_TASKS": str(self.tasks),
        }
        options = ClaudeAgentOptions(cli_path=CLI, system_prompt=SYSTEM, env=env)
        self.agent = arm(h.wrap(options, id=agent_id, **wrap), TOOLS)

    async def children(self, run_id: str) -> dict[str, RunRecord]:
        """The background runs a turn started, by what each was asked."""
        found = [r async for r in self.h.runs.iterate(parent_run_id=run_id)]
        return {str(r.input): r for r in [await record(self.h, c.run_id) for c in found]}

    async def call(self, step: dict[str, Any], task: str = "t1") -> None:
        """Task ``task`` makes the tool call ``step`` now."""
        (self.tasks / f"{task}.call").write_text(json.dumps(step))

    async def called(self, task: str = "t1", within: float = 60.0) -> dict[str, Any]:
        """What task ``task``'s call was answered, once it was."""
        answered = self.tasks / f"{task}.called"

        async def there() -> bool:
            return answered.exists()

        assert await eventually(there, within=within, every=0.1), (
            f"{task}'s call was never answered"
        )
        return dict(json.loads(answered.read_text()))

    async def finish(
        self, run_id: str, summary: str, task: str = "t1", within: float = 60.0
    ) -> RunRecord:
        """Task ``task`` ends with ``summary``; its run's record once it ended."""
        (self.tasks / f"{task}.done").write_text(summary)
        return await self.ended(run_id, within=within)

    async def ended(self, run_id: str, within: float = 60.0) -> RunRecord:
        async def over() -> bool:
            return (await record(self.h, run_id)).status.value not in (
                "RUNNING",
                "PAUSED",
                "QUEUED",
            )

        assert await eventually(over, within=within), await record(self.h, run_id)
        return await record(self.h, run_id)


def chat_app(agent: Any) -> Any:
    """A FastAPI app serving ``agent``'s chat (AG-UI), the caller named by the ``x-user`` header
    (a deployment's edge sets ``x-trellis-identity``)."""
    from fastapi import FastAPI, Request

    app = FastAPI()

    def user_of(request: Request) -> str:
        return request.headers["x-user"]

    agent.serve_chat(app, identity=user_of)
    return app


class Chat:
    """One user's chat with one agent over AG-UI, on whichever server ``port`` names."""

    def __init__(self, agent_id: str, user: str, thread: str) -> None:
        self.agent_id, self.thread = agent_id, thread
        self.client = httpx.AsyncClient(headers={"x-user": user}, timeout=300)

    async def post(self, port: int, **body: Any) -> list[tuple[int, dict[str, Any]]]:
        sent = await self.client.post(
            f"http://127.0.0.1:{port}/agui/{self.agent_id}/run",
            json={"threadId": self.thread, **body},
        )
        assert sent.status_code == 200, sent.text
        from trellis.harness.agui.sse import decode

        return decode(sent.text)

    async def say(self, port: int, text: str, **more: Any) -> list[tuple[int, dict[str, Any]]]:
        message = {"id": text[:16], "role": "user", "content": text}
        return await self.post(port, messages=[message], **more)
