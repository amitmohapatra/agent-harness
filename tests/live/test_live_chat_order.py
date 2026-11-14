"""A conversation's turns across real server processes: three harness servers
(``tests/support/chat_server.py``, every adapter that runs here) behind no balancer but the test,
sharing a real agent-runs (its API and its ticker, separate processes) on a real PostgreSQL. A
chat's messages, wherever each arrives, are answered one at a time in the order they were sent,
each turn seeing the earlier ones; ``interrupt`` cancels the turn working on another server,
``steer`` waits its turn there; a turn paused for a person holds nobody up; a queued turn and the
servers' turns take turns; two chats of a user, and two users' threads named alike, run side by
side; a server killed while it holds a chat's turn frees it once its hold lapses, its run ended
``holder_lost``. Every test's runs are checked against the test plan's invariants
(``tests/support/invariants.py``). No model is called: every model is scripted.

It starts what it needs, given two variables of its own: ``TRELLIS_LIVE_POSTGRES``, a PostgreSQL
admin URL (``postgresql://postgres@127.0.0.1:5432/postgres``: a database of its own is created and
dropped), and ``TRELLIS_LIVE_RUNS_PYTHON``, a Python that imports ``agent_runs`` (the service, from
its checkout). Unset, it is skipped. agent-runs asks a stand-in key registry who the key is."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Final

import httpx
import pytest

from tests.support.invariants import Invariants, invariants  # noqa: F401 - an autouse fixture
from trellis.harness.conversation import EARLIER_RUNS
from trellis.runs import RunsClient

POSTGRES = os.environ.get("TRELLIS_LIVE_POSTGRES", "").strip()
RUNS_PYTHON = os.environ.get("TRELLIS_LIVE_RUNS_PYTHON", "").strip()
pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not (POSTGRES and RUNS_PYTHON),
        reason="needs TRELLIS_LIVE_POSTGRES and TRELLIS_LIVE_RUNS_PYTHON (agent-runs)",
    ),
]
ROOT = Path(__file__).resolve().parents[2]
KEY: Final = "chat-order-key"
#: how long a server's hold on a run lasts (a killed server's turn is freed this soon, plus a tick)
HOLD_SECONDS: Final = 6
SERVERS: Final = 3


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class Registry(BaseHTTPRequestHandler):
    """The memory service's key registry, as agent-runs asks it (``GET /v1/keys/self``): the
    one key, of the tenant a harness with no memory service runs in."""

    def do_GET(self) -> None:
        known = self.path == "/v1/keys/self" and self.headers.get("X-API-Key") == KEY
        body = {"key_id": "key_chat", "tenant_id": "default", "principal": "svc"}
        body |= {"role": "service", "may_act_as": ["*"]}
        self.send_response(200 if known else 401)
        self.send_header("content-type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(body if known else {"detail": "unknown"}).encode())

    def log_message(self, format: str, *args: Any) -> None:
        return


@dataclass
class Stack:
    folder: Path
    runs_url: str
    servers: list[str]
    processes: dict[str, subprocess.Popen[bytes]] = field(default_factory=dict)
    traffic: Path | None = None

    def turns(self) -> list[dict[str, Any]]:
        log = self.folder / "turns.jsonl"
        return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []

    def hold(self, question: str) -> None:
        (self.folder / f"hold-{question}").touch()

    def release(self, question: str) -> None:
        (self.folder / f"hold-{question}").unlink(missing_ok=True)


def _runs_env(folder: Path, port: int, database: str, registry: int) -> dict[str, str]:
    url = POSTGRES.replace("postgresql://", "postgresql+psycopg://").rsplit("/", 1)[0]
    return {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(folder),
        "TMPDIR": str(folder),
        "NO_PROXY": "*",
        "PYTHONUNBUFFERED": "1",
        "RUNS__SERVICE__ENVIRONMENT": "dev",
        "RUNS__SERVICE__HOST": "127.0.0.1",
        "RUNS__SERVICE__PORT": str(port),
        "RUNS__SERVICE__WORKERS": "1",
        "RUNS__MEMORY__URL": f"http://127.0.0.1:{registry}",
        "RUNS__DATABASE__URL": f"{url}/{database}",
        "RUNS__BLOB__ROOT": str(folder / "blob"),
        "RUNS__TICKER__HEARTBEAT_FILE": str(folder / "ticker.beat"),
        "RUNS__RATE_LIMIT__PER_MINUTE": "0",
    }


def _database(sql: str) -> None:
    script = "import psycopg, sys\nwith psycopg.connect(sys.argv[1], autocommit=True) as c: c.execute(sys.argv[2])"
    subprocess.run([RUNS_PYTHON, "-c", script, POSTGRES, sql], check=True, timeout=60)


def _python(script: str) -> str:
    """What ``script`` prints, run by agent-runs' Python."""
    done = subprocess.run(
        [RUNS_PYTHON, "-c", script], check=True, capture_output=True, text=True, timeout=60
    )
    return done.stdout.strip()


async def _up(*urls: str, seconds: float = 180) -> None:
    """Wait until every one of ``urls`` answers 200."""
    async with httpx.AsyncClient() as client, asyncio.timeout(seconds):
        for url in urls:
            while True:
                try:
                    if (await client.get(url)).status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(0.25)


@pytest.fixture(scope="module")
def stack(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Stack]:
    folder = tmp_path_factory.mktemp("chat-order")
    database = f"trellis_chat_order_{uuid.uuid4().hex[:8]}"
    registry = ThreadingHTTPServer(("127.0.0.1", 0), Registry)
    threading.Thread(target=registry.serve_forever, daemon=True).start()
    runs_port = free_port()
    env = _runs_env(folder, runs_port, database, registry.server_address[1])
    runs_url = f"http://127.0.0.1:{runs_port}"
    s = Stack(folder, runs_url, [], traffic=folder / "servers.traffic")
    logs = (folder / "logs").resolve()
    logs.mkdir()

    def started(name: str, args: list[str], environ: dict[str, str], cwd: Path | str) -> None:
        out = (logs / f"{name}.log").open("wb")
        s.processes[name] = subprocess.Popen(args, env=environ, stdout=out, stderr=out, cwd=cwd)

    _database(f"CREATE DATABASE {database}")
    try:
        checkout = _python(  # the service reads its alembic.ini there
            "import importlib.util as u, pathlib as p\n"
            "print(p.Path(u.find_spec('agent_runs').origin).parents[2])"
        )
        migrate = (
            "from alembic import command\nfrom alembic.config import Config\n"
            "from agent_runs.store.database import ALEMBIC_INI\n"
            "command.upgrade(Config(str(ALEMBIC_INI)), 'head')"
        )
        subprocess.run([RUNS_PYTHON, "-c", migrate], env=env, check=True, timeout=120, cwd=checkout)
        started("runs-api", [RUNS_PYTHON, "-m", "agent_runs"], env, checkout)
        started("runs-ticker", [RUNS_PYTHON, "-m", "agent_runs.ticker"], env, checkout)
        asyncio.run(_up(f"{runs_url}/health/live"))
        served = {**os.environ, "HOLD_SECONDS": str(HOLD_SECONDS), "RUNS_URL": runs_url}
        served |= {"SCEN_TRAFFIC": str(s.traffic), "TMPDIR": str(folder)}
        served.pop("MEMORY_URL", None)
        if shutil.which("kimi"):
            served["CHAT_KIMI"] = "1"
        for n in range(SERVERS):
            port = free_port()
            command = [sys.executable, "-m", "tests.support.chat_server", runs_url, KEY, str(port)]
            started(f"server-{n}", [*command, str(folder)], served, ROOT)
            s.servers.append(f"http://127.0.0.1:{port}")
        asyncio.run(_up(*(f"{u}/ready" for u in s.servers)))
        os.environ["RUNS_URL"] = runs_url  # what the invariants read the runs back from
        yield s
    finally:
        os.environ.pop("RUNS_URL", None)
        for process in s.processes.values():
            process.terminate()
        for process in s.processes.values():
            try:
                process.wait(20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        registry.shutdown()
        _database(f"DROP DATABASE IF EXISTS {database} WITH (FORCE)")


@pytest.fixture(autouse=True)
def recorded(stack: Stack, invariants: Invariants) -> None:  # noqa: F811
    """The servers' exchanges with agent-runs are the test's too (the invariants read them)."""
    assert stack.traffic is not None
    invariants.files.append(stack.traffic)


@pytest.fixture
async def http() -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(timeout=300) as client:
        yield client


def chat() -> str:
    return f"chat-{uuid.uuid4().hex[:8]}"


async def send(
    http: httpx.AsyncClient, server: str, question: str, thread: str, **given: Any
) -> dict[str, Any]:
    body = {"adapter": "function", "input": question, "user": "ada", "thread": thread} | given
    response = await http.post(f"{server}/run", json=body)
    assert response.status_code == 200, response.text
    return response.json()


async def started(stack: Stack, question: str, seconds: float = 60) -> None:
    async with asyncio.timeout(seconds):
        while not any(t["event"] == "start" and t["question"] == question for t in stack.turns()):  # noqa: ASYNC110
            await asyncio.sleep(0.05)


def of(stack: Stack, questions: list[str]) -> list[dict[str, Any]]:
    return [t for t in stack.turns() if t.get("question") in questions]


def one_at_a_time_in_order(stack: Stack, questions: list[str]) -> None:
    """Each question's turn started after the one before ended, in the order sent, each seeing
    the earlier turns (its questions and answers, the newest ten)."""
    turns = [t for t in of(stack, questions) if t["event"] in ("start", "end")]
    assert [(t["event"], t["question"]) for t in turns] == [
        (e, q) for q in questions for e in ("start", "end")
    ]
    starts = [t for t in turns if t["event"] == "start"]
    for n, turn in enumerate(starts):
        if turn["history"] is None:  # the session its framework continued holds them
            continue
        asked = [said for said in turn["history"] if said in questions]
        assert asked == questions[max(0, n - EARLIER_RUNS) : n], turn


async def burst(
    http: httpx.AsyncClient, stack: Stack, questions: list[str], thread: str, **given: Any
) -> list[dict[str, Any]]:
    """``questions`` sent in order, round the servers, each once the one before was accepted."""
    sent = []
    for n, question in enumerate(questions):
        server = stack.servers[n % len(stack.servers)]
        sent.append(asyncio.create_task(send(http, server, question, thread, **given)))
        await _accepted(stack, thread, n + 1)
    return list(await asyncio.gather(*sent))


async def _accepted(stack: Stack, thread: str, count: int) -> None:
    """Wait until agent-runs holds ``count`` runs of ``thread`` (the next one is sent after)."""
    async with RunsClient(stack.runs_url, api_key=KEY) as runs, asyncio.timeout(60):
        while len([s async for s in runs.iterate(thread_id=thread)]) < count:  # noqa: ASYNC110
            await asyncio.sleep(0.02)


# --------------------------------------------------------------------------- one chat, in order
async def test_two_messages_on_two_servers_are_answered_one_at_a_time(
    stack: Stack, http: httpx.AsyncClient
) -> None:
    thread = chat()
    stack.hold("one")
    first = asyncio.create_task(send(http, stack.servers[0], "one", thread))
    await started(stack, "one")
    second = asyncio.create_task(send(http, stack.servers[1], "two", thread))
    await asyncio.sleep(1.5)
    assert [t["question"] for t in of(stack, ["one", "two"])] == ["one"], "two waits"
    stack.release("one")
    results = await asyncio.gather(first, second)
    assert [r["status"] for r in results] == ["SUCCESS", "SUCCESS"]
    one_at_a_time_in_order(stack, ["one", "two"])


@pytest.mark.parametrize("count", [5, 20])
async def test_a_burst_over_three_servers_is_answered_one_at_a_time_in_order(
    stack: Stack, http: httpx.AsyncClient, count: int
) -> None:
    thread = chat()
    questions = [f"{thread} message {n}" for n in range(count)]
    stack.hold(questions[0])
    sending = asyncio.create_task(burst(http, stack, questions, thread))
    await started(stack, questions[0])
    stack.release(questions[0])
    results = await sending
    assert all(r["status"] == "SUCCESS" for r in results), results
    assert len({t["server"] for t in of(stack, questions)}) == SERVERS
    one_at_a_time_in_order(stack, questions)


# --------------------------------------------------------------------------- modes
async def test_an_interrupt_cancels_the_turn_working_on_another_server(
    stack: Stack, http: httpx.AsyncClient
) -> None:
    thread = chat()
    stack.hold("long")
    first = asyncio.create_task(send(http, stack.servers[0], "long", thread))
    await started(stack, "long")
    second = await send(http, stack.servers[1], "correction", thread, mode="interrupt")
    stopped = await first
    stack.release("long")
    assert stopped["status"] == "CANCELLED" and second["status"] == "SUCCESS"
    events = [(t["event"], t["question"]) for t in of(stack, ["long", "correction"])]
    assert ("cancelled", "long") in events  # its server stopped it
    turn = next(t for t in stack.turns() if t["question"] == "correction")
    assert "long" in turn["history"]  # the conversation went on from the cancelled question


async def test_a_steer_to_a_turn_on_another_server_waits_its_turn(
    stack: Stack, http: httpx.AsyncClient
) -> None:
    thread = chat()
    stack.hold("first")
    first = asyncio.create_task(send(http, stack.servers[0], "first", thread, adapter="react"))
    await started(stack, "first")
    steered = asyncio.create_task(
        send(http, stack.servers[1], "also", thread, adapter="react", mode="steer")
    )
    await asyncio.sleep(1.5)
    assert [t["question"] for t in of(stack, ["first", "also"])] == ["first"]
    stack.release("first")
    results = await asyncio.gather(first, steered)
    assert [r["status"] for r in results] == ["SUCCESS", "SUCCESS"]
    one_at_a_time_in_order(stack, ["first", "also"])  # its own turn, after


async def test_a_turn_paused_for_a_person_holds_nobody_up(
    stack: Stack, http: httpx.AsyncClient
) -> None:
    thread = chat()
    asked = await send(http, stack.servers[0], "refund o-77", thread)
    assert asked["status"] == "PAUSED"
    stack.hold("meanwhile")
    meanwhile = asyncio.create_task(send(http, stack.servers[1], "meanwhile", thread))
    await started(stack, "meanwhile")  # it did not wait for the person
    body = {"adapter": "function", "interrupt_id": asked["interrupt"]["interrupt_id"]}
    resumed = asyncio.create_task(http.post(f"{stack.servers[2]}/resume", json=body))
    await asyncio.sleep(1.5)
    assert not [t for t in stack.turns() if t["event"] == "refunded"], "it continues after"
    stack.release("meanwhile")
    assert (await meanwhile)["status"] == "SUCCESS"
    assert (await resumed).json()["status"] == "SUCCESS"
    turns = [(t["event"], t["question"]) for t in stack.turns()]
    assert turns.index(("end", "meanwhile")) < turns.index(("refunded", "o-77"))


async def test_queued_turns_and_the_servers_turns_take_turns(
    stack: Stack, http: httpx.AsyncClient
) -> None:
    thread = chat()
    stack.hold("queued one")
    body = {"adapter": "function", "user": "ada", "thread": thread}
    await http.post(f"{stack.servers[0]}/start", json=body | {"input": "queued one"})
    await started(stack, "queued one")  # a server's worker claimed it
    direct = asyncio.create_task(send(http, stack.servers[1], "direct two", thread))
    await _accepted(stack, thread, 2)
    await http.post(f"{stack.servers[2]}/start", json=body | {"input": "queued three"})
    await asyncio.sleep(1.0)
    stack.release("queued one")
    assert (await direct)["status"] == "SUCCESS"
    await started(stack, "queued three")
    questions = ["queued one", "direct two", "queued three"]
    async with asyncio.timeout(60):
        while len([t for t in of(stack, questions) if t["event"] == "end"]) < 3:  # noqa: ASYNC110
            await asyncio.sleep(0.1)
    one_at_a_time_in_order(stack, questions)


# --------------------------------------------------------------------------- side by side
async def test_two_chats_of_a_user_and_two_users_with_one_thread_id_run_side_by_side(
    stack: Stack, http: httpx.AsyncClient
) -> None:
    thread, other = chat(), chat()
    for question in ("ada here", "ada there", "bob here"):
        stack.hold(question)
    sent = [
        asyncio.create_task(send(http, stack.servers[0], "ada here", thread)),
        asyncio.create_task(send(http, stack.servers[1], "ada there", other)),
        asyncio.create_task(send(http, stack.servers[2], "bob here", thread, user="bob")),
    ]
    for question in ("ada here", "ada there", "bob here"):
        await started(stack, question)  # all three at once
    bob = next(t for t in stack.turns() if t["question"] == "bob here")
    assert not bob["history"]  # ada's thread named alike is not bob's
    for question in ("ada here", "ada there", "bob here"):
        stack.release(question)
    assert all(r["status"] == "SUCCESS" for r in await asyncio.gather(*sent))


# --------------------------------------------------------------------------- every adapter
@pytest.mark.timeout(900)  # 180 turns, one at a time
@pytest.mark.parametrize("count", [5, 20])
async def test_every_adapter_answers_a_burst_over_three_servers_one_at_a_time_in_order(
    stack: Stack, http: httpx.AsyncClient, count: int
) -> None:
    adapters = (await http.get(f"{stack.servers[0]}/ready")).json()["adapters"]
    for adapter in adapters:
        thread = chat()
        questions = [f"{thread} {adapter} {n}" for n in range(count)]
        stack.hold(questions[0])
        sending = asyncio.create_task(
            burst(http, stack, questions, thread, adapter=adapter, stream=adapter != "function")
        )
        await started(stack, questions[0], seconds=120)
        stack.release(questions[0])
        results = await sending
        assert all(r["status"] == "SUCCESS" for r in results), (adapter, results)
        turns = [t for t in of(stack, questions) if t["event"] in ("start", "end")]
        if adapter == "kimi":  # its session may hold an earlier turn's result: it calls step less
            asked = [t["question"] for t in turns if t["event"] == "start"]
            assert asked == questions[: len(asked)]
            continue
        one_at_a_time_in_order(stack, questions)
    for server in stack.servers:  # the Claude processes the servers keep between turns
        assert (await http.post(f"{server}/close")).status_code == 200


async def test_a_chat_message_through_the_chat_protocol_waits_its_turn(
    stack: Stack, http: httpx.AsyncClient
) -> None:
    thread = chat()
    stack.hold("over chat one")
    first = asyncio.create_task(send(http, stack.servers[0], "over chat one", thread))
    await started(stack, "over chat one")
    identity = json.dumps({"tenant_id": "default", "user_id": "ada"})
    agui = asyncio.create_task(
        http.post(
            f"{stack.servers[1]}/agui/run",
            headers={"X-Trellis-Identity": identity},
            json={
                "threadId": thread,
                "messages": [{"id": "m1", "role": "user", "content": "over chat two"}],
            },
        )
    )
    await asyncio.sleep(1.5)
    assert [t["question"] for t in of(stack, ["over chat one", "over chat two"])] == [
        "over chat one"
    ]
    stack.release("over chat one")
    assert (await first)["status"] == "SUCCESS"
    assert (await agui).status_code == 200
    one_at_a_time_in_order(stack, ["over chat one", "over chat two"])


# --------------------------------------------------------------------------- a killed server
async def test_a_server_killed_holding_a_turn_frees_it_once_its_hold_lapses(
    stack: Stack,
    http: httpx.AsyncClient,
    invariants: Invariants,  # noqa: F811
) -> None:
    """Last: it kills the third server."""
    thread = chat()
    stack.hold("doomed")
    doomed = asyncio.create_task(send(http, stack.servers[2], "doomed", thread))
    await started(stack, "doomed")
    after = asyncio.create_task(send(http, stack.servers[0], "after the crash", thread))
    await _accepted(stack, thread, 2)
    killed = stack.processes["server-2"]
    killed.send_signal(signal.SIGKILL)
    killed.wait()
    stack.servers.pop()
    with pytest.raises(httpx.HTTPError):
        await doomed
    began = time.monotonic()
    answered = await asyncio.wait_for(after, 60)
    assert answered["status"] == "SUCCESS"
    assert time.monotonic() - began < 30
    async with RunsClient(stack.runs_url, api_key=KEY) as runs:
        found = [s async for s in runs.iterate(thread_id=thread)]
        lost = [await runs.get(s.run_id) for s in found if s.run_id != answered["run_id"]]
    assert [(r.status.value, r.error.code if r.error else None) for r in lost if r] == [
        ("ERROR", "holder_lost")
    ]
    stack.release("doomed")
