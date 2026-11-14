"""The feature suite: one purpose-level proof per documented feature, end to end against the
REAL memory service and the REAL agent-runs (two API replicas and the ticker), every model
scripted (``tests/support/planned.py``; the Claude Agent SDK drives the scripted CLI, Kimi Code
is the real ``kimi acp`` against a planned endpoint): nothing but the model is a stand-in, and
no model is paid for. Each test is tagged with its ID in the master test plan (``MEM-..``,
``RUN-..``, ``SCH``, ``FS``, ``SURF``, ``CON``, ``HAR-..``) by the ``plan`` marker, and
``docs/research/FEATURE_COVERAGE.md`` maps every plan ID to its tests.

A test proves what a user relies on, read back from the services' own state: the rows in the
memory and run databases (``FEAT_MEMORY_DSN`` / ``FEAT_RUNS_DSN``, read only, through ``psql``)
as well as their APIs.

``make test-features`` (``pytest -m features tests/features``) runs it with the deployment's
variables: ``MEMORY_URL``, ``RUNS_URL`` (and ``FEAT_RUNS_URL_2``, a second replica of the same
agent-runs), ``TRELLIS_API_KEY`` (a service key of the tenant), ``TRELLIS_ADMIN_KEY`` (its
administrator's), ``TRELLIS_API_KEY_OTHER`` (a service key of a second tenant),
``FEAT_KEY_ALICE`` (a service key that may act only as ``user:alice``), ``FEAT_PLATFORM_KEY``
(the memory service's platform key). Without them every test is skipped, never failed.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final
from urllib.parse import urlsplit

import httpx
import pytest

from tests.support.adapters import BUILDERS, SYSTEM
from tests.support.invariants import invariants  # noqa: F401  (INV-1..10 at every teardown)
from tests.support.planned import FINAL, Call, PlannedChat
from trellis import Harness, ReAct, Runtime, Settings
from trellis.harness.automatic import JournalBackend
from trellis.harness.tools.convert import text_of
from trellis.memory import MemoryClient, MemoryContext

#: How long a service call may take here (the services share one busy machine).
TIMEOUT: Final = 60.0


def _env(name: str) -> str | None:
    value = os.environ.get(name, "").strip()
    return value or None


def _reachable(url: str | None) -> bool:
    if url is None:
        return False
    try:
        return httpx.get(url.rstrip("/") + "/health/live", timeout=3.0).status_code < 500
    except httpx.HTTPError:
        return False


MEMORY_URL = _env("MEMORY_URL")
RUNS_URL = _env("RUNS_URL")
RUNS_URL_2 = _env("FEAT_RUNS_URL_2")
API_KEY = _env("TRELLIS_API_KEY")
ADMIN_KEY = _env("TRELLIS_ADMIN_KEY")
OTHER_KEY = _env("TRELLIS_API_KEY_OTHER")
ALICE_KEY = _env("FEAT_KEY_ALICE")
PLATFORM_KEY = _env("FEAT_PLATFORM_KEY")
MEMORY_DSN = _env("FEAT_MEMORY_DSN")
RUNS_DSN = _env("FEAT_RUNS_DSN")
UP = API_KEY is not None and _reachable(MEMORY_URL) and _reachable(RUNS_URL)

#: every adapter (``react_da``: ``ReAct`` on the Deep Agents backend)
ADAPTERS: Final = (
    "function",
    "react",
    "react_da",
    "langgraph",
    "middleware",
    "graph",
    "deepagents",
    "openai_agents",
    "claude_agent_sdk",
    "kimi",
)
#: the adapters whose model reads a pushed memory context (a function reads ``agent.context``)
PUSHED: Final = ADAPTERS


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    here = Path(__file__).parent
    skip = pytest.mark.skip(
        reason="needs the memory service and agent-runs (MEMORY_URL, RUNS_URL, TRELLIS_API_KEY)"
    )
    for item in items:
        if Path(str(item.fspath)).is_relative_to(here):
            item.add_marker(pytest.mark.features)
            if not UP:
                item.add_marker(skip)


# --------------------------------------------------------------------------- harnesses
def settings(**changes: object) -> Settings:
    return Settings(**{**Settings.from_env().model_dump(), **changes})


class Owned(Harness):
    """A harness that closes the memory client it was given (a deployment's would be its own)."""

    async def aclose(self) -> None:
        await super().aclose()
        if self.memory is not None:
            await self.memory.client.aclose()


def harness(*, key: str | None = None, memory: bool = True, **changes: object) -> Harness:
    """A harness for the stack (``key``: another key than the tenant's service key)."""
    if key is not None:
        changes["api_key"] = key
    s = settings(**changes)
    client: Any = (
        MemoryClient(s.memory_url, api_key=s.api_key, timeout=TIMEOUT)
        if memory and s.memory_url is not None
        else (False if not memory else None)
    )
    return Owned(config=s, memory=client)


@pytest.fixture
def uid() -> str:
    """A name of this test's own: the services keep everything across tests."""
    return uuid.uuid4().hex[:8]


@pytest.fixture
async def h() -> AsyncIterator[Harness]:
    async with harness() as made:
        yield made


async def scope(
    h: Harness, *, user: str, agent: str | None = None, thread: str | None = None
) -> MemoryContext:
    """The memory service in a user's (or a conversation's) scope, as the agent reads it."""
    assert h.memory is not None
    bound: dict[str, Any] = {"tenant_id": await h._tenant(), "user_id": user}
    if agent is not None:
        bound["agent_id"] = agent
    if thread is not None:
        bound["thread_id"] = await h.conversation(user=user, thread=thread)
    return h.memory.client.bind(**bound)


async def eventually(
    check: Callable[[], Awaitable[Any]], *, within: float = 90.0, every: float = 0.5
) -> Any:
    """``check``'s first truthy answer, or its last answer once ``within`` seconds passed."""
    deadline = time.monotonic() + within
    while True:
        got = await check()
        if got or time.monotonic() >= deadline:
            return got
        await asyncio.sleep(every)


# --------------------------------------------------------------------------- the databases
def sql(dsn: str | None, query: str, *params: str) -> list[dict[str, Any]]:
    """Rows of ``query`` read straight from a service's database (read only), as dicts.
    ``params`` replace ``:1``, ``:2``... as quoted literals."""
    assert dsn is not None, "FEAT_MEMORY_DSN / FEAT_RUNS_DSN name the services' databases"
    for i, value in enumerate(params, 1):
        query = query.replace(f":{i}", "'" + value.replace("'", "''") + "'")
    wrapped = f"select coalesce(json_agg(t), '[]'::json) from ({query}) t"
    out = subprocess.run(
        ["psql", dsn, "-qAt", "-v", "ON_ERROR_STOP=1", "-c", wrapped],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    return json.loads(out.stdout)


def memory_sql(query: str, *params: str) -> list[dict[str, Any]]:
    return sql(MEMORY_DSN, query, *params)


def runs_sql(query: str, *params: str) -> list[dict[str, Any]]:
    return sql(RUNS_DSN, query, *params)


# --------------------------------------------------------------------------- the adapters
@dataclass
class Built:
    """An adapter built to follow a plan: the target, the tools to arm it with, and what its
    model was sent (``said()``: one text)."""

    target: Any
    tools: list[Any]
    seen: list[Any] = field(default_factory=list)
    function_saw: list[str] = field(default_factory=list)

    def said(self) -> str:
        parts = list(self.function_saw)
        for model in self.seen:
            if isinstance(model, Path):
                parts.append(model.read_text() if model.exists() else "")
            else:
                parts.append(model.said())
        return "\n".join(parts)


async def build(
    adapter: str,
    h: Harness,
    tools: list[Any],
    tmp: Path,
    plan: list[Call],
    *,
    system: str = SYSTEM,
) -> Built:
    """``adapter`` following ``plan`` with ``tools``. A function target answers with what it
    read of the pushed context (``agent.context``), so its "model" said it too."""
    built = Built(target=None, tools=tools)
    if adapter == "function":
        saw = built.function_saw

        async def follow(input: Any, agent: Runtime) -> str:
            saw.append(f"{input}\n{agent.context or ''}")
            results = [text_of(await agent.tools.call(name, **args)) for name, args in plan]
            return FINAL.replace("{last}", results[-1] if results else "")

        built.target = follow
        return built
    if adapter == "react_da":
        model = PlannedChat(plan)
        built.seen.append(model)
        built.target = ReAct(system_prompt=system, model=model, backend=JournalBackend())
        return built
    built.target, built.tools = await BUILDERS[adapter](
        h, tools, tmp, plan, system=system, seen=built.seen
    )
    return built


# --------------------------------------------------------------------------- the wire
@dataclass
class Exchange:
    service: str
    method: str
    path: str
    status: int | None
    request: httpx.Request
    response: httpx.Response | None

    def body(self) -> Any:
        try:
            return json.loads(self.request.content or b"null")
        except ValueError:
            return None


class Traffic:
    """Every exchange this process had with the memory service and agent-runs."""

    def __init__(self) -> None:
        self.hosts = {
            urlsplit(url).netloc: name
            for url, name in ((MEMORY_URL, "memory"), (RUNS_URL, "runs"), (RUNS_URL_2, "runs"))
            if url
        }
        self.seen: list[Exchange] = []

    def note(self, request: httpx.Request, response: httpx.Response | None) -> None:
        service = self.hosts.get(request.url.netloc.decode())
        if service is None or request.url.path.startswith("/health"):
            return
        self.seen.append(
            Exchange(
                service,
                request.method,
                request.url.path,
                None if response is None else response.status_code,
                request,
                response,
            )
        )

    def of(self, service: str, method: str, path: str) -> list[Exchange]:
        return [x for x in self.seen if (x.service, x.method, x.path) == (service, method, path)]


@pytest.fixture
def traffic(monkeypatch: pytest.MonkeyPatch) -> Iterator[Traffic]:
    """What this test's harnesses sent the two services, and what they answered."""
    recorded = Traffic()
    send = httpx.AsyncClient.send

    async def noted(self: httpx.AsyncClient, request: httpx.Request, **kwargs: Any) -> Any:
        response = await send(self, request, **kwargs)
        recorded.note(request, response)
        return response

    monkeypatch.setattr(httpx.AsyncClient, "send", noted)
    yield recorded


def runs_http(key: str | None = None, *, url: str | None = None) -> httpx.AsyncClient:
    """agent-runs over its own HTTP API (``url``: another replica)."""
    return httpx.AsyncClient(
        base_url=url or RUNS_URL or "",
        headers={"X-API-Key": key or API_KEY or ""},
        timeout=TIMEOUT,
    )


def memory_http(key: str | None = None) -> httpx.AsyncClient:
    """The memory service over its own HTTP API."""
    return httpx.AsyncClient(
        base_url=MEMORY_URL or "", headers={"X-API-Key": key or API_KEY or ""}, timeout=TIMEOUT
    )


@pytest.fixture(autouse=True)
def fresh_ledger() -> Iterator[None]:
    """Each test books its own side effects (``tests.features.kit``)."""
    from tests.features import kit

    kit.LEDGER.clear()
    yield
    kit.LEDGER.clear()


# --------------------------------------------------------------------------- a faulty network
class FaultProxy:
    """A proxy in front of the memory service, on a port of its own, that injects what the
    network and an overloaded service do: ``rules[(method, path_prefix)]`` is ``"504"`` (the
    service's own deadline answer, the request never forwarded), ``"504-after"`` (forwarded
    and committed, then answered 504: the deadline firing after the work), or a count
    ``("504", n)`` for the first ``n`` matching requests. ``down()`` stops it (connections
    refused), ``up()`` starts it again on the same port."""

    def __init__(self, target: str) -> None:
        import socket

        self.target = target.rstrip("/")
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        self.port = sock.getsockname()[1]
        sock.close()
        self.url = f"http://127.0.0.1:{self.port}"
        self.rules: dict[tuple[str, str], Any] = {}
        self.hits: list[tuple[str, str, int]] = []
        self._server: Any = None
        self._thread: Any = None

    def _rule(self, method: str, path: str) -> str | None:
        for (m, prefix), rule in self.rules.items():
            if m == method and path.startswith(prefix):
                if isinstance(rule, tuple):
                    kind, left = rule
                    if left <= 0:
                        return None
                    self.rules[(m, prefix)] = (kind, left - 1)
                    return kind
                return rule
        return None

    def _app(self) -> Any:
        from starlette.applications import Starlette
        from starlette.requests import Request
        from starlette.responses import Response
        from starlette.routing import Route

        proxy = self

        def timeout(path: str) -> Response:
            body = {
                "type": "urn:trellis:problem:timeout",
                "title": "Timeout",
                "status": 504,
                "detail": "The request exceeded its 5 s read deadline",
                "instance": path,
                "code": "TIMEOUT",
                "retryable": True,
            }
            return Response(json.dumps(body), 504, media_type="application/problem+json")

        async def forward(request: Request) -> Response:
            path = request.url.path
            rule = proxy._rule(request.method, path)
            if rule == "504":
                proxy.hits.append((request.method, path, 504))
                return timeout(path)
            headers = {
                k: v
                for k, v in request.headers.items()
                if k.lower() not in ("host", "content-length")
            }
            async with httpx.AsyncClient(base_url=proxy.target, timeout=TIMEOUT) as client:
                got = await client.request(
                    request.method,
                    path,
                    params=request.query_params,
                    content=await request.body(),
                    headers=headers,
                )
            if rule == "504-after":
                proxy.hits.append((request.method, path, 504))
                return timeout(path)
            proxy.hits.append((request.method, path, got.status_code))
            drop = {"content-length", "content-encoding", "transfer-encoding", "connection"}
            return Response(
                got.content,
                got.status_code,
                headers={k: v for k, v in got.headers.items() if k.lower() not in drop},
            )

        methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]
        return Starlette(routes=[Route("/{path:path}", forward, methods=methods)])

    def up(self) -> None:
        import threading

        import uvicorn

        config = uvicorn.Config(self._app(), host="127.0.0.1", port=self.port, log_level="error")
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()
        deadline = time.monotonic() + 10
        while not self._server.started and time.monotonic() < deadline:
            time.sleep(0.05)

    def down(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
            self._thread.join(timeout=10)
            self._server = None


@pytest.fixture
def memory_proxy() -> Iterator[FaultProxy]:
    proxy = FaultProxy(MEMORY_URL or "")
    proxy.up()
    yield proxy
    proxy.down()


def move_clock(dsn: str | None, statement: str, *params: str) -> None:
    """The one write a test makes to a service's database: moving a row's time back (a
    schedule's next fire, an ended run's age), where waiting the real hours or days is the only
    other way. Never anything else."""
    assert dsn is not None
    assert statement.lstrip().lower().startswith("update"), statement
    for i, value in enumerate(params, 1):
        statement = statement.replace(f":{i}", "'" + value.replace("'", "''") + "'")
    subprocess.run(
        ["psql", dsn, "-qAt", "-v", "ON_ERROR_STOP=1", "-c", statement],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )


@pytest.fixture(autouse=True)
def claude_sessions_of_this_test(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The scripted Claude CLI keeps its sessions in the test's own folder (INV-9)."""
    monkeypatch.setenv("FAKE_CLAUDE_SESSIONS", str(tmp_path))


def run_periodic_now(task: str, queue: str) -> None:
    """Run one of the memory service's periodic jobs now, as an operator would run it on
    demand: one row in its job queue (Procrastinate's ``procrastinate_jobs``, which its worker
    listens on), the arguments a periodic fire carries. The retention sweep is daily and the
    archive purge hourly: waiting for their schedule is the only other way."""
    assert MEMORY_DSN is not None
    statement = (
        "insert into procrastinate_jobs (queue_name, task_name, args, priority) values"
        f" ('{queue}', '{task}', jsonb_build_object('timestamp', extract(epoch from now())::int),"
        " 40)"
    )
    subprocess.run(
        ["psql", MEMORY_DSN, "-qAt", "-v", "ON_ERROR_STOP=1", "-c", statement],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
