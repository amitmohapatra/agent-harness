"""The live suite: the harness against running services, configured by the same environment
a deployment uses (``.env.example``). ``make test-live`` runs it; a test whose service is
unset or unreachable is skipped, never failed.

Two variables of its own pick what it runs against: ``TRELLIS_LIVE_MODEL``, the gateway model
its agents use (default :data:`DEFAULT_MODEL`), and ``TRELLIS_LIVE_MCP_URL``, the wiki MCP
server (default the public DeepWiki, :data:`DEEPWIKI_URL`). The sandbox tests need only the
Docker daemon (its socket), and make their sandboxes of ``SANDBOX_IMAGE`` when it is set.

The gateway gets MCP clients of that server for the session, under :data:`WIKIS` names, and
virtual keys that allow some of their tools: an agent's MCP tools are exactly what its key
allows, so each test picks its key (:func:`live_harness`). A public server is registered for
the session and removed after it. A local one (``tests/live/mcp_fixture.py``, started here when
it is not running yet, with its ops server beside it) cannot be registered — the gateway
refuses loopback servers through its management API — so its clients are declared in the
gateway's ``config.json`` (the fixture module's docstring has them), and the tests that need
them skip when they are not."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Final
from urllib.parse import urlsplit

import httpx
import pytest
from bifrost_sdk import Bifrost, MCPClientConfig, MCPConnection
from bifrost_sdk.admin import Admin

from trellis import Harness, Settings
from trellis.harness.evals import Evaluator
from trellis.harness.sandbox.docker import DOCKER_SOCKET
from trellis.memory import MemoryClient

#: How long a memory call may take here (a deployment keeps the SDK's 10 s).
LIVE_TIMEOUT: Final = 60.0
#: Cheap, and reliable at tool calling through the gateway.
DEFAULT_MODEL: Final = "openrouter/openai/gpt-4.1-nano"
#: The public MCP server the suite registers in the gateway as Code Mode clients, and removes.
DEEPWIKI_URL: Final = "https://mcp.deepwiki.com/mcp"
#: Three servers are enough for Code Mode (a key that allows them all).
WIKIS: Final = ("trellislivewiki", "trellislivewiki2", "trellislivewiki3")
TOOLS_PER_WIKI: Final = 3
#: The one wiki tool the framework tests' key allows.
WIKI_TOOL: Final = "read_wiki_structure"
#: The local ops server's client (``mcp_fixture.py``): a write, an irreversible and a
#: header-authenticated tool.
OPS: Final = "trellisliveops"


def _env(name: str) -> str | None:
    value = os.environ.get(name, "").strip()
    return value or None


def _reachable(url: str | None, path: str) -> bool:
    if url is None:
        return False
    try:
        return httpx.get(url.rstrip("/") + path, timeout=3.0).status_code < 500
    except httpx.HTTPError:
        return False


def _docker_up() -> bool:
    """Whether the Docker daemon answers on its socket (the sandbox tests' provider)."""
    try:
        with httpx.Client(transport=httpx.HTTPTransport(uds=DOCKER_SOCKET)) as client:
            return client.get("http://docker/_ping", timeout=3.0).status_code == 200
    except httpx.HTTPError:
        return False


BIFROST_URL = _env("BIFROST_URL")
MODEL = _env("TRELLIS_LIVE_MODEL") or DEFAULT_MODEL
MCP_URL = _env("TRELLIS_LIVE_MCP_URL") or DEEPWIKI_URL
#: The wiki server is this machine's (``mcp_fixture.py``): declared in the gateway, not added.
LOCAL_MCP = urlsplit(MCP_URL).hostname in ("127.0.0.1", "localhost")
MEMORY_URL = _env("MEMORY_URL")
RUNS_URL = _env("RUNS_URL")
GATEWAY_UP = _reachable(BIFROST_URL.removesuffix("/v1") if BIFROST_URL else None, "/health")
MEMORY_UP = _reachable(MEMORY_URL, "/health/live")
RUNS_UP = _reachable(RUNS_URL, "/health/live")

needs_gateway = pytest.mark.skipif(not GATEWAY_UP, reason="needs a Bifrost gateway (BIFROST_URL)")
needs_memory = pytest.mark.skipif(not MEMORY_UP, reason="needs the memory service (MEMORY_URL)")
needs_runs = pytest.mark.skipif(not RUNS_UP, reason="needs agent-runs (RUNS_URL)")
needs_docker = pytest.mark.skipif(
    not _docker_up(), reason=f"needs a Docker daemon ({DOCKER_SOCKET})"
)


def settings(**changes: object) -> Settings:
    """The environment's deployment."""
    return Settings.from_env().model_copy(update=changes)


class LiveHarness(Harness):
    """A harness whose memory client is given (it waits longer than a deployment's would, since
    the services share one development machine), and closed with it."""

    async def aclose(self) -> None:
        await super().aclose()
        if self.memory is not None:
            await self.memory.client.aclose()


def live_harness(
    key: str | None = None, *, judges: Sequence[Evaluator] = (), **changes: object
) -> Harness:
    """A harness for the environment's deployment, with the virtual key ``key`` (a session
    key from the fixtures; the environment's otherwise) and the online ``judges``."""
    if key is not None:
        changes["bifrost_virtual_key"] = key
    s = settings(**changes)
    memory = (
        MemoryClient(s.memory_url, api_key=s.api_key, timeout=LIVE_TIMEOUT)
        if s.memory_url is not None
        else None
    )
    return LiveHarness(config=s, memory=memory, judges=judges)


@pytest.fixture
async def harness() -> AsyncIterator[Harness]:
    async with live_harness() as h:
        yield h


@pytest.fixture(scope="session")
def mcp_fixture() -> Iterator[str]:
    """The local MCP servers (``mcp_fixture.py``) at ``TRELLIS_LIVE_MCP_URL``, started for the
    session unless they already run, and the gateway's clients of them connected; the wiki
    server's URL. Skips when the URL is not this machine's, or the gateway declares none of
    the clients."""
    if not GATEWAY_UP or BIFROST_URL is None:
        pytest.skip("needs a Bifrost gateway (BIFROST_URL)")
    if not LOCAL_MCP:
        pytest.skip("needs the local MCP servers: TRELLIS_LIVE_MCP_URL=http://127.0.0.1:<port>/mcp")
    started = None
    if not _reachable(MCP_URL.rsplit("/", 1)[0], "/"):
        started = subprocess.Popen(
            [sys.executable, "-m", "tests.live.mcp_fixture", MCP_URL],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    try:
        asyncio.run(_connected(BIFROST_URL))
        yield MCP_URL
    finally:
        if started is not None:
            started.terminate()
            started.wait()


async def _connected(url: str) -> None:
    """The gateway's clients of the local servers, reconnected when they were down when the
    gateway started; skips when its ``config.json`` declares none."""
    async with Bifrost(url) as bf, httpx.AsyncClient(base_url=url.removesuffix("/v1")) as api:
        for _ in range(60):  # the servers start
            if _reachable(MCP_URL.rsplit("/", 1)[0], "/"):
                break
            await asyncio.sleep(0.5)
        declared = [c for c in await bf.mcp.clients() if c.config.name in (*WIKIS, OPS)]
        if not declared:
            pytest.skip("the gateway's config.json declares no client of the local MCP servers")
        for client in declared:
            if not client.tools:
                await api.post(f"/api/mcp/client/{client.id}/reconnect")
        for _ in range(60):
            clients = await bf.mcp.clients()
            if all(c.tools for c in clients if c.config.name in (*WIKIS, OPS)):
                return
            await asyncio.sleep(0.5)


@pytest.fixture(scope="session")
def wikis(request: pytest.FixtureRequest) -> Iterator[list[str]]:
    """Code Mode MCP clients of the wiki server in the gateway, under :data:`WIKIS` names:
    the local server's, declared in the gateway (``mcp_fixture``), or the public one's, added
    for the session and removed after it."""
    if not GATEWAY_UP or BIFROST_URL is None:
        pytest.skip("needs a Bifrost gateway (BIFROST_URL)")
    url = BIFROST_URL
    if LOCAL_MCP:
        request.getfixturevalue("mcp_fixture")
        yield list(WIKIS)
        return

    async def change(add: bool) -> None:
        async with Bifrost(url) as bf:
            for client in await bf.mcp.clients():
                if client.config.name in WIKIS:
                    await bf.mcp.remove(client.id)
            for name in WIKIS if add else ():
                await bf.mcp.add(
                    MCPClientConfig(
                        name=name,
                        connection=MCPConnection(type="http", url=MCP_URL),
                        tools_to_execute=("*",),
                        is_code_mode_client=True,
                    )
                )
            for _ in range(60 if add else 0):  # connected, with the server's tools listed
                if len(await bf.tools(clients=WIKIS)) == TOOLS_PER_WIKI * len(WIKIS):
                    return
                await asyncio.sleep(0.5)

    asyncio.run(change(add=True))
    try:
        yield list(WIKIS)
    finally:
        asyncio.run(change(add=False))


@pytest.fixture(scope="session")
def deepwiki(wikis: list[str]) -> str:
    return wikis[0]


def _virtual_key(name: str, mcp: list[dict[str, object]]) -> Iterator[str]:
    """A virtual key for the session: every provider, and the ``mcp`` allow-list. Its name
    is the session's own (``name`` and this process), so sessions running at the same time
    against one gateway never delete each other's key."""
    url = BIFROST_URL
    assert url is not None
    name = f"{name}-{os.getpid()}"

    async def create() -> tuple[str, str]:
        async with Admin(url) as admin:
            for existing in await admin.vk.list():
                if existing.get("name") == name:  # left by a crashed session of this pid
                    await admin.vk.delete(existing["id"])
            made = await admin.vk.create(
                name, provider_configs=[], mcp_configs=mcp, allow_all_providers=True
            )
            return made["virtual_key"]["id"], made["virtual_key"]["value"]

    async def delete(vk_id: str) -> None:
        async with Admin(url) as admin:
            await admin.vk.delete(vk_id)

    vk_id, value = asyncio.run(create())
    try:
        yield value
    finally:
        asyncio.run(delete(vk_id))


@pytest.fixture(scope="session")
def wiki_key(wikis: list[str]) -> Iterator[str]:
    """A key allowing one tool of one wiki: a small toolbox (no hints, no Code Mode)."""
    yield from _virtual_key(
        "trellis-live-one-wiki",
        [{"mcp_client_name": wikis[0], "tools_to_execute": [WIKI_TOOL]}],
    )


@pytest.fixture(scope="session")
def wikis_key(wikis: list[str]) -> Iterator[str]:
    """A key allowing every tool of the three wikis: enough for Code Mode, and for hints."""
    yield from _virtual_key(
        "trellis-live-all-wikis",
        [{"mcp_client_name": w, "tools_to_execute": ["*"]} for w in wikis],
    )


@pytest.fixture(scope="session")
def ops(mcp_fixture: str) -> str:
    """The local ops server's client in the gateway (``mcp_fixture.py``): a write, an
    irreversible and a header-authenticated tool."""
    return OPS


@pytest.fixture(scope="session")
def ops_key(ops: str) -> Iterator[str]:
    """A key allowing every tool of the ops server."""
    yield from _virtual_key(
        "trellis-live-ops", [{"mcp_client_name": ops, "tools_to_execute": ["*"]}]
    )
