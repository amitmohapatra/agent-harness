"""The live suite: the harness against running services, configured by the same environment
a deployment uses (``.env.example``). ``make test-live`` runs it; a test whose service is
unset or unreachable is skipped, never failed.

The gateway gets temporary MCP clients for the session (the public DeepWiki server, under
:data:`WIKIS` names) and virtual keys that allow some of their tools: an agent's MCP tools are
exactly what its key allows, so each test picks its key (:func:`live_harness`)."""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Final

import httpx
import pytest
from bifrost_sdk import Bifrost, MCPClientConfig, MCPConnection
from bifrost_sdk.admin import Admin

from trellis import Evaluator, Harness, Settings
from trellis.harness.clients.memory import Memory
from trellis.memory import MemoryClient

#: How long a memory call may take here (a deployment keeps the SDK's 10 s).
LIVE_TIMEOUT: Final = 60.0
#: Cheap, and reliable at tool calling through the gateway.
MODEL: Final = "openrouter/openai/gpt-4.1-nano"
#: A public MCP server the suite registers in the gateway as a Code Mode client, and removes.
DEEPWIKI_URL: Final = "https://mcp.deepwiki.com/mcp"
#: Three servers are enough for Code Mode (a key that allows them all).
WIKIS: Final = ("trellislivewiki", "trellislivewiki2", "trellislivewiki3")
TOOLS_PER_WIKI: Final = 3
#: The one wiki tool the framework tests' key allows.
WIKI_TOOL: Final = "read_wiki_structure"


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


BIFROST_URL = _env("BIFROST_URL")
MEMORY_URL = _env("MEMORY_URL")
RUNS_URL = _env("RUNS_URL")
GATEWAY_UP = _reachable(BIFROST_URL.removesuffix("/v1") if BIFROST_URL else None, "/health")
MEMORY_UP = _reachable(MEMORY_URL, "/health/live")
RUNS_UP = _reachable(RUNS_URL, "/health/live")

needs_gateway = pytest.mark.skipif(not GATEWAY_UP, reason="needs a Bifrost gateway (BIFROST_URL)")
needs_memory = pytest.mark.skipif(not MEMORY_UP, reason="needs the memory service (MEMORY_URL)")
needs_runs = pytest.mark.skipif(not RUNS_UP, reason="needs agent-runs (RUNS_URL)")


def settings(**changes: object) -> Settings:
    """The environment's deployment."""
    return Settings.from_env().model_copy(update=changes)


def live_harness(
    key: str | None = None, *, judges: Sequence[Evaluator] = (), **changes: object
) -> Harness:
    """A harness for the environment's deployment, with the virtual key ``key`` (a session
    key from the fixtures; the environment's otherwise) and the online ``judges``; its memory
    client waits longer than a deployment's would, since the services share one development
    machine."""
    if key is not None:
        changes["bifrost_virtual_key"] = key
    h = Harness(config=settings(**changes), judges=judges)
    s = h.settings
    if s.memory_url is not None:
        client = MemoryClient(s.memory_url, api_key=s.api_key, timeout=LIVE_TIMEOUT)
        h.memory = Memory(s.memory_url, s.api_key, client=client)
    return h


@pytest.fixture
async def harness() -> AsyncIterator[Harness]:
    async with live_harness() as h:
        yield h


@pytest.fixture(scope="session")
def wikis() -> Iterator[list[str]]:
    """Temporary Code Mode MCP clients in the gateway (the same public server under
    :data:`WIKIS` names), removed after the session."""
    if not GATEWAY_UP or BIFROST_URL is None:
        pytest.skip("needs a Bifrost gateway (BIFROST_URL)")
    url = BIFROST_URL

    async def change(add: bool) -> None:
        async with Bifrost(url) as bf:
            for client in await bf.mcp.clients():
                if client.config.name in WIKIS:
                    await bf.mcp.remove(client.id)
            for name in WIKIS if add else ():
                await bf.mcp.add(
                    MCPClientConfig(
                        name=name,
                        connection=MCPConnection(type="http", url=DEEPWIKI_URL),
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
    """A virtual key for the session: every provider, and the ``mcp`` allow-list."""
    url = BIFROST_URL
    assert url is not None

    async def create() -> tuple[str, str]:
        async with Admin(url) as admin:
            for existing in await admin.vk.list():
                if existing.get("name") == name:
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
