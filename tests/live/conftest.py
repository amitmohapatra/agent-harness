"""The live suite: the harness against running services, configured by the same environment
a deployment uses (``.env.example``). ``make test-live`` runs it; a test whose service is
unset or unreachable is skipped, never failed.

The model is reached the way a deployment reaches it: through Bifrost (``LLM_GATEWAY=bifrost``),
or — Bifrost being optional — directly, at the endpoint ``LLM_BASE_URL`` names with ``LLM_API_KEY``
(the ``openai-compatible`` route; the Claude Code CLI on the provider's own ``ANTHROPIC_API_KEY``
and ``ANTHROPIC_BASE_URL``). A test that needs only a real model runs either way
(:data:`needs_model`); only a test of Bifrost's own features — its MCP clients and virtual keys,
Code Mode, stored prompts, its MCP endpoint — needs the gateway (:data:`needs_gateway`).

Two variables of its own pick what it runs against: ``TRELLIS_LIVE_MODEL``, the model its
agents use (default :data:`DEFAULT_MODEL`; Bifrost's ``provider/model`` through the gateway, the
endpoint's own name directly), and ``TRELLIS_LIVE_MCP_URL``, the wiki MCP
server (default the public DeepWiki, :data:`DEEPWIKI_URL`); ``TRELLIS_LIVE_TIMEOUT`` and
``TRELLIS_LIVE_MAX_TOKENS`` size the model-driven tests' runs to the model
(``tests/live/proof.py``). The sandbox tests need only the Docker daemon (its socket); the
data ones also its data image (``SANDBOX_DATA_IMAGE``, built from ``tests/live/sandbox-data``). ``LIVE_SHARD=i/n`` runs the i-th of
n shards, packed by the last measured durations so each finishes under a job's limit on a CPU
model (``tests/live/shards.py``, ``make live-shard``).

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
import shutil
import subprocess
import sys
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Final
from urllib.parse import urlsplit

import httpx
import pytest
from bifrost_sdk import Bifrost, MCPClientConfig, MCPConnection
from bifrost_sdk.admin import Admin

from tests.live import shards
from trellis import Harness, Settings
from trellis.harness import agent as agent_module
from trellis.harness import evals, telemetry
from trellis.harness.agent import Agent
from trellis.harness.evals import Evaluator
from trellis.harness.sandbox.docker import docker_host
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


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """``LIVE_SHARD=i/n``: only the live tests the i-th of n shards holds (``shards.assign``)."""
    chosen = os.environ.get("LIVE_SHARD", "").strip()
    if not chosen:
        return
    try:
        which, total = shards.parse(chosen)
    except ValueError as exc:
        raise pytest.UsageError(str(exc)) from exc
    live = [item.nodeid for item in items if "live" in item.keywords]
    placed = shards.assign(live, total, shards.load())
    dropped = [i for i in items if "live" in i.keywords and placed[i.nodeid] != which]
    if dropped:
        config.hook.pytest_deselected(items=dropped)
        items[:] = [i for i in items if i not in dropped]


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


#: The Docker daemon the sandbox tests' provider reaches (``DOCKER_HOST``, the docker context,
#: else the local socket), and the Engine API's root on it.
DOCKER_HOST: Final = docker_host()
DOCKER_UDS: Final = (
    DOCKER_HOST.removeprefix("unix://") if DOCKER_HOST.startswith("unix://") else None
)
DOCKER_URL: Final = "http://docker" if DOCKER_UDS else f"http://{DOCKER_HOST.split('://', 1)[-1]}"


def docker_transport() -> httpx.HTTPTransport:
    return httpx.HTTPTransport(uds=DOCKER_UDS)


def _docker_up() -> bool:
    """Whether the Docker daemon answers (the sandbox tests' provider)."""
    try:
        with httpx.Client(transport=docker_transport()) as client:
            return client.get(f"{DOCKER_URL}/_ping", timeout=3.0).status_code == 200
    except httpx.HTTPError:
        return False


#: Bifrost's ``/v1`` root, when the environment names Bifrost (``LLM_GATEWAY=bifrost``).
GATEWAY_URL = _env("LLM_BASE_URL") if _env("LLM_GATEWAY") == "bifrost" else None
MODEL = _env("TRELLIS_LIVE_MODEL") or DEFAULT_MODEL
MCP_URL = _env("TRELLIS_LIVE_MCP_URL") or DEEPWIKI_URL
#: The wiki server is this machine's (``mcp_fixture.py``): declared in the gateway, not added.
LOCAL_MCP = urlsplit(MCP_URL).hostname in ("127.0.0.1", "localhost")
MEMORY_URL = _env("MEMORY_URL")
RUNS_URL = _env("RUNS_URL")
GATEWAY_UP = _reachable(GATEWAY_URL.removesuffix("/v1") if GATEWAY_URL else None, "/health")
MEMORY_UP = _reachable(MEMORY_URL, "/health/live")
RUNS_UP = _reachable(RUNS_URL, "/health/live")

#: The endpoint a model is reached at with no gateway (``LLM_BASE_URL``, the ``openai-compatible``
#: route), and whether it answers.
DIRECT_URL = _env("LLM_BASE_URL") if GATEWAY_URL is None else None
DIRECT_UP = _reachable(DIRECT_URL, "/models")
#: The provider's own Anthropic variables, as this process was started with them: what the
#: Claude Code CLI reads to reach the model with no gateway (``cli_home`` clears them from the
#: process; :func:`claude_cli_env` hands them to the CLI).
ANTHROPIC_ENV: Final = {
    name: value
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL")
    if (value := _env(name)) is not None
}

#: What a test of Bifrost's own features says when there is no gateway.
GATEWAY_REASON: Final = (
    "tests a Bifrost feature: needs a Bifrost gateway (LLM_BASE_URL, LLM_GATEWAY=bifrost)"
)
needs_gateway = pytest.mark.skipif(not GATEWAY_UP, reason=GATEWAY_REASON)
#: A real model, through the gateway or directly at ``LLM_BASE_URL``.
needs_model = pytest.mark.skipif(
    not (GATEWAY_UP or DIRECT_UP),
    reason="needs a model: a Bifrost gateway (LLM_BASE_URL, LLM_GATEWAY=bifrost) or the model's "
    "own OpenAI-compatible endpoint (LLM_BASE_URL, LLM_API_KEY)",
)
#: The live model for the Claude Code CLI, which speaks Anthropic's API: the gateway's
#: Anthropic route, or Anthropic's own API with the provider's key.
needs_claude_model = pytest.mark.skipif(
    not (GATEWAY_UP or "ANTHROPIC_API_KEY" in ANTHROPIC_ENV),
    reason="needs the live model on Anthropic's API: a Bifrost gateway (LLM_BASE_URL, "
    "LLM_GATEWAY=bifrost) or ANTHROPIC_API_KEY",
)
needs_memory = pytest.mark.skipif(not MEMORY_UP, reason="needs the memory service (MEMORY_URL)")
needs_runs = pytest.mark.skipif(not RUNS_UP, reason="needs agent-runs (RUNS_URL)")
needs_docker = pytest.mark.skipif(not _docker_up(), reason=f"needs a Docker daemon ({DOCKER_HOST})")


def settings(**changes: object) -> Settings:
    """The environment's deployment, with ``changes`` (each a setting: checked)."""
    return Settings(**{**Settings.from_env().model_dump(), **changes})


class LiveHarness(Harness):
    """A harness whose memory client is given (it waits longer than a deployment's would, since
    the services share one development machine), and closed with it; every agent it wraps is
    judged by its ``judges``; ``sampled`` is the share of runs grounded and judged while it is
    open (a fixed share in a deployment)."""

    def __init__(self, *, judges: Sequence[Evaluator], sampled: float | None, **given: Any) -> None:
        super().__init__(**given)
        self.judges = list(judges)
        self._samples = (agent_module.GROUNDING_SAMPLE, evals.JUDGE_SAMPLE)
        if sampled is not None:
            agent_module.GROUNDING_SAMPLE = evals.JUDGE_SAMPLE = sampled  # type: ignore[misc]

    def wrap(self, target: Any, **given: Any) -> Agent:  # type: ignore[override]
        given["judges"] = [*self.judges, *given.get("judges", ())]
        return super().wrap(target, **given)

    async def aclose(self) -> None:
        agent_module.GROUNDING_SAMPLE, evals.JUDGE_SAMPLE = self._samples  # type: ignore[misc]
        await super().aclose()
        if self.memory is not None:
            await self.memory.client.aclose()


def live_harness(
    key: str | None = None,
    *,
    judges: Sequence[Evaluator] = (),
    sampled: float | None = None,
    langfuse: Mapping[str, str] | None = None,
    **changes: object,
) -> Harness:
    """A harness for the environment's deployment, with the virtual key ``key`` (a session
    key from the fixtures; the environment's otherwise), the online ``judges`` on every agent,
    ``sampled`` (the share of runs grounded and judged) and Langfuse's variables
    (``langfuse``: its scores and datasets, not its traces)."""
    if key is not None:
        changes["llm_api_key"] = key
    s = settings(**changes)
    memory = (
        MemoryClient(s.memory_url, api_key=s.api_key, timeout=LIVE_TIMEOUT)
        if s.memory_url is not None
        else None
    )
    h = LiveHarness(config=s, memory=memory, judges=judges, sampled=sampled)
    if langfuse is not None:
        h.evals.langfuse = telemetry.Langfuse.of(langfuse)
    return h


@pytest.fixture
async def harness() -> AsyncIterator[Harness]:
    async with live_harness() as h:
        yield h


def claude_cli() -> str | None:
    """The Claude Code CLI the Claude Agent SDK runs when the options name none: the one
    bundled with the SDK, else a ``claude`` on PATH (``None``: neither)."""
    try:
        import claude_agent_sdk
    except ImportError:
        return shutil.which("claude")
    bundled = Path(claude_agent_sdk.__file__).parent / "_bundled" / "claude"
    return str(bundled) if bundled.is_file() else shutil.which("claude")


#: The Claude Code CLI, which the Claude Agent SDK runs (its tests skip without it).
needs_claude_cli = pytest.mark.skipif(claude_cli() is None, reason="needs the Claude Code CLI")


@pytest.fixture
def cli_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """The CLI as a deployment runs it: none of this process's own Claude settings (a Claude
    Code session's variables point a child CLI elsewhere), its sessions in a home of its own,
    and that home its working directory (give it as the options' ``cwd``): from this
    repository the CLI read the repository's CLAUDE.md and project settings into the model's
    context, and the local model followed their instructions (it searched memory for
    "gstack") instead of the task."""
    for name in list(os.environ):
        if name.startswith(("CLAUDE", "ANTHROPIC")):
            monkeypatch.delenv(name)
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


def claude_cli_env() -> dict[str, str]:
    """The CLI's environment for the live model: the gateway's Anthropic route and the key —
    with no gateway, Anthropic's API as the provider's own variables name it
    (:data:`ANTHROPIC_ENV`) — and replies capped at 512 tokens as the local server caps every
    other client's (its ``-n``). The CLI asks for tens of thousands, which the server honours:
    a reply the small model fell into repeating ran 2,100 tokens at ~4 token/s, past the
    test's ten minutes."""
    route = (
        {
            "ANTHROPIC_BASE_URL": f"{GATEWAY_URL.removesuffix('/v1')}/anthropic",
            "ANTHROPIC_API_KEY": settings().llm_api_key or "unused",
        }
        if GATEWAY_URL is not None
        else dict(ANTHROPIC_ENV)
    )
    return {
        **route,
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "512",
    }


@pytest.fixture(scope="session")
def mcp_fixture() -> Iterator[str]:
    """The local MCP servers (``mcp_fixture.py``) at ``TRELLIS_LIVE_MCP_URL``, started for the
    session unless they already run, and the gateway's clients of them connected; the wiki
    server's URL. Skips when the URL is not this machine's, or the gateway declares none of
    the clients."""
    if not GATEWAY_UP or GATEWAY_URL is None:
        pytest.skip(f"{GATEWAY_REASON} (its MCP clients)")
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
        asyncio.run(_connected(GATEWAY_URL))
        yield MCP_URL
    finally:
        if started is not None:
            started.terminate()
            started.wait()


async def _connected(url: str) -> None:
    """The gateway's clients of the local servers, reconnected (or their tools refreshed) when
    they were down when the gateway started, and waited for until their tools are listed;
    skips when its ``config.json`` declares none."""
    async with (
        Bifrost(url, gateway="bifrost") as bf,
        httpx.AsyncClient(base_url=url.removesuffix("/v1")) as api,
    ):
        for _ in range(60):  # the servers start
            if _reachable(MCP_URL.rsplit("/", 1)[0], "/"):
                break
            await asyncio.sleep(0.5)
        declared = [c for c in await bf.mcp.clients() if c.config.name in (*WIKIS, OPS)]
        if not declared:
            pytest.skip("the gateway's config.json declares no client of the local MCP servers")
        stale = [c.id for c in declared if not c.tools]
        for client_id in stale:
            # a persistent client reconnects; one that connects per call (the gateway answers
            # 400 to a reconnect) lists its tools again on refresh
            reconnected = await api.post(f"/api/mcp/client/{client_id}/reconnect")
            if reconnected.status_code == httpx.codes.BAD_REQUEST:
                await api.post(f"/api/mcp/client/{client_id}/refresh-tools")
        for _ in range(60):  # a reconnect lists the client's tools again in the background
            listed = {c.id: c.tools for c in await bf.mcp.clients()}
            if all(listed.get(client_id) for client_id in stale):
                break
            await asyncio.sleep(0.5)
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
    if not GATEWAY_UP or GATEWAY_URL is None:
        pytest.skip(f"{GATEWAY_REASON} (its MCP clients)")
    url = GATEWAY_URL
    if LOCAL_MCP:
        request.getfixturevalue("mcp_fixture")
        yield list(WIKIS)
        return

    async def change(add: bool) -> None:
        async with Bifrost(url, gateway="bifrost") as bf:
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
    url = GATEWAY_URL
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
