"""Credentials for the tools the harness calls out to, per run: ``Harness(credentials=hook)``,
``async (run, target) -> headers | None``, asked before each call of an ``mcp(...)`` server over
HTTP, an ``openapi(...)`` operation and an ``a2a(...)`` agent — the three places a harness tool
leaves the process, the same on every adapter (Claude and Kimi reach the harness's tools through
its loopback MCP server). No hook: each call is as it was, with its ``headers=`` only.

* The hook's headers go over the entry's fixed ``headers=``; ``None`` keeps those. Its answer is
  kept for the rest of the run, per target, and never past it: a scheduled run asks again, so it
  gets a token that is fresh (a refresh token's, RFC 6749 §6; or one exchanged for the user,
  RFC 8693). Nothing of it goes into a tool's arguments, the journal, the run's events, traces
  or memory: it is on the HTTP request only (traces mask ``Authorization`` besides).
* :class:`NeedsAuth` (``url``: where the user connects) pauses the run with a question, "Connect
  your <name>", as ``Runtime.ask``; on the answer the hook is asked once more and the call goes
  on (it runs once: the call is made when the run resumes, as any paused call is).
* Any other failure of the hook fails that call (a ``ToolError`` the model reads), with no other
  credentials tried.
* An MCP server's session may be bound to whoever opened it (the MCP authorization spec: a
  token is the user's, for that server), so with a hook the harness keeps one session per
  server, tenant, user and headers, and closes one idle for :data:`IDLE_SECONDS`
  (``tools.mcp``); without a hook, one per server, as before.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal

from trellis.contracts import Option, ToolError
from trellis.contracts.errors import is_pause_signal
from trellis.harness.asking import RunCancelled
from trellis.harness.runtime import current

if TYPE_CHECKING:
    from trellis.harness.runtime import Runtime

#: What a pooled MCP session may sit unused before it is closed, in seconds.
IDLE_SECONDS: Final = 300.0
#: The answer the user gives once they have connected.
CONNECTED: Final = "connected"


@dataclass(frozen=True)
class Target:
    """Where a tool call goes: the entry's name (an MCP server's, an operation's host, an A2A
    agent's), its URL and what kind of entry it is."""

    name: str
    url: str
    kind: Literal["mcp", "openapi", "a2a"]


class NeedsAuth(Exception):
    """Raised by the hook: the run's user has not connected ``target`` yet; ``url`` is where
    they do (an OAuth authorization link)."""

    def __init__(self, url: str) -> None:
        super().__init__(f"connect at {url}")
        self.url = url


#: ``Harness(credentials=)``.
CredentialsHook = Callable[["Runtime", Target], Awaitable[Mapping[str, str] | None]]


async def headers_for(target: Target, static: Mapping[str, str]) -> dict[str, str]:
    """The headers of one call to ``target``: ``static`` (the entry's ``headers=``), the hook's
    over them, the hook asked once per run and target."""
    runtime = current()
    hook = None if runtime is None else runtime.agent.harness.credentials
    if runtime is None or hook is None:
        return dict(static)
    key = (target.kind, target.url)
    if key not in runtime.credentials:
        runtime.credentials[key] = await _asked(hook, runtime, target)
    return {**static, **runtime.credentials[key]}


async def _asked(hook: CredentialsHook, runtime: Runtime, target: Target) -> dict[str, str]:
    try:
        try:
            given = await hook(runtime, target)
        except NeedsAuth as needed:
            await runtime.ask(
                f"Connect your {target.name}: {needed.url}",
                options=[Option(value=CONNECTED, label="Connected", recommended=True)],
            )
            given = await hook(runtime, target)
    except Exception as exc:
        if is_pause_signal(exc) or isinstance(exc, RunCancelled):
            raise
        # the hook's own words may hold a secret: only what failed is said
        raise ToolError(
            f"no credentials for {target.name} ({type(exc).__name__})", source="tools"
        ) from None
    return dict(given or {})
