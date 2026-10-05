"""Local MCP servers for the live suite (streamable HTTP), so it needs no public MCP host.

Two servers, run together by ``python -m tests.live.mcp_fixture URL`` (the session fixture
``mcp_fixture`` in ``conftest.py`` starts it when ``TRELLIS_LIVE_MCP_URL`` is a local URL):

* the **wiki** server at ``URL`` — DeepWiki's three tools (``read_wiki_structure``,
  ``read_wiki_contents``, ``ask_wiki_question``), with deterministic content and, like
  DeepWiki, no annotations;
* the **ops** server on the next port (:func:`ops_url`) — a tool that writes
  (``write_note``), one that is irreversible (``delete_records``, ``destructiveHint``) and one
  that answers who called it (``whoami``, from the ``x-trellis-identity`` header the gateway
  forwards; refused without one): a header-authenticated server, the per-user case.

The gateway refuses to register a loopback server through its management API, so the clients
are declared in its ``config.json`` (``mcp.client_configs``):

* ``trellislivewiki``, ``trellislivewiki2``, ``trellislivewiki3`` → ``URL``, Code Mode clients;
* ``trellisliveops`` → :func:`ops_url`, ``allowed_extra_headers: ["x-trellis-identity"]``.
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Final
from urllib.parse import urlsplit, urlunsplit

import uvicorn
from mcp.server.mcpserver import Context, MCPServer
from mcp.types import ToolAnnotations

from trellis.harness.identity import IDENTITY_HEADER

#: The repositories the wiki knows: each one's table of contents, and what it says.
WIKIS: Final[dict[str, tuple[list[str], str]]] = {
    "facebook/react": (
        ["1 React Overview", "2 Components", "3 Hooks", "4 Rendering"],
        "React is a library for building user interfaces from components.",
    ),
    "trellis/harness": (
        ["1 Harness Overview", "2 Tools", "3 Runs"],
        "The harness runs agents with memory, governed tools and durable runs.",
    ),
}
#: The ops server's name in the gateway.
OPS: Final = "trellisliveops"

wiki = MCPServer("trellislivewiki")
ops = MCPServer(OPS)


def _wiki(repo: str) -> tuple[list[str], str]:
    found = WIKIS.get(repo)
    if found is None:
        raise ValueError(f"no wiki for {repo!r}; known: {', '.join(WIKIS)}")
    return found


@wiki.tool()
def read_wiki_structure(repoName: str) -> str:
    """The table of contents of a repository's wiki."""
    return "\n".join(_wiki(repoName)[0])


@wiki.tool()
def read_wiki_contents(repoName: str) -> str:
    """The contents of a repository's wiki."""
    return _wiki(repoName)[1]


@wiki.tool()
def ask_wiki_question(repoName: str, question: str) -> str:
    """An answer to a question about a repository, from its wiki."""
    return f"About {repoName}: {_wiki(repoName)[1]} (asked: {question})"


@ops.tool()
def write_note(text: str) -> str:
    """Write a note (a write: it changes something, and can be undone)."""
    return f"noted: {text}"


@ops.tool(annotations=ToolAnnotations(destructive_hint=True))
def delete_records(table: str) -> str:
    """Delete every record of a table (irreversible)."""
    return f"deleted every record of {table}"


@ops.tool(annotations=ToolAnnotations(read_only_hint=True))
def whoami(ctx: Context) -> str:
    """Who the call is for: the user the gateway forwarded in ``x-trellis-identity``."""
    raw = (ctx.headers or {}).get(IDENTITY_HEADER)
    if not raw:
        raise ValueError(f"unauthenticated: no {IDENTITY_HEADER} header")
    return str(json.loads(raw)["user_id"])


def ops_url(url: str) -> str:
    """Where the ops server listens: the wiki's host, the next port."""
    parts = urlsplit(url)
    host = f"{parts.hostname}:{(parts.port or 80) + 1}"
    return urlunsplit((parts.scheme, host, parts.path, "", ""))


async def serve(url: str) -> None:
    servers = []
    for server, at in ((wiki, url), (ops, ops_url(url))):
        parts = urlsplit(at)
        app = server.streamable_http_app(stateless_http=True, json_response=True)
        config = uvicorn.Config(
            app, host=parts.hostname or "127.0.0.1", port=parts.port or 80, log_level="warning"
        )
        servers.append(uvicorn.Server(config))
    await asyncio.gather(*(s.serve() for s in servers))


if __name__ == "__main__":
    asyncio.run(serve(sys.argv[1]))
