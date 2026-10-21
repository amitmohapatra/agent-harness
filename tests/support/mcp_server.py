"""A real MCP server for the tests, on the official SDK: ``python -m tests.support.mcp_server``
serves it over stdio; ``... http PORT`` over streamable HTTP (``TOKEN`` in the environment: every
request needs ``Authorization: Bearer <TOKEN>``). ``MARK`` names a file: ``crash`` exits the
process the first time (the file not there yet), then answers."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

server = MCPServer("Notes Box")
served: list[str] = []


@server.tool(annotations=ToolAnnotations(read_only_hint=True))
def read_note(path: str) -> str:
    """Read a note.

    Args:
        path: The note's path.
    """
    served.append("read_note")
    return f"note at {path}"


@server.tool()
def add_note(text: str) -> str:
    """Add a note."""
    served.append("add_note")
    return f"added {text}"


@server.tool(annotations=ToolAnnotations(destructive_hint=True))
def delete_note(path: str) -> str:
    """Delete a note."""
    served.append("delete_note")
    return f"deleted {path}"


@server.tool(annotations=ToolAnnotations(read_only_hint=True, idempotent_hint=True))
def served_calls() -> list[str]:
    """The calls this process served."""
    return list(served)


@server.tool()
def fail(why: str) -> str:
    """Fail."""
    raise ValueError(why)


@server.tool(annotations=ToolAnnotations(read_only_hint=True))
def crash() -> str:
    """Exit the server the first time."""
    mark = Path(os.environ["MARK"])
    if not mark.exists():
        mark.write_text("crashed")
        os._exit(1)
    return "back"


def http(port: int) -> None:
    import uvicorn
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.responses import Response

    app = server.streamable_http_app()
    token = os.environ.get("TOKEN")
    if token:

        async def guard(request, call_next):  # type: ignore[no-untyped-def]
            if request.headers.get("authorization") != f"Bearer {token}":
                return Response(status_code=401)
            return await call_next(request)

        app.add_middleware(BaseHTTPMiddleware, dispatch=guard)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    if sys.argv[1:2] == ["http"]:
        http(int(sys.argv[2]))
    else:
        server.run("stdio")
