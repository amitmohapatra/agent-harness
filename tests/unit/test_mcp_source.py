"""``mcp(...)`` against a fake MCP client: every page of a listing, the server's name when it
gives none, the answers a call can have (structured, text and other content, a server error, a
connection that closed), and what a Virtual MCP entry is not."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from mcp import MCPError
from mcp.types import CONNECTION_CLOSED, ImageContent, TextContent

from trellis.contracts import ConfigurationError, ToolError
from trellis.harness.tools.mcp import MCPSource, _Connection, answer_of, mcp, stdio


def listed(name: str) -> Any:
    return SimpleNamespace(
        name=name, description=None, title=None, input_schema=None, annotations=None
    )


class FakeClient:
    def __init__(self, info: Any = None, answer: Any = None, raises: Exception | None = None):
        self.server_info = info
        self.answer = answer
        self.raises = raises
        self.cursors: list[Any] = []

    async def list_tools(self, cursor: Any = None) -> Any:
        self.cursors.append(cursor)
        if cursor is None:
            return SimpleNamespace(tools=[listed("a")], next_cursor="2")
        return SimpleNamespace(tools=[listed("b")], next_cursor=None)

    async def call_tool(self, name: str, args: dict[str, Any]) -> Any:
        if self.raises is not None:
            raise self.raises
        return self.answer


def connected(source: MCPSource, client: FakeClient) -> MCPSource:
    async def give() -> Any:
        return client

    source._connection.client = give  # type: ignore[method-assign]
    return source


async def test_every_page_is_listed_and_a_nameless_server_is_named_by_where_it_is() -> None:
    client = FakeClient()
    tools = await connected(mcp(stdio("npx", "@mcp/files-server")), client).resolve()
    assert [t.name for t in tools] == ["files_server-a", "files_server-b"]
    assert client.cursors == [None, "2"]
    assert tools[0].spec.description == "a" and tools[0].spec.input_schema == {"type": "object"}
    by_url = await connected(mcp("https://docs.example.com/mcp"), FakeClient()).resolve()
    assert by_url[0].name == "docs-a"
    bare = await connected(mcp(stdio("srv")), FakeClient()).resolve()
    assert bare[0].name == "srv-a"
    nothing = await connected(mcp("https://-/mcp"), FakeClient()).resolve()
    assert nothing[0].name == "mcp-a"


async def test_what_a_call_answers() -> None:
    def result(content: list[Any], structured: Any = None, error: bool = False) -> Any:
        return SimpleNamespace(content=content, structured_content=structured, is_error=error)

    text = TextContent(type="text", text="hi")
    image = ImageContent(type="image", data="AAAA", mime_type="image/png")
    assert answer_of(result([text], {"result": [1, 2]})) == [1, 2]
    assert answer_of(result([text], {"rows": 2})) == {"rows": 2}
    assert answer_of(result([text])) == "hi"
    both = answer_of(result([text, image]))
    assert both["text"] == "hi" and both["content"][0]["mime_type"] == "image/png"
    source = connected(mcp("https://h/mcp", name="h"), FakeClient(answer=result([], error=True)))
    with pytest.raises(ToolError, match="x failed"):
        await source._call("x", {})


async def test_a_server_error_is_the_calls_and_a_closed_connection_is_tried_again() -> None:
    refused = FakeClient(raises=MCPError(-32602, "bad arguments"))
    source = connected(mcp("https://h/mcp", name="h"), refused)
    with pytest.raises(ToolError, match="h: bad arguments") as said:
        await source._call("x", {})
    assert not said.value.retryable
    closed = FakeClient(raises=MCPError(CONNECTION_CLOSED, "Connection closed"))
    source = connected(mcp("https://h/mcp", name="h"), closed)
    with pytest.raises(ToolError, match="cannot be reached") as said:
        await source._call("x", {})
    assert said.value.retryable


async def test_a_virtual_mcp_is_not_resolved_here() -> None:
    with pytest.raises(ConfigurationError, match="listed by the gateway"):
        await mcp("finance").resolve()


async def test_a_connection_never_opened_closes_and_breaks_quietly() -> None:
    connection = _Connection(make=None)
    connection.broken()
    await connection.aclose()
    assert connection._client is None
