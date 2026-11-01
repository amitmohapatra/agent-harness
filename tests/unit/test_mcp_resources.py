"""A server's resources as one read tool, against a fake MCP client: offered only with the
``resources`` capability, never over a tool of the server's own of that name; its description
lists at most ``MAX_LISTED`` and says when there are more; a listing reads every page, up to
``MAX_PAGES``; a read's contents as the model reads them."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from tests.unit.test_mcp_source import FakeClient, connected, listed
from trellis.harness.tools import mcp as mcp_module
from trellis.harness.tools import resources
from trellis.harness.tools.mcp import mcp


def resource(n: int, **more: Any) -> Any:
    fields = {"uri": f"r://{n}", "name": f"r{n}", "title": None, "description": None}
    return SimpleNamespace(**{"mime_type": None, "size": None, **fields, **more})


def template(uri: str) -> Any:
    return SimpleNamespace(
        uri_template=uri, name="t", title="T", description="by id", mime_type=None
    )


class Resourceful(FakeClient):
    """A server with ``count`` resources, ``per_page`` a page, and one template."""

    def __init__(self, count: int, per_page: int, own: tuple[str, ...] = ()) -> None:
        super().__init__()
        self.server_capabilities = SimpleNamespace(resources=SimpleNamespace())
        self.items = [resource(n) for n in range(count)]
        self.per_page = per_page
        self.own = own
        self.pages = 0

    async def list_tools(self, cursor: Any = None) -> Any:
        return SimpleNamespace(tools=[listed(n) for n in self.own], next_cursor=None)

    async def list_resources(self, cursor: Any = None) -> Any:
        self.pages += 1
        start = int(cursor or 0)
        end = start + self.per_page
        more = str(end) if end < len(self.items) else None
        return SimpleNamespace(resources=self.items[start:end], next_cursor=more)

    async def list_resource_templates(self, cursor: Any = None) -> Any:
        return SimpleNamespace(resource_templates=[template("r://{kind}/{id}")], next_cursor=None)


async def test_a_long_list_is_cut_and_says_there_is_more() -> None:
    client = Resourceful(count=30, per_page=50)
    [reader] = await connected(mcp("https://h/mcp", name="h"), client).resolve()
    lines = reader.spec.description.splitlines()
    assert len(lines) == 1 + resources.MAX_LISTED + 1 and lines[-1] == resources.MORE
    paged = Resourceful(count=3, per_page=2)
    [reader] = await connected(mcp("https://h/mcp", name="h"), paged).resolve()
    lines = reader.spec.description.splitlines()
    assert lines[1:] == [
        "- r://0: r0",
        "- r://1: r1",
        "- r://{kind}/{id}: T, by id (fill in {kind} {id})",
        resources.MORE,  # the server had another page
    ]


async def test_a_listing_reads_every_page_up_to_the_most(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = Resourceful(count=5, per_page=2)
    source = connected(mcp("https://h/mcp", name="h"), client)
    [reader] = await source.resolve()
    rows = await reader.run({})
    assert [r.get("uri") or r["uri_template"] for r in rows] == [
        *[f"r://{n}" for n in range(5)],
        "r://{kind}/{id}",
    ]
    assert rows[0] == {"uri": "r://0", "name": "r0"}
    monkeypatch.setattr(mcp_module, "MAX_PAGES", 2)
    assert len(await reader.run({})) == 4 + 1  # two pages of resources, and the template


async def test_no_read_tool_over_the_servers_own() -> None:
    client = Resourceful(count=1, per_page=5, own=(resources.READ,))
    tools = await connected(mcp("https://h/mcp", name="h"), client).resolve()
    assert [t.name for t in tools] == ["h-read_resource"] and tools[0].spec.side_effects == "write"


def test_what_a_read_says() -> None:
    text = SimpleNamespace(uri="r://a", mime_type="text/csv", text="a,b")
    blob = SimpleNamespace(uri="r://b", mime_type=None, blob="AAAA")
    said = resources.text_of(SimpleNamespace(contents=[text, blob]))
    assert said == "a,b\n\n[r://b: binary, 3 bytes, not text]"
    one = resource(1, title="One", description="the first", mime_type="text/plain", size=9)
    assert resources.row(one) == {
        "uri": "r://1",
        "name": "r1",
        "title": "One",
        "description": "the first",
        "mime_type": "text/plain",
        "size": 9,
    }
