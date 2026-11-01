"""A server's resources (MCP resources), against a real server on the official SDK
(``tests/support/mcp_resources_server.py``) over stdio: offered as one read tool only when the
server has resources, its description the map of what there is; a resource read by its URI (a
template's filled in, a binary one described), every one listed without a URI, an unknown one
an error the agent reads; a large one kept and queried as any large result, on every adapter
that keeps large results (a function target's code gets the whole result)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Final

import pytest

from tests.support.adapters import BUILDERS
from tests.support.arm import arm
from tests.support.planned import Call
from trellis import Harness, Runtime, mcp, stdio
from trellis.contracts import RunStatus
from trellis.harness.journal import content_key

SERVER = Path(__file__).parents[1] / "support" / "mcp_resources_server.py"
#: The stock as the server gives it with :data:`ROWS` rows: over what the model reads as it is.
ROWS: Final = 20_000
STOCK: Final = "\n".join(["sku,on_hand", *[f"SKU-{i:05d},{i % 7}" for i in range(ROWS)]])
KEPT: Final = f"/large_tool_results/{content_key('result', STOCK)}"


def docs(**env: str) -> Any:
    return mcp(stdio(sys.executable, str(SERVER), env=env or None))


async def test_the_read_tool_maps_the_servers_resources(harness: Harness) -> None:
    seen: dict[str, Any] = {}

    async def reader(input: str, agent: Runtime) -> Any:
        tool = agent.toolbox["docs-read_resource"]
        seen["spec"] = tool.spec
        for uri in ("docs://handbook", "docs://pages/7", "docs://logo.png", "docs://nope"):
            seen[uri] = await agent.tools.call("docs-read_resource", uri=uri)
        seen["all"] = await agent.tools.call("docs-read_resource")
        return "done"

    result = await arm(harness.wrap(reader, id="reader"), [docs()]).run("go", user="u")
    assert result.status is RunStatus.SUCCESS, result.error
    spec = seen["spec"]
    assert spec.side_effects == "read" and spec.idempotent and spec.server == "docs"
    assert spec.description.splitlines()[1:] == [
        "- docs://handbook: handbook, How the team works "
        "(text/markdown; modified 2026-10-30T09:00:00Z)",
        "- data://stock.csv: stock (text/csv)",
        "- docs://logo.png: logo (image/png)",
        "- docs://pages/{page}: page, A page by its number (text/plain; fill in {page})",
    ]
    assert seen["docs://handbook"] == "# Handbook\nShip on Tuesdays."
    assert seen["docs://pages/7"] == "page 7"
    assert seen["docs://logo.png"] == "[docs://logo.png: image/png, 6 bytes, not text]"
    assert "docs-read_resource failed" in seen["docs://nope"]
    assert "Unknown resource: docs://nope" in seen["docs://nope"]
    listed = seen["all"]  # rows: a long listing is kept and queried as any large result
    assert [r.get("uri") or r["uri_template"] for r in listed] == [
        "docs://handbook",
        "data://stock.csv",
        "docs://logo.png",
        "docs://pages/{page}",
    ]
    assert listed[0] == {
        "uri": "docs://handbook",
        "name": "handbook",
        "description": "How the team works",
        "mime_type": "text/markdown",
        "last_modified": "2026-10-30T09:00:00Z",
    }


async def test_a_server_with_no_resource_has_no_read_tool(harness: Harness) -> None:
    async def toolbox(input: str, agent: Runtime) -> list[str]:
        return sorted(agent.toolbox)

    result = await arm(harness.wrap(toolbox, id="none"), [docs(NONE="1")]).run("go", user="u")
    assert result.answer == ["docs-ping"]


@pytest.mark.parametrize(
    "framework", ["react", "middleware", "openai_agents", "claude_agent_sdk", "kimi"]
)
async def test_a_large_resource_is_kept_and_queried(
    harness: Harness, framework: str, tmp_path: Path
) -> None:
    sql = "SELECT count(*) AS n, sum(on_hand) AS units FROM result"
    plan: list[Call] = [
        ("docs-read_resource", {"uri": "data://stock.csv"}),
        ("query_result", {"file_path": KEPT, "sql": sql}),
    ]
    source = docs(ROWS=str(ROWS))
    target, own = await BUILDERS[framework](harness, [source], tmp_path, plan)
    result = await arm(harness.wrap(target, id=f"res-{framework}"), own).run("q", user="u")
    assert result.status is RunStatus.SUCCESS, result.error
    units = sum(i % 7 for i in range(ROWS))
    assert f"{ROWS}" in str(result.answer) and f"{units}" in str(result.answer), result.answer
