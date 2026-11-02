"""What a user asks for and downloads, end to end over AG-UI, from a ``ReAct`` run: a result of
250,000 rows (five times the break-and-fix world's 50,000) kept, queried exactly over every row
and saved as an Excel workbook the user downloads; and a chart the run's code draws in its
sandbox, kept from ``outputs/`` and downloaded as an image."""

from __future__ import annotations

import io
from collections import defaultdict
from collections.abc import AsyncIterator
from decimal import Decimal
from pathlib import Path
from typing import Any, Final

import httpx
import pytest
from fastapi import FastAPI, Request
from openpyxl import load_workbook

from tests.support.arm import arm
from tests.support.models import ScriptedChat
from tests.support.sandbox import FakeSandboxes
from trellis import Harness, ReAct, Settings, sandbox, tool
from trellis.harness.agui.sse import decode
from trellis.harness.journal import content_key

PATH = "/agui"
#: 250,000 sales lines: a day, a region, units and revenue (about 7 MB of CSV)
REGIONS: Final = ("north", "south", "east", "west")
LINES: Final = [
    (
        f"2026-{1 + n % 9:02d}-{1 + n % 28:02d}",
        REGIONS[n % 4],
        1 + n % 7,
        f"{(n * 7919) % 10_000 / 100:.2f}",
    )
    for n in range(250_000)
]
CSV: Final = "day,region,units,revenue\n" + "\n".join(",".join(map(str, x)) for x in LINES)
KEPT: Final = f"/large_tool_results/{content_key('result', CSV)}"
BY_REGION: Final = (
    "SELECT region, sum(units) AS units, sum(revenue) AS revenue, count(*) AS lines "
    "FROM result GROUP BY region ORDER BY region"
)
#: A chart as the run's code draws it (an SVG: text, as the sandbox's write takes it)
CHART: Final = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="120" height="60">'
    '<rect x="0" y="10" width="100" height="10"/><rect x="0" y="30" width="60" height="10"/></svg>'
)


@tool(side_effects="read")
def sales_lines() -> str:
    """Every sales line of 2026 as CSV (day,region,units,revenue)."""
    return CSV


def truth() -> list[tuple[str, int, Decimal, int]]:
    """Each region's units, revenue (exact, as written) and lines, by region."""
    units: dict[str, int] = defaultdict(int)
    revenue: dict[str, Decimal] = defaultdict(Decimal)
    lines: dict[str, int] = defaultdict(int)
    for _, region, n, money in LINES:
        units[region] += n
        revenue[region] += Decimal(money)
        lines[region] += 1
    return [(r, units[r], revenue[r], lines[r]) for r in sorted(REGIONS)]


def user_of(request: Request) -> str:
    return request.headers.get("x-user", "")


@pytest.fixture
async def served(tmp_path: Path) -> AsyncIterator[tuple[Harness, FastAPI, httpx.AsyncClient]]:
    harness = Harness(config=Settings(files_url=tmp_path.as_uri()))
    app = FastAPI()  # each test serves its agent on it
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://agui", headers={"x-user": "u1"}
    ) as client:
        yield harness, app, client
    await harness.aclose()


async def asked(client: httpx.AsyncClient, question: str) -> tuple[str, list[dict[str, Any]]]:
    response = await client.post(
        f"{PATH}/run",
        json={"threadId": "t1", "messages": [{"id": "m1", "role": "user", "content": question}]},
    )
    events = [e for _, e in decode(response.text)]
    return events[-1]["runId"], events


def announced(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [e["value"] for e in events if e.get("name") == "file"]


async def test_a_250000_row_result_is_queried_exactly_and_downloaded_as_excel(
    served: tuple[Harness, FastAPI, httpx.AsyncClient],
) -> None:
    harness, app, client = served
    model = ScriptedChat(
        [
            ("sales_lines", {}),
            (
                "query_result",
                {"file_path": KEPT, "sql": BY_REGION, "save_as": "xlsx", "file_name": "by-region"},
            ),
            "Saved by-region.xlsx.",
        ]
    )
    agent = arm(harness.wrap(ReAct("s", model), id="big"), [sales_lines])
    agent.serve_chat(app, identity=user_of)
    run_id, events = await asked(client, "Units and revenue by region, as Excel")
    assert events[-1]["type"] == "RUN_FINISHED" and events[-1]["outcome"] == {"type": "success"}, (
        events[-1]
    )
    stub = str(model.requests[1]["messages"][-1]["content"])
    assert f"saved in the filesystem at this path: {KEPT}" in stub and len(stub) < 25_000
    shown = str(model.requests[2]["messages"][-1]["content"])
    expected = truth()
    head = shown.split("Its first rows:\n", 1)[1].splitlines()
    assert head[0] == "region,units,revenue,lines"
    rows = [line.split(",") for line in head[1:]]
    assert [(r, int(u), Decimal(v), int(n)) for r, u, v, n in rows] == expected, shown
    [file] = announced(events)
    assert file["file"] == "by-region.xlsx" and file["source"] == "query_result", file
    got = await client.get(f"{PATH}/runs/{run_id}/files/by-region.xlsx")
    assert got.status_code == 200, got.text[:300]
    book = load_workbook(io.BytesIO(got.content), read_only=True)
    rows = [tuple(r) for r in book.worksheets[0].iter_rows(values_only=True)]
    assert rows[0] == ("region", "units", "revenue", "lines")
    assert [(r, u, Decimal(str(v)), n) for r, u, v, n in rows[1:]] == expected


async def test_a_chart_the_runs_code_draws_is_downloaded_as_an_image(
    served: tuple[Harness, FastAPI, httpx.AsyncClient],
) -> None:
    harness, app, client = served
    provider = FakeSandboxes()
    model = ScriptedChat(
        [("sandbox_write", {"path": "outputs/units.svg", "content": CHART}), "Here is the chart."]
    )
    agent = arm(harness.wrap(ReAct("s", model), id="chart"), [sandbox(provider).tools()])
    agent.serve_chat(app, identity=user_of)
    run_id, events = await asked(client, "Chart units by region")
    assert events[-1]["outcome"] == {"type": "success"}, events[-1]
    [file] = announced(events)
    assert file == {
        "file": "units.svg",
        "size": len(CHART),
        "content_type": "image/svg+xml",
        "source": "sandbox_exec",
    }
    got = await client.get(f"{PATH}/runs/{run_id}/files/units.svg")
    assert got.status_code == 200 and got.text == CHART
    assert got.headers["content-type"].startswith("image/svg+xml")
    assert got.headers["content-disposition"] == 'attachment; filename="units.svg"'
    other = await client.get(f"{PATH}/runs/{run_id}/files/units.svg", headers={"x-user": "u2"})
    assert other.status_code == 404  # the run's own user's only
