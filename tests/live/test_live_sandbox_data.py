"""Data work in a real Docker sandbox, of a data image (``tests/live/sandbox-data``: Python 3.12
with pandas, openpyxl, matplotlib and statsmodels; ``SANDBOX_DATA_IMAGE`` names it, else
``trellis-sandbox-data``): a 50k-row result the run keeps is a file in its sandbox, which code
reads whole — through ``ReAct(backend=sandbox())`` and through ``sandbox().tools()`` on other
adapters — and writes an Excel workbook and a PNG chart from; a module the image lacks is said,
and cannot be installed without a network; two users' runs at once work in sandboxes of their
own. Skipped without the daemon or the image."""

from __future__ import annotations

import asyncio
import csv
import io
import json
import os
from pathlib import Path
from typing import Any, Final

import httpx
import pytest

from tests.support.adapters import BUILDERS
from tests.support.arm import arm
from tests.support.chat_model import ScriptedChatModel
from trellis import Harness, ReAct, Runtime, sandbox, tool
from trellis.contracts import RunEvent, RunEventType, RunStatus
from trellis.harness.sandbox import DockerSandbox
from trellis.harness.sandbox.base import TENANT
from trellis.harness.sandbox.docker import API_VERSION, docker_host

IMAGE: Final = os.environ.get("SANDBOX_DATA_IMAGE") or "trellis-sandbox-data"
PROVIDER: Final = DockerSandbox(IMAGE)
ROWS: Final = 50_000
REGIONS: Final = ("north", "south", "east", "west")


def _image_here() -> bool:
    """Whether the daemon ``DOCKER_HOST`` names answers and has the data image."""
    host = docker_host()
    socket = host.removeprefix("unix://") if host.startswith("unix://") else None
    base = "http://docker" if socket else f"http://{host.split('://', 1)[-1]}"
    try:
        with httpx.Client(transport=httpx.HTTPTransport(uds=socket), base_url=base) as client:
            return client.get(f"/{API_VERSION}/images/{IMAGE}/json", timeout=3).is_success
    except httpx.HTTPError:
        return False


pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not _image_here(), reason=f"needs a Docker daemon with {IMAGE}"),
]


def sales() -> str:
    out = io.StringIO()
    rows = csv.writer(out)
    rows.writerow(["order_id", "region", "units", "price"])
    rows.writerows([n, REGIONS[n % 4], n % 7 + 1, 1 + (n % 13) * 0.5] for n in range(ROWS))
    return out.getvalue()


def revenue() -> dict[str, float]:
    """The truth, computed here: each region's units times price, over every row."""
    totals = dict.fromkeys(REGIONS, 0.0)
    for n in range(ROWS):
        totals[REGIONS[n % 4]] += (n % 7 + 1) * (1 + (n % 13) * 0.5)
    return {region: round(total, 2) for region, total in totals.items()}


@tool
def sales_rows() -> str:
    """Every order of the quarter, as CSV (order_id, region, units, price)."""
    return sales()


#: Reads the result the run keeps (the one file the run's sandbox has of it), writes a workbook
#: and a chart, and reads them back: openpyxl opens the workbook, the chart has PNG's signature.
ANALYSE: Final = r"""
import glob, json, os
import matplotlib.pyplot as plt, openpyxl, pandas as pd
[path] = glob.glob("/large_tool_results/*")
df = pd.read_csv(path)
by = (df.units * df.price).groupby(df.region).sum().round(2)
os.makedirs("out", exist_ok=True)
by.rename("revenue").to_frame().to_excel("out/report.xlsx")
by.plot.bar().figure.savefig("out/chart.png")
print(json.dumps({
    "rows": len(df),
    "revenue": by.to_dict(),
    "sheets": openpyxl.load_workbook("out/report.xlsx").sheetnames,
    "png": open("out/chart.png", "rb").read(8) == b"\x89PNG\r\n\x1a\n",
}))
"""


def results(events: list[RunEvent]) -> list[tuple[str, Any]]:
    return [
        (e.data["tool"], e.data["output"])
        for e in events
        if e.type is RunEventType.TOOL_CALL_RESULT
    ]


async def left() -> list[str]:
    """The sandboxes of this tenant still there."""
    return [ref.id for ref in await PROVIDER.labelled({TENANT: "default"})]


async def test_react_runs_code_over_a_kept_50k_row_result_and_writes_a_workbook_and_a_chart(
    harness: Harness,
) -> None:
    model = ScriptedChatModel(
        turns=[
            ("sales_rows", {}),
            ("execute", {"command": f"python - <<'EOF'\n{ANALYSE}\nEOF"}),
            "done",
        ]
    )
    graph = ReAct("You analyse sales.", model, tools=[sales_rows], backend=sandbox(PROVIDER))
    agent = harness.wrap(graph, id="analyst")
    events = [e async for e in agent.stream("revenue by region, in Excel and a chart", user="u")]
    finished = events[-1]
    assert finished.outcome is not None and finished.outcome.value == "success", finished
    read = str(model.seen[1][-1].content)  # what the model read of the 50k rows
    assert "/large_tool_results/" in read and len(read) < 20_000
    [_, (_, ran)] = results(events)
    report = json.loads(str(ran)[str(ran).index("{") : str(ran).rindex("}") + 1])
    assert report == {"rows": ROWS, "revenue": revenue(), "sheets": ["Sheet1"], "png": True}
    assert await left() == []  # deleted with the run, its files with it


@pytest.mark.parametrize("framework", ["middleware", "openai_agents"])
async def test_an_adapter_with_the_sandbox_tools_runs_code_over_the_kept_result(
    harness: Harness, framework: str, tmp_path: Path
) -> None:
    plan = [
        ("sales_rows", {}),
        ("sandbox_write", {"path": "analyse.py", "content": ANALYSE}),
        ("sandbox_exec", {"command": "python analyse.py"}),
    ]
    box = sandbox(PROVIDER)
    target, tools = await BUILDERS[framework](harness, [sales_rows, box.tools()], tmp_path, plan)
    agent = arm(harness.wrap(target, id=f"analyst-{framework}"), tools)
    events = [e async for e in agent.stream("revenue by region", user="u")]
    [*_, (tool_name, ran)] = results(events)
    ran = ran if isinstance(ran, dict) else json.loads(ran)
    assert tool_name == "sandbox_exec" and ran["exit_code"] == 0, ran
    report = json.loads(ran["stdout"])
    assert report == {"rows": ROWS, "revenue": revenue(), "sheets": ["Sheet1"], "png": True}
    assert await left() == []


async def test_a_module_the_image_lacks_is_said_and_not_installed_without_a_network(
    harness: Harness,
) -> None:
    async def work(input: str, agent: Runtime) -> Any:
        missing = await agent.tools.call("sandbox_exec", command="python -c 'import duckdb'")
        pip = await agent.tools.call(
            "sandbox_exec", command="timeout 60 pip install --no-cache-dir duckdb 2>&1"
        )
        return [missing, pip["stdout"]]

    agent = arm(harness.wrap(work, id="modules"), [sandbox(PROVIDER).tools()])
    result = await agent.run("go", user="u")
    assert result.status is RunStatus.SUCCESS, result.error
    missing, pip = result.answer
    assert missing["exit_code"] == 1
    assert "No module named 'duckdb'" in missing["stderr"]
    assert "No matching distribution found for duckdb" in pip  # no network: no index


async def test_two_users_runs_at_once_work_in_sandboxes_of_their_own(harness: Harness) -> None:
    both = asyncio.Barrier(2)

    async def work(input: str, agent: Runtime) -> Any:
        await agent.tools.call("sandbox_write", path="secret.txt", content=f"secret of {input}")
        await asyncio.wait_for(both.wait(), 60)  # both sandboxes there at once
        seen = await agent.tools.call("sandbox_exec", command="cat /workspace/*.txt")
        return [seen["stdout"], await left()]

    agent = arm(harness.wrap(work, id="pair"), [sandbox(PROVIDER).tools()])
    ada, bob = await asyncio.gather(agent.run("ada", user="ada"), agent.run("bob", user="bob"))
    assert ada.answer[0] == "secret of ada" and bob.answer[0] == "secret of bob"
    assert sorted(ada.answer[1]) == sorted(bob.answer[1])  # two sandboxes, one per run
    assert sorted(ada.answer[1]) == sorted(f"trellis-{r.run_id}" for r in (ada, bob))
    assert await left() == []
