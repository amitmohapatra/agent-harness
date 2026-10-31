"""An earlier file compared with today's data in one query: ``query_result`` loads a file kept
for the user from the store as the table ``result`` (``tools.stored.table``: no copy kept in
the run), beside this run's results, and says where each table's rows come from — the file's
source and fetch time, this run's — so the comparison cites both runs. Every framework."""

from __future__ import annotations

import itertools
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import pytest

from tests.support.adapters import BUILDERS
from tests.support.arm import arm
from tests.support.models import ScriptedChat
from tests.support.planned import Call
from trellis import Harness, ReAct, tool
from trellis.contracts import RunStatus
from trellis.harness import snapshots
from trellis.harness.files import FileStore
from trellis.harness.journal import content_key
from trellis.harness.tools import stored

DAY1: Final = {f"SKU-{n:03d}": (n * 7) % 40 for n in range(1, 61)}
DAY2: Final = {**DAY1, "SKU-005": 99, "SKU-017": 0, "SKU-042": 3}
FIRST: Final = datetime(2026, 10, 1, 9, 0, tzinfo=UTC)
SECOND: Final = datetime(2026, 10, 2, 9, 30, tzinfo=UTC)
DIFF: Final = (
    "SELECT f.sku, f.stock AS was, i.stock AS now FROM result f JOIN inventory i "
    "ON i.sku = f.sku WHERE f.stock <> i.stock ORDER BY f.sku"
)
today: dict[str, int] = {}
IDS: Final = (f"cmp-{n}" for n in itertools.count())


def csv_of(stock: dict[str, int]) -> str:
    return "sku,stock\n" + "\n".join(f"{s},{u}" for s, u in stock.items())


@tool(side_effects="read")
def inventory() -> str:
    """Every SKU with its units in stock, as CSV."""
    return csv_of(today)


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> list[datetime]:
    times = [FIRST]
    monkeypatch.setattr(snapshots, "now", lambda: times[0])
    return times


@pytest.fixture
def root(harness: Harness, tmp_path: Path) -> Path:
    found = tmp_path / "files"
    harness.files = FileStore(found.as_uri())
    return found


async def saved_on_day_one(harness: Harness, kind: str = "xlsx") -> str:
    """Run 1: the whole inventory saved as a file. The path a later run reads it at."""
    today.clear()
    today.update(DAY1)
    kept = f"/large_tool_results/{content_key('result', csv_of(DAY1))}"
    save = {"file_path": kept, "sql": "SELECT * FROM result", "save_as": kind, "file_name": "stock"}
    model = ScriptedChat([("inventory", {}), ("query_result", save), "saved"])
    agent = arm(harness.wrap(ReAct(system_prompt="s", model=model), id=next(IDS)), [inventory])
    result = await agent.run("save the stock", user="u", thread="t")
    assert result.status is RunStatus.SUCCESS, result.error
    return f"/files/{result.run_id}/stock.{kind}"


def kept(root: Path, name: str) -> tuple[str, dict[str, Any]]:
    """A kept file's text, and its record."""
    [found] = list(root.rglob(name))
    return found.read_text(), json.loads((found.parent / f".{name}.json").read_text())


def deleted(root: Path, name: str) -> None:
    next(root.rglob(name)).unlink()


def python_diff() -> str:
    changed = [(s, DAY1[s], DAY2[s]) for s in sorted(DAY1) if DAY1[s] != DAY2[s]]
    return "sku,was,now\n" + "\n".join(f"{s},{a},{b}" for s, a, b in changed)


@pytest.mark.parametrize(
    "framework", ["react", "middleware", "openai_agents", "claude_agent_sdk", "kimi"]
)
async def test_a_later_run_diffs_the_file_against_todays_data_in_one_query(
    harness: Harness, root: Path, clock: list[datetime], framework: str, tmp_path: Path
) -> None:
    path = await saved_on_day_one(harness)
    today.clear()
    today.update(DAY2)
    clock[0] = SECOND
    plan: list[Call] = [("inventory", {}), ("query_result", {"file_path": path, "sql": DIFF})]
    target, own = await BUILDERS[framework](harness, [inventory], tmp_path, plan)
    agent = arm(harness.wrap(target, id=next(IDS)), own)
    result = await agent.run("what changed since the file?", user="u", thread="t")
    assert result.status is RunStatus.SUCCESS, result.error
    cited, _, rows = str(result.answer).removeprefix("Done. ").partition("\n")
    assert rows == python_diff()  # exact, against Python's own
    assert cited == (
        f"result is {path} (saved 2026-10-01 09:00 UTC by query_result from inventory fetched "
        "2026-10-01 09:00 UTC); inventory is this run's inventory result, fetched "
        "2026-10-02 09:30 UTC."
    ), cited


async def test_a_diff_saved_as_a_file_records_both_sources(
    harness: Harness, root: Path, clock: list[datetime]
) -> None:
    path = await saved_on_day_one(harness, "csv")
    today.clear()
    today.update(DAY2)
    clock[0] = SECOND
    save = {"file_path": path, "sql": DIFF, "save_as": "csv", "file_name": "changes"}
    model = ScriptedChat([("inventory", {}), ("query_result", save), "ok"])
    agent = arm(harness.wrap(ReAct(system_prompt="s", model=model), id=next(IDS)), [inventory])
    result = await agent.run("save what changed", user="u", thread="t")
    assert result.status is RunStatus.SUCCESS, result.error
    text, record = kept(root, "changes.csv")
    assert text.strip() == python_diff()
    assert stored.made(record) == (
        f"saved 2026-10-02 09:30 UTC by query_result from {path} fetched 2026-10-01 09:00 UTC, "
        "inventory fetched 2026-10-02 09:30 UTC"
    ), record


async def test_a_deleted_file_cannot_be_compared(
    harness: Harness, root: Path, clock: list[datetime]
) -> None:
    path = await saved_on_day_one(harness)
    deleted(root, "stock.xlsx")
    model = ScriptedChat([("query_result", {"file_path": path, "sql": DIFF}), "gone"])
    agent = arm(harness.wrap(ReAct(system_prompt="s", model=model), id=next(IDS)), [inventory])
    assert (await agent.run("compare", user="u", thread="t")).status is RunStatus.SUCCESS
    told = str(model.requests[1]["messages"][-1]["content"])
    made = "saved 2026-10-01 09:00 UTC by query_result from inventory fetched 2026-10-01 09:00 UTC"
    assert told == stored.GONE.format(path=path, made=made), told


async def test_another_users_file_cannot_be_queried(
    harness: Harness, root: Path, clock: list[datetime]
) -> None:
    path = await saved_on_day_one(harness)
    model = ScriptedChat([("query_result", {"file_path": path, "sql": "SELECT 1"}), "no"])
    agent = arm(harness.wrap(ReAct(system_prompt="s", model=model), id=next(IDS)), [inventory])
    assert (await agent.run("x", user="v", thread="t")).status is RunStatus.SUCCESS
    told = str(model.requests[1]["messages"][-1]["content"])
    assert told == stored.MISSING.format(path=path)
