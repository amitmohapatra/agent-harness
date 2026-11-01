"""The files kept for the user read back in a later run (``tools.stored``): a run of a chat with
files is told of them as it starts — each with what made it and when its sources were fetched,
from the file's record — ``read_file`` reads one (Excel as CSV rows, with how it was made) and
``/files`` lists every chat's; only ever the run's own user's, and a deleted file is said to be
gone."""

from __future__ import annotations

import io
import itertools
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Final

import pytest
from openpyxl import Workbook

from tests.support.arm import arm
from tests.support.models import ScriptedChat
from trellis import Harness, ReAct, tool
from trellis.contracts import RunStatus
from trellis.harness import runtime as runtime_module
from trellis.harness import snapshots
from trellis.harness.files import FileStore
from trellis.harness.journal import Journal, content_key
from trellis.harness.tools import results, stored

#: The SKUs on the first day, and the stock on the second (three of them changed)
DAY1: Final = {f"SKU-{n:03d}": (n * 7) % 40 for n in range(1, 61)}
DAY2: Final = {**DAY1, "SKU-005": 99, "SKU-017": 0, "SKU-042": 3}
FIRST: Final = datetime(2026, 10, 1, 9, 0, tzinfo=UTC)
SECOND: Final = datetime(2026, 10, 2, 9, 30, tzinfo=UTC)
LOW: Final = "SELECT sku, stock FROM result WHERE stock < 10 ORDER BY sku"
today: dict[str, int] = dict(DAY1)
#: what a run of a chat with files is told first
NOTED: Final = stored.NOTE
NOTED_SOME: Final = stored.NOTE.split(" (it is now")[0]
#: an id for each agent a test wraps
IDS: Final = (f"stock-{n}" for n in itertools.count())


def csv_of(stock: dict[str, int]) -> str:
    return "sku,stock\n" + "\n".join(f"{s},{u}" for s, u in stock.items())


@tool(side_effects="read")
def inventory() -> str:
    """Every SKU with its units in stock, as CSV."""
    return csv_of(today)


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> list[datetime]:
    """The time ``snapshots.now`` says: the first of the list."""
    times = [FIRST]
    monkeypatch.setattr(snapshots, "now", lambda: times[0])
    return times


@pytest.fixture
def root(harness: Harness, tmp_path: Path) -> Path:
    found = tmp_path / "files"
    harness.files = FileStore(found.as_uri())
    return found


def deleted(root: Path, name: str) -> None:
    next(root.rglob(name)).unlink()


def text(content: Any) -> str:
    """A message's content as text: a question's parts joined (the memory context is sent as
    the question's first part, audit K1)."""
    if isinstance(content, list):
        return "\n".join(str(part.get("text", "")) for part in content)
    return str(content)


def told(model: ScriptedChat, n: int = 0) -> str:
    """Everything request ``n`` holds, as text."""
    return "\n".join(text(m.get("content")) for m in model.requests[n]["messages"])


def last(model: ScriptedChat, n: int) -> str:
    return text(model.requests[n]["messages"][-1]["content"])


def offered(model: ScriptedChat, n: int) -> list[str]:
    return [t["function"]["name"] for t in model.requests[n].get("tools", [])]


async def first_run(harness: Harness, *, thread: str = "t", user: str = "u") -> str:
    """Run 1: the day's inventory, the low-stock rows saved as Excel. The file's path."""
    today.clear()
    today.update(DAY1)
    kept = f"/large_tool_results/{content_key('result', csv_of(DAY1))}"
    save = {"file_path": kept, "sql": LOW, "save_as": "xlsx", "file_name": "low"}
    model = ScriptedChat([("inventory", {}), ("query_result", save), "saved"])
    agent = arm(harness.wrap(ReAct(system_prompt="s", model=model), id=next(IDS)), [inventory])
    result = await agent.run("save the low stock as Excel", user=user, thread=thread)
    assert result.status is RunStatus.SUCCESS, result.error
    assert last(model, 2).startswith("Saved low.xlsx for the user"), last(model, 2)
    return f"/files/{result.run_id}/low.xlsx"


async def later_run(
    harness: Harness, turns: list[Any], *, thread: str = "t", user: str = "u"
) -> ScriptedChat:
    model = ScriptedChat(turns)
    agent = arm(harness.wrap(ReAct(system_prompt="s", model=model), id=next(IDS)), [inventory])
    result = await agent.run("what changed?", user=user, thread=thread)
    assert result.status is RunStatus.SUCCESS, result.error
    return model


async def test_a_later_run_is_told_of_its_chats_files_and_reads_how_one_was_made(
    harness: Harness, root: Path, clock: list[datetime]
) -> None:
    path = await first_run(harness)
    clock[0] = SECOND
    model = await later_run(harness, [("read_file", {"file_path": path}), "read"])
    said = told(model)
    line = (
        f"- {path} (",
        "bytes; saved 2026-10-01 09:00 UTC by query_result from inventory fetched "
        "2026-10-01 09:00 UTC)",
    )
    assert NOTED.format(now="2026-10-02 09:30 UTC") in said, said[-1_500:]
    assert line[0] in said and line[1] in said, said[-1_500:]
    assert {"read_file", "query_result"} <= set(offered(model, 0))  # from the first call
    page = last(model, 1)
    run = path.split("/")[2]
    assert page.startswith(
        f"{path} was saved 2026-10-01 09:00 UTC in run {run} by query_result with the query "
        f"{LOW}, over result = inventory fetched 2026-10-01 09:00 UTC. The run's tool calls: "
        "inventory() ok.\n\n"
    ), page[:600]
    rows = sorted((s, u) for s, u in DAY1.items() if u < 10)
    assert "1  sku,stock\n" in page and f"  {rows[-1][0]},{rows[-1][1]}" in page, page


async def test_another_users_files_are_neither_told_nor_read(
    harness: Harness, root: Path, clock: list[datetime]
) -> None:
    path = await first_run(harness, user="u")
    model = await later_run(harness, [("read_file", {"file_path": path}), "no"], user="v")
    assert NOTED_SOME not in told(model) and stored.ELSEWHERE not in told(model)
    assert last(model, 1) == stored.MISSING.format(path=path)  # v's own scope holds none
    model = await later_run(harness, [("read_file", {"file_path": "/files"}), "no"], user="v")
    assert last(model, 1) == "No files are kept for the user."


async def test_a_deleted_file_is_said_to_be_gone(
    harness: Harness, root: Path, clock: list[datetime]
) -> None:
    path = await first_run(harness)
    deleted(root, "low.xlsx")
    model = await later_run(harness, [("read_file", {"file_path": path}), "gone"])
    made = "saved 2026-10-01 09:00 UTC by query_result from inventory fetched 2026-10-01 09:00 UTC"
    assert f"- {path}: deleted, no longer kept ({made})" in told(model), told(model)
    assert last(model, 1) == stored.GONE.format(path=path, made=made)


async def test_another_chats_files_are_listed_on_asking(
    harness: Harness, root: Path, clock: list[datetime]
) -> None:
    path = await first_run(harness, thread="monday")
    model = await later_run(harness, [("read_file", {"file_path": "/files"}), "listed"])
    assert stored.ELSEWHERE in told(model) and NOTED_SOME not in told(model)
    listing = last(model, 1)
    run = path.split("/")[2]
    assert listing.startswith("The user's files, newest first:\n- /files/conv_"), listing
    assert f"/{run}/low.xlsx (" in listing, listing
    other = listing.split()[6]  # /files/<chat>/<run>/low.xlsx
    model = await later_run(harness, [("read_file", {"file_path": other, "limit": 2}), "r"])
    assert "  sku,stock\n" in last(model, 1), last(model, 1)


async def test_a_store_that_cannot_list_is_a_warning(
    harness: Harness, root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def broken(folder: str) -> list[str]:
        raise OSError("the bucket is gone")

    monkeypatch.setattr(harness.files, "folders", broken)
    model = ScriptedChat(["hi"])
    agent = arm(harness.wrap(ReAct(system_prompt="s", model=model), id="w"), [inventory])
    events = [e async for e in agent.stream("x", user="u")]
    warned = [e.data for e in events if e.data and e.data.get("name") == "warning"]
    assert [w["message"] for w in warned if w["code"] == "files"] == [
        "the files kept for the user: the bucket is gone"
    ], warned
    assert "read_file" not in offered(model, 0)


# --------------------------------------------------------------------------- reading
class Run:
    """A run, as ``tools.stored`` reads it."""

    def __init__(self, store: FileStore, **more: Any) -> None:
        self.tenant, self.user, self.conversation, self.run_id = "t", "u", "c_1", "r1"
        self.parent: Any = None
        self.replay = SimpleNamespace(journal=Journal())
        self.agent = SimpleNamespace(
            harness=SimpleNamespace(files=store), adapter=SimpleNamespace(name="langgraph")
        )
        self.events = SimpleNamespace(warning=lambda *a: None)
        self.__dict__.update(more)


@pytest.fixture
def run(tmp_path: Path) -> Iterator[Run]:
    given = Run(FileStore((tmp_path / "s").as_uri()))
    token = runtime_module._current.set(given)  # type: ignore[arg-type]
    yield given
    runtime_module._current.reset(token)


def sheet(*rows: tuple[Any, ...]) -> bytes:
    book = Workbook()
    for row in rows:
        book.active.append(row)  # type: ignore[union-attr]
    out = io.BytesIO()
    book.save(out)
    return out.getvalue()


async def test_an_excel_file_reads_as_csv_rows(run: Run) -> None:
    data = sheet(("when", None), (datetime(2026, 1, 2, 3, 4), 5), (None, None), ("x", 6))
    await run.agent.harness.files.put("t/u/c_1/r0", "a.xlsx", data, source="query_result")
    said = await results.aread({"file_path": "/files/r0/a.xlsx"})
    assert said == (
        "/files/r0/a.xlsx: how it was made is not recorded.\n\n"
        "1  when,c2\n2  2026-01-02T03:04:00,5\n3  x,6"
    ), said
    assert await results.aread({"file_path": "/files/r0/a.xlsx", "offset": 2}) == "3  x,6"


async def test_a_csv_file_reads_as_it_is_after_how_it_was_made(run: Run) -> None:
    files = run.agent.harness.files
    await files.put("t/u/c_1/r0", "d.csv", b"a,b\n1,2\n", source="sandbox_exec")
    await files.keep_record("t/u/c_1/r0", "d.csv", {"run": "r0", "source": "sandbox_exec"})
    said = await stored.read("/files/r0/d.csv")
    assert said == (
        "/files/r0/d.csv was saved an unknown time in run r0 by sandbox_exec. The run's tool "
        "calls: none.\n\n1  a,b\n2  1,2"
    ), said
    await files.put("t/u/c_1/r0", "e.xlsx", b"not a zip", source="sandbox_exec")
    said = await stored.read("/files/r0/e.xlsx")
    assert said.startswith("Error: /files/r0/e.xlsx is no table: BadZipFile"), said


async def test_a_file_that_is_no_table_or_too_large_says_so(
    run: Run, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = run.agent.harness.files
    await files.put("t/u/c_1/r0", "b.csv", b"\xff\xfe", source="sandbox_exec")
    said = await stored.read("/files/r0/b.csv")
    assert said.startswith("Error: /files/r0/b.csv is no table: UnicodeDecodeError"), said
    monkeypatch.setattr(stored, "MAX_BYTES", 1)
    said = await stored.read("/files/r0/b.csv")
    assert said == "Error: /files/r0/b.csv is 2 bytes, more than 1 read here", said


async def test_a_file_gone_between_its_size_and_its_bytes_is_missing(
    run: Run, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = run.agent.harness.files
    await files.put("t/u/c_1/r0", "c.csv", b"a\n1", source="x")

    async def none(folder: str, name: str) -> None:
        return None

    monkeypatch.setattr(files, "read", none)
    assert await stored.read("/files/r0/c.csv") == stored.MISSING.format(path="/files/r0/c.csv")


@pytest.mark.parametrize(
    "path",
    ["/files/a", "/files/a/b/c/d.csv", "/files/../r0/c.csv", "/files/r0/.c.csv.json"],
)
async def test_a_path_that_names_no_kept_file_is_refused(run: Run, path: str) -> None:
    said = await stored.read(path)
    assert said.startswith(f"Error: {path} is no kept file's path"), said


async def test_outside_a_run_nothing_is_read() -> None:
    assert await stored.read("/files/r0/a.csv") == "Error: File '/files/r0/a.csv' not found"
    assert await stored.table("/files/r0/a.csv") == "Error: File '/files/r0/a.csv' not found"
    assert results.kept() is False


async def test_bad_arguments_for_a_kept_file_are_read_files_own_refusal(run: Run) -> None:
    said = await results.aread({"file_path": "/files/r0/a.csv", "offset": "x"})
    assert said.startswith("read_file was not run: offset must be of type integer"), said


async def test_the_note_names_the_newest_and_counts_the_rest(
    run: Run, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = run.agent.harness.files
    monkeypatch.setattr(stored, "NOTED", 2)
    for n in range(3):
        await files.put("t/u/c_1/r0", f"f{n}.csv", b"a\n1", source="sandbox_exec")
        await files.keep_record(
            "t/u/c_1/r0", f"f{n}.csv", {"saved_at": f"2026-10-0{n + 1}T00:00:00+00:00"}
        )
    said = await stored.note(run)  # type: ignore[arg-type]
    assert said is not None
    assert said.splitlines()[1:] == [
        "- /files/r0/f2.csv (3 bytes; saved 2026-10-03 00:00 UTC by an unknown tool)",
        "- /files/r0/f1.csv (3 bytes; saved 2026-10-02 00:00 UTC by an unknown tool)",
        '- and 1 more: read_file(file_path="/files")',
    ], said
    assert stored.looks(run)  # type: ignore[arg-type]
    run.parent = object()
    assert not stored.looks(run)  # type: ignore[arg-type]  # its parent was told


async def test_how_a_file_was_made_says_what_is_known(run: Run) -> None:
    record = {
        "run": "r0",
        "saved_at": "not a time",
        "source": "sandbox_exec",
        "calls": [{"tool": "fetch", "args": {"q": "x" * 400}, "status": "ok"}],
    }
    said = stored.how("/files/r0/a.csv", record)
    assert said.startswith(
        "/files/r0/a.csv was saved an unknown time in run r0 by sandbox_exec. The run's tool "
        "calls: fetch(q='xxx"
    ), said
    assert said.endswith("x…) ok."), said
    assert stored.how("/p", {"run": "r", "source": "s"}).endswith("The run's tool calls: none.")


async def test_a_run_told_of_files_is_offered_the_readers(run: Run) -> None:
    run.files_note = "files"  # type: ignore[attr-defined]
    assert results.kept() is True
    stored.told(run, None)  # type: ignore[arg-type]
