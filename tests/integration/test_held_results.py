"""A result the run holds is queryable exactly, wherever it lives: every kind of backend
``ReAct`` takes (none, the graph's state, a disk, a store, a sandbox) crossed with every way a
result is held (shown whole, too large to show, cleared from the window, cut to fit a small
window, many in one step of one) and every shape its rows come in (CSV, a JSON array, a JSON
object). In each, the request that follows offers ``query_result``, the result's message names
its path, its table and its own columns, and one query over what it names gives the same exact
figure — with or without a sandbox."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, cast

import pytest
from deepagents.backends import FilesystemBackend, StoreBackend
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatResult
from langgraph.store.memory import InMemoryStore

from tests.support.arm import arm
from tests.support.chat_model import SAVED, KeyedChatModel
from tests.support.sandbox import FakeSandboxes
from trellis import Harness, ReAct, sandbox, tool
from trellis.harness.automatic import JournalBackend
from trellis.harness.react import CLEARED
from trellis.harness.tools import query, results

#: How a result is held, and the rows of the tool's result that make it so: few enough to show
#: whole (kept as well: a table), too many to show (kept, a preview shown), few rows of long
#: cells called often enough that the older ones are cleared from the window (``few``: too few
#: rows to be kept as a table, so the cleared one is read from the graph's state), few rows of
#: cells so long that one result does not fit a small window (cut to its head and tail), and
#: many results in one step of a small window (``crowded``: some cleared, the rest cut).
HELD = {"whole": 40, "large": 30_000, "cleared": 25, "few": 3, "cut": 25, "crowded": 24}
CALLS = {"whole": 1, "large": 1, "cleared": 8, "few": 8, "cut": 1, "crowded": 16}
CELL = {"cleared": 470, "few": 4_000, "cut": 2_400, "crowded": 300}
WINDOW = {
    "whole": 40_000, "large": 40_000, "cleared": 40_000, "few": 40_000, "cut": 16_000,
    "crowded": 16_000,
}  # fmt: skip
#: The ways whose asked result is the first call's (cleared by the time it is asked about).
FIRST = ("cleared", "few", "crowded")
#: The shapes a tool's rows come in: CSV, a JSON array of objects, a JSON object holding one.
SHAPES = ("csv", "array", "object")
QUESTION = "how many units?"
HELD_PATH, HELD_TABLE = "{held}", "{table}"
SQL = f"SELECT sum(units) AS u, count(*) AS n FROM {HELD_TABLE}"
#: What a held result's message says of it: its table, its columns, the path to query.
_SAID = re.compile(
    r"Its rows are the table (\w+) \(columns: ([^)]*)\)\. .*?"
    rf'{query.QUERY_RESULT}\(file_path="({results.PREFIX}/\w+)"'
)
#: How a mark starts: a cleared result's, and a stub's (``results.CLEARED``, ``EVICTED``).
MARKS = (CLEARED.split(":")[0], results.EVICTED)


class Asking(KeyedChatModel):
    """A scripted model that asks about the result one tool message of its request names:
    ``{held}`` and ``{table}`` in a call's arguments are the path and the table that message
    (the ``asks``-th tool message) gives for it — what a model reading it would write."""

    asks: int = -1

    def _generate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> ChatResult:
        made = super()._generate(messages, *args, **kwargs)
        told = [m.text for m in messages if isinstance(m, ToolMessage)]
        found = _SAID.search(told[self.asks]) if told else None
        for call in cast(AIMessage, made.generations[0].message).tool_calls:
            for key, value in call["args"].items():
                if found is not None and isinstance(value, str):
                    call["args"][key] = value.replace(HELD_PATH, found[3]).replace(
                        HELD_TABLE, found[1]
                    )
        return made


def backends(tmp_path: Path) -> dict[str, Any]:
    """Every kind of backend ``ReAct`` takes, by name (``None``: none given)."""
    return {
        "none": None,
        "state": JournalBackend(),
        "disk": FilesystemBackend(root_dir=tmp_path, virtual_mode=True),
        "store": StoreBackend(namespace=lambda _: ("held",), store=InMemoryStore()),
        "sandbox": sandbox(FakeSandboxes()),
    }


def table(way: str, shape: str = "csv", page: int = 0) -> str:
    """The tool's result for ``way`` in ``shape``: ``sku`` and ``units`` rows (a third column
    of long cells where the result is to be cleared or cut, each call's its own)."""
    cell = f"page {page} " + "w" * CELL[way] if way in CELL else None
    rows: list[dict[str, Any]] = [
        {"sku": f"s{n}", "units": n % 7} | ({} if cell is None else {"note": cell})
        for n in range(HELD[way])
    ]
    if shape == "array":
        return json.dumps(rows)
    if shape == "object":
        found = {"store": "S20", "page": page, "rows": rows, "units": sum(r["units"] for r in rows)}
        return json.dumps(found)
    return ",".join(rows[0]) + "\n" + "\n".join(",".join(map(str, r.values())) for r in rows)


def expected(way: str) -> str:
    return f"u,n\n{sum(n % 7 for n in range(HELD[way]))},{HELD[way]}"


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("way", list(HELD))
@pytest.mark.parametrize("kind", ["none", "state", "disk", "store", "sandbox"])
async def test_a_held_result_is_offered_the_query_with_its_table_named(
    harness: Harness, tmp_path: Path, kind: str, way: str, shape: str
) -> None:
    """Every backend, every way a result is held, every shape its rows come in: what its
    message says of it — shown with a note, a stub, a cleared result's mark, a cut — names the
    result's own columns, the path and the table it names give its rows to a query, exactly,
    and no request holds a mark of a mark (a mark is never read, kept or cut as if it were a
    result)."""

    @tool(side_effects="read")
    def sales(page: int = 0) -> str:
        """Sales rows."""
        return table(way, shape, page)

    fetch = [("sales", {"page": n}) for n in range(CALLS[way])]
    # the result asked about: the last one — and where older results are cleared, the first
    # call's — by the path and the table its message names
    ask = ("query_result", {"file_path": HELD_PATH, "sql": SQL})
    steps = [fetch, ask, "done"] if way == "crowded" else [*fetch, ask, "done"]
    model = Asking(turns=[], script=[(QUESTION, steps)], asks=0 if way in FIRST else -1)
    backend = backends(tmp_path)[kind]
    given = {} if backend is None else {"backend": backend}
    target = ReAct("s", model, context_window=WINDOW[way], **given)
    agent = arm(harness.wrap(target, id=f"held-{kind}-{way}-{shape}"), [sales])
    result = await agent.run(QUESTION, user="u")
    assert result.answer == "done", result.error

    asked, answered = model.seen[-2], model.seen[-1]
    # the request the query was made from offered it, and its message named the result's own
    # columns, whatever state it is in
    assert query.QUERY_RESULT in model.requests[-2]["tools"]
    told = [m.text for m in asked if isinstance(m, ToolMessage)]
    said = told[0 if way in FIRST else -1]
    found = _SAID.search(said)
    assert found is not None, said[-700:]
    columns = ["sku", "units", "note"] if way in CELL else ["sku", "units"]
    assert found[2].split(", ") == columns, said[:700]
    if way in FIRST:
        assert said.startswith(MARKS[0]), said[:200]  # the window no longer shows it
    if way in ("cut", "crowded"):
        assert any("cut to fit" in text for text in told)
    # the query over the path and the table it named is exact — the same on every backend
    assert answered[-1].text == expected(way)
    # and no mark is a result: none inside another message, none cut as a result is
    for request in model.seen:
        for text in (m.text for m in request if isinstance(m, ToolMessage)):
            marks = [mark for mark in MARKS if mark in text]
            assert not marks or text.startswith(tuple(marks)), text[:300]
            assert len(marks) < 2 and not (marks and "cut to fit" in text), text[:600]


async def test_with_a_sandbox_a_querys_rows_are_a_file_there_where_code_starts(
    harness: Harness,
) -> None:
    """The hand-off from the query to the sandbox: the query computes exactly, and what code
    makes of its result (a chart, a workbook) starts from the result as a file — the answer
    names it, the sandbox holds it at that path, and both tools' descriptions say which does
    what. Without a sandbox the same query answers with the same rows, and names no file."""
    rows = "SELECT sku, units FROM result WHERE units = 6 ORDER BY sku LIMIT 3"
    turns = [
        ("sales", {}),
        ("query_result", {"file_path": SAVED, "sql": rows}),
        ("execute", {"command": f"cat {SAVED}"}),
        "done",
    ]

    @tool(side_effects="read")
    def sales() -> str:
        """Sales rows."""
        return table("large")

    model = KeyedChatModel(turns=[], script=[(QUESTION, turns)])
    target = ReAct("s", model, backend=sandbox(FakeSandboxes()), context_window=128_000)
    agent = arm(harness.wrap(target, id="handed"), [sales])
    assert (await agent.run(QUESTION, user="u")).answer == "done"
    answered, read = model.seen[-2][-1].text, model.seen[-1][-1].text
    found = "sku,units\ns1000,6\ns10002,6\ns10009,6"
    path = re.search(rf"{results.PREFIX}/\w+", answered)
    assert answered.startswith(found) and path is not None, answered
    assert results.CODE.format(path=path.group(), tool="execute") in answered
    assert read.startswith(found), read  # the file the note names, read in the sandbox
    said = model.described[-1]
    assert "sandbox" in said[query.QUERY_RESULT] and query.QUERY_RESULT in said["execute"]

    plain = KeyedChatModel(turns=[], script=[(QUESTION, [*turns[:2], "done"])])
    alone = arm(harness.wrap(ReAct("s", plain, context_window=128_000), id="alone"), [sales])
    assert (await alone.run(QUESTION, user="u")).answer == "done"
    assert plain.seen[-1][-1].text == found  # the same rows, and no file named
    assert "sandbox" not in plain.described[-1][query.QUERY_RESULT]
