"""A result the run holds is queryable exactly, wherever it lives: every kind of backend
``ReAct`` takes (none, the graph's state, a disk, a store, a sandbox) crossed with every way a
result is held (shown whole, too large to show, cleared from the window, cut to fit a small
window). In each, the request that follows offers ``query_result``, the result's message names its path and its table, and
one query over it gives the same exact figure — with or without a sandbox."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
from deepagents.backends import FilesystemBackend, StoreBackend
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.store.memory import InMemoryStore

from tests.support.arm import arm
from tests.support.chat_model import SAVED, KeyedChatModel
from tests.support.sandbox import FakeSandboxes
from trellis import Harness, ReAct, sandbox, tool
from trellis.harness.automatic import JournalBackend
from trellis.harness.tools import query, results

#: How a result is held, and the rows of the tool's result that make it so: few enough to show
#: whole (kept as well: a table), too many to show (kept, a preview shown), few rows of long
#: cells called often enough that the older ones are cleared from the window, and few rows of
#: cells so long that one result does not fit a small window (cut to its head and tail).
HELD = {"whole": 40, "large": 30_000, "cleared": 25, "cut": 25}
CALLS = {"whole": 1, "large": 1, "cleared": 8, "cut": 1}
CELL = {"cleared": 470, "cut": 2_400}
WINDOW = {"whole": 40_000, "large": 40_000, "cleared": 40_000, "cut": 16_000}
QUESTION = "how many units?"
SQL = f"SELECT sum(units) AS u, count(*) AS n FROM {query.TABLE}"
_TABLE = re.compile(r"Its rows are the table (\w+) \(columns: sku, units")


def backends(tmp_path: Path) -> dict[str, Any]:
    """Every kind of backend ``ReAct`` takes, by name (``None``: none given)."""
    return {
        "none": None,
        "state": JournalBackend(),
        "disk": FilesystemBackend(root_dir=tmp_path, virtual_mode=True),
        "store": StoreBackend(namespace=lambda _: ("held",), store=InMemoryStore()),
        "sandbox": sandbox(FakeSandboxes()),
    }


def table(way: str, page: int = 0) -> str:
    """The tool's result for ``way``: ``sku,units`` rows (a third column of long cells where
    the result is to be cleared, each call's its own)."""
    if way not in CELL:
        return "sku,units\n" + "\n".join(f"s{n},{n % 7}" for n in range(HELD[way]))
    cell = f"page {page} " + "w" * CELL[way]
    return "sku,units,note\n" + "\n".join(f"s{n},{n % 7},{cell}" for n in range(HELD[way]))


def expected(way: str) -> str:
    return f"u,n\n{sum(n % 7 for n in range(HELD[way]))},{HELD[way]}"


@pytest.mark.parametrize("way", list(HELD))
@pytest.mark.parametrize("kind", ["none", "state", "disk", "store", "sandbox"])
async def test_a_held_result_is_offered_the_query_with_its_table_named(
    harness: Harness, tmp_path: Path, kind: str, way: str
) -> None:
    @tool(side_effects="read")
    def sales(page: int = 0) -> str:
        """Sales rows."""
        return table(way, page)

    fetch = [("sales", {"page": n}) for n in range(CALLS[way])]
    # the result asked about: the last one the model was told is held — and where older
    # results are cleared, the first call's, by the path its mark names (a table of 5 rows or
    # more is held from the start, so the last one shown names a path too: not the cleared one)
    held = f"{results.PREFIX}/call_1_0" if way == "cleared" else SAVED
    ask = ("query_result", {"file_path": held, "sql": SQL})
    model = KeyedChatModel(turns=[], script=[(QUESTION, [*fetch, ask, "done"])])
    backend = backends(tmp_path)[kind]
    given = {} if backend is None else {"backend": backend}
    target = ReAct("s", model, context_window=WINDOW[way], **given)
    agent = arm(harness.wrap(target, id=f"held-{kind}-{way}"), [sales])
    result = await agent.run(QUESTION, user="u")
    assert result.answer == "done", result.error

    asked, answered = model.seen[-2], model.seen[-1]
    call = next(m for m in reversed(answered) if isinstance(m, AIMessage)).tool_calls[0]
    path = call["args"]["file_path"]
    assert path.startswith(results.PREFIX + "/"), path  # a held result's path was named
    # the request the query was made from offered it, and named the path's table
    assert query.QUERY_RESULT in model.requests[-2]["tools"]
    said = next(m.text for m in asked if isinstance(m, ToolMessage) and path in m.text)
    assert _TABLE.search(said), said[-700:]
    if way in CELL:
        assert way in said  # the window no longer shows it whole: "cleared", "cut to fit"
    # and the query over it is exact — the same figure on every backend
    assert answered[-1].text == expected(way)


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
