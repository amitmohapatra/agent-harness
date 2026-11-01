"""Large data in plain ``ReAct`` (no backend: LangChain's ``create_agent`` with the harness's
middleware) — the SKU scenario: a tool answers 10,000 SKUs, and every row stays reachable.

* a result over ``tools.results.MAX_CHARS`` is kept in the run and previewed (the convention
  OpenAI Agents, Claude and MCP already read: ``tools.results``), ``read_file`` pages it to its
  last row;
* a result cut to fit a small window says where the rest is (``read_file`` from the line the
  cut starts on), so no row is silently gone.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Final

import pytest
from langchain_core.messages import convert_to_messages

from tests.support.adapters import BUILDERS
from tests.support.arm import arm
from tests.support.models import Script, ScriptedChat
from tests.support.planned import Call
from tests.support.sandbox import FakeSandboxes
from trellis import Harness, ReAct, sandbox, tool
from trellis.contracts import RunEventType, RunStatus
from trellis.harness.automatic import JournalBackend
from trellis.harness.journal import content_key
from trellis.harness.tools import query, results

#: 10,000 SKUs: id, units in stock, revenue — a CSV of about 250,000 characters
SKUS: Final = [(f"SKU-{n:05d}", (n * 37) % 500, (n * 7919) % 100_000) for n in range(1, 10_001)]
CSV: Final = "sku,stock,revenue\n" + "\n".join(f"{s},{u},{r}" for s, u, r in SKUS)
KEPT: Final = f"/large_tool_results/{content_key('result', CSV)}"
#: what a kept result's stub and a page say to read next
SAVED: Final = re.compile(r"saved in the filesystem at this path: (\S+)")
NEXT: Final = re.compile(r'read_file\(file_path="([^"]+)", offset=(\d+)\)')


@tool(side_effects="read")
def inventory() -> str:
    """Every SKU with its units in stock and its revenue, as CSV."""
    return CSV


def names(request: dict[str, Any]) -> list[str]:
    return [t["function"]["name"] for t in request.get("tools", [])]


def last(request: dict[str, Any]) -> str:
    return str(request["messages"][-1].get("content"))


class Reader(Script):
    """A model that calls ``first``, then follows every pointer a result gives — the stub's
    path, a page's next offset, a cut's offset — until none is left, then answers."""

    def __init__(self, first: tuple[str, dict[str, Any]]) -> None:
        super().__init__([first])
        self.pages: list[str] = []

    def next(self, body: dict[str, Any]) -> Any:
        if self.turns:
            return super().next(body)
        told = last(body)
        if body["messages"][-1].get("role") == "tool" and len(self.requests) > 2:
            self.pages.append(told)
        follow = NEXT.findall(told)
        if follow:
            path, offset = follow[-1]
            return ("read_file", {"file_path": path, "offset": int(offset), "limit": 1_000})
        saved = SAVED.search(told)
        if saved:
            return ("read_file", {"file_path": saved.group(1), "limit": 1_000})
        return "done"


def rows_read(pages: list[str]) -> set[str]:
    """The SKU ids the pages showed (each line numbered, as ``read_file`` numbers them)."""
    return set(re.findall(r"^\s*\d+\s+(SKU-\d{5}),", "\n".join(pages), re.MULTILINE))


async def test_a_10000_row_result_is_kept_and_read_to_its_last_row(harness: Harness) -> None:
    model = ScriptedChat(
        [("inventory", {}), ("read_file", {"file_path": KEPT, "offset": 9_995}), "done"]
    )
    agent = arm(harness.wrap(ReAct(system_prompt="s", model=model), id="skus"), [inventory])
    result = await agent.run("how many SKUs have stock < 10?", user="u")
    assert result.status is RunStatus.SUCCESS, result.error
    preview = last(model.requests[1])
    assert f"saved in the filesystem at this path: {KEPT}" in preview, preview[:300]
    assert len(preview) < 25_000  # a preview, not 250,000 characters
    assert "read_file" not in names(model.requests[0]) and "read_file" in names(model.requests[1])
    page = last(model.requests[2])
    assert "SKU-10000,0,90000" in page and "SKU-09995" in page, page[:300]


async def test_a_kept_result_says_it_is_the_runs_own_copy_not_a_file_for_the_user(
    harness: Harness,
) -> None:
    """The stub names a path in the run's own store: the model is told the user cannot open
    it (break-and-fix round 1: asked for an Excel export, a model told the user the rows were
    "saved as CSV at /large_tool_results/..., open it in Excel")."""
    model = ScriptedChat([("inventory", {}), "done"])
    agent = arm(harness.wrap(ReAct(system_prompt="s", model=model), id="own"), [inventory])
    await agent.run("export every SKU to a file I can download", user="u")
    preview = last(model.requests[1])
    assert results.PRIVATE in preview, preview[-600:]


async def test_every_row_of_a_kept_result_is_reachable_page_by_page(harness: Harness) -> None:
    script = Reader(("inventory", {}))
    model = ScriptedChat(script=script)
    target = ReAct(system_prompt="s", model=model, max_steps=40)
    result = await arm(harness.wrap(target, id="skus-all"), [inventory]).run("x", user="u")
    assert result.status is RunStatus.SUCCESS, result.error
    assert rows_read(script.pages) == {s for s, _, _ in SKUS}


async def test_a_result_cut_to_fit_the_window_says_where_the_rest_is(harness: Harness) -> None:
    rows = "\n".join(f"{s},{u},{r}" for s, u, r in SKUS[:2_000])  # ~38,000: under MAX_CHARS

    @tool(side_effects="read")
    def some_skus() -> str:
        """Two thousand SKUs, as CSV."""
        return rows

    script = Reader(("some_skus", {}))
    model = ScriptedChat(script=script)
    target = ReAct(system_prompt="s", model=model, context_window=8_192, max_steps=40)
    result = await arm(harness.wrap(target, id="skus-cut"), [some_skus]).run("x", user="u")
    assert result.status is RunStatus.SUCCESS, result.error
    told = last(model.requests[1])
    assert "characters cut to fit the window" in told and NEXT.search(told), told[-2_000:]
    assert "read_file" in names(model.requests[1])
    # the head the cut showed, then every row after it, read back
    shown = set(re.findall(r"^(SKU-\d{5}),", told, re.MULTILINE))
    assert shown | rows_read(script.pages) == {s for s, _, _ in SKUS[:2_000]}


async def test_a_long_text_cut_to_fit_the_window_is_offered_read_file(harness: Harness) -> None:
    """Not a table (no delimiter: nothing for ``query_result``), so nothing is kept until the
    cut: the cut keeps it and offers ``read_file`` with it."""
    text = "\n".join(
        f"note {n}: {s} has {u} units on hand" for n, (s, u, _) in enumerate(SKUS[:1_200])
    )

    @tool(side_effects="read")
    def notes() -> str:
        """Twelve hundred stock notes, as text."""
        return text

    script = Reader(("notes", {}))
    model = ScriptedChat(script=script)
    target = ReAct(system_prompt="s", model=model, context_window=8_192, max_steps=40)
    result = await arm(harness.wrap(target, id="notes-cut"), [notes]).run("x", user="u")
    assert result.status is RunStatus.SUCCESS, result.error
    told = last(model.requests[1])
    assert "characters cut to fit the window" in told and NEXT.search(told), told[-2_000:]
    assert "read_file" in names(model.requests[1])
    assert "query_result" not in names(model.requests[1])


# --------------------------------------------------------------------------- aggregation (D2)
@pytest.mark.parametrize(
    "framework", ["react", "middleware", "openai_agents", "claude_agent_sdk", "kimi"]
)
async def test_a_kept_result_is_a_file_in_the_runs_sandbox_for_code_to_compute_over(
    harness: Harness, framework: str, tmp_path: Path
) -> None:
    """A run with a sandbox (``sandbox().tools()``): the stub says the kept result is a file
    there to run code over (a count, a sum, a top-N over every row), and it is — whichever
    adapter runs the agent."""
    provider = FakeSandboxes()
    plan: list[Call] = [("inventory", {}), ("sandbox_exec", {"command": f"cat {KEPT}"})]
    tools = [inventory, sandbox(provider).tools()]
    target, own = await BUILDERS[framework](harness, tools, tmp_path, plan)
    events = [
        e async for e in arm(harness.wrap(target, id=f"agg-{framework}"), own).stream("x", user="u")
    ]
    assert events[-1].outcome is not None and events[-1].outcome.value == "success", events[-1]
    ran = [e.data for e in events if e.type is RunEventType.TOOL_CALL_RESULT]
    assert [r["tool"] for r in ran] == ["inventory", "sandbox_exec"]
    assert provider.commands == [f"cat {KEPT}"]
    said = str(ran[1]["output"])  # the stream's preview of what the command printed
    assert "'stdout': 'sku,stock,revenue\\nSKU-00001,37,7919\\n" in said, said[:300]


async def test_the_stub_points_a_run_with_a_sandbox_to_code(harness: Harness) -> None:
    model = ScriptedChat([("inventory", {}), "done"])
    target = ReAct(system_prompt="s", model=model)
    agent = arm(harness.wrap(target, id="agg-stub"), [inventory, sandbox(FakeSandboxes()).tools()])
    assert (await agent.run("x", user="u")).status is RunStatus.SUCCESS
    stub = last(model.requests[1])
    assert results.CODE.format(path=KEPT, tool="sandbox_exec") in stub, stub[:1_000]


# --------------------------------------------------------------------------- no sandbox: SQL
#: The SKU questions, the SQL a model asks them with, and the answer computed in Python over
#: every row (the ground truth the query must equal exactly).
QUESTIONS: Final = [
    (
        "how many SKUs have stock < 10",
        "SELECT count(*) AS n FROM result WHERE stock < 10",
        f"n\n{sum(u < 10 for _, u, _ in SKUS)}",
    ),
    (
        "top 5 by revenue",
        "SELECT sku FROM result ORDER BY revenue DESC, sku LIMIT 5",
        "sku\n" + "\n".join(s for s, _, _ in sorted(SKUS, key=lambda r: (-r[2], r[0]))[:5]),
    ),
    (
        "total units in stock",
        "SELECT sum(stock) AS total FROM result",
        f"total\n{sum(u for _, u, _ in SKUS)}",
    ),
    (
        "revenue of SKUs out of stock, grouped by whether revenue is over 50,000",
        "SELECT revenue > 50000 AS big, count(*) AS n, sum(revenue) AS r FROM result "
        "WHERE stock = 0 GROUP BY big ORDER BY big",
        "big,n,r\n"
        + "\n".join(
            f"{int(big)},{sum(1 for _, u, r in SKUS if u == 0 and (r > 50_000) == big)},"
            f"{sum(r for _, u, r in SKUS if u == 0 and (r > 50_000) == big)}"
            for big in (False, True)
            if any(u == 0 and (r > 50_000) == big for _, u, r in SKUS)
        ),
    ),
]


@pytest.mark.parametrize("question", range(len(QUESTIONS)))
@pytest.mark.parametrize(
    "framework", ["react", "middleware", "openai_agents", "claude_agent_sdk", "kimi"]
)
async def test_sku_questions_are_answered_exactly_over_every_row_with_no_sandbox(
    harness: Harness, framework: str, question: int, tmp_path: Path
) -> None:
    """No sandbox, no Code Mode: the kept result is a table ``query_result`` runs SQL over
    (``tools.query``, Python's own SQLite) — each answer equals the one computed in Python."""
    _, sql, answer = QUESTIONS[question]
    plan: list[Call] = [("inventory", {}), ("query_result", {"file_path": KEPT, "sql": sql})]
    target, own = await BUILDERS[framework](harness, [inventory], tmp_path, plan)
    result = await arm(harness.wrap(target, id=f"sql-{framework}"), own).run("q", user="u")
    assert result.status is RunStatus.SUCCESS, result.error
    assert result.answer == f"Done. {answer}"  # what the model read: the query's rows


async def test_the_stub_names_the_table_and_its_columns_before_the_preview(
    harness: Harness,
) -> None:
    model = ScriptedChat([("inventory", {}), "done"])
    agent = arm(harness.wrap(ReAct(system_prompt="s", model=model), id="sql-stub"), [inventory])
    assert (await agent.run("x", user="u")).status is RunStatus.SUCCESS
    stub = last(model.requests[1])
    # its table named by its tool: one query joins it with the run's other saved results
    said = results.QUERY.format(
        table="inventory", columns="sku, stock, revenue", path=KEPT, private=results.PRIVATE
    )
    assert said in stub and stub.index(said) < stub.index("Here is a preview")
    assert {"read_file", "query_result"} <= set(names(model.requests[1]))
    assert "query_result" not in names(model.requests[0])


async def test_a_result_deep_agents_saved_is_queried_too(harness: Harness) -> None:
    """``ReAct`` given a backend: Deep Agents saves the result (its own path, in the graph's
    state), and ``query_result`` runs over it the same way."""

    class Querying(Script):
        def next(self, body: dict[str, Any]) -> Any:
            if not self.requests[:-1]:
                return ("inventory", {})
            saved = SAVED.search(last(body))
            if saved and len(self.requests) == 2:
                sql = QUESTIONS[0][1]
                return ("query_result", {"file_path": saved.group(1), "sql": sql})
            return "done"

    model = ScriptedChat(script=Querying([]))
    target = ReAct(system_prompt="s", model=model, backend=JournalBackend())
    result = await arm(harness.wrap(target, id="sql-deep"), [inventory]).run("x", user="u")
    assert result.status is RunStatus.SUCCESS, result.error
    assert "query_result" in names(model.requests[1])
    assert last(model.requests[2]) == QUESTIONS[0][2]


async def test_a_tool_that_filters_on_its_own_side_answers_small_and_exact(
    harness: Harness,
) -> None:
    """The first choice: a tool whose arguments filter and rank — its schema, with each
    argument's description, is what the model is offered, and the answer is never paged."""

    @tool(side_effects="read")
    def find_skus(max_stock: int | None = None, top_by_revenue: int | None = None) -> str:
        """SKUs as CSV, filtered and ranked on the inventory's side.

        Args:
            max_stock: Only SKUs with fewer units in stock than this.
            top_by_revenue: Only this many SKUs, the highest revenue first.
        """
        rows = [r for r in SKUS if max_stock is None or r[1] < max_stock]
        if top_by_revenue is not None:
            rows = sorted(rows, key=lambda r: (-r[2], r[0]))[:top_by_revenue]
        return "\n".join(f"{s},{u},{r}" for s, u, r in rows)

    model = ScriptedChat([("find_skus", {"top_by_revenue": 5}), "done"])
    agent = arm(harness.wrap(ReAct(system_prompt="s", model=model), id="filtered"), [find_skus])
    assert (await agent.run("top 5 by revenue", user="u")).status is RunStatus.SUCCESS
    [offered] = model.requests[0]["tools"]
    described = offered["function"]["parameters"]["properties"]
    assert "highest revenue first" in described["top_by_revenue"]["description"]
    told = last(model.requests[1])
    top = sorted(SKUS, key=lambda r: (-r[2], r[0]))[:5]
    assert told == "\n".join(f"{s},{u},{r}" for s, u, r in top)  # whole: five rows, no stub
    assert "query_result" not in names(model.requests[1])  # nothing kept, nothing to query


async def test_the_harness_readers_answer_as_langchain_tools_too(harness: Harness) -> None:
    """``read_file`` and ``query_result`` as LangChain tools: what a tool node would run if one
    held them (the harness answers their calls itself)."""
    from trellis.harness import middleware as ours

    said = await ours.READERS["query_result"].ainvoke({"file_path": "/x", "sql": "SELECT 1"})
    assert said == "Error: File '/x' not found"  # outside a run, nothing is kept


# --------------------------------------------------------------------------- joins (50,000 rows)
#: 50,000 SKUs, their sales over 30 days (another tool), and their suppliers' lead times (a
#: third tool, which pages: 1,000 a page)
N: Final = 50_000
CATS: Final = ("garden", "kitchen", "toys", "tools")
STOCK: Final = {f"S{n:05d}": (n * 31) % 400 for n in range(N)}
SOLD: Final = {f"S{n:05d}": (n * 17) % 300 for n in range(N)}
LEAD: Final = {f"S{n:05d}": 3 + (n * 7) % 40 for n in range(N)}
CATEGORY: Final = {f"S{n:05d}": CATS[n % 4] for n in range(N)}
PAGE: Final = 1_000


@tool(side_effects="read")
def inventory50k() -> str:
    """Every SKU with its category and units in stock, as CSV."""
    return "sku,category,stock\n" + "\n".join(f"{s},{CATEGORY[s]},{u}" for s, u in STOCK.items())


@tool(side_effects="read")
def sales() -> str:
    """Units each SKU sold over the last 30 days, as JSON."""
    return json.dumps([{"sku_id": s, "units_30d": u} for s, u in SOLD.items()])


@tool(side_effects="read", paged=True)
def lead_times(cursor: str | None = None) -> dict[str, Any]:
    """Each SKU's supplier lead time in days, a page at a time.

    Args:
        cursor: Where the page starts (the last page's next_cursor).
    """
    start = int(cursor or 0)
    skus = list(LEAD)[start : start + PAGE]
    after = start + PAGE
    return {
        "items": [{"sku": s, "lead_days": LEAD[s]} for s in skus],
        "next_cursor": str(after) if after < N else None,
    }


#: Stock-out risk: a SKU sells out before more arrives (stock < 30-day sales / 30 x lead time):
#: the shortfall, in thirtieths of a unit (exact integers in SQL and in Python alike).
RISK_SQL: Final = (
    "SELECT i.sku, s.units_30d * l.lead_days - i.stock * 30 AS shortfall30 "
    "FROM inventory50k i JOIN sales s ON s.sku_id = i.sku JOIN lead_times l ON l.sku = i.sku "
    "WHERE i.stock * 30 < s.units_30d * l.lead_days ORDER BY shortfall30 DESC, i.sku LIMIT 5"
)
GROUP_SQL: Final = (
    "SELECT i.category, count(*) AS at_risk, sum(s.units_30d * l.lead_days - i.stock * 30) AS "
    "shortfall30 FROM inventory50k i JOIN sales s ON s.sku_id = i.sku JOIN lead_times l ON "
    "l.sku = i.sku WHERE i.stock * 30 < s.units_30d * l.lead_days GROUP BY i.category "
    "ORDER BY i.category"
)


def shortfalls() -> dict[str, int]:
    """Python's own answer: each SKU at risk, and its shortfall in thirtieths of a unit."""
    return {
        s: SOLD[s] * LEAD[s] - STOCK[s] * 30 for s in STOCK if STOCK[s] * 30 < SOLD[s] * LEAD[s]
    }


def expected_risks() -> str:
    top = sorted(shortfalls().items(), key=lambda kv: (-kv[1], kv[0]))[:5]
    return "sku,shortfall30\n" + "\n".join(f"{s},{v}" for s, v in top)


def expected_groups() -> str:
    found = shortfalls()
    rows = [
        f"{c},{sum(CATEGORY[s] == c for s in found)},"
        f"{sum(v for s, v in found.items() if CATEGORY[s] == c)}"
        for c in sorted(CATS)
    ]
    return "category,at_risk,shortfall30\n" + "\n".join(rows)


def kept_path(text: str) -> str:
    return f"/large_tool_results/{content_key('result', text)}"


@pytest.mark.parametrize("sql", ["risks", "groups"])
@pytest.mark.parametrize(
    "framework", ["react", "middleware", "openai_agents", "claude_agent_sdk", "kimi"]
)
async def test_three_tools_results_are_joined_in_one_query_exactly(
    harness: Harness, framework: str, sql: str, tmp_path: Path
) -> None:
    """50,000 SKUs from one tool, their sales from another, their lead times from a third that
    pages (fetched to its last page): every saved result is a table named by its tool, and one
    read-only query joins them — the answer equals Python's, the model reads only that."""
    query, answer = (
        (RISK_SQL, expected_risks()) if sql == "risks" else (GROUP_SQL, expected_groups())
    )
    path = kept_path(inventory50k.fn())
    plan: list[Call] = [("inventory50k", {}), ("sales", {}), ("lead_times", {})]
    plan.append(("query_result", {"file_path": path, "sql": query}))
    target, own = await BUILDERS[framework](
        harness, [inventory50k, sales, lead_times], tmp_path, plan
    )
    result = await arm(harness.wrap(target, id=f"join-{framework}"), own).run("q", user="u")
    assert result.status is RunStatus.SUCCESS, result.error
    assert result.answer == f"Done. {answer}"


async def test_a_join_over_50000_rows_costs_the_model_a_few_thousand_tokens(
    harness: Harness,
) -> None:
    """What the model is sent in all — three previews and one small answer, never the rows —
    and how long the query takes (``tools.query.SECONDS`` at most)."""
    from langchain_core.messages.utils import count_tokens_approximately

    path = kept_path(inventory50k.fn())
    model = ScriptedChat(
        [
            [("inventory50k", {}), ("sales", {}), ("lead_times", {})],
            ("query_result", {"file_path": path, "sql": RISK_SQL}),
            "done",
        ]
    )
    agent = arm(
        harness.wrap(ReAct(system_prompt="s", model=model), id="join-cost"),
        [inventory50k, sales, lead_times],
    )
    began = time.monotonic()
    result = await agent.run("top 5 stock-out risks", user="u")
    seconds = time.monotonic() - began
    assert result.status is RunStatus.SUCCESS, result.error
    assert last(model.requests[2]) == expected_risks()
    sent = sum(
        count_tokens_approximately(convert_to_messages(r["messages"])) for r in model.requests
    )
    raw = sum(len(t) for t in (inventory50k.fn(), sales.fn())) // 4
    print(f"\njoin over 3 x 50,000 rows: {sent} tokens sent in all ({raw}+ raw), {seconds:.2f} s")
    assert sent < 20_000 and seconds < query.SECONDS * 2
    previews = [m for m in model.requests[1]["messages"] if m["role"] == "tool"]
    assert [m["content"].count("query_result(") for m in previews] == [1, 1, 1]
    tables = {t for m in previews for t in re.findall(r"the table (\w+)", m["content"])}
    assert tables == {"inventory50k", "sales", "lead_times"}  # each its tool's, one query


#: 300 sales lines: a result well under ``tools.results.MAX_CHARS``, read whole, that a model
#: would otherwise count and add up itself
LINES: Final = "date,sku,units\n" + "\n".join(
    f"2026-09-{1 + n % 30:02d},SKU-{n % 41:05d},{1 + n % 7}" for n in range(300)
)
LINES_KEPT: Final = f"/large_tool_results/{content_key('result', LINES)}"
DISTINCT: Final = "SELECT COUNT(DISTINCT sku) FROM result"


@tool(side_effects="read")
def sales_lines() -> str:
    """September's sales lines, as CSV."""
    return LINES


@tool(side_effects="read")
def few_lines() -> str:
    """Three sales lines, as CSV."""
    return "date,sku,units\n" + "\n".join(LINES.splitlines()[1:4])


async def test_a_table_read_whole_is_also_queryable_so_no_row_is_counted_by_hand(
    harness: Harness,
) -> None:
    """A result of many rows that fits is read whole and also kept as a table: its message
    says so, ``query_result`` is offered and counts over every row exactly."""
    model = ScriptedChat(
        [("sales_lines", {}), ("query_result", {"file_path": LINES_KEPT, "sql": DISTINCT}), "41"]
    )
    agent = arm(harness.wrap(ReAct(system_prompt="s", model=model), id="lines"), [sales_lines])
    result = await agent.run("how many distinct SKUs sold?", user="u")
    assert result.status is RunStatus.SUCCESS, result.error
    told = last(model.requests[1])
    assert told.startswith(LINES)  # every row, as it came
    said = results.QUERY.format(
        table="sales_lines", columns="date, sku, units", path=LINES_KEPT, private=results.PRIVATE
    )
    assert said in told and "300 rows" in told
    assert "query_result" in names(model.requests[1])
    assert last(model.requests[2]).splitlines()[1] == "41"


async def test_a_few_rows_are_read_as_they_are(harness: Harness) -> None:
    model = ScriptedChat([("few_lines", {}), "done"])
    agent = arm(harness.wrap(ReAct(system_prompt="s", model=model), id="few"), [few_lines])
    assert (await agent.run("x", user="u")).status is RunStatus.SUCCESS
    assert last(model.requests[1]) == "date,sku,units\n" + "\n".join(LINES.splitlines()[1:4])
    assert "query_result" not in names(model.requests[1])


@pytest.mark.parametrize(
    "framework", ["react", "middleware", "openai_agents", "claude_agent_sdk", "kimi"]
)
async def test_every_framework_counts_a_table_read_whole_exactly(
    harness: Harness, framework: str, tmp_path: Path
) -> None:
    plan: list[Call] = [
        ("sales_lines", {}),
        ("query_result", {"file_path": LINES_KEPT, "sql": DISTINCT}),
    ]
    target, own = await BUILDERS[framework](harness, [sales_lines], tmp_path, plan)
    result = await arm(harness.wrap(target, id=f"lines-{framework}"), own).run("q", user="u")
    assert result.status is RunStatus.SUCCESS, result.error
    assert result.answer.endswith("41")
