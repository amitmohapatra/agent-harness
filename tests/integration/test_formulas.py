"""Formulas and statistics over 50,000 SKUs with no sandbox, asked with read-only SQL
(``query_result``: ``tools.query``) — margins, a weighted score, growth, percentiles, standard
deviation, correlation, regression, date math, a what-if, a moving average — each answer equal
to Python's own over every row (floating results to 12 significant digits)."""

from __future__ import annotations

import csv
import io
import math
import statistics
from datetime import date, timedelta
from typing import Any, Final

import pytest

from tests.support.arm import arm
from tests.support.models import ScriptedChat
from trellis import Harness, ReAct, tool
from trellis.contracts import RunStatus
from trellis.harness.journal import content_key

N: Final = 50_000
CATS: Final = ("garden", "kitchen", "toys", "tools")
TODAY: Final = date(2026, 10, 30)


def row(n: int) -> dict[str, Any]:
    revenue = 1_000 + (n * 7919) % 90_000
    return {
        "sku": f"S{n:05d}",
        "category": CATS[n % 4],
        "stock": (n * 31) % 400,
        "revenue": revenue,
        "cost": revenue * (40 + n % 37) // 100,
        "prev_revenue": 900 + (n * 104_729) % 90_000,
        "last_order": (TODAY - timedelta(days=(n * 13) % 365)).isoformat(),
    }


ROWS: Final = [row(n) for n in range(N)]
HEAD: Final = list(ROWS[0])
TEXT: Final = ",".join(HEAD) + "\n" + "\n".join(",".join(str(r[c]) for c in HEAD) for r in ROWS)
PATH: Final = f"/large_tool_results/{content_key('result', TEXT)}"


@tool(side_effects="read")
def ledger() -> str:
    """Every SKU: category, stock, revenue and cost this year, last year's revenue, last order."""
    return TEXT


def margin(r: dict[str, Any]) -> float:
    return (r["revenue"] - r["cost"]) / r["revenue"]


def percentile(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    at = p * (len(xs) - 1)
    low = math.floor(at)
    high = min(low + 1, len(xs) - 1)
    return xs[low] + (xs[high] - xs[low]) * (at - low)


def by_category(fn: Any) -> list[list[Any]]:
    return [[c, fn([r for r in ROWS if r["category"] == c])] for c in sorted(CATS)]


def top5_score() -> list[list[Any]]:
    scored = sorted(ROWS, key=lambda r: (-(0.6 * margin(r) + 0.4 * r["stock"] / 400), r["sku"]))
    return [[r["sku"]] for r in scored[:5]]


revenues = [float(r["revenue"]) for r in ROWS]
stocks = [float(r["stock"]) for r in ROWS]
cut = (TODAY - timedelta(days=90)).isoformat()

#: name: (Python's answer as rows, the SQL)
SUITE: Final[dict[str, tuple[list[list[Any]], str]]] = {
    "average margin by category": (
        by_category(lambda rs: statistics.fmean(margin(r) for r in rs)),
        "SELECT category, avg((revenue - cost) * 1.0 / revenue) FROM result GROUP BY category "
        "ORDER BY category",
    ),
    "top 5 by a weighted score": (
        top5_score(),
        "SELECT sku FROM result ORDER BY -(0.6 * (revenue - cost) * 1.0 / revenue + 0.4 * stock "
        "* 1.0 / 400), sku LIMIT 5",
    ),
    "growth over 10%": (
        [[sum((r["revenue"] - r["prev_revenue"]) / r["prev_revenue"] * 100 > 10 for r in ROWS)]],
        "SELECT count(*) FROM result WHERE (revenue - prev_revenue) * 100.0 / prev_revenue > 10",
    ),
    "90th percentile of revenue": (
        [[percentile(revenues, 0.9)]],
        "SELECT percentile_cont(revenue, 0.9) FROM result",
    ),
    "standard deviation of revenue": (
        [[statistics.stdev(revenues)]],
        "SELECT stddev(revenue) FROM result",
    ),
    "correlation of revenue and stock": (
        [[statistics.correlation(stocks, revenues)]],
        "SELECT corr(revenue, stock) FROM result",
    ),
    "regression of revenue on stock": (
        [list(statistics.linear_regression(stocks, revenues))],
        "SELECT regr_slope(revenue, stock), regr_intercept(revenue, stock) FROM result",
    ),
    "orders older than 90 days": (
        [[sum(r["last_order"] < cut for r in ROWS)]],
        "SELECT count(*) FROM result WHERE julianday('2026-10-30') - julianday(last_order) > 90",
    ),
    "what if toys sold for 5% more": (
        [[sum(r["revenue"] * (1.05 if r["category"] == "toys" else 1) for r in ROWS)]],
        "SELECT sum(revenue * CASE WHEN category = 'toys' THEN 1.05 ELSE 1 END) FROM result",
    ),
    "median cost by category": (
        by_category(lambda rs: statistics.median(r["cost"] for r in rs)),
        "SELECT category, median(cost) FROM result GROUP BY category ORDER BY category",
    ),
    "a 3-SKU moving average of revenue": (
        [[statistics.fmean(revenues[10:13])]],
        "SELECT ma FROM (SELECT sku, avg(revenue) OVER (ORDER BY sku ROWS BETWEEN 2 PRECEDING "
        "AND CURRENT ROW) AS ma FROM result) WHERE sku = 'S00012'",
    ),
}


def parsed(answer: str) -> list[list[Any]]:
    """The rows an answer (``Done. <csv>``) holds, numbers as numbers."""
    rows = list(csv.reader(io.StringIO(answer.removeprefix("Done. "))))[1:]
    return [[_number(v) for v in r] for r in rows]


def _number(value: str) -> Any:
    try:
        return float(value)
    except ValueError:
        return value


def same(got: list[list[Any]], want: list[list[Any]]) -> bool:
    if len(got) != len(want) or any(len(a) != len(b) for a, b in zip(got, want, strict=True)):
        return False
    return all(
        math.isclose(a, float(b), rel_tol=1e-12) if isinstance(a, float) else a == b
        for ga, wa in zip(got, want, strict=True)
        for a, b in zip(ga, wa, strict=True)
    )


@pytest.mark.parametrize("question", list(SUITE))
async def test_a_formula_or_a_statistic_over_50000_rows_equals_pythons(
    harness: Harness, question: str
) -> None:
    want, sql = SUITE[question]
    args = {"file_path": PATH, "sql": sql}
    model = ScriptedChat([("ledger", {}), ("query_result", args), "done"])
    agent = arm(harness.wrap(ReAct(system_prompt="s", model=model), id="formulas"), [ledger])
    result = await agent.run(question, user="u")
    assert result.status is RunStatus.SUCCESS, result.error
    got = model.requests[2]["messages"][-1]["content"]
    assert same(parsed(got), want), (got, want)
