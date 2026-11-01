"""The data and the questions of the query engine benchmark, and Python's own answer to each.

The data is the shape of the ``query_result`` tests (#66): the SKU ledger of
``tests/integration/test_formulas.py`` (one table, as CSV or as JSON) and the three tools' results
of ``tests/integration/test_large_data.py`` (inventory as CSV, sales as a JSON array, lead times
as a JSON object holding the array, as a paged tool's pages are joined) — at any row count.

Each question is the SQL the harness's tool documents (SQLite, with its statistics aggregates),
and, where an engine's dialect differs, that engine's own SQL for the same question. The answer
is computed in Python over every row: integers and text exactly, floating values with
``math.fsum`` and ``statistics``."""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Final

CATS: Final = ("garden", "kitchen", "toys", "tools")
TODAY: Final = date(2026, 10, 30)
LEDGER: Final[tuple[str, ...]] = (
    "sku",
    "category",
    "stock",
    "revenue",
    "cost",
    "prev_revenue",
    "last_order",
)


def ledger_row(n: int) -> tuple[Any, ...]:
    """Row ``n`` of the ledger, as ``test_formulas.row`` makes it."""
    revenue = 1_000 + (n * 7919) % 90_000
    return (
        f"S{n:05d}",
        CATS[n % 4],
        (n * 31) % 400,
        revenue,
        revenue * (40 + n % 37) // 100,
        900 + (n * 104_729) % 90_000,
        (TODAY - timedelta(days=(n * 13) % 365)).isoformat(),
    )


def stock(n: int) -> int:
    return (n * 31) % 400


def sold(n: int) -> int:
    return (n * 17) % 300


def lead(n: int) -> int:
    return 3 + (n * 7) % 40


def write(directory: Path, rows: int) -> dict[str, Path]:
    """The tables' texts for ``rows`` rows, written under ``directory`` (streamed: a 2M-row
    JSON is never held whole): ``ledger.csv``, ``ledger.json``, ``inventory.csv``,
    ``sales.json``, ``lead_times.json``."""
    directory.mkdir(parents=True, exist_ok=True)
    paths = {
        name: directory / name
        for name in ("ledger.csv", "ledger.json", "inventory.csv", "sales.json", "lead_times.json")
    }
    with paths["ledger.csv"].open("w") as out:
        out.write(",".join(LEDGER))
        for n in range(rows):
            out.write("\n" + ",".join(map(str, ledger_row(n))))
    ledger = (dict(zip(LEDGER, ledger_row(n), strict=True)) for n in range(rows))
    _json_array(paths["ledger.json"], ledger)
    with paths["inventory.csv"].open("w") as out:
        out.write("sku,category,stock")
        for n in range(rows):
            out.write(f"\nS{n:05d},{CATS[n % 4]},{stock(n)}")
    _json_array(
        paths["sales.json"], ({"sku_id": f"S{n:05d}", "units_30d": sold(n)} for n in range(rows))
    )
    _json_array(
        paths["lead_times.json"],
        ({"sku": f"S{n:05d}", "lead_days": lead(n)} for n in range(rows)),
        wrap="items",
    )
    return paths


def _json_array(path: Path, items: Iterator[Any], wrap: str | None = None) -> None:
    with path.open("w") as out:
        out.write(f'{{"{wrap}": [' if wrap else "[")
        for n, item in enumerate(items):
            out.write(("," if n else "") + json.dumps(item))
        out.write('], "next_cursor": null}' if wrap else "]")


# ------------------------------------------------------------------------------- the questions


@dataclass(frozen=True)
class Question:
    """One question: its kind, the harness's SQL, other engines' SQL where it differs, and
    Python's answer (rows) from the columns."""

    name: str
    kind: str
    sql: str
    truth: Callable[[Columns], list[list[Any]]]
    dialects: dict[str, str] = field(default_factory=dict)


class Columns:
    """The ledger and the join tables as Python lists, for the answers."""

    def __init__(self, rows: int) -> None:
        cols = list(zip(*(ledger_row(n) for n in range(rows)), strict=True))
        self.sku, self.category, self.stock, self.revenue, self.cost, self.prev, self.last = cols
        self.n = rows


def _by_cat(c: Columns, fn: Callable[[list[int]], Any]) -> list[list[Any]]:
    """``fn`` of each category's row indices, as rows ordered by category."""
    groups: dict[str, list[int]] = {k: [] for k in CATS}
    for i, k in enumerate(c.category):
        groups[k].append(i)
    return [[k, fn(groups[k])] for k in sorted(groups)]


def _margin(c: Columns, i: int) -> float:
    return (c.revenue[i] - c.cost[i]) / c.revenue[i]


def _percentile(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    at = p * (len(xs) - 1)
    low = math.floor(at)
    high = min(low + 1, len(xs) - 1)
    return xs[low] + (xs[high] - xs[low]) * (at - low)


def _top_toys(c: Columns) -> list[list[Any]]:
    hits = [i for i in range(c.n) if c.stock[i] < 50 and c.category[i] == "toys"]
    hits.sort(key=lambda i: (-c.revenue[i], c.sku[i]))
    return [[c.sku[i], c.revenue[i]] for i in hits[:10]]


def _sums(c: Columns) -> list[list[Any]]:
    out = _by_cat(
        c, lambda ix: (len(ix), sum(c.revenue[i] for i in ix), sum(c.cost[i] for i in ix))
    )
    return [[k, *v] for k, v in out]


def _shortfalls(c: Columns) -> dict[int, int]:
    return {
        n: sold(n) * lead(n) - stock(n) * 30
        for n in range(c.n)
        if stock(n) * 30 < sold(n) * lead(n)
    }


def _join_top(c: Columns) -> list[list[Any]]:
    top = sorted(_shortfalls(c).items(), key=lambda kv: (-kv[1], c.sku[kv[0]]))[:5]
    return [[c.sku[n], v] for n, v in top]


def _join_groups(c: Columns) -> list[list[Any]]:
    found = _shortfalls(c)
    return [
        [
            k,
            sum(c.category[n] == k for n in found),
            sum(v for n, v in found.items() if c.category[n] == k),
        ]
        for k in sorted(CATS)
    ]


def _weighted(c: Columns) -> list[list[Any]]:
    order = sorted(
        range(c.n),  # the SQL's own order of operations, so near-ties order alike
        key=lambda i: (
            -(0.6 * (c.revenue[i] - c.cost[i]) * 1.0 / c.revenue[i] + 0.4 * c.stock[i] * 1.0 / 400),
            c.sku[i],
        ),
    )
    return [[c.sku[i]] for i in order[:5]]


def _moving(c: Columns) -> list[list[Any]]:
    order = sorted(range(c.n), key=lambda i: c.sku[i])
    at = next(k for k, i in enumerate(order) if c.sku[i] == "S00012")
    return [[math.fsum(c.revenue[i] for i in order[max(0, at - 2) : at + 1]) / min(3, at + 1)]]


def _ranked(c: Columns) -> list[list[Any]]:
    out = []
    for k, ix in _by_cat(c, lambda ix: sorted(ix, key=lambda j: (-c.revenue[j], c.sku[j]))[:3]):
        out += [[k, c.sku[i], c.revenue[i], r] for r, i in enumerate(ix, start=1)]
    return out


def _growth(c: Columns) -> list[list[Any]]:
    return [[sum((c.revenue[i] - c.prev[i]) * 100.0 / c.prev[i] > 10 for i in range(c.n))]]


def _old(c: Columns) -> list[list[Any]]:
    return [[sum((TODAY - date.fromisoformat(d)).days > 90 for d in c.last)]]


def _months(c: Columns) -> list[list[Any]]:
    found: dict[str, int] = {}
    for d in c.last:
        found[d[:7]] = found.get(d[:7], 0) + 1
    return [[m, found[m]] for m in sorted(found)]


def _floats(xs: tuple[int, ...]) -> list[float]:
    return [float(x) for x in xs]


def _regression(c: Columns) -> list[list[Any]]:
    fit = statistics.linear_regression(_floats(c.stock), _floats(c.revenue))
    return [[fit.slope, fit.intercept]]


QUESTIONS: Final[tuple[Question, ...]] = (
    Question(
        "top_n",
        "filter + sort + top-N",
        "SELECT sku, revenue FROM ledger WHERE stock < 50 AND category = 'toys' "
        "ORDER BY revenue DESC, sku LIMIT 10",
        _top_toys,
    ),
    Question(
        "group_sums",
        "group-by sums",
        "SELECT category, count(*) AS n, sum(revenue) AS revenue, sum(cost) AS cost FROM ledger "
        "GROUP BY category ORDER BY category",
        _sums,
    ),
    Question(
        "join_top",
        "3-table join",
        "SELECT i.sku, s.units_30d * l.lead_days - i.stock * 30 AS shortfall30 FROM inventory i "
        "JOIN sales s ON s.sku_id = i.sku JOIN lead_times l ON l.sku = i.sku "
        "WHERE i.stock * 30 < s.units_30d * l.lead_days ORDER BY shortfall30 DESC, i.sku LIMIT 5",
        _join_top,
    ),
    Question(
        "join_groups",
        "3-table join",
        "SELECT i.category, count(*) AS at_risk, sum(s.units_30d * l.lead_days - i.stock * 30) "
        "AS shortfall30 FROM inventory i JOIN sales s ON s.sku_id = i.sku JOIN lead_times l ON "
        "l.sku = i.sku WHERE i.stock * 30 < s.units_30d * l.lead_days GROUP BY i.category "
        "ORDER BY i.category",
        _join_groups,
    ),
    Question(
        "margin",
        "formula (margin)",
        "SELECT category, avg((revenue - cost) * 1.0 / revenue) FROM ledger GROUP BY category "
        "ORDER BY category",
        lambda c: _by_cat(c, lambda ix: math.fsum(_margin(c, i) for i in ix) / len(ix)),
    ),
    Question(
        "growth",
        "formula (growth %)",
        "SELECT count(*) FROM ledger WHERE (revenue - prev_revenue) * 100.0 / prev_revenue > 10",
        _growth,
    ),
    Question(
        "weighted_top",
        "formula (weighted score)",
        "SELECT sku FROM ledger ORDER BY -(0.6 * (revenue - cost) * 1.0 / revenue + 0.4 * stock "
        "* 1.0 / 400), sku LIMIT 5",
        _weighted,
    ),
    Question(
        "what_if",
        "formula (what-if sum)",
        "SELECT sum(revenue * CASE WHEN category = 'toys' THEN 1.05 ELSE 1 END) FROM ledger",
        lambda c: [
            [
                math.fsum(
                    r * (1.05 if k == "toys" else 1)
                    for r, k in zip(c.revenue, c.category, strict=True)
                )
            ]
        ],
    ),
    Question(
        "moving_avg",
        "window (moving average)",
        "SELECT ma FROM (SELECT sku, avg(revenue) OVER (ORDER BY sku ROWS BETWEEN 2 PRECEDING "
        "AND CURRENT ROW) AS ma FROM ledger) WHERE sku = 'S00012'",
        _moving,
    ),
    Question(
        "rank",
        "window (rank)",
        "SELECT category, sku, revenue, r FROM (SELECT category, sku, revenue, rank() OVER "
        "(PARTITION BY category ORDER BY revenue DESC, sku) AS r FROM ledger) WHERE r <= 3 "
        "ORDER BY category, r",
        _ranked,
    ),
    Question(
        "stddev",
        "stats (stddev)",
        "SELECT stddev(revenue) FROM ledger",
        lambda c: [[statistics.stdev(_floats(c.revenue))]],
    ),
    Question(
        "median",
        "stats (median)",
        "SELECT category, median(cost) FROM ledger GROUP BY category ORDER BY category",
        lambda c: _by_cat(c, lambda ix: statistics.median(c.cost[i] for i in ix)),
    ),
    Question(
        "percentile",
        "stats (percentile)",
        "SELECT percentile_cont(revenue, 0.9) FROM ledger",
        lambda c: [[_percentile(_floats(c.revenue), 0.9)]],
        {
            "duckdb": "SELECT quantile_cont(revenue, 0.9) FROM ledger",
            "polars": "SELECT quantile_cont(revenue, 0.9) FROM ledger",
        },
    ),
    Question(
        "corr",
        "stats (correlation)",
        "SELECT corr(revenue, stock) FROM ledger",
        lambda c: [[statistics.correlation(_floats(c.stock), _floats(c.revenue))]],
    ),
    Question(
        "regression",
        "stats (regression)",
        "SELECT regr_slope(revenue, stock), regr_intercept(revenue, stock) FROM ledger",
        _regression,
        {
            "polars": "SELECT b, ay - b * ax FROM (SELECT corr(revenue, stock) * stddev(revenue) "
            "/ stddev(stock) AS b, avg(revenue) AS ay, avg(stock) AS ax FROM ledger)",
        },
    ),
    Question(
        "r2",
        "stats (regression)",
        "SELECT regr_r2(revenue, stock) FROM ledger",
        lambda c: [[statistics.correlation(_floats(c.stock), _floats(c.revenue)) ** 2]],
        {"polars": "SELECT power(corr(revenue, stock), 2) FROM ledger"},
    ),
    Question(
        "older_90d",
        "date math (age)",
        "SELECT count(*) FROM ledger WHERE julianday('2026-10-30') - julianday(last_order) > 90",
        _old,
        {
            "duckdb": "SELECT count(*) FROM ledger WHERE DATE '2026-10-30' - CAST(last_order AS "
            "DATE) > 90",
            "polars": "SELECT count(*) FROM ledger WHERE DATE '2026-10-30' - CAST(last_order AS "
            "DATE) > INTERVAL '90 days'",
        },
    ),
    Question(
        "by_month",
        "date math (by month)",
        "SELECT strftime('%Y-%m', last_order) AS m, count(*) FROM ledger GROUP BY m ORDER BY m",
        _months,
        {
            "duckdb": "SELECT strftime(CAST(last_order AS DATE), '%Y-%m') AS m, count(*) FROM "
            "ledger GROUP BY m ORDER BY m",
            "polars": "SELECT strftime(CAST(last_order AS DATE), '%Y-%m') AS m, count(*) FROM "
            "ledger GROUP BY m ORDER BY m",
        },
    ),
)


def truths(rows: int) -> dict[str, list[list[Any]]]:
    """Python's answer to every question at ``rows`` rows."""
    c = Columns(rows)
    return {q.name: q.truth(c) for q in QUESTIONS}


def same(got: list[list[Any]], want: list[list[Any]], rel: float = 1e-12) -> bool:
    """Whether an engine's rows equal Python's: text and integers exactly, a floating value
    to ``rel`` (12 significant digits, as ``test_formulas``)."""
    if len(got) != len(want) or any(len(a) != len(b) for a, b in zip(got, want, strict=True)):
        return False
    for row, expected in zip(got, want, strict=True):
        for a, b in zip(row, expected, strict=True):
            if isinstance(b, float) or isinstance(a, float):
                if not isinstance(a, int | float) or not math.isclose(a, b, rel_tol=rel):
                    return False
            elif a != b:
                return False
    return True
