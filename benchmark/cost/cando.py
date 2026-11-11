"""The ``complex`` suite's exact-answer cases for what a deployment *can* do (category
``cando``): the same data question must give the same figure whatever the agent was given — a
sandbox or none, a small window or the model's own — and a file asked for must be the user's,
with the right content.

Each case asks for a figure every variant can compute (the query over the kept result), so it
runs without a sandbox too; what needs code (a chart, a script) is checked only where code runs
(``Probe.coding``), and a case that is nothing without code (:attr:`Case.coding`: a clustering
from fixed starting points) applies only there. No case sets its window: ``--window`` runs each
at any size, on any variant (``--variant sandbox``, ``main``, ...), which is how the two are
compared (:mod:`benchmark.cost.parity`).

The scripted model replays the gold calls: the fetch, the query, and — without a sandbox — the
query saving its result as a file (``save_as``). It cannot write the code a sandbox runs (a fake
sandbox has no interpreter), so the files only code makes are proven on a real model.
"""

from __future__ import annotations

import statistics
from typing import Any, Final

from benchmark.cost import retail as r

Q3: Final = {"start": "2026-07-01", "end": "2026-09-30"}
SEP: Final = {"start": "2026-09-01", "end": "2026-09-30"}
YTD: Final = {"start": "2026-01-01", "end": "2026-09-30"}
#: The clustering's starting centres (prices in GBP) — fixed, so the result is one.
CENTRES: Final = (20.0, 60.0, 120.0)


def kmeans(values: list[float], centres: tuple[float, ...]) -> tuple[list[int], list[float]]:
    """Lloyd's algorithm in one dimension from ``centres``, until no value changes cluster (a
    value as near to two centres goes to the lower): each cluster's size and final centre."""
    now, assigned = list(centres), [-1] * len(values)
    while True:
        nearest = [min(range(len(now)), key=lambda k: (abs(v - now[k]), k)) for v in values]
        if nearest == assigned:
            sizes = [nearest.count(k) for k in range(len(now))]
            return sizes, now
        assigned = nearest
        for k in range(len(now)):
            members = [v for v, a in zip(values, assigned, strict=True) if a == k]
            now[k] = statistics.fmean(members) if members else now[k]


def cases() -> list[Any]:
    from benchmark.cost import complex as cx

    w = r.world()
    out: list[Any] = []
    saved = "{saved}"  # the scripted model's: the path of the result just kept

    # 1. a workbook with the right rows, and the figure at its top
    top = r.truth_top(w, "sku", "revenue", 20, **SEP)
    summary = {"group_by": "sku", "metric": "revenue", **SEP}
    ranked = "SELECT key AS sku, value AS revenue FROM result ORDER BY value DESC, key LIMIT 20"
    out.append(
        cx.Case(
            "cando-01", "cando", 3,
            (
                cx.Turn(
                    "Export the 20 SKUs with the highest September 2026 revenue as an Excel "
                    "file I can download (columns: sku, revenue), and tell me which SKU is "
                    "first and what its revenue was.",
                    (
                        (("sales_summary", summary),),
                        (("query_result", {"file_path": saved, "sql": ranked, "save_as": "xlsx",
                                           "file_name": "top20_skus_sep2026"}),),
                    ),
                    cx.record({"sku": top[0][0], "revenue": top[0][1]}, tol={"revenue": 0.006}),
                    sent=(
                        cx.Probe("file", r"\.xlsx$", "the workbook is the user's, with the 20 SKUs",
                                 holds=tuple(sku for sku, _ in top), limit=21),
                    ),
                ),
            ),
            ("a file with the right rows", "runs with or without a sandbox"),
            keyed=True,
        )
    )  # fmt: skip

    # 2. a chart file with the right labels (where code runs), and its largest bar's figure
    by_category = r.truth_top(w, "category", "revenue", len(r.CATEGORIES), **Q3)
    out.append(
        cx.Case(
            "cando-02", "cando", 3,
            (
                cx.Turn(
                    "Draw a bar chart of Q3 2026 revenue by category and save it as an SVG file "
                    "for me. Which category is largest, and what was its revenue?",
                    ((("sales_summary", {"group_by": "category", "metric": "revenue", **Q3}),),),
                    cx.record({"category": by_category[0][0], "revenue": by_category[0][1]},
                              tol={"revenue": 0.006}),
                    sent=(
                        cx.Probe("file", r"\.svg$", "the chart is the user's, every bar labelled",
                                 holds=tuple(c for c, _ in by_category), coding=True),
                    ),
                ),
            ),
            ("a chart file with the right labels", "the figure runs without a sandbox"),
        )
    )  # fmt: skip

    # 3. a script, kept and run: what it prints is the exact figure
    west = [s.units for s in r.selected(w, region="west", **SEP)]
    spread = round(statistics.pstdev(west), 3)
    spread_sql = "SELECT round(stddev_pop(units), 3) AS sd FROM result"
    out.append(
        cx.Case(
            "cando-03", "cando", 3,
            (
                cx.Turn(
                    "Write a Python script that reads our September 2026 west sales lines and "
                    "prints the population standard deviation of units per line, rounded to 3 "
                    "decimals. Save the script as a file for me, run it, and tell me the figure.",
                    (
                        (("sales_rows", {"region": "west", **SEP}),),
                        (("query_result", {"file_path": saved, "sql": spread_sql}),),
                    ),
                    cx.Expect("number", spread, tol=0.0006, reference=f"{spread:.3f}"),
                    sent=(cx.Probe("file", r"\.py$", "the script is the user's", coding=True),),
                ),
            ),
            ("a script with an exact figure", "the figure runs without a sandbox"),
            keyed=True,
        )
    )  # fmt: skip

    # 4. a clustering from fixed starting points: one right answer, and only code gives it
    sizes, centres = kmeans([s.price for s in w.skus.values()], CENTRES)
    fields = {f"cluster_{n + 1}_skus": size for n, size in enumerate(sizes)}
    fields |= {f"centre_{n + 1}": round(c, 2) for n, c in enumerate(centres)}
    out.append(
        cx.Case(
            "cando-04", "cando", 3,
            (
                cx.Turn(
                    "Cluster the whole product catalogue by current price into 3 groups with "
                    "k-means, starting from the centres 20, 60 and 120 and repeating until no "
                    "SKU changes group (a SKU equally near two centres goes to the lower one). "
                    "How many SKUs end in each group, lowest centre first, and what are the "
                    "three final centres (2 decimals)?",
                    ((("sku_catalog", {}),),),
                    cx.record(fields, tol={k: 0.006 if k.startswith("centre") else 0
                                           for k in fields}),
                ),
            ),
            ("clustering with fixed starting points",),
            coding=True,
        )
    )  # fmt: skip

    # 5. a file saved in one chat, read back in the next
    s14 = r.selected(w, store="S14", **YTD)
    line = "{0},{1},{2},{3},".format  # a line's start: its revenue's last digits vary by writer
    ends = tuple(line(s.day.isoformat(), s.store, s.sku, s.units) for s in (s14[0], s14[-1]))
    out.append(
        cx.Case(
            "cando-05", "cando", 3,
            (
                cx.Turn(
                    "Save every 2026 sales line of store S14 as a CSV file for me "
                    "(date, store, sku, units, revenue).",
                    (
                        (("sales_rows", {"store": "S14", **YTD}),),
                        (("query_result", {"file_path": saved, "sql": "SELECT * FROM result",
                                           "save_as": "csv", "file_name": "S14_sales_2026"}),),
                    ),
                    reply="Saved S14_sales_2026.csv for you.",
                    sent=(
                        cx.Probe("file", r"\.csv$", "every line of the store is in the user's file",
                                 holds=ends, limit=len(s14) + 1),
                    ),
                ),
                cx.Turn(
                    "In the CSV file you saved for me with store S14's 2026 sales lines: how "
                    "many lines have more than 3 units?",
                    expect=cx.count(sum(1 for s in s14 if s.units > 3)),
                    thread="b",
                ),
            ),
            ("a file saved in one chat, read back in the next",),
            keyed=True,
        )
    )  # fmt: skip
    return out


def names(case: Any) -> set[str]:
    """What ``--cases`` picks a ``cando`` case by besides its id and category."""
    return {"cando"} if case.id.startswith("cando-") else set()
