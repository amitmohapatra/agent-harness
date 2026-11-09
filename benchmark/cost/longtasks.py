"""The ``long`` set (``--cases long``): tasks a user gives an analytics assistant that take many
steps — a chain of tools repeated for each of several items (``long_chain``), a computation
over thousands of rows or a join of two large results (``long_data``), and a request made of
several sub-tasks, some across turns (``long_multi``). Each has exact answers computed from the
world (:mod:`benchmark.cost.retail`), as the rest of the suite's are; the gold calls are one
way to get them (a model may take another: a query over saved rows, a helper, a plan).

It measures ``ReAct`` as built against ``ReAct`` on Deep Agents (``--variant deep``) on the
work Deep Agents' planning, files and helpers are for.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from typing import Any, Final

from benchmark.cost import retail as r

CATEGORIES: Final = ("long_chain", "long_data", "long_multi")
TAG: Final = "long"

Q2: Final = {"start": "2026-04-01", "end": "2026-06-30"}
Q3: Final = {"start": "2026-07-01", "end": "2026-09-30"}
JUL: Final = {"start": "2026-07-01", "end": "2026-07-31"}
AUG: Final = {"start": "2026-08-01", "end": "2026-08-31"}
SEP: Final = {"start": "2026-09-01", "end": "2026-09-30"}
YTD: Final = {"start": "2026-01-01", "end": "2026-09-30"}
#: Days in Q3 and in September (an average daily figure's divisor).
Q3_DAYS: Final = 92
SEP_DAYS: Final = 30


def names(case: Any) -> set[str]:
    """What ``--cases`` picks a case of this set by besides its id and category."""
    return {TAG} if case.category in CATEGORIES else set()


def _summary(group_by: str, metric: str = "revenue", **f: Any) -> tuple[str, dict[str, Any]]:
    return ("sales_summary", {"group_by": group_by, "metric": metric, **f})


def _ids(rows: list[tuple[str, float]]) -> list[str]:
    return [k for k, _ in rows]


def cases() -> list[Any]:  # noqa: PLR0915 - one list of cases, each a few lines
    from benchmark.cost import complex as cx

    w = r.world()
    Case, Turn, record, idset = cx.Case, cx.Turn, cx.record, cx.idset
    out: list[Any] = []

    def add(cid: str, category: str, text: str, calls: Any, expect: Any, *edges: str) -> None:
        out.append(Case(cid, category, 3, (Turn(text, calls, expect),), edges))

    def gp(**f: Any) -> dict[str, float]:
        per: dict[str, float] = defaultdict(float)
        for s in r.selected(w, **f):
            per[w.skus[s.sku].category] += s.revenue - s.units * w.skus[s.sku].unit_cost
        return {k: round(v, 2) for k, v in per.items()}

    # ---------------------------------------------------------------- long_chain
    top6 = _ids(r.truth_top(w, "sku", "revenue", 6, **Q3))
    sup6 = {k: w.skus[k].supplier for k in top6}
    longest = max(set(sup6.values()), key=lambda s: (w.suppliers[s].lead_time_days, s))
    add(
        "long-chain-01", "long_chain",
        "Take the 6 SKUs with the highest revenue in Q3 2026 (July to September). For each one, "
        "find its supplier and that supplier's lead time in the supplier master. Which of those "
        "suppliers has the longest lead time, how many days is it, and what is the 6 SKUs' "
        "combined Q3 revenue?",
        ((_summary("sku", top=6, **Q3),), tuple(("sku_details", {"sku": k}) for k in top6),
         tuple(("supplier_info", {"supplier": s}) for s in sorted(set(sup6.values())))),
        record({"supplier": longest, "lead_days": w.suppliers[longest].lead_time_days,
                "revenue": round(sum(v for _, v in r.truth_top(w, "sku", "revenue", 6, **Q3)), 2)},
               tol={"lead_days": 0, "revenue": 1.0}),
        "a chain per item (6 x 2 tools)", "a max over the chain's results",
    )  # fmt: skip

    top5u = _ids(r.truth_top(w, "sku", "units", 5, **SEP))
    reorder = {}
    for k in top5u:
        avg = r.truth_avg_daily_units(w, k, **SEP)
        lead = w.suppliers[w.skus[k].supplier].lead_time_days
        on = sum(w.on_hand(k, g) for g in r.REGIONS)
        reorder[k] = r.reorder_qty(avg, lead, 14, on)
    add(
        "long-chain-02", "long_chain",
        "Build a reorder list for the 5 SKUs that sold the most units in September 2026. For each: "
        "its average daily units in September (units / 30), its supplier's lead time from the "
        "supplier master, and its total units on hand now. Reorder quantity = average daily units "
        "x (lead time + 14 review days) - on hand, rounded up, never below 0. Give each SKU's "
        "reorder quantity.",
        ((_summary("sku", "units", top=5, **SEP),),
         tuple(c for k in top5u
               for c in (("sku_details", {"sku": k}), ("stock_level", {"sku": k}))),
         tuple(("supplier_info", {"supplier": w.skus[k].supplier}) for k in top5u)),
        record(dict(reorder), tol=dict.fromkeys(reorder, 0.5)),
        "a chain per item (5 x 3 tools)", "a formula applied to each",
    )  # fmt: skip

    per_store = {}
    for g in r.REGIONS:
        stores = sum(1 for s in w.stores if s.region == g)
        per_store[g] = r.truth_total(w, "revenue", region=g, **Q3) / stores
    best = max(per_store, key=per_store.__getitem__)
    add(
        "long-chain-03", "long_chain",
        "For each of the four regions (north, south, east, west): its Q3 2026 revenue, how many "
        "stores it has, and its revenue per store. Which region has the highest revenue per "
        "store? Give all four revenue-per-store figures.",
        ((_summary("region", **Q3), *(("store_list", {"region": g}) for g in r.REGIONS)),),
        record({**{g: round(v, 2) for g, v in per_store.items()}, "best": best},
               tol=dict.fromkeys(per_store, 1.0)),
        "a chain per region", "a ratio per item",
    )  # fmt: skip

    top8u = _ids(r.truth_top(w, "sku", "units", 8, **Q3))
    sups = sorted({w.skus[k].supplier for k in top8u})
    gaps = {s: abs(w.suppliers[s].lead_time_days - w.suppliers[s].portal_lead_time_days)
            for s in sups}  # fmt: skip
    differ = [s for s in sups if gaps[s]]
    widest = max(sups, key=lambda s: (gaps[s], s))
    add(
        "long-chain-04", "long_chain",
        "Look at the suppliers of the 8 SKUs that sold the most units in Q3 2026. For each "
        "distinct supplier compare the lead time in the supplier master with the one quoted on "
        "the supplier portal. How many of them disagree, which supplier has the largest gap, and "
        "how many days is that gap?",
        ((_summary("sku", "units", top=8, **Q3),),
         tuple(("sku_details", {"sku": k}) for k in top8u),
         tuple(c for s in sups for c in (("supplier_info", {"supplier": s}),
                                          ("supplier_portal", {"supplier": s})))),
        record({"disagree": len(differ), "supplier": widest, "gap_days": gaps[widest]},
               tol={"disagree": 0, "gap_days": 0}),
        "two sources per item", "a chain three tools deep",
    )  # fmt: skip

    cut = [s for s in w.skus.values() if s.price_before is not None]
    cut_top = sorted(cut, key=lambda s: (-r.truth_total(w, "revenue", sku=s.sku, **Q3), s.sku))[:5]
    before = {"start": "2026-04-15", "end": "2026-05-14"}
    after = {"start": "2026-05-15", "end": "2026-06-13"}
    growth = {}
    for s in cut_top:
        b = r.truth_total(w, "units", sku=s.sku, **before)
        a = r.truth_total(w, "units", sku=s.sku, **after)
        growth[s.sku] = r.growth_pct(b, a) if b else math.inf
    grew = max(growth, key=lambda k: (growth[k], k))
    ids = ", ".join(s.sku for s in cut_top)
    add(
        "long-chain-05", "long_chain",
        f"These SKUs had a price change this year: {ids}. For each, get its price history (the "
        "price before and after the change) and its units sold in the 30 days before the change "
        "(2026-04-15 to 2026-05-14) and the 30 days from it (2026-05-15 to 2026-06-13). Which "
        "SKU's "
        "units grew the most in percent, and by what percentage?",
        (tuple(("price_history", {"sku": s.sku}) for s in cut_top),
         tuple(_summary("sku", "units", sku=s.sku, **p) for s in cut_top for p in (before, after))),
        record({"sku": grew, "growth_pct": round(growth[grew], 2)}, tol={"growth_pct": 0.06}),
        "two periods per item (5 x 3 tools)", "a before and after per item",
    )  # fmt: skip

    cat_top = {c: _ids(r.truth_top(w, "sku", "revenue", 1, category=c, **Q3))[0]
               for c in r.CATEGORIES}  # fmt: skip
    cat_sup = sorted({w.skus[k].supplier for k in cat_top.values()})
    scored = {s: r.supplier_score(w.suppliers[s]) for s in cat_sup}
    win = max(scored, key=lambda s: (scored[s], s))
    add(
        "long-chain-06", "long_chain",
        "For each of the eight categories, find its top SKU by Q3 2026 revenue and that SKU's "
        "supplier. Score each of those suppliers as 0.6 x on-time rate + 0.4 x (1 - lead time "
        "days / 60), using the supplier master. Which supplier scores highest, and what is its "
        "score (4 decimals)?",
        (tuple(_summary("sku", top=1, category=c, **Q3) for c in r.CATEGORIES),
         tuple(("sku_details", {"sku": k}) for k in cat_top.values()),
         tuple(("supplier_info", {"supplier": s}) for s in cat_sup)),
        record({"supplier": win, "score": round(scored[win], 4)}, tol={"score": 0.0006}),
        "a chain per category (8 x 3 tools)", "a weighted score",
    )  # fmt: skip

    west3 = _ids(r.truth_top(w, "store", "revenue", 3, region="west", **Q3))
    store_cat = {s: r.truth_top(w, "category", "revenue", 1, store=s, **Q3)[0] for s in west3}
    add(
        "long-chain-07", "long_chain",
        "Find the 3 west-region stores with the highest Q3 2026 revenue. For each, its top "
        "category by Q3 revenue in that store and that category's revenue there. Give each "
        "store's top category, and the sum of those three category revenues.",
        ((_summary("store", top=3, region="west", **Q3),),
         tuple(_summary("category", store=s, **Q3) for s in west3)),
        record({**{s: store_cat[s][0] for s in west3},
                "total": round(sum(v for _, v in store_cat.values()), 2)}, tol={"total": 1.0}),
        "a drill-down per item",
    )  # fmt: skip

    # ---------------------------------------------------------------- long_data
    knit = r.selected(w, category="knitwear", **Q3)
    knit_day = r.truth_top(w, "date", "revenue", 1, category="knitwear", **Q3)[0]
    add(
        "long-data-01", "long_data",
        "Over every knitwear sales line of Q3 2026: how many lines are there, what is the median "
        "revenue of a line, what is the 90th percentile of units per line (linear interpolation), "
        "and which day had the highest knitwear revenue?",
        ((("sales_rows", {"category": "knitwear", **Q3}),),),
        record({"lines": len(knit), "median": round(statistics.median(s.revenue for s in knit), 2),
                "p90_units": r.percentile([s.units for s in knit], 90), "day": knit_day[0]},
               tol={"lines": 0, "median": 0.01, "p90_units": 0.01}),
        "thousands of rows", "statistics over a large result",
    )  # fmt: skip

    gpc = sorted(gp(**Q3).items(), key=lambda kv: (-kv[1], kv[0]))[:3]
    add(
        "long-data-02", "long_data",
        "Gross profit is revenue minus units x the SKU's unit cost. Compute Q3 2026 gross profit "
        "for every category from the sales lines and the catalogue, and give the top 3 categories "
        "with their gross profit.",
        ((("sales_rows", Q3), ("sku_catalog", {})),),
        record(dict(gpc), tol={k: 1.0 for k, _ in gpc}),
        "a join of two large results", "a group sum after the join",
    )  # fmt: skip

    margins: dict[str, list[float]] = defaultdict(list)
    for s in w.skus.values():
        margins[s.category].append(r.margin_pct(s.price, s.unit_cost))
    avg_m = {c: statistics.fmean(v) for c, v in margins.items()}
    low = min(avg_m, key=avg_m.__getitem__)
    under = sum(1 for v in margins.values() for m in v if m < 40)
    add(
        "long-data-03", "long_data",
        "From the whole product catalogue: the margin of a SKU is (price - unit cost) / price. "
        "Which category has the lowest average margin, what is that average (in percent, 2 "
        "decimals), and how many SKUs in all have a margin under 40%?",
        ((("sku_catalog", {}),),),
        record({"category": low, "avg_margin_pct": round(avg_m[low], 2), "under_40": under},
               tol={"avg_margin_pct": 0.006, "under_40": 0}),
        "2,000 rows", "a per-row formula then a group mean",
    )  # fmt: skip

    fd = r.daily(w, category="footwear", **SEP)
    mu, sd = statistics.fmean(fd), statistics.stdev(fd)
    add(
        "long-data-04", "long_data",
        "Footwear's revenue for each day of September 2026: what is the mean daily revenue, the "
        "sample standard deviation, and on how many days was revenue above the mean plus one "
        "standard deviation?",
        ((_summary("date", category="footwear", **SEP),),),
        record({"mean": round(mu, 2), "sd": round(sd, 2),
                "days": sum(1 for v in fd if v > mu + sd)},
               tol={"mean": 0.5, "sd": 0.5, "days": 0}),
        "a series of 30 values", "mean, deviation and a count",
    )  # fmt: skip

    store_q3 = r.grouped(r.selected(w, **Q3), w, "store", "revenue")
    outl = r.iqr_outliers(store_q3)
    add(
        "long-data-05", "long_data",
        "Take every store's Q3 2026 revenue. Using quartiles by linear interpolation, which stores "
        "are outliers (above Q3 + 1.5 x IQR or below Q1 - 1.5 x IQR)? If none are, say none.",
        ((_summary("store", **Q3),),),
        idset(outl) if outl else cx.record({"outliers": 0}, tol={"outliers": 0}),
        "40 values", "quartiles and fences",
    )  # fmt: skip

    tps = r.truth_top_profit_skus(w, 5, **Q3)
    add(
        "long-data-06", "long_data",
        "Which 5 SKUs made the most gross profit (revenue - units x unit cost) in Q3 2026, and how "
        "much did each make?",
        ((("sales_rows", Q3), ("sku_catalog", {})),),
        record(dict(tps), tol={k: 1.0 for k, _ in tps}),
        "a join of two large results", "a top-N after the join",
    )  # fmt: skip

    cmgr = {}
    for g in r.REGIONS:
        months = [v for _, v in r.truth_monthly(w, region=g, **YTD)]
        cmgr[g] = ((months[-1] / months[0]) ** (1 / 8) - 1) * 100
    fast = max(cmgr, key=cmgr.__getitem__)
    add(
        "long-data-07", "long_data",
        "For each region, its monthly revenue from January to September 2026. Compute each "
        "region's compound monthly growth rate from January to September ((September / January) "
        "^ (1/8) - 1). Which region grew fastest, and at what rate (percent, 2 decimals)? Give all "
        "four rates.",
        (tuple(_summary("month", region=g, **YTD) for g in r.REGIONS),),
        record({**{g: round(v, 2) for g, v in cmgr.items()}, "fastest": fast},
               tol=dict.fromkeys(cmgr, 0.06)),
        "36 values", "a rate per item",
    )  # fmt: skip

    sku_rev = sorted(r.grouped(r.selected(w, **Q3), w, "sku", "revenue").values(), reverse=True)
    tenth = math.ceil(len(sku_rev) / 10)
    share = sum(sku_rev[:tenth]) / sum(sku_rev) * 100
    add(
        "long-data-08", "long_data",
        "Of the SKUs that sold anything in Q3 2026, take the top 10% by Q3 revenue (round the "
        "count up). How many SKUs is that, and what share of Q3 revenue (percent, 2 decimals) do "
        "they make?",
        ((_summary("sku", **Q3),),),
        record({"skus": tenth, "share_pct": round(share, 2)}, tol={"skus": 0, "share_pct": 0.06}),
        "a ranking of ~2,000 items", "a share of the top decile",
    )  # fmt: skip

    pivot = {
        (g, c): v
        for g in r.REGIONS
        for c, v in r.grouped(r.selected(w, region=g, **Q3), w, "category", "revenue").items()
    }
    hi, lo = max(pivot, key=pivot.__getitem__), min(pivot, key=pivot.__getitem__)
    add(
        "long-data-09", "long_data",
        "Make a region x category table of Q3 2026 revenue (4 regions x 8 categories). Which "
        "region and category pair has the highest revenue and which the lowest, with both "
        "figures?",
        (tuple(_summary("category", region=g, **Q3) for g in r.REGIONS),),
        record({"high_region": hi[0], "high_category": hi[1], "high": pivot[hi],
                "low_region": lo[0], "low_category": lo[1], "low": pivot[lo]},
               tol={"high": 1.0, "low": 1.0}),
        "a 32-cell pivot", "a max and a min",
    )  # fmt: skip

    # ---------------------------------------------------------------- long_multi
    ytd = r.truth_total(w, "revenue", **YTD)
    busiest = max(r.truth_monthly(w, "units", **YTD), key=lambda kv: kv[1])[0]
    outer = sum(1 for s in w.skus.values() if s.category == "outerwear")
    s07 = w.suppliers["SUP-07"]
    north = r.truth_total(w, "revenue", region="north", **Q3) / r.truth_total(w, "revenue", **Q3)
    add(
        "long-multi-01", "long_multi",
        "Five things for my weekly note: (1) total revenue for 2026 so far; (2) the month with the "
        "most units sold; (3) how many SKUs are in outerwear; (4) the gap in days between SUP-07's "
        "lead time in the supplier master and on its portal; (5) the north's share of Q3 2026 "
        "revenue in percent.",
        ((_summary("region", **YTD), _summary("month", "units", **YTD),
          ("sku_catalog", {"category": "outerwear"}), ("supplier_info", {"supplier": "SUP-07"}),
          ("supplier_portal", {"supplier": "SUP-07"}), _summary("region", **Q3)),),
        record({"ytd": ytd, "month": (busiest, busiest[5:]), "outerwear": outer,
                "gap": abs(s07.lead_time_days - s07.portal_lead_time_days),
                "north_pct": round(north * 100, 2)},
               tol={"ytd": 1.0, "outerwear": 0, "gap": 0, "north_pct": 0.06}),
        "five unrelated sub-tasks",
    )  # fmt: skip

    dq2, dq3 = (r.truth_total(w, "revenue", category="denim", **q) for q in (Q2, Q3))
    dtop = _ids(r.truth_top(w, "sku", "revenue", 1, category="denim", **Q3))[0]
    dsup = w.suppliers[w.skus[dtop].supplier]
    dstores = len({s.store for s in r.selected(w, category="denim", **Q3)})
    add(
        "long-multi-02", "long_multi",
        "Prepare a quarterly review of denim: Q2 and Q3 2026 revenue and the growth in percent; "
        "the top denim SKU of Q3 by revenue, its supplier and that supplier's on-time rate; how "
        "many stores sold denim in Q3; and denim's Q3 gross profit (revenue - units x unit cost).",
        ((_summary("category", category="denim", **Q2),
          _summary("sku", top=1, category="denim", **Q3),
          _summary("store", category="denim", **Q3), ("sales_rows", {"category": "denim", **Q3}),
          ("sku_catalog", {"category": "denim"})),
         (("sku_details", {"sku": dtop}),), (("supplier_info", {"supplier": dsup.supplier}),)),
        record({"q2": dq2, "q3": dq3, "growth_pct": round(r.growth_pct(dq2, dq3), 2), "sku": dtop,
                "supplier": dsup.supplier, "on_time": dsup.on_time_rate, "stores": dstores,
                "gross_profit": r.truth_gross_profit(w, category="denim", **Q3)},
               tol={"q2": 1.0, "q3": 1.0, "growth_pct": 0.06, "on_time": 0.0051, "stores": 0,
                    "gross_profit": 1.0}),
        "six sub-tasks", "a join among them",
    )  # fmt: skip

    def region_view(g: str) -> dict[str, Any]:
        rev = r.truth_total(w, "revenue", region=g, **Q3)
        units = r.truth_total(w, "units", region=g, **Q3)
        prev = r.truth_total(w, "revenue", region=g, **Q2)
        return {"rev": rev, "asp": rev / units, "growth": r.growth_pct(prev, rev),
                "top": r.truth_top(w, "category", "revenue", 1, region=g, **Q3)[0][0]}  # fmt: skip

    ev, so = region_view("east"), region_view("south")
    add(
        "long-multi-03", "long_multi",
        "Compare the east and the south for Q3 2026: each one's revenue, its average selling price "
        "(revenue / units), its top category, and its revenue growth vs Q2 in percent. Which of "
        "the two grew faster?",
        ((_summary("region", **Q3), _summary("region", "units", **Q3), _summary("region", **Q2),
          _summary("category", region="east", **Q3), _summary("category", region="south", **Q3)),),
        record({"east_revenue": ev["rev"], "south_revenue": so["rev"],
                "east_asp": round(ev["asp"], 2), "south_asp": round(so["asp"], 2),
                "east_top": ev["top"], "south_top": so["top"],
                "east_growth": round(ev["growth"], 2), "south_growth": round(so["growth"], 2),
                "faster": "east" if ev["growth"] > so["growth"] else "south"},
               tol={"east_revenue": 1.0, "south_revenue": 1.0, "east_asp": 0.01,
                    "south_asp": 0.01, "east_growth": 0.06, "south_growth": 0.06}),
        "two items x four measures",
    )  # fmt: skip

    n4 = _ids(r.truth_top(w, "sku", "units", 4, region="north", **Q3))
    nre = {}
    for k in n4:
        avg = r.truth_avg_daily_units(w, k, region="north", **Q3)
        lead = w.suppliers[w.skus[k].supplier].lead_time_days
        nre[k] = r.reorder_qty(avg, lead, 7, w.on_hand(k, "north"))
    add(
        "long-multi-04", "long_multi",
        "A restock plan for the north: the 4 SKUs that sold the most units in north stores in Q3 "
        "2026. For each: units on hand in the north now, its average daily units in the north "
        "over Q3 (units / 92), and its supplier's master lead time. Reorder quantity = average "
        "daily units x (lead time + 7) - north on hand, rounded up, never below 0. Give each "
        "SKU's reorder quantity.",
        ((_summary("sku", "units", top=4, region="north", **Q3),),
         tuple(c for k in n4 for c in (("sku_details", {"sku": k}),
                                       ("stock_level", {"sku": k, "region": "north"}))),
         tuple(("supplier_info", {"supplier": w.skus[k].supplier}) for k in n4)),
        record(nre, tol=dict.fromkeys(nre, 0.5)),
        "a chain per item, one region", "a formula per item",
    )  # fmt: skip

    mon = [v for _, v in r.truth_monthly(w, **YTD)]
    slope, icpt = r.regression(range(1, 10), mon)
    mean = statistics.fmean(mon)
    add(
        "long-multi-05", "long_multi",
        "Total revenue for each month from January to September 2026: fit a least-squares line of "
        "revenue on month number (January = 1). Give the slope (GBP per month), the projected "
        "October revenue (month 10) from that line, and how many of the nine months were above "
        "the nine-month mean.",
        ((_summary("month", **YTD),),),
        record({"slope": slope, "october": slope * 10 + icpt,
                "above": sum(1 for v in mon if v > mean)},
               tol={"slope": max(abs(slope) * 0.001, 0.5), "october": max(abs(icpt) * 0.001, 1.0),
                    "above": 0}),
        "a fit and a projection",
    )  # fmt: skip

    top3c = r.truth_top(w, "category", "revenue", 3, **Q3)
    pick = {c: _ids(r.truth_top(w, "sku", "revenue", 2, category=c, **Q3)) for c, _ in top3c}
    psup = sorted({w.skus[k].supplier for v in pick.values() for k in v})
    plead = max(psup, key=lambda s: (w.suppliers[s].portal_lead_time_days, s))
    out.append(Case(
        "long-multi-06", "long_multi", 3,
        (
            Turn("What were the top 3 categories by revenue in Q3 2026, with their revenue?",
                 ((_summary("category", top=3, **Q3),),),
                 record(dict(top3c), tol={c: 1.0 for c, _ in top3c})),
            Turn("For each of those three categories, take its 2 best SKUs by Q3 revenue, find "
                 "each SKU's supplier and that supplier's quoted lead time on the supplier portal. "
                 "Which "
                 "supplier quotes the longest lead time, and how many days?",
                 (tuple(_summary("sku", top=2, category=c, **Q3) for c, _ in top3c),
                  tuple(("sku_details", {"sku": k}) for v in pick.values() for k in v),
                  tuple(("supplier_portal", {"supplier": s}) for s in psup)),
                 record({"supplier": plead,
                         "days": w.suppliers[plead].portal_lead_time_days}, tol={"days": 0})),
        ),
        ("an earlier turn's list driving a chain", "a chain per item (6 x 2 tools)"),
    ))  # fmt: skip

    sr = r.truth_supplier_revenue(w, **Q3)
    total = sum(v for _, v in sr)
    s3 = [k for k, _ in sr[:3]]
    add(
        "long-multi-07", "long_multi",
        "Which 3 suppliers' products made the most revenue in Q3 2026, and what share of Q3 "
        "revenue (percent, 2 decimals) do the three make together? For each of the three, give "
        "its city and on-time rate from the supplier master.",
        ((_summary("supplier", **Q3),), tuple(("supplier_info", {"supplier": s}) for s in s3)),
        record({"share_pct": round(sum(v for _, v in sr[:3]) / total * 100, 2),
                **{f"{s}_city": w.suppliers[s].city for s in s3},
                **{f"{s}_on_time": w.suppliers[s].on_time_rate for s in s3}},
               tol={"share_pct": 0.06, **{f"{s}_on_time": 0.0051 for s in s3}}),
        "a ranking then a chain per item",
    )  # fmt: skip

    monthly_top = [set(_ids(r.truth_top(w, "store", "units", 3, **m))) for m in (JUL, AUG, SEP)]
    every = sorted(set.intersection(*monthly_top))
    add(
        "long-multi-08", "long_multi",
        "List the top 3 stores by units sold in each of July, August and September 2026. Which "
        "stores are in the top 3 in all three months? If none, say none.",
        (tuple(_summary("store", "units", top=3, **m) for m in (JUL, AUG, SEP)),),
        idset(every) if every else record({"stores": 0}, tol={"stores": 0}),
        "three rankings intersected",
    )  # fmt: skip
    return out
