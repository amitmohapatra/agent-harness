"""The ``long`` set (``--cases long``): tasks a user gives an analytics assistant that take many
steps — a chain of tools repeated for each of several items (``long_chain``), a computation
over thousands of rows or a join of two large results (``long_data``), and a request made of
several sub-tasks, some across turns (``long_multi``). Each has exact answers computed from the
world (:mod:`benchmark.cost.retail`), as the rest of the suite's are; the gold calls are one
way to get them (a model may take another: a query over saved rows, a helper, a plan).

``long_flow`` is the long workflows (:func:`flow_cases`): a job of fifty and more tool calls,
one whose context is cleared while it runs, one that waits for a person's approval in its
middle, one whose worker dies and another finishes it, a twelve-turn plan whose earlier
constraint a later turn changes, and jobs one of whose tools fails once or for good. Their
tools (:data:`TOOLS`) are the set's own, offered only to the cases that need them: each has no
shorter way round (an audit reads one store and one category a call), and what the writes did
is kept per person (:data:`TENDERS`, :data:`TRANSFERS`), so a write done twice shows.

It measures ``ReAct`` as built against ``ReAct`` on Deep Agents (``--variant deep``) on the
work Deep Agents' planning, files and helpers are for.
"""

from __future__ import annotations

import math
import random
import re
import statistics
from collections import defaultdict
from datetime import date, timedelta
from functools import cache
from typing import Any, Final

from benchmark.cost import retail as r
from trellis import current, tool
from trellis.harness.react import CLEARED

CATEGORIES: Final = ("long_chain", "long_data", "long_multi", "long_flow")
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


# --------------------------------------------------------------------------- shelf audits
#: The bays of one category's shelf in a store, as an audit lists them, and the day of the
#: latest audit.
BAYS: Final = 24
AUDITED: Final = "2026-10-27"


@cache
def audit(store: str, category: str) -> tuple[tuple[int, int, int], ...]:
    """Each bay of a category's shelf in a store: the facings planned, those found, and how
    many days before the audit it was last filled — fixed by the world's seed, the store and
    the category."""
    rng = random.Random(f"{r.SEED}:audit:{store}:{category}")
    bays = []
    for _ in range(BAYS):
        planned = rng.randint(4, 12)
        short = rng.randint(1, 3) if rng.random() < 0.35 else 0
        bays.append((planned, planned - short, rng.randint(0, 6)))
    return tuple(bays)


def gap(store: str, category: str) -> int:
    """A cell's gap: the facings planned less those found."""
    return sum(planned - found for planned, found, _ in audit(store, category))


@tool(side_effects="read")
def shelf_audit(store: str, category: str) -> str:
    """The latest shelf audit of one category in one store: the facings planned and found in
    each bay and the day it was last filled, and the totals. One store and one category a call.

    Args:
        store: the store id, e.g. S07
        category: the category
    """
    store, category = store.upper(), category.lower()
    if store not in {s.store for s in r.world().stores} or category not in r.CATEGORIES:
        return r._json({"error": f"no audit of {category} in {store}"})
    bays = audit(store, category)
    day = date.fromisoformat(AUDITED)
    return r._json(
        {
            "store": store,
            "category": category,
            "audited": AUDITED,
            "bays": [
                {"bay": f"B{n:02d}", "planned": planned, "found": found,
                 "last_filled": (day - timedelta(days=ago)).isoformat()}
                for n, (planned, found, ago) in enumerate(bays, 1)
            ],
            "planned": sum(planned for planned, _, _ in bays),
            "found": sum(found for _, found, _ in bays),
        }
    )  # fmt: skip


# --------------------------------------------------------------------------- what a job wrote
class WorkerDied(BaseException):
    """The worker process running the call dies: nothing more is written, its lease lapses
    (a ``Turn.restart`` turn's run is then finished by another worker)."""

    @classmethod
    def within(cls, exc: BaseException) -> bool:
        """Whether ``exc`` is the worker's death: as raised, or as a framework's task group
        raised it (the Claude Agent SDK runs a tool in one)."""
        if isinstance(exc, BaseExceptionGroup):
            return any(cls.within(e) for e in exc.exceptions)
        return isinstance(exc, cls)


#: Each person's stock transfers, in the order they were placed: the set's own side effects,
#: kept apart per person (a case's user is its own, ``complex.run_case``), so a write done
#: twice is two entries.
TRANSFERS: Final[dict[str, list[dict[str, Any]]]] = {}
#: Each person's freight tenders by id: the pallets, the lanes quoted and the awards made.
TENDERS: Final[dict[str, dict[str, dict[str, Any]]]] = {}
#: The calls that failed once for a person already: the next one of the same passes.
FAILED: Final[dict[str, set[str]]] = {}


def _person() -> str:
    runtime = current()
    return runtime.user if runtime is not None else ""


def _first(call: str) -> bool:
    """Whether this is the person's first such call (it is then no longer)."""
    seen = FAILED.setdefault(_person(), set())
    return call not in seen and not seen.add(call)


def transit_days(from_region: str, to_region: str) -> int:
    return 2 + abs(r.REGIONS.index(from_region) - r.REGIONS.index(to_region))


@tool(side_effects="irreversible")
def place_transfer(sku: str, from_region: str, to_region: str, units: int) -> str:
    """Move units of a SKU from one region's stock to another's: the transfer ships at once
    and cannot be undone. Returns the transfer's id.

    Args:
        sku: the SKU id
        from_region: the region the units leave (north, south, east or west)
        to_region: the region they go to
        units: how many units
    """
    w = r.world()
    sku, src, dst = sku.upper(), from_region.lower(), to_region.lower()
    if sku not in w.skus or src not in r.REGIONS or dst not in r.REGIONS or src == dst:
        return r._json({"error": f"no transfer of {sku} from {from_region} to {to_region}"})
    if not 0 < units <= w.on_hand(sku, src):
        return r._json({"error": f"{src} holds {w.on_hand(sku, src)} units of {sku}"})
    placed = TRANSFERS.setdefault(_person(), [])
    placed.append({"transfer": f"TR-{len(placed) + 1:04d}", "sku": sku, "from": src, "to": dst,
                   "units": units, "arrives_in_days": transit_days(src, dst)})  # fmt: skip
    return r._json({"transfer": placed[-1]["transfer"], "status": "shipped"})


@tool(side_effects="read")
def transfer_status(transfer: str) -> str:
    """A stock transfer's status: what it moves, and in how many days it arrives.

    Args:
        transfer: the transfer's id, as place_transfer gives it
    """
    found = [t for t in TRANSFERS.get(_person(), []) if t["transfer"] == transfer.upper()]
    if not found:
        return r._json({"error": f"no transfer {transfer}"})
    return r._json(found[0] | {"status": "in transit"})


#: The suppliers' lanes each freight case tenders, and the longest quoted lead time (the
#: supplier portal's) a lane may have to be awarded: the first group's cheapest lane fails
#: once (:data:`FLAKY`), an award to a lane of the second finds the desk busy once
#: (:data:`BUSY`), the third's cheapest lane is down for good (:data:`DOWN`), and the quote
#: of the fourth's cheapest kills the worker once (:data:`DEADLY`).
LANES: Final = tuple(
    tuple(f"SUP-{n:02d}" for n in range(first, first + 6)) for first in (1, 7, 13, 19)
)
LEAD_LIMITS: Final = (21, 35, 35, 35)
PALLETS: Final = 12


def quote(supplier: str) -> float:
    """A lane's quote per pallet (GBP): from its supplier's record in the world."""
    s = r.world().suppliers[supplier]
    return round(42 + 260 * (1 - s.on_time_rate) + 0.08 * s.min_order_qty, 2)


def eligible(group: int, *, without: str | None = None) -> list[str]:
    """A group's lanes that may be awarded (their portal lead time within its limit),
    cheapest first — ``without``: a lane that could not be quoted."""
    w = r.world()
    lanes = [s for s in LANES[group] if s != without
             and w.suppliers[s].portal_lead_time_days <= LEAD_LIMITS[group]]  # fmt: skip
    return sorted(lanes, key=lambda s: (quote(s), s))


@cache
def broken() -> dict[str, str]:
    """The lanes that fail, and how: ``flaky`` (its first quote is a 503), ``down`` (every
    quote is a 502) and ``deadly`` (its first quote kills the worker) — each its group's
    cheapest lane that may be awarded, so the job's answer needs it."""
    return {"flaky": eligible(0)[0], "down": eligible(2)[0], "deadly": eligible(3)[0]}


#: The lanes whose award finds the tender desk busy, once.
BUSY: Final = frozenset(LANES[1])


@tool(side_effects="write")
def open_tender(pallets: int) -> str:
    """Open a freight tender for a number of pallets: carriers quote under its id, and it is
    awarded to one lane. Returns the tender's id.

    Args:
        pallets: how many pallets are to be shipped
    """
    mine = TENDERS.setdefault(_person(), {})
    ident = f"TN-{len(mine) + 1:04d}"
    mine[ident] = {"pallets": pallets, "quotes": {}, "awards": []}
    return r._json({"tender": ident, "pallets": pallets, "status": "open"})


@tool(side_effects="read")
def carrier_quote(supplier: str, tender: str) -> str:
    """The carrier's quote (GBP per pallet) for the lane from a supplier, under an open
    tender. One lane a call.

    Args:
        supplier: the supplier id, e.g. SUP-04
        tender: the tender's id, as open_tender gives it
    """
    supplier, tender = supplier.upper(), tender.upper()
    held = TENDERS.get(_person(), {}).get(tender)
    if supplier not in r.world().suppliers or held is None:
        return r._json({"error": f"no lane {supplier} under tender {tender}"})
    fails = broken()
    if supplier == fails["deadly"] and _first(f"quote {supplier}"):
        raise WorkerDied(f"the worker died while {supplier}'s quote was read")
    if supplier == fails["flaky"] and _first(f"quote {supplier}"):
        raise RuntimeError("carrier API: 503 Service Unavailable (try again)")
    if supplier == fails["down"]:
        raise RuntimeError("carrier API: 502 Bad Gateway from this lane's carrier")
    held["quotes"][supplier] = quote(supplier)
    return r._json({"tender": tender, "supplier": supplier, "per_pallet": quote(supplier),
                    "currency": "GBP"})  # fmt: skip


@tool(side_effects="write")
def award_tender(tender: str, supplier: str) -> str:
    """Award an open tender to one supplier's lane at its quote: the freight is booked.

    Args:
        tender: the tender's id
        supplier: the supplier whose lane takes it
    """
    supplier, tender = supplier.upper(), tender.upper()
    held = TENDERS.get(_person(), {}).get(tender)
    if held is None or supplier not in held["quotes"]:
        return r._json({"error": f"no quote of {supplier} under tender {tender}"})
    if supplier in BUSY and _first("award"):
        raise RuntimeError("tender desk busy: nothing was awarded (try again)")
    held["awards"].append(supplier)
    total = round(held["quotes"][supplier] * held["pallets"], 2)
    return r._json({"tender": tender, "awarded_to": supplier, "total": total, "status": "booked"})


TOOLS: Final = (
    shelf_audit, place_transfer, transfer_status, open_tender, carrier_quote, award_tender,
)  # fmt: skip


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
    return [*out, *flow_cases()]


# --------------------------------------------------------------------------- long workflows
#: The window of the job whose context is cleared while it runs: its forty audits and the
#: tools offered do not fit half of it.
SMALL_WINDOW: Final = 32_000
#: The most model calls of a job that must end within the suite's step limit (``--max-steps``
#: 12, and the one that answers at the limit).
STEPS: Final = 13
#: The review days of the twelve-turn restock plan: as first asked, and as changed later.
REVIEW_DAYS: Final = (10, 21)


def _tender(group: int, *, without: str | None = None) -> tuple[str, Any, dict[str, Any]]:
    """A freight job on a group's lanes: what is asked, its gold rounds but the award, and the
    answer's fields — ``without``: the lane that cannot be quoted."""
    w = r.world()
    lanes, limit = LANES[group], LEAD_LIMITS[group]
    win = eligible(group, without=without)[0]
    text = (
        f"We need to ship {PALLETS} pallets. Open a freight tender for {PALLETS} pallets, then "
        f"get a carrier quote under that tender for each of these suppliers' lanes: "
        f"{', '.join(lanes)}. Check each supplier's quoted lead time on the supplier portal too. "
        f"Award the tender to the cheapest lane among those whose portal lead time is at most "
        f"{limit} days. Give me the tender's id, the supplier it went to, its quote per pallet "
        f"and the total for the {PALLETS} pallets."
    )
    rounds = (
        (("open_tender", {"pallets": PALLETS}),),
        tuple(("supplier_portal", {"supplier": s}) for s in lanes),
        tuple(("carrier_quote", {"supplier": s, "tender": "TN-0001"}) for s in lanes),
    )
    assert all(w.suppliers[s].portal_lead_time_days <= limit for s in (win,))
    fields = {"tender": "TN-0001", "supplier": win, "per_pallet": quote(win),
              "total": round(quote(win) * PALLETS, 2)}  # fmt: skip
    return text, rounds, fields


def flow_cases() -> list[Any]:  # noqa: PLR0915 - one list of cases, each a few lines
    """The long workflows (``long_flow``): the module says what each is for."""
    from benchmark.cost import complex as cx

    w = r.world()
    Case, Turn, Probe, record = cx.Case, cx.Turn, cx.Probe, cx.record
    out: list[Any] = []
    once = "the write is done once"

    def exactly(tool_name: str, n: int = 1) -> tuple[Any, ...]:
        return (Probe("called", tool_name, f"{tool_name} ran", limit=n),
                Probe("atmost", tool_name, once, limit=n))  # fmt: skip

    # 01: fifty and more tool calls, a ranked table at the end
    top7 = _ids(r.truth_top(w, "store", "revenue", 7, **Q3))
    gaps = {s: sum(gap(s, c) for c in r.CATEGORIES) for s in top7}
    ranked = sorted(gaps, key=lambda s: (-gaps[s], s))
    wide = sum(1 for s in top7 for c in r.CATEGORIES if gap(s, c) >= 15)
    out.append(Case(
        "long-flow-01", "long_flow", 3,
        (Turn(
            "A shelf audit sweep: take the 7 stores with the highest Q3 2026 revenue. For each of "
            "them, read the shelf audit of each of the eight categories "
            f"({', '.join(r.CATEGORIES)}): "
            "one store and one category a call, 56 audits. A cell's gap is its facings planned "
            "minus its facings found. Give a table of the 7 stores ranked by total gap, largest "
            "first, with each store's total gap; then the combined gap of the three worst stores, "
            "and how many of the 56 cells have a gap of 15 or more.",
            ((_summary("store", top=7, **Q3),),
             *(tuple(("shelf_audit", {"store": s, "category": c}) for c in r.CATEGORIES)
               for s in top7)),
            record({**gaps, "worst3": sum(gaps[s] for s in ranked[:3]), "cells": wide},
                   tol=dict.fromkeys([*gaps, "worst3", "cells"], 0)),
            sent=(Probe("called", "shelf_audit", "all 56 audits ran", limit=56),
                  Probe("distinct", "shelf_audit", "no audit is read twice"),
                  Probe("calls", why="the job ends within the step limit", limit=STEPS)),
        ),),
        ("57 tool calls", "a ranked table of every item"),
        tools=(shelf_audit,),
    ))  # fmt: skip

    # 02: the context cleared while the job runs; the answer needs the first step's figures
    west = r.truth_top(w, "store", "revenue", 5, region="west", **Q3)
    per = {s: sum(gap(s, c) for c in r.CATEGORIES) / revenue * 10_000 for s, revenue in west}
    out.append(Case(
        "long-flow-02", "long_flow", 3,
        (Turn(
            "Take the 5 west-region stores with the highest Q3 2026 revenue, with that revenue. "
            "Then read the shelf audit of each of the eight categories in each of them "
            f"({', '.join(r.CATEGORIES)}: one store and one category a call, 40 audits). A cell's "
            "gap is its facings planned minus its facings found. For each store give its total "
            "gap per GBP 10,000 of its Q3 revenue (2 decimals). Which store's is highest?",
            ((_summary("store", top=5, region="west", **Q3),),
             *(tuple(("shelf_audit", {"store": s, "category": c}) for c in r.CATEGORIES)
               for s, _ in west)),
            record({s: round(v, 2) for s, v in per.items()}, tol=dict.fromkeys(per, 0.006)),
            sent=(Probe("fits", why="every request fits the window", limit=SMALL_WINDOW),
                  Probe("has", re.escape(CLEARED.split(":")[0]),
                        "older results were cleared while the job ran", role="tool"),
                  Probe("called", "shelf_audit", "all 40 audits ran", limit=40),
                  Probe("distinct", "shelf_audit|sales_summary",
                        "a cleared result is read back, not fetched again")),
        ),),
        ("context cleared mid-run", "the first step's figures in the answer"),
        tools=(shelf_audit,),
        adapters=("react", "langgraph", "deepagents"),
        only_because="the window is the harness's middleware's to keep (ReAct, middleware()): "
                     "the other frameworks keep their own context",
        window=SMALL_WINDOW,
    ))  # fmt: skip

    # 03: a person's approval in the middle of the job
    n6 = _ids(r.truth_top(w, "sku", "units", 6, region="north", **SEP))
    spare = {k: w.on_hand(k, "south") - w.on_hand(k, "north") for k in n6}
    moved = max(n6, key=lambda k: (spare[k], k))
    units = spare[moved] // 2
    lead = {k: w.suppliers[w.skus[k].supplier].lead_time_days for k in n6}
    slow = max(n6, key=lambda k: (lead[k], k))
    out.append(Case(
        "long-flow-03", "long_flow", 3,
        (Turn(
            "Rebalance stock for the 6 SKUs that sold the most units in north stores in September "
            "2026. Read each one's stock per region. For the one whose south stock exceeds its "
            "north stock by the most, move half of that difference (rounded down) from the south "
            "to the north, then check the transfer's status. After that, find each of the six "
            "SKUs' supplier and its lead time in the supplier master. Give me: the SKU moved, the "
            "units, the transfer's id and in how many days it arrives, and which of the six SKUs "
            "has the longest supplier lead time, with its days.",
            ((_summary("sku", "units", top=6, region="north", **SEP),),
             tuple(("stock_level", {"sku": k}) for k in n6),
             (("place_transfer", {"sku": moved, "from_region": "south", "to_region": "north",
                                  "units": units}),),
             (("transfer_status", {"transfer": "TR-0001"}),),
             tuple(("sku_details", {"sku": k}) for k in n6),
             tuple(("supplier_info", {"supplier": s})
                   for s in sorted({w.skus[k].supplier for k in n6}))),
            record({"sku": moved, "units": units, "transfer": "TR-0001",
                    "arrives_in_days": transit_days("south", "north"), "longest": slow,
                    "lead_days": lead[slow]},
                   tol={"units": 0, "arrives_in_days": 0, "lead_days": 0},
                   forbidden=("TR-0002",)),
            sent=(Probe("asks", why="the transfer waits for a person, then goes on approved"),
                  *exactly("place_transfer"),
                  Probe("distinct", "sales_summary|sku_details|supplier_info|transfer_status",
                        "what was done before the pause is not done again")),
            approve=True,
        ),),
        ("an approval mid-run", "21 tool calls", "nothing repeated after the resume"),
        tools=(place_transfer, transfer_status),
    ))  # fmt: skip

    # 04: the worker dies mid-run; another finishes the run from its journal
    text, rounds, fields = _tender(3)
    quotes = rounds[-1]
    award = (("award_tender", {"tender": "TN-0001", "supplier": fields["supplier"]}),)
    out.append(Case(
        "long-flow-04", "long_flow", 3,
        (Turn(
            text,
            # the quotes are asked for again by the worker that takes the run over
            (*rounds, quotes, award),
            record(fields, tol={"per_pallet": 0.006, "total": 0.06}, forbidden=("TN-0002",)),
            sent=(Probe("restarts", why="the worker died mid-run and another finished the run"),
                  *exactly("open_tender"), *exactly("award_tender"),
                  Probe("called", "carrier_quote", "every lane was quoted", limit=len(LANES[3]))),
            restart=True,
        ),),
        ("a worker restart mid-run", "the write before the crash not done again"),
        tools=(open_tender, carrier_quote, award_tender),
    ))  # fmt: skip

    # 05: twelve turns; a later turn changes an earlier constraint
    n5 = _ids(r.truth_top(w, "sku", "units", 5, region="north", **Q3))
    avg = {k: r.truth_avg_daily_units(w, k, region="north", **Q3) for k in n5}
    north = {k: w.on_hand(k, "north") for k in n5}
    sup = {k: w.suppliers[w.skus[k].supplier] for k in n5}
    cost = {k: w.skus[k].unit_cost for k in n5}

    def plan(review: int) -> dict[str, int]:
        return {k: r.reorder_qty(avg[k], sup[k].lead_time_days, review, north[k]) for k in n5}

    def value(qty: dict[str, int]) -> float:
        return round(sum(qty[k] * cost[k] for k in qty), 2)

    # the review days are ones whose quantities are the same from the averages as fetched and
    # from the averages as an earlier turn showed them (2 decimals): either reading is right
    first, second = plan(REVIEW_DAYS[0]), plan(REVIEW_DAYS[1])
    least = min(n5, key=lambda k: (second[k], k))
    final = {k: v for k, v in second.items() if k != least}
    late = min(n5, key=lambda k: (sup[k].on_time_rate, k))
    north_q3 = {"region": "north", **Q3}
    stock = tuple(("stock_level", {"sku": k, "region": "north"}) for k in n5)
    details = tuple(("sku_details", {"sku": k}) for k in n5)
    masters = tuple(
        ("supplier_info", {"supplier": s}) for s in sorted({v.supplier for v in sup.values()})
    )
    units_n5 = (_summary("sku", "units", top=5, **north_q3),)
    again = (units_n5, (*stock, *details), masters)  # a turn that fetches what it needs anew
    out.append(Case(
        "long-flow-05", "long_flow", 3,
        (
            Turn("I'm planning a restock for the north. Which 5 SKUs sold the most units in north "
                 "stores in Q3 2026?", (units_n5,), cx.idset(n5)),
            Turn("What is each one's average daily units in the north over Q3 (units / 92, 2 "
                 "decimals)?", (units_n5,),
                 record({k: round(v, 2) for k, v in avg.items()}, tol=dict.fromkeys(avg, 0.006))),
            Turn("And how many units of each are on hand in the north now?", (stock,),
                 record(north, tol=dict.fromkeys(north, 0))),
            Turn("Who supplies each of the five, and what is each supplier's lead time in the "
                 "supplier master?", (details, masters),
                 record({k: v.supplier for k, v in sup.items()})),
            Turn("Use this: reorder quantity = average daily units x (lead time + "
                 f"{REVIEW_DAYS[0]} review days) - north on hand, rounded up, never below 0. What "
                 "is each SKU's reorder quantity?",
                 again, record(first, tol=dict.fromkeys(first, 0.5))),
            Turn("What does each of the five cost us per unit?", (details,),
                 record(cost, tol=dict.fromkeys(cost, 0.006))),
            Turn("So what is the order's total value (each reorder quantity x its unit cost, "
                 "summed)?", again, cx.money(value(first))),
            Turn("Which of the five SKUs' suppliers has the lowest on-time rate, and what is it?",
                 (details, masters),
                 record({"supplier": sup[late].supplier, "on_time": sup[late].on_time_rate},
                        tol={"on_time": 0.0051})),
            Turn(f"Change of plan: finance wants {REVIEW_DAYS[1]} review days, not "
                 f"{REVIEW_DAYS[0]}. Redo the five reorder quantities.", again,
                 record(second, tol=dict.fromkeys(second, 0.5))),
            Turn("And the order's total value now?", again, cx.money(value(second))),
            Turn("One more change: drop the SKU with the smallest reorder quantity from the order. "
                 "How many units are left in the order in all, and what is its total value?",
                 again,
                 record({"units": sum(final.values()), "value": value(final)},
                        tol={"units": 0.5, "value": max(0.5, value(final) * 0.001)})),
            Turn("Give me the final plan: each remaining SKU's reorder quantity.", again,
                 record(final, tol=dict.fromkeys(final, 0.5))),
        ),
        ("twelve turns", "an earlier constraint changed, the plan recomputed", "an item dropped"),
    ))  # fmt: skip

    # 06, 07: a tool that fails once in the middle of the job (a read, then a write)
    for n, group, edge in ((6, 0, "a read fails once (503)"), (7, 1, "a write fails once (busy)")):
        text, rounds, fields = _tender(group)
        award = (("award_tender", {"tender": "TN-0001", "supplier": fields["supplier"]}),)
        retried = (
            ((("carrier_quote", {"supplier": broken()["flaky"], "tender": "TN-0001"}),), award)
            if group == 0
            else (award, award)
        )
        out.append(Case(
            f"long-flow-{n:02d}", "long_flow", 3,
            (Turn(
                text, (*rounds, *retried),
                record(fields, tol={"per_pallet": 0.006, "total": 0.06}, forbidden=("TN-0002",)),
                sent=(*exactly("open_tender"), *exactly("award_tender"),
                      Probe("called", "carrier_quote", "every lane was quoted",
                            limit=len(LANES[group]))),
            ),),
            (edge, "tried again, nothing done twice"),
            tools=(open_tender, carrier_quote, award_tender),
        ))  # fmt: skip

    # 08: a tool that stays down; the answer says what was done and what was not
    down = broken()["down"]
    text, rounds, fields = _tender(2, without=down)
    out.append(Case(
        "long-flow-08", "long_flow", 3,
        (Turn(
            text + " If a lane cannot be quoted, award among the lanes that were quoted, and tell "
            "me every quote you got and which lane was not quoted.",
            (*rounds, (("carrier_quote", {"supplier": down, "tender": "TN-0001"}),),
             (("award_tender", {"tender": "TN-0001", "supplier": fields["supplier"]}),)),
            record({**fields, **{s: quote(s) for s in LANES[2] if s != down}, "not_quoted": down},
                   tol={"per_pallet": 0.006, "total": 0.06,
                        **{s: 0.006 for s in LANES[2] if s != down}},
                   forbidden=("TN-0002",)),
            sent=(*exactly("open_tender"), *exactly("award_tender"),
                  Probe("called", "carrier_quote", "every other lane was quoted",
                        limit=len(LANES[2]) - 1)),
        ),),
        ("a tool down for good", "what was done and what was not, said"),
        tools=(open_tender, carrier_quote, award_tender),
    ))  # fmt: skip

    # 09: a chain three tools deep for twenty items
    top20 = _ids(r.truth_top(w, "sku", "units", 20, **Q3))
    sups = sorted({w.skus[k].supplier for k in top20})
    gaps_ = {s: abs(w.suppliers[s].lead_time_days - w.suppliers[s].portal_lead_time_days)
             for s in sups}  # fmt: skip
    widest = max(sups, key=lambda s: (gaps_[s], s))
    mean_lead = statistics.fmean(w.suppliers[s].lead_time_days for s in sups)
    out.append(Case(
        "long-flow-09", "long_flow", 3,
        (Turn(
            "Take the 20 SKUs that sold the most units in Q3 2026 and find each one's supplier. "
            "For each distinct supplier, read its record in the supplier master and its quote on "
            "the supplier portal. How many distinct suppliers are there, what is their average "
            "master lead time (2 decimals), how many of them quote a different lead time on the "
            "portal, and which has the largest gap, with its days?",
            ((_summary("sku", "units", top=20, **Q3),),
             tuple(("sku_details", {"sku": k}) for k in top20),
             tuple(("supplier_info", {"supplier": s}) for s in sups),
             tuple(("supplier_portal", {"supplier": s}) for s in sups)),
            record({"suppliers": len(sups), "avg_lead": round(mean_lead, 2),
                    "differ": sum(1 for s in sups if gaps_[s]), "supplier": widest,
                    "gap_days": gaps_[widest]},
                   tol={"suppliers": 0, "avg_lead": 0.006, "differ": 0, "gap_days": 0}),
            sent=(Probe("called", "sku_details", "all 20 SKUs were read", limit=20),
                  Probe("distinct", why="no call is made twice"),
                  Probe("calls", why="the job ends within the step limit", limit=STEPS)),
        ),),
        (f"{21 + 2 * len(sups)} tool calls", "a chain three tools deep for twenty items"),
    ))  # fmt: skip
    return out
