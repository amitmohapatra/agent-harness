"""The ``complex`` suite's adversarial cases: what tries to break the harness before a user does —
conversations that twist (pronoun chains, corrections of corrections, contradictions, mixed
languages, blank and repeated messages, two people on one thread id, messages sent while a run
works), data that is wrong in every way a real source is (malformed CSV, wrong types, nulls,
unicode, duplicate keys, a 503 or a timeout mid-chain, pages that never end, exact money,
joins whose keys do not match), explanations after a failure or an interrupt, and helpers
fanned out by the twenty.

Each case is a :class:`~benchmark.cost.complex.Case` of the suite (category ``adv_*``; run them
alone with ``--cases adversarial``): its turns, the gold calls the scripted model replays, the
answer's ground truth (:class:`~benchmark.cost.complex.Expect`: strict, or a rubric for the
judge), and the harness's own checks (``Turn.sent``: what the model must and must not have been
sent; ``Turn.ends``: how the run may end). With ``--scripted`` the answer checks prove the
pipeline; the harness checks find harness bugs — a scripted model cannot hide what it was
sent. The judge packets carry both for the big-model pass.

The tools here (:data:`TOOLS`) are offered only to the cases that need them, beside the
retailer's: each is broken the way its docstring's first line says.
"""

from __future__ import annotations

import asyncio
import json
import math
from datetime import date, timedelta
from decimal import Decimal
from functools import cache
from typing import Any, Final

from benchmark.cost import retail as r
from trellis import tool

# --------------------------------------------------------------------------- broken sources
#: How long the slow inventory read takes: long enough for a second message to arrive.
SLOW_SECONDS: Final = 0.6


@tool(side_effects="read")
async def slow_stock(sku: str, region: str = "all") -> str:
    """Units on hand of a SKU, read from the inventory database (slow under load: it can take
    a while). Same answer as stock_level.

    Args:
        sku: the SKU id
        region: one region, or "all"
    """
    await asyncio.sleep(SLOW_SECONDS)
    return str(r.stock_level.fn(sku, region))


def returns_rows() -> list[tuple[str, str, int, str]]:
    """The well-formed rows of the returns export: (order, sku, units, reason)."""
    w = r.world()
    skus = list(w.skus)
    return [
        (
            f"R-{9000 + n}",
            skus[(n * 37) % len(skus)],
            1 + n % 4,
            ("size", "damaged", "colour")[n % 3],
        )
        for n in range(24)
    ]


@tool(side_effects="read")
def returns_export(region: str) -> str:
    """The returns team's CSV export for a region (order,sku,units,reason). Hand-edited in a
    spreadsheet: some lines are broken.

    Args:
        region: the region
    """
    rows = returns_rows()
    lines = ["order,sku,units,reason"]
    for n, (order, sku, units, reason) in enumerate(rows):
        lines.append(f"{order},{sku},{units},{reason}")
        if n == 5:
            lines.append('R-9105,SKU-00010,"2,damaged')  # an unclosed quote
        if n == 11:
            lines.append("order,sku,units,reason")  # the header pasted again
        if n == 17:
            lines.append("R-9117,SKU-00011")  # a ragged line
        if n == 20:
            lines.append("")
    lines.append("# exported by j.doe, totals not checked")
    return "\n".join(lines)


#: The on-time rates the vendor scorecard gives, as its owners typed them.
VENDOR_RATES: Final = (
    ("V-01", "Aldgate Knit", "0.91"),
    ("V-02", "Bursa Cotton", "93%"),
    ("V-03", "Cardiff Weave", 0.88),
    ("V-04", "Dhaka Stitch", "eighty"),
    ("V-05", "Essen Mills", None),
    ("V-06", "Faro Linen", 0.9),
)


@tool(side_effects="read")
def vendor_scorecard() -> str:
    """The procurement team's vendor scorecard (JSON): each vendor's on-time delivery rate, as
    typed by the vendor's account owner."""
    return json.dumps(
        [{"vendor": v, "name": n, "on_time": rate} for v, n, rate in VENDOR_RATES],
        ensure_ascii=False,
    )


#: Lead times the logistics sheet holds: some suppliers report none.
LEAD_TIMES: Final = (
    ("SUP-01", 14), ("SUP-02", None), ("SUP-03", 21), ("SUP-04", 9), ("SUP-05", None),
    ("SUP-06", 30), ("SUP-07", 12), ("SUP-08", None), ("SUP-09", 18), ("SUP-10", 25),
)  # fmt: skip


@tool(side_effects="read")
def logistics_sheet() -> str:
    """The logistics sheet (JSON): each supplier's reported shipping lead time in days (null
    where the supplier never reported one)."""
    return json.dumps([{"supplier": s, "lead_days": d} for s, d in LEAD_TIMES])


#: The vendor directory: names in their own scripts.
DIRECTORY: Final = (
    ("V-11", "Šťastný Textil s.r.o.", "Kraków", "zuzana@stastny.example"),
    ("V-12", "東京ニット株式会社", "Tokyo", "info@tokyo-knit.example"),
    ("V-13", "شركة القطن المصري", "Cairo", "sales@cotton-eg.example"),
    ("V-14", "Café́ Mills 🧶", "Lyon", "bonjour@cafemills.example"),
    ("V-15", "Ørsted & Søn ApS", "Aarhus", "hej@orsted.example"),
    ("V-16", "Zürich Strick‍waren AG", "Zürich", "kontakt@strick.example"),
)


@tool(side_effects="read")
def vendor_directory() -> str:
    """The vendor directory (JSON): each vendor's legal name, city and contact email."""
    return json.dumps(
        [{"vendor": v, "name": n, "city": c, "email": e} for v, n, c, e in DIRECTORY],
        ensure_ascii=False,
    )


#: What the bin counter sends for a bin: the same key twice, two counts.
BIN_COUNT: Final = '{"bin":"A-12","sku":"%s","on_hand":40,"counted_by":"scanner-3","on_hand":4}'


@tool(side_effects="read")
def bin_count(sku: str) -> str:
    """The warehouse bin counter's latest count of a SKU's bin (JSON, as the device sends it).

    Args:
        sku: the SKU id
    """
    return BIN_COUNT % sku.upper()


@tool(side_effects="read")
def inventory_service(sku: str) -> str:
    """The inventory microservice's live on-hand count of a SKU (all regions).

    Args:
        sku: the SKU id
    """
    raise RuntimeError("inventory service: 503 Service Unavailable (upstream overloaded)")


@tool(side_effects="read", call_timeout=0.2)
async def bin_scan(sku: str) -> str:
    """A live RFID scan of a SKU's bins (slow).

    Args:
        sku: the SKU id
    """
    await asyncio.sleep(5)
    return "{}"


#: Entries on each page of the audit log; it never says it is done.
PAGE_ENTRIES: Final = 5


@tool(side_effects="read")
def audit_log(page: int = 1) -> str:
    """The compliance audit log, one page at a time (newest first). Returns the page's entries
    and whether there are more pages.

    Args:
        page: the page number, from 1
    """
    entries = [
        {"entry": (page - 1) * PAGE_ENTRIES + n + 1, "event": "price override", "page": page}
        for n in range(PAGE_ENTRIES)
    ]
    return json.dumps({"page": page, "entries": entries, "has_more": True, "next_page": page + 1})


def ledger_rows() -> list[tuple[str, str]]:
    """The September refund ledger: (refund id, amount as written, two decimals). Large."""
    amounts = ("0.10", "0.20", "0.70", "19.99", "4.35", "0.01", "12.30")
    return [(f"RF-{n:05d}", amounts[(n * 5) % len(amounts)]) for n in range(6_000)]


def ledger_total() -> Decimal:
    return sum((Decimal(a) for _, a in ledger_rows()), Decimal("0"))


@tool(side_effects="read")
def refund_ledger(month: str) -> str:
    """Every refund of a month, as CSV (refund,amount_gbp). Large: thousands of lines.

    Args:
        month: YYYY-MM
    """
    return "\n".join(["refund,amount_gbp", *(f"{i},{a}" for i, a in ledger_rows())])


#: A transfer whose id is past 64 bits (the bank's ids are 20-digit numbers).
BIG_ID: Final = 18_446_744_073_709_551_621


def transfer_rows() -> list[tuple[int, str, str]]:
    rows = [
        (10_000_000_000_000_000_000 + n * 7_919, f"{(n * 13) % 997}.{n % 100:02d}", "ok")
        for n in range(3_000)
    ]
    rows[1_234] = (BIG_ID, "9120.55", "held")
    return rows


@tool(side_effects="read")
def bank_transfers(month: str) -> str:
    """Every outgoing bank transfer of a month (JSON array: id, amount_gbp, status). Large.

    Args:
        month: YYYY-MM
    """
    return json.dumps(
        [{"id": i, "amount_gbp": float(a), "status": s} for i, a, s in transfer_rows()]
    )


@tool(side_effects="read")
def store_targets(quarter: str) -> str:
    """Each store's revenue target for a quarter (GBP), from the planning tool (JSON). The
    planning tool names stores its own way.

    Args:
        quarter: e.g. 2026-Q3
    """
    w = r.world()
    rows = []
    for n, s in enumerate(w.stores):
        actual = r.truth_total(w, "revenue", start="2026-07-01", end="2026-09-30", store=s.store)
        rows.append(
            {
                "store_id": f"store-{s.store[1:]}",
                "target_gbp": round(actual * (0.9 if n % 3 else 1.1), -2),
            }
        )
    return json.dumps({"quarter": quarter, "targets": rows})


def beat_target(region: str) -> list[str]:
    """The stores of ``region`` whose Q3 revenue beat the planning tool's target."""
    w = r.world()
    out = []
    for n, s in enumerate(w.stores):
        actual = r.truth_total(w, "revenue", start="2026-07-01", end="2026-09-30", store=s.store)
        target = round(actual * (0.9 if n % 3 else 1.1), -2)
        if s.region == region and actual > target:
            out.append(s.store)
    return out


#: The size of the system event log of one day (lines): one of them is the failure.
EVENT_LINES: Final = 60_000
FAILED_ORDER: Final = "O-88412"


@tool(side_effects="read")
def event_log(day: str) -> str:
    """The order system's event log of one day, one line per event. Very large.

    Args:
        day: YYYY-MM-DD
    """
    lines = []
    for n in range(EVENT_LINES):
        if n == 41_207:
            lines.append(
                f"{day}T14:02:11Z ERROR order {FAILED_ORDER} payment declined: card expired"
            )
        else:
            lines.append(
                f"{day}T{(n // 3600) % 24:02d}:{(n // 60) % 60:02d}:{n % 60:02d}Z INFO order O-{70000 + n} ok"
            )
    return "\n".join(lines)


# --------------------------------------------------------------------------- scenario jobs
#: The quarter the scenario jobs plan from, and its days.
Q3: Final = ("2026-07-01", "2026-09-30")
Q3_DAYS: Final = 92
#: Scenario C's five new SKUs, as the question states them: units a quarter, price, unit cost,
#: units stocked, supplier lead time.
NEW_SKU: Final = {"units": 300, "price": 40.0, "cost": 18.0, "stock": 400, "lead": 30}
#: The price what-if's elasticity of units to price, as the question states it.
ELASTICITY: Final = -1.8


@cache
def assortment_rows() -> tuple[dict[str, Any], ...]:
    """One row per SKU, computed from the 50,000 sales lines, the catalogue, the stock and the
    supplier master: what ``assortment_data`` returns (its rounding is the truth's)."""
    w = r.world()
    units: dict[str, int] = dict.fromkeys(w.skus, 0)
    revenue: dict[str, float] = dict.fromkeys(w.skus, 0.0)
    for sale in r.selected(w, start=Q3[0], end=Q3[1]):
        units[sale.sku] += sale.units
        revenue[sale.sku] += sale.revenue
    return tuple(
        {
            "sku": k.sku,
            "category": k.category,
            "unit_cost": k.unit_cost,
            "price": k.price,
            "units_q3": units[k.sku],
            "revenue_q3": round(revenue[k.sku], 2),
            "on_hand": sum(w.on_hand(k.sku, g) for g in r.REGIONS),
            "lead_time_days": w.suppliers[k.supplier].lead_time_days,
            "avg_daily_units": round(units[k.sku] / Q3_DAYS, 4),
        }
        for k in w.skus.values()
    )


@tool(side_effects="read")
async def assortment_data() -> str:
    """Every SKU's Q3 2026 planning row, built from the 50,000 sales lines (slow: it scans them
    all), as CSV: sku, category, unit_cost, price, units_q3, revenue_q3 (GBP), on_hand (all
    regions, now), lead_time_days (supplier master), avg_daily_units (Q3 units / 92)."""
    await asyncio.sleep(SLOW_SECONDS)
    rows = assortment_rows()
    head = list(rows[0])
    return "\n".join([",".join(head), *(",".join(str(row[c]) for c in head) for row in rows)])


def _metrics(
    rows: list[dict[str, Any]], extra: int = 0, price: float | None = None
) -> dict[str, float]:
    """Gross margin (Q3 revenue less units x unit cost), sell-through % (units sold over sold
    plus on hand), SKUs at stock-out risk (on hand under lead-time demand), cash tied up (on
    hand x unit cost) and margin per pound of cash — ``extra`` new SKUs added as stated."""
    n = NEW_SKU | ({"price": price} if price is not None else {})
    margin = sum(x["revenue_q3"] - x["units_q3"] * x["unit_cost"] for x in rows)
    margin += extra * (n["units"] * n["price"] - n["units"] * n["cost"])
    sold = sum(x["units_q3"] for x in rows) + extra * n["units"]
    held = sum(x["on_hand"] for x in rows) + extra * n["stock"]
    risk = sum(x["on_hand"] < x["avg_daily_units"] * x["lead_time_days"] for x in rows)
    risk += extra * (n["stock"] < n["units"] / Q3_DAYS * n["lead"])
    cash = sum(x["on_hand"] * x["unit_cost"] for x in rows) + extra * n["stock"] * n["cost"]
    return {
        "margin": round(margin, 2),
        "sell_through": round(100.0 * sold / (sold + held), 2),
        "at_risk": risk,
        "cash": round(cash, 2),
        "per_cash": round(margin / cash, 4),
    }


def assortment(scenario: str, price: float | None = None) -> dict[str, float]:
    """Scenario ``keep``, ``cut`` (the bottom 10% of SKUs by Q3 revenue dropped) or ``add``
    (five new SKUs, :data:`NEW_SKU`; at ``price`` when given)."""
    rows = list(assortment_rows())
    if scenario == "cut":
        bottom = {
            x["sku"]
            for x in sorted(rows, key=lambda x: (x["revenue_q3"], x["sku"]))[: len(rows) // 10]
        }
        rows = [x for x in rows if x["sku"] not in bottom]
    return _metrics(rows, extra=5 if scenario == "add" else 0, price=price)


def price_cut_on_cut() -> dict[str, float]:
    """Scenario ``cut`` with a 5% price cut: units up by 1.8 x 5% (:data:`ELASTICITY`), revenue
    at 95% of the price; stock as it is."""
    keep = assortment_rows()
    bottom = {
        x["sku"] for x in sorted(keep, key=lambda x: (x["revenue_q3"], x["sku"]))[: len(keep) // 10]
    }
    lift = 1 + ELASTICITY * -0.05
    rows = [
        dict(x, units_q3=x["units_q3"] * lift, revenue_q3=x["revenue_q3"] * 0.95 * lift)
        for x in keep
        if x["sku"] not in bottom
    ]
    return _metrics(rows)


def replenishment(review: int, order_cost: float = 40) -> dict[str, float]:
    """The 50 fastest SKUs reordered with a ``review``-day review period: each order brings a
    SKU to its demand over lead time plus review (rounded up); ordering costs £40 per SKU
    ordered per order, ``92 / review`` orders a quarter; holding costs 25% a year of the
    average stock value (on hand plus half the order) over the quarter."""
    top = sorted(assortment_rows(), key=lambda x: (-x["units_q3"], x["sku"]))[:50]
    qty = [
        max(0, math.ceil(x["avg_daily_units"] * (x["lead_time_days"] + review) - x["on_hand"]))
        for x in top
    ]
    ordering = order_cost * sum(q > 0 for q in qty) * Q3_DAYS / review
    holding = (
        sum((x["on_hand"] + q / 2) * x["unit_cost"] for x, q in zip(top, qty, strict=True))
        * 0.25
        * Q3_DAYS
        / 365
    )
    return {"units": sum(qty), "ordering": round(ordering, 2), "holding": round(holding, 2),
            "total": round(ordering + holding, 2)}  # fmt: skip


def price_change(
    change: float, category: str = "knitwear", elasticity: float = ELASTICITY
) -> dict[str, float]:
    """A category's Q3 margin had its prices moved by ``change`` (units moved by
    :data:`ELASTICITY` x ``change``)."""
    lift = 1 + elasticity * change
    rows = [x for x in assortment_rows() if x["category"] == category]
    revenue = sum(x["revenue_q3"] * (1 + change) * lift for x in rows)
    margin = revenue - sum(x["units_q3"] * lift * x["unit_cost"] for x in rows)
    return {"revenue": round(revenue, 2), "margin": round(margin, 2)}


def allocation(sku: str, units: int = 1_200) -> dict[str, dict[str, float]]:
    """``units`` of ``sku`` spread over the regions three ways — ``equal``, ``by_sales`` (Q3
    units' share, rounded down, the rest to the biggest seller) and ``lowest_cover`` (all to the
    region with the fewest days of cover) — each plan's regions under 14 days of cover and its
    lowest cover (days), after the allocation."""
    w = r.world()
    sold = {
        g: sum(s.units for s in r.selected(w, sku=sku, region=g, start=Q3[0], end=Q3[1]))
        for g in r.REGIONS
    }
    held = {g: w.on_hand(sku, g) for g in r.REGIONS}
    daily = {g: sold[g] / Q3_DAYS for g in r.REGIONS}
    cover = {g: held[g] / daily[g] for g in r.REGIONS}
    total = sum(sold.values())
    share = {g: units * sold[g] // total for g in r.REGIONS}
    share[max(r.REGIONS, key=lambda g: sold[g])] += units - sum(share.values())
    plans = {
        "equal": dict.fromkeys(r.REGIONS, units // 4),
        "by_sales": share,
        "lowest_cover": {
            g: (units if g == min(r.REGIONS, key=lambda g: cover[g]) else 0) for g in r.REGIONS
        },
    }
    out = {}
    for name, plan in plans.items():
        after = {g: (held[g] + plan[g]) / daily[g] for g in r.REGIONS}
        out[name] = {
            "under_14": sum(c < 14 for c in after.values()),
            "lowest": round(min(after.values()), 1),
        }
    return out


TOOLS: Final = (
    slow_stock, returns_export, vendor_scorecard, logistics_sheet, vendor_directory, bin_count,
    inventory_service, bin_scan, audit_log, refund_ledger, bank_transfers, store_targets,
    event_log, assortment_data,
)  # fmt: skip


# --------------------------------------------------------------------------- files a run makes
#: The suite's file store: each person's files by name — ``(text, as_of)`` — so a run finds
#: only its own person's (the harness has no file store of its own yet: #91). Emptied for
#: every case (``complex.run_case``).
FILES: Final[dict[str, dict[str, tuple[str, str]]]] = {}


def _mine() -> dict[str, tuple[str, str]]:
    from trellis import current

    runtime = current()
    assert runtime is not None
    return FILES.setdefault(runtime.user, {})


@tool(side_effects="write")
def export_stock_report(skus: list[str]) -> str:
    """Export a stock report as a CSV file (sku,on_hand,as_of) for the person asking: what is
    on hand now (all regions) of each SKU. Returns the file's name.

    Args:
        skus: the SKU ids
    """
    w = r.world()
    mine = _mine()
    name = f"stock-report-{len(mine) + 1}.csv"
    rows = [f"{k},{sum(w.on_hand(k, g) for g in r.REGIONS)},{w.as_of()}" for k in skus]
    mine[name] = ("\n".join(["sku,on_hand,as_of", *rows]), w.as_of())
    return json.dumps({"file": name, "rows": len(rows), "as_of": w.as_of()})


@tool(side_effects="read")
def list_files() -> str:
    """The files made for the person asking: name and when each was made (as_of)."""
    return json.dumps([{"file": n, "as_of": a} for n, (_, a) in _mine().items()])


@tool(side_effects="read")
def open_file(name: str) -> str:
    """A file made for the person asking, whole.

    Args:
        name: the file's name, as list_files gives it
    """
    found = _mine().get(name)
    return (
        found[0]
        if found
        else json.dumps({"error": f"no file {name} (it does not exist or was deleted)"})
    )


@tool(side_effects="write")
def delete_file(name: str) -> str:
    """Delete a file made for the person asking.

    Args:
        name: the file's name
    """
    return json.dumps({"deleted": _mine().pop(name, None) is not None, "file": name})


FILE_TOOLS: Final = (export_stock_report, list_files, open_file, delete_file)


# --------------------------------------------------------------------------- allocation pipeline
#: The allocation pipeline's stages, in order, and the rule each applies (its rule text).
STAGES: Final = ("forecast", "preprocess", "optimiser", "postprocess", "final")
RULES: Final = {
    "R1": "forecast: 4-week demand from the store's last 12 weeks of sales of the SKU",
    "R2": "preprocess: add 15% safety stock for the region's top-20 SKUs, rounded down",
    "R3": "optimiser: shift units toward regions with higher sell-through (weight = region sell-through / chain average)",
    "R4": "postprocess: cap at the store's shelf space for the SKU",
    "R5": "final: the postprocess output, rounded to whole cases of 1",
}
PIPE_STORES: Final = tuple(f"ST-{n:02d}" for n in range(1, 49))
PIPE_SKU: Final = "SKU-00012"


def pipe_region(store: str) -> str:
    return r.REGIONS[(int(store[3:]) - 1) % 4]


@cache
def pipeline_records(run_id: str) -> tuple[dict[str, Any], ...]:
    """Every stage's record of ``run_id`` for every store (one SKU): units in and out, the rule
    and when it was logged. Store ST-42 of AL-1 is the worked example (96 -> 110 -> 135 -> 120)."""
    out = []
    salt = 7 if run_id == "AL-2" else 0
    for n, store in enumerate(PIPE_STORES, start=1):
        region = pipe_region(store)
        forecast = 60 + (n * 13 + salt) % 50
        pre = int(forecast * 1.15) if n % 3 == 0 else forecast
        weight = {"north": 1.25, "south": 0.9, "east": 1.0, "west": 0.95}[region]
        opt = round(pre * weight)
        shelf = 120 if n % 5 == 2 else 400
        post = min(opt, shelf)
        if run_id == "AL-1" and store == "ST-42":
            forecast, pre, opt, post = 96, 110, 135, 120
        units = [forecast, pre, opt, post, post]
        for k, stage in enumerate(STAGES):
            out.append({
                "run_id": run_id, "store": store, "region": region, "sku": PIPE_SKU, "stage": stage,
                "units_in": units[k - 1] if k else 0, "units_out": units[k], "rule": f"R{k + 1}",
                "logged_at": f"2026-10-{29 if run_id == 'AL-1' else 30}T10:{k * 7 + 1:02d}:{n:02d}Z",
            })  # fmt: skip
    return tuple(out)


def waterfall(run_id: str, store: str) -> dict[str, int]:
    rows = {x["stage"]: x for x in pipeline_records(run_id) if x["store"] == store}
    return {st: rows[st]["units_out"] - rows[st]["units_in"] for st in STAGES}


@tool(side_effects="read")
def allocation_log(run_id: str, stage: str, store: str | None = None) -> str:
    """The allocation pipeline's log of one stage of a run (JSON records: run_id, store,
    region, sku, stage, units_in, units_out, rule id, logged_at) — one store's, or every
    store's. Stages: forecast, preprocess, optimiser, postprocess, final. Run AL-2's
    postprocess stage logs plain text only (allocation_text_log).

    Args:
        run_id: the allocation run, e.g. AL-1
        stage: the stage
        store: one store, e.g. ST-42 (all stores when left out)
    """
    if run_id == "AL-2" and stage == "preprocess":
        return json.dumps({"error": "no preprocess log for run AL-2 (the stage's logging was off)"})
    if run_id == "AL-2" and stage == "postprocess":
        return json.dumps(
            {
                "error": "run AL-2's postprocess stage logged plain text only: see allocation_text_log"
            }
        )
    rows = [
        x
        for x in pipeline_records(run_id)
        if x["stage"] == stage and (store is None or x["store"] == store)
    ]
    return (
        json.dumps(rows)
        if rows
        else json.dumps({"error": f"no {stage} records for {run_id} {store or ''}".strip()})
    )


@tool(side_effects="read")
def allocation_text_log(run_id: str) -> str:
    """A run's postprocess stage as its plain-text log lines (one line per store and SKU, among
    the stage's other messages). Large.

    Args:
        run_id: the allocation run
    """
    lines = []
    for x in (x for x in pipeline_records(run_id) if x["stage"] == "postprocess"):
        for k in range(40):
            lines.append(
                f"{x['logged_at'][:-3]}:{k:02d}Z DEBUG postprocess heartbeat worker={k % 4}"
            )
        lines.append(
            f"{x['logged_at']} INFO postprocess {x['store']} {x['sku']} in={x['units_in']} out={x['units_out']} rule={x['rule']}"
        )
    return "\n".join(lines)


@tool(side_effects="read")
def allocation_rules() -> str:
    """The allocation pipeline's rule texts, by rule id (R1..R5)."""
    return json.dumps(RULES)


PIPELINE_TOOLS: Final = (allocation_log, allocation_text_log, allocation_rules)


# --------------------------------------------------------------------------- allocation domain
#: The allocation domain: 14 daily runs (AL-01 .. AL-14, today the last), 48 stores, 2 SKUs.
#: Its input and output tables answer across runs (``allocation_sql``); each run's stage logs,
#: in a bucket, answer why one run did what it did (``list_logs`` / ``read_log``) — some
#: stages log JSON lines, one plain text, one run's optimiser CSV, and two runs' preprocess
#: nothing; the rules are a document (``rules_doc``).
DOMAIN_RUNS: Final = tuple(f"AL-{d + 1:02d}" for d in range(14))
DOMAIN_SKUS: Final = ("SKU-00009", "SKU-00012")
DOMAIN_RULES: Final = {
    **RULES,
    "R6": "optimiser: no store gets more than 150 units of one SKU in one run (a cap per store and SKU)",
}
#: The runs whose preprocess stage logged nothing.
NO_PREPROCESS_LOG: Final = ("AL-07", "AL-13")
OPT_CAP: Final = 150
WEIGHTS: Final = {"north": 1.25, "south": 0.9, "east": 1.0, "west": 0.95}


def run_date(run_id: str) -> str:
    return (date(2026, 10, 17) + timedelta(days=DOMAIN_RUNS.index(run_id))).isoformat()


def stages_of(forecast: int, n: int, region: str) -> dict[str, int]:
    """One store-SKU through the stages: each stage's units out."""
    pre = int(forecast * 1.15) if n % 3 == 0 else forecast
    opt = min(round(pre * WEIGHTS[region]), OPT_CAP)
    post = min(opt, 120 if n % 5 == 2 else 400)
    return {
        "forecast": forecast,
        "preprocess": pre,
        "optimiser": opt,
        "postprocess": post,
        "final": post,
    }


@cache
def domain_rows() -> tuple[dict[str, Any], ...]:
    """Every run x store x SKU: its inputs (forecast, demand, shelf space) and each stage's
    units (the truth both the tables and the logs are written from)."""
    out = []
    for d, run in enumerate(DOMAIN_RUNS):
        for n in range(1, 49):
            store, region = f"ST-{n:02d}", r.REGIONS[(n - 1) % 4]
            for sku in DOMAIN_SKUS:
                base = 60 + (n * 13 + d * 5 + (7 if sku == "SKU-00009" else 0)) % 50
                base = round(base * 1.6) if sku == "SKU-00009" else base
                forecast = round(base * 1.35) if d == 13 and region == "north" else base
                demand = round(forecast * (1.1 if n % 7 in (0, 3) else 0.95))
                out.append({
                    "run_id": run, "run_date": run_date(run), "store": store, "region": region,
                    "sku": sku, "forecast": forecast, "demand": demand,
                    "shelf_space": 120 if n % 5 == 2 else 400,
                    "stages": stages_of(forecast, n, region),
                })  # fmt: skip
    return tuple(out)


def _domain_db() -> Any:
    import sqlite3

    db = sqlite3.connect(":memory:")
    db.execute(
        "CREATE TABLE inputs (run_id TEXT, run_date TEXT, store TEXT, region TEXT, sku TEXT, "
        "forecast INTEGER, demand INTEGER, shelf_space INTEGER)"
    )
    db.execute("CREATE TABLE outputs (run_id TEXT, store TEXT, sku TEXT, final_units INTEGER)")
    rows = domain_rows()
    db.executemany(
        "INSERT INTO inputs VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                x["run_id"],
                x["run_date"],
                x["store"],
                x["region"],
                x["sku"],
                x["forecast"],
                x["demand"],
                x["shelf_space"],
            )
            for x in rows
        ],
    )
    db.executemany(
        "INSERT INTO outputs VALUES (?, ?, ?, ?)",
        [(x["run_id"], x["store"], x["sku"], x["stages"]["final"]) for x in rows],
    )
    db.execute("PRAGMA query_only = ON")
    return db


@tool(side_effects="read")
def allocation_sql(sql: str) -> str:
    """Run one read-only SQL query (SQLite) over the allocation tables of every run: inputs
    (run_id, run_date, store, region, sku, forecast, demand, shelf_space) and outputs (run_id,
    store, sku, final_units). Runs AL-01..AL-14 are daily, 2026-10-17 to 2026-10-30 (today).
    Returns CSV.

    Args:
        sql: one SELECT
    """
    import sqlite3

    db = _domain_db()
    try:
        cursor = db.execute(sql)
        head = [c[0] for c in cursor.description or ()]
        found = cursor.fetchmany(500)
    except sqlite3.Error as exc:
        return json.dumps({"error": f"{exc}"})
    finally:
        db.close()
    return "\n".join([",".join(head), *(",".join(str(v) for v in row) for row in found)])


def _log_text(run_id: str, stage: str) -> str | None:
    if stage == "preprocess" and run_id in NO_PREPROCESS_LOG:
        return None
    rows = [x for x in domain_rows() if x["run_id"] == run_id]
    k = STAGES.index(stage)
    lines = []
    for x in rows:
        st = x["stages"]
        before, after = (st[STAGES[k - 1]] if k else 0), st[stage]
        rule = (
            "R6"
            if stage == "optimiser"
            and after == OPT_CAP
            and round(st["preprocess"] * WEIGHTS[x["region"]]) > OPT_CAP
            else f"R{k + 1}"
        )
        if stage == "postprocess":
            lines.append(
                f"{run_date(run_id)}T06:{k * 7:02d}:00Z INFO postprocess {x['store']} {x['sku']} in={before} out={after} rule={rule}"
            )
        elif stage == "optimiser" and run_id == "AL-14":
            lines.append(f"{x['store']},{x['sku']},{before},{after},{rule}")
        else:
            lines.append(json.dumps({"ts": f"{run_date(run_id)}T06:{k * 7:02d}:00Z", "store": x["store"], "sku": x["sku"],
                                     "in": before, "out": after, "rule": rule}))  # fmt: skip
    if stage == "optimiser" and run_id == "AL-14":
        lines.insert(0, "store,sku,in,out,rule")
    return "\n".join(lines)


def _log_path(run_id: str, stage: str) -> str:
    ext = (
        "txt"
        if stage == "postprocess"
        else ("csv" if stage == "optimiser" and run_id == "AL-14" else "jsonl")
    )
    return f"gs://alloc-logs/{run_id}/{stage}.{ext}"


@tool(side_effects="read")
def list_logs(run_id: str) -> str:
    """The stage logs one allocation run left in the log bucket (paths, read with read_log).

    Args:
        run_id: e.g. AL-13
    """
    return json.dumps([_log_path(run_id, st) for st in STAGES if _log_text(run_id, st) is not None])


@tool(side_effects="read")
def read_log(path: str) -> str:
    """One log file from the log bucket, whole (its shape is the stage's own: JSON lines, CSV
    or plain text).

    Args:
        path: a path list_logs gave, e.g. gs://alloc-logs/AL-13/optimiser.jsonl
    """
    parts = path.removeprefix("gs://alloc-logs/").split("/")
    if len(parts) == 2 and parts[0] in DOMAIN_RUNS:
        stage = parts[1].split(".")[0]
        if stage in STAGES and path == _log_path(parts[0], stage):
            text = _log_text(parts[0], stage)
            if text is not None:
                return text
    return json.dumps({"error": f"no such log: {path}"})


@tool(side_effects="read")
def rules_doc() -> str:
    """The allocation rules document: each rule's id and text."""
    return json.dumps(DOMAIN_RULES)


DOMAIN_TOOLS: Final = (allocation_sql, list_logs, read_log, rules_doc)


@tool(side_effects="irreversible")
def apply_assortment(scenario: str) -> str:
    """Apply an assortment plan to the live range (delists or lists SKUs: cannot be undone).

    Args:
        scenario: keep, cut or add
    """
    return f"applied the {scenario} plan"


#: The metrics every assortment query returns, as SQL over ``assortment_data``'s table.
_METRICS: Final = (
    "round(sum(revenue_q3 - units_q3*unit_cost){m}, 2) AS margin, "
    "round(100.0*(sum(units_q3){u})/(sum(units_q3){u} + sum(on_hand){h}), 2) AS sell_through, "
    "sum(on_hand < avg_daily_units*lead_time_days) AS at_risk, "
    "round(sum(on_hand*unit_cost){c}, 2) AS cash, "
    "round((sum(revenue_q3 - units_q3*unit_cost){m})/(sum(on_hand*unit_cost){c}), 4) AS per_cash"
)
BOTTOM: Final = "sku NOT IN (SELECT sku FROM result ORDER BY revenue_q3, sku LIMIT 200)"


def assortment_sql(scenario: str, price: float | None = None) -> str:
    n = NEW_SKU | ({"price": price} if price is not None else {})
    if scenario == "add":
        extra = {"m": f" + {5 * n['units'] * (n['price'] - n['cost']):.1f}", "u": f" + {5 * n['units']}",
                 "h": f" + {5 * n['stock']}", "c": f" + {5 * n['stock'] * n['cost']:.1f}"}  # fmt: skip
    else:
        extra = {"m": "", "u": "", "h": "", "c": ""}
    where = f" WHERE {BOTTOM}" if scenario == "cut" else ""
    return f"SELECT {_METRICS.format(**extra)} FROM result{where}"


def replenishment_sql(review: int, order_cost: float = 40) -> str:
    need = f"ceil(avg_daily_units*(lead_time_days+{review}) - on_hand)"
    return (
        "WITH top AS (SELECT * FROM result ORDER BY units_q3 DESC, sku LIMIT 50), "
        # never below 0: a CASE, as both engines write it (DuckDB has no two-argument max)
        f"q AS (SELECT *, CASE WHEN {need} > 0 THEN {need} ELSE 0 END AS qty FROM top) "
        f"SELECT sum(qty) AS units, round({order_cost}*sum(qty > 0)*92.0/{review}, 2) AS ordering, "
        "round(sum((on_hand + qty/2.0)*unit_cost)*0.25*92.0/365, 2) AS holding FROM q"
    )


def price_sql(change: float, elasticity: float = ELASTICITY) -> str:
    lift = f"(1 + {elasticity}*{change})"
    return (
        f"SELECT round(sum(revenue_q3*(1 + {change})*{lift}), 2) AS revenue, "
        f"round(sum(revenue_q3*(1 + {change})*{lift}) - sum(units_q3*{lift}*unit_cost), 2) AS margin "
        "FROM result WHERE category = 'knitwear'"
    )


# --------------------------------------------------------------------------- the cases
def cases() -> tuple[Any, ...]:
    """Every adversarial case, built from the world's data (``complex.cases`` adds them)."""
    from benchmark.cost.complex import (
        Case,
        Expect,
        Probe,
        Turn,
        abstain,
        cannot,
        count,
        flags,
        fmt,
        idset,
        judge,
        money,
        pct,
        record,
    )

    w = r.world()
    out: list[Case] = []
    add = out.append
    q3 = {"start": "2026-07-01", "end": "2026-09-30"}

    def total_on(sku: str) -> int:
        return sum(w.on_hand(sku, g) for g in r.REGIONS)

    def margin(sku: str) -> float:
        k = w.skus[sku]
        return r.margin_pct(k.price, k.unit_cost)

    def stores_in(region: str) -> int:
        return sum(s.region == region for s in w.stores)

    # what every turn's request must hold for a provider to take it
    nonblank = Probe("nonblank", why="a blank user message is refused by the provider")

    # ------------------------------------------------------------------ conversation
    parka = w.by_name("Waxed Parka Olive")
    sup = w.suppliers[parka.supplier]
    find = ("find_sku", {"query": parka.name})
    chain = (
        ("Which supplier makes the Waxed Parka Olive?", ((find,), (("sku_details", {"sku": parka.sku}),)),
         record({"supplier": (sup.supplier, sup.name)})),
        ("Where are they based?", ((("supplier_info", {"supplier": sup.supplier}),),),
         record({"city": sup.city})),
        ("And their lead time?", (), record({"lead_time_days": sup.lead_time_days}, tol={"lead_time_days": 0})),
        ("Minimum order?", (), record({"min_order_qty": sup.min_order_qty}, tol={"min_order_qty": 0})),
        ("How reliable are they on delivery?", (), record({"on_time_rate": sup.on_time_rate}, tol={"on_time_rate": 0.005})),
        ("What does that parka sell for now?", (), money(parka.price)),
        ("And what does it cost us?", (), money(parka.unit_cost)),
        ("So what's its gross margin %?", (), pct(margin(parka.sku))),
        ("How many of it are in the north?", ((("stock_level", {"sku": parka.sku, "region": "north"}),),),
         count(w.on_hand(parka.sku, "north"))),
        ("And in total?", ((("stock_level", {"sku": parka.sku}),),), count(total_on(parka.sku))),
        ("Who did you say makes it, again?", (), record({"supplier": (sup.supplier, sup.name)})),
        ("And how many days do they take to deliver?", (),
         record({"lead_time_days": sup.lead_time_days}, tol={"lead_time_days": 0})),
    )  # fmt: skip
    add(
        Case(
            "advconv-01", "adv_conversation", 3,
            tuple(
                Turn(text, calls, e, sent=(Probe("has", sup.supplier, "the antecedent of 'they' is in the context"),) if n == len(chain) - 1 else ())
                for n, (text, calls, e) in enumerate(chain)
            ),
            ("pronoun chain over 12 turns",),
        )
    )  # fmt: skip

    knit = {"group_by": "category", "metric": "revenue", "category": "knitwear"}
    jul, aug = (
        {"start": "2026-07-01", "end": "2026-07-31"},
        {"start": "2026-08-01", "end": "2026-08-31"},
    )
    t_jul = r.truth_total(w, "revenue", category="knitwear", **jul)
    t_aug = r.truth_total(w, "revenue", category="knitwear", **aug)
    t_jul_n = r.truth_total(w, "revenue", category="knitwear", region="north", **jul)
    add(
        Case(
            "advconv-02", "adv_conversation", 2,
            (
                Turn("Total knitwear revenue in July 2026?", ((("sales_summary", knit | jul),),), money(t_jul)),
                Turn("Sorry, I meant August.", ((("sales_summary", knit | aug),),),
                     money(t_aug, forbidden=(fmt(t_jul),))),
                Turn("No wait — July was right after all, but only the north region.",
                     ((("sales_summary", knit | jul | {"region": "north"}),),),
                     money(t_jul_n, forbidden=(fmt(t_jul), fmt(t_aug)))),
            ),
            ("correction of a correction",),
        )
    )  # fmt: skip

    north_q3 = r.truth_total(w, "revenue", region="north", **q3)
    n_north = stores_in("north")
    add(
        Case(
            "advconv-03", "adv_conversation", 2,
            (
                Turn(
                    "We have 12 stores in the north, right? What was the north's Q3 2026 revenue per store?",
                    ((("store_list", {"region": "north"}),
                      ("sales_summary", {"group_by": "region", "metric": "revenue", "region": "north", **q3})),),
                    record({"stores": n_north, "per_store": north_q3 / n_north}, tol={"stores": 0},
                           forbidden=(fmt(north_q3 / 12),)),
                ),
                Turn(
                    "Hmm — I said 12 just now, and last week I'd have told you 9. Which is it, and does it change the per-store figure?",
                    (),
                    judge(
                        f"says the store list has {n_north} north stores — neither 12 nor 9 — and that the "
                        f"per-store figure stays £{fmt(north_q3 / n_north)}; does not adopt the user's numbers",
                        f"The store list has {n_north} stores in the north, not 12 or 9, so £{fmt(north_q3 / n_north)} per store stands.",
                    ),
                ),
            ),
            ("contradictory user statements",),
        )
    )  # fmt: skip

    coat = w.by_name("Insulated Bomber Camel")
    sup2 = w.suppliers["SUP-02"]
    add(
        Case(
            "advconv-04", "adv_conversation", 2,
            (
                Turn(f"How many {coat.name} do we have in total?",
                     ((("find_sku", {"query": coat.name}),), (("stock_level", {"sku": coat.sku}),)),
                     count(total_on(coat.sku))),
                Turn("Different topic: which city is SUP-02 in?",
                     ((("supplier_info", {"supplier": "SUP-02"}),),), record({"city": sup2.city})),
                Turn("And another: how many stores are in the east?",
                     ((("store_list", {"region": "east"}),),), count(stores_in("east"))),
                Turn("Back to the jacket — what's its gross margin %?",
                     ((("sku_details", {"sku": coat.sku}),),), pct(margin(coat.sku))),
                Turn("And SUP-02 — what's their minimum order?", (),
                     record({"min_order_qty": sup2.min_order_qty}, tol={"min_order_qty": 0})),
            ),
            ("topic switch twice, returned to twice",),
        )
    )  # fmt: skip

    west_q1 = r.truth_total(w, "revenue", region="west", start="2026-01-01", end="2026-03-31")
    filler = []
    for category in list(r.CATEGORIES)[:6] * 2:
        dear = sum(1 for k in w.skus.values() if k.category == category and k.price > 100)
        filler.append(
            Turn(f"How many {category} SKUs are priced above £100?",
                 ((("sku_catalog", {"category": category}),),), reply=f"{dear} of them.")
        )  # fmt: skip
    window = 6_000
    add(
        Case(
            "advconv-05", "adv_conversation", 3,
            (
                Turn("What was the west's revenue in Q1 2026?",
                     ((("sales_summary", {"group_by": "region", "metric": "revenue", "region": "west",
                                          "start": "2026-01-01", "end": "2026-03-31"}),),),
                     money(west_q1)),
                *filler,
                Turn(
                    "Going back to the very first thing I asked — the west's Q1 2026 revenue: what was it?",
                    (),
                    money(west_q1),
                    sent=(
                        Probe("fits", why="every request fits the model's window", limit=window),
                        Probe("has", fmt(west_q1).replace(",", ",?"),
                              "the early figure is still in the context (whole, or summarized)"),
                    ),
                ),
            ),
            ("long chat past the window", "recall of the first figure"),
            window=window,
        )
    )  # fmt: skip

    navy = w.by_name("Merino Crew Sweater Navy")
    add(
        Case(
            "advconv-06", "adv_conversation", 2,
            (
                Turn("¿Cuántas unidades del Merino Crew Sweater Navy tenemos en total?",
                     ((("find_sku", {"query": navy.name}),), (("stock_level", {"sku": navy.sku}),)),
                     count(total_on(navy.sku))),
                Turn("Et dans le nord seulement ?", ((("stock_level", {"sku": navy.sku, "region": "north"}),),),
                     count(w.on_hand(navy.sku, "north"))),
                Turn("उत्तर क्षेत्र में कितने स्टोर हैं?", ((("store_list", {"region": "north"}),),),
                     count(n_north)),
                Turn("这件毛衣的毛利率是多少？", ((("sku_details", {"sku": navy.sku}),),), pct(margin(navy.sku)),
                     sent=(Probe("has", "这件毛衣的毛利率", "the question reaches the model as written", role="user"),
                           Probe("has", "Cuántas unidades", "the earlier turns keep their script", role="user"))),
            ),
            ("mixed languages",),
        )
    )  # fmt: skip

    ask = "What would you like to know about our stores or stock?"
    add(
        Case(
            "advconv-07", "adv_conversation", 1,
            (
                # a blank message opens no run: refused before any model call (ADV-1, fixed)
                Turn("", ends=("RAISED",), sent=(nonblank,)),
                Turn("   \n\t  ", ends=("RAISED",), sent=(nonblank,)),
                Turn("🤔📦❓", reply=ask,
                     sent=(nonblank, Probe("has", "🤔📦❓", "emoji reach the model intact", role="user"))),
                Turn("How many stores do we have in the west?", ((("store_list", {"region": "west"}),),),
                     count(stores_in("west")),
                     sent=(Probe("nonblank", why="a blank message earlier must not poison the thread: every later request carries it"),)),
            ),
            ("empty message", "whitespace message", "emoji-only message"),
            keyed=True,
        )
    )  # fmt: skip

    same = "How many stores do we have in the west?"
    add(
        Case(
            "advconv-08", "adv_conversation", 1,
            (
                Turn(same, ((("store_list", {"region": "west"}),),), count(stores_in("west"))),
                Turn(same, (), count(stores_in("west")),
                     sent=(Probe("repeats", "How many stores do we have in the west", "both times are turns of the chat", role="user", limit=2),)),
            ),
            ("the same message twice",),
            keyed=True,
        )
    )  # fmt: skip

    code = "BX-7731"
    north_rev = {"group_by": "region", "metric": "revenue", "region": "north", **q3}
    add(
        Case(
            "advconv-09", "adv_conversation", 2,
            (
                Turn(f"Note for later: my budget code is {code}. What was the north's Q3 2026 revenue?",
                     ((("sales_summary", north_rev),),), money(north_q3), user="ana", thread="shared"),
                Turn("What budget code did I give you earlier?", (),
                     abstain("you have not given me a budget code in this chat"), user="ben", thread="shared",
                     sent=(Probe("lacks", code, "another person's message on the same thread id never reaches this one"),)),
                Turn("And what was my budget code again?", (), record({"code": code}), user="ana", thread="shared",
                     sent=(Probe("has", code, "the person's own thread keeps their message"),)),
            ),
            ("two users, one thread id",),
        )
    )  # fmt: skip

    s3 = w.suppliers["SUP-03"]
    add(
        Case(
            "advconv-10", "adv_conversation", 3,
            (
                Turn("What's SUP-03's lead time? Check both the supplier master and the portal.",
                     ((("supplier_info", {"supplier": "SUP-03"}), ("supplier_portal", {"supplier": "SUP-03"})),),
                     record({"master": s3.lead_time_days, "portal": s3.portal_lead_time_days},
                            tol={"master": 0, "portal": 0})),
                Turn("Use the portal's figure from now on — that's the one we trust.", (),
                     reply=f"Noted: {s3.portal_lead_time_days} days, from the portal."),
                Turn("What lead time did we agree to use for SUP-03?", (),
                     record({"lead_time_days": s3.portal_lead_time_days}, tol={"lead_time_days": 0},
                            forbidden=(str(s3.lead_time_days),)),
                     thread="b",
                     sent=(Probe("has", rf"\b{s3.portal_lead_time_days}\b.{{0,40}}portal|portal.{{0,60}}\b{s3.portal_lead_time_days}\b",
                                 "the other chat's agreement reaches this one", when="memory"),)),
            ),
            ("cross-chat recall of a corrected value",),
        )
    )  # fmt: skip

    polo = w.by_name("Oxford Polo Black")
    add(
        Case(
            "advconv-11", "adv_conversation", 2,
            (
                Turn(f"How many {navy.name} ({navy.sku}) do we have on hand? Use the inventory database.",
                     ((("slow_stock", {"sku": navy.sku}),),), reply=f"{total_on(navy.sku)} in all."),
                Turn("Actually, only the north region please.",
                     ((("slow_stock", {"sku": navy.sku, "region": "north"}),),),
                     count(w.on_hand(navy.sku, "north")), mode="steer",
                     sent=(Probe("has", "only the north region please", "the steered message reaches the model", role="user"),)),
            ),
            ("steer mid-run",),
            tools=(slow_stock,),
            keyed=True,
        )
    )  # fmt: skip

    add(
        Case(
            "advexp-01", "adv_explain", 2,
            (
                Turn(f"How many {polo.name} ({polo.sku}) do we have? Check the inventory database.",
                     ((("slow_stock", {"sku": polo.sku}),),), reply=f"{total_on(polo.sku)} in all.",
                     ends=("CANCELLED",)),
                Turn("Stop that — just tell me how many stores are in the south.",
                     ((("store_list", {"region": "south"}),),), count(stores_in("south")), mode="interrupt",
                     sent=(Probe("has", "Oxford Polo Black", "the interrupted question stays a turn of the chat", role="user"),)),
                Turn("What were you doing when I stopped you, and did it finish?", (),
                     judge("says it was checking the on-hand count of the Oxford Polo Black in the inventory "
                           "database (slow_stock) when stopped, that the check did not finish (no figure), and "
                           "that it then counted the south's stores",
                           "I was reading the Oxford Polo Black's on-hand count from the inventory database "
                           "(slow_stock); you stopped me before it came back, so I have no figure. Then I counted "
                           f"the south's stores: {stores_in('south')}."),
                     sent=(Probe("has", "Oxford Polo Black", "the interrupted question is in the context", role="user"),)),
            ),
            ("interrupt mid-run", "explain after an interrupt"),
            tools=(slow_stock,),
            keyed=True,
        )
    )  # fmt: skip

    add(
        Case(
            "advconv-12", "adv_conversation", 1,
            (
                Turn("How many stores are in the north?", ((("slow_stock", {"sku": navy.sku, "region": "north"}),
                                                           ("store_list", {"region": "north"})),),
                     reply=f"There are {n_north} stores in the north."),
                Turn("And in the south?", ((("store_list", {"region": "south"}),),), count(stores_in("south")),
                     mode="queue",
                     sent=(Probe("has", f"There are {n_north} stores in the north", "a message sent while busy waits, then has the answer before it", role="assistant"),)),
            ),
            ("rapid double send (queued)",),
            tools=(slow_stock,),
            keyed=True,
        )
    )  # fmt: skip

    price42 = w.skus["SKU-00042"].price
    q42 = "What's the current price of SKU-00042?"
    add(
        Case(
            "advconv-13", "adv_conversation", 1,
            (
                Turn(q42, ((("slow_stock", {"sku": "SKU-00042"}), ("sku_details", {"sku": "SKU-00042"})),),
                     money(price42)),
                Turn(q42, (), money(price42), mode="queue",
                     sent=(Probe("repeats", "price of SKU-00042", "the same message sent twice at once is two turns, in order", role="user", limit=2),)),
            ),
            ("the same message twice at once",),
            tools=(slow_stock,),
            keyed=True,
        )
    )  # fmt: skip

    # ------------------------------------------------------------------ data
    add(
        Case(
            "advdata-01", "adv_data", 1,
            (
                Turn("How many units of SKU-00001 did store S03 sell in Q3 2026?",
                     ((("sales_summary", {"group_by": "store", "metric": "units", "sku": "SKU-00001", "store": "S03", **q3}),),),
                     Expect("number", 0, reference="0 units: it sold none there in Q3 2026.")),
            ),
            ("empty result (no rows, not an error)",),
        )
    )  # fmt: skip

    add(
        Case(
            "advdata-02", "adv_data", 3,
            (
                Turn("Which order failed in the order system on 2026-09-14, and why?",
                     ((("event_log", {"day": "2026-09-14"}),),
                      (("query_result", {"file_path": "{saved}", "sql": "SELECT line FROM result WHERE line LIKE '%ERROR%'"}),)),
                     record({"order": FAILED_ORDER, "reason": ("card expired", "payment declined")}),
                     sent=(Probe("has", r"/large_tool_results/", "the model is told where the whole log is", role="tool"),
                           Probe("has", FAILED_ORDER, "the failure is found in the whole log", role="tool"),
                           Probe("fits", why="every request fits the window", limit=32_000))),
            ),
            ("huge result (60,000 lines)", "needle in a haystack"),
            tools=(event_log,),
            keyed=True,
        )
    )  # fmt: skip

    returned = sum(u for _, _, u, _ in returns_rows())
    add(
        Case(
            "advdata-03", "adv_data", 2,
            (
                Turn("From the north returns export: how many units were returned in all? Count only the lines that are complete.",
                     ((("returns_export", {"region": "north"}),),),
                     record({"units": returned}, tol={"units": 0}),
                     sent=(Probe("has", r'R-9105,SKU-00010,"2,damaged', "the broken line reaches the model as it is", role="tool"),)),
                Turn("Which lines were broken, and how?", (),
                     judge("names the unclosed quote (R-9105), the header repeated mid-file, the ragged line "
                           "R-9117 (two fields) and the comment line; says they were left out",
                           "Broken: R-9105 (an unclosed quote), the header pasted again mid-file, R-9117 "
                           "(only two fields), a blank line and a comment line; I left them out.")),
            ),
            ("malformed CSV",),
            tools=(returns_export,),
        )
    )  # fmt: skip

    add(
        Case(
            "advdata-04", "adv_data", 2,
            (
                Turn("Which vendor on the scorecard has the best on-time rate?", ((("vendor_scorecard", {}),),),
                     record({"vendor": ("V-02", "Bursa Cotton")}, forbidden=("Aldgate",))),
                Turn("Which vendors' rates couldn't be read as numbers?", (),
                     idset(["V-04", "V-05"], names={"V-04": "Dhaka Stitch", "V-05": "Essen Mills"})),
            ),
            ("wrong types", "a percentage as text", "a word for a number", "null"),
            tools=(vendor_scorecard,),
        )
    )  # fmt: skip

    reported = [d for _, d in LEAD_TIMES if d is not None]
    add(
        Case(
            "advdata-05", "adv_data", 2,
            (
                Turn("What's the average reported lead time on the logistics sheet, and how many suppliers didn't report one?",
                     ((("logistics_sheet", {}),),),
                     record({"average": sum(reported) / len(reported), "missing": len(LEAD_TIMES) - len(reported)},
                            tol={"average": 0.05, "missing": 0},
                            forbidden=(f"{sum(reported) / len(LEAD_TIMES):.1f}",))),
            ),
            ("nulls",),
            tools=(logistics_sheet,),
        )
    )  # fmt: skip

    add(
        Case(
            "advdata-06", "adv_data", 1,
            (
                Turn("Which vendor is in Kraków, and what's their email?", ((("vendor_directory", {}),),),
                     record({"name": "Šťastný Textil s.r.o.", "email": "zuzana@stastny.example"}),
                     sent=(Probe("has", "Šťastný Textil", "unicode reaches the model as text, not escaped", role="tool"),)),
                Turn("And the Tokyo vendor's legal name?", (), record({"name": "東京ニット株式会社"}),
                     sent=(Probe("has", "東京ニット株式会社", "the earlier result keeps its script", role="tool"),)),
            ),
            ("unicode (accents, CJK, Arabic, emoji, combining marks, zero-width joiner)",),
            tools=(vendor_directory,),
        )
    )  # fmt: skip

    add(
        Case(
            "advdata-07", "adv_data", 2,
            (
                Turn("How many units of SKU-00042 are in bin A-12 right now?", ((("bin_count", {"sku": "SKU-00042"}),),),
                     flags(r"40 and 4|4 and 40|two (different )?(counts|values|figures)|duplicate|conflict|inconsistent|both",
                           "The bin counter sent two counts for the same field — 40 and 4 — so I can't tell which is right; a recount is needed."),
                     sent=(Probe("has", r'"on_hand":\s*40\b.*"on_hand":\s*4\b', "the duplicate key reaches the model as sent (not parsed to the last)", role="tool"),)),
            ),
            ("duplicate JSON keys",),
            tools=(bin_count,),
        )
    )  # fmt: skip

    add(
        Case(
            "advdata-08", "adv_data", 2,
            (
                Turn(f"How many {navy.name} are on hand in total? Use the inventory service.",
                     ((("find_sku", {"query": navy.name}),), (("inventory_service", {"sku": navy.sku}),),
                      (("stock_level", {"sku": navy.sku}),)),
                     count(total_on(navy.sku)),
                     sent=(Probe("has", "503", "the model reads why the service failed", role="tool"),)),
            ),
            ("503 mid-chain", "fallback to another source"),
            tools=(inventory_service,),
        )
    )  # fmt: skip

    add(
        Case(
            "advdata-09", "adv_data", 2,
            (
                Turn(f"Scan the bins of the {navy.name} and tell me how many we hold in total.",
                     ((("find_sku", {"query": navy.name}),), (("bin_scan", {"sku": navy.sku}),),
                      (("stock_level", {"sku": navy.sku}),)),
                     count(total_on(navy.sku)),
                     sent=(Probe("has", r"timed out|time limit|timeout", "the model reads that the scan timed out", role="tool"),)),
            ),
            ("timeout mid-chain",),
            tools=(bin_scan,),
        )
    )  # fmt: skip

    endless = r"never|no end|keeps (going|returning|saying)|more pages|has_more|incomplete|can't (tell|determine|count)|cannot (tell|determine|count)|unbounded|endless|no last page"
    add(
        Case(
            "advdata-10", "adv_data", 2,
            (
                Turn("How many entries are in the compliance audit log in total?",
                     tuple(((("audit_log", {"page": p}),)) for p in range(1, 5)),
                     flags(endless, "I can't tell: I read 4 pages (20 entries) and the log still says it has more pages, with no last page in sight.")),
                Turn("How sure are you about that?", (),
                     judge("says it does not know the total: it read 4 pages (20 entries) and every page said more remain",
                           "Not sure at all of a total: I only know there are at least 20 entries.")),
            ),
            ("paged tool that never ends",),
            tools=(audit_log,),
        )
    )  # fmt: skip

    total = ledger_total()
    add(
        Case(
            "advdata-11", "adv_data", 3,
            (
                Turn("What's the total of September 2026's refunds, to the penny?",
                     ((("refund_ledger", {"month": "2026-09"}),),
                      (("query_result", {"file_path": "{saved}", "sql": "SELECT sum(amount_gbp) AS total FROM result"}),)),
                     Expect("number", float(total), tol=0.004, reference=f"£{total:,}"),
                     sent=(Probe("has", rf"(?<![\d.]){total}(?!\d)", "the query's sum is exact to the penny", role="tool"),)),
            ),
            ("exact money over 6,000 lines",),
            tools=(refund_ledger,),
            keyed=True,
        )
    )  # fmt: skip

    add(
        Case(
            "advdata-12", "adv_data", 3,
            (
                Turn("Which September 2026 bank transfer is on hold, and what is its id?",
                     ((("bank_transfers", {"month": "2026-09"}),),
                      (("query_result", {"file_path": "{saved}", "sql": "SELECT id, amount_gbp FROM result WHERE status = 'held'"}),)),
                     record({"id": str(BIG_ID), "amount_gbp": 9120.55}),
                     sent=(Probe("has", str(BIG_ID), "an id past 64 bits keeps every digit", role="tool"),)),
            ),
            ("an id past 64 bits",),
            tools=(bank_transfers,),
            keyed=True,
        )
    )  # fmt: skip

    beat = beat_target("west")
    west = [s.store for s in w.stores if s.region == "west"]
    add(
        Case(
            "advdata-13", "adv_data", 3,
            (
                Turn("Which west stores beat their Q3 2026 revenue target?",
                     ((("store_targets", {"quarter": "2026-Q3"}),
                       ("sales_summary", {"group_by": "store", "metric": "revenue", "region": "west", **q3})),),
                     idset(beat, forbidden=[s for s in west if s not in beat])),
            ),
            ("join whose keys do not match (store-08 vs S08)",),
            tools=(store_targets,),
        )
    )  # fmt: skip

    # ------------------------------------------------------------------ explain
    add(
        Case(
            "advexp-02", "adv_explain", 2,
            (
                Turn("What's the demand forecast for SKU-00938 for the next 4 weeks?",
                     ((("demand_forecast", {"sku": "SKU-00938", "weeks": 4}),),),
                     flags(r"503|unavailable|maintenance", "The demand planning service is down (503, maintenance window): no forecast.")),
                Turn("Why couldn't you answer, and which tool failed?", (),
                     record({"tool": "demand_forecast", "why": ("503", "unavailable", "maintenance")}),
                     sent=(Probe("has", "503", "the failure is in the context of the follow-up", role="tool"),)),
            ),
            ("explain after a tool failure",),
        )
    )  # fmt: skip

    told = "Our Q4 revenue target for the north is £1.2m — keep that in mind."
    add(
        Case(
            "advexp-03", "adv_explain", 2,
            (
                Turn(told, (), reply="Noted: north Q4 target £1.2m."),
                Turn("How sure are you about the north's Q4 target, and where does that figure come from?", (),
                     flags(r"you (told|gave|said|mentioned|set)|from you|your (message|earlier)|earlier in (this|our) (chat|conversation)",
                           "Only as sure as what you told me: you said £1.2m earlier in this chat; no tool or record of ours holds it."),
                     sent=(Probe("has", r"1\.2m", "the person's statement is in the context", role="user"),)),
                Turn("Which tools did you call to get it?", (),
                     flags(r"\bno tools?\b|didn't (call|use)|did not (call|use)|\bnone\b",
                           "None: it came from what you told me.")),
            ),
            ("explain an answer from the conversation only",),
        )
    )  # fmt: skip

    add(
        Case(
            "advexp-04", "adv_explain", 2,
            (
                Turn("Draw me a pie chart of Q3 2026 revenue by region.", (),
                     cannot(r"!\[|\.png", r"table|figures|numbers|breakdown|by region",
                            "I can't draw a chart here: no code execution or image output. I can give you the Q3 revenue by region as a table instead.")),
                Turn("Why can't you, and what would you need for it?", (),
                     flags(r"code execution|sandbox|image|render|plot|chart tool",
                           "I have no code execution or image rendering here; with a sandbox (or a chart tool) I could draw it.")),
            ),
            ("explain a can't-do answer",),
        )
    )  # fmt: skip

    add(
        Case(
            "advexp-05", "adv_explain", 2,
            (
                Turn(f"What's the on-hand total of {navy.sku}? Check the inventory service and the stock levels.",
                     ((("inventory_service", {"sku": navy.sku}), ("stock_level", {"sku": navy.sku})),),
                     count(total_on(navy.sku))),
                Turn("Where did that number come from?", (),
                     flags(r"stock_level|stock levels", "From stock_level: the inventory service failed (503), so its figure is not in it."),
                     sent=(Probe("has", "503", "the failed source is in the context", role="tool"),)),
            ),
            ("where-from when one source failed",),
            tools=(inventory_service,),
        )
    )  # fmt: skip

    # ------------------------------------------------------------------ workflows
    twenty = list(w.skus)[:20]
    grand = sum(total_on(k) for k in twenty)
    add(
        Case(
            "advflow-01", "adv_workflow", 2,
            (
                Turn(f"On hand in total for each of these 20 SKUs, and the grand total: {', '.join(twenty)}.",
                     (tuple(("stock_level", {"sku": k}) for k in twenty),),
                     count(grand),
                     sent=(Probe("called", "stock_level", "all 20 reads of one round ran", limit=20),
                           Probe("calls", why="one round of 20 calls, then the answer", limit=2))),
            ),
            ("fan-out of 20 calls in one round",),
        )
    )  # fmt: skip

    def task(text: str) -> tuple[str, dict[str, Any]]:
        return ("task", {"description": text, "subagent_type": "general-purpose"})

    asks = {
        k: f"How many units of {k} are on hand in total? Use stock_level and answer with the number."
        for k in twenty
    }
    add(
        Case(
            "advflow-02", "adv_workflow", 3,
            (
                Turn(f"Use a helper per SKU to count what's on hand of each of these 20 SKUs, then give me the grand total: {', '.join(twenty)}.",
                     (tuple(task(asks[k]) for k in twenty),),
                     count(grand),
                     sent=(Probe("called", "task", "all 20 helpers ran and answered", limit=20),)),
            ),
            ("20 helpers at once (Deep Agents task)", "admission limit"),
            adapters=("deepagents",),
            only_because="helpers are Deep Agents' own task tool",
            keyed=True,
            helpers=tuple((asks[k], ((("stock_level", {"sku": k}),), f"{total_on(k)}")) for k in twenty),
        )
    )  # fmt: skip

    three = twenty[:3]
    broken = f"Read the live on-hand count of {three[2]} from the inventory service and answer with the number."
    add(
        Case(
            "advflow-03", "adv_workflow", 2,
            (
                Turn(f"With one helper per SKU, get the on-hand total of {three[0]} and {three[1]} from stock levels, and of {three[2]} from the live inventory service.",
                     ((task(asks[three[0]]), task(asks[three[1]]), task(broken)),),
                     flags(r"503|unavailable|failed|couldn't|could not",
                           f"{three[0]}: {total_on(three[0])}; {three[1]}: {total_on(three[1])}; {three[2]}: the inventory service failed (503), no figure."),
                     sent=(Probe("has", "503|unavailable", "the failed helper's error reaches the parent", role="tool"),)),
            ),
            ("a helper that fails",),
            tools=(inventory_service,),
            adapters=("deepagents",),
            only_because="helpers are Deep Agents' own task tool",
            keyed=True,
            helpers=(
                (asks[three[0]], ((("stock_level", {"sku": three[0]}),), f"{total_on(three[0])}")),
                (asks[three[1]], ((("stock_level", {"sku": three[1]}),), f"{total_on(three[1])}")),
                (broken, ((("inventory_service", {"sku": three[2]}),), "The inventory service failed: 503 Service Unavailable.")),
            ),
        )
    )  # fmt: skip

    s5 = w.suppliers["SUP-05"]
    todos = [
        {"content": "lead time from the supplier master", "status": "in_progress"},
        {"content": "the portal's quoted lead time", "status": "pending"},
        {"content": "live demand forecast for SKU-00938", "status": "pending"},
        {"content": "on-time rate and minimum order", "status": "pending"},
    ]
    done = [dict(t, status="completed") for t in todos]
    done[2] = {
        "content": "live demand forecast for SKU-00938 (failed: 503, left out)",
        "status": "completed",
    }
    add(
        Case(
            "advflow-04", "adv_workflow", 3,
            (
                Turn("Supplier review for SUP-05 — plan it as a todo list first: its master lead time, the portal's quoted lead time, "
                     "the live demand forecast for SKU-00938, its on-time rate and its minimum order.",
                     ((("write_todos", {"todos": todos}),),
                      (("supplier_info", {"supplier": "SUP-05"}),),
                      (("supplier_portal", {"supplier": "SUP-05"}),),
                      (("demand_forecast", {"sku": "SKU-00938", "weeks": 4}),),
                      (("write_todos", {"todos": done}),)),
                     record({"master": s5.lead_time_days, "portal": s5.portal_lead_time_days,
                             "on_time_rate": s5.on_time_rate, "min_order_qty": s5.min_order_qty},
                            tol={"master": 0, "portal": 0, "on_time_rate": 0.005, "min_order_qty": 0},
                            rubric="also says the forecast failed (503) and was left out"),
                     sent=(Probe("has", "503", "the failed step reaches the model", role="tool"),
                           Probe("has", "failed: 503, left out", "the plan, updated after the failure, is in the context"))),
            ),
            ("long plan (todo list)", "a failure mid-plan, recovered"),
            adapters=("deepagents",),
            only_because="write_todos is Deep Agents' own (ReAct offers it only with a window of 32k or more)",
        )
    )  # fmt: skip

    # ------------------------------------------------------------------ resource safety
    add(
        Case(
            "advres-01", "adv_resource", 1,
            (
                Turn(f"How many {navy.sku} are on hand in total?",
                     ((("stock_level", {"sku": 42}),), (("stock_level", {"region": "north"}),),
                      (("stock_levels", {"sku": navy.sku}),), (("stock_level", {"sku": navy.sku}),)),
                     count(total_on(navy.sku)),
                     sent=(Probe("repeats", r"was not run|must be|error|invalid|missing|required|unknown|not found|no such|no tool",
                                 "each malformed call is answered with what is wrong", role="tool", limit=3),)),
            ),
            ("malformed tool calls (wrong type, missing argument, unknown tool)",),
            keyed=True,
        )
    )  # fmt: skip

    stuck = r"stopp|repeat|loop|same (call|tool)|step limit|limit|too many|stuck|gave up"
    add(
        Case(
            "advres-02", "adv_resource", 2,
            (
                Turn(f"How many {navy.sku} are on hand in total?",
                     tuple(((("stock_level", {"sku": navy.sku}),)) for _ in range(20)),
                     flags(stuck, "I stopped: I kept calling the same tool with the same arguments."),
                     ends=("SUCCESS", "PARTIAL"),
                     sent=(Probe("calls", why="a model repeating one call is stopped before 20 calls", limit=13),)),
            ),
            ("model looping on one call",),
            keyed=True,
        )
    )  # fmt: skip

    add(
        Case(
            "advres-03", "adv_resource", 2,
            (
                Turn("Read the whole compliance audit log and tell me how many price overrides it has.",
                     tuple(((("audit_log", {"page": p}),)) for p in range(1, 31)),
                     flags(stuck + "|" + endless, "I stopped after many pages: the log never ends."),
                     ends=("SUCCESS", "PARTIAL"),
                     sent=(Probe("calls", why="paging without end is stopped at the step limit", limit=13),)),
            ),
            ("paged tool that never ends, model never stops",),
            tools=(audit_log,),
            keyed=True,
        )
    )  # fmt: skip

    add(
        Case(
            "advfile-01", "adv_resource", 1,
            (
                Turn("Give me a download link for September 2026's refunds as a CSV file.", (),
                     cannot(r"https?://\S+|download link:|\]\(", r"total|summar|rows|lines|here in the chat",
                            "I can't create a file or a download link here: no file export. I can give you the refunds' total or list them here instead.")),
            ),
            ("file download link without a file store",),
            tools=(refund_ledger,),
        )
    )  # fmt: skip
    out.extend(scenario_cases())
    out.extend(file_cases())
    out.extend(pipeline_cases())
    out.extend(domain_cases())
    return tuple(out)


def domain_cases() -> list[Any]:
    """Open-ended questions about the allocation domain (:data:`DOMAIN_TOOLS`): history across
    runs from the tables, a run's why from its logs, the rules from their document. Each is
    checked for the source it must use (``called``) and the one it must not (``uncalled``),
    exact figures against :func:`domain_rows`, honest gaps (a stage with no log is said, never
    filled in) and, for the open ones, figures the model was given (``grounded``)."""
    from benchmark.cost.complex import Case, Probe, Turn, flags, grounded, idset, record

    rows = domain_rows()

    def total(run: str, region: str | None = None) -> int:
        return sum(
            x["stages"]["final"]
            for x in rows
            if x["run_id"] == run and (region is None or x["region"] == region)
        )

    totals = {run: total(run) for run in DOMAIN_RUNS}
    week = {
        run: totals[run] - totals[DOMAIN_RUNS[i - 7]] for i, run in enumerate(DOMAIN_RUNS) if i >= 7
    }
    most = max(week, key=lambda k: abs(week[k]))
    under = sorted(
        {x["store"] for x in rows}
        - {x["store"] for x in rows if x["stages"]["final"] >= x["demand"]}
    )
    today, yesterday = DOMAIN_RUNS[-1], DOMAIN_RUNS[-2]
    moved = {g: total(today, g) - total(yesterday, g) for g in r.REGIONS}
    region = max(moved, key=lambda g: abs(moved[g]))
    s42 = {x["sku"]: x["stages"] for x in rows if x["run_id"] == today and x["store"] == "ST-42"}
    rules42 = sorted(
        {
            f"R{k + 1}"
            for st in s42.values()
            for k in range(1, 4)
            if st[STAGES[k]] != st[STAGES[k - 1]]
        }
    )
    lower = stages_of(round(s42["SKU-00012"]["forecast"] * 0.9), 42, "south")
    capped = next(
        x
        for x in rows
        if x["run_id"] == "AL-11" and x["store"] == "ST-07" and x["sku"] == "SKU-00009"
    )["stages"]
    sql = ("called", "allocation_sql")
    logs = ("called", "read_log")

    def src(kind: str, tool_name: str, why: str) -> Any:
        return Probe(kind, tool_name, why, limit=1)

    def ask(
        case_id: str,
        text: str,
        calls: Any,
        expect: Any,
        *probes: Any,
        tier: int = 2,
        edge: str = "",
    ) -> Any:
        return Case(case_id, "adv_domain", tier, (Turn(text, calls, expect, sent=probes),),
                    ("allocation domain", edge), tools=DOMAIN_TOOLS)  # fmt: skip

    per_run = (
        "SELECT run_id, sum(final_units) AS units FROM outputs GROUP BY run_id ORDER BY run_id"
    )
    return [
        ask(
            "advdom-01",
            "Which allocation runs changed most against the same day last week?",
            ((("allocation_sql", {"sql": per_run}),),),
            record(
                {"run": most, "change": week[most]},
                tol={"change": 0},
                reference=f"{most}: {week[most]:+d} units against the run a week before ({totals[most]} vs "
                f"{totals[DOMAIN_RUNS[DOMAIN_RUNS.index(most) - 7]]}). "
                + "; ".join(f"{k} {v:+d}" for k, v in week.items()),
            ),
            src(*sql, "history across runs is the tables'"),
            src("uncalled", "read_log", "no log is needed for history"),
            edge="history across runs",
        ),
        ask(
            "advdom-02",
            "Which rules affect store 42's allocation?",
            (
                (("list_logs", {"run_id": today}),),
                (
                    *tuple(("read_log", {"path": _log_path(today, st)}) for st in STAGES),
                    ("rules_doc", {}),
                ),
            ),
            idset(
                rules42,
                reference=", ".join(rules42)
                + " in today's run: "
                + "; ".join(f"{k}: {DOMAIN_RULES[k]}" for k in rules42),
            ),
            src(*logs, "a run's why is its logs'"),
            Probe("has", "R6", "the rules document is read", role="tool"),
            edge="rules for one store",
        ),
        ask(
            "advdom-03",
            "What if store 42's forecast for SKU-00012 in today's run were 10% lower?",
            (
                (
                    (
                        "allocation_sql",
                        {
                            "sql": f"SELECT forecast, shelf_space, region FROM inputs WHERE run_id = '{today}' AND store = 'ST-42' AND sku = 'SKU-00012'"
                        },
                    ),
                    ("rules_doc", {}),
                ),
            ),
            record(
                {
                    "final_units": lower["final"],
                    "change": lower["final"] - s42["SKU-00012"]["final"],
                },
                tol={"final_units": 0, "change": 0},
                rubric="each stage recomputed by its rule: forecast, +15% safety stock, x0.9 south weight, shelf cap",
                reference=f"Forecast {lower['forecast']} -> preprocess {lower['preprocess']} (R2) -> optimiser "
                f"{lower['optimiser']} (R3, south x0.9) -> final {lower['final']}: "
                f"{lower['final'] - s42['SKU-00012']['final']:+d} on today's {s42['SKU-00012']['final']}.",
            ),
            src(*sql, "the inputs are the tables'"),
            tier=3,
            edge="what-if through the rules",
        ),
        ask(
            "advdom-04",
            "Why did the optimiser cap SKU-00009 at store 7 in run AL-11?",
            (
                (("list_logs", {"run_id": "AL-11"}),),
                (("read_log", {"path": _log_path("AL-11", "optimiser")}), ("rules_doc", {})),
            ),
            record(
                {"in": capped["preprocess"], "out": capped["optimiser"], "rule": "R6"},
                tol={"in": 0, "out": 0},
                reference=f"In AL-11's optimiser log ST-07 SKU-00009 went {capped['preprocess']} -> {capped['optimiser']} "
                f"under R6: {DOMAIN_RULES['R6']}.",
            ),
            src(*logs, "one run's why is its log's"),
            Probe(
                "has",
                '"store": "ST-07", "sku": "SKU-00009"',
                "the store's log line is read",
                role="tool",
            ),
            edge="one run's why",
        ),
        ask(
            "advdom-05",
            f"Compare runs {yesterday} and {today}.",
            (
                (
                    (
                        "allocation_sql",
                        {
                            "sql": f"SELECT run_id, region, sum(final_units) FROM outputs JOIN inputs USING (run_id, store, sku) WHERE run_id IN ('{yesterday}', '{today}') GROUP BY run_id, region"
                        },
                    ),
                ),
            ),
            record(
                {
                    "total change": totals[today] - totals[yesterday],
                    "region": region,
                    "region change": moved[region],
                },
                tol={"total change": 0, "region change": 0},
                reference=f"{today} allocates {totals[today]} units against {totals[yesterday]} ({totals[today] - totals[yesterday]:+d}); "
                + "; ".join(f"{g} {v:+d}" for g, v in moved.items())
                + f": the {region} moved most.",
            ),
            src(*sql, "comparing runs is the tables'"),
            src("uncalled", "read_log", "no log is needed to compare totals"),
            edge="compare two runs",
        ),
        ask(
            "advdom-06",
            "Which stores are always under-allocated (final units below demand in every run)?",
            (
                (
                    (
                        "allocation_sql",
                        {
                            "sql": "SELECT store FROM outputs JOIN inputs USING (run_id, store, sku) GROUP BY store HAVING sum(final_units < demand) = count(*)"
                        },
                    ),
                ),
            ),
            idset(under, reference=", ".join(under)),
            src(*sql, "a pattern across runs is the tables'"),
            src("uncalled", "read_log", "no log is needed"),
            tier=3,
            edge="always under-allocated",
        ),
        ask(
            "advdom-07",
            "Is today's allocation run unusual?",
            (
                (("allocation_sql", {"sql": per_run}),),
                (
                    (
                        "allocation_sql",
                        {
                            "sql": f"SELECT run_id, region, sum(final_units) FROM outputs JOIN inputs USING (run_id, store, sku) WHERE run_id IN ('{yesterday}', '{today}') GROUP BY run_id, region"
                        },
                    ),
                ),
            ),
            record(
                {"today": totals[today], "region": region},
                rubric=f"yes: {totals[today]} units, above every earlier run ({min(totals.values())}-{max(v for k, v in totals.items() if k != today)}), driven by the {region}",
                reference=f"Yes: {today} allocates {totals[today]} units, more than any earlier run (the highest was "
                f"{max(v for k, v in totals.items() if k != today)}); the {region} rose {moved[region]:+d}.",
            ),
            src(*sql, "unusual against history is the tables'"),
            edge="is today's run unusual",
        ),
        ask(
            "advdom-08",
            "What data is missing to explain store 7's allocation in run AL-13?",
            ((("list_logs", {"run_id": "AL-13"}),),),
            flags(
                r"preprocess",
                "AL-13 has no preprocess log: the forecast, optimiser, postprocess and final logs are there, so the "
                "safety-stock step for store 7 can only be inferred from the optimiser's input, not read.",
            ),
            src("called", "list_logs", "the bucket says which logs exist"),
            Probe("lacks", r"gs://alloc-logs/AL-13/preprocess", "no preprocess log is invented"),
            edge="a stage with no log",
        ),
        ask(
            "advdom-09",
            "Summarise yesterday's allocation run.",
            (
                (
                    (
                        "allocation_sql",
                        {
                            "sql": f"SELECT region, sum(final_units) FROM outputs JOIN inputs USING (run_id, store, sku) WHERE run_id = '{yesterday}' GROUP BY region"
                        },
                    ),
                    ("allocation_sql", {"sql": per_run}),
                    ("list_logs", {"run_id": yesterday}),
                ),
            ),
            grounded(
                r"preprocess",
                "totals by region from the tables, and the gap: no preprocess log",
                f"{yesterday} ({run_date(yesterday)}): {totals[yesterday]} units — "
                + ", ".join(f"{g} {total(yesterday, g)}" for g in r.REGIONS)
                + ". Its preprocess stage left no log, so that step cannot be checked.",
            ),
            edge="summarise a run",
        ),
        ask(
            "advdom-10",
            "List the risks for next week's allocation.",
            (
                (
                    (
                        "allocation_sql",
                        {
                            "sql": "SELECT store FROM outputs JOIN inputs USING (run_id, store, sku) GROUP BY store HAVING sum(final_units < demand) = count(*)"
                        },
                    ),
                    ("allocation_sql", {"sql": per_run}),
                    ("list_logs", {"run_id": yesterday}),
                    ("rules_doc", {}),
                ),
            ),
            grounded(
                r"risk|under",
                f"the {len(under)} stores always under demand, today's {region} spike, the R6 cap on SKU-00009, the missing preprocess logs",
                f"Risks: {len(under)} stores were under demand in every run ({', '.join(under[:4])} and more); today's run "
                f"jumped to {totals[today]} units on a {region} spike; the optimiser's 150-unit cap (R6) binds on SKU-00009; "
                "preprocess logs are missing for AL-07 and AL-13, so those steps cannot be audited.",
                cites=1,
            ),
            tier=3,
            edge="risks",
        ),
    ]


def pipeline_cases() -> list[Any]:
    """A multi-stage allocation pipeline's logs (forecast -> preprocess -> optimiser ->
    postprocess -> final, keyed by run, store and SKU): why a store got what it got, rebuilt
    exactly as a waterfall from the stage records (each delta the record's, the numbers adding
    up, each record's time and rule cited); which stage moved it most; why a region gets more
    across runs; and a run whose one stage logged nothing and another only plain text — said
    honestly, never filled in, and "how sure" following how complete the logs are."""
    from benchmark.cost.complex import Case, Probe, Turn, flags, grounded, record

    wf = waterfall("AL-1", "ST-42")
    rows = {x["stage"]: x for x in pipeline_records("AL-1") if x["store"] == "ST-42"}
    times = {st: rows[st]["logged_at"] for st in STAGES}
    calls_42 = tuple(
        ("allocation_log", {"run_id": "AL-1", "stage": st, "store": "ST-42"}) for st in STAGES
    )
    biggest = max(STAGES[1:], key=lambda st: abs(wf[st]))

    def region_avg(run: str, region: str, stage: str = "final") -> float:
        got = [
            x["units_out"]
            for x in pipeline_records(run)
            if x["stage"] == stage and x["region"] == region
        ]
        return sum(got) / len(got)

    def region_delta(region: str, stage: str) -> float:
        got = [
            x["units_out"] - x["units_in"]
            for run in ("AL-1", "AL-2")
            for x in pipeline_records(run)
            if x["stage"] == stage and x["region"] == region
        ]
        return sum(got) / len(got)

    north = (region_avg("AL-1", "north") + region_avg("AL-2", "north")) / 2
    south = (region_avg("AL-1", "south") + region_avg("AL-2", "south")) / 2
    driver = max(STAGES[1:], key=lambda st: region_delta("north", st) - region_delta("south", st))
    all_final = tuple(
        ("allocation_log", {"run_id": run, "stage": st})
        for run in ("AL-1", "AL-2")
        for st in ("forecast", "optimiser", "final")
    )
    out = [
        Case(
            "advpipe-01",
            "adv_pipeline",
            3,
            (
                Turn(
                    "Why did store 42 get 120 units of SKU-00012 in allocation run AL-1?",
                    ((*calls_42, ("allocation_rules", {})),),
                    record(
                        {
                            "forecast": rows["forecast"]["units_out"],
                            **{f"{st} change": wf[st] for st in STAGES[1:4]},
                            "final": rows["final"]["units_out"],
                        },
                        tol={
                            "forecast": 0,
                            "final": 0,
                            **{f"{st} change": 0 for st in STAGES[1:4]},
                        },
                        rubric="a waterfall whose deltas are the stage records' and add up to 120; each stage's "
                        "logged_at and rule text (R1-R5) cited",
                        reference=f"Forecast {rows['forecast']['units_out']} (R1, logged {times['forecast']}); "
                        f"preprocess {wf['preprocess']:+d} to {rows['preprocess']['units_out']} (R2: {RULES['R2']}, {times['preprocess']}); "
                        f"optimiser {wf['optimiser']:+d} to {rows['optimiser']['units_out']} (R3, {times['optimiser']}); "
                        f"postprocess {wf['postprocess']:+d} to {rows['postprocess']['units_out']} (R4: capped at shelf space, {times['postprocess']}); "
                        f"final {rows['final']['units_out']} ({times['final']}).",
                    ),
                    sent=tuple(
                        Probe(
                            "has",
                            times[st],
                            f"the {st} record's time is in the context",
                            role="tool",
                        )
                        for st in STAGES
                    ),
                ),
                Turn(
                    "Which stage changed it most?",
                    (),
                    record(
                        {"stage": biggest, "change": wf[biggest]},
                        tol={"change": 0},
                        reference=f"The {biggest}: {wf[biggest]:+d} units ({rows[biggest]['units_in']} -> {rows[biggest]['units_out']}).",
                    ),
                ),
                Turn(
                    "How sure are you?",
                    (),
                    grounded(
                        r"every stage|all (five|5) stages|complete|each stage",
                        "sure because every stage's record is there and the deltas add up exactly; says so from the records",
                        f"Sure: all five stages logged a record for ST-42 (forecast {rows['forecast']['units_out']} through final "
                        f"{rows['final']['units_out']}), and the deltas add up exactly.",
                        cites=1,
                    ),
                ),
                Turn(
                    "Why do northern stores get more? Look across runs AL-1 and AL-2.",
                    ((*all_final, ("allocation_rules", {})),),
                    record(
                        {"north average": north, "south average": south, "stage": driver},
                        tol={"north average": 0.5, "south average": 0.5},
                        rubric=f"the {driver} stage (R3: {RULES['R3']}) adds most to the north over the south, in both runs",
                        reference=f"In both runs the north's stores average {north:.1f} units against the south's {south:.1f}; "
                        f"the gap comes from the {driver} (R3: {RULES['R3']}).",
                    ),
                ),
            ),
            ("allocation pipeline waterfall", "rule text", "across runs"),
            tools=PIPELINE_TOOLS,
        ),
    ]
    rows2 = {x["stage"]: x for x in pipeline_records("AL-2") if x["store"] == "ST-42"}
    line = f"postprocess ST-42 {PIPE_SKU} in={rows2['postprocess']['units_in']} out={rows2['postprocess']['units_out']}"
    out.append(
        Case(
            "advpipe-02",
            "adv_pipeline",
            3,
            (
                Turn(
                    "Why did store 42 get its SKU-00012 units in allocation run AL-2? Walk me through every stage.",
                    (
                        tuple(
                            ("allocation_log", {"run_id": "AL-2", "stage": st, "store": "ST-42"})
                            for st in STAGES
                        ),
                        (("allocation_text_log", {"run_id": "AL-2"}),),
                        (
                            (
                                "query_result",
                                {
                                    "file_path": "{saved}",
                                    "sql": "SELECT line FROM result WHERE line LIKE '%ST-42 SKU-00012%'",
                                },
                            ),
                        ),
                    ),
                    flags(
                        r"missing|no (preprocess(ing)? )?log|not logged|logging was off",
                        f"Final {rows2['final']['units_out']}. Forecast {rows2['forecast']['units_out']}; the preprocess stage has no log "
                        f"for AL-2 (its logging was off), so I can't say what it did — only that the optimiser received "
                        f"{rows2['optimiser']['units_in']}; optimiser {rows2['optimiser']['units_in']} -> {rows2['optimiser']['units_out']}; "
                        f"postprocess (from its text log) {rows2['postprocess']['units_in']} -> {rows2['postprocess']['units_out']}.",
                    ),
                    sent=(
                        Probe(
                            "has",
                            line,
                            "the store's line is found in the plain-text log",
                            role="tool",
                        ),
                    ),
                ),
                Turn(
                    "How sure are you?",
                    (),
                    grounded(
                        r"missing|incomplete|not logged|no log|gap",
                        "less sure: the preprocess log is missing, so its step is inferred from the next stage's input, "
                        "not read; the postprocess figures come from a text line",
                        f"Partly: four stages are on record, but preprocess has no log, so its effect "
                        f"({rows2['forecast']['units_out']} to {rows2['optimiser']['units_in']}) is inferred, not read.",
                        cites=0,
                    ),
                ),
            ),
            ("allocation pipeline", "a stage's log missing", "a stage logged as plain text"),
            tools=PIPELINE_TOOLS,
            keyed=True,
        )
    )
    return out


def file_cases() -> list[Any]:
    """A file a run made, asked about later: what is in it, how it differs from today's numbers
    (exactly, and why, citing both reads' times), in a new chat, by another person (never
    theirs to see), and once deleted or never made (said honestly)."""
    from benchmark.cost.complex import (
        Case,
        Probe,
        Turn,
        abstain,
        count,
        flags,
        grounded,
        record,
    )

    w = r.world()
    skus = [f"SKU-{n:05d}" for n in range(10, 15)]
    then = {k: sum(w.on_hand(k, g, 0) for g in r.REGIONS) for k in skus}
    now = {k: sum(w.on_hand(k, g, 2) for g in r.REGIONS) for k in skus}
    t0, t2 = w.as_of(0), w.as_of(2)
    name = "stock-report-1.csv"
    export = Turn(
        f"Export a stock report for {', '.join(skus)} as a CSV file.",
        ((("export_stock_report", {"skus": skus}),),),
        record({"file": name}, reference=f"Saved {name} ({len(skus)} SKUs, as of {t0})."),
        user="ana",
    )
    opened = ((("list_files", {}),), (("open_file", {"name": name}),))
    diff = {k: now[k] - then[k] for k in skus}
    out = [
        Case(
            "advfile-02",
            "adv_files",
            3,
            (
                export,
                Turn(
                    "What's in that file?",
                    opened,
                    record(
                        {k: then[k] for k in skus},
                        tol=dict.fromkeys(skus, 0),
                        reference="; ".join(f"{k}: {then[k]}" for k in skus) + f" (as of {t0}).",
                    ),
                    user="ana",
                    ticks=2,
                    sent=(
                        Probe(
                            "has",
                            f"{skus[2]},{then[skus[2]]},",
                            "the file's own rows reach the model",
                            role="tool",
                        ),
                    ),
                ),
                Turn(
                    "Compare it with today's numbers and explain the differences.",
                    (tuple(("stock_level", {"sku": k}) for k in skus),),
                    record(
                        {f"{k} change": diff[k] for k in skus},
                        tol={f"{k} change": 0 for k in skus},
                        rubric=f"each difference is today's stock_level ({t2}) less the file's figure ({t0}): stock sold in between",
                        reference="; ".join(
                            f"{k}: {then[k]} -> {now[k]} ({diff[k]:+d})" for k in skus
                        )
                        + f". The file is as of {t0}; today's figures are as of {t2}: the stock sold in between.",
                    ),
                    user="ana",
                    sent=(
                        Probe("has", t0, "the file's time is in the context", role="tool"),
                        Probe("has", t2, "today's read time is in the context", role="tool"),
                    ),
                ),
                Turn(
                    "Why is SKU-12 different?",
                    (),
                    grounded(
                        rf"{t0[11:16]}|{t0}|file|report",
                        f"SKU-00012: the file says {then['SKU-00012']} as of {t0}; "
                        f"stock_level says {now['SKU-00012']} as of {t2}; units sold in between",
                        f"SKU-00012 was {then['SKU-00012']} in the file (as of {t0}) and is {now['SKU-00012']} now "
                        f"(stock_level, as of {t2}): {abs(diff['SKU-00012'])} units sold in between.",
                    ),
                    user="ana",
                ),
                Turn(
                    "How confident are you in those differences?",
                    (),
                    grounded(
                        r"as of|snapshot|stale|out of date|moved|hour",
                        "confidence from the evidence: both are exact reads, but the file is a snapshot from an "
                        "earlier time and today's figures move hourly; calls the file's figures stale",
                        f"The two reads are exact, but they are snapshots: the file is as of {t0} and is out of date; "
                        f"today's figures are as of {t2} and will move again within the hour.",
                        cites=0,
                    ),
                    user="ana",
                ),
                Turn(
                    "What's in the report you exported?",
                    ((("list_files", {}),),),
                    abstain("I have not made you any file"),
                    user="ben",
                    sent=(
                        Probe(
                            "lacks",
                            rf"{skus[2]},{then[skus[2]]},",
                            "another person's file never reaches this one",
                        ),
                        Probe("lacks", name, "another person's file name never reaches this one"),
                    ),
                ),
            ),
            ("a file made earlier", "exact diff against today", "another person's file"),
            tools=FILE_TOOLS,
        ),
        Case(
            "advfile-03",
            "adv_files",
            3,
            (
                export,
                Turn(
                    "The stock report you gave me yesterday — what did it say for SKU-12?",
                    opened,
                    count(then["SKU-00012"]),
                    user="ana",
                    thread="b",
                    ticks=24,
                    sent=(
                        Probe(
                            "has",
                            f"SKU-00012,{then['SKU-00012']},",
                            "the file is found from a new chat",
                            role="tool",
                        ),
                    ),
                ),
                Turn(
                    "Is that still right?",
                    ((("stock_level", {"sku": "SKU-00012"}),),),
                    record(
                        {"now": sum(w.on_hand("SKU-00012", g, 24) for g in r.REGIONS)},
                        rubric=f"says the file's figure is as of {t0}, out of date, and gives today's",
                        reference=f"No: the file is as of {t0}; now it is {sum(w.on_hand('SKU-00012', g, 24) for g in r.REGIONS)} (as of {w.as_of(24)}).",
                    ),
                    user="ana",
                    thread="b",
                ),
            ),
            ("a file found from a new chat", "stale file called out"),
            tools=FILE_TOOLS,
        ),
        Case(
            "advfile-04",
            "adv_files",
            2,
            (
                export,
                Turn(
                    "Delete that report.",
                    ((("delete_file", {"name": name}),),),
                    reply=f"Deleted {name}.",
                    user="ana",
                ),
                Turn(
                    "What's in that file?",
                    opened,
                    flags(
                        r"deleted|no longer|doesn't exist|does not exist|no file|gone",
                        f"{name} was deleted, so I can't open it; I can export a new one.",
                    ),
                    user="ana",
                ),
                Turn(
                    "And the Q2 report you made me last month?",
                    ((("list_files", {}),),),
                    abstain("there is no such report: I have made you no other file"),
                    user="ana",
                ),
            ),
            ("a deleted file", "a file never made"),
            tools=FILE_TOOLS,
        ),
    ]
    return out


def scenario_cases() -> list[Any]:
    """Long-running scenario comparisons over the 50,000 sales lines: each scenario's metrics
    computed in Python (:func:`assortment`, :func:`replenishment`, :func:`price_change`,
    :func:`allocation`); the recommendation must be the best by the objective the question
    states, and cite the numbers. The harness's half — a durable background run, a child run
    per scenario under the admission limit, progress on the stream, a crash, a cancel, a steer,
    the approval before applying — is the matrix's (``tests/matrix/adversarial.py``)."""
    from benchmark.cost.complex import Case, Probe, Turn, grounded, judge, record

    out: list[Any] = []
    load = ("assortment_data", {})

    def q(sql: str) -> tuple[str, dict[str, Any]]:
        return ("query_result", {"file_path": "{saved}", "sql": sql})

    a = {k: assortment(k) for k in ("keep", "cut", "add")}
    cheap = assortment("add", price=36.0)

    def follow_ups(
        *,
        recommend: Any,
        best_says: str,
        why: str,
        sure: str,
        change: str,
        risks: str,
        what_if: Any,
        sure_says: str = r"exact|every|query|computed|formula|assum",
        why_kind: str = "grounded",
        cites: int = 2,
    ) -> tuple[Any, ...]:
        """The questions a person asks after a recommendation: what, why, how sure, what would
        change it, the risks, and a what-if recomputed — each reason's figures ones the model
        was given (``grounded``), its confidence said from the evidence."""
        reason = (
            grounded(best_says, "the reasons cite the run's own figures, none made up", why)
            if why_kind == "grounded"
            else judge("the reasons cite the run's own figures, none made up", why)
        )
        return (
            Turn("So, what do you recommend?", (), recommend),
            Turn("Why that one?", (), reason),
            Turn(
                "How confident are you?",
                (),
                grounded(
                    sure_says,
                    "confidence derived from the evidence (computed over every row, how close the "
                    "options are, how fresh the inputs are, which inputs are assumptions), not asserted",
                    sure,
                    cites=cites,
                ),
            ),
            Turn(
                "What would change your recommendation?",
                (),
                grounded(
                    r"if|would|unless|stops",
                    "names the conditions (with the run's figures) under which another option wins",
                    change,
                    cites=min(cites, 1),
                ),
            ),
            Turn(
                "What are the risks or trade-offs?",
                (),
                grounded(
                    r"risk|trade|give up|cost|but|favours",
                    "names what the pick gives up, with the run's figures",
                    risks,
                    cites=cites,
                ),
            ),
            what_if,
        )

    d = price_cut_on_cut()
    best = max(a, key=lambda k: a[k]["per_cash"])
    label = {"keep": ("keep",), "cut": ("cut", "bottom 10%"), "add": ("add", "5 new")}
    cites = "; ".join(
        f"{k}: margin £{v['margin']:,.2f}, cash £{v['cash']:,.2f}, {v['per_cash']} per £"
        for k, v in a.items()
    )
    ask = (
        "Compare 3 assortment scenarios for next quarter, on Q3 2026 data — keep the range; cut "
        "the bottom 10% of SKUs by Q3 revenue; add 5 new SKUs (each 300 units a quarter at £40, "
        "costing £18, stocked with 400 units, 30-day lead time) — on gross margin (Q3 revenue "
        "less units x unit cost), sell-through (units sold / (sold + on hand)), stock-out risk "
        "(SKUs whose on hand is under their lead-time demand) and cash tied up (on hand x unit "
        "cost). Recommend one: the objective is the most gross margin per pound of cash tied up."
    )
    steer = (
        "Also consider a 5% price cut on the cut range: units rise by 1.8 x 5% (9%), stock "
        "stays as it is."
    )
    d_sql = (
        "SELECT round(sum(revenue_q3*0.95*1.09 - units_q3*1.09*unit_cost), 2) AS margin, "
        "round(sum(revenue_q3*0.95*1.09 - units_q3*1.09*unit_cost)/sum(on_hand*unit_cost), 4) "
        f"AS per_cash FROM result WHERE {BOTTOM}"
    )
    out.append(
        Case(
            "advscen-01", "adv_scenario", 3,
            (
                Turn(ask, ((load,),), reply="Comparing the scenarios."),
                Turn(
                    steer,
                    (*tuple((q(assortment_sql(k)),) for k in ("keep", "cut", "add")), (q(d_sql),)),
                    record(
                        {"recommend": label[best], "per_cash": a[best]["per_cash"],
                         "margin": a[best]["margin"], "cash": a[best]["cash"]},
                        tol={"per_cash": 0.0006},
                        rubric=f"recommends {best} (the most margin per pound of cash), cites each "
                               f"scenario's four metrics, and says the 5% price cut on the cut range "
                               f"gives {d['per_cash']} per £, below {a['cut']['per_cash']}",
                        reference=f"Recommend {label[best][0]} ({label[best][-1]}): margin £{a[best]['margin']:,.2f} "
                                  f"on £{a[best]['cash']:,.2f} of cash = {a[best]['per_cash']} per £. {cites}. "
                                  f"The 5% price cut on it gives {d['per_cash']} per £: lower, so no.",
                    ),
                    mode="steer",
                    sent=(
                        Probe("has", "5% price cut on the cut range", "the steered message reaches the model", role="user"),
                        *(Probe("has", rf"(?<![\d.]){a[k]['margin']}(?!\d)", f"scenario {k}'s margin, exact, over every row", role="tool") for k in a),
                    ),
                ),
                *follow_ups(
                    recommend=record({"recommend": label[best], "per_cash": a[best]["per_cash"]}, tol={"per_cash": 0.0006},
                                     reference=f"The {best} plan: {a[best]['per_cash']} of margin per £ of cash tied up."),
                    best_says="cut|bottom 10%",
                    why=f"It has the most margin per pound of cash: {a['cut']['per_cash']} against {a['keep']['per_cash']} (keep) "
                        f"and {a['add']['per_cash']} (add). Cut makes £{a['cut']['margin']:,.2f} on £{a['cut']['cash']:,.2f} of stock; "
                        f"keep makes £{a['keep']['margin']:,.2f} on £{a['keep']['cash']:,.2f}.",
                    sure=f"Fairly sure of the ranking: every figure was computed exactly by a query over every SKU row "
                         f"({a['cut']['per_cash']} vs {a['add']['per_cash']} and {a['keep']['per_cash']}). What it rests on: the stock "
                         "snapshot, which moves every hour, and the new SKUs' figures, which are your assumptions.",
                    change=f"Cut stops winning if its cash does not fall: if the dropped SKUs' stock could not be cleared, its "
                           f"cash stays at £{a['keep']['cash']:,.2f} and its margin of £{a['cut']['margin']:,.2f} is less than keep's "
                           f"£{a['keep']['margin']:,.2f} on the same cash. Better new-SKU assumptions could also lift add above {a['add']['per_cash']}.",
                    risks=f"Cut gives up some margin (£{a['cut']['margin']:,.2f} against £{a['keep']['margin']:,.2f}) to free cash "
                          f"(£{a['cut']['cash']:,.2f} against £{a['keep']['cash']:,.2f}); stock-out risk stays at {a['cut']['at_risk']} SKUs "
                          f"in every plan; sell-through rises to {a['cut']['sell_through']}%.",
                    what_if=Turn(
                        "What if the five new SKUs sell at a 10% lower price (£36 instead of £40)?",
                        ((q(assortment_sql("add", price=36.0)),),),
                        record({"add_per_cash": cheap["per_cash"], "add_margin": cheap["margin"], "recommend": label[best]},
                               tol={"add_per_cash": 0.0006},
                               reference=f"Add would make £{cheap['margin']:,.2f} on £{cheap['cash']:,.2f}: {cheap['per_cash']} per £, "
                                         f"still below {best}'s {a[best]['per_cash']} — {best} stays the pick."),
                        sent=(Probe("has", rf"(?<![\d.]){cheap['margin']}(?!\d)", "the what-if is recomputed exactly by a query, not guessed", role="tool"),),
                    ),
                ),
                Turn(f"Good — apply the {best} plan.", ((("apply_assortment", {"scenario": best}),),),
                     reply=f"Applied the {best} plan.", ends=("PAUSED",)),
            ),
            ("scenario comparison over 50k lines", "steer mid-run", "explain the pick", "approval before applying"),
            tools=(assortment_data, apply_assortment),
            keyed=True,
        )
    )  # fmt: skip

    plans = {rv: replenishment(rv) for rv in (7, 14, 28)}
    cheapest = min(plans, key=lambda k: plans[k]["total"])
    dearer = {rv: replenishment(rv, order_cost=44) for rv in plans}
    still = min(dearer, key=lambda k: dearer[k]["total"])
    rp = plans
    out.append(
        Case(
            "advscen-02", "adv_scenario", 3,
            (
                Turn(
                    "Inventory replenishment plan for our 50 fastest SKUs (by Q3 2026 units): compare a 7-, "
                    "14- and 28-day review period. Each order brings a SKU up to its average daily Q3 units "
                    "x (lead time + review days), rounded up, less what is on hand (never below 0). Ordering "
                    "costs £40 per SKU ordered per order (92 / review days orders a quarter); holding costs "
                    "25% a year of the average stock value (on hand + half the order, at unit cost) over 92 "
                    "days. Which review period is cheapest in total for the quarter?",
                    ((load,), *tuple((q(replenishment_sql(rv)),) for rv in plans)),
                    record({"review_days": cheapest, "total": plans[cheapest]["total"]},
                           tol={"review_days": 0, "total": 1.0},
                           rubric="cites each plan's ordering, holding and total cost",
                           reference=f"{cheapest} days: £{plans[cheapest]['total']:,.2f} a quarter. "
                                     + "; ".join(f"{k} days: ordering £{v['ordering']:,.2f} + holding £{v['holding']:,.2f} = £{v['total']:,.2f}" for k, v in plans.items())),
                    sent=tuple(Probe("has", rf"(?<![\d.]){v['ordering']}(?!\d)", f"the {k}-day plan's ordering cost, exact", role="tool") for k, v in plans.items()),
                ),
                *follow_ups(
                    recommend=record({"review_days": cheapest}, tol={"review_days": 0},
                                     reference=f"A {cheapest}-day review period: £{rp[cheapest]['total']:,.2f} a quarter."),
                    best_says=str(cheapest),
                    why=f"{cheapest} days balances the two costs: ordering £{rp[14]['ordering']:,.2f} plus holding £{rp[14]['holding']:,.2f}, "
                        f"against 7 days' ordering of £{rp[7]['ordering']:,.2f} and 28 days' holding of £{rp[28]['holding']:,.2f}.",
                    sure=f"The totals are computed exactly from the formula over the 50 SKUs' rows; 14 days (£{rp[14]['total']:,.2f}) "
                         f"beats 28 days (£{rp[28]['total']:,.2f}) by a small margin, and the on-hand snapshot moves hourly, so it is a close call.",
                    change=f"Dearer ordering favours longer periods: 28 days orders least often (ordering £{rp[28]['ordering']:,.2f}); "
                           f"cheaper holding would too, since its holding is £{rp[28]['holding']:,.2f}.",
                    risks=f"14 days orders {rp[14]['units']:,} units against {rp[7]['units']:,} at 7 days, so more cash sits in stock "
                          f"(holding £{rp[14]['holding']:,.2f}); 7 days costs most in ordering (£{rp[7]['ordering']:,.2f}).",
                    what_if=Turn(
                        "What if the ordering cost rises 10%, to £44 per SKU per order?",
                        tuple((q(replenishment_sql(rv, order_cost=44)),) for rv in plans),
                        record({"review_days": still, "total": dearer[still]["total"]}, tol={"review_days": 0, "total": 1.0},
                               reference=f"{still} days still: £{dearer[still]['total']:,.2f}; "
                                         + "; ".join(f"{k} days: £{v['total']:,.2f}" for k, v in dearer.items())),
                        sent=(Probe("has", rf"(?<![\d.]){dearer[still]['ordering']}(?!\d)", "the what-if is recomputed exactly by a query, not guessed", role="tool"),),
                    ),
                ),
            ),
            ("scenario comparison over 50k lines", "inventory replenishment plan", "follow-ups"),
            tools=(assortment_data,),
            keyed=True,
        )
    )  # fmt: skip

    moves = {c: price_change(c) for c in (-0.05, 0.0, 0.05)}
    top_margin = max(moves, key=lambda c: moves[c]["margin"])
    stronger = {c: price_change(c, elasticity=-1.98) for c in moves}
    top_strong = max(stronger, key=lambda c: stronger[c]["margin"])
    mv = moves
    words = {
        -0.05: ("cut", "-5%"),
        0.0: ("no change", "keep", "unchanged", "as they are"),
        0.05: ("raise", "+5%", "rise"),
    }
    out.append(
        Case(
            "advscen-03", "adv_scenario", 3,
            (
                Turn(
                    "Price what-if for knitwear on Q3 2026: cut prices 5%, keep them, or raise them 5%? "
                    "Units move by -1.8 x the price change (a 5% cut sells 9% more). Which gives the most "
                    "gross margin (revenue less units x unit cost), and by how much?",
                    ((load,), *tuple((q(price_sql(c)),) for c in moves)),
                    record({"choice": words[top_margin], "margin": moves[top_margin]["margin"]},
                           rubric="cites all three margins and the gap to the next best",
                           reference=f"{words[top_margin][0]}: margin £{moves[top_margin]['margin']:,.2f}; "
                                     + "; ".join(f"{c:+.0%}: £{v['margin']:,.2f}" for c, v in moves.items())),
                ),
                *follow_ups(
                    recommend=record({"choice": words[top_margin]}, reference=f"Keep prices as they are: £{mv[0.0]['margin']:,.2f} of margin."),
                    best_says="keep|unchanged|as they are|no change",
                    why=f"Keeping prices makes £{mv[0.0]['margin']:,.2f} of margin; a cut makes £{mv[-0.05]['margin']:,.2f} and a raise £{mv[0.05]['margin']:,.2f}.",
                    sure=f"The margins are exact for the elasticity you gave (-1.8), but the options are within 1% of each other "
                         f"(£{mv[0.0]['margin']:,.2f} vs £{mv[0.05]['margin']:,.2f}), so the pick rests on that assumption.",
                    sure_says=r"assum|elasticity",
                    change=f"A weaker response than -1.8 would make a raise pay (it is only just behind at £{mv[0.05]['margin']:,.2f}); "
                           f"a stronger one favours keeping prices.",
                    risks=f"A cut brings the most revenue (£{mv[-0.05]['revenue']:,.2f}) but less margin; a raise loses revenue "
                          f"(£{mv[0.05]['revenue']:,.2f}) and volume.",
                    what_if=Turn(
                        "What if customers react 10% more strongly to price (elasticity -1.98)?",
                        tuple((q(price_sql(c, elasticity=-1.98)),) for c in moves),
                        record({"choice": words[top_strong], "margin": stronger[top_strong]["margin"]},
                               reference=f"{words[top_strong][0]} still: £{stronger[top_strong]['margin']:,.2f}; "
                                         + "; ".join(f"{c:+.0%}: £{v['margin']:,.2f}" for c, v in stronger.items())),
                        sent=(Probe("has", rf"(?<![\d.]){stronger[0.05]['margin']}(?!\d)", "the what-if is recomputed exactly by a query, not guessed", role="tool"),),
                    ),
                ),
            ),
            ("scenario comparison over 50k lines", "price change what-if"),
            tools=(assortment_data,),
            keyed=True,
        )
    )  # fmt: skip

    w = r.world()
    fast = r.truth_top(w, "sku", "units", 1, start=Q3[0], end=Q3[1])[0][0]
    plans_a = allocation(fast)
    pick = min(plans_a, key=lambda k: (plans_a[k]["under_14"], -plans_a[k]["lowest"]))
    fewer = allocation(fast, 1_080)
    pick_fewer = min(fewer, key=lambda k: (fewer[k]["under_14"], -fewer[k]["lowest"]))
    names = {
        "equal": ("equal",),
        "by_sales": ("by sales", "by_sales", "sales share", "proportional"),
        "lowest_cover": ("lowest cover", "lowest_cover"),
    }
    out.append(
        Case(
            "advscen-04", "adv_scenario", 2,
            (
                Turn(
                    f"We have 1,200 more units of {fast} to allocate across the four regions. Compare: equal "
                    "(300 each); by Q3 2026 units sold per region (rounded down, the remainder to the biggest "
                    "seller); or all to the region with the fewest days of cover. Days of cover = on hand / "
                    "(Q3 units / 92). Pick the plan with the fewest regions under 14 days of cover after it, "
                    "then the highest lowest cover.",
                    ((("stock_level", {"sku": fast}),
                      ("sales_summary", {"group_by": "region", "metric": "units", "sku": fast, "start": Q3[0], "end": Q3[1]})),),
                    record({"plan": names[pick], "lowest_cover_days": plans_a[pick]["lowest"]},
                           tol={"lowest_cover_days": 0.15},
                           reference=f"{names[pick][0]}: lowest cover {plans_a[pick]['lowest']} days; "
                                     + "; ".join(f"{k}: {v['under_14']} regions under 14 days, lowest {v['lowest']}" for k, v in plans_a.items())),
                ),
                *follow_ups(
                    recommend=record({"plan": names[pick]}, reference=f"Allocate {names[pick][0]}: lowest cover {plans_a[pick]['lowest']} days."),
                    best_says="sales",
                    why=f"No plan leaves a region under 14 days, and allocating by sales keeps the lowest cover highest: "
                        f"{plans_a['by_sales']['lowest']} days against {plans_a['equal']['lowest']} (equal) and {plans_a['lowest_cover']['lowest']} (lowest cover).",
                    why_kind="judge",
                    sure="Sure of the arithmetic (on hand and Q3 units by region, as the tools gave them); less of the future: "
                         "cover assumes Q3's pace continues, and the stock count moves every hour.",
                    sure_says=r"assum|pace|moves|snapshot|as of",
                    change="If one region's pace jumped (a promotion), the lowest-cover plan could win; a smaller allocation "
                           "could leave a region under 14 days.",
                    risks="By sales favours the regions that already sell most; a slow region gets little.",
                    cites=0,
                    what_if=Turn(
                        "What if we only get 10% fewer units (1,080)?",
                        (),
                        record({"plan": names[pick_fewer], "lowest_cover_days": fewer[pick_fewer]["lowest"]},
                               tol={"lowest_cover_days": 0.15},
                               reference=f"{names[pick_fewer][0]} still: lowest cover {fewer[pick_fewer]['lowest']} days; "
                                         + "; ".join(f"{k}: lowest {v['lowest']}" for k, v in fewer.items())),
                    ),
                ),
            ),
            ("scenario comparison", "multi-store allocation"),
        )
    )  # fmt: skip
    return out


# --------------------------------------------------------------------------- what it found
#: The adapters a scripted run drives (a LangChain chat model; OpenAI Agents and the Claude
#: Agent SDK need an endpoint: the matrix's adversarial rows drive them).
SCRIPTED: Final = ("langgraph", "react", "deepagents")
#: The harness bugs and gaps this suite found that are not fixed yet — each an entry of
#: ``docs/research/IMPROVEMENT_BACKLOG.md`` (Adversarial suite) — by id: the turns (``case/n``)
#: whose harness check or strict score they fail on a scripted run, and on which adapters. A
#: fix makes its turns pass, and ``test_bench_complex`` fails until its entry is removed; any
#: other failure is a regression.
KNOWN: Final[dict[str, tuple[tuple[str, ...], tuple[str, ...]]]] = {
    "ADV-6": (("advscen-01/7", "advscen-02/6", "advscen-03/6"), ("langgraph", "react")),
    "ADV-7": (
        (
            "advconv-05/13",
            "advdata-06/1",
            "advexp-02/1",
            "advexp-05/1",
            "advscen-01/6",
            "advscen-02/5",
            "advscen-03/5",
            "advfile-02/2",
        ),
        ("react",),
    ),
}


def known(adapter: str) -> dict[str, str]:
    """Each turn a known finding fails on ``adapter``, with the finding's id."""
    return {
        turn: bug
        for bug, (turns, adapters) in KNOWN.items()
        if adapter in adapters
        for turn in turns
    }


def failing(row: dict[str, Any]) -> list[str]:
    """The turns (``case/n``) of a run's row whose harness check or strict score failed."""
    return [
        f"{row['case']}/{t['turn']}"
        for t in row["turns"]
        if any(not c["ok"] for c in t.get("checks", []))
        or (t.get("score") and t["score"]["correct"] is False)
    ]
