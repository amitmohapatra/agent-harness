"""The ``complex`` suite's world: a fashion retailer's fake data, its read tools, and the
Python that computes every answer exactly (``benchmark.cost.complex`` asks the questions).

The data is generated from :data:`SEED`, so it is the same on every machine:

* 2,000 SKUs in eight categories (``SKU-00001``…), each with a supplier, a unit cost and a
  price; a fifth of them were 10% dearer before :data:`PRICE_CUT` (``price_history``);
* 25 suppliers, each with a lead time in the supplier master and, for some, a different one
  on the supplier portal (two sources that disagree, each dated);
* 40 stores in four regions;
* 50,000 sales lines from 2026-01-01 to 2026-09-30 (later months busier, a few SKUs and
  stores far busier than the rest);
* stock on hand per SKU and region, which moves when the world's clock ticks
  (:meth:`World.tick`: the freshness questions ask for the *current* stock after it did).

The tools only read. Some are edge cases on purpose: ``store_audit`` returns nothing,
``margin_feed`` returns cut-off JSON, ``demand_forecast`` fails, ``warehouse_scan`` times
out, and ``sales_rows``/``sku_catalog`` can return tens of thousands of lines.

The truth functions (``truth_*``, :func:`percentile`, :func:`reorder_qty`…) are what every
ground truth is computed with; ``tests/unit/test_bench_complex.py`` checks them against
hand-worked values.
"""

from __future__ import annotations

import bisect
import json
import math
import random
import statistics
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from functools import cache
from typing import Any, Final

from trellis import tool

SEED: Final = 2026
START: Final = date(2026, 1, 1)
END: Final = date(2026, 9, 30)
DAYS: Final = (END - START).days + 1
#: The day a fifth of the SKUs got cheaper (their price before it was 10% higher).
PRICE_CUT: Final = date(2026, 5, 15)
REGIONS: Final = ("north", "south", "east", "west")
N_SALES: Final = 50_000
N_SUPPLIERS: Final = 25
N_STORES: Final = 40
#: When the supplier master was last updated, and when the portal was.
MASTER_UPDATED: Final = "2026-03-02"
PORTAL_UPDATED: Final = "2026-10-28"

# fmt: off
CATEGORIES: Final = {
    "knitwear": (("Merino", "Cashmere", "Cotton", "Wool-Blend", "Chunky"),
                 ("Crew Sweater", "Cardigan", "Turtleneck", "Vest", "Hoodie")),
    "outerwear": (("Insulated", "Waxed", "Lightweight", "Quilted", "Classic"),
                  ("Parka", "Trench Coat", "Puffer Jacket", "Rain Shell", "Bomber")),
    "denim": (("Raw", "Stonewash", "Black", "Vintage", "Stretch"),
              ("Slim Jeans", "Straight Jeans", "Denim Jacket", "Shorts", "Skirt")),
    "footwear": (("Leather", "Suede", "Canvas", "Trail", "Knit"),
                 ("Sneaker", "Chelsea Boot", "Loafer", "Sandal", "Runner")),
    "accessories": (("Wool", "Vegan Leather", "Recycled", "Silk", "Fleece"),
                    ("Scarf", "Beanie", "Belt", "Tote Bag", "Gloves")),
    "shirts": (("Oxford", "Flannel", "Relaxed", "Tailored", "Heavyweight"),
               ("Button Shirt", "Overshirt", "Polo", "Camp Shirt", "Tee")),
    "dresses": (("Floral", "Satin", "Jersey", "Linen", "Pleated"),
                ("Midi Dress", "Wrap Dress", "Shirt Dress", "Slip Dress", "Maxi Dress")),
    "activewear": (("Seamless", "Thermal", "Mesh", "Compression", "Breathable"),
                   ("Leggings", "Track Jacket", "Sports Bra", "Running Shorts", "Training Tee")),
}
COLORS: Final = ("Navy", "Black", "Grey", "Cream", "Olive", "Burgundy", "Camel", "Forest",
                 "Charcoal", "Rust")
CITIES: Final = ("Leeds", "Porto", "Oslo", "Lyon", "Gdansk", "Bursa", "Prato", "Dhaka",
                 "Hanoi", "Izmir", "Braga", "Lodz", "Tirupur", "Biella", "Guimaraes")
STORE_CITIES: Final = {
    "north": ("Leeds", "York", "Newcastle", "Durham", "Hull", "Carlisle", "Lancaster", "Harrogate", "Bradford", "Sunderland"),
    "south": ("Brighton", "Southampton", "Exeter", "Bournemouth", "Plymouth", "Bath", "Truro", "Chichester", "Portsmouth", "Winchester"),
    "east": ("Norwich", "Cambridge", "Ipswich", "Colchester", "Peterborough", "Lincoln", "Ely", "Lowestoft", "Chelmsford", "Bury"),
    "west": ("Bristol", "Cardiff", "Gloucester", "Hereford", "Swansea", "Worcester", "Taunton", "Cheltenham", "Newport", "Shrewsbury"),
}
# fmt: on


@dataclass(frozen=True)
class Sku:
    sku: str
    name: str
    category: str
    supplier: str
    unit_cost: float
    price: float  # the current price (since PRICE_CUT for the SKUs that changed)
    price_before: float | None  # the price before PRICE_CUT, when it changed


@dataclass(frozen=True)
class Supplier:
    supplier: str
    name: str
    city: str
    lead_time_days: int  # the supplier master's
    portal_lead_time_days: int  # the supplier portal's (differs for some)
    on_time_rate: float
    min_order_qty: int


@dataclass(frozen=True)
class Store:
    store: str
    region: str
    city: str


@dataclass(frozen=True)
class Sale:
    day: date
    store: str
    sku: str
    units: int
    revenue: float


class World:
    """The generated data and the clock the stock moves with."""

    def __init__(self, seed: int = SEED) -> None:
        rng = random.Random(seed)
        self.suppliers = self._suppliers(rng)
        self.skus = self._skus(rng)
        self.stores = [
            Store(f"S{n + 1:02d}", REGIONS[n % 4], STORE_CITIES[REGIONS[n % 4]][n // 4])
            for n in range(N_STORES)
        ]
        self.sales = self._sales(rng)
        # stock sized to how fast each SKU sells in each region: 5 to 45 days of it, and a few
        rate: dict[tuple[str, str], float] = defaultdict(float)
        regions = {s.store: s.region for s in self.stores}
        for sale in self.sales:
            rate[(sale.sku, regions[sale.store])] += sale.units / DAYS
        self.base_stock = {
            (s.sku, r): int(rate[(s.sku, r)] * rng.uniform(5, 45)) + rng.randint(0, 20)
            for s in self.skus.values()
            for r in REGIONS
        }
        self.version = 0

    # ------------------------------------------------------------------ generation
    @staticmethod
    def _suppliers(rng: random.Random) -> dict[str, Supplier]:
        found = {}
        for n in range(N_SUPPLIERS):
            lead = rng.randint(7, 42)
            # every fifth supplier's portal says something else than the master
            portal = lead + (7 if n % 5 == 2 else 0)
            sid = f"SUP-{n + 1:02d}"
            found[sid] = Supplier(
                sid,
                f"{CITIES[n % len(CITIES)]} {('Textiles', 'Knit Co', 'Apparel', 'Mills', 'Goods')[n % 5]}",
                CITIES[n % len(CITIES)],
                lead,
                portal,
                round(rng.uniform(0.70, 0.99), 2),
                rng.choice((50, 100, 150, 200, 300)),
            )
        return found

    @staticmethod
    def _skus(rng: random.Random) -> dict[str, Sku]:
        found = {}
        n = 0
        for category, (adjectives, nouns) in CATEGORIES.items():
            for adjective in adjectives:
                for noun in nouns:
                    for color in COLORS:
                        n += 1
                        cost = round(rng.uniform(4.0, 90.0), 2)
                        price = round(cost * rng.uniform(1.6, 2.8), 2)
                        before = round(price * 1.1, 2) if n % 5 == 0 else None
                        sid = f"SKU-{n:05d}"
                        found[sid] = Sku(
                            sid,
                            f"{adjective} {noun} {color}",
                            category,
                            f"SUP-{rng.randint(1, N_SUPPLIERS):02d}",
                            cost,
                            price,
                            before,
                        )
        return found

    def _sales(self, rng: random.Random) -> list[Sale]:
        skus = list(self.skus.values())
        order = list(range(len(skus)))
        rng.shuffle(order)
        sku_weights = [0.0] * len(skus)
        for rank, index in enumerate(order):
            sku_weights[index] = 1.0 / (rank + 1) ** 0.8
        stores = self.stores
        store_weights = [1.0 + (3.0 if s.store in ("S07", "S22") else 0.0) for s in stores]
        day_weights = [1.0 + d / DAYS for d in range(DAYS)]  # later months busier
        picked_skus = rng.choices(range(len(skus)), weights=sku_weights, k=N_SALES)
        picked_stores = rng.choices(range(len(stores)), weights=store_weights, k=N_SALES)
        picked_days = rng.choices(range(DAYS), weights=day_weights, k=N_SALES)
        sales = []
        for i in range(N_SALES):
            sku = skus[picked_skus[i]]
            day = START + timedelta(days=picked_days[i])
            units = rng.randint(1, 6)
            discounted = rng.random() < 0.15
            unit_price = price_on(sku, day) * (0.8 if discounted else 1.0)
            sales.append(
                Sale(
                    day,
                    stores[picked_stores[i]].store,
                    sku.sku,
                    units,
                    round(units * unit_price, 2),
                )
            )
        sales.sort(key=lambda s: (s.day, s.store, s.sku))
        return sales

    # ------------------------------------------------------------------ stock
    def on_hand(self, sku: str, region: str, version: int | None = None) -> int:
        """Units on hand of ``sku`` in ``region`` at the clock's ``version`` (now by default):
        each tick sells one to six of it."""
        v = self.version if version is None else version
        base = self.base_stock[(sku, region)]
        index = int(sku.split("-")[1])
        step = 1 + (index * 7 + REGIONS.index(region) * 13) % 6
        return max(0, base - v * step)

    def as_of(self, version: int | None = None) -> str:
        v = self.version if version is None else version
        return f"2026-10-30T{9 + v:02d}:00:00Z"

    def tick(self) -> None:
        """An hour passes: stock moves."""
        self.version += 1

    def reset(self) -> None:
        self.version = 0

    # ------------------------------------------------------------------ lookups
    def region_of(self, store: str) -> str:
        return next(s.region for s in self.stores if s.store == store)

    def find(self, query: str) -> list[Sku]:
        words = [w for w in query.lower().replace("-", " ").split() if w]
        return [
            s
            for s in self.skus.values()
            if all(w in s.name.lower().replace("-", " ") or w == s.sku.lower() for w in words)
        ]

    def by_name(self, name: str) -> Sku:
        return next(s for s in self.skus.values() if s.name == name)


def price_on(sku: Sku, day: date) -> float:
    return sku.price_before if sku.price_before is not None and day < PRICE_CUT else sku.price


@cache
def world() -> World:
    """The suite's world (made once per process)."""
    return World()


# --------------------------------------------------------------------------- selection
def _d(text: str | None, default: date) -> date:
    return date.fromisoformat(text) if text else default


def selected(
    w: World,
    *,
    start: str | None = None,
    end: str | None = None,
    region: str | None = None,
    category: str | None = None,
    sku: str | None = None,
    store: str | None = None,
) -> list[Sale]:
    """The sales lines matching every filter given (dates inclusive)."""
    first, last = _d(start, START), _d(end, END)
    lo = bisect.bisect_left(w.sales, first, key=lambda s: s.day)
    hi = bisect.bisect_right(w.sales, last, key=lambda s: s.day)
    regions = {s.store: s.region for s in w.stores}
    out = []
    for s in w.sales[lo:hi]:
        if region and regions[s.store] != region.lower():
            continue
        if category and w.skus[s.sku].category != category.lower():
            continue
        if sku and s.sku != sku.upper():
            continue
        if store and s.store != store.upper():
            continue
        out.append(s)
    return out


def _key(w: World, sale: Sale, group_by: str) -> str:
    return {
        "sku": lambda: sale.sku,
        "store": lambda: sale.store,
        "region": lambda: w.region_of(sale.store),
        "category": lambda: w.skus[sale.sku].category,
        "month": lambda: sale.day.isoformat()[:7],
        "date": sale.day.isoformat,
        "supplier": lambda: w.skus[sale.sku].supplier,
    }[group_by]()


GROUPS: Final = ("sku", "store", "region", "category", "month", "date", "supplier")
METRICS: Final = ("revenue", "units", "lines")


def grouped(sales: Iterable[Sale], w: World, group_by: str, metric: str) -> dict[str, float]:
    """Each group's total of ``metric`` (revenue rounded to cents)."""
    totals: dict[str, float] = defaultdict(float)
    for s in sales:
        value = {"revenue": s.revenue, "units": s.units, "lines": 1}[metric]
        totals[_key(w, s, group_by)] += value
    return {k: (round(v, 2) if metric == "revenue" else v) for k, v in totals.items()}


# --------------------------------------------------------------------------- statistics
def percentile(values: Sequence[float], p: float) -> float:
    """The ``p``-th percentile, linear between closest ranks (numpy's default)."""
    ordered = sorted(values)
    if not ordered:
        raise ValueError("no values")
    rank = (len(ordered) - 1) * p / 100
    low = math.floor(rank)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def growth_pct(before: float, after: float) -> float:
    return (after - before) / before * 100


def margin_pct(price: float, cost: float) -> float:
    return (price - cost) / price * 100


def reorder_qty(avg_daily_units: float, lead_days: int, review_days: int, on_hand: int) -> int:
    """The suite's reorder formula (the user states it): average daily units x (lead time +
    review period) - on hand, rounded up, never below zero."""
    return max(0, math.ceil(avg_daily_units * (lead_days + review_days) - on_hand))


def supplier_score(s: Supplier) -> float:
    """The user's weighted score: 0.6 x on-time rate + 0.4 x (1 - lead time / 60)."""
    return 0.6 * s.on_time_rate + 0.4 * (1 - s.lead_time_days / 60)


def iqr_outliers(values: dict[str, float]) -> list[str]:
    """Keys above Q3 + 1.5 IQR or below Q1 - 1.5 IQR (percentiles as :func:`percentile`)."""
    q1, q3 = percentile(list(values.values()), 25), percentile(list(values.values()), 75)
    fence = 1.5 * (q3 - q1)
    return sorted(k for k, v in values.items() if v > q3 + fence or v < q1 - fence)


def daily(w: World, **filters: Any) -> list[float]:
    """Revenue per calendar day (zero on a day with no sale), in date order."""
    per = grouped(selected(w, **filters), w, "date", "revenue")
    first, last = _d(filters.get("start"), START), _d(filters.get("end"), END)
    days = [(first + timedelta(days=n)).isoformat() for n in range((last - first).days + 1)]
    return [per.get(d, 0.0) for d in days]


def regression(xs: Sequence[float], ys: Sequence[float]) -> tuple[float, float]:
    found = statistics.linear_regression(xs, ys)
    return found.slope, found.intercept


# --------------------------------------------------------------------------- the tools
def _json(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"))


@tool(side_effects="read")
def find_sku(query: str) -> str:
    """Search the product catalogue by words of a product's name (or its SKU id). Returns up
    to 10 matches: SKU id, name and category, and how many matched in all.

    Args:
        query: words of the name, e.g. "merino crew sweater navy"
    """
    found = world().find(query)
    return _json(
        {
            "query": query,
            "matches": len(found),
            "items": [{"sku": s.sku, "name": s.name, "category": s.category} for s in found[:10]],
        }
    )


@tool(side_effects="read")
def sku_details(sku: str) -> str:
    """One SKU's record: name, category, supplier id, unit cost and current price (GBP).

    Args:
        sku: the SKU id, e.g. SKU-00042
    """
    s = world().skus.get(sku.upper())
    if s is None:
        return _json({"error": f"no SKU {sku}"})
    return _json(
        {
            "sku": s.sku,
            "name": s.name,
            "category": s.category,
            "supplier": s.supplier,
            "unit_cost": s.unit_cost,
            "price": s.price,
        }
    )


@tool(side_effects="read")
def price_history(sku: str) -> str:
    """Every price a SKU has had this year, with the date each took effect.

    Args:
        sku: the SKU id
    """
    s = world().skus.get(sku.upper())
    if s is None:
        return _json({"error": f"no SKU {sku}"})
    changes = [{"from": START.isoformat(), "price": s.price_before or s.price}]
    if s.price_before is not None:
        changes.append({"from": PRICE_CUT.isoformat(), "price": s.price})
    return _json({"sku": s.sku, "prices": changes})


@tool(side_effects="read")
def stock_level(sku: str, region: str = "all") -> str:
    """Units on hand of a SKU right now, per region (north, south, east, west) or for one
    region, with the time the count was read.

    Args:
        sku: the SKU id
        region: one region, or "all"
    """
    w = world()
    s = w.skus.get(sku.upper())
    if s is None:
        return _json({"error": f"no SKU {sku}"})
    regions = REGIONS if region.lower() == "all" else (region.lower(),)
    if any(r not in REGIONS for r in regions):
        return _json({"error": f"no region {region}; regions are {', '.join(REGIONS)}"})
    by = {r: w.on_hand(s.sku, r) for r in regions}
    return _json({"sku": s.sku, "as_of": w.as_of(), "on_hand": by, "total": sum(by.values())})


@tool(side_effects="read")
def supplier_info(supplier: str) -> str:
    """A supplier's record from the supplier master: name, city, lead time in days, on-time
    delivery rate and minimum order quantity.

    Args:
        supplier: the supplier id, e.g. SUP-07
    """
    s = world().suppliers.get(supplier.upper())
    if s is None:
        return _json({"error": f"no supplier {supplier}"})
    return _json(
        {
            "supplier": s.supplier,
            "name": s.name,
            "city": s.city,
            "lead_time_days": s.lead_time_days,
            "on_time_rate": s.on_time_rate,
            "min_order_qty": s.min_order_qty,
            "source": f"supplier master, updated {MASTER_UPDATED}",
        }
    )


@tool(side_effects="read")
def supplier_portal(supplier: str) -> str:
    """What the supplier itself publishes on its portal: its quoted lead time in days.

    Args:
        supplier: the supplier id
    """
    s = world().suppliers.get(supplier.upper())
    if s is None:
        return _json({"error": f"no supplier {supplier}"})
    return _json(
        {
            "supplier": s.supplier,
            "quoted_lead_time_days": s.portal_lead_time_days,
            "source": f"supplier portal, updated {PORTAL_UPDATED}",
        }
    )


@tool(side_effects="read")
def store_list(region: str = "all") -> str:
    """The stores: id, region and city — all, or one region's.

    Args:
        region: north, south, east, west or "all"
    """
    stores = [
        {"store": s.store, "region": s.region, "city": s.city}
        for s in world().stores
        if region.lower() in ("all", s.region)
    ]
    return _json({"stores": stores})


@tool(side_effects="read")
def sales_summary(
    group_by: str,
    metric: str = "revenue",
    start: str = "2026-01-01",
    end: str = "2026-09-30",
    region: str | None = None,
    category: str | None = None,
    sku: str | None = None,
    store: str | None = None,
    top: int | None = None,
) -> str:
    """Sales totals grouped by one field, largest first (by date or month: in date order).
    Sales data covers 2026-01-01 to 2026-09-30 only. Revenue is in GBP after discounts.

    Args:
        group_by: sku, store, region, category, month, date or supplier
        metric: revenue, units or lines
        start: first day, YYYY-MM-DD (inclusive)
        end: last day, YYYY-MM-DD (inclusive)
        region: only this region's stores
        category: only this category
        sku: only this SKU
        store: only this store
        top: only the first N groups
    """
    w = world()
    if group_by not in GROUPS or metric not in METRICS:
        return _json({"error": f"group_by is one of {GROUPS}; metric one of {METRICS}"})
    found = grouped(
        selected(w, start=start, end=end, region=region, category=category, sku=sku, store=store),
        w,
        group_by,
        metric,
    )
    if group_by in ("date", "month"):
        rows = sorted(found.items())
    else:
        rows = sorted(found.items(), key=lambda kv: (-kv[1], kv[0]))
    if top:
        rows = rows[:top]
    filters = {
        k: v
        for k, v in {"region": region, "category": category, "sku": sku, "store": store}.items()
        if v
    }
    return _json(
        {
            "group_by": group_by,
            "metric": metric,
            "start": start,
            "end": end,
            "filters": filters,
            "groups": len(found),
            "rows": [{"key": k, "value": v} for k, v in rows],
        }
    )


@tool(side_effects="read")
def sales_rows(
    start: str = "2026-01-01",
    end: str = "2026-09-30",
    region: str | None = None,
    category: str | None = None,
    sku: str | None = None,
    store: str | None = None,
) -> str:
    """Every sales line matching the filters, as CSV (date,store,sku,units,revenue). Large:
    the whole period is 50,000 lines. Sales data covers 2026-01-01 to 2026-09-30 only.

    Args:
        start: first day, YYYY-MM-DD (inclusive)
        end: last day, YYYY-MM-DD (inclusive)
        region: only this region's stores
        category: only this category
        sku: only this SKU
        store: only this store
    """
    lines = ["date,store,sku,units,revenue"]
    for s in selected(
        world(), start=start, end=end, region=region, category=category, sku=sku, store=store
    ):
        lines.append(f"{s.day.isoformat()},{s.store},{s.sku},{s.units},{s.revenue:.2f}")
    return "\n".join(lines)


@tool(side_effects="read")
def sku_catalog(category: str | None = None) -> str:
    """The whole product catalogue (2,000 SKUs) or one category's, as CSV
    (sku,name,category,supplier,unit_cost,price).

    Args:
        category: only this category
    """
    lines = ["sku,name,category,supplier,unit_cost,price"]
    for s in world().skus.values():
        if category and s.category != category.lower():
            continue
        lines.append(f"{s.sku},{s.name},{s.category},{s.supplier},{s.unit_cost:.2f},{s.price:.2f}")
    return "\n".join(lines)


@tool(side_effects="read")
def store_audit(store: str) -> str:
    """The latest shelf audit of a store: the items found missing or damaged.

    Args:
        store: the store id, e.g. S07
    """
    return _json({"store": store.upper(), "audited": None, "items": []})


#: What ``margin_feed`` returns: the finance feed cut off mid-record.
MALFORMED: Final = '{"category":"%s","as_of":"2026-10-29","rows":[{"sku":"SKU-0'


@tool(side_effects="read")
def margin_feed(category: str) -> str:
    """The finance team's margin feed for a category (JSON).

    Args:
        category: the category
    """
    return MALFORMED % category.lower()


@tool(side_effects="read")
def demand_forecast(sku: str, weeks: int = 4) -> str:
    """The demand planning service's forecast of a SKU's weekly units.

    Args:
        sku: the SKU id
        weeks: how many weeks ahead
    """
    raise RuntimeError("demand planning service: 503 Service Unavailable (maintenance window)")


@tool(side_effects="read")
def warehouse_scan(region: str) -> str:
    """A live scan of a region's warehouse bins (slow).

    Args:
        region: the region
    """
    raise TimeoutError("warehouse_scan timed out after 30 seconds")


TOOLS: Final = (
    find_sku,
    sku_details,
    price_history,
    stock_level,
    supplier_info,
    supplier_portal,
    store_list,
    sales_summary,
    sales_rows,
    sku_catalog,
    store_audit,
    margin_feed,
    demand_forecast,
    warehouse_scan,
)

SYSTEM: Final = (
    "You are the analytics assistant of a fashion retailer (prices in GBP). Answer from the "
    "tools: fetch the figures you need, compute exactly, and give the number asked. Sales data "
    "covers 2026-01-01 to 2026-09-30. If a question is ambiguous, ask which one is meant. If "
    "the data needed is not available, say so: never invent a figure."
)


# --------------------------------------------------------------------------- truths
def truth_avg_daily_units(
    w: World, sku: str, *, start: str, end: str, region: str | None = None
) -> float:
    days = (_d(end, END) - _d(start, START)).days + 1
    units = sum(s.units for s in selected(w, sku=sku, start=start, end=end, region=region))
    return units / days


def truth_top(
    w: World, group_by: str, metric: str, n: int, **filters: Any
) -> list[tuple[str, float]]:
    found = grouped(selected(w, **filters), w, group_by, metric)
    return sorted(found.items(), key=lambda kv: (-kv[1], kv[0]))[:n]


def truth_total(w: World, metric: str, **filters: Any) -> float:
    found = grouped(selected(w, **filters), w, "region", metric)
    return round(sum(found.values()), 2)


def truth_gross_profit(w: World, **filters: Any) -> float:
    """Revenue less units x unit cost, over the lines selected (a join of 50k lines with the
    catalogue)."""
    return round(
        sum(s.revenue - s.units * w.skus[s.sku].unit_cost for s in selected(w, **filters)), 2
    )


def truth_top_profit_skus(w: World, n: int, **filters: Any) -> list[tuple[str, float]]:
    per: dict[str, float] = defaultdict(float)
    for s in selected(w, **filters):
        per[s.sku] += s.revenue - s.units * w.skus[s.sku].unit_cost
    return sorted(((k, round(v, 2)) for k, v in per.items()), key=lambda kv: (-kv[1], kv[0]))[:n]


def truth_supplier_revenue(w: World, **filters: Any) -> list[tuple[str, float]]:
    found = grouped(selected(w, **filters), w, "supplier", "revenue")
    return sorted(found.items(), key=lambda kv: (-kv[1], kv[0]))


def truth_monthly(w: World, metric: str = "revenue", **filters: Any) -> list[tuple[str, float]]:
    return sorted(grouped(selected(w, **filters), w, "month", metric).items())


def truth_correlation(xs: Sequence[float], ys: Sequence[float]) -> float:
    return statistics.correlation(xs, ys)


def pairs(
    rows: Iterable[tuple[str, float]], pick: Callable[[str], bool] = lambda _: True
) -> list[tuple[str, float]]:
    return [(k, v) for k, v in rows if pick(k)]
