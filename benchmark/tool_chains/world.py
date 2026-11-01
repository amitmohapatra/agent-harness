"""The retail allocation world the tool-chain cases run on: its data, and its read-only tools,
each with a typed output (a ``TypedDict``) the way a real API documents its responses, so the
harness's links (``tools.links``) see what each tool returns.

The ids are what a retail planning system uses — a product's ``sku_id`` and ``style_id``, a
store's ``store_id`` and region, a fiscal ``week_id``, an ``allocation_id`` and its shipment,
a supplier and the warehouse it ships from. No question names an id: each is read from one
tool's answer and given to the next (a date too: a warehouse's next truck is a date a fiscal
week is looked up by). Two tools fail as real ones do (:data:`DOWN`, :data:`FLAKY`), and one
answers a store's whole assortment, about 110,000 characters (:func:`store_assortment`), which
the harness keeps and pages rather than sends.
"""

from __future__ import annotations

import datetime as dt
import math
import random
from contextvars import ContextVar
from datetime import date, timedelta
from typing import Final, TypedDict

from trellis import tool

# --------------------------------------------------------------------------- the data
#: name -> (sku_id, style_id)
PRODUCTS: Final = {
    "harbor linen shirt": ("SKU-1041", "STY-210"),
    "alpine wool sweater": ("SKU-2207", "STY-330"),
    "canyon denim jacket": ("SKU-3150", "STY-415"),
    "meadow cotton tee": ("SKU-4012", "STY-210"),
    "summit rain shell": ("SKU-5233", "STY-520"),
    "coastal chino": ("SKU-6088", "STY-610"),
    "ember fleece hoodie": ("SKU-7120", "STY-330"),
    "prairie flannel shirt": ("SKU-8301", "STY-705"),
}
#: style_id -> (style name, category, supplier_id)
STYLES: Final = {
    "STY-210": ("Harbor basics", "tops", "SUP-14"),
    "STY-330": ("Alpine knits", "knitwear", "SUP-22"),
    "STY-415": ("Canyon denim", "outerwear", "SUP-31"),
    "STY-520": ("Summit tech", "outerwear", "SUP-47"),
    "STY-610": ("Coastal trousers", "bottoms", "SUP-14"),
    "STY-705": ("Prairie workwear", "tops", "SUP-58"),
}
#: supplier_id -> (name, lead time in days, contact email, warehouse_id)
SUPPLIERS: Final = {
    "SUP-14": ("Bayline Textiles", 9, "orders@bayline.example", "WH-03"),
    "SUP-22": ("Nordfell Knitting", 21, "supply@nordfell.example", "WH-07"),
    "SUP-31": ("Mesa Denim Works", 14, "trade@mesadenim.example", "WH-03"),
    "SUP-47": ("Peakform Apparel", 30, "b2b@peakform.example", "WH-11"),
    "SUP-58": ("Ridgeway Mills", 12, "sales@ridgeway.example", "WH-07"),
}
#: warehouse_id -> (city, next truck date)
WAREHOUSES: Final = {
    "WH-03": ("Stockton", date(2026, 11, 4)),
    "WH-07": ("Reno", date(2026, 11, 17)),
    "WH-11": ("Boise", date(2026, 12, 2)),
}
#: name -> (store_id, region)
STORES: Final = {
    "Riverside": ("ST-11", "north"),
    "Lakeview": ("ST-12", "north"),
    "Pinecrest": ("ST-13", "north"),
    "Old Town": ("ST-21", "south"),
    "Harbor Point": ("ST-22", "south"),
    "Midtown": ("ST-31", "east"),
    "Westgate": ("ST-41", "west"),
}
#: The fiscal year starts on this Monday (week 1); weeks run Monday to Sunday.
FY_START: Final = date(2026, 2, 2)
#: Transfer cost: a fixed handling fee per transfer plus a fee per unit, by whether the stores
#: share a region.
TRANSFER_FEE: Final = 35.0
PER_UNIT: Final = {True: 0.8, False: 2.25}
#: Local items a store sells that are not in the product catalogue (only its assortment has
#: them): store_id -> {item name: sku_id}.
LOCAL_ITEMS: Final = {
    "ST-11": {"Driftwood Candle": "LOC-11-0457", "Cedar Soap Bar": "LOC-11-0913"},
    "ST-21": {"Clay Mug Set": "LOC-21-0288", "Juniper Room Spray": "LOC-21-1102"},
    "ST-31": {"Copper Tea Tin": "LOC-31-0641"},
}
#: Rows of a store's assortment (about 110,000 characters: kept and paged, not sent whole).
ASSORTMENT_ROWS: Final = 1_200


def _seed(*parts: str) -> random.Random:
    return random.Random("|".join(parts))


def on_hand(store_id: str, sku_id: str) -> int:
    return _seed("stock", store_id, sku_id).randint(0, 60)


def week_of(day: date) -> tuple[str, date, date]:
    """A date's fiscal week: its id (``FY26-W14``), first and last day."""
    n = (day - FY_START).days // 7 + 1
    start = FY_START + timedelta(days=7 * (n - 1))
    return f"FY26-W{n:02d}", start, start + timedelta(days=6)


def units_sold(store_id: str, sku_id: str, week_id: str) -> int:
    return _seed("sales", store_id, sku_id, week_id).randint(3, 40)


def allocation(store_id: str, sku_id: str) -> dict[str, object]:
    r = _seed("allocation", store_id, sku_id)
    number = r.randint(1000, 9999)
    return {
        "allocation_id": f"AL-{number}",
        "store_id": store_id,
        "sku_id": sku_id,
        "units": r.randint(6, 48),
        "status": r.choice(["planned", "picked", "in_transit"]),
    }


CARRIERS: Final = ("Northline Freight", "Coastway Logistics", "Ridge Express")


def shipment(allocation_id: str) -> dict[str, object]:
    r = _seed("shipment", allocation_id)
    return {
        "allocation_id": allocation_id,
        "shipment_id": f"SH-{r.randint(10000, 99999)}",
        "carrier": r.choice(CARRIERS),
        "eta_date": (date(2026, 11, 2) + timedelta(days=r.randint(1, 20))).isoformat(),
    }


def assortment(store_id: str) -> list[dict[str, object]]:
    """A store's whole assortment: every catalogue SKU, its local items, and generic lines."""
    r = _seed("assortment", store_id)
    rows: list[dict[str, object]] = [
        {"sku_id": sku, "item_name": name.title(), "on_hand": on_hand(store_id, sku)}
        for name, (sku, _) in PRODUCTS.items()
    ]
    for name, sku in LOCAL_ITEMS.get(store_id, {}).items():
        rows.append({"sku_id": sku, "item_name": name, "on_hand": on_hand(store_id, sku)})
    kinds = ("Gift Card", "Tote Bag", "Sock Pack", "Water Bottle", "Notebook", "Beanie", "Scarf")
    while len(rows) < ASSORTMENT_ROWS:
        n = len(rows)
        sku = f"GEN-{store_id[3:]}-{n:04d}"
        rows.append(
            {
                "sku_id": sku,
                "item_name": f"{r.choice(kinds)} {n:04d}",
                "on_hand": on_hand(store_id, sku),
                "unit_price": round(r.uniform(3, 90), 2),
                "department": r.choice(["accessories", "home", "gifts", "seasonal"]),
            }
        )
    r.shuffle(rows)
    return rows


def transfer_price(from_store_id: str, to_store_id: str, units: int) -> float:
    same = STORES_BY_ID[from_store_id][1] == STORES_BY_ID[to_store_id][1]
    return round(TRANSFER_FEE + PER_UNIT[same] * units, 2)


def reorder(on_hand_units: int, weekly_units: int, lead_time_days: int) -> int:
    return max(0, math.ceil(weekly_units / 7 * lead_time_days) - on_hand_units)


STORES_BY_ID: Final = {sid: (name, region) for name, (sid, region) in STORES.items()}
SKUS: Final = {sku: (name, style) for name, (sku, style) in PRODUCTS.items()}


# --------------------------------------------------------------------------- failures
def _allocations(*pairs: tuple[str, str]) -> frozenset[str]:
    """The ids of the latest allocations of (store name, product name) pairs."""
    return frozenset(
        str(allocation(STORES[store][0], PRODUCTS[product][0])["allocation_id"])
        for store, product in pairs
    )


#: Allocations whose shipment lookup fails the first time it is asked in a run (a 503 from
#: the carrier gateway) and then answers.
FLAKY: Final = _allocations(
    ("Old Town", "coastal chino"),
    ("Lakeview", "summit rain shell"),
    ("Midtown", "meadow cotton tee"),
)
#: Allocations whose shipment lookup always fails (the carrier's system is down).
DOWN: Final = _allocations(
    ("Riverside", "alpine wool sweater"),
    ("Westgate", "canyon denim jacket"),
    ("Harbor Point", "ember fleece hoodie"),
)
#: How many times each flaky allocation was asked in the current case (:func:`reset`, in the
#: case's task: cases run at once keep their own counts).
_ASKED: ContextVar[dict[str, int]] = ContextVar("chains_asked")


def reset() -> None:
    _ASKED.set({})


# --------------------------------------------------------------------------- the outputs
class Product(TypedDict):
    sku_id: str
    style_id: str
    name: str


class Style(TypedDict):
    style_id: str
    style_name: str
    category: str
    supplier_id: str


class Supplier(TypedDict):
    supplier_id: str
    supplier_name: str
    lead_time_days: int
    contact_email: str
    warehouse_id: str


class Warehouse(TypedDict):
    warehouse_id: str
    city: str
    next_truck_date: str


class Store(TypedDict):
    store_id: str
    name: str
    region: str


class RegionStore(TypedDict):
    store_id: str
    name: str


class Region(TypedDict):
    region: str
    stores: list[RegionStore]


class StoreStock(TypedDict):
    store_id: str
    sku_id: str
    on_hand: int


class StockAt(TypedDict):
    store_id: str
    on_hand: int


class StockByStore(TypedDict):
    sku_id: str
    stores: list[StockAt]


class FiscalWeek(TypedDict):
    week_id: str
    start_date: str
    end_date: str


class WeeklySales(TypedDict):
    store_id: str
    sku_id: str
    week_id: str
    units: int


class Allocation(TypedDict):
    allocation_id: str
    store_id: str
    sku_id: str
    units: int
    status: str


class Shipment(TypedDict):
    allocation_id: str
    shipment_id: str
    carrier: str
    eta_date: str


class AssortmentItem(TypedDict):
    sku_id: str
    item_name: str
    on_hand: int


class Assortment(TypedDict):
    store_id: str
    items: list[AssortmentItem]


class TransferQuote(TypedDict):
    from_store_id: str
    to_store_id: str
    sku_id: str
    units: int
    cost_usd: float


# --------------------------------------------------------------------------- the tools
def _known(table: dict[str, object], key: str, what: str) -> str:
    found = key.strip().upper()
    if found not in table:
        raise ValueError(f"no {what} {key!r}")
    return found


@tool(side_effects="read")
def find_product(name: str) -> Product:
    """Find a catalogue product by its name: its SKU and its style.

    Args:
        name: The product's name, such as "coastal chino".
    """
    key = name.strip().casefold().removesuffix("s")
    hit = PRODUCTS.get(key) or PRODUCTS.get(name.strip().casefold())
    if hit is None:
        raise ValueError(f"no catalogue product is named {name!r}")
    return {"sku_id": hit[0], "style_id": hit[1], "name": name}


@tool(side_effects="read")
def get_style(style_id: str) -> Style:
    """A style's name, merchandise category and the supplier that makes it.

    Args:
        style_id: The style's id, such as STY-610.
    """
    sid = _known(STYLES, style_id, "style")  # type: ignore[arg-type]
    name, category, supplier = STYLES[sid]
    return {"style_id": sid, "style_name": name, "category": category, "supplier_id": supplier}


@tool(side_effects="read")
def supplier_info(supplier_id: str) -> Supplier:
    """A supplier's name, its lead time in days, its order contact and the warehouse it ships
    from.

    Args:
        supplier_id: The supplier's id, such as SUP-14.
    """
    sid = _known(SUPPLIERS, supplier_id, "supplier")  # type: ignore[arg-type]
    name, lead, email, warehouse = SUPPLIERS[sid]
    return {
        "supplier_id": sid,
        "supplier_name": name,
        "lead_time_days": lead,
        "contact_email": email,
        "warehouse_id": warehouse,
    }


@tool(side_effects="read")
def warehouse_info(warehouse_id: str) -> Warehouse:
    """A distribution warehouse's city and the date its next truck leaves.

    Args:
        warehouse_id: The warehouse's id, such as WH-03.
    """
    wid = _known(WAREHOUSES, warehouse_id, "warehouse")  # type: ignore[arg-type]
    city, truck = WAREHOUSES[wid]
    return {"warehouse_id": wid, "city": city, "next_truck_date": truck.isoformat()}


@tool(side_effects="read")
def find_store(name: str) -> Store:
    """Find a store by its name: its id and its region.

    Args:
        name: The store's name, such as "Old Town".
    """
    hit = next((v for k, v in STORES.items() if k.casefold() == name.strip().casefold()), None)
    if hit is None:
        raise ValueError(f"no store is named {name!r}")
    return {"store_id": hit[0], "name": name, "region": hit[1]}


@tool(side_effects="read")
def region_stores(region: str) -> Region:
    """Every store of a region.

    Args:
        region: The region: north, south, east or west.
    """
    wanted = region.strip().casefold()
    stores = [{"store_id": sid, "name": n} for n, (sid, reg) in STORES.items() if reg == wanted]
    if not stores:
        raise ValueError(f"no region {region!r}")
    return {"region": wanted, "stores": stores}  # type: ignore[typeddict-item]


@tool(side_effects="read")
def store_stock(store_id: str, sku_id: str) -> StoreStock:
    """The units of one SKU on hand at one store.

    Args:
        store_id: The store's id, such as ST-21.
        sku_id: The SKU's id, such as SKU-6088.
    """
    store = _known(STORES_BY_ID, store_id, "store")  # type: ignore[arg-type]
    sku = sku_id.strip().upper()
    return {"store_id": store, "sku_id": sku, "on_hand": on_hand(store, sku)}


@tool(side_effects="read")
def stock_by_store(sku_id: str) -> StockByStore:
    """The units of one catalogue SKU on hand at every store.

    Args:
        sku_id: The SKU's id, such as SKU-6088.
    """
    sku = _known(SKUS, sku_id, "catalogue SKU")  # type: ignore[arg-type]
    rows = [{"store_id": sid, "on_hand": on_hand(sid, sku)} for sid in STORES_BY_ID]
    return {"sku_id": sku, "stores": rows}  # type: ignore[typeddict-item]


@tool(side_effects="read")
def fiscal_week(date: str) -> FiscalWeek:
    """The fiscal week a calendar date falls in: its id and its first and last day.

    Args:
        date: The date, as YYYY-MM-DD.
    """
    try:
        day = dt.date.fromisoformat(date.strip())
    except ValueError as exc:
        raise ValueError(f"not a YYYY-MM-DD date: {date!r}") from exc
    week, start, end = week_of(day)
    return {"week_id": week, "start_date": start.isoformat(), "end_date": end.isoformat()}


@tool(side_effects="read")
def weekly_sales(store_id: str, sku_id: str, week_id: str) -> WeeklySales:
    """Units of one SKU sold at one store in one fiscal week.

    Args:
        store_id: The store's id, such as ST-21.
        sku_id: The SKU's id, such as SKU-6088.
        week_id: The fiscal week's id, such as FY26-W36.
    """
    store = _known(STORES_BY_ID, store_id, "store")  # type: ignore[arg-type]
    sku, week = sku_id.strip().upper(), week_id.strip().upper()
    if not week.startswith("FY26-W"):
        raise ValueError(f"not a fiscal week id: {week_id!r}")
    return {
        "store_id": store,
        "sku_id": sku,
        "week_id": week,
        "units": units_sold(store, sku, week),
    }


@tool(side_effects="read")
def latest_allocation(store_id: str, sku_id: str) -> Allocation:
    """The latest allocation of a SKU to a store: its id, units and status.

    Args:
        store_id: The store's id, such as ST-21.
        sku_id: The SKU's id (a catalogue SKU or a store's local item).
    """
    store = _known(STORES_BY_ID, store_id, "store")  # type: ignore[arg-type]
    return allocation(store, sku_id.strip().upper())  # type: ignore[return-value]


@tool(side_effects="read")
def shipment_status(allocation_id: str) -> Shipment:
    """Where an allocation's shipment is: its shipment id, carrier and estimated arrival date.

    Args:
        allocation_id: The allocation's id, such as AL-4821.
    """
    aid = allocation_id.strip().upper()
    if aid in DOWN:
        raise RuntimeError("carrier tracking system is down for maintenance (HTTP 502)")
    asked = _ASKED.get(None)
    if asked is None:
        asked = {}
        _ASKED.set(asked)
    asked[aid] = asked.get(aid, 0) + 1
    if aid in FLAKY and asked[aid] == 1:
        raise ConnectionError("carrier gateway: 503 Service Unavailable")
    return shipment(aid)  # type: ignore[return-value]


@tool(side_effects="read")
def store_assortment(store_id: str) -> Assortment:
    """Every item a store sells, its local items included, with the units on hand.

    Args:
        store_id: The store's id, such as ST-21.
    """
    store = _known(STORES_BY_ID, store_id, "store")  # type: ignore[arg-type]
    return {"store_id": store, "items": assortment(store)}  # type: ignore[typeddict-item]


@tool(side_effects="read")
def transfer_cost(from_store_id: str, to_store_id: str, sku_id: str, units: int) -> TransferQuote:
    """What moving units of a SKU from one store to another costs, in US dollars.

    Args:
        from_store_id: The id of the store the units leave, such as ST-11.
        to_store_id: The id of the store that receives them, such as ST-21.
        sku_id: The SKU's id, such as SKU-6088.
        units: How many units move.
    """
    source = _known(STORES_BY_ID, from_store_id, "store")  # type: ignore[arg-type]
    target = _known(STORES_BY_ID, to_store_id, "store")  # type: ignore[arg-type]
    if source == target:
        raise ValueError("a transfer needs two different stores")
    return {
        "from_store_id": source,
        "to_store_id": target,
        "sku_id": sku_id.strip().upper(),
        "units": units,
        "cost_usd": transfer_price(source, target, units),
    }


@tool(side_effects="read")
def reorder_quantity(on_hand: int, weekly_units: int, lead_time_days: int) -> int:
    """Units to reorder so stock lasts until the supplier delivers: the week's sales a day
    times the lead time, less what is on hand, rounded up, never below 0.

    Args:
        on_hand: Units on hand now.
        weekly_units: Units sold in a week.
        lead_time_days: Days the supplier takes to deliver.
    """
    return reorder(on_hand, weekly_units, lead_time_days)


TOOLS: Final = [
    find_product,
    get_style,
    supplier_info,
    warehouse_info,
    find_store,
    region_stores,
    store_stock,
    stock_by_store,
    fiscal_week,
    weekly_sales,
    latest_allocation,
    shipment_status,
    store_assortment,
    transfer_cost,
    reorder_quantity,
]
SYSTEM: Final = (
    "You are a retail allocation assistant. Answer from the tools: never guess an id, a date "
    "or a number. When a tool fails, retry it once if the error may pass; otherwise say what "
    "failed. Write numbers as plain digits and dates as YYYY-MM-DD."
)
