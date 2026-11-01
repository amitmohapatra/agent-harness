"""The tool-chain cases: fourteen shapes of dependent calls, each asked three times — twice in
the ``train`` split, once, reworded and about other records, in the ``heldout`` split (the
split a learned plan is measured on: the memory service learns from the train runs).

=================  ===========================================================================
shape              the chain (``→`` one call's output is the next one's argument)
=================  ===========================================================================
``d2-category``    find_product → style_id → get_style
``d2-region``      find_store → region → region_stores
``d3-lead``        find_product → get_style → supplier_id → supplier_info
``d3-contact``     the same, another field of its answer
``d5-truck``       find_product → get_style → supplier_info → warehouse_info → a date →
                   fiscal_week (five deep; a date read from one answer is the next argument)
``fanout-region``  find_product, region_stores → each store_id → store_stock (one result feeds
                   several calls)
``fanout-store``   find_product, find_store, fiscal_week → store_stock, weekly_sales,
                   latest_allocation (each id feeds three calls)
``fanin-reorder``  stock, a week's sales and the supplier's lead time → reorder_quantity
                   (four deep, three results combined)
``fanin-transfer`` two stores' ids and a SKU → transfer_cost
``error-retry``    … → latest_allocation → shipment_status, which fails once (a 503) and is
                   asked again
``error-report``   the same, the carrier's system down: said, with no date made up
``large-id``       find_store → store_assortment (about 110,000 characters, kept and paged) →
                   the local item's sku_id read from it → latest_allocation
``followup-ship``  turn 1: an allocation's id; turn 2 ("that shipment") → shipment_status
``followup-move``  turn 1: the store holding the most; turn 2: move some "from there"
=================  ===========================================================================

Each turn has its gold rounds of calls (a round's calls are independent: they may run at
once), the answer a scripted model gives (``reference``), and what a right answer holds:
every ``expect`` item — a figure (compared as a number), an id, a name or a date — and none of
the ``forbidden`` patterns; an ``error-report`` answer must also say the lookup failed.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from benchmark.tool_chains import world as w

Call = tuple[str, dict[str, Any]]
Round = tuple[Call, ...]
SPLITS: Final = ("train", "heldout")
#: What an answer saying a lookup failed holds (one of them).
FAILED: Final = re.compile(
    r"\b(down|unavailable|fail\w*|error|could ?n[o']t|cannot|can't|unable|maintenance|outage)\b",
    re.IGNORECASE,
)
#: A date written in an answer (an ``error-report`` must not make one up).
DATE: Final = r"\b20\d\d-\d\d-\d\d\b"


@dataclass(frozen=True)
class Turn:
    text: str
    rounds: tuple[Round, ...]
    reference: str
    #: each a figure, a text, or texts any one of which will do (a store's name or its id)
    expect: tuple[str | float | tuple[str, ...], ...] = ()
    forbidden: tuple[str, ...] = ()
    failed: bool = False  # the answer must say a lookup failed


@dataclass(frozen=True)
class Case:
    id: str
    shape: str
    split: str
    turns: tuple[Turn, ...]
    #: what the case is about (for the report)
    about: dict[str, Any] = field(default_factory=dict)

    @property
    def needed(self) -> list[str]:
        """Every tool its gold calls use, in order."""
        names = [c[0] for t in self.turns for r in t.rounds for c in r]
        return list(dict.fromkeys(names))


# --------------------------------------------------------------------------- gold helpers
def _product(p: str) -> tuple[Call, str, str]:
    sku, style = w.PRODUCTS[p]
    return ("find_product", {"name": p}), sku, style


def _store(s: str) -> tuple[Call, str, str]:
    sid, region = w.STORES[s]
    return ("find_store", {"name": s}), sid, region


def _supplier(style: str) -> tuple[Call, Call, str, tuple[str, int, str, str]]:
    supplier = w.STYLES[style][2]
    return (
        ("get_style", {"style_id": style}),
        ("supplier_info", {"supplier_id": supplier}),
        supplier,
        w.SUPPLIERS[supplier],
    )


def d2_category(p: str, text: str) -> Turn:
    find, _, style = _product(p)
    category = w.STYLES[style][1]
    return Turn(text, ((find,), (("get_style", {"style_id": style}),)), f"{category}.", (category,))


def d2_region(s: str, text: str) -> Turn:
    find, sid, region = _store(s)
    others = [n for n, (i, r) in w.STORES.items() if r == region and i != sid]
    return Turn(
        text,
        ((find,), (("region_stores", {"region": region}),)),
        ", ".join(others) + ".",
        tuple(others),
    )


def d3(p: str, text: str, field_: str) -> Turn:
    find, _, style = _product(p)
    get, info, _, (_, lead, email, _) = _supplier(style)
    value: str | float = float(lead) if field_ == "lead" else email
    return Turn(text, ((find,), (get,), (info,)), f"{value}", (value,))


def d5_truck(p: str, text: str) -> Turn:
    find, _, style = _product(p)
    get, info, _, (*_, warehouse) = _supplier(style)
    truck = w.WAREHOUSES[warehouse][1]
    week = w.week_of(truck)[0]
    return Turn(
        text,
        (
            (find,),
            (get,),
            (info,),
            (("warehouse_info", {"warehouse_id": warehouse}),),
            (("fiscal_week", {"date": truck.isoformat()}),),
        ),
        f"{week} (the truck leaves {truck.isoformat()}).",
        (week,),
    )


def fanout_region(p: str, region: str, text: str) -> Turn:
    find, sku, _ = _product(p)
    stores = [(n, sid) for n, (sid, r) in w.STORES.items() if r == region]
    counts = [(n, w.on_hand(sid, sku)) for n, sid in stores]
    return Turn(
        text,
        (
            (find, ("region_stores", {"region": region})),
            tuple(("store_stock", {"store_id": sid, "sku_id": sku}) for _, sid in stores),
        ),
        "; ".join(f"{n}: {c}" for n, c in counts) + ".",
        tuple(x for n, c in counts for x in (n, float(c))),
    )


def fanout_store(p: str, s: str, day: str, text: str) -> Turn:
    find, sku, _ = _product(p)
    found, sid, _ = _store(s)
    week = w.week_of(_date(day))[0]
    stock = w.on_hand(sid, sku)
    sold = w.units_sold(sid, sku, week)
    status = str(w.allocation(sid, sku)["status"])
    return Turn(
        text,
        (
            (find, found, ("fiscal_week", {"date": day})),
            (
                ("store_stock", {"store_id": sid, "sku_id": sku}),
                ("weekly_sales", {"store_id": sid, "sku_id": sku, "week_id": week}),
                ("latest_allocation", {"store_id": sid, "sku_id": sku}),
            ),
        ),
        f"On hand {stock}, sold {sold} in {week}, latest allocation {status}.",
        (float(stock), float(sold), status.replace("_", " ")),
    )


def fanin_reorder(p: str, s: str, day: str, text: str) -> Turn:
    find, sku, style = _product(p)
    found, sid, _ = _store(s)
    get, info, _, (_, lead, *_) = _supplier(style)
    week = w.week_of(_date(day))[0]
    stock, sold = w.on_hand(sid, sku), w.units_sold(sid, sku, week)
    qty = w.reorder(stock, sold, lead)
    return Turn(
        text,
        (
            (find, found, ("fiscal_week", {"date": day})),
            (
                get,
                ("store_stock", {"store_id": sid, "sku_id": sku}),
                ("weekly_sales", {"store_id": sid, "sku_id": sku, "week_id": week}),
            ),
            (info,),
            (
                (
                    "reorder_quantity",
                    {"on_hand": stock, "weekly_units": sold, "lead_time_days": lead},
                ),
            ),
        ),
        f"Reorder {qty} units.",
        (float(qty),),
    )


def fanin_transfer(p: str, a: str, b: str, units: int, text: str) -> Turn:
    find, sku, _ = _product(p)
    fa, ia, _ = _store(a)
    fb, ib, _ = _store(b)
    cost = w.transfer_price(ia, ib, units)
    move = {"from_store_id": ia, "to_store_id": ib, "sku_id": sku, "units": units}
    return Turn(text, ((fa, fb, find), (("transfer_cost", move),)), f"${cost:.2f}.", (cost,))


def _allocation_rounds(p: str, s: str) -> tuple[tuple[Round, ...], dict[str, Any]]:
    find, sku, _ = _product(p)
    found, sid, _ = _store(s)
    alloc = w.allocation(sid, sku)
    return (
        (find, found),
        (("latest_allocation", {"store_id": sid, "sku_id": sku}),),
    ), alloc


def eta(p: str, s: str, text: str) -> Turn:
    rounds, alloc = _allocation_rounds(p, s)
    aid = str(alloc["allocation_id"])
    ship = ("shipment_status", {"allocation_id": aid})
    if aid in w.DOWN:
        return Turn(
            text,
            (*rounds, (ship,)),
            f"The carrier's tracking system is down, so {aid}'s arrival date is unknown.",
            forbidden=(DATE,),
            failed=True,
        )
    date = str(w.shipment(aid)["eta_date"])
    retried = ((ship,), (ship,)) if aid in w.FLAKY else ((ship,),)
    return Turn(text, (*rounds, *retried), f"{date}.", (date,))


def large_id(s: str, item: str, text: str) -> Turn:
    found, sid, _ = _store(s)
    sku = w.LOCAL_ITEMS[sid][item]
    alloc = w.allocation(sid, sku)
    status, units = str(alloc["status"]), int(alloc["units"])  # type: ignore[call-overload]
    return Turn(
        text,
        (
            (found,),
            (("store_assortment", {"store_id": sid}),),
            (("latest_allocation", {"store_id": sid, "sku_id": sku}),),
        ),
        f"{status}, {units} units.",
        (status.replace("_", " "), float(units)),
    )


def followup_ship(p: str, s: str, first: str, second: str) -> tuple[Turn, Turn]:
    rounds, alloc = _allocation_rounds(p, s)
    aid, units = str(alloc["allocation_id"]), float(alloc["units"])  # type: ignore[arg-type]
    ship = w.shipment(aid)
    return (
        Turn(first, rounds, f"{aid}, {units:g} units.", (aid, units)),
        Turn(
            second,
            ((("shipment_status", {"allocation_id": aid}),),),
            f"{ship['carrier']}, arriving {ship['eta_date']}.",
            (str(ship["carrier"]), str(ship["eta_date"])),
        ),
    )


def followup_move(p: str, to: str, units: int, first: str, second: str) -> tuple[Turn, Turn]:
    find, sku, _ = _product(p)
    stocks = {sid: w.on_hand(sid, sku) for sid in w.STORES_BY_ID}
    top = max(stocks, key=lambda sid: (stocks[sid], sid))
    assert sorted(stocks.values())[-1] != sorted(stocks.values())[-2], "one store holds most"
    top_name = w.STORES_BY_ID[top][0]
    found, sid, _ = _store(to)
    cost = w.transfer_price(top, sid, units)
    move = {"from_store_id": top, "to_store_id": sid, "sku_id": sku, "units": units}
    return (
        Turn(
            first,
            ((find,), (("stock_by_store", {"sku_id": sku}),)),
            f"{top_name}, with {stocks[top]}.",
            ((top_name, top),),
        ),
        Turn(second, ((found,), (("transfer_cost", move),)), f"${cost:.2f}.", (cost,)),
    )


def _date(day: str) -> Any:
    return w.dt.date.fromisoformat(day)


# --------------------------------------------------------------------------- the cases
def _case(shape: str, split: str, n: int, *turns: Turn, **about: Any) -> Case:
    return Case(f"{shape}-{split[0]}{n}", shape, split, tuple(turns), about)


def cases() -> list[Case]:
    """Every case: 28 train (two per shape), 14 held out (one per shape, reworded)."""
    t, h = "train", "heldout"
    return [
        # depth 2
        _case("d2-category", t, 1, d2_category("coastal chino", "What merchandise category is the coastal chino in?")),
        _case("d2-category", t, 2, d2_category("summit rain shell", "Which merchandise category is the summit rain shell in?")),
        _case("d2-category", h, 1, d2_category("prairie flannel shirt", "Prairie flannel shirt: which category does it sit under?")),
        _case("d2-region", t, 1, d2_region("Riverside", "Which other stores are in the same region as Riverside?")),
        _case("d2-region", t, 2, d2_region("Old Town", "Which other stores are in the same region as Old Town?")),
        _case("d2-region", h, 1, d2_region("Pinecrest", "List the stores that share Pinecrest's region, Pinecrest aside.")),
        # depth 3
        _case("d3-lead", t, 1, d3("alpine wool sweater", "How many days does the supplier of the alpine wool sweater take to deliver?", "lead")),
        _case("d3-lead", t, 2, d3("canyon denim jacket", "How many days does the supplier of the canyon denim jacket take to deliver?", "lead")),
        _case("d3-lead", h, 1, d3("meadow cotton tee", "Lead time in days from whoever makes our meadow cotton tee?", "lead")),
        _case("d3-contact", t, 1, d3("summit rain shell", "What is the order contact email of the summit rain shell's supplier?", "email")),
        _case("d3-contact", t, 2, d3("ember fleece hoodie", "What is the order contact email of the ember fleece hoodie's supplier?", "email")),
        _case("d3-contact", h, 1, d3("coastal chino", "I need to email whoever supplies the coastal chino about an order. Which address?", "email")),
        # depth 5, a date passed on
        _case("d5-truck", t, 1, d5_truck("harbor linen shirt", "In which fiscal week does the next truck leave from the warehouse that supplies the harbor linen shirt?")),
        _case("d5-truck", t, 2, d5_truck("alpine wool sweater", "In which fiscal week does the next truck leave from the warehouse that supplies the alpine wool sweater?")),
        _case("d5-truck", h, 1, d5_truck("summit rain shell", "The summit rain shell's supplier ships from a warehouse: what fiscal week id is its next truck in?")),
        # fan-out
        _case("fanout-region", t, 1, fanout_region("canyon denim jacket", "north", "How many canyon denim jackets are on hand at each store in the north region?")),
        _case("fanout-region", t, 2, fanout_region("coastal chino", "south", "How many coastal chinos are on hand at each store in the south region?")),
        _case("fanout-region", h, 1, fanout_region("ember fleece hoodie", "north", "Ember fleece hoodie stock, store by store, across the north region?")),
        _case("fanout-store", t, 1, fanout_store("alpine wool sweater", "Lakeview", "2026-10-14", "For the alpine wool sweater at Lakeview: units on hand, units sold in the week of 2026-10-14, and the status of its latest allocation.")),
        _case("fanout-store", t, 2, fanout_store("coastal chino", "Midtown", "2026-09-30", "For the coastal chino at Midtown: units on hand, units sold in the week of 2026-09-30, and the status of its latest allocation.")),
        _case("fanout-store", h, 1, fanout_store("summit rain shell", "Westgate", "2026-10-21", "Summit rain shell at Westgate — what's on hand, how many sold the week of 2026-10-21, and where does its latest allocation stand?")),
        # fan-in
        _case("fanin-reorder", t, 1, fanin_reorder("summit rain shell", "Riverside", "2026-10-14", "How many summit rain shells should Riverside reorder, using its sales in the week of 2026-10-14 and the supplier's lead time?")),
        _case("fanin-reorder", t, 2, fanin_reorder("alpine wool sweater", "Old Town", "2026-10-07", "How many alpine wool sweaters should Old Town reorder, using its sales in the week of 2026-10-07 and the supplier's lead time?")),
        _case("fanin-reorder", h, 1, fanin_reorder("prairie flannel shirt", "Midtown", "2026-10-20", "Work out Midtown's reorder quantity for the prairie flannel shirt from the week of 2026-10-20's sales and its supplier's lead time.")),
        _case("fanin-transfer", t, 1, fanin_transfer("harbor linen shirt", "Riverside", "Old Town", 20, "What would it cost to move 20 harbor linen shirts from Riverside to Old Town?")),
        _case("fanin-transfer", t, 2, fanin_transfer("coastal chino", "Lakeview", "Pinecrest", 12, "What would it cost to move 12 coastal chinos from Lakeview to Pinecrest?")),
        _case("fanin-transfer", h, 1, fanin_transfer("canyon denim jacket", "Midtown", "Westgate", 8, "Quote me a transfer of 8 canyon denim jackets, Midtown to Westgate.")),
        # a middle step errors
        _case("error-retry", t, 1, eta("coastal chino", "Old Town", "When is the coastal chino allocation for Old Town expected to arrive?")),
        _case("error-retry", t, 2, eta("summit rain shell", "Lakeview", "When is the summit rain shell allocation for Lakeview expected to arrive?")),
        _case("error-retry", h, 1, eta("meadow cotton tee", "Midtown", "What's the arrival date of Midtown's latest meadow cotton tee allocation?")),
        _case("error-report", t, 1, eta("alpine wool sweater", "Riverside", "When is the alpine wool sweater allocation for Riverside expected to arrive?")),
        _case("error-report", t, 2, eta("canyon denim jacket", "Westgate", "When is the canyon denim jacket allocation for Westgate expected to arrive?")),
        _case("error-report", h, 1, eta("ember fleece hoodie", "Harbor Point", "What's the arrival date of Harbor Point's latest ember fleece hoodie allocation?")),
        # the id in a large kept result
        _case("large-id", t, 1, large_id("Riverside", "Driftwood Candle", "Riverside sells a local item called Driftwood Candle. What is the status of its latest allocation there, and how many units?")),
        _case("large-id", t, 2, large_id("Old Town", "Clay Mug Set", "Old Town sells a local item called Clay Mug Set. What is the status of its latest allocation there, and how many units?")),
        _case("large-id", h, 1, large_id("Midtown", "Copper Tea Tin", "Midtown stocks its own Copper Tea Tin. Where does that item's latest allocation stand, and for how many units?")),
        # a follow-up reusing the earlier chain's ids
        _case("followup-ship", t, 1, *followup_ship("harbor linen shirt", "Pinecrest", "What's the latest allocation of the harbor linen shirt to Pinecrest? Give its id and units.", "Which carrier has that shipment, and when does it arrive?")),
        _case("followup-ship", t, 2, *followup_ship("prairie flannel shirt", "Old Town", "What's the latest allocation of the prairie flannel shirt to Old Town? Give its id and units.", "Which carrier has that shipment, and when does it arrive?")),
        _case("followup-ship", h, 1, *followup_ship("canyon denim jacket", "Riverside", "Latest canyon denim jacket allocation to Riverside — id and units please.", "Who's carrying it and what's the ETA?")),
        _case("followup-move", t, 1, *followup_move("alpine wool sweater", "Westgate", 10, "Which store holds the most alpine wool sweaters?", "What would moving 10 of them from there to Westgate cost?")),
        _case("followup-move", t, 2, *followup_move("summit rain shell", "Old Town", 15, "Which store holds the most summit rain shells?", "What would moving 15 of them from there to Old Town cost?")),
        _case("followup-move", h, 1, *followup_move("coastal chino", "Old Town", 6, "Where do we have the most coastal chinos?", "Price a transfer of 6 from that store to Old Town.")),
    ]  # fmt: skip


# --------------------------------------------------------------------------- scoring
_NUMBER = re.compile(r"(?<![\w.-])-?\$?(\d[\d,]*(?:\.\d+)?)(?![\w]|-\d)")


def numbers(text: str) -> list[float]:
    """The figures an answer states (an id's or a date's digits are not figures)."""
    return [float(m.group(1).replace(",", "")) for m in _NUMBER.finditer(text)]


def _said(item: str | float | tuple[str, ...], text: str) -> bool:
    if isinstance(item, tuple):
        return any(_said(one, text) for one in item)
    if isinstance(item, int | float):
        return any(abs(n - item) < 0.005 for n in numbers(text))
    norm = text.casefold().replace("_", " ")
    return item.casefold() in norm


def scored(turn: Turn, answer: str | None) -> dict[str, Any]:
    """Whether ``answer`` is right for ``turn``, and what it lacks or wrongly holds."""
    text = answer or ""
    missing = [str(i) for i in turn.expect if not _said(i, text)]
    wrong = [p for p in turn.forbidden if re.search(p, text)]
    unsaid = turn.failed and not FAILED.search(text)
    return {
        "correct": bool(text) and not missing and not wrong and not unsaid,
        "missing": missing,
        "forbidden": wrong,
        "failure_unsaid": unsaid,
    }


def split(chosen: Sequence[str], every: Sequence[Case] | None = None) -> list[Case]:
    return [c for c in (every if every is not None else cases()) if c.split in chosen]
