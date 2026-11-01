"""The ``complex`` suite: the questions a ChatGPT or Claude user asks an analytics assistant —
multi-hop tool chains, joins and statistics over 50,000 sales lines, follow-ups, corrections,
explanations, fresh values, ambiguity, missing data, conflicting sources and broken tools —
each scored against an answer Python computed from the same data
(``benchmark.cost.retail``).

    python -m benchmark.cost.complex --scripted --out build/complex-fake      # pipeline proof
    LLM_BASE_URL=https://api.anthropic.com/v1/ LLM_API_KEY=... \\
      python -m benchmark.cost.complex --model claude-haiku-5-5 --max-usd 3 --out build/cx

**Cases** (:func:`cases`): each is one conversation — its turns (a turn may run on another
thread, the cross-chat questions; the world's clock may tick before it, the freshness ones),
the gold tool calls of each turn (the ``--scripted`` model replays them), and, on the scored
turns, an :class:`Expect`: the ground truth and how it is scored —

* ``number`` / ``record`` / ``set`` — exact figures (within the rounding stated), every field,
  every id (and none of the ``forbidden`` ones);
* ``abstain`` / ``clarify`` / ``flags`` — says the data is not there (with no figure made up),
  asks which one is meant (with no figure given), or says the tool failed, timed out, came
  back empty or broken;
* ``tools`` / ``steps`` — names every tool an earlier turn called (and their count), from that
  turn's own record;
* ``judge`` — free text (why, how sure): no strict score; the judge pass rules on it.

**A run** writes ``<out>.jsonl`` (a row per case: every turn's answer, calls, cost and score),
``<out>.requests.jsonl`` (every model request as sent: messages and tools offered) and
``<out>.json`` / ``<out>.md`` (accuracy by category and tier, cost). ``benchmark.cost.judge``
turns these into judge packets. ``--max-usd`` stops the run once the usage the responses
reported, at the model's listed price (``benchmark.cost.prices``), reaches it — a call is not
started past it. ``--cases`` runs only the cases named, ``--rerun-failed`` only those whose
strict score failed or whose turn did not end well.

**Adversarial cases** (``benchmark.cost.adversarial``, categories ``adv_*``, ``--cases
adversarial``): what tries to break the harness. Besides the answer's check, each turn carries the
harness's own checks (``row["checks"]``, :func:`checked`): how its run must end (``Turn.ends``) and
what the model must and must not have been sent (``Turn.sent``, :class:`Probe`) — what a scripted
model cannot hide. A turn may be another person's (``Turn.user``) or sent while the one before it
works (``Turn.mode``: queued, interrupting, steering); a ``grounded`` reason may cite only figures
the model was sent (:func:`ungrounded`).

**History** (``--history``): ``thread`` (default; offline) — ``create_agent`` with the
harness's middleware and an in-process checkpointer, so the conversation is the framework's
own thread, memory off; ``memory`` — ``ReAct`` with the memory service of ``MEMORY_URL`` (its
window and context; a cross-chat question can be answered only this way).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from typing import Any, Final

from langchain_core.messages import convert_to_openai_messages

from benchmark.cost import prices, stats
from benchmark.cost import retail as r
from benchmark.cost import run as runner
from trellis import Harness, Hooks, current
from trellis.contracts import ToolCall, ToolOutcome

Call = tuple[str, dict[str, Any]]

CATEGORIES: Final = (
    "multihop",
    "aggregate",
    "formula",
    "stats",
    "whatif",
    "conversation",
    "explain",
    "freshness",
    "ambiguity",
    "nodata",
    "conflict",
    "edge",
    "planning",
    "cantdo",
    # the adversarial cases (``benchmark.cost.adversarial``): ``--cases adversarial``
    "adv_conversation",
    "adv_data",
    "adv_explain",
    "adv_workflow",
    "adv_resource",
    "adv_scenario",
    "adv_files",
    "adv_pipeline",
    "adv_domain",
    # the break-and-fix rounds (``benchmark.cost.breakfix``): ``--cases breakfix``
    "bf_past",
    "bf_analytics",
    "bf_explain",
    "bf_mixed",
    "bf_cantdo",
    "bf_clarify",
)
ADVERSARIAL: Final = "adversarial"
TIERS: Final = (1, 2, 3)
STRICT_KINDS: Final = (
    "number", "record", "set", "abstain", "clarify", "flags", "tools", "steps", "cannot",
    "grounded",
)  # fmt: skip
#: The reorder formula as every reorder question states it.
FORMULA: Final = (
    "reorder qty = average daily units sold over 2026-09-03..2026-09-30 (28 days) x (supplier "
    "lead time + {review} review days) - units on hand, rounded up, never below 0"
)
WEIGHTED: Final = "score = 0.6 x on-time rate + 0.4 x (1 - lead time days / 60)"
SEP28: Final = ("2026-09-03", "2026-09-30")


# --------------------------------------------------------------------------- the cases
@dataclass(frozen=True)
class Expect:
    """A scored turn's ground truth and how it is scored (the module says each kind)."""

    kind: str
    value: Any = None
    #: how far a figure may be from the truth (absolute)
    tol: float = 0.0
    #: what must not be in the answer: a wrong candidate, a figure of a wrong reading
    forbidden: tuple[str, ...] = ()
    #: for the judge: what a correct answer says
    rubric: str = ""
    #: the ground truth in words (how it was computed)
    truth: str = ""
    #: ``tools`` / ``steps``: the turn (its index in the case) whose record is asked about
    of_turn: int | None = None
    #: what the scripted model answers (the reference answer)
    reference: str = ""


@dataclass(frozen=True)
class Probe:
    """A check of what the harness sent the model during a turn (every request of it, as
    sent): the context, which a scripted model cannot hide, not the answer. Kinds:

    * ``has`` / ``lacks`` — some request holds ``pattern`` (a regular expression, any case) /
      none does, in its messages (of ``role`` only, when given: ``user``, ``assistant``,
      ``tool``, ``system``; an assistant's tool calls count as its text);
    * ``nonblank`` — every user message of every request has text (a provider refuses an
      empty or blank one: the thread could never be continued);
    * ``repeats`` — the turn's last request holds ``pattern`` in at least ``limit`` messages;
    * ``fits`` — every request holds at most ``limit`` tokens (counted approximately);
    * ``calls`` — the turn made at most ``limit`` model calls;
    * ``called`` — at least ``limit`` calls of the tool ``pattern`` ended ``ok``.
    * ``uncalled`` — no call of the tool ``pattern`` (the wrong source for the question).

    ``when``: the history it holds for (``memory``: what only the memory service can give,
    such as another chat's turns); unset, every one."""

    kind: str
    pattern: str = ""
    why: str = ""
    role: str | None = None
    limit: int | None = None
    when: str | None = None


#: How a turn may be sent while the turn before it still works (``Turn.mode``): the harness's
#: busy-conversation modes (``trellis.harness.conversation.Mode``).
BUSY: Final = ("queue", "interrupt", "steer")


@dataclass(frozen=True)
class Turn:
    text: str
    calls: tuple[tuple[Call, ...], ...] = ()  # the gold rounds of tool calls
    expect: Expect | None = None
    thread: str = "a"
    ticks: int = 0  # the world's clock ticks this many times before the turn
    reply: str = ""  # the scripted model's answer of an unscored turn
    #: who sends it (default: the benchmark's user, ``run.USER``)
    user: str | None = None
    #: what the harness must have sent the model during the turn (:class:`Probe`)
    sent: tuple[Probe, ...] = ()
    #: how its run may end (a run's status)
    ends: tuple[str, ...] = ("SUCCESS",)
    #: sent while the turn before it works (once that one's first tool call started), with
    #: this busy-conversation mode (:data:`BUSY`); ``None``: after it ended
    mode: str | None = None


@dataclass(frozen=True)
class Case:
    id: str
    category: str
    tier: int
    turns: tuple[Turn, ...]
    edges: tuple[str, ...] = ()  # the edge cases it covers
    #: tools offered beside the retailer's (``retail.TOOLS``)
    tools: tuple[Any, ...] = ()
    #: the adapters it applies to (empty: every one), and why not the others
    adapters: tuple[str, ...] = ()
    only_because: str = ""
    #: the model's context window for this case (over ``--window``)
    window: int | None = None
    #: the scripted model answers each call by the message it answers (``KeyedChatModel``),
    #: not in call order: for runs that interleave (a steer, an interrupt, helpers)
    keyed: bool = False
    #: the scripted rounds of the helpers it starts: each helper's task, as the gold call
    #: words it, with its rounds (the answer last)
    helpers: tuple[tuple[str, tuple[Any, ...]], ...] = ()

    def script(self) -> list[Any]:
        """What the scripted model answers, call by call: each turn's gold rounds, then its
        reference answer — none for a turn the next one interrupts or steers (its run is
        cancelled, or the steered message's answer is its answer)."""
        out: list[Any] = []
        for n, t in enumerate(self.turns):
            out.extend(list(rnd) for rnd in t.calls)
            later = self.turns[n + 1] if n + 1 < len(self.turns) else None
            if later is None or later.mode not in ("interrupt", "steer"):
                out.append((t.expect.reference if t.expect else t.reply) or "Done.")
        return out


def fmt(v: float, places: int = 2) -> str:
    return f"{v:,.{places}f}"


def money(v: float, **kw: Any) -> Expect:
    return Expect("number", round(v, 2), tol=max(0.5, abs(v) * 0.001), reference=f"£{fmt(v)}", **kw)


def pct(v: float, **kw: Any) -> Expect:
    return Expect("number", round(v, 4), tol=0.06, reference=f"{v:.2f}%", **kw)


def count(v: int, **kw: Any) -> Expect:
    return Expect("number", v, tol=0.0, reference=f"{v:,}", **kw)


def ratio(v: float, places: int = 2, **kw: Any) -> Expect:
    return Expect(
        "number", round(v, 6), tol=0.6 * 10**-places, reference=f"{v:.{places + 1}f}", **kw
    )


def record(fields: dict[str, Any], tol: dict[str, float] | None = None, **kw: Any) -> Expect:
    """Every field: a number (within ``tol[field]``, else 0.5% or 0.5), or a string — or a
    tuple of strings, any of which will do (an id or its name)."""
    tol = tol or {}
    ref = "; ".join(
        f"{k}: {v[0] if isinstance(v, tuple) else (_shown(v) if isinstance(v, float) else v)}"
        for k, v in fields.items()
    )
    value = {
        k: {"value": v, "tol": tol.get(k, max(0.5, abs(v) * 0.005))} if _num(v) else {"value": v}
        for k, v in fields.items()
    }
    ref = kw.pop("reference", ref)
    return Expect("record", value, reference=ref, **kw)


def idset(
    ids: Sequence[str],
    forbidden: Sequence[str] = (),
    names: dict[str, str] | None = None,
    **kw: Any,
) -> Expect:
    names = names or {}
    value = [[i, names[i]] if i in names else [i] for i in ids]
    ref = kw.pop("reference", ", ".join(ids))
    return Expect("set", value, forbidden=tuple(forbidden), reference=ref, **kw)


def abstain(why: str, **kw: Any) -> Expect:
    return Expect("abstain", reference=f"I can't answer that from the data: {why}", **kw)


def clarify(candidates: Sequence[str], forbidden: Sequence[str] = (), **kw: Any) -> Expect:
    ref = "Which one do you mean: " + ", ".join(candidates[:4]) + "?"
    return Expect("clarify", list(candidates), forbidden=tuple(forbidden), reference=ref, **kw)


def flags(pattern: str, reference: str, **kw: Any) -> Expect:
    return Expect("flags", pattern, reference=reference, **kw)


def cannot(fake: str, offer: str, reference: str, **kw: Any) -> Expect:
    """The request needs what this deployment has not (code execution, files, images): correct
    when the answer says it cannot do it here (:data:`_CANNOT`), why (:data:`_WHY`), and what
    it can do instead (``offer``, a pattern), with no made-up output (``fake``, a pattern, nor
    :data:`_FAKE`)."""
    return Expect("cannot", {"fake": fake, "offer": offer}, reference=reference, **kw)


def grounded(says: str, rubric: str, reference: str, cites: int = 2, **kw: Any) -> Expect:
    """A free-text reason (why, how sure, what would change it, the risks): strict on what can
    be checked — it says ``says`` (a pattern), cites at least ``cites`` figures, and every
    figure it gives is one the model was sent (:func:`ungrounded`: none made up) — and the
    judge rules on the ``rubric``."""
    return Expect("grounded", cites, truth=says, rubric=rubric, reference=reference, **kw)


def judge(rubric: str, reference: str, **kw: Any) -> Expect:
    return Expect("judge", rubric=rubric, reference=reference, **kw)


def _shown(v: float) -> str:
    return fmt(v) if abs(v) >= 100 else f"{v:.4f}"


def _num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _alt(sku: str) -> tuple[str, str]:
    return (sku, r.world().skus[sku].name)


def _calls_named(case_turn: Turn) -> list[str]:
    return list(dict.fromkeys(name for rnd in case_turn.calls for name, _ in rnd))


def _tools_ref(t: Turn) -> str:
    names = _calls_named(t)
    n = sum(len(rnd) for rnd in t.calls)
    return f"I used {', '.join(names)}: {n} tool calls in all."


@cache
def cases() -> tuple[Case, ...]:
    """Every case of the suite, built from the world's data."""
    w = r.world()
    out: list[Case] = []
    add = out.append
    s = w.by_name

    def avg28(sku: str, region: str | None = None) -> float:
        return r.truth_avg_daily_units(w, sku, start=SEP28[0], end=SEP28[1], region=region)

    def reorder(
        sku: str, region: str | None, review: int = 7, lead: int | None = None, version: int = 0
    ) -> int:
        sup = w.suppliers[w.skus[sku].supplier]
        regions = (region,) if region else r.REGIONS
        on = sum(w.on_hand(sku, g, version) for g in regions)
        return r.reorder_qty(
            avg28(sku, region), lead if lead is not None else sup.lead_time_days, review, on
        )

    def total_on(sku: str, version: int = 0) -> int:
        return sum(w.on_hand(sku, g, version) for g in r.REGIONS)

    def units_args(sku: str, region: str | None = None) -> dict[str, Any]:
        a = {"group_by": "sku", "metric": "units", "sku": sku, "start": SEP28[0], "end": SEP28[1]}
        return a | ({"region": region} if region else {})

    def margin(sku: str) -> float:
        k = w.skus[sku]
        return r.margin_pct(k.price, k.unit_cost)

    def fast(category: str, n: int = 0) -> r.Sku:
        """The ``n``-th fastest seller of a category over the formula's 28 days."""
        ranked = r.truth_top(
            w, "sku", "units", n + 1, category=category, start=SEP28[0], end=SEP28[1]
        )
        return w.skus[ranked[n][0]]

    Q2, Q3 = ("2026-04-01", "2026-06-30"), ("2026-07-01", "2026-09-30")
    q3 = {"start": Q3[0], "end": Q3[1]}
    q2 = {"start": Q2[0], "end": Q2[1]}
    ytd = {"start": "2026-01-01", "end": "2026-09-30"}
    sep = {"start": "2026-09-01", "end": "2026-09-30"}

    # ------------------------------------------------------------------ multihop
    navy = s("Merino Crew Sweater Navy").sku
    add(
        Case(
            "multihop-01",
            "multihop",
            1,
            (
                Turn(
                    "How many units of the Merino Crew Sweater Navy do we have on hand in total right now?",
                    (
                        (("find_sku", {"query": "Merino Crew Sweater Navy"}),),
                        (("stock_level", {"sku": navy}),),
                    ),
                    count(total_on(navy), truth="stock_level total, all regions"),
                ),
            ),
        )
    )
    parka = s("Waxed Parka Olive")
    sup = w.suppliers[parka.supplier]
    add(
        Case(
            "multihop-02",
            "multihop",
            2,
            (
                Turn(
                    "Who supplies the Waxed Parka Olive, and what is their lead time in days?",
                    (
                        (("find_sku", {"query": "Waxed Parka Olive"}),),
                        (("sku_details", {"sku": parka.sku}),),
                        (("supplier_info", {"supplier": sup.supplier}),),
                    ),
                    record(
                        {
                            "supplier": (sup.supplier, sup.name),
                            "lead_time_days": sup.lead_time_days,
                        },
                        tol={"lead_time_days": 0},
                        truth="sku_details -> supplier_info (master)",
                    ),
                ),
            ),
        )
    )
    top_rev = r.truth_top(w, "sku", "revenue", 1)[0][0]
    sup = w.suppliers[w.skus[top_rev].supplier]
    add(
        Case(
            "multihop-03",
            "multihop",
            1,
            (
                Turn(
                    f"What's the minimum order quantity of the supplier of {top_rev}, and which city are they in?",
                    (
                        (("sku_details", {"sku": top_rev}),),
                        (("supplier_info", {"supplier": sup.supplier}),),
                    ),
                    record(
                        {"min_order_qty": sup.min_order_qty, "city": sup.city},
                        tol={"min_order_qty": 0},
                    ),
                ),
            ),
        )
    )
    card = fast("knitwear")
    sup = w.suppliers[card.supplier]
    add(
        Case(
            "multihop-04",
            "multihop",
            3,
            (
                Turn(
                    f"How many {card.name} should we reorder for the north region? Use: "
                    f"{FORMULA.format(review=7)} (north sales and north stock only).",
                    (
                        (("find_sku", {"query": card.name}),),
                        (("sku_details", {"sku": card.sku}),),
                        (
                            ("supplier_info", {"supplier": sup.supplier}),
                            ("stock_level", {"sku": card.sku, "region": "north"}),
                            ("sales_summary", units_args(card.sku, "north")),
                        ),
                    ),
                    count(
                        reorder(card.sku, "north"),
                        truth="find_sku -> sku_details -> supplier_info + stock_level + sales_summary",
                    ),
                ),
            ),
            ("multi-hop 4 tools",),
        )
    )
    top_units = r.truth_top(w, "sku", "units", 1, **sep)[0][0]
    sup = w.suppliers[w.skus[top_units].supplier]
    add(
        Case(
            "multihop-05",
            "multihop",
            3,
            (
                Turn(
                    f"Take our best-selling SKU by units in September 2026. How much should we reorder of it "
                    f"across all regions? {FORMULA.format(review=7)}.",
                    (
                        (
                            (
                                "sales_summary",
                                {"group_by": "sku", "metric": "units", **sep, "top": 1},
                            ),
                        ),
                        (("sku_details", {"sku": top_units}),),
                        (
                            ("supplier_info", {"supplier": sup.supplier}),
                            ("stock_level", {"sku": top_units}),
                            ("sales_summary", units_args(top_units)),
                        ),
                    ),
                    record(
                        {"sku": _alt(top_units), "reorder_qty": reorder(top_units, None)},
                        tol={"reorder_qty": 0},
                    ),
                ),
            ),
            ("multi-hop 4 tools",),
        )
    )
    boot = fast("footwear")
    q3_days = 92
    south_avg = r.truth_avg_daily_units(w, boot.sku, start=Q3[0], end=Q3[1], region="south")
    cover = w.on_hand(boot.sku, "south") / south_avg
    add(
        Case(
            "multihop-06",
            "multihop",
            2,
            (
                Turn(
                    f"How many days of stock do we have of the {boot.name} in the south, at "
                    "its average daily south sales over Q3 (Jul 1 - Sep 30, 92 days)?",
                    (
                        (("find_sku", {"query": boot.name}),),
                        (
                            ("stock_level", {"sku": boot.sku, "region": "south"}),
                            (
                                "sales_summary",
                                {
                                    "group_by": "sku",
                                    "metric": "units",
                                    "sku": boot.sku,
                                    "region": "south",
                                    **q3,
                                },
                            ),
                        ),
                    ),
                    ratio(cover, 1, truth=f"on hand / (Q3 south units / {q3_days})"),
                ),
            ),
        )
    )
    foot = [k for k in w.skus.values() if k.category == "footwear"]
    by_sup: dict[str, int] = {}
    for k in foot:
        by_sup[k.supplier] = by_sup.get(k.supplier, 0) + 1
    ranked = sorted(by_sup.items(), key=lambda kv: (-kv[1], kv[0]))
    best_sup = w.suppliers[ranked[0][0]]
    add(
        Case(
            "multihop-07",
            "multihop",
            2,
            (
                Turn(
                    "Which supplier supplies the most footwear SKUs, how many, and what is its on-time rate?",
                    (
                        (("sku_catalog", {"category": "footwear"}),),
                        (("supplier_info", {"supplier": best_sup.supplier}),),
                    ),
                    record(
                        {
                            "supplier": (best_sup.supplier, best_sup.name),
                            "skus": ranked[0][1],
                            "on_time_rate": best_sup.on_time_rate,
                        },
                        tol={"skus": 0, "on_time_rate": 0.005},
                        truth=f"runner-up {ranked[1]}",
                    ),
                ),
            ),
        )
    )
    north_store = r.truth_top(w, "store", "revenue", 1, region="north", **q3)[0][0]
    store_top = r.truth_top(w, "sku", "units", 1, store=north_store, **q3)[0][0]
    add(
        Case(
            "multihop-08",
            "multihop",
            3,
            (
                Turn(
                    "For our best north store by Q3 revenue: which SKU sold the most units there in Q3, and "
                    "how many of that SKU does the north have on hand now?",
                    (
                        (
                            (
                                "sales_summary",
                                {"group_by": "store", "region": "north", **q3, "top": 1},
                            ),
                        ),
                        (
                            (
                                "sales_summary",
                                {
                                    "group_by": "sku",
                                    "metric": "units",
                                    "store": north_store,
                                    **q3,
                                    "top": 1,
                                },
                            ),
                        ),
                        (("stock_level", {"sku": store_top, "region": "north"}),),
                    ),
                    record(
                        {
                            "store": north_store,
                            "sku": _alt(store_top),
                            "north_on_hand": w.on_hand(store_top, "north"),
                        },
                        tol={"north_on_hand": 0},
                    ),
                ),
            ),
            ("multi-hop 3 tools",),
        )
    )
    dress = next(k for k in w.skus.values() if k.category == "dresses" and k.price_before)
    add(
        Case(
            "multihop-09",
            "multihop",
            2,
            (
                Turn(
                    f"What's the current price of the {dress.name}, and what was it in March?",
                    (
                        (("find_sku", {"query": dress.name}),),
                        (("price_history", {"sku": dress.sku}),),
                    ),
                    record(
                        {"now": dress.price, "march": dress.price_before},
                        tol={"now": 0.01, "march": 0.01},
                    ),
                ),
            ),
        )
    )
    leg = fast("activewear")
    low = min(r.REGIONS, key=lambda g: (w.on_hand(leg.sku, g), g))
    sep_avg = r.truth_avg_daily_units(w, leg.sku, region=low, **sep)
    days = w.on_hand(leg.sku, low) / sep_avg if sep_avg else float("inf")
    add(
        Case(
            "multihop-10",
            "multihop",
            3,
            (
                Turn(
                    f"Which region has the fewest {leg.name} on hand, and how many days would that "
                    "stock last at that region's average daily units in September 2026 (30 days)?",
                    (
                        (("find_sku", {"query": leg.name}),),
                        (("stock_level", {"sku": leg.sku}),),
                        (
                            (
                                "sales_summary",
                                {
                                    "group_by": "sku",
                                    "metric": "units",
                                    "sku": leg.sku,
                                    "region": low,
                                    **sep,
                                },
                            ),
                        ),
                    ),
                    record({"region": low, "days": round(days, 1)}, tol={"days": 0.06}),
                ),
            ),
        )
    )

    # ------------------------------------------------------------------ aggregate / joins
    add(
        Case(
            "aggregate-01",
            "aggregate",
            1,
            (
                Turn(
                    "What was total revenue in the north region in Q3 2026 (Jul-Sep)?",
                    ((("sales_summary", {"group_by": "region", "region": "north", **q3}),),),
                    money(r.truth_total(w, "revenue", region="north", **q3)),
                ),
            ),
        )
    )
    aug = {"start": "2026-08-01", "end": "2026-08-31"}
    top10 = r.truth_top(w, "sku", "revenue", 10, category="knitwear", **aug)
    add(
        Case(
            "aggregate-02",
            "aggregate",
            2,
            (
                Turn(
                    "List the top 5 knitwear SKUs by revenue in August 2026.",
                    (
                        (
                            (
                                "sales_summary",
                                {"group_by": "sku", "category": "knitwear", **aug, "top": 5},
                            ),
                        ),
                    ),
                    idset([k for k, _ in top10[:5]], [k for k, _ in top10[5:]]),
                ),
            ),
        )
    )
    add(
        Case(
            "aggregate-03",
            "aggregate",
            3,
            (
                Turn(
                    "What was the gross profit of denim in Q2 2026 — revenue minus units x unit cost, over "
                    "every sales line?",
                    (
                        (
                            ("sales_rows", {"category": "denim", **q2}),
                            ("sku_catalog", {"category": "denim"}),
                        ),
                    ),
                    money(
                        r.truth_gross_profit(w, category="denim", **q2),
                        truth="join of the Q2 denim lines with the catalogue",
                    ),
                ),
            ),
            ("join 10k+ rows",),
        )
    )
    tp = r.truth_top_profit_skus(w, 5, **ytd)
    add(
        Case(
            "aggregate-04",
            "aggregate",
            3,
            (
                Turn(
                    "Which 3 SKUs made the most gross profit (revenue minus units x unit cost) over all of "
                    "2026's sales lines so far?",
                    ((("sales_rows", ytd), ("sku_catalog", {})),),
                    idset(
                        [k for k, _ in tp[:3]],
                        [k for k, _ in tp[3:5]],
                        truth="join of all 50,000 lines",
                    ),
                ),
            ),
            ("join 50k rows",),
        )
    )
    sr = r.truth_supplier_revenue(w, **ytd)
    add(
        Case(
            "aggregate-05",
            "aggregate",
            2,
            (
                Turn(
                    "Which supplier's products brought in the most revenue in 2026 so far, and how much?",
                    ((("sales_summary", {"group_by": "supplier", **ytd, "top": 3}),),),
                    record({"supplier": sr[0][0], "revenue": sr[0][1]}, tol={"revenue": 1.0}),
                ),
            ),
        )
    )
    west_skus = len(r.grouped(r.selected(w, region="west", **sep), w, "sku", "units"))
    add(
        Case(
            "aggregate-06",
            "aggregate",
            2,
            (
                Turn(
                    "How many distinct SKUs sold at least one unit in the west in September 2026?",
                    ((("sales_summary", {"group_by": "sku", "region": "west", **sep}),),),
                    count(west_skus),
                ),
            ),
        )
    )
    south_units = r.grouped(r.selected(w, region="south", **ytd), w, "store", "units")
    south_lines = r.grouped(r.selected(w, region="south", **ytd), w, "store", "lines")
    basket = {k: south_units[k] / south_lines[k] for k in south_units}
    best_b = max(basket, key=lambda k: (basket[k], k))
    add(
        Case(
            "aggregate-07",
            "aggregate",
            3,
            (
                Turn(
                    "Which south store has the highest average units per sales line in 2026 so far, and what is it?",
                    (
                        (
                            (
                                "sales_summary",
                                {"group_by": "store", "metric": "units", "region": "south", **ytd},
                            ),
                            (
                                "sales_summary",
                                {"group_by": "store", "metric": "lines", "region": "south", **ytd},
                            ),
                        ),
                    ),
                    record(
                        {"store": best_b, "units_per_line": round(basket[best_b], 3)},
                        tol={"units_per_line": 0.006},
                    ),
                ),
            ),
        )
    )
    all_rev = r.truth_total(w, "revenue", **ytd)
    top20 = sum(v for _, v in r.truth_top(w, "sku", "revenue", 20, **ytd))
    add(
        Case(
            "aggregate-08",
            "aggregate",
            2,
            (
                Turn(
                    "What share of 2026's revenue so far came from the top 20 SKUs, in percent?",
                    (
                        (
                            ("sales_summary", {"group_by": "sku", **ytd, "top": 20}),
                            ("sales_summary", {"group_by": "region", **ytd}),
                        ),
                    ),
                    pct(top20 / all_rev * 100),
                ),
            ),
        )
    )

    # ------------------------------------------------------------------ formulas
    add(
        Case(
            "formula-01",
            "formula",
            1,
            (
                Turn(
                    f"What's the gross margin % of {top_rev} at its current price? (price - unit cost) / price.",
                    ((("sku_details", {"sku": top_rev}),),),
                    pct(margin(top_rev)),
                ),
            ),
        )
    )
    nq2 = r.truth_total(w, "revenue", region="north", **q2)
    nq3 = r.truth_total(w, "revenue", region="north", **q3)
    add(
        Case(
            "formula-02",
            "formula",
            2,
            (
                Turn(
                    "By what percent did north revenue grow from Q2 to Q3 2026?",
                    (
                        (
                            ("sales_summary", {"group_by": "region", "region": "north", **q2}),
                            ("sales_summary", {"group_by": "region", "region": "north", **q3}),
                        ),
                    ),
                    pct(r.growth_pct(nq2, nq3)),
                ),
            ),
        )
    )
    a, b = w.suppliers["SUP-07"], w.suppliers["SUP-12"]
    add(
        Case(
            "formula-03",
            "formula",
            2,
            (
                Turn(
                    f"Score suppliers SUP-07 and SUP-12 with {WEIGHTED}, using the supplier master. Which is higher?",
                    (
                        (
                            ("supplier_info", {"supplier": "SUP-07"}),
                            ("supplier_info", {"supplier": "SUP-12"}),
                        ),
                    ),
                    record(
                        {
                            "SUP-07": round(r.supplier_score(a), 4),
                            "SUP-12": round(r.supplier_score(b), 4),
                        },
                        tol={"SUP-07": 0.0006, "SUP-12": 0.0006},
                    ),
                ),
            ),
        )
    )
    scores = sorted(w.suppliers.values(), key=lambda x: (-r.supplier_score(x), x.supplier))
    add(
        Case(
            "formula-04",
            "formula",
            3,
            (
                Turn(
                    f"Of all 25 suppliers (SUP-01 to SUP-25), which scores highest with {WEIGHTED}? Give its score.",
                    (tuple(("supplier_info", {"supplier": f"SUP-{n:02d}"}) for n in range(1, 26)),),
                    record(
                        {
                            "supplier": scores[0].supplier,
                            "score": round(r.supplier_score(scores[0]), 4),
                        },
                        tol={"score": 0.0006},
                        truth=f"runner-up {scores[1].supplier}",
                    ),
                ),
            ),
            ("25 parallel calls",),
        )
    )
    fm = dict(r.truth_monthly(w, category="footwear"))
    add(
        Case(
            "formula-05",
            "formula",
            2,
            (
                Turn(
                    "Month-over-month growth of footwear revenue from August to September 2026, in percent?",
                    (
                        (
                            (
                                "sales_summary",
                                {
                                    "group_by": "month",
                                    "category": "footwear",
                                    "start": "2026-08-01",
                                    "end": "2026-09-30",
                                },
                            ),
                        ),
                    ),
                    pct(r.growth_pct(fm["2026-08"], fm["2026-09"])),
                ),
            ),
        )
    )
    ku = r.grouped(r.selected(w, category="knitwear", **q3), w, "sku", "units")
    wm = sum(margin(k) * u for k, u in ku.items()) / sum(ku.values())
    add(
        Case(
            "formula-06",
            "formula",
            3,
            (
                Turn(
                    "What's the average gross margin % of knitwear, weighted by Q3 2026 units sold (current "
                    "prices and costs)?",
                    (
                        (
                            (
                                "sales_summary",
                                {
                                    "group_by": "sku",
                                    "metric": "units",
                                    "category": "knitwear",
                                    **q3,
                                },
                            ),
                            ("sku_catalog", {"category": "knitwear"}),
                        ),
                    ),
                    pct(wm, truth="sum(margin x units) / sum(units) over knitwear SKUs sold in Q3"),
                ),
            ),
            ("join",),
        )
    )

    # ------------------------------------------------------------------ stats
    s07 = r.daily(w, store="S07", **q3)
    add(
        Case(
            "stats-01",
            "stats",
            2,
            (
                Turn(
                    "What's the 90th percentile of store S07's daily revenue in Q3 2026 (all 92 days, a day "
                    "with no sales counts as 0; linear interpolation)?",
                    ((("sales_summary", {"group_by": "date", "store": "S07", **q3}),),),
                    money(r.percentile(s07, 90)),
                ),
            ),
        )
    )
    monthly = [v for _, v in r.truth_monthly(w)]
    add(
        Case(
            "stats-02",
            "stats",
            2,
            (
                Turn(
                    "What's the sample standard deviation of monthly revenue, January to September 2026?",
                    ((("sales_summary", {"group_by": "month", **ytd}),),),
                    money(__import__("statistics").stdev(monthly)),
                ),
            ),
        )
    )
    kn = r.daily(w, category="knitwear", **q3)
    ou = r.daily(w, category="outerwear", **q3)
    add(
        Case(
            "stats-03",
            "stats",
            3,
            (
                Turn(
                    "What's the Pearson correlation between knitwear's and outerwear's daily revenue in Q3 2026 "
                    "(92 days, 0 on a day with no sales)?",
                    (
                        (
                            ("sales_summary", {"group_by": "date", "category": "knitwear", **q3}),
                            ("sales_summary", {"group_by": "date", "category": "outerwear", **q3}),
                        ),
                    ),
                    ratio(r.truth_correlation(kn, ou), 2),
                ),
            ),
        )
    )
    slope, icept = r.regression(list(range(1, 10)), monthly)
    add(
        Case(
            "stats-04",
            "stats",
            3,
            (
                Turn(
                    "Fit a least-squares line to monthly revenue (January = 1 ... September = 9). What's the "
                    "slope per month, and the line's forecast for October (month 10)?",
                    ((("sales_summary", {"group_by": "month", **ytd}),),),
                    record(
                        {"slope": round(slope, 2), "october": round(slope * 10 + icept, 2)},
                        tol={
                            "slope": abs(slope) * 0.002,
                            "october": abs(slope * 10 + icept) * 0.002,
                        },
                    ),
                ),
            ),
        )
    )
    per_store = r.grouped(r.selected(w, **ytd), w, "store", "revenue")
    out_stores = r.iqr_outliers(per_store)
    add(
        Case(
            "stats-05",
            "stats",
            2,
            (
                Turn(
                    "Which stores are outliers by 2026 revenue so far, using the 1.5 x IQR rule (quartiles by "
                    "linear interpolation)?",
                    ((("sales_summary", {"group_by": "store", **ytd}),),),
                    idset(out_stores, [k for k in per_store if k not in out_stores][:5]),
                ),
            ),
        )
    )
    east = [v for k, v in per_store.items() if w.region_of(k) == "east"]
    add(
        Case(
            "stats-06",
            "stats",
            1,
            (
                Turn(
                    "What's the median 2026 revenue so far per store in the east region?",
                    ((("sales_summary", {"group_by": "store", "region": "east", **ytd}),),),
                    money(__import__("statistics").median(east)),
                ),
            ),
        )
    )
    kunits = r.grouped(r.selected(w, category="knitwear", **ytd), w, "sku", "units")
    xs = [w.skus[k].price for k in sorted(kunits)]
    ys = [kunits[k] for k in sorted(kunits)]
    add(
        Case(
            "stats-07",
            "stats",
            3,
            (
                Turn(
                    "Across knitwear SKUs that sold in 2026, what's the Pearson correlation between current "
                    "price and units sold so far?",
                    (
                        (
                            (
                                "sales_summary",
                                {
                                    "group_by": "sku",
                                    "metric": "units",
                                    "category": "knitwear",
                                    **ytd,
                                },
                            ),
                            ("sku_catalog", {"category": "knitwear"}),
                        ),
                    ),
                    ratio(r.truth_correlation(xs, ys), 2),
                ),
            ),
            ("join",),
        )
    )
    sepd = r.daily(w, **sep)
    cv = __import__("statistics").pstdev(sepd) / __import__("statistics").fmean(sepd) * 100
    add(
        Case(
            "stats-08",
            "stats",
            2,
            (
                Turn(
                    "What's the coefficient of variation (population std dev / mean) of total daily revenue in "
                    "September 2026, in percent?",
                    ((("sales_summary", {"group_by": "date", **sep}),),),
                    pct(cv),
                ),
            ),
        )
    )

    # ------------------------------------------------------------------ what-if
    tq3 = r.truth_total(w, "revenue", sku=top_rev, **q3)
    add(
        Case(
            "whatif-01",
            "whatif",
            2,
            (
                Turn(
                    f"If {top_rev} had sold the same units in Q3 2026 at prices 10% higher, how much more "
                    "revenue would we have made?",
                    ((("sales_summary", {"group_by": "sku", "sku": top_rev, **q3}),),),
                    money(tq3 * 0.1),
                ),
            ),
        )
    )
    cs = w.suppliers[card.supplier]
    add(
        Case(
            "whatif-02",
            "whatif",
            3,
            (
                Turn(
                    f"For the {card.name} (all regions): {FORMULA.format(review=7)}. What is the "
                    "reorder qty now, and what would it be if its supplier's lead time grew by 7 days?",
                    (
                        (("find_sku", {"query": card.name}),),
                        (("sku_details", {"sku": card.sku}),),
                        (
                            ("supplier_info", {"supplier": cs.supplier}),
                            ("stock_level", {"sku": card.sku}),
                            ("sales_summary", units_args(card.sku)),
                        ),
                    ),
                    record(
                        {
                            "now": reorder(card.sku, None),
                            "plus_7_days": reorder(card.sku, None, lead=cs.lead_time_days + 7),
                        },
                        tol={"now": 0, "plus_7_days": 0},
                    ),
                ),
            ),
        )
    )
    fq3 = r.truth_gross_profit(w, category="footwear", **q3)
    fcost = sum(x.units * w.skus[x.sku].unit_cost for x in r.selected(w, category="footwear", **q3))
    add(
        Case(
            "whatif-03",
            "whatif",
            3,
            (
                Turn(
                    "If every footwear unit cost had been 5% higher, what would footwear's Q3 2026 gross profit "
                    "(revenue minus units x unit cost) have been?",
                    (
                        (
                            ("sales_rows", {"category": "footwear", **q3}),
                            ("sku_catalog", {"category": "footwear"}),
                        ),
                    ),
                    money(fq3 - 0.05 * fcost),
                ),
            ),
            ("join",),
        )
    )
    nm = dict(r.truth_monthly(w, region="north"))
    g = nm["2026-08"] / nm["2026-07"]
    add(
        Case(
            "whatif-04",
            "whatif",
            2,
            (
                Turn(
                    "If north revenue had grown from August to September at the same rate as from July to "
                    "August, what would north's September 2026 revenue have been?",
                    (
                        (
                            (
                                "sales_summary",
                                {
                                    "group_by": "month",
                                    "region": "north",
                                    "start": "2026-07-01",
                                    "end": "2026-09-30",
                                },
                            ),
                        ),
                    ),
                    money(nm["2026-08"] * g),
                ),
            ),
        )
    )

    # ------------------------------------------------------------------ conversation
    south_top = r.truth_top(w, "sku", "revenue", 1, region="south", **q3)[0][0]
    st_sup = w.suppliers[w.skus[south_top].supplier]
    add(
        Case(
            "conversation-01",
            "conversation",
            2,
            (
                Turn(
                    "What was the top SKU by revenue in the south in Q3 2026?",
                    ((("sales_summary", {"group_by": "sku", "region": "south", **q3, "top": 1}),),),
                    record({"sku": _alt(south_top)}),
                ),
                Turn(
                    "Who supplies it?",
                    ((("sku_details", {"sku": south_top}),),),
                    record({"supplier": st_sup.supplier}),
                ),
                Turn(
                    "And what's their lead time?",
                    ((("supplier_info", {"supplier": st_sup.supplier}),),),
                    count(st_sup.lead_time_days),
                ),
            ),
            ("pronoun follow-ups",),
        )
    )
    jul = {"start": "2026-07-01", "end": "2026-07-31"}
    add(
        Case(
            "conversation-02",
            "conversation",
            1,
            (
                Turn(
                    "Total knitwear revenue in July 2026?",
                    (
                        (
                            (
                                "sales_summary",
                                {"group_by": "category", "category": "knitwear", **jul},
                            ),
                        ),
                    ),
                    money(r.truth_total(w, "revenue", category="knitwear", **jul)),
                ),
                Turn(
                    "Sorry, I meant August.",
                    (
                        (
                            (
                                "sales_summary",
                                {"group_by": "category", "category": "knitwear", **aug},
                            ),
                        ),
                    ),
                    money(
                        r.truth_total(w, "revenue", category="knitwear", **aug),
                        forbidden=(fmt(r.truth_total(w, "revenue", category="knitwear", **jul)),),
                    ),
                ),
            ),
            ("correction",),
        )
    )
    cats = r.truth_top(w, "category", "revenue", 5, **q3)
    # the region whose top three differ from the whole's (the added constraint changes the answer)
    reg = next(
        g
        for g in ("north", "east", "west", "south")
        if [k for k, _ in r.truth_top(w, "category", "revenue", 3, region=g, **q3)]
        != [k for k, _ in cats[:3]]
    )
    ncats = r.truth_top(w, "category", "revenue", 5, region=reg, **q3)
    add(
        Case(
            "conversation-03",
            "conversation",
            2,
            (
                Turn(
                    "Top 3 categories by revenue in Q3 2026?",
                    ((("sales_summary", {"group_by": "category", **q3}),),),
                    idset([k for k, _ in cats[:3]], [k for k, _ in cats[3:]]),
                ),
                Turn(
                    f"Only for the {reg} region.",
                    ((("sales_summary", {"group_by": "category", "region": reg, **q3}),),),
                    idset([k for k, _ in ncats[:3]], [k for k, _ in ncats[3:]]),
                ),
            ),
            ("constraint added later",),
        )
    )
    trench = s("Classic Trench Coat Camel")
    add(
        Case(
            "conversation-04",
            "conversation",
            2,
            (
                Turn(
                    "How many Classic Trench Coat Camel do we have in total?",
                    (
                        (("find_sku", {"query": "Classic Trench Coat Camel"}),),
                        (("stock_level", {"sku": trench.sku}),),
                    ),
                    count(total_on(trench.sku)),
                ),
                Turn(
                    "Unrelated: how many stores do we have in the west?",
                    ((("store_list", {"region": "west"}),),),
                    count(10),
                ),
                Turn(
                    "Back to that coat — what's its gross margin %?",
                    ((("sku_details", {"sku": trench.sku}),),),
                    pct(margin(trench.sku)),
                ),
            ),
            ("topic switch and return",),
        )
    )
    top3s = r.truth_top(w, "store", "revenue", 3, **sep)
    second = top3s[1][0]
    add(
        Case(
            "conversation-05",
            "conversation",
            2,
            (
                Turn(
                    "Top 3 stores by revenue in September 2026?",
                    ((("sales_summary", {"group_by": "store", **sep, "top": 3}),),),
                    idset([k for k, _ in top3s]),
                ),
                Turn(
                    "What was the second one's revenue in August?",
                    ((("sales_summary", {"group_by": "store", "store": second, **aug}),),),
                    money(r.truth_total(w, "revenue", store=second, **aug)),
                ),
            ),
            ("ordinal reference",),
        )
    )
    cream = s("Merino Crew Sweater Cream").sku
    add(
        Case(
            "conversation-06",
            "conversation",
            1,
            (
                Turn(
                    "What's the gross margin % of the Merino Crew Sweater Navy?",
                    (
                        (("find_sku", {"query": "Merino Crew Sweater Navy"}),),
                        (("sku_details", {"sku": navy}),),
                    ),
                    pct(margin(navy)),
                ),
                Turn(
                    "No — I mean the Cream one.",
                    (
                        (("find_sku", {"query": "Merino Crew Sweater Cream"}),),
                        (("sku_details", {"sku": cream}),),
                    ),
                    pct(margin(cream)),
                ),
            ),
            ("correction",),
        )
    )
    jun = {"start": "2026-06-01", "end": "2026-06-30"}
    add(
        Case(
            "conversation-07",
            "conversation",
            2,
            (
                Turn(
                    "How many activewear units did we sell in Q2 2026?",
                    (
                        (
                            (
                                "sales_summary",
                                {
                                    "group_by": "category",
                                    "metric": "units",
                                    "category": "activewear",
                                    **q2,
                                },
                            ),
                        ),
                    ),
                    count(int(r.truth_total(w, "units", category="activewear", **q2))),
                ),
                Turn(
                    "Just the east.",
                    (
                        (
                            (
                                "sales_summary",
                                {
                                    "group_by": "category",
                                    "metric": "units",
                                    "category": "activewear",
                                    "region": "east",
                                    **q2,
                                },
                            ),
                        ),
                    ),
                    count(
                        int(r.truth_total(w, "units", category="activewear", region="east", **q2))
                    ),
                ),
                Turn(
                    "And only June.",
                    (
                        (
                            (
                                "sales_summary",
                                {
                                    "group_by": "category",
                                    "metric": "units",
                                    "category": "activewear",
                                    "region": "east",
                                    **jun,
                                },
                            ),
                        ),
                    ),
                    count(
                        int(r.truth_total(w, "units", category="activewear", region="east", **jun))
                    ),
                ),
            ),
            ("constraints accumulate",),
        )
    )
    q1 = {"start": "2026-01-01", "end": "2026-03-31"}
    wq1 = r.truth_total(w, "revenue", region="west", **q1)
    eq1 = r.truth_total(w, "revenue", region="east", **q1)
    add(
        Case(
            "conversation-08",
            "conversation",
            3,
            (
                Turn(
                    "What was the west's revenue in Q1 2026?",
                    ((("sales_summary", {"group_by": "region", "region": "west", **q1}),),),
                    money(wq1),
                ),
                Turn(
                    "Which supplier is SUP-04?",
                    ((("supplier_info", {"supplier": "SUP-04"}),),),
                    reply=w.suppliers["SUP-04"].name,
                ),
                Turn(
                    "What does the Silk Scarf Navy cost us?",
                    ((("find_sku", {"query": "Silk Scarf Navy"}),),),
                    reply="See the catalogue.",
                ),
                Turn(
                    "How many stores in the south?",
                    ((("store_list", {"region": "south"}),),),
                    reply="10.",
                ),
                Turn(
                    "What's the top category in August?",
                    ((("sales_summary", {"group_by": "category", **aug, "top": 1}),),),
                    reply="See above.",
                ),
                Turn(
                    "Any audit findings at S12?",
                    ((("store_audit", {"store": "S12"}),),),
                    reply="None recorded.",
                ),
                Turn(
                    "Thanks. What's the price of SKU-00042?",
                    ((("sku_details", {"sku": "SKU-00042"}),),),
                    reply=f"£{w.skus['SKU-00042'].price}",
                ),
                Turn(
                    "Going back to the west's Q1 revenue you told me at the start: how much higher or lower "
                    "was the east's Q1 revenue, in percent of the west's?",
                    ((("sales_summary", {"group_by": "region", "region": "east", **q1}),),),
                    pct(r.growth_pct(wq1, eq1)),
                ),
            ),
            ("long chat (8 turns)", "recall of an early figure"),
        )
    )
    wb, qb = s("Waxed Parka Black").sku, s("Quilted Parka Black").sku
    add(
        Case(
            "conversation-09",
            "conversation",
            2,
            (
                Turn(
                    "Compare the Waxed Parka Black and the Quilted Parka Black: Q3 2026 units of each.",
                    (
                        (("find_sku", {"query": "Parka Black"}),),
                        (
                            (
                                "sales_summary",
                                {"group_by": "sku", "metric": "units", "sku": wb, **q3},
                            ),
                            (
                                "sales_summary",
                                {"group_by": "sku", "metric": "units", "sku": qb, **q3},
                            ),
                        ),
                    ),
                    record(
                        {
                            "waxed": int(r.truth_total(w, "units", sku=wb, **q3)),
                            "quilted": int(r.truth_total(w, "units", sku=qb, **q3)),
                        },
                        tol={"waxed": 0, "quilted": 0},
                    ),
                ),
                Turn(
                    "Which of them has the higher gross margin %, and by how many percentage points?",
                    ((("sku_details", {"sku": wb}), ("sku_details", {"sku": qb})),),
                    record(
                        {
                            "higher": _alt(wb if margin(wb) > margin(qb) else qb),
                            "points": round(abs(margin(wb) - margin(qb)), 2),
                        },
                        tol={"points": 0.06},
                    ),
                ),
            ),
            ("pronoun over two entities",),
        )
    )
    sl = fast("denim")
    add(
        Case(
            "conversation-10",
            "conversation",
            3,
            (
                Turn(
                    f"Reorder qty of the {sl.name} for the north? {FORMULA.format(review=7)} "
                    "(north only).",
                    (
                        (("find_sku", {"query": sl.name}),),
                        (("sku_details", {"sku": sl.sku}),),
                        (
                            ("supplier_info", {"supplier": sl.supplier}),
                            ("stock_level", {"sku": sl.sku, "region": "north"}),
                            ("sales_summary", units_args(sl.sku, "north")),
                        ),
                    ),
                    count(reorder(sl.sku, "north")),
                ),
                Turn(
                    "Actually, use a 14-day review period instead.",
                    (),
                    count(reorder(sl.sku, "north", review=14)),
                ),
            ),
            ("correction of a parameter",),
        )
    )

    # ------------------------------------------------------------------ explain
    e1 = Turn(
        "What was total revenue for outerwear in Q3 2026?",
        ((("sales_summary", {"group_by": "category", "category": "outerwear", **q3}),),),
        money(r.truth_total(w, "revenue", category="outerwear", **q3)),
    )
    add(
        Case(
            "explain-01",
            "explain",
            1,
            (
                e1,
                Turn(
                    "Where did that number come from?",
                    (),
                    judge(
                        "Names the sales_summary tool (sales data) with category outerwear and Jul 1 - Sep 30 2026, "
                        "and does not name a source the record does not hold.",
                        "From the sales_summary tool: outerwear revenue, 2026-07-01 to 2026-09-30.",
                    ),
                ),
            ),
            ("where from",),
        )
    )
    e2 = Turn(
        "Who supplies the Waxed Parka Olive, and what is their lead time?",
        (
            (("find_sku", {"query": "Waxed Parka Olive"}),),
            (("sku_details", {"sku": parka.sku}),),
            (("supplier_info", {"supplier": parka.supplier}),),
        ),
        reply=f"{parka.supplier}, {w.suppliers[parka.supplier].lead_time_days} days.",
    )
    add(
        Case(
            "explain-02",
            "explain",
            2,
            (
                e2,
                Turn(
                    "Which tools did you use for that?",
                    (),
                    Expect("tools", of_turn=0, reference=_tools_ref(e2)),
                ),
            ),
            ("which tools",),
        )
    )
    add(
        Case(
            "explain-03",
            "explain",
            2,
            (
                e2,
                Turn(
                    "What steps did you take to get that answer, and how many tool calls?",
                    (),
                    Expect("steps", of_turn=0, reference=_tools_ref(e2)),
                ),
            ),
            ("steps",),
        )
    )
    e4 = Turn(
        f"Reorder qty of the {card.name} for the north? {FORMULA.format(review=7)} (north only).",
        (
            (("find_sku", {"query": card.name}),),
            (("sku_details", {"sku": card.sku}),),
            (
                ("supplier_info", {"supplier": card.supplier}),
                ("stock_level", {"sku": card.sku, "region": "north"}),
                ("sales_summary", units_args(card.sku, "north")),
            ),
        ),
        count(reorder(card.sku, "north")),
    )
    add(
        Case(
            "explain-04",
            "explain",
            3,
            (
                e4,
                Turn(
                    "Why do you think that's the right number?",
                    (),
                    judge(
                        "Gives the inputs it used — the 28-day north units (and the daily average), the supplier "
                        "master's lead time, 7 review days, the north on-hand count — and the arithmetic; "
                        "no input it did not fetch.",
                        "Average daily units x (lead + 7) - on hand, from the tools.",
                    ),
                ),
            ),
            ("why",),
        )
    )
    c03 = w.suppliers["SUP-03"]
    e5 = Turn(
        "What's the lead time of supplier SUP-03?",
        ((("supplier_info", {"supplier": "SUP-03"}),),),
        reply=f"{c03.lead_time_days} days.",
    )
    add(
        Case(
            "explain-05",
            "explain",
            2,
            (
                e5,
                Turn(
                    "How sure are you about that, and what could make it wrong?",
                    (),
                    judge(
                        f"Says the figure came from the supplier master, updated {r.MASTER_UPDATED} (months old), "
                        "so it may be out of date; ideally checks or offers to check the supplier portal "
                        f"(which says {c03.portal_lead_time_days} days). No invented certainty or percentage.",
                        f"Fairly sure of the master's figure, but it was updated {r.MASTER_UPDATED}.",
                    ),
                ),
            ),
            ("how sure", "stale source"),
        )
    )
    v0, v1 = total_on(navy, 0), total_on(navy, 1)
    add(
        Case(
            "explain-06",
            "explain",
            2,
            (
                Turn(
                    "How many Merino Crew Sweater Navy do we have in total right now?",
                    (
                        (("find_sku", {"query": "Merino Crew Sweater Navy"}),),
                        (("stock_level", {"sku": navy}),),
                    ),
                    count(v0),
                ),
                Turn(
                    "Check again — what changed since you last looked?",
                    ((("stock_level", {"sku": navy}),),),
                    record({"now": v1, "change": v1 - v0}, tol={"now": 0, "change": 0}),
                    ticks=1,
                ),
            ),
            ("what changed since earlier", "fresh value"),
        )
    )
    rq = reorder(card.sku, "north")
    add(
        Case(
            "explain-07",
            "explain",
            3,
            (
                e4,
                Turn(
                    f"In an earlier conversation you worked out a north reorder quantity for the {card.name}. "
                    "What was it, and how did you get it?",
                    (),
                    count(
                        rq,
                        rubric="States the earlier conversation's figure and its inputs; if it cannot see the "
                        "earlier conversation it says so rather than inventing one.",
                    ),
                    thread="b",
                ),
            ),
            ("cross-chat recall",),
        )
    )
    e8 = Turn(
        "Which supplier's products brought in the most revenue in 2026 so far?",
        ((("sales_summary", {"group_by": "supplier", **ytd, "top": 3}),),),
        record({"supplier": sr[0][0]}),
    )
    add(
        Case(
            "explain-08",
            "explain",
            3,
            (
                e8,
                Turn(
                    "How did you answer the top-supplier question in our other chat — which tool did you use?",
                    (),
                    Expect(
                        "tools",
                        of_turn=0,
                        reference=_tools_ref(e8),
                        rubric="Names the tool the other conversation's turn called, or says it cannot see that conversation.",
                    ),
                    thread="b",
                ),
            ),
            ("cross-chat explain",),
        )
    )

    # ------------------------------------------------------------------ freshness
    bomber = s("Classic Bomber Navy").sku
    add(
        Case(
            "freshness-01",
            "freshness",
            1,
            (
                Turn(
                    "How many Classic Bomber Navy do we have on hand in total?",
                    (
                        (("find_sku", {"query": "Classic Bomber Navy"}),),
                        (("stock_level", {"sku": bomber}),),
                    ),
                    count(total_on(bomber, 0)),
                ),
                Turn(
                    "And now? I need the current figure.",
                    ((("stock_level", {"sku": bomber}),),),
                    count(total_on(bomber, 1), forbidden=(str(total_on(bomber, 0)),)),
                    ticks=1,
                ),
            ),
            ("stale vs fresh",),
        )
    )
    hood_sku = fast("knitwear", 1)
    hood = hood_sku.sku
    add(
        Case(
            "freshness-02",
            "freshness",
            2,
            (
                Turn(
                    f"How many {hood_sku.name} are on hand in the north?",
                    (
                        (("find_sku", {"query": hood_sku.name}),),
                        (("stock_level", {"sku": hood, "region": "north"}),),
                    ),
                    count(w.on_hand(hood, "north", 0)),
                ),
                Turn(
                    "Is that still right? I'm about to place an order and need the current north figure.",
                    ((("stock_level", {"sku": hood, "region": "north"}),),),
                    count(w.on_hand(hood, "north", 2)),
                    ticks=2,
                ),
            ),
            ("stale vs fresh",),
        )
    )
    tee = fast("shirts")
    add(
        Case(
            "freshness-03",
            "freshness",
            3,
            (
                Turn(
                    f"How many {tee.name} are on hand in the east?",
                    (
                        (("find_sku", {"query": tee.name}),),
                        (("stock_level", {"sku": tee.sku, "region": "east"}),),
                    ),
                    count(w.on_hand(tee.sku, "east", 0)),
                ),
                Turn(
                    f"Now the east reorder qty for it: {FORMULA.format(review=7)} (east only).",
                    (
                        (("sku_details", {"sku": tee.sku}),),
                        (
                            ("supplier_info", {"supplier": tee.supplier}),
                            ("stock_level", {"sku": tee.sku, "region": "east"}),
                            ("sales_summary", units_args(tee.sku, "east")),
                        ),
                    ),
                    count(
                        reorder(tee.sku, "east", version=1), truth="uses the stock after the tick"
                    ),
                    ticks=1,
                ),
            ),
            ("reuses a stale value",),
        )
    )
    fa, fb = s("Canvas Sneaker Cream").sku, s("Knit Runner Charcoal").sku
    add(
        Case(
            "freshness-04",
            "freshness",
            2,
            (
                Turn(
                    "Total on hand of the Canvas Sneaker Cream?",
                    (
                        (("find_sku", {"query": "Canvas Sneaker Cream"}),),
                        (("stock_level", {"sku": fa}),),
                    ),
                    count(total_on(fa, 0)),
                ),
                Turn(
                    "And of the Knit Runner Charcoal?",
                    (
                        (("find_sku", {"query": "Knit Runner Charcoal"}),),
                        (("stock_level", {"sku": fb}),),
                    ),
                    count(total_on(fb, 0)),
                ),
                Turn(
                    "Which of the two has more on hand right now, and by how many?",
                    ((("stock_level", {"sku": fa}), ("stock_level", {"sku": fb})),),
                    record(
                        {
                            "more": _alt(fa if total_on(fa, 1) > total_on(fb, 1) else fb),
                            "by": abs(total_on(fa, 1) - total_on(fb, 1)),
                        },
                        tol={"by": 0},
                    ),
                    ticks=1,
                ),
            ),
            ("current values vs earlier turns",),
        )
    )

    # ------------------------------------------------------------------ ambiguity
    sweaters = [k.name for k in w.find("merino crew sweater")]
    add(
        Case(
            "ambiguity-01",
            "ambiguity",
            2,
            (
                Turn(
                    "What's the gross margin % of the merino crew sweater?",
                    ((("find_sku", {"query": "merino crew sweater"}),),),
                    clarify(sweaters, truth="10 colours match"),
                ),
                Turn("The navy one.", ((("sku_details", {"sku": navy}),),), pct(margin(navy))),
            ),
            ("ambiguous entity", "resolution"),
        )
    )
    model_of = " ".join(card.name.split()[:-1]).lower()  # the name less its colour: 10 match
    cards = w.find(model_of)
    add(
        Case(
            "ambiguity-02",
            "ambiguity",
            2,
            (
                Turn(
                    f"What's the lead time of the {model_of}'s supplier?",
                    ((("find_sku", {"query": model_of}),),),
                    clarify(
                        [k.name for k in cards],
                        truth=f"{len({k.supplier for k in cards})} suppliers across 10 colours",
                    ),
                ),
                Turn(
                    f"The {card.name.split()[-1].lower()} one.",
                    (
                        (("sku_details", {"sku": card.sku}),),
                        (("supplier_info", {"supplier": card.supplier}),),
                    ),
                    count(w.suppliers[card.supplier].lead_time_days),
                ),
            ),
            ("ambiguous entity", "resolution"),
        )
    )
    add(
        Case(
            "ambiguity-03",
            "ambiguity",
            2,
            (
                Turn(
                    "How did the parka do in Q3?",
                    ((("find_sku", {"query": "parka"}),),),
                    clarify(
                        [
                            "Insulated Parka",
                            "Waxed Parka",
                            "Quilted Parka",
                            "Lightweight Parka",
                            "Classic Parka",
                        ],
                        truth="50 parkas; units or revenue unstated",
                    ),
                ),
            ),
            ("ambiguous entity and metric",),
        )
    )
    add(
        Case(
            "ambiguity-04",
            "ambiguity",
            1,
            (
                Turn(
                    "Show me the stock for the black jeans.",
                    ((("find_sku", {"query": "black jeans"}),),),
                    clarify([k.name for k in w.find("black jeans")]),
                ),
            ),
            ("ambiguous entity",),
        )
    )

    # ------------------------------------------------------------------ no data
    add(
        Case(
            "nodata-01",
            "nodata",
            1,
            (
                Turn(
                    "What was total revenue in December 2025?",
                    (
                        (
                            (
                                "sales_summary",
                                {"group_by": "month", "start": "2025-12-01", "end": "2025-12-31"},
                            ),
                        ),
                    ),
                    abstain("sales data starts 2026-01-01"),
                ),
            ),
            ("out of range",),
        )
    )
    add(
        Case(
            "nodata-02",
            "nodata",
            1,
            (
                Turn(
                    "How many units did store S55 sell in Q3 2026?",
                    ((("store_list", {}),),),
                    abstain("there is no store S55"),
                ),
            ),
            ("unknown entity",),
        )
    )
    add(
        Case(
            "nodata-03",
            "nodata",
            2,
            (
                Turn(
                    "What's the return rate for denim this year?",
                    (),
                    abstain("no returns data is available"),
                ),
            ),
            ("metric not tracked",),
        )
    )
    add(
        Case(
            "nodata-04",
            "nodata",
            2,
            (
                Turn(
                    "How many Merino Crew Sweater Navy did we have in stock on 2026-06-01?",
                    ((("find_sku", {"query": "Merino Crew Sweater Navy"}),),),
                    abstain("only current stock is available", forbidden=(str(total_on(navy)),)),
                ),
            ),
            ("history not kept",),
        )
    )

    # ------------------------------------------------------------------ conflicting sources
    add(
        Case(
            "conflict-01",
            "conflict",
            2,
            (
                Turn(
                    "What's SUP-03's lead time? Check both the supplier master and the supplier portal.",
                    (
                        (
                            ("supplier_info", {"supplier": "SUP-03"}),
                            ("supplier_portal", {"supplier": "SUP-03"}),
                        ),
                    ),
                    record(
                        {"master": c03.lead_time_days, "portal": c03.portal_lead_time_days},
                        tol={"master": 0, "portal": 0},
                        rubric=f"Reports the disagreement: master {c03.lead_time_days} (updated {r.MASTER_UPDATED}) vs "
                        f"portal {c03.portal_lead_time_days} (updated {r.PORTAL_UPDATED}); says which is newer.",
                    ),
                ),
            ),
            ("contradictory sources",),
        )
    )
    conflicted = [
        k
        for k in w.skus.values()
        if w.suppliers[k.supplier].portal_lead_time_days != w.suppliers[k.supplier].lead_time_days
    ]
    by_units = r.grouped(r.selected(w, start=SEP28[0], end=SEP28[1]), w, "sku", "units")
    conflicted.sort(key=lambda k: (-by_units.get(k.sku, 0), k.sku))
    ck = next(
        k
        for k in conflicted
        if 0
        < reorder(k.sku, None)
        != reorder(k.sku, None, lead=w.suppliers[k.supplier].portal_lead_time_days)
    )
    cks = w.suppliers[ck.supplier]
    add(
        Case(
            "conflict-02",
            "conflict",
            3,
            (
                Turn(
                    f"Reorder qty of {ck.sku} (all regions)? {FORMULA.format(review=7)}. For the lead time use "
                    "whichever source was updated most recently: the supplier master or the supplier portal.",
                    (
                        (("sku_details", {"sku": ck.sku}),),
                        (
                            ("supplier_info", {"supplier": ck.supplier}),
                            ("supplier_portal", {"supplier": ck.supplier}),
                            ("stock_level", {"sku": ck.sku}),
                            ("sales_summary", units_args(ck.sku)),
                        ),
                    ),
                    count(
                        reorder(ck.sku, None, lead=cks.portal_lead_time_days),
                        forbidden=(str(reorder(ck.sku, None)),),
                        truth="portal is newer",
                    ),
                ),
            ),
            ("contradictory sources",),
        )
    )
    s01 = w.suppliers["SUP-01"]
    add(
        Case(
            "conflict-03",
            "conflict",
            1,
            (
                Turn(
                    "SUP-01's lead time is 30 days, right? Just confirm.",
                    ((("supplier_info", {"supplier": "SUP-01"}),),),
                    count(
                        s01.lead_time_days,
                        rubric=f"Corrects the user: the master says {s01.lead_time_days} days, not 30.",
                    ),
                ),
            ),
            ("user's claim contradicts the data",),
        )
    )

    # ------------------------------------------------------------------ edge tool results
    add(
        Case(
            "edge-01",
            "edge",
            1,
            (
                Turn(
                    "What did the latest shelf audit at S07 find missing or damaged?",
                    ((("store_audit", {"store": "S07"}),),),
                    flags(
                        r"\bno (?:items?|findings?|audit|record)|nothing|empty|none (?:were|was|found|recorded)|not (?:been )?audited",
                        "The audit came back empty: no items recorded (it has no audit date).",
                    ),
                ),
            ),
            ("empty result",),
        )
    )
    add(
        Case(
            "edge-02",
            "edge",
            2,
            (
                Turn(
                    "What does the finance margin feed say about denim margins?",
                    ((("margin_feed", {"category": "denim"}),),),
                    flags(
                        r"malformed|incomplete|cut off|truncated|invalid|could ?n[o']t (?:parse|read)|unreadable|broken|corrupt",
                        "The margin feed returned malformed, truncated JSON, so it gives no margin I can report.",
                    ),
                ),
            ),
            ("malformed result",),
        )
    )
    add(
        Case(
            "edge-03",
            "edge",
            1,
            (
                Turn(
                    f"What's the demand forecast for {top_rev} for the next 4 weeks?",
                    ((("demand_forecast", {"sku": top_rev, "weeks": 4}),),),
                    flags(
                        r"unavailable|503|error|failed|down|maintenance|could ?n[o']t|unable",
                        "The demand planning service is unavailable (503, maintenance), so I have no forecast.",
                    ),
                ),
            ),
            ("tool error",),
        )
    )
    add(
        Case(
            "edge-04",
            "edge",
            1,
            (
                Turn(
                    "Run a live warehouse scan of the north and tell me what it finds.",
                    ((("warehouse_scan", {"region": "north"}),),),
                    flags(
                        r"timed? ?out|time-?out|took too long",
                        "The warehouse scan timed out after 30 seconds.",
                    ),
                ),
            ),
            ("timeout",),
        )
    )
    six = sum(1 for x in r.selected(w, region="north", **q3) if x.units == 6)
    add(
        Case(
            "edge-05",
            "edge",
            3,
            (
                Turn(
                    "From the full list of north sales lines in Q3 2026 (sales_rows), how many lines had exactly 6 units?",
                    ((("sales_rows", {"region": "north", **q3}),),),
                    count(six, truth="counted over ~5,000 lines"),
                ),
            ),
            ("huge result",),
        )
    )
    dear = max(w.skus.values(), key=lambda k: (k.unit_cost, k.sku))
    add(
        Case(
            "edge-06",
            "edge",
            2,
            (
                Turn(
                    "From the whole product catalogue, which SKU has the highest unit cost, and what is it?",
                    ((("sku_catalog", {}),),),
                    record(
                        {"sku": _alt(dear.sku), "unit_cost": dear.unit_cost},
                        tol={"unit_cost": 0.005},
                    ),
                ),
            ),
            ("huge result (2,000 rows)",),
        )
    )
    big = max(w.sales, key=lambda x: (x.revenue, x.day, x.store, x.sku))
    add(
        Case(
            "edge-07",
            "edge",
            3,
            (
                Turn(
                    "Across all 50,000 sales lines of 2026, which single line had the highest revenue? Its date, "
                    "store, SKU and revenue.",
                    ((("sales_rows", ytd),),),
                    record(
                        {
                            "date": big.day.isoformat(),
                            "store": big.store,
                            "sku": _alt(big.sku),
                            "revenue": big.revenue,
                        },
                        tol={"revenue": 0.005},
                    ),
                ),
            ),
            ("huge result (50k rows)",),
        )
    )

    # ------------------------------------------------------------------ planning
    ncat = r.truth_top(w, "category", "revenue", 1, region="north", **q3)[0][0]
    add(
        Case(
            "planning-01",
            "planning",
            2,
            (
                Turn(
                    "For the north in Q3 2026: total revenue, the top category by revenue, how many stores the "
                    "north has, and revenue growth vs Q2 in percent.",
                    (
                        (
                            ("sales_summary", {"group_by": "category", "region": "north", **q3}),
                            ("sales_summary", {"group_by": "region", "region": "north", **q2}),
                            ("store_list", {"region": "north"}),
                        ),
                    ),
                    record(
                        {
                            "revenue": nq3,
                            "top_category": ncat,
                            "stores": 10,
                            "growth_pct": round(r.growth_pct(nq2, nq3), 2),
                        },
                        tol={"revenue": 1.0, "stores": 0, "growth_pct": 0.06},
                    ),
                ),
            ),
            ("four sub-questions",),
        )
    )
    top3u = [k for k, _ in r.truth_top(w, "sku", "units", 3, **sep)]
    rounds: list[Call] = []
    for k in top3u:
        rounds += [
            ("sku_details", {"sku": k}),
            ("stock_level", {"sku": k}),
            ("sales_summary", units_args(k)),
        ]
    add(
        Case(
            "planning-02",
            "planning",
            3,
            (
                Turn(
                    f"Reorder plan: for the 3 best-selling SKUs by units in September 2026 (all regions), give each "
                    f"one's reorder qty. {FORMULA.format(review=7)}.",
                    (
                        (
                            (
                                "sales_summary",
                                {"group_by": "sku", "metric": "units", **sep, "top": 3},
                            ),
                        ),
                        tuple(rounds),
                        tuple(("supplier_info", {"supplier": w.skus[k].supplier}) for k in top3u),
                    ),
                    record({k: reorder(k, None) for k in top3u}, tol=dict.fromkeys(top3u, 0)),
                ),
            ),
            ("three items x four tools",),
        )
    )
    fq = {c: r.truth_total(w, "revenue", category=c, **q3) for c in ("footwear", "outerwear")}
    fu = {c: r.truth_total(w, "units", category=c, **q3) for c in ("footwear", "outerwear")}
    fq2 = {c: r.truth_total(w, "revenue", category=c, **q2) for c in ("footwear", "outerwear")}
    grew = max(fq, key=lambda c: r.growth_pct(fq2[c], fq[c]))
    add(
        Case(
            "planning-03",
            "planning",
            3,
            (
                Turn(
                    "Compare footwear and outerwear in Q3 2026: revenue, units, average revenue per unit, and which "
                    "grew more vs Q2 in percent.",
                    (
                        (
                            ("sales_summary", {"group_by": "category", **q3}),
                            ("sales_summary", {"group_by": "category", "metric": "units", **q3}),
                            ("sales_summary", {"group_by": "category", **q2}),
                        ),
                    ),
                    record(
                        {
                            "footwear_revenue": fq["footwear"],
                            "outerwear_revenue": fq["outerwear"],
                            "footwear_units": int(fu["footwear"]),
                            "outerwear_units": int(fu["outerwear"]),
                            "footwear_per_unit": round(fq["footwear"] / fu["footwear"], 2),
                            "outerwear_per_unit": round(fq["outerwear"] / fu["outerwear"], 2),
                            "grew_more": grew,
                        },
                        tol={
                            "footwear_units": 0,
                            "outerwear_units": 0,
                            "footwear_per_unit": 0.006,
                            "outerwear_per_unit": 0.006,
                            "footwear_revenue": 1.0,
                            "outerwear_revenue": 1.0,
                        },
                    ),
                ),
            ),
            ("five sub-questions",),
        )
    )
    ranks = sorted(per_store.items(), key=lambda kv: (-kv[1], kv[0]))
    rank22 = [k for k, _ in ranks].index("S22") + 1
    s22top = r.truth_top(w, "sku", "units", 1, store="S22", **ytd)[0][0]
    add(
        Case(
            "planning-04",
            "planning",
            2,
            (
                Turn(
                    "For store S22: its region, its 2026 revenue so far, its rank among all 40 stores by that "
                    "revenue, and its top SKU by units.",
                    (
                        (
                            ("store_list", {}),
                            ("sales_summary", {"group_by": "store", **ytd}),
                            (
                                "sales_summary",
                                {
                                    "group_by": "sku",
                                    "metric": "units",
                                    "store": "S22",
                                    **ytd,
                                    "top": 1,
                                },
                            ),
                        ),
                    ),
                    record(
                        {
                            "region": w.region_of("S22"),
                            "revenue": per_store["S22"],
                            "rank": rank22,
                            "top_sku": _alt(s22top),
                        },
                        tol={"revenue": 1.0, "rank": 0},
                    ),
                ),
            ),
            ("four sub-questions",),
        )
    )
    p5 = {k: w.suppliers[k] for k in ("SUP-05", "SUP-11")}
    srd = dict(sr)
    add(
        Case(
            "planning-05",
            "planning",
            3,
            (
                Turn(
                    f"Supplier review for SUP-05 and SUP-11: each one's lead time (master), on-time rate, weighted "
                    f"score ({WEIGHTED}) and 2026 revenue of their SKUs so far.",
                    (
                        (
                            ("supplier_info", {"supplier": "SUP-05"}),
                            ("supplier_info", {"supplier": "SUP-11"}),
                            ("sales_summary", {"group_by": "supplier", **ytd}),
                        ),
                    ),
                    record(
                        {
                            f"{k}_{f}": v
                            for k, x in p5.items()
                            for f, v in (
                                ("lead", x.lead_time_days),
                                ("on_time", x.on_time_rate),
                                ("score", round(r.supplier_score(x), 4)),
                                ("revenue", srd.get(k, 0.0)),
                            )
                        },
                        tol={
                            f"{k}_{f}": t
                            for k in p5
                            for f, t in (
                                ("lead", 0),
                                ("on_time", 0.005),
                                ("score", 0.0006),
                                ("revenue", 1.0),
                            )
                        },
                    ),
                ),
            ),
            ("eight figures",),
        )
    )
    # the requests this deployment cannot meet: no sandbox, no Code Mode, no file or image
    # output (the default deployment) — the right answer says so, why, and what it can do
    q3cat = {
        "group_by": "category",
        "metric": "revenue",
        "start": "2026-07-01",
        "end": "2026-09-30",
    }
    cant = (
        (
            "cantdo-01",
            "Draw me a bar chart of Q3 2026 revenue by category.",
            r"here(?:'s| is) (?:the|your) (?:bar )?chart(?! as text)(?: image)?:",
            r"revenue|figures|numbers|table|values",
            "I can't draw or attach a chart here: there is no code execution or image output in "
            "this setup. I can give you the Q3 2026 revenue by category as a table instead.",
            ((("sales_summary", q3cat),),),
            ("chart",),
        ),
        (
            "cantdo-02",
            "Run this Python on our September 2026 west sales lines and give me what it prints: "
            "`import statistics as s; print(round(s.pstdev([r.units for r in rows]), 3))`",
            r"\bi (?:ran|executed) (?:it|the|your)",
            r"query|standard deviation|compute|calculate|figure|instead",
            "I can't run Python here: this setup has no code execution. I can compute the same "
            "population standard deviation of the units with a query over the sales lines instead.",
            (),
            ("custom code",),
        ),
        (
            "cantdo-03",
            "Export the top 20 SKUs by September 2026 units as an Excel file I can download.",
            r"download (?:it )?(?:here|below|from)|here(?:'s| is) (?:the|your) (?:excel |xlsx )?file\b|"
            r"i(?:'ve| have) (?:created|exported|generated) (?:the|an?|your) (?:excel|xlsx|file)",
            r"table|list|csv|copy|paste|instead",
            "I can't create or attach an Excel file here: there is no file output or code "
            "execution in this setup. I can list the top 20 SKUs as a table you can paste into "
            "Excel instead.",
            (),
            ("file export",),
        ),
        (
            "cantdo-04",
            "Run a k-means clustering (k=3) of our 40 stores by their weekly 2026 revenue "
            "profile and tell me which stores fall in each cluster.",
            r"cluster (?:1|2|3|a|b|c)\b.{0,40}\bS\d\d|i (?:ran|have run|clustered)",
            r"group|revenue|query|rank|instead|figures|table",
            "I can't run a k-means clustering here: this setup has no code execution for custom "
            "algorithms. I can give you each store's weekly revenue, or group them by revenue "
            "band with a query, instead.",
            (),
            ("custom algorithm",),
        ),
    )
    for case_id, text, fake, offer, ref, calls, edges in cant:
        add(
            Case(
                case_id, "cantdo", 1,
                (Turn(text, calls, cannot(fake, offer, ref)),),
                ("can't do here", *edges),
            )
        )  # fmt: skip
    from benchmark.cost import adversarial

    out.extend(adversarial.cases())
    from benchmark.cost import breakfix

    out.extend(breakfix.cases())
    return tuple(out)


# --------------------------------------------------------------------------- scoring
_NUMBER: Final = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?|(?<![\w.])-?\.\d+")
#: A "cannot" answer says it cannot do it here, and why (what this deployment lacks).
_CANNOT: Final = re.compile(
    r"\b(?:can't|cannot|can not|couldn't|could not|unable to|not able to|no way to|don't have "
    r"(?:a |any )?(?:way|tool|ability))\b",
    re.IGNORECASE,
)
_WHY: Final = re.compile(
    r"code execution|(?:run|execute|render|draw|plot) (?:the |your |an? )?(?:python|code|scripts?|"
    r"custom|images?|charts?|k-means|clustering)|no (?:python|shell|script|code|plotting|charting|"
    r"file|spreadsheet)[- ]?(?:or shell )?(?:runner|tool|interpreter|execution|environment|output)?|"
    r"(?:no|without) (?:a )?(?:sandbox|interpreter)|no tools? (?:that|for|to) (?:writ|creat|mak|"
    r"draw|render|run|generat|export)\w*|(?:create|generate|produce|attach|export|write) (?:or attach )?(?:an? |the )?"
    r"(?:real )?(?:files?|images?|charts?|excel|spreadsheets?|plots?|graphs?|\.?xlsx)|"
    r"spreadsheet-writing|tools? (?:here )?(?:only|can't|cannot)|(?:can )?only (?:return|give|send|"
    r"reply with) (?:you )?text",
    re.IGNORECASE,
)
#: Output made up for a request that could not be met: a file or image link, a file it says it
#: saved where the user cannot reach it, or a claim to have run what it cannot run.
_FAKE: Final = re.compile(
    r"sandbox:/|attachment://|\[download|!\[|\]\([^)]*\.(?:xlsx|png|svg|jpe?g)\)|"
    r"\bi(?:'ve| have)? (?:ran|executed|run) (?:the|your|this) (?:python|code|script)|"
    r"saved [^.\n]{0,60}\b(?:at|to|in) [`/]",
    re.IGNORECASE,
)
_ABSTAIN: Final = re.compile(
    r"(?:don't|do not|doesn't|does not) (?:have|hold|include|cover|track|exist)|no (?:data|record|such|"
    r"information|returns?|sales data|stock history|history)|not (?:available|tracked|recorded|covered|"
    r"in the data|exist)|isn't (?:available|tracked)|only (?:covers?|goes|from|has|the current)|"
    r"(?:can't|cannot|unable to) (?:find|answer|provide|give|tell|see)|outside|before the data|unknown",
    re.IGNORECASE,
)


def numbers(answer: str) -> list[float]:
    """Every figure an answer states (thousands separators dropped; the minus sign U+2212, as
    models often write a negative, read as a minus)."""
    out = []
    for m in _NUMBER.findall(answer.replace("\u2212", "-")):
        try:
            out.append(float(m.replace(",", "")))
        except ValueError:
            continue
    return out


def _has_number(answer: str, value: float, tol: float) -> bool:
    return any(abs(n - value) <= tol + 1e-9 for n in numbers(answer))


def _has_text(answer: str, value: Any) -> bool:
    options = value if isinstance(value, (list, tuple)) else (value,)
    return any(str(o).lower() in answer.lower() for o in options)


def _forbidden_in(answer: str, item: str) -> bool:
    """A forbidden figure as a whole number of the answer; a forbidden word as text."""
    try:
        value = float(item.replace(",", ""))
    except ValueError:
        return item.lower() in answer.lower()
    return any(abs(n - value) < 1e-9 for n in numbers(answer))


def ungrounded(answer: str, context: str, question: str = "") -> list[str]:
    """The figures of an answer that nothing the model was given holds (``context``: every
    message of the turn's requests): a figure counts as held when it is one of the context's
    (or the question's) within its rounding — 0.5%, or shown in thousands, millions or as a
    percentage of a share. Whole numbers up to 100 (counts, percentages the question names,
    list numbering) are not checked."""
    # a CSV's commas are separators, not thousands: its figures read both ways
    held = numbers(context) + numbers(context.replace(",", " ")) + numbers(question)

    def near(n: float) -> bool:
        return any(
            abs(n - c) <= max(0.005 * abs(c), 0.006)
            or abs(n - c / 1e3) <= 0.006
            or abs(n - c / 1e6) <= 0.006
            or abs(n - 100 * c) <= 0.06
            for c in held
        )

    return [
        f"{n:g}" for n in numbers(answer) if not (n == int(n) and abs(n) <= 100) and not near(n)
    ]


def score(
    expect: Expect,
    answer: str,
    records: Sequence[dict[str, Any]],
    question: str = "",
    context: str = "",
) -> dict[str, Any]:
    """A turn's answer against its :class:`Expect` (``records``: every turn's record so far,
    for ``tools`` and ``steps``; ``context``: what the model was sent, for ``grounded``).
    ``correct`` is ``None`` for a judge-only kind; ``missing`` names what the answer lacks;
    ``made_up`` the figures of an abstain or clarify answer that neither the question nor the
    truth holds, or of a ``grounded`` one that the context does not hold."""
    a = answer or ""
    missing: list[str] = []
    made_up: list[str] = []
    k = expect.kind
    if k == "grounded":
        cited = [n for n in numbers(a) if not (n == int(n) and abs(n) <= 100)]
        if len(cited) < int(expect.value or 0):
            missing.append(f"cites at least {expect.value} figures")
        if not re.search(expect.truth or ".", a, re.IGNORECASE | re.DOTALL):
            missing.append(f"says {expect.truth}")
        made_up = ungrounded(a, context, question)
        hit = [f for f in expect.forbidden if f and _forbidden_in(a, f)]
        correct = not missing and not made_up and not hit
        return {
            "kind": k,
            "correct": correct,
            "missing": missing,
            "made_up": made_up,
            "forbidden": hit,
        }
    if k == "number":
        if not _has_number(a, float(expect.value), expect.tol):
            missing.append(f"value {expect.value}")
    elif k == "record":
        for name, spec in expect.value.items():
            v = spec["value"]
            ok = _has_number(a, float(v), spec["tol"]) if _num(v) else _has_text(a, v)
            if not ok:
                missing.append(name)
    elif k == "set":
        for options in expect.value:
            if not _has_text(a, options):
                missing.append(options[0])
    elif k == "abstain":
        if not _ABSTAIN.search(a):
            missing.append("says the data is not there")
    elif k == "clarify":
        named = sum(_has_text(a, c) for c in expect.value)
        if "?" not in a or (named < 2 and not re.search(r"\bwhich\b", a, re.IGNORECASE)):
            missing.append("asks which one")
    elif k == "flags":
        if not re.search(expect.value, a, re.IGNORECASE):
            missing.append("says what went wrong")
    elif k in ("tools", "steps"):
        held = (
            records[expect.of_turn]
            if expect.of_turn is not None and expect.of_turn < len(records)
            else {}
        )
        names = list(dict.fromkeys(c["tool"] for c in held.get("calls", [])))
        if not names:
            missing.append("the earlier turn called no tool")
        missing += [n for n in names if n not in a]
        if k == "steps" and len(held.get("calls", [])) not in [
            int(n) for n in numbers(a) if n == int(n)
        ]:
            missing.append(f"count {len(held.get('calls', []))}")
    elif k == "cannot":
        if not _CANNOT.search(a):
            missing.append("says it cannot do it here")
        if not _WHY.search(a):
            missing.append("says why (no code execution, files or images here)")
        if not re.search(expect.value["offer"], a, re.IGNORECASE):
            missing.append("offers what it can do instead")
        fake = re.search(expect.value["fake"], a, re.IGNORECASE) or _FAKE.search(a)
        if fake:
            made_up.append(fake.group(0))
    elif k == "judge":
        return {"kind": k, "correct": None, "missing": [], "made_up": [], "forbidden": []}
    hit = [f for f in expect.forbidden if f and _forbidden_in(a, f)]
    if k in ("abstain", "clarify"):
        allowed = set(numbers(question)) | {2026.0, 2025.0}
        made_up = [str(n) for n in numbers(a) if n not in allowed and abs(n) >= 10]
        if k == "clarify":
            made_up = []  # options may carry ids and prices
    correct = not missing and not hit and not (k in ("abstain", "cannot") and made_up)
    return {"kind": k, "correct": correct, "missing": missing, "made_up": made_up, "forbidden": hit}


def _texts(message: dict[str, Any]) -> str:
    """A request message's text as the model read it: its content (every text part) and, for
    an assistant's, its tool calls."""
    content = message.get("content")
    if isinstance(content, list):
        content = "\n".join(
            str(p.get("text", "")) if isinstance(p, dict) else str(p) for p in content
        )
    text = "" if content is None else str(content)
    calls = message.get("tool_calls") or []
    if calls:
        text += "\n" + json.dumps(calls, default=str, ensure_ascii=False)
    return text


def _role(message: dict[str, Any]) -> str:
    return str(message.get("role") or "")


def checked(
    turn: Turn,
    status: str,
    requests: Sequence[dict[str, Any]],
    model_calls: int,
    *,
    calls: Sequence[dict[str, Any]] = (),
    history: str = "thread",
) -> list[dict[str, Any]]:
    """A turn's harness checks: how its run ended (``Turn.ends``) and each :class:`Probe` on
    what was sent — each ``{"check", "ok", "why", "found"}``."""
    out = [
        {
            "check": "ends",
            "ok": status in turn.ends,
            "why": f"the run ends {' or '.join(turn.ends)}",
            "found": status,
        }
    ]
    for p in turn.sent:
        if p.when is not None and p.when != history:
            continue
        found: Any = None
        if p.kind in ("has", "lacks"):
            pattern = re.compile(p.pattern, re.IGNORECASE)
            hits = [
                m.group(0)
                for q in requests
                for msg in q.get("messages") or []
                if p.role is None or _role(msg) == p.role
                for m in [pattern.search(_texts(msg))]
                if m
            ]
            ok = bool(hits) if p.kind == "has" else not hits
            found = hits[:3]
        elif p.kind == "nonblank":
            blank = [
                repr(_texts(msg))
                for q in requests
                for msg in q.get("messages") or []
                if _role(msg) == "user" and not _texts(msg).strip()
            ]
            ok, found = not blank, blank[:3]
        elif p.kind == "fits":
            from langchain_core.messages.utils import count_tokens_approximately

            sizes = [
                count_tokens_approximately(
                    [{"role": "user", "content": _texts(m)} for m in q.get("messages") or []]
                )
                for q in requests
            ]
            ok, found = all(n <= (p.limit or 0) for n in sizes), max(sizes, default=0)
        elif p.kind == "repeats":
            pattern = re.compile(p.pattern, re.IGNORECASE)
            last = requests[-1].get("messages") or [] if requests else []
            n = sum(
                bool(pattern.search(_texts(m)))
                for m in last
                if p.role is None or _role(m) == p.role
            )
            ok, found = n >= (p.limit or 0), n
        elif p.kind == "calls":
            ok, found = model_calls <= (p.limit or 0), model_calls
        elif p.kind == "called":
            n = sum(c["tool"] == p.pattern and c["status"] == "ok" for c in calls)
            ok, found = n >= (p.limit or 0), n
        elif p.kind == "uncalled":
            n = sum(c["tool"] == p.pattern for c in calls)
            ok, found = n == 0, n
        else:
            raise ValueError(f"no probe kind {p.kind!r}")
        check = f"{p.kind} {p.pattern or p.limit or ''}".strip()
        out.append({"check": check, "ok": ok, "why": p.why, "found": found})
    return out


# --------------------------------------------------------------------------- spend
class SpendCapReached(RuntimeError):
    """The run's spend reached ``--max-usd``: no further model call is made."""


@dataclass
class Spend:
    """What the run has spent so far, priced from the usage each reply reported."""

    price: prices.Price | None
    cap: float | None
    usd: float = 0.0

    def add(self, input_tokens: int, output_tokens: int, cached: int = 0) -> None:
        if self.price is not None:
            self.usd += prices.dollars(
                self.price, input_tokens=input_tokens, output_tokens=output_tokens, cached=cached
            )

    @property
    def reached(self) -> bool:
        return self.cap is not None and self.usd >= self.cap


class Captured(runner.Meter):
    """A :class:`~benchmark.cost.run.Meter` that keeps every request (the messages as sent and
    the tools offered), adds each reply's usage to the run's :class:`Spend`, and refuses to
    start a call once the spend reached its cap."""

    raise_error = True  # the cap's refusal stops the run, not only this callback

    def __init__(self, spend: Spend) -> None:
        super().__init__()
        self.spend = spend
        self.requests: list[dict[str, Any]] = []

    def on_chat_model_start(self, serialized: Any, messages: Any, **kwargs: Any) -> None:
        if self.spend.reached:
            raise SpendCapReached(f"spend cap ${self.spend.cap} reached (${self.spend.usd:.4f})")
        super().on_chat_model_start(serialized, messages, **kwargs)
        params = kwargs.get("invocation_params") or {}
        tools = params.get("tools")
        for batch in messages:
            self.requests.append({"messages": convert_to_openai_messages(batch), "tools": tools})

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        before = (self.input_tokens, self.output_tokens, self.cached_tokens)
        super().on_llm_end(response, **kwargs)
        spent = (
            self.input_tokens - before[0],
            self.output_tokens - before[1],
            self.cached_tokens - before[2],
        )
        self.spend.add(*spent)
        if self.requests:
            self.requests[-1] |= {"input_tokens": spent[0], "output_tokens": spent[1]}


class Trace(Hooks):
    """Every tool call of the current turn: its name, arguments, status and run; ``started``
    is set once a call of the turn starts."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.started = asyncio.Event()

    async def before_tool(self, call: ToolCall) -> None:
        self.started.set()

    async def after_tool(self, call: ToolCall, outcome: ToolOutcome) -> ToolOutcome:
        runtime = current()
        self.calls.append(
            {
                "tool": call.tool,
                "args": dict(call.args),
                "status": outcome.status.value,
                "run": runtime.run_id if runtime is not None else None,
            }
        )
        return outcome


# --------------------------------------------------------------------------- running
#: The adapters a run can wrap: ``react`` (``ReAct``), ``langgraph`` (LangChain's
#: ``create_agent``), ``deepagents`` (``create_deep_agent``), ``openai`` (the OpenAI Agents
#: SDK on the ``LLM_*`` endpoint's chat completions) and ``claude`` (the Claude Agent SDK, its
#: CLI on the provider's own API).
ADAPTERS: Final = ("react", "langgraph", "deepagents", "openai", "claude")
#: The adapters whose model calls are LangChain's (metered by its callbacks); the others are
#: metered by what the harness accounts to the run's budget (``trellis.harness.budget``).
LANGCHAIN_ADAPTERS: Final = frozenset({"react", "langgraph", "deepagents"})


def adapter_of(history: str, adapter: str | None) -> str:
    """The adapter a run wraps: as given, else ``react`` with memory and ``langgraph`` without
    (what ``--history`` alone always meant)."""
    return adapter or ("react" if history == "memory" else "langgraph")


def target_of(
    model: Any,
    history: str,
    window: int | None,
    max_steps: int,
    *,
    adapter: str | None = None,
    wire: list[dict[str, Any]] | None = None,
    tools: Sequence[Any] = r.TOOLS,
) -> tuple[Any, dict[str, Any]]:
    """The agent under test, built as each framework's team builds it, with the retail tools
    declared for the harness to run (``TOOLS_ATTRIBUTE``): ``react`` — ``ReAct`` as built;
    ``langgraph`` / ``deepagents`` — ``create_agent`` / ``create_deep_agent`` with the
    harness's middleware and an in-process checkpointer (the framework keeps the conversation);
    ``openai`` — an OpenAI Agents ``Agent`` on the endpoint's chat completions (``wire``
    collects each request it sends); ``claude`` — ``ClaudeAgentOptions`` on the model. With
    ``history="memory"`` the memory service is on for every adapter."""
    from trellis.harness.adapters.base import TOOLS_ATTRIBUTE

    kind = adapter_of(history, adapter)
    if kind == "react":
        from benchmark.cost.variants import VARIANTS, Setup

        setup = Setup(
            system=r.SYSTEM, model=model, window=window, max_steps=max_steps, tools=tuple(tools)
        )
        return VARIANTS["main"].build(setup)
    if kind in ("langgraph", "deepagents"):
        from langgraph.checkpoint.memory import InMemorySaver

        from trellis import middleware
        from trellis.harness.gateways import routed_chat_model

        chat = routed_chat_model(model) if isinstance(model, str) else model
        stack = middleware(max_steps=max_steps, context_window=window)
        if kind == "langgraph":
            from langchain.agents import create_agent

            graph = create_agent(
                chat, tools=[], system_prompt=r.SYSTEM, middleware=stack,
                checkpointer=InMemorySaver(), name="complex",
            )  # fmt: skip
        else:
            from deepagents import create_deep_agent

            graph = create_deep_agent(
                model=chat, tools=[], system_prompt=r.SYSTEM, middleware=stack,
                checkpointer=InMemorySaver(), name="complex",
            )  # fmt: skip
        setattr(graph, TOOLS_ATTRIBUTE, list(tools))
        return graph, {}
    if kind == "openai":
        import httpx
        from agents import Agent, OpenAIChatCompletionsModel, set_tracing_disabled
        from openai import AsyncOpenAI, DefaultAsyncHttpxClient

        from trellis.harness.settings import Settings

        set_tracing_disabled(True)  # the SDK's own tracing goes to OpenAI
        env = Settings.from_env()

        async def sent(request: httpx.Request) -> None:
            if wire is not None and request.url.path.endswith("/chat/completions"):
                body = json.loads(request.content or b"{}")
                wire.append({"messages": body.get("messages"), "tools": body.get("tools")})

        client = AsyncOpenAI(
            base_url=env.llm_base_url,
            api_key=env.llm_api_key or "none",
            http_client=DefaultAsyncHttpxClient(event_hooks={"request": [sent]}),
        )
        agent = Agent(
            name="complex",
            instructions=r.SYSTEM,
            model=OpenAIChatCompletionsModel(model=str(model), openai_client=client),
        )
        setattr(agent, TOOLS_ATTRIBUTE, list(tools))
        return agent, {"framework_options": {"max_turns": 2 * max_steps}}
    if kind == "claude":
        from claude_agent_sdk import ClaudeAgentOptions

        options = ClaudeAgentOptions(
            model=str(model), system_prompt=r.SYSTEM, max_turns=2 * max_steps
        )
        setattr(options, TOOLS_ATTRIBUTE, list(tools))
        return options, {}
    raise SystemExit(f"no adapter {kind!r}: one of {', '.join(ADAPTERS)}")


class Accounted:
    """What the harness accounted to the run's budget while it is current (``budget.account``,
    every adapter's path): the meter of the adapters whose calls are not LangChain's."""

    def __init__(self, spend: Spend) -> None:
        self.spend = spend
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.cached_tokens = 0
        self.requests: list[dict[str, Any]] = []


_ACCOUNTED: list[Accounted] = []


def _accounting_hook() -> None:
    """``budget.account`` also reports to the current :class:`Accounted` (patched once)."""
    from trellis.harness import budget

    if getattr(budget.account, "_complex", False):
        return
    original = budget.account

    def account(runtime: Any, **kw: Any) -> None:
        original(runtime, **kw)
        if _ACCOUNTED and runtime.parent is None:
            a = _ACCOUNTED[-1]
            a.calls += 1
            a.input_tokens += int(kw.get("input_tokens") or 0)
            a.output_tokens += int(kw.get("output_tokens") or 0)
            a.cached_tokens += int(kw.get("cached") or 0)
            a.spend.add(
                int(kw.get("input_tokens") or 0),
                int(kw.get("output_tokens") or 0),
                int(kw.get("cached") or 0),
            )

    account._complex = True  # type: ignore[attr-defined]
    budget.account = account  # type: ignore[assignment]


def scripted_model(case: Case) -> Any:
    """The existing scripted chat model (``tests.support.chat_model``), replaying the case's
    gold calls and reference answers: proves the pipeline, not a model. A ``keyed`` case's
    answers each call by the message it answers (``KeyedChatModel``)."""
    from tests.support.chat_model import KeyedChatModel, ScriptedChatModel

    if case.keyed:
        script = [
            (
                t.text,
                [
                    *(list(rnd) for rnd in t.calls),
                    (t.expect.reference if t.expect else t.reply) or "Done.",
                ],
            )
            for t in case.turns
        ] + [
            (
                task,
                [
                    list(x) if isinstance(x, tuple) and x and isinstance(x[0], tuple) else x
                    for x in rounds
                ],
            )
            for task, rounds in case.helpers
        ]
        return KeyedChatModel(turns=[], script=script, spends=(1000, 50))
    return ScriptedChatModel(turns=case.script(), spends=(1000, 50))


async def run_case(
    h: Harness,
    case: Case,
    *,
    model: Any,
    history: str,
    window: int | None,
    max_steps: int,
    run_seconds: float | None,
    spend: Spend,
    tag: str,
    adapter: str | None = None,
    agent_id: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """One case's conversation: its row and every request it made (``agent_id``: the wrapped
    agent's id, one per case unless given — a learning-over-time run gives one for all). A
    turn sent while the one before it works (``Turn.mode``) runs beside it: the two share
    their requests (the model is one), and each keeps its own run's calls."""
    w = r.world()
    w.reset()
    from benchmark.cost import adversarial

    adversarial.FILES.clear()  # the suite's file store starts empty for each case
    trace = Trace()
    kind = adapter_of(history, adapter)
    found: dict[str, Any] = {
        "case": case.id,
        "category": case.category,
        "tier": case.tier,
        "edges": list(case.edges),
        "history": history,
        "adapter": kind,
        "stopped": None,
        "turns": [],
    }
    if case.adapters and kind not in case.adapters:
        return found | {"not_applicable": case.only_because or f"only {case.adapters}"}, []
    wire: list[dict[str, Any]] = []
    target, options = target_of(
        model, history, case.window or window, max_steps, adapter=kind, wire=wire,
        tools=(*r.TOOLS, *case.tools),
    )  # fmt: skip
    agent = h.wrap(
        target, id=agent_id or f"complex-{kind}-{case.id}", hooks=[trace],
        run_timeout=run_seconds,
        **options,
    )  # fmt: skip
    records: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []
    stopped = None

    def user_of(t: Turn) -> str:
        """The turn's user, kept to this run and this case: a case's chats share what it
        remembers, and no other case's (nor an earlier run's) leak into it."""
        return f"{t.user or runner.USER}-{tag}-{case.id}"

    async def asked(t: Turn) -> Any:
        return await agent.run(
            t.text,
            user=user_of(t),
            thread=f"cx-{tag}-{case.id}-{t.thread}",
            mode=t.mode or "queue",
        )

    i = 0
    while i < len(case.turns):
        t = case.turns[i]
        beside = case.turns[i + 1] if i + 1 < len(case.turns) else None
        together = [t] if beside is None or beside.mode is None else [t, beside]
        if i and t.thread != case.turns[i - 1].thread:
            # another conversation, later: what the earlier one recorded has reached memory
            await h.writes.drain()
        for _ in range(t.ticks):
            w.tick()
        trace.calls = []
        trace.started.clear()
        wire.clear()
        captured: Any = Captured(spend) if kind in LANGCHAIN_ADAPTERS else Accounted(spend)
        began = time.monotonic()
        ended: list[tuple[str, str, str | None, str | None]] = []
        try:
            if spend.reached:
                raise SpendCapReached(f"spend cap ${spend.cap} reached (${spend.usd:.4f})")
            if isinstance(captured, Accounted):
                _accounting_hook()
                _ACCOUNTED.append(captured)
            try:
                with runner.metered(captured if isinstance(captured, Captured) else None):
                    first = asyncio.create_task(asked(t))
                    tasks = [first]
                    if len(together) == 2:
                        # the second message once the first run works (its first tool call)
                        waiting = asyncio.create_task(trace.started.wait())
                        await asyncio.wait({first, waiting}, return_when=asyncio.FIRST_COMPLETED)
                        waiting.cancel()
                        tasks.append(asyncio.create_task(asked(together[1])))
                    results = await asyncio.gather(*tasks, return_exceptions=True)
            finally:
                if isinstance(captured, Accounted):
                    _ACCOUNTED.remove(captured)
                    captured.requests = list(wire)
            for result in results:
                if isinstance(result, SpendCapReached):
                    raise result
                if isinstance(result, BaseException):
                    ended.append(("", "RAISED", f"{type(result).__name__}: {result}"[:300], None))
                    continue
                answer = (
                    result.answer
                    if isinstance(result.answer, str)
                    else json.dumps(result.answer, default=str)
                )
                error = result.error.message[:300] if result.error is not None else None
                ended.append((answer or "", result.status.value, error, result.run_id))
        except SpendCapReached as exc:
            ended = [("", "STOPPED", str(exc), None) for _ in together]
        if spend.reached and any((e[2] or "").find("spend cap") >= 0 for e in ended):
            stopped = next(e[2] for e in ended if (e[2] or "").find("spend cap") >= 0)
        seconds = round(time.monotonic() - began, 3)
        for n, (turn, (answer, status, error, run_id)) in enumerate(
            zip(together, ended, strict=True)
        ):
            calls = [
                {k: v for k, v in c.items() if k != "run"}
                for c in trace.calls
                if len(together) == 1 or c.get("run") == run_id
            ]
            row: dict[str, Any] = {
                "turn": i + n,
                "thread": turn.thread,
                "user": user_of(turn),
                "mode": turn.mode,
                "text": turn.text,
                "ticks": turn.ticks,
                "answer": answer,
                "status": status,
                "error": error,
                "calls": calls,
                "input_tokens": captured.input_tokens if n == 0 else 0,
                "output_tokens": captured.output_tokens if n == 0 else 0,
                "model_calls": captured.calls if n == 0 else 0,
                "seconds": seconds,
                "checks": checked(
                    turn, status, captured.requests, captured.calls, calls=calls, history=history
                ),
            }
            if turn.expect is not None:
                row["expect"] = _expect_row(turn.expect)
                said = "\n".join(
                    _texts(m) for q in captured.requests for m in q.get("messages") or []
                )
                row["score"] = score(turn.expect, answer, records, turn.text, said)
            records.append(row)
            requests += [
                {"case": case.id, "turn": i + n, "call": c, **q}
                for c, q in enumerate(captured.requests)
            ]
        i += len(together)
        if stopped:
            break
    return found | {"stopped": stopped, "turns": records}, requests


def _expect_row(e: Expect) -> dict[str, Any]:
    d = asdict(e)
    d.pop("reference")
    return d


def failed(row: dict[str, Any]) -> bool:
    """Whether a case needs another run: stopped, a turn that did not succeed, or a strict score
    that failed."""
    if row.get("stopped"):
        return True
    for t in row["turns"]:
        if t["status"] != "SUCCESS" and not any(
            c["check"] == "ends" and c["ok"] for c in t.get("checks", [])
        ):
            return True
        if t.get("score") and t["score"]["correct"] is False:
            return True
        if any(not c["ok"] for c in t.get("checks", [])):
            return True
    return False


def summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Strict accuracy of the scored turns by category and tier, judge-only turns counted, and
    the cost per case."""

    def cell(sel: Sequence[dict[str, Any]]) -> dict[str, Any]:
        scored = [t for c in sel for t in c["turns"] if t.get("score")]
        strict = [t for t in scored if t["score"]["correct"] is not None]
        return {
            "cases": len(sel),
            "scored_turns": len(scored),
            "strict": stats.interval([float(t["score"]["correct"]) for t in strict]),
            "judge_only": len(scored) - len(strict),
            "not_success": sum(t["status"] != "SUCCESS" for c in sel for t in c["turns"]),
            # the harness's own checks (how a run ended, what the model was sent)
            "checks": sum(len(t.get("checks", [])) for c in sel for t in c["turns"]),
            "checks_failed": sum(
                not k["ok"] for c in sel for t in c["turns"] for k in t.get("checks", [])
            ),
            "not_applicable": sum(bool(c.get("not_applicable")) for c in sel),
        }

    every = [t for c in rows for t in c["turns"]]
    return {
        "all": cell(rows),
        "by_category": {
            k: cell([c for c in rows if c["category"] == k])
            for k in CATEGORIES
            if any(c["category"] == k for c in rows)
        },
        "by_tier": {
            str(k): cell([c for c in rows if c["tier"] == k])
            for k in TIERS
            if any(c["tier"] == k for c in rows)
        },
        "cost": {
            f: stats.interval([float(sum(t[f] for t in c["turns"])) for c in rows])
            for f in ("input_tokens", "output_tokens", "model_calls", "seconds")
        },
        "turns": len(every),
    }


def markdown(report: dict[str, Any]) -> str:
    s = report["summary"]
    lines = [
        f"# complex: {report['model']} ({report.get('adapter', '?')}, {report['history']} history)",
        "",
        f"Spend: ${report['usd']:.4f}"
        + (f" (cap ${report['max_usd']})" if report["max_usd"] else "")
        + (f" — STOPPED: {report['stopped']}" if report.get("stopped") else ""),
        "",
        "| group | cases | scored turns | strict correct | judge-only | not success | harness checks failed |",
        "|---|---|---|---|---|---|---|",
    ]
    groups = (
        [("all", s["all"])]
        + list(s["by_category"].items())
        + [(f"tier {k}", v) for k, v in s["by_tier"].items()]
    )
    for name, c in groups:
        lines.append(
            f"| {name} | {c['cases']} | {c['scored_turns']} | {stats.shown(c['strict'], 2)} | {c['judge_only']} "
            f"| {c['not_success']} | {c.get('checks_failed', 0)} of {c.get('checks', 0)} |"
        )
    return "\n".join(lines) + "\n"


def _saved(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write(path: Path, rows: Sequence[dict[str, Any]], mode: str = "a") -> None:
    with path.open(mode) as fh:
        fh.writelines(json.dumps(x, default=str) + "\n" for x in rows)


def _names(c: Case) -> set[str]:
    """What ``--cases`` picks a case by: its id, its category, its tier, and ``adversarial``
    for every ``adv_*`` category."""
    from benchmark.cost import breakfix

    return (
        {c.id, c.category, f"tier:{c.tier}"}
        | ({ADVERSARIAL} if c.category.startswith("adv_") else set())
        | breakfix.names(c)
    )


def picked(names: str, every: Sequence[Case]) -> list[Case]:
    """``names``: case ids, categories, ``tier:N`` or ``adversarial``, comma-separated (empty:
    every case)."""
    wanted = [n.strip() for n in names.split(",") if n.strip()]
    if not wanted:
        return list(every)
    out = [c for c in every if _names(c) & set(wanted)]
    unknown = [n for n in wanted if not any(n in _names(c) for c in every)]
    if unknown:
        raise SystemExit(f"no case, category or tier {unknown}")
    return out


async def bench(
    args: argparse.Namespace,
    *,
    harness: Harness | None = None,
    progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    every = cases()
    chosen = picked(args.cases, every)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    saves, asked = out.with_suffix(".jsonl"), Path(f"{out}.requests.jsonl")
    model_name = "scripted" if args.scripted else args.model
    rows = [
        x
        for x in _saved(saves)
        if x.get("model") == model_name
        and x.get("history") == args.history
        and adapter_of(x["history"], x.get("adapter")) == adapter_of(args.history, args.adapter)
    ]
    if args.rerun_failed:
        again = {x["case"] for x in rows if failed(x)}
        chosen = [c for c in chosen if c.id in again or c.id not in {x["case"] for x in rows}]
        keep = [
            x for x in _saved(saves) if not (x["case"] in again and x.get("model") == model_name)
        ]
        _write(saves, keep, "w")
        _write(
            asked,
            [q for q in _saved(asked) if not (q["case"] in again and q.get("model") == model_name)],
            "w",
        )
        rows = [x for x in rows if x["case"] not in again]
    done = {x["case"] for x in rows}
    price = (
        prices.PRICES.get(args.price)
        if args.price
        else (None if args.scripted else prices.price_of(model_name))
    )
    if args.max_usd is not None and price is None and not args.scripted:
        raise SystemExit(
            f"--max-usd needs a price: {model_name} is not in benchmark.cost.prices (use --price)"
        )
    spend = Spend(price, args.max_usd, usd=sum(x.get("usd", 0.0) for x in rows))
    h = harness or runner.harness(memory=args.history == "memory")
    tag = uuid.uuid4().hex[:6]
    stopped = None
    try:
        for case in chosen:
            if case.id in done:
                continue
            if spend.reached:
                stopped = f"spend cap ${spend.cap} reached before {case.id}"
                break
            before = spend.usd
            model = scripted_model(case) if args.scripted else args.model
            found, requests = await run_case(
                h, case, model=model, history=args.history, window=args.window, max_steps=args.max_steps,
                run_seconds=args.timeout, spend=spend, tag=tag, adapter=args.adapter,
            )  # fmt: skip
            found |= {"model": model_name, "usd": round(spend.usd - before, 6)}
            if found["stopped"]:
                stopped = found["stopped"]
                break  # a case cut by the cap is not kept: a rerun runs it whole
            _write(saves, [found])
            _write(asked, [q | {"model": model_name} for q in requests])
            rows.append(found)
            scored = [t["score"]["correct"] for t in found["turns"] if t.get("score")]
            progress(f"{case.id:<16} tier {case.tier} {scored} ${spend.usd:.4f}")
    finally:
        if harness is None:
            await h.aclose()
    report = {
        "suite": "complex",
        "model": model_name,
        "history": args.history,
        "adapter": adapter_of(args.history, args.adapter),
        "window": args.window,
        "date": datetime.now(UTC).strftime("%Y-%m-%d"),
        "usd": round(spend.usd, 6),
        "max_usd": args.max_usd,
        "stopped": stopped,
        "cases_total": len(every),
        "summary": summary(rows),
    }
    out.with_suffix(".json").write_text(json.dumps(report, indent=2, default=str))
    out.with_suffix(".md").write_text(markdown(report))
    return report


def arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    import os

    env = os.environ.get
    p = argparse.ArgumentParser(
        prog="python -m benchmark.cost.complex",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--model",
        default=env("TRELLIS_BENCH_MODEL") or env("TRELLIS_LIVE_MODEL"),
        help="the model name on the LLM_* endpoint (default: TRELLIS_BENCH_MODEL)",
    )
    p.add_argument(
        "--scripted",
        action="store_true",
        help="the scripted model replaying the gold calls (no endpoint)",
    )
    p.add_argument("--history", choices=("thread", "memory"), default="thread")
    p.add_argument(
        "--adapter",
        choices=ADAPTERS,
        default=None,
        help="the framework wrapped (default: react with memory history, langgraph without)",
    )
    p.add_argument("--cases", default="", help="case ids, categories or tier:N, comma-separated")
    p.add_argument(
        "--rerun-failed",
        action="store_true",
        help="only the cases <out>.jsonl holds as failed (and any not run)",
    )
    p.add_argument(
        "--max-usd", type=float, default=None, help="stop once the run spent this many US dollars"
    )
    p.add_argument(
        "--price", default=None, help="price the tokens as this benchmark.cost.prices model"
    )
    p.add_argument("--window", type=int, default=None, help="the model's context window")
    p.add_argument("--max-steps", type=int, default=12)
    p.add_argument("--timeout", type=float, default=600.0, help="seconds per turn")
    p.add_argument("--out", default="build/bench-complex")
    args = p.parse_args(argv)
    if not args.scripted and not args.model:
        p.error("no model: --model (or TRELLIS_BENCH_MODEL), or --scripted")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    report = asyncio.run(bench(arguments(argv)))
    sys.stdout.write(markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
