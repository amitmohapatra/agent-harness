"""The ``complex`` suite's break-and-fix rounds: long, multi-turn conversations that mix past chats
(two or more earlier conversations of the same user, a decision or a correction made there,
another user's statement that must not leak), analytics over two or three tools (group sums,
joins, % change, scenarios), explanations (which steps, which source, how sure, on the turn just
done and on an earlier one), follow-ups that switch topic, corrections, ambiguity and requests
this deployment cannot do.

Each round adds cases no earlier round used (ids ``bf<round>-…``, tag ``round:<n>``); the
held-out cases (ids ``bfh-…``, tag ``heldout``) are run every round and never fixed against, so
their score stays honest. Categories: ``bf_past``, ``bf_analytics``, ``bf_explain``,
``bf_mixed``, ``bf_cantdo``, ``bf_clarify``. Every figure's truth is computed from the world
(:mod:`benchmark.cost.retail`), as the rest of the suite's is.
"""

from __future__ import annotations

import functools
import itertools
from collections.abc import Callable
from typing import Any, Final

from benchmark.cost import retail as r

CATEGORIES: Final = ("bf_past", "bf_analytics", "bf_explain", "bf_mixed", "bf_cantdo", "bf_clarify")
#: Another person of the same tenant: what they said is never this user's.
OTHER_USER: Final = "trellis-bench-colleague"
#: The memory of bf3-past-02's approval as the scripted model names it (a model reads the id
#: its memory context shows).
SUP12_APPROVAL: Final = "mem_sup12_approval"

Q2: Final = {"start": "2026-04-01", "end": "2026-06-30"}
Q3: Final = {"start": "2026-07-01", "end": "2026-09-30"}
AUG: Final = {"start": "2026-08-01", "end": "2026-08-31"}
SEP: Final = {"start": "2026-09-01", "end": "2026-09-30"}
YTD: Final = {"start": "2026-01-01", "end": "2026-09-30"}
#: An answer that claims a chart it could not make: "here is your chart", "I made the chart".
FAKE_CHART: Final = (
    r"\b(?:here(?: is|'s) (?:the|your|a)|i(?:'ve| have)? ?(?:attached|created|made|generated|saved|"
    r"drew))(?: (?:the|your|a|an))?(?: [\w-]+)? (?:chart|image|png|graph|svg)"
    # a chart given as its SVG source, or drawn as text, in the answer itself, is no claimed image
    r"(?!\s+(?:as\s+)?(?:svg\s+)?(?:source|code|markup)\b|\s+(?:as|in)\s+(?:plain\s+)?(?:text|ascii)\b)"
)
MONTHS: Final = {
    f"2026-{n:02d}": name
    for n, name in enumerate(
        ("January", "February", "March", "April", "May", "June", "July", "August", "September"), 1
    )
}


def cases() -> list[Any]:
    from benchmark.cost import complex as cx

    w = r.world()
    by_name = w.by_name
    Case, Turn = cx.Case, cx.Turn
    out: list[Any] = []

    def formula(review: int = 7) -> str:
        return cx.FORMULA.format(review=review)

    @functools.cache  # each SKU's average once a build: a scan of the sales lines
    def avg28(sku: str, region: str | None) -> float:
        return r.truth_avg_daily_units(w, sku, start=cx.SEP28[0], end=cx.SEP28[1], region=region)

    def reorder(sku: str, region: str | None, *, lead: int | None = None, review: int = 7,
                version: int = 0) -> int:  # fmt: skip
        sup = w.suppliers[w.skus[sku].supplier]
        regions = (region,) if region else r.REGIONS
        on = sum(w.on_hand(sku, g, version) for g in regions)
        return r.reorder_qty(avg28(sku, region), lead if lead is not None else sup.lead_time_days,
                             review, on)  # fmt: skip

    def total_on(sku: str, version: int = 0) -> int:
        return sum(w.on_hand(sku, g, version) for g in r.REGIONS)

    def rev(**f: Any) -> float:
        return r.truth_total(w, "revenue", **f)

    def units(**f: Any) -> float:
        return r.truth_total(w, "units", **f)

    def named(sku: str) -> tuple[str, str]:
        return (sku, w.skus[sku].name)

    def first(pick: Callable[[r.Sku], bool], skip: int = 0) -> r.Sku:
        """The SKU after ``skip`` others that ``pick`` takes, in SKU order: the rest unchecked."""
        found = (k for k in sorted(w.skus.values(), key=lambda k: k.sku) if pick(k))
        return next(itertools.islice(found, skip, None))

    def moves_north(k: r.Sku) -> bool:
        """Its north reorder quantity changes when the clock ticks once (and is above 0)."""
        a, b = reorder(k.sku, "north"), reorder(k.sku, "north", version=1)
        return a > 0 and b > 0 and a != b and avg28(k.sku, "north") > 0.5

    def portal_differs(k: r.Sku) -> bool:
        """Its supplier's portal lead time differs from the master's, and the north reorder
        quantity is above 0 with either."""
        s = w.suppliers[k.supplier]
        return (
            s.portal_lead_time_days != s.lead_time_days
            and avg28(k.sku, "north") > 0.5
            and reorder(k.sku, "north") > 0
            and reorder(k.sku, "north", lead=s.portal_lead_time_days) > 0
        )

    # ================================================================== round 1
    # ---------------------------------------------------------------- past chats
    a = first(lambda k: k.category == "knitwear" and avg28(k.sku, "north") > 0.5, 3)
    out.append(Case(
        "bf1-past-01", "bf_past", 3,
        (
            Turn(f"For the record: I approved a north reorder of 120 units of the {a.name}. "
                 "Please keep that in mind.", reply="Noted: 120 units, north."),
            Turn(f"Change of plan on the {a.name}: make my approved north order 90 units, not 120.",
                 reply="Updated: 90 units.", thread="b"),
            Turn(f"What north order quantity do I currently have approved for the {a.name}, "
                 "and what was it before I changed it?",
                 expect=cx.record({"now": 90, "before": 120}, tol={"now": 0, "before": 0},
                                  forbidden=()), thread="c"),
        ),
        ("approval recalled across chats", "a correction made in another chat: the latest wins"),
    ))  # fmt: skip
    west, east = rev(region="west", **AUG), rev(region="east", **AUG)
    out.append(Case(
        "bf1-past-02", "bf_past", 3,
        (
            Turn("What was the west region's revenue in August 2026?", expect=cx.money(west)),
            Turn("What was the east region's revenue in August 2026?", expect=cx.money(east),
                 thread="b"),
            Turn("In my last two chats I asked about August revenue for two regions. Which two "
                 "regions were they, and how much higher was the larger one, in pounds and as a "
                 "% of the smaller one?",
                 expect=cx.record({
                     "difference": abs(west - east),
                     "pct": abs(west - east) / min(west, east) * 100,
                 }, tol={"pct": 0.06}), thread="c"),
        ),
        ("recall across two earlier chats", "compute over recalled figures"),
    ))  # fmt: skip
    out.append(Case(
        "bf1-past-03", "bf_past", 2,
        (
            Turn("Note for me: our target gross margin for denim is 61%.", user=OTHER_USER,
                 reply="Noted."),
            Turn("What target gross margin did I set for denim?",
                 expect=cx.Expect("abstain", forbidden=("61",),
                                  reference="I can't answer that from the data: you haven't set a denim margin target with me."),
                 thread="b"),
        ),
        ("another user's statement never answers this user",),
    ))  # fmt: skip
    fw_q3 = rev(category="footwear", **Q3)
    out.append(Case(
        "bf1-past-04", "bf_past", 2,
        (
            Turn("From now on, whenever I ask for revenue, give it in thousands of pounds rounded "
                 "to one decimal, like £708.4k.", reply="Will do."),
            Turn("What was footwear revenue in Q3 2026?",
                 expect=cx.Expect("number", round(fw_q3 / 1000, 1), tol=0.06,
                                  reference=f"£{fw_q3 / 1000:.1f}k",
                                  rubric="Gives the figure in thousands of pounds with one "
                                         "decimal, as the user asked in an earlier chat."),
                 thread="b"),
        ),
        ("a preference from an earlier chat applied",),
    ))  # fmt: skip

    # ---------------------------------------------------------------- analytics
    growth = {g: r.growth_pct(rev(region=g, **Q2), rev(region=g, **Q3)) for g in r.REGIONS}
    fastest = max(growth, key=lambda g: growth[g])
    slowest = min(growth, key=lambda g: growth[g])
    out.append(Case(
        "bf1-analytics-01", "bf_analytics", 3,
        (
            Turn("For each region, give Q3 2026 revenue and its % change versus Q2 2026.",
                 expect=cx.record({f"{g}_pct": growth[g] for g in r.REGIONS},
                                  tol={f"{g}_pct": 0.06 for g in r.REGIONS})),
            Turn("Which region grew fastest, and by how many percentage points more than the "
                 "slowest?",
                 expect=cx.record({"fastest": fastest,
                                   "points": growth[fastest] - growth[slowest]},
                                  tol={"points": 0.06})),
        ),
        ("% change per group", "rank and difference of the computed values"),
    ))  # fmt: skip
    gp = {c: r.truth_gross_profit(w, category=c, **Q3) for c in ("knitwear", "outerwear")}
    mp = {c: gp[c] / rev(category=c, **Q3) * 100 for c in gp}
    out.append(Case(
        "bf1-analytics-02", "bf_analytics", 3,
        (
            Turn("What was knitwear's gross profit in Q3 2026 (revenue minus units x unit cost)?",
                 expect=cx.money(gp["knitwear"])),
            Turn("Now as a margin % of its revenue.", expect=cx.pct(mp["knitwear"])),
            Turn("Same two figures for outerwear, and which category has the higher margin %?",
                 expect=cx.record({"profit": gp["outerwear"], "margin": mp["outerwear"],
                                   "higher": max(mp, key=lambda c: mp[c])},
                                  tol={"margin": 0.06})),
        ),
        ("join of sales lines with the catalogue", "follow-ups that reuse the method"),
    ))  # fmt: skip
    top3 = r.truth_supplier_revenue(w, **YTD)[:3]

    def score(s: r.Supplier, lead: int) -> float:
        return 0.6 * s.on_time_rate + 0.4 * (1 - lead / 60)

    sups = [w.suppliers[s] for s, _ in top3]
    out.append(Case(
        "bf1-analytics-03", "bf_analytics", 3,
        (
            Turn("Top 3 suppliers by 2026 revenue so far: for each, its master lead time, on-time "
                 f"rate and weighted score ({cx.WEIGHTED}).",
                 expect=cx.record({f"{s.supplier}_score": score(s, s.lead_time_days)
                                   for s in sups},
                                  tol={f"{s.supplier}_score": 0.0006 for s in sups})),
            Turn("Sorry — use the supplier portal's lead time instead of the master's for the "
                 "score. What are the three scores now, and does the order change?",
                 expect=cx.record({f"{s.supplier}_score": score(s, s.portal_lead_time_days)
                                   for s in sups},
                                  tol={f"{s.supplier}_score": 0.0006 for s in sups})),
        ),
        ("join of three tools", "a correction that changes the source"),
    ))  # fmt: skip
    sc_a, sc_b = fw_q3 * 1.08 * 0.95, fw_q3 * 1.03
    out.append(Case(
        "bf1-analytics-04", "bf_analytics", 2,
        (
            Turn("Two scenarios on Q3 2026 footwear revenue. A: prices up 8% and units down 5%. "
                 "B: prices unchanged and units up 3%. What would revenue be under each, and "
                 "which is higher by how much?",
                 expect=cx.record({"A": sc_a, "B": sc_b, "difference": abs(sc_a - sc_b)})),
        ),
        ("scenario comparison",),
    ))  # fmt: skip

    # ---------------------------------------------------------------- explanation
    south_sep = r.truth_top(w, "store", "revenue", 1, region="south", **SEP)[0]
    out.append(Case(
        "bf1-explain-01", "bf_explain", 2,
        (
            Turn("Which south store had the highest revenue in September 2026, and how much?",
                 ((("sales_summary", {"group_by": "store", "metric": "revenue",
                                      "region": "south", "top": 1, **SEP}),),),
                 cx.record({"store": south_sep[0], "revenue": south_sep[1]})),
            Turn("Which tool calls did you make for that, in order, and with which filters?",
                 expect=cx.Expect("tools", of_turn=0, rubric="Names the calls the record "
                                  "holds, with their filters; nothing it did not call.",
                                  reference="One call: sales_summary, grouped by store, "
                                  "revenue, region south, 2026-09-01 to 2026-09-30, top 1.")),
            Turn("How sure are you of that store and figure, and what could make them wrong?",
                 expect=cx.judge("Grounded in the call it made (the sales data for the south "
                                 "in September); names a real risk (data covers to 2026-09-30, "
                                 "store revenue after discounts); no invented certainty.",
                                 "Sure: one sales_summary call for the south in September.")),
        ),
        ("steps from the run record", "how sure, from the evidence"),
    ))  # fmt: skip
    pk = first(portal_differs, 2)
    ps = w.suppliers[pk.supplier]
    out.append(Case(
        "bf1-explain-02", "bf_explain", 3,
        (
            Turn(f"Reorder qty of the {pk.name} for the north? {formula()} (north only).",
                 expect=cx.count(reorder(pk.sku, "north"))),
            Turn("Unrelated: how many stores do we have in the east?", expect=cx.count(10)),
            Turn("Going back to the reorder number two questions ago: which lead time did you use "
                 "and from which source, and what would the number be with the supplier portal's "
                 "lead time instead?",
                 expect=cx.record({"lead_used": ps.lead_time_days,
                                   "with_portal": reorder(pk.sku, "north",
                                                          lead=ps.portal_lead_time_days)},
                                  tol={"lead_used": 0, "with_portal": 0})),
        ),
        ("explain a turn two back", "redo with the other source"),
    ))  # fmt: skip
    d2, d3 = units(category="denim", **Q2), units(category="denim", **Q3)
    out.append(Case(
        "bf1-explain-03", "bf_explain", 2,
        (
            Turn("Total denim units sold in Q2 2026?", expect=cx.count(int(d2))),
            Turn("Sorry, I meant Q3.", expect=cx.count(int(d3))),
            Turn("Walk me through the two numbers you gave me: which quarter each was, and the % "
                 "change from the first to the second.",
                 expect=cx.record({"q2": d2, "q3": d3, "pct": r.growth_pct(d2, d3)},
                                  tol={"q2": 0, "q3": 0, "pct": 0.06})),
        ),
        ("explain across a correction",),
    ))  # fmt: skip

    # ---------------------------------------------------------------- mixed
    m1 = first(moves_north, 1)
    q0, q1 = reorder(m1.sku, "north"), reorder(m1.sku, "north", version=1)
    out.append(Case(
        "bf1-mixed-01", "bf_mixed", 3,
        (
            Turn(f"Reorder qty of the {m1.name} for the north? {formula()} (north only).",
                 expect=cx.count(q0)),
            Turn("Approve that quantity — note it as my approved north order for this item.",
                 reply="Noted as approved."),
            Turn(f"Compare the north order I approved for the {m1.name} with what the same "
                 "formula gives with today's stock, and explain why they differ.",
                 expect=cx.record({"approved": q0, "today": q1}, tol={"approved": 0, "today": 0},
                                  rubric="Says the approved figure (from the earlier chat) and "
                                         "today's, and that north stock on hand changed since."),
                 thread="b", ticks=1),
        ),
        ("an approval from an earlier chat against today's data", "why they differ"),
    ))  # fmt: skip
    m2 = first(lambda k: k.category == "denim" and total_on(k.sku) != total_on(k.sku, 1), 4)
    out.append(Case(
        "bf1-mixed-02", "bf_mixed", 3,
        (
            Turn(f"Total on hand of the {m2.name} across all regions?",
                 expect=cx.count(total_on(m2.sku))),
            Turn("Redo the stock total from my earlier chat with today's data: what is it now, "
                 "what was it then, and how sure are you of each?",
                 expect=cx.record({"now": total_on(m2.sku, 1), "then": total_on(m2.sku)},
                                  tol={"now": 0, "then": 0},
                                  rubric="Now from a fresh read, then from the earlier chat; "
                                         "sure of each as of its read time."),
                 thread="b", ticks=1),
        ),
        ("redo an earlier chat's figure on new data", "how sure"),
    ))  # fmt: skip
    kn = r.truth_top(w, "sku", "revenue", 3, category="knitwear", **Q3)
    dn = r.truth_top(w, "sku", "units", 3, category="denim", **Q3)
    second = w.skus[dn[1][0]]
    ss = w.suppliers[second.supplier]
    out.append(Case(
        "bf1-mixed-03", "bf_mixed", 3,
        (
            Turn("Top 3 knitwear SKUs by Q3 2026 revenue?",
                 expect=cx.idset([k for k, _ in kn],
                                 names={k: w.skus[k].name for k, _ in kn})),
            Turn("Same for denim, but by units.",
                 expect=cx.idset([k for k, _ in dn],
                                 names={k: w.skus[k].name for k, _ in dn})),
            Turn("For the second one in that list, who's the supplier and what's their on-time "
                 "rate?",
                 expect=cx.record({"supplier": (ss.supplier, ss.name),
                                   "on_time": ss.on_time_rate},
                                  tol={"on_time": 0.005})),
            Turn("And that SKU's gross margin %?",
                 expect=cx.pct(r.margin_pct(second.price, second.unit_cost))),
        ),
        ("a category switch mid-conversation", "pronoun chain"),
    ))  # fmt: skip
    z = r.truth_top(w, "sku", "revenue", 5, category="activewear", **Q3)[4]
    zk = w.skus[z[0]]
    scen = z[1] * 0.9 * 1.15
    out.append(Case(
        "bf1-mixed-04", "bf_mixed", 3,
        (
            Turn(f"What was the {zk.name}'s revenue in Q3 2026?", expect=cx.money(z[1])),
            Turn(f"Take the Q3 revenue you gave me for the {zk.name} in our other chat. If its "
                 "price had been 10% lower and its units 15% higher, what would that revenue have "
                 "been, and the change in pounds?",
                 expect=cx.record({"scenario": scen, "change": scen - z[1]}), thread="b"),
        ),
        ("a what-if on a figure from an earlier chat",),
    ))  # fmt: skip
    navy = by_name("Merino Crew Sweater Navy")
    nq3 = {"units": units(sku=navy.sku, **Q3), "revenue": rev(sku=navy.sku, **Q3)}
    months = r.truth_monthly(w, "units", sku=navy.sku, **Q3)
    best = max(months, key=lambda kv: kv[1])
    out.append(Case(
        "bf1-mixed-05", "bf_mixed", 3,
        (
            Turn("How did the sweater do last quarter?",
                 expect=cx.clarify(["Merino Crew Sweater", "Cashmere Crew Sweater",
                                    "Cotton Crew Sweater", "Chunky Crew Sweater"])),
            Turn("The Merino Crew Sweater Navy, Q3 2026: units and revenue.",
                 expect=cx.record(nq3, tol={"units": 0})),
            Turn("Plot its weekly units as a chart image I can paste into a slide.",
                 expect=cx.cannot(FAKE_CHART,
                                  r"table|weekly|figures|numbers|instead|list|here(?: is|'s) the",
                                  "I can't make an image here: there is no code execution, files or images in this setup. "
                                  "I can give the weekly figures as a table instead.")),
            Turn("Fine — then its best month of Q3 by units, and that month's units.",
                 expect=cx.record({"month": (best[0], MONTHS[best[0]]), "units": best[1]},
                                  tol={"units": 0})),
        ),
        ("clarify, then analytics, then can't-do, then a follow-up",),
    ))  # fmt: skip
    m6 = first(lambda k: avg28(k.sku, "north") > 0.2 and k.category == "outerwear", 1)
    s6 = w.suppliers[m6.supplier]
    told = s6.lead_time_days + 9
    out.append(Case(
        "bf1-mixed-06", "bf_mixed", 3,
        (
            Turn(f"Note: {s6.supplier}'s lead time is {told} days, from my call with them today.",
                 reply="Noted."),
            Turn(f"Reorder qty of the {m6.name} for the north, using the lead time I told you for "
                 f"its supplier. {formula()} (north only).",
                 expect=cx.count(reorder(m6.sku, "north", lead=told)), thread="b"),
            Turn("And with the supplier master's lead time instead?",
                 expect=cx.count(reorder(m6.sku, "north")), thread="b"),
            Turn("Which of the two should I trust, and why?",
                 expect=cx.judge("Weighs the user's fresh call against the master's update date; "
                                 "names both figures; no invented certainty.",
                                 "The call is newer than the master record."),
                 thread="b"),
        ),
        ("a user-stated value from an earlier chat against the system's",),
    ))  # fmt: skip

    # ---------------------------------------------------------------- can't-do
    dq3 = {"units": units(category="denim", **Q3), "revenue": rev(category="denim", **Q3)}
    out.append(Case(
        "bf1-cantdo-01", "bf_cantdo", 2,
        (
            Turn("Export all Q3 2026 denim sales lines to an Excel file I can download.",
                 expect=cx.cannot(r"\b(?:here(?: is|'s) (?:the|your|a)|i(?:'ve| have) (?:attached|created|exported|saved))\b.{0,40}(?:xlsx|excel|file)",
                                  r"total|summar|figures|table|instead|units|revenue",
                                  "I can't make a file here: there is no code execution, files or images in this setup. "
                                  "I can give the totals instead.")),
            Turn("Then just tell me Q3 2026 denim units and revenue.",
                 expect=cx.record(dq3, tol={"units": 0})),
        ),
        ("can't-do, then what it can do",),
    ))  # fmt: skip
    legs = [k.sku for k in w.skus.values() if "Leggings" in k.name]
    leg_rev = sum(rev(sku=k, **Q3) for k in legs)
    leg_units = sum(units(sku=k, **Q3) for k in legs)
    out.append(Case(
        "bf1-cantdo-02", "bf_cantdo", 3,
        (
            Turn("What's our competitors' average price for leggings this quarter?",
                 expect=cx.abstain("no competitor prices in the data")),
            Turn("OK — what was our own average selling price for leggings (all of them) in Q3 "
                 "2026, as revenue divided by units?",
                 expect=cx.money(leg_rev / leg_units)),
        ),
        ("no data for it, then an aggregate over a product type",),
    ))  # fmt: skip

    # ---------------------------------------------------------------- clarify
    tc = by_name("Classic Trench Coat Camel")
    out.append(Case(
        "bf1-clarify-01", "bf_clarify", 2,
        (
            Turn("What's the lead time of the trench coat's supplier?",
                 expect=cx.clarify(["Insulated Trench Coat", "Waxed Trench Coat",
                                    "Lightweight Trench Coat", "Quilted Trench Coat",
                                    "Classic Trench Coat"])),
            Turn("The Classic one, in Camel.",
                 expect=cx.record({"lead": w.suppliers[tc.supplier].lead_time_days},
                                  tol={"lead": 0})),
        ),
        ("one short question, then the answer",),
    ))  # fmt: skip

    # ================================================================== round 2
    # ---------------------------------------------------------------- analytics
    def asp(**f: Any) -> float:
        return rev(**f) / units(**f)

    fa = {"rev": rev(category="footwear", **AUG), "units": units(category="footwear", **AUG)}
    ka = {"rev": rev(category="knitwear", **AUG), "units": units(category="knitwear", **AUG)}
    asp_aug = {c: asp(category=c, **AUG) for c in ("footwear", "knitwear")}
    asp_sep = {c: asp(category=c, **SEP) for c in ("footwear", "knitwear")}
    out.append(Case(
        "bf2-analytics-01", "bf_analytics", 3,
        (
            Turn("Revenue and units for footwear in August 2026?",
                 expect=cx.record({"revenue": fa["rev"], "units": fa["units"]}, tol={"units": 0})),
            Turn("Same for knitwear.",
                 expect=cx.record({"revenue": ka["rev"], "units": ka["units"]}, tol={"units": 0})),
            Turn("Which of the two had the higher average selling price (revenue / units), and "
                 "by how much per unit?",
                 expect=cx.record({"higher": max(asp_aug, key=lambda c: asp_aug[c]),
                                   "per_unit": abs(asp_aug["footwear"] - asp_aug["knitwear"])},
                                  tol={"per_unit": 0.01})),
            Turn("Now the same comparison for September.",
                 expect=cx.record({"higher": max(asp_sep, key=lambda c: asp_sep[c]),
                                   "per_unit": abs(asp_sep["footwear"] - asp_sep["knitwear"])},
                                  tol={"per_unit": 0.01})),
        ),
        ("the period carried through four follow-ups",),
    ))  # fmt: skip
    top_u = r.truth_top(w, "supplier", "units", 3, **YTD)
    tsups = [w.suppliers[s] for s, _ in top_u]
    worst = min(tsups, key=lambda s: s.on_time_rate)
    share = dict(top_u)[worst.supplier] / sum(u for _, u in top_u) * 100
    out.append(Case(
        "bf2-analytics-02", "bf_analytics", 3,
        (
            Turn("The 3 suppliers with the most units sold in 2026 so far (all their SKUs): each "
                 "one's units, on-time rate and minimum order quantity.",
                 expect=cx.record({**{f"{s.supplier}_units": dict(top_u)[s.supplier]
                                      for s in tsups},
                                   **{f"{s.supplier}_moq": s.min_order_qty for s in tsups}},
                                  tol={**{f"{s.supplier}_units": 0 for s in tsups},
                                       **{f"{s.supplier}_moq": 0 for s in tsups}})),
            Turn("Which of those three has the lowest on-time rate, and what share of the three's "
                 "combined units is theirs, in %?",
                 expect=cx.record({"supplier": (worst.supplier, worst.name), "share": share},
                                  tol={"share": 0.06})),
        ),
        ("a group, then a join with the supplier master, then a share",),
    ))  # fmt: skip
    em = dict(r.truth_monthly(w, "revenue", region="east", **Q3))
    jul, aug, sep = em["2026-07"], em["2026-08"], em["2026-09"]
    out.append(Case(
        "bf2-analytics-03", "bf_analytics", 3,
        (
            Turn("East region revenue for July, August and September 2026, and the "
                 "month-on-month % changes.",
                 expect=cx.record({"jul": jul, "aug": aug, "sep": sep,
                                   "aug_pct": r.growth_pct(jul, aug),
                                   "sep_pct": r.growth_pct(aug, sep)},
                                  tol={"aug_pct": 0.06, "sep_pct": 0.06})),
            Turn("If September had grown at July-to-August's rate instead, what would September "
                 "revenue have been, and how far from the actual?",
                 expect=cx.record({"would_be": aug * aug / jul,
                                   "gap": abs(aug * aug / jul - sep)})),
        ),
        ("a monthly series, its changes, and a what-if on them",),
    ))  # fmt: skip
    wu = [s for s, _ in r.truth_top(w, "store", "units", 5, region="west", **Q3)]
    wr = r.truth_top(w, "store", "revenue", 5, region="west", **Q3)
    out.append(Case(
        "bf2-analytics-04", "bf_analytics", 3,
        (
            Turn("Top 5 west stores by Q3 2026 units?", expect=cx.idset(wu)),
            Turn("Sorry, by revenue, not units.", expect=cx.idset([s for s, _ in wr])),
            Turn("What share of the west's Q3 revenue do those five make up, in %?",
                 expect=cx.pct(sum(v for _, v in wr) / rev(region="west", **Q3) * 100)),
        ),
        ("a correction of the metric, then a share over the corrected list",),
    ))  # fmt: skip

    # ---------------------------------------------------------------- past chats
    out.append(Case(
        "bf2-past-01", "bf_past", 3,
        (
            Turn("My budget cap for the autumn knitwear reorder is £40,000.", reply="Noted."),
            Turn("Raise the knitwear reorder budget to £55,000.", reply="Noted.", thread="b"),
            Turn("Actually cut the knitwear reorder budget back to £48,000; finance pushed back.",
                 reply="Noted.", thread="c"),
            Turn("What is my current knitwear reorder budget, and what were the two earlier "
                 "figures, in order?",
                 expect=cx.record({"now": 48000, "first": 40000, "second": 55000},
                                  tol={"now": 0, "first": 0, "second": 0}), thread="d"),
        ),
        ("a value corrected twice across three chats: the latest wins, the history kept",),
    ))  # fmt: skip
    out.append(Case(
        "bf2-past-02", "bf_past", 3,
        (
            Turn("The approved denim markdown for October is 25%.", user=OTHER_USER,
                 reply="Noted."),
            Turn("Our approved denim markdown for October is 15%.", reply="Noted.", thread="b"),
            Turn("What denim markdown did I approve for October?",
                 expect=cx.Expect("number", 15, tol=0, forbidden=("25",), reference="15%."),
                 thread="c"),
        ),
        ("a colleague's conflicting value never answers this user",),
    ))  # fmt: skip
    nq, sq = rev(region="north", **Q3), rev(region="south", **Q3)
    out.append(Case(
        "bf2-past-03", "bf_past", 3,
        (
            Turn("What was the north region's revenue in Q3 2026?", expect=cx.money(nq)),
            Turn("What was the south region's revenue in Q3 2026?", expect=cx.money(sq),
                 thread="b"),
            Turn("Using the two regional Q3 figures you gave me in our earlier chats, what's "
                 "their combined total, and the north's share of it in %?",
                 expect=cx.record({"total": nq + sq, "north_share": nq / (nq + sq) * 100},
                                  tol={"north_share": 0.06}), thread="c"),
        ),
        ("the assistant's own figures from two earlier chats",),
    ))  # fmt: skip
    big3 = sum(rev(category=c, **Q3) for c in ("footwear", "knitwear", "outerwear"))
    out.append(Case(
        "bf2-past-04", "bf_past", 2,
        (
            Turn("When I say 'the big three' I mean footwear, knitwear and outerwear.",
                 reply="Noted."),
            Turn("What was the big three's combined revenue in Q3 2026?", expect=cx.money(big3),
                 thread="b"),
        ),
        ("a term the user defined in an earlier chat",),
    ))  # fmt: skip

    # ---------------------------------------------------------------- explanation
    ex = first(lambda k: k.category == "footwear" and avg28(k.sku, "north") > 0.5
               and reorder(k.sku, "north") > 0, 2)  # fmt: skip
    exs = w.suppliers[ex.supplier]
    u28 = sum(s.units for s in r.selected(w, sku=ex.sku, region="north",
                                          start=cx.SEP28[0], end=cx.SEP28[1]))  # fmt: skip
    out.append(Case(
        "bf2-explain-01", "bf_explain", 3,
        (
            Turn(f"Reorder qty of the {ex.name} for the north? {formula()} (north only).",
                 expect=cx.count(reorder(ex.sku, "north"))),
            Turn(f"Yesterday you gave me a north reorder number for the {ex.name}. What was it, "
                 "and how did you get it: the units sold over the 28 days, the lead time and its "
                 "source, and the units on hand?",
                 expect=cx.record({"qty": reorder(ex.sku, "north"), "units_28d": u28,
                                   "lead": exs.lead_time_days,
                                   "on_hand": w.on_hand(ex.sku, "north", 0)},
                                  tol={"qty": 0, "units_28d": 0, "lead": 0, "on_hand": 0}),
                 thread="b"),
        ),
        ("explain an earlier chat's figure from its inputs",),
    ))  # fmt: skip
    jtop = r.truth_top(w, "category", "revenue", 2, start="2026-07-01", end="2026-07-31")
    out.append(Case(
        "bf2-explain-02", "bf_explain", 3,
        (
            Turn("Top category by revenue in July 2026?", expect=cx.idset([jtop[0][0]])),
            Turn("How many stores are in the north?", expect=cx.count(10)),
            Turn("What's the current price of SKU-00042?",
                 expect=cx.money(w.skus["SKU-00042"].price)),
            Turn("Go back to the first answer: which tool and filters did you use, and what was "
                 "the runner-up category's revenue?",
                 expect=cx.record({"runner_up": jtop[1][0], "revenue": jtop[1][1]})),
        ),
        ("explain a turn three back, after two topic switches",),
    ))  # fmt: skip
    s3 = w.suppliers["SUP-08"]
    out.append(Case(
        "bf2-explain-03", "bf_explain", 2,
        (
            Turn("What's SUP-08's lead time?",
                 expect=cx.judge("Gives the master's figure and, if it checked, the portal's; "
                                 "says which is which.", f"{s3.lead_time_days} days (master).")),
            Turn("How sure are you, and which figure would you use for a reorder placed today?",
                 # the master named by its figure or by its date, as the rubric asks
                 expect=cx.record({"master": (f"{s3.lead_time_days} days", f"{s3.lead_time_days}-day",
                                              r.MASTER_UPDATED),
                                   "portal": s3.portal_lead_time_days},
                                  tol={"master": 0, "portal": 0},
                                  rubric="Names both sources and their update dates; prefers the "
                                         "newer portal figure for today, or says why not.")),
        ),
        ("how sure, when two sources disagree",),
    ))  # fmt: skip

    # ---------------------------------------------------------------- mixed
    mx = first(moves_north, 4)
    s_mx = w.suppliers[mx.supplier]
    out.append(Case(
        "bf2-mixed-01", "bf_mixed", 3,
        (
            Turn(f"Reorder qty of the {mx.name} for the north? {formula()} (north only).",
                 expect=cx.count(reorder(mx.sku, "north"))),
            Turn("Approve it as my north order for this item.", reply="Approved."),
            Turn(f"Today's stock is in. How far is my approved quantity for the {mx.name} from "
                 "what the formula gives today, and what would today's number be if the lead "
                 "time were 5 days longer?",
                 expect=cx.record({"approved": reorder(mx.sku, "north"),
                                   "today": reorder(mx.sku, "north", version=1),
                                   "longer": reorder(mx.sku, "north", version=1,
                                                     lead=s_mx.lead_time_days + 5)},
                                  tol={"approved": 0, "today": 0, "longer": 0}),
                 thread="b", ticks=1),
        ),
        ("an approval, new data and a what-if in one question",),
    ))  # fmt: skip
    ou = r.truth_top(w, "sku", "units", 3, category="outerwear", **Q3)
    ac = r.truth_top(w, "sku", "revenue", 3, category="accessories", **Q3)
    o1 = w.suppliers[w.skus[ou[0][0]].supplier]
    out.append(Case(
        "bf2-mixed-02", "bf_mixed", 3,
        (
            Turn("Top 3 outerwear SKUs by Q3 2026 units?",
                 expect=cx.idset([k for k, _ in ou], names={k: w.skus[k].name for k, _ in ou})),
            Turn("And the top 3 accessories by Q3 revenue?",
                 expect=cx.idset([k for k, _ in ac], names={k: w.skus[k].name for k, _ in ac})),
            Turn("Who supplies the first one?",
                 expect=cx.record({"supplier": (w.skus[ac[0][0]].supplier,
                                                w.suppliers[w.skus[ac[0][0]].supplier].name)})),
            Turn("No, I meant the first outerwear one.",
                 expect=cx.record({"supplier": (o1.supplier, o1.name)})),
        ),
        ("anaphora after a switch, then a correction of which list",),
    ))  # fmt: skip
    dg3, dg2 = (r.truth_gross_profit(w, category="denim", **q) for q in (Q3, Q2))
    out.append(Case(
        "bf2-mixed-03", "bf_mixed", 3,
        (
            Turn("Q3 2026 gross profit for denim (revenue minus units x unit cost)?",
                 expect=cx.money(dg3)),
            Turn("Redo the denim gross-profit figure from our other chat but for Q2 2026, and "
                 "give the % change from Q2 to Q3.",
                 expect=cx.record({"q2": dg2, "pct": r.growth_pct(dg2, dg3)},
                                  tol={"pct": 0.06}), thread="b"),
        ),
        ("redo an earlier chat's method on another period",),
    ))  # fmt: skip
    s14 = r.selected(w, store="S14", **YTD)
    out.append(Case(
        "bf2-mixed-04", "bf_mixed", 3,
        (
            Turn("Give me every 2026 sales line for store S14 as a CSV I can download.",
                 # the lines given inline, said to be no file, are not made up
                 expect=cx.cannot(r"\b(?:here(?: is|'s) (?:the|your|a) (?:csv )?file|i(?:'ve| "
                                  r"have) (?:attached|created|exported|saved))\b.{0,40}"
                                  r"(?:csv|file)",
                                  r"count|total|summar|figures|instead|lines|revenue",
                                  "I can't make a file here: there is no code execution, files "
                                  "or images in this setup. I can give totals instead.")),
            Turn("Then: how many sales lines does S14 have in 2026, and what was its single "
                 "biggest line's revenue?",
                 expect=cx.record({"lines": len(s14), "max": max(s.revenue for s in s14)},
                                  tol={"lines": 0, "max": 0.005})),
        ),
        ("an export refused honestly, then exact compute over the large result",),
    ))  # fmt: skip
    fl = {m: rev(store="S22", start=f"2026-{m}-01", end=f"2026-{m}-{d}")
          for m, d in (("08", "31"), ("09", "30"))}  # fmt: skip
    out.append(Case(
        "bf2-mixed-05", "bf_mixed", 3,
        (
            Turn("Note: 'our flagship' means store S22.", reply="Noted."),
            Turn("How did our flagship do in September 2026 versus August 2026: revenue in each "
                 "month and the % change?",
                 expect=cx.record({"aug": fl["08"], "sep": fl["09"],
                                   "pct": r.growth_pct(fl["08"], fl["09"])},
                                  tol={"pct": 0.06},
                                  rubric="Says September fell slightly versus August."),
                 thread="b"),
        ),
        ("a name defined in an earlier chat, then compute",),
    ))  # fmt: skip
    kr = rev(category="knitwear", **Q3)
    out.append(Case(
        "bf2-mixed-06", "bf_mixed", 3,
        (
            Turn("Scenario: knitwear Q3 2026 units 10% higher at the same prices. What would Q3 "
                 "knitwear revenue have been?", expect=cx.money(kr * 1.1)),
            Turn("Make it 10% more units but prices 4% lower.", expect=cx.money(kr * 1.1 * 0.96)),
            Turn("Unit costs don't change. Which of the two scenarios gives more gross profit, "
                 "and by how much?",
                 expect=cx.record({"difference": kr * 1.1 * 0.04},
                                  rubric="Says the first scenario (same prices) gives more.")),
        ),
        ("a scenario corrected, then compared on another measure",),
    ))  # fmt: skip

    # ---------------------------------------------------------------- can't-do
    fq = {"revenue": fw_q3, "units": units(category="footwear", **Q3)}
    out.append(Case(
        "bf2-cantdo-01", "bf_cantdo", 2,
        (
            Turn("Email the Q3 2026 footwear numbers to my manager.",
                 expect=cx.judge("Says it cannot send email here (no email tool), says so "
                                 "without pretending it sent anything, and offers the numbers.",
                                 "I can't send email here; here are the numbers to forward.")),
            Turn("OK, what are they: revenue and units?",
                 expect=cx.record(fq, tol={"units": 0})),
        ),
        ("no tool for the action, then the data",),
    ))  # fmt: skip
    wk = r.truth_top(w, "sku", "units", 5, start="2026-09-24", end="2026-09-30")
    out.append(Case(
        "bf2-cantdo-02", "bf_cantdo", 2,
        (
            Turn("Set this up to run every Monday morning and send me the top 5 SKUs by units.",
                 expect=cx.judge("Says it cannot schedule or send anything here; offers to give "
                                 "the list now.", "I can't schedule; here is this week's list.")),
            Turn("Fine: the top 5 SKUs by units for the last week of the data, 24 to 30 "
                 "September 2026.", expect=cx.idset([k for k, _ in wk])),
        ),
        ("no scheduler, then the list",),
    ))  # fmt: skip

    # ---------------------------------------------------------------- clarify
    bj = by_name("Black Slim Jeans Navy")
    out.append(Case(
        "bf2-clarify-01", "bf_clarify", 2,
        (
            Turn("What's the north stock of the black slim jeans?",
                 expect=cx.clarify(["Black Slim Jeans Navy", "Raw Slim Jeans Black",
                                    "Black Slim Jeans Black", "Stretch Slim Jeans Black"])),
            Turn("The Black Slim Jeans in Navy.",
                 expect=cx.count(w.on_hand(bj.sku, "north", 0))),
        ),
        ("a name that is both a material and a colour",),
    ))  # fmt: skip

    # ================================================================== round 3
    import statistics

    # ---------------------------------------------------------------- past chats
    wq2, wq3, eq3 = (units(region=g, **q) for g, q in (("west", Q2), ("west", Q3), ("east", Q3)))
    figs = {"west_q2": wq2, "west_q3": wq3, "east_q3": eq3}
    hi, lo = max(figs.values()), min(figs.values())
    out.append(Case(
        "bf3-past-01", "bf_past", 3,
        (
            Turn("Units sold in the west region in Q2 2026?", expect=cx.count(int(wq2))),
            Turn("Units sold in the west region in Q3 2026?", expect=cx.count(int(wq3)),
                 thread="b"),
            Turn("Units sold in the east region in Q3 2026?", expect=cx.count(int(eq3)),
                 thread="c"),
            Turn("Of the three unit figures you gave me in our last three chats, which was the "
                 "largest, and how much bigger was it than the smallest, in %?",
                 expect=cx.record({"largest": hi, "pct": (hi - lo) / lo * 100},
                                  tol={"largest": 0, "pct": 0.06}), thread="d"),
        ),
        ("the assistant's figures from three earlier chats",),
    ))  # fmt: skip
    out.append(Case(
        "bf3-past-02", "bf_past", 3,
        (
            Turn("I'm approving the SUP-12 contract renewal.", reply="Noted."),
            # forgetting what the user said asks a person (harness._forgotten): the run pauses,
            # is approved, and goes on; the approve-then-revoke history stays for the last turn
            Turn("Hold on: revoke my SUP-12 renewal approval until the audit is done.",
                 ((("memory_forget", {"id": SUP12_APPROVAL}),),),
                 reply="Noted: revoked pending the audit.", thread="b", approve=True),
            Turn("SUP-12's renewal is approved.", user=OTHER_USER, reply="Noted.", thread="c"),
            Turn("Have I approved the SUP-12 renewal right now?",
                 expect=cx.judge("Says no: the user approved it, then revoked the approval "
                                 "pending the audit; nothing another person said counts as the "
                                 "user's approval.", "No: you revoked it pending the audit."),
                 thread="d"),
        ),
        ("a decision revoked in a later chat; another user's claim",),
    ))  # fmt: skip
    kn_north = units(region="north", category="knitwear", **Q3)
    out.append(Case(
        "bf3-past-03", "bf_past", 2,
        (
            Turn("Unless I say otherwise, I only care about the north region.", reply="Noted."),
            Turn("What were knitwear units in Q3 2026?", expect=cx.count(int(kn_north)),
                 thread="b"),
        ),
        ("a default scope from an earlier chat",),
    ))  # fmt: skip
    south_sep = r.truth_top(w, "store", "revenue", 10, region="south", **SEP)
    under = [s for s, v in south_sep if v < 45_000]
    # an honest answer may name the store just above the line as just above it
    nearest_above = min(((v, s) for s, v in south_sep if v >= 45_000), default=(0, None))[1]
    out.append(Case(
        "bf3-past-04", "bf_past", 3,
        (
            Turn("Flag any store whose monthly revenue is below £45,000: that's my alert line.",
                 reply="Noted."),
            Turn("Which south stores were under my alert line in September 2026?",
                 expect=cx.idset(under, forbidden=[s for s, _ in south_sep
                                                   if s not in under and s != nearest_above])
                 if under else cx.abstain("none under the line"), thread="b"),
        ),
        ("a user-set threshold applied in a later chat",),
    ))  # fmt: skip

    # ---------------------------------------------------------------- analytics
    cats = sorted({k.category for k in w.skus.values()})
    margins = {c: r.truth_gross_profit(w, category=c, **Q3) / rev(category=c, **Q3) * 100
               for c in cats}  # fmt: skip
    top_m = sorted(margins, key=lambda c: -margins[c])[:3]
    out.append(Case(
        "bf3-analytics-01", "bf_analytics", 3,
        (
            Turn("Rank every category by Q3 2026 gross margin % (revenue minus units x unit "
                 "cost, over revenue), highest first. Give the top 3 with their margins.",
                 expect=cx.record({c: margins[c] for c in top_m},
                                  tol=dict.fromkeys(top_m, 0.06))),
        ),
        ("a join of the sales lines with the catalogue, per group, ranked",),
    ))  # fmt: skip
    nd = r.daily(w, region="north", **SEP)
    nu = [sum(s.units for s in r.selected(w, region="north", start=d, end=d))
          for d in (f"2026-09-{n:02d}" for n in range(1, 31))]  # fmt: skip
    out.append(Case(
        "bf3-analytics-02", "bf_analytics", 3,
        (
            Turn("North region units per day in September 2026 (count a day with no sale as 0): "
                 "the daily average and the population standard deviation?",
                 expect=cx.record({"mean": statistics.mean(nu), "sd": statistics.pstdev(nu)},
                                  tol={"mean": 0.01, "sd": 0.01})),
            Turn("And the day with the most revenue in the north that month, with its revenue?",
                 expect=cx.record({"day": (f"2026-09-{nd.index(max(nd)) + 1:02d}",
                                           f"{nd.index(max(nd)) + 1} September"),
                                   "revenue": max(nd)})),
        ),
        ("statistics over a daily series with empty days",),
    ))  # fmt: skip
    st = {s: (rev(store=s, **Q2), rev(store=s, **Q3)) for s in ("S05", "S06", "S07")}
    gr = {s: r.growth_pct(*v) for s, v in st.items()}
    out.append(Case(
        "bf3-analytics-03", "bf_analytics", 3,
        (
            Turn("Revenue for store S05 in Q2 and Q3 2026, and the % change.",
                 expect=cx.record({"q2": st["S05"][0], "q3": st["S05"][1], "pct": gr["S05"]},
                                  tol={"pct": 0.06})),
            Turn("Same for S06 and S07.",
                 expect=cx.record({"S06": gr["S06"], "S07": gr["S07"]},
                                  tol={"S06": 0.06, "S07": 0.06})),
            Turn("Which of the three grew most, and by how many points more than the least?",
                 expect=cx.record({"store": max(gr, key=lambda s: gr[s]),
                                   "points": max(gr.values()) - min(gr.values())},
                                  tol={"points": 0.06})),
        ),
        ("a method repeated over a list, then compared",),
    ))  # fmt: skip
    rq3 = {g: rev(region=g, **Q3) for g in r.REGIONS}
    shares = {g: v / sum(rq3.values()) * 100 for g, v in rq3.items()}
    out.append(Case(
        "bf3-analytics-04", "bf_analytics", 2,
        (
            Turn("Each region's share of Q3 2026 revenue in %, and the gap in share points "
                 "between the largest and the smallest.",
                 expect=cx.record({**{g: shares[g] for g in r.REGIONS},
                                   "gap": max(shares.values()) - min(shares.values())},
                                  tol={**dict.fromkeys(r.REGIONS, 0.06), "gap": 0.06})),
        ),
        ("shares and a gap",),
    ))  # fmt: skip

    # ---------------------------------------------------------------- explanation
    acc, dr = units(category="accessories", **Q3), units(category="dresses", **Q3)
    out.append(Case(
        "bf3-explain-01", "bf_explain", 2,
        (
            Turn("Q3 2026 units for accessories?", expect=cx.count(int(acc))),
            Turn("And dresses?", expect=cx.count(int(dr))),
            Turn("Their sum?", expect=cx.count(int(acc + dr))),
            Turn("How did you get that sum: which two inputs, and from which calls?",
                 expect=cx.record({"accessories": acc, "dresses": dr},
                                  tol={"accessories": 0, "dresses": 0},
                                  rubric="Names the two sales_summary calls (accessories, "
                                         "dresses, Q3) and adds them; nothing it did not call.")),
        ),
        ("explain a sum of two earlier answers",),
    ))  # fmt: skip
    e2 = first(lambda k: k.category == "shirts" and total_on(k.sku) != total_on(k.sku, 1), 2)
    out.append(Case(
        "bf3-explain-02", "bf_explain", 3,
        (
            Turn(f"Total on hand of the {e2.name} across all regions?",
                 expect=cx.count(total_on(e2.sku))),
            Turn("How sure are you that the stock figure you gave me in our other chat is still "
                 "right? Check, and tell me the difference.",
                 expect=cx.record({"then": total_on(e2.sku), "now": total_on(e2.sku, 1),
                                   "difference": abs(total_on(e2.sku, 1) - total_on(e2.sku))},
                                  tol={"then": 0, "now": 0, "difference": 0}),
                 thread="b", ticks=1),
        ),
        ("how sure about an earlier chat's figure, checked",),
    ))  # fmt: skip
    out.append(Case(
        "bf3-explain-03", "bf_explain", 2,
        (
            Turn(f"What's the 4-week demand forecast for {navy.sku}?",
                 expect=cx.abstain("the demand planning service is unavailable")),
            Turn("What exactly did you try, and what failed?",
                 expect=cx.judge("Names the demand_forecast call(s) it made and the 503 / "
                                 "maintenance window it got; offers an alternative (recent "
                                 "sales); nothing it did not try.",
                                 "I called demand_forecast; it returned 503 (maintenance).")),
        ),
        ("explain a failed call",),
    ))  # fmt: skip

    # ---------------------------------------------------------------- mixed
    m3 = first(moves_north, 5)
    a0, a1 = reorder(m3.sku, "north"), reorder(m3.sku, "north", version=1)
    out.append(Case(
        "bf3-mixed-01", "bf_mixed", 3,
        (
            Turn(f"Reorder qty of the {m3.name} for the north? {formula()} (north only).",
                 expect=cx.count(a0)),
            Turn("Approve it.", reply="Approved."),
            Turn(f"New stock is in. Compare my approved north order for the {m3.name} with "
                 "today's formula result; what is the difference worth at unit cost?",
                 expect=cx.record({"approved": a0, "today": a1,
                                   "cost": abs(a1 - a0) * m3.unit_cost},
                                  tol={"approved": 0, "today": 0, "cost": 0.01}),
                 thread="b", ticks=1),
            Turn("Put that in a PDF for finance.",
                 expect=cx.cannot(r"\b(?:here(?: is|'s) (?:the|your|a)|i(?:'ve| have) "
                                  r"(?:attached|created|made|generated))\b.{0,30}pdf",
                                  r"text|table|summary|copy|paste|instead",
                                  "I can't make a PDF here: no code execution, files or images. "
                                  "Here is the text instead."), thread="b"),
        ),
        ("approval, new data, cost, then can't-do",),
    ))  # fmt: skip
    core = r.truth_top(w, "sku", "revenue", 3, **Q3, category="knitwear")
    both = sorted(
        r.truth_top(w, "sku", "revenue", 3, category="knitwear", **Q3)
        + r.truth_top(w, "sku", "revenue", 3, category="denim", **Q3),
        key=lambda kv: (-kv[1], kv[0]),
    )[:3]
    out.append(Case(
        "bf3-mixed-02", "bf_mixed", 3,
        (
            Turn("'Core lines' means knitwear and denim.", reply="Noted."),
            Turn("Top 3 SKUs by Q3 2026 revenue across our core lines?",
                 expect=cx.idset([k for k, _ in both]), thread="b"),
            Turn("Drop denim: just knitwear.", expect=cx.idset([k for k, _ in core]),
                 thread="b"),
        ),
        ("a term from an earlier chat, then narrowed",),
    ))  # fmt: skip
    m14 = first(lambda k: k.category == "knitwear" and avg28(k.sku, "north") > 0.5
                and reorder(k.sku, "north", review=14) != reorder(k.sku, "north", review=10)
                and reorder(k.sku, "north", review=10) > 0, 6)  # fmt: skip
    q14, q10 = reorder(m14.sku, "north", review=14), reorder(m14.sku, "north", review=10)
    lead14 = w.suppliers[m14.supplier].lead_time_days
    # the colleague's result, unless the right working shows the same number (its cover
    # days, lead time + 14): then the colleague's cover days, lead time + 10
    wrong14 = str(q10) if q10 != lead14 + 14 else str(lead14 + 10)
    out.append(Case(
        "bf3-mixed-03", "bf_mixed", 3,
        (
            Turn("Use a 10-day review period for every reorder.", user=OTHER_USER,
                 reply="Noted."),
            Turn("For my reorders, use a 14-day review period.", reply="Noted.", thread="b"),
            Turn(f"Reorder qty of the {m14.name} for the north using my review period: average "
                 "daily units sold over 2026-09-03..2026-09-30 (28 days) x (supplier lead time + "
                 "review period) - units on hand, rounded up, never below 0 (north only).",
                 expect=cx.Expect("number", q14, forbidden=(wrong14,), reference=f"{q14}"),
                 thread="c"),
        ),
        ("this user's own parameter, not a colleague's",),
    ))  # fmt: skip
    srev = r.truth_supplier_revenue(w, **YTD)
    topsup, topv = srev[0]
    rank = 1 + sum(1 for s, v in srev[1:] if v > topv * 0.8)
    out.append(Case(
        "bf3-mixed-04", "bf_mixed", 3,
        (
            Turn("Top supplier by 2026 revenue so far?",
                 expect=cx.record({"supplier": (topsup, w.suppliers[topsup].name)})),
            Turn("Their on-time rate, in %?",
                 expect=cx.pct(w.suppliers[topsup].on_time_rate * 100)),
            Turn("How many stores in the east?", expect=cx.count(10)),
            Turn("Price of SKU-00100?", expect=cx.money(w.skus["SKU-00100"].price)),
            Turn("Back to that top supplier: if their 2026 revenue had been 20% lower, where "
                 "would they rank among all suppliers?", expect=cx.count(rank)),
        ),
        ("a scenario on a topic four turns back",),
    ))  # fmt: skip
    colour = next(c for c in r.COLORS
                  if rev(sku=by_name(f"Cashmere Cardigan {c}").sku, **Q3) > 500
                  and rev(sku=by_name(f"Cotton Cardigan {c}").sku, **Q3) > 500)  # fmt: skip
    cashmere, cotton = by_name(f"Cashmere Cardigan {colour}"), by_name(f"Cotton Cardigan {colour}")
    out.append(Case(
        "bf3-mixed-05", "bf_mixed", 3,
        (
            Turn(f"What was revenue for the {cashmere.name} in Q3 2026?",
                 expect=cx.money(rev(sku=cashmere.sku, **Q3))),
            Turn("And for the cardigan in that same colour but in cotton, same quarter?",
                 expect=cx.money(rev(sku=cotton.sku, **Q3)), thread="b"),
        ),
        ("an attribute carried over from an earlier chat",),
    ))  # fmt: skip
    f2 = rev(category="footwear", **Q2)
    out.append(Case(
        "bf3-mixed-06", "bf_mixed", 3,
        (
            Turn("Footwear revenue in Q2 2026?", expect=cx.money(f2)),
            Turn("Sorry, Q3.", expect=cx.money(fw_q3)),
            Turn("No wait, I did mean Q2. And what if Q2 prices had been 5% higher?",
                 expect=cx.record({"q2": f2, "what_if": f2 * 1.05})),
        ),
        ("a correction of a correction, then a what-if",),
    ))  # fmt: skip

    # ---------------------------------------------------------------- can't-do
    q3m = r.truth_monthly(w, "revenue", **Q3)
    out.append(Case(
        "bf3-cantdo-01", "bf_cantdo", 2,
        (
            Turn("Forecast Q4 2026 revenue with a proper time-series model (ARIMA).",
                 expect=cx.cannot(r"\b(?:arima|the model) (?:forecast|predicts?)\b.{0,30}£",
                                  r"average|trend|monthly|figures|instead|simple",
                                  "I can't fit an ARIMA model here: no code execution. I can "
                                  "give the monthly trend instead.")),
            Turn("OK, then the average monthly revenue of Q3 2026.",
                 expect=cx.money(sum(v for _, v in q3m) / 3)),
        ),
        ("a model it cannot run, then a simple figure",),
    ))  # fmt: skip
    out.append(Case(
        "bf3-cantdo-02", "bf_cantdo", 2,
        (
            Turn("What's the weather forecast for Leeds tomorrow, and will it hurt footfall?",
                 expect=cx.abstain("no weather or footfall data")),
            Turn("OK. How many of our stores are in Leeds?", expect=cx.count(1)),
        ),
        ("outside data it has not",),
    ))  # fmt: skip

    # ---------------------------------------------------------------- clarify
    wpo, wpb = by_name("Waxed Parka Olive"), by_name("Waxed Parka Black")
    out.append(Case(
        "bf3-clarify-01", "bf_clarify", 2,
        (
            Turn("Compare the two parkas.",
                 expect=cx.clarify(["Insulated Parka", "Waxed Parka", "Lightweight Parka",
                                    "Quilted Parka", "Classic Parka"])),
            Turn("The Waxed Parka Olive and the Waxed Parka Black: Q3 2026 units.",
                 expect=cx.record({"olive": units(sku=wpo.sku, **Q3),
                                   "black": units(sku=wpb.sku, **Q3)},
                                  tol={"olive": 0, "black": 0})),
        ),
        ("which two, then the comparison",),
    ))  # fmt: skip

    # ================================================================== held out
    h1 = first(lambda k: k.category == "shirts" and avg28(k.sku, "north") > 0.5, 2)
    out.append(Case(
        "bfh-past-01", "bf_past", 3,
        (
            Turn(f"I'm approving a north order of 75 units of the {h1.name}; remember it.",
                 reply="Noted."),
            Turn("Unrelated: what was total revenue in July 2026?",
                 expect=cx.money(rev(start="2026-07-01", end="2026-07-31")), thread="b"),
            Turn(f"What order did I approve for the {h1.name}, and in which region?",
                 expect=cx.record({"units": 75, "region": "north"}, tol={"units": 0}),
                 thread="c"),
        ),
        ("heldout",),
    ))  # fmt: skip
    hq = {g: units(region=g, **SEP) for g in r.REGIONS}
    out.append(Case(
        "bfh-analytics-01", "bf_analytics", 3,
        (
            Turn("Units sold per region in September 2026, and each region's share of the total "
                 "in %.",
                 expect=cx.record({f"{g}_share": hq[g] / sum(hq.values()) * 100
                                   for g in r.REGIONS},
                                  tol={f"{g}_share": 0.06 for g in r.REGIONS})),
        ),
        ("heldout",),
    ))  # fmt: skip
    hm = first(moves_north, 3)
    out.append(Case(
        "bfh-mixed-01", "bf_mixed", 3,
        (
            Turn(f"Reorder qty of the {hm.name} for the north? {formula()} (north only).",
                 expect=cx.count(reorder(hm.sku, "north"))),
            Turn(f"Last time you worked out a north reorder for the {hm.name}. Redo it with "
                 "today's stock, give both numbers, and say what changed.",
                 expect=cx.record({"then": reorder(hm.sku, "north"),
                                   "now": reorder(hm.sku, "north", version=1)},
                                  tol={"then": 0, "now": 0}),
                 thread="b", ticks=1),
        ),
        ("heldout",),
    ))  # fmt: skip
    hs = r.truth_top(w, "category", "revenue", 2, **AUG)
    out.append(Case(
        "bfh-mixed-02", "bf_mixed", 3,
        (
            Turn("Top 2 categories by revenue in August 2026?",
                 expect=cx.idset([c for c, _ in hs])),
            Turn("If the first one's revenue had been 12% lower, would it still be on top? By how "
                 "much would it lead or trail the second?",
                 expect=cx.record({"gap": abs(hs[0][1] * 0.88 - hs[1][1])})),
            Turn("Make that a chart I can download.",
                 expect=cx.cannot(FAKE_CHART,
                                  # the data given to chart elsewhere ("here is the data") is an offer too
                                  r"figures|numbers|table|instead|here(?: is|'s) the|give you the data",
                                  "I can't make a chart here: there is no code execution, files or images in this setup. "
                                  "Here are the figures instead.")),
        ),
        ("heldout",),
    ))  # fmt: skip
    out.append(Case(
        "bfh-clarify-01", "bf_clarify", 2,
        (
            Turn("How many of the hoodie do we have?",
                 expect=cx.clarify(["Merino Hoodie", "Cashmere Hoodie", "Cotton Hoodie",
                                    "Wool-Blend Hoodie", "Chunky Hoodie"])),
        ),
        ("heldout",),
    ))  # fmt: skip
    return out


def names(case: Any) -> set[str]:
    """What ``--cases`` picks a break-and-fix case by besides its id and category."""
    if not case.id.startswith(("bf1-", "bf2-", "bf3-", "bf4-", "bf5-", "bfh-")):
        return set()
    tags = {"breakfix"}
    if case.id.startswith("bfh-"):
        tags.add("heldout")
    else:
        tags.add(f"round:{case.id[2]}")
    return tags
