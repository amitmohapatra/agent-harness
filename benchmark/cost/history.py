"""The ``history`` set (``--cases history``): what a user asks an assistant that remembers them,
as they would ask Claude — what it told them before, a new figure against one it gave, a trend
over figures fetched across chats, two chats' results merged, why a number changed since last
time, an earlier run's long tool output recalled; and how it got an answer (which tools and
with what, which part came from memory and which from a tool, where a number is from and when
it was read), after a resume and in a new chat.

It is the cases below (categories ``hist_past``, ``hist_analytics``, ``hist_explain``) and the
suite's existing cases of the same kinds (:data:`REUSED`), which it picks rather than repeats.
A cross-chat turn (``thread`` other than ``a``) can be answered only with the memory service on
(``--history memory``). The chats are run one after another, so a question names the earlier
chat ("our earlier chat"), not a date: the memory service would see them minutes apart.
Every figure's truth is computed from the world (:mod:`benchmark.cost.retail`).
"""

from __future__ import annotations

from typing import Any, Final

from benchmark.cost import retail as r

CATEGORIES: Final = ("hist_past", "hist_analytics", "hist_explain")
TAG: Final = "history"
#: The suite's cases this set picks as they are: each asks one of the questions above.
REUSED: Final = (
    "explain-01",  # where did that number come from
    "explain-03",  # which steps, how many calls
    "explain-04",  # why that number
    "explain-06",  # what changed since you last looked
    "explain-08",  # a new chat: how did you answer it there
    "bf1-explain-01",  # which calls, in order, with which filters
    "bf2-explain-01",  # an earlier chat's figure, from its inputs
    "bf3-explain-01",  # which inputs, from which calls
    "bf1-mixed-02",  # an earlier chat's result redone with today's data
    "bf2-mixed-03",  # an earlier chat's method on another period, and the % change
    "bf2-past-03",  # two figures from earlier chats combined
    "bfh-mixed-01",  # last time's reorder figure
)
#: The memory of histx-03's approval as the scripted model names it.
SUP07_APPROVAL: Final = "mem_sup07_approval"

JUN_AUG: Final = {"start": "2026-06-01", "end": "2026-08-31"}
JUN_SEP: Final = {"start": "2026-06-01", "end": "2026-09-30"}
AUG: Final = {"start": "2026-08-01", "end": "2026-08-31"}
SEP: Final = {"start": "2026-09-01", "end": "2026-09-30"}
Q3: Final = {"start": "2026-07-01", "end": "2026-09-30"}
#: Says where a figure came from: memory, an earlier chat or the user's own note.
FROM_MEMORY: Final = (
    r"memory|earlier (?:chat|conversation)|(?:previous|last|other) (?:chat|conversation)|"
    r"you (?:told|said|noted|set|gave)|your (?:note|budget)"
)


def names(case: Any) -> set[str]:
    """What ``--cases`` picks a case of this set by besides its id and category."""
    return {TAG} if case.id.startswith(("hist-", "histx-")) or case.id in REUSED else set()


def cases() -> list[Any]:
    from benchmark.cost import breakfix
    from benchmark.cost import complex as cx

    w = r.world()
    Case, Turn = cx.Case, cx.Turn
    out: list[Any] = []

    def month_args(**f: Any) -> dict[str, Any]:
        return {"group_by": "month", "metric": "revenue", **f}

    # ---------------------------------------------------------------- analytics over history
    # a trend over three months fetched and this month's, by least squares and correlation
    ou = [v for _, v in r.truth_monthly(w, category="outerwear", **JUN_SEP)]
    slope, _ = r.regression(range(1, 5), ou)
    out.append(Case(
        "hist-trend-01", "hist_analytics", 3,
        (
            Turn("Fetch outerwear's revenue for the three months before September 2026 (June, "
                 "July, August) and for September 2026. Then fit the least-squares trend of "
                 "revenue on month number (June = 1 ... September = 4): the slope in £ per "
                 "month, and the Pearson correlation of month number and revenue.",
                 ((("sales_summary", month_args(category="outerwear", **JUN_SEP)),),),
                 cx.record({"slope": slope, "r": r.truth_correlation(range(1, 5), ou)},
                           tol={"slope": max(abs(slope) * 0.001, 0.5), "r": 0.006})),
        ),
        ("a trend over 4 fetched months", "slope and correlation computed, not guessed"),
    ))  # fmt: skip
    # the same, its first three months given in an earlier chat
    kn = dict(r.truth_monthly(w, category="knitwear", **JUN_SEP))
    kv = list(kn.values())
    kslope, _ = r.regression(range(1, 5), kv)
    out.append(Case(
        "hist-trend-02", "hist_analytics", 3,
        (
            Turn("What was knitwear revenue in June, July and August 2026, month by month?",
                 ((("sales_summary", month_args(category="knitwear", **JUN_AUG)),),),
                 cx.record({"june": kv[0], "july": kv[1], "august": kv[2]})),
            Turn("Take the three monthly knitwear figures you gave me in our earlier chat, add "
                 "September 2026's, and give me the trend: the least-squares slope in £ per "
                 "month (June = 1 ... September = 4) and the correlation with month number.",
                 ((("sales_summary", month_args(category="knitwear", **SEP)),),),
                 cx.record({"september": kv[3], "slope": kslope,
                            "r": r.truth_correlation(range(1, 5), kv)},
                           tol={"slope": max(abs(kslope) * 0.001, 0.5), "r": 0.006}),
                 thread="b"),
        ),
        ("an earlier chat's figures and a new one in one trend",),
    ))  # fmt: skip
    # two chats' lists merged, and each shared store's change
    aug = dict(r.truth_top(w, "store", "revenue", 5, **AUG))
    sep = dict(r.truth_top(w, "store", "revenue", 5, **SEP))
    both = sorted(set(aug) & set(sep))
    assert both, "the August and September top-5 stores share none"
    out.append(Case(
        "hist-merge-01", "hist_analytics", 3,
        (
            Turn("Top 5 stores by revenue in August 2026, with their revenue?",
                 ((("sales_summary", {"group_by": "store", "metric": "revenue", "top": 5,
                                      **AUG}),),),
                 cx.idset(list(aug), reference="; ".join(f"{s} £{cx.fmt(v)}"
                                                         for s, v in aug.items()))),
            Turn("Merge the August top-5 store list you gave me in our other chat with the same "
                 "list for September 2026: which stores are on both, and how much did each of "
                 "those change in revenue from August to September?",
                 ((("sales_summary", {"group_by": "store", "metric": "revenue", "top": 5,
                                      **SEP}),),),
                 cx.record({s: sep[s] - aug[s] for s in both}),
                 thread="b"),
        ),
        ("an earlier chat's list merged with today's", "the change per shared item"),
    ))  # fmt: skip

    # ---------------------------------------------------------------- recall
    # an earlier run's long tool output, recalled: not fetched again
    kq3 = r.truth_top(w, "sku", "units", 60, category="knitwear", **Q3)
    rank = next(n for n in range(6, len(kq3) - 1)
                if kq3[n - 1][1] != kq3[n][1] != kq3[n + 1][1])  # fmt: skip
    nth, nth_units = kq3[rank]
    out.append(Case(
        "hist-long-01", "hist_past", 3,
        (
            Turn("Which knitwear SKU sold the most units in Q3 2026? Look at the whole list.",
                 ((("sales_summary", {"group_by": "sku", "metric": "units",
                                      "category": "knitwear", **Q3}),),),
                 cx.record({"sku": (kq3[0][0], w.skus[kq3[0][0]].name), "units": kq3[0][1]},
                           tol={"units": 0})),
            Turn(f"In our earlier chat you looked up every knitwear SKU's Q3 2026 units. On that "
                 f"same list, which SKU was number {rank + 1}, and how many units did it sell? "
                 "Use that list; don't look it up again.",
                 expect=cx.record({"sku": (nth, w.skus[nth].name), "units": nth_units},
                                  tol={"units": 0}),
                 thread="b",
                 sent=(cx.Probe("uncalled", "sales_summary",
                                "the earlier run's tool output, recalled", when="memory"),)),
        ),
        ("a long tool output of an earlier run, recalled",),
    ))  # fmt: skip
    # nothing was said about it: no memory made up
    out.append(Case(
        "hist-none-01", "hist_past", 2,
        (
            Turn("How many stores are in the west?", ((("store_list", {"region": "west"}),),),
                 cx.count(sum(1 for s in w.stores if s.region == "west"))),
            Turn("What did you tell me about denim return rates in our earlier chat?",
                 expect=cx.abstain("we have not discussed denim return rates, and returns are "
                                   "not in the data"),
                 thread="b"),
        ),
        ("asked what it never said",),
    ))  # fmt: skip
    # its own figure, not another person's
    fq3 = r.truth_total(w, "revenue", category="footwear", **Q3)
    out.append(Case(
        "hist-own-01", "hist_past", 3,
        (
            Turn("What was footwear revenue in Q3 2026?",
                 ((("sales_summary", {"group_by": "category", "metric": "revenue",
                                      "category": "footwear", **Q3}),),),
                 cx.money(fq3)),
            Turn("For the record, Q3 2026 footwear revenue was £1,234,567.", reply="Noted.",
                 user=breakfix.OTHER_USER, thread="c"),
            Turn("What Q3 2026 footwear revenue figure did you give me before?",
                 expect=cx.money(fq3, forbidden=("1,234,567",)), thread="b"),
        ),
        ("its own earlier figure, not another user's",),
    ))  # fmt: skip

    # ---------------------------------------------------------------- explain
    # why a figure changed since the earlier chat: the stock moved
    sku = next(k for k in sorted(w.skus.values(), key=lambda k: k.sku)
               if w.on_hand(k.sku, "north", 0) != w.on_hand(k.sku, "north", 1) > 0)  # fmt: skip
    now, then = w.on_hand(sku.sku, "north", 1), w.on_hand(sku.sku, "north", 0)
    out.append(Case(
        "hist-why-01", "hist_explain", 3,
        (
            Turn(f"How many {sku.name} ({sku.sku}) do we have in the north right now?",
                 ((("stock_level", {"sku": sku.sku, "region": "north"}),),),
                 cx.count(then)),
            Turn(f"Check the north stock of the {sku.name} again. What is it now, how far is it "
                 "from what you told me in our earlier chat, and why did it change?",
                 ((("stock_level", {"sku": sku.sku, "region": "north"}),),),
                 cx.record({"now": now, cx.DIFFERENCE: now - then, "then": then},
                           tol={"now": 0, cx.DIFFERENCE: 0, "then": 0},
                           rubric="Says the count was read again at a later time "
                                  f"({w.as_of(1)} vs {w.as_of(0)}), so stock moved; no "
                                  "invented cause."),
                 thread="b", ticks=1),
        ),
        ("a figure from an earlier chat against today's, and why",),
    ))  # fmt: skip
    # which part came from memory, which from a tool
    knit = next(k for k in sorted(w.skus.values(), key=lambda k: k.sku)
                if k.category == "knitwear")  # fmt: skip
    left = 50_000 - 120 * knit.unit_cost
    out.append(Case(
        "histx-01", "hist_explain", 3,
        (
            Turn("Note for later: my knitwear reorder budget is £50,000.", reply="Noted."),
            Turn(f"If I buy 120 more units of the {knit.name} ({knit.sku}) at its unit cost, how "
                 "much of my knitwear budget is left?",
                 ((("sku_details", {"sku": knit.sku}),),), cx.money(left), thread="b"),
            Turn("Which part of that came from memory and which from a tool call?",
                 expect=cx.grounded(
                     FROM_MEMORY,
                     "The £50,000 budget from memory (the user's note in an earlier chat); the "
                     "unit cost from the sku_details call; the 120 units from the question. No "
                     "step it did not take.",
                     f"The £50,000 budget came from memory, your note in our earlier chat; the "
                     f"unit cost £{cx.fmt(knit.unit_cost)} from sku_details; 120 x "
                     f"£{cx.fmt(knit.unit_cost)} leaves £{cx.fmt(left)}."),
                 thread="b"),
        ),
        ("which part from memory, which from a tool",),
    ))  # fmt: skip
    # where a figure came from, and when it was read
    east = next(k for k in sorted(w.skus.values(), key=lambda k: k.sku)
                if w.on_hand(k.sku, "east", 0) > 100)  # fmt: skip
    out.append(Case(
        "histx-02", "hist_explain", 2,
        (
            Turn(f"How many {east.name} ({east.sku}) are on hand in the east right now?",
                 ((("stock_level", {"sku": east.sku, "region": "east"}),),),
                 cx.count(w.on_hand(east.sku, "east", 0))),
            Turn("Where is that number from, and when was it fetched?",
                 expect=cx.grounded(
                     # the source and the read time, in either order
                     r"(?=.*(?:stock[_ -]level|stock (?:count|tool|system)))"
                     r"(?=.*(?:2026-10-30|30 Oct|October 30|09:00))",
                     f"Names the stock_level call for {east.sku} in the east and the count's "
                     f"read time {w.as_of(0)} as the tool gave it; no invented source or time.",
                     f"From a stock_level call for {east.sku} in the east: "
                     f"{w.on_hand(east.sku, 'east', 0)} on hand, read at {w.as_of(0)}.",
                     cites=1)),
        ),
        ("where a figure is from, and when it was read",),
    ))  # fmt: skip
    # after a resume: what the resumed run did
    out.append(Case(
        "histx-03", "hist_explain", 3,
        (
            Turn("I'm approving the SUP-07 contract renewal.", reply="Noted."),
            Turn("Revoke my SUP-07 renewal approval: the audit isn't done.",
                 ((("memory_forget", {"id": SUP07_APPROVAL}),),),
                 reply="Done: I've removed your SUP-07 renewal approval.", thread="b",
                 approve=True),
            Turn("What exactly did you do for that just now, with which tool, and did it go "
                 "through?",
                 expect=cx.Expect("tools", of_turn=1,
                                  rubric="Names the memory_forget call of the resumed run, says "
                                         "it was approved and went through; no other step.",
                                  reference="I called memory_forget on your SUP-07 approval; "
                                            "you approved it and it went through."),
                 thread="b"),
        ),
        ("the steps of a resumed run",),
    ))  # fmt: skip
    return out
