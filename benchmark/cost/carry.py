"""The ``complex`` suite's conversations whose later turns need a file or a todo list an earlier
turn of the same conversation made (category ``carry``, ``--cases carry``): Deep Agents' own
file tools and todo list, carried from turn to turn.

A file written in turn 1 is read back, edited or overwritten in turn 2 or 3; a todo list planned
in turn 1 is worked through in the turns after it; one file outlives a turn that never touches
it. Each later turn's answer is exact (the figures the earlier turn wrote, computed from the
world), and each turn that uses an earlier file carries a probe that no file tool said the file
was not found (``Probe("lacks", ...)``): an answer the model gave from the conversation's
messages while the file itself was lost is right, and still a failed check.

The cases are for an agent with Deep Agents' tools from its first call: ``create_deep_agent``
(``--adapter deepagents``: built with no checkpointer, so the harness carries its files and
todos; with ``--variant threaded``, an ``InMemorySaver`` keeps them on its thread) or ``ReAct``
with a backend (``--adapter react --variant deep``). ``ReAct`` as built (``main``) offers the
file tools only for a long task.
"""

from __future__ import annotations

from typing import Any, Final

from benchmark.cost import retail as r

#: What a file tool says of a file the conversation lost (Deep Agents' ``StateBackend``:
#: "File '/x' not found").
LOST: Final = r"not found"
#: The adapters the cases apply to: Deep Agents' tools from the first call.
ADAPTERS: Final = (
    "deepagents",
    # ``ReAct`` with a backend: Deep Agents' tools from the first call (``main`` offers the
    # file tools only on a long task, and said honestly it had none: react-plain carry-01/02/05)
    *(f"react:{v}" for v in ("deep", "deep-paging", "deep-context_editing")),
    *(f"react:{v}" for v in ("deep-summarization", "sandbox", "memory-files", "permissions")),
)
WHY: Final = "the files and todos are Deep Agents' own (ReAct offers them with a backend)"


def cases() -> list[Any]:
    from benchmark.cost import complex as cx

    w = r.world()
    out: list[Any] = []

    def total(sku: str) -> int:
        return sum(w.on_hand(sku, g) for g in r.REGIONS)

    def kept(why: str) -> Any:
        return cx.Probe("lacks", LOST, why, role="tool")

    trail, wrap, trench = "SKU-00938", "SKU-01517", "SKU-00412"
    three = (trail, wrap, trench)
    lines = "\n".join(f"{s}: {total(s)}" for s in three)

    # 1. a file written in turn 1, read back in turn 2
    out.append(
        cx.Case(
            "carry-01", "carry", 2,
            (
                cx.Turn(
                    f"Look up how many units of {trail}, {wrap} and {trench} are on hand in "
                    "total right now and write them to the file /notes/stock.md, one line per "
                    "SKU (SKU: units). Just tell me the file is saved; don't repeat the figures.",
                    (tuple(("stock_level", {"sku": s}) for s in three),
                     (("write_file", {"file_path": "/notes/stock.md", "content": lines}),)),
                    reply="Saved /notes/stock.md.",
                ),
                cx.Turn(
                    "Read /notes/stock.md back: which SKU in it has the most units on hand, and "
                    "what is the total of all the SKUs in it?",
                    ((("read_file", {"file_path": "/notes/stock.md"}),),),
                    cx.record({"sku": wrap, "total": sum(map(total, three))}, tol={"total": 0}),
                    sent=(kept("the file turn 1 wrote is there"),),
                ),
            ),
            ("a file written in one turn, read in the next",),
            adapters=ADAPTERS, only_because=WHY,
        )
    )  # fmt: skip

    # 2. a file written, edited in turn 2, read in turn 3
    sups = ("SUP-02", "SUP-05")
    lead = {s: w.suppliers[s].lead_time_days for s in (*sups, "SUP-09")}
    first = "\n".join(f"{s}: {lead[s]} days" for s in sups)
    longest = max(lead, key=lambda s: lead[s])
    out.append(
        cx.Case(
            "carry-02", "carry", 2,
            (
                cx.Turn(
                    "Write the supplier-master lead times of SUP-02 and SUP-05 to the file "
                    "/notes/suppliers.md, one line per supplier (SUP-xx: N days). Just confirm "
                    "it's saved.",
                    (tuple(("supplier_info", {"supplier": s}) for s in sups),
                     (("write_file", {"file_path": "/notes/suppliers.md", "content": first}),)),
                    reply="Saved /notes/suppliers.md.",
                ),
                cx.Turn(
                    "Add SUP-09's lead time to /notes/suppliers.md: edit the file and keep the "
                    "lines already in it. Just confirm it's done.",
                    ((("supplier_info", {"supplier": "SUP-09"}),),
                     (("edit_file", {"file_path": "/notes/suppliers.md",
                                     "old_string": f"SUP-05: {lead['SUP-05']} days",
                                     "new_string": f"SUP-05: {lead['SUP-05']} days\n"
                                                   f"SUP-09: {lead['SUP-09']} days"}),)),
                    reply="Added SUP-09 to /notes/suppliers.md.",
                    sent=(kept("the file turn 1 wrote is there to edit"),),
                ),
                cx.Turn(
                    "Going by /notes/suppliers.md: which supplier in it has the longest lead "
                    "time, and what is the average lead time of the suppliers in the file "
                    "(1 decimal)?",
                    ((("read_file", {"file_path": "/notes/suppliers.md"}),),),
                    cx.record({"supplier": longest, "average": round(sum(lead.values()) / 3, 1)},
                              tol={"average": 0.06}),
                    sent=(kept("the file, as turn 2 left it, is there"),),
                ),
            ),
            ("a file edited across turns",),
            adapters=ADAPTERS, only_because=WHY,
        )
    )  # fmt: skip

    # 3. a todo list planned in turn 1, worked in turns 2 and 3
    plan = [{"content": f"stock check of {s}", "status": "pending"} for s in three]
    one = [dict(plan[0], status="completed"), *plan[1:]]
    out.append(
        cx.Case(
            "carry-03", "carry", 2,
            (
                cx.Turn(
                    f"Plan a stock check as a todo list, one todo per SKU: {trail}, {wrap}, "
                    f"{trench}, in that order. Don't do any of them yet.",
                    ((("write_todos", {"todos": plan}),),),
                    reply="Planned: three stock checks, none done yet.",
                ),
                cx.Turn(
                    "Do the first todo still pending on our list, mark it completed, and tell "
                    "me that SKU's total units on hand.",
                    ((("stock_level", {"sku": trail}),), (("write_todos", {"todos": one}),)),
                    cx.count(total(trail)),
                ),
                cx.Turn(
                    "Which SKUs on our todo list are still pending?",
                    (),
                    cx.idset([wrap, trench]),
                ),
            ),
            ("a todo list worked across turns",),
            adapters=ADAPTERS, only_because=WHY,
        )
    )  # fmt: skip

    # 4. a todo list and a file, together across three turns
    s3 = w.suppliers["SUP-03"]
    review = [
        {"content": "SUP-03 master lead time", "status": "in_progress"},
        {"content": "SUP-03 portal quoted lead time", "status": "pending"},
        {"content": "the difference", "status": "pending"},
    ]
    after = [dict(review[0], status="completed"), dict(review[1], status="completed"), review[2]]
    master = f"master: {s3.lead_time_days} days"
    out.append(
        cx.Case(
            "carry-04", "carry", 3,
            (
                cx.Turn(
                    "Supplier review of SUP-03. Make a todo list: (1) its supplier-master lead "
                    "time, (2) its portal's quoted lead time, (3) the difference. Do only the "
                    "first item now and save its figure in /review/sup03.md.",
                    ((("write_todos", {"todos": review}),),
                     (("supplier_info", {"supplier": "SUP-03"}),),
                     (("write_file", {"file_path": "/review/sup03.md", "content": master}),)),
                    reply="Item 1 done and saved in /review/sup03.md.",
                ),
                cx.Turn(
                    "Carry on with the next item on the list and add its figure to "
                    "/review/sup03.md.",
                    ((("supplier_portal", {"supplier": "SUP-03"}),),
                     (("edit_file", {"file_path": "/review/sup03.md", "old_string": master,
                                     "new_string": f"{master}\n"
                                                   f"portal: {s3.portal_lead_time_days} days"}),),
                     (("write_todos", {"todos": after}),)),
                    reply="Item 2 done and added to /review/sup03.md.",
                    sent=(kept("the file turn 1 wrote is there to add to"),),
                ),
                cx.Turn(
                    "Finish the list from /review/sup03.md: the master lead time, the portal's "
                    "and how many days longer the portal's is.",
                    ((("read_file", {"file_path": "/review/sup03.md"}),),),
                    cx.record({"master": s3.lead_time_days, "portal": s3.portal_lead_time_days,
                               "difference": s3.portal_lead_time_days - s3.lead_time_days},
                              tol={"master": 0, "portal": 0, "difference": 0}),
                    sent=(kept("the file, as turn 2 left it, is there"),),
                ),
            ),
            ("a todo list and a file across turns",),
            adapters=ADAPTERS, only_because=WHY,
        )
    )  # fmt: skip

    # 5. a file outlives a turn that never touches it
    west = {s: w.on_hand(s, "west") for s in (trail, wrap)}
    west_lines = "\n".join(f"{s}: {n}" for s, n in west.items())
    north = sum(1 for s in w.stores if s.region == "north")
    out.append(
        cx.Case(
            "carry-05", "carry", 2,
            (
                cx.Turn(
                    f"Write the west region's units on hand of {trail} and {wrap} to "
                    "/notes/west.md, one line each. Just confirm it's saved.",
                    (tuple(("stock_level", {"sku": s, "region": "west"}) for s in west),
                     (("write_file", {"file_path": "/notes/west.md", "content": west_lines}),)),
                    reply="Saved /notes/west.md.",
                ),
                cx.Turn(
                    "Something else: how many stores are in the north region?",
                    ((("store_list", {"region": "north"}),),),
                    cx.count(north),
                ),
                cx.Turn(
                    "Back to /notes/west.md: read it and tell me the sum of the figures in it.",
                    ((("read_file", {"file_path": "/notes/west.md"}),),),
                    cx.count(sum(west.values())),
                    sent=(kept("the file outlived the turn that did not touch it"),),
                ),
            ),
            ("a file kept through a turn that does not use it",),
            adapters=ADAPTERS, only_because=WHY,
        )
    )  # fmt: skip

    # 6. a file overwritten in turn 2: turn 3 reads the new text, not the old
    out.append(
        cx.Case(
            "carry-06", "carry", 2,
            (
                cx.Turn(
                    f"Write {trail}'s total units on hand to /notes/watch.md as one line "
                    "(SKU: units). Just confirm it's saved.",
                    ((("stock_level", {"sku": trail}),),
                     (("write_file", {"file_path": "/notes/watch.md",
                                      "content": f"{trail}: {total(trail)}"}),)),
                    reply="Saved /notes/watch.md.",
                ),
                cx.Turn(
                    f"I'm watching {trench} instead: replace what /notes/watch.md says with "
                    f"{trench}'s total units on hand, the same way. Just confirm.",
                    ((("stock_level", {"sku": trench}),),
                     (("write_file", {"file_path": "/notes/watch.md",
                                      "content": f"{trench}: {total(trench)}"}),)),
                    reply="Replaced: /notes/watch.md now holds the new line.",
                ),
                cx.Turn(
                    "Read /notes/watch.md: which SKU does it hold now, and what figure?",
                    ((("read_file", {"file_path": "/notes/watch.md"}),),),
                    cx.record({"sku": trench, "units": total(trench)}, tol={"units": 0}),
                    sent=(kept("the file, as turn 2 left it, is there"),),
                ),
            ),
            ("a file overwritten across turns",),
            adapters=ADAPTERS, only_because=WHY,
        )
    )  # fmt: skip
    return out


def names(case: Any) -> set[str]:
    """What ``--cases`` picks a ``carry`` case by besides its id and category."""
    return {"carry"} if case.id.startswith("carry-") else set()
