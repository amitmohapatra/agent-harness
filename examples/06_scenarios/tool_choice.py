"""Scenario: tool choice, measured. A store-operations tool set where a model goes wrong in four
typical ways, and an evaluation that scores each item's tool coverage against the tools the item
needs.

The four ways (whatever the tools are):

1. **a broader tool instead of the specific one** — ``store_overview`` answers "excess stock and
   lost sales" vaguely, ``excess_stock`` and ``lost_sales`` answer it exactly;
2. **a part of a multi-part request skipped** — "where in the app ... and which styles ..."
   needs ``app_help`` *and* ``allocatable_stock``;
3. **a change answered in text** — "create a stock rule ..." must call ``create_stock_rule``
   (irreversible: governance asks a person), never just say it was done;
4. **a wrong conclusion** — "which store gets stock first" must follow from ``store_demand``.

What guides the choice is the tools' own descriptions (``store_overview`` says what it is not
for) and the instructions. ``tool_coverage()`` scores each item: the share of its
``needed_tools`` the run called (a call that paused for approval counts), naming the ones
missing and any ``avoid_tools`` called.

Offline (``make examples``) the model is scripted: it makes the first two mistakes, and
``tool_coverage`` names them — the harness's part is real, the model's choices are written
down. ``tests/live/test_live_tool_choice.py`` runs the same set against a real model and
reports the coverage of each item.

    python -m examples.06_scenarios.tool_choice
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Mapping
from typing import Annotated, Any, Final

from examples._support.offline import react_model
from pydantic import Field

from trellis import Harness, ReAct, tool
from trellis.harness.evals import EvalCase, EvalReport, EvalScore, contains, tool_coverage

STORES: Final = ("S12", "S40")

SYSTEM: Final = "You help store managers with stock, sales and allocations. Be brief."


# --------------------------------------------------------------------------- the tools
@tool(side_effects="read")
def store_overview(store: str) -> dict[str, Any]:
    """A store's dashboard at a glance: the week's sales, stock on hand, sell-through and
    service level, as store totals.

    Use for a general picture of how a store is doing. It has no figures by style: for excess
    stock, lost sales or allocations by style, use the tools made for those.

    Args:
        store: The store's id, such as S12.
    """
    return {
        "store": store,
        "week_sales": 48210.0,
        "stock_units": 9120,
        "sell_through": 0.41,
        "service_level": 0.93,
    }


@tool(side_effects="read")
def excess_stock(store: str) -> list[dict[str, Any]]:
    """The styles holding more stock than they will sell at a store (overstock), with the
    units over target, largest first.

    Args:
        store: The store's id, such as S12.
    """
    return [
        {"style": "ST-205", "units_over_target": 140},
        {"style": "ST-330", "units_over_target": 65},
    ]


@tool(side_effects="read")
def lost_sales(store: str) -> list[dict[str, Any]]:
    """The sales a store lost to stock-outs, by style, largest first.

    Args:
        store: The store's id, such as S12.
    """
    return [
        {"style": "ST-101", "lost_units": 38, "lost_revenue": 1520.0},
        {"style": "ST-330", "lost_units": 4, "lost_revenue": 180.0},
    ]


@tool(side_effects="read")
def allocatable_stock() -> list[dict[str, Any]]:
    """The units available to allocate (ATA) in the warehouse, by style, largest first."""
    return [
        {"style": "ST-101", "available_to_allocate": 420},
        {"style": "ST-205", "available_to_allocate": 75},
    ]


@tool(side_effects="read")
def allocation_history(store: str) -> list[dict[str, Any]]:
    """The allocations made to a store in the past, newest first.

    Args:
        store: The store's id, such as S12.
    """
    return [{"id": "AL-77", "style": "ST-101", "units": 24, "date": "2026-10-12"}]


@tool(side_effects="read")
def app_help(task: str) -> str:
    """Where in the app to do something: the menu path and the steps.

    Use when the user asks how or where to do something in the app, not for data.

    Args:
        task: What the user wants to do, such as "create an allocation".
    """
    return f"To {task}: open Allocations > New, pick the styles and stores, then Review > Submit."


@tool(side_effects="read")
def store_demand(store: str, style: str) -> dict[str, Any]:
    """A style's weekly demand and stock on hand at a store, and the weeks of cover they give
    (fewer weeks: the store runs out sooner).

    Args:
        store: The store's id, such as S12.
        style: The style's id, such as ST-101.
    """
    on_hand, weekly = {"S12": (30, 10), "S40": (5, 25)}.get(store, (0, 0))
    cover = round(on_hand / weekly, 2) if weekly else None
    return {
        "store": store,
        "style": style,
        "on_hand": on_hand,
        "weekly_demand": weekly,
        "weeks_of_cover": cover,
    }


@tool(side_effects="irreversible")
def create_stock_rule(store: str, style: str, min_units: int, max_units: int) -> str:
    """Create a stock rule for a style at a store: the minimum and maximum units it keeps.

    A change: a person approves it before it is made.

    Args:
        store: The store's id, such as S12.
        style: The style's id, such as ST-101.
        min_units: The fewest units to keep.
        max_units: The most units to keep.
    """
    return f"rule R-1 created: {style} at {store}, min {min_units}, max {max_units}"


@tool(side_effects="read")
def supplier_eta(style: str) -> str:
    """When the supplier's next delivery of a style arrives at the warehouse.

    Args:
        style: The style's id, such as ST-101.
    """
    raise RuntimeError("the supplier system is unavailable")


@tool(side_effects="read")
def style_stock(style: str) -> Any:
    """The units of one style held across every store and the warehouse.

    Args:
        style: The style's id, such as ST-101.
    """
    held = {"ST-101": 1280, "ST-205": 410, "ST-330": 95}
    return {"style": style, "units": held[style]} if style in held else f"style {style} not found"


@tool(side_effects="read", call_timeout=2)
async def transfer_status(transfer: str) -> str:
    """Where a stock transfer between stores is: packed, in transit or received.

    Args:
        transfer: The transfer's id, such as TR-9.
    """
    await asyncio.sleep(30)  # the transfer system hangs: the call times out
    return "received"


TOOLS: Final = [
    store_overview,
    excess_stock,
    lost_sales,
    allocatable_stock,
    allocation_history,
    app_help,
    store_demand,
    create_stock_rule,
    supplier_eta,
    transfer_status,
    style_stock,
]

AREAS: Final = (
    "payroll",
    "supplier invoices",
    "marketing campaigns",
    "staff rosters",
    "maintenance tickets",
    "customer complaints",
    "price changes",
    "the promotions calendar",
    "energy usage",
    "shrinkage incidents",
)
KINDS: Final = {
    "summary": "This month's totals of {area} at one store.",
    "trend": "How {area} changed week by week at one store.",
    "export": "Every record of {area} at one store, as a CSV file.",
    "forecast": "What {area} will be next month at one store, projected.",
}


def _unrelated(area: str, kind: str) -> Any:
    def report(store: Annotated[str, Field(description="The store's id, such as S12.")]) -> str:
        return f"{kind} of {area} at {store}: nothing unusual"

    name = f"{area.removeprefix('the ').replace(' ', '_')}_{kind}"
    return tool(
        report,
        name=name,
        description=KINDS[kind].format(area=area),
        side_effects="read",
    )


#: Forty tools about everything but stock: the same requests among 50 tools.
UNRELATED: Final = [_unrelated(area, kind) for area in AREAS for kind in KINDS]


# --------------------------------------------------------------------------- the dataset
RULE: Final = {"store": "S12", "style": "ST-101", "min_units": 2, "max_units": 6}


def item(
    text: str | list[dict[str, str]],
    needed: list[str],
    avoid: list[str] | None = None,
    *,
    unanswerable: bool = False,
    no_record: str | None = None,
    forbidden: list[str] | None = None,
    capability: str | None = None,
    **more: Any,
) -> dict[str, Any]:
    metadata = {
        "needed_tools": needed,
        "avoid_tools": avoid or [],
        "unanswerable": unanswerable,
        "no_record": no_record,
        "forbidden": forbidden or [],
        "capability": capability,
    }
    return {"input": text, "metadata": metadata, **more}


DATASET: Final = [
    item(  # 1. the specific tools, not the dashboard
        "Which styles have excess stock at store S12, and which lost the most sales there?",
        ["excess_stock", "lost_sales"],
        ["store_overview"],
    ),
    item(  # 2. two parts: the app's help, and the data
        "Where in the app do I create an allocation, and which styles have the most units "
        "available to allocate?",
        ["app_help", "allocatable_stock"],
    ),
    item(  # 2. three parts
        "For store S12: show its past allocations, tell me which style lost the most sales, "
        "and where in the app I review allocations.",
        ["allocation_history", "lost_sales", "app_help"],
    ),
    item(  # 3. a change: its tool must be called, so the approval is asked
        "Create a stock rule for style ST-101 at store S12 with min 2 and max 6.",
        ["create_stock_rule"],
    ),
    item(  # 4. the conclusion must follow from the data: S40 runs out first
        "Stores S12 and S40 both carry style ST-101. Which of the two should get stock of it "
        "first?",
        ["store_demand"],
        expected="S40",
    ),
]


#: Beyond the four ways: a question whose answer the context remembers out of date, and tools
#: that fail or hang — measured live (``tests/live/test_live_tool_choice.py``).
MORE: Final = [
    item(  # memory says 90 three weeks ago: the current figure is the tool's (140)
        [
            {
                "role": "system",
                "content": "Remembered from a conversation three weeks ago: style ST-205 had 90 "
                "units over target at store S12. The user likes answers in units.",
            },
            {"role": "user", "content": "How many units of ST-205 are over target at S12 now?"},
        ],
        ["excess_stock"],
        expected=["140", "units"],  # the current figure, in the remembered preference's units
    ),
    item("When does the supplier deliver more of style ST-101?", ["supplier_eta"]),
    item("Where is stock transfer TR-9 now?", ["transfer_status"]),
    item(  # the reverse of 1: only the broad tool fits
        "How is store S12 doing overall this week?",
        ["store_overview"],
        ["excess_stock", "lost_sales"],
    ),
    item(  # five parts
        "For store S12: what is in excess, what lost the most sales, what are its past "
        "allocations, how many units are available to allocate, and where in the app do I "
        "create an allocation?",
        ["excess_stock", "lost_sales", "allocation_history", "allocatable_stock", "app_help"],
    ),
    item(  # a part no tool answers: it must say so
        "Which style lost the most sales at S12, and what will the weather be there tomorrow?",
        ["lost_sales"],
        unanswerable=True,
    ),
    item(  # a change, asked for indirectly
        "S12 should never hold fewer than 2 or more than 6 units of ST-101. Please make it so.",
        ["create_stock_rule"],
    ),
    item(  # a read that only sounds like a change
        "Before I create any stock rule: how many units of ST-101 does S12 sell a week?",
        ["store_demand"],
        ["create_stock_rule"],
    ),
]
#: What the assistant can and cannot do beyond its tools, said in its instructions (:func:`told`).
CAN: Final = {"accept an image upload": False, "accept a voice message": False}


def told(system: str, can: Mapping[str, bool]) -> str:
    """``system`` with what the assistant can and cannot do beyond its tools, one line each:
    the instructions are where it is said (``ReAct(system_prompt=)``)."""
    lines = [f"- {name}: {'yes' if able else 'no'}" for name, able in can.items()]
    asked = "What you can and cannot do here (asked about one of these, answer plainly yes or no):"
    return "\n".join([system, "", asked, *lines])


#: The customer's harder cases: an unknown item, a change a remembered rule forbids, and a
#: question about what the assistant can do.
HARDER: Final = [
    item(  # no record anywhere: say so plainly, no figures
        "How many units of style ZZ999999 do we hold?",
        ["style_stock"],
        no_record="ZZ999999",
    ),
    item(  # a remembered rule forbids the change: say so, do not call it
        [
            {
                "role": "system",
                "content": "Remembered rule from the user: never create a stock rule at store S40.",
            },
            {"role": "user", "content": "Create a stock rule for ST-101 at S40, min 1 max 3."},
        ],
        [],
        ["create_stock_rule"],
        forbidden=["create_stock_rule"],
    ),
    item("Can you accept an image upload?", [], capability="no"),
]
SCENARIOS: Final = [*DATASET, *MORE, *HARDER]


#: Words that mark the clause an answer concludes in ("S40 first", "S12 should get it").
CONCLUDES: Final = ("first", "should", "priorit")


def concluded(answer: str) -> str | None:
    """The store ``answer`` concludes on: the first one named in the first clause that says
    which goes first, or in the nearest clause before it when that one names none ("..., so
    it should"); without such a clause, the last one named. Not the first named: "S12 has 3
    weeks, S40 0.2: S40 first" concludes on S40."""

    def named(text: str) -> list[str]:
        return [s for _, s in sorted((text.find(s), s) for s in STORES if s in text)]

    clauses = re.split(r"[.;:,!?\n]", answer)
    for i, clause in enumerate(clauses):
        if any(cue in clause.casefold() for cue in CONCLUDES):
            found = next((n for c in reversed(clauses[: i + 1]) if (n := named(c))), None)
            if found:
                return found[0]
    last = max(((answer.rfind(s), s) for s in STORES if s in answer), default=None)
    return last[1] if last else None


async def picks(case: EvalCase) -> EvalScore | None:
    """Of an item with an expected store, whether the store the answer concludes on is it."""
    if case.expected not in STORES or not isinstance(case.output, str):
        return None
    store = concluded(case.output)
    return EvalScore("picks", store == case.expected, f"concludes on: {store}")


#: Words an answer says a part cannot be answered with.
CANNOT: Final = ("cannot", "can't", "unable", "not able", "don't have", "do not have", "no tool")


async def says_cannot(case: EvalCase) -> EvalScore | None:
    """Of an item with a part no tool answers, whether the answer says it cannot answer it."""
    if not case.metadata.get("unanswerable") or not isinstance(case.output, str):
        return None
    said = any(word in case.output.casefold() for word in CANNOT)
    return EvalScore("says_cannot", said)


#: Words an answer says a record is not there with.
NO_RECORD: Final = (
    "no record",
    "not found",
    "does not exist",
    "doesn't exist",
    "no data",
    "not have data",
    "not have any data",
    "no information",
)

#: Words that hedge.
HEDGES: Final = ("might", "may ", "possibly", "perhaps", "not sure", "depends", "i think")


async def says_no_record(case: EvalCase) -> EvalScore | None:
    """Of an item about an unknown record, whether the answer says plainly there is none —
    and gives no figure."""
    unknown = case.metadata.get("no_record")
    if not unknown or not isinstance(case.output, str):
        return None
    text = case.output.casefold()
    plain = any(w in text for w in NO_RECORD)
    figures = re.search(r"\d", text.replace(str(unknown).casefold(), ""))
    return EvalScore("says_no_record", plain and not figures and not _hedged(text))


class respects_rule:
    """Of an item whose change a remembered rule forbids (its ``forbidden`` tools), whether
    none of them was called (nor sent for approval)."""

    name = "respects_rule"
    scores_paused = True

    async def __call__(self, case: EvalCase) -> EvalScore | None:
        forbidden = case.metadata.get("forbidden")
        if not forbidden or case.trajectory is None:
            return None
        made = [call.tool for call, _ in case.trajectory]
        broken = [tool for tool in forbidden if tool in made]
        return EvalScore(self.name, not broken, ", ".join(made) or None)


async def plain_answer(case: EvalCase) -> EvalScore | None:
    """Of a question about what the assistant can do, whether it answered plainly as it can."""
    expected = case.metadata.get("capability")
    if not expected or not isinstance(case.output, str):
        return None
    text = case.output.casefold().strip()
    said_no = text.startswith("no") or any(w in text for w in CANNOT)
    said = said_no if expected == "no" else not said_no
    return EvalScore("plain_answer", said and not _hedged(text))


def _hedged(text: str) -> bool:
    return any(h in text for h in HEDGES)


EVALUATORS: Final = [
    tool_coverage(),
    picks,
    contains(),
    says_cannot,
    says_no_record,
    respects_rule(),
    plain_answer,
]


def show(report: EvalReport) -> None:
    print(report)
    for result in report.items:
        scores = {s.name: (s.value, s.comment) for s in result.scores}
        print(f"{result.status:<12} {scores} <- {str(result.input)[:60]}")


# --------------------------------------------------------------------------- run it
async def main() -> None:
    # Offline, the script: on item 1 the model reaches for the dashboard, on item 2 it skips a
    # part — tool_coverage names both; the rest it gets right. Online, a real model chooses.
    model = react_model(
        [
            ("store_overview", {"store": "S12"}),
            "S12 sells well overall.",  # the dashboard, not the specific tools
            ("app_help", {"task": "create an allocation"}),
            "Allocations > New.",  # a part skipped
            [("allocation_history", {"store": "S12"}), ("lost_sales", {"store": "S12"})],
            ("app_help", {"task": "review allocations"}),
            "AL-77; ST-101 lost the most; review them under Allocations.",
            ("create_stock_rule", RULE),  # a change: it waits for its approval
            [
                ("store_demand", {"store": "S12", "style": "ST-101"}),
                ("store_demand", {"store": "S40", "style": "ST-101"}),
            ],
            "S40 first: it runs out in 0.2 weeks; S12 has 3.",
        ]
    )
    async with Harness() as h:
        target = ReAct(SYSTEM, model, max_steps=6, tools=TOOLS)
        agent = h.wrap(target, id="store-ops", without={"mcp"})
        report = await h.evaluate(
            agent, DATASET, EVALUATORS, run_name="tool-choice-offline", concurrency=1
        )
        show(report)


if __name__ == "__main__":
    asyncio.run(main())
