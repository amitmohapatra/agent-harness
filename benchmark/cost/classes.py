"""The ReAct loop's task classes (``--suite react``): the class suite's ``simple``, ``multi``
and ``wide`` tasks with ten more each, and four classes of their own, each a different task:

* ``simple`` (20) — one tool, a change behind approval, a read that sounds like a change;
* ``multi`` (20) — two or three parts, each its own tool;
* ``wide`` (20) — four to six parts, or one tool for four to six items (each item named in
  the answer: ``contains``);
* ``error`` (20) — ten simple and ten multi-part tasks whose first call of a needed tool fails
  (:class:`Fault`: ten a refused request, five a timeout, five a schema the service no longer
  takes), the next call of it passing: the run recovers or not;
* ``memory`` (20) — ten questions only the seeded notes answer (:data:`NOTES`, added for the
  benchmark's user by :func:`seed`; some need two facts of them), five about records the
  notes do not hold (it says so, with no figure: ``says_no_record``) and five
  general-knowledge controls (answered without memory);
* ``ambiguous`` (20) — ten changes missing a value their tool requires (it asks for it and
  changes nothing: :func:`asks_back`), and the same ten fully given (controls: the change
  asked, behind approval);
* ``helpers`` (10) — eight to twelve items of one kind, for the helper arms (``helpers``,
  ``helpers-admitted``: wall time, model errors, memory and CPU);
* ``reasoning`` (40) — maths word problems, dates, multi-hop and logic with exact answers,
  half ``dev`` and half ``test`` (``benchmark.cost.reasoning``).

The large-catalog class is the class suite's ``mix`` at 50 and 100 tools
(``--suite classes --classes mix --sizes 50,100``).
"""

from __future__ import annotations

import dataclasses
import importlib
from dataclasses import dataclass
from typing import Any, Final

from benchmark.cost import catalog, reasoning
from trellis.contracts import ToolStatus
from trellis.harness.evals import EvalScore

scenario = importlib.import_module("examples.06_scenarios.tool_choice")

#: The classes of the ``react`` suite, in the order a report lists them.
CLASSES: Final = (
    "simple",
    "multi",
    "wide",
    "error",
    "memory",
    "ambiguous",
    "helpers",
    "reasoning",
)
#: The metadata key of a task whose needed tool fails once (:class:`Fault`).
INJECT: Final = "inject"


# --------------------------------------------------------------------------- faults
@dataclass(frozen=True)
class Fault:
    """The first call of ``tool`` in a run fails as ``kind`` says; its next call passes."""

    tool: str
    kind: str  # "error" | "timeout" | "schema"

    def outcome(self, args: dict[str, Any]) -> tuple[ToolStatus, str]:
        """The failed call's status and what the model reads."""
        if self.kind == "timeout":
            return ToolStatus.TIMEOUT, f"{self.tool} timed out after 30s"
        if self.kind == "schema":
            names = ", ".join(sorted(args)) or "none"
            return (
                ToolStatus.ERROR,
                f"{self.tool} failed: 422 Unprocessable Entity: the service's schema changed "
                f"and the request did not match it (fields sent: {names})",
            )
        return (
            ToolStatus.ERROR,
            f"{self.tool} failed: 409 Conflict: the record was being updated and the request "
            "was not applied",
        )


#: The kinds of fault, ten, five and five of twenty.
FAULTS: Final = ("error",) * 10 + ("timeout",) * 5 + ("schema",) * 5


def fault_of(task: dict[str, Any]) -> Fault | None:
    """The fault a task injects, if any."""
    found = task["metadata"].get(INJECT)
    return Fault(**found) if found else None


# --------------------------------------------------------------------------- memory
#: What the memory class's notes hold: a document added for the benchmark's user.
NOTES: Final = """# Store operations notes

- Store S12 is managed by Priya Rao. Store S40 is managed by Tomas Lind.
- Store S7 is managed by Amara Osei and opened in 2019.
- S40 is twinned with S12: transfers between them need no approval.
- The supplier of style ST-101 is Nordwear, based in Oslo.
- The supplier of style ST-205 is Calder Knit, based in Leeds.
- Calder Knit delivers on Thursdays; Nordwear delivers on Mondays.
- Store S7's stock count is done on the last Friday of each month.
- The regional buyer for the north is Ines Moreau.
"""
#: The notes' title, as the document is added.
NOTES_TITLE: Final = "Store operations notes"


async def seed(h: Any, user: str) -> None:
    """The memory class's notes added for ``user`` (indexed before it returns)."""
    await h.add_document(
        ("store-notes.md", NOTES.encode(), "text/markdown"), user=user, title=NOTES_TITLE
    )


def _memory() -> list[dict[str, Any]]:
    item = scenario.item
    known = [
        ("Who manages store S12?", "Priya"),
        ("Who manages store S40?", "Tomas"),
        ("When did store S7 open?", "2019"),
        ("Who supplies style ST-205?", "Calder"),
        ("In which city is the supplier of style ST-101 based?", "Oslo"),  # two facts
        ("On which day does the supplier of ST-205 deliver?", "Thursday"),  # two facts
        ("Who manages the store that S12 is twinned with?", "Tomas"),  # two facts
        ("Between which two stores do transfers need no approval?", ["S40", "S12"]),
        ("When is the stock count at store S7?", "last Friday"),
        ("Who is the regional buyer for the north?", "Ines"),
    ]
    unknown = [
        ("Who manages store S99?", "S99"),
        ("Who supplies style ST-999?", "ST-999"),
        ("When did store S88 open?", "S88"),
        ("On which day does the supplier of ST-777 deliver?", "ST-777"),
        ("Who is the regional buyer for the south-east?", "south-east"),
    ]
    controls = [
        ("What is 15% of 240?", "36"),
        ("How many days are there in a leap year?", "366"),
        ("What is the capital of Norway?", "Oslo"),
        ("How many metres are there in 2 kilometres?", "2000"),
        ("What is 7 times 8?", "56"),
    ]
    return [
        *(_classed(item(q, [], expected=a), f"memory: {q.lower()}", "memory") for q, a in known),
        *(
            _classed(item(q, [], no_record=r), f"memory: no record of {r}", "memory")
            for q, r in unknown
        ),
        *(
            _classed(item(q, [], expected=a), f"memory: a control, {q.lower()}", "memory")
            for q, a in controls
        ),
    ]


# --------------------------------------------------------------------------- ambiguous
#: The tool a run asks its user a question with (``techniques.ask_user``).
ASK_TOOL: Final = "ask_user"


def asked_in(output: Any, made: list[str]) -> bool:
    """Whether a run asked its user: the question tool called, or a question in its answer."""
    return ASK_TOOL in made or (isinstance(output, str) and "?" in output)


def gave_the_value(meta: dict[str, Any], args: Any) -> bool:
    """Whether a change's arguments hold the value the task left out (``metadata["gold"]``)."""
    text = str(args).casefold()
    return all(token.casefold() in text for token in meta.get("gold") or [])


async def asks_back(case: Any) -> EvalScore | None:
    """Of a change missing a value its tool requires (``metadata["ambiguous"]``), whether the
    run asked for it — the question tool, or a question in its answer — and then made no change
    (nor asked to: a call paused for approval is one) or made it with the value the user gave
    (``metadata["gold"]``); a change with a guessed value fails."""
    if not case.metadata.get("ambiguous"):
        return None
    steps = list(case.trajectory or [])
    made = [call.tool for call, _ in steps]
    avoid = case.metadata.get("avoid_tools", [])
    changes = [call for call, _ in steps if call.tool in avoid]
    right = all(gave_the_value(case.metadata, getattr(c, "args", {})) for c in changes)
    return EvalScore("asks_back", asked_in(case.output, made) and right, ", ".join(made) or None)


asks_back.scores_paused = True  # type: ignore[attr-defined]


#: The ambiguous class's changes: what is said without a required value, and the change in
#: full (its control), by the tool that makes it and the task set it is of.
AMBIGUOUS: Final = (
    ("store", "create_stock_rule", "Create a stock rule for ST-101 at S12.",
     "Create a stock rule for ST-101 at S12 with min 2 and max 6."),
    ("store", "create_stock_rule", "Create a stock rule at S40 with min 2 and max 6.",
     "Create a stock rule for ST-205 at S40 with min 2 and max 6."),
    ("store", "create_stock_rule", "Set a stock rule for ST-205: min 1, max 4.",
     "Set a stock rule for ST-205 at S7: min 1, max 4."),
    ("store", "create_stock_rule", "Create a stock rule for S7 and ST-310 with a max of 8.",
     "Create a stock rule for S7 and ST-310 with a min of 3 and a max of 8."),
    ("help desk", "reset_password", "Reset the password, please.",
     "Reset the password of user u-2041, please."),
    ("help desk", "reset_password", "My colleague is locked out: reset their password.",
     "My colleague u-3307 is locked out: reset their password."),
    ("travel", "submit_expense", "Submit an expense of 42 EUR for e-401.",
     "Submit a meals expense of 42 EUR for e-401."),
    ("travel", "submit_expense", "Submit a taxi expense for e-512.",
     "Submit a taxi expense of 18 EUR for e-512."),
    ("travel", "submit_expense", "Submit a meal expense of 30 EUR.",
     "Submit a meal expense of 30 EUR for e-618."),
    ("travel", "submit_expense", "File a hotel expense for e-618.",
     "File a hotel expense of 120 EUR for e-618."),
)  # fmt: skip
#: What each ambiguous change leaves out, by its place in :data:`AMBIGUOUS`: what the user
#: answers when asked, the words a question about it names (any one), and what the change's
#: arguments then hold (each) — the gold data a scripted answer and the scores read.
MISSING: Final = (
    ("min 2 and max 6", ("min", "max", "how many", "units"), ("2", "6")),
    ("style ST-205", ("style",), ("ST-205",)),
    ("store S7", ("store",), ("S7",)),
    ("a min of 3", ("min",), ("3",)),
    ("user u-2041", ("user", "whose", "who", "account"), ("u-2041",)),
    ("user u-3307", ("user", "whose", "who", "account"), ("u-3307",)),
    ("meals", ("category", "type", "kind", "what for", "what was"), ("meal",)),
    ("18 EUR", ("amount", "how much", "cost"), ("18",)),
    ("employee e-618", ("employee", "who", "whose", "which"), ("e-618",)),
    ("120 EUR", ("amount", "how much", "cost"), ("120",)),
)  # fmt: skip


def _ambiguous(domain: str) -> list[dict[str, Any]]:
    item = scenario.item
    found = []
    for n, (where, change, vague, full) in enumerate(AMBIGUOUS):
        if where != domain:
            continue
        asked = item(vague, [], [change])
        answer, about, gold = MISSING[n]
        asked["metadata"].update(
            ambiguous=True, answer_given=answer, about=list(about), gold=list(gold)
        )
        found += [
            _classed(asked, f"{domain}: ambiguous {n}, a value missing", "ambiguous"),
            _classed(item(full, [change]), f"{domain}: ambiguous {n}, in full", "ambiguous"),
        ]
    return found


# --------------------------------------------------------------------------- the classes
def _classed(item: dict[str, Any], name: str, kind: str) -> dict[str, Any]:
    return {**item, "metadata": {**item["metadata"], "task": name, "length": kind, "mix": False}}


def _faulted(task: dict[str, Any], kind: str) -> dict[str, Any]:
    needed = task["metadata"]["needed_tools"]
    meta = {
        **task["metadata"],
        "task": f"{task['metadata']['task']} ({kind} on {needed[0]})",
        "length": "error",
        INJECT: {"tool": needed[0], "kind": kind},
    }
    return {**task, "metadata": meta}


def _more(domain: str) -> list[dict[str, Any]]:
    """The ten simple, multi and wide tasks beyond the class suite's, of ``domain``."""
    item = scenario.item
    stores, styles = (
        ["S12", "S40", "S7", "S3", "S21", "S9"],
        ["ST-101", "ST-205", "ST-310", "ST-412"],
    )
    teams, laptops = (
        ["network", "desktop", "security", "servers"],
        ["LT-2290", "LT-1001", "LT-3307", "LT-4410"],
    )
    people = ["e-401", "e-512", "e-618", "e-733", "e-801"]
    rows = {
        "store": [
            (
                "When does the supplier deliver more of style ST-205?",
                ["supplier_eta"],
                None,
                "simple",
            ),
            (
                "How many units of ST-101 are in stock at each store?",
                ["style_stock"],
                None,
                "simple",
            ),
            ("What can be allocated right now?", ["allocatable_stock"], None, "simple"),
            ("Show store S40's past allocations.", ["allocation_history"], None, "simple"),
            (
                "For S40: what is in excess, and how many units of ST-101 does it sell a week?",
                ["excess_stock", "store_demand"],
                None,
                "multi",
            ),
            (
                "Which styles lost the most sales at S7, and where in the app do I review "
                "lost sales?",
                ["lost_sales", "app_help"],
                None,
                "multi",
            ),
            (
                "When does ST-205 arrive from the supplier, and how much of it is in stock?",
                ["supplier_eta", "style_stock"],
                None,
                "multi",
            ),
            (
                "Show S12's overview and its past allocations.",
                ["store_overview", "allocation_history"],
                None,
                "multi",
            ),
            (
                f"What is in excess at {', '.join(stores[:3])} and {stores[3]}?",
                ["excess_stock"],
                stores[:4],
                "wide",
            ),
            (
                f"Which styles lost the most sales at {', '.join(stores[:4])} and {stores[4]}?",
                ["lost_sales"],
                stores[:5],
                "wide",
            ),
            (
                f"Show the past allocations of {', '.join(stores[:3])} and {stores[3]}.",
                ["allocation_history"],
                stores[:4],
                "wide",
            ),
            (
                f"When does the supplier deliver {', '.join(styles[:3])} and {styles[3]}?",
                ["supplier_eta"],
                styles,
                "wide",
            ),
            (
                f"Give me the overview of {', '.join(stores[:5])} and {stores[5]}.",
                ["store_overview"],
                stores,
                "wide",
            ),
        ],
        "help desk": [
            ("Show me the network team's desk dashboard.", ["desk_dashboard"], None, "simple"),
            (
                "Which of the desktop team's tickets breach their SLA?",
                ["sla_breaches"],
                None,
                "simple",
            ),
            ("Who is laptop LT-3307 assigned to?", ["asset_owner"], None, "simple"),
            (
                "Show the security team's dashboard and its SLA breaches.",
                ["desk_dashboard", "sla_breaches"],
                None,
                "multi",
            ),
            (
                "List the network team's high-priority tickets, and tell me from the knowledge "
                "base how to reset MFA.",
                ["tickets_by_priority", "knowledge_base"],
                None,
                "multi",
            ),
            (
                "Who is LT-1001 assigned to, and what does the desktop team's dashboard show?",
                ["asset_owner", "desk_dashboard"],
                None,
                "multi",
            ),
            (
                f"SLA breaches for the {', '.join(teams[:3])} and {teams[3]} teams.",
                ["sla_breaches"],
                teams,
                "wide",
            ),
            (
                f"Who are laptops {', '.join(laptops[:3])} and {laptops[3]} assigned to?",
                ["asset_owner"],
                laptops,
                "wide",
            ),
        ],
        "travel": [
            (
                "What does the travel policy say about rail travel?",
                ["travel_policy"],
                None,
                "simple",
            ),
            ("Which flights does e-512 have booked?", ["flight_bookings"], None, "simple"),
            ("How many trips has e-401 taken this year?", ["trip_summary"], None, "simple"),
            (
                "How many trips has e-512 taken, and which expense claims does it have pending?",
                ["trip_summary", "expense_claims"],
                None,
                "multi",
            ),
            (
                "What does the policy say about taxis, and which flights does e-401 have booked?",
                ["travel_policy", "flight_bookings"],
                None,
                "multi",
            ),
            (
                "For e-401: trips this year, booked flights and the policy on hotels.",
                ["trip_summary", "flight_bookings", "travel_policy"],
                None,
                "multi",
            ),
            (
                f"Booked flights for {', '.join(people[:3])} and {people[3]}.",
                ["flight_bookings"],
                people[:4],
                "wide",
            ),
            (
                f"Pending expense claims for {', '.join(people[:4])} and {people[4]}.",
                ["expense_claims"],
                people,
                "wide",
            ),
            (
                f"Trips this year for {', '.join(people[:3])} and {people[3]}.",
                ["trip_summary"],
                people[:4],
                "wide",
            ),
        ],
    }
    return [
        _classed(
            item(text, needed, **({"expected": named} if named else {})),
            f"{domain}: {kind} {n}",
            kind,
        )
        for n, (text, needed, named, kind) in enumerate(rows[domain])
    ]


def _helpers(domain: str) -> list[dict[str, Any]]:
    """Eight to twelve items of one kind, one helper each."""
    item = scenario.item
    ask = "Hand each one to a helper, all at once, then sum up."
    stores = [f"S{n}" for n in (12, 40, 7, 3, 21, 9, 15, 30, 44, 51, 60, 72)]
    styles = [f"ST-{n}" for n in (101, 205, 310, 412, 518, 620, 733, 845)]
    people = [f"e-{n}" for n in (401, 512, 618, 733, 801, 902, 913, 924, 935)]
    teams = [*catalog.TEAMS[:8]]
    rows = {
        "store": [
            (
                f"The overview of each store: {', '.join(stores[:8])}.",
                ["store_overview"],
                stores[:8],
            ),
            (
                f"What is in excess at each store: {', '.join(stores[:10])}.",
                ["excess_stock"],
                stores[:10],
            ),
            (f"The overview of each store: {', '.join(stores)}.", ["store_overview"], stores),
            (
                f"The supplier's next delivery of each style: {', '.join(styles)}.",
                ["supplier_eta"],
                styles,
            ),
        ],
        "help desk": [
            (f"SLA breaches of each team: {', '.join(teams)}.", ["sla_breaches"], teams),
            (f"The desk dashboard of each team: {', '.join(teams)}.", ["desk_dashboard"], teams),
        ],
        "travel": [
            (f"Booked flights of each employee: {', '.join(people)}.", ["flight_bookings"], people),
            (
                f"Trips this year of each employee: {', '.join(people[:8])}.",
                ["trip_summary"],
                people[:8],
            ),
            (
                f"Pending expense claims of each employee: {', '.join(people)}.",
                ["expense_claims"],
                people,
            ),
            (
                f"Booked flights of each employee: {', '.join(people[:8])}.",
                ["flight_bookings"],
                people[:8],
            ),
        ],
    }
    return [
        _classed(item(f"{text} {ask}", needed, expected=named), f"{domain}: helpers {n}", "helpers")
        for n, (text, needed, named) in enumerate(rows[domain])
    ]


def react_sets() -> list[catalog.TaskSet]:
    """The ``react`` suite: the class suite's three domains, their large tasks left out, each
    with the tasks of every class above (:data:`CLASSES`)."""
    found = []
    for taskset in catalog.class_sets():
        kept = [t for t in taskset.tasks if t["metadata"]["length"] in ("simple", "multi", "wide")]
        mine = [*kept, *_more(taskset.name)]
        found.append(dataclasses.replace(taskset, tasks=mine))
    # the error class: ten simple and ten multi tasks that only read, the first call of a
    # needed tool failing (a change pauses for approval before it runs: never one of them)
    chosen = [*_spread(_reading(found, "simple"), 10), *_spread(_reading(found, "multi"), 10)]
    faulted = {id(t): _faulted(t, kind) for t, kind in zip(chosen, FAULTS, strict=True)}
    for taskset in found:
        mine = [faulted[id(t)] for t in taskset.tasks if id(t) in faulted]
        memory = _memory() if taskset.name == "store" else []
        taskset.tasks.extend([*mine, *memory, *_ambiguous(taskset.name), *_helpers(taskset.name)])
    return [*found, catalog.TaskSet("reasoning", reasoning.SYSTEM, [], reasoning.items())]


def _reading(sets: list[catalog.TaskSet], kind: str) -> list[dict[str, Any]]:
    """The tasks of class ``kind`` that need tools and change nothing."""
    changes = {t.spec.name for s in sets for t in s.tools if t.spec.side_effects != "read"}
    return [
        t
        for s in sets
        for t in s.tasks
        if t["metadata"]["length"] == kind
        and t["metadata"]["needed_tools"]
        and not changes & set(t["metadata"]["needed_tools"])
    ]


def _spread(tasks: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    """``count`` of ``tasks``, evenly apart (every domain among them)."""
    if len(tasks) < count:
        raise ValueError(f"{count} tasks wanted, {len(tasks)} found")
    return [tasks[n * len(tasks) // count] for n in range(count)]


#: The evaluators of the ``react`` suite: the tool-choice scenario's, :func:`asks_back` and
#: the reasoning class's ``exact_answer``.
EVALUATORS: Final = [*scenario.EVALUATORS, asks_back, reasoning.exact_answer]

__all__ = ["CLASSES", "EVALUATORS", "Fault", "fault_of", "react_sets", "seed"]
