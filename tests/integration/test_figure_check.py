"""``ReAct``'s figure check (``trellis.harness.figures.FigureCheck``) in a run: an answer holding
a figure worked out by hand from many is asked once more, offered ``query_result`` over this
turn's results (kept as tables, however small), and the answer computed with it is the run's;
an answer whose figures trace costs nothing more; the draft and the note never reach the
conversation the graph keeps."""

from __future__ import annotations

import json
import re
from typing import Any, Final

from tests.support.arm import arm
from tests.support.models import Script, ScriptedChat
from trellis import Harness, ReAct, middleware, tool
from trellis.contracts import RunEventType, RunStatus
from trellis.harness import figures

MONTHS: Final = [1407, 1540, 1589, 1802, 2203, 2246, 2420, 2472, 2424]
HAND: Final = "The slope is 151.3 units a month; Q4 comes to about 8,760 units."
SQL: Final = (
    "SELECT regr_slope(value, n) AS slope, regr_intercept(value, n) AS icept FROM "
    "(SELECT value, row_number() OVER (ORDER BY key) AS n FROM monthly_units)"
)
#: Where the note says a result was kept.
KEPT: Final = re.compile(r"monthly_units \((/large_tool_results/[^)]+)\)")


@tool(side_effects="read")
def monthly_units() -> str:
    """Outerwear's units sold per month of 2026."""
    rows = [{"key": f"2026-{n:02d}", "value": v} for n, v in enumerate(MONTHS, 1)]
    return json.dumps({"group_by": "month", "rows": rows})


@tool(side_effects="read")
def note() -> str:
    """A note on outerwear."""
    return "Outerwear sells more in autumn."


def last(request: dict[str, Any]) -> str:
    return str(request["messages"][-1].get("content"))


def offered(request: dict[str, Any]) -> list[str]:
    return [t["function"]["name"] for t in request.get("tools", [])]


class Recomputes(Script):
    """Fetches the months, answers by hand, then — told to compute — queries the table the
    note names and answers with what the query gave."""

    def __init__(self) -> None:
        super().__init__([("monthly_units", {}), HAND])

    def next(self, body: dict[str, Any]) -> Any:
        if self.turns:
            return super().next(body)
        told = last(body)
        if told == "Thanks.":
            return "You're welcome."
        kept = KEPT.search(told)
        if kept:
            return ("query_result", {"file_path": kept.group(1), "sql": SQL})
        found = re.search(r"slope,icept\n([\d.]+),([\d.]+)", told)
        assert found, told
        return f"Slope {float(found[1]):.1f} units a month (intercept {float(found[2]):.1f})."


async def test_a_figure_worked_out_by_hand_is_computed_with_a_tool(harness: Harness) -> None:
    script = Recomputes()
    model = ScriptedChat(script=script)
    agent = arm(harness.wrap(ReAct("s", model), id="fit"), [monthly_units])
    events = [e async for e in agent.stream("Fit a trend to monthly units; Q4?", user="u")]
    done = events[-1]
    assert done.outcome is not None and done.outcome.value == "success", done
    asked = model.requests
    assert len(asked) == 4  # the call, the draft, the query, the answer
    told = last(asked[2])
    assert told.startswith("Before you answer: 151.3, 8,760 in your answer are"), told
    assert "regr_slope(y, x)" in told and KEPT.search(told), told
    assert "query_result" not in offered(asked[1]) and "query_result" in offered(asked[2])
    assert asked[2]["messages"][-2]["content"] == HAND  # the draft, then the note
    answer = next(e for e in reversed(events) if e.type is RunEventType.RUN_FINISHED)
    assert "Slope 149.5 units a month (intercept 1263.9)" in str(answer.data), answer.data
    assert any("figure_check" in str(e.data) for e in events if e.type is RunEventType.CUSTOM)


async def test_an_answer_whose_figures_trace_costs_no_more_calls(harness: Harness) -> None:
    model = ScriptedChat(
        [("monthly_units", {}), "September sold 2,424 units, 52 fewer than August's 2,476."]
    )
    agent = arm(harness.wrap(ReAct("s", model), id="traced"), [monthly_units])
    # 2,476 is not August's (2,472): a figure one step from two the answer states is traced,
    # right or not — the check finds hand work, the scorer finds wrong answers
    result = await agent.run("September's units?", user="u")
    assert result.status is RunStatus.SUCCESS, result.error
    assert len(model.requests) == 2


async def test_a_turn_is_checked_once(harness: Harness) -> None:
    model = ScriptedChat([("monthly_units", {}), HAND, "Still 151.3 a month; about 8,760 in Q4."])
    agent = arm(harness.wrap(ReAct("s", model), id="once"), [monthly_units])
    result = await agent.run("Fit a trend; Q4?", user="u")
    assert result.status is RunStatus.SUCCESS, result.error
    assert len(model.requests) == 3
    assert result.answer == "Still 151.3 a month; about 8,760 in Q4."


async def test_with_nothing_to_compute_with_the_answer_stands(harness: Harness) -> None:
    """No table among this turn's results and no code tool: nothing would compute it."""
    model = ScriptedChat([("note", {}), HAND])
    agent = arm(harness.wrap(ReAct("s", model), id="nothing"), [note])
    result = await agent.run("Fit a trend; Q4?", user="u")
    assert result.status is RunStatus.SUCCESS, result.error
    assert len(model.requests) == 2 and result.answer == HAND


async def test_the_conversation_keeps_the_answer_not_the_draft_or_the_note(
    harness: Harness,
) -> None:
    model = ScriptedChat(script=Recomputes())
    agent = arm(harness.wrap(ReAct("s", model), id="kept"), [monthly_units])
    first = await agent.run("Fit a trend; Q4?", user="u", thread="t")
    assert first.status is RunStatus.SUCCESS, first.error
    later = await agent.run("Thanks.", user="u", thread="t")
    assert later.status is RunStatus.SUCCESS, later.error
    history = json.dumps(model.requests[-1]["messages"])
    assert "Before you answer" not in history and "151.3" not in history
    assert "Slope 149.5" in history


def test_the_check_is_react_s_own() -> None:
    """``middleware()`` for a team's own ``create_agent`` graph leaves it out: an answer's
    figures are checked in the loop ``ReAct`` adds on top of the frameworks."""
    assert all(type(m).__name__ != "FigureCheck" for m in middleware())
    assert figures.FigureCheck().name == "FigureCheck"
