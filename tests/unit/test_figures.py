"""``trellis.harness.figures``: which figures of an answer trace to what the model was given —
held as given or rounded, worked a step or two from traced figures, a total of a result's rows
or of a list the answer gives, a record's margin — and which were worked out by hand from many
figures at once (a fitted line, a compound rate, an unshown sum)."""

from __future__ import annotations

import json
from typing import Final

from trellis.harness import figures

#: Outerwear's 2026 units by month, as the retail tool returns them (9 rows).
MONTHS: Final = json.dumps(
    {
        "group_by": "month",
        "rows": [
            {"key": f"2026-{n:02d}", "value": v}
            for n, v in enumerate((1407, 1540, 1589, 1802, 2203, 2246, 2420, 2472, 2424), 1)
        ],
    }
)
FIT: Final = (
    "Fit a least-squares trend to monthly units (January = 1 ... September = 9). The slope, and "
    "the total forecast for Q4 (months 10 to 12)?"
)
#: 92 days of units: a table (more than ``figures.RUN`` figures), never totalled by the check.
DAILY: Final = "date,units\n" + "\n".join(
    f"2026-07-{d % 28 + 1:02d},{40 + d % 17}" for d in range(92)
)


def test_figures_read_separators_signs_and_precision_and_skip_dates_times_and_years() -> None:
    found = figures.figures(
        "On 2026-09-30 at 09:00 (Q3 2026, 2026-07, 30 Sep): £1,234.5, \u22127.25%, 7,800 and 0.214."
    )
    assert found == [(30.0, 5.0), (1234.5, 0.05), (-7.25, 0.005), (7800.0, 50.0), (0.214, 0.0005)]
    assert figures.figures("SKU-00602 and S14") == [(602.0, 0.5)]  # an id's digits, read alone
    assert figures.figures("2,026 units") == [(2026.0, 0.5)]  # with a separator, a figure


def test_a_figure_given_as_it_is_rounded_or_rescaled_is_traced() -> None:
    given = [MONTHS, '{"revenue": 2897540.12, "rate": 0.0693}']
    for answer in (
        "September sold 2,424 units.",
        "About 2.9m in revenue (£2,897.5k).",
        "Revenue was £2,897,540.",
        "A rate of 6.93%.",
    ):
        assert figures.untraced(answer, given) == [], answer


def test_working_shown_step_by_step_is_traced() -> None:
    given = ['{"units_28d": 6, "lead_time_days": 32}', '{"on_hand": 5}']
    answer = "0.214 a day x 39 days = 8.36, less 5 on hand: 3.36, so order 4."
    assert figures.untraced(answer, given, "Review period 7 days; 28 days of sales.") == []
    assert (
        figures.untraced("0.214 x 39 = 8.346, so about 8.35.", [], "6 sold in 28 days; 39 days")
        == []
    )
    # a share, a % change, a gap between two figures it was given
    assert figures.untraced("1,802 is 2.40% above 1,759.75.", ['{"a": 1802, "b": 1759.75}']) == []
    record = '{"price": 79.0, "unit_cost": 29.8}'
    fee = "Break-even units for a £6,000 fee?"
    assert (
        figures.untraced("A margin of £49.20 a unit, so 6,000 / 49.20 = 122 units.", [record], fee)
        == []
    )


def test_a_total_of_a_results_rows_or_of_a_list_the_answer_gives_is_traced() -> None:
    q3 = '{"rows": [{"m": "2026-07", "v": 2280481}, {"m": "2026-08", "v": 2279729}, {"m": "2026-09", "v": 2380783}]}'
    assert figures.untraced("Q3 revenue was £6,940,993.", [q3]) == []
    listed = "North 1,523.5, south 1,947.25, east 1,972.75 and west 1,497.5: 6,941 in all."
    given = ['{"north": 1523.5, "south": 1947.25, "east": 1972.75, "west": 1497.5, "x": 9}']
    assert figures.untraced(listed, given) == []


def test_a_figure_worked_from_many_at_once_is_not_traced() -> None:
    """A least-squares line, a compound rate, a total of a table's rows the check does not add
    up (a table is computed over with a tool): each is named, as written."""
    hand = "The slope is 151.3 units a month, so Q4 comes to about 8,760 units."
    assert figures.untraced(hand, [MONTHS], FIT) == ["151.3", "8,760"]
    rate = "Revenue grew 6.93% a month, so December would be £2,911,135."
    revenue = '{"rows": [{"m": "2026-01", "v": 1392535}, {"m": "2026-09", "v": 2380783}]}'
    assert figures.untraced(rate, [revenue]) == ["6.93", "2,911,135"]
    assert figures.untraced("They sold 4,320 units in Q3.", [DAILY]) == ["4,320"]


def test_figures_leaning_on_each_other_do_not_trace_one_another() -> None:
    """Forecasts made from a slope agree with it, and it with them: none is traced unless the
    slope is (grounded, not circular)."""
    answer = (
        "Slope +149.5 units a month (intercept 1,263.9). October 2,758.9, November 2,908.4, "
        "December 3,057.9: 8,725.2 in Q4."
    )
    assert figures.untraced(answer, [MONTHS], FIT) == [
        "149.5", "1,263.9", "2,758.9", "2,908.4", "3,057.9", "8,725.2",
    ]  # fmt: skip
    computed = "regr_slope,regr_intercept\n149.5,1263.94"
    assert figures.untraced(answer, [MONTHS, computed], FIT) == []


def test_counts_and_figures_the_question_gives_are_not_checked() -> None:
    assert figures.untraced("Top 3 of 40 stores; 97 units.", []) == []
    assert (
        figures.untraced("At £95 the fit gives 36.24.", [], "Predict at a price of £95 (36.24)")
        == []
    )


def test_a_long_answer_of_figures_is_read_not_searched() -> None:
    """Past ``MOST_STATED`` traced figures, a figure one step from none is taken as traced:
    a table is read, and the two-step search would be slow and trace anything."""
    table = "\n".join(f"| S{n:02d} | {1000 + 7 * n} |" for n in range(1, 50))
    given = ["\n".join(str(1000 + 7 * n) for n in range(1, 50))]
    assert figures.untraced(f"{table}\nOne more: 98,765,432.1.", given) == []


def test_a_figure_past_a_floats_range_is_no_figure_and_one_said_twice_is_named_once() -> None:
    assert figures.figures("1" + "0" * 400) == []
    assert figures.untraced("8,760 units; yes, 8,760.", ["1407, 1540"]) == ["8,760"]


def test_no_operation_divides_by_zero() -> None:
    """A zero operand, a zero target and a -100% change are skipped where they would divide."""
    assert len(list(figures._inverses(5.0, 0.1, 0.0))) == 6
    assert len(list(figures._inverses(0.0, 0.1, 2.0))) == 8
    assert len(list(figures._inverses(-100.0, 0.1, 2.0))) == 9
    assert len(list(figures._made((3.0, 0.0), (0.0, 0.0)))) == 3


def test_a_table_is_kept_to_compute_over_only_with_a_column_of_numbers() -> None:
    """A search's notes, ids and dates are no figures to compute; a CSV's numbers as text are."""
    assert figures._numeric([["2026-07-01", "41"], ["2026-07-02", "38.5"]])
    assert figures._numeric([[1, "a"], [2.5, "b"]])
    assert not figures._numeric([["mem_1", "true", None], ["mem_2", "x", "2026-09-01"]])
