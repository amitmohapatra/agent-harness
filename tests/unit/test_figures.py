"""``figures.unbacked``: the figures of an answer that nothing its model was given holds."""

from __future__ import annotations

import pytest

from trellis.harness.figures import numbers, unbacked

GIVEN = (
    'sales_summary: {"rows": [{"key": "east", "value": 157733.89}, {"key": "west", "value": 216700}]}\n'
    "sku,units,price\nSKU-00012,1141,39.50\nSKU-00034,253,12.00\n"
    "stock as of 2026-10-30T09:00:00Z: 965 units, share 0.1234"
)


@pytest.mark.parametrize(
    "answer",
    [
        "East sold £157,733.89 and west £216,700.",
        "East is about £158k; west £216.7k (0.2167 million).",  # shown in thousands, millions
        "SKU-00012 sold 1,141 units at £39.50; SKU-00034 sold 253.",
        "The share is 12.3% (0.1234), counted on 2026-10-30 at 09:00.",  # a share as a percent
        "There are 3 reasons, and the top 5 are listed; 40 of 100 stores.",  # counts: not checked
        "Run AL-13 in 2026 used version v2.6.2.",  # ids, a year, a version: no figures
        "",
    ],
)
def test_a_figure_the_model_was_given_is_backed_within_its_rounding(answer: str) -> None:
    assert unbacked(answer, GIVEN) == []


def test_a_figure_worked_out_by_hand_is_named_once_as_the_answer_wrote_it() -> None:
    answer = (
        "1.09 x 0.95 = 1.0355, so east would need about £188k; that is 1.0355 again, "
        "and 373,000 in total."
    )
    assert unbacked(answer, GIVEN) == ["1.09", "0.95", "1.0355", "188", "373000"]


def test_plain_arithmetic_on_held_figures_is_checked_here_not_sent_back() -> None:
    """A difference, a share, a growth rate and a table's total of figures the answer has a
    source for are done here: right, they need no tool (four answers in ten state one); wrong,
    they match nothing and are named."""
    given = "east 653411.68 west 484723.88 north 1523104.74 south 1947395.24"
    right = (
        "East made £653,411.68 against west's £484,723.88: higher by £168,687.80, about "
        "34.8% more. With north's £1,523,104.74 and south's £1,947,395.24 the four total "
        "£4,608,635.54, an average of £1,152,158.89."
    )
    assert unbacked(right, given) == []
    wrong = right.replace("168,687.80", "186,687.80").replace("34.8%", "38.5%")
    assert unbacked(wrong, given) == ["186687.80", "38.5"]
    assert unbacked("An average of 4.7 a month.", "rows: 3, 5, 6") == ["4.7"]
    assert unbacked("An average of 4.7 a month.", "avg: 4.6667") == []  # the same, rounded


def test_numbers_are_read_through_thousands_commas_and_csv_cells() -> None:
    assert 1141.0 in numbers("SKU-00012,1141,39.50") and 39.5 in numbers("SKU-00012,1141,39.50")
    assert 157733.89 in numbers("£157,733.89") and -12.0 in numbers("a change of -12 units")
    assert numbers("on 2026-10-30 at 09:00, SKU-00012") == []


def test_a_zero_among_the_held_figures_divides_nothing() -> None:
    given = "north 0.00 south 412.50"
    assert unbacked("North made £0.00 and south £412.50; twice south is £825.00.", given) == []


def test_arithmetic_on_arithmetic_is_followed_so_far_and_no_further() -> None:
    """A figure worked from one that was itself worked out is checked too, a few steps deep
    (``PASSES``): past that it is named, and a tool computes it."""
    given = "start 100.5 factor 3.7"
    answer = (
        "It reaches 18,835.32, after 5,090.63 and 1,375.85, from 371.85: 100.5 times 3.7, "
        "and again each step."
    )
    assert unbacked(answer, given) == ["18835.32"]


def test_a_change_said_as_how_far_under_is_the_change() -> None:
    """ "4.7% under forecast": the change's size, its direction in words."""
    given = "forecast 6535, allocated 6230"
    assert unbacked("Allocated 6,230 against 6,535, about 4.7% under forecast.", given) == []
    assert unbacked("Allocated 6,230 against 6,535, about 5.7% under forecast.", given) == ["5.7"]


def test_a_change_said_as_its_size_is_the_signed_figure_it_was_given() -> None:
    """ "Margin falls by about £4,880" from a result reading -4880.38613: a fall said in words
    is its size, so the figure is held (advscen-03's answers were sent back, and scored made
    up, for restating a tool's negative change as a fall). A size no figure rounds to is still
    named."""
    given = "cut5_delta,raise5_delta\n-4880.38613,-4710.12493"
    said = "A 5% cut: margin falls by about £4,880. A 5% raise: it still falls, by £4,710."
    assert unbacked(said, given) == []
    assert unbacked(said.replace("4,880", "4,980"), given) == ["4980"]
    assert unbacked("A rise of 12.5 points.", "change: -12.5") == []


def test_a_number_too_large_for_a_float_is_no_figure_and_breaks_nothing() -> None:
    """A run of digits past what a float holds (an id, a hash, a large result's padding) is
    read as infinity: it is no figure the answer states or a tool returned, and checking an
    answer that holds one ran into ``int(inf)`` and failed the run (the matrix's large-result
    cells)."""
    huge = "9" * 400
    assert numbers(f"rows: {huge} and 12") == [12.0, 12.0]
    assert unbacked(f"It is {huge}, and 12 units.", "12 units") == []
    assert unbacked(f"It is {huge}, and £13.47.", "£12.50") == ["13.47"]  # still checked
