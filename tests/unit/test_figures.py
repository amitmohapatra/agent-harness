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
    assert unbacked("An average of 4.7 a month.", "rows: 3, 5, 6") == ["4.7"]
    assert unbacked("An average of 4.7 a month.", "avg: 4.6667") == []  # the same, rounded


def test_numbers_are_read_through_thousands_commas_and_csv_cells() -> None:
    assert 1141.0 in numbers("SKU-00012,1141,39.50") and 39.5 in numbers("SKU-00012,1141,39.50")
    assert 157733.89 in numbers("£157,733.89") and -12.0 in numbers("a change of -12 units")
    assert numbers("on 2026-10-30 at 09:00, SKU-00012") == []
