"""The figures of an answer that nothing the model was given holds.

A model asked a follow-up ("why that one?", "how confident are you?") often answers with a
figure it worked out in its head — a break-even, a share, a difference — and such a figure is
the commonest wrong one (the hard set's reviews: seven to twelve of each adapter's misses).
:func:`unbacked` names them: a figure of the answer that no message of its request states,
within its rounding. What the run does with them is its loop's (``middleware.figures_checked``:
the model is asked once to compute them with a tool, or to answer without them).

A figure is a number an answer cites as a quantity: one with a decimal part, or a whole number
past :data:`COUNTED` (a count, a rank, a list's numbering, a percentage the question names are
not checked). A date, a clock time and an id's digits ("SKU-00012", "AL-13") are none.
"""

from __future__ import annotations

import re
from typing import Final

#: Whole numbers up to this are not checked: counts, ranks, the numbering of a list.
COUNTED: Final = 100
#: A number as text writes it: a sign, digits with thousands' commas, a decimal part.
_NUMBER: Final = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?")
#: What holds digits that are no figure: an ISO date or time, a date in words' day and year, a
#: clock time, an id written with a prefix, a version.
_NOT_FIGURES: Final = re.compile(
    r"\b\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?Z?)?\b"  # 2026-10-30T09:00:00Z
    r"|\b\d{1,2}:\d{2}(?::\d{2})?\b"  # 09:00
    r"|\b[A-Za-z]{1,6}[-_]?\d+(?:[-_.]\d+)*\b"  # SKU-00012, S04, AL-13, v2.6.2
    r"|\b(?:19|20)\d{2}\b"  # a year
)
#: The scales a figure is shown in beside the one it was given in: thousands, millions, and a
#: share as a percentage (0.123 as 12.3).
_SCALES: Final = (1.0, 1e-3, 1e-6, 100.0)
#: How far a figure may be from the one it restates and still be that figure: half a percent
#: (a rounding to three significant figures), or half a unit of its last place shown.
TOLERANCE: Final = 0.005
#: How many times over a figure worked out from held ones may itself be worked from (a
#: difference, then its share of a total), the most figures of an answer looked at, and the
#: longest run of them summed as a table's total.
PASSES: Final = 3
POOL: Final = 120
RUN: Final = 40


def _read(text: str) -> list[tuple[float, int]]:
    """Each number ``text`` states, with the decimal places it is written to."""
    found = []
    for m in _NUMBER.finditer(_NOT_FIGURES.sub(" ", text)):
        said = m.group(0).replace(",", "")
        found.append((float(said), len(said.partition(".")[2])))
    return found


def numbers(text: str) -> list[float]:
    """The numbers ``text`` states — read twice where commas may separate a CSV's cells rather
    than a number's thousands."""
    return [n for n, _ in (*_read(text), *_read(text.replace(",", " ")))]


def _near(value: float, other: float, step: float) -> bool:
    return abs(value - other) <= max(TOLERANCE * abs(other), step)


def worked(value: float, step: float, known: list[float]) -> bool:
    """Whether ``value`` is plain arithmetic on figures the answer already has a source for
    (``known``): the sum, the difference, the product or the ratio of two of them, one as a
    percentage of or a percentage change on another (said with its sign or as how far
    under or over), their mean, or the sum or the mean of a
    run of them as the answer lists them (a table's total). Such a figure is checked here, by
    doing the sum: right, it needs no tool; wrong, it matches nothing and is named."""
    for i, a in enumerate(known):
        for b in known[i:]:
            made = [a + b, a - b, b - a, a * b, (a + b) / 2]
            if b:
                made += [a / b, 100 * a / b, 100 * (a - b) / b, 100 * (b - a) / b]
            if a:
                made += [b / a, 100 * b / a, 100 * (b - a) / a, 100 * (a - b) / a]
            if any(_near(value, m, step) for m in made):
                return True
    for start in range(len(known)):
        total = known[start]
        for end in range(start + 1, min(len(known), start + RUN)):
            total += known[end]
            if _near(value, total, step) or _near(value, total / (end - start + 1), step):
                return True
    return False


def unbacked(answer: str, given: str) -> list[str]:
    """The figures ``answer`` states that ``given`` — everything its model was sent — does not
    hold: each as the answer wrote it, once, in the answer's order. A figure is held when one
    of ``given``'s numbers is it within :data:`TOLERANCE`, or rounds to it at the places the
    answer shows, in the scale given or another (:data:`_SCALES`) — or when it is plain
    arithmetic on figures of the answer that are held (:func:`worked`: a difference, a share,
    a total; and on what that gives in turn, :data:`PASSES` deep)."""
    held = numbers(given)
    stated = _read(answer)
    known: list[float] = []
    open_: list[tuple[float, int]] = []
    for value, places in stated:
        step = 0.5 * 10.0**-places + 1e-9
        if (value == int(value) and abs(value) <= COUNTED) or any(
            _near(value, c * scale, step) for c in held for scale in _SCALES
        ):
            known.append(value)
        else:
            open_.append((value, places))
    for _ in range(PASSES):
        still = []
        for value, places in open_:
            step = 0.5 * 10.0**-places + 1e-9
            if worked(value, step, known[:POOL]):
                known.append(value)
            else:
                still.append((value, places))
        if len(still) == len(open_):
            break
        open_ = still
    out: list[str] = []
    for value, places in open_:
        said = f"{value:.{places}f}" if places else f"{value:g}"
        if said not in out:
            out.append(said)
    return out


__all__ = ["COUNTED", "PASSES", "TOLERANCE", "numbers", "unbacked", "worked"]
