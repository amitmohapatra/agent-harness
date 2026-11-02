"""``ReAct``'s check of an answer's figures before it is given: a figure the model worked out by
hand — a fitted line, a sum of many rows, a growth rate compounded — is computed with a tool
instead.

A small model reads figures well and computes many of them badly: a least-squares line or a sum
of a dozen rows worked out in its reply drifts, and nothing in the loop sees it (program-aided
answers — PAL, Program of Thoughts — and tool-checked ones — CRITIC — are more accurate for that
reason). LangChain's loop ends on a reply with no tool call; :class:`FigureCheck` reads that
reply first, deterministically and without a model call:

* every figure of the answer that is **held** — a figure of what the model was given (the
  instructions, the memory context, the history, the question, every tool result), as written
  or rounded to the answer's precision, in thousands or millions, or as a percentage — is
  traced;
* so is one worked from traced figures — each step grounded, never from figures that lean on
  it: one rounded (up, down, to a whole number); one operation (``+ - x /``, a share in %, a %
  change) on a traced figure and a held one; two operations on three traced figures — the
  working a model shows ("0.214 a day x 39 days = 8.36, less 5 on hand: 3.36, so 4"); a total of
  one short result's rows, or of a list of traced figures the answer gives; one operation on
  two figures of one record (a price less a cost);
* any other figure was computed from many figures at once, and not by a tool: the model is told
  which, and asked once — offered ``query_result`` over this turn's results, each kept as a table
  (``tools.results.keep``), or the run's code tool — to compute it with a tool and answer again.

A turn is checked once: the second answer is the answer. Most answers hold no such figure:
replayed on the break-and-fix gate's recorded ``ReAct`` turns, it asks again on 1 of 336 with
memory on and 3 of 168 with it off, so most turns cost nothing more.
The draft and the note go only to that one model call; the conversation the graph keeps holds
the answer alone.
"""

from __future__ import annotations

import bisect
import itertools
import math
import re
import weakref
from collections.abc import Awaitable, Callable, Iterable, Iterator, Sequence
from typing import Any, Final

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage

from trellis.harness.runtime import Runtime, current
from trellis.harness.tools import query, results
from trellis.harness.tools.convert import text_of

#: The minus sign (U+2212), as a model often writes a negative.
MINUS: Final = "\u2212"
#: A figure as written: an optional sign (the minus sign too), digits with thousands
#: separators or without, a fraction — not part of a word ("SKU-00602"'s id is its digits).
_FIGURE: Final = re.compile(
    r"(?<![\w.])[-" + MINUS + r"]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?![\d])"
)
#: What is not a figure: a date, a month (``2026-07``), a time, a year written as one (four
#: digits, no separator).
_NOT_FIGURES: Final = re.compile(
    r"\b\d{4}-\d{2}(?:-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?\S*)?)?\b|\b\d{1,2}:\d{2}\b"
    r"|(?<![\d,.])(?:19|20)\d{2}(?!\d|,\d{3}|\.\d)"
)
#: A percentage as a question gives it ("12%", "5 per cent").
_PERCENT: Final = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s?(?:%|per ?cent)")
#: Figures every answer may use besides what it states, exact: a whole, a percentage's 100.
_CONSTANTS: Final = ((1.0, 0.0), (100.0, 0.0))
#: A whole number up to this is a count, a rank, a list's number, a percentage the question
#: names: not checked (as the suite's scorer reads made-up figures).
SMALL: Final = 100
#: The most traced figures the two-step search runs over: past it (an answer restating a
#: table) the pairs of them are dense enough to trace any figure by chance, so a figure
#: one step from none of them is not traced.
MOST_STATED: Final = 12
#: The most figures a text may hold for the runs of them to be totalled (a result's rows, a
#: list), and for two of them to be worked into one (a record: a price and a cost).
RUN: Final = 50
RECORD: Final = 12
#: The most a figure worked from two stated ones may be off, as a share of it, to trace
#: another: past it (a small difference of rounded figures) it could explain anything.
PRECISE: Final = 0.01
#: How far apart two figures may be and still be one (a float's own error).
EPSILON: Final = 1e-9
#: The most figures a note names.
NAMED: Final = 5

NOTE: Final = (
    "Before you answer: {figures} in your answer {verb} not in any tool result, nor a step or "
    "two of arithmetic from the figures you state, so {subject} {was} worked out by hand. "
    "Compute {pronoun} with a tool instead: {how}. Then give your answer again. If a tool "
    "result does hold {pronoun}, say which."
)
#: How the model computes, by what the run has.
QUERY_HOW: Final = (
    "query_result runs SQL over a saved result's rows (sum, avg, count, power, ln, exp, "
    "regr_slope(y, x), regr_intercept(y, x), regr_r2(y, x), corr(y, x), window functions); "
    "this turn's results are saved: {tables}"
)
CODE_HOW: Final = "run the computation as code with {tool}"


# --------------------------------------------------------------------------- figures
#: A figure: its value and half a unit of its last written place (its precision).
Figure = tuple[float, float]


def figures(text: str) -> list[Figure]:
    """Every figure ``text`` states, each with half a unit of its last written place (of its
    last digit that is not a zero, for a whole number): ``"£1,234.5"`` is ``(1234.5, 0.05)``,
    ``"7,800"`` ``(7800.0, 50.0)``. Dates, times and years are not figures."""
    out = []
    for m in _FIGURE.finditer(_NOT_FIGURES.sub(" ", text).replace(MINUS, "-")):
        written = m.group(0).replace(",", "")
        value = float(written)
        if math.isfinite(value):
            whole, _, fraction = written.partition(".")
            # "7,800" or "£2,000" said of 7,810.73: a whole number ending in zeros is rounded to
            # its last digit that is not one
            zeros = len(whole) - len(whole.rstrip("0")) if not fraction and value else 0
            out.append((value, 0.5 * 10.0 ** (zeros - len(fraction))))
    return out


def _within(given: Sequence[float], at: float, spread: float) -> list[int]:
    """Where the figures of ``given`` (sorted) within ``spread`` of ``at`` are."""
    lo = bisect.bisect_left(given, at - spread)
    hi = bisect.bisect_right(given, at + spread, lo=lo)
    return list(range(lo, hi))


def _held(value: float, half: float, given: Sequence[float]) -> bool:
    """Whether ``value`` is a figure of ``given`` (sorted) rounded to ``value``'s precision
    (``half``) — as it is, in thousands or millions, or a fraction said as a percentage."""
    return any(
        _within(given, value * scale, half * scale + EPSILON * abs(value * scale))
        for scale in (1.0, 1e3, 1e6, 0.01)
    )


def _inverses(target: float, spread: float, a: float) -> Iterator[tuple[float, float]]:
    """Each ``x`` (and how far it may be off) for which ``a`` and ``x`` make ``target`` (to
    within ``spread``) by one operation, either way round: ``+ - x /``, a share in % and a %
    change."""
    yield target - a, spread  # a + x
    yield a - target, spread  # a - x
    yield target + a, spread  # x - a
    if a:
        yield target / a, spread / abs(a)  # a * x
        yield target * a, spread * abs(a)  # x / a
        yield target * a / 100, abs(a) * spread / 100  # x as % of a
        yield a * (1 + target / 100), abs(a) * spread / 100  # x, a % change from a
    if target:
        yield a / target, abs(a) * spread / (target * target)  # a / x
        yield 100 * a / target, 100 * abs(a) * spread / (target * target)  # a as % of x
    grown = 1 + target / 100
    if grown:
        yield a / grown, abs(a) * spread / (100 * grown * grown)  # a, a % change from x


def _made(a: Figure, b: Figure) -> Iterator[Figure]:
    """What ``a`` and ``b`` make by one operation, each with how far off the operands'
    precision may leave it."""
    (x, hx), (y, hy) = a, b
    yield x + y, hx + hy
    yield x - y, hx + hy
    yield x * y, abs(x) * hy + abs(y) * hx
    if y:
        yield x / y, (hx + abs(x / y) * hy) / abs(y)
        yield 100 * (x / y - 1), 100 * (hx + abs(x / y) * hy) / abs(y)


def _spread(value: float, half: float, a: Figure) -> float:
    """How far a figure derived from ``a`` may land from ``value``: ``value``'s own precision,
    and ``a``'s carried through the operation (at most by ``(|value| + 100) / |a|``: a product,
    a ratio, a % change)."""
    reach = max(1.0, (abs(value) + 100) / abs(a[0])) if a[0] else 1.0
    return half + a[1] * reach + EPSILON * abs(value)


def _derived(
    value: float,
    half: float,
    stated: Sequence[Figure],
    held: Sequence[float],
    factors: Sequence[Figure] = (),
) -> bool:
    """Whether ``value`` (said to ``half``) is worked from the ``stated`` figures (those traced
    so far): one of them rounded (up or down, to a whole number too), one operation on one of
    them (or of ``factors``: a what-if's 0.88) and a figure of ``held`` (sorted: what the
    model was given) or of ``stated``, or two operations on three stated figures."""
    whole = value == int(value)
    if any(abs(v - value) <= h + half or (whole and abs(v - value) < 1) for v, h in stated):
        return True
    pool = sorted({*held, *(v for v, _ in stated)})
    for a in [*stated, *factors]:
        spread = _spread(value, half, a)
        for x, off in _inverses(value, spread, a[0]):
            if _within(pool, x, off):
                return True
    operands = [f for f in stated if f not in _CONSTANTS]  # 1 and 100 convert, never add
    if len(operands) > MOST_STATED:
        return False
    made = sorted(
        (w, off)
        for i, a in enumerate(operands)
        for j, b in enumerate(operands)
        if i != j
        for w, off in _made(a, b)
        if w and off <= PRECISE * abs(w)  # a figure that vague traces nothing
    )
    values = [w for w, _ in made]
    widest = max((h for _, h in made), default=0.0)
    for c in stated:
        spread = _spread(value, half, c)
        for u, off in _inverses(value, spread, c[0]):
            for i in _within(values, u, off + widest):
                if abs(values[i] - u) <= off + made[i][1]:
                    return True
    return False


def untraced(answer: str, given: Iterable[str], question: str = "") -> list[str]:
    """The figures of ``answer`` (as written) not traced to what the model was given: neither
    held — a figure of ``given`` (everything the model was told) or of ``question`` — nor
    worked from traced figures (:func:`_derived`), each step grounded: a figure is traced only
    from figures traced before it, never from ones that lean on it (a line's slope "checked"
    by the forecasts made from it). What is left was computed from many figures at once, and
    not by a tool. Whole numbers up to :data:`SMALL` are counts and are not checked."""
    said = figures(answer)
    asked = figures(question)
    held = sorted({v for text in given for v, _ in figures(text)} | {v for v, _ in asked})
    # an operand written whole is taken as exact (a count, a month, a fee, a rounded-up
    # quantity), one with a fraction as rounded to its places
    traced = [(v, 0.0 if v == int(v) else h) for v, h in asked] + list(_CONSTANTS)
    # a percentage the question gives, as the factor a what-if multiplies by (12% lower: 0.88):
    # one step's operand, never the two steps' (it would only add chances)
    shares = [float(p) for p in _PERCENT.findall(question)]
    factors = [(w, 0.0) for v in shares for w in (v / 100, 1 + v / 100, 1 - v / 100) if w]
    pending: list[Figure] = []
    for value, half in said:
        if (value == int(value) and abs(value) <= SMALL) or _held(value, half, held):
            traced.append((value, 0.0 if value == int(value) else half))
        else:
            pending.append((value, half))
    if pending:  # what one result holds: a total of its rows, a record's margin
        worked = _worked(figures(text) for text in given)
        found = [(v, h) for v, h in pending if _within(worked, v, h + EPSILON * abs(v))]
        pending = [f for f in pending if f not in found]
        traced.extend((v, 0.0 if v == int(v) else h) for v, h in found)
    grew = True
    while pending and grew:
        grew = False
        # a total of figures the answer lists, each traced already
        listed = _worked(_listed(said, {v for v, _ in traced}), pairs=False)
        for f in list(pending):
            if _within(listed, f[0], f[1] + EPSILON * abs(f[0])) or _derived(
                f[0], f[1], traced, held, factors
            ):
                pending.remove(f)
                traced.append((f[0], 0.0 if f[0] == int(f[0]) else f[1]))
                grew = True
    out: list[str] = []
    for value, half in pending:
        shown = f"{value:,.{_places(half)}f}"
        if shown not in out:
            out.append(shown)
    return out


def _listed(said: Sequence[Figure], traced: set[float]) -> list[list[Figure]]:
    """The answer's figures in the order written, in runs of figures traced already (one not
    traced yet ends a run)."""
    runs: list[list[Figure]] = [[]]
    for f in said:
        if f[0] in traced:
            runs[-1].append(f)
        elif runs[-1]:
            runs.append([])
    return runs


def _worked(texts: Iterable[Sequence[Figure]], *, pairs: bool = True) -> list[float]:
    """What the figures of one short text make (sorted), a text at a time: the sum of every
    run of figures written one after another in a text of at most :data:`RUN` (a total over a
    result's rows, or over figures an answer lists), and one operation on two figures of a
    text of at most :data:`RECORD` (a record's margin: its price less its cost). A longer text
    is a table: read for its figures, computed over with a tool (``query_result`` keeps one of
    ``tools.results.TABLE_ROWS`` rows), never worked here: its many figures would trace
    anything."""
    out: list[float] = []
    for found in texts:
        values = [v for v, _ in found]
        if len(values) > RUN:
            continue
        for i, start in enumerate(values):
            total = start
            for v in values[i + 1 :]:
                total += v
                out.append(total)
        if pairs and len(values) <= RECORD:
            out.extend(w for a in found for b in found if a is not b for w, _ in _made(a, b))
    out.sort()
    return out


def _places(half: float) -> int:
    return max(0, -math.floor(math.log10(2 * half) + 1e-9))


# --------------------------------------------------------------------------- the check
class FigureCheck(AgentMiddleware):
    """``ReAct``'s check of an answer's figures before it is given (this module): an answer
    holding a figure the model worked out by hand (:func:`untraced`) is asked once more, told
    which figures and offered a tool that computes them — ``query_result`` over this turn's
    results, each kept as a table, or the run's code tool. A run with neither, an answer that
    is a structured response, and a turn already checked are left as they are. Outside a
    harness run it steps aside."""

    def __init__(self) -> None:
        super().__init__()
        #: each run's turns already checked (by their question's id, else its text)
        self._checked: weakref.WeakKeyDictionary[Runtime, set[str]] = weakref.WeakKeyDictionary()

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        reply = await handler(request)
        runtime = current()
        draft = next((m for m in reply.result if isinstance(m, AIMessage)), None)
        if (
            runtime is None
            or draft is None
            or draft.tool_calls
            or not draft.text
            or request.response_format is not None
        ):
            return reply
        asked = next((m for m in reversed(request.messages) if isinstance(m, HumanMessage)), None)
        checked = self._checked.setdefault(runtime, set())
        turn = (asked.id or asked.text) if asked is not None else ""
        if asked is None or turn in checked:
            return reply
        given = [text_of(m.content) for m in request.messages if m is not asked]
        found = untraced(draft.text, given, asked.text)
        how = _how(runtime, request.messages) if found else None
        if how is None:
            return reply
        checked.add(turn)
        note = _note(found, how)
        runtime.events.warning("figure_check", f"asked to compute {', '.join(found)} with a tool")
        return await handler(
            request.override(messages=[*request.messages, draft, HumanMessage(content=note)])
        )


def _how(runtime: Runtime, messages: Sequence[AnyMessage]) -> str | None:
    """How the model can compute in this run: ``query_result`` over this turn's results that
    are tables with a column of numbers, each kept (or results kept already), or the run's code
    tool; ``None`` when neither."""
    kept = []
    for m in _since_question(messages):
        said = text_of(m.content)
        columns, rows = query.rows_of(said) if len(said) <= results.MAX_CHARS else (["line"], [])
        if len(rows) >= 2 and columns != ["line"] and _numeric(rows):
            kept.append(results.keep(said, tool=m.name))
    tables = query.tables(runtime)
    if tables:
        shown = [p for p in tables if p in kept] or list(tables)
        named = ", ".join(f"{tables[p]} ({p})" for p in shown[:NAMED])
        return QUERY_HOW.format(tables=named)
    code = results.coding_tool()
    return CODE_HOW.format(tool=code) if code is not None else None


def _numeric(rows: Sequence[Sequence[Any]]) -> bool:
    """Whether a table has a column of numbers (a figure to compute over: not a list of notes,
    ids and dates)."""
    return any(all(_number(row[c]) for row in rows) for c in range(len(rows[0])))


def _number(cell: Any) -> bool:
    """Whether a cell is a number: a JSON one, or a CSV one's text (a JSON ``true`` is its
    text, ``query.rows_of``: not one)."""
    try:
        float(cell)
    except (TypeError, ValueError):
        return False
    return True


def _since_question(messages: Sequence[AnyMessage]) -> Iterator[ToolMessage]:
    """This turn's tool results that succeeded (after the latest user message)."""
    turn = itertools.takewhile(lambda m: not isinstance(m, HumanMessage), reversed(messages))
    return (m for m in turn if isinstance(m, ToolMessage) and m.status != "error")


def _note(found: Sequence[str], how: str) -> str:
    shown = found[:NAMED]
    one = len(shown) == 1
    return NOTE.format(
        figures=", ".join(shown),
        verb="is" if one else "are",
        pronoun="it" if one else "them",
        subject="it" if one else "they",
        was="was" if one else "were",
        how=how,
    )
