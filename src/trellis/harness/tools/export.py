"""A query's whole result as a file the user downloads (``query_result(save_as=...)``:
``tools.query``) — CSV with the standard library, Excel with the ``xlsx`` extra (openpyxl's
write-only workbook) — written as the rows come, and kept in the file store
(``Runtime.save_file``).

Saving is offered to a run with no tool that runs code (:func:`runs_code`, read from the run's
own tools, never configured): a run with one writes files with its code (a sandbox's
``outputs/``), so one purpose never has two ways.
"""

from __future__ import annotations

import csv
import functools
import importlib.util
import io
from collections.abc import Iterable, Sequence
from decimal import Decimal
from typing import Any, Final

from trellis.harness.runtime import Runtime

#: The tools that run code able to write a file, by the name each framework gives it: the
#: harness's sandbox (``sandbox_exec``), Deep Agents' ``execute`` on a sandbox backend, Claude
#: Code's ``Bash``, OpenAI's hosted code interpreter. Bifrost's Code Mode runs Starlark with no
#: file system: it computes, it writes no file.
CODE_TOOLS: Final = ("sandbox_exec", "execute", "Bash", "code_interpreter")
#: The formats, by a file name's extension; Excel only with openpyxl installed (the extra).
CSV: Final = "csv"
XLSX: Final = "xlsx"
#: The most rows one file holds (Excel's own limit, less its header).
MAX_ROWS: Final = 1_048_575


class TooManyRows(ValueError):
    """More rows than a file holds (:data:`MAX_ROWS`)."""


@functools.cache
def formats() -> tuple[str, ...]:
    """The formats this process writes: CSV always, Excel with openpyxl."""
    return (CSV, XLSX) if importlib.util.find_spec("openpyxl") is not None else (CSV,)


def runs_code(runtime: Runtime) -> str | None:
    """The tool the run runs code with, when it has one (:data:`CODE_TOOLS`): its own toolbox's
    or its framework's."""
    return next((n for n in CODE_TOOLS if n in runtime.toolbox or n in runtime.native), None)


def written(kind: str, columns: Sequence[str], rows: Iterable[Sequence[Any]]) -> tuple[bytes, int]:
    """``rows`` under a header of ``columns`` as a ``kind`` file (:func:`formats`), streamed —
    never held as cells — and how many rows it holds; :class:`TooManyRows` past
    :data:`MAX_ROWS`."""
    if kind == XLSX:
        from openpyxl import Workbook  # noqa: PLC0415 - the xlsx extra

        book = Workbook(write_only=True)
        sheet = book.create_sheet("Table")
        sheet.append(list(columns))
        count = 0
        try:
            for count, row in enumerate(rows, start=1):
                _under(count)
                sheet.append([_cell(v) for v in row])
        finally:
            sheet.close()  # its temporary file, whatever happens
        out = io.BytesIO()
        book.save(out)
        return out.getvalue(), count
    text = io.StringIO()
    writer = csv.writer(text, lineterminator="\n")
    writer.writerow(columns)
    count = 0
    for count, row in enumerate(rows, start=1):
        _under(count)
        writer.writerow([_text(v) for v in row])
    return text.getvalue().encode(), count


def _under(count: int) -> None:
    if count > MAX_ROWS:
        raise TooManyRows(f"more than {MAX_ROWS} rows: a file holds at most that many")


def _cell(value: Any) -> Any:
    """A value as an Excel cell takes it: a number (DuckDB's decimal too), text or a boolean as
    it is, else its text."""
    if isinstance(value, Decimal):
        return float(value)
    return value if value is None or isinstance(value, int | float | str | bool) else str(value)


def _text(value: Any) -> Any:
    """A value as a CSV cell writes it: a boolean as SQL and JSON write it (DuckDB's; SQLite's
    are 0 and 1), else as it is (a decimal exact, as the engine computed it)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


__all__ = [
    "CODE_TOOLS",
    "CSV",
    "MAX_ROWS",
    "XLSX",
    "TooManyRows",
    "formats",
    "runs_code",
    "written",
]
