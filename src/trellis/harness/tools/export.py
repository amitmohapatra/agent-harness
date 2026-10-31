"""``export_table``: a table the user asked for as a file (CSV, or Excel with the ``xlsx`` extra),
kept in the file store (``Runtime.save_file``) and offered for download. It is offered to a run
that has no tool that runs code: a run that has one writes files with its code instead
(:data:`CODE_TOOLS`), so one purpose never has two tools.

The rows are the model's (``rows``: what it read or computed) or a result the run saved whole
(``file_path``: ``/large_tool_results/...``, read as ``query_result`` reads it, every row),
so a large table is exported without passing through the model's context.
"""

from __future__ import annotations

import csv
import functools
import importlib.util
import io
from typing import Any, Final

from trellis.contracts import ToolSpec
from trellis.harness.runtime import Runtime, current
from trellis.harness.tools import query
from trellis.harness.tools.base import Tool, arguments_problem, not_run

EXPORT_TABLE: Final = "export_table"
#: The tools that run code able to write a file, by the name each framework gives it: the
#: harness's sandbox (``sandbox_exec``), Deep Agents' ``execute`` on a sandbox backend, Claude
#: Code's ``Bash``, OpenAI's hosted code interpreter. Bifrost's Code Mode runs Starlark with no
#: file system: it computes, it writes no file.
CODE_TOOLS: Final = ("sandbox_exec", "execute", "Bash", "code_interpreter")
#: The formats, by a file name's extension; Excel only with openpyxl installed (the extra).
CSV: Final = "csv"
XLSX: Final = "xlsx"
#: The most rows one export holds (Excel's own limit, less its header).
MAX_ROWS: Final = 1_048_575


def formats() -> tuple[str, ...]:
    """The formats this process can write: CSV always, Excel with openpyxl."""
    return (CSV, XLSX) if importlib.util.find_spec("openpyxl") is not None else (CSV,)


def runs_code(runtime: Runtime) -> str | None:
    """The tool the run runs code with, when it has one (:data:`CODE_TOOLS`): its own toolbox's
    or its framework's."""
    return next((n for n in CODE_TOOLS if n in runtime.toolbox or n in runtime.native), None)


def offered(runtime: Runtime) -> Tool | None:
    """``export_table`` for a run with no tool that runs code; ``None`` for one with one."""
    return None if runs_code(runtime) is not None else _tool()


@functools.cache
def _tool() -> Tool:
    """The tool, made once per process (its formats are the process's): the same object every
    run, so a run's tool set is converted for its framework once."""
    kinds = formats()
    named = " or ".join(f".{k}" for k in kinds)
    schema = {
        "type": "object",
        "properties": {
            "file_name": {
                "type": "string",
                "description": f"The file's name, ending {named} (its format), e.g. "
                f"sales-2026.{kinds[-1]}",
            },
            "rows": {
                "type": "array",
                "items": {"type": "object"},
                "description": "The rows, each an object of column -> value",
            },
            "file_path": {
                "type": "string",
                "description": "Instead of rows: a saved result's path (/large_tool_results/...), "
                "exported whole",
            },
        },
        "required": ["file_name"],
        "additionalProperties": False,
    }
    spec = ToolSpec(
        name=EXPORT_TABLE,
        description=f"Save a table as a file ({named}) the user downloads. Other file kinds "
        "(PDF, charts, documents) cannot be made in this chat: no code runs here.",
        input_schema=schema,
        side_effects="write",
        idempotent=True,  # the same table under the same name has one effect
    )
    return Tool(spec, _export)


async def _export(args: dict[str, Any]) -> str:
    tool = _tool()
    problem = arguments_problem(tool.spec.input_schema, args)
    runtime = current()
    if problem is None and ("rows" in args) == ("file_path" in args):
        problem = "give either rows or file_path"
    if problem is not None or runtime is None:
        return not_run(EXPORT_TABLE, problem or "it runs inside a run")
    name = str(args["file_name"])
    kind = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if kind not in formats():
        return not_run(EXPORT_TABLE, f"the file name ends .{' or .'.join(formats())}")
    if "file_path" in args:
        path = "/" + str(args["file_path"]).lstrip("/")
        text = runtime.replay.journal.results.get(path)
        if text is None:
            return not_run(EXPORT_TABLE, f"no saved result at {path}")
        columns, rows = query.rows_of(text)
    else:
        given = [r for r in args["rows"] if isinstance(r, dict)]
        columns = list(dict.fromkeys(k for row in given for k in row))
        rows = [[row.get(c) for c in columns] for row in given]
    if len(rows) > MAX_ROWS:
        return not_run(EXPORT_TABLE, f"{len(rows)} rows: a file holds at most {MAX_ROWS}")
    data = _xlsx(columns, rows) if kind == XLSX else _csv(columns, rows)
    stored = await runtime.save_file(name, data, source=EXPORT_TABLE)
    return (
        f"Saved {stored.name}: {len(rows)} rows, {len(columns)} columns, {stored.size} bytes. "
        "The user downloads it from the link the chat shows next to your answer."
    )


def _csv(columns: list[str], rows: list[list[Any]]) -> bytes:
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(columns)
    writer.writerows(["" if v is None else v for v in row] for row in rows)
    return out.getvalue().encode()


def _xlsx(columns: list[str], rows: list[list[Any]]) -> bytes:
    from openpyxl import Workbook  # noqa: PLC0415 - the xlsx extra

    book = Workbook(write_only=True)  # streamed: a large table never held as cells
    sheet = book.create_sheet("Table")
    sheet.append(columns)
    for row in rows:
        sheet.append(
            [v if v is None or isinstance(v, int | float | str | bool) else str(v) for v in row]
        )
    out = io.BytesIO()
    book.save(out)
    return out.getvalue()


__all__ = ["CODE_TOOLS", "EXPORT_TABLE", "formats", "offered", "runs_code"]
