"""Large tool results where the framework has no cut of its own (OpenAI Agents, Claude): the
convention Deep Agents' ``FilesystemMiddleware`` gives ``ReAct`` and Deep Agents, so a model
reads one convention everywhere — without importing Deep Agents (an optional extra).

A result over :data:`MAX_CHARS` (Deep Agents' 20,000 tokens, at 4 characters each) is kept
whole in the run's journal (``Journal.results``: it stays readable for the rest of the run,
across a pause and on another worker), and the model reads Deep Agents' stub in its place: the
path the result was saved at (``/large_tool_results/<id>``, the id naming its content) and its
first and last lines, numbered. ``read_file`` (Deep Agents' contract: ``file_path``,
``offset`` the line to start from, 0-indexed, ``limit`` the most lines, 100 by default) pages
it, at most :data:`PAGE_CHARS` at a time, each page saying where the next starts. A line
longer than :data:`MAX_LINE` characters reads as several (Deep Agents' ``read_file`` numbers
their parts 5.1, 5.2… in one page): a one-line JSON result pages too.
"""

from __future__ import annotations

from typing import Any, Final

from trellis.harness.journal import content_key
from trellis.harness.runtime import current
from trellis.harness.tools.base import arguments_problem, not_run
from trellis.harness.tools.convert import text_of

#: The most characters a result is read as it is; a longer one is kept and previewed.
MAX_CHARS: Final = 20_000 * 4
#: Where a kept result is read (Deep Agents' ``/large_tool_results/<call id>``).
PREFIX: Final = "/large_tool_results"
#: The longest line read as one, and the most of a line a preview shows.
MAX_LINE: Final = 5_000
PREVIEW_LINE: Final = 1_000
#: The lines a preview shows from the head and from the tail.
HEAD: Final = 5
TAIL: Final = 5
#: The tool that pages a kept result, the lines it reads by default, and the most characters
#: one page holds (as ``read_result`` reads).
READ_FILE: Final = "read_file"
READ_LINES: Final = 100
PAGE_CHARS: Final = 20_000
READ_FILE_DESCRIPTION: Final = (
    "Reads a large tool result that was saved to a file: the tool message gives its path "
    f"(under {PREFIX}/) and a preview. By default it reads up to {READ_LINES} lines from the "
    "start; use offset (the line to start from, 0-indexed) and limit (the most lines) to page "
    "through it instead of reading it whole."
)
READ_FILE_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "file_path": {"type": "string", "description": "The path the tool message gives."},
        "offset": {"type": "integer", "description": "The line to start from (0-indexed)."},
        "limit": {"type": "integer", "description": "The most lines to read."},
    },
    "required": ["file_path"],
    "additionalProperties": False,
}

#: Deep Agents' stub for a large result (``_message_eviction._TOO_LARGE_TOOL_MSG``).
STUB: Final = """Tool result too large, the result of this tool call {tool_call_id} was saved in the filesystem at this path: {file_path}

You can read the result from the filesystem by using the read_file tool, but make sure to only read part of the result at a time.

You can do this by specifying an offset and limit in the read_file tool call. For example, to read the first 100 lines, you can use the read_file tool with offset=0 and limit=100.

{preview_note}

{content_sample}
"""  # noqa: E501 - Deep Agents' text, as it is
MARKER: Final = "... [{omitted_lines} lines truncated] ..."
OMITTED: Final = (
    f"lines of the form `{MARKER.format(omitted_lines='N')}` indicate omitted lines in the "
    "middle of the content"
)
CLIPPED: Final = (
    f"the output contains lines longer than {PREVIEW_LINE} characters; this preview shows only "
    f"their first {PREVIEW_LINE} characters"
)


def readable(output: Any) -> str:
    """A tool's result as the model of the current run reads it: its text, or — longer than
    :data:`MAX_CHARS` — the stub naming where it is kept, with a preview."""
    text = text_of(output)
    runtime = current()
    if len(text) <= MAX_CHARS or runtime is None:
        return text
    ident = content_key("result", text)
    path = f"{PREFIX}/{ident}"
    runtime.replay.journal.results[path] = text
    return _stub(ident, path, _lines(text))


def kept() -> bool:
    """Whether the current run keeps a result to read (``read_file`` is offered then)."""
    runtime = current()
    return runtime is not None and bool(runtime.replay.journal.results)


def read(args: dict[str, Any]) -> str:
    """``read_file``: a page of a kept result, as the model asked for it (``args``)."""
    problem = arguments_problem(READ_FILE_SCHEMA, args)
    if problem is not None:
        return not_run(READ_FILE, problem)
    return read_file(**args)


def read_file(file_path: str, offset: int = 0, limit: int = READ_LINES) -> str:
    """Lines ``offset`` on (at most ``limit``, and :data:`PAGE_CHARS`) of the result kept at
    ``file_path``, numbered, then where the next page starts."""
    runtime = current()
    path = "/" + file_path.lstrip("/")
    text = None if runtime is None else runtime.replay.journal.results.get(path)
    if text is None:
        return f"Error: File '{path}' not found"
    lines = _lines(text)
    start = max(0, offset)
    if start >= len(lines):
        return f"Error: Line offset {offset} exceeds file length ({len(lines)} lines)"
    last = min(len(lines), start + max(1, limit))
    width = len(str(last))
    shown: list[str] = []
    size = 0
    for number in range(start, last):
        row = f"{number + 1:>{width}}  {lines[number]}"
        if shown and size + len(row) > PAGE_CHARS:
            break
        shown.append(row)
        size += len(row) + 1
    end = start + len(shown)
    page = "\n".join(shown)
    if end < len(lines):
        page += f'\n…[{len(lines) - end} more lines: {READ_FILE}(file_path="{path}", offset={end})]'
    return page


def _lines(text: str) -> list[str]:
    """The lines a kept result is read by: a longer line than :data:`MAX_LINE`, in parts."""
    return [
        line[start : start + MAX_LINE]
        for line in text.splitlines()
        for start in range(0, max(1, len(line)), MAX_LINE)
    ]


def _stub(ident: str, path: str, lines: list[str]) -> str:
    """Deep Agents' stub: the path, and the head and tail lines, each at most
    :data:`PREVIEW_LINE` characters (a kept result has more lines than :data:`HEAD` and
    :data:`TAIL`: none is longer than :data:`MAX_LINE`)."""
    head, tail = lines[:HEAD], lines[-TAIL:]
    marker = MARKER.format(omitted_lines=len(lines) - HEAD - TAIL)
    sample = f"{_numbered(head, 1)}\n{marker}\n{_numbered(tail, len(lines) - TAIL + 1)}"
    clipped = any(len(line) > PREVIEW_LINE for line in (*head, *tail))
    caveats = "; ".join([OMITTED, CLIPPED] if clipped else [OMITTED])
    note = f"Here is a preview showing the head and tail of the result ({caveats}):"
    return STUB.format(tool_call_id=ident, file_path=path, preview_note=note, content_sample=sample)


def _numbered(lines: list[str], first: int) -> str:
    """Lines numbered from ``first``, as Deep Agents numbers them, each clipped for a preview."""
    width = len(str(first + len(lines) - 1))
    return "\n".join(f"{first + n:>{width}}  {line[:PREVIEW_LINE]}" for n, line in enumerate(lines))
