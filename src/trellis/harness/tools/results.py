"""Large tool results where the framework has no cut of its own (OpenAI Agents, Claude, Kimi,
``ReAct`` without a backend, a ``create_agent`` graph with ``middleware()``): the convention
Deep Agents' ``FilesystemMiddleware`` gives ``ReAct`` with a backend and Deep Agents, so a
model reads one convention everywhere — without importing Deep Agents (an optional extra).

A result over :data:`MAX_CHARS` (Deep Agents' 20,000 tokens, at 4 characters each) is kept
whole in the run's journal (``Journal.results``: it stays readable for the rest of the run,
across a pause and on another worker), and the model reads Deep Agents' stub in its place: the
path the result was saved at (``/large_tool_results/<id>``, the id naming its content) and its
first and last lines, numbered. ``read_file`` (Deep Agents' contract: ``file_path``,
``offset`` the line to start from, 0-indexed, ``limit`` the most lines, 100 by default) pages
it, at most :data:`PAGE_CHARS` at a time, each page saying where the next starts. A line
longer than :data:`MAX_LINE` characters reads as several (Deep Agents' ``read_file`` numbers
their parts 5.1, 5.2… in one page): a one-line JSON result pages too.

``query_result`` (``tools.query``) runs read-only SQL over its rows — a count, a sum, a filter,
a top-N or a group over every row, exact — and the stub says so before its preview, naming the
table's columns; a run with a sandbox is also told the result is a file there to run code over
(:data:`CODE`: the sandbox writes every kept result in, ``SandboxSource.with_results``).
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from trellis.harness.history import Said
from trellis.harness.journal import content_key
from trellis.harness.runtime import current
from trellis.harness.tools import query
from trellis.harness.tools.base import arguments_problem, not_run
from trellis.harness.tools.convert import text_of

#: The most characters a result is read as it is; a longer one is kept and previewed.
MAX_CHARS: Final = 20_000 * 4
#: The rows from which a table read whole is also kept, ``query_result`` computing over it
#: (:func:`tabled`): a small model counts, adds up and ranks a few dozen rows by hand wrongly
#: (the holistic run: 748 distinct SKUs counted where there were 740, a 237-row join summed
#: £18,533 off), and the query is exact.
TABLE_ROWS: Final = 30
#: What a table read whole says after its rows (then :data:`QUERY`).
TABLED: Final = "These {rows} rows are also saved at {path}. "
#: Where a kept result is read (Deep Agents' ``/large_tool_results/<call id>``).
PREFIX: Final = query.KEPT_PREFIX.rstrip("/")
#: The most characters of results the conversations' earlier turns saved that an agent keeps
#: in this process for their later turns (:class:`Conversations`: what one query may load).
KEPT_CHARS: Final = query.LOAD_CHARS
#: The longest line read as one, and the most of a line a preview shows.
MAX_LINE: Final = 5_000
PREVIEW_LINE: Final = 1_000
#: The lines a preview shows from the head and from the tail.
HEAD: Final = 5
TAIL: Final = 5
#: The tool that pages a kept result, the lines it reads by default, and the most characters
#: one page holds — a kept result's and a cleared one's alike.
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

#: How a result Deep Agents (or :func:`readable`) saved to a file reads in its place: its stub.
EVICTED: Final = "Tool result too large"
#: What a stub says first: the kept rows are a table that ``query_result`` computes over
#: (``tools.query``) — a count, a sum, a filter, a top-N or a group, exact, with no sandbox.
QUERY: Final = (
    "Its rows are the table {table} (columns: {columns}). For a count, a sum, an average, a "
    'filter, a ranking or a group over them, call query_result(file_path="{path}", '
    'sql="SELECT ... FROM {table} ..."): it computes over every row, exactly. Every result '
    "this run saved is a table of the same query, named by its tool, so one query can JOIN "
    "them. Do not add up or rank rows yourself. {private}"
)
#: What a kept result's note says of its path: the run's own working copy, which the user
#: cannot reach (a model offered one as an export to open in Excel).
PRIVATE: Final = (
    "The path is this run's own working copy, for read_file and query_result: the user cannot "
    "open or download it, so never offer it to them as a file."
)
#: Deep Agents' tool that runs a command on its backend (a sandbox's: ``ReAct(backend=sandbox())``).
EXECUTE: Final = "execute"
#: What a stub adds for a run with a sandbox: the kept result is a file there, which code reads
#: whole — a count, a sum, a filter, a top-N or a group over every row, without paging it.
CODE: Final = (
    "The whole result is also the file {path} in your sandbox: to count, sum, filter, rank or "
    "group its rows, run code over that file with {tool} instead of reading it page by page."
)
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


def readable(output: Any, tool: str | None = None) -> str:
    """A tool's result (``tool``'s, when named) as the model of the current run reads it: its
    text, or — longer than :data:`MAX_CHARS` — the stub naming where it is kept, with a
    preview."""
    text = text_of(output)
    if len(text) <= MAX_CHARS:
        return tabled(text, tool)
    path = keep(text, tool=tool)
    runtime = current()
    if path is None or runtime is None:
        return text
    columns = ", ".join(query.columns_of(text))
    name = query.tables(runtime)[path]
    table = QUERY.format(table=name, columns=columns, path=path, private=PRIVATE)
    return _stub(path.rsplit("/", 1)[1], path, _lines(text), code=coding_tool(), table=table)


def tabled(text: str, tool: str | None = None) -> str:
    """``text`` read whole, as it is — and, when it is a table of :data:`TABLE_ROWS` rows or
    more (``tools.query.rows_of``: CSV with a header, or JSON objects), kept as well, followed
    by where and how ``query_result`` computes over its rows, so a count, a sum or a ranking is
    exact rather than done by hand. Parsed only when long enough to hold that many rows; a text
    over :data:`MAX_CHARS` is the cut's (:func:`readable`, or the framework's own), not this."""
    runtime = current()
    if runtime is None or not TABLE_ROWS * MIN_ROW_CHARS <= len(text) <= MAX_CHARS:
        return text
    columns, rows = query.rows_of(text)
    if len(rows) < TABLE_ROWS or columns == ["line"]:
        return text
    path = keep(text, tool=tool)
    assert path is not None  # in a run
    name = query.tables(runtime)[path]
    table = QUERY.format(
        table=name, columns=", ".join(query.column_names(columns)), path=path, private=PRIVATE
    )
    return f"{text}\n\n{TABLED.format(rows=len(rows), path=path)}{table}"


#: The fewest characters a table's row takes (``a,1`` and its line break): a shorter text
#: cannot hold :data:`TABLE_ROWS` rows, and is not parsed.
MIN_ROW_CHARS: Final = 4


def keep(text: str, *, tool: str | None = None) -> str | None:
    """``text`` kept whole in the current run's journal, where ``read_file`` pages it (and the
    run's sandbox finds it: ``sandbox.SandboxSource``), and the path it is kept at — named by
    its content, so the same text is kept once; ``None`` outside a run. ``tool``: the tool that
    returned it, whose name its table has in a query (``tools.query.tables``)."""
    runtime = current()
    if runtime is None:
        return None
    path = f"{PREFIX}/{content_key('result', text)}"
    runtime.replay.journal.results[path] = text
    if tool is not None and getattr(runtime, "kept_from", None) is not None:
        runtime.kept_from.setdefault(path, tool)
    return path


def line_at(text: str, chars: int) -> int:
    """The line (0-indexed, as ``read_file`` numbers them: :func:`_lines`) that character
    ``chars`` of ``text`` is on — where reading on from the first ``chars`` characters starts."""
    head = text[:chars]
    if not head or head.endswith(("\n", "\r")):
        return len(_lines(head))
    return len(_lines(head)) - 1


def coding_tool() -> str | None:
    """The tool the current run runs code with in its sandbox — the sandbox's own tool
    (``sandbox().tools()``), else Deep Agents' ``execute`` on the run's sandbox — or ``None``:
    a kept result is a file there too, which code counts, sums, filters and ranks whole."""
    runtime = current()
    if runtime is None:
        return None
    from trellis.harness.sandbox import EXEC  # noqa: PLC0415 - the sandbox imports the tools

    if EXEC in runtime.toolbox:
        return EXEC
    return EXECUTE if EXECUTE in runtime.native else None


def kept() -> bool:
    """Whether the current run keeps a result to read (``read_file`` is offered then)."""
    runtime = current()
    return runtime is not None and bool(query.saved(runtime))


class Conversations:
    """What each conversation's turns left for its later turns, kept in this process
    (``Agent.results``) — as a framework's own thread keeps it (Deep Agents' files in its
    state, a checkpointer's messages): the large results they saved, and, where nothing else
    keeps the conversation (memory off), each turn's transcript — its question, tool calls,
    their results and its answer (``conversation.earlier``). The least recently used
    conversation is forgotten past :data:`KEPT_CHARS` in all. A turn on another worker, or
    after a restart, finds none: a path read then says so (``query.GONE``), and its history is
    each turn's question and answer from the run store."""

    def __init__(self, most: int = KEPT_CHARS) -> None:
        self.most = most
        self.chars = 0
        self._kept: OrderedDict[str, _Kept] = OrderedDict()

    def of(self, conversation: str) -> tuple[dict[str, str], dict[str, str]]:
        """The results ``conversation``'s earlier turns saved, by path, and the tool each came
        from (its table's name in a query)."""
        found = self._found(conversation)
        return (found.texts, found.tools) if found is not None else ({}, {})

    def turns(self, conversation: str) -> dict[str, list[Said]]:
        """The transcripts of ``conversation``'s earlier turns kept here, by run."""
        found = self._found(conversation)
        return found.turns if found is not None else {}

    def add(self, conversation: str, texts: Mapping[str, str], tools: Mapping[str, str]) -> None:
        """A turn's saved results (``texts``) and their tools, kept for ``conversation``."""
        kept = self._kept.setdefault(conversation, _Kept())
        self._kept.move_to_end(conversation)
        for path, text in texts.items():
            if path not in kept.texts:
                kept.texts[path] = text
                self.chars += len(text)
        kept.tools.update({p: t for p, t in tools.items() if p in kept.texts})
        self._bounded()

    def said(self, conversation: str, run_id: str, transcript: Sequence[Said]) -> None:
        """Run ``run_id``'s transcript, kept for ``conversation`` (an attempt's replaces the
        one before it)."""
        kept = self._kept.setdefault(conversation, _Kept())
        self._kept.move_to_end(conversation)
        self.chars -= _size(kept.turns.get(run_id, ()))
        kept.turns[run_id] = list(transcript)
        self.chars += _size(transcript)
        self._bounded()

    def _found(self, conversation: str) -> _Kept | None:
        found = self._kept.get(conversation)
        if found is not None:
            self._kept.move_to_end(conversation)
        return found

    def _bounded(self) -> None:
        while self.chars > self.most:
            _, gone = self._kept.popitem(last=False)
            self.chars -= sum(map(len, gone.texts.values())) + sum(map(_size, gone.turns.values()))


@dataclass
class _Kept:
    """One conversation's: its saved results, their tools, its turns' transcripts by run."""

    texts: dict[str, str] = field(default_factory=dict)
    tools: dict[str, str] = field(default_factory=dict)
    turns: dict[str, list[Said]] = field(default_factory=dict)


def _size(transcript: Iterable[Said]) -> int:
    return sum(len(s.content) for s in transcript)


def read(args: dict[str, Any], *, text: str | None = None) -> str:
    """``read_file``: a page of a kept result, as the model asked for it (``args``) — or of
    ``text``, a cleared result the graph's state keeps (:func:`read_file`)."""
    problem = arguments_problem(READ_FILE_SCHEMA, args)
    if problem is not None:
        return not_run(READ_FILE, problem)
    return read_file(**args, text=text)


def read_file(
    file_path: str, offset: int = 0, limit: int = READ_LINES, *, text: str | None = None
) -> str:
    """Lines ``offset`` on (at most ``limit``, and :data:`PAGE_CHARS`) of the result kept at
    ``file_path`` — or of ``text`` given for it (a result cleared from the request, which the
    graph's state keeps: ``middleware.HarnessTools``) — numbered, then where the next page
    starts. The one paging every read-back uses."""
    runtime = current()
    path = "/" + file_path.lstrip("/")
    if text is None and runtime is not None:
        text = query.saved(runtime).get(path)
    if text is None:
        return query.not_found(path)
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


def _stub(ident: str, path: str, lines: list[str], *, code: str | None, table: str) -> str:
    """Deep Agents' stub: the path, and the head and tail lines, each at most
    :data:`PREVIEW_LINE` characters (a kept result has more lines than :data:`HEAD` and
    :data:`TAIL`: none is longer than :data:`MAX_LINE`) — and, before the preview, its rows as
    the table ``query_result`` computes over (``table``: :data:`QUERY`) and, for a run with a
    sandbox (``code``: its tool), that the file is there to run code over (:data:`CODE`)."""
    head, tail = lines[:HEAD], lines[-TAIL:]
    marker = MARKER.format(omitted_lines=len(lines) - HEAD - TAIL)
    sample = f"{_numbered(head, 1)}\n{marker}\n{_numbered(tail, len(lines) - TAIL + 1)}"
    clipped = any(len(line) > PREVIEW_LINE for line in (*head, *tail))
    caveats = "; ".join([OMITTED, CLIPPED] if clipped else [OMITTED])
    note = f"Here is a preview showing the head and tail of the result ({caveats}):"
    if code is not None:
        note = f"{CODE.format(path=path, tool=code)}\n\n{note}"
    note = f"{table}\n\n{note}"
    return STUB.format(tool_call_id=ident, file_path=path, preview_note=note, content_sample=sample)


def _numbered(lines: list[str], first: int) -> str:
    """Lines numbered from ``first``, as Deep Agents numbers them, each clipped for a preview."""
    width = len(str(first + len(lines) - 1))
    return "\n".join(f"{first + n:>{width}}  {line[:PREVIEW_LINE]}" for n, line in enumerate(lines))


#: The tools that read a kept result — ``read_file`` (its pages) and ``query_result`` (SQL over
#: its rows: ``tools.query``) — as each adapter offers them: name, description, JSON schema, and
#: what answers a call (its arguments, as the model wrote them).
async def aread(args: dict[str, Any]) -> str:
    """:func:`read`, as a reader answers (``READERS``: each awaited)."""
    return read(args)


READERS: Final[
    tuple[tuple[str, str, dict[str, Any], Callable[[dict[str, Any]], Awaitable[str]]], ...]
] = (
    (READ_FILE, READ_FILE_DESCRIPTION, READ_FILE_SCHEMA, aread),
    (query.QUERY_RESULT, query.QUERY_RESULT_DESCRIPTION, query.QUERY_RESULT_SCHEMA, query.query),
)
