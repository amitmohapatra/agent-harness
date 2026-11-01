"""A tool that returns its rows a page at a time, fetched to completion: one call of it — as
the model makes it — runs the tool page after page, each next page asked with the cursor the
last one gave, and the model is given every row as one result (a large one is kept and
queried like any: ``tools.results``, ``tools.query``) — so a count, a sum or a join is over
all of the rows, never over the first page.

Only for a tool that says it pages (``PAGED``: ``tool(paged=True)``, a LangChain tool's
``metadata={"paged": True}``, an MCP tool's ``_meta``): asking the next page of any tool with a
cursor argument could cost far more than the model asked for. The conventions read: the cursor
argument is the first of :data:`CURSOR_ARGS` its schema has; a page is a JSON object (or JSON
text of one) whose next cursor is the first of :data:`NEXT_KEYS` it holds (none, empty or null:
the last page), and whose rows are its first list (:data:`ROW_KEYS` first). At most
:data:`MAX_PAGES` pages are fetched (the result says when there were more); a call the model
gives a cursor itself is run as it is (it asked for that page). The pages are one call of the
tool (``tools.base.execute``): within its timeout and the run's time, retried as one, and
governed, journaled and recorded as one.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Final

from trellis.contracts import ToolSpec

#: A tool's arguments that take the cursor of the page to fetch, and a page's keys holding the
#: next one.
CURSOR_ARGS: Final = ("cursor", "page_token", "pageToken", "next_cursor", "after", "page")
NEXT_KEYS: Final = ("next_cursor", "nextCursor", "next_page_token", "nextPageToken", "next_page")
#: The keys a page's rows are under, looked for first (else its first list).
ROW_KEYS: Final = ("items", "data", "results", "records", "rows")
#: The most pages one call fetches.
MAX_PAGES: Final = 200
#: What the merged result says of a tool that had more pages than :data:`MAX_PAGES`.
MORE: Final = "more_pages"


def cursor_arg(spec: ToolSpec) -> str | None:
    """The argument a paged tool takes its cursor in (``None``: the tool does not page)."""
    if not (spec.model_extra or {}).get("paged"):
        return None
    properties = (spec.input_schema or {}).get("properties") or {}
    return next((a for a in CURSOR_ARGS if a in properties), None)


async def every_page(
    spec: ToolSpec, args: dict[str, Any], run: Callable[[dict[str, Any]], Awaitable[Any]]
) -> Any:
    """``run`` (the tool, once) for ``args``, and — the tool paged (:func:`cursor_arg`), no
    cursor given — for each next page, the pages' rows merged into the first page's object (its
    next cursor gone, ``pages`` how many; :data:`MORE` when the last page fetched still had
    one). A page that is not one ends it, as it is; a page that fails fails the call."""
    arg = cursor_arg(spec)
    output = await run(args)
    if arg is None or args.get(arg) not in (None, ""):
        return output
    first = _page(output)
    if first is None:
        return output
    key, rows = first
    merged = dict(_object(output) or {})
    after = _next(merged)
    pages = 1
    while after is not None and pages < MAX_PAGES:
        output = await run({**args, arg: after})
        found = _page(output)
        if found is None:
            return output
        rows.extend(found[1])
        after = _next(_object(output) or {})
        pages += 1
    for name in NEXT_KEYS:
        merged.pop(name, None)
    merged[key] = rows
    merged["pages"] = pages
    if after is not None:
        merged[MORE] = True
    return merged


def _object(output: Any) -> dict[str, Any] | None:
    """A page as a JSON object: as it is, or parsed from its text."""
    if isinstance(output, str):
        try:
            output = json.loads(output)
        except ValueError:
            return None
    return output if isinstance(output, dict) else None


def _page(output: Any) -> tuple[str, list[Any]] | None:
    """A page's rows, and the key they are under; ``None``: not a page."""
    found = _object(output)
    if found is None:
        return None
    key = next((k for k in ROW_KEYS if isinstance(found.get(k), list)), None)
    key = key or next((k for k, v in found.items() if isinstance(v, list)), None)
    return None if key is None else (key, list(found[key]))


def _next(page: Mapping[str, Any]) -> Any:
    """A page's next cursor (``None``: it is the last)."""
    after = next((page[k] for k in NEXT_KEYS if k in page), None)
    return None if after in (None, "", False) else after
