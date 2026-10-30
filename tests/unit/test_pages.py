"""``tools.pages``: a tool that says it pages is fetched to its last page in one call."""

from __future__ import annotations

import json
from typing import Any

import pytest

from trellis import tool
from trellis.contracts import ToolSpec
from trellis.harness.tools import pages
from trellis.harness.tools.base import PAGED, data_of

SCHEMA = {"properties": {"cursor": {"type": "string"}, "q": {"type": "string"}}}


def paged(**more: Any) -> ToolSpec:
    return ToolSpec(name="t", input_schema=SCHEMA, **{PAGED: True}, **more)  # type: ignore[arg-type]


def book(n: int, size: int = 2, *, text: bool = False, key: str = "items") -> Any:
    """A tool answering ``n`` pages of ``size`` rows, and the cursors it was asked for."""
    asked: list[Any] = []

    async def run(args: dict[str, Any]) -> Any:
        asked.append(args.get("cursor"))
        at = int(args.get("cursor") or 0)
        rows = [{"n": at * size + i} for i in range(size)]
        page = {"total": n * size, key: rows, "next_cursor": str(at + 1) if at + 1 < n else None}
        return json.dumps(page) if text else page

    run.asked = asked  # type: ignore[attr-defined]
    return run


async def test_every_page_is_fetched_and_merged_into_one_result() -> None:
    run = book(3)
    merged = await pages.every_page(paged(), {"q": "x"}, run)
    assert merged == {"total": 6, "items": [{"n": i} for i in range(6)], "pages": 3}
    assert run.asked == [None, "1", "2"]


async def test_pages_as_json_text_with_rows_under_any_list_key() -> None:
    merged = await pages.every_page(paged(), {}, book(2, text=True, key="skus"))
    assert merged["skus"] == [{"n": i} for i in range(4)] and merged["pages"] == 2


async def test_a_tool_that_does_not_page_or_was_given_a_cursor_runs_once() -> None:
    run = book(3)
    assert (await pages.every_page(paged(), {"cursor": "1"}, run))["next_cursor"] == "2"
    plain = ToolSpec(name="t", input_schema=SCHEMA)
    assert (await pages.every_page(plain, {}, run))["next_cursor"] == "1"
    no_arg = ToolSpec(name="t", input_schema={"properties": {}}, **{PAGED: True})  # type: ignore[arg-type]
    assert pages.cursor_arg(no_arg) is None and pages.cursor_arg(paged()) == "cursor"


@pytest.mark.parametrize("first", ["not json", "[1, 2]", {"no": "rows"}])
async def test_an_answer_that_is_not_a_page_is_returned_as_it_is(first: Any) -> None:
    async def run(args: dict[str, Any]) -> Any:
        return first

    assert await pages.every_page(paged(), {}, run) == first


async def test_a_later_answer_that_is_not_a_page_ends_it_as_it_is() -> None:
    answers: list[Any] = [{"items": [1], "next_cursor": "a"}, "the service is down"]

    async def run(args: dict[str, Any]) -> Any:
        return answers.pop(0)

    assert await pages.every_page(paged(), {}, run) == "the service is down"


async def test_at_most_max_pages_are_fetched_and_the_result_says_there_were_more(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pages, "MAX_PAGES", 2)
    merged = await pages.every_page(paged(), {}, book(5))
    assert merged["pages"] == 2 and merged[pages.MORE] is True and len(merged["items"]) == 4


def test_paging_is_declared_as_data_beside_freshness() -> None:
    assert data_of({"paged": True}) == {PAGED: True}
    assert data_of({"paged": "yes"}) == {}  # only true pages
    assert data_of({"paged": True, "freshness": "live"}) == {"max_age": 0.0, PAGED: True}
    assert data_of({"paged": True, "max_age": 5}) == {"max_age": 5.0, PAGED: True}

    @tool(side_effects="read", paged=True)
    def rows(cursor: str | None = None) -> dict[str, Any]:
        """Rows, a page at a time."""
        return {"items": [], "next_cursor": None}

    assert pages.cursor_arg(rows.spec) == "cursor"
