"""Large tool results where the framework has no cut of its own (``tools.results``): kept in
the run's journal, previewed as Deep Agents previews them, read back in pages."""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest

from trellis.harness import runtime as runtime_module
from trellis.harness.journal import Journal, content_key
from trellis.harness.tools import results
from trellis.harness.tools.results import MAX_CHARS, PAGE_CHARS, kept, read, read_file, readable


@pytest.fixture
def journal() -> Iterator[Journal]:
    """A run in progress: what ``tools.results`` reads of it is its journal."""
    journal = Journal()
    run = SimpleNamespace(replay=SimpleNamespace(journal=journal), toolbox={}, native={})
    token = runtime_module._current.set(run)  # type: ignore[arg-type]
    yield journal
    runtime_module._current.reset(token)


def kept_path(stub: str) -> str:
    return stub.split("at this path: ", 1)[1].split("\n", 1)[0]


def test_a_result_within_the_limit_is_read_as_it_is(journal: Journal) -> None:
    assert readable({"rows": [1, 2]}) == '{"rows": [1, 2]}'
    assert readable("x" * MAX_CHARS) == "x" * MAX_CHARS
    assert journal.results == {} and not kept()


def test_outside_a_run_nothing_is_cut_or_kept() -> None:
    assert readable("x" * (MAX_CHARS + 1)) == "x" * (MAX_CHARS + 1)
    assert not kept()
    assert read_file("/large_tool_results/abc") == "Error: File '/large_tool_results/abc' not found"


def test_a_large_result_is_kept_and_previewed_by_its_head_and_tail(journal: Journal) -> None:
    text = "\n".join(f"line {n} " + "y" * 900 for n in range(1, 201))
    stub = readable(text)
    path = kept_path(stub)
    assert path.startswith("/large_tool_results/") and journal.results == {path: text}
    assert kept() and len(stub) < 25_000
    assert "the read_file tool" in stub and "showing the head and tail" in stub
    assert "... [190 lines truncated] ..." in stub and "longer than" not in stub
    assert "\n1  line 1 " in stub and "\n200  line 200 " in stub and "line 6 " not in stub
    assert readable(text) == stub  # the same result, the same path: kept once


def test_a_one_line_result_is_previewed_and_paged_by_parts(journal: Journal) -> None:
    text = "".join(f"{n:04d}" for n in range(25_000))  # 100,000 characters on one line
    stub = readable(text)
    path = kept_path(stub)
    assert len(stub) < 25_000 and "this preview shows only their first 1000" in stub
    first = read_file(path)
    assert len(first) <= PAGE_CHARS + 200 and first.startswith(" 1  00000001")
    assert f'more lines: read_file(file_path="{path}", offset=3)]' in first
    pages, offset = [], 0
    while True:
        page = read_file(path, offset=offset, limit=4)
        pages.append(page)
        if "more lines" not in page:
            break
        offset = int(page.rsplit("offset=", 1)[1].rstrip("])"))
    rows = [row.split("  ", 1)[1] for page in pages for row in page.splitlines() if "  " in row]
    assert "".join(rows) == text  # every part read, in order


def test_read_file_pages_by_lines_from_an_offset(journal: Journal) -> None:
    text = "\n".join(f"row {n}" for n in range(1, 30_001))
    path = kept_path(readable(text))
    page = read_file(path.lstrip("/"), offset=10, limit=3)  # a path without its slash too
    assert page.splitlines() == [
        "11  row 11",
        "12  row 12",
        "13  row 13",
        f'…[29987 more lines: read_file(file_path="{path}", offset=13)]',
    ]
    assert read_file(path, offset=29_999) == "30000  row 30000"
    assert read_file(path, offset=-5, limit=0).splitlines()[0] == "1  row 1"
    assert read_file(path, offset=30_000) == (
        "Error: Line offset 30000 exceeds file length (30000 lines)"
    )


def test_read_checks_the_arguments_the_model_wrote(journal: Journal) -> None:
    path = kept_path(readable("z\n" * MAX_CHARS))
    assert read({"file_path": path, "limit": 1}).startswith("1  z\n…[")
    assert read({"offset": 1}).startswith(f"{results.READ_FILE} was not run: missing required")
    assert "unknown" in read({"file_path": path, "page": 2})


@pytest.fixture
def run(journal: Journal) -> SimpleNamespace:
    """The run in progress the ``journal`` fixture set (its tools: ``toolbox``, ``native``)."""
    return runtime_module._current.get()  # type: ignore[return-value]


def test_a_kept_text_is_named_by_its_content_and_read_on_from_where_a_cut_starts(
    journal: Journal,
) -> None:
    text = "\n".join(f"row {n}" for n in range(1, 101))
    path = results.keep(text)
    assert path == f"/large_tool_results/{content_key('result', text)}"
    assert journal.results == {path: text} and results.keep(text) == path  # kept once
    two = len("row 1\nrow 2\n")
    assert results.line_at(text, 0) == 0
    assert results.line_at(text, two) == 2  # on a line's end: the next line
    assert results.line_at(text, two + 2) == 2  # within a line: that line, again
    assert read_file(path, offset=2, limit=1).startswith("3  row 3")


def test_outside_a_run_nothing_is_kept_and_no_code_runs() -> None:
    assert results.keep("x") is None and results.coding_tool() is None


@pytest.mark.parametrize(
    ("toolbox", "native", "said"),
    [({"sandbox_exec": 1}, {}, "sandbox_exec"), ({}, {"execute": 1}, "execute"), ({}, {}, None)],
)
def test_a_run_with_a_sandbox_is_told_the_kept_result_is_a_file_there_to_run_code_over(
    run: SimpleNamespace, toolbox: dict[str, Any], native: dict[str, Any], said: str | None
) -> None:
    run.toolbox, run.native = toolbox, native
    stub = readable("y\n" * MAX_CHARS)
    path = kept_path(stub)
    code = results.CODE.format(path=path, tool=said)
    assert (code in stub) is (said is not None)
    if said is not None:  # said before the preview, which a small model reads past
        assert stub.index(code) < stub.index("Here is a preview")
