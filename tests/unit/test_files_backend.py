"""Where a Deep Agents file tool keeps its files, read from the tool itself
(``middleware.files_backend``): a tool whose code closes over no ``FilesystemMiddleware`` has no
backend to read, and its calls are then governed (it fails closed)."""

from __future__ import annotations

import types
from typing import Any

from langchain_core.tools import StructuredTool

from trellis.harness import middleware


def _closing_over_nothing_readable() -> Any:
    """A tool's code whose closure holds an empty cell (a name it reads that is never bound)
    and nothing else: what a tool of another middleware may close over."""

    def read_file(file_path: str) -> str:
        """Read a file."""
        return file_path

    empty = types.CellType()  # a cell with no value: reading it raises ValueError
    return types.FunctionType(
        _reads_a_free_name().__code__, read_file.__globals__, "read_file", None, (empty,)
    )


def _reads_a_free_name() -> Any:
    unbound = None

    def read_file(file_path: str) -> str:
        """Read a file."""
        return f"{unbound} {file_path}"

    return read_file


async def _read_file(file_path: str) -> str:
    """Read a file."""
    return file_path


def test_a_tool_that_closes_over_no_filesystem_middleware_has_no_backend_to_read() -> None:
    sync = StructuredTool.from_function(func=_closing_over_nothing_readable(), name="read_file")
    both = StructuredTool.from_function(
        func=_closing_over_nothing_readable(), coroutine=_read_file, name="read_file"
    )
    assert middleware.files_backend(sync) is None
    assert middleware.files_backend(both) is None  # its coroutine read first, then its func
    # and so its calls are the run's: governed, as a file tool outside the graph's state is
    assert middleware._file_marks(sync, "read_file") == middleware.outside_state("read_file")
