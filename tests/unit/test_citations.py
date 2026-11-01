"""A tool's artifact as its sources (``tools/base.py``: ``citations``), and the tool's
``response_format``."""

from __future__ import annotations

import pytest

from trellis import Citation, tool
from trellis.contracts import ConfigurationError
from trellis.harness.tools.base import citations

NASDAQ = Citation(name="Nasdaq Last Sale feed", date="2026-10-27", link="https://nasdaq.test/a")


def test_citations_are_a_citation_or_a_list_of_them() -> None:
    assert citations(NASDAQ) == [NASDAQ]
    assert citations([{"name": "a"}, {"name": "b", "link": "x"}]) == [
        Citation(name="a"),
        Citation(name="b", link="x"),
    ]
    assert citations({"rows": 1}) == [] and citations(None) == [] and citations([]) == []


def test_a_tools_response_format_is_langchains() -> None:
    with pytest.raises(ConfigurationError, match="content_and_artifact"):
        tool(lambda: "x", response_format="artifact")  # type: ignore[arg-type]
    assert tool(lambda: "x").response_format == "content"
