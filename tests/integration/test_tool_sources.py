"""Where a tool's result came from: a ``content_and_artifact`` tool (the harness's ``tool`` or
LangChain's) returns its content and a ``Citation`` artifact; the model reads the content alone
and the call's outcome keeps the citations in ``metadata["sources"]`` (``tools/base.py``)."""

from __future__ import annotations

import json
from typing import Any

import pytest
from langchain_core.tools import tool as lc_tool

from tests.support.arm import arm
from tests.support.models import ScriptedChat
from trellis import Citation, Harness, Hooks, ReAct, tool
from trellis.contracts import ToolCall, ToolOutcome
from trellis.harness.tools.base import SOURCES

pytestmark = pytest.mark.asyncio
NASDAQ = Citation(name="Nasdaq Last Sale feed", date="2026-10-27", link="https://nasdaq.test/a")
CONTENT = json.dumps({"symbol": "ACME", "close": 123.45})


@tool(side_effects="read", response_format="content_and_artifact")
def stock_quote(symbol: str) -> tuple[str, Citation]:
    """The latest closing quote of a stock symbol.

    Args:
        symbol: the ticker, e.g. ACME
    """
    return CONTENT, NASDAQ


class Outcomes(Hooks):
    def __init__(self) -> None:
        self.seen: list[ToolOutcome] = []

    async def after_tool(self, call: ToolCall, outcome: ToolOutcome) -> ToolOutcome:
        self.seen.append(outcome)
        return outcome


def _tool_results(request: dict[str, Any]) -> list[str]:
    return [m["content"] for m in request["messages"] if m["role"] == "tool"]


async def _ran(harness: Harness, tools: list[Any]) -> tuple[ScriptedChat, list[ToolOutcome]]:
    model = ScriptedChat([("stock_quote", {"symbol": "ACME"}), "123.45"])
    seen = Outcomes()
    agent = arm(harness.wrap(ReAct("Quotes.", model), id="stock", hooks=[seen]), tools)
    await agent.run("ACME?", user="u1")
    return model, seen.seen


async def test_the_artifact_is_kept_beside_the_result_never_read(harness: Harness) -> None:
    model, seen = await _ran(harness, [stock_quote])
    assert seen[0].metadata[SOURCES] == [NASDAQ.model_dump(mode="json", exclude_none=True)]
    assert _tool_results(model.script.requests[-1]) == [CONTENT]


async def test_an_artifact_that_names_no_source_is_not_one(harness: Harness) -> None:
    @tool(name="stock_quote", side_effects="read", response_format="content_and_artifact")
    def raw(symbol: str) -> tuple[str, Any]:
        """A lookup whose artifact is a raw payload."""
        return CONTENT, {"rows": [1, 2]}

    _, seen = await _ran(harness, [raw])
    assert SOURCES not in seen[0].metadata


async def test_a_content_and_artifact_tool_that_returns_no_pair_fails_its_call(
    harness: Harness,
) -> None:
    @tool(name="stock_quote", side_effects="read", response_format="content_and_artifact")
    def broken(symbol: str) -> str:
        """Returns the content alone, by mistake."""
        return CONTENT

    model, _ = await _ran(harness, [broken])
    [said] = _tool_results(model.script.requests[-1])
    assert "content_and_artifact" in said and "(content, artifact)" in said


async def test_a_langchain_tools_artifact_is_its_source(harness: Harness) -> None:
    @lc_tool(response_format="content_and_artifact")
    def stock_quote(symbol: str) -> tuple[str, Citation]:
        """The latest closing quote of a stock symbol."""
        return CONTENT, NASDAQ

    stock_quote.metadata = {"side_effects": "read"}
    model = ScriptedChat([("stock_quote", {"symbol": "ACME"}), "123.45"])
    seen = Outcomes()
    agent = harness.wrap(ReAct("Quotes.", model, tools=[stock_quote]), id="stock", hooks=[seen])
    await agent.run("ACME?", user="u1")
    assert seen.seen[0].metadata[SOURCES][0]["name"] == NASDAQ.name
