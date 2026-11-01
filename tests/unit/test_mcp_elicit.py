"""A server's question (MCP elicitation) as the person is asked it, and the answer the server
gets back: a field that picks from values a choice (titled, several, the older ``enumNames``),
any other form a form, a link a choice of opened or declined; and a question no call can be
told to own."""

from __future__ import annotations

import asyncio
from typing import Any

from mcp.types import (
    INVALID_REQUEST,
    ElicitRequestFormParams,
    ElicitRequestURLParams,
    ErrorData,
)

from trellis.contracts import Option
from trellis.harness.tools.elicit import DECLINED, OPENED, elicited
from trellis.harness.tools.mcp import WHOSE, _Call, mcp


def form(**properties: Any) -> ElicitRequestFormParams:
    schema = {"type": "object", "properties": properties}
    return ElicitRequestFormParams(message="Which?", requested_schema=schema)  # type: ignore[arg-type]


def test_a_field_that_picks_from_values_is_a_choice() -> None:
    plain = elicited(form(size={"type": "string", "enum": ["S", "L"]}))
    assert plain.options == ["S", "L"] and not plain.multiple
    assert plain.expects is None
    assert plain.result("L") == {"action": "accept", "content": {"size": "L"}}
    titled = elicited(
        form(region={"type": "string", "oneOf": [{"const": "eu", "title": "Europe"}]})
    )
    assert titled.options == [Option(value="eu", label="Europe")]
    named = elicited(form(tier={"type": "string", "enum": ["a", "b"], "enumNames": ["Gold"]}))
    assert named.options == [Option(value="a", label="Gold"), "b"]
    several = elicited(
        form(
            tags={
                "type": "array",
                "items": {"anyOf": [{"const": "x", "title": "X"}, {"const": "y", "title": "Y"}]},
            }
        )
    )
    assert several.multiple and several.options == [
        Option(value="x", label="X"),
        Option(value="y", label="Y"),
    ]
    assert several.result(["x"]) == {"action": "accept", "content": {"tags": ["x"]}}
    assert several.asking == {"options": several.options, "multiple": True, "expects": None}


def test_any_other_form_is_a_form_with_the_servers_schema() -> None:
    one = elicited(form(day={"type": "string", "format": "date"}))
    assert one.options == [] and one.expects == {
        "type": "object",
        "properties": {"day": {"type": "string", "format": "date"}},
    }
    assert one.result({"day": "2026-11-02"}) == {
        "action": "accept",
        "content": {"day": "2026-11-02"},
    }
    assert one.result(False) == {"action": "decline"}  # rejected
    two = elicited(form(a={"type": "string", "enum": ["x"]}, b={"type": "number"}))
    assert two.expects is not None and two.options == []


def test_a_link_to_open_is_a_choice_of_opened_or_declined() -> None:
    params = ElicitRequestURLParams(
        message="Sign in to the bank", url="https://bank.example/auth", mode="url"
    )
    link = elicited(params)
    assert link.question == "Sign in to the bank\nhttps://bank.example/auth"
    assert link.options == [OPENED, DECLINED]
    assert link.result("accept") == {"action": "accept"}
    assert link.result("decline") == {"action": "decline"}


async def test_a_question_while_several_calls_are_under_way_is_refused() -> None:
    source = mcp("https://h/mcp", name="h")
    source._calls = {_Call(), _Call()}
    refused = await source._elicited(None, form(a={"type": "string"}))
    assert refused == ErrorData(code=INVALID_REQUEST, message=WHOSE)
    source._calls = set()
    assert (await source._elicited(None, form(a={"type": "string"}))).message == WHOSE


async def test_the_only_call_under_way_gets_the_question() -> None:
    source = mcp("https://h/mcp", name="h")
    call = _Call()
    source._calls = {call}
    asking = asyncio.ensure_future(source._elicited(None, form(a={"type": "string"})))
    params, answered = await call.asked.get()
    assert params.message == "Which?"
    answered.set_result("the answer")
    assert await asking == "the answer"
