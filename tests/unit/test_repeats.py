"""A call a model step repeats (the same tool, the same arguments): which calls are repeats
(``tools.bridge.firsts``), read from an OpenAI Agents reply (``hooks.openai_agents``), and a
harness tool called outside a run, which has no step to read."""

from __future__ import annotations

from typing import Any

import pytest
from agents.tool_context import ToolContext
from openai.types.responses import ResponseFunctionToolCall, ResponseOutputMessage

from trellis.contracts import ToolError, ToolSpec
from trellis.harness.hooks.openai_agents import _repeats
from trellis.harness.tools.base import Tool
from trellis.harness.tools.bridge import firsts
from trellis.harness.tools.convert import openai_agents


def test_a_repeat_is_the_same_tool_with_the_same_arguments_in_any_key_order() -> None:
    calls = [
        ("c1", "stock", {"style": "ZZ9", "store": "S1"}),
        ("c2", "stock", {"store": "S1", "style": "ZZ9"}),  # the same arguments
        ("c3", "stock", {"style": "ZZ9", "store": "S2"}),  # others
        ("c4", "price", {"style": "ZZ9", "store": "S1"}),  # another tool
        ("c5", "stock", {"style": "ZZ9", "store": "S2"}),
        ("c6", "stock", {"style": "ZZ9", "store": "S1"}),
    ]
    assert firsts(calls) == {"c2": "c1", "c5": "c3", "c6": "c1"}
    assert firsts([]) == {}


def _call(call_id: str, arguments: str) -> ResponseFunctionToolCall:
    return ResponseFunctionToolCall(
        id=f"fc_{call_id}", call_id=call_id, name="stock", arguments=arguments, type="function_call"
    )


def test_an_openai_reply_repeats_by_its_arguments_parsed_or_as_written() -> None:
    said = ResponseOutputMessage(
        id="msg_1", role="assistant", status="completed", type="message", content=[]
    )
    items = [
        said,
        _call("c1", '{"a": 1, "b": 2}'),
        _call("c2", '{"b": 2, "a": 1}'),  # the same object, written otherwise
        _call("c3", "{not json"),
        _call("c4", "{not json"),  # the same text that is not an object
        _call("c5", "[1]"),
    ]
    assert _repeats(items) == {
        ("c2", "stock", '{"b": 2, "a": 1}'): "c1",
        ("c4", "stock", "{not json"): "c3",
    }


async def test_a_harness_tool_called_outside_a_run_has_no_step_and_says_where_it_runs() -> None:
    async def stock(args: dict[str, Any]) -> int:
        return 7

    made = openai_agents.convert([Tool(ToolSpec(name="stock"), stock)])[0]
    context = ToolContext(
        context=None, tool_name="stock", tool_call_id="c1", tool_arguments='{"style": "a"}'
    )
    with pytest.raises(ToolError, match="stock is a trellis tool: it runs inside a Harness run"):
        await made.on_invoke_tool(context, '{"style": "a"}')
