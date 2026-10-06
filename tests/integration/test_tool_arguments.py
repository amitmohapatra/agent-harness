"""A tool call whose arguments are not a JSON object — the model wrote broken JSON, or a list —
on every adapter whose framework hands the harness the text the model wrote: an error the
model reads (the call is not run), then the call made again, and the run goes on.

LangChain's agents (``create_agent``, Deep Agents) parse the arguments in the chat model
(``AIMessage.invalid_tool_calls``) and end the run on a message whose calls all failed to
parse, without a word to the model: the harness's middleware (``ModelHooks``, which a graph is
given for its model hooks, and ``ReAct`` has) answers them and asks the model again."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

import pytest
from agents import Agent, function_tool
from deepagents import create_deep_agent
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from tests.support.chat_model import ScriptedChatModel
from tests.support.models import ScriptedChat
from tests.support.openai_model import ScriptedModel
from trellis import Harness, ReAct, tool
from trellis.contracts import RunEventType, RunStatus
from trellis.harness.middleware import ModelHooks
from trellis.harness.tools.convert.openai_agents import FIX_ARGUMENTS, arguments_of

ran: list[str] = []


@tool(side_effects="read")
def stock(sku: str) -> int:
    """Units of a SKU in stock."""
    ran.append(sku)
    return 42


@pytest.fixture(autouse=True)
def _reset() -> None:
    ran.clear()


#: What the model writes, and what it is told about it.
BROKEN = {
    "not json": ("{sku: A-1", "its arguments are not valid JSON (Expecting property name"),
    "not an object": ('["A-1"]', "its arguments must be a JSON object"),
}


class _Openai:
    def __init__(self, raw: str) -> None:
        self.model = ScriptedModel([("stock", raw), ("stock", {"sku": "A-1"}), "42 units"])

    async def target(self, h: Harness) -> tuple[Any, list[Any]]:
        return Agent(name="stock", model=self.model), [stock]

    def told(self) -> str:
        [read] = [i for i in self.model.inputs[1] if i.get("type") == "function_call_output"]
        return str(read["output"])


class _React:
    #: streamed, LangChain reads the chunks of broken JSON leniently (``parse_partial_json``:
    #: ``{}``), so the call reaches the harness, whose schema check refuses it (on the stream);
    #: run, it is told what the JSON is (``test_react.py``)
    streamed: ClassVar[dict[str, str]] = {"not json": "missing required argument(s): sku"}

    def __init__(self, raw: str) -> None:
        function = {"name": "stock", "arguments": raw}
        broken = {
            "role": "assistant",
            "tool_calls": [{"id": "c1", "type": "function", "function": function}],
        }
        self.model = ScriptedChat([broken, ("stock", {"sku": "A-1"}), "42 units"])

    async def target(self, h: Harness) -> tuple[Any, list[Any]]:
        return ReAct(system="s", model=self.model), [stock]

    def told(self) -> str:
        return str(self.model.requests[1]["messages"][-1]["content"])


class _Unparsed(ScriptedChatModel):
    """A chat model whose first call is one the provider could not parse (as langchain-openai
    reports it)."""

    raw: str = ""

    def _generate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> ChatResult:
        if self.seen:
            return super()._generate(messages, *args, **kwargs)
        self.seen.append(list(messages))
        self.turns.pop(0)
        bad = {
            "name": "stock",
            "args": self.raw,
            "id": "c1",
            "error": "bad JSON",
            "type": "invalid_tool_call",
        }
        message = AIMessage(content="", invalid_tool_calls=[bad])  # type: ignore[list-item]
        return ChatResult(generations=[ChatGeneration(message=message)])


class _LangChain:
    def __init__(self, raw: str, build: Callable[[Any, list[Any]], Any]) -> None:
        turns: list[Any] = ["broken", ("stock", {"sku": "A-1"}), "42 units"]
        self.model, self.build = _Unparsed(turns=turns, raw=raw), build

    async def target(self, h: Harness) -> tuple[Any, list[Any]]:
        return self.build(self.model, await h.tools(stock, framework="langgraph")), []

    def told(self) -> str:
        assert len(self.model.seen) > 1, "the model was never called again"
        return str([m for m in self.model.seen[1] if isinstance(m, ToolMessage)][-1].content)


def _langgraph(raw: str) -> _LangChain:
    return _LangChain(
        raw, lambda model, tools: create_agent(model, tools=tools, middleware=[ModelHooks()])
    )


def _deepagents(raw: str) -> _LangChain:
    return _LangChain(
        raw,
        lambda model, tools: create_deep_agent(model=model, tools=tools, middleware=[ModelHooks()]),
    )


ADAPTERS = [
    pytest.param(_Openai, id="openai_agents"),
    pytest.param(_React, id="react"),
    pytest.param(_langgraph, id="langgraph"),
    pytest.param(_deepagents, id="deepagents"),
]


@pytest.mark.parametrize("broken", list(BROKEN))
@pytest.mark.parametrize("adapter", ADAPTERS)
async def test_arguments_that_are_not_an_object_are_an_error_the_model_reads(
    harness: Harness, adapter: Callable[[str], Any], broken: str
) -> None:
    raw, problem = BROKEN[broken]
    case = adapter(raw)
    refused = broken in getattr(case, "streamed", {})
    problem = getattr(case, "streamed", {}).get(broken, problem)
    target, tools = await case.target(harness)
    events = [
        e async for e in harness.wrap(target, id="args", tools=tools).stream("A-1?", user="u")
    ]
    finished = events[-1]
    assert finished.type is RunEventType.RUN_FINISHED and finished.error is None, finished
    assert finished.data.get("result") == "42 units"
    assert case.told().startswith(f"stock was not run: {problem}")
    assert ran == ["A-1"]  # the broken call never ran; the one made again did
    # only the call that ran is on the run's stream
    started = [e for e in events if e.type is RunEventType.TOOL_CALL_START]
    assert [e.data["tool"] for e in started] == ["stock"] * (2 if refused else 1)


def test_the_arguments_of_a_call_are_an_object_or_what_is_wrong_with_them() -> None:
    assert arguments_of(None) == arguments_of("") == {}
    assert arguments_of('{"sku": "A-1"}') == {"sku": "A-1"}
    assert arguments_of("7") == "its arguments must be a JSON object"
    assert str(arguments_of("{")).startswith("its arguments are not valid JSON")
    assert FIX_ARGUMENTS.startswith("Call it again")


async def test_an_sdk_approval_of_arguments_that_are_not_json_pauses_with_them_as_written(
    harness: Harness,
) -> None:
    @function_tool(needs_approval=True)
    def send(order: str) -> str:
        raise AssertionError("arguments that are not JSON never reach the function")

    model = ScriptedModel([("send", "{order: o9"), "could not send"])
    agent = harness.wrap(Agent(name="s", model=model, tools=[send]), id="sender")
    paused = await agent.run("send o9", user="u1")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
    assert paused.interrupt.tool_call is not None
    assert paused.interrupt.tool_call.args == {"arguments": "{order: o9"}
    # approved, the SDK's own tool tells the model, and the run goes on
    finished = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="u1")
    assert finished.status is RunStatus.SUCCESS and finished.answer == "could not send"
    told = next(i for i in model.inputs[-1] if i.get("type") == "function_call_output")
    assert "valid JSON" in told["output"]
