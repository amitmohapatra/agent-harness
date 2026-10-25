"""The quality audit's fixes in the middleware: every model call accounted whatever graph made
it (P8), the narrowing's reliance on LangChain's model node said aloud (R3), and the per-call
work bounded by the latest step rather than the whole conversation (A1-A6)."""

from __future__ import annotations

import logging
import uuid
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

import pytest
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from opentelemetry.sdk.trace import TracerProvider

from tests.support.arm import arm, native_tools
from tests.support.models import ScriptedChat
from trellis import Budget, Harness, ReAct, current, tool
from trellis.contracts import RunStatus
from trellis.harness import middleware as mw
from trellis.harness import telemetry
from trellis.harness.tools.convert import langchain as converted


@tool(side_effects="read")
def stock(sku: str) -> int:
    """Units of a SKU in stock.

    Args:
        sku: The SKU, such as a-1.
    """
    return 7


class Ahead(Sequence[Any]):
    """A conversation whose messages before ``start`` must not be read: reading one fails."""

    def __init__(self, messages: list[Any], start: int) -> None:
        self.messages = messages
        self.start = start

    def __len__(self) -> int:
        return len(self.messages)

    def __getitem__(self, n: Any) -> Any:
        if isinstance(n, slice) or (n if n >= 0 else len(self) + n) < self.start:
            raise AssertionError(f"read message {n}: before the latest turn")
        return self.messages[n]


def conversation(steps: int) -> list[Any]:
    """An old turn of ``steps`` steps, then a new one of one failed step."""
    old: list[Any] = [HumanMessage(content="old")]
    for n in range(steps):
        call = {"name": "t", "args": {"n": n}, "id": f"o{n}"}
        old += [AIMessage(content="", tool_calls=[call]), ToolMessage("x", tool_call_id=f"o{n}")]
    call = {"name": "t", "args": {}, "id": "new"}
    return [
        *old,
        HumanMessage(content="new"),
        AIMessage(content="", tool_calls=[call]),
        ToolMessage("bad", tool_call_id="new", status="error"),
    ]


# --------------------------------------------------------------------------- P8
async def test_a_graph_without_model_hooks_spends_its_budget(harness: Harness) -> None:
    model = ScriptedChat([("stock", {"sku": "a"}), ("stock", {"sku": "b"}), "7"])
    tools = await native_tools(harness, stock, framework="langgraph")
    graph = create_agent(model, tools=tools, system_prompt="s")
    agent = harness.wrap(graph, id="plain", budget=Budget(max_tokens=20))
    result = await agent.run("a?", user="u")
    # each reply is 15 tokens: the second call's tools are refused, the budget spent
    assert result.status is RunStatus.PARTIAL, result.error
    assert len(model.requests) <= 3


@tool(side_effects="read")
def spent() -> int:
    """The tokens this run has spent so far."""
    runtime = current()
    assert runtime is not None
    return int(runtime.replay.journal.spent.get("tokens", 0))


async def test_react_accounts_each_model_call_once(harness: Harness) -> None:
    model = ScriptedChat([("spent", {}), "done"])
    agent = arm(harness.wrap(ReAct(system_prompt="s", model=model), id="once"), [spent])
    assert (await agent.run("a?", user="u")).answer == "done"
    assert model.requests[1]["messages"][-1]["content"] == "15"  # one reply of 15 tokens


def test_outside_a_run_nothing_is_accounted() -> None:
    from langchain_core.outputs import ChatGeneration, LLMResult

    from trellis.harness.accounting import UsageAccounting

    reply = LLMResult(generations=[[ChatGeneration(message=AIMessage(content="x"))]])
    UsageAccounting().on_llm_end(reply, run_id=uuid.uuid4())  # no run: nothing to account


# --------------------------------------------------------------------------- R3
def test_a_model_node_of_another_shape_is_said_once(caplog: pytest.LogCaptureFixture) -> None:
    async def node(state: Any) -> Any:
        return state

    node.__qualname__ = "create_agent.<locals>.amodel_node.r3test"
    with caplog.at_level(logging.WARNING, logger="trellis.run"):
        assert mw.narrowing(object(), node) is None
        assert mw.narrowing(object(), node) is None
    said = [r for r in caplog.records if "not of the shape" in r.getMessage()]
    assert len(said) == 1


# --------------------------------------------------------------------------- A1, A2, A4
def test_the_stall_look_back_reads_only_the_latest_turn() -> None:
    messages = conversation(500)
    ahead = Ahead(messages, start=len(messages) - 3)
    assert mw._failing(ahead) == 1
    assert [len(s) for s in mw._steps(ahead)] == [1]


def test_a_call_is_found_from_the_end() -> None:
    messages = conversation(500)
    ahead = Ahead(messages, start=len(messages) - 3)
    found = mw._calling(ahead, "new")
    assert found is not None and found[0] == len(messages) - 2
    assert mw._calling(messages, "missing") is None
    # an id a model used again: the latest call is the one
    again = [*messages, AIMessage(content="", tool_calls=[{"name": "t", "args": {}, "id": "new"}])]
    assert mw._calling(again, "new") == (len(again) - 1, again[-1])


async def test_read_result_reads_the_latest_result_of_an_id() -> None:
    read = mw.read_result()
    messages = [ToolMessage("first", tool_call_id="c"), ToolMessage("again", tool_call_id="c")]
    assert await read.coroutine(id="c", state={"messages": messages}) == "again"  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- A5
def test_writes_are_kept_once_and_a_special_write_replaces() -> None:
    saver = mw.RunCheckpointer()
    config: Any = {"configurable": {"thread_id": "t", "checkpoint_ns": "", "checkpoint_id": "1"}}
    saver.put_writes(config, [("a", 1), ("b", 2)], "task")
    saver.put_writes(config, [("a", 9)], "task")  # the same write again: kept as it was
    saver.put_writes(config, [("__error__", "x")], "task")
    saver.put_writes(config, [("__error__", "y")], "task")  # special: replaced
    kept = saver._threads["t"][""]["writes"]
    assert [(w[2], w[3], saver._loaded(w[4])) for w in kept] == [
        (0, "a", 1),
        (1, "b", 2),
        (-1, "__error__", "y"),
    ]


# --------------------------------------------------------------------------- A3
def test_the_spans_messages_are_made_only_when_it_records(monkeypatch: Any) -> None:
    made: list[int] = []

    def messages() -> list[dict[str, str]]:
        made.append(1)
        return [{"role": "user", "content": "hi"}]

    with telemetry.model_span("m", messages):
        pass
    assert made == []
    monkeypatch.setattr(telemetry, "_tracer", TracerProvider().get_tracer("t"))
    with telemetry.model_span("m", messages) as span:
        assert span.is_recording()
    assert made == [1]


# --------------------------------------------------------------------------- A6
async def test_a_tool_set_is_converted_once_for_its_runs(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []
    real = converted.convert

    def counting(tools: Any) -> Any:
        calls.append(len(tools))
        return real(tools)

    monkeypatch.setattr(converted, "convert", counting)
    model = ScriptedChat(["one", "two", "three"])
    agent = arm(harness.wrap(ReAct(system_prompt="s", model=model), id="conv"), [stock])
    for n in range(3):
        await agent.run(f"run {n}", user="u")
    assert calls == [1]


def test_a_text_reply_or_a_note_in_the_turn_changes_nothing(monkeypatch: Any) -> None:
    from langchain_core.messages import SystemMessage
    from langchain_core.outputs import Generation, LLMResult

    from trellis.harness import accounting

    accounted: list[Any] = []
    run = SimpleNamespace(model_ended=lambda key: None)
    monkeypatch.setattr(accounting, "current", lambda: run)
    monkeypatch.setattr(accounting.budget, "account_reply", lambda *a: accounted.append(a))
    accounting.UsageAccounting().on_llm_end(
        LLMResult(generations=[[Generation(text="x")]]), run_id=uuid.uuid4()
    )
    assert accounted == []  # a completion model's text reply carries no chat usage
    noted = [*conversation(1), SystemMessage(content="a note")]
    assert mw._failing(noted) == 1
