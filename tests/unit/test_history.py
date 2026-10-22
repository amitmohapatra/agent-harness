"""Who owns a conversation's history (one policy, a registry of RAM-only stores), and its
messages between the memory service and each framework: the window as the
framework's own messages, and what a run said back as the service stores it."""

from __future__ import annotations

import json
from typing import Any

import pytest
from agents import Agent as OpenAIAgent
from agents import MessageOutputItem, SQLiteSession, ToolCallItem, ToolCallOutputItem
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver, MemorySaver
from openai.types.responses import (
    ResponseFileSearchToolCall,
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
)

from trellis.harness.adapters.langgraph import LangGraphAdapter, added, langchain_message
from trellis.harness.adapters.openai_agents import OpenAIAgentsAdapter, said_in
from trellis.harness.bridges.session import input_items
from trellis.harness.history import (
    INTERNAL,
    Called,
    Owner,
    Said,
    Speaker,
    answered,
    from_window,
    in_ram,
    owner,
    recorded,
    said_of,
)
from trellis.memory.models import WindowMessage

CALL = Called("c1", "stock", {"sku": "a"})
TURN = [
    Said(Speaker.USER, "How many a?"),
    Said(Speaker.ASSISTANT, "", (CALL,)),
    Said(Speaker.TOOL, "7", tool_call_id="c1"),
    Said(Speaker.ASSISTANT, "7 units."),
]


def test_the_window_becomes_messages_in_its_order() -> None:
    window = [
        WindowMessage(id="m1", role="USER", text="How many a?"),
        WindowMessage.model_validate(
            {
                "id": "m2",
                "role": "ASSISTANT",
                "text": "",
                "tool_calls": [{"id": "c1", "name": "stock", "arguments": {"sku": "a"}}],
            }
        ),
        WindowMessage(id="m3", role="TOOL", text="7", tool_call_id="c1"),
        WindowMessage(id="m4", role="ASSISTANT", text="7 units."),
        WindowMessage(id="m5", role="EVENT", text="not replayed"),
    ]
    assert from_window(window) == TURN


def test_a_message_is_stored_as_written_and_read_back() -> None:
    stored = [recorded(s) for s in TURN]
    assert stored[1] == {
        "role": "ASSISTANT",
        "content": "",
        "tool_calls": [{"id": "c1", "name": "stock", "arguments": {"sku": "a"}}],
    }
    assert stored[2] == {
        "role": "TOOL",
        "content": "7",
        "kind": INTERNAL,
        "tool_call_id": "c1",
    }
    assert stored[0] == {"role": "USER", "content": "How many a?"}
    # through the spool's JSON and back, and the pairs an older spool holds
    assert [said_of(m) for m in json.loads(json.dumps(stored))] == TURN
    assert said_of(["assistant", "hi"]) == Said(Speaker.ASSISTANT, "hi")


def test_langgraph_gets_the_window_as_its_own_messages() -> None:
    history = [Said(Speaker.SYSTEM, "earlier summary"), *TURN]
    state = LangGraphAdapter().prepare_input(None, "And b?", "context", history)
    system, summary, user, call, result, answer, asked = state["messages"]
    assert isinstance(system, SystemMessage) and system.content == "context"
    assert isinstance(summary, SystemMessage) and summary.content == "earlier summary"
    assert isinstance(user, HumanMessage) and user.content == "How many a?"
    assert isinstance(call, AIMessage)
    assert call.tool_calls == [
        {"id": "c1", "name": "stock", "args": {"sku": "a"}, "type": "tool_call"}
    ]
    assert isinstance(result, ToolMessage) and result.tool_call_id == "c1"
    assert isinstance(answer, AIMessage) and answer.content == "7 units."
    assert asked == {"role": "user", "content": "And b?"}
    assert langchain_message(Said(Speaker.TOOL, "x")).tool_call_id == ""


def test_langgraph_reports_what_the_run_added_after_its_question() -> None:
    messages = [
        HumanMessage("earlier"),
        AIMessage("earlier answer"),
        HumanMessage("How many a?"),
        AIMessage("", tool_calls=[{"id": "c1", "name": "stock", "args": {"sku": "a"}}]),
        ToolMessage("7", tool_call_id="c1"),
        AIMessage(""),  # said nothing, called nothing: not a message worth replaying
        SystemMessage("a note a middleware added"),
        AIMessage([{"type": "text", "text": "7 units."}]),
    ]
    assert added(messages) == TURN[1:]
    assert added([AIMessage("no question")]) == []


def test_openai_agents_read_the_thread_as_input_items() -> None:
    assert [i for said in answered(TURN) for i in input_items(said)] == [
        {"role": "user", "content": "How many a?"},
        {"type": "function_call", "call_id": "c1", "name": "stock", "arguments": '{"sku": "a"}'},
        {"type": "function_call_output", "call_id": "c1", "output": "7"},
        {"role": "assistant", "content": "7 units."},
    ]
    spoke = Said(Speaker.ASSISTANT, "Let me check.", (CALL,))
    assert [i.get("role") or i["type"] for i in input_items(spoke)] == [
        "assistant",
        "function_call",
    ]
    # a message that said nothing and called nothing is still a message
    assert input_items(Said(Speaker.USER, "")) == [{"role": "user", "content": ""}]
    # the session carries the history: the input stays as it was given
    assert OpenAIAgentsAdapter().prepare_input(None, "hi", None, ()) == "hi"


def test_a_call_and_its_result_are_never_parted() -> None:
    unanswered = Said(Speaker.ASSISTANT, "Checking.", (CALL,))
    silent = Said(Speaker.ASSISTANT, "", (CALL,))
    orphan = Said(Speaker.TOOL, "7", tool_call_id="gone")
    assert answered([unanswered, silent, orphan]) == [Said(Speaker.ASSISTANT, "Checking.")]


def _message(agent: OpenAIAgent, text: str) -> MessageOutputItem:
    content: list[Any] = [ResponseOutputText(type="output_text", text=text, annotations=[])]
    raw = ResponseOutputMessage(
        id="m", role="assistant", status="completed", type="message", content=content
    )
    return MessageOutputItem(agent=agent, raw_item=raw)


def _call(agent: OpenAIAgent, call_id: str, arguments: str) -> ToolCallItem:
    raw = ResponseFunctionToolCall(
        id=f"fc_{call_id}",
        call_id=call_id,
        type="function_call",
        name="stock",
        arguments=arguments,
        status="completed",
    )
    return ToolCallItem(agent=agent, raw_item=raw)


def test_openai_agents_report_calls_results_and_answers_in_order() -> None:
    agent = OpenAIAgent(name="a")
    hosted = ResponseFileSearchToolCall(
        id="fs", queries=["q"], status="completed", type="file_search_call"
    )
    items = [
        _message(agent, "Let me check."),
        _call(agent, "c1", '{"sku": "a"}'),
        _call(agent, "c2", "not json"),
        ToolCallItem(agent=agent, raw_item=hosted),  # the SDK's own tool: not replayed
        ToolCallOutputItem(
            agent=agent,
            raw_item={"type": "function_call_output", "call_id": "c1", "output": "7"},
            output=7,
        ),
        ToolCallOutputItem(
            agent=agent,
            raw_item={"type": "function_call_output", "call_id": "c2", "output": "x"},
            output="x",
        ),
        ToolCallOutputItem(agent=agent, raw_item={"type": "computer_call_output"}, output="png"),
        _message(agent, ""),
        _call(agent, "c3", "{}"),
        _message(agent, "7 units."),
    ]
    assert said_in(items) == [
        Said(
            Speaker.ASSISTANT,
            "Let me check.",
            (Called("c1", "stock", {"sku": "a"}), Called("c2", "stock", {"arguments": "not json"})),
        ),
        Said(Speaker.TOOL, "7", tool_call_id="c1"),
        Said(Speaker.TOOL, "x", tool_call_id="c2"),
        Said(Speaker.ASSISTANT, "", (Called("c3", "stock", {}),)),
        Said(Speaker.ASSISTANT, "7 units."),
    ]


@pytest.mark.parametrize("role", list(Speaker))
def test_every_speaker_goes_both_ways(role: Speaker) -> None:
    said = Said(role, "text")
    assert said_of(recorded(said)) == said
    assert langchain_message(said).content == "text"


def test_no_store_leaves_the_history_to_memory() -> None:
    assert owner(None, agent_id="a") is Owner.MEMORY


@pytest.mark.parametrize(
    "store", [InMemorySaver(), MemorySaver(), SQLiteSession("s")], ids=["saver", "alias", "sqlite"]
)
def test_a_store_kept_in_ram_leaves_the_history_to_memory_said_once(
    store: object, caplog: pytest.LogCaptureFixture
) -> None:
    assert in_ram(store)
    with caplog.at_level("WARNING", logger="trellis.harness"):
        assert owner(store, agent_id=f"ram_{id(store)}") is Owner.MEMORY
        assert owner(store, agent_id=f"ram_{id(store)}") is Owner.MEMORY
    said = [r for r in caplog.records if "memory only" in r.getMessage()]
    assert len(said) == 1 and said[0].store == type(store).__name__  # type: ignore[attr-defined]


def test_a_durable_store_owns_the_history(tmp_path: pytest.TempPathFactory) -> None:
    on_disk = SQLiteSession("s", f"{tmp_path}/s.db")
    assert not in_ram(on_disk) and owner(on_disk, agent_id="a") is Owner.FRAMEWORK
    assert owner(object(), agent_id="a") is Owner.FRAMEWORK  # unknown: taken as durable
    on_disk.close()
