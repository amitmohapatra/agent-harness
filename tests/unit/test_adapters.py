"""Each adapter's four functions on their own: what the framework is given, what is read back
from it, and what a resume continues with — the shapes the integration tests do not reach."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from typing import Any

import pytest
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
)
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import GraphOutput

from trellis.contracts import (
    ConfigurationError,
    Interrupt,
    InterruptDecision,
    InterruptReason,
    InterruptResolution,
    ToolCall,
)
from trellis.harness.adapters import convert, detect
from trellis.harness.adapters.base import Invocation, Output, query_of
from trellis.harness.adapters.claude import (
    CLI_MESSAGE_BYTES,
    ClaudeAdapter,
    ClaudeRunError,
    _with_context,
    configured_servers,
)
from trellis.harness.adapters.function import FunctionAdapter
from trellis.harness.adapters.langgraph import (
    CONTEXT_MESSAGE_ID,
    LangGraphAdapter,
    bound_tools,
    hitl_response,
    is_hitl,
)
from trellis.harness.adapters.openai_agents import (
    EDITED,
    OpenAIAgentsAdapter,
    _arguments,
    _Continue,
)
from trellis.harness.adapters.react import ReAct, ReActAdapter, ReActResult, _unfenced
from trellis.harness.journal import Journal, Pending

# --------------------------------------------------------------------------- the query


def test_the_query_is_the_text_the_user_asked() -> None:
    assert query_of("plain") == "plain"
    assert query_of({"messages": [{"role": "user", "content": "in a state"}]}) == "in a state"
    assert query_of({"question": "  ", "prompt": "the first text field"}) == "the first text field"
    assert query_of({"count": 3}) == ""
    messages = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "latest"},
    ]
    assert query_of(messages) == "latest"
    assert query_of([HumanMessage(content="a LangChain message"), AIMessage(content="x")]) == (
        "a LangChain message"
    )
    assert query_of([{"role": "user", "content": [{"type": "image"}]}]) == ""
    assert query_of(42) == ""


# --------------------------------------------------------------------------- detection


@pytest.mark.parametrize("module", ["langgraph.fake", "agents.fake", "claude_agent_sdk.fake"])
def test_a_lookalike_from_a_framework_module_is_not_wrapped(module: str) -> None:
    lookalike = type("Graph", (), {"__module__": module})()
    with pytest.raises(ConfigurationError, match="cannot wrap Graph"):
        detect(lookalike)


def test_an_object_with_an_async_call_is_a_function_target() -> None:
    class Callable_:
        async def __call__(self, input: Any, agent: Any) -> Any:
            return input

    assert isinstance(detect(Callable_()), FunctionAdapter)


def test_no_tools_or_no_format_converts_to_nothing() -> None:
    assert convert("langchain", []) is None
    assert convert("none", [object()]) is None  # type: ignore[list-item]


# --------------------------------------------------------------------------- function


def test_a_function_gets_the_context_before_a_message_list() -> None:
    adapter = FunctionAdapter()
    messages = [{"role": "user", "content": "hi"}]
    assert adapter.prepare_input(None, messages, "ctx") == [
        {"role": "system", "content": "ctx"},
        *messages,
    ]
    assert adapter.prepare_input(None, "hi", "ctx") == "hi"  # read from agent.context instead
    assert adapter.extract(None, {"a": 1}).transcript == []


# --------------------------------------------------------------------------- Claude


@pytest.mark.parametrize(
    ("system_prompt", "expected"),
    [
        (None, "CTX"),
        ("You ship.", "You ship.\n\nCTX"),
        (
            {"type": "preset", "preset": "claude_code"},
            {"type": "preset", "preset": "claude_code", "append": "CTX"},
        ),
        (
            {"type": "preset", "preset": "claude_code", "append": "Be brief."},
            {"type": "preset", "preset": "claude_code", "append": "Be brief.\n\nCTX"},
        ),
        ({"type": "custom", "prompt": "Mine."}, {"type": "custom", "prompt": "Mine.\n\nCTX"}),
        ({"type": "file", "path": "prompt.md"}, {"type": "file", "path": "prompt.md"}),
    ],
    ids=["none", "text", "preset", "preset-appended", "custom", "file-untouched"],
)
def test_the_context_joins_whatever_system_prompt_the_options_have(
    system_prompt: Any, expected: Any
) -> None:
    assert _with_context(system_prompt, "CTX") == expected


def test_a_structured_input_is_the_prompt_as_json() -> None:
    prepared = ClaudeAdapter().prepare_input(None, {"order": 7}, None)
    assert (prepared.prompt, prepared.context) == ('{"order": 7}', None)


def _result(**fields: Any) -> ResultMessage:
    base: dict[str, Any] = {
        "subtype": "success",
        "duration_ms": 1,
        "duration_api_ms": 1,
        "is_error": False,
        "num_turns": 1,
        "session_id": "s",
    }
    return ResultMessage(**{**base, **fields})


def test_a_failed_cli_run_raises_with_its_errors() -> None:
    adapter = ClaudeAdapter()
    with pytest.raises(ClaudeRunError, match="rate limited; retry later"):
        adapter.extract(None, [_result(is_error=True, errors=["rate limited", "retry later"])])
    with pytest.raises(ClaudeRunError, match="error_max_turns"):
        adapter.extract(None, [_result(is_error=True, subtype="error_max_turns")])


def test_structured_output_outranks_the_result_text() -> None:
    said = AssistantMessage(
        content=[TextBlock(text="thinking"), ToolUseBlock("t", "x", {})], model="m"
    )
    extracted = ClaudeAdapter().extract(
        None, [said, _result(result="text", structured_output={"n": 1})]
    )
    assert extracted.answer == {"n": 1}
    assert extracted.transcript == [("assistant", "thinking")]


async def test_the_stream_yields_only_the_assistant_text(monkeypatch: pytest.MonkeyPatch) -> None:
    import claude_agent_sdk

    messages = [
        AssistantMessage(content=[ToolUseBlock("t1", "mcp__trellis__x", {})], model="m"),
        AssistantMessage(content=[TextBlock(text=""), TextBlock(text="done")], model="m"),
        _result(result="done"),
    ]
    seen: list[ClaudeAgentOptions] = []

    async def query(*, prompt: str, options: ClaudeAgentOptions) -> Any:
        seen.append(options)
        for message in messages:
            yield message

    monkeypatch.setattr(claude_agent_sdk, "query", query)
    replay = SimpleNamespace(journal=Journal())
    run = Invocation(runtime=SimpleNamespace(pending=None, replay=replay), tools=[])  # type: ignore[arg-type]
    adapter = ClaudeAdapter()
    native = adapter.prepare_input(None, "go", None)
    options = ClaudeAgentOptions(system_prompt="Mine.")
    items = [i async for i in adapter.stream(options, native, run)]
    assert items[:-1] == ["done"]
    assert isinstance(items[-1], Output) and items[-1].value == messages
    # no context and no harness tools: the options as they were, the permission check and
    # the room for a large tool result added
    [sent] = seen
    assert sent.can_use_tool is not None and sent.resume is None
    assert sent.max_buffer_size == CLI_MESSAGE_BYTES
    assert dataclasses.replace(sent, can_use_tool=None, max_buffer_size=None) == options


# --------------------------------------------------------------------------- LangGraph


def test_the_context_message_leads_every_input_shape() -> None:
    adapter = LangGraphAdapter()
    listed = adapter.prepare_input(None, [{"role": "user", "content": "hi"}], "ctx")
    first = listed["messages"][0]
    assert isinstance(first, SystemMessage) and first.id == CONTEXT_MESSAGE_ID
    state = adapter.prepare_input(
        None, {"messages": [{"role": "user", "content": "hi"}], "n": 1}, "ctx"
    )
    assert state["n"] == 1 and isinstance(state["messages"][0], SystemMessage)
    assert adapter.prepare_input(None, {"topic": "tides"}, "ctx") == {"topic": "tides"}


def test_the_answer_is_the_structured_response_or_the_last_ai_text() -> None:
    adapter = LangGraphAdapter()
    structured = adapter.extract(None, GraphOutput(value={"structured_response": {"n": 1}}))
    assert structured.answer == {"n": 1} and structured.transcript == []
    messages = [AIMessage(content="first"), HumanMessage(content="q"), AIMessage(content="")]
    assert adapter.extract(None, GraphOutput(value={"messages": messages})).answer == "first"
    assert adapter.extract(None, GraphOutput(value={"messages": []})).answer is None
    assert adapter.extract(None, GraphOutput(value=None)).answer is None


async def test_the_stream_ignores_parts_it_does_not_carry() -> None:
    class Graph:
        checkpointer = None

        async def astream(self, *args: Any, **kwargs: Any) -> Any:
            yield {"type": "messages", "data": (AIMessage(content="hel"), {})}
            yield {"type": "messages", "data": (HumanMessage(content="not ours"), {})}
            yield {"type": "custom", "data": "progress"}
            yield {"type": "values", "data": {"messages": []}, "interrupts": ()}

    runtime = SimpleNamespace(thread=None, run_id="run_1")
    run = Invocation(runtime=runtime, tools=[])  # type: ignore[arg-type]
    items = [i async for i in LangGraphAdapter().stream(Graph(), {}, run)]
    assert items[0] == "hel" and len(items) == 2
    assert items[1].value == GraphOutput(value={"messages": []}, interrupts=())


def test_a_graph_without_tool_nodes_binds_no_tools() -> None:
    assert bound_tools(object()) == []
    node = SimpleNamespace(bound=SimpleNamespace(tools_by_name={"a": "tool-a"}))
    assert bound_tools(SimpleNamespace(nodes={"tools": node, "agent": SimpleNamespace()})) == [
        "tool-a"
    ]


# --------------------------------------------------------------------------- OpenAI Agents


def test_the_context_is_the_first_system_message_of_every_input_shape() -> None:
    adapter = OpenAIAgentsAdapter()
    system = {"role": "system", "content": "ctx"}
    assert adapter.prepare_input(None, "hi", None) == "hi"
    assert adapter.prepare_input(None, "hi", "ctx") == [system, {"role": "user", "content": "hi"}]
    assert adapter.prepare_input(None, [{"role": "user", "content": "hi"}], "ctx")[0] == system
    assert adapter.prepare_input(None, {"order": 7}, None) == [
        {"role": "user", "content": '{"order": 7}'}
    ]


def _pending(**fields: Any) -> Pending:
    interrupt = Interrupt(interrupt_id="run_1.1.1", tenant_id="t", run_id="run_1", question="?")
    return Pending(key="k", interrupt=interrupt, **fields)


def test_only_an_sdk_approval_resumes_from_its_run_state() -> None:
    adapter = OpenAIAgentsAdapter()
    approve = InterruptResolution(
        interrupt_id="run_1.1.1", run_id="run_1", decision=InterruptDecision.APPROVE
    )
    reject = approve.model_copy(update={"decision": InterruptDecision.REJECT})
    assert adapter.resume_input(None, ["input"], _pending(), approve) == ["input"]
    state = {"serialised": True}
    pending = _pending(native_id="call_1", native_state=state)
    assert adapter.resume_input(None, ["input"], pending, approve) == _Continue(
        state, "call_1", True
    )
    assert adapter.resume_input(None, ["input"], pending, reject).approve is False


def _resolved(decision: InterruptDecision, **fields: Any) -> InterruptResolution:
    return InterruptResolution(
        interrupt_id="run_1.1.1", run_id="run_1", decision=decision, **fields
    )


def test_an_sdk_approval_maps_every_decision_to_what_the_sdk_takes() -> None:
    adapter = OpenAIAgentsAdapter()
    approval = Interrupt(
        interrupt_id="run_1.1.1",
        tenant_id="t",
        run_id="run_1",
        reason=InterruptReason.APPROVAL,
        question="?",
        tool_call=ToolCall(tool="send", args={"order": "o1"}),
    )
    pending = Pending(key="send", interrupt=approval, native_id="c", native_state={"s": 1})
    edit = adapter.resume_input(
        None, [], pending, _resolved(InterruptDecision.EDIT, payload={"order": "o2"})
    )
    assert (edit.approve, edit.tool, edit.edited) == (False, "send", {"order": "o2"})
    assert edit.message == EDITED.format(tool="send", args='{"order": "o2"}')
    answer = adapter.resume_input(
        None, [], pending, _resolved(InterruptDecision.ANSWER, answer=[1])
    )
    assert (answer.approve, answer.message) == (False, "[1]")
    said = adapter.resume_input(None, [], pending, _resolved(InterruptDecision.ANSWER, answer="x"))
    assert said.message == "x"
    silent = adapter.resume_input(None, [], pending, _resolved(InterruptDecision.ANSWER))
    assert (silent.approve, silent.message) == (False, None)
    reason = adapter.resume_input(
        None, [], pending, _resolved(InterruptDecision.REJECT, answer=" no ")
    )
    assert reason.message == "no" and reason.edited is None
    # an edit with no call to name is a plain reject
    unnamed = _pending(native_id="c", native_state={"s": 1})
    blind = adapter.resume_input(
        None, [], unnamed, _resolved(InterruptDecision.EDIT, payload={"a": 1})
    )
    assert (blind.approve, blind.edited) == (False, None)


def test_a_call_with_unreadable_arguments_is_not_the_edited_one() -> None:
    assert _arguments(SimpleNamespace(arguments="{not json")) is None
    assert _arguments(SimpleNamespace(arguments="")) == {}


HITL_REQUEST: dict[str, Any] = {
    "action_requests": [{"name": "a", "args": {"x": 1}}, {"name": "b", "args": {}}],
    "review_configs": [
        {"action_name": "a", "allowed_decisions": ["approve", "edit", "reject", "respond"]},
        {"action_name": "b", "allowed_decisions": ["approve", "reject", "respond"]},
    ],
}


def test_a_middleware_request_is_recognised_by_its_shape() -> None:
    assert is_hitl(HITL_REQUEST)
    assert not is_hitl({"action_requests": [], "review_configs": []})
    assert not is_hitl({"action_requests": [{"name": "a"}]})
    assert not is_hitl("approve?") and not is_hitl(None)


def test_each_harness_decision_is_one_middleware_decision_per_call() -> None:
    approve = hitl_response(HITL_REQUEST, _resolved(InterruptDecision.APPROVE))
    assert approve == {"decisions": [{"type": "approve"}] * 2}
    edit = hitl_response(HITL_REQUEST, _resolved(InterruptDecision.EDIT, payload={"x": 2}))
    assert edit["decisions"] == [
        {"type": "edit", "edited_action": {"name": "a", "args": {"x": 2}}},
        {"type": "approve"},
    ]
    reject = hitl_response(HITL_REQUEST, _resolved(InterruptDecision.REJECT, answer="why"))
    assert reject["decisions"] == [{"type": "reject", "message": "why"}] * 2
    respond = hitl_response(HITL_REQUEST, _resolved(InterruptDecision.ANSWER, answer={"k": 1}))
    assert respond["decisions"] == [{"type": "respond", "message": '{"k": 1}'}] * 2
    raw = {"decisions": [{"type": "approve"}, {"type": "reject"}]}
    assert hitl_response(HITL_REQUEST, _resolved(InterruptDecision.EDIT, payload=raw)) == raw
    only_b = {**HITL_REQUEST, "review_configs": [HITL_REQUEST["review_configs"][1]] * 2}
    with pytest.raises(ConfigurationError, match="does not allow the decision 'edit'"):
        hitl_response(only_b, _resolved(InterruptDecision.EDIT, payload={"x": 2}))


# --------------------------------------------------------------------------- ReAct


def test_react_puts_its_system_prompt_and_the_context_first() -> None:
    adapter = ReActAdapter()
    target = ReAct(system="You help.", model="m")
    listed = adapter.prepare_input(target, [{"role": "user", "content": "hi"}], "ctx")
    assert listed[0] == {"role": "system", "content": "You help.\n\nctx"}
    structured = adapter.prepare_input(target, {"n": 1}, None)
    assert structured[1] == {"role": "user", "content": '{"n": 1}'}
    pending = _pending()
    answer = InterruptResolution(
        interrupt_id="run_1.1.1", run_id="run_1", decision=InterruptDecision.ANSWER
    )
    assert adapter.resume_input(target, listed, pending, answer) is listed


async def test_a_reply_without_a_message_is_a_model_error(harness: Any) -> None:
    class Broken:
        async def complete(self, messages: Any, **body: Any) -> dict[str, Any]:
            return {"choices": []}

    result = await harness.wrap(ReAct(system="s", model=Broken()), id="broken").run("q", user="u")
    assert result.error is not None and "returned no message" in result.error.message


def test_a_fenced_json_answer_is_unfenced() -> None:
    assert _unfenced('```json\n{"n": 1}\n```') == '{"n": 1}'
    assert _unfenced("```") == ""
    assert _unfenced(' {"n": 1} ') == '{"n": 1}'


def test_react_extracts_the_assistant_text_as_its_transcript() -> None:
    result = ReActResult(
        messages=[
            {"role": "system", "content": "s"},
            {"role": "assistant", "tool_calls": []},
            {"role": "assistant", "content": "done"},
        ],
        answer="done",
    )
    assert ReActAdapter().extract(ReAct(system="s", model="m"), result).transcript == [
        ("assistant", "done")
    ]


def test_the_context_window_is_read_where_a_target_says_it() -> None:
    from types import SimpleNamespace

    from trellis.harness.adapters.base import context_window
    from trellis.harness.clients.memory import context_budget

    assert context_window(SimpleNamespace(context_window=200_000)) == 200_000
    assert context_window(SimpleNamespace(model=SimpleNamespace(max_input_tokens=32_000))) == 32_000
    profiled = SimpleNamespace(model=SimpleNamespace(profile={"max_input_tokens": 1_000_000}))
    assert context_window(profiled) == 1_000_000
    assert context_window(SimpleNamespace(model="gpt-x", context_window=True)) is None
    assert context_window(object()) is None
    assert context_budget(None) == 2000
    assert context_budget(8_000) == 2000  # never less than the default
    assert context_budget(1_000_000) == 8000  # never more than the cap


def test_mcp_servers_given_as_text_or_a_file_are_read() -> None:
    assert configured_servers({"a": {"type": "stdio"}}) == {"a": {"type": "stdio"}}
    assert configured_servers(None) == {} and configured_servers("") == {}
    assert configured_servers('{"mcpServers": {"b": {"type": "http"}}}') == {"b": {"type": "http"}}
    with pytest.raises(ConfigurationError, match="no mcpServers"):
        configured_servers('{"servers": {}}')
