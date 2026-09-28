"""What a run puts on its event stream: text, tools, steps, context, results, envelopes.

Every behaviour here is one a UI renders; each test would fail if the emission were removed.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from trellis.contracts import (
    AgentExecutionContext,
    ErrorCategory,
    HarnessError,
    RunEventType,
    RunOutcome,
    ToolError,
)
from trellis.contracts.model import ModelResponse

from trellis.harness import (
    AgentHarness,
    CallablePolicyProvider,
    CollectingEventSink,
    ContextAssembler,
    LocalToolClient,
    RetryPolicy,
    react,
)
from trellis.harness.config.settings import RetryConfig
from trellis.harness.telemetry.redaction import REDACTED

TENANT = "acme"


class StreamingModel:
    async def ainvoke(self, prompt, **kwargs):
        return {"text": "answer", "model": "m"}

    async def astream(self, prompt, **kwargs):
        for chunk in ("Hel", "lo"):
            await asyncio.sleep(0)
            yield chunk


def _harness(sink: CollectingEventSink, **options: Any) -> AgentHarness:
    return AgentHarness(memory=None, event_sinks=[sink], defaults={"tenant_id": TENANT}, **options)


def _events(sink: CollectingEventSink, run_id: str, *types: RunEventType) -> list[Any]:
    return [e for e in sink.for_run(run_id) if not types or e.type in types]


async def test_a_streamed_answer_is_one_message_of_deltas_and_an_invoke_is_one_of_text() -> None:
    sink = CollectingEventSink()
    harness = _harness(sink, model=StreamingModel())

    async def agent(payload: Any, runtime: Any) -> str:
        chunks = [c async for c in runtime.model.stream("q")]
        response = await runtime.model.invoke("q")
        return "".join(chunks) + "|" + (response.text or "")

    result = await harness.run(agent, None, agent_id="talker")
    assert result.data == "Hello|answer"
    run_id = sink.events[0].run_id
    text = _events(
        sink,
        run_id,
        RunEventType.TEXT_MESSAGE_START,
        RunEventType.TEXT_MESSAGE_CONTENT,
        RunEventType.TEXT_MESSAGE_END,
    )
    shape = [(e.type.value, e.message_id, e.data.get("delta")) for e in text]
    first = text[0].message_id
    second = text[-1].message_id
    assert first != second and first.startswith("msg_") and second.startswith("msg_")
    assert shape == [
        ("TEXT_MESSAGE_START", first, None),
        ("TEXT_MESSAGE_CONTENT", first, "Hel"),
        ("TEXT_MESSAGE_CONTENT", first, "lo"),
        ("TEXT_MESSAGE_END", first, None),
        ("TEXT_MESSAGE_START", second, None),
        ("TEXT_MESSAGE_CONTENT", second, "answer"),
        ("TEXT_MESSAGE_END", second, None),
    ]


async def test_a_structured_tool_result_is_json_on_the_stream_and_redacted() -> None:
    sink = CollectingEventSink()

    def lookup(order_id: int, api_key: str = "") -> dict[str, Any]:
        return {"order_id": order_id, "ok": True, "api_key": "sk-live-1234567890abcdef"}

    harness = _harness(sink, tools=LocalToolClient({"lookup": lookup}))

    async def agent(payload: Any, runtime: Any) -> Any:
        return (await runtime.tools.call("lookup", order_id=91, api_key="sk-x-abcdefghijkl")).output

    await harness.run(agent, None, agent_id="looker")
    run_id = sink.events[0].run_id
    args = _events(sink, run_id, RunEventType.TOOL_CALL_ARGS)[0].data["args"]
    assert args == {"order_id": 91, "api_key": REDACTED}
    result = _events(sink, run_id, RunEventType.TOOL_CALL_RESULT)[0].data
    assert result["status"] == "ok"
    assert result["output"] == {"order_id": 91, "ok": True, "api_key": REDACTED}  # JSON, not repr
    finished = _events(sink, run_id, RunEventType.RUN_FINISHED)[0]
    assert finished.data["result"]["api_key"] == REDACTED


async def test_a_failing_tool_still_closes_its_call_on_the_stream() -> None:
    sink = CollectingEventSink()

    async def flaky() -> None:
        raise RuntimeError("upstream exploded")

    harness = _harness(sink, tools=[flaky])

    async def agent(payload: Any, runtime: Any) -> str:
        with pytest.raises(ToolError):
            await runtime.tools.call("flaky")
        return "handled"

    await harness.run(agent, None, agent_id="tolerant")
    run_id = sink.events[0].run_id
    tool_events = _events(
        sink,
        run_id,
        RunEventType.TOOL_CALL_START,
        RunEventType.TOOL_CALL_ARGS,
        RunEventType.TOOL_CALL_END,
        RunEventType.TOOL_CALL_RESULT,
    )
    assert [e.type.value for e in tool_events] == [
        "TOOL_CALL_START",
        "TOOL_CALL_ARGS",
        "TOOL_CALL_END",
        "TOOL_CALL_RESULT",
    ]
    assert len({e.tool_call_id for e in tool_events}) == 1
    result = tool_events[-1].data
    assert result["status"] == "error" and result["error_class"] == "RuntimeError"
    assert result["output"] == "upstream exploded"


async def test_a_retry_is_a_new_attempt_on_the_stream() -> None:
    sink = CollectingEventSink()
    harness = _harness(sink)
    attempts: list[int] = []

    class Flaky(HarnessError):
        code = "UPSTREAM"
        category = ErrorCategory.DEPENDENCY
        retryable = True

    async def agent(payload: Any, runtime: Any) -> str:
        attempts.append(1)
        if len(attempts) == 1:
            raise Flaky("first try fails")
        return "second try"

    policy = RetryPolicy(
        RetryConfig(enabled=True, max_attempts=2, backoff_seconds=0.0), idempotent=True
    )
    result = await harness.run(agent, None, agent_id="retrier", retry=policy)
    assert result.data == "second try" and len(attempts) == 2
    run_id = sink.events[0].run_id
    shape = [(e.attempt, e.sequence, e.type.value) for e in sink.for_run(run_id)]
    assert shape == [
        (1, 0, "RUN_STARTED"),
        (1, 1, "STEP_STARTED"),
        (1, 2, "STEP_FINISHED"),
        (2, 0, "STEP_STARTED"),
        (2, 1, "STEP_FINISHED"),
        (2, 2, "RUN_FINISHED"),
    ]


async def test_the_requests_own_webhook_url_rides_on_the_notified_events() -> None:
    sink = CollectingEventSink()
    harness = _harness(sink)

    async def agent(payload: Any, runtime: Any) -> str:
        return "ok"

    await harness.run(
        agent, None, agent_id="notify", metadata={"webhook_url": "https://own.example/h"}
    )
    run_id = sink.events[0].run_id
    by_type = {e.type: e.data.get("webhook_url") for e in sink.for_run(run_id)}
    assert by_type[RunEventType.RUN_STARTED] == "https://own.example/h"
    assert by_type[RunEventType.RUN_FINISHED] == "https://own.example/h"
    assert by_type[RunEventType.STEP_STARTED] is None


async def test_an_execution_block_is_a_run_on_the_stream() -> None:
    sink = CollectingEventSink()
    harness = _harness(sink, tools=LocalToolClient({"double": lambda n: n * 2}))
    ctx = AgentExecutionContext.create(tenant_id=TENANT, agent_id="block", turn_id="t1")
    async with harness.execution(ctx, agent_id="block", input="go") as runtime:
        outcome = await runtime.tools.call("double", n=21)
        runtime.state["result"] = {"answer": outcome.output}
    types = sink.types(ctx.agent_run_id)
    assert types[0] is RunEventType.RUN_STARTED and types[-1] is RunEventType.RUN_FINISHED
    assert RunEventType.TOOL_CALL_RESULT in types
    finished = sink.for_run(ctx.agent_run_id)[-1]
    assert finished.outcome is RunOutcome.SUCCESS and finished.data["result"] == {"answer": 42}

    failing = AgentExecutionContext.create(tenant_id=TENANT, agent_id="block", turn_id="t2")
    with pytest.raises(RuntimeError):
        async with harness.execution(failing, agent_id="block", input="go"):
            raise RuntimeError("inside the block")
    failed = sink.for_run(failing.agent_run_id)[-1]
    assert failed.type is RunEventType.RUN_FINISHED and failed.outcome is RunOutcome.ERROR
    assert failed.error is not None and "inside the block" in failed.error.message


class ScriptedModel:
    def __init__(self, *replies: ModelResponse) -> None:
        self._replies = list(replies)
        self.requests: list[Any] = []

    async def invoke(self, request: Any, /, **kwargs: Any) -> ModelResponse:
        self.requests.append(request)
        return self._replies.pop(0)

    async def structured(self, request: Any, /, schema: Any, **kwargs: Any) -> ModelResponse:
        return await self.invoke(request, **kwargs)


async def test_react_compacts_over_budget_and_the_summary_is_not_announced_as_an_answer() -> None:
    sink = CollectingEventSink()
    tools = LocalToolClient({"lookup": lambda: "stock is 42 units across three warehouses"})
    call = ModelResponse(
        text=None,
        model="m",
        tool_calls=[
            {"id": "a", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
        ],
    )
    model = ScriptedModel(
        call,
        ModelResponse(text="The user asked about stock; lookup answered.", model="m"),  # summary
        ModelResponse(text="42 units", model="m"),
    )
    harness = _harness(sink, model=model, tools=tools)
    assembler = ContextAssembler("prompt", budget_tokens=25, keep_recent=2)

    async def agent(question: str, runtime: Any) -> Any:
        return await react(runtime, question, max_steps=3, assembler=assembler)

    result = await harness.run(agent, "how much stock?", agent_id="reactor")
    assert result.data == "42 units" and assembler.compactions == 1
    assert model.requests[1].metadata == {"internal": "compaction"}
    assert model.requests[2].messages[0]["content"] == "prompt"  # the assembler's prompt
    assert any("Summary of the conversation" in m["content"] for m in model.requests[2].messages)
    run_id = sink.events[0].run_id
    starts = _events(sink, run_id, RunEventType.TEXT_MESSAGE_START)
    assert len(starts) == 1  # the answer; the summary the harness asked for is not a message


async def test_context_loaded_precedes_the_first_step(memory, context) -> None:
    sink = CollectingEventSink()
    harness = AgentHarness(memory=memory, event_sinks=[sink], defaults={"tenant_id": TENANT})

    async def agent(payload: Any, runtime: Any) -> str:
        return "ok"

    await harness.wrap(agent, agent_id="test-agent")("how much stock?", context=context)
    types = sink.types(context.agent_run_id)
    assert types.index(RunEventType.CONTEXT_LOADED) < types.index(RunEventType.STEP_STARTED)
    loaded = _events(sink, context.agent_run_id, RunEventType.CONTEXT_LOADED)[0].data
    assert loaded["has_context"] is True and "facts" in loaded


async def test_a_dead_memory_service_reports_no_context_and_the_run_goes_on(
    dead_memory, context
) -> None:
    sink = CollectingEventSink()
    harness = AgentHarness(memory=dead_memory, event_sinks=[sink], defaults={"tenant_id": TENANT})

    async def agent(payload: Any, runtime: Any) -> str:
        return "ok"

    result = await harness.wrap(agent, agent_id="test-agent")("how much stock?", context=context)
    assert result.data == "ok"
    loaded = _events(sink, context.agent_run_id, RunEventType.CONTEXT_LOADED)[0].data
    assert loaded["has_context"] is False


async def test_memory_tools_are_offered_only_when_asked_for(memory, context) -> None:
    async def agent(payload: Any, runtime: Any) -> list[str]:
        return sorted(s.name for s in await runtime.tools.list_tools())

    plain = AgentHarness(memory=memory, defaults={"tenant_id": TENANT})
    assert (await plain.wrap(agent, agent_id="test-agent")(None, context=context)).data == []
    offered = AgentHarness(
        memory=memory, defaults={"tenant_id": TENANT}, config={"memory": {"as_tools": True}}
    )
    names = (await offered.wrap(agent, agent_id="test-agent")(None, context=context)).data
    assert names == ["memory.recall", "memory.remember"]

    async def recalling(payload: Any, runtime: Any) -> Any:
        outcome = await runtime.tools.call("memory.recall", query="stock levels")
        assert outcome.status == "ok"
        remembered = await runtime.tools.call(
            "memory.remember", content="the warehouse closes at 18:00", kind="fact"
        )
        return remembered.status

    assert (
        await offered.wrap(recalling, agent_id="test-agent")(None, context=context)
    ).data == "ok"
    await offered.drain()
    assert memory.of("recall")[-1]["query"] == "stock levels"
    written = [o for o in memory.observations if "warehouse closes" in o["content"]]
    assert written and written[-1]["source"] == "memory.remember"  # provenance travels


async def test_an_approval_is_recorded_as_feedback(memory, context) -> None:
    refunds: list[int] = []

    def refund(amount: int) -> dict[str, Any]:
        refunds.append(amount)
        return {"refunded": amount}

    sink = CollectingEventSink()
    harness = AgentHarness(
        memory=memory,
        tools=LocalToolClient({"refund": refund}),
        policy=CallablePolicyProvider(tool=lambda c, call: "require_approval"),
        event_sinks=[sink],
        defaults={"tenant_id": TENANT},
        error_mode="raise",
    )

    async def agent(payload: Any, runtime: Any) -> Any:
        return (await runtime.tools.call("refund", amount=240)).output

    from trellis.contracts import InterruptDecision, InterruptResolution

    from trellis.harness.interrupts import ApprovalRequired

    wrapped = harness.wrap(agent, agent_id="test-agent")
    with pytest.raises(ApprovalRequired):
        await wrapped("refund", context=context)
    interrupt = harness.resolutions.announced(TENANT)[-1]
    edit = InterruptResolution(
        interrupt_id=interrupt.interrupt_id,
        run_id=interrupt.run_id,
        decision=InterruptDecision.EDIT,
        payload={"amount": 200},
        reviewer="reviewer-1",
    )
    result = await harness.resume(interrupt, edit, context=context, agent=wrapped)
    assert result.data == {"refunded": 200} and refunds == [200]
    await harness.drain()
    submitted = memory.of("feedback.submit")[-1]
    assert submitted["target_kind"] == "tool_call" and submitted["verdict"] == "edit"
    assert submitted["source"] == "interrupt" and submitted["correction"] == {"amount": 200}
    assert submitted["reviewer"] == "reviewer-1"
    assert submitted["metadata"]["interrupt_id"] == interrupt.interrupt_id
