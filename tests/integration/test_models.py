"""Model instrumentation (§15, §16, §17, §73).

No LLM provider is wired into the harness yet — the gateway (Bifrost) comes later — so there
is no live model to call. What stands in is not a mock of one: ``DeterministicModel`` is a
real implementation of the model port that really executes, returns a real usage payload and
really streams. Everything under test here — the span, the meter, the timeout, the error
classification — is the harness's own code running for real against it.
"""

from __future__ import annotations

import asyncio

import pytest
from tests.support import span_by_name, span_names

from universal_agent_harness import AgentHarness, ModelError, ModelRequest, ModelUsage
from universal_agent_harness.contracts.errors import ConfigurationError


class DeterministicModel:
    """A provider-shaped object: ``ainvoke`` plus a usage-carrying response."""

    def __init__(self, text: str = "answer") -> None:
        self.text = text
        self.calls: list[object] = []

    async def ainvoke(self, prompt, **kwargs):
        self.calls.append(prompt)
        return {
            "text": self.text,
            "model": "test-model-1",
            "usage": {"prompt_tokens": 11, "completion_tokens": 7, "cost_usd": 0.0004},
        }

    async def astream(self, prompt, **kwargs):
        for chunk in ("an", "sw", "er"):
            await asyncio.sleep(0)
            yield chunk


async def test_model_call_is_traced_and_metered(memory, context, spans):
    model = DeterministicModel()
    harness = AgentHarness(memory=memory, model=model, defaults={"tenant_id": "acme"})

    async def agent(payload, runtime):
        response = await runtime.model.invoke("how much stock?")
        assert response.text == "answer"
        assert response.usage == ModelUsage(input_tokens=11, output_tokens=7, cost_usd=0.0004)
        return response.text

    result = await harness.wrap(agent, agent_id="inv")(None, context=context)
    assert result.metrics["total_tokens"] == 18
    assert result.metrics["cost_usd"] == pytest.approx(0.0004)

    span = span_by_name(spans, "agent.model.invoke")
    assert span.attributes["gen_ai.usage.input_tokens"] == 11
    assert span.attributes["gen_ai.usage.output_tokens"] == 7
    assert span.attributes["gen_ai.response.model"] == "test-model-1"
    assert span.attributes["duration_ms"] > 0


async def test_prompts_are_not_captured_by_default(memory, context, spans):
    harness = AgentHarness(memory=memory, model=DeterministicModel(), defaults={"tenant_id": "acme"})

    async def agent(payload, runtime):
        return (await runtime.model.invoke("patient record: John Doe")).text

    await harness.wrap(agent, agent_id="inv")(None, context=context)
    span = span_by_name(spans, "agent.model.invoke")
    assert "John Doe" not in str(dict(span.attributes))


async def test_prompts_are_captured_when_explicitly_enabled(memory, context, spans):
    harness = AgentHarness(
        memory=memory,
        model=DeterministicModel(),
        defaults={"tenant_id": "acme"},
        config={"telemetry": {"capture": {"inputs": True, "outputs": True}}},
    )

    async def agent(payload, runtime):
        return (await runtime.model.invoke("visible prompt")).text

    await harness.wrap(agent, agent_id="inv")(None, context=context)
    span = span_by_name(spans, "agent.model.invoke")
    assert "visible prompt" in span.attributes["input.value"]
    assert "answer" in span.attributes["output.value"]


async def test_model_request_metadata_reaches_the_span(memory, context, spans):
    harness = AgentHarness(memory=memory, model=DeterministicModel(), defaults={"tenant_id": "acme"})

    async def agent(payload, runtime):
        request = ModelRequest(
            prompt="q", model="gpt-x", provider="acme-ai", prompt_id="inventory/v3",
            prompt_version="3",
        )
        return (await runtime.model.invoke(request)).text

    await harness.wrap(agent, agent_id="inv")(None, context=context)
    span = span_by_name(spans, "agent.model.invoke")
    assert span.attributes["gen_ai.request.model"] == "gpt-x"
    assert span.attributes["gen_ai.system"] == "acme-ai"
    assert span.attributes["gen_ai.prompt.id"] == "inventory/v3"


async def test_streaming_records_time_to_first_token_without_buffering(memory, context, spans):
    harness = AgentHarness(memory=memory, model=DeterministicModel(), defaults={"tenant_id": "acme"})
    received: list[str] = []

    async def agent(payload, runtime):
        async for chunk in runtime.model.stream("q"):
            received.append(chunk)
        return "".join(received)

    result = await harness.wrap(agent, agent_id="inv")(None, context=context)
    assert result.data == "answer"
    span = span_by_name(spans, "agent.model.invoke")
    assert span.attributes["gen_ai.request.streaming"] is True
    assert span.attributes["chunks"] == 3
    assert span.attributes["gen_ai.response.time_to_first_token_ms"] >= 0
    assert any(event.name == "first_token" for event in span.events)


async def test_model_failures_are_normalized(memory, context, spans):
    class Failing:
        async def ainvoke(self, prompt, **kwargs):
            raise ConnectionError("provider down")

    harness = AgentHarness(memory=memory, model=Failing(), defaults={"tenant_id": "acme"})

    async def agent(payload, runtime):
        with pytest.raises(ModelError, match="provider down"):
            await runtime.model.invoke("q")
        return "handled"

    await harness.wrap(agent, agent_id="inv")(None, context=context)
    assert span_by_name(spans, "agent.model.invoke").attributes["status"] == "error"


async def test_unconfigured_model_fails_with_guidance(harness, context):
    async def agent(payload, runtime):
        with pytest.raises(ConfigurationError, match="no model client is configured"):
            await runtime.model.invoke("q")
        return "ok"

    await harness.wrap(agent, agent_id="inv")(None, context=context)


async def test_plain_callables_work_as_models(memory, context):
    async def call_model(prompt):
        return f"echo: {prompt}"

    harness = AgentHarness(memory=memory, model=call_model, defaults={"tenant_id": "acme"})

    async def agent(payload, runtime):
        return (await runtime.model.invoke("hi")).text

    assert (await harness.wrap(agent, agent_id="inv")(None, context=context)).data == "echo: hi"


async def test_model_deadline_is_bounded_by_the_agent_deadline(memory, context):
    class Slow:
        async def ainvoke(self, prompt, **kwargs):
            await asyncio.sleep(5)
            return "never"

    harness = AgentHarness(
        memory=memory,
        model=Slow(),
        defaults={"tenant_id": "acme"},
        config={"timeouts": {"default_seconds": 0.1, "model_seconds": 30}},
    )

    async def agent(payload, runtime):
        await runtime.model.invoke("q")
        return "should not get here"

    from universal_agent_harness import AgentTimeoutError

    with pytest.raises((AgentTimeoutError, ModelError)):
        await harness.wrap(agent, agent_id="inv")(None, context=context)


async def test_usage_extraction_handles_missing_usage():
    assert ModelUsage.extract({"text": "x"}) is None
    assert ModelUsage.extract({"usage": {"input_tokens": 3}}).input_tokens == 3
    assert ModelUsage.extract({"usage": {"prompt_tokens": 5, "completion_tokens": 2}}).tokens == 7


async def test_uninstrumented_model_calls_are_simply_not_traced(harness, context, spans):
    """Bypassing the runtime is allowed — and honestly reported as uninstrumented (§13)."""

    async def agent(payload, runtime):
        return await DeterministicModel().ainvoke("direct call")

    await harness.wrap(agent, agent_id="inv")(None, context=context)
    assert "agent.model.invoke" not in span_names(spans)
    assert "agent.run" in span_names(spans)
