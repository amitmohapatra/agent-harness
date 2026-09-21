"""The ReAct loop, driven by a scripted model.

ReAct is four lines of control flow and a great deal of care about when to stop, so every
test here is about a stopping condition or an observation, not about "does it call a tool".
The model is scripted rather than mocked at the transport: what matters is how the loop
reacts to what a model says, and a real gateway cannot be made to say these things on cue.
"""

from __future__ import annotations

from typing import Any

import pytest
from universal_agent_contracts.model import ModelResponse
from universal_agent_contracts.tool import ToolSpec

from universal_agent_harness import AgentHarness
from universal_agent_harness.reasoning import react


class ScriptedModel:
    """Answers with whatever the script says next, and records what it was asked.

    Implements ``invoke`` **and** ``structured`` deliberately: ``AgentHarness`` passes a model
    straight through only when it has both, and otherwise wraps it in ``DirectModelClient``,
    which calls the target with the bare message list and drops ``tools``. A fake with only
    ``invoke`` would therefore never see the tool schemas, and every assertion about what the
    loop offered the model would be vacuously true.
    """

    def __init__(self, *replies: ModelResponse) -> None:
        self._replies = list(replies)
        self.requests: list[Any] = []

    async def invoke(self, request: Any, /, **kwargs: Any) -> ModelResponse:
        self.requests.append(request)
        if not self._replies:
            # The loop asked for more than the script provides, which is itself a finding:
            # a test that silently repeated the last answer would hide a runaway loop.
            raise AssertionError("the loop made more model calls than the script allows")
        return self._replies.pop(0)

    async def structured(self, request: Any, /, schema: Any, **kwargs: Any) -> ModelResponse:
        return await self.invoke(request, **kwargs)


def says(text: str) -> ModelResponse:
    """A model turn with no tool call: the answer."""
    return ModelResponse(text=text, finish_reason="stop")


def calls(tool: str, **arguments: Any) -> ModelResponse:
    """A model turn asking for a tool."""
    import json

    return ModelResponse(
        text=None,
        finish_reason="tool_calls",
        tool_calls=[
            {
                "id": "c1",
                "type": "function",
                "function": {"name": tool, "arguments": json.dumps(arguments)},
            }
        ],
    )


async def stock(sku: str) -> dict[str, Any]:
    """On-hand units for a SKU."""
    return {"sku": sku, "on_hand": 7}


async def explodes(sku: str) -> dict[str, Any]:
    """A tool that fails the way a real one does — bad argument, not a crash in the loop."""
    raise ValueError(f"unknown sku {sku!r}")


def build(model: ScriptedModel, tools: list[Any] | None = None) -> AgentHarness:
    return AgentHarness(
        model=model,
        tools=tools if tools is not None else [stock],
        defaults={"tenant_id": "acme", "user_id": "u1"},
        config={"memory": {"enabled": False}},
    )


async def run(harness: AgentHarness, question: str, **kwargs: Any):
    """Run the loop inside a real execution, so the runtime is the instrumented one."""
    async with harness.execution(agent_id="researcher") as runtime:
        response = await react(runtime, question, **kwargs)
        return response, runtime


# ----------------------------------------------------------------- stopping


async def test_a_model_that_asks_for_no_tool_answers_immediately() -> None:
    """One call, one answer. A loop that always takes a tool step pays for reasoning the
    question never needed."""
    model = ScriptedModel(says("seven units"))
    response, runtime = await run(build(model), "how much stock?")

    assert response.data == "seven units"
    assert len(model.requests) == 1
    trace = runtime.state["react"]
    assert trace.stopped_because == "answered"
    assert trace.tool_names == []


async def test_a_tool_result_comes_back_as_an_observation_and_is_answered() -> None:
    model = ScriptedModel(calls("stock", sku="SKU-1"), says("seven on hand"))
    response, runtime = await run(build(model), "how much stock of SKU-1?")

    assert response.data == "seven on hand"
    trace = runtime.state["react"]
    assert trace.tool_names == ["stock"]
    assert "on_hand" in (trace.steps[0].observation or "")
    assert trace.steps[0].failed is False
    # the observation has to reach the model, or the second turn is reasoning blind
    second = model.requests[1]
    assert any("Observation:" in str(m.get("content", "")) for m in second.messages)


async def test_a_failing_tool_is_an_observation_not_a_failure() -> None:
    """The whole point of ReAct. A loop that aborts on the first bad argument is a chain
    with extra steps — the model has to see the error and get a chance to adapt."""
    model = ScriptedModel(
        calls("explodes", sku="NOPE"),
        calls("stock", sku="SKU-1"),
        says("seven, after a wrong guess"),
    )
    response, runtime = await run(build(model, tools=[stock, explodes]), "stock?")

    trace = runtime.state["react"]
    assert trace.tool_names == ["explodes", "stock"]
    assert trace.steps[0].failed is True
    assert "unknown sku" in (trace.steps[0].observation or "")
    assert response.data == "seven, after a wrong guess"


async def test_the_loop_is_bounded_by_max_steps() -> None:
    """A model that never stops asking must not hold the execution open forever."""
    model = ScriptedModel(
        calls("stock", sku="A"),
        calls("stock", sku="B"),
        says("giving you what I have"),  # the forced final answer
    )
    response, runtime = await run(build(model), "stock?", max_steps=2)

    trace = runtime.state["react"]
    assert len(trace.steps) == 2
    assert trace.stopped_because == "max_steps"
    # out of steps with observations in hand: ask once more rather than return nothing
    assert response.data == "giving you what I have"


async def test_the_final_answer_call_withholds_tools() -> None:
    """Asking again with tools still offered invites another tool call and the loop never
    terminates."""
    model = ScriptedModel(calls("stock", sku="A"), says("done"))
    await run(build(model), "stock?", max_steps=1)

    final = model.requests[-1]
    assert not final.tools


async def test_max_steps_must_be_at_least_one() -> None:
    model = ScriptedModel(says("unused"))
    with pytest.raises(ValueError, match="at least 1"):
        await run(build(model), "q", max_steps=0)


# ----------------------------------------------------------------- what reaches the model


async def test_the_tools_offered_are_the_runtime_s_own() -> None:
    """The loop must not invent a tool list: what it advertises has to be what the harness
    will actually execute, or the model is told about tools that do not exist."""
    model = ScriptedModel(says("fine"))
    await run(build(model), "q")

    offered = model.requests[0].tools
    assert [t["function"]["name"] for t in offered] == ["stock"]
    assert offered[0]["function"]["description"]


async def test_an_explicit_tool_list_overrides_discovery() -> None:
    model = ScriptedModel(says("fine"))
    spec = ToolSpec(name="only_this", description="a narrowed tool")
    async with build(model).execution(agent_id="r") as runtime:
        await react(runtime, "q", tools=[spec])
    assert [t["function"]["name"] for t in model.requests[0].tools] == ["only_this"]


async def test_no_tools_at_all_is_not_an_error() -> None:
    """An agent with no tools is a legitimate configuration; the loop degrades to one call."""
    model = ScriptedModel(says("from what I know"))
    response, _runtime = await run(build(model, tools=[]), "q")
    assert response.data == "from what I know"
    assert not model.requests[0].tools


async def test_the_system_prompt_is_replaceable() -> None:
    model = ScriptedModel(says("ok"))
    async with build(model).execution(agent_id="r") as runtime:
        await react(runtime, "q", system="You are terse.")
    first = model.requests[0].messages[0]
    assert first == {"role": "system", "content": "You are terse."}


# ----------------------------------------------------------------- observability


async def test_each_step_is_a_span_under_the_agent_run(spans) -> None:
    """Without this the reasoning trace is invisible in Langfuse and OTel, and the only way
    to see why an agent took six steps is to add print statements."""
    from tests.support import span_names

    model = ScriptedModel(calls("stock", sku="A"), says("seven"))
    await run(build(model), "q")

    names = span_names(spans)
    assert names.count("agent.react.step") == 2
    assert "agent.run" in names
