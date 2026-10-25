"""One approval per call: a tool held by the framework's own gate (LangChain's
``HumanInTheLoopMiddleware``, an OpenAI Agents ``needs_approval`` tool) and by governance is
asked once — governance counts the framework's approval, journaled with the run — and a hook's
denial still wins."""

from __future__ import annotations

import dataclasses
import logging
from typing import Any

import pytest
from agents import Agent
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver

from tests.support.arm import native_tools
from tests.support.chat_model import ScriptedChatModel
from tests.support.openai_model import ScriptedModel
from trellis import Deny, Harness, Hooks, Settings, tool
from trellis.contracts import RunStatus
from trellis.harness.agent import gated_tools

sent: list[dict[str, Any]] = []


@tool(side_effects="irreversible")
def wire(to: str, amount: int) -> str:
    """Wire money."""
    sent.append({"to": to, "amount": amount})
    return f"wired {amount} to {to}"


@tool(side_effects="write")
def note(text: str) -> str:
    """Take a note."""
    sent.append({"note": text})
    return "noted"


@pytest.fixture(autouse=True)
def _reset() -> None:
    sent.clear()


async def gated(
    h: Harness, model: Any, *tools: Any, gate: tuple[str, ...] = ("wire",), saver: Any = None
) -> Any:
    return create_agent(
        model,
        tools=await native_tools(h, *tools, framework="langgraph"),
        middleware=[HumanInTheLoopMiddleware(interrupt_on=dict.fromkeys(gate, True))],
        checkpointer=saver or InMemorySaver(),
    )


async def test_a_call_gated_in_both_places_is_asked_once(harness: Harness) -> None:
    model = ScriptedChatModel(turns=[("wire", {"to": "ada", "amount": 5}), "wired"])
    agent = harness.wrap(await gated(harness, model, wire), id="payer")
    paused = await agent.run("pay ada", user="u", thread="t")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
    done = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="lead")
    assert done.status is RunStatus.SUCCESS and done.answer == "wired", done
    assert sent == [{"to": "ada", "amount": 5}]  # run once, asked once


async def test_an_edit_in_the_framework_gate_is_the_call_governance_counts(
    harness: Harness,
) -> None:
    model = ScriptedChatModel(turns=[("wire", {"to": "ada", "amount": 5}), "wired"])
    agent = harness.wrap(await gated(harness, model, wire), id="payer")
    paused = await agent.run("pay ada", user="u", thread="t")
    assert paused.interrupt is not None
    done = await agent.resume(
        paused.interrupt.interrupt_id, "edit", answer={"to": "ada", "amount": 3}, reviewer="l"
    )
    assert done.status is RunStatus.SUCCESS and sent == [{"to": "ada", "amount": 3}]


async def test_a_hook_denies_what_the_framework_gate_approved(harness: Harness) -> None:
    class NoBigWires(Hooks):
        async def before_tool(self, call: Any) -> Any:
            return Deny("over the limit") if call.args.get("amount", 0) > 1 else None

    model = ScriptedChatModel(turns=[("wire", {"to": "ada", "amount": 5}), "not wired"])
    agent = harness.wrap(await gated(harness, model, wire), id="payer", hooks=[NoBigWires()])
    paused = await agent.run("pay ada", user="u", thread="t")
    assert paused.interrupt is not None
    done = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="lead")
    assert done.answer == "not wired" and sent == []  # the stricter decision wins


async def test_a_tool_gated_only_by_governance_is_asked_as_before(harness: Harness) -> None:
    model = ScriptedChatModel(turns=[("wire", {"to": "ada", "amount": 5}), "wired"])
    agent = harness.wrap(await gated(harness, model, wire, gate=("other",)), id="payer")
    paused = await agent.run("pay ada", user="u", thread="t")
    assert paused.interrupt is not None and paused.interrupt.question.startswith("Approve wire?")
    done = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="lead")
    assert done.answer == "wired" and sent == [{"to": "ada", "amount": 5}]


async def test_a_tool_gated_only_by_the_framework_is_one_approval(harness: Harness) -> None:
    model = ScriptedChatModel(turns=[("note", {"text": "a"}), "noted"])
    agent = harness.wrap(await gated(harness, model, note, gate=("note",)), id="noter")
    paused = await agent.run("note a", user="u", thread="t")
    assert paused.interrupt is not None
    done = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="lead")
    assert done.answer == "noted" and sent == [{"note": "a"}]


async def test_a_resume_in_another_process_still_asks_once() -> None:
    model = ScriptedChatModel(turns=[("wire", {"to": "ada", "amount": 5}), "wired"])
    saver = InMemorySaver()  # the graph's own checkpoints, where both processes reach them
    async with Harness(config=Settings()) as first:
        graph = await gated(first, model, wire, saver=saver)
        paused = await first.wrap(graph, id="payer").run("pay ada", user="u", thread="t")
        assert paused.interrupt is not None
        async with Harness(config=Settings(), runs=first.runs) as second:
            graph = await gated(second, model, wire, saver=saver)
            agent = second.wrap(graph, id="payer")
            done = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="l")
    assert done.answer == "wired" and sent == [{"to": "ada", "amount": 5}]


async def test_an_sdk_approval_counts_for_governance(harness: Harness) -> None:
    tools = await native_tools(harness, wire, framework="openai_agents")
    wired = next(t for t in tools if t.name == "wire")
    model = ScriptedModel([("wire", {"to": "ada", "amount": 5}), "wired"])
    target = Agent(name="s", model=model, tools=[dataclasses.replace(wired, needs_approval=True)])
    agent = harness.wrap(target, id="sdk")
    paused = await agent.run("pay ada", user="u")
    assert paused.interrupt is not None
    done = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="lead")
    assert done.status is RunStatus.SUCCESS and done.answer == "wired", done
    assert sent == [{"to": "ada", "amount": 5}]


async def test_the_tools_both_gate_are_said_at_wrap(
    harness: Harness, caplog: pytest.LogCaptureFixture
) -> None:
    graph = await gated(harness, ScriptedChatModel(turns=[]), wire)
    assert gated_tools(graph) == {"wire"}
    with caplog.at_level(logging.WARNING):
        harness.wrap(graph, id="said")
    assert any("wire is approved in the framework's own gate" in r.message for r in caplog.records)
    assert gated_tools(object()) == set()
