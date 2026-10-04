"""LangChain's ``HumanInTheLoopMiddleware`` and Deep Agents' ``interrupt_on``: their own pause
answered with the harness's decisions (approve, edit, reject, answer), the real middleware
driven by a scripted chat model."""

from __future__ import annotations

from typing import Any

import pytest
from deepagents import create_deep_agent
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, InterruptOnConfig
from langchain_core.messages import ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from tests.support.chat_model import ScriptedChatModel
from trellis import Harness, tool
from trellis.contracts import ConfigurationError, InterruptReason, RunStatus

sent: list[dict[str, Any]] = []


@tool(side_effects="write")
def email(to: str, body: str) -> str:
    """Send an email."""
    sent.append({"to": to, "body": body})
    return f"sent to {to}"


@pytest.fixture(autouse=True)
def _reset() -> None:
    sent.clear()


async def mailer(
    harness: Harness, model: ScriptedChatModel, review: InterruptOnConfig | None = None
) -> Any:
    graph = create_agent(
        model,
        tools=await harness.tools(email, framework="langgraph"),
        middleware=[HumanInTheLoopMiddleware(interrupt_on={"email": review or True})],
        checkpointer=InMemorySaver(),
    )
    return harness.wrap(graph, id="mailer")


def tool_messages(model: ScriptedChatModel) -> list[ToolMessage]:
    return [m for m in model.seen[-1] if isinstance(m, ToolMessage)]


async def test_the_middleware_pause_is_an_approval_of_the_tool_call(harness: Harness) -> None:
    model = ScriptedChatModel(turns=[("email", {"to": "ada", "body": "hi"}), "sent"])
    agent = await mailer(harness, model)
    paused = await agent.run("email ada", user="u1", thread="t1")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
    interrupt = paused.interrupt
    assert interrupt.reason is InterruptReason.APPROVAL and interrupt.ui == "approve"
    assert interrupt.question == "Approve email?"
    assert interrupt.tool_call is not None
    assert (interrupt.tool_call.tool, interrupt.tool_call.args) == (
        "email",
        {"to": "ada", "body": "hi"},
    )
    assert interrupt.payload is not None
    assert interrupt.payload["review_configs"][0]["allowed_decisions"] == [
        "approve",
        "edit",
        "reject",
        "respond",
    ]
    assert sent == []


async def test_approve_runs_the_call(harness: Harness) -> None:
    model = ScriptedChatModel(turns=[("email", {"to": "ada", "body": "hi"}), "sent"])
    agent = await mailer(harness, model)
    paused = await agent.run("email ada", user="u1", thread="t1")
    assert paused.interrupt is not None
    done = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="lead")
    assert done.status is RunStatus.SUCCESS and done.answer == "sent"
    assert sent == [{"to": "ada", "body": "hi"}]


async def test_edit_runs_the_call_with_the_edited_arguments(harness: Harness) -> None:
    model = ScriptedChatModel(turns=[("email", {"to": "ada", "body": "hi"}), "sent"])
    agent = await mailer(harness, model)
    paused = await agent.run("email ada", user="u1", thread="t1")
    assert paused.interrupt is not None
    done = await agent.resume(
        paused.interrupt.interrupt_id, "edit", answer={"to": "ada", "body": "hello"}, reviewer="l"
    )
    assert done.status is RunStatus.SUCCESS
    assert sent == [{"to": "ada", "body": "hello"}]


async def test_reject_tells_the_model_the_reason(harness: Harness) -> None:
    model = ScriptedChatModel(turns=[("email", {"to": "ada", "body": "hi"}), "not sent"])
    agent = await mailer(harness, model)
    paused = await agent.run("email ada", user="u1", thread="t1")
    assert paused.interrupt is not None
    done = await agent.resume(
        paused.interrupt.interrupt_id, "reject", answer="wrong recipient", reviewer="l"
    )
    assert done.status is RunStatus.SUCCESS and sent == []
    [message] = tool_messages(model)
    assert "wrong recipient" in str(message.content) and message.status == "error"


async def test_an_answer_responds_for_the_tool(harness: Harness) -> None:
    model = ScriptedChatModel(turns=[("email", {"to": "ada", "body": "hi"}), "ok"])
    agent = await mailer(harness, model)
    paused = await agent.run("email ada", user="u1", thread="t1")
    assert paused.interrupt is not None
    await agent.resume(
        paused.interrupt.interrupt_id, "answer", answer="I'll call her", reviewer="l"
    )
    assert sent == []
    [message] = tool_messages(model)
    assert message.content == "I'll call her" and message.status == "success"


async def test_raw_decisions_still_pass_through(harness: Harness) -> None:
    model = ScriptedChatModel(turns=[("email", {"to": "ada", "body": "hi"}), "sent"])
    agent = await mailer(harness, model)
    paused = await agent.run("email ada", user="u1", thread="t1")
    assert paused.interrupt is not None
    raw = {"decisions": [{"type": "approve"}]}
    await agent.resume(paused.interrupt.interrupt_id, "answer", answer=raw, reviewer="l")
    assert sent == [{"to": "ada", "body": "hi"}]


async def test_a_decision_the_tool_does_not_allow_is_refused_before_resuming(
    harness: Harness,
) -> None:
    model = ScriptedChatModel(turns=[("email", {"to": "ada", "body": "hi"}), "sent"])
    agent = await mailer(harness, model, InterruptOnConfig(allowed_decisions=["approve", "reject"]))
    paused = await agent.run("email ada", user="u1", thread="t1")
    assert paused.interrupt is not None
    with pytest.raises(ConfigurationError, match="edit"):
        await agent.resume(
            paused.interrupt.interrupt_id, "edit", answer={"to": "bob", "body": "x"}, reviewer="l"
        )
    with pytest.raises(ConfigurationError, match="respond"):
        await agent.resume(paused.interrupt.interrupt_id, "answer", answer="no", reviewer="l")
    # still paused on the same interrupt: an allowed decision goes through
    done = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="l")
    assert done.status is RunStatus.SUCCESS and sent == [{"to": "ada", "body": "hi"}]


async def test_a_batch_of_calls_is_one_approval_and_one_decision_for_all(
    harness: Harness,
) -> None:
    calls = [("email", {"to": "ada", "body": "hi"}), ("email", {"to": "bob", "body": "hi"})]
    model = ScriptedChatModel(turns=[calls, "sent both"])
    agent = await mailer(harness, model)
    paused = await agent.run("email both", user="u1", thread="t1")
    assert paused.interrupt is not None
    assert paused.interrupt.question == "Approve email? (and 1 more call)"
    assert paused.interrupt.payload is not None
    assert len(paused.interrupt.payload["action_requests"]) == 2
    done = await agent.resume(
        paused.interrupt.interrupt_id, "edit", answer={"to": "cy", "body": "hi"}, reviewer="l"
    )
    assert done.status is RunStatus.SUCCESS
    assert sorted(m["to"] for m in sent) == ["bob", "cy"]  # the first edited, the second approved


async def test_a_rejected_batch_rejects_every_call(harness: Harness) -> None:
    calls = [("email", {"to": "ada", "body": "hi"})] * 3
    model = ScriptedChatModel(turns=[calls, "none sent"])
    agent = await mailer(harness, model)
    paused = await agent.run("email", user="u1", thread="t1")
    assert paused.interrupt is not None
    assert paused.interrupt.question == "Approve email? (and 2 more calls)"
    await agent.resume(paused.interrupt.interrupt_id, "reject", reviewer="l")
    assert sent == []
    assert all("not executed" in str(m.content) for m in tool_messages(model))


async def test_a_deep_agent_interrupt_on_is_answered_with_harness_decisions(
    harness: Harness,
) -> None:
    model = ScriptedChatModel(
        turns=[("email", {"to": "ada", "body": "hi"}), "done"],
    )
    graph = create_deep_agent(
        model=model,
        tools=await harness.tools(email, framework="langgraph"),
        system_prompt="You send emails.",
        interrupt_on={"email": True},
        checkpointer=InMemorySaver(),
    )
    agent = harness.wrap(graph, id="deep-mailer")
    paused = await agent.run("email ada", user="u1", thread="d1")
    assert paused.interrupt is not None and paused.interrupt.reason is InterruptReason.APPROVAL
    done = await agent.resume(
        paused.interrupt.interrupt_id, "edit", answer={"to": "bob", "body": "hi"}, reviewer="l"
    )
    assert done.status is RunStatus.SUCCESS and sent == [{"to": "bob", "body": "hi"}]
