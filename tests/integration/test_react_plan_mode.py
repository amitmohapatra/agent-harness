"""Plan mode for ``ReAct`` (``plan_mode.PlanMode``): entered on request, as Claude's and Kimi's
is (``framework_options``), never by default — the run reads only, presents its plan with
LangChain's ``write_todos``, a person approves it (LangChain's ``HumanInTheLoopMiddleware``:
one question), then the agent acts under the usual governance."""

from __future__ import annotations

from typing import Any

import pytest

from tests.support.arm import arm
from tests.support.models import ScriptedChat
from trellis import Harness, ReAct, tool
from trellis.contracts import RunEventType, RunStatus
from trellis.harness.plan_mode import PlanMode

PLAN = {"configurable": {"plan_mode": True}}
TODOS = {"todos": [{"content": "refund o1", "status": "pending"}]}
done: list[str] = []


@tool(side_effects="read")
def order(order_id: str) -> str:
    """An order's state."""
    return f"{order_id}: paid"


@tool(side_effects="write")
def refund(order_id: str) -> str:
    """Refund an order."""
    done.append(order_id)
    return f"refunded {order_id}"


@pytest.fixture(autouse=True)
def nothing_done() -> None:
    done.clear()


def planner(harness: Harness, turns: list[Any], **wrap: Any) -> tuple[Any, ScriptedChat]:
    model = ScriptedChat(turns)
    graph = ReAct("You handle refunds.", model, middleware=[PlanMode()])
    return arm(harness.wrap(graph, id="refunds", **wrap), [order, refund]), model


def offered(request: dict[str, Any]) -> set[str]:
    return {t["function"]["name"] for t in request.get("tools", [])}


def system(request: dict[str, Any]) -> str:
    return str(request["messages"][0]["content"])


async def test_plan_mode_is_not_entered_by_default(harness: Harness) -> None:
    agent, model = planner(harness, [("refund", {"order_id": "o1"}), "refunded"])
    result = await agent.run("refund o1", user="u")
    # the write is governed as ever (it asks): no plan, no plan-mode instructions
    assert result.interrupt is not None and result.interrupt.tool_call is not None
    assert result.interrupt.tool_call.tool == "refund"
    assert "write_todos" not in offered(model.requests[0])
    assert "Plan mode" not in system(model.requests[0])


async def test_in_plan_mode_the_plan_is_approved_once_then_the_agent_acts(
    harness: Harness, writes_unasked: None
) -> None:
    agent, model = planner(
        harness,
        [
            ("order", {"order_id": "o1"}),
            ("write_todos", TODOS),
            ("refund", {"order_id": "o1"}),
            "refunded o1",
        ],
    )
    paused = await agent.run("refund o1", user="u", framework_options=PLAN)
    assert "write_todos" in offered(model.requests[0])
    assert "Plan mode" in system(model.requests[0])
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
    assert paused.interrupt.tool_call is not None
    assert (paused.interrupt.tool_call.tool, paused.interrupt.tool_call.args) == (
        "write_todos",
        TODOS,
    )
    assert done == []
    result = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="lead")
    assert result.status is RunStatus.SUCCESS and result.answer == "refunded o1"
    assert done == ["o1"]
    assert "Plan mode" not in system(model.requests[-1])  # approved: it acts


async def test_while_planning_a_call_that_does_more_than_read_is_refused_and_recorded(
    harness: Harness, writes_unasked: None
) -> None:
    agent, model = planner(
        harness, [[("order", {"order_id": "o1"}), ("refund", {"order_id": "o1"})], "planning"]
    )
    events = [e async for e in agent.stream("refund o1", user="u", framework_options=PLAN)]
    assert done == []
    results = {
        e.data["tool"]: e.data["status"] for e in events if e.type is RunEventType.TOOL_CALL_RESULT
    }
    assert results == {"order": "ok", "refund": "rejected"}
    told = [m for m in model.requests[1]["messages"] if m["role"] == "tool"]
    assert any("plan" in str(m["content"]).lower() for m in told)


async def test_a_rejected_plan_is_revised_and_asked_again(
    harness: Harness, writes_unasked: None
) -> None:
    revised = {"todos": [{"content": "ask before refunding o1", "status": "pending"}]}
    agent, model = planner(harness, [("write_todos", TODOS), ("write_todos", revised), "ok"])
    first = await agent.run("refund o1", user="u", framework_options=PLAN)
    assert first.interrupt is not None
    second = await agent.resume(
        first.interrupt.interrupt_id, "reject", answer="ask the customer first", reviewer="lead"
    )
    assert second.status is RunStatus.PAUSED and second.interrupt is not None
    assert second.interrupt.tool_call is not None and second.interrupt.tool_call.args == revised
    assert "Plan mode" in system(model.requests[1])  # still planning


async def test_an_agent_is_always_in_plan_mode_with_its_wrap_options(harness: Harness) -> None:
    agent, _ = planner(harness, [("write_todos", TODOS), "ok"], framework_options=PLAN)
    paused = await agent.run("refund o1", user="u")
    assert paused.interrupt is not None and paused.interrupt.tool_call is not None
    assert paused.interrupt.tool_call.tool == "write_todos"


async def test_a_hand_over_waits_for_the_plan(harness: Harness) -> None:
    model = ScriptedChat(
        [("task", {"description": "refund o1", "subagent_type": "general-purpose"}), "planning"]
    )
    helper = {"name": "clerk", "description": "Does refunds.", "system_prompt": "s"}
    graph = ReAct("s", model, subagents=[helper], middleware=[PlanMode()])
    agent = arm(harness.wrap(graph, id="refunds"), [refund])
    result = await agent.run("refund o1", user="u", framework_options=PLAN)
    assert result.answer == "planning" and len(model.requests) == 2  # the helper never ran
