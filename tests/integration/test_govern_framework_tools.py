"""The framework's own tools governed: a ``create_agent`` graph's tools through
``h.governance_middleware()`` (``ReAct`` has it), an OpenAI Agents agent's ``FunctionTool``\\ s
through ``h.wrap`` on a copy — each call decided by governance (a read runs, a change is
announced, an irreversible one asks), journaled (a resume does not run it again) and seen on
the run's events; the harness's own tools and the graph's state tools are let through."""

from __future__ import annotations

from typing import Any

from agents import Agent, function_tool
from langchain.agents import create_agent
from langchain_core.tools import tool as lc_tool
from langgraph.checkpoint.memory import InMemorySaver

from tests.support.chat_model import ScriptedChatModel
from tests.support.openai_model import ScriptedModel
from trellis import Harness, ReAct, tool
from trellis.contracts import RunEventType, RunStatus
from trellis.harness.middleware import GovernTools, side_effects_of

ran: list[str] = []


@lc_tool
def lookup(sku: str) -> str:
    """Look a SKU up."""
    ran.append(f"lookup {sku}")
    return f"{sku}: 7"


lookup.metadata = {"side_effects": "read"}


@lc_tool
async def refund(order: str) -> str:
    """Refund an order."""
    ran.append(f"refund {order}")
    return f"refunded {order}"


refund.metadata = {"side_effects": "irreversible"}


@lc_tool
def broken(x: str) -> str:
    """Fail."""
    raise ValueError("the service is down")


@tool(side_effects="irreversible")
def wire(to: str) -> str:
    """Wire money."""
    ran.append(f"wire {to}")
    return f"wired {to}"


def graph(model: Any, *tools: Any) -> Any:
    return create_agent(
        model, tools=list(tools), middleware=[GovernTools()], checkpointer=InMemorySaver()
    )


def results(events: list[Any]) -> dict[str, Any]:
    return {
        e.data["tool"]: e.data["status"] for e in events if e.type is RunEventType.TOOL_CALL_RESULT
    }


def setup_function() -> None:
    ran.clear()


async def test_a_graphs_own_read_runs_and_is_on_the_runs_events(harness: Harness) -> None:
    model = ScriptedChatModel(turns=[("lookup", {"sku": "a"}), "a: 7"])
    agent = harness.wrap(graph(model, lookup), id="g")
    events = [e async for e in agent.stream("a?", user="u", thread="t")]
    assert results(events) == {"lookup": "ok"} and ran == ["lookup a"]
    assert harness.governance_middleware().__class__ is GovernTools


async def test_an_irreversible_own_tool_asks_and_a_resume_runs_nothing_twice(
    harness: Harness,
) -> None:
    model = ScriptedChatModel(
        turns=[[("lookup", {"sku": "o1"}), ("refund", {"order": "o1"})], "refunded"]
    )
    agent = harness.wrap(graph(model, lookup, refund), id="g")
    paused = await agent.run("refund o1", user="u", thread="t")
    assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
    assert paused.interrupt.question.startswith("Approve refund?")
    done = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="r")
    assert done.answer == "refunded"
    assert ran == ["lookup o1", "refund o1"]  # the read was journaled, not run again


async def test_an_own_tool_that_fails_is_an_error_the_model_reads(harness: Harness) -> None:
    model = ScriptedChatModel(turns=[("broken", {"x": "1"}), "it failed"])
    agent = harness.wrap(graph(model, broken), id="g")
    events = [e async for e in agent.stream("x", user="u", thread="t")]
    assert results(events) == {"broken": "error"}
    assert "the service is down" in str(model.seen[-1][-1].content)


async def test_the_harness_tools_and_state_tools_go_through_as_they_are(
    harness: Harness,
) -> None:
    from langchain.agents.middleware import TodoListMiddleware

    model = ScriptedChatModel(
        turns=[("write_todos", {"todos": []}), ("wire", {"to": "ada"}), "wired"]
    )
    target = create_agent(  # type: ignore[call-overload]
        model,
        tools=await harness.tools(wire, framework="langgraph"),
        middleware=[TodoListMiddleware(), GovernTools()],  # type: ignore[list-item]
        checkpointer=InMemorySaver(),
    )
    agent = harness.wrap(target, id="g")
    paused = await agent.run("wire ada", user="u", thread="t")
    assert paused.interrupt is not None  # asked once, by the harness tool's own governance
    done = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="r")
    assert done.answer == "wired" and ran == ["wire ada"]


async def test_react_governs_a_tool_its_given_middleware_brings(harness: Harness) -> None:
    from langchain.agents.middleware import AgentMiddleware

    class Brings(AgentMiddleware):
        def __init__(self) -> None:
            super().__init__()
            self.tools = [refund]

    model = ScriptedChatModel(turns=[("refund", {"order": "o2"}), "refunded"])
    target = ReAct("s", model, middleware=[Brings()])
    paused = await harness.wrap(target, id="r").run("refund o2", user="u")
    assert paused.interrupt is not None and ran == []


def test_a_tools_risk_is_declared_else_annotated_else_write() -> None:
    assert side_effects_of({"side_effects": "read"}) == "read"
    assert side_effects_of({"annotations": {"readOnlyHint": True}}) == "read"
    assert side_effects_of({"annotations": {"destructiveHint": True}}) == "irreversible"
    assert side_effects_of(None) == "write"


# --------------------------------------------------------------------------- OpenAI Agents
async def test_an_sdk_agents_own_tool_is_governed_on_a_copy(harness: Harness) -> None:
    @function_tool
    def ship(order: str) -> str:
        """Ship an order."""
        ran.append(f"ship {order}")
        return f"shipped {order}"

    target = Agent(name="s", model=ScriptedModel([("ship", {"order": "o1"}), "shipped"]))
    target.tools.append(ship)
    tools_before = list(target.tools)
    agent = harness.wrap(target, id="sdk")
    events = [e async for e in agent.stream("ship o1", user="u")]
    assert results(events) == {"ship": "ok"} and ran == ["ship o1"]
    notices = [
        e for e in events if e.type is RunEventType.CUSTOM and e.data.get("name") == "tool_notice"
    ]
    assert notices  # a change, announced (no declared risk: write)
    assert target.tools == tools_before and target.tools[0] is ship  # yours, unchanged


async def test_an_error_the_tool_itself_reports_is_an_error_outcome(harness: Harness) -> None:
    from langchain_core.tools import ToolException

    @lc_tool
    def picky(x: str) -> str:
        """Refuse."""
        raise ToolException("not that one")

    picky.handle_tool_error = True
    model = ScriptedChatModel(turns=[("picky", {"x": "1"}), "refused"])
    events = [e async for e in harness.wrap(graph(model, picky), id="g").stream("x", user="u")]
    assert results(events) == {"picky": "error"}
