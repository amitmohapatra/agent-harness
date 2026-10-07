"""The integration matrix against the real services — the memory service, and agent-runs with
its ticker — with every model planned (``tests/support/planned.py``; the Claude target drives
the scripted Claude Code CLI): everything but the model is real, and every assertion reads the
services' own state back.

For each target — a function, ``ReAct``, LangChain's ``create_agent`` (a LangGraph graph), a
hand-built LangGraph ``StateGraph`` with a checkpointer and an ``interrupt()`` node, Deep
Agents with planning and a sub-agent, the OpenAI Agents SDK, the Claude Agent SDK:

* the run's context is pushed from memory: a memory seeded for the user reaches what the model
  is sent;
* the memory tools are the agent's: its ``memory_remember`` lands in the memory service;
* an irreversible harness tool pauses the run in agent-runs, and an approval or a rejection
  with a reason completes it — for LangGraph and Deep Agents also through their own
  ``HumanInTheLoopMiddleware`` / ``interrupt_on``, for the OpenAI Agents SDK its own
  ``needs_approval``, for a hand-built graph its own ``interrupt()``;
* the transcript, the tool calls (the catalog's statistics, approvals included) and the run's
  outcome are in the memory service, and the run record in agent-runs has the run's status.

Then the features around a run, each with memory on: a worker (claim, pause, a resume another
worker continues, a crash after a side effect that the next attempt does not repeat), a
schedule fired by the ticker to a worker, AG-UI, A2A, documents, a person's feedback, offline
evaluation and online judges."""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
import uuid
from collections.abc import Awaitable, Callable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Final

import httpx
import pytest
import uvicorn
from agents import Agent as OpenAIAgent
from agents import function_tool
from claude_agent_sdk import ClaudeAgentOptions
from deepagents import create_deep_agent
from fastapi import FastAPI, Request
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, TodoListMiddleware
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from tests.live.conftest import RUNS_URL, live_harness, needs_memory, needs_runs
from tests.live.support import StubLangfuse, eventually, memory_scope
from tests.support.planned import FINAL, Call, PlannedChat, PlannedChatModel, PlannedModel
from trellis import Agent, Harness, ReAct, Runtime, a2a, tool
from trellis.contracts import (
    InterruptReason,
    RunEvent,
    RunEventType,
    RunOutcome,
    RunRecord,
    RunStatus,
)
from trellis.harness import telemetry
from trellis.harness.agui.sse import decode
from trellis.harness.evals import exact_match, grounding, llm_judge
from trellis.harness.governance import catalog as governance_catalog
from trellis.harness.tools.convert import text_of
from trellis.harness.tools.sources import FunctionTool
from trellis.memory import MemoryContext
from trellis.memory.models import Feedback

pytestmark = [pytest.mark.live, needs_memory, needs_runs]

#: A run here waits on both services (and a worker on the ticker's sweep): well above that.
TIMEOUT_SECONDS: Final = 300
CLI: Final = str(Path(__file__).resolve().parents[1] / "support" / "fake_claude_cli.py")
SYSTEM: Final = "You keep SKU A-1 in stock. Use the tools you are asked to use."
QUESTION: Final = "Is SKU A-1 low in my warehouse? Remember how I reorder, then check its stock."
REASON: Final = "over this month's budget"
SUBAGENT: Final = "stock-checker"
FRAMEWORKS: Final = (
    "function",
    "react",
    "langgraph",
    "stategraph",
    "deepagents",
    "openai_agents",
    "claude_agent_sdk",
)
#: The targets whose framework pauses for approval itself.
NATIVE: Final = ("langgraph", "deepagents", "openai_agents")
#: The graph targets: tools are built in with ``h.tools`` (a compiled graph binds them).
GRAPHS: Final = ("langgraph", "stategraph", "deepagents")


# --------------------------------------------------------------------------- the targets
@dataclass
class Case:
    """One test's agent: its names (unique, so the services' state is this test's own), the
    harness tools, and a ledger each side effect appends to."""

    h: Harness
    framework: str
    tmp: Path
    suffix: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    ledger: list[str] = field(default_factory=list)

    @property
    def user(self) -> str:
        return f"fv-user-{self.suffix}"

    @property
    def thread(self) -> str:
        return f"fv-thread-{self.suffix}"

    @property
    def agent_id(self) -> str:
        return f"fv-{self.framework}-{self.suffix}"

    @property
    def fact(self) -> str:
        return f"My SKU A-1 stock is kept in the Berlin warehouse, bay {self.suffix}."

    @property
    def note(self) -> str:
        return f"{self.user} reorders in batches of 20 ({self.suffix})."

    @property
    def stock(self) -> str:
        return f"stock_{self.suffix}"

    @property
    def reorder(self) -> str:
        return f"reorder_{self.suffix}"

    @property
    def email(self) -> str:
        return f"email_{self.suffix}"

    def tools(self, *, native_email: bool = False) -> list[FunctionTool]:
        ledger = self.ledger

        def stock(sku: str) -> int:
            """Units of a SKU in stock."""
            return 3

        def reorder(sku: str, qty: int) -> str:
            """Order more units of a SKU from the supplier."""
            ledger.append(f"reorder:{sku}:{qty}")
            return f"ordered {qty} x {sku}"

        def email(to: str, body: str) -> str:
            """Email someone."""
            ledger.append(f"email:{to}")
            return f"sent to {to}"

        made = [
            tool(stock, name=self.stock, side_effects="read"),
            tool(reorder, name=self.reorder, side_effects="irreversible"),
        ]
        if not native_email:
            made.append(tool(email, name=self.email, side_effects="write"))
        return made


@dataclass
class Built:
    target: Any
    #: the harness tools passed to ``wrap`` (none for a graph: it binds them when built)
    tools: list[Any]
    #: everything the model was sent, as text
    said: Callable[[], str]


async def build(case: Case, plan: Sequence[Call], *, hitl: bool = False) -> Built:
    """``case.framework``'s target, its model following ``plan``. ``hitl``: approvals of
    ``case.email`` by the framework itself."""
    tools = case.tools(native_email=hitl and case.framework == "openai_agents")
    return await BUILDERS[case.framework](case, list(plan), tools, hitl)


Builder = Callable[[Case, list[Call], list[Any], bool], Awaitable[Built]]


async def _function(case: Case, plan: list[Call], tools: list[Any], hitl: bool) -> Built:
    seen: list[str] = []

    async def follow(input: Any, agent: Runtime) -> str:
        seen.append(agent.context or "")
        results = [text_of(await agent.tools.call(name, **args)) for name, args in plan]
        return FINAL.replace("{last}", results[-1] if results else "")

    return Built(follow, tools, lambda: "\n".join(seen))


async def _react(case: Case, plan: list[Call], tools: list[Any], hitl: bool) -> Built:
    chat = PlannedChat(plan)
    return Built(ReAct(system=SYSTEM, model=chat), tools, chat.said)


async def _langgraph(case: Case, plan: list[Call], tools: list[Any], hitl: bool) -> Built:
    """LangChain's ``create_agent``: without ``hitl`` it has no checkpointer, so a resume
    re-runs the graph against the run's journal; with it, the middleware's pause resumes in
    place."""
    model = PlannedChatModel(plan=plan)
    middleware = [HumanInTheLoopMiddleware(interrupt_on={case.email: True})] if hitl else []
    graph = create_agent(
        model,
        tools=await case.h.tools(*tools, framework="langgraph"),
        system_prompt=SYSTEM,
        middleware=middleware,
        checkpointer=InMemorySaver() if hitl else None,
    )
    return Built(graph, [], model.said)


async def _state_graph(case: Case, plan: list[Call], tools: list[Any], hitl: bool) -> Built:
    model = PlannedChatModel(plan=plan)
    native = await case.h.tools(*tools, framework="langgraph")
    return Built(state_graph(model, native, confirm=False), [], model.said)


async def _openai_agents(case: Case, plan: list[Call], tools: list[Any], hitl: bool) -> Built:
    model = PlannedModel(plan)
    native_tools = [_sdk_email(case)] if hitl else []
    agent = OpenAIAgent(name="stock", instructions=SYSTEM, model=model, tools=native_tools)
    return Built(agent, tools, model.said)


async def _claude(case: Case, plan: list[Call], tools: list[Any], hitl: bool) -> Built:
    record = case.tmp / f"cli-{uuid.uuid4().hex[:6]}.json"
    script = [{"tool": name, "args": args} for name, args in plan] + [{"text": FINAL}]
    options = ClaudeAgentOptions(
        cli_path=CLI,
        system_prompt=SYSTEM,
        env={"FAKE_CLAUDE_SCRIPT": json.dumps(script), "FAKE_CLAUDE_RECORD": str(record)},
    )
    return Built(options, tools, lambda: record.read_text() if record.exists() else "")


def state_graph(model: PlannedChatModel, tools: list[Any], *, confirm: bool) -> Any:
    """A hand-built graph: a model node, a tool node, and — with ``confirm`` — a node that asks
    with LangGraph's own ``interrupt()`` before it answers. Compiled with a checkpointer, so a
    pause (the graph's own, or a harness approval inside the tool node) resumes in place."""
    bound = model.bind_tools(tools)

    async def think(state: MessagesState) -> dict[str, Any]:
        return {"messages": [await bound.ainvoke(state["messages"])]}

    def route(state: MessagesState) -> str:
        if getattr(state["messages"][-1], "tool_calls", None):
            return "tools"
        return "confirm" if confirm else END

    def confirm_node(state: MessagesState) -> dict[str, Any]:
        answer = interrupt({"question": "Send the stock report?"})
        said = state["messages"][-1].content
        return {"messages": [AIMessage(content=f"{said} (confirmed: {answer})")]}

    graph = StateGraph(MessagesState)
    graph.add_node("model", think)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", route)
    graph.add_edge("tools", "model")
    if confirm:
        graph.add_node("confirm", confirm_node)
        graph.add_edge("confirm", END)
    return graph.compile(checkpointer=InMemorySaver())


async def _deep_agent(case: Case, plan: list[Call], tools: list[Any], hitl: bool) -> Built:
    """Deep Agents: it plans (``write_todos``), hands the stock check to a sub-agent (``task``)
    — which calls the harness's stock tool itself — then follows ``plan``."""
    native = await case.h.tools(*tools, framework="deepagents")
    checker = PlannedChatModel(plan=[(case.stock, {"sku": "A-1"})], final="A-1: {last} units.")
    model = PlannedChatModel(
        plan=[
            ("write_todos", {"todos": [{"content": "check A-1", "status": "in_progress"}]}),
            ("task", {"description": "Check the stock of A-1.", "subagent_type": SUBAGENT}),
            *plan,
        ]
    )
    graph = create_deep_agent(
        model=model,
        tools=native,
        system_prompt=SYSTEM,
        middleware=[TodoListMiddleware()],
        subagents=[
            {
                "name": SUBAGENT,
                "description": "Checks how many units of a SKU are in stock.",
                "system_prompt": "Check the stock with the stock tool.",
                "tools": native,
                "model": checker,
            }
        ],
        interrupt_on={case.email: True} if hitl else None,
        checkpointer=InMemorySaver(),
    )
    return Built(graph, [], lambda: model.said() + checker.said())


def _sdk_email(case: Case) -> Any:
    """The OpenAI Agents SDK's own tool, which asks for approval itself (``needs_approval``)."""
    ledger = case.ledger

    @function_tool(name_override=case.email, needs_approval=True)
    def email(to: str, body: str) -> str:
        """Email someone."""
        ledger.append(f"email:{to}")
        return f"sent to {to}"

    return email


BUILDERS: Final[dict[str, Builder]] = {
    "function": _function,
    "react": _react,
    "langgraph": _langgraph,
    "stategraph": _state_graph,
    "deepagents": _deep_agent,
    "openai_agents": _openai_agents,
    "claude_agent_sdk": _claude,
}


# --------------------------------------------------------------------------- reading back
async def stored(h: Harness, run_id: str) -> RunRecord:
    """The run as agent-runs keeps it, read over its own API and parsed as the contract."""
    assert RUNS_URL is not None
    async with httpx.AsyncClient(
        base_url=RUNS_URL, headers={"X-Api-Key": h.settings.api_key or ""}
    ) as client:
        response = await client.get(f"/v1/runs/{run_id}")
        response.raise_for_status()
        return RunRecord.model_validate(response.json())


async def stats(scope: MemoryContext, name: str, **at_least: int) -> bool:
    """Whether the catalog's statistics of ``name`` reach ``at_least`` within the settle time."""

    async def reached() -> bool:
        entries = await scope.advanced.tools.catalog(names=[name])
        return bool(entries) and all(
            getattr(entries[0].stats, key) >= value for key, value in at_least.items()
        )

    return await eventually(reached)


async def outcome_of(scope: MemoryContext, run_id: str) -> list[Feedback]:
    return [f for f in (await scope.feedback.page_for("run", run_id)).items if f.source == "system"]


async def written_back(
    h: Harness, case: Case, run_id: str, answer: str, *, tool_name: str, **at_least: int
) -> None:
    """The run's transcript, a tool's statistics (``at_least``; by default one call) and the
    run's outcome, in the memory service."""
    await h.writes.drain()
    assert h.writes.failed == 0
    thread = await memory_scope(h, user=case.user, agent_id=case.agent_id, thread=case.thread)
    history = await thread.history()
    assert history and history[0].role == "USER" and history[0].content == QUESTION, history
    assert history[-1].role == "ASSISTANT" and history[-1].content == answer, history
    assert await stats(thread, tool_name, **(at_least or {"calls": 1}))
    [outcome] = await outcome_of(thread, run_id)
    assert outcome.verdict == "confirm"


def ran(events: list[RunEvent]) -> set[str]:
    return {str(e.data.get("tool")) for e in events if e.type is RunEventType.TOOL_CALL_START}


# --------------------------------------------------------------------------- per target
@pytest.mark.timeout(TIMEOUT_SECONDS)
@pytest.mark.parametrize("framework", FRAMEWORKS)
async def test_memory_is_pushed_pulled_and_written_back(framework: str, tmp_path: Path) -> None:
    async with live_harness() as h:
        case = Case(h, framework, tmp_path)
        plan: list[Call] = [("memory_remember", {"content": case.note})]
        if framework != "deepagents":  # a Deep Agent's sub-agent checks the stock
            plan.append((case.stock, {"sku": "A-1"}))
        built = await build(case, plan)
        agent = h.wrap(built.target, id=case.agent_id, tools=built.tools)
        user_scope = await memory_scope(h, user=case.user, agent_id=agent.id)
        await user_scope.remember(case.fact, visibility="USER")

        events = [e async for e in agent.stream(QUESTION, user=case.user, thread=case.thread)]
        finished = events[-1]
        assert finished.type is RunEventType.RUN_FINISHED, finished
        assert finished.outcome is RunOutcome.SUCCESS, finished.error
        answer = finished.data["result"]
        last = "3" if framework != "deepagents" else ""  # the stock, or what remember said
        assert answer.startswith(f"Done. {last}"), answer

        # (a) pushed: the seeded memory reached what the model was sent
        [loaded] = [e for e in events if e.type is RunEventType.CONTEXT_LOADED]
        assert loaded.data["chars"] > 0
        assert case.fact in built.said()
        # (b) pulled: the model's memory_remember is in the memory service
        assert {"memory_remember", case.stock} <= ran(events)

        async def remembered() -> bool:
            found = await user_scope.search(case.note)
            return any(item.text == case.note for item in found)

        assert await eventually(remembered)
        # (d) the transcript, the stock call and the outcome; (e) the run record
        await written_back(h, case, finished.run_id, answer, tool_name=case.stock)
        record = await stored(h, finished.run_id)
        assert record.status is RunStatus.SUCCESS and record.output == answer
        assert (record.agent_id, record.user_id, record.thread_id) == (
            agent.id,
            case.user,
            case.thread,
        )


@pytest.mark.timeout(TIMEOUT_SECONDS)
@pytest.mark.parametrize("decision", ["approve", "reject"])
@pytest.mark.parametrize("framework", FRAMEWORKS)
async def test_an_irreversible_call_waits_in_agent_runs_for_a_person(
    framework: str, decision: str, tmp_path: Path
) -> None:
    async with live_harness() as h:
        case = Case(h, framework, tmp_path)
        built = await build(case, [(case.reorder, {"sku": "A-1", "qty": 20})])
        agent = h.wrap(built.target, id=case.agent_id, tools=built.tools)

        paused = await agent.run(QUESTION, user=case.user, thread=case.thread)
        assert paused.status is RunStatus.PAUSED and paused.interrupt is not None, paused.error
        asked = paused.interrupt
        assert asked.reason is InterruptReason.APPROVAL and asked.tool_call is not None
        assert asked.tool_call.tool == case.reorder and case.ledger == []
        record = await stored(h, paused.run_id)
        assert record.status is RunStatus.PAUSED and record.checkpoint is not None
        assert record.awaiting is not None and record.awaiting.interrupt_id == asked.interrupt_id
        assert paused.run_id in [r.run_id for r in await h.inbox()]

        reason = REASON if decision == "reject" else None
        done = await agent.resume(asked.interrupt_id, decision, answer=reason, reviewer="fv-lead")
        assert done.status is RunStatus.SUCCESS, done.error
        if decision == "approve":
            assert case.ledger == ["reorder:A-1:20"] and "ordered 20 x A-1" in done.answer
        else:
            assert case.ledger == [] and REASON in done.answer  # the model read the reason
        assert (await stored(h, done.run_id)).status is RunStatus.SUCCESS
        # a rejected call never ran: it is a rejection in the statistics, not a call
        counted = {"approvals": 1, "calls": 1} if decision == "approve" else {"rejections": 1}
        await written_back(h, case, done.run_id, done.answer, tool_name=case.reorder, **counted)


@pytest.mark.timeout(TIMEOUT_SECONDS)
@pytest.mark.parametrize("decision", ["approve", "reject"])
@pytest.mark.parametrize("framework", NATIVE)
async def test_the_frameworks_own_approval_is_a_harness_interrupt(
    framework: str, decision: str, tmp_path: Path
) -> None:
    async with live_harness() as h:
        case = Case(h, framework, tmp_path)
        built = await build(case, [(case.email, {"to": "ops", "body": "A-1 is low"})], hitl=True)
        agent = h.wrap(built.target, id=case.agent_id, tools=built.tools)

        paused = await agent.run(QUESTION, user=case.user, thread=case.thread)
        assert paused.status is RunStatus.PAUSED and paused.interrupt is not None, paused.error
        asked = paused.interrupt
        assert asked.reason is InterruptReason.APPROVAL and asked.tool_call is not None
        assert asked.tool_call.tool == case.email and asked.tool_call.args["to"] == "ops"
        if framework != "openai_agents":  # the middleware's whole request travels along
            assert asked.payload is not None and asked.payload["action_requests"]
        record = await stored(h, paused.run_id)
        assert record.status is RunStatus.PAUSED and record.awaiting is not None

        reason = REASON if decision == "reject" else None
        done = await agent.resume(asked.interrupt_id, decision, answer=reason, reviewer="fv-lead")
        assert done.status is RunStatus.SUCCESS, done.error
        if decision == "approve":
            assert case.ledger == ["email:ops"] and "sent to ops" in done.answer
        else:
            assert case.ledger == [] and REASON in done.answer
        assert (await stored(h, done.run_id)).status is RunStatus.SUCCESS
        if framework == "openai_agents":  # the SDK's own tool: no harness record to count
            await h.writes.drain()
            return
        counted = {"approvals": 1, "calls": 1} if decision == "approve" else {"rejections": 1}
        await written_back(h, case, done.run_id, done.answer, tool_name=case.email, **counted)


@pytest.mark.timeout(TIMEOUT_SECONDS)
async def test_a_hand_built_graphs_own_interrupt_pauses_in_agent_runs(tmp_path: Path) -> None:
    async with live_harness() as h:
        case = Case(h, "stategraph", tmp_path)
        model = PlannedChatModel(plan=[(case.stock, {"sku": "A-1"})])
        native = await h.tools(*case.tools(), framework="langgraph")
        agent = h.wrap(state_graph(model, native, confirm=True), id=case.agent_id)
        scope = await memory_scope(h, user=case.user, agent_id=agent.id)
        await scope.remember(case.fact, visibility="USER")

        paused = await agent.run(QUESTION, user=case.user, thread=case.thread)
        assert paused.status is RunStatus.PAUSED and paused.interrupt is not None, paused.error
        assert paused.interrupt.question == "Send the stock report?"
        assert paused.interrupt.reason is InterruptReason.QUESTION
        assert (await stored(h, paused.run_id)).status is RunStatus.PAUSED
        done = await agent.resume(
            paused.interrupt.interrupt_id, "answer", answer="yes", reviewer=case.user
        )
        assert done.status is RunStatus.SUCCESS and done.answer == "Done. 3 (confirmed: yes)"
        assert case.fact in model.said()
        assert len(model.seen) == 2  # resumed in place: the model was not asked again
        await written_back(h, case, done.run_id, done.answer, tool_name=case.stock)


@pytest.mark.timeout(TIMEOUT_SECONDS)
@pytest.mark.parametrize("framework", ["langgraph", "deepagents"])
async def test_a_catalog_rule_set_after_the_graph_was_built_decides_its_calls(
    framework: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A compiled graph holds the tools it was built with; an administrator's rule set in the
    catalog afterwards still decides whether a call waits, once the rules governance read
    while building it are older than their TTL."""
    async with live_harness() as h:
        case = Case(h, framework, tmp_path)
        built = await build(case, [(case.email, {"to": "ops", "body": "A-1 is low"})])
        await h.writes.drain()  # the build published the tools to the catalog
        agent = h.wrap(built.target, id=case.agent_id)
        scope = await memory_scope(h, user=case.user, agent_id=agent.id)
        rule = 'to == "ops"'
        await scope.advanced.tools.put_catalog(
            [{"name": case.email, "side_effects": "write", "approve_when": rule}]
        )
        ttl = governance_catalog.GOVERNANCE_TTL_SECONDS
        monkeypatch.setattr(governance_catalog, "_now", lambda: time.monotonic() + ttl + 1)
        paused = await agent.run(QUESTION, user=case.user, thread=case.thread)
        assert paused.status is RunStatus.PAUSED and paused.interrupt is not None, paused.error
        assert rule in paused.interrupt.question and case.ledger == []
        done = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="fv-lead")
        assert done.status is RunStatus.SUCCESS and case.ledger == ["email:ops"]


@pytest.mark.timeout(TIMEOUT_SECONDS)
async def test_a_deep_agents_sub_agent_call_waits_in_agent_runs(tmp_path: Path) -> None:
    """The irreversible call is the sub-agent's (``task``): it pauses the run like any other,
    and the approval resumes the sub-agent in place."""
    async with live_harness() as h:
        case = Case(h, "deepagents", tmp_path)
        native = await h.tools(*case.tools(), framework="deepagents")
        buyer = PlannedChatModel(plan=[(case.reorder, {"sku": "A-1", "qty": 20})], final="{last}")
        model = PlannedChatModel(
            plan=[("task", {"description": "Reorder 20 x A-1.", "subagent_type": "buyer"})]
        )
        graph = create_deep_agent(
            model=model,
            tools=native,
            subagents=[
                {
                    "name": "buyer",
                    "description": "Reorders stock.",
                    "system_prompt": "Reorder what you are asked to.",
                    "tools": native,
                    "model": buyer,
                }
            ],
            checkpointer=InMemorySaver(),
        )
        agent = h.wrap(graph, id=case.agent_id)
        paused = await agent.run(QUESTION, user=case.user, thread=case.thread)
        assert paused.status is RunStatus.PAUSED and paused.interrupt is not None, paused.error
        assert paused.interrupt.tool_call is not None
        assert paused.interrupt.tool_call.tool == case.reorder and case.ledger == []
        assert (await stored(h, paused.run_id)).status is RunStatus.PAUSED
        done = await agent.resume(paused.interrupt.interrupt_id, "approve", reviewer="fv-lead")
        assert done.status is RunStatus.SUCCESS and done.answer == "Done. ordered 20 x A-1"
        assert case.ledger == ["reorder:A-1:20"] and len(buyer.seen) == 2  # resumed in place
        await written_back(h, case, done.run_id, done.answer, tool_name=case.reorder, approvals=1)


# --------------------------------------------------------------------------- workers
@pytest.mark.timeout(TIMEOUT_SECONDS)
async def test_workers_claim_pause_resume_and_survive_a_crash(tmp_path: Path) -> None:
    case = Case(live_harness(), "worker", tmp_path)
    charged: list[str] = []
    crashed = asyncio.Event()
    contexts: list[str] = []

    @tool(name=f"charge_{case.suffix}", side_effects="write")
    def charge(order: str, amount: int) -> str:
        """Charge an order."""
        charged.append(order)
        return f"charged {order} {amount}"

    async def billing(input: dict[str, Any], agent: Runtime) -> str:
        contexts.append(agent.context or "")
        receipt = await agent.tools.call(charge.spec.name, order=input["order"], amount=40)
        if agent.attempt == 1:  # the worker process dies here, after the side effect
            crashed.set()
            await asyncio.Event().wait()
        size = await agent.ask("Which size?", assignee="role:fv-ops")
        return f"{receipt}; size {size}"

    async with case.h as h:
        agent = h.wrap(billing, id=case.agent_id, tools=[charge])
        scope = await memory_scope(h, user=case.user, agent_id=agent.id)
        await scope.remember(case.fact, visibility="USER")
        # the question memory is asked about is the input's text field (a dict's "question")
        question = "Charge order o-7 for my SKU A-1 stock from the warehouse."
        handle = await agent.start({"order": "o-7", "question": question}, user=case.user)
        assert (await stored(h, handle.run_id)).status is RunStatus.QUEUED

        first = h.worker([agent], concurrency=1)
        # a short lease, so the run the crashed worker held is queued again within the test
        first.loop.lease_seconds = 6
        working = asyncio.create_task(first.run())
        await asyncio.wait_for(crashed.wait(), 60)
        # the charge was saved as the run's progress before the crash
        progress = (await stored(h, handle.run_id)).checkpoint
        assert progress is not None and progress.get("calls"), progress
        first.stop()
        first.stop()  # at once: the run is released, nothing written, its lease lapses
        await working

        async def requeued() -> bool:
            return (await stored(h, handle.run_id)).status is RunStatus.QUEUED

        assert await eventually(requeued, within=60)
        second = h.worker([agent], concurrency=1)
        assert await second.run_once()
        paused = await handle.result(timeout=30)
        assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
        assert paused.interrupt.question == "Which size?" and charged == ["o-7"]
        assert handle.run_id in [r.run_id for r in await h.inbox("role:fv-ops")]

        queued = await agent.resume(
            paused.interrupt.interrupt_id, "answer", answer="L", reviewer="fv-ops"
        )
        assert queued.status is RunStatus.QUEUED  # it came from the queue: a worker continues
        third = h.worker([agent], concurrency=1)
        assert await third.run_once()
        done = await handle.result(timeout=30)
        assert done.status is RunStatus.SUCCESS, done.error
        assert done.answer == "charged o-7 40; size L"
        record = await stored(h, handle.run_id)
        assert record.attempt == 3 and record.checkpoint is None
        assert charged == ["o-7"]  # three attempts, one charge
        assert all(case.fact in c for c in contexts)  # each attempt had the user's memory

        await h.writes.drain()
        assert await stats(scope, charge.spec.name, calls=1)
        thread = await memory_scope(h, user=case.user, agent_id=agent.id, thread=handle.run_id)
        assert [m.content for m in await thread.history()][-1] == done.answer


@pytest.mark.timeout(TIMEOUT_SECONDS)
async def test_a_queued_graph_paused_in_one_worker_is_continued_by_another(
    tmp_path: Path,
) -> None:
    """Two worker processes, each with the graph and its own InMemorySaver: the approval asked
    in the first is answered from the run's journal in the second (its checkpointer does not
    hold the pause), and the call runs once."""
    async with live_harness() as first, live_harness() as second:
        case = Case(first, "langgraph", tmp_path)
        plan = [(case.reorder, {"sku": "A-1", "qty": 20})]

        async def worker_agent(h: Harness) -> Agent:
            model = PlannedChatModel(plan=plan)
            tools = await h.tools(*case.tools(), framework="langgraph")
            graph = create_agent(model, tools=tools, checkpointer=InMemorySaver())
            return h.wrap(graph, id=case.agent_id)

        here, there = await worker_agent(first), await worker_agent(second)
        handle = await here.start(QUESTION, user=case.user, thread=case.thread)
        assert await first.worker([here], concurrency=1).run_once()
        paused = await handle.result(timeout=30)
        assert paused.status is RunStatus.PAUSED and paused.interrupt is not None
        queued = await here.resume(paused.interrupt.interrupt_id, "approve", reviewer="fv-lead")
        assert queued.status is RunStatus.QUEUED
        assert await second.worker([there], concurrency=1).run_once()
        done = await handle.result(timeout=30)
        assert done.status is RunStatus.SUCCESS, done.error
        assert done.answer == "Done. ordered 20 x A-1" and case.ledger == ["reorder:A-1:20"]
        assert (await stored(first, handle.run_id)).attempt == 2


# longer than the others: the ticker fires the schedule within the next two minutes
@pytest.mark.timeout(420)
async def test_a_schedule_fires_through_the_ticker_to_a_worker(tmp_path: Path) -> None:
    async with live_harness() as h:
        case = Case(h, "schedule", tmp_path)

        async def briefing(input: str, agent: Runtime) -> str:
            return f"{input} for {agent.user}: {agent.context or 'nothing'}"

        agent = h.wrap(briefing, id=case.agent_id)
        scope = await memory_scope(h, user=case.user, agent_id=agent.id)
        await scope.remember(case.fact, visibility="USER")
        minute = (datetime.now(UTC) + timedelta(minutes=2)).minute  # agent-runs: one an hour
        schedule = await agent.schedule(f"{minute} * * * *", "briefing", on_behalf_of=case.user)
        worker = h.worker([agent], concurrency=1)
        working = asyncio.create_task(worker.run())
        try:

            async def fired() -> RunRecord | None:
                for row in await _runs_of(h, agent.id):
                    if row["status"] == RunStatus.SUCCESS.value:
                        return await stored(h, row["run_id"])
                return None

            assert await eventually(lambda: _some(fired()), within=240, every=5)
            run = await fired()
            assert run is not None and run.metadata["schedule_id"] == schedule.schedule_id
            assert run.on_behalf_of == case.user and case.fact in str(run.output)
        finally:
            worker.stop()
            await working
            async with _runs_client(h) as client:
                await client.delete(f"/v1/schedules/{schedule.schedule_id}")
        await h.writes.drain()
        thread = await memory_scope(h, user=case.user, agent_id=agent.id, thread=run.run_id)
        assert [m.content for m in await thread.history()] == ["briefing", run.output]


async def _some(value: Any) -> bool:
    return (await value) is not None


def _runs_client(h: Harness) -> httpx.AsyncClient:
    assert RUNS_URL is not None
    return httpx.AsyncClient(base_url=RUNS_URL, headers={"X-Api-Key": h.settings.api_key or ""})


async def _runs_of(h: Harness, agent_id: str) -> list[dict[str, Any]]:
    async with _runs_client(h) as client:
        response = await client.get("/v1/runs", params={"agent_id": agent_id})
        response.raise_for_status()
        return list(response.json())


# --------------------------------------------------------------------------- surfaces
def _user_of(request: Request) -> str:
    return request.headers.get("x-user", "anonymous")


@pytest.mark.timeout(TIMEOUT_SECONDS)
async def test_agui_runs_a_graph_pauses_resumes_and_replays(tmp_path: Path) -> None:
    async with live_harness() as h:
        case = Case(h, "langgraph", tmp_path)
        built = await build(case, [(case.reorder, {"sku": "A-1", "qty": 20})])
        agent = h.wrap(built.target, id=case.agent_id)
        scope = await memory_scope(h, user=case.user, agent_id=agent.id)
        await scope.remember(case.fact, visibility="USER")
        app = FastAPI()
        agent.serve_chat(app, identity=_user_of)
        run_id = f"run_{uuid.uuid4().hex}"
        message = [{"id": "m1", "role": "user", "content": QUESTION}]
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://agui",
            headers={"x-user": case.user},
        ) as client:
            first = await client.post(
                "/agui/run", json={"threadId": case.thread, "runId": run_id, "messages": message}
            )
            events = decode(first.text)
            names = [e[1]["type"] for e in events]
            assert "CUSTOM" in names and "TOOL_CALL_START" not in names  # context, no call yet
            assert {e[1].get("name") for e in events} >= {"context_loaded"}
            paused = events[-1][1]
            assert paused["type"] == "RUN_FINISHED" and paused["outcome"]["type"] == "interrupt"
            [entry] = paused["outcome"]["interrupts"]
            record = await stored(h, run_id)
            assert record.status is RunStatus.PAUSED and record.user_id == case.user

            # a client that dropped reconnects after the last event it saw
            replay = await client.get(f"/agui/runs/{run_id}/events", headers={"Last-Event-ID": "1"})
            assert decode(replay.text) == events[2:]

            resumed = await client.post(
                "/agui/run",
                json={
                    "threadId": case.thread,
                    "resume": [{"interruptId": entry["id"], "payload": True}],
                },
            )
            done = decode(resumed.text)
        assert done[-1][1]["outcome"] == {"type": "success"}
        assert done[-1][1]["result"] == "Done. ordered 20 x A-1"
        assert done[0][0] > events[-1][0]  # numbering continues across the attempts
        assert case.ledger == ["reorder:A-1:20"] and case.fact in built.said()
        assert (await stored(h, run_id)).status is RunStatus.SUCCESS
        await h.writes.drain()
        thread = await memory_scope(h, user=case.user, agent_id=agent.id, thread=case.thread)
        assert (await thread.history())[-1].content == "Done. ordered 20 x A-1"
        assert await stats(thread, case.reorder, approvals=1)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@dataclass
class Served:
    url: str
    harness: Harness
    agent: Agent


@pytest.fixture
def planner_served() -> Iterator[Served]:
    """An agent that asks before it answers, served over A2A by its own harness (memory on)
    and uvicorn, on a real port."""

    async def planner(input: Any, agent: Runtime) -> str:
        region = await agent.ask("Which region?", options=["eu", "us"])
        return f"deploying {input} to {region}"

    port = _free_port()
    url = f"http://127.0.0.1:{port}/a2a"
    harness = live_harness()
    app = FastAPI()
    agent = harness.wrap(planner, id=f"fv-planner-{uuid.uuid4().hex[:6]}")
    agent.serve_a2a(app, url)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        threading.Event().wait(0.05)
    try:
        yield Served(url, harness, agent)
    finally:
        server.should_exit = True
        thread.join(timeout=10)


@pytest.mark.timeout(TIMEOUT_SECONDS)
async def test_a2a_an_agent_served_and_consumed_as_a_tool(
    planner_served: Served, tmp_path: Path
) -> None:
    async with live_harness() as h:
        case = Case(h, "a2a", tmp_path)
        chat = PlannedChat([("planner", {"message": "the shop"})])
        agent = h.wrap(
            ReAct(system=SYSTEM, model=chat),
            id=case.agent_id,
            tools=[a2a(planner_served.url, name="planner")],
        )
        paused = await agent.run(QUESTION, user=case.user, thread=case.thread)
        # the remote agent's question became this run's own pause
        assert paused.status is RunStatus.PAUSED and paused.interrupt is not None, paused.error
        assert paused.interrupt.question == "Which region?"
        assert (await stored(h, paused.run_id)).status is RunStatus.PAUSED
        done = await agent.resume(
            paused.interrupt.interrupt_id, "answer", answer="eu", reviewer=case.user
        )
        assert done.status is RunStatus.SUCCESS, done.error
        assert done.answer == "Done. deploying the shop to eu"
        assert (await stored(h, done.run_id)).status is RunStatus.SUCCESS
        await h.writes.drain()
        scope = await memory_scope(h, user=case.user, agent_id=agent.id, thread=case.thread)
        assert await stats(scope, "planner", calls=1)
        assert (await scope.history())[-1].content == done.answer
    # the remote side, in agent-runs: the task the pause cancelled, and the one the resumed run
    # answered (the journal gave it the answer)
    remote_runs = await _runs_of(planner_served.harness, planner_served.agent.id)
    assert sorted(r["status"] for r in remote_runs) == ["CANCELLED", "SUCCESS"]


# --------------------------------------------------------------------------- memory features
@pytest.mark.timeout(TIMEOUT_SECONDS)
async def test_a_document_added_is_cited_by_the_next_context(tmp_path: Path) -> None:
    async with live_harness() as h:
        case = Case(h, "documents", tmp_path)
        policy = f"Returns policy {case.suffix}: damaged pallets are refunded within 14 days."
        info = await h.add_document(
            (f"returns-{case.suffix}.txt", policy.encode(), "text/plain"),
            user=case.user,
            title=f"Returns {case.suffix}",
        )
        assert info.status == "READY", info
        seen: list[str] = []

        async def answer(input: str, agent: Runtime) -> str:
            seen.append(agent.context or "")
            return "Within 14 days."

        agent = h.wrap(answer, id=case.agent_id)
        result = await agent.run("How fast are damaged pallets refunded?", user=case.user)
        assert result.status is RunStatus.SUCCESS
        assert "14 days" in seen[0] and case.suffix in seen[0], seen


@pytest.mark.timeout(TIMEOUT_SECONDS)
async def test_feedback_waits_for_review_in_memory_and_is_scored() -> None:
    with StubLangfuse() as langfuse:
        # no grounding check: a sampled run's /v1/verify would add a ``judge`` record
        async with live_harness(grounding_sample=0.0, **langfuse.settings()) as h:
            case = Case(h, "feedback", Path("."))

            async def answer(input: str, agent: Runtime) -> str:
                return "Your warehouse is in Munich."

            agent = h.wrap(answer, id=case.agent_id)
            result = await agent.run("Where is my warehouse?", user=case.user)
            stored_feedback = await h.feedback(result.run_id, "correct", correction="Berlin")
            await h.writes.drain()
            assert stored_feedback is not None and stored_feedback.source == "human"
            assert stored_feedback.review is not None and stored_feedback.review.state == "pending"
            scope = await memory_scope(h, user=case.user, agent_id=agent.id)
            sources = {
                f.source for f in (await scope.feedback.page_for("run", result.run_id)).items
            }
            assert sources == {"system", "human"}
    [score] = [s for s in langfuse.posted("/api/public/scores") if s["name"] == "feedback"]
    assert score["value"] == 0.0 and score["comment"] == "Berlin"
    assert score["traceId"] == telemetry.trace_hex(result.run_id)


# --------------------------------------------------------------------------- evaluation
class Judge:
    """A judge model: 1 when the graded answer names what was expected, else 0 — the strict
    JSON ``llm_judge`` asks for. It keeps every prompt it was sent."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def complete(self, messages: list[dict[str, Any]], **body: Any) -> dict[str, Any]:
        prompt = str(messages[-1]["content"])
        self.prompts.append(prompt)
        sections = dict(part.split("\n", 1) for part in prompt.split("## ")[1:])
        expected = sections.get("Expected answer", "").strip().casefold()
        graded = sections.get("Answer to grade", "").strip().casefold()
        good = bool(graded) and expected in graded
        verdict = {"score": 1.0 if good else 0.0, "reasoning": "names the expected answer"}
        return {"choices": [{"message": {"role": "assistant", "content": json.dumps(verdict)}}]}


ANSWERS: Final = {
    "Who supplies steel to the Berlin office?": "Acme Steel supplies the Berlin office.",
    "What is Acme Steel's supplier id?": "Acme Steel's supplier id is SUP-40.",
}


async def answer_from_table(question: str, agent: Runtime) -> str:
    return ANSWERS[question]


@pytest.fixture
def spans(monkeypatch: pytest.MonkeyPatch) -> InMemorySpanExporter:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(telemetry, "_tracer", provider.get_tracer("fv"))
    return exporter


@pytest.mark.timeout(TIMEOUT_SECONDS)
async def test_offline_evaluation_scores_a_langfuse_dataset(spans: InMemorySpanExporter) -> None:
    suffix = uuid.uuid4().hex[:8]
    items = [
        {"id": f"i-{n}", "status": "ACTIVE", "input": q, "expectedOutput": a, "metadata": {}}
        for n, (q, a) in enumerate(ANSWERS.items())
    ]
    with StubLangfuse(f"fv-golden-{suffix}", items) as langfuse:
        async with live_harness(grounding_sample=0.0, **langfuse.settings()) as h:
            h.evals.judge_model = judge = Judge()
            agent = h.wrap(answer_from_table, id=f"fv-eval-{suffix}")
            user = f"fv-eval-{suffix}"
            scope = await memory_scope(h, user=user, agent_id=agent.id)
            # unique per test: the memory SDK's default idempotency key leaves the user out
            await scope.remember(
                f"The Berlin office buys its steel from Acme Steel, supplier id SUP-40 ({suffix}).",
                visibility="USER",
            )
            report = await h.evaluate(
                agent,
                f"fv-golden-{suffix}",
                [grounding(), exact_match(), llm_judge("Names the supplier or its id.")],
                run_name=f"fv-{suffix}",
                user=user,
            )
            await h.writes.drain()
            for item in report.items:  # each item is a real run, recorded in memory
                assert item.run_id is not None
                [outcome] = await outcome_of(scope, item.run_id)
                assert outcome.verdict == "confirm"
                assert (await stored(h, item.run_id)).status is RunStatus.SUCCESS
    assert [i.status for i in report.items] == ["success", "success"], report
    assert report.summary["exact_match"].mean == 1.0
    assert report.summary["llm_judge"].mean == 1.0 and len(judge.prompts) == 2
    grounded = report.summary["grounding"]
    assert grounded.count == 2 and grounded.mean is not None and grounded.mean > 0.5, report
    assert report.experiment_id == StubLangfuse.DATASET_RUN
    traces = {telemetry.trace_hex(i.run_id or "") for i in report.items}
    links = langfuse.posted("/api/public/dataset-run-items")
    assert sorted(link["datasetItemId"] for link in links) == ["i-0", "i-1"]
    assert {link["traceId"] for link in links} == traces
    scored = {(s["name"], s["traceId"]) for s in langfuse.posted("/api/public/scores")}
    assert scored >= {(n, t) for n in ("grounding", "exact_match", "llm_judge") for t in traces}
    for item in report.items:  # Langfuse v4 builds the experiment from the spans
        trace_id = telemetry.trace_id_of(item.run_id or "")
        [root] = [
            s
            for s in spans.get_finished_spans()
            if s.context is not None
            and s.context.trace_id == trace_id
            and s.name.startswith("invoke_agent")
        ]
        attributes = dict(root.attributes or {})
        assert attributes["langfuse.experiment.id"] == StubLangfuse.DATASET_RUN
        assert attributes["langfuse.experiment.name"] == f"fv-{suffix}"
        assert attributes["langfuse.experiment.item.id"] in ("i-0", "i-1")


@pytest.mark.timeout(TIMEOUT_SECONDS)
async def test_online_judges_score_every_sampled_run() -> None:
    suffix = uuid.uuid4().hex[:8]
    judge = Judge()
    with StubLangfuse() as langfuse:
        async with live_harness(
            judge_sample=1.0, judges=[llm_judge("Answers the question.")], **langfuse.settings()
        ) as h:
            h.evals.judge_model = judge
            agent = h.wrap(answer_from_table, id=f"fv-judged-{suffix}")
            user = f"fv-judged-{suffix}"
            scope = await memory_scope(h, user=user, agent_id=agent.id)
            await scope.remember(
                f"Acme Steel's supplier id is SUP-40 ({suffix}).", visibility="USER"
            )
            runs = [await agent.run(q, user=user) for q in ANSWERS]
            await h.writes.drain()
            assert all(r.status is RunStatus.SUCCESS for r in runs)
    judged = [s for s in langfuse.posted("/api/public/scores") if s["name"] == "llm_judge"]
    assert {s["traceId"] for s in judged} == {telemetry.trace_hex(r.run_id) for r in runs}
    assert all(s["value"] == 1.0 for s in judged)
    assert any("SUP-40" in p for p in judge.prompts)  # the judge read the run's memory context
