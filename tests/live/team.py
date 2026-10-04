"""A team's own agent, built from Trellis's blocks with no harness agent (Way 2): a LangGraph
graph — a planned model, the team's tools, LangGraph's own checkpointer and ``interrupt()`` —
whose prompt gets the memory service's context, whose tools are checked by governance, whose
turn is recorded in memory, and whose pauses its runner makes visible in agent-runs.

Nothing here is the harness's runtime: the graph is plain LangGraph, and the runner is the few
lines a team writes around ``graph.ainvoke`` (start the run, pause it on an interrupt, resume
it from the checkpointer, finish it). ``test_live_pluggable`` and ``test_live_mixed`` drive it.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Final

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, interrupt

from tests.live.conftest import LIVE_TIMEOUT, MEMORY_URL
from tests.support.planned import PlannedChatModel
from trellis.contracts import (
    Interrupt,
    InterruptReason,
    RunRecord,
    RunStart,
    RunStatus,
    ToolCall,
    new_id,
)
from trellis.harness.governance import Decision, Governance, Rejected
from trellis.memory import MemoryClient, MemoryContext
from trellis.runs import RunsClient

#: What the model is told, above the memory service's context.
SYSTEM: Final = "You are the procurement assistant. Use the tools you are asked to use."
Tools = Mapping[str, Callable[..., Awaitable[Any]]]


class State(MessagesState):
    """The graph's state: the conversation, the context pushed into it, and the answer."""

    bundle_id: str | None
    context: str
    answer: str


def graph(
    memory: MemoryContext, model: PlannedChatModel, tools: Tools, checkpointer: InMemorySaver
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """recall (the memory service's context) → model ⇄ tools → remember (the turn)."""

    async def recall(state: State) -> dict[str, Any]:
        pushed = await memory.context(_question(state))
        return {"bundle_id": pushed.bundle_id, "context": pushed.rendered}

    async def think(state: State) -> dict[str, Any]:
        prompt = SystemMessage(f"{SYSTEM}\n\n{state['context']}")
        return {"messages": [await model.ainvoke([prompt, *state["messages"]])]}

    async def act(state: State) -> dict[str, Any]:
        last = state["messages"][-1]
        assert isinstance(last, AIMessage)
        results = []
        for call in last.tool_calls:
            try:
                output = str(await tools[call["name"]](**call["args"]))
            except Rejected as exc:
                output = str(exc)
            results.append(ToolMessage(output, tool_call_id=call["id"]))
        return {"messages": results}

    async def remember(state: State) -> dict[str, Any]:
        answer = str(state["messages"][-1].content)
        await memory.history.add([("USER", _question(state)), ("ASSISTANT", answer)])
        return {"answer": answer}

    def route(state: State) -> str:
        last = state["messages"][-1]
        return "act" if isinstance(last, AIMessage) and last.tool_calls else "remember"

    builder = StateGraph(State)
    builder.add_node("recall", recall)
    builder.add_node("think", think)
    builder.add_node("act", act)
    builder.add_node("remember", remember)
    builder.add_edge(START, "recall")
    builder.add_edge("recall", "think")
    builder.add_conditional_edges("think", route, ["act", "remember"])
    builder.add_edge("act", "think")
    builder.add_edge("remember", END)
    return builder.compile(checkpointer=checkpointer)


def _question(state: State) -> str:
    first = next(m for m in state["messages"] if isinstance(m, HumanMessage))
    return str(first.content)


@dataclass
class Team:
    """The team's wiring of the blocks: the memory service, agent-runs and governance from
    the environment (``MEMORY_URL``, ``RUNS_URL``, ``TRELLIS_API_KEY``), in the key's tenant;
    the graphs' checkpointer; and the decisions governance asked a person about."""

    agent_id: str
    tenant: str
    memory: MemoryClient
    runs: RunsClient
    governance: Governance
    checkpointer: InMemorySaver = field(default_factory=InMemorySaver)
    asked: list[Decision] = field(default_factory=list)

    @classmethod
    async def open(cls, agent_id: str) -> Team:
        memory = MemoryClient(MEMORY_URL, timeout=LIVE_TIMEOUT)
        tenant = (await memory.tenant.keys.whoami()).tenant_id
        assert tenant is not None, "the live suite's key is a tenant's"
        governance = Governance.from_env(agent_id=agent_id, tenant=tenant)
        return cls(agent_id, tenant, memory, RunsClient(), governance)

    async def aclose(self) -> None:
        await self.governance.aclose()
        await self.runs.aclose()
        await self.memory.aclose()

    def scope(
        self, user: str, thread: str | None = None, run_id: str | None = None
    ) -> MemoryContext:
        """The memory service for one of the agent's runs (or, without ``run_id``, its user)."""
        scope: dict[str, Any] = {
            "tenant_id": self.tenant,
            "user_id": user,
            "agent_id": self.agent_id,
        }
        if thread is not None:
            scope["thread_id"] = thread
        if run_id is not None:
            scope["agent_run_id"] = run_id
        return self.memory.bind(**scope)

    def ask(self, decision: Decision) -> Any:
        """``on_ask`` for ``governed``: LangGraph's own ``interrupt()`` — the graph stops here,
        and on resume the person's answer (``True``, ``False`` or edited arguments) is the
        verdict."""
        self.asked.append(decision)
        return interrupt(
            {"question": decision.question, "tool": decision.tool, "args": dict(decision.args)}
        )

    async def start(self, question: str, *, user: str, thread: str) -> RunRecord:
        """The run in agent-runs, ``RUNNING`` in this process."""
        start = RunStart(
            run_id=new_id("run_"),
            tenant_id=self.tenant,
            agent_id=self.agent_id,
            user_id=user,
            thread_id=thread,
            input=question,
        )
        return await self.runs.start(start)

    async def advance(
        self,
        app: CompiledStateGraph[Any, Any, Any, Any],
        record: RunRecord,
        *,
        resume: Any = None,
        assignee: str | None = None,
    ) -> RunRecord:
        """Run the graph for ``record`` (from its question, or with ``resume`` from its
        checkpoint): a graph that interrupts pauses the run for approval in agent-runs, one
        that ends finishes it with its answer."""
        config: Any = {"configurable": {"thread_id": record.run_id}}
        given: Any = (
            Command(resume=resume)
            if resume is not None
            else {"messages": [HumanMessage(str(record.input))]}
        )
        out = await app.ainvoke(given, config)
        waiting = out.get("__interrupt__")
        if waiting:
            asked = waiting[0].value
            current = await self.runs.get(record.run_id, tenant=self.tenant)
            assert current is not None
            return await self.runs.pause(
                Interrupt(
                    interrupt_id=f"{record.run_id}.{current.attempt}.1",
                    tenant_id=self.tenant,
                    run_id=record.run_id,
                    reason=InterruptReason.APPROVAL,
                    question=asked["question"],
                    tool_call=ToolCall(tool=asked["tool"], args=asked["args"]),
                    assignee=assignee,
                ),
                checkpoint={"langgraph_thread": record.run_id},
            )
        return await self.runs.finish(
            record.run_id, RunStatus.SUCCESS, output=out["answer"], tenant=self.tenant
        )
