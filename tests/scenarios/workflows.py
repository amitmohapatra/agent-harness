"""Workflows, as each framework builds them, for scenario 12 (WF) — registered as adapters of
:mod:`tests.scenarios.brains` so a worker process (``tests.scenarios.serve``) builds the same:

* ``wf_graph`` / ``wf_graph_cp``: a hand-built LangGraph graph — a plan node fanning out with
  ``Send`` to three ``fetch`` nodes (a read each, at once), a subgraph (a write, then a slow
  read), a branch (``pay``: an irreversible call that asks a person; else straight on), and a
  last node (a write) that answers. ``_cp`` is compiled with the run's checkpointer
  (``RunCheckpointer``: it continues from the node it paused in), the other with none (it is run
  again from its input, the journal replaying what ran). A message with ``confirm`` passes a node
  that asks with LangGraph's own ``interrupt()``.
* ``wf_deepagents``: Deep Agents planning (``write_todos``) and ``task`` sub-agents (a dict
  sub-agent with a brain model of its own), each task a child run.
* ``wf_handoffs``: an OpenAI Agents SDK handoff chain (triage → billing → triage) and
  ``wf_tools``: an orchestrator calling a specialist agent as a tool (``Agent.as_tool``).

Each step calls the scenarios' tools (``tests.scenarios.kit``), so the ledger counts what ran.
"""

from __future__ import annotations

import operator
from pathlib import Path
from typing import Annotated, Any, cast

from agents import Agent as OpenAIAgent
from agents import handoff
from deepagents import create_deep_agent
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Send, interrupt

from tests.scenarios import brains
from tests.scenarios.brains import TOKEN, BrainChatModel, BrainModel, Built, text
from tests.support.arm import native_tools
from trellis import Harness
from trellis.harness.middleware import RunCheckpointer

#: the tool calls a full ``wf_graph`` run makes: three fetches, a note, a wait, a transfer, a note
GRAPH_CALLS = 7


class Flow(MessagesState):
    fetched: Annotated[list[str], operator.add]
    key: str


def _asked(state: Any) -> str:
    said = [text(m.content) for m in state["messages"] if getattr(m, "type", "") == "human"]
    return said[-1] if said else ""


async def _graph(
    h: Harness, tools: list[Any], tmp: Path, claude: Any, *, checkpointed: bool
) -> Built:
    by_name = {t.name: t for t in await native_tools(h, *tools, framework="langgraph")}

    async def plan(state: Flow) -> dict[str, Any]:
        return {}

    def fan_out(state: Flow) -> list[Send]:
        return [Send("fetch", {"key": f"k{n}", "messages": []}) for n in range(3)]

    async def fetch(state: Flow) -> dict[str, Any]:
        return {"fetched": [str(await by_name["lookup"].ainvoke({"key": state["key"]}))]}

    async def step1(state: Flow) -> dict[str, Any]:
        await by_name["note"].ainvoke({"text": f"step1 {TOKEN.findall(_asked(state))[-1]}"})
        return {}

    async def slow(state: Flow) -> dict[str, Any]:
        seconds = 8.0 if "slowly" in _asked(state) else 0.2
        await by_name["wait"].ainvoke({"seconds": seconds})
        return {}

    sub = StateGraph(Flow)
    sub.add_node("step1", step1)
    sub.add_node("slow", slow)
    sub.add_edge(START, "step1")
    sub.add_edge("step1", "slow")
    sub.add_edge("slow", END)

    def branch(state: Flow) -> str:
        asked = _asked(state)
        return "confirm" if "confirm" in asked else "pay" if "pay" in asked else "finish"

    async def confirm(state: Flow) -> dict[str, Any]:
        said = interrupt("Go on with the payment?")
        return {"fetched": [f"confirmed:{said}"]}

    async def pay(state: Flow) -> dict[str, Any]:
        await by_name["transfer"].ainvoke({"amount": 9, "to": "wf", "meta": {"step": "pay"}})
        return {}

    async def finish(state: Flow) -> dict[str, Any]:
        mine = TOKEN.findall(_asked(state))[-1]
        await by_name["note"].ainvoke({"text": f"final {mine}"})
        got = ",".join(sorted(state.get("fetched") or []))
        return {"messages": [AIMessage(f"ANS {mine.upper()} SEEN  MEM  LAST {got}")]}

    graph = StateGraph(Flow)
    graph.add_node("plan", plan)
    graph.add_node("fetch", fetch)
    graph.add_node("process", sub.compile())
    graph.add_node("confirm", confirm)
    graph.add_node("pay", pay)
    graph.add_node("finish", finish)
    graph.add_edge(START, "plan")
    graph.add_conditional_edges("plan", fan_out, ["fetch"])
    graph.add_edge("fetch", "process")
    graph.add_conditional_edges("process", branch, ["confirm", "pay", "finish"])
    graph.add_edge("confirm", "pay")
    graph.add_edge("pay", "finish")
    graph.add_edge("finish", END)
    compiled = graph.compile(checkpointer=RunCheckpointer()) if checkpointed else graph.compile()
    return Built(compiled, tools)


async def _wf_graph(h: Harness, tools: list[Any], tmp: Path, claude: Any) -> Built:
    return await _graph(h, tools, tmp, claude, checkpointed=False)


async def _wf_graph_cp(h: Harness, tools: list[Any], tmp: Path, claude: Any) -> Built:
    return await _graph(h, tools, tmp, claude, checkpointed=True)


async def _wf_deepagents(h: Harness, tools: list[Any], tmp: Path, claude: Any) -> Built:
    built = await native_tools(h, *tools, framework="deepagents")
    helper = {
        "name": "helper",
        "description": "Does one step of the work.",
        "system_prompt": "You do one step.",
        "model": BrainChatModel(),
        "tools": built,
    }
    graph = create_deep_agent(
        model=BrainChatModel(),
        tools=built,
        subagents=cast(Any, [helper]),
        system_prompt=brains.SYSTEM,
    )
    return Built(graph, tools)


async def _own(h: Harness, tools: list[Any]) -> list[Any]:
    """``tools`` as an OpenAI agent's own function tools (through the harness's bridge): the
    tools of an agent handed off to are its own — the run's tools are the wrapped agent's."""
    names = {getattr(t, "name", None) or t.spec.name for t in tools}
    return [t for t in await native_tools(h, *tools, framework="openai_agents") if t.name in names]


async def _wf_handoffs(h: Harness, tools: list[Any], tmp: Path, claude: Any) -> Built:
    billing = OpenAIAgent(
        name="billing",
        instructions="You take payments.",
        model=BrainModel(),
        tools=await _own(h, tools),
    )
    triage = OpenAIAgent(
        name="triage", instructions="You route.", model=BrainModel(), handoffs=[handoff(billing)]
    )
    billing.handoffs = [handoff(triage)]  # a cycle: billing hands back
    return Built(triage, tools)


async def _wf_tools(h: Harness, tools: list[Any], tmp: Path, claude: Any) -> Built:
    specialist = OpenAIAgent(
        name="specialist",
        instructions="You look things up.",
        model=BrainModel(),
        tools=await _own(h, tools),
    )
    orchestrator = OpenAIAgent(
        name="orchestrator",
        instructions="You orchestrate.",
        model=BrainModel(),
        tools=[specialist.as_tool(tool_name="specialist", tool_description="Asks the specialist.")],
    )
    return Built(orchestrator, tools)


brains.BUILDERS.update(  # type: ignore[attr-defined]
    {
        "wf_graph": _wf_graph,
        "wf_graph_cp": _wf_graph_cp,
        "wf_deepagents": _wf_deepagents,
        "wf_handoffs": _wf_handoffs,
        "wf_tools": _wf_tools,
    }
)
