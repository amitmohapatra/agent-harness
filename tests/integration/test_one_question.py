"""One question a call, on every adapter, whatever asks: each tool call that needs a person's
approval is ONE question, runs at most once (once approved, never rejected), and its decision
is recorded once and counted once.

For each case the ledger (:func:`ledger`) checks, across every attempt of every run the case
made:

* the questions a person was shown (the runs' pauses; on a stream, its ``INTERRUPT`` events)
  are the expected number;
* the tool's body ran exactly once per approved call (with an edit's arguments), never for a
  rejected one;
* the run's trajectory (the journal) holds one record per call that ran, the memory service one
  tool record (``record_tool``), and the run's budget counts one tool call (``spent``);
* each decision is one feedback record (``target_kind="tool_call"``) sent to the memory service
  once, on the run that recorded the call.

The core matrix (:func:`test_one_question_a_call`) runs every adapter of
``tests.support.adapters.BUILDERS`` — the function target, ``ReAct``, ``create_agent`` (on its
own and with ``trellis.middleware()``), a hand-built graph, Deep Agents, OpenAI Agents, the
Claude Agent SDK (the real SDK driving the scripted CLI) and Kimi Code over ACP — against each
way a harness tool's call asks (:data:`SOURCES`) and each flow (:data:`FLOWS`). The cases after
it are the library's own approvals and the paths only one adapter has.

Not run here, and why:

* Always allow where a catalog rule or a hook's ``Ask`` decides the call: it is not offered
  (``governance.decision.choices``; ``test_write_asks``: a rule and a hook decide before
  always);
* a hand-built graph's ``interrupt_on``: LangChain's HITL middleware is ``create_agent``'s (the
  hand-built graph is in the core matrix for every other source);
* an edit of an ACP agent's own call: the agent makes it again with the edited arguments
  (``test_acp.test_an_edited_call_is_made_again_with_its_arguments_and_runs``); an ACP
  sub-agent's call shows no update, so it is not recorded — its question and feedback are
  checked;
* Kimi Code itself for its own Bash call (``test_kimi``: approved, rejected, streamed) — its
  permission requests run against the scripted ACP agent here;
* a hosted MCP tool's approval (``test_openai_agents_features``);
* a built-in's count in the run's budget: ``Budget.max_tool_calls`` counts harness tool calls,
  so a framework's own built-in counts none (checked: zero, never twice).

What other tests already assert of one approval is not asked again here:
``test_one_approval`` (a framework gate and governance on one call, a hook's denial after the
gate), ``test_claude_sdk_features.test_a_built_in_governance_asked_about_is_one_question_
whatever_made_it_ask`` (a catalog rule, a hook's ``Ask``, a rewrite x main agent / subagent x
query / live), ``test_openai_agents_mcp`` (a server's policy and governance),
``test_hitl_feedback`` (the feedback's id, a retried send, a refused resume)."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final, cast

import pytest
from agents import Agent, LocalShellTool, function_tool
from deepagents import create_deep_agent
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from openai.types.responses.response_output_item import LocalShellCall, LocalShellCallAction

from tests.integration.test_openai_agents_mcp import results, stdio
from tests.support.adapters import BUILDERS, SYSTEM
from tests.support.arm import arm, declaring, native_tools, with_governance
from tests.support.catalog import FakeCatalog
from tests.support.kimi import detached
from tests.support.memory import FakeMemoryService
from tests.support.openai_model import Items, ScriptedModel
from tests.support.planned import Call, PlannedChat, PlannedChatModel, PlannedModel
from trellis import Ask, Harness, Hooks, ReAct, Result, Runtime, Settings, current, tool
from trellis.contracts import Interrupt, RunEvent, RunEventType, RunStatus, ToolCall
from trellis.harness.adapters.acp import KIMI, ACPAgent
from trellis.harness.automatic import JournalBackend
from trellis.harness.bridges.helpers import asked_inside
from trellis.harness.governance import Governance
from trellis.harness.governance.catalog import Rule
from trellis.harness.governance.decision import ALWAYS
from trellis.harness.subagents import asked_by

#: The tool bodies that ran, in order: (tool, amount).
RAN: list[tuple[str, int]] = []
THREAD: Final = "chat-1"


@tool(side_effects="write")
def pay(amount: int) -> str:
    """Pay an amount."""
    RAN.append(("pay", amount))
    return f"paid {amount}"


@tool
def poke(amount: int) -> str:
    """Poke an amount (it says nothing of what it does)."""
    RAN.append(("poke", amount))
    return f"poked {amount}"


@tool(side_effects="read")
def look(amount: int) -> str:
    """Look at an amount."""
    RAN.append(("look", amount))
    return f"looked {amount}"


class Asking(Hooks):
    """A run hook asking about every call of ``look``: your rule in code."""

    async def before_tool(self, call: ToolCall) -> Ask | None:
        return Ask(f"Look at {call.args['amount']}?") if call.tool == "look" else None


class Watch(Hooks):
    """Every attempt's runtime, by run, as it ended: what its journal holds."""

    def __init__(self) -> None:
        self.ended: dict[str, Runtime] = {}

    async def on_run_end(self, run: Runtime, result: Result) -> None:
        self.ended[run.run_id] = run


@dataclasses.dataclass(frozen=True)
class Source:
    """A way a harness tool's call asks a person: the tool, the catalog's rule, the hook."""

    tool: Any
    rule: str | None = None
    hooked: bool = False
    #: whether Always allow is offered (no rule, no hook decides the call)
    always: bool = True

    @property
    def name(self) -> str:
        return str(self.tool.spec.name)


#: Each way a harness tool's call needs a person's approval.
SOURCES: Final = {
    # a tool that declares no side effects: nothing known is never a reason to run
    "undeclared": Source(poke),
    # a catalog rule that asks (``approve_when``) on a tool that only reads
    "rule": Source(look, rule="amount > 0", always=False),
    # a write: asks by default
    "write": Source(pay),
    # a run hook's ``Ask`` on a tool that only reads
    "hook": Source(look, hooked=True, always=False),
}


@pytest.fixture(autouse=True)
def _reset() -> None:
    RAN.clear()


# --------------------------------------------------------------------------- the ledger


@dataclasses.dataclass
class Asked:
    """What one case made of its runs: every result a person saw, and each question."""

    results: list[Result] = dataclasses.field(default_factory=list)
    #: the ``INTERRUPT`` events its streams showed (a stream flow only)
    streamed: list[RunEvent] = dataclasses.field(default_factory=list)

    @property
    def questions(self) -> list[Any]:
        return [r.interrupt for r in self.results if r.status is RunStatus.PAUSED]

    @property
    def runs(self) -> list[str]:
        return list(dict.fromkeys(r.run_id for r in self.results))


async def settled(
    agent: Any,
    result: Result,
    asked: Asked,
    answer: Callable[[Any], dict[str, Any]],
    *,
    resumer: Any = None,
    watch_events: bool = False,
) -> Result:
    """``result`` and the resumes that answer each question (``answer(interrupt)``: the
    resume's arguments) until the run ends; ``resumer``: the agent that resumes (another
    worker's); ``watch_events``: each resume followed on the run's event stream."""
    asked.results.append(result)
    while result.status is RunStatus.PAUSED:
        assert result.interrupt is not None
        assert len(asked.questions) < 8, [q.question for q in asked.questions]
        by = resumer or agent
        kwargs = {"reviewer": "lee", **answer(result.interrupt)}
        if watch_events:
            following = asyncio.create_task(_interrupts(by, result.run_id, asked))
            await asyncio.sleep(0)
            result = await by.resume(result.interrupt.interrupt_id, **kwargs)
            await asyncio.wait_for(following, 30)
        else:
            result = await by.resume(result.interrupt.interrupt_id, **kwargs)
        asked.results.append(result)
    return result


async def _interrupts(agent: Any, run_id: str, asked: Asked) -> None:
    async for event in agent.events(run_id):
        if event.type is RunEventType.INTERRUPT:
            asked.streamed.append(event)


async def streamed(agent: Any, input: str, asked: Asked, **kwargs: Any) -> Result:
    """``agent.stream``'s run, as the :class:`Result` its ``RUN_FINISHED`` says."""
    events = [e async for e in agent.stream(input, **kwargs)]
    asked.streamed.extend(e for e in events if e.type is RunEventType.INTERRUPT)
    last = events[-1]
    assert last.type is RunEventType.RUN_FINISHED, last
    interrupt = last.data.get("interrupt")
    status = RunStatus.PAUSED if interrupt is not None else RunStatus.SUCCESS

    return Result(
        run_id=last.run_id,
        status=status,
        interrupt=None if interrupt is None else Interrupt.model_validate(interrupt),
        answer=last.data.get("result"),
    )


def asked_on(interrupt: Any) -> str:
    """The run a question's call is recorded under: a helper's the framework spawned (its
    payload names it), a sub-agent's whose question its parent carries, else its own."""
    child = asked_by(interrupt)
    return asked_inside(interrupt) or (child or {}).get("run_id") or interrupt.run_id


async def ledger(
    h: Harness,
    memory: FakeMemoryService,
    watch: Watch,
    asked: Asked,
    tool: str,
    *,
    questions: int,
    ran: int,
    harness_tool: bool = True,
    counted: int | None = None,
) -> None:
    """The case's accounting, the tool's runs checked by the case: ``questions`` asked; per
    call that ran (``ran``), one record on the run's trajectory (a harness tool's call: a
    framework's own built-in is not journaled), one tool record in the memory service, one
    count in the run's budget (``counted``: as the top runs count it — a harness tool call,
    each counted on its run and the runs it works inside; a built-in is not a harness tool
    call); and one feedback record per decision, sent once, on the run that recorded the call
    (:func:`asked_on`) — where the memory service records it too."""
    await h.writes.drain()
    shown = asked.questions
    assert len(shown) == questions, [q.question for q in shown]
    journals = [r.replay.journal for r in watch.ended.values()]
    trajectory = [c for j in journals for c, _ in j.trajectory if c.tool == tool]
    assert len(trajectory) == (ran if harness_tool else 0), trajectory
    tops = [watch.ended[r].replay.journal for r in asked.runs if r in watch.ended]
    spent = sum(j.spent.get("tool_calls", 0) for j in tops)
    expected = counted if counted is not None else ran if harness_tool else 0
    assert spent == expected, [j.spent for j in tops]
    records = [c for c in memory.named("record_tool") if c.body["tool"] == tool]
    assert len(records) == ran, [c.body for c in records]
    sent = [c for c in memory.named("feedback") if c.body["target_kind"] == "tool_call"]
    assert len(sent) == questions, [c.body for c in sent]
    assert len({c.body["feedback_id"] for c in sent}) == questions
    where = [asked_on(q) for q in shown]
    assert [c.body.get("agent_run_id") or c.scope.get("agent_run_id") for c in sent] == where
    if ran == questions:  # each call asked about ran: recorded where its decision was fed back
        recorded_on = [str(c.scope.get("agent_run_id")) for c in records]
        assert sorted(recorded_on) == sorted(where), (recorded_on, where)
    stored = [f for f in memory.stored_feedback.values() if f["target_kind"] == "tool_call"]
    assert len(stored) == questions


# --------------------------------------------------------------------------- the core matrix


@dataclasses.dataclass(frozen=True)
class Flow:
    """How a case runs: its turns (each a plan of amounts), how each question is answered,
    and what it must come to for a source."""

    turns: tuple[tuple[int, ...], ...]
    decision: str = "approve"
    remember: str = "once"
    stream: bool = False
    #: resumed by another harness on the same run store (another worker)
    fresh: bool = False
    #: the second turn in another conversation of the same user
    elsewhere: bool = False
    #: each question's own decision, in order (else ``decision`` for every one)
    decisions: tuple[str, ...] = ()


FLOWS: Final = {
    "approve": Flow(((5,),)),
    "reject": Flow(((5,),), decision="reject"),
    "edit": Flow(((5,),), decision="edit"),
    "stream": Flow(((5,),), stream=True),
    "fresh-harness": Flow(((5,),), fresh=True),
    # two calls needing approval in one turn
    "two-calls": Flow(((5, 6),)),
    # the same call made again later in the turn: asked again
    "again-in-turn": Flow(((5, 5),)),
    # ... each its own decision: the second does not take the first's answer
    "again-approve-reject": Flow(((5, 5),), decisions=("approve", "reject")),
    "again-reject-edit": Flow(((5, 5),), decisions=("reject", "edit")),
    # a rejection is the call's it answered: the same call made again later is asked again
    "again-reject-approve": Flow(((5, 5),), decisions=("reject", "approve")),
    # the same call in the next turn of the conversation: asked again
    "next-turn": Flow(((5,), (5,))),
    # allowed for this conversation: asked zero times after (a hook's Ask: once again)
    "conversation": Flow(((5,), (6,)), remember="run"),
    "conversation-in-turn": Flow(((5, 6),), remember="run"),
    # always allowed: asked zero times in another conversation
    "always": Flow(((5,), (6,)), remember="always", elsewhere=True),
}


def expected(source: Source, flow: Flow) -> tuple[int, list[int]]:
    """The questions a person is asked in ``flow`` about ``source``'s calls, and the amounts
    the tool runs with."""
    calls = [a for turn in flow.turns for a in turn]
    if flow.decisions:
        decided = zip(calls, flow.decisions, strict=True)
        return len(calls), [3 if d == "edit" else a for a, d in decided if d != "reject"]
    if flow.decision == "reject":
        return len(calls), []
    if flow.decision == "edit":
        return len(calls), [3] * len(calls)
    if flow.remember == "once" or source.hooked:
        return len(calls), calls  # a hook's Ask is asked whatever was allowed
    return 1, calls  # allowed beyond its call: the later calls ask nobody


def answering(source: Source, flow: Flow) -> Callable[[Any], dict[str, Any]]:
    left = iter(flow.decisions)

    def answer(interrupt: Any) -> dict[str, Any]:
        decision = next(left, flow.decision)
        if decision == "edit":
            return {"decision": "edit", "answer": {"amount": 3}}
        if flow.remember == "always":
            assert ALWAYS.value in interrupt.option_values, interrupt.options
            return {"decision": "answer", "answer": ALWAYS.value}
        return {"decision": decision, "remember": flow.remember}

    return answer


async def _react_deep(
    h: Harness, tools: list[Any], tmp: Path, plan: list[Call], **_: Any
) -> tuple[Any, list[Any]]:
    """``ReAct`` with the Deep Agents backend (``backend=``): Deep Agents from the first step."""
    return ReAct(system_prompt=SYSTEM, model=PlannedChat(plan), backend=JournalBackend()), tools


#: Every adapter, and ``ReAct`` on the Deep Agents backend.
FRAMEWORKS: Final[dict[str, Any]] = {**BUILDERS, "react_deepagents": _react_deep}
#: The adapters whose model is given the conversation's earlier turns, their tool results
#: included (the function target has no model; the scripted Claude CLI and Kimi's planned
#: endpoint follow each turn's own plan).
CARRIED: Final = {
    "react",
    "react_deepagents",
    "langgraph",
    "middleware",
    "graph",
    "deepagents",
    "openai_agents",
}


def cases() -> list[Any]:
    found = []
    for framework in FRAMEWORKS:
        for source_name, source in SOURCES.items():
            for flow_name, flow in FLOWS.items():
                if flow.remember == "always" and not source.always:
                    continue  # not offered where a rule or a hook decides the call
                found.append(pytest.param(framework, source_name, flow_name))
    return found


@pytest.mark.parametrize(("framework", "source_name", "flow_name"), cases())
async def test_one_question_a_call(
    memory_service: FakeMemoryService,
    framework: str,
    source_name: str,
    flow_name: str,
    tmp_path: Path,
) -> None:
    source, flow = SOURCES[source_name], FLOWS[flow_name]
    if source.rule is not None:
        memory_service.catalog = {
            source.name: {"side_effects": "read", "approve_when": source.rule}
        }
    watch = Watch()
    hooks: list[Hooks] = [watch, *([Asking()] if source.hooked else [])]
    asked = Asked()
    answer = answering(source, flow)
    async with Harness(config=Settings(), memory=memory_service.client()) as h:

        async def built(on: Harness, n: int) -> Any:
            # each turn's agent plans that turn's calls (any agent of the conversation reads
            # what it allowed); a planned model that reads the conversation's earlier turns
            # counts their results too: its plan holds their calls
            said = flow.turns[: n + 1] if framework in CARRIED and not flow.elsewhere else [()]
            turns = [*said[:-1], flow.turns[n]]
            plan: list[Call] = [(source.name, {"amount": a}) for t in turns for a in t]
            target, tools = await FRAMEWORKS[framework](on, [source.tool], tmp_path, plan)
            return arm(on.wrap(target, id=f"payer-{framework}-{n}", hooks=hooks), tools)

        for n in range(len(flow.turns)):
            agent = await built(h, n)
            thread = f"{THREAD}-{n}" if flow.elsewhere and n else THREAD
            if flow.stream:
                first = await streamed(agent, "go", asked, user="ada", thread=thread)
            else:
                first = await agent.run("go", user="ada", thread=thread)
            if flow.fresh and first.status is RunStatus.PAUSED:
                async with Harness(
                    config=Settings(), memory=memory_service.client(), runs=h.runs
                ) as other:
                    resumer = await built(other, n)
                    done = await settled(agent, first, asked, answer, resumer=resumer)
                    await other.writes.drain()
            else:
                done = await settled(agent, first, asked, answer, watch_events=flow.stream)
            assert done.status is RunStatus.SUCCESS, done
        questions, ran = expected(source, flow)
        await ledger(
            h, memory_service, watch, asked, source.name, questions=questions, ran=len(ran)
        )
    assert [a for t, a in RAN if t == source.name] == ran, RAN
    if flow.stream:
        assert len(asked.streamed) == questions, asked.streamed

    await detached()


# --------------------------------------------------------------------------- the library's gate
# LangChain's ``HumanInTheLoopMiddleware`` (``create_agent``) and Deep Agents' ``interrupt_on``:
# the graph's own pause is the one question for its call, whether governance asks too (a
# write: "both") or not (a read: "library"). ``test_one_approval`` asks the same of an approval
# (once, an edit, a hook's denial, another process); here every decision is accounted.

GATES: Final = {"both": pay, "library": look}


async def _gated(h: Harness, lib: str, fn: Any, plan: list[Call], saver: Any) -> Any:

    name = fn.spec.name
    model = PlannedChatModel(plan=plan)
    if lib == "deepagents":
        built = await native_tools(h, fn, framework="deepagents")
        return create_deep_agent(
            model=model, tools=built, interrupt_on={name: True}, checkpointer=saver
        )
    built = await native_tools(h, fn, framework="langgraph")
    gate = HumanInTheLoopMiddleware(interrupt_on={name: True})
    return create_agent(model, tools=built, middleware=[gate], checkpointer=saver)


@pytest.mark.parametrize("lib", ["create_agent", "deepagents"])
@pytest.mark.parametrize("gate", list(GATES))
@pytest.mark.parametrize("flow", ["approve", "reject", "edit", "two-calls", "fresh-harness"])
async def test_the_librarys_own_gate_is_the_one_question(
    memory_service: FakeMemoryService, lib: str, gate: str, flow: str
) -> None:

    fn = GATES[gate]
    name = fn.spec.name
    amounts = (5, 6) if flow == "two-calls" else (5,)
    plan: list[Call] = [(name, {"amount": a}) for a in amounts]
    decision = flow if flow in ("reject", "edit") else "approve"
    saver, watch, asked = InMemorySaver(), Watch(), Asked()

    def answer(interrupt: Any) -> dict[str, Any]:
        edited = {"answer": {"amount": 3}} if decision == "edit" else {}
        return {"decision": decision, **edited}

    async with Harness(config=Settings(), memory=memory_service.client()) as h:
        graph = await _gated(h, lib, fn, plan, saver)
        agent = arm(h.wrap(graph, id=f"gated-{lib}", hooks=[watch]), [fn])
        first = await agent.run("go", user="ada", thread=THREAD)
        if flow == "fresh-harness":
            async with Harness(
                config=Settings(), memory=memory_service.client(), runs=h.runs
            ) as other:
                again = await _gated(other, lib, fn, plan, saver)
                resumer = arm(other.wrap(again, id=f"gated-{lib}", hooks=[watch]), [fn])
                done = await settled(agent, first, asked, answer, resumer=resumer)
                await other.writes.drain()
        else:
            done = await settled(agent, first, asked, answer)
        assert done.status is RunStatus.SUCCESS, done
        ran = {"approve": list(amounts), "reject": [], "edit": [3]}[decision]
        await ledger(h, memory_service, watch, asked, name, questions=len(amounts), ran=len(ran))
    assert [a for t, a in RAN if t == name] == ran, RAN


# --------------------------------------------------------------------------- inside a sub-agent
# Deep Agents' ``task``: a sub-agent of ``ReAct`` (on the Deep Agents backend) or of
# ``create_deep_agent`` calls the run's write; the question, its run once, its record and its
# feedback are the run's.


async def _tasked(h: Harness, lib: str, plan: list[Call]) -> tuple[Any, list[Any]]:

    helper = PlannedChatModel(plan=plan)
    told = {"description": "pay them", "subagent_type": "payer"}
    parent = PlannedChatModel(plan=[("task", told)])
    payer = {"name": "payer", "description": "Pays.", "model": helper}
    if lib == "react":  # the sub-agent is offered the run's tools
        return ReAct("s", parent, subagents=[payer]), [pay]

    built = await native_tools(h, pay, framework="deepagents")
    given = {**payer, "system_prompt": "Pay.", "tools": built}
    return create_deep_agent(model=parent, tools=built, subagents=[cast(Any, given)]), [pay]


@pytest.mark.parametrize("lib", ["react", "deepagents"])
@pytest.mark.parametrize("flow", ["approve", "reject", "two-calls"])
async def test_a_task_sub_agents_call_is_one_question(
    memory_service: FakeMemoryService, lib: str, flow: str
) -> None:
    amounts = (5, 6) if flow == "two-calls" else (5,)
    decision = "reject" if flow == "reject" else "approve"
    watch, asked = Watch(), Asked()
    async with Harness(config=Settings(), memory=memory_service.client()) as h:
        target, tools = await _tasked(h, lib, [("pay", {"amount": a}) for a in amounts])
        agent = arm(h.wrap(target, id=f"tasks-{lib}", hooks=[watch]), tools)
        first = await agent.run("pay them", user="ada", thread=THREAD)
        done = await settled(agent, first, asked, lambda _: {"decision": decision})
        assert done.status is RunStatus.SUCCESS, done
        ran = list(amounts) if decision == "approve" else []
        await ledger(
            h,
            memory_service,
            watch,
            asked,
            "pay",
            questions=len(amounts),
            ran=len(ran),
            counted=len(ran) + 1,  # and the task call, a harness tool call too
        )
    assert [a for t, a in RAN if t == "pay"] == ran, RAN


# --------------------------------------------------------------------------- OpenAI Agents
# An OpenAI agent's own tools: a function tool — behind the SDK's ``needs_approval`` alone
# ("library": the catalog says it reads), behind it and governance ("both": a write), or
# governance's alone ("governance") —, an agent called as a tool whose own tool is behind the
# SDK's approval, an agent handed off to whose tool asks, a local shell command. A run the SDK
# paused goes on from its saved state; one the harness paused runs again from its input.


@function_tool
def settle(amount: int) -> str:
    """Settle an amount."""
    RAN.append(("settle", amount))
    return f"settled {amount}"


#: The function tool's gates: the SDK's own approval, and what the catalog says it does.
OPENAI_GATES: Final = {
    "library": (True, "read"),
    "both": (True, "write"),
    "governance": (False, "write"),
}


@pytest.mark.parametrize("gate", list(OPENAI_GATES))
@pytest.mark.parametrize(
    "flow", ["approve", "reject", "two-calls", "again-in-turn", "fresh-harness", "stream"]
)
async def test_an_openai_function_tools_call_is_one_question(
    memory_service: FakeMemoryService, gate: str, flow: str
) -> None:

    needs, risk = OPENAI_GATES[gate]
    amounts = {"two-calls": (5, 6), "again-in-turn": (5, 5)}.get(flow, (5,))
    decision = "reject" if flow == "reject" else "approve"
    gated = dataclasses.replace(settle, needs_approval=needs)
    watch, asked = Watch(), Asked()

    def built(on: Harness) -> Any:
        model = PlannedModel([("settle", {"amount": a}) for a in amounts])
        target = Agent(name="desk", model=model, tools=[gated])
        return declaring(on, settle=risk).wrap(target, id="desk", hooks=[watch])

    async with Harness(config=Settings(), memory=memory_service.client()) as h:
        agent = built(h)
        if flow == "stream":
            first = await streamed(agent, "settle", asked, user="ada", thread=THREAD)
        else:
            first = await agent.run("settle", user="ada", thread=THREAD)
        answer = lambda _: {"decision": decision}  # noqa: E731
        if flow == "fresh-harness":
            async with Harness(
                config=Settings(), memory=memory_service.client(), runs=h.runs
            ) as other:
                done = await settled(agent, first, asked, answer, resumer=built(other))
                await other.writes.drain()
        else:
            done = await settled(agent, first, asked, answer, watch_events=flow == "stream")
        assert done.status is RunStatus.SUCCESS, done
        ran = list(amounts) if decision == "approve" else []
        await ledger(
            h, memory_service, watch, asked, "settle", questions=len(amounts), ran=len(ran)
        )
    assert [a for t, a in RAN if t == "settle"] == ran, RAN
    if flow == "stream":
        assert len(asked.streamed) == len(amounts), asked.streamed


@pytest.mark.parametrize("decision", ["approve", "reject"])
async def test_an_sdk_approval_inside_an_agent_tool_is_one_question(
    memory_service: FakeMemoryService, decision: str
) -> None:
    """The agent tool's own child run records the call: its decision is fed back there."""

    inner = Agent(
        name="checker",
        model=PlannedModel([("settle", {"amount": 5})]),
        tools=[dataclasses.replace(settle, needs_approval=True)],
    )
    outer = PlannedModel([("check", {"input": "settle 5"})])
    desk = Agent(name="desk", model=outer, tools=[inner.as_tool("check", "Check.")])
    watch, asked = Watch(), Asked()
    async with Harness(config=Settings(), memory=memory_service.client()) as h:
        agent = declaring(h, settle="read", check="read").wrap(desk, id="desk", hooks=[watch])
        first = await agent.run("settle 5", user="ada", thread=THREAD)
        done = await settled(agent, first, asked, lambda _: {"decision": decision})
        assert done.status is RunStatus.SUCCESS, done
        ran = 1 if decision == "approve" else 0
        await ledger(
            h, memory_service, watch, asked, "settle", questions=1, ran=ran, counted=ran + 1
        )
    assert [a for t, a in RAN if t == "settle"] == [5] * ran, RAN


@pytest.mark.parametrize("decision", ["approve", "reject"])
async def test_a_call_of_an_agent_handed_off_to_is_one_question(
    memory_service: FakeMemoryService, decision: str
) -> None:

    refunds = Agent(
        name="refunds",
        # the agent handed off to reads the conversation so far: the handoff's result counts
        model=PlannedModel([("transfer_to_refunds", {}), ("settle", {"amount": 5})]),
        tools=[settle],
        handoff_description="Settles.",
    )
    desk = Agent(name="desk", model=PlannedModel([("transfer_to_refunds", {})]), handoffs=[refunds])
    watch, asked = Watch(), Asked()
    async with Harness(config=Settings(), memory=memory_service.client()) as h:
        # a write of the agent's own is announced (its SDK's permission flow is its own): an
        # irreversible one asks
        agent = declaring(h, settle="irreversible").wrap(desk, id="desk", hooks=[watch])
        first = await agent.run("settle 5", user="ada", thread=THREAD)
        done = await settled(agent, first, asked, lambda _: {"decision": decision})
        assert done.status is RunStatus.SUCCESS, done
        ran = 1 if decision == "approve" else 0
        await ledger(h, memory_service, watch, asked, "settle", questions=1, ran=ran)
    assert [a for t, a in RAN if t == "settle"] == [5] * ran, RAN


@pytest.mark.parametrize("decision", ["approve", "reject"])
async def test_a_local_shell_command_is_one_question(
    memory_service: FakeMemoryService, decision: str
) -> None:

    def execute(request: Any) -> str:
        RAN.append(("local_shell", len(request.data.action.command)))
        return "removed"

    call = LocalShellCall(
        id="ls_1",
        call_id="call_ls",
        status="completed",
        type="local_shell_call",
        action=LocalShellCallAction(type="exec", command=["rm", "-rf", "x"], env={}),
    )
    # the harness's pause runs the run again from its input: the model makes the call again
    model = ScriptedModel([Items(call), Items(call), "done"])
    target = Agent(name="ops", model=model, tools=[LocalShellTool(executor=execute)])
    watch, asked = Watch(), Asked()
    async with Harness(config=Settings(), memory=memory_service.client()) as h:
        agent = h.wrap(target, id="ops", hooks=[watch], without={"schedules"})
        first = await agent.run("remove x", user="ada", thread=THREAD)
        done = await settled(agent, first, asked, lambda _: {"decision": decision})
        assert done.status is RunStatus.SUCCESS, done
        ran = 1 if decision == "approve" else 0
        await ledger(h, memory_service, watch, asked, "local_shell", questions=1, ran=ran)
    assert [t for t, _ in RAN] == ["local_shell"] * ran, RAN


@pytest.mark.parametrize("policy", ["always", "never"])
@pytest.mark.parametrize("decision", ["approve", "reject"])
async def test_an_mcp_server_tool_is_one_question(
    memory_service: FakeMemoryService, policy: str, decision: str
) -> None:
    """The server's own ``require_approval`` and governance (``add_note`` writes, as its
    annotations say) ask about one call: one question, whichever asks."""

    watch, asked = Watch(), Asked()
    async with (
        stdio(require_approval=policy) as server,
        Harness(config=Settings(), memory=memory_service.client()) as h,
    ):
        model = PlannedModel([("add_note", {"text": "milk"})], final="ok")
        target = Agent(name="desk", model=model, mcp_servers=[server])
        agent = h.wrap(target, id="desk", hooks=[watch], without={"schedules"})
        first = await agent.run("note milk", user="ada", thread=THREAD)
        done = await settled(agent, first, asked, lambda _: {"decision": decision})
        assert done.status is RunStatus.SUCCESS, done
        ran = 1 if decision == "approve" else 0
        await ledger(h, memory_service, watch, asked, "add_note", questions=1, ran=ran)
        said = results(model)
    assert said.count("added milk") == ran, said


# --------------------------------------------------------------------------- nested arguments


@tool(side_effects="write")
def tag(items: list[str], where: dict[str, str]) -> str:
    """Tag items somewhere."""
    RAN.append(("tag", len(items)))
    return f"tagged {len(items)}"


@pytest.mark.parametrize("decision", ["approve", "reject", "edit"])
async def test_a_call_whose_arguments_nest_is_one_decision_counted(
    memory_service: FakeMemoryService, decision: str
) -> None:
    """A call whose arguments nest (a list, an object) is fed back as any call: the memory
    service takes the decision, its arguments as they are, and counts it."""
    args = {"items": ["a", "b"], "where": {"shelf": "s1"}}

    async def follow(input: Any, agent: Runtime) -> Any:
        return await agent.tools.call("tag", **args)

    watch, asked = Watch(), Asked()
    edited = {"items": ["a"], "where": {"shelf": "s2"}}
    answer = {"decision": decision, **({"answer": edited} if decision == "edit" else {})}
    async with Harness(config=Settings(), memory=memory_service.client()) as h:
        agent = arm(h.wrap(follow, id="tagger", hooks=[watch]), [tag])
        first = await agent.run("tag", user="ada", thread=THREAD)
        done = await settled(agent, first, asked, lambda _: answer)
        assert done.status is RunStatus.SUCCESS, done
        ran = 0 if decision == "reject" else 1
        await ledger(h, memory_service, watch, asked, "tag", questions=1, ran=ran)
        assert h.writes.failed == 0
    [stored] = [
        f for f in memory_service.stored_feedback.values() if f["target_kind"] == "tool_call"
    ]
    # the arguments as the call was asked about, structured as they are
    assert stored["metadata"]["tool"] == "tag" and stored["metadata"]["args"] == args
    assert [n for t, n in RAN if t == "tag"] == {"approve": [2], "reject": [], "edit": [1]}[
        decision
    ]


# --------------------------------------------------------------------------- Claude Code's own
# A built-in Claude Code asks its user about (``can_use_tool``: ``Bash`` is not in its allowed
# tools): one question, on the query path and in a conversation's live session, the main
# agent's call or a foreground subagent's. A built-in is Claude's: not journaled on the run's
# trajectory and not a harness tool call of the budget — it is recorded once as the run's.
# ``test_claude_sdk_features.test_a_built_in_governance_asked_about_is_one_question_whatever_
# made_it_ask`` asks the same of a call governance also asks about (a catalog rule, a hook's
# Ask, a rewrite) on those paths; the background subagent's is below.

CLI: Final = str(Path(__file__).resolve().parents[1] / "support" / "fake_claude_cli.py")
DEPLOY: Final = {"command": "make deploy"}
DRY: Final = {"command": "make deploy -n"}
#: Each flow: the built-in calls the "model" makes, the person's decisions, the commands run.
CLAUDE_FLOWS: Final[dict[str, tuple[list[dict[str, str]], list[str], list[dict[str, str]]]]] = {
    "approve": ([DEPLOY], ["approve"], [DEPLOY]),
    "reject": ([DEPLOY], ["reject"], []),
    "edit": ([DEPLOY], ["edit"], [DRY]),
    "two-calls": ([DEPLOY, DRY], ["approve", "approve"], [DEPLOY, DRY]),
    # the same call made again in the turn: each its own question, each its own decision
    "again-approve-reject": ([DEPLOY, DEPLOY], ["approve", "reject"], [DEPLOY]),
    "again-reject-edit": ([DEPLOY, DEPLOY], ["reject", "edit"], [DRY]),
    "again-reject-approve": ([DEPLOY, DEPLOY], ["reject", "approve"], [DEPLOY]),
}


def claude_options(tmp: Path, script: list[dict[str, Any]], **fields: Any) -> Any:
    from claude_agent_sdk import ClaudeAgentOptions

    env = {
        "FAKE_CLAUDE_SCRIPT": json.dumps(script),
        "FAKE_CLAUDE_RECORD": str(tmp / "cli.json"),
        "FAKE_CLAUDE_SESSIONS": str(tmp),
        "FAKE_CLAUDE_BUILTINS": str(tmp / "builtins.jsonl"),
    }
    return ClaudeAgentOptions(cli_path=CLI, env=env, **fields)


def built_ins(tmp: Path) -> list[dict[str, Any]]:
    log = tmp / "builtins.jsonl"
    return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []


def deciding(decisions: list[str]) -> Callable[[Any], dict[str, Any]]:
    left = iter(decisions)

    def answer(interrupt: Any) -> dict[str, Any]:
        decision = next(left)
        return {"decision": decision, **({"answer": DRY} if decision == "edit" else {})}

    return answer


@pytest.mark.parametrize("path", ["query", "live"])
@pytest.mark.parametrize("place", ["main-agent", "subagent"])
@pytest.mark.parametrize("flow", list(CLAUDE_FLOWS))
async def test_a_built_in_claude_code_asks_about_is_one_question(
    memory_service: FakeMemoryService, tmp_path: Path, path: str, place: str, flow: str
) -> None:
    from claude_agent_sdk import AgentDefinition

    calls, decisions, run = CLAUDE_FLOWS[flow]
    steps = [{"builtin": "Bash", "args": c} for c in calls]
    deploys = {"agent": "deployer", "prompt": "deploy", "steps": [*steps, {"text": "deployed"}]}
    script = [deploys] if place == "subagent" else steps
    agents = {"deployer": AgentDefinition(description="Deploys.", prompt="Deploy it.")}
    watch, asked = Watch(), Asked()
    async with Harness(config=Settings(), memory=memory_service.client()) as h:
        target = claude_options(tmp_path, [*script, {"text": "{last}"}], agents=agents)
        agent = h.wrap(target, id="deploys", hooks=[watch])
        thread = THREAD if path == "live" else None
        first = await agent.run("deploy", user="ada", thread=thread)
        done = await settled(agent, first, asked, deciding(decisions))
        assert done.status is RunStatus.SUCCESS, done
        await ledger(
            h,
            memory_service,
            watch,
            asked,
            "Bash",
            questions=len(calls),
            ran=len(run),
            harness_tool=False,
        )
    assert [c["args"] for c in built_ins(tmp_path)] == run


# --------------------------------------------------------------------------- ACP permissions
# An ACP agent's own call it asks permission for (``session/request_permission``: the scripted
# agent, speaking ACP as ``kimi acp`` does). ``test_kimi`` runs Kimi Code itself for a Bash
# call approved and rejected; the harness's tools on Kimi are in the core matrix above.

FAKE_ACP: Final = str(Path(__file__).resolve().parents[1] / "support" / "fake_acp_agent.py")
ACP_FLOWS: Final[dict[str, tuple[list[dict[str, str]], list[str], list[dict[str, str]]]]] = {
    "approve": ([DEPLOY], ["approve"], [DEPLOY]),
    "reject": ([DEPLOY], ["reject"], []),
    "two-calls": ([DEPLOY, DRY], ["approve", "approve"], [DEPLOY, DRY]),
    "again-approve-reject": ([DEPLOY, DEPLOY], ["approve", "reject"], [DEPLOY]),
    "again-reject-approve": ([DEPLOY, DEPLOY], ["reject", "approve"], [DEPLOY]),
}


def acp_agent(tmp: Path, script: list[dict[str, Any]]) -> Any:
    env = {
        "FAKE_ACP_SCRIPT": json.dumps(script),
        "FAKE_ACP_RECORD": str(tmp / "record.json"),
        "KIMI_CODE_HOME": str(tmp / "home"),
    }
    return ACPAgent(command=(sys.executable, FAKE_ACP), env=env, cwd=str(tmp), profile=KIMI)


@pytest.mark.parametrize("flow", [*ACP_FLOWS, "subagent-approve", "subagent-reject"])
async def test_an_acp_agents_permission_request_is_one_question(
    memory_service: FakeMemoryService, tmp_path: Path, flow: str
) -> None:
    if flow.startswith("subagent"):
        decision = flow.removeprefix("subagent-")
        calls, decisions, run = [DEPLOY], [decision], [DEPLOY] if decision == "approve" else []
        agent_step = {"description": "deploy", "prompt": "deploy", "subagent_type": "coder"}
        asks = {"builtin": "Bash", "args": DEPLOY}
        script = [{"agent": agent_step, "asks": asks, "result": "deployed"}, {"text": "done"}]
    else:
        calls, decisions, run = ACP_FLOWS[flow]
        steps = [{"builtin": "Bash", "args": c, "ask": True, "output": "ok"} for c in calls]
        script = [*steps, {"text": "done"}]
    watch, asked = Watch(), Asked()
    async with Harness(config=Settings(), memory=memory_service.client()) as h:
        agent = h.wrap(acp_agent(tmp_path, script), id="ops", hooks=[watch])
        first = await agent.run("deploy", user="ada", thread=THREAD)
        done = await settled(agent, first, asked, deciding(decisions))
        assert done.status is RunStatus.SUCCESS, done
        if flow.startswith("subagent"):  # the sub-agent's call shows no update: not recorded
            await h.writes.drain()
            assert len(asked.questions) == 1
            sent = [
                c for c in memory_service.named("feedback") if c.body["target_kind"] == "tool_call"
            ]
            assert [c.body.get("agent_run_id") or c.scope.get("agent_run_id") for c in sent] == [
                asked_on(q) for q in asked.questions
            ]
        else:
            await ledger(
                h,
                memory_service,
                watch,
                asked,
                "Bash",
                questions=len(calls),
                ran=len(run),
                harness_tool=False,
            )
            ran = [c.body["args"] for c in memory_service.named("record_tool")]
            assert ran == run
    answered = json.loads((tmp_path / "record.json").read_text()).get("permissions", [])
    assert [p["answer"] for p in answered].count("approve_once") == len(run)


# --------------------------------------------------------------------------- Claude, background
# The combination no test had: a background subagent of a conversation's live session makes a
# built-in call the catalog asks about (``Bash``, irreversible) AND that Claude Code asks about
# too (``Bash`` is not in its allowed tools). The question waits in the subagent's run and is
# asked in the conversation's next turn: one question; approved, it runs once.


@pytest.mark.parametrize("decision", ["approve", "reject"])
async def test_a_background_subagents_built_in_both_ask_about_is_one_question(
    memory_service: FakeMemoryService, tmp_path: Path, decision: str
) -> None:
    from tests.integration.test_claude_background_session import (
        BACKGROUND,
        options,
        until,
    )

    async def when(run_id: str, status: RunStatus) -> None:
        async with asyncio.timeout(10):
            while (record := await h.runs.get(run_id)) is None or record.status is not status:  # noqa: ASYNC110
                await asyncio.sleep(0.02)

    scripts = [[BACKGROUND, {"text": "started it"}], [{"text": "ok"}]]
    catalog = FakeCatalog({"Bash": Rule(risk="irreversible")})
    watch, asked = Watch(), Asked()
    governed = Harness(config=Settings(), memory=memory_service.client())
    async with with_governance(governed, Governance(catalog)) as h:
        agent = h.wrap(options(tmp_path, scripts), id="bg", hooks=[watch])
        first = await agent.run("research the market", user="ada", thread="t1")
        asked.results.append(first)
        [child] = [r async for r in h.runs.iterate(parent_run_id=first.run_id)]
        (tmp_path / "tasks" / "t1.call").write_text(json.dumps({"builtin": "Bash", "args": DEPLOY}))
        await when(child.run_id, RunStatus.PAUSED)
        turn = await agent.run("how is it going?", user="ada", thread="t1")
        done = await settled(agent, turn, asked, lambda _: {"decision": decision})
        assert done.status is RunStatus.SUCCESS, done
        await until((tmp_path / "tasks" / "t1.called").exists)
        await h.writes.drain()
        assert len(asked.questions) == 1, [q.question for q in asked.questions]
        assert "Claude Code asks" not in asked.questions[0].question  # governance's question
        ran = 1 if decision == "approve" else 0
        assert built_ins(tmp_path) == [{"tool": "Bash", "args": DEPLOY}] * ran
        records = [c for c in memory_service.named("record_tool") if c.body["tool"] == "Bash"]
        assert len(records) == ran
        sent = [c for c in memory_service.named("feedback") if c.body["target_kind"] == "tool_call"]
        # fed back once, on the subagent's run, which asked and records the call
        assert [c.body.get("agent_run_id") or c.scope.get("agent_run_id") for c in sent] == [
            child.run_id
        ]
        assert {c.scope.get("agent_run_id") for c in records} <= {child.run_id}
        (tmp_path / "tasks" / "t1.done").write_text("done")


# --------------------------------------------------------------------------- a hook's question
# A run hook that asks the person a question of its own (``current().ask`` in ``before_tool``)
# before a call Claude Code asks about: the hook's question is not the call's approval — the
# approval is still asked, once.


class Confirming(Hooks):
    """A hook asking its own question about each ``Bash`` call, then leaving it be."""

    async def before_tool(self, call: ToolCall) -> None:
        if call.tool == "Bash":
            runtime = current()
            assert runtime is not None
            await runtime.ask("Which environment?", options=["staging", "production"])


@pytest.mark.parametrize("path", ["query", "live"])
async def test_a_hooks_own_question_is_not_the_approval_of_the_call(
    memory_service: FakeMemoryService, tmp_path: Path, path: str
) -> None:
    touch = {"command": "touch notes.txt"}
    watch, asked = Watch(), Asked()
    async with Harness(config=Settings(), memory=memory_service.client()) as h:
        script = [{"builtin": "Bash", "args": touch}, {"text": "{last}"}]
        agent = h.wrap(claude_options(tmp_path, script), id="hooked", hooks=[watch, Confirming()])
        first = await agent.run("touch it", user="ada", thread=THREAD if path == "live" else None)

        def answer(interrupt: Any) -> dict[str, Any]:
            if interrupt.tool_call is None:
                return {"decision": "answer", "answer": "staging"}
            return {"decision": "approve"}

        done = await settled(agent, first, asked, answer)
        assert done.status is RunStatus.SUCCESS, done
    approvals = [q for q in asked.questions if q.tool_call is not None]
    assert len(approvals) == 1, [q.question for q in asked.questions]
    assert built_ins(tmp_path) == [{"tool": "Bash", "args": touch}]
