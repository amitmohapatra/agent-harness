"""The FEATURES table: every feature the matrix runs, how it is turned on and off, where it does
not apply (and why), where it is not there yet (the audit's gap), and the scenario that checks
it — the behaviour when it is on, and no trace when it is off.

Feature ids are the audit's (``AUDIT_MATRIX.md``, section 1), a letter added where one audit
row is checked by two scenarios (``F30p`` a sub-agent called by the adapter under test, ``F30c``
the adapter under test as the sub-agent). Extension points (features in flight: ``without=``,
hooks, sandbox, HITL v2...) are at the end: each a probe of the API the plan proposes, a strict
xfail until it lands.
"""

from __future__ import annotations

import asyncio
import copy
import json
from collections.abc import Sequence
from typing import Any, Final

import pytest
from agents import Agent as OpenAIAgent
from agents import function_tool
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver

from tests.matrix import way2
from tests.matrix.kit import EMAIL, SECRET, UNKNOWN, Desk
from tests.matrix.model import NA, Feature, Gap, Note
from tests.matrix.models import GroupedChat, GroupedChatModel, GroupedModel, Step
from tests.matrix.world import (
    CONFIRM,
    KEY_TOOL,
    USER,
    NotTimedOut,
    OffButOffered,
    World,
    approve_all,
)
from tests.support.gateway import SkillVersions
from tests.support.memory import MEMORY_TOOLS
from tests.support.planned import Call, PlannedChat, PlannedChatModel
from tests.support.sandbox import PausingSandboxes
from trellis import Harness, ReAct, Runtime, Settings, a2a, skills
from trellis.contracts import (
    InterruptReason,
    RunEventType,
    RunStatus,
    ToolCall,
    ToolOutcome,
    ToolSpec,
)
from trellis.harness.a2a import client as a2a_client
from trellis.harness.governance.catalog import Rule
from trellis.harness.hooks import Ask, Deny, Hooks, ModelCall, Rewrite, Verdict
from trellis.harness.sandbox import sandbox
from trellis.harness.skills import LOAD_SKILL
from trellis.harness.tools.base import Tool

# --------------------------------------------------------------------------- not applicable
NO_MODEL: Final = NA("a function target has no model: your code calls the tools itself")
#: What a LangChain graph is built with natively (ReAct's defaults), when it wants it.
NATIVE_CONTEXT: Final = NA(
    "native: ContextEditingMiddleware and Deep Agents' summarization, which ReAct is built "
    "with (Deep Agents has its summarization by default; add them to create_agent)"
)
OWN_CONTEXT: Final = NA("the framework keeps its own context (its compaction, its trimming)")
ONE_CALL_AT_A_TIME: Final = NA(
    "the scripted Claude CLI calls one tool at a time (the real CLI's concurrency is unverified)"
)
NO_TEXT_OUTSIDE_STREAMS: Final = NA("nothing is streamed: no listener in this mode")


def only_modes(*modes: str, reason: str) -> dict[str, Note]:
    from tests.matrix.model import MODES

    return {m: NA(reason) for m in MODES if m not in modes}


# --------------------------------------------------------------------------- run, events
async def events(w: World) -> None:
    d = Desk()
    o = await w.go(
        [d.lookup(), d.note()], [("lookup", {"topic": "tides"}), ("note", {"text": "high"})]
    )
    o.succeeded()
    assert o.answer == "Done. noted high", o.answer
    called = [c for c in o.called() if c != CONFIRM[0]]
    assert called == ["lookup", "note"], called
    assert [n["tool"] for n in o.custom("tool_notice")] == ["note"]  # a write is announced
    assert d.ran("note") == [{"text": "high"}]
    if w.mode == "stream":
        assert [e.type for e in o.surface][-1] is RunEventType.RUN_FINISHED
    if w.mode == "agui":
        numbers = [n for n, _ in o.surface]
        assert numbers == list(range(len(numbers))), numbers
        assert o.surface[-1][1]["type"] == "RUN_FINISHED" and o.surface[-1][1]["result"] == o.answer
    if w.mode == "a2a":
        assert _a2a_states(o.surface)[-1] == "TASK_STATE_COMPLETED"


async def streamed_text(w: World) -> None:
    d = Desk()
    o = (await w.go([d.lookup()], [("lookup", {"topic": "tides"})])).succeeded()
    deltas = [e.data["delta"] for e in o.events if e.type is RunEventType.TEXT_MESSAGE_CONTENT]
    assert deltas, "no text was streamed"
    assert "".join(deltas).endswith(o.answer), ("".join(deltas), o.answer)


# --------------------------------------------------------------------------- governance
async def tiers(w: World) -> None:
    d = Desk()
    plan: list[Call] = [
        ("lookup", {"topic": "o1"}),
        ("note", {"text": "refund o1"}),
        ("refund", {"order": "o1"}),
    ]
    o = (await w.go([d.lookup(), d.note(), d.refund()], plan)).succeeded()
    asked = [p.tool_call.tool for p in o.pauses if p.tool_call is not None]
    assert asked == ["refund"], asked  # only the irreversible call asks
    assert o.pauses[0].reason is InterruptReason.APPROVAL
    assert [n["tool"] for n in o.custom("tool_notice")] == ["note"]
    assert d.ran("refund") == [{"order": "o1"}] and d.ran("note") == [{"text": "refund o1"}]
    assert o.answer == "Done. refunded o1"


async def approve_when(w: World) -> None:
    d = Desk()
    w.rules({"note": Rule("write", 'text == "big"')})
    plan: list[Call] = [("note", {"text": "small"}), ("note", {"text": "big"})]
    o = (await w.go([d.note()], plan)).succeeded()
    asked = [p.tool_call.args for p in o.pauses if p.tool_call is not None]
    if w.on:
        assert asked == [{"text": "big"}], asked  # the rule, not the tool's risk, asks
        assert 'text == "big"' in o.pauses[0].question
    else:
        assert asked == [], asked  # no catalog: a write runs, announced
    assert d.ran("note") == [{"text": "small"}, {"text": "big"}]


async def ask(w: World) -> None:
    d = Desk()

    def answer(interrupt: Any) -> tuple[str, Any]:
        if interrupt.reason is InterruptReason.CHOICE:
            return "answer", "L"
        return approve_all(interrupt)

    o = (await w.go([d.size()], [("size", {"item": "shirt"})], answer=answer)).succeeded()
    [asked] = o.pauses
    assert asked.reason is InterruptReason.CHOICE and asked.options == ["S", "L"], asked
    assert asked.question == "Which size of shirt?"
    assert d.ran("size") == [{"item": "shirt", "size": "L"}]
    assert o.answer == "Done. shirt in size L", o.answer


async def reject(w: World) -> None:
    d = Desk()

    def answer(interrupt: Any) -> tuple[str, Any]:
        return "reject", "not this order"

    o = (await w.go([d.refund()], [("refund", {"order": "o1"})], answer=answer)).succeeded()
    assert len(o.pauses) == 1 and d.ran("refund") == []  # rejected: never ran
    assert "not this order" in o.text, o.answer  # the model read why


async def native_approval(w: World) -> None:
    d = Desk()
    note = d.note()

    async def target(h: Harness, tools: list[Any], plan: list[Call]) -> tuple[Any, list[Any]]:
        if w.adapter == "openai_agents":

            @function_tool(needs_approval=True)
            async def send(order: str) -> str:
                d.done.append(("send", {"order": order}))
                return f"sent {order}"

            return OpenAIAgent(name="sender", model=GroupedModel(plan), tools=[send]), tools
        native = await h.tools(*tools, framework=w.adapter)  # type: ignore[arg-type]
        model = GroupedChatModel(plan=[], steps=list(plan))
        if w.adapter == "deepagents":
            from deepagents import create_deep_agent

            graph = create_deep_agent(
                model=model, tools=native, interrupt_on={"note": True}, checkpointer=InMemorySaver()
            )
            return graph, []
        graph = create_agent(
            model,
            tools=native,
            middleware=[HumanInTheLoopMiddleware(interrupt_on={"note": True})],
            checkpointer=InMemorySaver(),
        )
        return graph, []

    name = "send" if w.adapter == "openai_agents" else "note"
    args = {"order": "o9"} if name == "send" else {"text": "o9"}
    o = (await w.go([note], [(name, args)], target=target)).succeeded()
    assert [p.tool_call.tool for p in o.pauses if p.tool_call] == [name]
    assert len(d.done) == 1, d.done  # approved, ran once, continued in place


# --------------------------------------------------------------------------- the journal
async def journal(w: World) -> None:
    d = Desk()
    plan: list[Call] = [("note", {"text": "a"}), ("refund", {"order": "o1"})]
    o = (await w.go([d.note(), d.refund()], plan)).succeeded()
    assert d.ran("note") == [{"text": "a"}], d.done  # replayed after the pause, not run again
    assert d.ran("refund") == [{"order": "o1"}]
    assert o.answer == "Done. refunded o1"


async def large_journal(w: World) -> None:
    d = Desk()
    plan: list[Call] = [("export", {"rows": 3}), ("refund", {"order": "o1"})]
    o = (await w.go([d.export(), d.refund()], plan)).succeeded()
    assert d.ran("export") == [{"rows": 3}], len(d.done)  # kept as an artifact, replayed
    assert o.answer == "Done. refunded o1"


# --------------------------------------------------------------------------- reliability
async def read_timeout(w: World) -> None:
    d = Desk()
    o = (await w.go([d.slow()], [("slow", {"sku": "A"})])).succeeded()
    assert o.answer == "Done. slow timed out after 0.05s", o.answer
    assert [r["status"] for r in o.results("slow")] == ["timeout"]


async def unknown_outcome(w: World) -> None:
    d = Desk()
    o = (await w.go([d.transfer()], [("transfer", {"amount": 5})])).succeeded()
    assert o.text.endswith(UNKNOWN), o.answer
    assert d.ran("transfer") == [{"amount": 5}]  # once: a write is never tried again
    assert [r["status"] for r in o.results("transfer")] == ["timeout"]


async def retries(w: World) -> None:
    d = Desk()
    o = (await w.go([d.quote()], [("quote", {"sku": "A-1"})])).succeeded()
    assert o.answer == "Done. A-1 costs 7" and d.quotes == 3, (o.answer, d.quotes)


async def idempotency(w: World) -> None:
    d = Desk()
    o = (await w.go([d.note()], [("note", {"text": "x"})])).succeeded()
    [key] = d.keys
    assert key is not None and key.startswith(f"{o.run_id}:"), key


async def cancel(w: World) -> None:
    d = Desk()

    async def cancelling(run_id: str) -> None:
        await asyncio.wait_for(d.started.wait(), 10)
        await w.cancel(run_id, "a duplicate")

    o = await w.go([d.wait()], [("wait", {"seconds": 30})], during=cancelling)
    assert o.status is RunStatus.CANCELLED, (o.status, o.record.error)
    assert d.ran("wait") == []  # stopped while it waited
    finished = [e for e in o.events if e.type is RunEventType.RUN_FINISHED]
    assert finished[-1].outcome is not None and finished[-1].outcome.value == "cancelled"


async def run_timeout(w: World) -> None:
    d = Desk()
    o = await w.go([d.wait()], [("wait", {"seconds": 30})], timeout=0.4)
    if o.status is not RunStatus.TIMEOUT:
        raise NotTimedOut((o.status, o.record.error))
    assert o.record.error is not None and o.record.error.code == "run_timeout"
    assert o.record.timeout_seconds == 0.4


# --------------------------------------------------------------------------- sub-agents
def _researcher(w: World, d: Desk, process: int = 0) -> Any:
    async def research(input: str, agent: Runtime) -> str:
        """Research a topic and report what was found."""
        return str(await agent.tools.call("lookup", topic=input))

    return w.harness(process).wrap(research, id=f"research{process}", tools=[d.lookup()])


async def subagent_parent(w: World) -> None:
    d = Desk()
    child = _researcher(w, d).as_tool(name="research")
    o = (await w.go([child], [("research", {"message": "tides"})])).succeeded()
    assert o.answer == "Done. facts about tides", o.answer
    kids = [r for r in w.store._runs.values() if r.parent_run_id == o.run_id]
    assert [k.status for k in kids] == [RunStatus.SUCCESS], kids
    assert kids[0].user_id == USER and kids[0].output == "facts about tides"


async def subagent_child(w: World) -> None:
    d = Desk()
    child = await w.agent([d.lookup()], [("lookup", {"topic": "tides"})])
    o = await w.go(
        [child.as_tool(name="research")], [("research", {"message": "tides"})], parent="function"
    )
    o.succeeded()
    assert o.answer == "Done. Done. facts about tides", o.answer
    kids = [r for r in w.store._runs.values() if r.parent_run_id == o.run_id]
    assert [k.agent_id for k in kids] == [child.id] and kids[0].status is RunStatus.SUCCESS


# --------------------------------------------------------------------------- parallel calls
def _parallel_target(w: World):
    async def target(h: Harness, tools: list[Any], plan: list[Step]) -> tuple[Any, list[Any]]:
        if w.adapter == "react":
            return ReAct(system="You work.", model=GroupedChat(plan)), tools
        if w.adapter == "openai_agents":
            return OpenAIAgent(name="worker", model=GroupedModel(plan)), tools
        native = await h.tools(*tools, framework=w.adapter)  # type: ignore[arg-type]
        model = GroupedChatModel(plan=[], steps=list(plan))
        if w.adapter == "deepagents":
            from deepagents import create_deep_agent

            return create_deep_agent(model=model, tools=native), []
        return create_agent(model, tools=native), []

    return target


async def parallel_reads(w: World) -> None:
    d = Desk()
    step: list[Call] = [("pa", {"key": "1"}), ("pa", {"key": "2"})]
    o = await w.go([d.parallel("pa", "read")], [step], target=_parallel_target(w))  # type: ignore[list-item]
    o.succeeded()
    assert sorted(a["key"] for a in d.ran("pa")) == ["1", "2"]
    assert d.overlapped, "the two reads of one turn ran one after the other"


async def parallel_writes(w: World) -> None:
    d = Desk()
    step: list[Call] = [("pw", {"key": "1"}), ("pw", {"key": "2"})]
    o = await w.go([d.parallel("pw", "write")], [step], target=_parallel_target(w))  # type: ignore[list-item]
    o.succeeded()
    assert [a["key"] for a in d.ran("pw")] == ["1", "2"]  # in the order the model made them
    assert not d.overlapped, "two writes of one turn ran at the same time"


# --------------------------------------------------------------------------- context
async def large_result(w: World) -> None:
    d = Desk()
    o = (await w.go([d.report(100_000)], [("report", {"name": "q3"})])).succeeded()
    assert "/large_tool_results/" in o.text, f"the model read all {len(o.text)} characters"
    assert len(o.text) < 25_000


async def context_management(w: World) -> None:
    d = Desk()

    async def target(h: Harness, tools: list[Any], plan: list[Call]) -> tuple[Any, list[Any]]:
        model = PlannedChat(plan)
        return ReAct(system="You read.", model=model, context_window=6000), tools

    plan: list[Call] = [("report", {"name": f"r{n}"}) for n in range(6)]
    o = (await w.go([d.report(2000)], plan, target=target)).succeeded()
    assert "result cleared to keep the context small" in w.said(), "nothing was cleared"
    assert len(d.ran("report")) == 6 and o.answer.startswith("Done. ")


# --------------------------------------------------------------------------- memory
async def push(w: World) -> None:
    d = Desk()
    o = (await w.go([d.lookup()], [("lookup", {"topic": "contact"})])).succeeded()
    told = w.memory_service.context_text
    if w.adapter == "function":
        return  # its code reads agent.context (no model); the event is checked for every run
    if w.on:
        assert told in w.said(), "the context never reached the model"
    else:
        assert told not in w.said()
    assert o.answer == "Done. facts about contact"


async def push_function(w: World) -> None:
    async def target(h: Harness, tools: list[Any], plan: list[Call]) -> tuple[Any, list[Any]]:
        async def reads(input: str, agent: Runtime) -> str:
            for name, args in plan:
                await agent.tools.call(name, **args)
            return str(agent.context)

        return reads, tools

    o = (await w.go([Desk().lookup()], [], target=target)).succeeded()
    assert (w.memory_service.context_text in o.text) is w.on, o.answer


async def pull(w: World) -> None:
    d = Desk()
    if not w.on:
        o = (await w.go([d.lookup()], [("lookup", {"topic": "x"})])).succeeded()
        offered = {n for names in w.models.offered() for n in names}
        if offered & set(MEMORY_TOOLS):
            raise OffButOffered(sorted(offered & set(MEMORY_TOOLS)))
        return
    o = (await w.go([d.lookup()], [("memory_search", {"query": "contact"})])).succeeded()
    assert "the user prefers email" in o.text, o.answer
    [searched] = w.memory_service.named("call_agent_tool")
    assert searched.path["name"] == "memory_search"


async def records(w: World) -> None:
    d = Desk()
    plan: list[Call] = [("lookup", {"topic": "a"}), ("note", {"text": "b"})]
    o = (await w.go([d.lookup(), d.note()], plan, input="note it down")).succeeded()
    if not w.on:
        return  # nothing to record to: World.verify checks no call was made
    tools = [c.body["tool"] for c in w.memory_service.named("record_tool")]
    assert sorted(t for t in tools if t != CONFIRM[0]) == ["lookup", "note"], tools
    said = [m["content"] for c in w.memory_service.named("messages") for m in c.body["messages"]]
    assert "note it down" in said and o.answer in said, said
    verdicts = [
        c.body["verdict"]
        for c in w.memory_service.named("feedback")
        if c.body.get("source") == "system" and c.body.get("target_id") == o.run_id
    ]
    assert verdicts == ["confirm"], verdicts


async def decisions_fed_back(w: World) -> None:
    d = Desk()
    (await w.go([d.refund()], [("refund", {"order": "o1"})])).succeeded()
    decided = [
        c.body["verdict"]
        for c in w.memory_service.named("feedback")
        if c.body.get("target_kind") == "tool_call"
    ]
    if w.on:
        assert "approve" in decided, decided
    else:
        assert decided == []


async def hints(w: World) -> None:
    d = Desk()
    w.memory_service.candidates = ["lookup", CONFIRM[0]]
    tools = [
        d.lookup(),
        d.note(),
        d.refund(),
        d.parallel("t4", "read"),
        d.parallel("t5", "read"),
    ]
    (await w.go(tools, [("lookup", {"topic": "x"})])).succeeded()
    offered = w.models.offered() or w.claude_offered
    assert offered, "the model was never offered anything"
    narrowed = all("note" not in names for names in offered)
    if w.on:
        assert narrowed and all("lookup" in names for names in offered), offered
    else:
        assert not narrowed, offered


# --------------------------------------------------------------------------- gateway
SQL: Final = SkillVersions(
    {"1.1.0": ("Reviews SQL.", "Read rules.md first.", {"rules.md": "No SELECT *."})}, "1.1.0"
)


async def key_tools(w: World) -> None:
    d = Desk()
    if not w.on:
        (await w.go([d.lookup()], [("lookup", {"topic": "x"})])).succeeded()
        return
    o = (await w.go([d.lookup()], [(KEY_TOOL, {"q": "tides"})])).succeeded()
    assert "tides" in o.text, o.answer
    assert [r["status"] for r in o.results(KEY_TOOL)] == ["ok"]


async def virtual_mcps(w: World) -> None:
    d = Desk()
    w.fake_gateway.bundles["billing"] = ["billing-invoice"]
    if not w.on:
        (await w.go([d.lookup()], [("lookup", {"topic": "x"})])).succeeded()
        return
    plan: list[Call] = [("billing-invoice", {"invoice": "7"})]
    o = (await w.go([d.lookup()], plan, mcp=["billing"])).succeeded()
    assert '"invoice": "7"' in o.text, o.answer
    called = {r.url.path for r in w.fake_gateway.requests}
    assert "/mcp/billing" in called and "/mcp" not in called, called  # its bundle only


async def skills_(w: World) -> None:
    d = Desk()
    w.fake_gateway.skills = {"sql": copy.deepcopy(SQL)}
    if not w.on:
        o = (await w.go([d.lookup()], [("lookup", {"topic": "x"})])).succeeded()
        assert not o.custom("skills")
        return
    o = (await w.go([d.lookup(), skills("sql")], [(LOAD_SKILL, {"name": "sql"})])).succeeded()
    assert "Read rules.md first." in o.text, o.answer
    pinned = o.custom("skills")  # each attempt says which versions it pinned
    assert pinned and all(p == {"name": "skills", "versions": {"sql": "1.1.0"}} for p in pinned)
    assert "## Skills" in w.said() or w.adapter == "function"


async def prompts(w: World) -> None:
    d = Desk()
    system = [{"role": "system", "content": "You triage."}]
    w.fake_gateway.prompts = {"triage": [system]}
    if not w.on:
        (await w.go([d.lookup()], [("lookup", {"topic": "x"})])).succeeded()
        return
    if w.adapter == "react":

        async def target(h: Harness, tools: list[Any], plan: list[Call]) -> tuple[Any, list[Any]]:
            w.fake_gateway.chat = PlannedChat(plan)
            w.fake_gateway.serve_models(w.monkeypatch)
            return ReAct(system="Tickets.", model="local/small", prompt="triage"), tools

        o = (await w.go([d.lookup()], [("lookup", {"topic": "x"})], target=target)).succeeded()
        assert {r.headers["x-bf-prompt-id"] for r in w.fake_gateway.completions} == {"p-triage"}
        assert o.answer == "Done. facts about x"
        return
    # every other framework: its model client is given the prompt's headers; the run must pin
    # the version it started with and say so on its agent span (G10)
    if "tracing" not in w.switched:
        pytest.skip("n.a.: a pinned prompt shows on the run's agent span, and tracing is off")
    headers = await w.harness().model_headers(prompt="triage")
    assert headers["x-bf-prompt-id"] == "p-triage"
    o = (await w.go([d.lookup()], [("lookup", {"topic": "x"})])).succeeded()
    pinned = o.custom("prompt") or [
        s for s in w.spans() if (s.attributes or {}).get("trellis.prompt.name") == "triage"
    ]
    assert pinned, "the run neither pinned nor recorded the prompt its model used"


class _Script:
    """A Code Mode meta-tool (what the gateway offers in place of many read-only tools)."""

    async def resolve(self) -> list[Tool]:
        spec = ToolSpec(name="executeToolCode", description="run a script", side_effects="read")

        async def run(args: dict[str, Any]) -> Any:
            return "printed"

        return [Tool(spec, run, feature="code_mode")]


async def code_mode(w: World) -> None:
    d = Desk()
    asked: list[str] = []

    async def logged(run_id: str, since: Any) -> list[Any]:
        from types import SimpleNamespace

        asked.append(run_id)
        nested = SimpleNamespace(
            name="wiki-search", arguments={"q": "x"}, result="found", error=None, latency_ms=4
        )
        return [nested]

    if not w.on:
        (await w.go([d.lookup()], [("lookup", {"topic": "x"})])).succeeded()
        return
    for process in (0, 1):
        gateway = w.harness(process).gateway
        assert gateway is not None
        gateway.code_mode_calls = logged  # type: ignore[method-assign]
    o = (await w.go([_Script()], [("executeToolCode", {"code": "print(1)"})])).succeeded()
    assert asked == [o.run_id], asked
    recorded = [c.body["tool"] for c in w.memory_service.named("record_tool")]
    assert "wiki-search" in recorded, recorded  # the script's nested call, from the log


# --------------------------------------------------------------------------- evaluation
async def judges(w: World) -> None:
    d = Desk()
    o = (await w.go([d.lookup()], [("lookup", {"topic": "x"})])).succeeded()
    seen = [c for c in w.judged if c.run_id == o.run_id]
    if w.on:
        [case] = seen
        assert case.output == o.answer and case.input == "do the task", case
    else:
        assert seen == []


async def grounding(w: World) -> None:
    d = Desk()
    o = (await w.go([d.lookup()], [("lookup", {"topic": "x"})])).succeeded()
    verified = w.memory_service.named("verify")
    if w.on:
        assert [c.body.get("answer") or c.body.get("text") for c in verified] in (
            [o.answer],
            [None],
        ), [c.body for c in verified]
        assert len(verified) == 1
    else:
        assert verified == []


async def tracing(w: World) -> None:
    d = Desk()
    o = (await w.go([d.lookup()], [("lookup", {"topic": "x"})])).succeeded()
    if not w.on:
        return
    names = [s.name for s in w.spans()]
    assert any(n.startswith("invoke_agent") for n in names), names
    assert "execute_tool lookup" in names, names
    del o


async def redaction(w: World) -> None:
    d = Desk()
    plan: list[Call] = [
        ("notify", {"email": EMAIL, "api_key": SECRET}),
        ("lookup", {"topic": "x"}),
    ]
    o = (await w.go([d.notify(), d.lookup()], plan)).succeeded()
    assert d.ran("notify") == [{"email": EMAIL, "api_key": SECRET}]  # the tool had them as is
    leaks = {
        "events": json.dumps([e.model_dump(mode="json") for e in o.events]),
        "surface": json.dumps(o.surface, default=str),
        "spans": json.dumps([dict(s.attributes or {}) for s in w.spans()], default=str),
        "memory tool records": json.dumps(
            [c.body for c in w.memory_service.named("record_tool")], default=str
        ),
        "memory transcript": json.dumps(
            [c.body for c in w.memory_service.named("messages")], default=str
        ),
    }
    leaked = [where for where, text in leaks.items() if EMAIL in text or SECRET in text]
    assert not leaked, f"left the process unredacted in: {leaked}"


# --------------------------------------------------------------------------- tools
async def a2a_tool(w: World) -> None:
    from fastapi import FastAPI

    async def greeter(input: Any, agent: Runtime) -> str:
        """Greets."""
        return f"hello {input}"

    url = "http://a2a.remote/agents/greeter"
    # the remote agent is served in the same tenant (a deployment's own agents call each other)
    memory = w.memory_service.client() if "memory" in w.switched else False
    remote_h = Harness(config=Settings(), runs=False, memory=memory, gateway=False)
    w.others.append(remote_h)  # closed with the world
    app = FastAPI()
    remote_h.wrap(greeter, id="greeter").serve_a2a(app, url)
    w.monkeypatch.setattr(
        a2a_client,
        "_http",
        lambda timeout: __import__("httpx").AsyncClient(
            transport=__import__("httpx").ASGITransport(app=app), base_url="http://a2a.remote"
        ),
    )
    o = (await w.go([a2a(url)], [("greeter", {"message": "world"})])).succeeded()
    assert o.answer == "Done. hello world", o.answer


# --------------------------------------------------------------------------- surfaces, modes
async def agui_surface(w: World) -> None:
    await events(w)
    run_id = w.outcomes[-1].run_id
    handle = w._current
    assert handle is not None and handle.http is not None
    replay = await handle.http.get(f"/agui/runs/{run_id}/events", params={"after": 0})
    from trellis.harness.agui.sse import decode

    assert decode(replay.text) == w.outcomes[-1].surface[1:]


async def a2a_surface(w: World) -> None:
    await events(w)
    states = _a2a_states(w.outcomes[-1].surface)
    assert states[0] == "TASK_STATE_SUBMITTED" and states[-1] == "TASK_STATE_COMPLETED", states


async def scheduled(w: World) -> None:
    d = Desk()
    o = (await w.go([d.lookup()], [("lookup", {"topic": "x"})])).succeeded()
    assert o.record.on_behalf_of == USER and o.record.metadata.get("schedule_id"), o.record


async def durable(w: World) -> None:
    d = Desk()
    o = (await w.go([d.note()], [("note", {"text": "x"})])).succeeded()
    assert o.record.attempt >= 1 and o.record.checkpoint is None, o.record


def _a2a_states(responses: Sequence[Any]) -> list[str]:
    from a2a.types import TaskState

    found = []
    for response in responses:
        which = response.WhichOneof("payload")
        if which == "task":
            found.append(TaskState.Name(response.task.status.state))
        elif which == "status_update":
            found.append(TaskState.Name(response.status_update.status.state))
    return found


# --------------------------------------------------------------------------- the table
FIXED_ONLY: Final = {"function": NO_MODEL, "react": NO_MODEL}
STREAMING: Final = only_modes("stream", "agui", reason=NO_TEXT_OUTSIDE_STREAMS.reason)

FEATURES: Final[list[Feature]] = [
    Feature(
        "F60",
        "events: a run's numbered events, tool calls, announcements, its end",
        "F60, F02, F01",
        "always on (a listener: stream, AG-UI, A2A)",
        events,
        way2=way2.proposed("trellis.harness.blocks", "events"),
        way2_gap=Gap("G6", "Way 2 has no event block"),
    ),
    Feature(
        "F02",
        "streamed text: the answer as text deltas",
        "F02",
        "always on when someone listens",
        streamed_text,
        adapters={"function": NO_MODEL},
        modes=STREAMING,
        way2=way2.proposed("trellis.harness.blocks", "events"),
        way2_gap=Gap("G6", "Way 2 has no event block"),
    ),
    Feature(
        "F32",
        "governance tiers: read runs, write is announced, irreversible asks",
        "F32",
        "always on; per tool through side_effects",
        tiers,
        way2=way2.tiers,
    ),
    Feature(
        "F33",
        "approval rules (approve_when) from the catalog",
        "F33",
        "an administrator's rule in the memory service's catalog (memory on)",
        approve_when,
        needs=frozenset({"memory"}),
        way2=way2.approve_when,
    ),
    Feature(
        "F36",
        "ask: a question pauses the run, the answer resumes it",
        "F36, F04",
        "explicit (current().ask in a tool)",
        ask,
        way2=way2.ask,
    ),
    Feature(
        "F04",
        "a rejected approval: the call never runs and the model reads why",
        "F04, F32",
        "always on",
        reject,
        way2=way2.reject,
    ),
    Feature(
        "F35",
        "framework-native approvals (HITL middleware, interrupt_on, needs_approval)",
        "F35",
        "the framework's own setting",
        native_approval,
        adapters={
            "react": NA("ReAct's approvals are the harness's own (F32)"),
            "function": NA("a function target has no framework approvals"),
            "claude": NA(
                "G5 (can_use_tool, session resume) concerns Claude's built-in tools, which "
                "the scripted CLI cannot call: the live suite's"
            ),
        },
        modes={
            "elsewhere": NA("the graph's in-memory checkpointer is the pausing process's own"),
            "schedule": NA("a scheduled run's pause is the queued run's (covered by worker)"),
        },
        way2=NA("Way 2 recipes use the native approvals themselves (W2R)"),
    ),
    Feature(
        "F23",
        "journal: what ran before a pause is replayed, not run again",
        "F23",
        "automatic",
        journal,
        way2=NA("Way 2 continues from your framework's own checkpoint"),
    ),
    Feature(
        "F12",
        "a journal larger than a checkpoint is kept as an artifact",
        "F12",
        "automatic",
        large_journal,
        way2=NA("Way 2 keeps your framework's checkpoint (the artifacts API by hand)"),
    ),
    Feature(
        "F20",
        "per-tool timeout: a read past its time limit says so",
        "F20",
        "@tool(timeout=)",
        read_timeout,
        way2=way2.read_timeout,
    ),
    Feature(
        "F24",
        "unknown outcome: a write past its time limit is never run again",
        "F24",
        "automatic (a write's timeout)",
        unknown_outcome,
        way2=way2.unknown_outcome,
    ),
    Feature(
        "F21",
        "retries: a read is tried again after an error that may pass",
        "F21",
        "automatic",
        retries,
        way2=way2.retries,
    ),
    Feature(
        "F22",
        "idempotency key: a write gets its run's key",
        "F22",
        "automatic (current().idempotency_key)",
        idempotency,
        way2=way2.proposed("trellis.runs", "idempotency_key"),
        way2_gap=Gap("G6", "no idempotency_key(run_id, call) block outside a run"),
    ),
    Feature(
        "F05",
        "cancel: a running run stops, CANCELLED with its reason",
        "F05",
        "agent.cancel / handle.cancel / A2A tasks/cancel",
        cancel,
        modes={"agui": Gap("G29", "AG-UI has no cancel route")},
        way2=way2.cancel,
        way2_modes=("worker",),
    ),
    Feature(
        "F09",
        "run time limit: past it the run ends TIMEOUT",
        "F09",
        "h.wrap(timeout=) / run(timeout=)",
        run_timeout,
        way2=way2.run_timeout,
        way2_modes=("worker",),
    ),
    Feature(
        "F30p",
        "sub-agent: the adapter under test calls a child agent (as_tool)",
        "F30",
        "opt-in (agent.as_tool())",
        subagent_parent,
        way2=way2.proposed("trellis.harness.blocks", "subagent"),
        way2_gap=Gap("G6", "no sub-agent block (use remote())"),
    ),
    Feature(
        "F30c",
        "sub-agent: the adapter under test is the child of a function parent",
        "F30",
        "opt-in (agent.as_tool())",
        subagent_child,
        way2=way2.proposed("trellis.harness.blocks", "subagent"),
        way2_gap=Gap("G6", "no sub-agent block (use remote())"),
    ),
    Feature(
        "F25r",
        "parallel calls: the reads of one turn run together",
        "F25",
        "automatic",
        parallel_reads,
        adapters={
            "function": NA("a function target runs its own calls (asyncio.gather)"),
            "claude": ONE_CALL_AT_A_TIME,
        },
        way2=NA("your framework runs its calls"),
    ),
    Feature(
        "F25w",
        "parallel calls: the writes of one turn run one at a time, in order",
        "F25",
        "automatic",
        parallel_writes,
        adapters={
            "function": NA("a function target runs its own calls"),
            "claude": ONE_CALL_AT_A_TIME,
            "langgraph": NA(
                "native: the graph runs a turn's calls its own way (N3); HarnessTools orders "
                "the writes (ReAct has it)"
            ),
            "deepagents": NA(
                "native: the graph runs a turn's calls its own way (N3); HarnessTools orders "
                "the writes (ReAct has it)"
            ),
            "openai_agents": NA("native: the framework runs a turn's calls its own way (N3)"),
        },
        way2=NA("your framework runs its calls"),
    ),
    Feature(
        "F26",
        "large results: saved as a file, a head-and-tail preview read in pages (read_file)",
        "F26",
        "automatic: Deep Agents' FilesystemMiddleware (ReAct's, Deep Agents' by default)",
        large_result,
        adapters={
            "function": NA("a function target gets the whole result: it has no context to fill"),
            "langgraph": NA(
                "native: add FilesystemMiddleware(tools=['read_file']) to create_agent (ReAct "
                "and Deep Agents have it by default)"
            ),
            "openai_agents": Gap("G8", "nothing cuts a large result before the model reads it"),
            "claude": Gap("G8", "nothing cuts a large result before the model reads it"),
        },
        way2=way2.proposed("trellis.harness.blocks", "bounded"),
        way2_gap=Gap("G8", "no cut-and-keep block"),
    ),
    Feature(
        "F65",
        "context management: older results cleared past the window's share",
        "F65",
        "automatic in ReAct (native ContextEditingMiddleware, sized from context_window=)",
        context_management,
        adapters={
            "function": NA("a function target has no model context"),
            "langgraph": NATIVE_CONTEXT,
            "deepagents": NATIVE_CONTEXT,
            "openai_agents": OWN_CONTEXT,
            "claude": OWN_CONTEXT,
        },
        way2=NA("native middleware: a graph is built with it"),
    ),
    Feature(
        "F41",
        "memory push: the context reaches the model",
        "F41",
        "the memory service (MEMORY_URL / Harness(memory=))",
        push,
        needs=frozenset({"memory", "memory_push"}),
        adapters={"function": NA("checked by F41f: a function target reads agent.context")},
        way2=way2.push,
    ),
    Feature(
        "F41f",
        "memory push: a function target reads agent.context",
        "F41",
        "the memory service",
        push_function,
        needs=frozenset({"memory", "memory_push"}),
        adapters={
            a: NA("checked by F41 (what the model is sent)")
            for a in ("react", "langgraph", "deepagents", "openai_agents", "claude")
        },
        way2=NA("checked by F41"),
    ),
    Feature(
        "F42",
        "memory pull: the memory tools are the agent's",
        "F42",
        "the memory service",
        pull,
        needs=frozenset({"memory", "memory_pull"}),
        adapters={"function": NA("a function target calls the memory tools like any other (F43)")},
        way2=way2.proposed("trellis.harness.blocks", "agent_tools"),
        way2_gap=Gap("G6", "agent_tools() is raw: no conversion block"),
    ),
    Feature(
        "F43",
        "memory records: transcript, tool calls, outcome",
        "F43",
        "the memory service",
        records,
        needs=frozenset({"memory", "records"}),
        way2=way2.records,
    ),
    Feature(
        "F40",
        "a decision on a call is fed back to memory",
        "F40",
        "the memory service",
        decisions_fed_back,
        needs=frozenset({"memory", "records"}),
        way2=way2.decided,
    ),
    Feature(
        "F28",
        "tool hints narrow what the model is offered",
        "F28",
        "automatic from 5 tools, with memory",
        hints,
        needs=frozenset({"memory", "memory_push", "hints"}),
        adapters={
            "function": NO_MODEL,
            "langgraph": Gap("G12", "tools are bound when the graph is built"),
            "deepagents": Gap("G12", "tools are bound when the graph is built"),
        },
        way2=way2.proposed("trellis.harness.blocks", "tool_hints"),
        way2_gap=Gap("G6", "memory.tool_hints is raw: no narrowing block"),
    ),
    Feature(
        "F17",
        "the key's MCP tools (through the gateway)",
        "F17",
        "the gateway (BIFROST_URL / Harness(gateway=))",
        key_tools,
        needs=frozenset({"gateway", "mcp"}),
        way2=way2.key_tools,
    ),
    Feature(
        "F18",
        "Virtual MCPs: only the named bundles' tools",
        "F18",
        "mcp=[...] (wrap, or h.tools for a graph)",
        virtual_mcps,
        needs=frozenset({"gateway", "mcp"}),
        way2=way2.virtual_mcps,
    ),
    Feature(
        "F19",
        "Code Mode: a script's nested calls are recorded from the gateway's log",
        "F19",
        "automatic with the gateway (and memory to record)",
        code_mode,
        needs=frozenset({"gateway", "memory", "records", "mcp", "code_mode"}),
        way2=NA("Code Mode is the gateway's: Way 2 calls its meta-tools through bifrost-sdk"),
    ),
    Feature(
        "F50",
        "skills: disclosed, pinned, loaded through the bridge",
        "F50",
        "skills=[...] / skills(...) with the gateway",
        skills_,
        needs=frozenset({"gateway", "skills"}),
        way2=way2.proposed("trellis.harness.skills", "disclose"),
        way2_gap=Gap("G6", "no skills block (disclose(refs))"),
    ),
    Feature(
        "F49",
        "stored prompts: pinned per run and recorded",
        "F49",
        "ReAct(prompt=) / model_headers(prompt=) with the gateway",
        prompts,
        needs=frozenset({"gateway"}),
        adapters={
            "function": NO_MODEL,
            "langgraph": Gap("G10", "model_headers resolves the prompt once, unpinned, unrecorded"),
            "deepagents": Gap(
                "G10", "model_headers resolves the prompt once, unpinned, unrecorded"
            ),
            "openai_agents": Gap(
                "G10", "model_headers resolves the prompt once, unpinned, unrecorded"
            ),
            "claude": Gap("G10", "model_headers resolves the prompt once, unpinned, unrecorded"),
        },
        way2=way2.proposed("trellis.harness.blocks", "prompt_pin"),
        way2_gap=Gap("G6", "no prompt_pin(ref) block"),
    ),
    Feature(
        "F55",
        "online judges score sampled runs",
        "F55",
        "Harness(judges=[...]) (+ TRELLIS_JUDGE_SAMPLE)",
        judges,
        needs=frozenset({"judges"}),
        way2=way2.judges,
    ),
    Feature(
        "F54",
        "grounding: the answer checked against its context",
        "F54",
        "TRELLIS_GROUNDING_SAMPLE with memory",
        grounding,
        needs=frozenset({"grounding", "memory", "memory_push"}),
        way2=way2.grounding,
    ),
    Feature(
        "F57",
        "tracing: run and tool spans",
        "F57",
        "OTEL_EXPORTER_OTLP_ENDPOINT (an OTel provider)",
        tracing,
        needs=frozenset({"tracing"}),
        way2=way2.proposed("trellis.harness.tracing", "agent_span"),
        way2_gap=Gap("G6", "no tracing block (agent_span/tool_span)"),
    ),
    Feature(
        "F61",
        "redaction: nothing secret leaves the process",
        "F61",
        "always on",
        redaction,
        way2=way2.redaction,
    ),
    Feature(
        "F16",
        "a2a(): a remote A2A agent as a tool",
        "F16",
        "tools=[a2a(url)]",
        a2a_tool,
        way2=way2.remote_agent,
    ),
    Feature(
        "F62",
        "AG-UI: numbered events, the answer, a reconnect replays",
        "F62",
        "agent.serve_chat(app)",
        agui_surface,
        modes=only_modes("agui", reason="the AG-UI surface is its own mode"),
        way2=NA("Way 1 only by design (README: surfaces are the harness's)"),
    ),
    Feature(
        "F63",
        "A2A: the task is the run, submitted to completed",
        "F63",
        "agent.serve_a2a(app, url)",
        a2a_surface,
        modes=only_modes("a2a", reason="the A2A surface is its own mode"),
        way2=NA("Way 1 only by design (README: surfaces are the harness's)"),
    ),
    Feature(
        "F06",
        "schedules: a schedule fires a run for its person",
        "F06",
        "agent.schedule(cron, input, on_behalf_of=)",
        scheduled,
        modes=only_modes("schedule", reason="schedules are their own mode"),
        way2=way2.scheduled,
        way2_modes=("schedule",),
    ),
    Feature(
        "F03",
        "durable workers: a queued run is claimed and finished by a worker",
        "F03",
        "agent.start + a worker (RUNS_URL for agent-runs)",
        durable,
        modes=only_modes("worker", "elsewhere", reason="queued runs are the worker modes"),
        way2=way2.durable,
        way2_modes=("worker", "elsewhere"),
    ),
]


# --------------------------------------------------------------------------- hooks
class _Guard(Hooks):
    """The hooks the scenarios give: deny refunds, rewrite notes, ask before lookups, replace
    a quote's outcome; and every run and model call, noted."""

    def __init__(
        self, deny: bool = False, rewrite: bool = False, ask: bool = False, replace: bool = False
    ) -> None:
        self.deny, self.rewrite, self.ask, self.replace = deny, rewrite, ask, replace
        self.runs: list[tuple[str, str, str | None]] = []
        self.models: list[str] = []
        self.errors: list[str] = []

    async def on_run_start(self, run: Runtime) -> None:
        self.runs.append(("start", run.run_id, None))

    async def on_run_end(self, run: Runtime, result: Any) -> None:
        self.runs.append(("end", run.run_id, result.status.value))

    async def before_model(self, call: ModelCall) -> None:
        self.models.append(f"before {call.framework}")

    async def after_model(self, call: ModelCall, reply: Any) -> None:
        self.models.append(f"after {call.framework}")

    async def before_tool(self, call: ToolCall) -> Verdict:
        if self.deny and call.tool == "refund":
            return Deny("refunds are closed today")
        if self.rewrite and call.tool == "note":
            return Rewrite({"text": "[rewritten]"})
        if self.ask and call.tool == "lookup":
            return Ask("Look this up?")
        return None

    async def after_tool(self, call: ToolCall, outcome: ToolOutcome) -> ToolOutcome:
        if self.replace and call.tool == "lookup":
            return outcome.model_copy(update={"output": "replaced by a hook"})
        return outcome

    async def on_error(self, stage: str, error: Exception) -> None:
        self.errors.append(stage)


async def hook_deny(w: World) -> None:
    d, guard = Desk(), _Guard(deny=True)
    o = (await w.go([d.refund()], [("refund", {"order": "o1"})], hooks=[guard])).succeeded()
    assert d.ran("refund") == [] and not o.pauses, (d.done, o.pauses)  # never ran, never asked
    assert "refunds are closed today" in o.text, o.answer


async def hook_rewrite(w: World) -> None:
    d, guard = Desk(), _Guard(rewrite=True)
    o = (await w.go([d.note()], [("note", {"text": "secret"})], hooks=[guard])).succeeded()
    assert d.ran("note") == [{"text": "[rewritten]"}], d.done
    assert o.answer == "Done. noted [rewritten]", o.answer


async def hook_ask(w: World) -> None:
    d, guard = Desk(), _Guard(ask=True)
    o = (await w.go([d.lookup()], [("lookup", {"topic": "x"})], hooks=[guard])).succeeded()
    assert [p.question for p in o.pauses] == ["Approve lookup? Look this up?"] or (
        len(o.pauses) == 1 and "Look this up?" in o.pauses[0].question
    ), o.pauses
    assert d.ran("lookup") == [{"topic": "x"}]  # once, after the approval


async def hook_after(w: World) -> None:
    d, guard = Desk(), _Guard(replace=True)
    o = (await w.go([d.lookup()], [("lookup", {"topic": "x"})], hooks=[guard])).succeeded()
    assert o.answer == "Done. replaced by a hook", o.answer


async def hook_runs(w: World) -> None:
    d, guard = Desk(), _Guard()
    o = (await w.go([d.lookup()], [("lookup", {"topic": "x"})], hooks=[guard])).succeeded()
    attempts = sorted({e.attempt for e in o.events})
    starts = [r for r in guard.runs if r[:2] == ("start", o.run_id)]
    ends = [r for r in guard.runs if r[:2] == ("end", o.run_id)]
    assert len(starts) == len(ends) == len(attempts), (guard.runs, attempts)
    assert ends[-1][2] == "SUCCESS"


async def hook_models(w: World) -> None:
    d, guard = Desk(), _Guard()

    async def target(h: Harness, tools: list[Any], plan: list[Call]) -> tuple[Any, list[Any]]:
        from trellis.harness.middleware import ModelHooks

        native = await h.tools(*tools, framework=w.adapter)  # type: ignore[arg-type]
        model = PlannedChatModel(plan=plan)
        if w.adapter == "deepagents":
            from deepagents import create_deep_agent

            return create_deep_agent(model=model, tools=native, middleware=[ModelHooks()]), []
        return create_agent(model, tools=native, middleware=[ModelHooks()]), []

    built = target if w.adapter in ("langgraph", "deepagents") else None
    plan: list[Call] = [("lookup", {"topic": "x"})]
    (await w.go([d.lookup()], plan, hooks=[guard], target=built)).succeeded()
    assert guard.models, "no model call was hooked"
    assert len([m for m in guard.models if m.startswith("before")]) >= 2, guard.models
    assert len([m for m in guard.models if m.startswith("after")]) >= 2, guard.models


# --------------------------------------------------------------------------- a run's own options
async def run_options_timeout(w: World) -> None:
    d = Desk()
    o = await w.go([d.wait()], [("wait", {"seconds": 30})], run={"timeout": 0.4})
    if o.status is not RunStatus.TIMEOUT:
        raise NotTimedOut((o.status, o.record.error))
    assert o.record.timeout_seconds == 0.4


async def run_options_queue(w: World) -> None:
    d = Desk()
    run = {"priority": 7, "concurrency_key": "nightly"}
    o = (await w.go([d.lookup()], [("lookup", {"topic": "x"})], run=run)).succeeded()
    assert (o.record.priority, o.record.concurrency_key) == (7, "nightly"), o.record


async def run_options_without(w: World) -> None:
    d = Desk()
    o = (
        await w.go([d.lookup()], [("lookup", {"topic": "x"})], run={"without": ["memory"]})
    ).succeeded()
    assert not [e for e in o.events if e.type is RunEventType.CONTEXT_LOADED]
    mine = [c for c in w.memory_service.calls if c.scope.get("agent_run_id") == o.run_id]
    assert [c.name for c in mine if c.name in ("messages", "record_tool", "context")] == [], [
        c.name for c in mine
    ]


#: Each framework's own limit on a run's steps, set below what a plan of one call needs (and
#: what its error says), and set far above it.
TIGHT: Final[dict[str, tuple[dict[str, Any], str]]] = {
    "react": ({"recursion_limit": 3}, "Recursion limit of 3 reached"),
    "langgraph": ({"recursion_limit": 3}, "Recursion limit of 3 reached"),
    "deepagents": ({"recursion_limit": 3}, "Recursion limit of 3 reached"),
    "openai_agents": ({"max_turns": 1}, "Max turns (1) exceeded"),
    "claude": ({"max_turns": 1}, "Reached maximum number of turns"),
}
ROOMY: Final[dict[str, dict[str, Any]]] = {
    "react": {"recursion_limit": 50},
    "langgraph": {"recursion_limit": 50},
    "deepagents": {"recursion_limit": 50},
    "openai_agents": {"max_turns": 20},
    "claude": {"max_turns": 20},
}


async def framework_options(w: World) -> None:
    """The framework's own limit reaches its run call: a run's own over its agent's (kept with
    its record — a scheduled run's with its schedule — for a resume and a worker), or — a
    surface — the agent's."""
    tight, said = TIGHT[w.adapter]
    d, plan = Desk(), [("lookup", {"topic": "x"})]
    if w.mode in ("run", "stream", "worker", "elsewhere", "schedule"):
        run = {"framework_options": tight}
        o = await w.go([d.lookup()], plan, run=run, framework_options=ROOMY[w.adapter])
        assert o.record.metadata.get("framework_options") == tight, o.record.metadata
    else:
        o = await w.go([d.lookup()], plan, framework_options=tight)
    assert o.status is RunStatus.ERROR and o.record.error is not None, o.status
    assert said in o.record.error.message, o.record.error.message


# --------------------------------------------------------------------------- sandboxes
async def sandboxed(w: World) -> None:
    provider = PausingSandboxes()
    plan: list[Call] = [
        ("sandbox_write", {"path": "a.txt", "content": "hello"}),
        ("refund", {"order": "o1"}),
        ("sandbox_exec", {"command": "cat a.txt"}),
    ]
    d = Desk()
    o = (await w.go([sandbox(provider), d.refund()], plan)).succeeded()
    assert "hello" in o.text, o.answer
    assert len(provider.made()) == 1, provider.calls  # one sandbox, kept across the pause
    assert provider.calls[-1][0] == "delete" and not provider.boxes, provider.calls
    assert [c for c, _ in provider.calls].count("attach") >= 1, provider.calls


# --------------------------------------------------------------------------- the selection
async def selection(w: World) -> None:
    """One run that does a bit of everything (a read retried, a write announced, an approval)
    under the cell's selection: what is on works together, what is off leaves no trace
    (``World.verify``, after every scenario)."""
    d = Desk()
    plan: list[Call] = [
        ("quote", {"sku": "A-1"}),
        ("note", {"text": "A-1 costs 7"}),
        ("refund", {"order": "o1"}),
    ]
    o = (await w.go([d.quote(), d.note(), d.refund()], plan)).succeeded()
    assert o.answer == "Done. refunded o1", o.answer
    assert d.quotes == 3 and d.ran("refund") == [{"order": "o1"}]
    assert [p.tool_call.tool for p in o.pauses if p.tool_call is not None] == ["refund"]


SELECTION: Final = Feature(
    "SEL",
    "the selection: everything on works together, everything off leaves no trace",
    "Definition of done (selection)",
    "the switches (dimensions.SWITCHES)",
    selection,
    way2=NA("Way 2 selects by import: each block is used or not by your own code"),
)


async def versioned(w: World) -> None:
    d = Desk()
    o = (await w.go([d.lookup()], [("lookup", {"topic": "x"})])).succeeded()
    from tests.matrix.world import AGENT_TIMEOUT, VERSION

    assert o.record.agent_version == (VERSION if w.on else None), o.record.agent_version
    limit = AGENT_TIMEOUT if "agent_timeout" in w.switched else None
    assert o.record.timeout_seconds == limit, o.record.timeout_seconds


FEATURES.append(
    Feature(
        "F10",
        "the agent's version (and time limit) recorded with every run it starts",
        "F10, F09",
        "h.wrap(version=, timeout=) / TRELLIS_AGENT_VERSION",
        versioned,
        needs=frozenset({"version"}),
        way2=NA("RunStart.agent_version: your code sets it"),
    )
)

PER_RUN: Final = only_modes(
    "run",
    "stream",
    "worker",
    "elsewhere",
    "schedule",
    reason="a surface takes no per-run option: the agent's own (h.wrap) applies",
)
QUEUED: Final = only_modes(
    "worker",
    "elsewhere",
    "schedule",
    reason="a queue order is a queued run's: start and schedule take it",
)
NO_MODEL_HOOKS: Final = NA(
    "no model call the harness can hook (a function makes none; the CLI's are its own)"
)
TOOL_HOOK_ROWS: Final = (
    (
        "F71d",
        "hooks: before_tool denies a call (never run, the model reads why)",
        hook_deny,
        way2.hook_deny,
    ),
    ("F71w", "hooks: before_tool rewrites a call's arguments", hook_rewrite, way2.hook_rewrite),
    ("F71a", "hooks: before_tool asks a person first", hook_ask, way2.hook_ask),
    ("F71t", "hooks: after_tool replaces what the model reads", hook_after, way2.hook_after),
)
FEATURES.extend(
    [
        *(
            Feature(
                fid,
                title,
                "F71 (W6)",
                "h.wrap(hooks=[...]) / Harness(hooks=); Way 2 governed(hooks=)",
                scenario,
                way2=block,
            )
            for fid, title, scenario, block in TOOL_HOOK_ROWS
        ),
        Feature(
            "F71r",
            "hooks: on_run_start / on_run_end around every attempt",
            "F71 (W6)",
            "h.wrap(hooks=[...]) / Harness(hooks=)",
            hook_runs,
            way2=NA("no run of the harness's in Way 2: your code has its own"),
        ),
        Feature(
            "F71m",
            "hooks: before_model / after_model around every model call",
            "F71 (W6)",
            "hooks=; LangChain: middleware=[ModelHooks()] (ReAct's own); OpenAI: automatic",
            hook_models,
            adapters={"function": NO_MODEL_HOOKS, "claude": NO_MODEL_HOOKS},
            way2=way2.model_hooks,
        ),
        Feature(
            "F09r",
            "a run's own time limit: run/stream/start/schedule(timeout=)",
            "F09",
            "agent.run(..., timeout=)",
            run_options_timeout,
            modes=PER_RUN,
            way2=NA("RunStart.timeout_seconds: your code sets it (F09's Way 2 row)"),
        ),
        Feature(
            "F09q",
            "a queued run's own queue order: start/schedule(priority=, concurrency_key=)",
            "F09 (ADR 0006, 0007)",
            "agent.start/schedule(..., priority=, concurrency_key=)",
            run_options_queue,
            modes=QUEUED,
            way2=NA("RunStart/ScheduleSpec.priority: your code sets it (F09's Way 2 row)"),
        ),
        Feature(
            "F75r",
            "a run's own without=: memory off for one run, kept across its attempts",
            "F75 (G2)",
            "agent.run(..., without={...})",
            run_options_without,
            needs=frozenset({"memory"}),
            modes=PER_RUN,
            way2=NA("Way 2 selects by import"),
        ),
        Feature(
            "F70",
            "the framework's own run options: a run's over its agent's, kept across attempts",
            "F70 (G13)",
            "h.wrap(framework_options=) / agent.run(..., framework_options=)",
            framework_options,
            adapters={"function": NA("no framework run call: framework_options= is refused")},
            way2=NA("Way 2 calls the framework itself, with its own options"),
        ),
        Feature(
            "F73",
            "sandbox: write, exec across a pause, one sandbox per run, deleted at its end",
            "F73 (W7)",
            "tools=[sandbox(provider)] (h.tools for a graph); SANDBOX=docker",
            sandboxed,
            way2=NA(
                "sandbox tools are harness tools: they run in a harness run (Way 1, with blocks)"
            ),
        ),
    ]
)
