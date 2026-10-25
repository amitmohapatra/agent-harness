"""What the memory service offers, used on every run, the same through every framework:

* a run's own success is its outcome only; the sampled share is verified, and the judge's
  verdict is what teaches;
* a sub-agent's run is derived from its parent's memory scope (it names the parent's run and
  its delegation tree's agent group);
* the framework's own tools count among the agent's tools for its skills and hints;
* a person's verdict cites the memories the run's answer rested on;
* a run may narrow its context to given documents;
* a context that cannot reach the agent is said, once per agent, with the fix.
"""

from __future__ import annotations

import logging
from typing import Any, TypedDict

import pytest
from agents import Agent as OpenAIAgent
from agents import function_tool
from langchain.agents import create_agent
from langchain_core.tools import tool as lc_tool
from langgraph.graph import END, START, StateGraph

from tests.support.arm import arm
from tests.support.chat_model import ScriptedChatModel
from tests.support.memory import FakeMemoryService
from tests.support.models import ScriptedChat
from tests.support.openai_model import ScriptedModel
from trellis import Harness, ReAct, Runtime, Settings, current, tool
from trellis.contracts import ConfigurationError, RunEventType, RunStatus
from trellis.harness.agent import UNREACHED_FIX
from trellis.harness.clients.memory import CITED_MAX, PUSHED_BUNDLE, PUSHED_MEMORIES
from trellis.memory.errors import DependencyUnavailableError


@tool(side_effects="read")
def stock(sku: str) -> int:
    """Units in stock."""
    return 7


def harness_with(service: FakeMemoryService, **settings: Any) -> Harness:
    return Harness(config=Settings(**settings), memory=service.client())


def judged(service: FakeMemoryService) -> list[dict[str, Any]]:
    return [f for f in service.stored_feedback.values() if f["source"] == "judge"]


def warned(events: list[Any], code: str) -> list[dict[str, Any]]:
    return [
        e.data
        for e in events
        if e.type is RunEventType.CUSTOM and e.data["name"] == "warning" and e.data["code"] == code
    ]


# --------------------------------------------------------------------------- 1. learning signal
@pytest.mark.parametrize("framework", ["function", "react", "openai_agents"])
async def test_a_sampled_run_is_verified_whatever_the_framework(
    memory_service: FakeMemoryService, framework: str, sampled: Any
) -> None:
    async def fn(input: str, agent: Runtime) -> str:
        return "you prefer email"

    targets: dict[str, Any] = {
        "function": fn,
        "react": ReAct(system_prompt="s", model=ScriptedChat(["you prefer email"])),
        "openai_agents": OpenAIAgent(name="o", model=ScriptedModel(["you prefer email"])),
    }
    sampled(1.0)
    async with harness_with(memory_service) as h:
        result = await h.wrap(targets[framework], id="v").run("contact?", user="u")
        await h.writes.drain()
    [verified] = memory_service.named("verify")
    assert verified.body["answer"] == "you prefer email"
    assert verified.scope["agent_run_id"] == result.run_id
    assert [f["target_id"] for f in judged(memory_service)] == [result.run_id]


@pytest.mark.parametrize(("sample", "verified"), [(0.0, False), (1.0, True)])
async def test_a_runs_own_success_is_its_outcome_and_only_a_sampled_run_is_verified(
    memory_service: FakeMemoryService, sample: float, verified: bool, sampled: Any
) -> None:
    """The run's ``system`` confirm cites nothing (the service keeps it as the outcome only);
    the judge's verdict, on the sampled share, is what teaches."""

    async def fn(input: str, agent: Runtime) -> str:
        return "an answer"

    sampled(sample)
    async with harness_with(memory_service) as h:
        result = await h.wrap(fn, id="g").run("q", user="u")
        await h.writes.drain()
    own = [f for f in memory_service.stored_feedback.values() if f["source"] == "system"]
    assert [(f["target_id"], f["verdict"]) for f in own] == [(result.run_id, "confirm")]
    assert not own[0].get("evidence_refs")
    assert bool(memory_service.named("verify")) is verified
    assert bool(judged(memory_service)) is verified


# --------------------------------------------------------------------------- 2. sub-agents
async def test_a_sub_agents_run_reads_its_parents_notes_and_binds_its_group(
    memory_service: FakeMemoryService,
) -> None:
    """The child's scope is derived from its parent's (``MemoryContext.agent``): it names the
    parent's run and the tree's agent group, the run at its top; a grandchild names its own
    parent and the same group. The top run names neither."""

    async def grandchild(input: str, agent: Runtime) -> str:
        await agent.tools.call("memory_remember", content="found it", scope="group")
        return "deep"

    async def child(input: str, agent: Runtime) -> Any:
        await agent.tools.call("memory_search", query="the plan")
        return await agent.tools.call("deeper", message=input)

    async def parent(input: str, agent: Runtime) -> Any:
        await agent.tools.call("memory_remember", content="the plan", scope="run")
        return await agent.tools.call("helper", message=input)

    async with harness_with(memory_service) as h:
        deeper = h.wrap(grandchild, id="deeper")
        helper = arm(h.wrap(child, id="helper"), [deeper])
        top = arm(h.wrap(parent, id="lead"), [helper])
        result = await top.run("plan it", user="ada", thread="t1")
        await h.writes.drain()
    assert result.status is RunStatus.SUCCESS, result.error
    scopes = {c.scope["agent_id"]: c.scope for c in memory_service.named("call_agent_tool")}
    lead, helper_scope, deeper_scope = scopes["lead"], scopes["helper"], scopes["deeper"]
    assert "parent_agent_run_id" not in lead and "agent_group_id" not in lead
    assert helper_scope["parent_agent_run_id"] == lead["agent_run_id"] == result.run_id
    assert helper_scope["agent_group_id"] == result.run_id
    assert deeper_scope["parent_agent_run_id"] == helper_scope["agent_run_id"]
    assert deeper_scope["agent_group_id"] == result.run_id
    for scope in (helper_scope, deeper_scope):  # the same person and conversation
        assert (scope["user_id"], scope["thread_id"], scope["tenant_id"]) == ("ada", "t1", "acme")
    # every call of the child's run is in that scope: its context and its records too
    contexts = [c.scope for c in memory_service.named("context") if c.scope["agent_id"] == "helper"]
    assert contexts and all(s["agent_group_id"] == result.run_id for s in contexts)


async def test_a_sub_agents_run_keeps_its_scope_when_it_continues(
    memory_service: FakeMemoryService,
) -> None:
    """A child paused on a person continues in the scope it started in: its parent and its
    group come from its record."""

    @tool(side_effects="irreversible")
    def pay(order: str) -> str:
        """Pay an order."""
        return f"paid {order}"

    async def child(input: str, agent: Runtime) -> Any:
        return await agent.tools.call("pay", order=input)

    async def parent(input: str, agent: Runtime) -> Any:
        return await agent.tools.call("payer", message=input)

    async with harness_with(memory_service) as h:
        payer = arm(h.wrap(child, id="payer"), [pay])
        boss = arm(h.wrap(parent, id="boss"), [payer])
        paused = await boss.run("o-1", user="ada")
        assert paused.interrupt is not None
        done = await boss.resume(paused.run_id, "approve", reviewer="cfo")
        await h.writes.drain()
    assert done.answer == "paid o-1"
    asked = [c.scope for c in memory_service.named("context") if c.scope["agent_id"] == "payer"]
    assert len(asked) == 2  # before the pause and after the answer
    assert all(s["parent_agent_run_id"] == done.run_id for s in asked)
    assert all(s["agent_group_id"] == done.run_id for s in asked)
    [decision] = [
        f for f in memory_service.named("feedback") if f.body["target_kind"] == "tool_call"
    ]
    assert decision.body["agent_id"] == "payer"


# --------------------------------------------------------------------------- 3. native tools
async def test_an_openai_agents_own_tools_count_for_its_skills_and_hints(
    memory_service: FakeMemoryService,
) -> None:
    def make(i: int) -> Any:
        def lookup(key: str) -> str:
            return key

        lookup.__name__ = f"own{i}"
        return function_tool(lookup)

    model = ScriptedModel(["done"])
    target = OpenAIAgent(name="o", model=model, tools=[make(i) for i in range(4)])
    async with harness_with(memory_service) as h:
        await arm(h.wrap(target, id="native"), [stock]).run("how many?", user="u")
    [asked] = memory_service.named("context")
    # the harness's tool and the agent's own four: five, the hints' threshold
    assert asked.body["tools"]["available"] == ["stock", "own0", "own1", "own2", "own3"]
    told = model.inputs[0][0]
    assert told["role"] == "system" and "## Tools" in told["content"]


async def test_a_graphs_own_tools_count_and_never_narrow_it(
    memory_service: FakeMemoryService,
) -> None:
    """A ``create_agent`` graph built with its own tools: they count for the hints; the
    candidates the service names among them narrow none of its tools (narrowing is for the
    harness's tools alone)."""

    @lc_tool
    def lookup_price(sku: str) -> str:
        """Price of a SKU."""
        return "9"

    @lc_tool
    def lookup_size(sku: str) -> str:
        """Size of a SKU."""
        return "M"

    memory_service.candidates = ["lookup_price"]
    graph = create_agent(ScriptedChatModel(turns=["done"]), tools=[lookup_price, lookup_size])
    async with harness_with(memory_service) as h:
        agent = h.wrap(graph, id="graph")
        result = await agent.run("price?", user="u")
    assert result.status is RunStatus.SUCCESS
    [asked] = memory_service.named("context")
    assert asked.body["tools"]["available"] == ["lookup_price", "lookup_size"]


async def test_react_counts_its_toolbox_not_its_automatic_tools(
    memory_service: FakeMemoryService,
) -> None:
    """``ReAct`` takes the run's tools per call: its own nodes hold only what its middleware
    offers on a long task (the files), which are not the agent's tools."""
    model = ScriptedChat(["seven"])
    async with harness_with(memory_service) as h:
        await arm(h.wrap(ReAct(system_prompt="s", model=model), id="r"), [stock]).run("q", user="u")
    [asked] = memory_service.named("context")
    assert asked.body["tools"]["available"] == ["stock"]


async def test_claudes_tools_are_its_built_ins_and_the_mcp_tools_it_names_lists_and_calls() -> None:
    from claude_agent_sdk import ClaudeAgentOptions, create_sdk_mcp_server
    from claude_agent_sdk import tool as sdk_tool

    from trellis.harness.adapters.claude import BUILTIN_SIDE_EFFECTS, ClaudeAdapter

    async def names(options: ClaudeAgentOptions, adapter: ClaudeAdapter | None = None) -> list[str]:
        return [s.name for s in await (adapter or ClaudeAdapter()).native(options)]

    assert await names(ClaudeAgentOptions()) == sorted(BUILTIN_SIDE_EFFECTS)
    narrowed = ClaudeAgentOptions(
        tools=["Read", "Bash"],
        disallowed_tools=["Bash", "mcp__crm__delete"],
        allowed_tools=["Read", "mcp__crm__find", "mcp__crm__delete", "mcp__trellis__stock"],
    )
    assert await names(narrowed) == ["Read", "mcp__crm__find"]
    assert await names(ClaudeAgentOptions(tools=[])) == []
    preset = ClaudeAgentOptions(tools={"type": "preset", "preset": "claude_code"})
    assert await names(preset) == sorted(BUILTIN_SIDE_EFFECTS)

    @sdk_tool("lookup", "Look an order up.", {"order": str})
    async def lookup(args: dict[str, Any]) -> dict[str, Any]:
        return {"content": [{"type": "text", "text": "found"}]}

    # an in-process server's tools are listed; a tool the agent was seen calling joins the set
    server = create_sdk_mcp_server("orders", tools=[lookup])
    adapter = ClaudeAdapter()
    listing = ClaudeAgentOptions(tools=[], mcp_servers={"orders": server})
    [found] = await adapter.native(listing)
    assert (found.name, found.source, found.server) == ("mcp__orders__lookup", "mcp", "orders")
    adapter.observed("mcp__wiki__search")
    adapter.observed("Read")  # a built-in is declared, not learned
    assert await names(listing, adapter) == ["mcp__orders__lookup", "mcp__wiki__search"]
    odd: Any = {"type": "sdk", "name": "odd", "instance": object()}
    broken = ClaudeAgentOptions(tools=[], mcp_servers={"odd": odd})
    assert await names(broken) == []  # cannot be listed: said once, and left out


async def test_a_function_has_no_tools_of_its_own(memory_service: FakeMemoryService) -> None:
    async def fn(input: str, agent: Runtime) -> str:
        return "x"

    async with harness_with(memory_service) as h:
        await arm(h.wrap(fn, id="f"), [stock]).run("q", user="u")
    [asked] = memory_service.named("context")
    assert asked.body["tools"]["available"] == ["stock"]


async def test_without_tool_search_the_own_tools_still_bring_the_skills(
    memory_service: FakeMemoryService,
) -> None:
    def make(i: int) -> Any:
        def act(key: str) -> str:
            return key

        act.__name__ = f"own{i}"
        return function_tool(act)

    target = OpenAIAgent(name="o", model=ScriptedModel(["ok"]), tools=[make(i) for i in range(6)])
    async with harness_with(memory_service) as h:
        await h.wrap(target, id="o").run("q", user="u", without={"tool_search"})
    [asked] = memory_service.named("context")
    assert asked.body["tools"]["hints"] is False and len(asked.body["tools"]["available"]) == 6


# --------------------------------------------------------------------------- 5. feedback
async def test_a_persons_verdict_cites_the_memories_the_answer_rested_on(
    memory_service: FakeMemoryService,
    sampled: Any,
) -> None:
    async def fn(input: str, agent: Runtime) -> str:
        return "you prefer email"

    sampled(1.0)
    async with harness_with(memory_service) as h:  # a verified run
        result = await h.wrap(fn, id="f").run("contact?", user="u")
        await h.writes.drain()  # the check, and with it the judge's verdict
        # another voice on the run citing something else is not what the answer rested on
        memory_service.stored_feedback["fb_other"] = {
            **next(iter(judged(memory_service))),
            "feedback_id": "fb_other",
            "source": "human",
            "evidence_refs": [{"source_type": "memory", "source_id": "mem_99"}],
        }
        stored = await h.feedback(result.run_id, "reject")
    [sent] = [f for f in memory_service.named("feedback") if f.body["source"] == "human"]
    cited = [(r["source_type"], r["source_id"]) for r in sent.body["evidence_refs"]]
    assert cited == [("memory", "mem_7"), ("memory", "mem_9")]  # a passage is no memory
    assert stored is not None and [r.source_id for r in stored.evidence_refs] == ["mem_7", "mem_9"]


async def test_a_verdict_on_a_run_never_checked_cites_nothing(
    memory_service: FakeMemoryService,
) -> None:
    async def fn(input: str, agent: Runtime) -> str:
        return "fine"

    async with harness_with(memory_service) as h:
        result = await h.wrap(fn, id="f").run("q", user="u", without={"memory"})
        await h.feedback(result.run_id, "confirm")
    [sent] = [f for f in memory_service.named("feedback") if f.body["source"] == "human"]
    assert sent.body["evidence_refs"] == []


@pytest.mark.parametrize("listed", [200, 404])
async def test_a_verdict_on_an_unverified_run_cites_what_its_context_showed(
    memory_service: FakeMemoryService, listed: int, sampled: Any
) -> None:
    """Outside the sample the run has no judge's verdict: the person's verdict cites the
    memories the run's pushed context showed, which the run's own outcome record keeps with
    its ``bundle_id`` — and is sent citing nothing when the service knows nothing of the
    run."""
    if listed == 404:
        memory_service.failures["feedback_list"] = 404

    async def fn(input: str, agent: Runtime) -> str:
        return "an answer"

    sampled(0.0)
    async with harness_with(memory_service) as h:
        result = await h.wrap(fn, id="f").run("q", user="u")
        await h.writes.drain()
        stored = await h.feedback(result.run_id, "confirm")
    assert memory_service.named("verify") == []
    [outcome] = [f for f in memory_service.named("feedback") if f.body["source"] == "system"]
    kept = outcome.body["metadata"]  # flat: the service keeps metadata two levels deep
    assert kept[PUSHED_BUNDLE].startswith("bnd_") and kept[PUSHED_MEMORIES] == ["mem_1", "mem_2"]
    [sent] = [f for f in memory_service.named("feedback") if f.body["source"] == "human"]
    assert stored is not None
    if listed == 404:
        assert sent.body["evidence_refs"] == []
        return
    [asked] = memory_service.named("feedback_list")  # as the service's GET /v1/feedback takes it
    assert asked.query["target_kind"] == "run"
    assert asked.query["target_id"] == asked.query["agent_run_id"] == result.run_id
    cited = [(r["source_type"], r["source_id"]) for r in sent.body["evidence_refs"]]
    assert cited == [("memory", "mem_1"), ("memory", "mem_2")]


async def test_a_run_whose_context_showed_no_memory_cites_nothing(
    memory_service: FakeMemoryService,
    sampled: Any,
) -> None:
    memory_service.shown = []

    async def fn(input: str, agent: Runtime) -> str:
        return "an answer"

    sampled(0.0)
    async with harness_with(memory_service) as h:
        result = await h.wrap(fn, id="f").run("q", user="u")
        await h.writes.drain()
        await h.feedback(result.run_id, "confirm")
    [outcome] = [f for f in memory_service.named("feedback") if f.body["source"] == "system"]
    assert outcome.body["metadata"][PUSHED_MEMORIES] == []
    [sent] = [f for f in memory_service.named("feedback") if f.body["source"] == "human"]
    assert sent.body["evidence_refs"] == []


async def test_the_cited_memories_are_each_once_and_bounded(
    memory_service: FakeMemoryService,
    sampled: Any,
) -> None:
    memory_service.cited = [f"mem_{i}" for i in range(CITED_MAX + 10)] + ["mem_0"]

    async def fn(input: str, agent: Runtime) -> str:
        return "an answer"

    sampled(1.0)
    async with harness_with(memory_service) as h:  # a verified run
        result = await h.wrap(fn, id="f").run("q", user="u")
        await h.writes.drain()
        await h.feedback(result.run_id, "confirm")
    [sent] = [f for f in memory_service.named("feedback") if f.body["source"] == "human"]
    ids = [r["source_id"] for r in sent.body["evidence_refs"]]
    assert ids == [f"mem_{i}" for i in range(CITED_MAX)]


async def test_a_verdict_is_not_sent_without_its_citations(
    memory_service: FakeMemoryService,
) -> None:
    """The memories cannot be read: the verdict is refused, not sent citing nothing."""

    async def fn(input: str, agent: Runtime) -> str:
        return "an answer"

    async with harness_with(memory_service) as h:
        result = await h.wrap(fn, id="f").run("q", user="u")
        await h.writes.drain()
        memory_service.fail = {"feedback_list"}
        with pytest.raises(DependencyUnavailableError):
            await h.feedback(result.run_id, "confirm")
    assert not [f for f in memory_service.named("feedback") if f.body["source"] == "human"]


# --------------------------------------------------------------------------- 6. documents
async def test_a_run_draws_its_context_from_the_documents_it_names(
    memory_service: FakeMemoryService,
) -> None:
    async def fn(input: str, agent: Runtime) -> str:
        return "30 days"

    async with harness_with(memory_service) as h:
        info = await h.add_document(b"Returns: 30 days.", user="ada", wait=None)
        agent = h.wrap(fn, id="docs")
        await agent.run("returns?", user="ada", document_ids=[info.document_id, info.document_id])
        await agent.run("anything?", user="ada")
    scoped, unscoped = memory_service.named("context")
    assert scoped.body["document_ids"] == [info.document_id]
    assert "document_ids" not in unscoped.body


async def test_the_documents_are_kept_with_the_run_and_its_sub_agents(
    memory_service: FakeMemoryService,
) -> None:
    """A resume, a worker's attempt and a sub-agent's run draw on the same documents."""

    @tool(side_effects="irreversible")
    def send(text: str) -> str:
        """Send it."""
        return "sent"

    async def child(input: str, agent: Runtime) -> str:
        return "summary"

    async def parent(input: str, agent: Runtime) -> Any:
        await agent.tools.call("summarize", message=input)
        return await agent.tools.call("send", text="summary")

    async with harness_with(memory_service) as h:
        summarize = h.wrap(child, id="summarize")
        lead = arm(h.wrap(parent, id="lead"), [summarize, send])
        handle = await lead.start("sum up", user="ada", document_ids=["doc_1"])
        assert await h.worker([lead]).run_once()
        paused = await handle.result(wait=5)
        assert paused.interrupt is not None
        await lead.resume(paused.run_id, "approve", reviewer="ada")
        assert await h.worker([lead]).run_once()
        done = await handle.result(wait=5)
    assert done.status is RunStatus.SUCCESS
    contexts = memory_service.named("context")
    assert {c.scope["agent_id"] for c in contexts} == {"lead", "summarize"}
    assert len(contexts) == 3  # the first attempt, the child, the attempt after the answer
    assert all(c.body["document_ids"] == ["doc_1"] for c in contexts)


@pytest.mark.parametrize("given", ["doc_1", [], [""], [3]])
async def test_document_ids_are_a_list_of_ids(harness: Harness, given: Any) -> None:
    async def fn(input: str, agent: Runtime) -> str:
        return "x"

    with pytest.raises(ConfigurationError, match="document_ids="):
        await harness.wrap(fn, id="d").run("q", user="u", document_ids=given)


async def test_a_schedule_keeps_its_documents(harness: Harness) -> None:
    async def fn(input: str, agent: Runtime) -> str:
        return "x"

    agent = harness.wrap(fn, id="nightly")
    assert agent._metadata(frozenset(), None, ["doc_9"]) == {"document_ids": ["doc_9"]}


# --------------------------------------------------------------------------- 7. unreached
async def test_a_function_that_never_reads_its_context_is_told_once(
    memory_service: FakeMemoryService, caplog: pytest.LogCaptureFixture
) -> None:
    async def ignores(input: str, agent: Runtime) -> str:
        return "answered blind"

    async with harness_with(memory_service) as h:
        agent = h.wrap(ignores, id="blind")
        first = [e async for e in agent.stream("contact?", user="u")]
        second = [e async for e in agent.stream("again?", user="u")]
    [said] = warned(first, "memory_context")
    assert "blind" in said["message"] and UNREACHED_FIX["function"] in said["message"]
    assert warned(second, "memory_context") == [], "once per agent"
    assert caplog.text.count("reaches neither its input nor its code") == 1


@pytest.mark.parametrize("given", ["reads", "reads current()", "messages", "no context"])
async def test_a_function_given_the_context_is_not_told(
    memory_service: FakeMemoryService, given: str
) -> None:
    def helper() -> str:  # code the function calls: the run's context through current()
        runtime = current()
        assert runtime is not None
        return f"{runtime.context}"

    async def reads(input: Any, agent: Runtime) -> str:
        if given == "reads current()":
            assert helper() == memory_service.context_text
        return f"{agent.context}" if given == "reads" else "fine"

    if given == "no context":
        memory_service.fail = {"context"}
    question: Any = [{"role": "user", "content": "contact?"}] if given == "messages" else "contact?"
    async with harness_with(memory_service) as h:
        events = [e async for e in h.wrap(reads, id="ok").stream(question, user="u")]
    assert warned(events, "memory_context") == []


class Ticket(TypedDict):
    question: str
    answer: str


def ticket_graph(reads: bool) -> Any:
    def answer(state: Ticket) -> dict[str, str]:
        runtime = current()
        assert runtime is not None
        context = runtime.context if reads else None
        return {"answer": f"{state['question']} handled{' with memory' if context else ''}"}

    builder = StateGraph(Ticket)
    builder.add_node("answer", answer)
    builder.add_edge(START, "answer")
    builder.add_edge("answer", END)
    return builder.compile()


@pytest.mark.parametrize("reads", [False, True])
async def test_a_graph_whose_input_has_no_messages_is_told_unless_a_node_reads_it(
    memory_service: FakeMemoryService, reads: bool, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING, logger="trellis.run")
    async with harness_with(memory_service) as h:
        agent = h.wrap(ticket_graph(reads), id="tickets")
        question = {"question": "Is T-1 urgent?", "answer": ""}
        events = [e async for e in agent.stream(question, user="u")]
    said = warned(events, "memory_context")
    if reads:
        assert said == []
    else:
        assert UNREACHED_FIX["langgraph"] in said[0]["message"]
